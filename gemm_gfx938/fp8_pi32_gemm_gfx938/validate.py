"""Validate the gfx938 FP8 GEMM against sampled FP32 products."""

import argparse

import torch

from .python_api import SHAPES, gemm, get_config, pack_weight, unpack_weight


SMOKE = (1, 16, 24, 64, 96, 128)
GRAPH_SMOKE = (1, 96, 128)
FP8 = torch.float8_e4m3fn


def activation(m, k, device):
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device).to(FP8)
    scale = torch.rand((m, 1), dtype=torch.float32, device=device).mul_(0.02).add_(0.01)
    return x, scale


def reference_weights(weight, scale):
    n = weight.shape[0]
    device = weight.device
    spread = torch.arange(128, device=device) * (n - 1) // 127
    edges = torch.arange(64, device=device)
    cols = torch.cat((spread, edges, n - 1 - edges)).unique()
    selected = weight.view(torch.uint8).index_select(0, cols).view(FP8).float()
    return cols, selected, scale.index_select(0, cols).T


def check_result(x, scale, output, sampled, where):
    cols, weight, weight_scale = sampled
    assert output.dtype == torch.bfloat16 and output.is_contiguous(), where
    assert torch.isfinite(output).all().item(), f"{where}: nonfinite output"
    expected = ((x.float() @ weight.T) * scale * weight_scale).to(torch.bfloat16).float()
    actual = output.index_select(1, cols).float()
    error = torch.linalg.vector_norm(actual - expected)
    relative_l2 = (error / torch.linalg.vector_norm(expected).clamp_min(1e-12)).item()
    if relative_l2 > 0.004:
        raise AssertionError(f"{where}: sampled relative L2 {relative_l2:.6g} > 0.004")
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=5e-5, msg=where)
    return relative_l2


def validate_graph(role, x, scale, packed, weight_scale, sampled, config):
    m, k = x.shape
    n = weight_scale.numel()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    workspace = None
    if config.splits > 1:
        workspace = torch.empty((config.splits, m, n), dtype=torch.float32, device=x.device)
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        gemm(x, packed, scale, weight_scale, role=role, out=out, workspace=workspace)
    torch.cuda.current_stream(x.device).wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        returned = gemm(x, packed, scale, weight_scale, role=role, out=out, workspace=workspace)
    assert returned.data_ptr() == out.data_ptr(), "GEMM did not use the supplied output buffer"

    worst = 0.0
    for replay in range(2):
        new_x, new_scale = activation(m, k, x.device)
        if replay == 1:
            new_scale[0] = 0
        x.view(torch.uint8).copy_(new_x.view(torch.uint8))
        scale.copy_(new_scale)
        out.fill_(float("nan"))
        graph.replay()
        where = f"{role} M={m} graph replay {replay + 1}"
        assert torch.equal(x.view(torch.uint8), new_x.view(torch.uint8)), where
        assert torch.equal(scale, new_scale), where
        worst = max(worst, check_result(new_x, new_scale, out, sampled, where))
        if replay == 1:
            assert torch.count_nonzero(out[0]).item() == 0, f"{where}: zero input scale produced nonzero output"
    return worst


@torch.no_grad()
def run(roles, sizes, graph_sizes, device):
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if not getattr(props, "gcnArchName", "").startswith("gfx938"):
        raise RuntimeError(f"Validation requires a gfx938 device, got {props.name}")
    torch.manual_seed(3141)
    torch.set_float32_matmul_precision("highest")

    for role in roles:
        k, n = SHAPES[role]
        weight = torch.randn((n, k), dtype=torch.bfloat16, device=device).to(FP8)
        weight_scale = torch.rand((n, 1), dtype=torch.float32, device=device).mul_(0.02).add_(0.01)
        packed = pack_weight(weight)
        assert packed.shape == (k // 64, n // 16, 4, 16, 16)
        restored = torch.empty_like(weight)
        returned = unpack_weight(packed, out=restored)
        assert returned.data_ptr() == restored.data_ptr()
        assert torch.equal(restored.view(torch.uint8), weight.view(torch.uint8)), f"{role}: W round trip differs"
        del restored, returned
        sampled = reference_weights(weight, weight_scale)
        del weight

        for m in sizes:
            config = get_config(m, k, n, role=role)
            x, scale = activation(m, k, device)
            original_x = x.view(torch.uint8).clone()
            original_scale = scale.clone()
            out = gemm(x, packed, scale, weight_scale, role=role)
            where = f"{role} M={m} eager"
            assert out.shape == (m, n), where
            eager_l2 = check_result(x, scale, out, sampled, where)
            assert torch.equal(x.view(torch.uint8), original_x), where
            assert torch.equal(scale, original_scale), where
            graph = ""
            if m in graph_sizes:
                graph_l2 = validate_graph(role, x, scale, packed, weight_scale, sampled, config)
                graph = f" graph_l2={graph_l2:.3g}"
            print(
                f"{role} M={m:3d} {config.provider} split={config.splits} "
                f"depth={config.depth} tile_n={config.tile_n} eager_l2={eager_l2:.3g}{graph}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=SMOKE, help="M values between 1 and 128")
    parser.add_argument("--roles", nargs="+", choices=tuple(SHAPES), default=tuple(SHAPES))
    parser.add_argument("--device", type=int, default=0, help="HIP device index after visibility filtering")
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--graph", action="store_true", help="Check graph replay at every selected M")
    capture.add_argument("--no-graph", action="store_true", help="Skip graph replay")
    args = parser.parse_args()
    sizes = sorted(set(args.m))
    if any(not 1 <= m <= 128 for m in sizes):
        parser.error("--m values must be between 1 and 128")
    graph_sizes = () if args.no_graph else sizes if args.graph else GRAPH_SMOKE
    run(args.roles, sizes, graph_sizes, torch.device("cuda", args.device))


if __name__ == "__main__":
    main()
