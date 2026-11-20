"""Compare GDN FP8 fusion with the two-stage LightOp path on gfx938."""

import argparse
from statistics import median

import torch
import triton

from . import gated_rms_norm_per_token_fp8_quant


def benchmark(tokens, repeats):
    from lightop.norm import layer_norm_fwd_1pass_opt
    from lightop.quant import per_token_quant_fp8

    heads, width = 24, 128
    rows = tokens * heads
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(width, device=x.device, dtype=x.dtype) * 0.25 + 1
    y = torch.empty_like(x)
    rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
    units = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_rows = min(4, triton.next_power_of_2(triton.cdiv(rows, 2 * units)))

    def baseline():
        layer_norm_fwd_1pass_opt(
            x, y, weight, None, gate, None, rstd,
            width, width, width, rows, width, 1e-6, width, tile_rows,
            False, True, True, True, "swish",
        )
        return per_token_quant_fp8(y.view(tokens, heads * width))

    def fused():
        return gated_rms_norm_per_token_fp8_quant(x, gate, weight, 1e-6, num_heads=heads)

    q, scale = baseline()
    fused_q, fused_scale = fused()
    torch.testing.assert_close(fused_scale, scale, rtol=8e-3, atol=0)
    values = q.float() * scale
    torch.testing.assert_close(fused_q.float() * fused_scale, values, rtol=0.12, atol=0.002)
    assert (fused_q.float() * fused_scale - values).norm() <= 8e-3 * values.norm()

    graphs = []
    iterations = 128
    for fn in (baseline, fused):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                fn()
        graph.replay()
        graphs.append(graph)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = [[], []]
    for repeat in range(repeats):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            graphs[index].replay()
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / iterations)
    baseline_us, fused_us = map(median, samples)
    print(f"{tokens:>5}  {baseline_us:>8.3f}  {fused_us:>8.3f}  {1-fused_us/baseline_us:>7.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 64, 128, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if min(*args.tokens, args.repeats) < 1:
        parser.error("tokens and repeats must be positive")
    if (
        torch.version.hip is None or not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx938"
    ):
        parser.error("requires a gfx938 GPU")
    torch.manual_seed(417)
    print("tokens   split us  fused us  reduction")
    for tokens in args.tokens:
        benchmark(tokens, args.repeats)


if __name__ == "__main__":
    main()
