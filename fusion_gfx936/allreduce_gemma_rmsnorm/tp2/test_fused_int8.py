"""Check TP2 fused AR + Gemma RMSNorm + INT8 on two gfx936 ranks."""

import argparse
import datetime
import json
import os
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist

from tp2_ar import HipOneShotAllreduce, TunedOneShotAllreduce


WIDTH = 5120
EPS = 1e-6
WEIGHT_KEY = "model.language_model.layers.0.post_attention_layernorm.weight"
DEFAULT_SIZES = (1, 16, 65, 129, 512, 4096)


def inputs(rows, device, rank, seed, pattern):
    if pattern == "random":
        gen = torch.Generator(device=device).manual_seed(seed + 233 * rank)
        x = torch.randn((rows, WIDTH), device=device, dtype=torch.bfloat16, generator=gen)
        residual = torch.randn(x.shape, device=device, dtype=x.dtype, generator=gen)
        return x, residual

    x = torch.zeros((rows, WIDTH), device=device, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    if pattern == "ties" and rank == 0:
        # The row RMS is close enough to 1 that the BF16 norm reproduces these
        # entries. Multiplying by 127/(254/128) produces even and odd half-ties.
        x.fill_(1)
        x[:, :5] = x.new_tensor([254 / 128, 253 / 128, -253 / 128, 127 / 128, -127 / 128])
    elif pattern == "tiny" and rank == 0:
        x.fill_(2.0**-55)
    elif pattern == "rounding":
        if rank == 0:
            x.fill_(1)
        residual.fill_(1 / 256)
    elif pattern == "signed_zero":
        x.fill_(-0.0)
        residual.fill_(-0.0)
        x[:, 1] = 0.0
        residual[:, 2] = 0.0
    elif pattern == "subnormal" and rank == 0:
        x.view(torch.int16)[:, :4] = torch.tensor(
            [1, -32767, 127, -32641], device=device, dtype=torch.int16
        )
    elif pattern == "bf16_max" and rank == 0:
        maximum = torch.finfo(x.dtype).max
        x[:, :2] = x.new_tensor([maximum, -maximum])
    elif pattern == "infinity" and rank == 0:
        x[:, :2] = x.new_tensor([float("inf"), float("-inf")])
    elif pattern == "nan" and rank == 0:
        x.fill_(1)
        x.view(torch.int16)[:, 2] = 0x7FC1
    return x, residual


def fast_reference(reduced, residual, weight):
    added = reduced.float() + residual.float()
    weighted = added * (1 + weight.float())
    rstd = (added.square().mean(dim=1, keepdim=True) + EPS).rsqrt()
    absmax = (weighted.abs().amax(dim=1, keepdim=True) * rstd).clamp_min(1e-10)
    q = (weighted * rstd * (127 / absmax)).round().to(torch.int8)
    return q, absmax / 127, added.bfloat16()


def bit_mismatches(actual, expected):
    bits = {torch.int8: torch.int8, torch.bfloat16: torch.int16, torch.float32: torch.int32}
    return [
        int(torch.count_nonzero(value.view(bits[value.dtype]) != gold.view(bits[gold.dtype])))
        for value, gold in zip(actual, expected)
    ]


def check_outputs(actual, expected, fast, label):
    for value, gold in zip(actual, expected):
        if value.shape != gold.shape or value.dtype != gold.dtype:
            return f"{label}: got {value.shape} {value.dtype}, expected {gold.shape} {gold.dtype}"

    counts = bit_mismatches(actual, expected)
    if not fast:
        return f"{label}: q/scale/residual mismatches {counts}" if any(counts) else None

    q, scale, _ = actual
    max_delta = int((q.int() - expected[0].int()).abs().max())
    scale_matches = torch.allclose(scale, expected[1], rtol=2e-6, atol=0)
    if max_delta > 1 or counts[0] > max(4, q.numel() // 10_000) or not scale_matches or counts[2]:
        return (
            f"{label}: q/scale/residual bit mismatches {counts}, max q delta {max_delta}, "
            f"scale within tolerance {scale_matches}"
        )
    return None


def verify_cases(comm, ref, norm, quantize, local, rows, patterns, *, check_fast=True):
    rank, device = local
    shape = (rows, WIDTH)
    x = torch.empty(shape, device=device, dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    reduced = torch.empty_like(x)
    errors = []

    def execute():
        exact = comm.fused_int8(x, residual, norm.weight, EPS)
        if not check_fast:
            return (exact,)
        fast = comm.fused_int8(x, residual, norm.weight, EPS, fast=True)
        return exact, fast

    for step, pattern in enumerate(patterns):
        a, r = inputs(rows, device, rank, 1303 + step * 271, pattern)
        x.copy_(a)
        residual.copy_(r)
        ref.all_reduce(x, out=reduced, registered_input=False)
        fast_expected = fast_reference(reduced, residual, norm.weight) if check_fast else None
        y, updated = norm(reduced, residual.clone())
        q, scale = quantize(y)
        exact_expected = (q, scale, updated)

        if step == 0:
            actual = execute()
            phase = "eager"
        else:
            graph.replay()
            actual = captured
            phase = f"graph-{step}"

        scenario = f"{phase} M={rows} {pattern}"
        for fast, observed, expected in zip((False, True), actual, (exact_expected, fast_expected)):
            label = f"{scenario} {'fast' if fast else 'exact'}"
            if error := check_outputs(observed, expected, fast, label):
                errors.append(error)
        if pattern == "ties" and q[0, :5].tolist() != [127, 126, -126, 64, -64]:
            errors.append(f"{scenario}: LightOp reference did not exercise the intended half-ties")
        if pattern == "rounding" and torch.equal(actual[0][1], actual[1][1]):
            errors.append(f"{scenario}: the two modes used the same activation scale")
        if any(bit_mismatches((x, residual), (a, r))):
            errors.append(f"{scenario}: fused call mutated an input")

        if step == 0:
            with comm.capture():
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin()
                captured = execute()
                graph.capture_end()
    return errors


def make_norm(norm_type, device, model, zero_weight=False):
    norm = norm_type(WIDTH, eps=EPS).to(device=device, dtype=torch.bfloat16)
    if zero_weight:
        weight = torch.zeros_like(norm.weight)
    elif model is None:
        gen = torch.Generator(device=device).manual_seed(4618)
        weight = torch.randn(WIDTH, device=device, dtype=torch.bfloat16, generator=gen) * 0.1
    else:
        from safetensors import safe_open

        index = json.loads((model / "model.safetensors.index.json").read_text())
        with safe_open(model / index["weight_map"][WEIGHT_KEY], framework="pt") as weights:
            weight = weights.get_tensor(WEIGHT_KEY).to(device)
    norm._weight_loader(norm.weight, weight)
    return norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--model", type=Path, help="optional Qwen checkpoint with the Gemma norm weight")
    parser.add_argument("--variant", choices=("fixed", "tuned", "both"), default="tuned")
    args = parser.parse_args()
    if not all(1 <= rows <= 4096 for rows in args.sizes):
        parser.error("--sizes must be in 1..4096")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        parser.error("run with torchrun --standalone --nproc-per-node=2")

    os.environ.setdefault("HCU_ADAPTER_HOST", "sglang")
    os.environ.setdefault("SGLANG_PLUGINS", "hcu")
    os.environ.setdefault("AITER_AR_TRANSPORT", "ipc")
    os.environ.setdefault("AITER_AR_MAX_SIZE_MB", "128")
    os.environ.setdefault("AITER_AR_SMALL_MESSAGE", "off")
    os.environ.setdefault("HCU_FUSED_ADD_RMS_IMPL", "triton")
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=240))

    with ExitStack() as resources:
        resources.callback(dist.destroy_process_group)
        from sglang.srt.plugins import load_plugins

        load_plugins()
        from hcu_sglang.distributed_aiter import dispatch_custom_allreduce
        from lightop.quant import per_token_quant_int8
        from sgl_kernel.layernorm import _resolve_fused_add_impl
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        if _resolve_fused_add_impl() != "triton":
            raise RuntimeError("reference requires the HCU Triton Gemma fused-add path")
        group = dist.group.WORLD
        ref = dispatch_custom_allreduce(group, device, backend="aiter")(group=group, device=device)
        resources.callback(ref.close)
        if ref.disabled or ref.transport != "ipc" or ref.enable_register_for_capturing:
            raise RuntimeError("reference requires the enabled IPC AITER copy-in path")
        norm = make_norm(GemmaRMSNorm, device, args.model)
        zero_norm = make_norm(GemmaRMSNorm, device, None, zero_weight=True)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        variants = {"fixed": HipOneShotAllreduce, "tuned": TunedOneShotAllreduce}
        names = tuple(variants) if args.variant == "both" else (args.variant,)
        for name in names:
            with ExitStack() as variant_resources:
                comm = variants[name](group, device, max_size=max(args.sizes) * WIDTH * 2)
                variant_resources.callback(comm.close)
                with torch.cuda.stream(stream):
                    errors = []
                    for rows in args.sizes:
                        errors.extend(verify_cases(comm, ref, norm, per_token_quant_int8,
                                                   (rank, device), rows, ("random",) * 3))
                    errors.extend(verify_cases(comm, ref, zero_norm, per_token_quant_int8,
                                               (rank, device), 1, ("ties", "zero", "tiny", "rounding")))
                    errors.extend(verify_cases(
                        comm, ref, zero_norm, per_token_quant_int8, (rank, device), 1,
                        ("signed_zero", "subnormal", "bf16_max", "infinity", "nan"), check_fast=False,
                    ))
                stream.synchronize()
            all_errors = [None, None]
            dist.all_gather_object(all_errors, errors)
            if any(all_errors):
                raise AssertionError(f"{name} validation by rank: {all_errors}")
            if rank == 0:
                print(f"{name}: exact and fast checks passed on both ranks for M={list(args.sizes)} "
                      "and special-value rows", flush=True)


if __name__ == "__main__":
    main()
