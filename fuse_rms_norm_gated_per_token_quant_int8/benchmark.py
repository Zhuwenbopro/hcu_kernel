"""GPU-event benchmark of existing GDN norm + quant versus the new fusion.

Run from the repository: python test/benchmark_gdn_rms_norm_gated_int8.py --output results.json
Input generation and correctness comparisons are outside timed regions.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from lightop import fuse_rms_norm_gated_per_token_quant_int8
from lightop.norm import layer_norm_fwd_1pass_opt
from lightop.quant import per_token_quant_int8


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def measure(fn, warmup, repetitions, graph):
    for _ in range(warmup):
        outputs = fn()
    torch.cuda.synchronize()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            outputs = fn()
        run = capture.replay
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
    else:
        run = fn
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends):
        start.record()
        outputs = run()
        end.record()
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.1),
        "p90_us": percentile(values, 0.9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    records = []
    for dtype in (torch.bfloat16, torch.float16):
        for tokens in (1, 4, 26, 104, 257, 1024, 20000):
            torch.manual_seed(20261107 + tokens)
            x = torch.randn((tokens, 4096), device="cuda", dtype=dtype)
            z = torch.randn_like(x)
            weight = torch.randn(128, device=x.device, dtype=dtype)
            head_x, head_z = (t.view(-1, 128) for t in (x, z))
            norm = torch.empty_like(head_x)
            rstd = torch.empty(head_x.shape[0], device=x.device, dtype=torch.float32)

            def old():
                layer_norm_fwd_1pass_opt(
                    head_x, norm, weight, None, head_z, None, rstd,
                    128, 128, 128, head_x.shape[0], 128, 1e-6, 128, 16,
                    False, True, True, True, "swish",
                )
                return per_token_quant_int8(norm.view_as(x))

            def new():
                return fuse_rms_norm_gated_per_token_quant_int8(x, weight, z, 1e-6)

            q, scale = new()
            ref_q, ref_scale = old()
            mismatches = int((q != ref_q).sum())
            scale_mismatches = int((scale.view(torch.int32) != ref_scale.view(torch.int32)).sum())
            assert not mismatches and not scale_mismatches, (dtype, tokens, mismatches, scale_mismatches)
            for graph in (False, True):
                before = measure(old, args.warmup, args.repetitions, graph)
                after = measure(new, args.warmup, args.repetitions, graph)
                row = {
                    "dtype": str(dtype), "tokens": tokens, "heads": 32, "head_dim": 128,
                    "mode": "graph" if graph else "eager", "old": before, "new": after,
                    "speedup_pct": 100 * (before["median_us"] / after["median_us"] - 1),
                    "int8_elements": q.numel(), "int8_mismatches": mismatches,
                    "scale_bit_mismatches": scale_mismatches,
                }
                records.append(row)
                Path(args.output).write_text(json.dumps({
                    "warmup": args.warmup, "repetitions": args.repetitions,
                    "torch": torch.__version__, "hip": torch.version.hip,
                    "device": torch.cuda.get_device_name(), "records": records,
                }, indent=2))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
