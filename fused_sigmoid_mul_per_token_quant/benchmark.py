"""GPU-event comparison: original sigmoid-mul + quant versus attention fusion."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from lightop.quant import per_token_quant_int8

if __package__:
    from .attention_sigmoid_int8_reference import assert_accuracy, comparison_stats, sigmoid_mul_reference
    from .python_api import fuse_sigmoid_mul_int8_quant
else:
    from attention_sigmoid_int8_reference import assert_accuracy, comparison_stats, sigmoid_mul_reference
    from python_api import fuse_sigmoid_mul_int8_quant


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def measure(fn, warmup, repetitions, graph, graph_repeats):
    for _ in range(warmup):
        outputs = fn()
    torch.cuda.synchronize()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            for _ in range(graph_repeats):
                outputs = fn()
        _captured_outputs = outputs
        run = capture.replay
        for _ in range(10):
            run()
        torch.cuda.synchronize()
    else:
        run = fn
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    # Event objects allocate native resources at their first record. Initialize
    # them before measuring, and amortize graph submission over many operators.
    for start, end in zip(starts, ends):
        start.record()
        end.record()
    torch.cuda.synchronize()
    for start, end in zip(starts, ends):
        start.record()
        outputs = run()
        end.record()
    torch.cuda.synchronize()
    divisor = graph_repeats if graph else 1
    values = [start.elapsed_time(end) * 1000 / divisor for start, end in zip(starts, ends)]
    return {"median_us": statistics.median(values), "p10_us": percentile(values, 0.1), "p90_us": percentile(values, 0.9)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--graph-repeats", type=int, default=128)
    parser.add_argument("--hidden", type=int, choices=(3072, 4096), default=4096)
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 4, 26, 104, 257, 1024, 20000))
    args = parser.parse_args()
    if min(args.tokens) < 1:
        parser.error("--tokens must be positive")
    if args.graph_repeats < 1:
        parser.error("--graph-repeats must be positive")
    records = []
    for dtype in (torch.bfloat16, torch.float16):
        for tokens in args.tokens:
            torch.manual_seed(20261107 + tokens)
            x = torch.randn((tokens, args.hidden), device="cuda", dtype=dtype)
            gate = torch.randn((tokens, args.hidden // 256, 2, 256), device=x.device, dtype=dtype)[:, :, 1, :]

            def old():
                return per_token_quant_int8(sigmoid_mul_reference(x, gate))

            def new():
                return fuse_sigmoid_mul_int8_quant(x, gate)

            stats = comparison_stats(new(), old())
            assert_accuracy(stats, dtype)
            for graph in (False, True):
                # Keep capture/measurement work bounded for the 20K-token case.
                batch = min(args.graph_repeats, max(4, 131072 // tokens))
                before = measure(old, args.warmup, args.repetitions, graph, batch)
                after = measure(new, args.warmup, args.repetitions, graph, batch)
                row = {
                    "dtype": str(dtype), "tokens": tokens, "hidden": args.hidden,
                    "gate_shape": list(gate.shape), "gate_stride": list(gate.stride()),
                    "mode": "graph" if graph else "eager", "old": before, "new": after,
                    "graph_repeats": batch if graph else 1,
                    "speedup_pct": 100 * (before["median_us"] / after["median_us"] - 1),
                    **stats,
                }
                records.append(row)
                Path(args.output).write_text(json.dumps({
                    "warmup": args.warmup, "repetitions": args.repetitions,
                    "torch": torch.__version__, "hip": torch.version.hip,
                    "device": torch.cuda.get_device_name(), "records": records,
                    "timing_method": "preinitialized GPU events; repeated operators inside each measured Graph",
                }, indent=2))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
