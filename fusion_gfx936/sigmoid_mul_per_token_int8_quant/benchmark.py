"""Compare the original fused attention kernel and its tuned dispatch on the same inputs."""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch

if __package__:
    from .python_api import _launch, _launch_original
else:
    from python_api import _launch, _launch_original


def measure(tokens, width, dtype, layout, warmup, rounds, repeats):
    x = torch.randn(tokens, width, device="cuda", dtype=dtype)
    if layout == "projection":
        # The 24/4-head attention model stores Q/gate pairs before the K/V tail.
        row_stride = 7168 if width == 3072 else 14336 if width == 6144 else 2 * width
        backing = torch.randn(tokens, row_stride, device=x.device, dtype=dtype)
        gate = backing[:, :2 * width].view(tokens, width // 256, 2, 256)[:, :, 1, :]
    else:
        gate = torch.randn_like(x)
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(tokens, 1, device=x.device, dtype=torch.float32)
    ref_q, ref_s = torch.empty_like(q), torch.empty_like(s)

    def before():
        _launch_original(x, gate, ref_q, ref_s)

    def after():
        _launch(x, gate, q, s)

    for _ in range(max(1, warmup)):
        before()
        after()
    if not torch.equal(q, ref_q) or not torch.equal(s.view(torch.int32), ref_s.view(torch.int32)):
        raise RuntimeError(f"Bitwise mismatch at M={tokens}, D={width}, {layout}, {dtype}")
    graphs = []
    for fn in (before, after):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        graph.replay()
        graphs.append(graph)
    torch.cuda.synchronize()
    start, stop = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    times = [[], []]
    rng = random.Random(tokens)
    for _ in range(rounds):
        for i in rng.sample(range(2), 2):
            start.record()
            graphs[i].replay()
            stop.record()
            stop.synchronize()
            times[i].append(start.elapsed_time(stop) * 1000 / repeats)
    old, new = map(statistics.median, times)
    useful = tokens * (5 * width + 4)
    return dict(tokens=tokens, width=width, dtype=str(dtype), layout=layout,
                original_us=old, optimized_us=new, speedup=old / new,
                original_TBps=useful / old / 1e6, optimized_TBps=useful / new / 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hidden", type=int, choices=(3072, 4096, 6144), default=3072)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 64, 128, 512, 1024, 4096, 16384])
    parser.add_argument("--layout", nargs="+", choices=("contiguous", "projection"), default=["contiguous", "projection"])
    parser.add_argument("--dtype", nargs="+", choices=("bfloat16", "float16"), default=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--graph-repeats", type=int, default=16)
    args = parser.parse_args()
    if min(args.tokens) < 1 or min(args.repetitions, args.graph_repeats) < 1 or args.warmup < 0:
        parser.error("tokens and repetitions must be positive and warmup nonnegative")
    torch.manual_seed(20261107)
    records = []
    for m in args.tokens:
        for dtype in args.dtype:
            for layout in args.layout:
                row = measure(m, args.hidden, getattr(torch, dtype), layout,
                              args.warmup, args.repetitions, args.graph_repeats)
                records.append(row)
                print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"torch": str(torch.__version__), "hip": torch.version.hip,
                                          "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
