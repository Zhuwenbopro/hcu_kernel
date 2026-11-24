"""Compare the original and tuned 24-head gfx936 kernels on preallocated outputs."""
import argparse
import csv
import random
import statistics
import sys

import torch

from .python_api import _gated_exact_heads_kernel, _launch_gfx936_24, _is_gfx936


def benchmark(m, layout, rounds):
    h, d = 24, 128
    x = torch.randn(m * h, d, dtype=torch.bfloat16, device="cuda")
    if not _is_gfx936(x.device):
        raise RuntimeError("This comparison requires gfx936")
    if layout == "projection":
        backing = torch.randn(m, 8192, dtype=x.dtype, device=x.device)
        z = backing[:, 5120:].view(m, h, d)
    else:
        z = torch.randn(m, h, d, dtype=x.dtype, device=x.device)
    weight = torch.randn(d, dtype=x.dtype, device=x.device) * 0.25 + 1
    q = torch.empty(m, h * d, dtype=torch.int8, device=x.device)
    scale = torch.empty(m, 1, dtype=torch.float32, device=x.device)
    ref_q, ref_s = torch.empty_like(q), torch.empty_like(scale)
    warps = 8 if m <= 64 else 4 if m <= 512 else 2

    def before():
        _gated_exact_heads_kernel[(m,)](
            x, z, weight, ref_q, ref_s, 1e-6, x.stride(0), z.stride(0), z.stride(1),
            num_warps=warps,
        )

    def after():
        _launch_gfx936_24(x, z, weight, q, scale, 1e-6, z.stride(0), z.stride(1))

    calls = (before, after)
    for fn in calls:
        fn()
    # Allow the small-token SiLU table to finish initializing before capture.
    torch.cuda.synchronize()
    after()
    if not torch.equal(ref_q, q) or not torch.equal(ref_s.view(torch.int32), scale.view(torch.int32)):
        raise RuntimeError(f"Bitwise mismatch at M={m}, {layout}")
    repeats = 32 if m <= 256 else 8
    graphs = []
    for fn in calls:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        graph.replay()
        graphs.append(graph)
    torch.cuda.synchronize()
    start, stop = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    times = [[], []]
    rng = random.Random(m)
    for _ in range(rounds):
        for i in rng.sample(range(2), 2):
            start.record()
            graphs[i].replay()
            stop.record()
            stop.synchronize()
            times[i].append(start.elapsed_time(stop) * 1000 / repeats)
    old, new = map(statistics.median, times)
    useful = m * (5 * h * d + 4) + 2 * d
    return old, new, old / new, useful / old / 1e6, useful / new / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 64, 128, 512, 1024, 4096, 16384])
    parser.add_argument("--layout", nargs="+", choices=("contiguous", "projection"), default=["contiguous", "projection"])
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if min(*args.tokens, args.repeats) < 1:
        parser.error("tokens and repeats must be positive")
    torch.manual_seed(417)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("M", "layout", "before_us", "after_us", "speedup", "before_TBps", "after_TBps"))
    for m in args.tokens:
        for layout in args.layout:
            writer.writerow([m, layout, *[f"{v:.4f}" for v in benchmark(m, layout, args.repeats)]])


if __name__ == "__main__":
    main()
