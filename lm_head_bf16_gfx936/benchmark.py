"""Compare the gfx936 BF16 LM head with torch.mm on reference [N,K] weights."""

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

import torch

from . import K, MAX_M, N, LMHead, route


SMOKE = (1, 2, 3, 4, 15, 65, 129, 256, 257, 4096)
EVENTS_PER_SHAPE = 5
HBM_BYTES_PER_US = 1.5e6
L2_BYTES_PER_US = 6.0e6
BF16_BYTES = 2


def long_shapes():
    boundaries = {
        edge + delta
        for edge in range(4096, MAX_M + 1, 256)
        for delta in (-1, 1)
        if 4096 < edge + delta <= MAX_M
    }
    return sorted(
        {4096, 20_000, MAX_M - 1, MAX_M}
        | set(range(4224, MAX_M + 1, 128))
        | boundaries
    )


def selected_shapes(args, parser):
    if args.m is not None:
        try:
            sizes = sorted({int(value) for value in args.m.split(",")})
        except ValueError:
            parser.error("--m expects a comma-separated list of M values")
        if sizes[0] < 1 or sizes[-1] > MAX_M:
            parser.error(f"--m values must be between 1 and {MAX_M}")
        return sizes
    if args.preset == "medium":
        return list(range(256, 1025))
    if args.preset == "dense-4k":
        return list(range(1, 4097))
    if args.preset == "long-sample":
        return long_shapes()
    return list(SMOKE)


def event_intervals(call, events):
    call()  # Untimed warmup for this shape and layout.
    for start, end in events:
        start.record()
        call()
        end.record()
    events[-1][1].synchronize()
    return [start.elapsed_time(end) * 1000 for start, end in events]


def io_floor(rows, tm, tn, transpose_rows=None):
    w_bytes = BF16_BYTES * K * N
    x_bytes = BF16_BYTES * rows * K
    y_bytes = BF16_BYTES * rows * N
    hbm_bytes = w_bytes + x_bytes + y_bytes
    if transpose_rows is not None:
        # Read the original X and write its padded, transposed workspace.
        hbm_bytes += BF16_BYTES * transpose_rows * K + x_bytes
    l2_bytes = (math.ceil(rows / tm) - 1) * w_bytes + (math.ceil(N / tn) - 1) * x_bytes
    return hbm_bytes / HBM_BYTES_PER_US, l2_bytes / L2_BYTES_PER_US


def summarize(m, baseline, optimized):
    method, tm, tn, p = route(m)
    t0 = statistics.median(baseline)
    t1 = statistics.median(optimized)
    hbm, l2 = io_floor(p, tm, tn, transpose_rows=m if method == "large" else None)
    reference_hbm, reference_l2 = io_floor(m, 256, 256)
    return {
        "m": m,
        "method": method,
        "p": p,
        "tm": tm,
        "tn": tn,
        "baseline_us": t0,
        "actual_us": t1,
        "speedup": t0 / t1,
        "hbm_us": hbm,
        "l2_us": l2,
        "floor_us": hbm + l2,
        "reference_floor_us": reference_hbm + reference_l2,
        "floor_pct": 100 * (hbm + l2) / t1,
        **{f"baseline_{i}_us": v for i, v in enumerate(baseline)},
        **{f"optimized_{i}_us": v for i, v in enumerate(optimized)},
    }


@torch.no_grad()
def run(sizes, block_size, order_seed):
    torch.cuda.set_device(0)
    torch.set_num_threads(6)
    torch.manual_seed(22011)
    torch.backends.cuda.matmul.allow_tf32 = False

    # Two numerically equivalent host arrangements share one HCU allocation.
    weight = torch.randn(K * N, device="cuda:0", dtype=torch.bfloat16)
    w_nk = weight.view(N, K)
    w_kn = weight.view(K, N)
    cpu_nk = w_nk.cpu()
    cpu_kn = cpu_nk.T.contiguous()
    head = LMHead(w_kn)
    assert head.weight is w_kn and head.weight.data_ptr() == w_nk.data_ptr()

    x = torch.randn((max(sizes), K), device=weight.device, dtype=weight.dtype)
    routes = {m: route(m) for m in sizes}
    y = weight.new_empty((max(spec[3] for spec in routes.values()), N))
    max_workspace = max((p for method, _, _, p in routes.values() if method == "large"), default=0)
    work = weight.new_empty(max_workspace * K)
    weight_baseline = w_nk.T
    intervals = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(EVENTS_PER_SHAPE)
    ]

    order = sizes.copy()
    random.Random(order_seed).shuffle(order)
    measurements = {}
    for offset in range(0, len(order), block_size):
        block = order[offset:offset + block_size]
        phases = (False, True) if (offset // block_size) % 2 == 0 else (True, False)
        for use_optimized in phases:
            if use_optimized:
                w_kn.copy_(cpu_kn)
            else:
                w_nk.copy_(cpu_nk)
            torch.cuda.synchronize()

            for m in block:
                method, _, _, p = routes[m]
                activation = x[:m]
                if use_optimized:
                    output = y[:p]
                    workspace = work[:K * p].view(K, p) if method == "large" else None

                    def call():
                        return head(activation, out=output, workspace=workspace)
                else:
                    output = y[:m]

                    def call():
                        return torch.mm(activation, weight_baseline, out=output)

                measurements[use_optimized, m] = event_intervals(call, intervals)

    return [summarize(m, measurements[False, m], measurements[True, m]) for m in sizes]



@torch.no_grad()
def run_tuned(sizes):
    """Pair original, incumbent, and opt-in steady Graph timings."""
    torch.cuda.set_device(0)
    torch.manual_seed(22011)
    torch.backends.cuda.matmul.allow_tf32 = False
    w = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    original = w.T.contiguous().T
    off = LMHead(w)
    on = LMHead(w, tuned=True)
    x = w.new_empty((max(sizes), K)).normal_()
    y = w.new_empty((max(route(m, tuned=enabled)[-1] for m in sizes
                        for enabled in (False, True)), N))
    stream = torch.cuda.Stream()
    timers = {key: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for key in ("original", "off", "on")}

    def capture(call):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        return graph

    rng = random.Random(46120480)
    order = sizes.copy()
    rng.shuffle(order)
    results = {}
    for m in order:
        p_off = route(m)[-1]
        tuned = route(m, tuned=True)
        p_on = tuned[-1]
        graphs = {
            "original": capture(lambda: torch.mm(x[:m], original, out=y[:m])),
            "off": capture(lambda: off(x[:m], out=y[:p_off])),
        }
        if tuned != route(m):
            graphs["on"] = capture(lambda: on(x[:m], out=y[:p_on]))
        cases = list(graphs)
        samples = {key: [] for key in cases}
        for sample in range(11):
            rng.shuffle(cases)
            for key in cases:
                begin, end = timers[key]
                graphs[key].replay()
                begin.record()
                graphs[key].replay()
                end.record()
            timers[cases[-1]][1].synchronize()
            if sample >= 2:
                for key in cases:
                    begin, end = timers[key]
                    samples[key].append(begin.elapsed_time(end)*1000)
        times = {key: statistics.median(ts) for key, ts in samples.items()}
        times.setdefault("on", times["off"])
        results[m] = {"m": m, "tuned_method": tuned[0],
                      **{key + "_us": value for key, value in times.items()},
                      "added_speedup": times["off"] / times["on"],
                      "total_speedup": times["original"] / times["on"]}
    return [results[m] for m in sizes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", help="Comma-separated M values, overriding the preset")
    parser.add_argument("--preset", choices=("smoke", "dense-4k", "long-sample", "medium"), default="smoke")
    parser.add_argument("--csv", type=Path, help="Write medians, floors, and individual event times")
    parser.add_argument("--tuned", action="store_true", help="Run a steady Graph comparison with tuning off and on")
    args = parser.parse_args()
    sizes = selected_shapes(args, parser)
    if args.tuned:
        rows = run_tuned(sizes)
        print("M tuned_method original_us off_us on_us added_speedup total_speedup")
        for row in rows:
            print(f"{row['m']:5d} {row['tuned_method']:9s} {row['original_us']:12.3f} "
                  f"{row['off_us']:12.3f} {row['on_us']:12.3f} "
                  f"{row['added_speedup']:7.3f}x {row['total_speedup']:7.3f}x")
    else:
        long_sweep = args.preset == "long-sample" and args.m is None
        block_size = 16 if long_sweep else 32
        order_seed = 46120480 if long_sweep else 272011
        rows = run(sizes, block_size, order_seed)
        print("M method baseline_us optimized_us speedup io_floor_us floor_pct")
        for row in rows:
            print(
                f"{row['m']:5d} {row['method']:7s} {row['baseline_us']:12.3f} "
                f"{row['actual_us']:12.3f} {row['speedup']:7.3f}x "
                f"{row['floor_us']:12.3f} {row['floor_pct']:6.2f}%"
            )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
