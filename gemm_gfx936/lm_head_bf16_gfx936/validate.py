"""Check gfx936 BF16 LM-head dispatches against independent FP32 products."""

import argparse
from itertools import chain

import torch

from . import K, MAX_M, N, LMHead, route, workspace_rows


INT32_M = (2**31 - 1) // N
SMOKE = (1, 2, 3, 4, 15, 64, 65, 128, 129, 256, 257, 4096)
TUNED = (1, 2, 4, 5, 8, 9, 32, 33, 48, 49, 64, 65, 128, 129, 192,
         193, 255, 256, 257, 272, 273, 384, 512, 513, 640, 752, 753, 754, 768, 769,
         928, 1008, 1009, 1024, 1025, 1125, 1379, 2047, 2048, 2049)
LONG = (INT32_M, INT32_M + 1, 20_000, MAX_M - 1, MAX_M)
ATOL, RTOL = 0.04, 0.0044


def selected_shapes(args, parser):
    if args.m is None:
        return list({"smoke": SMOKE, "20k": LONG, "tuned": TUNED}[args.preset])
    try:
        sizes = sorted({int(value) for value in args.m.split(",")})
    except ValueError:
        parser.error("--m expects a comma-separated list of M values")
    if sizes[0] < 1 or sizes[-1] > MAX_M:
        parser.error(f"--m values must be between 1 and {MAX_M}")
    return sizes


def indices(size, step, extra, device):
    values = sorted({i for i in chain(range(0, size, step), extra, (size - 1,)) if 0 <= i < size})
    return torch.tensor(values, device=device, dtype=torch.long)


def error_ratio(actual, reference):
    ratio = (actual.float() - reference).abs() / (ATOL + RTOL * reference.abs())
    return ratio.nan_to_num(nan=float("inf")).max().item()


def check_accuracy(x, w, output):
    m = x.shape[0]
    rows = indices(m, 701, (1, 255, 256, 257, m // 2, m - 2, INT32_M - 1, INT32_M), x.device)
    cols = indices(N, 1241, (1, 255, 256, 257, N // 2, N - 2), x.device)
    selected_x = x.index_select(0, rows).float()
    selected_output = output.index_select(0, rows)
    worst_ratio = 0.0
    for start in range(0, N, 8192):
        end = min(start + 8192, N)
        reference = selected_x @ w[:, start:end].float()
        worst_ratio = max(worst_ratio, error_ratio(selected_output[:, start:end], reference))
    del selected_x, selected_output, reference

    selected_w = w.index_select(1, cols).float()
    for start in range(0, m, 1024):
        end = min(start + 1024, m)
        reference = x[start:end].float() @ selected_w
        actual = output[start:end].index_select(1, cols)
        worst_ratio = max(worst_ratio, error_ratio(actual, reference))

    if worst_ratio > 1.0:
        raise AssertionError(f"M={m}: sampled FP32 comparison exceeded tolerance ({worst_ratio:.5f}x)")
    for start in range(0, m, 512):
        if not torch.isfinite(output[start:start + 512]).all().item():
            raise AssertionError(f"M={m}: nonfinite output in chunk starting at row {start}")
    compared = rows.numel() * N + (m - rows.numel()) * cols.numel()
    return compared, worst_ratio


@torch.no_grad()
def run(sizes, tuned=False, graph=False):
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(4020834)

    w = torch.randn((K, N), device="cuda:0", dtype=torch.bfloat16)
    allocated = torch.cuda.memory_allocated()
    head = LMHead(w, tuned=tuned)
    if torch.cuda.memory_allocated() != allocated or head.weight is not w:
        raise AssertionError("LMHead must retain the supplied W without allocating another weight")
    x = w.new_empty((max(sizes), K)).normal_()

    for m in sizes:
        method, _, _, p = route(m, tuned=tuned)
        activation = x[:m]
        storage = None
        workspace = None
        # Exercise the legacy padded output as well as the exact-row output.
        storage_rows = (m + 15) // 16 * 16 if method == "active_x" and m == 1009 else p
        if storage_rows > m or method in ("rolling", "active_x"):
            # Start with NaN to expose unwritten output or workspace rows.
            storage = w.new_full((storage_rows, N), float("nan"))
        work_rows = workspace_rows(m, tuned=tuned)
        if work_rows:
            workspace = w.new_full((K, work_rows), float("nan"))

        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                head(activation, out=storage, workspace=workspace)
            stream.synchronize()
            replay = torch.cuda.CUDAGraph()
            with torch.cuda.graph(replay, stream=stream):
                output = head(activation, out=storage, workspace=workspace)
            replay.replay()
            torch.cuda.synchronize()
        else:
            output = head(activation, out=storage, workspace=workspace)
        if output.shape != (m, N) or output.stride() != (N, 1) or output.dtype != torch.bfloat16:
            raise AssertionError(f"M={m}: unexpected output shape, layout or dtype")
        if head.weight.data_ptr() != w.data_ptr():
            raise AssertionError(f"M={m}: W pointer changed after dispatch")
        if storage is not None:
            if output.data_ptr() != storage.data_ptr():
                raise AssertionError(f"M={m}: output does not use the supplied storage")
            if storage_rows > m and torch.count_nonzero(storage[m:]).item():
                raise AssertionError(f"M={m}: nonzero output in padding")
        if workspace is not None:
            if not torch.equal(workspace[:, :m], activation.T):
                raise AssertionError(f"M={m}: workspace contents disagree with the transposed input")
            if torch.count_nonzero(workspace[:, m:]).item():
                raise AssertionError(f"M={m}: nonzero workspace in padding")

        count, ratio = check_accuracy(activation, w, output)
        print(f"M={m:5d} {method:7s} P={p:5d} passed {count:,} FP32 checks, max error/limit {ratio:.4f}",
              flush=True)
        del output, storage, workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", help="Comma-separated M values, overriding the preset")
    parser.add_argument("--preset", choices=("smoke", "20k", "tuned"), default="smoke")
    parser.add_argument("--tuned", action="store_true", help="Use the merged TP2 dispatch")
    parser.add_argument("--graph", action="store_true", help="Validate after HIP Graph replay")
    args = parser.parse_args()
    run(selected_shapes(args, parser), tuned=args.tuned or args.preset == "tuned", graph=args.graph)


if __name__ == "__main__":
    main()
