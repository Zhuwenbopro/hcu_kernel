"""Cold-pool HIP Graph benchmark; reports useful total IO and auxiliary W throughput."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import statistics

import torch

try:
    from .python_api import VARIANTS, choose_variant, gemm_packed, kernel_info, pack_activation, pack_weight
    from .validation_utils import SENTINEL, cpu_check, guarded_output, guards_ok, reference
except ImportError:
    from python_api import VARIANTS, choose_variant, gemm_packed, kernel_info, pack_activation, pack_weight
    from validation_utils import SENTINEL, cpu_check, guarded_output, guards_ok, reference

ROOT = Path(__file__).resolve().parent

def batch_sizes(text):
    if ":" in text:
        a, b = map(int, text.split(":"))
        if a < 1 or b < a:
            raise ValueError("batch range must be positive and increasing")
        return list(range(a, b + 1))
    values = [int(v) for v in text.split(",")]
    if not values or min(values) < 1:
        raise ValueError("batch sizes must be positive")
    return values

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", default="32,128")
    p.add_argument("--n", type=int, nargs="+", default=[34816, 17408])
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["E023", "E030", "E038", "E047"])
    p.add_argument("--pool-gib", type=float, default=6)
    p.add_argument("--graph-calls", type=int, default=0, help="0: one full W pool; otherwise at least the pool copy count")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=202611103)
    p.add_argument("--reference-gbs", type=float, default=1363.5)
    p.add_argument("--output", default=str(ROOT / "build" / "benchmark.json"))
    a = p.parse_args()
    batches = batch_sizes(a.batches)
    if a.rounds < 1 or a.warmup < 1 or a.pool_gib <= 0 or a.reference_gbs <= 0 or a.k < 64 or a.k % 64:
        p.error("invalid rounds/warmup/pool/reference/K")
    if len(a.variants) != len(set(a.variants)):
        p.error("duplicate variants")
    torch.set_num_threads(8)
    props = torch.cuda.get_device_properties(0)
    assert "gfx936" in props.gcnArchName
    for variant in a.variants:
        info = kernel_info(variant, 0)
        assert all(n > 0 and n % info[4] == 0 for n in a.n), (variant, info[4], a.n)
    records = []
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    def save(status):
        data = {"status": status, "config": vars(a), "records": records,
                "environment": {"gpu": props.name, "arch": props.gcnArchName, "cu": props.multi_processor_count,
                                "torch": torch.__version__, "hip": torch.version.hip},
                "metric": "useful bytes MK+NK+4MN / time; copy reference is not a physical HBM peak",
                "physical_HBM_counters_measured": False,
                "scope": "prepacked INT8 X/W -> INT32; packing/scales/dequantization excluded"}
        out.write_text(json.dumps(data, indent=2) + "\n")
    try:
        save("running")
        for n in a.n:
            copies = max(2, math.floor(a.pool_gib * (1 << 30) / (n * a.k)))
            needed = 2 * copies * n * a.k + 2 * (1 << 30)
            assert torch.cuda.mem_get_info()[0] > needed, "not enough free memory for the cold W pool"
            torch.manual_seed(a.seed + n)
            raw = [torch.randint(-128, 128, (n, a.k), device="cuda", dtype=torch.int8) for _ in range(copies)]
            weights = [pack_weight(w) for w in raw]
            calls = a.graph_calls or copies
            if calls < copies:
                raise ValueError("graph-calls must cover the entire weight pool")
            sequence = np.random.default_rng(a.seed + n).permutation(copies).tolist()
            order = [sequence[i % copies] for i in range(calls)]
            for m in batches:
                torch.manual_seed(a.seed + m)
                x = torch.randint(-128, 128, (m, a.k), device="cuda", dtype=torch.int8)
                packed = {v: pack_activation(x, v) for v in a.variants}
                backing, y = guarded_output(m, n, x.device)
                wi = (m * 7) % copies
                expected = reference(x, raw[wi])
                final = reference(x, raw[order[-1]])
                cpu_bad, _ = cpu_check(x, raw[wi], expected)
                cpu_bad_final, _ = cpu_check(x, raw[order[-1]], final)
                assert cpu_bad == cpu_bad_final == 0
                for v in a.variants:
                    backing.fill_(SENTINEL)
                    gemm_packed(packed[v], weights[wi], v, out=y)
                    assert torch.equal(y, expected) and guards_ok(backing), (m, n, v)
                graphs = {}
                for v in a.variants:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for wi in order:
                            gemm_packed(packed[v], weights[wi], v, out=y)
                    graphs[v] = graph
                for _ in range(a.warmup):
                    for graph in graphs.values():
                        graph.replay()
                torch.cuda.synchronize()
                samples = {v: [] for v in a.variants}
                for rd in range(a.rounds):
                    ids = a.variants[:]; shift = (m + rd) % len(ids); ids = ids[shift:] + ids[:shift]
                    for v in ids:
                        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                        start.record(); graphs[v].replay(); end.record(); end.synchronize()
                        us = start.elapsed_time(end) * 1000 / calls
                        assert torch.equal(y, final) and guards_ok(backing), (m, n, v, rd)
                        samples[v].append(us)
                variants = {}
                for v, samples_us in samples.items():
                    us = statistics.median(samples_us)
                    useful = (m * a.k + n * a.k + 4 * m * n) / us / 1000
                    variants[v] = {"median_us": us, "min_us": min(samples_us), "max_us": max(samples_us),
                                   "samples_us": samples_us, "logical_total_GBs": useful,
                                   "total_effective_io_pct": useful / a.reference_gbs * 100,
                                   "weight_GBs": n * a.k / us / 1000,
                                   "weight_util_pct": n * a.k / us / 1000 / a.reference_gbs * 100,
                                   "TOPs": 2 * m * n * a.k / us / 1e6}
                winner = min(variants, key=lambda v: variants[v]["median_us"])
                rec = {"M": m, "N": n, "K": a.k, "copies": copies, "pool_bytes": copies * n * a.k,
                       "graph_calls": calls, "winner": winner, "variants": variants, "correct": True,
                       "guards": True, "within_one_percent": [v for v in variants if variants[v]["median_us"] <= variants[winner]["median_us"] * 1.01]}
                records.append(rec); save("running")
                print(json.dumps({"M": m, "N": n, "winner": winner, **variants[winner]}), flush=True)
                del graphs, graph, x, packed, backing, y, expected, final
            del weights, raw
            torch.cuda.empty_cache()
        save("complete")
        with out.with_suffix(".csv").open("w", newline="") as f:
            fields = ["M", "N", "K", "variant", "median_us", "total_effective_io_pct", "weight_util_pct", "TOPs"]
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for r in records:
                for v, stats in r["variants"].items():
                    writer.writerow({**{k:r[k] for k in ("M", "N", "K")}, "variant": v, **{k:stats[k] for k in fields if k in stats}})
    except Exception:
        save("failed")
        raise

if __name__ == "__main__":
    main()
