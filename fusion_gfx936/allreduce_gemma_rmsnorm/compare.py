"""Check the published dense sweeps and render the cross-TP speedup figure."""

import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SERIES = [
    (2, "tuned_us", "TP2   grid-tuned pull", "#066e78"),
    (4, "selected_us", "TP4   direct through M=19", "#374fa2"),
    (8, "selected_us", "TP8   direct through M=12", "#b34a17"),
]


def measured_speedups(tp, latency_column):
    results = HERE / f"tp{tp}" / "results"
    path = (results / "dense" if tp == 2 else results) / "by-m.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["m"]) for row in rows] != list(range(1, 4097)):
        raise ValueError(f"{path}: expected the 4096 measured integer shapes in order")
    error_fields = ["bit_mismatches"] if tp == 2 else [
        f"{method}_{check}_mismatches"
        for method in ("aiter", "direct", "sharded")
        for check in ("eager_norm", "eager_residual", "graph_changed_norm", "graph_changed_residual")
    ]
    ratios = []
    for row in rows:
        if any(int(row[field]) != 0 for field in error_fields):
            raise ValueError(f"{path}: accuracy failure at M={row['m']}")
        baseline, fused = float(row["aiter_us"]), float(row[latency_column])
        if not all(math.isfinite(t) and t > 0 for t in (baseline, fused)):
            raise ValueError(f"{path}: invalid latency at M={row['m']}")
        ratios.append(baseline / fused)
    return ratios


def main():
    figure, (full, zoom) = plt.subplots(
        1, 2, figsize=(12, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1.55, 1]}
    )
    for tp, column, label, color in SERIES:
        ratios = measured_speedups(tp, column)
        full.plot(range(1, 4097), ratios, color=color, lw=1.4, label=label)
        zoom.plot(range(1, 33), ratios[:32], color=color, lw=1.5)
        print(f"TP{tp}   {sum(r > 1 for r in ratios)}/4096 faster   "
              f"geometric mean {statistics.geometric_mean(ratios):.6f}x")
    full.set_xscale("log", base=2)
    ticks = [1, 4, 16, 64, 256, 1024, 4096]
    full.set_xticks(ticks, labels=[str(t) for t in ticks])
    zoom.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    for axes, maximum, title in ((full, 4096, "All measured sizes"), (zoom, 32, "Small inputs")):
        axes.set_xlim(1, maximum)
        axes.set_ylim(0.9, 1.85)
        axes.set_title(title, loc="left")
        axes.set_xlabel("Token rows per rank (M)")
        axes.axhline(1, color="#333333", ls="--", lw=0.9)
        axes.grid(axis="y", color="#e2e5e9")
        axes.spines[["top", "right"]].set_visible(False)
    full.set_ylabel("Speedup over the same-TP AITER + add/RMSNorm")
    full.legend(frameon=False, loc="upper right", fontsize=9)
    figure.suptitle("BF16 AllReduce + residual add + Gemma RMSNorm  |  gfx936, H=5120")
    figure.savefig(HERE / "speedup-by-tp.png", dpi=175)
    plt.close(figure)


if __name__ == "__main__":
    main()
