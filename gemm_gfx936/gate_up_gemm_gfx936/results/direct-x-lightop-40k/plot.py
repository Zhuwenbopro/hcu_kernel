"""Redraw the fixed-clock gfx936 GateUp comparison. Requires NumPy and Matplotlib."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter


HERE = Path(__file__).resolve().parent
METHODS = ("direct", "lightop")
COLORS = ("#005564", "#C95118")
N, K = 17408, 5120


def load_series():
    # genfromtxt transparently decompresses the sibling gzip file.
    rows = np.genfromtxt(HERE / "per-m.csv.gz", delimiter=",", names=True)
    m = rows["M"]
    if not np.array_equal(m, np.arange(1, 40961)):
        raise ValueError("Expected ordered measurements for every M from 1 through 40960")
    x, w, y = m * K, N * K, 2 * m * N
    hbm_s = (x + w + y) / 1.5e12
    compute_s = (2 * m * N * K) / 784e12
    series = {"M": m}
    for name in METHODS:
        us = rows[f"{name}_us"]
        tm, tn = (rows[f"{name}_tile_{axis}"] for axis in ("m", "n"))
        if not all(np.all(np.isfinite(v) & (v > 0)) for v in (us, tm, tn)):
            raise ValueError(f"Expected finite positive {name} timings and tile sizes")
        if not all(np.array_equal(v, np.floor(v)) for v in (tm, tn)):
            raise ValueError(f"Expected integer {name} tile sizes")
        bm, bn = np.ceil(m / tm), np.ceil(N / tn)
        l2_s = ((bm - 1) * w + (bn - 1) * x) / 6e12
        series[name] = dict(us=us, bw=(hbm_s + l2_s) / (us * 1e-6),
                            compute=compute_s / (us * 1e-6))
    return series


def draw(series, kind, output_dir):
    latency = kind == "latency"
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.6),
                             gridspec_kw={"width_ratios": [1, 2.8]})
    fig.subplots_adjust(left=.065, right=.92, top=.79, bottom=.16, hspace=.26, wspace=.30)
    fig.text(.066, .938, "GPU latency" if latency else "Direct X speedup over LightOP",
             size=24, weight="bold", color="#172F39")
    fig.text(.067, .892, "gfx936   /   TP2 INT8 GateUp GEMM   /   N = 17,408, K = 5,120"
             "   /   Core 1,499 MHz, HBM 1,800 MHz", size=11, color="#405565")
    speed_color = "#594084"
    signals = list(zip(COLORS, ("Direct X", "LightOP W8A8 ASM"))) if latency else [
        (speed_color, "LightOP time / Direct X time")]
    handles = [Line2D([], [], lw=2.3, color=c, label=label) for c, label in signals]
    handles.extend(Line2D([], [], lw=1.5, ls=ls, color="#667F8B", label=label)
                   for ls, label in [("-", "Direct X estimate"), ("--", "LightOP estimate")])
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.93, .875),
               ncol=len(handles), frameon=False, fontsize=10)

    estimates = [("bw", "Modeled bandwidth time / GPU time", "#5687A0", "#CEE4F0"),
                 ("compute", "INT8 compute utilization", "#67856E", "#D4E4D5")]
    for j, count in enumerate((256, len(series["M"]))):
        m = series["M"][:count]
        group = 1 if j == 0 else 128
        center = m.reshape(-1, group).mean(axis=1)
        d, l = (series[name]["us"][:count] for name in METHODS)
        scale = 1000 if j == 1 else 1
        measured = [d / scale, l / scale] if latency else [l / d]
        for i, (metric, label, edge, fill) in enumerate(estimates):
            ax = axes[i, j]
            back = ax.twinx()
            ax.set_zorder(back.get_zorder() + 1)
            ax.patch.set_visible(False)
            pct = [series[name][metric][:count] for name in METHODS]
            medians = [np.median(v.reshape(-1, group), axis=1) for v in pct]
            bottom, top = np.minimum(*medians), np.maximum(*medians)
            back.fill_between(center, 0, bottom, color=fill, alpha=.45, lw=0)
            back.fill_between(center, bottom, top, color=fill, alpha=.86, lw=0)
            for values, med, style in zip(pct, medians, ("-", "--")):
                if group > 1:
                    back.plot(m, values, color=edge, alpha=.24, lw=.4)
                back.plot(center, med, ls=style, color=edge, lw=1.3)
            ymax = max(1.0, np.ceil(np.max(pct) * 4) / 4)
            back.set_ylim(0, ymax)
            back.set_yticks(np.arange(0, ymax + .01, .25))
            back.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            back.set_ylabel(label, color=edge)
            back.tick_params(axis="y", colors=edge, labelsize=9)
            if metric == "bw":
                back.axhline(1, color=edge, lw=.8, ls=":")
            for values, (color, _) in zip(measured, signals):
                ax.plot(m, values, color=color, lw=1.1 if group == 1 else .48,
                        alpha=1 if group == 1 else .58)
                if group > 1:
                    ax.plot(center, np.median(values.reshape(-1, group), axis=1),
                            color=color, lw=1.6)
            if latency:
                ax.set_ylabel("GPU latency (ms)" if scale == 1000 else "GPU latency (µs)")
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylabel("LightOP time / Direct X time")
                ax.axhline(1, color="#56616B", lw=.8, ls="--")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
            ax.grid(axis="y", lw=.6, color="#C8D3D8", alpha=.6)
            ax.set_xlim(1, count)
            ax.set_xticks([1, 64, 128, 192, 256] if j == 0 else
                          [1, 4096, 10240, 20480, 30720, 40960])
            if i == 0:
                ax.set_title("Small M  ·  1–256" if j == 0 else "All 40,960 measured shapes",
                             loc="left", pad=10, weight="bold", size=12)
            else:
                ax.set_xlabel("M (input rows)")
            ax.spines[["right", "top"]].set_visible(False)
            back.spines[["left", "top", "bottom"]].set_visible(False)
    fig.text(.065, .080, "Full-range panels show every measured point. Bold lines and shaded "
             "boundaries show medians in consecutive, nonoverlapping groups of 128 M values.", size=10)
    fig.text(.065, .050, "HBM (1.5 TB/s) counts X, W and one BF16 output write. L2 (6 TB/s) counts "
             "W rereads across M tiles and X rereads across N tiles. INT8 peak: 784 TOPS.", size=10)
    fig.savefig(output_dir / f"{kind}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "redrawn",
                        help="Output directory (default: redrawn beside the published charts)")
    args = parser.parse_args()
    series = load_series()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("latency", "speedup"):
        draw(series, kind, args.output_dir)


if __name__ == "__main__":
    main()
