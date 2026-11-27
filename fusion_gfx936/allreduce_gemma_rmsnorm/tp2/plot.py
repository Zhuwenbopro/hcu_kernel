#!/usr/bin/env python3
"""Render BW47 TP2 all-size latency and speedup using only measured samples."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np


def load_data(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        required = ['m', 'original_us', 'tuned_us', 'aiter_us']
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {required}')
        rows = sorted(reader, key=lambda r: int(r['m']))
        has_decode = 'decode_us' in reader.fieldnames
    if [int(r['m']) for r in rows] != list(range(1, 4097)):
        raise ValueError('Exactly one measured sample per integer M from 1 to 4096 is required')
    data = {k: np.array([float(r[k]) for r in rows]) for k in required}
    if any(not np.all(np.isfinite(arr) & (arr > 0)) for arr in data.values()):
        raise ValueError('Expected all positive finite values')
    if has_decode:
        arr = np.array([float(r['decode_us']) if r['decode_us'].strip() else np.nan for r in rows])
        if not np.all(np.isfinite(arr[:128]) & (arr[:128] > 0)) or not np.all(np.isnan(arr[128:])):
            raise ValueError('The decode-only path must contain exactly M=1..128')
        data['decode_us'] = arr
    return data


def configure_font(tmp, source):
    # The server's CJK font defaults to a very thin variable-font axis.
    # Use a regular-weight subset for the report and avoid writing a font cache.
    if not source:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans SC', 'Microsoft YaHei', 'DejaVu Sans']
        return
    # Matplotlib holds open the registered font on Windows. Register the
    # user-supplied file there so the temporary directory can be removed.
    target = source
    if os.name != 'nt':
        from fontTools import subset
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
        with TTFont(source) as font:
            subsetter = subset.Subsetter()
            subsetter.populate(text=Path(__file__).read_text(encoding='utf-8') + ''.join(chr(i) for i in range(32, 127)))
            subsetter.subset(font)
            if 'fvar' in font:
                instantiateVariableFont(font, {'wght': 500}, inplace=True)
            target = Path(tmp) / 'noto-medium.ttf'
            font.save(target)
    font_manager.fontManager.addfont(str(target))
    name = font_manager.FontProperties(fname=str(target)).get_name()
    plt.rcParams.update({'font.family': [name, 'DejaVu Sans'], 'font.size': 11,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})


def axes_pair(title):
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                                     gridspec_kw={'width_ratios': [1.15, 1]},
                                     layout='constrained')
    fig.suptitle(title + '    |    BW47 · TP2 · BF16 [M, 5120]', fontsize=17)
    for ax in (full, zoom):
        ax.grid(axis='y', color='#d8dfe6', lw=.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10, length=3)
        ax.spines[['bottom','left']].set_color('#9aa4af')
        ax.set_xlabel('M（本次算子的 token 行数）')
    full.set_title('全范围  M=1–4096', loc='left', fontsize=12)
    full.set_xscale('log', base=2)
    full.set_xlim(1,4096)
    full.xaxis.set_major_locator(FixedLocator([1,4,16,64,256,1024,4096]))
    full.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    zoom.set_title('调优区间放大  M=48–280', loc='left', fontsize=12)
    zoom.set_xlim(48,280)
    zoom.set_xticks([48,64,96,128,160,192,224,256,280])
    for b in (64,128,192,256):
        zoom.axvline(b,color='#a3aab2',ls=':',lw=.8,zorder=0)
    return fig, full, zoom


def legend(ax, loc):
    ax.legend(loc=loc, framealpha=.96, edgecolor='#d6dee3',
              fontsize=9, handlelength=2.3, handletextpad=.5, borderpad=.5)


def render(data, dest):
    x = data['m']
    original, tuned, aiter = (data[k] for k in ('original_us','tuned_us','aiter_us'))
    m = (x >=48) & (x<=280)
    indigo, green, orange, plum = '#345a92', '#007a5b', '#b5551b', '#9c3e7d'

    fig, full, zoom = axes_pair('AR + residual add + RMSNorm 延迟')
    for ax, mask in ((full, np.ones(x.size,dtype=bool)), (zoom,m)):
        ax.plot(x[mask], aiter[mask], color=orange, lw=1.3, label='AITER 两段（含输入拷贝）')
        ax.plot(x[mask], tuned[mask], color=green, lw=1.85, label='调优融合（单通信器）')
        ax.plot(x[mask], original[mask], color=indigo, ls=(0,(3,2.2)), lw=1.15, label='原融合（64 块上限）')
        ax.set_ylabel('延迟（µs）')
        if ax is zoom and 'decode_us' in data:
            ax.plot(x[mask],data['decode_us'][mask],color=plum,ls=(0,(5,2)),lw=1.4,
                    label='decode 双通信器（M≤128）')
    full.set_yscale('log')
    full.yaxis.set_major_locator(FixedLocator([10,20,50,100,200,500]))
    full.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f'{y:g}'))
    legend(full,'upper left')
    if 'decode_us' in data:
        legend(zoom,'upper left')
    fig.savefig(dest/'latency.png', dpi=155)
    plt.close(fig)

    fig, full, zoom = axes_pair('融合加速比')
    full.plot(x, aiter/original, color=indigo, ls=(0,(3,2.2)), lw=1.2, label='AITER / 原融合')
    full.plot(x, aiter/tuned, color=green, lw=1.25, label='AITER / 调优融合')
    full.plot(x, original/tuned, color='#633b82', lw=1.05, label='原融合 / 调优融合')
    zoom.plot(x[m], (original/tuned)[m], color='#633b82', lw=1.65, label='原融合 / 调优融合')
    if 'decode_us' in data:
        zoom.plot(x[m], (original/data['decode_us'])[m],color=plum,ls=(0,(5,2)),lw=1.5,
                  label='原融合 / decode 双通信器')
    for ax in (full,zoom):
        ax.axhline(1, color='#484e55', ls='--', lw=.95)
        ax.yaxis.set_major_locator(MaxNLocator(8))
        ax.set_ylabel('加速比（×，越高越快）')
    legend(full,'best')
    legend(zoom,'upper right')
    fig.savefig(dest/'speedup.png', dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--font', type=Path, help='optional CJK TTF file')
    args = parser.parse_args()
    data = load_data(args.csv)
    out = args.out or args.csv.resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        configure_font(tmp, args.font)
        render(data, out)


if __name__ == '__main__':
    main()
