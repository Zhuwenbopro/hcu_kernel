"""Validate both rank samples and turn the dense run into one row per integer M."""
import collections
import csv
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

def open_csv(path):
    return path.open(newline="") if path.exists() else gzip.open(str(path) + ".gz", "rt", newline="")


run = Path(sys.argv[1])
rows = collections.defaultdict(dict)
raw = collections.defaultdict(dict)
with open_csv(run / 'full-raw.csv') as f:
    for row in csv.DictReader(f):
        point = int(row['m']), row['method']
        t = int(row['trial'])
        assert t not in raw[point]
        delays = float(row['rank0']), float(row['rank1'])
        assert all(math.isfinite(x) and x > 0 for x in delays), (point, t)
        raw[point][t] = max(delays)
with open_csv(run / 'full.csv') as f:
    for row in csv.DictReader(f):
        m, key = int(row['m']), row['method']
        assert key not in rows[m]
        samples = raw[m, key]
        assert set(samples) == set(range(9)) and int(row['k']) == 8 and int(row['rounds']) == 9
        assert abs(float(row['us']) - statistics.median(samples.values())) < 1e-8
        assert int(row['mismatch']) == sum(int(row[k]) for k in ('eager_norm','eager_residual','replay_norm','replay_residual'))
        rows[m][key] = row
assert set(rows) == set(range(1,4097))
assert len(raw) == 4096*3 + 128
out = []
def cap(m):
    if 65<=m<=96: return 128
    if 129<=m<=160: return 80
    if 193<=m<=216: return 72
    if 257<=m<=272: return 96
    return 64
for m in sorted(rows):
    rr = rows[m]
    methods = ['original','tuned','aiter'] + (['decode'] if m<=128 else [])
    assert set(rr)==set(methods), m
    us = {k:float(rr[k]['us']) for k in methods}
    out.append(dict(m=m, bytes_per_rank=m*10240, tuned_grid_cap=cap(m),
        original_us=us['original'], tuned_us=us['tuned'], aiter_us=us['aiter'], decode_us=us.get('decode',''),
        aiter_over_tuned=us['aiter']/us['tuned'],original_over_tuned=us['original']/us['tuned'],
        aiter_over_original=us['aiter']/us['original'],
        original_over_decode=us['original']/us['decode'] if m<=128 else '',
        bit_mismatches=sum(int(rr[k]['mismatch']) for k in methods)))
with (run/'by-m.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
def stats(group):
    ratios = [r['aiter_over_tuned'] for r in group]
    rel = [r['original_over_tuned'] for r in group]
    diff = [r['original_us']-r['tuned_us'] for r in group]
    return dict(n=len(group),aiter_median=statistics.median(ratios),aiter_min=min(ratios),aiter_max=max(ratios),aiter_wins=sum(v>1 for v in ratios),
        relative_median=statistics.median(rel),relative_min=min(rel),relative_max=max(rel),
        saved_us_median=statistics.median(diff),original_wins=sum(v>1 for v in rel),original_ties=sum(v==1 for v in rel))
groups={'all':out,'selected_windows':[r for r in out if cap(r['m'])!=64],'default_windows':[r for r in out if cap(r['m'])==64]}
for lo,hi in [(1,64),(65,96),(129,160),(193,216),(257,272),(273,4096),(2049,4096)]:
    groups[f'{lo}-{hi}']=[r for r in out if lo<=r['m']<=hi]
def median_small(lo,hi):
    vals=[r for r in out if lo<=r['m']<=hi]
    return dict(median_saved=statistics.median(r['original_us']-r['decode_us'] for r in vals))
summary=dict(n_raw=sum(len(v) for v in raw.values()),mismatches=sum(r['bit_mismatches'] for r in out),
    groups={k:stats(v) for k,v in groups.items()},small_decode={f'{a}-{b}':median_small(a,b) for a,b in [(1,64),(65,96),(97,128)]},
    points={r['m']:[round(r[k],4) for k in ('original_us','tuned_us','aiter_us','original_over_tuned','aiter_over_tuned')]
        for r in out if r['m'] in (1,16,32,64,65,68,80,96,128,129,160,193,256,257,272,512,1024,2048,4096)})
(run/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
