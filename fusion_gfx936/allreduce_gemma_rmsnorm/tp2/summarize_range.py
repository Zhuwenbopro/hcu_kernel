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


out=Path(sys.argv[1])
samples=collections.defaultdict(dict)
for r in csv.DictReader(open_csv(out / 'range-raw.csv')):
    key=(int(r['m']),r['method'])
    trial=int(r['trial'])
    assert trial not in samples[key]
    t=max(float(r['rank0']),float(r['rank1']))
    assert math.isfinite(t) and t>0
    samples[key][trial]=t
by_m=collections.defaultdict(dict)
mismatch=0
for r in csv.DictReader((out/'range.csv').open()):
    m=int(r['m']); key=(m,r['method'])
    assert set(samples[key])==set(range(15))
    assert int(r['rounds'])==15 and int(r['k'])==8
    assert statistics.median(samples[key].values())==float(r['us'])
    assert r['method'] not in by_m[m]
    by_m[m][r['method']]=float(r['us'])
    mismatch+=int(r['mismatch'])
assert len(by_m)==521 and all(set(rr)=={'original','65-512','four_windows'} for rr in by_m.values())
assert all(m in by_m for m in range(1,515))
def windows(m):
    return 65<=m<=96 or 129<=m<=160 or 193<=m<=216 or 257<=m<=272
def summarize(ms):
    res={}
    for name,k0,k1 in [('wide_vs_old','original','65-512'),('narrow_vs_old','original','four_windows'),('narrow_vs_wide','65-512','four_windows')]:
        dv=[by_m[m][k0]-by_m[m][k1] for m in ms]
        res[name]=dict(n=len(dv), median_saved_us=statistics.median(dv),mean_saved_us=statistics.mean(dv),
                       faster=sum(x>0 for x in dv),tie=dv.count(0),slower=sum(x<0 for x in dv))
    return res
ms=list(range(65,513))
stats={k:summarize(g) for k,g in [('all_middle',ms),('calibrated',[m for m in ms if windows(m)]),('extra',[m for m in ms if not windows(m)]),('low',list(range(1,65))),('high',[m for m in sorted(by_m) if m>=513])]}
mixed={k:json.loads((out/(k+'-mixed.json')).read_text()) for k in ('proposed','windows')}
assert not mismatch and all(not r['mismatches'] for job in mixed.values() for r in job)
summary={'shapes':len(by_m),'mismatches':mismatch,'stats':stats,'mixed':mixed,'points':{m:by_m[m] for m in (16,64,65,96,128,129,257,272,273,511,512,513,1024,4096)}}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
