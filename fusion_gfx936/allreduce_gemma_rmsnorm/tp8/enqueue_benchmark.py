import argparse,json,os,random,re,time
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('stage',choices=['smoke','pilot','confirm','full','stop'])
p.add_argument('--run',type=Path)
p.add_argument('--tag')
p.add_argument('--sizes')
p.add_argument('--methods',default=None)
p.add_argument('--k',type=int)
p.add_argument('--rounds',type=int)
p.add_argument('--cutoff',type=int)
p.add_argument('--blocks',type=int,default=64)
a=p.parse_args()
run=a.run or Path((root/'active-run.txt').read_text().strip())
tag=a.tag or a.stage
if not re.fullmatch('[a-zA-Z0-9_-]+',tag):p.error('invalid tag')
if a.stage=='stop':
    task={'stage':'stop'}
else:
    if a.stage=='smoke':
        sizes=[1,6,7,8,9,15,16,17,31,65,129,4096];k,rounds=2,3
    elif a.stage=='pilot':
        sizes=list(range(1,33))+[40,48,56,63,64,65,72,80,96,112,127,128,129,144,160,192,224,255,256,257,320,384,448,511,512,513,640,768,896,1023,1024,1025,1280,1536,1792,2047,2048,2049,2560,3072,3584,4095,4096];k,rounds=8,9
    elif a.stage=='confirm':
        sizes=[1,4,7,8,9,15,16,17,32,64,128,256,512,1024,2048,4096];k,rounds=12,15
    else:
        sizes=list(range(1,4097));k,rounds=8,9
    if a.sizes:
        sizes=[]
        for part in a.sizes.split(','):
            if '-' in part:
                l,h=map(int,part.split('-'));sizes.extend(range(l,h+1))
            else:sizes.append(int(part))
        sizes=list(dict.fromkeys(sizes))
    if not sizes or min(sizes)<1 or max(sizes)>4096:p.error('invalid M')
    if a.stage=='full':
        random.Random(13479).shuffle(sizes)
    methods=a.methods.split(',') if a.methods else (['aiter','selected'] if a.cutoff is not None else ['aiter','direct','sharded'])
    if not set(methods)<=set(['direct','sharded','aiter','selected']) or len(methods)!=len(set(methods)):p.error('invalid method')
    if 'selected' in methods and (a.cutoff is None or 'sharded' in methods or 'direct' in methods):p.error('selected requires --cutoff and no overlapping method')
    task=dict(stage=tag,sizes=sizes,k=a.k or k,rounds=a.rounds or rounds,methods=methods,blocks=a.blocks,cutoff=a.cutoff)
    if min(task['k'],task['rounds'],task['blocks'])<1:p.error('k,rounds,blocks must be positive')
job=run/'jobs'/f'{time.time_ns()}-{tag}.json'
tmp=job.with_suffix('.tmp')
tmp.write_text(json.dumps(task))
os.replace(tmp,job)
print(job,flush=True)
