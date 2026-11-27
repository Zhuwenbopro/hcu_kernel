"""Render measured TP4 results. Recompute every reported median from all four ranks."""
import argparse
import gzip
import os
import csv
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

METHODS=('aiter','direct','sharded')
NAMES={'aiter':'AITER 两段','direct':'直接 pull 融合','sharded':'分片归约融合'}
COLORS={'aiter':'#323b48','direct':'#007c73','sharded':'#bc5329'}
STYLE={'aiter':'-','direct':'-','sharded':'-'}

def read_stage(directory, all_sizes=False):
    cases={}
    if directory.is_file():
        with gzip.open(directory, 'rt', encoding='utf-8') as handle:
            source = [(f'{directory.name}:{i}', json.loads(line)) for i, line in enumerate(handle, 1)]
    else:
        source = ((p, json.loads(p.read_text())) for p in directory.glob('m*.json'))
    for path, record in source:
        m,k,n=record['m'],record['k'],record['rounds']
        if m in cases or min(m,k,n)<1 or m>4096: raise ValueError(f'invalid sample {path}')
        cases[m]=record
        for method, stats in record['methods'].items():
            ranks=stats['rank_us']
            if len(ranks)!=4 or any(len(r)!=n for r in ranks):
                raise ValueError(f'rank/trial shape in {path}')
            if not all(math.isfinite(v) and v>0 for r in ranks for v in r):
                raise ValueError(f'invalid times in {path}')
            if len(stats['errors'])!=4 or any(stats['errors']):
                raise ValueError(f'bitwise mismatch {path} {method} {stats["errors"]}')
            critical=list(map(max,zip(*ranks)))
            if not math.isclose(stats['us'],st.median(critical)):
                raise ValueError(f'inconsistent summary in {path}')
            stats['p25_us'],stats['p75_us']=map(float,np.percentile(critical,[25,75]))
    if all_sizes and sorted(cases)!=list(range(1,4097)):
        raise ValueError(f'expected 4096 unique measured M; found {len(cases)}')
    if not cases:raise ValueError(f'no measurements under {directory}')
    return dict(sorted(cases.items()))

def fit_cutoff(pilot):
    # Fix a threshold using only the independent pilot. Each pilot M has equal weight.
    usable={m:o['methods'] for m,o in pilot.items() if all(n in o['methods'] for n in METHODS)}
    if not usable: raise ValueError('pilot must measure both TP4 algorithms and AITER')
    def loss(c):
        return st.mean(values['direct' if m<=c else 'sharded']['us']/values['aiter']['us']
                       for m,values in usable.items())
    cutoff=min([0,*usable],key=loss)
    if cutoff==max(usable): cutoff=4096
    return cutoff,loss(cutoff),len(usable)

def export(cases,dest,cutoff):
    rows=[]
    for m,record in cases.items():
        data=record['methods']
        if 'aiter' not in data:raise ValueError(f'missing reference M={m}')
        row={'m':m,'k':record['k'],'rounds':record['rounds']}
        for name in METHODS:
            stats=data.get(name,{})
            for field in ('us','p25_us','p75_us'):
                row[name+'_'+field]=stats.get(field,'')
            for i,check in enumerate(['eager_norm','eager_residual','graph_changed_norm','graph_changed_residual']):
                row[name+'_'+check+'_mismatches']=stats['errors'][i] if stats else ''
            if name!='aiter':
                row['speedup_'+name]=data['aiter']['us']/stats['us'] if stats else ''
        if cutoff is not None:
            method='direct' if m<=cutoff else 'sharded'
            if method not in data:raise ValueError(f'missing selected {method} at M={m}')
            row.update(selected_method=method, selected_us=data[method]['us'],
                       speedup_selected=data['aiter']['us']/data[method]['us'])
        rows.append(row)
    with (dest/'by-m.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    summary={}
    for lo,hi in [(1,4096),(1,15),(16,64),(65,128),(129,256),(257,512),(513,1024),(1025,2048),(2049,4096)]:
        group=[r for r in rows if lo<=r['m']<=hi]
        res={}
        for method in ['direct','sharded','selected']:
            samples=[(r['m'],r['speedup_'+method]) for r in group if r.get('speedup_'+method,'')!='']
            if not samples:continue
            ratios=[v for _,v in samples]
            res[method]={'n':len(ratios),'wins':sum(v>1 for v in ratios),
                         'geomean':math.exp(st.mean(map(math.log,ratios))),
                         'p50':st.median(ratios),'min':min(samples,key=lambda i:i[1]),
                         'max':max(samples,key=lambda i:i[1])}
        summary[f'{lo}-{hi}']=res
    return rows,summary

def configure_font():
    font_path = os.environ.get('CJK_FONT')
    if not font_path:
        names = ['Noto Sans CJK SC', 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', 'PingFang SC']
        font_path = font_manager.findfont(font_manager.FontProperties(family=names), fallback_to_default=False)
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'font.weight':500,'axes.titleweight':500,'axes.labelweight':500,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def draw(rows,dest,cutoff,zoom_max):
    x=np.array([r['m'] for r in rows])
    for label,title in [('latency','AR + residual add + RMSNorm 延迟'),('speedup','相对 AITER 两段的加速比')]:
        fig,axes=plt.subplots(1,2,figsize=(13,5.35),layout='constrained',gridspec_kw={'width_ratios':[1.3,1]})
        fig.suptitle(title+'     |     BW47 · TP4 · BF16 [M, 5120]',fontsize=16)
        for j,ax in enumerate(axes):
            mask=x<= (4096 if j==0 else zoom_max)
            ax.set_xlabel('M（每卡输入的 token 行数）')
            ax.set_title('全范围  M=1–4096' if j==0 else f'小批量  M=1–{zoom_max}',loc='left',fontsize=12)
            ax.set_axisbelow(True)
            ax.grid(axis='y',color='#dbe0e5',lw=.7)
            ax.tick_params(labelsize=10,length=3)
            ax.spines[['bottom','left']].set_color('#a3abb2')
            ax.set_xlim(1,4096 if j==0 else zoom_max)
            if j==0:
                ax.set_xscale('log',base=2)
                ax.xaxis.set_major_locator(FixedLocator([1,4,16,64,256,1024,4096]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
            else:
                ax.set_xticks(sorted(set([1,zoom_max]+list(range(max(1,zoom_max//8),zoom_max,max(1,zoom_max//8))))))
            if cutoff is not None and 1<cutoff<max(x[mask]):
                ax.axvline(cutoff,color='#888e93',ls=':',lw=.95)
            if label=='speedup':
                ax.axhline(1,color=COLORS['aiter'],ls='--',lw=1.0,label='AITER 基线  1×')
                ax.set_ylabel('加速比（AITER 延迟 / 融合延迟）')
            else:
                ax.set_ylabel('每次操作延迟（µs）')
                if j==0:
                    ax.set_yscale('log')
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
            for name in METHODS:
                if label=='speedup' and name=='aiter':continue
                col=name+'_us' if label=='latency' else 'speedup_'+name
                vals=np.array([r[col] if r[col]!='' else np.nan for r in rows])
                if np.any(np.isfinite(vals[mask])):
                    ax.plot(x[mask],vals[mask],color=COLORS[name],lw=.95 if j==0 else 1.35,
                            ls=STYLE[name],alpha=.93,label=NAMES[name],zorder=2)
            if cutoff is not None:
                col='selected_us' if label=='latency' else 'speedup_selected'
                selected=np.array([r[col] for r in rows])
                name=(f'固定选路（M≤{cutoff} 直拉，其余分片）' if 0<cutoff<4096 else '固定选路')
                ax.plot(x[mask],selected[mask],color='#774391',lw=1.8,ls=(0,(4,3)),label=name,zorder=3)
            if j==0:ax.legend(loc='best',fontsize=9,framealpha=.96,edgecolor='#d7dde2')
        fig.savefig(dest/(label+'.png'),dpi=170)
        plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    p.add_argument('--dest',type=Path)
    p.add_argument('--zoom-max',type=int,default=32,choices=range(2,4097),metavar='M')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--pilot',type=Path)
    group.add_argument('--cutoff',type=int)
    a=p.parse_args()
    dest=a.dest or (a.stage.parent if a.stage.is_file() else a.stage)
    dest.mkdir(exist_ok=True,parents=True)
    cases=read_stage(a.stage,all_sizes=True)
    cutoff=a.cutoff
    policy={}
    if a.pilot:
        cutoff,penalty,n=fit_cutoff(read_stage(a.pilot))
        policy=dict(pilot=a.pilot.name,n=n,mean_selected_over_aiter=penalty)
    if cutoff is not None:policy['direct_through_m']=cutoff
    rows,summary=export(cases,dest,cutoff)
    (dest/'summary.json').write_text(json.dumps(dict(policy=policy,summary=summary),indent=2))
    configure_font()
    draw(rows,dest,cutoff,a.zoom_max)
    print(json.dumps(dict(shapes=len(rows),summary=summary['1-4096'],policy=policy)),flush=True)
if __name__=='__main__':main()
