from pathlib import Path
import json,csv,statistics,collections,hashlib,os,sys
R=Path(__file__).resolve().parent
data=json.loads((R/'benchmark-raw.json').read_text())
assert data['status']=='complete' and len(data['groups'])==2*len(data['points'])*len(data['seeds'])
F=['fused','current_GEMM','current_plus_aiter','current_plus_triton','same_tile_plus_triton']
points=[]
for precision in ['int8','bf16']:
    for m in data['points']:
        gs=[g for g in data['groups'] if g['precision']==precision and g['M']==m]
        assert len(gs)==3
        row={'precision':precision,'M':m,'fused_family':gs[0]['fused_family'],'current_family':gs[0]['current_family']}
        perseed={g['seed']:{f:statistics.median(s['us'][f] for s in g['samples']) for f in F} for g in gs}
        for f in F:row[f+'_us']=statistics.median(p[f] for p in perseed.values())
        best=min(['current_plus_aiter','current_plus_triton'],key=lambda k:row[k+'_us'])
        row['faster_separate_activation']=best
        row['faster_separate_us']=row[best+'_us']
        row['speedup_vs_faster_separate']=row['faster_separate_us']/row['fused_us']
        row['latency_reduction_vs_faster_separate_pct']=(1-row['fused_us']/row['faster_separate_us'])*100
        row['speedup_vs_AITER_separate']=row['current_plus_aiter_us']/row['fused_us']
        row['speedup_vs_same_tile']=row['same_tile_plus_triton_us']/row['fused_us']
        row['seed_median_us']={str(k):v for k,v in perseed.items()}
        points.append(row)
fields=[k for k in points[0] if k!='seed_median_us']
for precision in ['int8','bf16','all']:
    with (R/f'latency-{precision}.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
        w.writerows({k:v for k,v in p.items() if k in fields} for p in points if precision=='all' or p['precision']==precision)
v=json.loads((R/'validation.json').read_text())
accuracy={}
for precision in ['int8','bf16']:
    rs=[z for z in v['records'] if z['precision']==precision]
    ac={'M_cases':len(rs),'seeds':sorted({z['seed'] for z in rs}),'gate_up_current_bit_exact':all(z['gate_up_vs_current_bit_exact'] for z in rs),'guards':all(z['guards'] for z in rs)}
    for ref in ['fused_vs_AITER','fused_vs_Triton','fused_vs_FP64_activation','GU_vs_FP64']:
        mm=[z[ref] for z in rs]
        ac[ref]={'max_abs':max(z['max_abs'] for z in mm),'max_relative_L2':max(z['relative_L2'] for z in mm),'max_BF16_ULP':max(z['max_BF16_ULP'] for z in mm),'mismatches':sum(z['mismatches'] for z in mm),'elements':sum(z['elements'] for z in mm)}
        if ref in ['fused_vs_AITER','fused_vs_Triton']:
            ac[ref].update(max_BF16_ULP_gate_ge_neg80=max(z['max_BF16_ULP_gate_ge_neg80'] for z in mm),max_abs_extra_ULP_gate_lt_neg80=max(z.get('max_abs_extra_ULP',0) for z in mm))
    accuracy[precision]=ac
ranges={}
for precision in ['int8','bf16']:
    ps=[p for p in points if p['precision']==precision]
    ranges[precision]={'measured_points':len(ps),'wins_vs_faster_separate':sum(p['speedup_vs_faster_separate']>1 for p in ps),'min_speedup_vs_faster_separate':min(p['speedup_vs_faster_separate'] for p in ps),'max_speedup_vs_faster_separate':max(p['speedup_vs_faster_separate'] for p in ps),'min_reduction_pct':min(p['latency_reduction_vs_faster_separate_pct'] for p in ps),'max_reduction_pct':max(p['latency_reduction_vs_faster_separate_pct'] for p in ps),'min_speedup_same_tile':min(p['speedup_vs_same_tile'] for p in ps),'max_speedup_same_tile':max(p['speedup_vs_same_tile'] for p in ps)}
cl=[json.loads(t) for t in (R/'clocks.jsonl').read_text().splitlines()]
clocks={k:dict(collections.Counter(z.get(k,'missing') for z in cl)) for k in ['freq1_input','freq2_input','power_dpm_force_performance_level']}
res={'status':'complete','scope':{'M':[1,256],'N':17408,'K':5120,'H':8704,'weight_layout':'packed G16,U16 with PI16 inside each block','output':'BF16 [M,8704]','INT8_path':'INT8 A/W -> INT32 accumulation -> (sx*sw)*float(acc) -> BF16 RNE in registers -> FP32 SiLU*up -> BF16 RNE store','BF16_path':'BF16 A/W -> FP32 accumulation -> BF16 RNE in registers -> FP32 SiLU*up -> BF16 RNE store','activation_pack_each_call':True,'weight_prepared_once':True,'model_dispatch_modified':False},'method':{'formal_seeds':data['seeds'],'rounds':data['rounds'],'copies':data['copies'],'measured_M':data['points'],'aggregation':'median of seven rounds per seed, then median of three held-out seed medians','main_baseline':'lower aggregate median of frozen GEMM+AITER and frozen GEMM+single-kernel Triton','same_tile_control':'same packed G16/U16 W, same family and K loop; materialize BF16 gate_up, then standalone Triton SiLU*up','timed_scope':data['environment']['scope'],'timed_groups':len(data['groups']),'timed_replays_checked':sum(len(g['samples'])*len(g['samples'][0]['us']) for g in data['groups'])},'ranges':ranges,'points':points,'accuracy':accuracy,'environment':data['environment'],'clocks':clocks,'clock_samples':len(cl),'instruction_resource_audit':json.loads((R/'instruction-resource-audit.json').read_text()),'policy':json.loads((R/'fused-policy.json').read_text()),'preserved':json.loads((R/'PRESERVED.json').read_text())}
(R/'RESULT.json').write_text(json.dumps(res,indent=2)+'\n')
sys.path.insert(0,'/home/zmoe/experiments/int8-bw90-20261110/nodes/S001/plot-deps')
os.environ['MPLCONFIGDIR']=str(R/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13.4,8),sharex='col',gridspec_kw={'height_ratios':[2,1]})
lines=[('current_plus_aiter_us','Current GEMM + AITER','#888888','-'),('current_plus_triton_us','Current GEMM + Triton','#d77935','-'),('same_tile_plus_triton_us','Same tile GEMM + Triton','#9a75b5','--'),('fused_us','Fused GEMM + SiLU*up','#1673b7','-')]
handles=[]
for i,precision in enumerate(['int8','bf16']):
    ps=[p for p in points if p['precision']==precision];ms=[p['M'] for p in ps]
    for key,label,color,ls in lines:
        h,=axs[0,i].plot(ms,[p[key] for p in ps],label=label,color=color,ls=ls,lw=1.6,marker='o',ms=2.5)
        if i==0:handles.append(h)
    axs[0,i].set_title(('INT8 x INT8' if precision=='int8' else 'BF16 x BF16')+' -> BF16',fontsize=13)
    axs[0,i].set_ylabel('GPU latency per call (us)')
    axs[1,i].plot(ms,[p['speedup_vs_faster_separate'] for p in ps],color='#1673b7',lw=1.6,marker='o',ms=2.5,label='Faster current separate / fused')
    axs[1,i].plot(ms,[p['speedup_vs_same_tile'] for p in ps],color='#9a75b5',ls='--',lw=1.4,label='Same tile separate / fused')
    axs[1,i].axhline(1,color='#777',ls=':',lw=1)
    axs[1,i].set_ylabel('Speedup');axs[1,i].set_xlabel('M (token rows)')
    for j in range(2):
        axs[j,i].grid(alpha=.18);axs[j,i].set_xlim(1,256)
    axs[1,i].set_xticks([1,32,64,96,128,160,192,224,256])
axs[1,0].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('Packed gate/up GEMM fused with SiLU(gate) * up',fontsize=16,y=.984)
fig.text(.5,.943,'N=17408, K=5120, output H=8704 | BW47 GPU5 | 3 seeds x 7 rounds x 72 weights | 33 measured M values',ha='center',fontsize=10,color='#444')
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.919),ncol=4,frameon=False,fontsize=9)
fig.tight_layout(rect=[.01,.015,.995,.875])
fig.savefig(R/'fused-gate-up.png',dpi=180);fig.savefig(R/'fused-gate-up.svg');plt.close(fig)
print(json.dumps({'ranges':ranges,'accuracy':accuracy,'clocks':clocks,'points':[{k:v for k,v in p.items() if k!='seed_median_us'} for p in points if p['M'] in [1,2,16,32,64,128,129,192,256]]},indent=2))
