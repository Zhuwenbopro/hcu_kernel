from test_common import *
COPIES=72
ROUNDS=5
POINTS=[1,2,4,8,10,12,16,32,64,96,128,129,144,160,176,192,193,208,224,240,256]
def pool(precision,seed):
    torch.manual_seed(seed)
    if precision=='int8':
        x=torch.randint(-127,128,(256,fg.K),device='cuda',dtype=torch.int8)
        sx=torch.rand((256,1),device='cuda')*.008+.004
    else:
        x=torch.randn((256,fg.K),device='cuda',dtype=torch.bfloat16);sx=None
    weights=[];old=[];new=[];scales=[]
    for _ in range(COPIES):
        w=torch.randint(-127,128,(fg.N,fg.K),device='cuda',dtype=torch.int8) if precision=='int8' else torch.randn((fg.N,fg.K),device='cuda',dtype=torch.bfloat16)
        b=w.t()
        sw=torch.rand((fg.N,1),device='cuda')*.006+.002 if precision=='int8' else None
        weights.append(b);scales.append(sw)
        old.append(prepared_baseline(precision,b));new.append(fg.prepare_weight(b,sw))
    gu=baseline_gemm(precision,x,weights[-1],old[-1],sx,scales[-1])
    ref=activation(gu)
    torch.cuda.synchronize()
    return x,sx,weights,scales,old,new,ref
def capture(fn):
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):ys=fn()
    return g,ys
def measure(g):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize()
    return a.elapsed_time(b)*1000/COPIES
def eligible(e,m,precision,f):
    k=e.kernel(m,precision,f)
    if k.rec.get('fixed_M'):return m==k.rec['fixed_M']
    return math.ceil(m/k.bm)*k.bm<=k.mp_for(m)
def main():
    assert json.loads((R/'smoke.json').read_text())['status']=='complete'
    e=fg.engine();groups=[]
    prior=R/'screen-result.json'
    if prior.exists() and '--fresh' not in sys.argv:groups=json.loads(prior.read_text())['groups']
    completed={(g['precision'],g['M']) for g in groups}
    for pi,precision in enumerate(['int8','bf16']):
        seed=202611153021+pi
        if all((precision,m) in completed for m in POINTS):continue
        x,sx,weights,scales,old,new,ref=pool(precision,seed)
        fams=sorted({z['base_family'] for z in e.records if z['input_dtype']==precision and not (precision=='bf16' and z['base_family']=='E030')})
        print('POOL_READY',precision,flush=True)
        for m in POINTS:
            if (precision,m) in completed:continue
            a=x[:m];ss=None if sx is None else sx[:m];graphs={};outputs={}
            families=[f for f in fams if eligible(e,m,precision,f)]
            for f in families:tmp=e.mm(a,new[0],ss,family=f)
            for kind in ['aiter','triton']:tmp=baseline(precision,a,weights[0],old[0],ss,scales[0],kind)
            torch.cuda.synchronize();del tmp
            for f in families:
                graphs['fused:'+f],outputs['fused:'+f]=capture(lambda f=f:[e.mm(a,p,ss,family=f) for p in new])
            for kind in ['aiter','triton']:
                graphs['baseline:'+kind],outputs['baseline:'+kind]=capture(lambda kind=kind:[baseline(precision,a,b,p,ss,sw,kind) for b,p,sw in zip(weights,old,scales)])
            for g in graphs.values():g.replay()
            torch.cuda.synchronize()
            for f,ys in outputs.items():close(ys[-1],ref[:m],{'screen':precision,'M':m,'family':f})
            group={'precision':precision,'seed':seed,'M':m,'samples':[]}
            rng=random.Random(seed+m)
            for rnd in range(ROUNDS):
                order=list(graphs);rng.shuffle(order);row={'round':rnd,'order':order,'us':{}}
                for f in order:
                    row['us'][f]=measure(graphs[f]);close(outputs[f][-1],ref[:m],{'screen':precision,'M':m,'family':f})
                group['samples'].append(row)
            med={f:statistics.median(s['us'][f] for s in group['samples']) for f in graphs}
            best=min((f for f in med if f.startswith('fused:')),key=med.get)
            group.update(median_us=med,selected=best,accuracy={f:comparison(ys[-1],ref[:m]) for f,ys in outputs.items()})
            groups.append(group)
            save('screen-result.json',{'status':'running','groups':groups,'copies':COPIES,'rounds':ROUNDS})
            save('screen-status.json',{'status':'running','precision':precision,'M':m,'groups':len(groups),'best':best,'us':med[best],'baseline_aiter':med['baseline:aiter'],'baseline_triton':med['baseline:triton']})
            print('SCREEN',precision,m,best,round(med[best],3),'AITER',round(med['baseline:aiter'],3),'TRITON',round(med['baseline:triton'],3),flush=True)
            del graphs,outputs,g,ys
            gc.collect();torch.cuda.empty_cache()
        del x,sx,weights,scales,old,new,ref
        gc.collect();torch.cuda.empty_cache()
    save('screen-result.json',{'status':'complete','groups':groups,'copies':COPIES,'rounds':ROUNDS})
    save('screen-status.json',{'status':'complete','groups':len(groups)})
    print('SCREEN_DONE',len(groups),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:
        save('screen-failure.json',{'error':str(ex),'traceback':traceback.format_exc()});traceback.print_exc();sys.exit(1)
