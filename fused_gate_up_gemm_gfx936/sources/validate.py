from test_common import *
import numpy as np
def dataset(precision,seed):
    torch.manual_seed(seed);dtype=torch.int8 if precision=='int8' else torch.bfloat16
    if precision=='int8':
        x=torch.randint(-127,128,(256,fg.K),device='cuda',dtype=dtype)
        w=torch.randint(-127,128,(fg.N,fg.K),device='cuda',dtype=dtype)
        sx=torch.rand((256,1),device='cuda')*.012+.002
        sw=torch.rand((fg.N,1),device='cuda')*.008+.001
    else:
        scale=1.0 if seed%2 else .125
        x=torch.randn((256,fg.K),device='cuda',dtype=dtype)*scale
        w=torch.randn((fg.N,fg.K),device='cuda',dtype=dtype)
        sx=sw=None
    return x,w,sx,sw
def main():
    assert json.loads((R/'smoke.json').read_text())['status']=='complete'
    e=fg.engine();records=[];family_records=[]
    for precision in ['int8','bf16']:
        for seed in [202611153031,202611153032]:
            x,w,sx,sw=dataset(precision,seed);b=w.t();p=fg.prepare_weight(b,sw);pb=prepared_baseline(precision,b)
            acc64=x.double()@b.double()
            rs=[0,15,16,127,128,255];cs=[0,15,16,8703,8704,17407]
            cpu=x[rs].double().cpu().numpy()@w[cs].double().cpu().numpy().T
            np.testing.assert_allclose(cpu,acc64[rs][:,cs].cpu().numpy(),rtol=1e-12,atol=1e-12)
            refgu=((sx*sw.t())*acc64.float()).bfloat16() if precision=='int8' else acc64.bfloat16()
            for fam in sorted({z['base_family'] for z in e.records}):
                k=e.kernel(1,precision,fam);m=k.rec.get('fixed_M') or min(256,k.bm)
                a=x[:m];ss=None if sx is None else sx[:m]
                gu=fg.debug_gate_up(a,p,ss,family=fam);y=e.mm(a,p,ss,family=fam);ay=activation(gu)
                close(gu,refgu[:m],{'family_GU':fam,'precision':precision})
                cm=activation_comparison(y,ay,gu,{'precision':precision,'seed':seed,'family':fam})
                same_bit=None
                if precision=='bf16' and (fam,'bf16') in b16.engine().kernels:
                    control=b16.engine().mm(a,pb,family=fam)
                    identical(gu,control,{'same_family_GU':fam,'precision':precision,'seed':seed})
                    same_bit=True
                elif precision=='int8':
                    identical(gu,refgu[:m],{'exact_integer_GU':fam,'seed':seed});same_bit=True
                family_records.append({'precision':precision,'seed':seed,'family':fam,'M':m,'same_family_or_exact_reference_bit_exact':same_bit,'vs_same_family_AITER':cm})
            for m in range(1,257):
                a=x[:m];ss=None if sx is None else sx[:m];fam=e.key(m,precision)
                gb,gu=backing(m,fg.N);e.mm(a,p,ss,out=gu,debug=True);guard(gb)
                old=baseline_gemm(precision,a,b,pb,ss,sw)
                identical(gu,old,{'default_gate_up':precision,'seed':seed,'M':m,'family':fam})
                yb,y=backing(m);e.mm(a,p,ss,out=y);guard(yb)
                ay=activation(old);ty=activation(old,'triton');ref=reference_activation(old)
                close(y,ay,{'fused_AITER':precision,'seed':seed,'M':m})
                ca=activation_comparison(y,ay,old,{'precision':precision,'seed':seed,'M':m,'reference':'AITER'})
                ct=activation_comparison(y,ty,old,{'precision':precision,'seed':seed,'M':m,'reference':'Triton'})
                close(y,ref,{'fused_FP64_activation':precision,'seed':seed,'M':m})
                close(gu,refgu[:m],{'GEMM_FP64':precision,'seed':seed,'M':m})
                if m in [1,2,16,32,64,128,129,160,192,193,256]:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):gy=fg.silu_mul(a,p,ss)
                    for _ in range(3):graph.replay()
                    torch.cuda.synchronize()
                    identical(gy,y,{'Graph':precision,'M':m});del graph,gy
                records.append({'precision':precision,'seed':seed,'M':m,'family':fam,'gate_up_vs_current_bit_exact':True,'GU_vs_FP64':comparison(gu,refgu[:m]),'fused_vs_AITER':ca,'fused_vs_Triton':ct,'fused_vs_FP64_activation':comparison(y,ref),'guards':True})
                if m%32==0 or m in [1,128,129,193,256]:
                    save('validation-status.json',{'status':'running','precision':precision,'seed':seed,'M':m,'cases':len(records)})
                    print('VALIDATE',precision,seed,m,'maxULP',ca['max_BF16_ULP'],'relL2',round(ca['relative_L2'],8),flush=True)
            save('validation-partial.json',{'status':'running','records':records,'family_records':family_records})
            del x,w,b,p,pb,acc64,refgu,gb,gu,yb,y,old,ay,ty,ref
            gc.collect();torch.cuda.empty_cache()
    # BF16 tail semantics including saturation and nonfinite values.
    a=torch.zeros((1,fg.K),device='cuda',dtype=torch.bfloat16);a[0,0]=1
    w=torch.zeros((fg.N,fg.K),device='cuda',dtype=torch.bfloat16)
    vals=torch.tensor([-100,-90,-88.5,-80,-30,-10,-1,-0.,0.,1,10,30,80,float('inf'),float('-inf'),float('nan')],device='cuda',dtype=torch.bfloat16)
    w[:fg.H,0]=vals[torch.arange(fg.H,device='cuda')%vals.numel()]
    w[fg.H:,0]=(torch.arange(fg.H,device='cuda')%9-4).bfloat16()
    p=fg.prepare_weight(w.t());special=[]
    for fam in ['N128_SAB','M128_SAB','E047','E030']:
        gu=fg.debug_gate_up(a,p,family=fam);y=e.mm(a,p,family=fam)
        ay=activation(gu);close(y,ay,{'nonfinite':fam});special.append({'family':fam,'vs_AITER':comparison(y,ay)})
    # Worst full-range signed INT8 dot products, with exact power-of-two scales.
    a=torch.full((1,fg.K),-128,device='cuda',dtype=torch.int8)
    w=torch.full((fg.N,fg.K),127,device='cuda',dtype=torch.int8);w[:fg.H]=-128
    sx=torch.full((1,1),1/256,device='cuda');sw=torch.full((fg.N,1),1/256,device='cuda')
    p=fg.prepare_weight(w.t(),sw)
    gu=fg.debug_gate_up(a,p,sx);expect=((sx*sw.t())*(a.double()@w.double().T).float()).bfloat16()
    identical(gu,expect,{'full_range_INT8':True});close(fg.silu_mul(a,p,sx),activation(gu))
    sw[0,0]+=1
    try:p.check();raise AssertionError('Missing scale mutation guard')
    except ValueError:pass
    save('validation.json',{'status':'complete','records':records,'family_records':family_records,'special':special,'full_range_INT8':True,'scale_mutation_rejected':True,'FP64_reference_CPU_spots_per_seed':36,'activation_max_ULP_gate_ge_neg80':1,'FP32_underflow_region_absolute_tolerance':1e-20,'policy_sha256':hashlib.sha256((R/'fused-policy.json').read_bytes()).hexdigest()})
    save('validation-status.json',{'status':'complete','cases':len(records),'family_cases':len(family_records)})
    print('VALIDATION_DONE',len(records),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:
        save('validation-failure.json',{'error':str(ex),'traceback':traceback.format_exc()});traceback.print_exc();sys.exit(1)
