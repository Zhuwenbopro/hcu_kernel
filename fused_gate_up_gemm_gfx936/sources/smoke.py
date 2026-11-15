from test_common import *
def main():
    e=fg.engine();records=[]
    for precision in ['int8','bf16']:
        torch.manual_seed(202611153011)
        dtype=torch.int8 if precision=='int8' else torch.bfloat16
        x=torch.randint(-2,3,(256,fg.K),device='cuda',dtype=torch.int8).to(dtype)
        w=torch.randint(-2,3,(fg.N,fg.K),device='cuda',dtype=torch.int8).to(dtype);b=w.t()
        sx=torch.linspace(.003,.02,256,device='cuda').view(-1,1) if precision=='int8' else None
        sw=torch.linspace(.007,.023,fg.N,device='cuda').view(-1,1) if precision=='int8' else None
        p=fg.prepare_weight(b,sw);pb=prepared_baseline(precision,b)
        acc=x.double()@b.double()
        ref=((sx*sw.t())*acc.float()).bfloat16() if precision=='int8' else acc.bfloat16()
        for fam in sorted({r['base_family'] for r in e.records}):
            k=e.kernel(1,precision,fam)
            ms=[k.rec['fixed_M']] if k.rec.get('fixed_M') else sorted({1,min(256,k.bm),min(256,k.bm)-1})
            for m in ms:
                a=x[:m];ss=None if sx is None else sx[:m]
                g=fg.debug_gate_up(a,p,ss,family=fam)
                identical(g,ref[:m],{'precision':precision,'family':fam,'M':m,'phase':'G/U pairing'})
                rb,y=backing(m);e.mm(a,p,ss,family=fam,out=y);guard(rb)
                ra=reference_activation(ref[:m]);ay=activation(ref[:m])
                close(y,ra,{'precision':precision,'family':fam,'M':m,'vs':'FP64 SiLU'})
                close(y,ay,{'precision':precision,'family':fam,'M':m,'vs':'AITER'})
                records.append({'precision':precision,'family':fam,'M':m,'gate_up_bit_exact':True,'guard':True,'vs_FP64':comparison(y,ra),'vs_AITER':comparison(y,ay)})
            print('SMOKE',precision,fam,flush=True)
        # Explicit K tails and gate/up scales with patterned channel values.
        x.zero_();w.zero_()
        positions=[0,7,8,15,16,31,32,63,64,2559,2560,5088,5119]
        for i,kpos in enumerate(positions):
            x[i,kpos]=1
            w[:,kpos]=(torch.arange(fg.N,device='cuda')%7-3).to(dtype)
        p=fg.prepare_weight(b,sw)
        m=len(positions);ss=None if sx is None else sx[:m]
        rr=((ss*sw.t())*(x[:m].double()@b.double()).float()).bfloat16() if precision=='int8' else (x[:m].double()@b.double()).bfloat16()
        for fam in ['N128_SAB','M128_SAB','N256_SP11_SAB']:
            gg=fg.debug_gate_up(x[:m],p,ss,family=fam);identical(gg,rr,{'one_hot':precision,'family':fam})
            yy=e.mm(x[:m],p,ss,family=fam);close(yy,reference_activation(rr),{'one_hot_fused':precision})
        # Prepared weights must detect mutations.
        w[0,0]+=1
        try:p.check();raise AssertionError('missing weight mutation check')
        except ValueError:pass
        del x,w,b,p,pb,acc,ref,rb,y,g,ra,ay,rr,gg,yy
        torch.cuda.empty_cache()
    save('smoke.json',{'status':'complete','records':records,'one_hot_positions':positions,'weight_mutation_rejected':True})
    print('SMOKE_DONE',len(records),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:
        save('smoke-failure.json',{'error':str(ex),'traceback':traceback.format_exc()});traceback.print_exc();sys.exit(1)
