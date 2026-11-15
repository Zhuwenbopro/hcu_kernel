import os
os.environ['HIP_VISIBLE_DEVICES']='5'
for k in ['CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:os.environ.pop(k,None)
os.environ['OMP_NUM_THREADS']='8'
from pathlib import Path
import sys,json,traceback,statistics,random,gc,math,importlib.util,hashlib
import torch
import triton
import triton.language as tl
torch.set_num_threads(8)
import fused_gemm as fg
R=Path(__file__).resolve().parent
def imp(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
b8=imp('_frozen_aligned_int8',R/'baseline/int8/aligned-bf16/aligned_gemm.py')
b16=imp('_frozen_packed_bf16',R/'baseline/bf16/bf16-gemm/bf16_gemm.py')
from aiter import silu_and_mul as aiter_silu_and_mul
@triton.jit
def _act(X,Y,TOTAL:tl.constexpr,BLOCK:tl.constexpr):
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=z//8704;col=z%8704
    g=tl.load(X+row*17408+col,z<TOTAL,other=0).to(tl.float32)
    u=tl.load(X+row*17408+8704+col,z<TOTAL,other=0).to(tl.float32)
    v=(g*tl.sigmoid(g))*u
    tl.store(Y+z,v,z<TOTAL)
def activation(x,kind='aiter',out=None):
    if out is None:out=torch.empty((x.shape[0],fg.H),dtype=torch.bfloat16,device=x.device)
    if kind=='aiter':aiter_silu_and_mul(out,x,0.0)
    elif kind=='triton':_act[(triton.cdiv(out.numel(),1024),)](x,out,out.numel(),1024,num_warps=4,enable_fp_fusion=False)
    else:raise ValueError(kind)
    return out
def prepared_baseline(precision,b):return (b8 if precision=='int8' else b16).prepare_weight(b)
def baseline_gemm(precision,a,b,p,sx=None,sw=None):
    return p.scaled_mm(a,b,sx,sw) if precision=='int8' else p.mm(a,b)
def baseline(precision,a,b,p,sx=None,sw=None,kind='aiter'):
    return activation(baseline_gemm(precision,a,b,p,sx,sw),kind)
def save(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
def identical(y,z,ctx):
    assert y.shape==z.shape
    bad=int((y.view(torch.int16)!=z.view(torch.int16)).sum().item())
    if bad:
        ix=torch.nonzero(y!=z)[:5].cpu().tolist()
        raise RuntimeError({'context':ctx,'bad':bad,'first':[(i,float(y[tuple(i)]),float(z[tuple(i)])) for i in ix]})
def backing(m,n=8704):
    b=torch.full((m*n+2048,),0x5a5a,dtype=torch.int16,device='cuda')
    return b,b[1024:-1024].view(torch.bfloat16).view(m,n)
def guard(b):assert bool((b[:1024]==0x5a5a).all().item() and (b[-1024:]==0x5a5a).all().item())
def reference_activation(gate_up):
    g=gate_up[:,:fg.H].double();u=gate_up[:,fg.H:].double()
    return (torch.nn.functional.silu(g)*u).bfloat16()
def comparison(y,z):
    yf=y.float();zf=z.float();finite=torch.isfinite(yf)&torch.isfinite(zf)
    for fn in [torch.isnan,torch.isposinf,torch.isneginf]:assert torch.equal(fn(yf),fn(zf))
    d=(yf[finite]-zf[finite]).double()
    yi=y.view(torch.int16).int();zi=z.view(torch.int16).int()
    yo=torch.where(yi<0,-32768-yi,yi);zo=torch.where(zi<0,-32768-zi,zi)
    ulp=(yo-zo).abs()[finite]
    return {'max_abs':float(d.abs().max().item()) if d.numel() else 0,'relative_L2':float(torch.linalg.vector_norm(d).item()/max(torch.linalg.vector_norm(zf[finite].double()).item(),1e-30)) if d.numel() else 0,'max_BF16_ULP':int(ulp.max().item()) if ulp.numel() else 0,'mismatches':int((y.view(torch.int16)!=z.view(torch.int16)).sum().item()),'elements':y.numel()}
def ulp_distance(y,z):
    yi=y.view(torch.int16).int();zi=z.view(torch.int16).int()
    yo=torch.where(yi<0,-32768-yi,yi);zo=torch.where(zi<0,-32768-zi,zi)
    return (yo-zo).abs()
def close(y,z,ctx=None):
    torch.testing.assert_close(y,z,rtol=1/128,atol=1/256,equal_nan=True,msg=lambda x:str(ctx)+' '+x)

def activation_comparison(y,z,gate_up,ctx=None):
    close(y,z,ctx)
    cm=comparison(y,z)
    dist=ulp_distance(y,z)
    finite=torch.isfinite(y.float())&torch.isfinite(z.float())
    normal=finite&(gate_up[:,:fg.H].float()>=-80)
    cm["max_BF16_ULP_gate_ge_neg80"]=int(dist[normal].max().item()) if bool(normal.any().item()) else 0
    assert cm["max_BF16_ULP_gate_ge_neg80"]<=1,(ctx,cm)
    bad=finite&(dist>1)
    if bool(bad.any().item()):
        g=gate_up[:,:fg.H].float()
        assert bool((g[bad] < -80).all().item()),(ctx,cm)
        tiny_diff=float((y.float()-z.float()).abs()[bad].max().item())
        assert tiny_diff<1e-20,(ctx,cm)
        cm.update(extra_ULP_only_gate_lt_neg80=True,max_abs_extra_ULP=tiny_diff,extra_ULP_elements=int(bad.sum().item()))
    return cm
