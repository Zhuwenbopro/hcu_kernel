"""Packed G16/U16 weight GEMM fused with SiLU(gate)*up; BF16 output, gfx936."""
from pathlib import Path
from dataclasses import dataclass
import ctypes as C
import math,json,hashlib
import torch
import triton
import triton.language as tl
R=Path(__file__).resolve().parent
N,K,H=17408,5120,8704
@triton.jit
def _pack(SRC,DST,rows,PAD:tl.constexpr,STAGE:tl.constexpr,PI:tl.constexpr,INTERLEAVE:tl.constexpr,BLOCK:tl.constexpr):
    E:tl.constexpr=STAGE//4
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    rr=(z//E)%16
    if PI:rr=4*(rr%4)+rr//4
    row=((z//(STAGE*16))%(PAD//16))*16+rr
    if INTERLEAVE:row=(row//32)*16+row%16+tl.where((row%32)>=16,8704,0)
    col=(z//(PAD*STAGE))*STAGE+((z//(16*E))%4)*E+z%E
    v=tl.load(SRC+row*5120+col,(z<PAD*5120)&(row<rows),other=0)
    tl.store(DST+z,v,z<PAD*5120)
@triton.jit
def _scale_interleave(SRC,DST,BLOCK:tl.constexpr):
    z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=(z//32)*16+z%16+tl.where((z%32)>=16,8704,0)
    v=tl.load(SRC+row,z<17408,other=0)
    tl.store(DST+z,v,z<17408)
def pack_a(a,mp,out=None):
    stage=64 if a.dtype==torch.int8 else 32
    if out is None:out=torch.empty((K//stage,mp//16,4,16,stage//4),device=a.device,dtype=a.dtype)
    _pack[(triton.cdiv(mp*K,4096),)](a,out,a.shape[0],mp,stage,False,False,4096,num_warps=4)
    return out
def pack_w(b,out=None):
    stage=64 if b.dtype==torch.int8 else 32
    if out is None:out=torch.empty((K//stage,N//16,4,16,stage//4),device=b.device,dtype=b.dtype)
    _pack[(triton.cdiv(N*K,4096),)](b,out,N,N,stage,True,True,4096,num_warps=4)
    return out
def pack_scale(sw):
    out=torch.empty_like(sw)
    _scale_interleave[(triton.cdiv(N,1024),)](sw,out,1024,num_warps=4)
    return out
HIP=C.CDLL('libamdhip64.so')
HIP.hipGetErrorString.restype=C.c_char_p
HIP.hipModuleLoad.argtypes=[C.POINTER(C.c_void_p),C.c_char_p]
HIP.hipModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
HIP.hipModuleLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
HIP.hipFuncGetAttribute.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p]
def ok(rc):
    if rc:raise RuntimeError((rc,HIP.hipGetErrorString(rc).decode()))
class Kernel:
    def __init__(self,rec):
        self.rec=rec;self.bm=rec['BM'];self.bn=rec['BN'];self.nw=rec['NW']
        path=R/'kernels'/rec['family']/'kernel.hsaco'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['hsaco_sha256']:raise RuntimeError('Kernel checksum mismatch')
        self.module=C.c_void_p();ok(HIP.hipModuleLoad(C.byref(self.module),str(path).encode()))
        self.fn=C.c_void_p();ok(HIP.hipModuleGetFunction(C.byref(self.fn),self.module,rec['function'].encode()))
        self.info={}
        for attr in [0,1,2,3,4]:
            v=C.c_int();ok(HIP.hipFuncGetAttribute(C.byref(v),attr,self.fn));self.info[str(attr)]=v.value
        if self.info['3']!=0:raise RuntimeError('Unexpected scratch')
    def mp_for(self,m):return math.ceil(m/32)*32 if self.rec['base_family']=='E030' else 128 if self.bm==128 and m<=128 else 256
    def launch(self,xp,wp,y,m,sx=None,sw=None):
        mp=self.mp_for(m)
        if self.rec.get('fixed_M') and m!=self.rec['fixed_M']:raise ValueError('Fixed kernel requires M128')
        if math.ceil(m/self.bm)*self.bm>mp:raise ValueError('M does not fit this tile')
        kb=K if self.rec['input_dtype']=='int8' else K*2
        vals=[C.c_void_p(t.data_ptr()) for t in [xp,wp,y]]+[C.c_int(v) for v in [m,N,kb,mp]]
        if self.rec['input_dtype']=='int8':vals += [C.c_void_p(sx.data_ptr()),C.c_void_p(sw.data_ptr())]
        args=(C.c_void_p*len(vals))(*[C.addressof(v) for v in vals])
        ok(HIP.hipModuleLaunchKernel(self.fn,N//self.bn,math.ceil(m/self.bm),1,self.nw*64,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),args,None))
class Engine:
    def __init__(self):
        self.records=json.loads((R/'kernels/manifest.json').read_text())
        self.index={(z['base_family'],z['input_dtype'],z['mode']):z for z in self.records}
        self.kernels={}
        p=R/'fused-policy.json'
        self.routes=json.loads(p.read_text())['by_dtype'] if p.exists() else None
    def key(self,m,precision,family=None):
        if not 1<=m<=256:raise ValueError('Supported M is 1..256')
        if family:return family
        if self.routes:return self.routes[precision][str(m)]
        return 'N128_SAB' if m<=128 else 'M192_PF_SAB' if m<=192 else 'N208_SP7_SAB' if m<=208 else 'N224_SP9_SAB' if m<=224 else 'N240_SP12_SAB' if m<=240 else 'N256_SP11_SAB'
    def kernel(self,m,precision,family=None,debug=False):
        f=self.key(m,precision,family);key=f,precision,'debug' if debug else 'fused'
        if key not in self.kernels:self.kernels[key]=Kernel(self.index[key])
        return self.kernels[key]
    def mm(self,a,prepared,scale_a=None,*,family=None,out=None,debug=False):
        validate(a,prepared,scale_a);prepared.check()
        precision='int8' if a.dtype==torch.int8 else 'bf16';m=a.shape[0]
        k=self.kernel(m,precision,family,debug)
        xp=pack_a(a,k.mp_for(m))
        n=N if debug else H
        if out is None:out=torch.empty((m,n),dtype=torch.bfloat16,device=a.device)
        if out.shape!=(m,n) or out.dtype!=torch.bfloat16 or out.device!=a.device or not out.is_contiguous():raise ValueError('Invalid output')
        k.launch(xp,prepared.packed,out,m,scale_a,prepared.packed_scale)
        return out
_ENGINE=None
def engine():
    global _ENGINE
    if _ENGINE is None:_ENGINE=Engine()
    return _ENGINE
def version(t):
    try:return t._version
    except RuntimeError:return None
@dataclass
class PreparedWeight:
    original:torch.Tensor
    packed:torch.Tensor
    source_version:object
    original_scale:object=None
    packed_scale:object=None
    scale_version:object=None
    def check(self):
        if version(self.original)!=self.source_version:raise ValueError('Weight changed; prepare again')
        if self.original_scale is not None and version(self.original_scale)!=self.scale_version:raise ValueError('Scale changed; prepare again')
    def mm(self,a,scale_a=None):return engine().mm(a,self,scale_a)
def prepare_weight(b,scale_b=None):
    if b.shape!=(K,N) or b.stride()!=(1,K) or b.dtype not in [torch.int8,torch.bfloat16] or not b.is_cuda:
        raise ValueError('B must be INT8/BF16 [5120,17408], stride=(1,5120), CUDA')
    if b.dtype==torch.int8:
        if scale_b is None or scale_b.shape!=(N,1) or scale_b.dtype!=torch.float32 or scale_b.device!=b.device or not scale_b.is_contiguous():raise ValueError('INT8 scale_b must be contiguous FP32 [17408,1]')
    elif scale_b is not None:raise ValueError('BF16 input has no quantization scales')
    return PreparedWeight(b,pack_w(b),version(b),scale_b,None if scale_b is None else pack_scale(scale_b),None if scale_b is None else version(scale_b))
def validate(a,p,sx):
    if a.ndim!=2 or a.shape[1]!=K or not 1<=a.shape[0]<=256 or a.dtype!=p.original.dtype or a.device!=p.original.device or not a.is_contiguous():raise ValueError('A must match prepared W dtype/device, contiguous [M,5120], M1..256')
    if a.dtype==torch.int8:
        if sx is None or sx.shape!=(a.shape[0],1) or sx.dtype!=torch.float32 or sx.device!=a.device or not sx.is_contiguous():raise ValueError('INT8 scale_a must be contiguous FP32 [M,1]')
    elif sx is not None:raise ValueError('BF16 input has no quantization scales')
def silu_mul(a,prepared_weight,scale_a=None):
    return engine().mm(a,prepared_weight,scale_a)
def debug_gate_up(a,prepared_weight,scale_a=None,*,family=None):
    return engine().mm(a,prepared_weight,scale_a,family=family,debug=True)
