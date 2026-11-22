import torch
import triton
import triton.language as tl

N, K = 124160, 5120

@triton.jit
def _head(X,W,Y,M,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    p = tl.program_id(0)
    gm = tl.cdiv(M, BM)
    # Adjacent CTAs access the same weight panel for different token blocks.
    pn = p // gm
    pm = p % gm
    i = pm * BM + tl.arange(0,BM)
    j = pn * BN + tl.arange(0,BN)
    k = tl.arange(0,BK)
    xp = X + i[:,None]*K + k[None,:]
    wp = W + k[:,None]*N + j[None,:]
    acc = tl.full((BM,BN),0,tl.float32)
    for _ in range(K//BK):
        x=tl.load(xp,mask=i[:,None]<M,other=0.)
        w=tl.load(wp)
        acc=tl.dot(x,w,acc)
        xp += BK
        wp += BK*N
    tl.store(Y+i[:,None]*N+j[None,:],acc,mask=i[:,None]<M)

def launch(x,w,y,config):
    bm,bn,bk,nw=config
    return _head[(triton.cdiv(x.shape[0],bm)*(N//bn),)](x,w,y,x.shape[0],N,K,bm,bn,bk,num_warps=nw,num_stages=2,matrix_instr_nonkdim=16)


@triton.jit
def _transpose_active_x(X, Packed, M, P, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inner = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + row[:, None] * K + inner[None, :], row[:, None] < M, 0)
    tl.store(Packed + inner[None, :] * P + row[:, None], values, row[:, None] < P)


def pack_active_x(x, workspace):
    """Transpose X and zero its aligned tail in one GPU launch."""
    m, k = x.shape
    p = workspace.shape[1]
    block = 32
    return _transpose_active_x[(triton.cdiv(p, block), k // block)](
        x, workspace, m, p, k, block, num_warps=4,
    )
