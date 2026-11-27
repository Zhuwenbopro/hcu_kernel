"""Reproduce the gfx936 TP2 microbenchmark under torchrun."""
import argparse
import csv
import datetime
import gc
import json
import os
import random
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from tp2_ar import HipOneShotAllreduce, TunedOneShotAllreduce, RangeSelector, fused_grid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True, help="Qwen checkpoint directory")
parser.add_argument("--output", type=Path, required=True, help="new output directory")
parser.add_argument("--mode", choices=("dense", "range"), default="dense")
parser.add_argument("--sizes", type=int, nargs="+", help="override scan with selected M")
args = parser.parse_args()
if args.sizes and (len(set(args.sizes)) != len(args.sizes) or not all(1 <= m <= 4096 for m in args.sizes)):
    parser.error("--sizes must be unique integers in 1..4096")
outdir = args.output
outdir.mkdir(parents=True, exist_ok=True)
if any(outdir.iterdir()):
    parser.error("--output must be empty")
if int(os.environ.get("WORLD_SIZE", "0")) != 2:
    parser.error("run with torchrun --standalone --nproc-per-node=2")

os.environ.setdefault("HCU_ADAPTER_HOST", "sglang")
os.environ.setdefault("SGLANG_PLUGINS", "hcu")
os.environ.setdefault("AITER_AR_TRANSPORT", "auto")
os.environ.setdefault("AITER_AR_MAX_SIZE_MB", "128")
os.environ.setdefault("AITER_AR_SMALL_MESSAGE", "off")
rank = int(os.environ["RANK"])
dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(dev)
torch.set_grad_enabled(False)
dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=240))

from sglang.srt.plugins import load_plugins
load_plugins()
from hcu_sglang.distributed_aiter import dispatch_custom_allreduce
from sglang.srt.layers.layernorm import GemmaRMSNorm

group = dist.group.WORLD
resources = ExitStack()
resources.callback(dist.destroy_process_group)
ref = dispatch_custom_allreduce(group, dev, backend="aiter")(group=group, device=dev)
resources.callback(ref.close)
if ref.disabled or ref.transport != "ipc" or ref.enable_register_for_capturing:
    raise RuntimeError("reference requires the enabled IPC AITER copy-in path")

def create(cls, max_rows, **options):
    comm = cls(group, dev, max_size=max_rows * 5120 * 2, **options)
    resources.callback(comm.close)
    return comm

fixed = create(HipOneShotAllreduce, 4096)
norm = GemmaRMSNorm(5120, eps=1e-6).to(device=dev, dtype=torch.bfloat16)
key = "model.language_model.layers.0.post_attention_layernorm.weight"
mapping = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(args.model / mapping[key], framework="pt") as f:
    norm._weight_loader(norm.weight, f.get_tensor(key).to(dev))

if args.mode == "dense":
    tuned = create(TunedOneShotAllreduce, 4096)
    decode_default = create(HipOneShotAllreduce, 128)
    decode_window = create(HipOneShotAllreduce, 96, blocks=96)
else:
    proposed = RangeSelector(fixed, create(TunedOneShotAllreduce, 512))
    windows = RangeSelector(fixed, create(TunedOneShotAllreduce, 272), windows_only=True)

def communicator(name, m):
    if name == "original":
        return fixed
    if name == "tuned":
        return tuned
    if name == "decode":
        return decode_window if 65 <= m <= 96 else decode_default
    if name == "65-512":
        return proposed.communicator(m)
    if name == "four_windows":
        return windows.communicator(m)
    raise ValueError(f"unknown TP2 method {name}")

K = 8
shape = (4096, 5120)
def tensor(seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn(shape, dtype=torch.bfloat16, generator=gen, device=dev)

a, b = tensor(67258), tensor(1383)
gold_positive = (a.float() + b.float()).bfloat16()
gold_negative = ((-a).float() + (-b).float()).bfloat16()
local = (a, b)[rank]
neg_local = -local
residual = tensor(2127 + rank)
xs, rf, ra = (torch.empty_like(local) for _ in range(3))
# Stable graph outputs. Separate ping-pong outputs retain the out-of-place
# contract of native fused() and the in-place residual of GemmaRMSNorm.
fused_y = [torch.empty_like(local) for _ in range(2)]
fused_r = [torch.empty_like(local) for _ in range(2)]
aiter_y = [torch.empty_like(local) for _ in range(2)]
assert ref.should_custom_ar(xs)
compute = torch.cuda.Stream(device=dev)
torch.cuda.synchronize()

def aiter_op(x, r):
    y = ref.custom_all_reduce(x)
    assert y is not None
    return norm(y, r)

def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with ref.capture():
        graph.capture_begin()
        val = fn()
        graph.capture_end()
    return graph, val

def chain(which, x, r):
    m = x.shape[0]
    for step in range(K):
        if which == "aiter":
            y = ref.all_reduce(x, out=aiter_y[step % 2][:m], registered_input=False)
            y, r = norm(y, r)
            continue
        comm = communicator(which, m)
        y, next_r = fused_y[step % 2][:m], fused_r[step % 2][:m]
        params = (comm._ctx, x.data_ptr(), r.data_ptr(), norm.weight.data_ptr(),
                  y.data_ptr(), next_r.data_ptr(), m, 5120, 2, 1e-6, compute.cuda_stream)
        if isinstance(comm, TunedOneShotAllreduce):
            rc = comm._osar.lib.osar_run_fused_grid(*params, fused_grid(m))
        else:
            rc = comm._osar.lib.osar_run_fused(*params)
        if rc:
            raise RuntimeError(f"{which} launch returned {rc}")
        r = next_r
    return y, r


def mismatches(vals, gold):
    return [int(torch.count_nonzero(v.view(torch.int16) != g.view(torch.int16))) for v, g in zip(vals, gold)]

def write_rows(path, rows):
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)

def bench_sizes(label,ms,options,ntr,steps):
    global K
    K=steps
    all_keys=list(options)
    events={(t,k):(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for t in range(ntr) for k in all_keys}
    start=time.monotonic()
    for idx,m in enumerate(ms):
        keys=[k for k in all_keys if k!='decode' or m<=128]
        x,fr,ar,ri=xs[:m],rf[:m],ra[:m],residual[:m]
        gg,errs={},{}
        gold1=norm(gold_positive[:m].clone(),ri.clone())
        rr=ri.clone()
        for j in range(K):
            yy,rr=norm(gold_negative[:m].clone(),rr)
        for k in keys:
            x.copy_(local[:m])
            fr.copy_(ri); ar.copy_(ri)
            f=aiter_op(x,ar) if k=='aiter' else communicator(k, m).fused(x, fr, norm.weight, 1e-6)
            e=mismatches(f,gold1)
            compute.synchronize()
            g,vals=capture(lambda: chain(k, x, ar if k == 'aiter' else fr))
            x.copy_(neg_local[:m]); fr.copy_(ri); ar.copy_(ri)
            g.replay()
            e+=mismatches(vals,(yy,rr))
            gg[k]=g
            errs[k]=e
        x.copy_(local[:m])
        for k in keys:
            fr.copy_(ri); ar.copy_(ri)
            gg[k].replay()
        compute.synchronize()
        dist.barrier()
        for t in range(ntr):
            order=keys[:]
            random.Random(14821+idx*997+(t//len(keys))).shuffle(order)
            pos=t%len(keys)
            for k in order[pos:]+order[:pos]:
                res=ar if k=='aiter' else fr
                res.copy_(ri)
                ev0,ev1=events[t,k]
                prefix.replay(); ev0.record(); gg[k].replay(); ev1.record()
        compute.synchronize()
        tt={k:[events[t,k][0].elapsed_time(events[t,k][1])*1000/K for t in range(ntr)] for k in keys}
        gathered=[None,None]
        dist.all_gather_object(gathered,(tt,errs))
        if rank==0:
            raws=[]; summaries=[]
            for k in keys:
                v0=gathered[0][0][k]; v1=gathered[1][0][k]
                vv=[max(a,b) for a,b in zip(v0,v1)]
                c=[sum(r[1][k][i] for r in gathered) for i in range(4)]
                mismatch=sum(c)
                summaries.append(dict(m=m,method=k,us=statistics.median(vv),lo=sorted(vv)[1],hi=sorted(vv)[-2],mismatch=mismatch,eager_norm=c[0],eager_residual=c[1],replay_norm=c[2],replay_residual=c[3],rounds=ntr,k=K))
                raws.extend(dict(m=m,method=k,trial=i,rank0=a,rank1=b) for i,(a,b) in enumerate(zip(v0,v1)))
            write_rows(outdir/(label+'.csv'),summaries)
            write_rows(outdir/(label+'-raw.csv'),raws)
        if any(any(r[1][k]) for r in gathered for k in keys):
            raise RuntimeError(f'bitwise mismatch at M={m}; see {label}.csv')
        for g in gg.values(): g.reset()
        if idx%40==39:
            gc.collect(); torch.cuda.empty_cache()
        if rank==0 and ((idx+1)%100==0 or idx==0 or idx+1==len(ms)):
            print(label,idx+1,len(ms),round(time.monotonic()-start,1),flush=True)

with torch.cuda.stream(compute):
    anchor=local[:16]
    ref.custom_all_reduce(anchor)
    compute.synchronize()
    def anchor_calls():
        for _ in range(8): y=ref.custom_all_reduce(anchor)
        return y
    prefix,prefix_out=capture(anchor_calls)
    print('READY',rank,torch.cuda.get_device_properties(dev),flush=True)
    if args.mode == "dense":
        first=[1,16,32,64,65,68,80,96,128,129,193,256,257,272,512,1024,2048,4096]
        tail=[m for m in range(1,4097) if m not in first]
        random.Random(329485).shuffle(tail)
        bench_sizes('full',args.sizes or first+tail,['original','tuned','aiter','decode'],9,8)
    else:
        sizes = list(range(1, 515)) + [515, 520, 640, 768, 1024, 2048, 4096]
        random.Random(6907).shuffle(sizes)
        bench_sizes("range", args.sizes or sizes, ["original", "65-512", "four_windows"], 15, 8)
        if not args.sizes:
            from check_switching import check_switching
            for name, route in [("proposed", proposed), ("windows", windows)]:
                check_switching(name, route, a, b, residual, rank, dev, norm, capture, compute, outdir)
    compute.synchronize()
dist.barrier()
resources.close()
