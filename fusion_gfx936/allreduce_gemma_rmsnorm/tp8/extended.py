"""TP8 special-value, switching and eager-dispatch checks on the BW47 experiment."""
import contextlib
import ctypes
import datetime
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.append(str(Path(__file__).resolve().parent))
from tp8_ar import TP8Allreduce


WIDTH = 5120
RANKS = 8
CHECK_ROWS = (1, 7, 8, 9, 65, 257, 4096)


def pilot_cutoff():
    override = os.environ.get("TP8_DIRECT_THROUGH")
    if override is not None:
        cutoff = int(override)
    else:
        path = Path(os.environ["TP8_ROUTING_FILE"])
        cutoff = json.loads(path.read_text())["cutoff"]
    if type(cutoff) is not int or not 0 <= cutoff <= 4096:
        raise ValueError(f"invalid TP8 pilot cutoff {cutoff}")
    return cutoff

MODES = ("direct", "sharded", "auto")
CLASSES = ("order", "ordinary", "subnormal", "signed_zero", "finite_extreme", "inf", "qnan", "snan")
# Bit patterns keep signed zero, negative/subnormal values and the original NaN payloads.
PATTERNS = (
    (0x4c80, 0xcc80, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0),
    (0x3f81, 0xc123, 0x3a88, 0xb881, 0x4140, 0x4201, 0xbf90, 0xb042),
    (0x0001, 0x807f, 0x0080, 0x8001, 0x0002, 0x8002, 0x0040, 0x8040),
    (0x0000, 0x8000, 0x8000, 0x8000, 0x0000, 0x8000, 0x8000, 0x0000),
    (0x7f7f, 0xff7f, 0x7f7e, 0x0001, 0x7f7f, 0xff7f, 0xff7e, 0x8001),
    (0x7f80, 0xff80, 0x3f80, 0x0000, 0x4000, 0x0000, 0x7f80, 0xff80),
    (0x7fc1, 0xffd3, 0x3f00, 0x4004, 0x7ff7, 0xffcc, 0xc806, 0x8000),
    (0x7f82, 0xffb5, 0x3f80, 0x0001, 0x7fb1, 0xff91, 0x3f80, 0x0002),
)
RESIDUALS = (
    (0,) * RANKS,
    (0x3b84, 0xbe0d, 0x4002, 0x3fc4, 0x3f81, 0xbf82, 0x4031, 0xc040),
    PATTERNS[2][::-1],
    PATTERNS[3][::-1],
    (0xff7f, 0x7f7f, 0, 0x3f81, 0x0080, 0x8001, 0x7f7f, 0xff7f),
    (0x0000, 0x3f82, 0xff80, 0x7f80, 0x0000, 0xff80, 0xc110, 0x7f80),
    (0x3f80, 0x7fc7, 0x0000, 0xffea, 0x3f80, 0x7fc9, 0xffc1, 0x8000),
    (0x3f80, 0x7f84, 0x0000, 0xff93, 0x3f80, 0x7fa8, 0xff9c, 0x0000),
)
MODES_SEQ = ("auto", "direct", "sharded", "auto", "sharded", "direct", "auto", "sharded")


def tile(tuples, device, shift, *, inputs=False):
    def signed(x):
        return x if x < 32768 else x - 65536

    # Include all-positive / all-negative zero and single-inf / single-NaN
    # lanes alongside opposing infinities and mixed payload lanes.
    rows = []
    for cls, pattern in enumerate(tuples):
        bits = []
        for i in range(32):
            lane = (shift + i) % RANKS
            b = pattern[lane]
            if cls == 3 and i >= 8:
                b = 0x8000 if i < 16 else 0 if i < 24 else b
            if inputs and cls >= 5 and i >= 8:
                if i < 16:
                    b = pattern[0] if lane == 0 else 0x3f80
                elif i < 24:
                    b = pattern[1] if lane == 1 else 0x3f80
                elif cls == 5:
                    b = 0x7f80 if lane == 0 else 0x3f80
            bits.append(signed(b))
        rows.append(bits)
    return torch.tensor(rows, dtype=torch.int16, device=device).view(torch.bfloat16).repeat(1, WIDTH//32)


def fill(x, r, rank, version, kind):
    xx = tile(PATTERNS, x.device, rank + version, inputs=True)
    rr = tile(RESIDUALS, x.device, version*3 + rank)
    if kind == "order":
        x.copy_(xx[:1].expand_as(x))
        r.zero_()
    else:
        for k in range(len(CLASSES)):
            x[k::len(CLASSES)].copy_(xx[k].expand_as(x[k::len(CLASSES)]))
            r[k::len(CLASSES)].copy_(rr[k].expand_as(r[k::len(CLASSES)]))


def check_record(a, b, classes=False):
    aa, bb = a.view(torch.int16), b.view(torch.int16)
    mask = aa != bb
    n = int(mask.count_nonzero())
    info = {"elements": a.numel(), "mismatch_elements": n}
    if n:
        i = int(mask.flatten().to(torch.uint8).argmax())
        info["first"] = [i//WIDTH, i%WIDTH, int(aa.flatten()[i]) & 65535, int(bb.flatten()[i]) & 65535]
        if classes:
            info["by_row_class"] = {name:int(mask[k::8].count_nonzero()) for k,name in enumerate(CLASSES)}
    return info


def order_oracle(rows, version, device):
    total = rows*WIDTH
    flat = torch.empty((total,), device=device, dtype=torch.bfloat16)
    # Each rotation is unique. Round each scalar addition in FP32:
    # [2^26, -2^26, 1, 2, 3, 4, 5, 6], rotated left from 0 to 7.
    values = (21, 16, 24, 17, 19, 22, 18, 23)
    for owner in range(RANKS):
        # Each range starts on a multiple of eight columns.
        part = flat[total*owner//RANKS:total*(owner+1)//RANKS]
        start = owner if rows >= 8 else 0
        for c in range(RANKS):
            part[c::RANKS].fill_(values[(start+version+c)%RANKS])
    return flat.view(rows,WIDTH)


def create_ref(device):
    from sglang.srt.plugins import load_plugins
    load_plugins()

    from benchmark_runtime import load_norm_weight
    from hcu_sglang.distributed_aiter import dispatch_custom_allreduce
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    os.environ["AITER_AR_TRANSPORT"] = "auto"
    factory = dispatch_custom_allreduce(dist.group.WORLD, device, backend="aiter")
    ref = factory(group=dist.group.WORLD, device=device)
    assert not ref.disabled and ref.transport == "ipc" and ref.fully_connected and not ref.enable_register_for_capturing
    norm = GemmaRMSNorm(WIDTH, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    norm._weight_loader(norm.weight, load_norm_weight(device))
    return ref, norm


def reference(ref, norm, x, residual, scratch):
    ar = ref.all_reduce(x, out=scratch, registered_input=False)
    return norm(ar, residual.clone())


def profile_eager(out, rank, device, ref, norm, stream):
    from torch.profiler import profile, ProfilerActivity
    profile_data = []
    for m in (7, 8, 257):
        x, r, scratch = [torch.empty((m, WIDTH),dtype=torch.bfloat16,device=device) for _ in range(3)]
        fill(x,r,rank,0,"order")
        reference(ref,norm,x,r,scratch)
        stream.synchronize()
        dist.barrier()
        prof = profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) if rank == 0 else None
        with prof if prof is not None else contextlib.nullcontext():
            dist.barrier()
            # The reference mutates both x (AR output) and residual. They are disposable here.
            ar = ref.all_reduce(x,out=scratch,registered_input=False)
            norm(ar,r)
            stream.synchronize()
        dist.barrier()
        if rank == 0:
            path = out/f"eager-M{m}-rank0.json"
            prof.export_chrome_trace(str(path))
            e = json.loads(path.read_text())["traceEvents"]
            events = []
            for ev in e:
                cat = str(ev.get("cat","")).lower()
                if ev.get("ph") == "X" and ("kernel" in cat or "gpu_mem" in cat or "gpu_memcpy" in cat):
                    args = ev.get("args",{})
                    events.append({"category":cat,"name":ev["name"],"ts":ev["ts"],"dur_us":ev.get("dur"),
                                   "bytes":args.get("bytes",args.get("Bytes")),"stream":args.get("stream")})
            events.sort(key=lambda ev:ev["ts"])
            profile_data.append({"M":m,"gpu_events":events,"captured":bool(events)})
    if rank == 0:
        (out/"profile-summary.json").write_text(json.dumps(profile_data,indent=2))
    dist.barrier()


def validate(out, rank, device, ref, norm, native, stream):
    cutoff = native.direct_through
    sizes = sorted(set(CHECK_ROWS).union(m for m in (cutoff,cutoff+1) if 1 <= m <= 4096))
    with (out/f"raw-rank{rank}.jsonl").open("w", buffering=1) as log:
        for m in sizes:
            x,r,scratch,actual,updated = [torch.empty((m,WIDTH),device=device,dtype=torch.bfloat16)
                                        for _ in range(5)]
            fill(x,r,rank,0,"order")
            ref.all_reduce(x,out=scratch,registered_input=False)
            norm(scratch,r.clone())
            for mode in MODES:
                native.fused_into(x,r,norm.weight,actual,updated,1e-6,mode)
            stream.synchronize()
            graphs = {}
            for mode in MODES:
                g = torch.cuda.CUDAGraph()
                g.capture_begin()
                native.fused_into(x,r,norm.weight,actual,updated,1e-6,mode)
                g.capture_end()
                graphs[mode] = g
            for kind in ("order","mixed"):
                for seq,mode in enumerate(MODES_SEQ):
                    version = (seq*3 + m) % RANKS
                    fill(x,r,rank,version,kind)
                    y_ref,r_ref = reference(ref,norm,x,r,scratch)
                    oracle = check_record(r_ref, order_oracle(m,version,device)) if kind == "order" else None
                    for eager in (True,False):
                        if eager:
                            native.fused_into(x,r,norm.weight,actual,updated,1e-6,mode)
                        else:
                            graphs[mode].replay()
                            graphs[mode].replay()
                        row = {"M":m,"case":kind,"mode":mode,"step":seq,"version":version,
                               "execution":"eager" if eager else "graph_double_replay",
                               "norm":check_record(actual,y_ref,kind=="mixed"),
                               "residual":check_record(updated,r_ref,kind=="mixed"),"ref_order":oracle}
                        log.write(json.dumps(row)+"\n")
            for g in graphs.values():
                g.reset()
            stream.synchronize()
            if rank == 0:
                print("validated",m,flush=True)


def summarize(out):
    states = [json.loads(line) for i in range(RANKS)
              for line in (out/f"raw-rank{i}.jsonl").read_text().splitlines()]
    sums = {}
    for row in states:
        key = row["case"] + "_" + row["execution"]
        into = sums.setdefault(key, {"rows":0,"norm":0,"residual":0,"ref_order":0})
        into["rows"] += 1
        for item in ("norm","residual","ref_order"):
            if row[item]:
                into[item] += row[item]["mismatch_elements"]
    (out/"validation-summary.json").write_text(json.dumps(sums,indent=2))


def main():
    rank = int(os.environ["LOCAL_RANK"])
    out = Path(os.environ["EXTENDED_OUT"])
    device = torch.device("cuda",rank)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    buf = ctypes.create_string_buffer(80)
    assert ctypes.CDLL("libamdhip64.so").hipDeviceGetPCIBusId(buf,len(buf),rank) == 0
    dist.init_process_group("gloo",timeout=datetime.timedelta(seconds=240))
    from benchmark_runtime import check_device_mapping
    check_device_mapping(buf.value.decode().lower(), 8)
    ref,norm = create_ref(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        profile_eager(out,rank,device,ref,norm,stream)
        native = TP8Allreduce(dist.group.WORLD,device,max_rows=4096,blocks=64,direct_through=pilot_cutoff())
        validate(out,rank,device,ref,norm,native,stream)
        native.close()
    ref.close()
    dist.barrier()
    if rank == 0:
        summarize(out)
    dist.destroy_process_group()
    if rank == 0:
        summary = json.loads((out/"validation-summary.json").read_text())
        if any(section[field] for section in summary.values() for field in ("norm", "residual", "ref_order")):
            raise RuntimeError("raw bit differences, see validation-summary.json and raw-rank*.jsonl")
        dispatch = json.loads((out/"profile-summary.json").read_text())
        for row in dispatch:
            expected = "cross_device_reduce_1stage" if row["M"] < 8 else "cross_device_reduce_2stage"
            events = row["gpu_events"]
            if not (any(expected in e["name"] for e in events) and
                    any("fused_add_rmsnorm" in e["name"] for e in events) and
                    any("Memcpy DtoD" in e["name"] and e["bytes"] == row["M"]*WIDTH*2 for e in events)):
                raise RuntimeError(f"unexpected AITER operator profile for M={row['M']}")


if __name__ == "__main__":
    main()
