import ctypes, datetime, gc, json, os, random, statistics, sys, time
from pathlib import Path
import torch
import torch.distributed as dist

root=Path(__file__).resolve().parent
out=Path(os.environ['OUTDIR'])
rank=int(os.environ['RANK'])
dev=torch.device('cuda',rank)
torch.cuda.set_device(dev)
torch.set_grad_enabled(False)
buf=ctypes.create_string_buffer(80)
assert ctypes.CDLL('libamdhip64.so').hipDeviceGetPCIBusId(buf,len(buf),rank)==0
assert dist.is_gloo_available() and int(os.environ["WORLD_SIZE"]) == 8
dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=360))
from benchmark_runtime import check_device_mapping
check_device_mapping(buf.value.decode().lower(), 8)
from sglang.srt.plugins import load_plugins
load_plugins()

from benchmark_runtime import load_norm_weight
from hcu_sglang.distributed_aiter import dispatch_custom_allreduce
from sglang.srt.layers.layernorm import GemmaRMSNorm
# IPC may query the active SGLang server config; the earlier validated bench set auto
# which resolves to IPC locally and avoids tying Framework state to admission.
os.environ['AITER_AR_TRANSPORT']='auto'
factory=dispatch_custom_allreduce(dist.group.WORLD,dev,backend='aiter')
ref=factory(group=dist.group.WORLD,device=dev)
assert not ref.disabled, (rank, 'AITER TP8 rejected the selected physical group', getattr(ref,'fully_connected',None))
assert ref.world_size==8 and ref.transport=='ipc' and ref.fully_connected and not ref.enable_register_for_capturing
assert ref.small_ar_mode=='off', ref.small_ar_mode
norm=GemmaRMSNorm(5120,eps=1e-6).to(device=dev,dtype=torch.bfloat16)
norm._weight_loader(norm.weight,load_norm_weight(dev))
compute=torch.cuda.Stream(device=dev)
compute.wait_stream(torch.cuda.current_stream(dev))
native=None
print('READY_AITER',rank,buf.value,ref.transport,ref.max_size,flush=True)
with torch.cuda.stream(compute):
    stage_loaded=False
    active_blocks=None
    while True:
        name=None
        if rank==0:
            pending=sorted(p for p in (out/'jobs').glob('*.json') if not (out/(p.stem+'.done')).is_file())
            if pending: name=pending[0].name
        box=[name];dist.broadcast_object_list(box,src=0)
        if box[0] is None:
            time.sleep(.25);continue
        task=json.loads((out/'jobs'/box[0]).read_text())
        if task['stage']=='stop': break
        if not stage_loaded:
            exec(compile((root/'suite.py').read_text(),str(root/'suite.py'),'exec'),globals())
            stage_loaded=True
        blocks=int(task.get('blocks',64))
        if native is not None and blocks!=active_blocks:
            native.close()
            native=None
        if native is None and task.get('methods')!=['aiter']:
            active_blocks=blocks
            from tp8_ar import TP8Allreduce
            native=TP8Allreduce(dist.group.WORLD,dev,max_rows=4096,blocks=blocks)
        run_stage(task)
        compute.synchronize();dist.barrier()
        if rank==0: (out/(Path(box[0]).stem+'.done')).touch()
if native is not None:native.close()
ref.close()
dist.destroy_process_group()
