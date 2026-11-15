from screen import pool,capture,measure,COPIES
from test_common import *
import threading,datetime,ctypes as C,importlib.metadata as md
ROUNDS=7
POINTS=sorted(set(range(1,17))|{32,48,64,80,96,112,128,129,144,160,176,192,193,208,224,240,256})
SEEDS=[202611153041,202611153042,202611153043]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def clocks(stop):
    root=Path('/sys/bus/pci/devices/0000:c1:00.0')
    with (R/'clocks.jsonl').open('w') as out:
        while not stop.is_set():
            rec={'time':now()}
            for n in ['pp_dpm_sclk','pp_dpm_mclk','power_dpm_force_performance_level','mem_info_vram_used']:
                try:rec[n]=(root/n).read_text().strip()
                except OSError:pass
            for p in (root/'hwmon').glob('hwmon*/*'):
                if p.name in ['freq1_input','freq2_input','temp1_input','power1_average']:
                    try:rec[p.name]=p.read_text().strip()
                    except OSError:pass
            out.write(json.dumps(rec)+'\n');out.flush();stop.wait(1)
def main():
    v=json.loads((R/'validation.json').read_text())
    assert v['status']=='complete'
    assert v['policy_sha256']==hashlib.sha256((R/'fused-policy.json').read_bytes()).hexdigest()
    e=fg.engine()
    fg.HIP.hipDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int]
    pci=C.create_string_buffer(64);fg.ok(fg.HIP.hipDeviceGetPCIBusId(pci,64,0))
    env={'torch':torch.__version__,'HIP':torch.version.hip,'device':str(torch.cuda.get_device_properties(0)),'PCI':pci.value.decode(),'node':'bw47','container':'zhuwenbo_sgl_0522','GPU':5,'formal_seeds':SEEDS,'rounds':ROUNDS,'copies':COPIES,'points':POINTS,'policy_sha256':v['policy_sha256'],'activation':'AITER silu_and_mul(out,input,limit=0) and single-kernel Triton sigmoid(g)*g*u, FP32 arithmetic','aiter_distribution':[{d.metadata.get('Name'):d.version} for d in md.distributions() if 'aiter' in d.metadata.get('Name','').lower()],'env':{k:v for k,v in os.environ.items() if k in ['TORCH_BLAS_PREFER_HIPBLASLT','PYTORCH_TUNABLEOP_ENABLED','PYTORCH_TUNABLEOP_TUNING','HIP_VISIBLE_DEVICES']},'scope':'A pack + GEMM + activation/BF16 output. Static weight/scale preparation excluded; input INT8 quantization excluded.'}
    save('environment.json',env)
    stop=threading.Event();thread=threading.Thread(target=clocks,args=(stop,),daemon=True);thread.start()
    groups=[]
    try:
        for precision in ['int8','bf16']:
            for si,seed in enumerate(SEEDS):
                x,sx,weights,scales,old,new,ref=pool(precision,seed)
                refgu=baseline_gemm(precision,x,weights[-1],old[-1],sx,scales[-1])
                print('POOL_READY',precision,seed,flush=True)
                orderM=POINTS.copy();random.Random(seed).shuffle(orderM)
                with (R/f'benchmark-{precision}-seed-{seed}.jsonl').open('w') as out:
                    for mi,m in enumerate(orderM):
                        a=x[:m];ss=None if sx is None else sx[:m]
                        tmp=fg.silu_mul(a,new[0],ss)
                        tmp=activation(fg.debug_gate_up(a,new[0],ss),'triton')
                        for kind in ['aiter','triton']:tmp=baseline(precision,a,weights[0],old[0],ss,scales[0],kind)
                        torch.cuda.synchronize();del tmp
                        graphs={};outputs={}
                        graphs['fused'],outputs['fused']=capture(lambda:[fg.silu_mul(a,p,ss) for p in new])
                        graphs['current_GEMM'],outputs['current_GEMM']=capture(lambda:[baseline_gemm(precision,a,b,p,ss,sw) for b,p,sw in zip(weights,old,scales)])
                        for kind in ['aiter','triton']:
                            graphs['current_plus_'+kind],outputs['current_plus_'+kind]=capture(lambda kind=kind:[baseline(precision,a,b,p,ss,sw,kind) for b,p,sw in zip(weights,old,scales)])
                        graphs['same_tile_plus_triton'],outputs['same_tile_plus_triton']=capture(lambda:[activation(fg.debug_gate_up(a,p,ss),'triton') for p in new])
                        for _ in range(2):
                            for g in graphs.values():g.replay()
                        torch.cuda.synchronize()
                        for f,ys in outputs.items():close(ys[-1],refgu[:m] if f=='current_GEMM' else ref[:m],{'warmup':precision,'M':m,'path':f})
                        group={'precision':precision,'seed':seed,'M':m,'fused_family':e.key(m,precision),'current_family':(b8 if precision=='int8' else b16).engine().key(m),'samples':[],'addresses':{'A':a.data_ptr(),'original_W0':weights[0].data_ptr(),'original_W71':weights[-1].data_ptr(),'old_packed_W0':old[0].packed.data_ptr(),'new_packed_W0':new[0].packed.data_ptr(),'old_packed_W71':old[-1].packed.data_ptr(),'new_packed_W71':new[-1].packed.data_ptr()}}
                        rng=random.Random(seed+m)
                        for rnd in range(ROUNDS):
                            order=list(graphs);rng.shuffle(order);row={'round':rnd,'order':order,'us':{}}
                            for f in order:
                                row['us'][f]=measure(graphs[f])
                                close(outputs[f][-1],refgu[:m] if f=='current_GEMM' else ref[:m],{'timing':precision,'M':m,'path':f})
                            group['samples'].append(row)
                        group['accuracy']={f:comparison(ys[-1],refgu[:m] if f=='current_GEMM' else ref[:m]) for f,ys in outputs.items()}
                        groups.append(group);out.write(json.dumps(group,separators=(',',':'))+'\n');out.flush()
                        med={f:statistics.median(s['us'][f] for s in group['samples']) for f in graphs}
                        save('benchmark-status.json',{'status':'running','precision':precision,'seed':seed,'M':m,'groups':len(groups),'completed_M_this_seed':mi+1,'median_us':med})
                        print('TIMING',precision,seed,mi+1,m,e.key(m,precision),{k:round(v,3) for k,v in med.items()},flush=True)
                        del graphs,outputs,g,ys
                        gc.collect();torch.cuda.empty_cache()
                del x,sx,weights,scales,old,new,ref,refgu
                gc.collect();torch.cuda.empty_cache()
        save('benchmark-raw.json',{'status':'complete','finished':now(),'groups':groups,'seeds':SEEDS,'rounds':ROUNDS,'copies':COPIES,'points':POINTS,'environment':env})
        save('benchmark-status.json',{'status':'complete','groups':len(groups),'time':now()})
        print('BENCHMARK_DONE',len(groups),flush=True)
    finally:stop.set();thread.join(timeout=3)
if __name__=='__main__':
    try:main()
    except BaseException as ex:
        save('benchmark-failure.json',{'error':str(ex),'traceback':traceback.format_exc()});traceback.print_exc();sys.exit(1)
