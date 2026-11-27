# Executed in the owning worker's globals after device, stream and AITER initialization.
def sample(seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn((4096,5120), dtype=torch.bfloat16, device=dev, generator=gen)

inputs = [sample(1001+rank), sample(45637+rank)]
residuals = [sample(987+rank), sample(71292+rank)]
# Cancellation probes distinguish rank order in FP32. Interleave probes with random data.
for version, source in enumerate(inputs):
    probes = ([2**26,-2**26,1.,2.,3.,4.,5.,6.],
              [2**28,1.,2.,4.,-2**28,8.,16.,32.])
    for base,values in enumerate(probes):
        for phase in range(8):
            source[:,base*8+phase::128] = values[(rank+phase+version)%8] * (1 if version==0 else -2)
xbuf, rbuf, redbuf, goldbuf = [torch.empty_like(inputs[0]) for _ in range(4)]
ys = [torch.empty_like(xbuf) for _ in range(2)]
rs = [torch.empty_like(xbuf) for _ in range(2)]
anchor = inputs[0][:8]
anchor_out = torch.empty_like(anchor)
assert ref.should_custom_ar(xbuf)

# On this HIP allocator a pool-handle tuple does not keep its use count
# positive after the last captured graph is reset. Pin each reusable
# method pool with its own small graph and captured buffer for this worker.
# Every per-method timed graph is synchronized and reset before its successor.
sequential_pools = os.environ.get('PT8_EXPERIMENT_SEQUENTIAL_POOLS','0') == '1'
graph_pools = {}
pool_keepers = {}
if not sequential_pools:
    graph_pools = {name: torch.cuda.graph_pool_handle() for name in ('prefix','aiter','direct','sharded')}
    for method in ('aiter','direct','sharded'):
        keeper = torch.cuda.CUDAGraph()
        keeper.capture_begin(pool=graph_pools[method])
        keep_tensor = torch.empty(8, device=dev, dtype=torch.int32)
        keep_tensor.fill_(rank)
        keeper.capture_end()
        pool_keepers[method] = (keeper, keep_tensor)

def record_graph(name, fn):
    g = torch.cuda.CUDAGraph()
    with ref.capture():
        g.capture_begin(pool=graph_pools.get(name))
        vals = fn()
        g.capture_end()
    return g, vals

def anchor_calls():
    for _ in range(10):
        ref.all_reduce(anchor, out=anchor_out, registered_input=False)
ref.all_reduce(anchor, out=anchor_out, registered_input=False)
compute.synchronize()
prefix, _ = record_graph('prefix', anchor_calls)

def execute(name, x, residual, steps):
    m = x.shape[0]
    r = residual
    for step in range(steps):
        y = ys[step%2][:m]
        if name == 'aiter':
            y = ref.all_reduce(x, out=y, registered_input=False)
            y, r = norm(y,r)
        else:
            updated = rs[step%2][:m]
            native.fused_into(x,r,norm.weight,y,updated,1e-6,algorithm=name)
            r = updated
    return y,r

def gold(m, version, steps):
    # AR input remains fixed over the chain. The service's residual update is in-place.
    x, r = inputs[version][:m], residuals[version][:m].clone()
    reduced = ref.all_reduce(x, out=redbuf[:m], registered_input=False)
    for step in range(steps):
        y = goldbuf[:m].copy_(reduced)
        y,r = norm(y,r)
    return y.clone(),r

def compare(vals,gold_vals):
    errors,first = [],[]
    for v,g in zip(vals,gold_vals):
        a,b=v.view(torch.int16),g.view(torch.int16)
        bad=a!=b
        count=int(torch.count_nonzero(bad))
        errors.append(count)
        if count:
            i=int(bad.flatten().to(torch.uint8).argmax())
            first.append([i//5120,i%5120,int(a.flatten()[i])&65535,int(b.flatten()[i])&65535])
        else:
            first.append(None)
    return errors,first

def one_shape(m, methods, k, rounds, events):
    x,r = xbuf[:m],rbuf[:m]
    expected = gold(m,0,1),gold(m,1,k)
    graphs, errors, details = {},{},{}
    for name in methods:
        x.copy_(inputs[0][:m]);r.copy_(residuals[0][:m])
        eager=execute(name,x,r,1)
        errs,diag=compare(eager,expected[0])
        compute.synchronize()
        g,vals=record_graph(name, lambda: execute(name,x,r,k))
        x.copy_(inputs[1][:m]);r.copy_(residuals[1][:m])
        g.replay()
        errs2,diag2=compare(vals,expected[1])
        graphs[name]=g
        errors[name]=errs+errs2
        details[name]=diag+diag2
    x.copy_(inputs[0][:m])
    for name in methods:
        r.copy_(residuals[0][:m])
        graphs[name].replay()
    compute.synchronize()
    dist.barrier()
    for t in range(rounds):
        order=list(methods)
        random.Random(98456+m*331+t//len(order)).shuffle(order)
        rotate=t%len(order)
        for name in order[rotate:]+order[:rotate]:
            r.copy_(residuals[0][:m])
            begin,end=events[t,name]
            prefix.replay()
            begin.record()
            graphs[name].replay()
            end.record()
    compute.synchronize()
    times={name:[events[t,name][0].elapsed_time(events[t,name][1])*1000/k for t in range(rounds)] for name in methods}
    for g in graphs.values():g.reset()
    return times,errors,details

def run_stage(task):
    tag=task['stage']
    stage_dir=out/tag
    stage_dir.mkdir(exist_ok=True)
    cfg=stage_dir/'config.json'
    if rank==0:
        if cfg.exists() and json.loads(cfg.read_text())!=task:
            raise RuntimeError('existing stage has different parameters '+str(cfg))
        cfg.write_text(json.dumps(task))
    dist.barrier()
    k,rounds=task['k'],task['rounds']
    events={(t,name):(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for t in range(rounds) for name in ['aiter','direct','sharded']}
    sizes=task['sizes']
    todo=[m for m in sizes if not (stage_dir/f'm{m:04d}.json').is_file()] if rank==0 else None
    obj=[todo];dist.broadcast_object_list(obj,src=0)
    todo=obj[0]
    stamp=time.monotonic()
    for i,m in enumerate(todo):
        names=list(task['methods'])
        if 'selected' in names:
            names[names.index('selected')]='direct' if m<=task['cutoff'] else 'sharded'
        stats=one_shape(m,names,k,rounds,events)
        peers=[None]*dist.get_world_size()
        dist.all_gather_object(peers,stats)
        if rank==0:
            result={}
            for name in names:
                timings=[p[0][name] for p in peers]
                max_us=[max(row) for row in zip(*timings)]
                counts=[sum(col) for col in zip(*(p[1][name] for p in peers))]
                result[name]=dict(us=statistics.median(max_us),rank_us=timings,errors=counts)
                if sum(counts):
                    result[name]['details']=[p[2][name] for p in peers]
                    print('MISMATCH',tag,m,name,counts,result[name]['details'],flush=True)
            path=stage_dir/f'm{m:04d}.json'
            tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(m=m,k=k,rounds=rounds,methods=result)))
            tmp.replace(path)
            if i==0 or (i+1)%100==0 or i+1==len(todo) or len(todo)<100:
                print('MEASURED',tag,i+1,len(todo),'M',m,
                      {n:round(v['us'],3) for n,v in result.items()},round(time.monotonic()-stamp,2),flush=True)
        if any(sum(sum(errors) for errors in p[1].values()) for p in peers):
            raise RuntimeError(f'{tag} bitwise validation failed at M={m}; details saved by rank 0')
        if rank in (4,5) and (i==0 or (i+1)%100==0):
            memory = torch.cuda.memory_stats(dev)
            metrics = {key: int(memory[key]) for key in ('allocated_bytes.all.current','reserved_bytes.all.current',
                                                         'active_bytes.all.current','inactive_split_bytes.all.current') if key in memory}
            metrics['driver_free_bytes'] = int(torch.cuda.mem_get_info(dev)[0])
            print('MEMORY',tag,i+1,m,json.dumps(metrics),flush=True)
        if (i+1)%64==0:
            gc.collect()
