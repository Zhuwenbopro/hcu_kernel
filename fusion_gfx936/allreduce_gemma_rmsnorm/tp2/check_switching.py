import json
import torch
import torch.distributed as dist


def check_switching(name, route, a, b, residual, rank, dev, norm, capture, compute, outdir):
    # Replay alternating shape sequences. Adjacent calls can change communicator
    # and both ranks can arrive at the transition at different times.
    seq=(64,65,512,513,129,1,4096,193,257,272,273,65,63,511,514,128,513,256)
    xs=[torch.empty_like(a[:m]) for m in seq]
    rs=[torch.empty_like(a[:m]) for m in seq]
    scratch=torch.ones((512,5120),device=dev)
    def execute():
        results=[]
        for i,(x,r) in enumerate(zip(xs,rs)):
            if i%3==1 and i%2==rank:
                scratch.mul_(.99999)
            results.append(route.fused(x,r,norm.weight,1e-6))
        return results
    compute.synchronize()
    graph,outs=capture(execute)
    mismatches=0
    total=0
    for t in range(27):
        refs=[]
        for j,m in enumerate(seq):
            aa=(a[:m].float()+(t+j)*.03125).bfloat16()
            bb=(b[:m].float()-(2*t-j)*.03125).bfloat16()
            xs[j].copy_((aa,bb)[rank]);rs[j].copy_(residual[:m]+t*.125)
            refs.append(norm((aa.float()+bb.float()).bfloat16(),rs[j].clone()))
        if t%3==1: actual=execute()
        else:
            graph.replay()
            actual=outs
        counts=[(v.view(torch.int16)!=w.view(torch.int16)).sum() for pair,expect in zip(actual,refs) for v,w in zip(pair,expect)]
        mismatches+=int(torch.stack(counts).sum())
        total+=sum(v.numel() for pair in refs for v in pair)
    both=[None,None]
    dist.all_gather_object(both,dict(rank=rank,bf16_values=total,mismatches=mismatches,calls=27*len(seq)))
    if rank==0:
        (outdir/(name+'-mixed.json')).write_text(json.dumps(both,indent=2))
    assert all(r['mismatches']==0 for r in both)
