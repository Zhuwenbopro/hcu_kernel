from pathlib import Path
import json,sys,collections,hashlib
R=Path(__file__).resolve().parent
def select():
    d=json.loads((R/'screen-result.json').read_text());assert d['status']=='complete'
    recs={(z['base_family'],z['input_dtype']):z for z in json.loads((R/'kernels/manifest.json').read_text())}
    policy={'training_seeds':{'int8':202611153021,'bf16':202611153022},'source':'screen-result.json','by_dtype':{},'selection':'Small M nearest measured point; M129+ upper 16-row endpoint. Minimum measured fused latency among legal tiles. BF16 E030 excluded because split-K changes intermediate BF16 rounding relative to current baseline.'}
    for precision in ['int8','bf16']:
        groups=[g for g in d['groups'] if g['precision']==precision];route={}
        for m in range(1,257):
            target=m if m<=128 else ((m+15)//16)*16
            chosen=None
            for g in sorted(groups,key=lambda g:(abs(g['M']-target),g['M'])):
                for f in sorted((k for k in g['median_us'] if k.startswith('fused:')),key=g['median_us'].get):
                    name=f.split(':',1)[1];rec=recs[name,precision];bm=rec['BM']
                    mp=((m+31)//32)*32 if name=='E030' else 128 if bm==128 and m<=128 else 256
                    if (not rec.get('fixed_M') or rec['fixed_M']==m) and ((m+bm-1)//bm)*bm<=mp:
                        chosen=name;break
                if chosen:break
            assert chosen is not None
            route[str(m)]=chosen
        policy['by_dtype'][precision]=route
    return policy
if __name__=='__main__':
    out=json.dumps(select(),indent=2)+'\n'
    if '--check' in sys.argv:assert out==(R/'fused-policy.json').read_text()
    else:(R/'fused-policy.json').write_text(out)
    print('POLICY',hashlib.sha256(out.encode()).hexdigest(),flush=True)
