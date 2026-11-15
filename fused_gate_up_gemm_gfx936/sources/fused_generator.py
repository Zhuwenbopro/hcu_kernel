"""Gate16/up16 interleaving; gfx936 native INT8/BF16 GEMM with BF16 SwiGLU output."""
from pathlib import Path
import re,json,sys,hashlib,subprocess,importlib.util,collections,yaml
R=Path(__file__).resolve().parent
BASE=R/'baseline/bf16/bf16-gemm'
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('_frozen_tile_generator',BASE/'int8_tile_generator.py')
i8=importlib.util.module_from_spec(spec);spec.loader.exec_module(i8)
MC='/opt/dtk/aillvm/bin/llvm-mc';LD='/opt/dtk/aillvm/bin/ld.lld';OD='/opt/dtk/aillvm/bin/llvm-objdump'
N,K,H=17408,5120,8704
def run(args):
    p=subprocess.run(args,capture_output=True,text=True)
    if p.returncode:raise RuntimeError(' '.join(args)+'\n'+p.stdout+p.stderr)
    return p.stdout
def round_fp32(v,tmp,nan):
    return [
        f'v_cmp_u_f32_e32 vcc, v{v}, v{v}',
        f'v_lshrrev_b32_e32 v{tmp}, 16, v{v}',
        f'v_and_b32_e32 v{tmp}, 1, v{tmp}',
        f'v_add_u32_e32 v{v}, 0x7fff, v{v}',
        f'v_add_u32_e32 v{v}, v{tmp}, v{v}',
        f'v_and_b32_e32 v{v}, 0xffff0000, v{v}',
        f'v_cndmask_b32_e32 v{v}, v{v}, v{nan}, vcc']
def store4(a,tmp,nan,addr,offset):
    code=[]
    for v in range(a,a+4):code+=round_fp32(v,tmp,nan)
    code += [
        f'v_lshrrev_b32_e32 v{a}, 16, v{a}',
        f'v_or_b32_e32 v{a}, v{a}, v{a+1}',
        f'v_lshrrev_b32_e32 v{a+2}, 16, v{a+2}',
        f'v_or_b32_e32 v{a+1}, v{a+2}, v{a+3}',
        f'global_store_dwordx2 v[{addr}:{addr+1}], v[{a}:{a+1}], off offset:{offset} glc slc']
    return code
def swiglu(g,u,tmp,nan):
    code=[]
    for e in range(4):
        code+=round_fp32(g+e,tmp,nan)+round_fp32(u+e,tmp,nan)
        code += [
            f'v_mul_f32_e32 v{tmp}, 0xbfb8aa3b, v{g+e}',
            f'v_exp_f32_e32 v{tmp}, v{tmp}',
            f'v_add_f32_e32 v{tmp}, 1.0, v{tmp}',
            f'v_rcp_f32_e32 v{tmp}, v{tmp}',
            f'v_mul_f32_e32 v{g+e}, v{g+e}, v{tmp}',
            f'v_mul_f32_e32 v{g+e}, v{g+e}, v{u+e}']
    return code
def epilogue(cfg,precision,mode):
    t=cfg['address_registers'][0];ac=cfg['accumulator_VGPR']
    direct=cfg['direction'];pf=cfg['PF'];sf=cfg['SF']
    cols=sf if direct=='M' else pf
    rows=pf if direct=='M' else sf
    sw=ac;sx=sw+4*cols;tmp=sx+1;nan=sx+2
    assert nan<t,(cfg['family'],nan,t)
    code=['GEMM_WAIT_VM 0']
    if precision=='int8':
        code+=['s_load_dwordx4 s[0:3], s[4:5], 0x28','GEMM_WAIT_LGKM 0']
    code += [
        f'v_lshrrev_b32_e32 v{t+2}, 4, v{t}',f'v_and_b32_e32 v{t+2}, 15, v{t+2}',
        f'v_lshrrev_b32_e32 v{t+3}, 8, v{t}',
        f'v_and_b32_e32 v{t+3}, 3, v{t+3}',
        f'v_lshlrev_b32_e32 v{t+3}, 2, v{t+3}',
        f'v_lshlrev_b32_e32 v{t+4}, 2, v{t+3}',
        f'v_add_u32_e32 v{t+4}, s23, v{t+4}']
    if direct=='M':
        code += [i8.scale_u32(f'v{t+10}',f'v{t+1}',pf*16),f'v_add_u32_e32 v{t+2}, v{t+10}, v{t+2}']
    else:
        code += [f'v_lshlrev_b32_e32 v{t+10}, 7, v{t+1}',f'v_add_u32_e32 v{t+4}, v{t+10}, v{t+4}',
                 f'v_lshlrev_b32_e32 v{t+10}, 4, v{t+1}',f'v_add_u32_e32 v{t+3}, v{t+10}, v{t+3}']
    code += [f'v_add_u32_e32 v{t+2}, s22, v{t+2}',
             's_lshl_b32 s26, s6, 6',f'v_add_u32_e32 v{t+3}, s26, v{t+3}',
             f'v_mov_b32_e32 v{nan}, 0x7fff0000']
    if precision=='int8':
        for c in range(cols):code += [f'global_load_dwordx4 v[{sw+4*c}:{sw+4*c+3}], v{t+4}, s[2:3] offset:{c*64}']
    for row in range(rows):
        skip=f'.L_fused_skip_{row}'
        code += [f'v_add_u32_e32 v{t+7}, {row*16}, v{t+2}',
                 f'v_cmp_gt_u32_e32 vcc, s14, v{t+7}','s_and_saveexec_b64 s[28:29], vcc',f's_cbranch_execz {skip}']
        if precision=='int8':code += [f'v_lshlrev_b32_e32 v{tmp}, 2, v{t+7}',f'global_load_dword v{sx}, v{tmp}, s[0:1]']
        stride=H if mode=='fused' else N
        code += [f'v_mul_u32_u24_e32 v{t+8}, {stride}, v{t+7}',f'v_add_u32_e32 v{t+8}, v{t+3}, v{t+8}',
                 f'v_lshlrev_b32_e32 v{t+8}, 1, v{t+8}',
                 f'v_add_co_u32_e32 v{t+5}, vcc, s12, v{t+8}',
                 f'v_mov_b32_e32 v{t+6}, s13',f'v_addc_co_u32_e32 v{t+6}, vcc, 0, v{t+6}, vcc']
        if precision=='int8':code += ['GEMM_WAIT_VM 0']
        for pair in range(cols//2):
            c=2*pair
            g=4*(c*pf+row) if direct=='M' else 4*(row*pf+c)
            u=4*((c+1)*pf+row) if direct=='M' else g+4
            if precision=='int8':
                for start,scale in [(g,sw+4*c),(u,sw+4*(c+1))]:
                    for e in range(4):
                        code += [f'v_cvt_f32_i32 v{start+e}, v{start+e}',
                                 f'v_mul_f32_e32 v{tmp}, v{sx}, v{scale+e}',
                                 f'v_mul_f32_e32 v{start+e}, v{tmp}, v{start+e}']
            if mode=='fused':
                code+=swiglu(g,u,tmp,nan)+store4(g,tmp,nan,t+5,pair*32)
            else:
                code+=store4(g,tmp,nan,t+5,pair*32)
                code += [f'v_add_co_u32_e32 v{t+9}, vcc, {H*2}, v{t+5}',f'v_addc_co_u32_e32 v{t+10}, vcc, 0, v{t+6}, vcc']
                code+=store4(u,tmp,nan,t+9,pair*32)
        code += [skip+':','s_or_b64 exec, exec, s[28:29]']
    return '\n'.join(code)+'\n'
def int8_args(source):
    source=source.replace('.amdhsa_kernarg_size 40','.amdhsa_kernarg_size 56')
    a,rest=source.split('.amdgpu_metadata',1);doc,b=rest.split('.end_amdgpu_metadata',1)
    m=yaml.safe_load(doc.expandtabs(8))
    for rec in m['amdhsa.kernels']:
        for key in ['.args','.unfolded_args']:
            if key in rec:rec[key]+=[{'.address_space':'global','.offset':o,'.size':8,'.value_kind':'global_buffer'} for o in [40,48]]
        rec['.kernarg_segment_size']=56
    return a+'.amdgpu_metadata\n---\n'+yaml.safe_dump(m,sort_keys=False)+'...\n.end_amdgpu_metadata'+b
def render(direction,bm,read_first,scalar_address,prefetch_order,precision,mode):
    source,cfg=i8.render_kernel(direction,bm,read_first,scalar_address,prefetch_order)
    old=cfg['family'];base=old
    if precision=='bf16':
        source=source.replace('GEMM_MMA_I8','GEMM_MMA_BF16')
        assert source.count('s_mov_b32 s24, 40\n')==1
        source=source.replace('s_mov_b32 s24, 40\n','s_mov_b32 s24, 80\n')
    t=cfg['address_registers'][0]
    marker=f'GEMM_WAIT_VM 0\nv_mov_b32_e32 v{t+11}, s13\n'
    assert source.count(marker)==1
    start=source.index(marker);end=source.index('s_endpgm',start)
    loop=source[source.index('.L_pair:\n'):start]
    source=source[:start]+epilogue(dict(cfg,direction=direction),precision,mode)+source[end:]
    new=old+'_'+precision.upper()+'_'+mode.upper()
    source=re.sub(r'\b'+re.escape(old)+r'\b',new,source)
    if precision=='int8':source=int8_args(source)
    cfg.update(base_family=base,family=new,function=new,direction=direction,input_dtype=precision,output_dtype='bf16',mode=mode,
        stage_K=64 if precision=='int8' else 32,total_stages=80 if precision=='int8' else 160,
        N=N,K=K,H=H,output_N=H if mode=='fused' else N,interleaved_W='G16,U16',intermediate_rounding='BF16 RNE before SiLU',
        K_loop_source_sha256=hashlib.sha256(loop.encode()).hexdigest(),extra_VGPR=0,extra_LDS=0)
    return source,cfg
def emit(source,rec):
    d=R/'kernels'/rec['family'];d.mkdir(parents=True,exist_ok=True)
    (d/'kernel.s').write_text(source)
    run([MC,'--save-temp-labels','-I',str(R),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx936','-mattr=+sramecc','-filetype=obj',str(d/'kernel.s'),'-o',str(d/'kernel.o')])
    run([LD,'-shared',str(d/'kernel.o'),'-o',str(d/'kernel.hsaco')])
    dis=run([OD,'-d','--mcpu=gfx936',str(d/'kernel.hsaco')]);(d/'device.disasm').write_text(dis)
    op='v_mmac_i32_16x16x32_i8' if rec['input_dtype']=='int8' else 'v_mmac_f32_16x16x16_bf16'
    assert op in dis
    assert ('v_exp_f32' in dis)==(rec['mode']=='fused')
    (d/'notes.txt').write_text(run(['/opt/dtk/aillvm/bin/llvm-readelf','--notes',str(d/'kernel.hsaco')]))
    rec.update(hsaco_sha256=hashlib.sha256((d/'kernel.hsaco').read_bytes()).hexdigest())
    (d/'plan.json').write_text(json.dumps(rec,indent=2)+'\n')
    return rec
def layout_proof(cfg):
    out=collections.Counter()
    for wave in range(4):
        for lane in range(64):
            rows=cfg['PF'] if cfg['direction']=='M' else cfg['SF']
            pairs=cfg['SF']//2 if cfg['direction']=='M' else 1
            for rr in range(rows):
                row=lane%16+rr*16+(wave*cfg['PF']*16 if cfg['direction']=='M' else 0)
                for pp in range(pairs):
                    for e in range(4):
                        col=(lane//16)*4+e+pp*16+(wave*16 if cfg['direction']=='N' else 0)
                        out[row,col]+=1
    assert len(out)==cfg['BM']*64 and set(out.values())=={1}
    assert set(out)=={(i,j) for i in range(cfg['BM']) for j in range(64)}
    return {'family':cfg['base_family'],'output_cells':len(out),'coverage_exact_once':True,'wave_N':32 if cfg['direction']=='N' else 128}
def main():
    (R/'common_gfx936.inc').write_text((BASE/'common_gfx936.inc').read_text())
    configs=[('M',128,0,'legacy'),('M',192,0,'private_first'),('M',256,0,'private_first')]
    configs += [('N',bm,0,'legacy') for bm in [128,144,160,176,192]]
    configs += [('N',208,k,'split_private_then_shared') for k in [5,7]]
    configs += [('N',224,k,'split_private_then_shared') for k in [9,10]]
    configs += [('N',240,12,'split_private_then_shared'),('N',256,11,'split_private_then_shared')]
    records=[];proof=[]
    for direction,bm,rf,po in configs:
        for precision in ['int8','bf16']:
            for mode in ['fused','debug']:
                source,rec=render(direction,bm,rf,'both',po,precision,mode)
                records.append(emit(source,rec))
                if precision=='bf16' and mode=='fused':proof.append(layout_proof(rec))
                print('BUILT',rec['family'],flush=True)
    (R/'kernels/manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    (R/'layout-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('BUILD_DONE',len(records),flush=True)
if __name__=='__main__':main()
