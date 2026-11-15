"""Fuse the saved small-M E047 and split-K E030 paths without changing their K loops."""
from pathlib import Path
import json,re,hashlib,yaml
import fused_generator as g
R=g.R;N,K,H=g.N,g.K,g.H
def legacy_ep(family,groups,precision,mode,*,sw,sx,tmp,nan,row,col,colbyte,addr,mreg,yreg,row_base=None,wave=None):
    code=['s_waitcnt vmcnt(0)',f'v_and_b32_e32 v{row}, 15, v0',f'v_lshrrev_b32_e32 v{col}, 4, v0',f'v_and_b32_e32 v{col}, 3, v{col}',f'v_lshlrev_b32_e32 v{col}, 2, v{col}']
    if family=='E030':
        code += [f'v_add_u32_e32 v{row}, s5, v{row}',f'v_lshlrev_b32_e32 v{colbyte}, 2, v{col}',
                 's_lshl_b32 s2, s4, 2',f'v_add_u32_e32 v{colbyte}, s2, v{colbyte}',
                 's_lshr_b32 s2, s4, 1',f'v_add_u32_e32 v{col}, s2, v{col}']
    else:
        code += [f'v_lshlrev_b32_e32 v{colbyte}, 2, v{col}','s_lshl_b32 s2, s6, 9',f'v_add_u32_e32 v{colbyte}, s2, v{colbyte}',
                 's_lshl_b32 s2, s6, 6',f'v_add_u32_e32 v{col}, s2, v{col}']
        if wave is not None:code += [f'v_lshlrev_b32_e32 v{tmp}, 5, v{wave}',f'v_add_u32_e32 v{row}, v{tmp}, v{row}']
        else:code += [f'v_add_u32_e32 v{row}, v{row_base}, v{row}']
    if precision=='int8':
        for j in range(len(groups[0])):code += [f'global_load_dwordx4 v[{sw+4*j}:{sw+4*j+3}], v{colbyte}, s[26:27] offset:{j*64}']
    code += [f'v_mov_b32_e32 v{nan}, 0x7fff0000']
    for i,accs in enumerate(groups):
        skip=f'.L_legacy_fuse_skip_{family}_{i}'
        code += [f'v_cmp_gt_u32_e32 vcc, {mreg}, v{row}','s_and_saveexec_b64 s[2:3], vcc',f's_cbranch_execz {skip}']
        if precision=='int8':code += [f'v_lshlrev_b32_e32 v{tmp}, 2, v{row}',f'global_load_dword v{sx}, v{tmp}, s[24:25]']
        code += [f'v_mul_u32_u24_e32 v{tmp}, {H if mode=="fused" else N}, v{row}',f'v_add_u32_e32 v{tmp}, v{col}, v{tmp}',
                 f'v_lshlrev_b32_e32 v{tmp}, 1, v{tmp}',f'v_add_co_u32_e32 v{addr}, vcc, s{yreg}, v{tmp}',
                 f'v_mov_b32_e32 v{addr+1}, s{yreg+1}',f'v_addc_co_u32_e32 v{addr+1}, vcc, 0, v{addr+1}, vcc']
        if precision=='int8':code += ['s_waitcnt vmcnt(0)']
        for j in range(0,len(accs),2):
            gate,up=accs[j:j+2]
            if precision=='int8':
                for start,scale in [(gate,sw+4*j),(up,sw+4*(j+1))]:
                    for e in range(4):code += [f'v_cvt_f32_i32 v{start+e}, v{start+e}',f'v_mul_f32_e32 v{tmp}, v{sx}, v{scale+e}',f'v_mul_f32_e32 v{start+e}, v{tmp}, v{start+e}']
            if mode=='fused':code += g.swiglu(gate,up,tmp,nan)+g.store4(gate,tmp,nan,addr,(j//2)*32)
            else:
                code += g.store4(gate,tmp,nan,addr,(j//2)*32)
                code += [f'v_add_co_u32_e32 v{addr}, vcc, {H*2}, v{addr}',f'v_addc_co_u32_e32 v{addr+1}, vcc, 0, v{addr+1}, vcc']
                code += g.store4(up,tmp,nan,addr,(j//2)*32)
                code += [f'v_add_co_u32_e32 v{addr}, vcc, {hex((-H*2)&0xffffffff)}, v{addr}',f'v_addc_co_u32_e32 v{addr+1}, vcc, -1, v{addr+1}, vcc']
        code += [skip+':','s_or_b64 exec, exec, s[2:3]']
        if i+1<len(groups):code += [f'v_add_u32_e32 v{row}, 16, v{row}']
    return '\n'.join(code)+'\n'
def render(family,precision,mode):
    base=R/('baseline/int8/aligned-bf16/bf16_kernels' if precision=='int8' else 'baseline/bf16/bf16-gemm/kernels')
    path=base/(family if precision=='int8' else family+'_BF16_BF16')/('bf16.s' if precision=='int8' else 'kernel.s')
    source=path.read_text();original=source
    if family=='E047':
        args=[('.LBB0_4:',[[57,61,65,69,53,49,41,33],[29,25,21,17,13,9,5,1]],dict(family='E047_fixed',sw=84,sx=116,tmp=117,nan=82,row=120,col=121,colbyte=122,addr=118,mreg='128',yreg=8,wave=83)),
              ('.LBB1_12:',[[77,69,61,57,53,49,45,41],[37,33,21,17,13,9,5,1]],dict(family='E047_generic',sw=86,sx=118,tmp=119,nan=82,row=122,col=123,colbyte=124,addr=120,mreg='s0',yreg=16,row_base=85))]
        for marker,groups,kw in args:
            start=source.index(marker);end=source.index('s_endpgm',start)
            source=source[:start]+marker+'\n'+legacy_ep(groups=groups,precision=precision,mode=mode,**kw)+source[end:]
    else:
        anchor=source.index('; %bb.9:')
        start=source.index('s_waitcnt vmcnt(0)\nv_and_b32_e32 v51, 15, v0',anchor)
        end=source.index('.LBB0_13:',start)
        source=source[:start]+legacy_ep('E030',[[17,21,25,29],[5,9,13,1]],precision,mode,sw=33,sx=49,tmp=56,nan=50,row=51,col=52,colbyte=53,addr=54,mreg='s8',yreg=12)+source[end:]
    doc=source.split('.amdgpu_metadata',1)[1].split('.end_amdgpu_metadata',1)[0]
    meta=yaml.safe_load(doc.expandtabs(8));newnames=[]
    for entry in meta['amdhsa.kernels']:
        old=entry['.name']
        fixed=('Li128ELi17408ELi5120' in old or '_fixed_' in old)
        name=family+'_'+precision.upper()+'_'+mode.upper()+('_fixed' if fixed else '_generic')
        source=re.sub(r'\b'+re.escape(old)+r'\b',name,source)
        rec=dict(base_family=family+('_fixed' if fixed else ''),family=family+'_'+precision.upper()+'_'+mode.upper(),function=name,
                 BM=128 if family=='E047' else 32,BN=128 if family=='E047' else 64,NW=4 if family=='E047' else 2,
                 VGPR=entry['.vgpr_count'],SGPR=entry['.sgpr_count'],LDS=entry['.group_segment_fixed_size'],scratch=0,
                 direction='M' if family=='E047' else 'splitK',input_dtype=precision,output_dtype='bf16',mode=mode,
                 N=N,K=K,H=H,output_N=H if mode=='fused' else N,interleaved_W='G16,U16',
                 intermediate_rounding='BF16 RNE before SiLU',extra_VGPR=0,extra_LDS=0,
                 fixed_M=128 if fixed else None,legacy_source=str(path),source_sha256=hashlib.sha256(original.encode()).hexdigest())
        newnames.append(rec)
    return source,newnames
def main():
    old=json.loads((R/'kernels/manifest.json').read_text())
    records=[z for z in old if not z['base_family'].startswith(('E047','E030'))]
    for family in ['E047','E030']:
        for precision in ['int8','bf16']:
            for mode in ['fused','debug']:
                source,recs=render(family,precision,mode)
                first=g.emit(source,recs[0]);records.append(first)
                for rec in recs[1:]:rec['hsaco_sha256']=first['hsaco_sha256'];records.append(rec)
                print('LEGACY_BUILT',family,precision,mode,flush=True)
    (R/'kernels/manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print('LEGACY_DONE',len(records),flush=True)
if __name__=='__main__':main()
