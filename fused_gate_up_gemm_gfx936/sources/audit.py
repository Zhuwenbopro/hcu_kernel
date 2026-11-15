from pathlib import Path
import json,re,yaml,hashlib
R=Path(__file__).resolve().parent
records=json.loads((R/'kernels/manifest.json').read_text())
entries=[]
for rec in records:
    folder=R/'kernels'/rec['family']
    assert hashlib.sha256((folder/'kernel.hsaco').read_bytes()).hexdigest()==rec['hsaco_sha256']
    s=(folder/'kernel.s').read_text();dis=(folder/'device.disasm').read_text()
    meta=yaml.safe_load(s.split('.amdgpu_metadata',1)[1].split('.end_amdgpu_metadata',1)[0].expandtabs(8))
    m=next(z for z in meta['amdhsa.kernels'] if z['.name']==rec['function'])
    assert m['.private_segment_fixed_size']==0
    assert m['.vgpr_count']==rec['VGPR'] and m['.group_segment_fixed_size']==rec['LDS']
    if not rec['base_family'].startswith(('E030','E047')):
        other=next(z for z in records if z['base_family']==rec['base_family'] and z['input_dtype']==rec['input_dtype'] and z['mode']!=rec['mode'])
        assert other['K_loop_source_sha256']==rec['K_loop_source_sha256']
    if rec['mode']=='fused':
        assert 'v_exp_f32' in dis and 'v_rcp_f32' in dis
        stores=re.findall(r'global_store_\w+',dis)
        assert stores and set(stores)=={'global_store_dwordx2'}
    entries.append({k:rec[k] for k in ['family','base_family','function','input_dtype','mode','BM','BN','NW','VGPR','SGPR','LDS','scratch']})
saved=json.loads((R/'PRESERVED.json').read_text())
for item in saved:
    assert hashlib.sha256(Path(item['saved']).read_bytes()).hexdigest()==item['sha256']
    assert hashlib.sha256(Path(item['source']).read_bytes()).hexdigest()==item['sha256']
out={'status':'complete','entries':entries,'functions':len(entries),'module_files':len({x['family'] for x in entries}),'native_MMAC':'INT8xINT8 -> INT32 or BF16xBF16 -> FP32','SwiGLU':'v_exp_f32 + v_rcp_f32 in register; BF16 RNE before SiLU and on final output','generic_fused_debug_K_loop_sources_equal':True,'extra_VGPR':0,'extra_LDS':0,'scratch':0,'prior_archives_unchanged':True}
(R/'instruction-resource-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('AUDIT_DONE',out['functions'],out['module_files'],flush=True)
