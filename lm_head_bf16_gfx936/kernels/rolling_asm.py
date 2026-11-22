"""Generate the verified gfx936 wave64 rolling-MMAC assembly."""

from dataclasses import dataclass
import itertools
import math

K, N = 5120, 124160
ROW_W, ROW_X = N*2, K*2
TARGET = "amdgcn-amd-amdhsa--gfx936:sramecc+"


@dataclass(frozen=True)
class Config:
    bm: int
    bn: int
    nw: int
    wm: int
    depth: int
    cache: int = 3
    bk: int = 32
    xt: int = 0
    wpad: int = 0

    @property
    def name(self):
        return f"m{self.bm}-n{self.bn}-w{self.nw}x{self.wm}-d{self.depth}-c{self.cache}-k{self.bk}-x{self.xt}"+(f"-p{self.wpad}" if self.wpad else "")

    @property
    def wn(self):
        return self.nw // self.wm

    @property
    def mb(self):
        return self.bm // (16 * self.wm)

    @property
    def nb(self):
        return self.bn // (32 * self.wn)

    @property
    def slot(self):
        return self.wstage+self.xwaves*1024

    @property
    def wstage(self):
        return self.wloads*self.nw*(1024+self.wpad)

    @property
    def lds(self):
        return self.depth * self.slot

    @property
    def vgprs(self):
        return 24 + 8*self.mb*self.nb + (4*self.nb+2*self.mb)*(self.bk//16)

    @property
    def wloads(self):
        return self.bn*self.bk // (self.nw*512)

    @property
    def xwaves(self):
        return math.ceil(self.bm*self.bk / 512)

    @property
    def xrows(self):
        return 512//self.bk


def lg(n):
    assert n > 0 and n & (n-1) == 0, n
    return n.bit_length()-1


def metadata(symbol, q):
    types = ("global_buffer", "global_buffer", "global_buffer") + ("by_value",)*4
    groups = []
    for i,t in enumerate(types):
        isptr = i < 3
        size = 8 if isptr else 4
        offset = i*8 if isptr else 24+(i-3)*4
        addr = "      - .address_space: global\n        .offset:" if isptr else "      - .offset:"
        groups.append(f"{addr} {offset}\n        .size: {size}\n        .value_kind: {t}")
    return f"""
.section .rodata
.p2align 6
.amdhsa_kernel {symbol}
  .amdhsa_group_segment_fixed_size {q.lds}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 6
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr {q.vgprs}
  .amdhsa_next_free_sgpr 43
  .amdhsa_reserve_flat_scratch 0
  .amdhsa_reserve_xnack_mask 1
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_dx10_clamp 1
  .amdhsa_ieee_mode 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
{chr(10).join(groups)}
    .group_segment_fixed_size: {q.lds}
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language: OpenCL C
    .language_version: [2, 0]
    .max_flat_workgroup_size: {q.nw*64}
    .name: {symbol}
    .private_segment_fixed_size: 0
    .sgpr_count: 47
    .sgpr_spill_count: 0
    .symbol: {symbol}.kd
    .uniform_work_group_size: 1
    .vgpr_count: {q.vgprs}
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target: '{TARGET}'
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
"""


def render(q):
    assert q.wloads and q.nb and q.mb
    assert q.lds <= 65536 and q.vgprs <= 256
    sym = "rolling_mmac"
    acc = 24
    wf = acc + 8*q.mb*q.nb
    k_halves = q.bk//16
    xf = wf + 4*q.nb*k_halves
    assert xf + 2*q.mb*k_halves == q.vgprs
    x_base = q.wstage
    w_lds_k16=(q.bn//32)*(1024+q.wpad)
    x_increment = q.bm*2*q.bk if q.xt else q.bk*2
    lines = []
    def w(*strings):
        lines.extend(strings)
    def buf(x, offset, stage_w, cache):
        w(f"s_add_u32 m0, s{31 if stage_w and q.wpad else 30}, {offset}")
        desc, step, vec = ("[16:19]", "s32", "v9") if stage_w else ("[20:23]", "s33", "v10")
        if x:
            # One vector per lane advances by an entire block of lanes.
            addr_delta = q.nw*1024 if q.xt else q.nw*q.xrows*ROW_X
            if stage_w:
                addr_delta = (q.nw*64 // (q.bn//8))*ROW_W
            w(f"v_add_u32_e32 v11, {x*addr_delta}, {vec}")
            vec = "v11"
        w(f"buffer_load_dwordx4 {vec}, s{desc}, {step} offen{cache} lds")
    def x_issue(slot):
        rounds, tail = divmod(q.xwaves, q.nw)
        offset = slot*q.slot + x_base
        for x in range(rounds):
            buf(x, offset+x*q.nw*1024, False, "")
        if tail:
            w(f"s_cmp_lt_u32 s29, {tail}", f"s_cbranch_scc0 .Lskip_x{issue.num}")
            buf(rounds, offset+rounds*q.nw*1024, False, "")
            w(f".Lskip_x{issue.num}:")
    def issue(slot):
        w(f".Lissue{issue.num}:")
        for x in range(q.wloads):
            buf(x, slot*q.slot+x*q.nw*(1024+q.wpad), True, " glc slc" if q.cache else "")
        x_issue(slot)
        w("s_waitcnt vmcnt(63)")
        w("s_add_u32 s32, s32, s40", f"s_add_u32 s33, s33, {x_increment}")
        issue.num += 1
    issue.num = 0
    def ready(future):
        w(f".Lwait{ready.num}:")
        base, tail = divmod(q.xwaves,q.nw)
        count = future * (q.wloads+base)
        if future and tail:
            w(f"s_cmp_lt_u32 s29, {tail}", f"s_cbranch_scc0 .Lwait_w{ready.num}",
              f"s_waitcnt vmcnt({count+future})", f"s_branch .Lbar{ready.num}",
              f".Lwait_w{ready.num}:")
        w(f"s_waitcnt vmcnt({count})", f".Lbar{ready.num}:", "s_barrier")
        ready.num += 1
    ready.num = 0
    def read_compute(slot):
        origin = 0 if slot is None else slot*q.slot
        w_addr,x_addr = ("v17","v18") if slot is None else ("v12","v13")
        w(f"s_mov_b32 m0, s41")
        for step in range(k_halves):
            for j in range(q.nb):
                dest = wf+4*(k_halves*j+step)
                off = origin+64*j+w_lds_k16*step
                w(f"ds_read_m32x16_b16 v[{dest}:{dest+3}], {w_addr} offset:{off}")
            if q.xt:
                for i in range(0,q.mb,2):
                    dest=xf+2*k_halves*i+4*step
                    off=origin+x_base+i*32+q.bm*32*step
                    w(f"ds_read_m32x16_b16 v[{dest}:{dest+3}], {x_addr} offset:{off}")
            else:
                for i in range(q.mb):
                    dest=xf+2*(k_halves*i+step)
                    off=origin+x_base+i*q.bk*32+step*32
                    w(f"ds_read_b64 v[{dest}:{dest+1}], {x_addr} offset:{off}")
            w("s_waitcnt lgkmcnt(0)")
            for i,j,half in itertools.product(range(q.mb),range(q.nb),range(2)):
                ag = acc + 8*(i*q.nb+j)+4*half
                wa = wf+4*(k_halves*j+step)+2*half
                xa = (xf+4*(k_halves*(i//2)+step)+2*(i%2) if q.xt
                      else xf+2*(k_halves*i+step))
                w(f"v_mmac_f32_16x16x16_bf16 v[{ag}:{ag+3}], v[{wa}:{wa+1}], v[{xa}:{xa+1}], v[{ag}:{ag+3}]")
    x_arg = (["v_lshlrev_b32_e32 v10, 4, v0"] if q.xt else
             [f"v_lshrrev_b32_e32 v7, {lg(q.bk//8)}, v0",
              f"v_and_b32_e32 v8, {q.bk//8-1}, v0",
              f"v_mul_u32_u24_e32 v10, {ROW_X}, v7",
              "v_lshlrev_b32_e32 v8, 4, v8","v_add_u32_e32 v10, v10, v8"])
    pow2_m = q.bm & (q.bm-1) == 0
    x_ds = ([(f"v_lshlrev_b32_e32 v13, {lg(q.bm*2)}, v7" if pow2_m else
              f"v_mul_u32_u24_e32 v13, {q.bm*2}, v7"),"v_add_u32_e32 v13, v13, v8",
             f"v_mul_u32_u24_e32 v8, {q.mb*32}, v4","v_add_u32_e32 v13, v13, v8"] if q.xt else
            [f"v_mul_u32_u24_e32 v13, {q.mb*16}, v4","v_add_u32_e32 v13, v13, v5",
             f"v_lshlrev_b32_e32 v13, {lg(q.bk*2)}, v13","v_lshlrev_b32_e32 v8, 3, v6",
             "v_add_u32_e32 v13, v13, v8"])
    w(f'.amdgcn_target "{TARGET}"', '.text', f'.protected {sym}', f'.globl {sym}',
      '.p2align 8', f'.type {sym},@function', f'{sym}:',
      's_load_dwordx4 s[8:11], s[4:5], 0',
      's_load_dwordx4 s[12:15], s[4:5], 16',
      's_load_dword s24, s[4:5], 32',
      's_load_dword s25, s[4:5], 36',
      'v_and_b32_e32 v1, 63, v0','v_lshrrev_b32_e32 v2, 6, v0',
      f'v_and_b32_e32 v3, {q.wn-1}, v2',f'v_lshrrev_b32_e32 v4, {lg(q.wn)}, v2',
      'v_and_b32_e32 v5, 15, v1','v_lshrrev_b32_e32 v6, 4, v1',
      'v_readfirstlane_b32 s29, v2','s_lshl_b32 s30, s29, 10',
      # Loader W v9 = row * global stride + vector_in_row * 16.
      f'v_lshrrev_b32_e32 v7, {lg(q.bn//8)}, v0',f'v_and_b32_e32 v8, {q.bn//8-1}, v0',
      f'v_mul_u32_u24_e32 v9, {ROW_W}, v7','v_lshlrev_b32_e32 v8, 4, v8',
      'v_add_u32_e32 v9, v9, v8',
      *x_arg,
      # DS matrix read accepts 8 contiguous BF16 per source lane.
      'v_lshrrev_b32_e32 v7, 2, v1','v_and_b32_e32 v8, 3, v1',
      f'v_lshlrev_b32_e32 v12, {lg(q.bn*2)}, v7','v_lshlrev_b32_e32 v8, 4, v8',
      'v_add_u32_e32 v12, v12, v8',
      *x_ds,
      *([f"v_lshrrev_b32_e32 v23, {lg(512//q.bn)}, v7",
         f"v_mul_u32_u24_e32 v23, {q.wpad}, v23",
         "v_add_u32_e32 v12, v12, v23",
         f"s_mul_i32 s31, s29, {1024+q.wpad}"] if q.wpad else []),
      f'v_lshlrev_b32_e32 v8, {lg(q.nb*64)}, v3',
      'v_add_u32_e32 v12, v12, v8',
      's_waitcnt lgkmcnt(0)',
      # W and X SRDs. The bound on X zeros the padded rows in LDS.
      's_mov_b64 s[16:17], s[10:11]',f's_mov_b32 s18, {K*ROW_W}','s_mov_b32 s19, 0x20000',
      's_mov_b64 s[20:21], s[8:9]',
      (f's_mov_b32 s22, {K*q.bm*2}' if q.xt else f's_mul_i32 s22, s14, {ROW_X}'),
      's_mov_b32 s23, 0x20000',
      f's_lshl_b32 s26, s6, {lg(q.bn)}',
      (f's_mul_i32 s33, s15, {q.bm*2}' if q.xt and not pow2_m else
       f's_lshl_b32 s33, s15, {lg(q.bm*2) if q.xt else 1}'),
      f's_mul_i32 s32, s15, {ROW_W}', 's_lshl_b32 s27, s26, 1', 's_add_u32 s32, s32, s27',
      f's_sub_u32 s34, s24, {q.depth-1}',f's_mov_b32 s40, {q.bk*ROW_W}', f's_mov_b32 s41, {q.lds}')
    for slot in range(q.depth-1):
        issue(slot)
    for a in range(acc,wf):
        w(f"v_mov_b32_e32 v{a}, 0")
    w(".p2align 5")
    for slot in range(q.depth):
        w(f".Lroll{slot}:")
        ready(q.depth-2)
        issue((slot+q.depth-1)%q.depth)
        read_compute(slot)
        w("s_sub_u32 s34, s34, 1", "s_cmp_eq_u32 s34, 0")
        if slot+1 < q.depth:
            w(f"s_cbranch_scc1 .Ldrain{slot+1}")
        else:
            w("s_cbranch_scc0 .Lroll0")
    for slot in range(q.depth):
        w(f".Ldrain{slot}:", f"s_mov_b32 s35, {slot*q.slot}")
        if slot+1 < q.depth:
            w("s_branch .Ldrain")
    w(".Ldrain:")
    ready(0)
    w(f"s_mov_b32 s34, {q.depth-1}", ".Ltail:",
      "v_add_u32_e32 v17, s35, v12","v_add_u32_e32 v18, s35, v13")
    read_compute(None)
    w(f"s_add_u32 s35, s35, {q.slot}", f"s_cmp_eq_u32 s35, {q.lds}",
      "s_cselect_b32 s35, 0, s35", "s_sub_u32 s34, s34, 1",
      "s_cmp_lg_u32 s34, 0", "s_cbranch_scc1 .Ltail")

    w("v_add_u32_e32 v16, s26, v5",
      f"v_lshlrev_b32_e32 v8, {lg(q.nb*32)}, v3", "v_add_u32_e32 v16, v16, v8",
      "v_mov_b32_e32 v14, v6", f"v_mul_u32_u24_e32 v8, {q.mb*16}, v4",
      "v_add_u32_e32 v14, v14, v8", "s_mov_b32 s39, 0x7fff","v_mov_b32_e32 v21, 0x7fc0")
    for partial in (True,False):
        size = 4 if partial else 2
        suffix = "partial" if partial else "bf16"
        if not partial:
            w(".Lstore_bf16:")
        else:
            w("s_cmp_eq_u32 s25, 0","s_cbranch_scc1 .Lstore_bf16")
        for i,e in itertools.product(range(q.mb),range(4)):
            label=f".Lskip_{suffix}_{i}_{e}"
            w(f"v_add_u32_e32 v15, {i*16+4*e}, v14","v_cmp_gt_u32_e32 vcc, s14, v15",
              "s_and_saveexec_b64 s[36:37], vcc",f"s_cbranch_execz {label}",
              f"v_mul_u32_u24_e32 v17, {N*size}, v15",
              f"v_lshlrev_b32_e32 v18, {lg(size)}, v16","v_add_u32_e32 v18, v18, v17")
            for j,half in itertools.product(range(q.nb),range(2)):
                a=acc+8*(i*q.nb+j)+4*half+e
                offset=(j*32+half*16)*size
                if partial:
                    w(f"global_store_dword v18, v{a}, s[12:13] offset:{offset}")
                else:
                    w(f"v_cmp_u_f32_e32 vcc, v{a}, v{a}",f"v_bfe_u32 v19, v{a}, 16, 1",
                      f"v_add3_u32 v20, v{a}, v19, s39","v_lshrrev_b32_e32 v20, 16, v20",
                      "v_cndmask_b32_e32 v20, v20, v21, vcc",
                      f"global_store_short v18, v20, s[12:13] offset:{offset}")
            w(f"{label}:","s_or_b64 exec, exec, s[36:37]")
        w("s_waitcnt vmcnt(0)","s_endpgm")
    w(".Lkernel_end:", f".size {sym}, .Lkernel_end-{sym}")
    return "\n".join(lines)+metadata(sym,q)
