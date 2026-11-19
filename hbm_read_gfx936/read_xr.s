// gfx936 / wave64. 80 CTAs of 256 threads read two 40 MiB buffers.
// Each pointer addresses exactly 128 contiguous stages of 327680 bytes.
.amdgcn_target "amdgcn-amd-amdhsa--gfx936:sramecc+"

.macro read_and_advance base, lo, hi
  .if CACHE == 3
    global_load_dwordx4 v[dst:dst+3], v1, \base glc slc
  .else
    global_load_dwordx4 v[dst:dst+3], v1, \base
  .endif
  s_add_u32 \lo, \lo, 327680
  s_addc_u32 \hi, \hi, 0
.endm

.text
.protected read_xr
.globl read_xr
.p2align 8
.type read_xr,@function
read_xr:
  s_load_dwordx2 s[8:9], s[4:5], 0
  s_load_dwordx2 s[12:13], s[4:5], 8
  s_lshl_b32 s10, s6, 12
  v_lshlrev_b32 v1, 4, v0
  s_waitcnt lgkmcnt(0)
  s_add_u32 s8, s8, s10
  s_addc_u32 s9, s9, 0
  s_add_u32 s12, s12, s10
  s_addc_u32 s13, s13, 0

  .set stage, 0
  .rept 256
    .if stage >= 8
      .if BATCH == 0
        s_waitcnt vmcnt(7)
      .elseif stage % 8 == 0
        s_waitcnt vmcnt(0)
      .else
        s_waitcnt vmcnt(63)
      .endif
    .endif
    .set dst, 4 + 4 * (stage % 8)
    .if (SERIAL == 0 && stage % 2 == 0) || (SERIAL == 1 && stage < 128)
      read_and_advance s[8:9], s8, s9
    .else
      read_and_advance s[12:13], s12, s13
    .endif
    .set stage, stage + 1
  .endr
  s_waitcnt vmcnt(0)
  s_endpgm
.Lend:
.size read_xr, .Lend-read_xr

.section .rodata
.p2align 6
.amdhsa_kernel read_xr
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_count 6
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 36
  .amdhsa_next_free_sgpr 14
  .amdhsa_reserve_flat_scratch 0
  .amdhsa_reserve_xnack_mask 1
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_dx10_clamp 1
  .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .offset: 8
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language: OpenCL C
    .language_version: [2, 0]
    .max_flat_workgroup_size: 256
    .name: read_xr
    .private_segment_fixed_size: 0
    .sgpr_count: 18
    .sgpr_spill_count: 0
    .symbol: read_xr.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 36
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target: 'amdgcn-amd-amdhsa--gfx936:sramecc+'
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
