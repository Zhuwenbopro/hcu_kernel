// One half of a contiguous BF16 LM head, split into two address ranges.
.amdgcn_target "amdgcn-amd-amdhsa--gfx938:sramecc+"

.set GROUP_BYTES, 4096
.set STEP_BYTES, BLOCKS * GROUP_BYTES
.set STAGES, 2 * (TOTAL_BYTES / (2 * STEP_BYTES))
.set TAIL_BLOCKS, (TOTAL_BYTES / (2 * GROUP_BYTES)) % BLOCKS
.if (TOTAL_BYTES % (2 * GROUP_BYTES)) != 0
  .error "payload must consist of two whole CTA-granularity ranges"
.endif
.if DEPTH == 8
  .set UNROLL, 64
.elseif DEPTH == 12
  .set UNROLL, 48
.else
  .error "unsupported rolling load depth"
.endif
.set LOOPS, STAGES / UNROLL
.set TAIL, STAGES % UNROLL
.if LOOPS < 1
  .error "payload too small for the unrolled main loop"
.endif

// A wave covers 1 KiB. The oldest VGPR group becomes reusable when at most
// DEPTH-1 loads remain in flight. UNROLL is divisible by DEPTH and by two.
.macro read_stage
  s_waitcnt vmcnt(DEPTH - 1)
  .set dst, 4 + 4 * (stage % DEPTH)
  .if stage % 2 == 0
    global_load_dwordx4 v[dst:dst+3], v1, s[8:9] glc slc
    s_add_u32 s8, s8, STEP_BYTES
    s_addc_u32 s9, s9, 0
  .else
    global_load_dwordx4 v[dst:dst+3], v1, s[12:13] glc slc
    s_add_u32 s12, s12, STEP_BYTES
    s_addc_u32 s13, s13, 0
  .endif
  .set stage, stage + 1
.endm

.text
.protected read_xr
.globl read_xr
.p2align 8
.type read_xr,@function
read_xr:
  .if TIMESTAMPS
    s_getreg_b32 s20, hwreg(HW_REG_HW_ID)
    s_memrealtime s[16:17]
    s_load_dwordx2 s[14:15], s[4:5], 16
  .endif
  s_load_dwordx2 s[8:9], s[4:5], 0
  s_load_dwordx2 s[12:13], s[4:5], 8
  s_lshl_b32 s10, s6, 12
  v_lshlrev_b32 v1, 4, v0
  s_waitcnt lgkmcnt(0)
  s_add_u32 s8, s8, s10
  s_addc_u32 s9, s9, 0
  s_add_u32 s12, s12, s10
  s_addc_u32 s13, s13, 0

  s_mov_b32 s11, LOOPS
.Lread_loop:
  .set stage, 0
  .rept UNROLL
    read_stage
  .endr
  s_sub_u32 s11, s11, 1
  s_cmp_lg_u32 s11, 0
  s_cbranch_scc1 .Lread_loop

  .set stage, 0
  .rept TAIL
    read_stage
  .endr

  // Assign the final partial stage to the first TAIL_BLOCKS workgroups.
  .if TAIL_BLOCKS
    s_cmp_lt_u32 s6, TAIL_BLOCKS
    s_cbranch_scc0 .Lafter_partial_stage
    s_waitcnt vmcnt(0)
    global_load_dwordx4 v[4:7], v1, s[8:9] glc slc
    global_load_dwordx4 v[8:11], v1, s[12:13] glc slc
.Lafter_partial_stage:
  .endif
  s_waitcnt vmcnt(0)

  // The separate instrumented build emits 32-byte records per wave. Only
  // lane zero writes, after the timed reads have completed.
  .if TIMESTAMPS
    s_memrealtime s[18:19]
    s_waitcnt lgkmcnt(0)
    v_and_b32 v2, 63, v0
    v_cmp_eq_u32_e64 s[22:23], 0, v2
    s_and_b64 exec, exec, s[22:23]
    s_cbranch_execz .Ldone
    v_readfirstlane_b32 s24, v0
    s_lshr_b32 s24, s24, 1
    s_lshl_b32 s21, s6, 7
    s_add_u32 s21, s21, s24
    v_mov_b32 v1, s21
    v_mov_b32 v4, s16
    v_mov_b32 v5, s17
    v_mov_b32 v6, s18
    v_mov_b32 v7, s19
    global_store_dwordx4 v1, v[4:7], s[14:15]
    v_mov_b32 v8, s20
    global_store_dword v1, v8, s[14:15] offset:16
    s_waitcnt vmcnt(0)
.Ldone:
  .endif
  s_endpgm
.Lend:
.size read_xr, .Lend-read_xr

.section .rodata
.p2align 6
.amdhsa_kernel read_xr
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size @KERNARG_SIZE@
  .amdhsa_user_sgpr_count 6
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr @VGPRS@
  .amdhsa_next_free_sgpr @NEXT_SGPR@
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
@TIMESTAMP_ARG@    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: @KERNARG_SIZE@
    .language: OpenCL C
    .language_version: [2, 0]
    .max_flat_workgroup_size: 256
    .name: read_xr
    .private_segment_fixed_size: 0
    .sgpr_count: @SGPRS@
    .sgpr_spill_count: 0
    .symbol: read_xr.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: @VGPRS@
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target: 'amdgcn-amd-amdhsa--gfx938:sramecc+'
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
