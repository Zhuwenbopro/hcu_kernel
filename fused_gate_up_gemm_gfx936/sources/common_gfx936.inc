; gfx936 INT8 GEMM shared assembly primitives, ABI v1.
; Explicit physical operands; inline expansion, no calls.
; No implicit waits, barriers, address changes, scratch registers or m0 writes.
; GLOBAL may hit a cache; HBM source is determined by counters.
; GLOBAL_TO_LDS requires caller-prepared descriptor/offset/m0 and slot readiness.
; LDS_TO_REG requires caller-owned destination and lgkm readiness.
; MMA takes matching packed K32 halves; caller selects accumulator registers.
; EPILOGUE requires valid coordinate/address and caller row mask.

.macro GEMM_GLOBAL_TO_REG dst, addr
    global_load_dwordx4 \dst, \addr, off
.endm

.macro GEMM_W_GLOBAL_TO_REG dst, addr
    GEMM_GLOBAL_TO_REG \dst, \addr
.endm

.macro GEMM_X_GLOBAL_TO_REG dst, addr
    GEMM_GLOBAL_TO_REG \dst, \addr
.endm

.macro GEMM_GLOBAL_TO_LDS vaddr, srd, soffset
    buffer_load_dwordx4 \vaddr, \srd, \soffset offen lds
.endm

.macro GEMM_W_GLOBAL_TO_LDS vaddr, srd, soffset
    GEMM_GLOBAL_TO_LDS \vaddr, \srd, \soffset
.endm

.macro GEMM_X_GLOBAL_TO_LDS vaddr, srd, soffset
    GEMM_GLOBAL_TO_LDS \vaddr, \srd, \soffset
.endm

.macro GEMM_LDS_TO_REG dst, addr, offset=0
    ds_read_b128 \dst, \addr offset:\offset
.endm

.macro GEMM_MMA_I8 acc_out, x, w, acc_in
    v_mmac_i32_16x16x32_i8 \acc_out, \x, \w, \acc_in
.endm

.macro GEMM_EPILOGUE_STORE addr, src, offset=0
    global_store_dwordx4 \addr, \src, off offset:\offset
.endm

.macro GEMM_EPILOGUE_STORE_GLC_SLC addr, src, offset=0
    global_store_dwordx4 \addr, \src, off offset:\offset glc slc
.endm

.macro GEMM_WAIT_VM count
    s_waitcnt vmcnt(\count)
.endm

.macro GEMM_WAIT_LGKM count
    s_waitcnt lgkmcnt(\count)
.endm

.macro GEMM_CTA_BARRIER
    s_barrier
.endm

; Scalar64 base plus lane-varying32 offset; no implicit pointer advance.
.macro GEMM_GLOBAL_SADDR_TO_REG dst, voffset, saddr, offset=0
    global_load_dwordx4 \dst, \voffset, \saddr offset:\offset
.endm

.macro GEMM_W_GLOBAL_SADDR_TO_REG dst, voffset, saddr, offset=0
    GEMM_GLOBAL_SADDR_TO_REG \dst, \voffset, \saddr, \offset
.endm

.macro GEMM_X_GLOBAL_SADDR_TO_REG dst, voffset, saddr, offset=0
    GEMM_GLOBAL_SADDR_TO_REG \dst, \voffset, \saddr, \offset
.endm

.macro GEMM_MMA_BF16 dst, a, b, acc
 v_mmac_f32_16x16x16_bf16 \dst, \a, \b, \acc
.endm
