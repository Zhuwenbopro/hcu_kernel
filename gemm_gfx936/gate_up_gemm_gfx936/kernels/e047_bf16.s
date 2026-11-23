; Shared primitive refactor. Original addresses/registers/schedule retained.
.include "common_gfx936.inc"
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx936:sramecc+"
	.section	.text._Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii,#alloc,#execinstr
	.protected	_Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii ; -- Begin function _Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii
	.globl	_Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii
	.p2align	8
	.type	_Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii,@function
_Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii: ; @_Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii
s_load_dwordx4 s[24:27], s[4:5], 0x28
; %bb.0:
	s_load_dwordx4 s[12:15], s[4:5], 0x0
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	v_lshrrev_b32_e32 v83, 6, v0
	v_lshlrev_b32_e32 v4, 11, v83
	v_and_b32_e32 v2, 63, v0
	s_mov_b32 s16, 0
	GEMM_WAIT_LGKM 0
	v_mov_b32_e32 v5, s13
	v_add_co_u32_e32 v3, vcc, s12, v4
	v_addc_co_u32_e32 v6, vcc, 0, v5, vcc
	v_lshlrev_b32_e32 v84, 4, v2
	s_mov_b32 s18, s16
	s_mov_b32 s19, s16
	v_add_co_u32_e32 v2, vcc, v3, v84
	s_mov_b32 s17, s16
	v_mov_b64_e32 v[39:40], s[18:19]
	v_addc_co_u32_e32 v3, vcc, 0, v6, vcc
	v_lshlrev_b32_e32 v6, 4, v0
	v_mov_b64_e32 v[37:38], s[16:17]
	v_mov_b64_e32 v[47:48], v[39:40]
	v_mov_b32_e32 v1, 0
	s_add_i32 s1, s15, 4.0
	s_mov_b32 s3, 0x20000
	s_brev_b32 s2, 1
	s_mov_b32 s0, s14
	v_mov_b64_e32 v[45:46], v[37:38]
	v_lshl_or_b32 v6, s6, 13, v6
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[45:48], v[2:3]
	;;#ASMEND
	s_movk_i32 s4, 0x400
	v_add_co_u32_e32 v2, vcc, s4, v2
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[37:40], v[2:3]
	;;#ASMEND
	v_and_b32_e32 v2, 0xc0, v0
	v_lshlrev_b32_e32 v85, 4, v2
	v_readfirstlane_b32 s4, v85
	v_or_b32_e32 v86, 0x1000, v85
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v86
	GEMM_W_GLOBAL_TO_LDS v6, s[0:3], 0
	v_or_b32_e32 v2, 0x1000, v6
	s_mov_b32 m0, s4
	s_movk_i32 s4, 0x4000
	GEMM_W_GLOBAL_TO_LDS v2, s[0:3], 0
	v_or_b32_e32 v2, v4, v84
	v_add_co_u32_e32 v2, vcc, s12, v2
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	v_add_co_u32_e32 v81, vcc, s4, v2
	v_addc_co_u32_e32 v82, vcc, 0, v3, vcc
	v_add_u32_e32 v87, 0x221000, v6
	s_mov_b32 s4, -2
	s_movk_i32 s5, 0xe000
	s_movk_i32 s7, 0xe400
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v8, v1
	v_mov_b32_e32 v9, v1
	v_mov_b32_e32 v10, v1
	v_mov_b32_e32 v11, v1
	v_mov_b32_e32 v12, v1
	v_mov_b32_e32 v13, v1
	v_mov_b32_e32 v14, v1
	v_mov_b32_e32 v15, v1
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v17, v1
	v_mov_b32_e32 v18, v1
	v_mov_b32_e32 v19, v1
	v_mov_b32_e32 v20, v1
	v_mov_b32_e32 v21, v1
	v_mov_b32_e32 v22, v1
	v_mov_b32_e32 v23, v1
	v_mov_b32_e32 v24, v1
	v_mov_b32_e32 v25, v1
	v_mov_b32_e32 v26, v1
	v_mov_b32_e32 v27, v1
	v_mov_b32_e32 v28, v1
	v_mov_b32_e32 v29, v1
	v_mov_b32_e32 v30, v1
	v_mov_b32_e32 v31, v1
	v_mov_b32_e32 v32, v1
	v_mov_b32_e32 v33, v1
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v35, v1
	v_mov_b32_e32 v36, v1
	v_mov_b32_e32 v41, v1
	v_mov_b32_e32 v42, v1
	v_mov_b32_e32 v43, v1
	v_mov_b32_e32 v44, v1
	v_mov_b32_e32 v49, v1
	v_mov_b32_e32 v50, v1
	v_mov_b32_e32 v51, v1
	v_mov_b32_e32 v52, v1
	v_mov_b32_e32 v53, v1
	v_mov_b32_e32 v54, v1
	v_mov_b32_e32 v55, v1
	v_mov_b32_e32 v56, v1
	v_mov_b32_e32 v57, v1
	v_mov_b32_e32 v58, v1
	v_mov_b32_e32 v59, v1
	v_mov_b32_e32 v60, v1
	v_mov_b32_e32 v61, v1
	v_mov_b32_e32 v62, v1
	v_mov_b32_e32 v63, v1
	v_mov_b32_e32 v64, v1
	v_mov_b32_e32 v65, v1
	v_mov_b32_e32 v66, v1
	v_mov_b32_e32 v67, v1
	v_mov_b32_e32 v68, v1
	v_mov_b32_e32 v69, v1
	v_mov_b32_e32 v70, v1
	v_mov_b32_e32 v71, v1
	v_mov_b32_e32 v72, v1
	v_mov_b32_e32 v73, v1
	v_mov_b32_e32 v74, v1
	v_mov_b32_e32 v75, v1
	v_mov_b32_e32 v76, v1
	v_mov_b32_e32 v77, v1
	v_mov_b32_e32 v78, v1
	v_mov_b32_e32 v79, v1
	v_mov_b32_e32 v80, v1
	v_or_b32_e32 v88, 0x2000, v85
	v_or_b32_e32 v89, 0x3000, v85
	v_or_b32_e32 v90, 0x2000, v84
	s_branch .LBB0_2
.LBB0_1:                                ;   in Loop: Header=BB0_2 Depth=1
	;;#ASMSTART
	GEMM_LDS_TO_REG v[91:94], v90, 0
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[95:98], v90, 0x400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[99:102], v90, 0x800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[103:106], v90, 0xc00
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[107:110], v90, 0x1000
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[111:114], v90, 0x1400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[115:118], v90, 0x1800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[119:122], v90, 0x1c00
	;;#ASMEND
	;;#ASMSTART
	GEMM_WAIT_LGKM 7
	;;#ASMEND
	GEMM_MMA_I8 v[57:60], v[77:78], v[91:92], v[57:60]
	GEMM_MMA_I8 v[29:32], v[73:74], v[91:92], v[29:32]
	GEMM_MMA_I8 v[57:60], v[79:80], v[93:94], v[57:60]
	GEMM_MMA_I8 v[29:32], v[75:76], v[93:94], v[29:32]
	;;#ASMSTART
	GEMM_WAIT_LGKM 6
	;;#ASMEND
	GEMM_MMA_I8 v[61:64], v[77:78], v[95:96], v[61:64]
	GEMM_MMA_I8 v[25:28], v[73:74], v[95:96], v[25:28]
	GEMM_MMA_I8 v[61:64], v[79:80], v[97:98], v[61:64]
	GEMM_MMA_I8 v[25:28], v[75:76], v[97:98], v[25:28]
	;;#ASMSTART
	GEMM_WAIT_LGKM 5
	;;#ASMEND
	GEMM_MMA_I8 v[65:68], v[77:78], v[99:100], v[65:68]
	GEMM_MMA_I8 v[21:24], v[73:74], v[99:100], v[21:24]
	GEMM_MMA_I8 v[65:68], v[79:80], v[101:102], v[65:68]
	GEMM_MMA_I8 v[21:24], v[75:76], v[101:102], v[21:24]
	;;#ASMSTART
	GEMM_WAIT_LGKM 4
	;;#ASMEND
	GEMM_MMA_I8 v[69:72], v[77:78], v[103:104], v[69:72]
	GEMM_MMA_I8 v[17:20], v[73:74], v[103:104], v[17:20]
	GEMM_MMA_I8 v[69:72], v[79:80], v[105:106], v[69:72]
	GEMM_MMA_I8 v[17:20], v[75:76], v[105:106], v[17:20]
	;;#ASMSTART
	GEMM_WAIT_LGKM 3
	;;#ASMEND
	GEMM_MMA_I8 v[53:56], v[77:78], v[107:108], v[53:56]
	GEMM_MMA_I8 v[13:16], v[73:74], v[107:108], v[13:16]
	GEMM_MMA_I8 v[53:56], v[79:80], v[109:110], v[53:56]
	GEMM_MMA_I8 v[13:16], v[75:76], v[109:110], v[13:16]
	;;#ASMSTART
	GEMM_WAIT_LGKM 2
	;;#ASMEND
	GEMM_MMA_I8 v[49:52], v[77:78], v[111:112], v[49:52]
	GEMM_MMA_I8 v[9:12], v[73:74], v[111:112], v[9:12]
	GEMM_MMA_I8 v[49:52], v[79:80], v[113:114], v[49:52]
	GEMM_MMA_I8 v[9:12], v[75:76], v[113:114], v[9:12]
	;;#ASMSTART
	GEMM_WAIT_LGKM 1
	;;#ASMEND
	GEMM_MMA_I8 v[41:44], v[77:78], v[115:116], v[41:44]
	GEMM_MMA_I8 v[5:8], v[73:74], v[115:116], v[5:8]
	GEMM_MMA_I8 v[41:44], v[79:80], v[117:118], v[41:44]
	GEMM_MMA_I8 v[5:8], v[75:76], v[117:118], v[5:8]
	;;#ASMSTART
	GEMM_WAIT_LGKM 0
	;;#ASMEND
	GEMM_MMA_I8 v[33:36], v[77:78], v[119:120], v[33:36]
	GEMM_MMA_I8 v[1:4], v[73:74], v[119:120], v[1:4]
	s_add_i32 s4, s4, 2
	v_add_co_u32_e32 v81, vcc, 0x4000, v81
	GEMM_MMA_I8 v[33:36], v[79:80], v[121:122], v[33:36]
	GEMM_MMA_I8 v[1:4], v[75:76], v[121:122], v[1:4]
	v_addc_co_u32_e32 v82, vcc, 0, v82, vcc
	s_cmpk_lt_u32 s4, 0x4d
	v_add_u32_e32 v87, 0x220000, v87
	s_cbranch_scc0 .LBB0_4
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	;;#ASMSTART
	GEMM_WAIT_VM 0
	;;#ASMEND
	GEMM_WAIT_LGKM 0
	;;#ASMSTART
	GEMM_CTA_BARRIER
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_add_co_u32_e32 v91, vcc, s5, v81
	v_addc_co_u32_e32 v92, vcc, -1, v82, vcc
	s_add_i32 s10, s4, 3
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[77:80], v[91:92]
	;;#ASMEND
	v_add_co_u32_e32 v91, vcc, s7, v81
	v_addc_co_u32_e32 v92, vcc, -1, v82, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[73:76], v[91:92]
	;;#ASMEND
	v_readfirstlane_b32 s11, v88
	v_add_u32_e32 v91, 0xffeef000, v87
	s_mov_b32 m0, s11
	v_readfirstlane_b32 s11, v89
	GEMM_W_GLOBAL_TO_LDS v91, s[0:3], 0
	v_add_u32_e32 v91, 0xffef0000, v87
	s_mov_b32 m0, s11
	s_nop 0
	GEMM_W_GLOBAL_TO_LDS v91, s[0:3], 0
	;;#ASMSTART
	GEMM_LDS_TO_REG v[91:94], v84, 0
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[95:98], v84, 0x400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[99:102], v84, 0x800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[103:106], v84, 0xc00
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[107:110], v84, 0x1000
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[111:114], v84, 0x1400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[115:118], v84, 0x1800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[119:122], v84, 0x1c00
	;;#ASMEND
	;;#ASMSTART
	GEMM_WAIT_LGKM 7
	;;#ASMEND
	GEMM_MMA_I8 v[57:60], v[45:46], v[91:92], v[57:60]
	GEMM_MMA_I8 v[29:32], v[37:38], v[91:92], v[29:32]
	GEMM_MMA_I8 v[57:60], v[47:48], v[93:94], v[57:60]
	GEMM_MMA_I8 v[29:32], v[39:40], v[93:94], v[29:32]
	;;#ASMSTART
	GEMM_WAIT_LGKM 6
	;;#ASMEND
	GEMM_MMA_I8 v[61:64], v[45:46], v[95:96], v[61:64]
	GEMM_MMA_I8 v[25:28], v[37:38], v[95:96], v[25:28]
	GEMM_MMA_I8 v[61:64], v[47:48], v[97:98], v[61:64]
	GEMM_MMA_I8 v[25:28], v[39:40], v[97:98], v[25:28]
	;;#ASMSTART
	GEMM_WAIT_LGKM 5
	;;#ASMEND
	GEMM_MMA_I8 v[65:68], v[45:46], v[99:100], v[65:68]
	GEMM_MMA_I8 v[21:24], v[37:38], v[99:100], v[21:24]
	GEMM_MMA_I8 v[65:68], v[47:48], v[101:102], v[65:68]
	GEMM_MMA_I8 v[21:24], v[39:40], v[101:102], v[21:24]
	;;#ASMSTART
	GEMM_WAIT_LGKM 4
	;;#ASMEND
	GEMM_MMA_I8 v[69:72], v[45:46], v[103:104], v[69:72]
	GEMM_MMA_I8 v[17:20], v[37:38], v[103:104], v[17:20]
	GEMM_MMA_I8 v[69:72], v[47:48], v[105:106], v[69:72]
	GEMM_MMA_I8 v[17:20], v[39:40], v[105:106], v[17:20]
	;;#ASMSTART
	GEMM_WAIT_LGKM 3
	;;#ASMEND
	GEMM_MMA_I8 v[53:56], v[45:46], v[107:108], v[53:56]
	GEMM_MMA_I8 v[13:16], v[37:38], v[107:108], v[13:16]
	GEMM_MMA_I8 v[53:56], v[47:48], v[109:110], v[53:56]
	GEMM_MMA_I8 v[13:16], v[39:40], v[109:110], v[13:16]
	;;#ASMSTART
	GEMM_WAIT_LGKM 2
	;;#ASMEND
	GEMM_MMA_I8 v[49:52], v[45:46], v[111:112], v[49:52]
	GEMM_MMA_I8 v[9:12], v[37:38], v[111:112], v[9:12]
	GEMM_MMA_I8 v[49:52], v[47:48], v[113:114], v[49:52]
	GEMM_MMA_I8 v[9:12], v[39:40], v[113:114], v[9:12]
	;;#ASMSTART
	GEMM_WAIT_LGKM 1
	;;#ASMEND
	GEMM_MMA_I8 v[41:44], v[45:46], v[115:116], v[41:44]
	GEMM_MMA_I8 v[5:8], v[37:38], v[115:116], v[5:8]
	GEMM_MMA_I8 v[41:44], v[47:48], v[117:118], v[41:44]
	GEMM_MMA_I8 v[5:8], v[39:40], v[117:118], v[5:8]
	;;#ASMSTART
	GEMM_WAIT_LGKM 0
	;;#ASMEND
	GEMM_MMA_I8 v[33:36], v[45:46], v[119:120], v[33:36]
	GEMM_MMA_I8 v[1:4], v[37:38], v[119:120], v[1:4]
	GEMM_MMA_I8 v[33:36], v[47:48], v[121:122], v[33:36]
	GEMM_MMA_I8 v[1:4], v[39:40], v[121:122], v[1:4]
	;;#ASMSTART
	GEMM_WAIT_VM 0
	;;#ASMEND
	GEMM_WAIT_LGKM 0
	;;#ASMSTART
	GEMM_CTA_BARRIER
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_cmpk_gt_u32 s10, 0x4e
	s_cbranch_scc1 .LBB0_1
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[45:48], v[81:82]
	;;#ASMEND
	v_add_co_u32_e32 v91, vcc, 0x400, v81
	v_addc_co_u32_e32 v92, vcc, 0, v82, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[37:40], v[91:92]
	;;#ASMEND
	v_readfirstlane_b32 s10, v85
	v_add_u32_e32 v91, 0xfffff000, v87
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v86
	GEMM_W_GLOBAL_TO_LDS v91, s[0:3], 0
	s_mov_b32 m0, s10
	s_nop 0
	GEMM_W_GLOBAL_TO_LDS v87, s[0:3], 0
	s_branch .LBB0_1
.LBB0_4:
s_waitcnt vmcnt(0)
v_and_b32_e32 v120, 15, v0
v_lshrrev_b32_e32 v121, 4, v0
v_and_b32_e32 v121, 3, v121
v_lshlrev_b32_e32 v121, 2, v121
s_lshl_b32 s2, s6, 7
v_add_u32_e32 v121, s2, v121
v_lshlrev_b32_e32 v117, 5, v83
v_add_u32_e32 v120, v117, v120
v_lshlrev_b32_e32 v122, 2, v121
global_load_dwordx4 v[84:87], v122, s[26:27] offset:0
global_load_dwordx4 v[88:91], v122, s[26:27] offset:64
global_load_dwordx4 v[92:95], v122, s[26:27] offset:128
global_load_dwordx4 v[96:99], v122, s[26:27] offset:192
global_load_dwordx4 v[100:103], v122, s[26:27] offset:256
global_load_dwordx4 v[104:107], v122, s[26:27] offset:320
global_load_dwordx4 v[108:111], v122, s[26:27] offset:384
global_load_dwordx4 v[112:115], v122, s[26:27] offset:448
v_cmp_gt_u32_e32 vcc, 128, v120
s_and_saveexec_b64 s[2:3], vcc
s_cbranch_execz .L_E047_fixed_bf16_skip_0
v_lshlrev_b32_e32 v117, 2, v120
global_load_dword v116, v117, s[24:25]
v_mul_u32_u24_e32 v117, 17408, v120
v_add_u32_e32 v117, v121, v117
v_lshlrev_b32_e32 v117, 1, v117
v_add_co_u32_e32 v118, vcc, s8, v117
v_mov_b32_e32 v119, s9
v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
s_waitcnt vmcnt(0)
v_cvt_f32_i32 v57, v57
v_mul_f32_e32 v117, v116, v84
v_mul_f32_e32 v57, v117, v57
v_lshrrev_b32_e32 v117, 16, v57
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v57, 0x7fff, v57
v_add_u32_e32 v57, v117, v57
v_cvt_f32_i32 v58, v58
v_mul_f32_e32 v117, v116, v85
v_mul_f32_e32 v58, v117, v58
v_lshrrev_b32_e32 v117, 16, v58
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v58, 0x7fff, v58
v_add_u32_e32 v58, v117, v58
v_cvt_f32_i32 v59, v59
v_mul_f32_e32 v117, v116, v86
v_mul_f32_e32 v59, v117, v59
v_lshrrev_b32_e32 v117, 16, v59
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v59, 0x7fff, v59
v_add_u32_e32 v59, v117, v59
v_cvt_f32_i32 v60, v60
v_mul_f32_e32 v117, v116, v87
v_mul_f32_e32 v60, v117, v60
v_lshrrev_b32_e32 v117, 16, v60
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v60, 0x7fff, v60
v_add_u32_e32 v60, v117, v60
v_lshrrev_b32_e32 v57, 16, v57
v_and_b32_e32 v58, 0xffff0000, v58
v_or_b32_e32 v57, v57, v58
v_lshrrev_b32_e32 v59, 16, v59
v_and_b32_e32 v60, 0xffff0000, v60
v_or_b32_e32 v58, v59, v60
global_store_dwordx2 v[118:119], v[57:58], off offset:0 glc slc
v_cvt_f32_i32 v61, v61
v_mul_f32_e32 v117, v116, v88
v_mul_f32_e32 v61, v117, v61
v_lshrrev_b32_e32 v117, 16, v61
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v61, 0x7fff, v61
v_add_u32_e32 v61, v117, v61
v_cvt_f32_i32 v62, v62
v_mul_f32_e32 v117, v116, v89
v_mul_f32_e32 v62, v117, v62
v_lshrrev_b32_e32 v117, 16, v62
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v62, 0x7fff, v62
v_add_u32_e32 v62, v117, v62
v_cvt_f32_i32 v63, v63
v_mul_f32_e32 v117, v116, v90
v_mul_f32_e32 v63, v117, v63
v_lshrrev_b32_e32 v117, 16, v63
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v63, 0x7fff, v63
v_add_u32_e32 v63, v117, v63
v_cvt_f32_i32 v64, v64
v_mul_f32_e32 v117, v116, v91
v_mul_f32_e32 v64, v117, v64
v_lshrrev_b32_e32 v117, 16, v64
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v64, 0x7fff, v64
v_add_u32_e32 v64, v117, v64
v_lshrrev_b32_e32 v61, 16, v61
v_and_b32_e32 v62, 0xffff0000, v62
v_or_b32_e32 v61, v61, v62
v_lshrrev_b32_e32 v63, 16, v63
v_and_b32_e32 v64, 0xffff0000, v64
v_or_b32_e32 v62, v63, v64
global_store_dwordx2 v[118:119], v[61:62], off offset:32 glc slc
v_cvt_f32_i32 v65, v65
v_mul_f32_e32 v117, v116, v92
v_mul_f32_e32 v65, v117, v65
v_lshrrev_b32_e32 v117, 16, v65
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v65, 0x7fff, v65
v_add_u32_e32 v65, v117, v65
v_cvt_f32_i32 v66, v66
v_mul_f32_e32 v117, v116, v93
v_mul_f32_e32 v66, v117, v66
v_lshrrev_b32_e32 v117, 16, v66
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v66, 0x7fff, v66
v_add_u32_e32 v66, v117, v66
v_cvt_f32_i32 v67, v67
v_mul_f32_e32 v117, v116, v94
v_mul_f32_e32 v67, v117, v67
v_lshrrev_b32_e32 v117, 16, v67
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v67, 0x7fff, v67
v_add_u32_e32 v67, v117, v67
v_cvt_f32_i32 v68, v68
v_mul_f32_e32 v117, v116, v95
v_mul_f32_e32 v68, v117, v68
v_lshrrev_b32_e32 v117, 16, v68
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v68, 0x7fff, v68
v_add_u32_e32 v68, v117, v68
v_lshrrev_b32_e32 v65, 16, v65
v_and_b32_e32 v66, 0xffff0000, v66
v_or_b32_e32 v65, v65, v66
v_lshrrev_b32_e32 v67, 16, v67
v_and_b32_e32 v68, 0xffff0000, v68
v_or_b32_e32 v66, v67, v68
global_store_dwordx2 v[118:119], v[65:66], off offset:64 glc slc
v_cvt_f32_i32 v69, v69
v_mul_f32_e32 v117, v116, v96
v_mul_f32_e32 v69, v117, v69
v_lshrrev_b32_e32 v117, 16, v69
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v69, 0x7fff, v69
v_add_u32_e32 v69, v117, v69
v_cvt_f32_i32 v70, v70
v_mul_f32_e32 v117, v116, v97
v_mul_f32_e32 v70, v117, v70
v_lshrrev_b32_e32 v117, 16, v70
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v70, 0x7fff, v70
v_add_u32_e32 v70, v117, v70
v_cvt_f32_i32 v71, v71
v_mul_f32_e32 v117, v116, v98
v_mul_f32_e32 v71, v117, v71
v_lshrrev_b32_e32 v117, 16, v71
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v71, 0x7fff, v71
v_add_u32_e32 v71, v117, v71
v_cvt_f32_i32 v72, v72
v_mul_f32_e32 v117, v116, v99
v_mul_f32_e32 v72, v117, v72
v_lshrrev_b32_e32 v117, 16, v72
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v72, 0x7fff, v72
v_add_u32_e32 v72, v117, v72
v_lshrrev_b32_e32 v69, 16, v69
v_and_b32_e32 v70, 0xffff0000, v70
v_or_b32_e32 v69, v69, v70
v_lshrrev_b32_e32 v71, 16, v71
v_and_b32_e32 v72, 0xffff0000, v72
v_or_b32_e32 v70, v71, v72
global_store_dwordx2 v[118:119], v[69:70], off offset:96 glc slc
v_cvt_f32_i32 v53, v53
v_mul_f32_e32 v117, v116, v100
v_mul_f32_e32 v53, v117, v53
v_lshrrev_b32_e32 v117, 16, v53
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v53, 0x7fff, v53
v_add_u32_e32 v53, v117, v53
v_cvt_f32_i32 v54, v54
v_mul_f32_e32 v117, v116, v101
v_mul_f32_e32 v54, v117, v54
v_lshrrev_b32_e32 v117, 16, v54
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v54, 0x7fff, v54
v_add_u32_e32 v54, v117, v54
v_cvt_f32_i32 v55, v55
v_mul_f32_e32 v117, v116, v102
v_mul_f32_e32 v55, v117, v55
v_lshrrev_b32_e32 v117, 16, v55
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v55, 0x7fff, v55
v_add_u32_e32 v55, v117, v55
v_cvt_f32_i32 v56, v56
v_mul_f32_e32 v117, v116, v103
v_mul_f32_e32 v56, v117, v56
v_lshrrev_b32_e32 v117, 16, v56
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v56, 0x7fff, v56
v_add_u32_e32 v56, v117, v56
v_lshrrev_b32_e32 v53, 16, v53
v_and_b32_e32 v54, 0xffff0000, v54
v_or_b32_e32 v53, v53, v54
v_lshrrev_b32_e32 v55, 16, v55
v_and_b32_e32 v56, 0xffff0000, v56
v_or_b32_e32 v54, v55, v56
global_store_dwordx2 v[118:119], v[53:54], off offset:128 glc slc
v_cvt_f32_i32 v49, v49
v_mul_f32_e32 v117, v116, v104
v_mul_f32_e32 v49, v117, v49
v_lshrrev_b32_e32 v117, 16, v49
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v49, 0x7fff, v49
v_add_u32_e32 v49, v117, v49
v_cvt_f32_i32 v50, v50
v_mul_f32_e32 v117, v116, v105
v_mul_f32_e32 v50, v117, v50
v_lshrrev_b32_e32 v117, 16, v50
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v50, 0x7fff, v50
v_add_u32_e32 v50, v117, v50
v_cvt_f32_i32 v51, v51
v_mul_f32_e32 v117, v116, v106
v_mul_f32_e32 v51, v117, v51
v_lshrrev_b32_e32 v117, 16, v51
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v51, 0x7fff, v51
v_add_u32_e32 v51, v117, v51
v_cvt_f32_i32 v52, v52
v_mul_f32_e32 v117, v116, v107
v_mul_f32_e32 v52, v117, v52
v_lshrrev_b32_e32 v117, 16, v52
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v52, 0x7fff, v52
v_add_u32_e32 v52, v117, v52
v_lshrrev_b32_e32 v49, 16, v49
v_and_b32_e32 v50, 0xffff0000, v50
v_or_b32_e32 v49, v49, v50
v_lshrrev_b32_e32 v51, 16, v51
v_and_b32_e32 v52, 0xffff0000, v52
v_or_b32_e32 v50, v51, v52
global_store_dwordx2 v[118:119], v[49:50], off offset:160 glc slc
v_cvt_f32_i32 v41, v41
v_mul_f32_e32 v117, v116, v108
v_mul_f32_e32 v41, v117, v41
v_lshrrev_b32_e32 v117, 16, v41
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v41, 0x7fff, v41
v_add_u32_e32 v41, v117, v41
v_cvt_f32_i32 v42, v42
v_mul_f32_e32 v117, v116, v109
v_mul_f32_e32 v42, v117, v42
v_lshrrev_b32_e32 v117, 16, v42
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v42, 0x7fff, v42
v_add_u32_e32 v42, v117, v42
v_cvt_f32_i32 v43, v43
v_mul_f32_e32 v117, v116, v110
v_mul_f32_e32 v43, v117, v43
v_lshrrev_b32_e32 v117, 16, v43
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v43, 0x7fff, v43
v_add_u32_e32 v43, v117, v43
v_cvt_f32_i32 v44, v44
v_mul_f32_e32 v117, v116, v111
v_mul_f32_e32 v44, v117, v44
v_lshrrev_b32_e32 v117, 16, v44
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v44, 0x7fff, v44
v_add_u32_e32 v44, v117, v44
v_lshrrev_b32_e32 v41, 16, v41
v_and_b32_e32 v42, 0xffff0000, v42
v_or_b32_e32 v41, v41, v42
v_lshrrev_b32_e32 v43, 16, v43
v_and_b32_e32 v44, 0xffff0000, v44
v_or_b32_e32 v42, v43, v44
global_store_dwordx2 v[118:119], v[41:42], off offset:192 glc slc
v_cvt_f32_i32 v33, v33
v_mul_f32_e32 v117, v116, v112
v_mul_f32_e32 v33, v117, v33
v_lshrrev_b32_e32 v117, 16, v33
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v33, 0x7fff, v33
v_add_u32_e32 v33, v117, v33
v_cvt_f32_i32 v34, v34
v_mul_f32_e32 v117, v116, v113
v_mul_f32_e32 v34, v117, v34
v_lshrrev_b32_e32 v117, 16, v34
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v34, 0x7fff, v34
v_add_u32_e32 v34, v117, v34
v_cvt_f32_i32 v35, v35
v_mul_f32_e32 v117, v116, v114
v_mul_f32_e32 v35, v117, v35
v_lshrrev_b32_e32 v117, 16, v35
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v35, 0x7fff, v35
v_add_u32_e32 v35, v117, v35
v_cvt_f32_i32 v36, v36
v_mul_f32_e32 v117, v116, v115
v_mul_f32_e32 v36, v117, v36
v_lshrrev_b32_e32 v117, 16, v36
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v36, 0x7fff, v36
v_add_u32_e32 v36, v117, v36
v_lshrrev_b32_e32 v33, 16, v33
v_and_b32_e32 v34, 0xffff0000, v34
v_or_b32_e32 v33, v33, v34
v_lshrrev_b32_e32 v35, 16, v35
v_and_b32_e32 v36, 0xffff0000, v36
v_or_b32_e32 v34, v35, v36
global_store_dwordx2 v[118:119], v[33:34], off offset:224 glc slc
.L_E047_fixed_bf16_skip_0:
s_or_b64 exec, exec, s[2:3]
v_add_u32_e32 v120, 16, v120
v_cmp_gt_u32_e32 vcc, 128, v120
s_and_saveexec_b64 s[2:3], vcc
s_cbranch_execz .L_E047_fixed_bf16_skip_1
v_lshlrev_b32_e32 v117, 2, v120
global_load_dword v116, v117, s[24:25]
v_mul_u32_u24_e32 v117, 17408, v120
v_add_u32_e32 v117, v121, v117
v_lshlrev_b32_e32 v117, 1, v117
v_add_co_u32_e32 v118, vcc, s8, v117
v_mov_b32_e32 v119, s9
v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
s_waitcnt vmcnt(0)
v_cvt_f32_i32 v29, v29
v_mul_f32_e32 v117, v116, v84
v_mul_f32_e32 v29, v117, v29
v_lshrrev_b32_e32 v117, 16, v29
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v29, 0x7fff, v29
v_add_u32_e32 v29, v117, v29
v_cvt_f32_i32 v30, v30
v_mul_f32_e32 v117, v116, v85
v_mul_f32_e32 v30, v117, v30
v_lshrrev_b32_e32 v117, 16, v30
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v30, 0x7fff, v30
v_add_u32_e32 v30, v117, v30
v_cvt_f32_i32 v31, v31
v_mul_f32_e32 v117, v116, v86
v_mul_f32_e32 v31, v117, v31
v_lshrrev_b32_e32 v117, 16, v31
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v31, 0x7fff, v31
v_add_u32_e32 v31, v117, v31
v_cvt_f32_i32 v32, v32
v_mul_f32_e32 v117, v116, v87
v_mul_f32_e32 v32, v117, v32
v_lshrrev_b32_e32 v117, 16, v32
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v32, 0x7fff, v32
v_add_u32_e32 v32, v117, v32
v_lshrrev_b32_e32 v29, 16, v29
v_and_b32_e32 v30, 0xffff0000, v30
v_or_b32_e32 v29, v29, v30
v_lshrrev_b32_e32 v31, 16, v31
v_and_b32_e32 v32, 0xffff0000, v32
v_or_b32_e32 v30, v31, v32
global_store_dwordx2 v[118:119], v[29:30], off offset:0 glc slc
v_cvt_f32_i32 v25, v25
v_mul_f32_e32 v117, v116, v88
v_mul_f32_e32 v25, v117, v25
v_lshrrev_b32_e32 v117, 16, v25
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v25, 0x7fff, v25
v_add_u32_e32 v25, v117, v25
v_cvt_f32_i32 v26, v26
v_mul_f32_e32 v117, v116, v89
v_mul_f32_e32 v26, v117, v26
v_lshrrev_b32_e32 v117, 16, v26
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v26, 0x7fff, v26
v_add_u32_e32 v26, v117, v26
v_cvt_f32_i32 v27, v27
v_mul_f32_e32 v117, v116, v90
v_mul_f32_e32 v27, v117, v27
v_lshrrev_b32_e32 v117, 16, v27
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v27, 0x7fff, v27
v_add_u32_e32 v27, v117, v27
v_cvt_f32_i32 v28, v28
v_mul_f32_e32 v117, v116, v91
v_mul_f32_e32 v28, v117, v28
v_lshrrev_b32_e32 v117, 16, v28
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v28, 0x7fff, v28
v_add_u32_e32 v28, v117, v28
v_lshrrev_b32_e32 v25, 16, v25
v_and_b32_e32 v26, 0xffff0000, v26
v_or_b32_e32 v25, v25, v26
v_lshrrev_b32_e32 v27, 16, v27
v_and_b32_e32 v28, 0xffff0000, v28
v_or_b32_e32 v26, v27, v28
global_store_dwordx2 v[118:119], v[25:26], off offset:32 glc slc
v_cvt_f32_i32 v21, v21
v_mul_f32_e32 v117, v116, v92
v_mul_f32_e32 v21, v117, v21
v_lshrrev_b32_e32 v117, 16, v21
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v21, 0x7fff, v21
v_add_u32_e32 v21, v117, v21
v_cvt_f32_i32 v22, v22
v_mul_f32_e32 v117, v116, v93
v_mul_f32_e32 v22, v117, v22
v_lshrrev_b32_e32 v117, 16, v22
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v22, 0x7fff, v22
v_add_u32_e32 v22, v117, v22
v_cvt_f32_i32 v23, v23
v_mul_f32_e32 v117, v116, v94
v_mul_f32_e32 v23, v117, v23
v_lshrrev_b32_e32 v117, 16, v23
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v23, 0x7fff, v23
v_add_u32_e32 v23, v117, v23
v_cvt_f32_i32 v24, v24
v_mul_f32_e32 v117, v116, v95
v_mul_f32_e32 v24, v117, v24
v_lshrrev_b32_e32 v117, 16, v24
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v24, 0x7fff, v24
v_add_u32_e32 v24, v117, v24
v_lshrrev_b32_e32 v21, 16, v21
v_and_b32_e32 v22, 0xffff0000, v22
v_or_b32_e32 v21, v21, v22
v_lshrrev_b32_e32 v23, 16, v23
v_and_b32_e32 v24, 0xffff0000, v24
v_or_b32_e32 v22, v23, v24
global_store_dwordx2 v[118:119], v[21:22], off offset:64 glc slc
v_cvt_f32_i32 v17, v17
v_mul_f32_e32 v117, v116, v96
v_mul_f32_e32 v17, v117, v17
v_lshrrev_b32_e32 v117, 16, v17
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v17, 0x7fff, v17
v_add_u32_e32 v17, v117, v17
v_cvt_f32_i32 v18, v18
v_mul_f32_e32 v117, v116, v97
v_mul_f32_e32 v18, v117, v18
v_lshrrev_b32_e32 v117, 16, v18
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v18, 0x7fff, v18
v_add_u32_e32 v18, v117, v18
v_cvt_f32_i32 v19, v19
v_mul_f32_e32 v117, v116, v98
v_mul_f32_e32 v19, v117, v19
v_lshrrev_b32_e32 v117, 16, v19
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v19, 0x7fff, v19
v_add_u32_e32 v19, v117, v19
v_cvt_f32_i32 v20, v20
v_mul_f32_e32 v117, v116, v99
v_mul_f32_e32 v20, v117, v20
v_lshrrev_b32_e32 v117, 16, v20
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v20, 0x7fff, v20
v_add_u32_e32 v20, v117, v20
v_lshrrev_b32_e32 v17, 16, v17
v_and_b32_e32 v18, 0xffff0000, v18
v_or_b32_e32 v17, v17, v18
v_lshrrev_b32_e32 v19, 16, v19
v_and_b32_e32 v20, 0xffff0000, v20
v_or_b32_e32 v18, v19, v20
global_store_dwordx2 v[118:119], v[17:18], off offset:96 glc slc
v_cvt_f32_i32 v13, v13
v_mul_f32_e32 v117, v116, v100
v_mul_f32_e32 v13, v117, v13
v_lshrrev_b32_e32 v117, 16, v13
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v13, 0x7fff, v13
v_add_u32_e32 v13, v117, v13
v_cvt_f32_i32 v14, v14
v_mul_f32_e32 v117, v116, v101
v_mul_f32_e32 v14, v117, v14
v_lshrrev_b32_e32 v117, 16, v14
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v14, 0x7fff, v14
v_add_u32_e32 v14, v117, v14
v_cvt_f32_i32 v15, v15
v_mul_f32_e32 v117, v116, v102
v_mul_f32_e32 v15, v117, v15
v_lshrrev_b32_e32 v117, 16, v15
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v15, 0x7fff, v15
v_add_u32_e32 v15, v117, v15
v_cvt_f32_i32 v16, v16
v_mul_f32_e32 v117, v116, v103
v_mul_f32_e32 v16, v117, v16
v_lshrrev_b32_e32 v117, 16, v16
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v16, 0x7fff, v16
v_add_u32_e32 v16, v117, v16
v_lshrrev_b32_e32 v13, 16, v13
v_and_b32_e32 v14, 0xffff0000, v14
v_or_b32_e32 v13, v13, v14
v_lshrrev_b32_e32 v15, 16, v15
v_and_b32_e32 v16, 0xffff0000, v16
v_or_b32_e32 v14, v15, v16
global_store_dwordx2 v[118:119], v[13:14], off offset:128 glc slc
v_cvt_f32_i32 v9, v9
v_mul_f32_e32 v117, v116, v104
v_mul_f32_e32 v9, v117, v9
v_lshrrev_b32_e32 v117, 16, v9
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v9, 0x7fff, v9
v_add_u32_e32 v9, v117, v9
v_cvt_f32_i32 v10, v10
v_mul_f32_e32 v117, v116, v105
v_mul_f32_e32 v10, v117, v10
v_lshrrev_b32_e32 v117, 16, v10
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v10, 0x7fff, v10
v_add_u32_e32 v10, v117, v10
v_cvt_f32_i32 v11, v11
v_mul_f32_e32 v117, v116, v106
v_mul_f32_e32 v11, v117, v11
v_lshrrev_b32_e32 v117, 16, v11
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v11, 0x7fff, v11
v_add_u32_e32 v11, v117, v11
v_cvt_f32_i32 v12, v12
v_mul_f32_e32 v117, v116, v107
v_mul_f32_e32 v12, v117, v12
v_lshrrev_b32_e32 v117, 16, v12
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v12, 0x7fff, v12
v_add_u32_e32 v12, v117, v12
v_lshrrev_b32_e32 v9, 16, v9
v_and_b32_e32 v10, 0xffff0000, v10
v_or_b32_e32 v9, v9, v10
v_lshrrev_b32_e32 v11, 16, v11
v_and_b32_e32 v12, 0xffff0000, v12
v_or_b32_e32 v10, v11, v12
global_store_dwordx2 v[118:119], v[9:10], off offset:160 glc slc
v_cvt_f32_i32 v5, v5
v_mul_f32_e32 v117, v116, v108
v_mul_f32_e32 v5, v117, v5
v_lshrrev_b32_e32 v117, 16, v5
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v5, 0x7fff, v5
v_add_u32_e32 v5, v117, v5
v_cvt_f32_i32 v6, v6
v_mul_f32_e32 v117, v116, v109
v_mul_f32_e32 v6, v117, v6
v_lshrrev_b32_e32 v117, 16, v6
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v6, 0x7fff, v6
v_add_u32_e32 v6, v117, v6
v_cvt_f32_i32 v7, v7
v_mul_f32_e32 v117, v116, v110
v_mul_f32_e32 v7, v117, v7
v_lshrrev_b32_e32 v117, 16, v7
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v7, 0x7fff, v7
v_add_u32_e32 v7, v117, v7
v_cvt_f32_i32 v8, v8
v_mul_f32_e32 v117, v116, v111
v_mul_f32_e32 v8, v117, v8
v_lshrrev_b32_e32 v117, 16, v8
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v8, 0x7fff, v8
v_add_u32_e32 v8, v117, v8
v_lshrrev_b32_e32 v5, 16, v5
v_and_b32_e32 v6, 0xffff0000, v6
v_or_b32_e32 v5, v5, v6
v_lshrrev_b32_e32 v7, 16, v7
v_and_b32_e32 v8, 0xffff0000, v8
v_or_b32_e32 v6, v7, v8
global_store_dwordx2 v[118:119], v[5:6], off offset:192 glc slc
v_cvt_f32_i32 v1, v1
v_mul_f32_e32 v117, v116, v112
v_mul_f32_e32 v1, v117, v1
v_lshrrev_b32_e32 v117, 16, v1
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v1, 0x7fff, v1
v_add_u32_e32 v1, v117, v1
v_cvt_f32_i32 v2, v2
v_mul_f32_e32 v117, v116, v113
v_mul_f32_e32 v2, v117, v2
v_lshrrev_b32_e32 v117, 16, v2
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v2, 0x7fff, v2
v_add_u32_e32 v2, v117, v2
v_cvt_f32_i32 v3, v3
v_mul_f32_e32 v117, v116, v114
v_mul_f32_e32 v3, v117, v3
v_lshrrev_b32_e32 v117, 16, v3
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v3, 0x7fff, v3
v_add_u32_e32 v3, v117, v3
v_cvt_f32_i32 v4, v4
v_mul_f32_e32 v117, v116, v115
v_mul_f32_e32 v4, v117, v4
v_lshrrev_b32_e32 v117, 16, v4
v_and_b32_e32 v117, 1, v117
v_add_u32_e32 v4, 0x7fff, v4
v_add_u32_e32 v4, v117, v4
v_lshrrev_b32_e32 v1, 16, v1
v_and_b32_e32 v2, 0xffff0000, v2
v_or_b32_e32 v1, v1, v2
v_lshrrev_b32_e32 v3, 16, v3
v_and_b32_e32 v4, 0xffff0000, v4
v_or_b32_e32 v2, v3, v4
global_store_dwordx2 v[118:119], v[1:2], off offset:224 glc slc
.L_E047_fixed_bf16_skip_1:
s_or_b64 exec, exec, s[2:3]
s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 123
		.amdhsa_next_free_sgpr 28
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii,#alloc,#execinstr
.Lfunc_end0:
	.size	_Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii, .Lfunc_end0-_Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 1892
; NumSgprs: 24
; NumVgprs: 123
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 123
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii,#alloc,#execinstr
	.protected	_Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii ; -- Begin function _Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii
	.globl	_Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii
	.p2align	8
	.type	_Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii,@function
_Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii: ; @_Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii
s_load_dwordx4 s[24:27], s[4:5], 0x28
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x18
	s_load_dwordx4 s[12:15], s[4:5], 0x0
	s_load_dwordx2 s[16:17], s[4:5], 0x10
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v1, 0x60, v1
	GEMM_WAIT_LGKM 0
	s_ashr_i32 s4, s2, 31
	v_lshl_or_b32 v85, s7, 7, v1
	s_lshr_b32 s4, s4, 26
	v_ashrrev_i32_e32 v1, 4, v85
	s_add_i32 s4, s2, s4
	v_ashrrev_i32_e32 v2, 31, v1
	s_ashr_i32 s21, s4, 6
	s_ashr_i32 s4, s1, 31
	v_lshlrev_b64 v[1:2], 10, v[1:2]
	s_lshr_b32 s4, s4, 28
	s_add_i32 s4, s1, s4
	v_add_co_u32_e32 v1, vcc, s12, v1
	s_mov_b32 s12, 0
	v_and_b32_e32 v86, 63, v0
	s_ashr_i32 s20, s4, 4
	s_ashr_i32 s4, s3, 31
	v_mov_b32_e32 v3, s13
	s_add_i32 s9, s15, 4.0
	s_mov_b32 s8, s14
	s_mov_b32 s14, s12
	s_mov_b32 s15, s12
	s_lshr_b32 s4, s4, 28
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc
	v_lshlrev_b32_e32 v90, 4, v86
	s_mov_b32 s13, s12
	v_mov_b64_e32 v[27:28], s[14:15]
	s_add_i32 s3, s3, s4
	v_add_co_u32_e32 v81, vcc, v1, v90
	s_lshl_b32 s22, s6, 13
	v_lshlrev_b32_e32 v91, 4, v0
	v_mov_b64_e32 v[25:26], s[12:13]
	v_mov_b64_e32 v[31:32], v[27:28]
	s_ashr_i32 s4, s3, 4
	v_addc_co_u32_e32 v82, vcc, 0, v2, vcc
	s_mov_b32 s11, 0x20000
	s_brev_b32 s10, 1
	v_mov_b64_e32 v[29:30], v[25:26]
	v_or_b32_e32 v87, s22, v91
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[29:32], v[81:82]
	;;#ASMEND
	s_movk_i32 s3, 0x400
	v_add_co_u32_e32 v1, vcc, s3, v81
	v_addc_co_u32_e32 v2, vcc, 0, v82, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[25:28], v[1:2]
	;;#ASMEND
	v_and_b32_e32 v1, 0xc0, v0
	v_lshlrev_b32_e32 v88, 4, v1
	v_readfirstlane_b32 s3, v88
	v_or_b32_e32 v92, 0x1000, v88
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v92
	GEMM_W_GLOBAL_TO_LDS v87, s[8:11], 0
	v_or_b32_e32 v89, 0x1000, v87
	s_mov_b32 m0, s3
	s_cmpk_lt_i32 s2, 0x80
	GEMM_W_GLOBAL_TO_LDS v89, s[8:11], 0
	s_cbranch_scc1 .LBB1_7
; %bb.1:
	s_mov_b32 s14, s12
	s_mov_b32 s15, s12
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s13, s12
	v_mov_b64_e32 v[67:68], s[14:15]
	s_lshl_b64 s[2:3], s[4:5], 10
	s_lshl_b64 s[18:19], s[4:5], 11
	s_lshl_b32 s5, s20, 10
	s_lshl_b32 s7, s20, 11
	v_mov_b64_e32 v[65:66], s[12:13]
	v_mov_b32_e32 v1, 0
	v_mov_b64_e32 v[75:76], v[67:68]
	v_or_b32_e32 v93, 0x2000, v88
	v_or_b32_e32 v94, 0x3000, v88
	v_or_b32_e32 v95, 0x2000, v90
	s_add_i32 s5, s22, s5
	s_add_i32 s22, s22, s7
	v_mov_b64_e32 v[83:84], v[81:82]
	v_mov_b64_e32 v[73:74], v[65:66]
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v8, v1
	v_mov_b32_e32 v9, v1
	v_mov_b32_e32 v10, v1
	v_mov_b32_e32 v11, v1
	v_mov_b32_e32 v12, v1
	v_mov_b32_e32 v13, v1
	v_mov_b32_e32 v14, v1
	v_mov_b32_e32 v15, v1
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v17, v1
	v_mov_b32_e32 v18, v1
	v_mov_b32_e32 v19, v1
	v_mov_b32_e32 v20, v1
	v_mov_b32_e32 v21, v1
	v_mov_b32_e32 v22, v1
	v_mov_b32_e32 v23, v1
	v_mov_b32_e32 v24, v1
	v_mov_b32_e32 v33, v1
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v35, v1
	v_mov_b32_e32 v36, v1
	v_mov_b32_e32 v37, v1
	v_mov_b32_e32 v38, v1
	v_mov_b32_e32 v39, v1
	v_mov_b32_e32 v40, v1
	v_mov_b32_e32 v41, v1
	v_mov_b32_e32 v42, v1
	v_mov_b32_e32 v43, v1
	v_mov_b32_e32 v44, v1
	v_mov_b32_e32 v45, v1
	v_mov_b32_e32 v46, v1
	v_mov_b32_e32 v47, v1
	v_mov_b32_e32 v48, v1
	v_mov_b32_e32 v49, v1
	v_mov_b32_e32 v50, v1
	v_mov_b32_e32 v51, v1
	v_mov_b32_e32 v52, v1
	v_mov_b32_e32 v53, v1
	v_mov_b32_e32 v54, v1
	v_mov_b32_e32 v55, v1
	v_mov_b32_e32 v56, v1
	v_mov_b32_e32 v57, v1
	v_mov_b32_e32 v58, v1
	v_mov_b32_e32 v59, v1
	v_mov_b32_e32 v60, v1
	v_mov_b32_e32 v61, v1
	v_mov_b32_e32 v62, v1
	v_mov_b32_e32 v63, v1
	v_mov_b32_e32 v64, v1
	v_mov_b32_e32 v69, v1
	v_mov_b32_e32 v70, v1
	v_mov_b32_e32 v71, v1
	v_mov_b32_e32 v72, v1
	v_mov_b32_e32 v77, v1
	v_mov_b32_e32 v78, v1
	v_mov_b32_e32 v79, v1
	v_mov_b32_e32 v80, v1
	s_branch .LBB1_3
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	;;#ASMSTART
	GEMM_LDS_TO_REG v[96:99], v95, 0
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[100:103], v95, 0x400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[104:107], v95, 0x800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[108:111], v95, 0xc00
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[112:115], v95, 0x1000
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[116:119], v95, 0x1400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[120:123], v95, 0x1800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[124:127], v95, 0x1c00
	;;#ASMEND
	;;#ASMSTART
	GEMM_WAIT_LGKM 7
	;;#ASMEND
	GEMM_MMA_I8 v[77:80], v[73:74], v[96:97], v[77:80]
	GEMM_MMA_I8 v[37:40], v[65:66], v[96:97], v[37:40]
	GEMM_MMA_I8 v[77:80], v[75:76], v[98:99], v[77:80]
	GEMM_MMA_I8 v[37:40], v[67:68], v[98:99], v[37:40]
	;;#ASMSTART
	GEMM_WAIT_LGKM 6
	;;#ASMEND
	GEMM_MMA_I8 v[69:72], v[73:74], v[100:101], v[69:72]
	GEMM_MMA_I8 v[33:36], v[65:66], v[100:101], v[33:36]
	GEMM_MMA_I8 v[69:72], v[75:76], v[102:103], v[69:72]
	GEMM_MMA_I8 v[33:36], v[67:68], v[102:103], v[33:36]
	;;#ASMSTART
	GEMM_WAIT_LGKM 5
	;;#ASMEND
	GEMM_MMA_I8 v[61:64], v[73:74], v[104:105], v[61:64]
	GEMM_MMA_I8 v[21:24], v[65:66], v[104:105], v[21:24]
	GEMM_MMA_I8 v[61:64], v[75:76], v[106:107], v[61:64]
	GEMM_MMA_I8 v[21:24], v[67:68], v[106:107], v[21:24]
	;;#ASMSTART
	GEMM_WAIT_LGKM 4
	;;#ASMEND
	GEMM_MMA_I8 v[57:60], v[73:74], v[108:109], v[57:60]
	GEMM_MMA_I8 v[17:20], v[65:66], v[108:109], v[17:20]
	GEMM_MMA_I8 v[57:60], v[75:76], v[110:111], v[57:60]
	GEMM_MMA_I8 v[17:20], v[67:68], v[110:111], v[17:20]
	;;#ASMSTART
	GEMM_WAIT_LGKM 3
	;;#ASMEND
	GEMM_MMA_I8 v[53:56], v[73:74], v[112:113], v[53:56]
	GEMM_MMA_I8 v[13:16], v[65:66], v[112:113], v[13:16]
	GEMM_MMA_I8 v[53:56], v[75:76], v[114:115], v[53:56]
	GEMM_MMA_I8 v[13:16], v[67:68], v[114:115], v[13:16]
	;;#ASMSTART
	GEMM_WAIT_LGKM 2
	;;#ASMEND
	GEMM_MMA_I8 v[49:52], v[73:74], v[116:117], v[49:52]
	GEMM_MMA_I8 v[9:12], v[65:66], v[116:117], v[9:12]
	GEMM_MMA_I8 v[49:52], v[75:76], v[118:119], v[49:52]
	GEMM_MMA_I8 v[9:12], v[67:68], v[118:119], v[9:12]
	;;#ASMSTART
	GEMM_WAIT_LGKM 1
	;;#ASMEND
	GEMM_MMA_I8 v[45:48], v[73:74], v[120:121], v[45:48]
	GEMM_MMA_I8 v[5:8], v[65:66], v[120:121], v[5:8]
	GEMM_MMA_I8 v[45:48], v[75:76], v[122:123], v[45:48]
	GEMM_MMA_I8 v[5:8], v[67:68], v[122:123], v[5:8]
	;;#ASMSTART
	GEMM_WAIT_LGKM 0
	;;#ASMEND
	GEMM_MMA_I8 v[41:44], v[73:74], v[124:125], v[41:44]
	GEMM_MMA_I8 v[1:4], v[65:66], v[124:125], v[1:4]
	s_add_i32 s13, s12, 2
	v_mov_b32_e32 v96, s19
	s_add_i32 s5, s5, s7
	s_add_i32 s22, s22, s7
	v_add_co_u32_e32 v83, vcc, s18, v83
	GEMM_MMA_I8 v[41:44], v[75:76], v[126:127], v[41:44]
	GEMM_MMA_I8 v[1:4], v[67:68], v[126:127], v[1:4]
	s_cmp_le_i32 s13, s21
	v_addc_co_u32_e32 v84, vcc, v84, v96, vcc
	s_cbranch_scc0 .LBB1_8
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	;;#ASMSTART
	GEMM_WAIT_VM 0
	;;#ASMEND
	GEMM_WAIT_LGKM 0
	;;#ASMSTART
	GEMM_CTA_BARRIER
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s13, s12, 1
	s_cmp_ge_i32 s13, s21
	s_cbranch_scc1 .LBB1_5
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b32_e32 v97, s3
	v_add_co_u32_e32 v96, vcc, s2, v83
	v_addc_co_u32_e32 v97, vcc, v84, v97, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[73:76], v[96:97]
	;;#ASMEND
	v_add_co_u32_e32 v96, vcc, 0x400, v96
	v_addc_co_u32_e32 v97, vcc, 0, v97, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[65:68], v[96:97]
	;;#ASMEND
	v_readfirstlane_b32 s13, v93
	v_add_u32_e32 v96, s5, v91
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v94
	GEMM_W_GLOBAL_TO_LDS v96, s[8:11], 0
	v_add_u32_e32 v96, 0x1000, v96
	s_mov_b32 m0, s13
	s_nop 0
	GEMM_W_GLOBAL_TO_LDS v96, s[8:11], 0
.LBB1_5:                                ;   in Loop: Header=BB1_3 Depth=1
	;;#ASMSTART
	GEMM_LDS_TO_REG v[96:99], v90, 0
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[100:103], v90, 0x400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[104:107], v90, 0x800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[108:111], v90, 0xc00
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[112:115], v90, 0x1000
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[116:119], v90, 0x1400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[120:123], v90, 0x1800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[124:127], v90, 0x1c00
	;;#ASMEND
	;;#ASMSTART
	GEMM_WAIT_LGKM 7
	;;#ASMEND
	GEMM_MMA_I8 v[77:80], v[29:30], v[96:97], v[77:80]
	GEMM_MMA_I8 v[37:40], v[25:26], v[96:97], v[37:40]
	GEMM_MMA_I8 v[77:80], v[31:32], v[98:99], v[77:80]
	GEMM_MMA_I8 v[37:40], v[27:28], v[98:99], v[37:40]
	;;#ASMSTART
	GEMM_WAIT_LGKM 6
	;;#ASMEND
	GEMM_MMA_I8 v[69:72], v[29:30], v[100:101], v[69:72]
	GEMM_MMA_I8 v[33:36], v[25:26], v[100:101], v[33:36]
	GEMM_MMA_I8 v[69:72], v[31:32], v[102:103], v[69:72]
	GEMM_MMA_I8 v[33:36], v[27:28], v[102:103], v[33:36]
	;;#ASMSTART
	GEMM_WAIT_LGKM 5
	;;#ASMEND
	GEMM_MMA_I8 v[61:64], v[29:30], v[104:105], v[61:64]
	GEMM_MMA_I8 v[21:24], v[25:26], v[104:105], v[21:24]
	GEMM_MMA_I8 v[61:64], v[31:32], v[106:107], v[61:64]
	GEMM_MMA_I8 v[21:24], v[27:28], v[106:107], v[21:24]
	;;#ASMSTART
	GEMM_WAIT_LGKM 4
	;;#ASMEND
	GEMM_MMA_I8 v[57:60], v[29:30], v[108:109], v[57:60]
	GEMM_MMA_I8 v[17:20], v[25:26], v[108:109], v[17:20]
	GEMM_MMA_I8 v[57:60], v[31:32], v[110:111], v[57:60]
	GEMM_MMA_I8 v[17:20], v[27:28], v[110:111], v[17:20]
	;;#ASMSTART
	GEMM_WAIT_LGKM 3
	;;#ASMEND
	GEMM_MMA_I8 v[53:56], v[29:30], v[112:113], v[53:56]
	GEMM_MMA_I8 v[13:16], v[25:26], v[112:113], v[13:16]
	GEMM_MMA_I8 v[53:56], v[31:32], v[114:115], v[53:56]
	GEMM_MMA_I8 v[13:16], v[27:28], v[114:115], v[13:16]
	;;#ASMSTART
	GEMM_WAIT_LGKM 2
	;;#ASMEND
	GEMM_MMA_I8 v[49:52], v[29:30], v[116:117], v[49:52]
	GEMM_MMA_I8 v[9:12], v[25:26], v[116:117], v[9:12]
	GEMM_MMA_I8 v[49:52], v[31:32], v[118:119], v[49:52]
	GEMM_MMA_I8 v[9:12], v[27:28], v[118:119], v[9:12]
	;;#ASMSTART
	GEMM_WAIT_LGKM 1
	;;#ASMEND
	GEMM_MMA_I8 v[45:48], v[29:30], v[120:121], v[45:48]
	GEMM_MMA_I8 v[5:8], v[25:26], v[120:121], v[5:8]
	GEMM_MMA_I8 v[45:48], v[31:32], v[122:123], v[45:48]
	GEMM_MMA_I8 v[5:8], v[27:28], v[122:123], v[5:8]
	;;#ASMSTART
	GEMM_WAIT_LGKM 0
	;;#ASMEND
	GEMM_MMA_I8 v[41:44], v[29:30], v[124:125], v[41:44]
	GEMM_MMA_I8 v[1:4], v[25:26], v[124:125], v[1:4]
	GEMM_MMA_I8 v[41:44], v[31:32], v[126:127], v[41:44]
	GEMM_MMA_I8 v[1:4], v[27:28], v[126:127], v[1:4]
	;;#ASMSTART
	GEMM_WAIT_VM 0
	;;#ASMEND
	GEMM_WAIT_LGKM 0
	;;#ASMSTART
	GEMM_CTA_BARRIER
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s12, s12, 2
	s_cmp_ge_i32 s12, s21
	s_cbranch_scc1 .LBB1_2
; %bb.6:                                ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b32_e32 v97, s19
	v_add_co_u32_e32 v96, vcc, s18, v83
	v_addc_co_u32_e32 v97, vcc, v84, v97, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[29:32], v[96:97]
	;;#ASMEND
	v_add_co_u32_e32 v96, vcc, 0x400, v96
	v_addc_co_u32_e32 v97, vcc, 0, v97, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[25:28], v[96:97]
	;;#ASMEND
	v_readfirstlane_b32 s13, v88
	v_add_u32_e32 v96, s22, v91
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v92
	GEMM_W_GLOBAL_TO_LDS v96, s[8:11], 0
	v_add_u32_e32 v96, 0x1000, v96
	s_mov_b32 m0, s13
	s_nop 0
	GEMM_W_GLOBAL_TO_LDS v96, s[8:11], 0
	s_branch .LBB1_2
.LBB1_7:
	v_mov_b64_e32 v[1:2], s[12:13]
	v_mov_b64_e32 v[3:4], s[14:15]
	v_mov_b64_e32 v[7:8], v[3:4]
	v_mov_b64_e32 v[11:12], v[3:4]
	v_mov_b64_e32 v[15:16], v[3:4]
	v_mov_b64_e32 v[19:20], v[3:4]
	v_mov_b64_e32 v[23:24], v[3:4]
	v_mov_b64_e32 v[35:36], v[3:4]
	v_mov_b64_e32 v[39:40], v[3:4]
	v_mov_b64_e32 v[43:44], v[3:4]
	v_mov_b64_e32 v[47:48], v[3:4]
	v_mov_b64_e32 v[51:52], v[3:4]
	v_mov_b64_e32 v[55:56], v[3:4]
	v_mov_b64_e32 v[59:60], v[3:4]
	v_mov_b64_e32 v[63:64], v[3:4]
	v_mov_b64_e32 v[71:72], v[3:4]
	v_mov_b64_e32 v[79:80], v[3:4]
	v_mov_b64_e32 v[67:68], v[3:4]
	v_mov_b64_e32 v[75:76], v[3:4]
	v_mov_b64_e32 v[5:6], v[1:2]
	v_mov_b64_e32 v[9:10], v[1:2]
	v_mov_b64_e32 v[13:14], v[1:2]
	v_mov_b64_e32 v[17:18], v[1:2]
	v_mov_b64_e32 v[21:22], v[1:2]
	v_mov_b64_e32 v[33:34], v[1:2]
	v_mov_b64_e32 v[37:38], v[1:2]
	v_mov_b64_e32 v[41:42], v[1:2]
	v_mov_b64_e32 v[45:46], v[1:2]
	v_mov_b64_e32 v[49:50], v[1:2]
	v_mov_b64_e32 v[53:54], v[1:2]
	v_mov_b64_e32 v[57:58], v[1:2]
	v_mov_b64_e32 v[61:62], v[1:2]
	v_mov_b64_e32 v[69:70], v[1:2]
	v_mov_b64_e32 v[77:78], v[1:2]
	v_mov_b64_e32 v[65:66], v[1:2]
	v_mov_b64_e32 v[73:74], v[1:2]
.LBB1_8:
	s_cmp_ge_i32 s12, s21
	s_cbranch_scc1 .LBB1_12
; %bb.9:
	;;#ASMSTART
	GEMM_WAIT_VM 0
	;;#ASMEND
	GEMM_WAIT_LGKM 0
	;;#ASMSTART
	GEMM_CTA_BARRIER
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_or_b32 s2, s12, 1
	s_cmp_ge_i32 s2, s21
	s_cbranch_scc1 .LBB1_11
; %bb.10:
	s_ashr_i32 s3, s4, 31
	v_mov_b32_e32 v83, s4
	v_alignbit_b32 v83, s3, v83, 22
	s_lshl_b32 s3, s4, 10
	v_mov_b32_e32 v84, s2
	v_mul_lo_u32 v83, v83, s2
	v_mad_u64_u32 v[81:82], s[4:5], s3, v84, v[81:82]
	v_add_u32_e32 v82, v83, v82
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[73:76], v[81:82]
	;;#ASMEND
	v_add_co_u32_e32 v81, vcc, 0x400, v81
	v_addc_co_u32_e32 v82, vcc, 0, v82, vcc
	;;#ASMSTART
	GEMM_X_GLOBAL_TO_REG v[65:68], v[81:82]
	;;#ASMEND
	v_or_b32_e32 v81, 0x2000, v88
	s_mul_i32 s2, s2, s20
	s_lshl_b32 s2, s2, 10
	v_readfirstlane_b32 s3, v81
	v_add_u32_e32 v82, s2, v87
	s_mov_b32 m0, s3
	v_or_b32_e32 v81, 0x3000, v88
	GEMM_W_GLOBAL_TO_LDS v82, s[8:11], 0
	v_add_u32_e32 v82, s2, v89
	v_readfirstlane_b32 s2, v81
	s_mov_b32 m0, s2
	s_nop 0
	GEMM_W_GLOBAL_TO_LDS v82, s[8:11], 0
.LBB1_11:
	v_lshlrev_b32_e32 v110, 4, v86
	;;#ASMSTART
	GEMM_LDS_TO_REG v[81:84], v110, 0
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[86:89], v110, 0x400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[90:93], v110, 0x800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[94:97], v110, 0xc00
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[98:101], v110, 0x1000
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[102:105], v110, 0x1400
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[106:109], v110, 0x1800
	;;#ASMEND
	;;#ASMSTART
	GEMM_LDS_TO_REG v[110:113], v110, 0x1c00
	;;#ASMEND
	;;#ASMSTART
	GEMM_WAIT_LGKM 7
	;;#ASMEND
	GEMM_MMA_I8 v[77:80], v[29:30], v[81:82], v[77:80]
	GEMM_MMA_I8 v[37:40], v[25:26], v[81:82], v[37:40]
	GEMM_MMA_I8 v[77:80], v[31:32], v[83:84], v[77:80]
	GEMM_MMA_I8 v[37:40], v[27:28], v[83:84], v[37:40]
	;;#ASMSTART
	GEMM_WAIT_LGKM 6
	;;#ASMEND
	GEMM_MMA_I8 v[69:72], v[29:30], v[86:87], v[69:72]
	GEMM_MMA_I8 v[33:36], v[25:26], v[86:87], v[33:36]
	GEMM_MMA_I8 v[69:72], v[31:32], v[88:89], v[69:72]
	GEMM_MMA_I8 v[33:36], v[27:28], v[88:89], v[33:36]
	;;#ASMSTART
	GEMM_WAIT_LGKM 5
	;;#ASMEND
	GEMM_MMA_I8 v[61:64], v[29:30], v[90:91], v[61:64]
	GEMM_MMA_I8 v[21:24], v[25:26], v[90:91], v[21:24]
	GEMM_MMA_I8 v[61:64], v[31:32], v[92:93], v[61:64]
	GEMM_MMA_I8 v[21:24], v[27:28], v[92:93], v[21:24]
	;;#ASMSTART
	GEMM_WAIT_LGKM 4
	;;#ASMEND
	GEMM_MMA_I8 v[57:60], v[29:30], v[94:95], v[57:60]
	GEMM_MMA_I8 v[17:20], v[25:26], v[94:95], v[17:20]
	GEMM_MMA_I8 v[57:60], v[31:32], v[96:97], v[57:60]
	GEMM_MMA_I8 v[17:20], v[27:28], v[96:97], v[17:20]
	;;#ASMSTART
	GEMM_WAIT_LGKM 3
	;;#ASMEND
	GEMM_MMA_I8 v[53:56], v[29:30], v[98:99], v[53:56]
	GEMM_MMA_I8 v[13:16], v[25:26], v[98:99], v[13:16]
	GEMM_MMA_I8 v[53:56], v[31:32], v[100:101], v[53:56]
	GEMM_MMA_I8 v[13:16], v[27:28], v[100:101], v[13:16]
	;;#ASMSTART
	GEMM_WAIT_LGKM 2
	;;#ASMEND
	GEMM_MMA_I8 v[49:52], v[29:30], v[102:103], v[49:52]
	GEMM_MMA_I8 v[9:12], v[25:26], v[102:103], v[9:12]
	GEMM_MMA_I8 v[49:52], v[31:32], v[104:105], v[49:52]
	GEMM_MMA_I8 v[9:12], v[27:28], v[104:105], v[9:12]
	;;#ASMSTART
	GEMM_WAIT_LGKM 1
	;;#ASMEND
	GEMM_MMA_I8 v[45:48], v[29:30], v[106:107], v[45:48]
	GEMM_MMA_I8 v[5:8], v[25:26], v[106:107], v[5:8]
	GEMM_MMA_I8 v[45:48], v[31:32], v[108:109], v[45:48]
	GEMM_MMA_I8 v[5:8], v[27:28], v[108:109], v[5:8]
	;;#ASMSTART
	GEMM_WAIT_LGKM 0
	;;#ASMEND
	GEMM_MMA_I8 v[41:44], v[29:30], v[110:111], v[41:44]
	GEMM_MMA_I8 v[1:4], v[25:26], v[110:111], v[1:4]
	GEMM_MMA_I8 v[41:44], v[31:32], v[112:113], v[41:44]
	GEMM_MMA_I8 v[1:4], v[27:28], v[112:113], v[1:4]
.LBB1_12:
s_waitcnt vmcnt(0)
v_and_b32_e32 v122, 15, v0
v_lshrrev_b32_e32 v123, 4, v0
v_and_b32_e32 v123, 3, v123
v_lshlrev_b32_e32 v123, 2, v123
s_lshl_b32 s2, s6, 7
v_add_u32_e32 v123, s2, v123
v_add_u32_e32 v122, v85, v122
v_lshlrev_b32_e32 v124, 2, v123
global_load_dwordx4 v[86:89], v124, s[26:27] offset:0
global_load_dwordx4 v[90:93], v124, s[26:27] offset:64
global_load_dwordx4 v[94:97], v124, s[26:27] offset:128
global_load_dwordx4 v[98:101], v124, s[26:27] offset:192
global_load_dwordx4 v[102:105], v124, s[26:27] offset:256
global_load_dwordx4 v[106:109], v124, s[26:27] offset:320
global_load_dwordx4 v[110:113], v124, s[26:27] offset:384
global_load_dwordx4 v[114:117], v124, s[26:27] offset:448
v_cmp_gt_u32_e32 vcc, s0, v122
s_and_saveexec_b64 s[2:3], vcc
s_cbranch_execz .L_E047_generic_bf16_skip_0
v_lshlrev_b32_e32 v119, 2, v122
global_load_dword v118, v119, s[24:25]
v_mul_u32_u24_e32 v119, 17408, v122
v_add_u32_e32 v119, v123, v119
v_lshlrev_b32_e32 v119, 1, v119
v_add_co_u32_e32 v120, vcc, s16, v119
v_mov_b32_e32 v121, s17
v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
s_waitcnt vmcnt(0)
v_cvt_f32_i32 v77, v77
v_mul_f32_e32 v119, v118, v86
v_mul_f32_e32 v77, v119, v77
v_lshrrev_b32_e32 v119, 16, v77
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v77, 0x7fff, v77
v_add_u32_e32 v77, v119, v77
v_cvt_f32_i32 v78, v78
v_mul_f32_e32 v119, v118, v87
v_mul_f32_e32 v78, v119, v78
v_lshrrev_b32_e32 v119, 16, v78
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v78, 0x7fff, v78
v_add_u32_e32 v78, v119, v78
v_cvt_f32_i32 v79, v79
v_mul_f32_e32 v119, v118, v88
v_mul_f32_e32 v79, v119, v79
v_lshrrev_b32_e32 v119, 16, v79
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v79, 0x7fff, v79
v_add_u32_e32 v79, v119, v79
v_cvt_f32_i32 v80, v80
v_mul_f32_e32 v119, v118, v89
v_mul_f32_e32 v80, v119, v80
v_lshrrev_b32_e32 v119, 16, v80
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v80, 0x7fff, v80
v_add_u32_e32 v80, v119, v80
v_lshrrev_b32_e32 v77, 16, v77
v_and_b32_e32 v78, 0xffff0000, v78
v_or_b32_e32 v77, v77, v78
v_lshrrev_b32_e32 v79, 16, v79
v_and_b32_e32 v80, 0xffff0000, v80
v_or_b32_e32 v78, v79, v80
global_store_dwordx2 v[120:121], v[77:78], off offset:0 glc slc
v_cvt_f32_i32 v69, v69
v_mul_f32_e32 v119, v118, v90
v_mul_f32_e32 v69, v119, v69
v_lshrrev_b32_e32 v119, 16, v69
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v69, 0x7fff, v69
v_add_u32_e32 v69, v119, v69
v_cvt_f32_i32 v70, v70
v_mul_f32_e32 v119, v118, v91
v_mul_f32_e32 v70, v119, v70
v_lshrrev_b32_e32 v119, 16, v70
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v70, 0x7fff, v70
v_add_u32_e32 v70, v119, v70
v_cvt_f32_i32 v71, v71
v_mul_f32_e32 v119, v118, v92
v_mul_f32_e32 v71, v119, v71
v_lshrrev_b32_e32 v119, 16, v71
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v71, 0x7fff, v71
v_add_u32_e32 v71, v119, v71
v_cvt_f32_i32 v72, v72
v_mul_f32_e32 v119, v118, v93
v_mul_f32_e32 v72, v119, v72
v_lshrrev_b32_e32 v119, 16, v72
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v72, 0x7fff, v72
v_add_u32_e32 v72, v119, v72
v_lshrrev_b32_e32 v69, 16, v69
v_and_b32_e32 v70, 0xffff0000, v70
v_or_b32_e32 v69, v69, v70
v_lshrrev_b32_e32 v71, 16, v71
v_and_b32_e32 v72, 0xffff0000, v72
v_or_b32_e32 v70, v71, v72
global_store_dwordx2 v[120:121], v[69:70], off offset:32 glc slc
v_cvt_f32_i32 v61, v61
v_mul_f32_e32 v119, v118, v94
v_mul_f32_e32 v61, v119, v61
v_lshrrev_b32_e32 v119, 16, v61
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v61, 0x7fff, v61
v_add_u32_e32 v61, v119, v61
v_cvt_f32_i32 v62, v62
v_mul_f32_e32 v119, v118, v95
v_mul_f32_e32 v62, v119, v62
v_lshrrev_b32_e32 v119, 16, v62
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v62, 0x7fff, v62
v_add_u32_e32 v62, v119, v62
v_cvt_f32_i32 v63, v63
v_mul_f32_e32 v119, v118, v96
v_mul_f32_e32 v63, v119, v63
v_lshrrev_b32_e32 v119, 16, v63
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v63, 0x7fff, v63
v_add_u32_e32 v63, v119, v63
v_cvt_f32_i32 v64, v64
v_mul_f32_e32 v119, v118, v97
v_mul_f32_e32 v64, v119, v64
v_lshrrev_b32_e32 v119, 16, v64
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v64, 0x7fff, v64
v_add_u32_e32 v64, v119, v64
v_lshrrev_b32_e32 v61, 16, v61
v_and_b32_e32 v62, 0xffff0000, v62
v_or_b32_e32 v61, v61, v62
v_lshrrev_b32_e32 v63, 16, v63
v_and_b32_e32 v64, 0xffff0000, v64
v_or_b32_e32 v62, v63, v64
global_store_dwordx2 v[120:121], v[61:62], off offset:64 glc slc
v_cvt_f32_i32 v57, v57
v_mul_f32_e32 v119, v118, v98
v_mul_f32_e32 v57, v119, v57
v_lshrrev_b32_e32 v119, 16, v57
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v57, 0x7fff, v57
v_add_u32_e32 v57, v119, v57
v_cvt_f32_i32 v58, v58
v_mul_f32_e32 v119, v118, v99
v_mul_f32_e32 v58, v119, v58
v_lshrrev_b32_e32 v119, 16, v58
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v58, 0x7fff, v58
v_add_u32_e32 v58, v119, v58
v_cvt_f32_i32 v59, v59
v_mul_f32_e32 v119, v118, v100
v_mul_f32_e32 v59, v119, v59
v_lshrrev_b32_e32 v119, 16, v59
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v59, 0x7fff, v59
v_add_u32_e32 v59, v119, v59
v_cvt_f32_i32 v60, v60
v_mul_f32_e32 v119, v118, v101
v_mul_f32_e32 v60, v119, v60
v_lshrrev_b32_e32 v119, 16, v60
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v60, 0x7fff, v60
v_add_u32_e32 v60, v119, v60
v_lshrrev_b32_e32 v57, 16, v57
v_and_b32_e32 v58, 0xffff0000, v58
v_or_b32_e32 v57, v57, v58
v_lshrrev_b32_e32 v59, 16, v59
v_and_b32_e32 v60, 0xffff0000, v60
v_or_b32_e32 v58, v59, v60
global_store_dwordx2 v[120:121], v[57:58], off offset:96 glc slc
v_cvt_f32_i32 v53, v53
v_mul_f32_e32 v119, v118, v102
v_mul_f32_e32 v53, v119, v53
v_lshrrev_b32_e32 v119, 16, v53
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v53, 0x7fff, v53
v_add_u32_e32 v53, v119, v53
v_cvt_f32_i32 v54, v54
v_mul_f32_e32 v119, v118, v103
v_mul_f32_e32 v54, v119, v54
v_lshrrev_b32_e32 v119, 16, v54
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v54, 0x7fff, v54
v_add_u32_e32 v54, v119, v54
v_cvt_f32_i32 v55, v55
v_mul_f32_e32 v119, v118, v104
v_mul_f32_e32 v55, v119, v55
v_lshrrev_b32_e32 v119, 16, v55
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v55, 0x7fff, v55
v_add_u32_e32 v55, v119, v55
v_cvt_f32_i32 v56, v56
v_mul_f32_e32 v119, v118, v105
v_mul_f32_e32 v56, v119, v56
v_lshrrev_b32_e32 v119, 16, v56
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v56, 0x7fff, v56
v_add_u32_e32 v56, v119, v56
v_lshrrev_b32_e32 v53, 16, v53
v_and_b32_e32 v54, 0xffff0000, v54
v_or_b32_e32 v53, v53, v54
v_lshrrev_b32_e32 v55, 16, v55
v_and_b32_e32 v56, 0xffff0000, v56
v_or_b32_e32 v54, v55, v56
global_store_dwordx2 v[120:121], v[53:54], off offset:128 glc slc
v_cvt_f32_i32 v49, v49
v_mul_f32_e32 v119, v118, v106
v_mul_f32_e32 v49, v119, v49
v_lshrrev_b32_e32 v119, 16, v49
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v49, 0x7fff, v49
v_add_u32_e32 v49, v119, v49
v_cvt_f32_i32 v50, v50
v_mul_f32_e32 v119, v118, v107
v_mul_f32_e32 v50, v119, v50
v_lshrrev_b32_e32 v119, 16, v50
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v50, 0x7fff, v50
v_add_u32_e32 v50, v119, v50
v_cvt_f32_i32 v51, v51
v_mul_f32_e32 v119, v118, v108
v_mul_f32_e32 v51, v119, v51
v_lshrrev_b32_e32 v119, 16, v51
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v51, 0x7fff, v51
v_add_u32_e32 v51, v119, v51
v_cvt_f32_i32 v52, v52
v_mul_f32_e32 v119, v118, v109
v_mul_f32_e32 v52, v119, v52
v_lshrrev_b32_e32 v119, 16, v52
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v52, 0x7fff, v52
v_add_u32_e32 v52, v119, v52
v_lshrrev_b32_e32 v49, 16, v49
v_and_b32_e32 v50, 0xffff0000, v50
v_or_b32_e32 v49, v49, v50
v_lshrrev_b32_e32 v51, 16, v51
v_and_b32_e32 v52, 0xffff0000, v52
v_or_b32_e32 v50, v51, v52
global_store_dwordx2 v[120:121], v[49:50], off offset:160 glc slc
v_cvt_f32_i32 v45, v45
v_mul_f32_e32 v119, v118, v110
v_mul_f32_e32 v45, v119, v45
v_lshrrev_b32_e32 v119, 16, v45
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v45, 0x7fff, v45
v_add_u32_e32 v45, v119, v45
v_cvt_f32_i32 v46, v46
v_mul_f32_e32 v119, v118, v111
v_mul_f32_e32 v46, v119, v46
v_lshrrev_b32_e32 v119, 16, v46
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v46, 0x7fff, v46
v_add_u32_e32 v46, v119, v46
v_cvt_f32_i32 v47, v47
v_mul_f32_e32 v119, v118, v112
v_mul_f32_e32 v47, v119, v47
v_lshrrev_b32_e32 v119, 16, v47
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v47, 0x7fff, v47
v_add_u32_e32 v47, v119, v47
v_cvt_f32_i32 v48, v48
v_mul_f32_e32 v119, v118, v113
v_mul_f32_e32 v48, v119, v48
v_lshrrev_b32_e32 v119, 16, v48
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v48, 0x7fff, v48
v_add_u32_e32 v48, v119, v48
v_lshrrev_b32_e32 v45, 16, v45
v_and_b32_e32 v46, 0xffff0000, v46
v_or_b32_e32 v45, v45, v46
v_lshrrev_b32_e32 v47, 16, v47
v_and_b32_e32 v48, 0xffff0000, v48
v_or_b32_e32 v46, v47, v48
global_store_dwordx2 v[120:121], v[45:46], off offset:192 glc slc
v_cvt_f32_i32 v41, v41
v_mul_f32_e32 v119, v118, v114
v_mul_f32_e32 v41, v119, v41
v_lshrrev_b32_e32 v119, 16, v41
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v41, 0x7fff, v41
v_add_u32_e32 v41, v119, v41
v_cvt_f32_i32 v42, v42
v_mul_f32_e32 v119, v118, v115
v_mul_f32_e32 v42, v119, v42
v_lshrrev_b32_e32 v119, 16, v42
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v42, 0x7fff, v42
v_add_u32_e32 v42, v119, v42
v_cvt_f32_i32 v43, v43
v_mul_f32_e32 v119, v118, v116
v_mul_f32_e32 v43, v119, v43
v_lshrrev_b32_e32 v119, 16, v43
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v43, 0x7fff, v43
v_add_u32_e32 v43, v119, v43
v_cvt_f32_i32 v44, v44
v_mul_f32_e32 v119, v118, v117
v_mul_f32_e32 v44, v119, v44
v_lshrrev_b32_e32 v119, 16, v44
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v44, 0x7fff, v44
v_add_u32_e32 v44, v119, v44
v_lshrrev_b32_e32 v41, 16, v41
v_and_b32_e32 v42, 0xffff0000, v42
v_or_b32_e32 v41, v41, v42
v_lshrrev_b32_e32 v43, 16, v43
v_and_b32_e32 v44, 0xffff0000, v44
v_or_b32_e32 v42, v43, v44
global_store_dwordx2 v[120:121], v[41:42], off offset:224 glc slc
.L_E047_generic_bf16_skip_0:
s_or_b64 exec, exec, s[2:3]
v_add_u32_e32 v122, 16, v122
v_cmp_gt_u32_e32 vcc, s0, v122
s_and_saveexec_b64 s[2:3], vcc
s_cbranch_execz .L_E047_generic_bf16_skip_1
v_lshlrev_b32_e32 v119, 2, v122
global_load_dword v118, v119, s[24:25]
v_mul_u32_u24_e32 v119, 17408, v122
v_add_u32_e32 v119, v123, v119
v_lshlrev_b32_e32 v119, 1, v119
v_add_co_u32_e32 v120, vcc, s16, v119
v_mov_b32_e32 v121, s17
v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
s_waitcnt vmcnt(0)
v_cvt_f32_i32 v37, v37
v_mul_f32_e32 v119, v118, v86
v_mul_f32_e32 v37, v119, v37
v_lshrrev_b32_e32 v119, 16, v37
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v37, 0x7fff, v37
v_add_u32_e32 v37, v119, v37
v_cvt_f32_i32 v38, v38
v_mul_f32_e32 v119, v118, v87
v_mul_f32_e32 v38, v119, v38
v_lshrrev_b32_e32 v119, 16, v38
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v38, 0x7fff, v38
v_add_u32_e32 v38, v119, v38
v_cvt_f32_i32 v39, v39
v_mul_f32_e32 v119, v118, v88
v_mul_f32_e32 v39, v119, v39
v_lshrrev_b32_e32 v119, 16, v39
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v39, 0x7fff, v39
v_add_u32_e32 v39, v119, v39
v_cvt_f32_i32 v40, v40
v_mul_f32_e32 v119, v118, v89
v_mul_f32_e32 v40, v119, v40
v_lshrrev_b32_e32 v119, 16, v40
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v40, 0x7fff, v40
v_add_u32_e32 v40, v119, v40
v_lshrrev_b32_e32 v37, 16, v37
v_and_b32_e32 v38, 0xffff0000, v38
v_or_b32_e32 v37, v37, v38
v_lshrrev_b32_e32 v39, 16, v39
v_and_b32_e32 v40, 0xffff0000, v40
v_or_b32_e32 v38, v39, v40
global_store_dwordx2 v[120:121], v[37:38], off offset:0 glc slc
v_cvt_f32_i32 v33, v33
v_mul_f32_e32 v119, v118, v90
v_mul_f32_e32 v33, v119, v33
v_lshrrev_b32_e32 v119, 16, v33
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v33, 0x7fff, v33
v_add_u32_e32 v33, v119, v33
v_cvt_f32_i32 v34, v34
v_mul_f32_e32 v119, v118, v91
v_mul_f32_e32 v34, v119, v34
v_lshrrev_b32_e32 v119, 16, v34
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v34, 0x7fff, v34
v_add_u32_e32 v34, v119, v34
v_cvt_f32_i32 v35, v35
v_mul_f32_e32 v119, v118, v92
v_mul_f32_e32 v35, v119, v35
v_lshrrev_b32_e32 v119, 16, v35
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v35, 0x7fff, v35
v_add_u32_e32 v35, v119, v35
v_cvt_f32_i32 v36, v36
v_mul_f32_e32 v119, v118, v93
v_mul_f32_e32 v36, v119, v36
v_lshrrev_b32_e32 v119, 16, v36
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v36, 0x7fff, v36
v_add_u32_e32 v36, v119, v36
v_lshrrev_b32_e32 v33, 16, v33
v_and_b32_e32 v34, 0xffff0000, v34
v_or_b32_e32 v33, v33, v34
v_lshrrev_b32_e32 v35, 16, v35
v_and_b32_e32 v36, 0xffff0000, v36
v_or_b32_e32 v34, v35, v36
global_store_dwordx2 v[120:121], v[33:34], off offset:32 glc slc
v_cvt_f32_i32 v21, v21
v_mul_f32_e32 v119, v118, v94
v_mul_f32_e32 v21, v119, v21
v_lshrrev_b32_e32 v119, 16, v21
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v21, 0x7fff, v21
v_add_u32_e32 v21, v119, v21
v_cvt_f32_i32 v22, v22
v_mul_f32_e32 v119, v118, v95
v_mul_f32_e32 v22, v119, v22
v_lshrrev_b32_e32 v119, 16, v22
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v22, 0x7fff, v22
v_add_u32_e32 v22, v119, v22
v_cvt_f32_i32 v23, v23
v_mul_f32_e32 v119, v118, v96
v_mul_f32_e32 v23, v119, v23
v_lshrrev_b32_e32 v119, 16, v23
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v23, 0x7fff, v23
v_add_u32_e32 v23, v119, v23
v_cvt_f32_i32 v24, v24
v_mul_f32_e32 v119, v118, v97
v_mul_f32_e32 v24, v119, v24
v_lshrrev_b32_e32 v119, 16, v24
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v24, 0x7fff, v24
v_add_u32_e32 v24, v119, v24
v_lshrrev_b32_e32 v21, 16, v21
v_and_b32_e32 v22, 0xffff0000, v22
v_or_b32_e32 v21, v21, v22
v_lshrrev_b32_e32 v23, 16, v23
v_and_b32_e32 v24, 0xffff0000, v24
v_or_b32_e32 v22, v23, v24
global_store_dwordx2 v[120:121], v[21:22], off offset:64 glc slc
v_cvt_f32_i32 v17, v17
v_mul_f32_e32 v119, v118, v98
v_mul_f32_e32 v17, v119, v17
v_lshrrev_b32_e32 v119, 16, v17
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v17, 0x7fff, v17
v_add_u32_e32 v17, v119, v17
v_cvt_f32_i32 v18, v18
v_mul_f32_e32 v119, v118, v99
v_mul_f32_e32 v18, v119, v18
v_lshrrev_b32_e32 v119, 16, v18
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v18, 0x7fff, v18
v_add_u32_e32 v18, v119, v18
v_cvt_f32_i32 v19, v19
v_mul_f32_e32 v119, v118, v100
v_mul_f32_e32 v19, v119, v19
v_lshrrev_b32_e32 v119, 16, v19
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v19, 0x7fff, v19
v_add_u32_e32 v19, v119, v19
v_cvt_f32_i32 v20, v20
v_mul_f32_e32 v119, v118, v101
v_mul_f32_e32 v20, v119, v20
v_lshrrev_b32_e32 v119, 16, v20
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v20, 0x7fff, v20
v_add_u32_e32 v20, v119, v20
v_lshrrev_b32_e32 v17, 16, v17
v_and_b32_e32 v18, 0xffff0000, v18
v_or_b32_e32 v17, v17, v18
v_lshrrev_b32_e32 v19, 16, v19
v_and_b32_e32 v20, 0xffff0000, v20
v_or_b32_e32 v18, v19, v20
global_store_dwordx2 v[120:121], v[17:18], off offset:96 glc slc
v_cvt_f32_i32 v13, v13
v_mul_f32_e32 v119, v118, v102
v_mul_f32_e32 v13, v119, v13
v_lshrrev_b32_e32 v119, 16, v13
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v13, 0x7fff, v13
v_add_u32_e32 v13, v119, v13
v_cvt_f32_i32 v14, v14
v_mul_f32_e32 v119, v118, v103
v_mul_f32_e32 v14, v119, v14
v_lshrrev_b32_e32 v119, 16, v14
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v14, 0x7fff, v14
v_add_u32_e32 v14, v119, v14
v_cvt_f32_i32 v15, v15
v_mul_f32_e32 v119, v118, v104
v_mul_f32_e32 v15, v119, v15
v_lshrrev_b32_e32 v119, 16, v15
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v15, 0x7fff, v15
v_add_u32_e32 v15, v119, v15
v_cvt_f32_i32 v16, v16
v_mul_f32_e32 v119, v118, v105
v_mul_f32_e32 v16, v119, v16
v_lshrrev_b32_e32 v119, 16, v16
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v16, 0x7fff, v16
v_add_u32_e32 v16, v119, v16
v_lshrrev_b32_e32 v13, 16, v13
v_and_b32_e32 v14, 0xffff0000, v14
v_or_b32_e32 v13, v13, v14
v_lshrrev_b32_e32 v15, 16, v15
v_and_b32_e32 v16, 0xffff0000, v16
v_or_b32_e32 v14, v15, v16
global_store_dwordx2 v[120:121], v[13:14], off offset:128 glc slc
v_cvt_f32_i32 v9, v9
v_mul_f32_e32 v119, v118, v106
v_mul_f32_e32 v9, v119, v9
v_lshrrev_b32_e32 v119, 16, v9
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v9, 0x7fff, v9
v_add_u32_e32 v9, v119, v9
v_cvt_f32_i32 v10, v10
v_mul_f32_e32 v119, v118, v107
v_mul_f32_e32 v10, v119, v10
v_lshrrev_b32_e32 v119, 16, v10
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v10, 0x7fff, v10
v_add_u32_e32 v10, v119, v10
v_cvt_f32_i32 v11, v11
v_mul_f32_e32 v119, v118, v108
v_mul_f32_e32 v11, v119, v11
v_lshrrev_b32_e32 v119, 16, v11
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v11, 0x7fff, v11
v_add_u32_e32 v11, v119, v11
v_cvt_f32_i32 v12, v12
v_mul_f32_e32 v119, v118, v109
v_mul_f32_e32 v12, v119, v12
v_lshrrev_b32_e32 v119, 16, v12
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v12, 0x7fff, v12
v_add_u32_e32 v12, v119, v12
v_lshrrev_b32_e32 v9, 16, v9
v_and_b32_e32 v10, 0xffff0000, v10
v_or_b32_e32 v9, v9, v10
v_lshrrev_b32_e32 v11, 16, v11
v_and_b32_e32 v12, 0xffff0000, v12
v_or_b32_e32 v10, v11, v12
global_store_dwordx2 v[120:121], v[9:10], off offset:160 glc slc
v_cvt_f32_i32 v5, v5
v_mul_f32_e32 v119, v118, v110
v_mul_f32_e32 v5, v119, v5
v_lshrrev_b32_e32 v119, 16, v5
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v5, 0x7fff, v5
v_add_u32_e32 v5, v119, v5
v_cvt_f32_i32 v6, v6
v_mul_f32_e32 v119, v118, v111
v_mul_f32_e32 v6, v119, v6
v_lshrrev_b32_e32 v119, 16, v6
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v6, 0x7fff, v6
v_add_u32_e32 v6, v119, v6
v_cvt_f32_i32 v7, v7
v_mul_f32_e32 v119, v118, v112
v_mul_f32_e32 v7, v119, v7
v_lshrrev_b32_e32 v119, 16, v7
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v7, 0x7fff, v7
v_add_u32_e32 v7, v119, v7
v_cvt_f32_i32 v8, v8
v_mul_f32_e32 v119, v118, v113
v_mul_f32_e32 v8, v119, v8
v_lshrrev_b32_e32 v119, 16, v8
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v8, 0x7fff, v8
v_add_u32_e32 v8, v119, v8
v_lshrrev_b32_e32 v5, 16, v5
v_and_b32_e32 v6, 0xffff0000, v6
v_or_b32_e32 v5, v5, v6
v_lshrrev_b32_e32 v7, 16, v7
v_and_b32_e32 v8, 0xffff0000, v8
v_or_b32_e32 v6, v7, v8
global_store_dwordx2 v[120:121], v[5:6], off offset:192 glc slc
v_cvt_f32_i32 v1, v1
v_mul_f32_e32 v119, v118, v114
v_mul_f32_e32 v1, v119, v1
v_lshrrev_b32_e32 v119, 16, v1
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v1, 0x7fff, v1
v_add_u32_e32 v1, v119, v1
v_cvt_f32_i32 v2, v2
v_mul_f32_e32 v119, v118, v115
v_mul_f32_e32 v2, v119, v2
v_lshrrev_b32_e32 v119, 16, v2
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v2, 0x7fff, v2
v_add_u32_e32 v2, v119, v2
v_cvt_f32_i32 v3, v3
v_mul_f32_e32 v119, v118, v116
v_mul_f32_e32 v3, v119, v3
v_lshrrev_b32_e32 v119, 16, v3
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v3, 0x7fff, v3
v_add_u32_e32 v3, v119, v3
v_cvt_f32_i32 v4, v4
v_mul_f32_e32 v119, v118, v117
v_mul_f32_e32 v4, v119, v4
v_lshrrev_b32_e32 v119, 16, v4
v_and_b32_e32 v119, 1, v119
v_add_u32_e32 v4, 0x7fff, v4
v_add_u32_e32 v4, v119, v4
v_lshrrev_b32_e32 v1, 16, v1
v_and_b32_e32 v2, 0xffff0000, v2
v_or_b32_e32 v1, v1, v2
v_lshrrev_b32_e32 v3, 16, v3
v_and_b32_e32 v4, 0xffff0000, v4
v_or_b32_e32 v2, v3, v4
global_store_dwordx2 v[120:121], v[1:2], off offset:224 glc slc
.L_E047_generic_bf16_skip_1:
s_or_b64 exec, exec, s[2:3]
s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 28
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii,#alloc,#execinstr
.Lfunc_end1:
	.size	_Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii, .Lfunc_end1-_Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 2728
; NumSgprs: 27
; NumVgprs: 128
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 27
; NumVGPRsForWavesPerEU: 128
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.no_dead_strip	_ZZ8reg_E047ILi0ELi0ELi0EEvPKaS1_PiiiiiE5lwbuf
	.ident	"clang version 17.0.0 (http://10.16.1.181/dcutoolkit/driverruntime/llvm-project.git b8c5e6d901d63a10d73acf48579414a1ea463c62)"
	.section	".note.GNU-stack"
	.addrsig
	.addrsig_sym _ZZ8reg_E047ILi0ELi0ELi0EEvPKaS1_PiiiiiE5lwbuf
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
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .offset: 24
    .size: 4
    .value_kind: by_value
  - .offset: 28
    .size: 4
    .value_kind: by_value
  - .offset: 32
    .size: 4
    .value_kind: by_value
  - .offset: 36
    .size: 4
    .value_kind: by_value
  - .address_space: global
    .offset: 40
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 48
    .size: 8
    .value_kind: global_buffer
  .fp64_status: 0
  .group_segment_fixed_size: 16384
  .kernarg_segment_align: 8
  .kernarg_segment_size: 56
  .language: OpenCL C
  .language_version:
  - 2
  - 0
  .max_flat_workgroup_size: 256
  .name: _Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii
  .private_segment_fixed_size: 0
  .sgpr_count: 32
  .sgpr_spill_count: 0
  .symbol: _Z8reg_E047ILi128ELi17408ELi5120EEvPKaS1_Piiiii.kd
  .unfolded_args:
  - .address_space: global
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .offset: 24
    .size: 4
    .value_kind: by_value
  - .offset: 28
    .size: 4
    .value_kind: by_value
  - .offset: 32
    .size: 4
    .value_kind: by_value
  - .offset: 36
    .size: 4
    .value_kind: by_value
  - .address_space: global
    .offset: 40
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 48
    .size: 8
    .value_kind: global_buffer
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 123
  .vgpr_spill_count: 0
  .wavefront_size: 64
- .args:
  - .address_space: global
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .offset: 24
    .size: 4
    .value_kind: by_value
  - .offset: 28
    .size: 4
    .value_kind: by_value
  - .offset: 32
    .size: 4
    .value_kind: by_value
  - .offset: 36
    .size: 4
    .value_kind: by_value
  - .address_space: global
    .offset: 40
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 48
    .size: 8
    .value_kind: global_buffer
  .fp64_status: 0
  .group_segment_fixed_size: 16384
  .kernarg_segment_align: 8
  .kernarg_segment_size: 56
  .language: OpenCL C
  .language_version:
  - 2
  - 0
  .max_flat_workgroup_size: 256
  .name: _Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii
  .private_segment_fixed_size: 0
  .sgpr_count: 32
  .sgpr_spill_count: 0
  .symbol: _Z8reg_E047ILi0ELi0ELi0EEvPKaS1_Piiiii.kd
  .unfolded_args:
  - .address_space: global
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .offset: 24
    .size: 4
    .value_kind: by_value
  - .offset: 28
    .size: 4
    .value_kind: by_value
  - .offset: 32
    .size: 4
    .value_kind: by_value
  - .offset: 36
    .size: 4
    .value_kind: by_value
  - .address_space: global
    .offset: 40
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 48
    .size: 8
    .value_kind: global_buffer
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 128
  .vgpr_spill_count: 0
  .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx936:sramecc+
amdhsa.version:
- 1
- 2
...
.end_amdgpu_metadata
