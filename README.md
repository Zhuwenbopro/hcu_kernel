# hcu_kernel

海光 HCU/DTK 算子源码归档。每个算子放在独立目录；LightOp 融合算子来自已验证实现，GEMM 包含独立 HIP 与模块化汇编实现。

## 算子

- [fused_gate_up_gemm_gfx936/](fused_gate_up_gemm_gfx936/)：gfx936 INT8/BF16 特殊布局 gate/up GEMM 融合 SiLU×up，直接输出 BF16 [M,8704]；含源码快照、完整可校验复现包和正确性/性能记录

- `decode_gemm_int8_gfx936/`：gfx936 INT8 decode GEMM，含14种HIP配置、独立编译/Python入口、正确性与冷权重池基准，以及TP1/2/4/8、bs1–1024结果摘要

- `gemma_rms_norm_dynamic_per_token_quant/`：Gemma RMSNorm + per-token INT8 quant
- [gemma_fused_add_rmsnorm_int8_fp32_gfx936/](gemma_fused_add_rmsnorm_int8_fp32_gfx936/) 支持 5120 维的 Gemma 残差加法、RMSNorm 和 per-token INT8 量化，在 FP32 中合并平方和与带权最大值规约，支持独立 HIP 编译
- `fuse_rms_norm_gated_per_token_quant_int8/`：GDN RMSNorm + SiLU gate + per-token INT8 quant
- `fused_sigmoid_mul_per_token_quant/`：Attention sigmoid-mul + per-token INT8 quant

LightOp 目录保留 kernel、Python 接口、必要的 `reduce.h` 和 focused tests/benchmark helper；完整注册、编译系统和运行时依赖仍由上游 LightOp 工程提供。GEMM 目录可用 DTK hipcc 独立编译，调用依赖 HCU 版 PyTorch。

本仓库不包含 LightOp PR 的原始 profiling JSON、trace 或长篇证据文档。
