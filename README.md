# hcu_kernel

海光 HCU/DTK 算子源码归档。每个算子放在独立目录；融合算子来自 LightOp 已验证实现，decode GEMM 为独立 HIP 实现。

## 算子

- `decode_gemm_int8_gfx936/`：gfx936 INT8 decode GEMM，含14种HIP配置、独立编译/Python入口、正确性与冷权重池基准，以及TP1/2/4/8、bs1–1024结果摘要

- `gemma_rms_norm_dynamic_per_token_quant/`：Gemma RMSNorm + per-token INT8 quant
- `fuse_rms_norm_gated_per_token_quant_int8/`：GDN RMSNorm + SiLU gate + per-token INT8 quant
- `fused_sigmoid_mul_per_token_quant/`：Attention sigmoid-mul + per-token INT8 quant

LightOp 目录保留 kernel、Python 接口、必要的 `reduce.h` 和 focused tests/benchmark helper；完整注册、编译系统和运行时依赖仍由上游 LightOp 工程提供。GEMM 目录可用 DTK hipcc 独立编译，调用依赖 HCU 版 PyTorch。

本仓库不包含 LightOp PR 的原始 profiling JSON、trace 或长篇证据文档。
