# hcu_kernel

海光 HCU/DTK 上的三个融合算子源码归档。每个算子放在独立目录，代码来自 LightOp 中已完成验证的实现。

## 算子

- `gemma_rms_norm_dynamic_per_token_quant/`：Gemma RMSNorm + per-token INT8 quant
- `fuse_rms_norm_gated_per_token_quant_int8/`：GDN RMSNorm + SiLU gate + per-token INT8 quant
- `fused_sigmoid_mul_per_token_quant/`：Attention sigmoid-mul + per-token INT8 quant

目录中保留 kernel、Python 接口、必要的 `reduce.h` 和 focused tests/benchmark helper；LightOp 的完整注册、编译系统和运行时依赖仍由上游 LightOp 工程提供。

本仓库不包含 LightOp PR 的原始 profiling JSON、trace 或长篇证据文档。
