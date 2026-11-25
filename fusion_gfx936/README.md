# gfx936 融合算子

这四个算子都输出逐 token 量化的 INT8 激活和 FP32 scale。子目录按 `<算子>_per_token_int8_quant` 命名。下表链接到各算子的调用示例、构建或测试命令及性能图。

| 算子 | 实现 | 输入与行为 |
| --- | --- | --- |
| [Attention sigmoid × mul](sigmoid_mul_per_token_int8_quant/) | Triton | 用 sigmoid gate 乘 attention 输出，支持连续 gate 和 QKV 投影视图。 |
| [GDN gated RMSNorm](gated_rmsnorm_per_token_int8_quant/) | Triton | 逐 head 做 RMSNorm 和 SiLU 门控，跨本地 heads 计算每个 token 的 scale。 |
| [Gemma add + RMSNorm](gemma_add_rmsnorm_per_token_int8_quant/) | HIP、Python 接口 | 融合 5,120 维残差加法和 RMSNorm，保留 FP32 中间值，并原地写回 BF16 残差。 |
| [SiLU × up](silu_mul_per_token_int8_quant/) | HIP、C 接口 | 分别提供 gate/up 分开存放和逐元素交错的入口，直接使用上游写好的布局。 |
