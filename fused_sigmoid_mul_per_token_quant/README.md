# Attention sigmoid-mul + per-token INT8

对应 LightOp kernel：`fused_sigmoid_mul_per_token_quant.cu`。

计算 `attn_output * sigmoid(gate)`，保留 BF16/FP16 中间舍入，然后进行 per-token INT8 量化。对对齐的 `hidden=4096, head_dim=256` 生产布局使用 Vec8 快路径，其余布局走通用 fallback。`reduce.h` 是该 kernel 所需的共享归约辅助头文件。

来源：LightOp Attention PR !75，最终提交 `6866db940198c266e1688fd797879ced17ea132a`。