# GDN RMSNorm + SiLU gate + per-token INT8

对应 LightOp kernel：`fuse_rms_norm_gated_per_token_quant_int8.cu`。

输入为 `[tokens, heads * 128]`，每个 head 做 128 维 RMSNorm 和 SiLU gate，scale 在 token 的所有 head 上归约。`reduce.h` 是该 kernel 所需的共享归约辅助头文件。

来源：LightOp GDN PR !74，最终提交 `449cb9e8e7e4c914d20f8feb10ed82f1c9a0a8f1`。