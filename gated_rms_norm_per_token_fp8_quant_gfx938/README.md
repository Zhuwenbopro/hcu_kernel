# gfx938 GDN gated RMSNorm + per-token FP8

Triton 实现取自 [LightOp PR #183](https://github.com/HYGON-AI/lightop/pull/183) 的 gfx938 路径。每个 Triton program 处理一个 token。先按 128 维 value head 做 RMSNorm，乘 SiLU gate 后舍入为 BF16，再跨该 token 在本 rank 上的全部 heads 计算一个 FP32 scale。输出采用 E4M3FN，scale 为 `absmax / 448`，全零 token 的 scale 为零。

```python
from gated_rms_norm_per_token_fp8_quant_gfx938 import gated_rms_norm_per_token_fp8_quant

q, scale = gated_rms_norm_per_token_fp8_quant(x, z, weight, 1e-6, num_heads=24)
```

输入为同一 gfx938 设备上的 BF16 张量。支持 1 至 64 个 value heads。输入 `x` 的形状为 `[tokens * num_heads, 128]`，`z` 可以与 `x` 同形状，也可以是 `[tokens, num_heads, 128]` 的带 stride 投影视图。两者最后一维都需要连续，权重为共享的 `[128]` 向量。返回 `[tokens, num_heads * 128]` 的 FP8 激活及 `[tokens, 1]` 的 FP32 scale。

在 DTK、HIP PyTorch 和 Triton 环境中，从仓库根目录运行以下命令。benchmark 另需 LightOp 提供两段式对照 `layer_norm_fwd_1pass_opt` 和 `per_token_quant_fp8`，在同卡交错重放 Graph。

```bash
python -m pytest -q gated_rms_norm_per_token_fp8_quant_gfx938

HIP_GRAPH_ACCUMULATE_DISPATCH=1 ACCUMULATE_DISPATCH_PROFILE_LEVEL=FULL_PROFILE \
python -m gated_rms_norm_per_token_fp8_quant_gfx938.benchmark
```

gfx938 已验证 1、12、24、48 和 64 heads 的离线编译，其中 24 heads 覆盖 4 和 8 warps。gfx938 硬件上的数值和性能待验证。
