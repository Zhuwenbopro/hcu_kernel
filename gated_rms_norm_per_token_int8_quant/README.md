# GDN gated RMSNorm + per-token INT8

Triton 实现与 [LightOp PR #181](https://github.com/HYGON-AI/lightop/pull/181) 的
`043d42d` 对齐。逐个 128 维 value head 做 RMSNorm，乘 SiLU gate 后舍入为
BF16，再跨本 rank 的全部 heads 计算每 token 的 INT8 scale。gfx936 的
24-head 路径将 heads 分成三组，每组 8 个。源码可直接由 Triton JIT 编译。

```python
from gated_rms_norm_per_token_int8_quant import gated_rms_norm_per_token_int8_quant

q, scale = gated_rms_norm_per_token_int8_quant(x, z, weight, 1e-6, num_heads=24)
```

`x` 为 `[tokens * num_heads, 128]`，`z` 与 `x` 同形状或为
`[tokens, num_heads, 128]` 的投影视图，最后一维连续。共享权重为 `[128]`。
输入均为同卡 BF16，输出为 `[tokens, num_heads * 128]` 的 INT8 和
`[tokens, 1]` 的 FP32 scale。

在 DTK、HIP PyTorch 和 Triton 环境中，从仓库根目录运行：

```bash
python -m pytest -q gated_rms_norm_per_token_int8_quant
python -m gated_rms_norm_per_token_int8_quant.benchmark
```

benchmark 使用已安装 LightOp 的 `layer_norm_fwd_1pass_opt` 和
`per_token_quant_int8` 作为两段式基线。基线与融合版在同卡交错 Graph 重放。
