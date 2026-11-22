# gfx936 非融合 GateUp GEMM

输入 K=5120。TP1 的 N=34816，TP2 每 rank 的 N=17408。支持 M=1–4096，按输入权重的列顺序写出 BF16 `Y[M,N]`。

## BF16 单份 W[K,N]

`BF16GateUpGemm` 直接持有调用方传入的连续 BF16 `W[5120,N]`。如果 checkpoint 保存为 `W[N,5120]`，加载时转置一次即可。

M=1–256 使用带实测路由的 GEMV、rolling MMAC、`torch.mm` 和 rocBLAS。M=257–4096 使用已有的单 W `torch.mm` / rocBLAS 路由。部分小 M 路由会补齐 X 和输出行数，再返回前 M 行。传入精确形状的 `out[M,N]` 时使用对应的实测路由。

从仓库根目录构建。所需环境为 gfx936 BW1000、DTK 和 HCU 版 PyTorch / Triton。

```bash
source /opt/dtk/env.sh
python -m gate_up_gemm_gfx936.build_bf16
```

```python
from gate_up_gemm_gfx936.bf16 import BF16GateUpGemm

op = BF16GateUpGemm(weight_kn)
y = op(x)
```

使用预分配缓冲区时，`workspace_elements(M, exact=True)` 返回精确 `[M,N]` 输出路径的 X 工作区元素数。用 `output_rows(M)` 查询默认路径的物理输出行数。

推理时可以在目标 stream 预热后捕获 HIP Graph。所有工作完成并释放 Graph 后，用 `op.close()` 释放该实例的 rocBLAS handle。

`bf16-policy.json` 每条记录包含当前路由的 M 上界，该条覆盖上一条上界之后的所有 M。`exact` 只记录 M=1–256，之后沿用 `default`。rolling 汇编复用仓库中的 LM head 生成器。

在 BW47 GPU2 上重建后，8 个 rolling HSACO 与实验版逐字节一致。TP1/TP2 的全量路由和缓冲区尺寸一致，184 组实跑输出逐元素一致。各 TP 的 Graph 时延中位数变化均在 0.02% 内。
