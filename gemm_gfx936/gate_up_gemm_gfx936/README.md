# gfx936 非融合 GateUp GEMM

输入 K=5120。TP1 的 N=34816，TP2 每 rank 的 N=17408。支持 M=1–4096，按输入权重的列顺序写出 BF16 `Y[M,N]`。

## BF16 单份 W[K,N]

`BF16GateUpGemm` 直接持有调用方传入的连续 BF16 `W[5120,N]`。如果 checkpoint 保存为 `W[N,5120]`，加载时转置一次即可。

M=1–256 使用带实测路由的 GEMV、rolling MMAC、`torch.mm` 和 rocBLAS。M=257–4096 使用单 W `torch.mm` / rocBLAS，M=257–2048 的部分尾块在默认路径补齐实际计算和输出行数。补齐保留原 X 的物理 pitch、分段方式和本轮实测的内核及 CTA 网格。输出返回前 M 行。传入精确形状的 `out[M,N]` 时走精确输出路径。

从仓库根目录构建。所需环境为 gfx936 BW1000、DTK 和 HCU 版 PyTorch / Triton。

```bash
source /opt/dtk/env.sh
python -m gemm_gfx936.gate_up_gemm_gfx936.build_bf16
```

```python
from gemm_gfx936.gate_up_gemm_gfx936.bf16 import BF16GateUpGemm

op = BF16GateUpGemm(weight_kn)
y = op(x)
```

使用预分配缓冲区时，`workspace_elements(M, exact=True)` 返回精确 `[M,N]` 输出路径的 X 工作区元素数。用 `output_rows(M)` 查询默认路径的物理输出行数。

推理时可以在目标 stream 预热后捕获 HIP Graph。所有工作完成并释放 Graph 后，用 `op.close()` 释放该实例的 rocBLAS handle。

`bf16-policy.json` 每条记录包含当前路由的 M 上界，该条覆盖上一条上界之后的所有 M。`exact` 记录 M=1–256 的精确输出路由，后续精确输出沿用基础 `default`。`tail` 记录只有默认输出路径应用的逐 M 对齐粒度。rolling 汇编复用仓库中的 LM head 生成器。

在 BW47 GPU2 上重建后，8 个 rolling HSACO 与实验版逐字节一致。TP1/TP2 的全量路由和缓冲区尺寸一致，184 组实跑输出逐元素一致。各 TP 的 Graph 时延中位数变化均在 0.02% 内。

BW47 尾块实验在 TP1 和 TP2 各逐点复测 M=1–2048。TP1 保留 332 个补齐点，TP2 保留 114 个。M=257–2048 的 Graph 平均时延分别从 1.262 降至 1.212 ms、从 0.643 降至 0.638 ms。计时包含在线 X 转置和 BF16 写回。

## INT8 单份 packed W

`Int8GateUpGemm` 用一份 packed INT8 W 和 FP32 `scale_w[N,1]`。每次调用接受连续 INT8 `X[M,5120]` 和 FP32 `scale_x[M,1]`。固定寄存器汇编完成 INT32 累加，乘以两份 scale 后写回 BF16 `Y[M,N]`。

`pack_int8_weight` 在权重加载时把连续 `W[N,5120]` 转成不透明的一维存储。布局按 K64 分段，每段按 16 行一组重新排列。初始化算子后即可释放原始 INT8 W。

```bash
source /opt/dtk/env.sh
python -m gemm_gfx936.gate_up_gemm_gfx936.build_int8
```

```python
from gemm_gfx936.gate_up_gemm_gfx936.int8 import Int8GateUpGemm, pack_int8_weight

packed_w = pack_int8_weight(weight_nk)
op = Int8GateUpGemm(packed_w, scale_w)
y = op(x, scale_x)
```

可以传入精确形状的 `out[M,N]` 和工作区。`workspace_elements(M)` 返回当前路由所需的 INT8 元素数，包含补齐和已选路由的 X pitch 偏移。权重准备、X 打包和 GEMM 会提交到调用方当前 stream。跨 stream 交接权重或复用工作区时，由调用方同步。

`int8-policy.json` 记录 M=1–256 的原有 packed 路由和 M=257–4096 的已验证优化路由。汇编构建使用 PyYAML 并复用 `fused_gate_up_gemm_gfx936` 的公共装载、MMAC 和同步宏。

在 BW47 GPU2 上重建后，50 个 INT8 HSACO 与实验版逐字节一致。TP1/TP2 的全量路由和 X 工作区尺寸一致，168 组实跑输出逐元素一致。各 TP 的 Graph 时延中位数变化均在 0.02% 内。TP1 的 M=4095 首轮慢 4.4%，6 次交替重捕获的时延比中位数为 +0.76%。
