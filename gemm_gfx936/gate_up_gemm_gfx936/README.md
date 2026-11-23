# gfx936 非融合 GateUp GEMM

输入 K=5120。TP1 的 N=34816，TP2 每 rank 的 N=17408。支持 M=1–4096，按输入权重的列顺序写出 BF16 `Y[M,N]`。

## BF16 单份 W[K,N]

`BF16GateUpGemm` 直接持有调用方传入的连续 BF16 `W[5120,N]`。如果 checkpoint 保存为 `W[N,5120]`，加载时转置一次即可。

M=1–256 使用实测选择的 GEMV、rolling MMAC、`torch.mm` 或 rocBLAS。M=257–4096 使用同一份 W 的 `torch.mm` 或 rocBLAS。TP2 默认路径按 M 选择是否转置 X、沿 N 分段，以及是否将 X 的 pitch 和计算行数补齐到 16、32、64 或 128 的倍数。补齐路径只返回前 M 行。显式传入 `out[M,N]` 时只用物理输出恰好为 M 行的路由。

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

`bf16-policy.json` 中，`default` 和 `exact` 记录一段连续 M 的上界。精确输出在 M≤256 使用 `exact`，之后以 `default` 为基础。`tail` 记录默认路径的尾块对齐。TP2 的 `overrides` 按闭区间 `[first, last, route]` 覆盖默认路径。精确输出可以复用其中物理输出为 M 行、且 X 工作区需求不增加的路由。 `exact_overrides` 另外记录单独用 `out[M,N]` 实测的路由。rolling 汇编复用仓库中的 LM head 生成器。

在 BW47 GPU2 上重建后，8 个 rolling HSACO 与实验版逐字节一致。TP1/TP2 的全量路由和缓冲区尺寸一致，184 组实跑输出逐元素一致。各 TP 的 Graph 时延中位数变化均在 0.02% 内。

BW47 尾块实验在 TP1 和 TP2 各逐点复测 M=1–2048。TP1 保留 332 个补齐点，TP2 保留 114 个。M=257–2048 的 Graph 平均时延分别从 1.262 降至 1.212 ms、从 0.643 降至 0.638 ms。计时包含在线 X 转置和 BF16 写回。

BW46 定频复测中，TP2 使用 Qwen3.8-27B 第 0 层权重和随机输入。相对 ce256fe，默认路径修改的 637 个 M 的几何平均加速为 1.078×，4096 个 M 的等权平均延迟从 1062.5 降到 1051.6 µs。预分配精确输出的 198 个变更点几何平均加速为 1.201×。按 M 预分配默认工作区的调用方应查询 `workspace_elements(M)`，覆盖所有 M 的最大容量仍为 40 MiB。计时包含在线 X 处理和 BF16 写回。

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

TP2 的 main 路由在 M≥257 时以 16×256 分块打包 X；合入的调优路由使用连续遍历打包输出的一维 kernel，并保留各自实测的 X pitch。两种打包方式生成相同的 K64 片段格式。main 中沿 M 方向遍历、每块 256 列的 TP2 内核用普通 `global_store_dwordx2` 写回 BF16；调优内核保留其原有的写回指令。

`int8-policy.json` 记录 M=1–4096 的 packed 路由。TP2 按整次 GPU 调用耗时合并 main 和调优版，计时包含在线 X 打包、GEMM、缩放及 BF16 写回，不含离线 W 准备。TP1 的路由保持原样。汇编构建使用 PyYAML 并复用 `fused_gate_up_gemm_gfx936` 的公共装载、MMAC 和同步宏。

在 BW47 GPU2 上重建后，50 个 INT8 HSACO 与实验版逐字节一致。TP1/TP2 的全量路由和 X 工作区尺寸一致，168 组实跑输出逐元素一致。各 TP 的 Graph 时延中位数变化均在 0.02% 内。TP1 的 M=4095 首轮慢 4.4%，6 次交替重捕获的时延比中位数为 +0.76%。

BW46 定频复测使用 Qwen3.8-27B 第 0 层 BF16 权重，离线逐输出通道量化为 INT8。每个 M 使用随机 INT8 输入和逐 token FP32 scale。相对 ce256fe，4096 个 M 的等权平均延迟从 601.5 降到 584.0 µs，M=257–4096 的几何平均加速为 1.027×。M=4096 从 1175.5 降到 1117.7 µs。计时包含在线 X 打包、GEMM、缩放和 BF16 写回。

### packed X 调优版合并

TP2 在两次独立的 M=1–4096 扫描中选择稳定受益的整次调用：一轮至少快 2%，另一轮同样快于 main；其余位置保留 main 路由。合入的内核包括 M=65–128 的 E047、部分 M=193–224 的拆分预取、8-wave 的 144×256 / 192×256 tile，以及 256×128 的 MMAC 分组和成对 FP32 缩放。

`kernels/int8_tuned.json` 保存这些内核的生成参数、wave 数和打包方式。固定寄存器内核由 `packed_asm.py` 生成；E047 的两个入口保存在 `kernels/e047_bf16.s`，保留编译器分配的寄存器和原有执行顺序。构建过程不依赖历史实验目录或预编译二进制，公开调用方式与 W 格式不变。工作区大小仍应通过 `workspace_elements(M)` 查询。

BW47 GPU3、核心 1499 MHz / 显存 1800 MHz 定频复测，以 2bb232a 为基线。每个 M 轮换 4 份独立随机 INT8 W，使用 HIP Graph 做 5 轮交替计时。合并版逐点通过 M=1–4096 的逐元素正确性及输出边界校验，另通过 31 个 one-hot 形状。53 个重建内核的函数机器码均与对应的 main 或调优原版逐字节一致。

200 个变更点的整次调用几何平均加速为 1.074×，纯 GEMM 为 1.087×；全部 4096 个 M 的整次调用为 1.0035×。其余 3896 个点保留原路由和工作区大小，整次调用时延比的几何平均变化小于 0.01%。以下是同轮全量扫描的整次调用中位数，单位 µs：

| M | main | 合并版 |
|---:|---:|---:|
| 128 | 88.72 | 80.48 |
| 208 | 94.24 | 88.56 |
| 224 | 94.60 | 90.84 |
| 384 | 149.40 | 137.44 |
| 512 | 191.68 | 183.60 |

另在全部 200 个变更点和 22 个边界点用 3 个独立随机种子、每种子 7 轮交替复测。变更点整次调用几何平均加速为 1.073×，全部仍快于 main；合并版相对原调优版的整次调用时延比为 1.0009，重建未改变其性能。TP1 全部 4096 个 M 的路由及工作区尺寸保持一致，覆盖 19 个内核的 38 个实跑形状逐元素正确。
