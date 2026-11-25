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

### Direct X（TP2，M=1–4096）

`DirectXInt8GateUpGemm` 是独立入口，支持 TP2 的 N=17408、K=5120、M=1–4096。输入是连续、16 字节对齐的 INT8 `X[M,5120]` 和 FP32 `scale_x[M,1]`，输出 BF16 `Y[M,17408]`。每次调用只提交一个 GEMM kernel，`workspace_elements(M)` 为 0，无需补齐或单独打包 X。支持区间外的 M 会报错。

构建不需要先生成 packed-X 代码对象。结果写入独立的 `build/int8_direct`，不修改 `build/int8`：

```bash
source /opt/dtk/env.sh
python -m gemm_gfx936.gate_up_gemm_gfx936.build_int8_direct
```

```python
from gemm_gfx936.gate_up_gemm_gfx936.int8 import pack_int8_weight
from gemm_gfx936.gate_up_gemm_gfx936.int8_direct import DirectXInt8GateUpGemm

packed_w = pack_int8_weight(weight_nk)
direct = DirectXInt8GateUpGemm(packed_w, scale_w)
y = direct(x, scale_x)
```

packed W 格式与 `Int8GateUpGemm` 完全相同。若同时使用两种入口，可以将同一个 `packed_w` 传给两个实例，共享权重存储。原 `Int8GateUpGemm`、`int8-policy.json` 及其 M=1–4096 分发保持原样。可以传入精确形状的 `out[M,N]`；输出存储必须与输入分离。两种入口都使用调用方当前 stream，跨 stream 交接由调用方同步。

`int8-direct-policy.json` 独立保存逐 M 路由及内核的生成参数，不读取 packed-X 路由表或历史实验二进制。每个 M 直接索引预先测定的内核，选项包括 tile 尺寸、读取方式、拆分预取和 MMAC 分组。沿 N 分工时，coalesced 方式让相邻四个 lane 读取同一行的 K64 片段，并通过 LDS 读地址的逆映射恢复 MMAC 操作数。部分内核复用地址寄存器，或在同一个 GEMM 内通过寄存器预取 X、在原有同步点写入 LDS。

这里“纯 GEMM＝整次”指预分配输出时的 GPU 执行时间，不包含输入量化、静态 W 准备或 CPU 调用开销。M 接近 128 时 direct 与 packed 整次调用接近，独立入口便于调用方按实际形状选择；默认入口仍为 packed-X 方案。

#### M=513–4096 的 Direct X 路由

新增区间覆盖全部 3,584 个整数 M。按实测结果将它们合并为 42 段，共用 15 个内核，其中 1 个与 M≤512 共用。原有 512 个路由和 73 个代码对象均与上一版一致。完整构建生成 87 个 HSACO，各自的 SHA256 与参与测试的版本一致。

M=4096 选择沿 N 分工的 BM128×BN512、8-wave 内核。一个 CTA 负责一个 128×512 输出块。每个 wave 用 4 个分片槽轮换读取 LDS 中的 8 个 X 分片，每个槽占用 4 个 VGPR。上一个分片的最后一条 MMAC 发射后，该槽即可读入新分片。固定 tile、CTA 顺序、预取点、MMAC 分组和写回方式，在 BW47 GPU7 上比较不同窗口，M=4096 的 8/4/2 分片窗口分别为 1362.56/1070.18/1095.30 µs，对应 204/188/180 个 VGPR。此项独立对照使用 8 份权重、7 轮交替测量，窗口大小同时改变 LDS 读指令的发射位置、等待值和寄存器占用。

在 BW47 GPU7 上固定核心 1499 MHz、显存 1800 MHz，以两个独立种子逐点测试全部 4096 个 M。每个种子轮换 4 份权重，四条路径随机交错回放 HIP Graph 5 轮，取每轮的单次 GPU 时延，再按轮和种子取中位数。packed X 包含每次调用时的 X 打包，各路径均从已量化的 X 和准备好的 W 开始。LightOP 使用容器内的 gfx936 W8A8 汇编 GEMM。M≤512 的旧 Direct X 是上一版公开入口，M≥513 的旧 Direct X 是此前覆盖到 M=4096 的实验路由。

| M | 旧 Direct X / 新 Direct X | packed X / 新 Direct X | LightOP / 新 Direct X | 新 Direct X 快于 packed X 的点数 |
|---|---:|---:|---:|---:|
| 1–512 | 1.003× | 1.030× | 1.351× | 350/512 |
| 513–1024 | 1.178× | 1.098× | 1.270× | 512/512 |
| 1025–2048 | 1.156× | 1.058× | 1.201× | 971/1,024 |
| 2049–3072 | 1.261× | 1.022× | 1.204× | 777/1,024 |
| 3073–4096 | 1.424× | 1.014× | 1.161× | 668/1,024 |
| 513–4096 | 1.261× | 1.040× | 1.200× | 2,928/3,584 |

表内加速比是逐 M 比值的几何平均。M=4096 时，新 Direct X / 旧 Direct X / packed X / LightOP 分别为 1069.38 / 1484.36 / 1112.82 / 1233.24 µs。另取 10 个 M，以相同的两个种子各轮换 32 份权重、交错回放 7 轮，该口径下 M=4096 分别为 1075.75 / 1488.47 / 1116.27 / 1253.91 µs。

对全部 4096 个 M，随机输入和 one-hot 输入的 BF16 结果均与独立 INT32 参考逐位相同，紧凑输入与输出边界哨兵检查通过。另在 22 个边界及代表点验证了独立 stream 上的 HIP Graph 回放。

#### 此前 M=1–512 版本的定频验证

BW47 GPU7（gfx936）在核心 1499 MHz、显存 1800 MHz 下逐点比较三个入口。M≤128 的旧 Direct X 取自 `90ab436`，M=129–256 使用此前独立调优的实验路由，M=257–512 使用此前 M=1–4096 实验的对应路由。packed X 使用 `90ab436` 的 `Int8GateUpGemm`，计时包含在线打包。三条路径共用 X、权重及 scale，预分配输出，用 HIP Graph 测 GPU 时间。每个 M 使用两个独立种子，每种子轮换 32 份随机 W，交替计时 7 轮。下表用每个种子的中位数汇总各 M，再对逐点加速比取几何平均。

| M | 旧 Direct X / 新 Direct X | packed 整次 / 新 Direct X | 新 Direct X 快于 packed 的点数 |
|---|---:|---:|---:|
| 1–128 | 1.0014× | 1.0585× | 124/128 |
| 129–256 | 1.0012× | 0.9389× | 0/128 |
| 257–384 | 1.0535× | 1.0320× | 96/128 |
| 385–512 | 1.2229× | 1.1098× | 128/128 |
| 1–512 | 1.0661× | 1.0329× | 348/512 |

M=512 使用 8-wave、BM128×BN512 的 N-split 内核。每个 K 阶段保留 4 个共享 X 分片的寄存器槽，最后一条消费旧分片的 MMAC 发射后，再把下一个 LDS 分片读入该槽，并按已发射读请求数更新 `lgkmcnt` 等待值。相同 tile、权重和 CTA 顺序下，8 分片与 4 分片的独立对照分别为 188.76 和 162.80 µs，声明的 VGPR 从 204 减到 188。最终公开入口的两个种子汇总值为 162.72 µs，旧 Direct X 为 237.81 µs，packed X 整次为 184.58 µs。

选定的 M=1、2、3、6、289、305 使用 K 循环入口填充。首次进入循环执行填充指令，循环回跳直接跳到其后的标签。M=361–380 在公共入口复测后采用旧实验路由。M=129–256 的 128 个形状均由 packed X 整次调用取得更低延迟。

从公开构建入口生成的 73 个 HSACO 的 SHA256 均与实测候选一致。M=1–512 全部通过紧凑随机 X 和逐行 one-hot X 的 BF16 位级对比、输出前后哨兵检查及零工作区检查。15 个代表形状通过非默认 stream 上的 HIP Graph 回放验证。

#### M=1–128 首版（90ab436）

在 BW47 GPU7 独占、核心1499 MHz / 显存1800 MHz 下验证：33 个 direct 代码对象与保存的实验方案逐字节一致，M=1–128 的随机输入、紧凑 X one-hot、输出边界、Graph 及非默认 stream 校验通过。重新构建 direct 前后，packed 构建目录内 157 个文件的哈希和修改时间均不变，其 53 个内核与 12bf4c2 一致。

逐点使用 3 个独立种子、每种子 7 轮交替计时，每轮轮换 32 份独立 W。仓库入口相对实验入口的时延比几何平均为 0.99998，最慢点为 +0.31%；相对当前 packed 整次调用，M=1–128 的几何平均加速为 1.057×。同轮结果如下，单位 µs：

| M | Direct X（纯＝整次） | packed X 整次 |
|---:|---:|---:|
| 1 | 66.08 | 68.70 |
| 32 | 65.07 | 69.41 |
| 64 | 66.71 | 72.83 |
| 65 | 67.63 | 75.70 |
| 96 | 74.79 | 77.24 |
| 112 | 77.01 | 78.61 |
| 128 | 79.27 | 79.85 |

### Direct X、packed X 与 LightOP 性能对比（M=1–4096）

图中 Direct X 使用覆盖 M=1–4096 的扩展版。在 BW47 GPU7 上逐点测试 TP2 的每个整数 M，每个 rank 的 N=17408、K=5120。上图为 GPU 延迟，下图为对照延迟除以 Direct X 延迟，高于 1 表示 Direct X 更快。packed X 的计时包含在线打包 X。

![gfx936 TP2 上 Direct X、packed X 与 LightOP 在 M=1–4096 的 GPU 延迟和加速比](direct-x-packed-x-lightop-M1-4096.png)
