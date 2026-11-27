# TP2 · gfx936 AllReduce + residual add + Gemma RMSNorm

两张 gfx936 卡各自输入 BF16 `[M, 5120]` 的局部和及同形状 residual。TP2 提供两个入口，按下游需要的激活类型选择。

| 下游需要 | 调用 | 返回值 |
|---|---|---|
| BF16 归一化激活 | `comm.fused(partial, residual, weight, eps)` | `(norm, updated_residual)`，两者都是 BF16 `[M, 5120]` |
| INT8 激活，默认逐比特模式 | `comm.fused_int8(partial, residual, weight, eps)` | `(q, scale, updated_residual)`，分别是 INT8 `[M, 5120]`、FP32 `[M, 1]` 和 BF16 `[M, 5120]` |
| INT8 激活，FP32 单轮归约 | `comm.fused_int8(partial, residual, weight, eps, fast=True)` | 与默认 INT8 模式相同的形状和类型 |

两个入口都返回新张量。调用方可以继续把 `fused()` 的 BF16 `norm` 交给原下游。使用 `fused_int8()` 时，支持预量化输入的 W8A8 下游直接消费 `q` 和 `scale`。两种入口的 `updated_residual` 都供下一层使用。

## 构建并在两张卡上运行

两张卡需要同机 HIP IPC 和 P2P，PyTorch 需要支持 HIP 和 Gloo。两端使用同一次构建的库。从仓库根目录进入算子目录，在含 gfx936 目标的 DTK 中构建。

```bash
cd fusion_gfx936/allreduce_gemma_rmsnorm/tp2
bash build.sh
```

将以下示例保存为当前目录下的 `example_tp2.py`。每个 rank 的 `partial` 代表本地计算结果。实际调用传入本层原始 Gemma 权重 `w`，kernel 内计算 `1 + w`。

```python
import os
from contextlib import ExitStack

import torch
import torch.distributed as dist

from tp2_ar import TunedOneShotAllreduce

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

with ExitStack() as cleanup:
    dist.init_process_group("gloo")
    cleanup.callback(dist.destroy_process_group)
    comm = TunedOneShotAllreduce(dist.group.WORLD, device, max_size=128 * 5120 * 2)
    cleanup.callback(comm.close)

    partial = torch.full((128, 5120), local_rank + 1, device=device, dtype=torch.bfloat16)
    residual = torch.zeros_like(partial)
    weight = torch.zeros(5120, device=device, dtype=torch.bfloat16)
    norm, updated_bf16 = comm.fused(partial, residual, weight, eps=1e-6)
    q, scale, updated_int8 = comm.fused_int8(partial, residual, weight, eps=1e-6)
    q_fast, scale_fast, updated_fast = comm.fused_int8(
        partial, residual, weight, eps=1e-6, fast=True,
    )
    if local_rank == 0:
        print(norm.dtype, q.dtype, scale.dtype, updated_int8.dtype)
```

从当前目录启动两个 rank。将示例设备号改为预留的两张卡。

```bash
HIP_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 example_tp2.py
```

`partial` 和 `residual` 必须是本卡连续且 16 字节对齐的 BF16 `[M, 5120]`，其中 `1 <= M <= 4096`。原始 `weight` 是本卡连续且 16 字节对齐的 BF16 或 FP32 `[5120]`。按批量上限设置 `max_size`，至少为 `M_max * 5120 * 2` 字节。每卡有两个通信槽和 64 KiB 信号区。M 上限为 4096 时，每槽 40 MiB，每卡共 80 MiB 数据槽。

两个 rank 共同初始化和关闭通信器。每次调用在两端使用同一入口、形状和模式，并按相同顺序提交到各自的单一有序流。Graph capture 时选定入口和 `fast`，重放用的是捕获时的模式。通信器和捕获的输入输出缓冲在最后一次 Graph 重放完成前需保持有效。

| 实现 | 调度 | 记录中的列 |
|---|---|---|
| `HipOneShotAllreduce(group, device, max_size, blocks=64)` | 固定 grid 上限 64 个线程块 | `original_us` |
| `TunedOneShotAllreduce(group, device, max_size)` | 单个通信器，M 在 65–96、129–160、193–216、257–272 四个区间分别使用上限 128、80、72、96，其余使用 64 | `tuned_us` |
| decode 双通信器实验 | M=65–96 使用独立的 96 块通信器，其他 M 使用固定 64，仅扫描 M=1–128 | `decode_us` |

`RangeSelector(fixed, tuned)` 在 M=65–512 时调用 tuned，其余调用 fixed。`RangeSelector(fixed, tuned, windows_only=True)` 只在上述四个区间调用 tuned。调用方在两端分别创建并持有两个通信器。单独的 [521 形状切换实验](results/range/summary.json) 比较了这两种路由与 fixed，并验证了多种 M 在同一张 Graph 中交替重放。主图的 tuned 曲线是单个调优通信器的全量实测。

已有服务集成对符合条件的 decode 和 target verify 使用固定 64 块的 BF16 融合，上限 M=1024，每卡两个 10 MiB 槽。服务把超出上限的整批交给框架原有路径。这里的 M=4096 是独立算子配置。INT8 由下游显式调用 `fused_int8()` 启用。

## INT8 模式和验证

固定和调优通信器都支持 INT8。默认模式沿用 BF16 `fused()` 的 Gemma RMSNorm 计算顺序，保留归一化激活的 BF16 舍入，再用舍入后的值求每行 `amax` 和 INT8 码。逐比特对照是 AITER IPC AllReduce、Triton Gemma RMSNorm、LightOp INT8。显式传入 `fast=True` 时，归一化分子和量化使用 FP32 中间值，在一轮 CTA 归约内计算平方和与带权 `amax`。这个模式的 INT8 码和 scale 可能不同。两种模式都以 FP32 累加 AllReduce 并舍入至 BF16，residual 也写成 BF16，INT8 转换均按最近偶数舍入。

通信器初始化时按模式交换两张卡的 CTA 常驻上限。若所选 grid 超过较小的上限，两端都会在提交 INT8 kernel 前报错。

在装有 AITER、SGLang HCU 插件和 LightOp 的环境中，从 `tp2` 目录运行以下命令。默认模式分别比较 INT8、FP32 scale 和 BF16 residual 的原始位，快速模式与 FP32 参考对拍。测试覆盖 eager 和更换输入后的 Graph 重放。

```bash
torchrun --standalone --nproc-per-node=2 test_fused_int8.py --variant both
```

## 结果与复现

以下延迟和正确性记录对应 `fused` 的 BF16 输出路径。

[逐 M 数据](results/dense/by-m.csv)覆盖 1–4096 的所有整数 M。[延迟图](results/dense/latency.png)与[加速比图](results/dense/speedup.png)对应同一次采样。相对于 AITER IPC AllReduce 加 SGLang Gemma residual RMSNorm，单通信器调优版本在 4096 个点均更快，几何平均 1.357×，中位数 1.349×。在 104 个调优窗口点上，调优相对原固定融合均更快，中位加速 1.088×。两种输出在 eager 和更换输入后的 Graph 重放中逐 BF16 位比较，记录的差异均为零。

复现基线另需适配 gfx936 的 AITER、SGLang、`hcu_sglang` 插件和 `safetensors`。设置 `MODEL_DIR` 指向 Qwen3.6-27B-Channel-INT8-w8a8 checkpoint。选择两张预留的空闲卡，从算子目录 `fusion_gfx936/allreduce_gemma_rmsnorm` 运行下列命令。输出目录应为空。BW47 记录使用物理 1、2 卡。

```bash
export HIP_VISIBLE_DEVICES=1,2
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu
export AITER_AR_TRANSPORT=auto AITER_AR_MAX_SIZE_MB=128 AITER_AR_SMALL_MESSAGE=off
torchrun --standalone --nproc-per-node=2 tp2/benchmark.py --model "$MODEL_DIR" --output my-tp2-dense
python tp2/summarize.py my-tp2-dense
python tp2/plot.py my-tp2-dense/by-m.csv --out my-tp2-dense
```

`--sizes 1 65 257 4096` 可只做若干形状的正确性与计时检查。全量计时每张图连续执行 8 次，两个 rank 每轮取较慢者，最终取 9 轮的中位数。前置的公共 AITER 重放放在计时区间外。两端各自的原始延迟见 [full-raw.csv.gz](results/dense/full-raw.csv.gz)，两项输出各阶段的错误计数见 [full.csv.gz](results/dense/full.csv.gz)。图表依赖 Matplotlib、NumPy 和 FontTools，可用 `--font` 指定中文字体。

```bash
torchrun --standalone --nproc-per-node=2 tp2/benchmark.py --mode range --model "$MODEL_DIR" --output my-tp2-range
python tp2/summarize_range.py my-tp2-range
```

切换实验每个形状、每种方法取 15 轮。可将保存的 `results/dense` 或 `results/range` 复制到空目录，分别运行对应的汇总脚本，从压缩的逐 rank、逐轮 CSV 重建汇总。源码与实测二进制的设备代码校验见 [provenance.json](provenance.json)。
