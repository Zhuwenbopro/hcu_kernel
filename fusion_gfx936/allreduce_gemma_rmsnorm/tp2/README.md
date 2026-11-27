# TP2 · gfx936 AllReduce + residual add + Gemma RMSNorm

每张卡输入连续 BF16 `[M, 5120]` 的局部和、同形状 residual，以及宽度 5120 的 Gemma 权重。返回两个新 BF16 张量，分别为归一化结果和更新后的 residual。两张 gfx936 卡需支持同机 HIP IPC，并在各自的单一流上按相同顺序进入 collective。两个 rank 共同构造通信器并在退出前共同调用 `close()`。

## 构建和调用

在含 gfx936 目标的 DTK 中，从 `fusion_gfx936/allreduce_gemma_rmsnorm` 运行 `bash tp2/build.sh`。独立算子只需要 HIP 运行时和支持 HIP、Gloo 的 PyTorch。以下示例从 `tp2` 目录导入，在两个已经初始化的 Gloo rank 上运行。

```python
from tp2_ar import TunedOneShotAllreduce

comm = TunedOneShotAllreduce(group, device, max_size=4096 * 5120 * 2)
try:
    norm_output, updated_residual = comm.fused(partial, residual, weight, eps=1e-6)
finally:
    comm.close()
```

`max_size` 是每个通信槽的最大字节数。每卡使用两个 uncached 槽和 64 KiB 信号。M 上限为 4096 时，每槽 40 MiB，每卡共 80 MiB 数据槽。各线程块从对端槽拉取数据，按 AITER 的 FP32 加法和 BF16 舍入得到 AR 结果，再按 Gemma 的计算顺序处理 residual 和归一化。测试使用 BF16 权重。

| 实现 | 调度 | 记录中的列 |
|---|---|---|
| `HipOneShotAllreduce(group, device, max_size, blocks=64)` | 固定 grid 上限 64 个线程块 | `original_us` |
| `TunedOneShotAllreduce(group, device, max_size)` | 单个通信器，M 在 65–96、129–160、193–216、257–272 四个区间分别使用上限 128、80、72、96，其余使用 64 | `tuned_us` |
| decode 双通信器实验 | M=65–96 使用独立的 96 块通信器，其他 M 使用固定 64，仅扫描 M=1–128 | `decode_us` |

`RangeSelector(fixed, tuned)` 在 M=65–512 时调用 tuned，其余调用 fixed。`RangeSelector(fixed, tuned, windows_only=True)` 只在上述四个区间调用 tuned。调用方在两端分别创建并持有两个通信器。单独的 [521 形状切换实验](results/range/summary.json) 比较了这两种路由与 fixed，并验证了多种 M 在同一张 Graph 中交替重放。主图的 tuned 曲线是单个调优通信器的全量实测。

已有服务集成对符合条件的 decode 和 target verify 使用固定 64 块的融合，上限 M=1024，每卡两个 10 MiB 槽。服务把超出上限的整批交给框架原有路径。这里的 M=4096 是独立实验配置。

## 结果与复现

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
