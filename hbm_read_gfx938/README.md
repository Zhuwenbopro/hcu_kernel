# gfx938 LM head 权重读取带宽

在 `nmz26` 的 BW1102（gfx938、72 CU、wave64）HCU 7 上，**64 个可用 CU、64 CTA × 256 线程、12 组滚动加载**达到 **2.066 TB/s** 的有效读带宽。XProf 把同次 dispatch 的显存读计数和执行时间配对，测得中位数 **2.087 TB/s**。另一组 72 CTA 在开放 72 CU 时也达到约 2.03 TB/s。

## 实现与测量口径

读取指令和 VGPR 轮转沿用 [gfx936 纯读实现](../hbm_read_gfx936/read_xr.s)。本目录以 gfx938 为汇编目标。

- Qwen3.8-27B-Channel-FP8 的 `lm_head.weight` 在 checkpoint 中是连续 BF16 `[248320, 5120]`，共 **2,542,796,800 B**。按行分两半分别装入显存，每次 dispatch 完整读取 **1,271,398,400 B**。
- 每 lane 用 `global_load_dwordx4 ... glc slc` 读取 16 B，每个 256 线程 CTA 在一个阶段覆盖 4096 B。两个连续地址范围交替推进。Q8 和 Q12 分别有 8 和 12 组轮转 VGPR，复用前执行 `vmcnt(Q-1)`。末尾的不足整阶段部分按实际切片数读取。
- 每种配置、每半权重预热 10 次，测量 61 次。表中的有效读带宽是两半权重总字节数除以两次无 profiler 的 dispatch p50 之和，采用十进制 TB/s。64 CTA/Q12 在 64 CU 上两半合计 **1.231 ms**，400 CTA/Q8 在 72 CU 上合计 **1.290 ms**。基准只执行权重读取，实际 LM head GEMM 还包含矩阵乘法和输出写回。

## CTA 数与可用 CU 数

统一用 256 线程和 Q8 读取同一组权重。CU mask 通过 HIP stream 设置并读回复核。

| CTA 数 | 开放 72 CU | 限制 64 CU |
| ---: | ---: | ---: |
| 64 | 2.055 TB/s | **2.053 TB/s** |
| 65 | 1.966 TB/s | 1.370 TB/s |
| 72 | **2.027 TB/s** | 1.521 TB/s |
| 73 | 1.410 TB/s | 1.523 TB/s |
| 80 | 1.554 TB/s | 1.680 TB/s |
| 128 | 1.980 TB/s | **2.014 TB/s** |
| 144 | **1.987 TB/s** | 1.805 TB/s |
| 400，原配置 | 1.971 TB/s | 1.919 TB/s |

**80 CTA 在同一张 72-CU 卡上形成了长尾。** 独立的硬件 `HW_ID` 和起止时间戳采样显示，64 个 CU 各跑 1 个 CTA，8 个 CU 各跑 2 个。CTA 几乎同时开始。单 CTA 的 CU 全部完成后，双 CTA 的 CU 继续执行的尾段约占总时间的 31%。9 次采样中，最后完成的 CTA 均位于双 CTA 的 CU。相同的 72-CTA kernel 在限制到 64 CU 时降到 1.521 TB/s，64-CTA kernel 在两种 CU 数下均保持约 2.05 TB/s。64 CTA 的 Q12 配置比 Q8 快约 0.4%。两种配置的循环展开长度分别是 48 和 64 条读取。

XProf 单独复测了第一半权重，每种配置 3 次，显存读计数约 1.2714 GB，`WriteSize` 为 0。

| 配置 | 有效读带宽 | XProf 显存读带宽中位数 |
| --- | ---: | ---: |
| 64 CTA/Q12，64 CU | 2.066 TB/s | 2.087 TB/s |
| 72 CTA/Q12，72 CU | 2.028 TB/s | 2.005 TB/s |

实测时原 SGLang 服务常驻，测试窗口内无请求，测后 `/health` 返回 200。

## 复现

需要 gfx938、DTK、Python 3.10 或以上和 HCU 版 PyTorch。读取真实 checkpoint 时还需要 `safetensors`。以下命令从仓库根目录运行，将 `7` 换成目标卡编号。复用带 SGLang adapter 的环境时，测量进程使用 `HCU_ADAPTER_HOST=none SGLANG_PLUGINS=none PYTHONPATH=`。

```bash
python3 hbm_read_gfx938/build.py

for cu in 72 64; do
  HIP_VISIBLE_DEVICES=7 python3 hbm_read_gfx938/benchmark.py \
    --model-dir "$MODEL_DIR" --blocks 64 65 72 73 80 128 144 400 \
    --depth 8 --cu-count "$cu" --output "/tmp/hbm${cu}.json"
done

HIP_VISIBLE_DEVICES=7 python3 hbm_read_gfx938/benchmark.py \
  --model-dir "$MODEL_DIR" --blocks 64 --depth 12 --cu-count 64 \
  --output /tmp/hbm64-q12.json
```

`MODEL_DIR` 指向包含 `model.safetensors.index.json` 的模型目录。省略 `--model-dir` 时使用等字节数的合成数据。默认依次测两半，可用 `--half 0` 或 `--half 1` 只测一半。追加 `--timestamps` 可单独采集 instrumented kernel 的 CU 分配与时间戳。

```bash
HIP_VISIBLE_DEVICES=7 xprof --replay-mode off --kernels read_xr \
  --metrics FetchSize,WriteSize,TCC_EA_RDREQ_sum,TCC_EA_RDREQ_32B_sum \
  --output-dir /tmp/hbm64-xprof \
  python3 hbm_read_gfx938/benchmark.py --model-dir "$MODEL_DIR" \
    --half 0 --blocks 64 --depth 12 --cu-count 64 --warmup 0 --samples 3 \
    --output /tmp/hbm64-xprof.json
```

按 DTK 的 gfx938 指标定义，显存读字节数为 `64 × TCC_EA_RDREQ_sum − 32 × TCC_EA_RDREQ_32B_sum`。`FetchSize` 按 KiB 输出，换算为字节时可能有不足 1 KiB 的舍入差。带宽使用 XProf 同次 dispatch 的起止时间计算，独立于无 profiler 的事件计时。
