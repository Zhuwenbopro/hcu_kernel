# gfx938 PI32 Channel FP8 GEMM

面向 Qwen3.8-27B Channel FP8 TP1 的小 M 投影，支持每个整数 `M=1–128`。六类投影使用同一套 PI32 权重布局，每层只常驻一份打包权重。

| 投影 | K | N |
|---|---:|---:|
| GDN QKVZ | 5120 | 16384 |
| GDN Out | 6144 | 5120 |
| Full Attention QKV | 5120 | 14336 |
| Full Attention Out | 6144 | 5120 |
| MLP GateUp | 5120 | 34816 |
| MLP Down | 17408 | 5120 |

## 数据布局与计算

- X 是连续行主序 E4M3FN `[M,K]`，激活 scale 是每行一个 FP32 值。内核直接读取调用方的 X。
- `pack_weight` 将连续 checkpoint E4M3FN `[N,K]` 逐字节重排为 `uint8 [K/64,N/16,4,16,16]`。每 32 个 N 通道内，存储位置 `i` 对应原通道 `8*(i%4) + 4*(i//16) + (i%16)//4`。N 方向的 FP32 scale 维持原顺序。
- CTA 协作把 X 的 K64 片段放进 LDS，由四个 wave 复用。W 以 16 字节向量预取到寄存器，FP8 MMAC 用 FP32 累加，随后施加行、列 scale，写出连续 BF16 `[M,N]`。
- `dispatch.json` 保存各角色逐 M 的实测配置。配置字段为 `provider:splits:depth[:tile_n]`，省略 tile 时为 N256。分派在 N128/N256、预取深度和 split-K 间选择。向量归约每个 lane 处理 8 个相邻输出。
- 默认用 `(K,N)` 分派，两类 Out 沿用服务的 GDN Out 表。可传入 `role="attention_out"` 使用它独立测量的表。

## 构建与验证

需要 gfx938、包含 gfx938 FP8 MMAC builtin 的 DTK hipcc，以及对应的 HCU 版 PyTorch。从仓库根目录执行：

```bash
source /opt/dtk/env.sh
python3 -m gemm_gfx938.fp8_pi32_gemm_gfx938.build --hipcc /opt/dtk/bin/hipcc
HIP_VISIBLE_DEVICES=6 python3 -m gemm_gfx938.fp8_pi32_gemm_gfx938.validate
```

验证使用合成 E4M3FN 输入和权重，检查六类投影、PI32 字节往返、FP32 抽样参考和 CUDA Graph 动态输入重放。通过 `--m` 和 `--roles` 可选择用例。 `--no-graph` 只跑 eager。

```python
from gemm_gfx938.fp8_pi32_gemm_gfx938 import gemm, pack_weight

packed_w = pack_weight(w_nk)  # 权重加载时执行一次
y = gemm(x, packed_w, x_scale, weight_scale)
```

调用方可传入 `out=` 和 `workspace=` 复用缓冲区。用 `get_config(M, K, N)` 查询分片数，split-K workspace 需容纳 `config.splits * M * N` 个 FP32 值。所有设备缓冲区需在同一张 gfx938 卡上、连续且 16 字节对齐。输出和 workspace 需要独立的存储区域，计算进入 PyTorch 当前 HIP stream。SGLang 的长 prefill 在 M>128 时用 `unpack_weight(packed_w)` 生成临时 NK，再交给原有 DeepGEMM。

## 实测记录

BW1102 gfx938，72 CU，核心 1300 MHz、HBM 875 MHz。使用真实模型权重，计时轮换至少 384 MiB 权重副本。延迟为 Graph 事件中位数，包含 split-K 归约和 BF16 写回。地板为一次 FP8 权重字节数除以 1.9 TB/s。

| 投影 | M96 原 NK (µs) | M96 PI32 (µs) | 权重读取地板 (µs) |
|---|---:|---:|---:|
| GDN QKVZ | 75.94 | 61.69 | 44.15 |
| GDN Out | 46.54 | 36.82 | 16.56 |
| Full Attention QKV | 74.61 | 57.44 | 38.63 |
| Full Attention Out | 46.66 | 36.74 | 16.56 |
| MLP GateUp | 159.11 | 128.08 | 93.82 |
| MLP Down | 123.04 | 73.70 | 46.91 |

六类投影的 768 个 M 点全部通过校验，1,536 次动态 Graph 重放的最大相对 L2 为 1.48×10⁻⁴。按这 768 个点等权累计，延迟减少 23.2%。

同卡 SGLang A/B 使用 4096 输入、1024 输出、24 并发与 MTP NEXTN 3/1/4。平均 TPOT 从 29.52 ms 降到 27.48 ms，平均 TTFT 从 8.14 s 增到 8.72 s。HumanEval pass@1 从 162/164 降到 160/164。
