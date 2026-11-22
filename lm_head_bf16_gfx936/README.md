# gfx936 BF16 LM head

Qwen3.8-27B TP2 的单 rank 前向算子。计算 `Y = X @ W`，覆盖 `1 <= M <= 20480`。X、W 和 Y 均为 BF16，乘加或规约使用 FP32。列数固定为每 rank 124160，输入维度固定为 5120，输出可交给上层 TP 通讯和 logits 处理。

## 构建与调用

在 gfx936 HCU 上使用含 HIP、rocBLAS 和 LLVM 工具的 DTK 环境，以及配套的 HCU 版 PyTorch 和 Triton。实测使用 DTK 26.10 nightly-1112、PyTorch 2.11.0 和 Triton 3.6.0。显式 rocBLAS 方案编号 2534、2218 依赖该版本的方案表。在更换 DTK、编译器或框架后，用审计入口核对新生成的 ISA 和实际派发，再比较性能。

从仓库根目录运行。

```bash
python3 -m lm_head_bf16_gfx936.build
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.validate
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.validate --preset 20k
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.audit
```

```python
from lm_head_bf16_gfx936 import K, MAX_M, N, LMHead, route

# 在加载 rank 的 [N,K] checkpoint 分片时生成一份连续 [K,N] GPU 权重。
weight_kn = weight_nk_cpu.T.contiguous().to("cuda")
head = LMHead(weight_kn)
logits = head(hidden_states)  # BF16 [M,N]
```

W 必须是行连续 `[K,N]`，stride 为 `(N,1)`。X 必须是连续 `[M,K]`。张量同设备、16 字节对齐。算子持有传入的 W，构造时新增权重显存为 0 B。每 rank 的 W 为 1,271,398,400 B，所有派发路径共用其存储。

调用方可预分配连续 BF16 `out[P,N]`。每点的 `P` 由 `route(M)` 的第 4 项给出，返回 `out[:M]`。转置 X 路径可另外传入连续 BF16 `workspace[K,P]`。这两个缓冲区与 X、W 以及彼此的存储必须分离。算子在输入设备的当前流派发。跨流共享权重、输入或缓冲区时，由调用方建立生产、调用、消费之间的流依赖，保证各张量在 GPU 工作完成前有效。并发调用各自持有输出和工作区。

## 派发

| M | 实现 | 输出 tile `T_M × T_N` | K 步长 |
|---:|---|---:|---:|
| 1 | HIP GEMV | 1 × 512 | 全 K |
| 2 | HIP GEMV | 2 × 512 | 全 K |
| 3 | Triton GEMM | 16 × 128 | 128 |
| 4–14 | rocBLAS 2534 | 64 × 448 | 16 |
| 15–64 | rocBLAS 2218 | 64 × 256 | 32 |
| 65–128 | Triton GEMM | 128 × 256 | 32 |
| 129–20480 | 转置 X + PyTorch GEMM | 256 × 256 | 16 |

前两条 GEMV 路径在一个 workgroup 内沿 K 规约，N 向每组覆盖 512 列。M≥129 时，`P=16*ceil(M/16)`，X 在线转置到 `workspace[K,P]`，尾部补零，再调用 `torch.mm(workspace.T, W, out=out)`。其余路径直接使用 X，`P=M`。大 M 实测目标为 `256×256×16`，原服务目标为 `256×256×32`。在 M=20480，两者各派发 38800 个 workgroup。

## 性能复现

```bash
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.benchmark --m 1,2,3,4,15,65,129,4096,20479,20480
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.benchmark --preset dense-4k --csv build/lm-head-4k.csv
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.benchmark --preset long-sample --csv build/lm-head-20k.csv
```

每个测量使用等值的合成输入和权重。原服务基线为 `torch.mm(X, W_NK.T)`，物理 W 为连续 `[N,K]`。优化版与基线在同一卡上交替测试，重用同一块 GPU 权重分配。每个 M 预热一次，取 5 个 HIP event 区间的中位数。在打乱 M 后按批次交替基线与优化的执行顺序。计时覆盖 X 在线转置、尾部补零和输出写回。权重布局转换、上传和分配在计时区间外。

以字节计算 I/O 地板。令 `Xb=2PK`、`Wb=2KN`、`Yb=2PN`、`gM=ceil(P/T_M)`、`gN=ceil(N/T_N)`。原服务基线取 `P=M`，输出 tile 为 `256×256`。

```text
T_floor = (Xb + Wb + Yb + E) / 1.5e12 + ((gM - 1)*Wb + (gN - 1)*Xb) / 6e12
```

E 为在线 X 变换的 HBM 字节数。转置路径读原 X 并写入补齐后的工作区，`E=2K(M+P)`，其余路径 E 为 0。比值 `T_floor / T_measured` 由模型计算。

### BW1000 实测

| M | 原服务 | 统一 W | I/O 地板 | 地板 / 实测 |
|---:|---:|---:|---:|---:|
| 1 | 1.522 ms | 0.994 ms | 0.848 ms | 85.3% |
| 64 | 1.563 ms | 1.065 ms | 0.911 ms | 85.6% |
| 4096 | 15.057 ms | 13.445 ms | 8.171 ms | 60.8% |
| 8192 | 30.596 ms | 27.263 ms | 15.707 ms | 57.6% |
| 16384 | 60.966 ms | 54.419 ms | 30.779 ms | 56.6% |
| 20000 | 75.621 ms | 67.304 ms | 37.617 ms | 55.9% |
| 20480 | 76.170 ms | 68.035 ms | 38.315 ms | 56.3% |

M=1–4096 全整数测量，首轮 4053 点更快，中位加速 1.104×。对首轮 43 个较慢点另做三轮复测，129 次配对均更快。长段在同型号的另一张卡上采样 257 个 M，包含 128 行间隔及 256 行边界两侧，全部更快，中位加速 1.114×。4K 交接点分别在两张卡上实测。

M=20480 的 X 转置独立计时为 0.567 ms。M=20479 与它使用相同的 P 和 GEMM 目标，但 X 转置走通用跨步拷贝，转置加补零独立计时为 2.827 ms。M=20480 的 BF16 输出为 5,085,593,600 B，PyTorch 峰值分配为 6,809,976,832 B。

独立 FP32 参考对比覆盖了 M=1–256 和 4 个较长 M 的全部约 48.16 亿个输出。20K 校验覆盖 `2^31` 个输出元素边界两侧、补零尾行和隐式分配路径，抽查 47,331,712 个输出。两组 FP32 对比均零超阈，容限为 `abs(got-ref) <= 0.04 + 0.0044*abs(ref)`。实测摘要、依赖版本和 ISA 资源见 [results.json](results.json)。

## 后续调优

在已验证的 DTK 26.10 nightly-1112 环境下构建滚动 MMAC 扩展。每次构建都会反汇编 7 个 code object，检查固定寄存器、LDS、scratch 和稳态循环中 W 的发射顺序。审计记录位于 `build/rolling/audit.json`。

```bash
python3 -m lm_head_bf16_gfx936.build --tuned
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.validate --preset tuned --graph
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.audit --tuned --m 1,4,33,65,129,192,256,257,512,768,1024
HIP_VISIBLE_DEVICES=0 python3 -m lm_head_bf16_gfx936.benchmark --tuned --preset medium --csv build/lm-head-medium-ab.csv
```

```python
head = LMHead(weight_kn, tuned=True)
logits = head(hidden_states)
```

开关启用后，`route(M, tuned=True)` 给出当前派发和输出行数 P。滚动路径的 P=M。若预分配其 X 转置工作区，行高采用 `route` 中的 BM。分段 rocBLAS 路径的 P 仍向上对齐到 16，X 在 `workspace[K,P]` 转置、补零一次，再由各子 grid 写入各自的 Y 列。每个路径直接读取同一份连续 `W[K,N]`。

- M=2–192 使用固定寄存器的双槽汇编，W 经向量 load 直达 LDS。下一段 W 的读取与当前 K 步的 BF16 MMAC 重叠。每个 M 按选定的 tile 沿 N 发射 3 或 4 段。
- M=256 时，原 `256×256×16` GEMM 的 485 个 CTA 沿 N 分 7 段，每段 69–70 个。
- M=257–512 采用分段分发表。M=512 将列切成 160/160/160/5 个 256 列单元。前三段使用 `BM256×BN256×BK16`，末段 1,280 列由库选择 `BM64×BN128×BK32`。
- M=513–1024 中的 96 个形状采用已复测的切分或显式方案编号 25440、25442。更换 DTK 时，按上述入口复核显式方案及实际派发。

BW1000 gfx936 双卡逐点对照使用等值 X 和 W。每卡采集九次同路径预热后的 HIP Graph 区间，取两卡 p50 的均值。时间计入 X 转置、补零和 Y 写回。原始布置为 `torch.mm(X, W_NK.T)`，关闭本轮调优时用此前的 packed-X GEMM。

| M | 原始 `torch.mm` | 调优关 | 调优开 |
|---:|---:|---:|---:|
| 256 | 1.640 ms | 1.255 ms | 1.084 ms |
| 257 | 2.639 ms | 1.695 ms | 1.630 ms |
| 512 | 2.112 ms | 1.720 ms | 1.656 ms |
| 768 | 2.926 ms | 2.633 ms | 2.550 ms |
| 1024 | 3.941 ms | 3.456 ms | 3.402 ms |

M=257–512 的 256 个形状全部提速，中位省 62 µs。M=256–1024 的 769 个形状等权汇总，开启本轮调优额外减少 0.94% 延迟。派发表命中的 353 个形状中，349 个提速，4 个形状的回退均小于 0.1%。完整摘要见 [tuned-results.json](tuned-results.json)。
