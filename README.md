# hcu_kernel

海光 HCU/DTK 算子源码归档。每个算子放在独立目录；LightOp 融合算子来自已验证实现，GEMM 包含独立 HIP 与模块化汇编实现。

## 算子

- [gemm_gfx936/](gemm_gfx936/) 收录 gfx936 GEMM。
  - [lm_head_bf16_gfx936/](gemm_gfx936/lm_head_bf16_gfx936/) 提供 Qwen3.8-27B TP2 的单 rank BF16 LM head，所有路径共用连续 W[K,N]。支持 M=1–20480，附独立构建、正确性、基准和 ISA 审计入口。 提供可选的滚动 MMAC 与中段 CTA 分组调优。
  - [gate_up_gemm_gfx936/](gemm_gfx936/gate_up_gemm_gfx936/) 提供 gfx936 非融合 GateUp GEMM。BF16 使用连续 W[K,N]，INT8 使用 packed W，各自用一份权重覆盖 TP1/TP2 的 M=1–4096。TP2 的 INT8 Direct X 入口支持 M=1–40960，直接读取连续 X 并输出 BF16 [M,N]。

- [gemm_gfx938/](gemm_gfx938/) 收录 gfx938 GEMM。
  - [fp8_pi32_gemm_gfx938/](gemm_gfx938/fp8_pi32_gemm_gfx938/) 提供 Qwen3.8-27B Channel FP8 TP1 六类投影，复用一份 PI32 W 并直接读取行连续 X，支持 M=1–128，含独立构建与 Graph 验证。

- [fused_gate_up_gemm_gfx936/](fused_gate_up_gemm_gfx936/)：gfx936 INT8/BF16 特殊布局 gate/up GEMM 融合 SiLU×up，直接输出 BF16 [M,8704]；含源码快照、完整可校验复现包和正确性/性能记录

- `decode_gemm_int8_gfx936/`：gfx936 INT8 decode GEMM，含14种HIP配置、独立编译/Python入口、正确性与冷权重池基准，以及TP1/2/4/8、bs1–1024结果摘要

- [fusion_gfx936/](fusion_gfx936/) 收录四个逐 token INT8 融合算子，包括 Attention sigmoid × mul、GDN gated RMSNorm、Gemma add + RMSNorm 和 SiLU × up。各子目录提供调用说明与性能对比图。
- [gated_rms_norm_per_token_fp8_quant_gfx938/](gated_rms_norm_per_token_fp8_quant_gfx938/) 提供 gfx938 的 GDN gated RMSNorm、SiLU gate 和 per-token FP8 E4M3FN 融合，含正确性测试和两段式对照基准。
- [hbm_read_gfx938/](hbm_read_gfx938/) 提供 gfx938 的 BF16 LM head 权重纯读基准，含 CU mask、CTA 时间戳和 XProf 复测。

LightOp 目录保留 kernel、Python 接口、必要的 `reduce.h` 和 focused tests/benchmark helper；完整注册、编译系统和运行时依赖仍由上游 LightOp 工程提供。GEMM 目录可用 DTK hipcc 独立编译，调用依赖 HCU 版 PyTorch。

本仓库不包含 LightOp PR 的原始 profiling JSON、trace 或长篇证据文档。

## gfx936 HBM 带宽经验

**先用纯读确认硬件和读取调度，再用同读写量的控制版评估融合算子。** 在 BW1000（80 CU、wave64）上，两块连续的 `[4096, 5120]` BF16 输入共 80 MiB。纯读基准用 80 CTA × 256 线程，每条 `global_load_dwordx4` 读 16 B/lane，一个 wave 覆盖连续 1 KiB。lane 不需要各读 64 B 才能合并访存。

| 同一输入、GPU dispatch p50 | 有效纯读带宽 |
| --- | ---: |
| 默认 load 标志，8 组 VGPR 滑动窗口 | 1.31 TB/s |
| `glc slc`，每 8 条 load 用 `vmcnt(0)` 等清 | 1.34 TB/s |
| `glc slc`，滑动窗口，每次复用最老一组 VGPR 前用 `vmcnt(7)` | 1.47–1.49 TB/s |
| 同一滑动窗口，每个 wave 先发完 X 的读取再发残差的读取 | 1.50 TB/s |

这里的带宽是有效字节数除以无插桩的 dispatch p50。另用 XProf 核对最快路径的 `FETCH_SIZE ≈ 81920 KiB`、`WRITE_SIZE = 0`。用插桩 dispatch 自己的计数和时间折算约 1.37 TB/s。缓存标志在这里作为一组做 A/B，单独移植到旧读写控制时，4K 耗时从 116.32 增至 118.08 µs。大流量基准中，只平移热循环代码的地址相位，3150 MiB 纯读主体带宽从 1.393 变到 1.505 TB/s。核对的是指令地址相位，具体前端机制尚未隔离。

**迁移到融合算子时要重新测。** 5120 维 Add + RMSNorm + INT8 在 4K 时读 80 MiB、写约 60 MiB。旧版无归约读写控制约 116–119 µs，适合当这套 I/O 布局的实测参照。历史对照中，同为 256 线程、2 wave/token，普通网格 148.50 µs，固定 80 CTA 的持久化网格 263.14 µs，HBM 字节数接近。持久化流水还要承担每对 token 的规约、同步和写回。先保留普通网格，再用相同读写语义验证预取收益。[当前融合实现和复测入口](fusion_gfx936/gemma_add_rmsnorm_per_token_int8_quant/README.md)

本仓库 80 CU 复跑中，默认标志版为 1.42 TB/s，最快版为 1.51 TB/s。绝对值随设备和运行条件波动，比较时以同轮 A/B 为准。

### 复现纯读 A/B

需要 gfx936、DTK 和 HCU 版 PyTorch。从仓库根目录运行，将 `6` 换成空闲设备编号。脚本对四版用同一组缓冲区，预热后交替测 61 次并取 p50。原实验记录见 [第 1 轮](hbm_read_gfx936/wave-read-1.log) 和 [第 2 轮](hbm_read_gfx936/wave-read-2.log)。

```bash
source /opt/dtk/env.sh
bash hbm_read_gfx936/build.sh
HIP_VISIBLE_DEVICES=6 python3 hbm_read_gfx936/benchmark.py

HIP_VISIBLE_DEVICES=6 xprof --replay-mode off --metrics FETCH_SIZE,WRITE_SIZE \
  --kernels read_xr --output-dir hbm_read_gfx936/build/pmc \
  python3 hbm_read_gfx936/benchmark.py --case xr_c3_serial --samples 3 --warmup 0
```
