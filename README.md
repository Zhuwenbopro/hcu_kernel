# hcu_kernel

海光 HCU/DTK 算子源码归档。每个算子放在独立目录；LightOp 融合算子来自已验证实现，GEMM 包含独立 HIP 与模块化汇编实现。

## 算子

- [fused_gate_up_gemm_gfx936/](fused_gate_up_gemm_gfx936/)：gfx936 INT8/BF16 特殊布局 gate/up GEMM 融合 SiLU×up，直接输出 BF16 [M,8704]；含源码快照、完整可校验复现包和正确性/性能记录

- `decode_gemm_int8_gfx936/`：gfx936 INT8 decode GEMM，含14种HIP配置、独立编译/Python入口、正确性与冷权重池基准，以及TP1/2/4/8、bs1–1024结果摘要

- [gemma_fused_add_rmsnorm_int8_fp32_gfx936/](gemma_fused_add_rmsnorm_int8_fp32_gfx936/) 支持 5120 维的 Gemma 残差加法、RMSNorm 和 per-token INT8 量化，在 FP32 中合并平方和与带权最大值规约，支持独立 HIP 编译
- `fuse_rms_norm_gated_per_token_quant_int8/`：GDN RMSNorm + SiLU gate + per-token INT8 quant
- `fused_sigmoid_mul_per_token_quant/`：Attention sigmoid-mul + per-token INT8 quant

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

**迁移到融合算子时要重新测。** 5120 维 Add + RMSNorm + INT8 在 4K 时读 80 MiB、写约 60 MiB。旧版无归约读写控制约 116–119 µs，适合当这套 I/O 布局的实测参照。历史对照中，同为 256 线程、2 wave/token，普通网格 148.50 µs，固定 80 CTA 的持久化网格 263.14 µs，HBM 字节数接近。持久化流水还要承担每对 token 的规约、同步和写回。先保留普通网格，再用相同读写语义验证预取收益。[当前融合实现和复测入口](gemma_fused_add_rmsnorm_int8_fp32_gfx936/README.md)

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
