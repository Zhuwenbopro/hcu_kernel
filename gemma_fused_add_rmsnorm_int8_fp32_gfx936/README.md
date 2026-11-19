# 5120 维 Gemma Add + RMSNorm + Per-token INT8

适用于 gfx936。每个 CTA 处理一个 token。输入、残差和 Gemma 原始权重为 BF16，残差原地写回，返回 INT8 和每行 FP32 scale。

对每一列先计算 $s_i=x_i+r_i$ 和 $p_i=s_i(1+w_i)$。同一轮 CTA 规约同时求出 $\sum_i s_i^2$ 和 $\max_i |p_i|$。再计算

$$
rstd=\operatorname{rsqrt}\left(\frac{\sum_i s_i^2}{5120}+\epsilon\right),\qquad
scale=\frac{\max\left(rstd\cdot\max_i |p_i|,10^{-10}\right)}{127}.
$$

将 $p_i\cdot rstd$ 按 scale 做对称 INT8 量化。归一化与量化之间保留 FP32 中间值，Qwen3.8-27B-int8 的双核路径在这里使用 BF16。在合成 BF16 输入和四层真实权重的对比中，两版的 INT8 码值有 7.24% 不同，反量化结果的相对 L2 差异为 0.78%–0.86%。

## 线程与读取调度

| Token 数 | 线程 / CTA | 每线程读取 | 调度 |
| --- | ---: | --- | --- |
| 1–63 | 640 | 8 列 | 编译器调度 |
| 64–159 | 512 | 8 列，前 128 线程再读 8 列 | 编译器调度 |
| 160–351 | 320 | 8+8 列 | 编译器调度 |
| 352–703 | 320 | 8+8 列 | 两组提前发射，分批等待，`glc slc` |
| 704–1407 | 512 | 8 列，前 128 线程再读 8 列 | 编译器调度 |
| 1408 及以上 | 320 | 8+8 列 | 两组提前发射，分批等待，默认缓存配置 |

320 线程的分批读取先发射两组向量 load，用 `vmcnt(3)` 等第一组。第一组的加法、平方和及带权绝对最大值计算完成后，再用 `vmcnt(0)` 等第二组。最后用一次 CTA barrier 汇总两组的 wave 部分结果。

## 构建与调用

在 gfx936 机器上加载 DTK 环境，从仓库根目录构建。Python 入口使用 HCU 版 PyTorch。

```bash
make -C gemma_fused_add_rmsnorm_int8_fp32_gfx936
python3 -m pytest -q gemma_fused_add_rmsnorm_int8_fp32_gfx936/test_gemma_int8_5120.py
```

```python
from gemma_fused_add_rmsnorm_int8_fp32_gfx936.python_api import gemma_fused_add_rmsnorm_int8_fp32

q, scale = gemma_fused_add_rmsnorm_int8_fp32(x, residual, weight, epsilon=1e-6)
```

`x` 和 `residual` 为 `[tokens, 5120]`，`weight` 为 `[5120]`。三个 BF16 张量需要在同一张卡上连续存放并按 16 字节对齐，残差与另外两个张量的存储区域需要互不重叠。算子在当前 PyTorch stream 上异步执行，返回的 `q` 和 `scale` 分别为 `[tokens, 5120]` 和 `[tokens, 1]`。

## 复测结果

在 BW1000 上用 DTK 26.10 nightly 1112 重建。用相同的 BF16 输入和 Qwen 第 0 层权重，在 13 个 token 数下核对了原 FP32 实验二进制和独立版。INT8、scale 和 BF16 残差全部逐位一致。

同卡 GPU Graph p50，单位为 µs。

| Token 数 | Qwen 实际双核 | 原 FP32 实验版 | 独立构建版 |
| ---: | ---: | ---: | ---: |
| 1 | 7.60 | 5.22 | 5.22 |
| 256 | 19.81 | 13.39 | 13.74 |
| 512 | 31.40 | 21.25 | 21.43 |
| 1024 | 55.91 | 36.24 | 36.14 |
| 4096 | 193.64 | 119.82 | 120.10 |
| 16384 | 743.84 | 470.80 | 471.76 |
