# TP4 · 单机 gfx936

在此目录执行 `bash native/build.sh`，从 [tp4_ar.py](tp4_ar.py) 导入 `TP4Allreduce`。每个 rank 用同一 Gloo 组构造通信器，`fused(..., algorithm="auto")` 在 M≤19 时直接 pull，其余走分片归约。完整的输入约定、调用示例和测试环境见[算子说明](../README.md#构建和使用)。

[逐 M 实测](results/by-m.csv)覆盖 M=1–4096。[预扫记录](results/routing.json)给出切换点的选择依据，[每轮原始延迟](results/full-timings.jsonl.gz)记录 AITER、直接 pull 和分片在四张卡上的计时。[专项检查](results/validation-summary.json)包含 eager 和连续两次 Graph 重放的逐比特结果。
