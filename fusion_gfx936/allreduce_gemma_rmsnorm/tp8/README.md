# TP8 · 单机 gfx936

在此目录执行 `bash native/build.sh`，从 [tp8_ar.py](tp8_ar.py) 导入 `TP8Allreduce`。默认 `direct_through=12`，`algorithm="auto"` 在 M≤12 时直接 pull，其余走分片归约。可在构造时指定其他阈值来复测。完整的输入约定、调用示例和测试环境见[算子说明](../README.md#构建和使用)。

[逐 M 实测](results/by-m.csv)覆盖 M=1–4096。[预扫记录](results/routing.json)给出切换点的选择依据，[每轮原始延迟](results/full-timings.jsonl.gz)记录 AITER、直接 pull 和分片在八张卡上的计时。可用[落后点复测](results/loss_recheck-timings.jsonl.gz)核对两个接近 1× 的区间，用[内存池复测](results/pool_recheck-timings.jsonl.gz)核对测试进程重启前后的计时一致性。
