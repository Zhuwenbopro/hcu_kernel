# gfx936 INT8 decode GEMM

独立 HIP 实现：X[M,K] × W[N,K].T → Y[M,N]，输入为有符号 INT8，输出为 INT32；M 是 batch size，N 是输出维度，K 是输入维度。主要测量 K=5120、N=34816（full）、17408（TP2）、8704（TP4）或4352（TP8），覆盖每个整数 M=1..1024。

## 编译和调用

示例中的 HIP_VISIBLE_DEVICES=0 请替换为实际空闲卡编号。

需要 Linux、gfx936 HCU、支持该目标的 DTK hipcc，以及 HCU 版 PyTorch 和 NumPy。验证环境为 BW1000 64G / 80 CU、PyTorch 2.11.0、HIP 7.2.26365；hipcc 报告 dcc 26.04.0 / clang17。

    cd decode_gemm_int8_gfx936
    python build.py
    HIP_VISIBLE_DEVICES=0 python test_decode_gemm_int8.py

从仓库根目录调用：

    import torch
    from decode_gemm_int8_gfx936 import decode_gemm_int8, pack_weight

    x = torch.randint(-128, 128, (128, 5120), device="cuda", dtype=torch.int8)
    w = torch.randint(-128, 128, (17408, 5120), device="cuda", dtype=torch.int8)
    packed_w = pack_weight(w)                 # 权重只预打包一次
    y = decode_gemm_int8(x, packed_w, variant="E047")
    assert y.shape == (128, 17408) and y.dtype == torch.int32

`decode_gemm_int8` 每次打包 X；`gemm_packed` 接收 `pack_activation(x, variant)` 的结果，用于纯 kernel 计时或复用已打包输入。打包后的 X 带有 block-M 元数据，不可在不同 M tile 的变体间直接混用。所有路径共用 PI16 权重排列，输出恢复正常 N 顺序。输出不包含量化 scale、bias、激活或反量化。

K 必须为64的正倍数，N 必须是所选变体 CTA N tile 的整数倍。输入和输出应在同一设备，kernel 使用当前 Torch stream；跨 stream 的输入依赖由调用方按通常的 Torch 规则安排。

## 六份源码、十四种配置

GPU 源码按验证版本原样保留，编译宏与原始 SHA256 在 `variants.json` 中。Python接口、编译入口和测试/基准为独立封装。

|ID|原父节点|X / W 路径|CTA M×N|主循环 barrier|
|---|---|---|---|---|
|E023|E016|X共享LDS / W直接REG|128×256|每K64一次|
|E025|E012|X/W直接REG，单wave|32×64|0|
|E030|E028|X/W直接REG，两wave分K|32×64|0；末尾归约一次|
|E038|E027，参考E029|X共享LDS / W直接REG，M优先派发|64×256|每K64一次|
|E047|E046|X直接REG / W共享LDS，双缓冲|128×128|每K64一次|
|E048|E023，参考E006/E025|同tile对照：X/W直接REG|128×256|0|

E023/E048是同tile的X交付路径对照。E038/E047还改变tile、wave映射、W共享、预取或输出策略，其排名不能只归因于X是否进REG。E047仅在M128/N17408/K5120使用常量特化，其余走通用实现；两者均验证无spill。

`auto` 对TP1/TP2沿用原保守规则：N34816且M≤128选E023；N17408且M≤32选E030、M33–128选E047；两者M>128选E038。TP4/TP8使用后文的新规则。对四种已测N提供auto，具体区间见 `dispatch.json`。该性能规则主要针对K5120；其他K可显式选择变体。大M下E023在部分区间仍更快，可显式选择并用基准比较；auto不承诺每个点均为全局最优。

## 正确性和基准

测试包含INT8极值、零、onehot、非32倍数M、K尾部、M128/129等tile边界，以及两种真实N到M1024。每组逐元素比较GPU INT32参考，参考另用CPU INT64采样核验；输出前后设置guard并每次重置哨兵，检查完整输出与越界。输出是4字节INT32，不能沿用BF16的2字节拷贝长度。

    # 约6GiB权重池，36/72份W轮转，每次graph完整遍历一次池
    HIP_VISIBLE_DEVICES=0 python benchmark.py --batches 32,128,129,1024

    # 重现扩展扫描的范围与候选；写入JSON和CSV
    HIP_VISIBLE_DEVICES=0 python benchmark.py --batches 128:1024 --variants E023 E038 E047 E048 --rounds 3 --output build/sweep.json

    # 原1..128概览的5条候选和72次graph调用
    HIP_VISIBLE_DEVICES=0 python benchmark.py --batches 1:128 --variants E023 E025 E030 E038 E047 --graph-calls 72 --rounds 5 --output build/small_sweep.json

基准只测预打包 INT8 kernel；通过HIP Graph和GPU event计时，每轮旋转候选顺序，warmup与CPU/参考计算在计时外。每个参与计时的候选先做完整正确性检查，每次计时graph的最终输出与guard再次核对。约6GiB权重池需要足够显存；用可见设备变量选择空闲卡。

## 指标与结果

主指标为总有效逻辑读写：

    useful_GB/s = (M*K + N*K + 4*M*N) / kernel_us / 1000
    useful_IO_pct = useful_GB/s / 1363.5 * 100

辅助W吞吐只计N*K。1363.5GB/s来自D2D copy基准，读写组合不同，超过100%的点按实测保留。分子只计唯一有效输入/权重和输出；padding、跨M tile重读W、LDS/REG内部搬运不加入有效分子。未采集真实HBM事务计数器，不能由此百分比推断总线空闲比例。INT8算力使用TOP/s，每次乘加计两次整数操作。

四个原正式形状的独立大池验收（3种子×7轮）：

|M|N|路径|kernel µs|总有效IO/基准|
|---:|---:|---|---:|---:|
|32|34816|E023|131.646|101.88%|
|128|34816|E023|144.027|100.18%|
|32|17408|E030|70.096|95.76%|
|128|17408|E047|74.234|97.51%|

扩展的关键点复核（3个新种子×7轮）：

|M|N|本轮最快|kernel µs|总有效IO/基准|
|---:|---:|---|---:|---:|
|129|34816|E038|226.510|63.75%|
|129|17408|E038|98.875|73.26%|
|1024|34816|E038|857.693|27.89%|
|1024|17408|E038|427.753|28.41%|

TP2的同tile X路径对照：M128时E048/E023约104.384/115.971µs，直接REG更快；M129时变为150.869/109.262µs，共享LDS更快。该方向在三个独立种子中一致；密集扫描M129..1024均为E023领先E048。两者静态VGPR为183/184，均0spill，变化发生在M tile数/工作集/调度边界，不能解释为REG容量随总batch增长而不足。

实验次序为S001（1..128概览）→S002（扩到1024代表点、剪枝明显落后路径）→S003（128..1024逐整数，加入E023→E048匹配对照）→S004（关键点独立复核）。E047曾预测常量特化提速2.5%，实际约0.26%，该性能预测未成立；源码和结果均保留。

`results/batch_sweep.csv`覆盖四种N的每个整数M=1..1024，只列当时参与候选中最快的中位数；full/TP2的M<128来自S001的5轮/5候选、M≥128来自S003的3轮/4候选；TP4/TP8来自S007的3轮/分区候选，TP与来源列明确区分。1%以内看作近似平手，不宜据逐点小幅胜出硬编码分派。`x_delivery_comparison.csv`保留全部同tile对照；`independent_validation.json`给出S004的24个关键形状、21样本聚合及种子范围。原四个形状达90%的结论只覆盖M32/M128，不外推到1024。

## 首次归档自测（TP1 / TP2）

六个源码在该目录独立重编译后，288项正确性/auto入口检查全部通过，覆盖767,502,336个输出元素；全部guard完好，6个源码SHA与原始验收版本一致。另对M32/128/129/1024×两种N×6条路径做48组约6GiB冷池基准检查，并验证72次graph调用入口。环境、资源、重编译哈希与本轮计时见 `results/rebuild_validation.json`；历史结果文件保留原值。

## TP4 / TP8 扩展
这里TP按输出N切分：N=34816/TP，K=5120不变，X仍为完整M×K；基准为单卡本地GEMM，不包含多卡通信。
在原有六个配置上增加八个编译配置，复用同样的六份GPU源码，源码SHA不变：

|ID|父节点|唯一概念变化|CTA M×N|深度|K分片|
|---|---|---|---|---:|---:|
|E049|E025|单wave M32→64|64×64|1|1|
|E050|E049|单wave M64→128|128×64|1|1|
|E051|E030|预取深度1→2|32×64|2|2|
|E052|E051|预取深度2→4|32×64|4|2|
|E053|E030|保留K二分，M32→64|64×64|1|2|
|E054|E051|N64→32，增加CTA|32×32|2|2|
|E055|E053|M64路径深度1→2|64×64|2|2|
|E056|E051|K二分→四分，NW/KS保持同Ntile|32×64|2|4|

每步先登记原因与量化预测，再测46组INT8极值/K尾/M边界/真实N正确性，以及同父节点约6GiB池配对性能。预测与失败结果见results/tp48_experiments.json：单wave扩大Mtile失败；双槽在小M有收益但在部分大M退化；深度4、N32、K四分均未达到各自预设增益；M64保留K二分才在中M明显改善。E055的TP4 M128预测失败，但TP8 M128有收益，保留为分段候选。

经过S005冻结旧候选、S006的14候选同池代表点比较，S007逐整数扫描M1..1024：2048个形状、11264项参与候选检查全部正确。S007为一次完整权重池warmup和3轮；候选按运行前登记区间剪枝，具体条件在results/tp48_schedule.json。S008用3个新种子、每种子7轮/2次warmup复核56个关键形状，共936项候选检查，四个家族边界预测均成立。

|TP|主循环REG领先的连续区间|下一行|独立复核结论|
|---:|---|---:|---|
|4|bs1–128|129|128时REG快20.8%；129时共享LDS快18.6%|
|8|bs1–256|257|256时REG快24.8%；257时共享LDS快11.0%|

主循环REG包括末尾使用LDS整数归约的K分片路径，仍有一次CTA内同步；完全0LDS的单wave控制并非最优。固定tile的VGPR不会随总bs成比例增加，这些边界不是“REG容量装不下”的边界。

独立复核中位数：
|TP|M|最快|us|总有效IO/基准|
|---:|---:|---|---:|---:|
|4|1|E051|33.908|96.48%|
|4|32|E051|35.771|93.99%|
|4|64|E030|38.206|90.45%|
|4|128|E053|44.696|81.51%|
|4|129|E047|59.212|61.58%|
|4|256|E047|62.452|64.34%|
|4|257|E047|77.670|51.77%|
|4|1024|E047|208.001|30.13%|
|8|1|E052|18.315|89.32%|
|8|32|E052|19.627|85.96%|
|8|64|E051|22.145|78.57%|
|8|128|E055|29.626|62.30%|
|8|129|E053|37.491|49.27%|
|8|256|E053|41.963|49.02%|
|8|257|E047|55.807|36.89%|
|8|1024|E023|110.579|30.08%|

90%总有效IO目标没有在全范围达到，TP8仍有明显差距。范围后移不等于vmcnt/barrier的全部开销已经隐藏，也不等于物理HBM利用率已达90%。

auto按dispatch.json选择稳定tile区间；TP8的bs65–96及769–832没有为约1%的差异增加细分规则。独立复核点中保守分派相对本轮最快值的最大额外耗时约1.15%，不承诺逐点全局最优。

    # 显式比较TP4/8；--tp仅将N切分，仍在当前可见的一张卡运行
    HIP_VISIBLE_DEVICES=0 python benchmark.py --tp 4 8 --batches 1,32,64,128,256,1024 --variants E030 E047 E051 E052 E053 E055 --rounds 3 --output build/tp48.json

TP4/TP8全部逐整数家族对照在results/tp48_reg_vs_lds.csv；独立种子聚合在tp48_independent_validation.json。所有14种配置独立重编译、正确性/auto入口检查及TP1/2/4/8冷池基准记录在tp48_rebuild_validation.json。
