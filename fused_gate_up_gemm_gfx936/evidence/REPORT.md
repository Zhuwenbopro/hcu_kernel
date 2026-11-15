# Packed gate/up GEMM 融合 SiLU(gate) * up

已完成INT8和BF16两条原生计算路径，直接输出BF16 [M,8704]。固定M1..256、K5120、gate/up合并N17408，对应TP2后的MLP gate/up投影。静态权重按G16/U16交错打包。此前已验证的INT8/BF16成果包独立封存，本轮未替换LightOp或模型服务入口。

## 正式性能

BW47 / zhuwenbo_sgl_0522 / GPU5 / gfx936。33个实际M测点，三个独立于训练和精度验证的seed，各七轮，每图72份不同W轮转。每seed取七轮中位数，再取三个seed的中位数。GPU Graph内计时，两边均包含A在线重排，排除静态W/权重scale准备、输入INT8量化、首次模块加载/JIT和Graph构建。

主对照是此前冻结的当前最佳GEMM，加AITER silu_and_mul(limit=0)或单核Triton SiLU*up，选两者聚合中位数中更快者作保守比较；两组完整结果均保留。另测同一交错W、同一tile、同一K主循环输出BF16完整gate_up后接Triton激活，作为同tile对照。

INT8的33点：33点更快，加速比 1.024–1.132x，延迟减少 2.31%–11.65%。
BF16的33点：33点更快，加速比 1.014–1.139x，延迟减少 1.41%–12.21%。

### INT8 x INT8 -> BF16

| M | Current GEMM us | + AITER us | + Triton us | Same tile + Triton us | Fused us | Speedup vs faster current separate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70.527 | 76.349 | 72.713 | 72.791 | 71.035 | 1.024x |
| 2 | 70.567 | 76.684 | 72.900 | 72.953 | 71.053 | 1.026x |
| 4 | 70.767 | 76.902 | 72.998 | 72.953 | 71.064 | 1.027x |
| 8 | 71.364 | 77.549 | 73.673 | 73.409 | 71.431 | 1.031x |
| 16 | 73.220 | 79.280 | 75.671 | 74.231 | 71.700 | 1.055x |
| 32 | 74.036 | 80.724 | 77.929 | 76.120 | 72.633 | 1.073x |
| 64 | 75.773 | 83.093 | 81.611 | 80.129 | 75.691 | 1.078x |
| 128 | 79.242 | 89.980 | 88.173 | 87.871 | 80.018 | 1.102x |
| 129 | 79.889 | 90.538 | 88.604 | 88.755 | 83.749 | 1.058x |
| 160 | 82.267 | 93.104 | 92.067 | 92.182 | 86.629 | 1.063x |
| 176 | 84.160 | 97.047 | 94.569 | 95.951 | 86.469 | 1.094x |
| 192 | 84.704 | 98.707 | 96.727 | 96.658 | 86.967 | 1.112x |
| 193 | 88.562 | 102.087 | 100.289 | 100.671 | 94.344 | 1.063x |
| 224 | 91.509 | 104.793 | 103.915 | 104.338 | 97.296 | 1.068x |
| 256 | 98.131 | 114.613 | 112.160 | 129.120 | 99.098 | 1.132x |

### BF16 x BF16 -> BF16

| M | Current GEMM us | + AITER us | + Triton us | Same tile + Triton us | Fused us | Speedup vs faster current separate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 135.782 | 141.578 | 137.962 | 137.624 | 135.947 | 1.015x |
| 2 | 135.944 | 141.733 | 137.993 | 137.687 | 136.049 | 1.014x |
| 4 | 136.022 | 141.822 | 138.127 | 137.884 | 136.162 | 1.014x |
| 8 | 136.375 | 142.580 | 138.702 | 138.429 | 136.340 | 1.017x |
| 16 | 137.124 | 143.262 | 139.811 | 139.429 | 136.764 | 1.022x |
| 32 | 137.782 | 143.862 | 141.513 | 141.240 | 137.644 | 1.028x |
| 64 | 139.304 | 146.353 | 144.895 | 144.738 | 139.655 | 1.038x |
| 128 | 143.404 | 153.895 | 151.655 | 151.440 | 143.751 | 1.055x |
| 129 | 147.220 | 157.751 | 155.795 | 155.609 | 148.295 | 1.051x |
| 160 | 150.731 | 161.920 | 160.522 | 160.282 | 151.593 | 1.059x |
| 176 | 153.400 | 166.869 | 163.984 | 163.738 | 154.162 | 1.064x |
| 192 | 154.613 | 168.338 | 166.204 | 166.198 | 154.949 | 1.073x |
| 193 | 166.091 | 179.342 | 178.158 | 176.838 | 166.342 | 1.071x |
| 224 | 169.300 | 183.458 | 182.793 | 181.227 | 169.558 | 1.078x |
| 256 | 182.318 | 200.242 | 198.531 | 207.327 | 174.291 | 1.139x |

完整33点见latency-int8.csv / latency-bf16.csv；图fused-gate-up.png / .svg。曲线仅连接实际测点，不表示已逐M测量性能。计时原始样本benchmark-raw.json及benchmark-*-seed-*.jsonl，198组、6930次实现回放，每次检查最后一份W的完整输出。

这些是GPU算子路径时间，非模型请求端到端时间。选路是本候选集的测量结果，不是任意tile/调度的全局最优；与冻结当前最佳的比较包含tile选择变化，同tile对照也已单独列出。

## 权重与寄存器路径

外部W仍按正常gate_up拼接：[K,17408]，stride=(1,5120)，底层连续[N,K]，前8704行gate、后8704行up。prepare_weight在同一次静态W打包中变成 G[0:16],U[0:16],G[16:32],U[16:32]...，每个N16内部保留原PI16排列。INT8的per-channel权重scale同步按G16/U16交错。

N方向划分的tile每wave恰好负责32维：16gate、16up；每lane的四个gate累加器与四个up累加器直接配对。也保留M方向和原有E047/E030候选，其中每wave可处理多组G16/U16，配对仍在该wave内完成。大M选中的M256_PF_SAB属于这种每wave多组配对。没有为激活增加wave间shuffle、LDS发布或barrier。

INT8路径：INT8 A/W -> v_mmac_i32_16x16x32_i8 / INT32 accum -> FP32(sx*sw)*float(acc) -> 寄存器内BF16 RNE -> FP32 SiLU*up -> BF16 RNE直接写回。
BF16路径：BF16 A/W -> v_mmac_f32_16x16x16_bf16 / FP32 accum -> 寄存器内BF16 RNE -> FP32 SiLU*up -> BF16 RNE直接写回。

保留GEMM原本输出BF16的舍入点，未把未舍入FP32累加器直接送进SiLU。SiLU尾部使用v_exp_f32及v_rcp_f32，保留NaN分类；final使用global_store_dwordx2写四个BF16。现有同tile的VGPR/LDS预算未增加，scratch为0。64个模块、68个函数配置，静态审计见instruction-resource-audit.json，通用融合/未融合debug对照的K主循环源代码相同。旧E047/E030保留其原K循环和必要的split-K归约，仅替换输出尾部。

直接省去[M,17408] BF16中间矩阵的写回及独立激活对它的读取，最终只写[M,8704]。这是每M行减少68KiB逻辑中间数据IO，M256为17MiB；不是物理HBM流量实测值，不将整体加速单独归因于某项资源或某条指令。

## 正确性

两dtype各两个随机seed，逐M覆盖全部256点，共1024组。另有98组整数/边界smoke，68组逐family随机检查，覆盖最后K段one-hot、guard、Graph重复回放、极值INT8、NaN/Inf类别，以及权重/scale版本变化的拒绝检查。FP64完整矩阵乘作GEMM参考，每个随机seed再以CPU FP64点检36个元素；SiLU也以FP64逐元素参考核对。

- int8: 512组，两seed逐M覆盖1..256；融合前gate/up与当前GEMM逐位一致，guard均通过。
- 对AITER：最大相对L2 3.34306986e-07，最大绝对差 0.03125；gate>=-80区域最多1个BF16 ULP。
- 对Triton：最大相对L2 1.77243198e-05，最大绝对差 1；gate>=-80区域最多1个BF16 ULP。
- 更负gate下的额外ULP差异：对AITER最大绝对差 0，对Triton 5.49190961e-35。全局ULP最大值分别 1 / 1682；这些大ULP距来自接近零的FP32下溢区，没有将其描述为全域逐位一致。

- bf16: 512组，两seed逐M覆盖1..256；融合前gate/up与当前GEMM逐位一致，guard均通过。
- 对AITER：最大相对L2 2.14889727e-06，最大绝对差 0.015625；gate>=-80区域最多1个BF16 ULP。
- 对Triton：最大相对L2 8.3410854e-06，最大绝对差 4；gate>=-80区域最多1个BF16 ULP。
- 更负gate下的额外ULP差异：对AITER最大绝对差 2.4795584e-39，对Triton 2.25694915e-34。全局ULP最大值分别 27 / 1942；这些大ULP距来自接近零的FP32下溢区，没有将其描述为全域逐位一致。


BF16 E030 split-K会改变FP32累加顺序，少数中间值会跨越BF16舍入边界；同一中间gate_up上融合/独立激活一致，原始核对见bf16-small-M-order-check.json。为了对齐当前BF16中间结果，它不进入BF16性能筛选/默认路由；INT8 E030仍可用且在M1选中。默认逐M选路在fused-policy.json，完整默认策略检查确认两dtype融合前的gate/up均与当前GEMM逐位一致。

激活对照默认rtol=1/128、atol=1/256；同时单独施加更严格的gate>=-80区域最多1个BF16 ULP要求。更负gate的额外ULP差异必须满足绝对差<1e-20，原始实测值远小于此限。上述阈值和全部原始误差保留在validate.py / validation.json，未用宽松阈值替代中间矩阵逐位检查。

## 计时与环境

72份数值对应的原W、旧packed-W、新packed-W同时准备，INT8每份85MiB，BF16每份170MiB；各实现各自轮转全部72份。每轮随机实现顺序，每seed随机M顺序。输入A、sx和数值相同的W/sw一一对应；静态W及sw准备不计时，A在线打包计时。INT8公开接口直接接收xq/sx，本轮不含之前的per-token量化。

原始环境见environment.json。频率采样159次：
```json
{
  "freq1_input": {
    "1499000000": 159
  },
  "freq2_input": {
    "1800000000": 159
  },
  "power_dpm_force_performance_level": {
    "manual": 159
  }
}
```
本轮未改动频率设置。运行结束的显存/进程检查另存postflight.txt。

## 保存与接口

此前两个成果包保存在frozen-int8.tar.gz和frozen-bf16.tar.gz，原始目录及SHA256由PRESERVED.json记录；原包校验未变。新接口fused_gemm.py，生成器fused_generator.py / fused_legacy_generator.py，所有汇编、反汇编与属性在kernels/。

```python
from fused_gemm import prepare_weight, silu_mul

# BF16 A [M,5120]; BF16 W [5120,17408], stride=(1,5120)
p16 = prepare_weight(w16)
y16 = silu_mul(x16, p16)  # BF16 [M,8704]

# INT8 A/W; sx FP32 [M,1]; sw FP32 [17408,1]
p8 = prepare_weight(w8, scale_b=sw)
y8 = silu_mul(xq, p8, scale_a=sx)  # BF16 [M,8704]
```

权重及其scale修改后需重新prepare；prepare后的同一份W可供重复调用，普通带version的tensor会检测更新。A和scale必须连续，同设备；当前固定N/K与M1..256，不支持bias、clipping或autograd。热身所需M后可捕获Graph。模块继承调用方GPU映射，验证与benchmark脚本显式选择BW47物理GPU5。

复现顺序：fused_generator.py，fused_legacy_generator.py，smoke.py，screen.py --fresh，select_policy.py，validate.py，benchmark.py，analyze.py，report.py。候选和正式计时使用不同seed。独立激活两种实现及测试输入分布在test_common.py / screen.py中；所有脚本和分析工件留在服务器。
