from pathlib import Path
import json
R=Path(__file__).resolve().parent
d=json.loads((R/'RESULT.json').read_text())
fence=chr(96)*3
def table(precision):
    rows=[p for p in d['points'] if p['precision']==precision and p['M'] in [1,2,4,8,16,32,64,128,129,160,176,192,193,224,256]]
    out=['| M | Current GEMM us | + AITER us | + Triton us | Same tile + Triton us | Fused us | Speedup vs faster current separate |','|---:|---:|---:|---:|---:|---:|---:|']
    for p in rows:out.append(f"| {p['M']} | {p['current_GEMM_us']:.3f} | {p['current_plus_aiter_us']:.3f} | {p['current_plus_triton_us']:.3f} | {p['same_tile_plus_triton_us']:.3f} | {p['fused_us']:.3f} | {p['speedup_vs_faster_separate']:.3f}x |")
    return '\n'.join(out)
a=d['accuracy'];q=d['ranges']
def accuracy_text(precision):
    r=a[precision];ai=r['fused_vs_AITER'];tr=r['fused_vs_Triton']
    return f"""- {precision}: {r['M_cases']}组，两seed逐M覆盖1..256；融合前gate/up与当前GEMM逐位一致，guard均通过。
- 对AITER：最大相对L2 {ai['max_relative_L2']:.9g}，最大绝对差 {ai['max_abs']:.9g}；gate>=-80区域最多1个BF16 ULP。
- 对Triton：最大相对L2 {tr['max_relative_L2']:.9g}，最大绝对差 {tr['max_abs']:.9g}；gate>=-80区域最多1个BF16 ULP。
- 更负gate下的额外ULP差异：对AITER最大绝对差 {ai['max_abs_extra_ULP_gate_lt_neg80']:.9g}，对Triton {tr['max_abs_extra_ULP_gate_lt_neg80']:.9g}。全局ULP最大值分别 {ai['max_BF16_ULP']} / {tr['max_BF16_ULP']}；这些大ULP距来自接近零的FP32下溢区，没有将其描述为全域逐位一致。
"""
report=f"""# Packed gate/up GEMM 融合 SiLU(gate) * up

已完成INT8和BF16两条原生计算路径，直接输出BF16 [M,8704]。固定M1..256、K5120、gate/up合并N17408，对应TP2后的MLP gate/up投影。静态权重按G16/U16交错打包。此前已验证的INT8/BF16成果包独立封存，本轮未替换LightOp或模型服务入口。

## 正式性能

BW47 / zhuwenbo_sgl_0522 / GPU5 / gfx936。33个实际M测点，三个独立于训练和精度验证的seed，各七轮，每图72份不同W轮转。每seed取七轮中位数，再取三个seed的中位数。GPU Graph内计时，两边均包含A在线重排，排除静态W/权重scale准备、输入INT8量化、首次模块加载/JIT和Graph构建。

主对照是此前冻结的当前最佳GEMM，加AITER silu_and_mul(limit=0)或单核Triton SiLU*up，选两者聚合中位数中更快者作保守比较；两组完整结果均保留。另测同一交错W、同一tile、同一K主循环输出BF16完整gate_up后接Triton激活，作为同tile对照。

INT8的33点：{q['int8']['wins_vs_faster_separate']}点更快，加速比 {q['int8']['min_speedup_vs_faster_separate']:.3f}–{q['int8']['max_speedup_vs_faster_separate']:.3f}x，延迟减少 {q['int8']['min_reduction_pct']:.2f}%–{q['int8']['max_reduction_pct']:.2f}%。
BF16的33点：{q['bf16']['wins_vs_faster_separate']}点更快，加速比 {q['bf16']['min_speedup_vs_faster_separate']:.3f}–{q['bf16']['max_speedup_vs_faster_separate']:.3f}x，延迟减少 {q['bf16']['min_reduction_pct']:.2f}%–{q['bf16']['max_reduction_pct']:.2f}%。

### INT8 x INT8 -> BF16

{table('int8')}

### BF16 x BF16 -> BF16

{table('bf16')}

完整33点见latency-int8.csv / latency-bf16.csv；图fused-gate-up.png / .svg。曲线仅连接实际测点，不表示已逐M测量性能。计时原始样本benchmark-raw.json及benchmark-*-seed-*.jsonl，{d['method']['timed_groups']}组、{d['method']['timed_replays_checked']}次实现回放，每次检查最后一份W的完整输出。

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

{accuracy_text('int8')}
{accuracy_text('bf16')}

BF16 E030 split-K会改变FP32累加顺序，少数中间值会跨越BF16舍入边界；同一中间gate_up上融合/独立激活一致，原始核对见bf16-small-M-order-check.json。为了对齐当前BF16中间结果，它不进入BF16性能筛选/默认路由；INT8 E030仍可用且在M1选中。默认逐M选路在fused-policy.json，完整默认策略检查确认两dtype融合前的gate/up均与当前GEMM逐位一致。

激活对照默认rtol=1/128、atol=1/256；同时单独施加更严格的gate>=-80区域最多1个BF16 ULP要求。更负gate的额外ULP差异必须满足绝对差<1e-20，原始实测值远小于此限。上述阈值和全部原始误差保留在validate.py / validation.json，未用宽松阈值替代中间矩阵逐位检查。

## 计时与环境

72份数值对应的原W、旧packed-W、新packed-W同时准备，INT8每份85MiB，BF16每份170MiB；各实现各自轮转全部72份。每轮随机实现顺序，每seed随机M顺序。输入A、sx和数值相同的W/sw一一对应；静态W及sw准备不计时，A在线打包计时。INT8公开接口直接接收xq/sx，本轮不含之前的per-token量化。

原始环境见environment.json。频率采样{d['clock_samples']}次：
{fence}json
{json.dumps(d['clocks'],indent=2)}
{fence}
本轮未改动频率设置。运行结束的显存/进程检查另存postflight.txt。

## 保存与接口

此前两个成果包保存在frozen-int8.tar.gz和frozen-bf16.tar.gz，原始目录及SHA256由PRESERVED.json记录；原包校验未变。新接口fused_gemm.py，生成器fused_generator.py / fused_legacy_generator.py，所有汇编、反汇编与属性在kernels/。

{fence}python
from fused_gemm import prepare_weight, silu_mul

# BF16 A [M,5120]; BF16 W [5120,17408], stride=(1,5120)
p16 = prepare_weight(w16)
y16 = silu_mul(x16, p16)  # BF16 [M,8704]

# INT8 A/W; sx FP32 [M,1]; sw FP32 [17408,1]
p8 = prepare_weight(w8, scale_b=sw)
y8 = silu_mul(xq, p8, scale_a=sx)  # BF16 [M,8704]
{fence}

权重及其scale修改后需重新prepare；prepare后的同一份W可供重复调用，普通带version的tensor会检测更新。A和scale必须连续，同设备；当前固定N/K与M1..256，不支持bias、clipping或autograd。热身所需M后可捕获Graph。模块继承调用方GPU映射，验证与benchmark脚本显式选择BW47物理GPU5。

复现顺序：fused_generator.py，fused_legacy_generator.py，smoke.py，screen.py --fresh，select_policy.py，validate.py，benchmark.py，analyze.py，report.py。候选和正式计时使用不同seed。独立激活两种实现及测试输入分布在test_common.py / screen.py中；所有脚本和分析工件留在服务器。
"""
(R/'REPORT.md').write_text(report)
readme=f"""# Fused packed gate/up GEMM

Fixed gfx936 N=17408, K=5120, M=1..256. Output BF16 [M,8704].
W source: [K,N], stride=(1,K), ordinary concatenation [gate | up].
prepare_weight packs G16/U16, with PI16 inside each block. Static preparation is once per W.
INT8 uses per-row/per-column FP32 scales and INT32 accumulation.
BF16 uses native BF16 MMAC with FP32 accumulation.
Both round gate/up to BF16 in registers before FP32 SiLU*up and final BF16 RNE.

{fence}python
from fused_gemm import prepare_weight, silu_mul
p = prepare_weight(w16)
y = silu_mul(x16, p)
p = prepare_weight(w8, scale_b=sw)
y = silu_mul(xq, p, scale_a=sx)
{fence}

All calls include online A packing. Re-prepare after changing W or sw.
No bias, clipping, arbitrary N/K, autograd or model-dispatch integration.
Warm up required M before Graph capture.
Standalone activations are AITER(limit=0) and a single-kernel Triton implementation.
Full validation, tiny-value numerical limits, independent formal timing and same-tile control are in REPORT.md / RESULT.json.
Tests and benchmarks select physical GPU5 on BW47; the runtime inherits the caller's GPU mapping.
Old packages and source hashes are independently preserved in PRESERVED.json.
"""
(R/'README.md').write_text(readme)
print('REPORT_DONE',flush=True)
