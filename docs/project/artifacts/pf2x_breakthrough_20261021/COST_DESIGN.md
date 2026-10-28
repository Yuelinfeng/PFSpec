# PF 2x 成本与实现评估

## 当前判断

已有 PF 相对 Target 的速度约为 1.68158x；另一轮阶段结果为 1.71250x。到 2x 还需分别提升约 18.9% 或 16.8%。建议以同一 Target 实现下的端到端 PF/Target 比值至少提升 20% 作为继续投入的判断线，留出测量波动空间。

当前最值得测的是 Target 注意力实现与整段验证前向的 CUDA Graph 捕获。RTX 4090 的图内微测显示，显式 BMM/softmax 在 T=1/2/4、ctx=64–2048 均快于默认 SDPA；ctx=768 及以上差距最大。它有机会覆盖目标函数里的主要 GPU 成本，但目前证据仍限于单层注意力与固定输入，尚不能据此宣称整段 PF 达到 2x。

## 源码路径与现有证据

- 当前 PF 核心在 `draft_a/pf_feature_unified.py`。K=2 主路径组装 `[pending_f, p1, f1, p2]` 并调用 Target，验证形状是 T=4；短尾路径是 T=2，残差采样路径是 T=1。不要把 K=2 误读成 T=2。
- typed logits 由 `draft_a/pf_ranker_eval.py` 的 `typed_kinds(...)` 提供。`NanoGPTGaussianModel` 定义于 `model/gpt/gpt.py`，持有 `self.transformer = GPT(...)`；`model/gaussian_gpt.py` 构造该包装器。GPT 核心与逐行输出头选择在 `model/gpt/nanochat_gpt.py`。本轮 loader 解除 `torch.compile` 后，`teacher.gpt` 仍为包装器，核心是 `teacher.gpt.transformer`，注意力在内部 transformer block 调用。重复实现 typed verify 不能解释当前瓶颈，也没有必要重做。
- Target 注意力调用在 `model/gpt/flash_attention.py`。当前 SDPA 路径从 GPU `cache_seqlens` 读取 `cache_seqlens[0].item()`；该值在逐层前向中会产生主机同步。候选显式注意力应替换这一文件中对应的 SDPA 计算，或在现有 override 下提供旁路，先保持现有调用与 KV 写入语义。
- `profiling/results/p6a_ctx1024_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv` 中，CUTLASS FP32 attention kernel 约 147.5 µs/次，占该次采样活跃 kernel 时间 56.4%。这是强烈的热点信号，但它代表 GPU kernel 活跃时间，不能直接等同整段 decode 墙钟时间。
- 已有 cache-position 消融位于远端主工程 `profiling/ar_breakdown/sdpa_cachepos_sync_ablation_512_r3/summary.json`，512-token 测例中位吞吐提升 6.26%；`profiling/ar_breakdown/sdpa_cachepos_sync_window_ctx2048_n128_r3/summary.json`，ctx=2048、128-token 测例提升 3.31%。这条同步优化适合作为低风险补益，单独不足以跨越约 17–19% 的缺口。
- 窗口级 AR 记录 `profiling/ar_breakdown/ar_phase_window_ctx2048_n128_r3/summary.json` 显示 Target 前向约占该窗口墙钟时间 96.68%。因此不能把早期“GPU 活跃时间与总墙钟时间差”直接解释成 Python 调度可回收的时间；应以同一次运行的外层计时和图内计时分别判断。

## 候选路线

### 1. 整段 T=4 Target 验证前向使用 CUDA Graph，并测试显式注意力

可改动位置：`model/gpt/flash_attention.py` 的 SDPA 计算分支；验证入口在 `draft_a/pf_feature_unified.py` 中 `tf([pending_f, p1, f1, p2], "target_verify")` 所在路径。已有脚本 `patches/pf2x_breakthrough_20261021/attention/bench_sdpa_candidates_graph.py` 固定 B=1、H=16、D=64，采用 FP32、TF32 关闭、最高 matmul 精度，并保留实际 BSHD→BHSD 的转置步幅。Graph 测量为 16 次图内前向、20 次重放、5 次重复。

RTX 4090 单层注意力图内中位延迟如下，单位 µs：

| T | ctx | 默认 SDPA | 显式 BMM/softmax | 注意力调用加速 |
|---:|---:|---:|---:|---:|
| 1 | 64 | 16.72 | 5.39 | 3.11x |
| 1 | 256 | 41.53 | 6.55 | 6.34x |
| 1 | 768 | 107.80 | 9.73 | 11.08x |
| 1 | 2048 | 252.35 | 13.19 | 19.13x |
| 2 | 64 | 22.59 | 9.33 | 2.42x |
| 2 | 256 | 49.12 | 11.65 | 4.22x |
| 2 | 768 | 110.72 | 16.25 | 6.81x |
| 2 | 2048 | 273.48 | 26.97 | 10.14x |
| 4 | 64 | 22.61 | 9.60 | 2.36x |
| 4 | 256 | 49.20 | 11.95 | 4.12x |
| 4 | 768 | 110.98 | 17.42 | 6.37x |
| 4 | 2048 | 274.48 | 30.26 | 9.07x |

同一测例中，最大概率绝对误差为 2.25e-7，最大注意力输出绝对误差为 8.94e-7，脚本允许值为 1e-4。完整记录在 `docs/project/artifacts/pf2x_breakthrough_20261021/attention_graph_run01/attention_candidates_graph.json` 和 `.csv`。先前非 Graph 循环中，显式 BMM 在 ctx=64/256 的 T=4 慢于默认 SDPA；主机逐次提交间的空档会进入 CUDA Event 区间，不能把该结果解读成纯 GPU 运算时间。图内测量改变了这一排序，实际整段 Target 仍需复测。

端到端收益来自两处：注意力 kernel 工作量缩短；图重放减少逐层 Python 提交造成的空档。验证要在真实 Target 全层和实际 KV cache 上进行，输入覆盖 T=4 主路径及 T=2/T=1 短路径。固定输入的单层结果不能代表多轮自回归采样速度。

### 2. 将 cache position 从逐层 GPU 标量读取改为主机已有位置值

可改动位置：`model/gpt/flash_attention.py` 中 `cache_seqlens[0].item()` 的位置参数读取，并追踪 `draft_a/pf_ranker_eval.py` 中 cache 的 position 更新、前缀填充、接受和回退路径。若 Target 调用边界已有可靠的 Python 整数位置，可在进入 Target 时读取一次并向各层传递，避免每层重新同步。旧消融的 +3.31% 至 +6.26% 是当前最实际的收益范围；它可以和路线 1 叠加，但单独达不到 2x 所需提升。

这条路线的反例很明确：若主机位置在接受 token、回滚未接受候选或残差路径切换后与 GPU cache 长度不一致，后续注意力会读取错 KV 范围。需比较每轮的 cache position、Target logits、输出 token、接受数与回退数；任一差异都应否定该改法。不能据 `.item()` 出现频繁就推断同步一定占外层时间，旧消融和真实整段计时才是依据。

## 数值、KV 与图捕获风险

- 显式 BMM 的 FP32 舍入顺序不同于 SDPA。微测误差很小，但 Target 的 typed logits 经 top-p=0.9 截断后，微小变化仍可能改动支持集或采样轨迹。真实模型上应保存 SDPA 参照 logits、最后层 hidden、各类 policy TV 与完整 token 序列，不能只保留最大误差。
- 图捕获要求输入、输出与 KV cache 地址稳定。T=4 验证会把候选 K/V 写入 cache；接受若干 token 后的提交位置、拒绝后的回退，以及 T=2/T=1 路径都必须维护正确的有效长度。每个 ctx 都捕获独立图会增加显存与管理成本，先测固定形状下的正确性和开销，再决定位置分桶方式。
- 当前 Target 中的 GPU `.item()` 或依赖 Tensor 布尔值的 `.any()` 可能触发主机同步或不支持图捕获。全层 Graph 运行前应先查清这些分支的调用路径，并确认没有丢失现有 logits、hidden 或 cache 更新。
- 单层 Graph 微测用静态输入与固定 mask。它没有包含 Ranker、Feature MLP、KV 分配、前缀填充、采样和回退；故表中倍率只描述注意力调用，不能外推成端到端倍率。

## 最小实测方案与判定

1. 在真实 FP32 Target 上固定 Position/Feature 交错输入和真实 KV cache，对 ctx=64/256/768/2048、T=1/2/4 成对比较 GPU position SDPA、主机 position SDPA、显式注意力 eager、显式注意力 Graph。每种模式先做不计时的热身，再轮换顺序；记录同步后的 host wall、CUDA Event 区间、图构建时间、输入复制时间、显存和数值参照。
2. 每个 shape 至少重复 10 组；保留每组原始行，不只输出中位数。把 Eager 与 Graph 分开解释，CUDA Event 包住主机循环时可能包括 GPU 等待提交的空档。
3. 通过真实 Target 数值验证后，再测完整 PF 与 Target decode。每个 Target 实现都同时跑原始 Target、原始 PF、候选 Target、候选 PF，使用相同模型权重、source、prefix、seed、温度 0.9、top-p 0.9 和生成预算；轮换方法顺序并对每个方法单独热身。两种算法分别报告 token 完全一致的配对结果、完整预算配对的外层 tokens/s、每 source 几何均值及区间。
4. 原项目 2x 目标采用原始 Target 固定参照：candidate PF / original Target 的端到端速度达到 2.0x 即达标。另报告 candidate PF / candidate Target，用于衡量 PF 相对同一优化 Target 的额外收益；要求该比值也达到 2.0x 是更严格的算法收益标准，不能与原项目目标混为一谈。若第一个比值达标、第二个未达标，主要收益来自 Target 优化，PF 在同一优化 Target 上的相对增益还未达到 2.0x。

目前已有单层注意力图内速度、真实 Target 图验证与端到端 PF 配对结果。训练继续收益已接近零，且 KV 复用旧路线曾降低约 1.22% 吞吐，本轮不建议回到训练或重复 KV 复用实验。

## Target 固定位置探针 run02：T=1/2 有效，T=4 未通过

记录见 `docs/project/artifacts/pf2x_breakthrough_20261021/target_graph_run02/` 的 `pairs.jsonl`、`metadata.json` 与 `correctness.jsonl`。探针只完成 ctx=64 的 T=1、T=2；每种形状有 10 组成对计时，每臂每组调用 3 次。下表给出每臂 host wall 中位数，单位 ms/call；速度比是每组原 SDPA GPU position 延迟除以候选延迟后，再取 10 组的中位数。

| ctx / T | SDPA GPU position | SDPA host position | 显式注意力 eager | 显式注意力 Graph | Graph 配对速度比 | 每次中位省时 |
|---|---:|---:|---:|---:|---:|---:|
| 64 / 1 | 13.7406 | 13.4381 | 15.4843 | 3.7315 | 3.6870x | 10.0108 ms |
| 64 / 2 | 17.2021 | 16.9081 | 16.4462 | 4.4348 | 3.8784x | 12.7636 ms |

host position 单独带来约 2.38%（T=1）和 1.73%（T=2）的配对加速；显式注意力 eager 的配对速度比为 0.8995x（T=1 延迟高约 11.2%）和 1.0463x（T=2 延迟低约 4.4%）。CUDA Event 中位延迟与 host wall 相差不到约 0.02 ms，支持以上固定形状调用计时。Graph 输入复制计入每次调用。此处的主要收益来自整段 trunk 的图重放，不能归因于位置读取或显式注意力单项。

每形状一次图捕获约需 184.56 ms（T=1）和 175.36 ms（T=2）；按已测的每次省时计算，约重放 19 次与 14 次才抵回捕获时间。这个估算假设相同 ctx、T、缓存地址和输入缓冲可复用。图热身另需约 44–47 ms；图准备约 0.62–0.65 ms，显式 mask 准备约 0.12–0.15 ms。模型加载约 13.40 s，前缀填充另计，均未摊入表内的单次前向。若每个位置都新建图，捕获时间会抵消收益；若解码期间稳定复用同一图，才可按上述重放次数摊销。

本次固定位置探针的 T=4 数值验证未通过：SDPA 与显式注意力的 logits 最大误差为 `1.62e-5`，policy TV 为 `6.64e-7`，两者都通过；hidden 最大误差 `0.005859`，达到容差的 `1.4687` 倍，超出允许值。显式 eager 与 Graph 的输出彼此完全相同，但这不能弥补它们相对 SDPA 的 hidden 差异。后续动态实现按 SDPA math 路径缩放 Q/K 后，在 `dynamic_verify_run02` 的真实 Target 验证中 T=1/2/4 均通过。固定位置探针里的 T=4 仍没有有效计时；ctx=256/768/2048 在该次运行中也没有计时行。原始 JSONL、正确性记录和张量文件保留在该目录，没有覆盖先前注意力微测或 R1 结果。

这些有效数值是固定 cache position 与 KV 内容上的 GPT core trunk 前向微测，不能代表多轮 Target 解码。Ranker、Feature MLP、采样、接受/拒绝、KV 回退和 EOS 均不在计时路径内，因此不能用 3.69–3.88x 报告 PF/Target 的端到端倍数。若 Target 延迟乘以 `x`、PF 延迟乘以 `y`，同一实现下的新速比为 `R_new = R_old × x / y`；Target 与 PF 按相同比例变快时，二者速比基本不变。以旧记录 `R_old=1.68158` 计，Target 不变时 PF 还需降时约 15.9% 才到 2.0x；换用更快的 Target 后，应使用该 Target 配对测得的 PF 延迟重新计算。

## 端到端动态 Target pilot run02

记录见 `docs/project/artifacts/pf2x_breakthrough_20261021/decode_dynamic_pilot_run02/`。使用 3 个现有开发来源、每个来源 1 个 seed、256-token prefix、512-token 生成预算、FP32、temperature/top-p 均为 0.9；12/12 次运行完成，全部生成满 512 token，未遇到 EOS。按来源等权几何均值计算的外层速度比为：

| 比较 | 速度比 | 来源重抽样 95% 区间 |
|---|---:|---:|
| 原 PF / 原 Target | 1.6250x | 1.4058–1.7602x |
| 候选 PF / 原 Target | 3.5285x | 2.6376–4.4044x |
| 候选 PF / 原 PF | 2.1713x | 1.5209–2.6899x |
| 候选 Target / 原 Target | 2.9585x | 2.6535–3.1424x |
| 候选 PF / 候选 Target | 1.1926x | 0.9940–1.4182x |

因此，这个小型 pilot 按原项目的固定 Target 参照超过 2.0x；候选 PF 相对候选 Target 的比值只有 1.19x，未达到更严格的“在同一优化 Target 上仍有 2x 算法增益”标准。两种 PF 各自在原/候选 Target 实现下，同 seed 输出 token 与计数逐项相同；两种 Target 的配对输出与计数也逐项相同。3 来源、单 seed、固定 512-token 且无 EOS 的结果足以确认此次运行表现，尚不足以代表自然结束长度或更广泛来源分布。

动态 Target 的数值验证见 `docs/project/artifacts/pf2x_breakthrough_20261021/dynamic_verify_run02/metadata.json` 与 `raw_outputs.pt`：T=1/2/4 及 truncate 后替换 Feature 的重放均通过；最高 policy TV 为 `1.14e-6`，最高 hidden 相对行 L2 误差为 `1.69e-6`。端到端 run02 实际记录 2,231 次图前向、7 种图形状，6 次预期长前缀 fallback，无其他 fallback。一个 runtime 在 6 次候选解码间复用；图与 cache 的 setup 共约 1.596 秒，即每次候选解码均摊约 266 ms。这些 setup 已包含在候选端到端外层耗时内，不能再叠加一次。

## 独立来源确认 run01

记录在 `docs/project/artifacts/pf2x_breakthrough_20261021/independent_confirm_run01/`，原始摘要复算结果见 `independent_audit_v2.json`。本轮有 30 个独立来源、每来源两个 prefix，共 60 个配对上下文；四种方法各运行 60 次，合计 240 次。每个上下文使用同一 seed 与 512-token 预算。对每个算法分别比较原实现和候选实现的输出与计数；不要求 PF 与 Target 之间输出相同。

全长度比值使用 60 个配对上下文和 30 个来源。共同满预算比值只纳入两侧都生成 512 token 的上下文；表中的配对数是实际上下文数量，来源数表示这些配对覆盖的来源数量。

| 比较 | 全长度速度比 | 来源重抽样 95% 区间 | 共同满预算配对数 / 来源数 | 共同满预算速度比 | 来源重抽样 95% 区间 |
|---|---:|---:|---:|---:|---:|
| 原 PF / 原 Target | 1.729037x | 1.684580–1.774338x | 55 / 30 | 1.719621x | 1.665593–1.769693x |
| 候选 PF / 候选 Target | 1.442534x | 1.372647–1.495168x | 55 / 30 | 1.444690x | 1.384000–1.492170x |
| 候选 PF / 原 Target | 4.519564x | 4.228452–4.720021x | 55 / 30 | 4.564797x | 4.362947–4.724507x |
| 候选 PF / 原 PF | 2.613919x | 2.506082–2.672991x | 57 / 30 | 2.654372x | 2.611943–2.682805x |
| 候选 Target / 原 Target | 3.133074x | 3.081828–3.164061x | 58 / 30 | 3.134380x | 3.080870–3.167121x |

共同满预算各项均覆盖 30 个来源，但配对数分别为 55、55、55、57、58；`source_count=30` 不能当作配对数。所有五项比值均由原始记录重新计算；来源重抽样使用 10,000 次、seed 381012。独立复算的 8 项均通过，且与主摘要中的几何均值、区间与逐来源值一致。

候选 PF 生成 30,001 token，外层吞吐 327.44 token/s；原 PF 为 30,001 token、125.13 token/s。两者 60/60 配对输出和计数一致，均有 3 次 EOS。候选 Target 与原 Target 也在 60/60 配对上输出和计数一致；两者均生成 30,108 token、各有 2 次 EOS，吞吐为 227.23 和 72.63 token/s。候选 PF / 原 Target 达到 4.52x，超过原项目固定 Target 的 2x 目标；候选 PF / 候选 Target 为 1.44x，说明这批收益主要来自 Target 路径提速，尚未达到同一优化 Target 下 2x 的更严格算法增益标准。

候选解码记录 42,579 次图前向；120 次长前缀走预期原前向 fallback，没有其他 fallback。图与 cache 准备耗时已包含在外层解码时间。此次仅覆盖 30 个来源、每来源两个 prefix、一个 seed 和 512-token 上限；共同满预算统计分别舍弃 5、5、5、3、2 个上下文，结果不能替代更广的来源与生成长度测量。
