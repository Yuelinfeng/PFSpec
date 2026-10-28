# 论文证据索引

## 2026-10-23 v3研究边界与重复证据

单备用新增吞吐约1.81%已在同卡四进程、两个新seed上重复；来源等权1.0180919、95%来源区间约[1.0148,1.0213]，480配对输出接受相同、少2908恢复。28/30来源正，首次图子集慢，不能写普遍每次请求都快，也不能替代已有端到端区间跨1的增量结论。见[最终复核](artifacts/rescue_repeat_20261022/RESULTS_REVIEW_20261023.md)。

v3接受率反例为局部存在性命题；完整历史的统计信息与可付费计算不同，需限定选择器输入与预算。多分支、成本选树、廉价上下文估值在SpecBranch/Sequoia/EAGLE-2已有，真实q=δa与Ranker评分不可混用。当前支持类型条件恢复复用，不支持已完成动态选择或一般最优；k1..4负结果与已完成归因/三维端到端应并列引用。39固定前缀同池排序只有k1的2/39上下文增加覆盖质量，来源等权+0.6764个百分点；k2..4无增加，不能转成在线速度。最近原文、可用字段与论文主张边界见[v3决定](artifacts/v3_review_20261023/DECISION.md)。

## 2026-10-22 新接受率方向与备用继续执行证据

三十来源新种子纯备用确认支持小幅执行增益：备用/同轮W1=1.017481 [1.002716,1.026950]，备用/原Target=5.056076 [4.553989,5.386374]；60/60输出和alpha计数相同、少388恢复。主倍率为执行优化加投机总收益，新增增量约1.75%，不宣称原接受率提高。固定Feature历史policy混合微平均接受率33.2333%→34.5925%，但来源等权差区间[-1.6660,+3.5001]pp，组合/W1区间跨1，不能主张可靠接受或净速度改善。

混合重合度的凹性、精确残差抽样与通用成本比值不作为首次理论贡献。三十来源历史用于旧方法，纯备用追加发生在组合结果读出后；没有新训练或三维端到端计时。数值证据有固定范围及121/182逐元素allclose张量组差异，不能写全局逐位等价。原始数据、重采样seed、源码版本与可主张范围见[最终结果](artifacts/spec_next_20261022/RESULTS.md)和METHOD_NOTES.md。

## 2026-10-21 同前缀预测验证与相关工作结论

[完成报告](artifacts/recovery_model_20261021/RESULTS.md)对应十来源39前缀、八种子、三个动作、三次重复的2808行主测，另480行诊断。恢复事件模型/W1局部进展率0.987458 [0.953654,1.020480]，与分别预测R/C的39/39动作相同；直接R/C为1.010234 [0.984362,1.043486]，尚无稳定选择收益。成本预测MAE仅由1.3352变为1.3324ms。最终统计为dev_analysis_v2.json，旧版重采样区间不采用；八模型原始评分独立复算相符，最终32项GPU对照与480/480诊断输出一致。此次无EOS、无在线选择、无新端到端结果，未启用本轮外部测试。九篇原始论文与一综述的[比对](artifacts/recovery_model_20261021/NOVELTY_REVIEW.md)表明通用奖励/成本与动态选树已有覆盖，pair公式保留为状态机计数表达，暂不列为新的通用理论或有效选择机制。

## 2026-10-21 类型化进展与成本理论草稿

新增[论文设计](artifacts/paper_design_20261021/RESEARCH_STORY.md)、[带前提和证明的理论草稿](artifacts/paper_design_20261021/THEORY_NOTES.md)及[CPU复算](artifacts/paper_design_20261021/count_and_algebra_audit.json)。main_run02的W1/W2各18条满足generated=2*(tree_verify+feature_hits+k1)，Target_calls=tree_verify+recovery+k1+prefill，每条prefill=1且position_fallbacks=0。恢复感知模型给出局部增宽条件，未证明全局最优，未有新的在线选择实现。精确算术采样证明属于标准方法的P/F实例化；浮点数值范围与BOS限制沿用真实记录。原Target合计4.966× token及4.335×端到端不作为算法独立倍率。文献范围为六篇相关正文，尚不能支持首次主张。本轮只做文档与CPU复算，无新GPU速度证据。

更新：2026-10-17。本文把可用于论文的主张对应到正式或开发记录、统计定义、原始产物和可用表述。路径以项目根目录为起点；明确标注“远端”的产物位于 Autodl，不在本地交接副本中。

2026-10-16 远端逐项核对与复算见 [研究工作与真实证据对应](RESEARCH_EVIDENCE_AUDIT_2026-10-16.md)。该记录补齐 Draft-A、GPT-D8、D4、Position Ranker、Feature MLP 与 K2 的实际来源；阶段 E 的 2,160 行吞吐和接受率汇总、阶段 F 的 360 行完整时间及相关区间均已复现。本轮未运行新的 GPU 实验。

## 统计口径

- 阶段 E 是固定前缀、最多续写 512 token 的 FP32 解码比较；它不包含 VQ-VAE 解码、Gaussian 导出或渲染。
- 阶段 F 是自然 EOS 结束的全长单 chunk 流程；时间从已准备的前缀计至 Gaussian NPZ 和六张 PNG 写入。两阶段的时间、吞吐和接受率分别列示。
- 阶段 E 的“微平均”按所有记录的新增 token 总数除以总耗时；“来源组几何平均”先在来源组内汇总，再让来源组等权。接受率为已接受数/已核验数，汇总时按计数合并。
- 阶段 F 的真实补全时间比按相同样本的 Target 完整耗时除以提议模式完整耗时；先在同一场景的三个 seed 内取几何平均，再让 30 个场景等权。95% 区间以场景为单位重采样 10,000 次，并保留每场景三个 seed。
- 本研究的“独立测试场景”指未用于本轮方法训练、调参和选择的场景。Target 与 D8 历史训练来源另有其既往接触范围，故不概括成所有模型均未见过这些来源。

## 2026-10-17 新增机制与成本证据

[候选成本 A1–A3](artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md)汇总 56 个相同上下文的八 seed 单轮对照与固定窗口记录：PF/P-only 固定窗口速度比为 1.171× [1.130, 1.211]，单轮推进效率比为 1.141× [1.085, 1.195]。P2 拒绝后整段处理占复测时长 1.337%，将其完全删除且其余路径不变的估算为 1.0136×，仅代表当前路径的乐观时间相减。

[Feature 条件收益分析](artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md)在十个开发来源上补采 5,078 个候选轮次标签并重放 448 次单轮。候选间隔为 1 与大于 1 的分组速度比为 1.190× [1.117, 1.261] 和 0.970× [0.934, 1.015]。生成前特征预测 Feature 接受概率的留一来源 R² 为 0.264；额外耗时预测弱于均值参照，尚无在线选择策略。

[第二轮执行记录](ROUND2_EXECUTION_2026-10-17.md)对应 A0/A1/E0、B0/B1 与 A2。首训练 seed 的 392 条在线比较中，旧 PF 为 1.624× [1.557, 1.682]，B1-E 为 1.506× [1.450, 1.558]，相对旧 PF 为 0.927× [0.905, 0.948]。另外两个训练 seed 只有固定上下文离线重合值，不能用于推断跨 seed 的在线速度。

## 理论接受率、实测接受率与净速度

给定已提交历史 `h`，设 Target 与 Draft 在相同合法 Position/EOS 候选集上的分布为 `p(v|h)` 和 `q(v|h)`，且两边使用同一位置约束、温度与 top-p 处理。若 Draft 从 `q` 抽取候选 `y`，按

\[
\Pr(\text{接受 }y\mid h,y)=\min\left(1,\frac{p(y\mid h)}{q(y\mid h)}\right)
\]

接受，则该固定上下文下的一步期望接受率为

\[
\alpha_P(h)=\sum_v\min\bigl(p(v\mid h),q(v\mid h)\bigr)
=1-\operatorname{TV}(p,q),\qquad
\operatorname{TV}(p,q)=\frac12\sum_v|p(v\mid h)-q(v\mid h)|.
\]

拒绝时按未覆盖的 Target 概率质量抽样：

\[
r(v\mid h)=\frac{[p(v\mid h)-q(v\mid h)]_+}
{\sum_u[p(u\mid h)-q(u\mid h)]_+}.
\]

接受候选与拒绝后的残差抽样合并后，输出仍服从 `p(·|h)`。当 `p=q` 时拒绝概率为零，残差式不需计算。这是固定历史下的一步分布结论，不直接给出长轨迹速度或生成质量。PF 中的 Feature 提案依赖已到达的 Position 条件，不能把 P、F 的汇总接受率直接相乘来声称成对接受概率。

**净速度由工作量和时间共同决定。** 可用“每轮平均提交 token 数 / 每轮平均耗时”描述投机轮次的吞吐；Draft 前向、Target 验证、拒绝后采样、缓存处理与调度都影响分母。接受率只影响接受路径与提交量，单独不能决定速度。本项目目前没有一套由逐轮路径和各项成本计算、并经独立数据校准的理论速度预测；下表是实测结果，不能从上面的单步公式推导出来。

| 观测 | 接受率 | 实测速度比 | 解释范围 |
| --- | ---: | ---: | --- |
| 阶段 E 轻量 PF | F 33.2582% | 来源组几何平均 1.71250× | 30 个场景的固定窗口 FP32 比较 |
| 阶段 E PF-D8 | F 55.4712% | 来源组几何平均 1.46646× | 同轮多场景对照；较高 F 接受率未对应更高总速度 |
| 阶段 F 轻量 PF | P/F 91.720%/23.256%（逐样本比率均值） | PF/Target 1.569×；PF/P-only 1.070× [0.912, 1.252] | 完整单 chunk 输出；PF 相对 P-only 的区间跨过 1 |

阶段 F 的 360 份样本还记录了端到端时间构成。下表每格为“组内平均秒数 / 该阶段总耗时占比”；占比按该阶段秒数总和除以 `total_seconds` 总和计算。真实补全各模式 `n=90`，BOS 各模式 `n=30`。

| 样本组 | 模式 | AR 生成 | VQ 解码 | Gaussian 导出 | 渲染 | PNG 写入 | 完整时间均值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实补全 | Target | 45.1636 / 98.038% | 0.2040 / 0.443% | 0.5856 / 1.271% | 0.0025 / 0.005% | 0.1117 / 0.243% | 46.0674 秒 |
| 真实补全 | P-only | 30.4898 / 97.197% | 0.1975 / 0.630% | 0.5676 / 1.809% | 0.0024 / 0.008% | 0.1118 / 0.357% | 31.3692 秒 |
| 真实补全 | 轻量 PF | 28.6040 / 96.896% | 0.2052 / 0.695% | 0.5949 / 2.015% | 0.0024 / 0.008% | 0.1139 / 0.386% | 29.5203 秒 |
| BOS | Target | 69.8616 / 98.396% | 0.2594 / 0.365% | 0.7360 / 1.037% | 0.0039 / 0.005% | 0.1394 / 0.196% | 71.0003 秒 |
| BOS | P-only | 47.3974 / 97.631% | 0.2587 / 0.533% | 0.7435 / 1.532% | 0.0028 / 0.006% | 0.1450 / 0.299% | 48.5475 秒 |
| BOS | 轻量 PF | 39.2434 / 97.301% | 0.2440 / 0.605% | 0.7007 / 1.737% | 0.0028 / 0.007% | 0.1412 / 0.350% | 40.3320 秒 |

AR 占各组总时间的 96.9%–98.4%；这批设置下主要等待来自 token 生成。阶段总耗时中其余环节合计约 1.6%–3.1%。逐样本五项时间之和与 `total_seconds` 一致，最大差异约 `1.3e-14` 秒，属于浮点末位。CSV 的轻量 PF 模式原标签是 `step12000`，依据对应 Feature 接受率列完整且与正式权重记录相符进行归组。

另一个单场景、四裁剪的 D8 对照显示其 Feature 接受率为 53.2552%，轻量 MLP 为 20.2967%；D8 将 Target 调用从 9,288 次降至 7,639 次，但实测速度比分别为 1.3561× 和 1.4483×。这与多场景观察方向一致，且提示额外草稿计算会抵消部分验证收益；单场景数据只作机制说明。真实来源适配 D2 也呈现类似区分：固定前缀重合度提高，但 B/A 在线速度比 1.00688× [0.99368, 1.02163] 的区间跨过 1。D2 的重合度指标未在结果摘要中给出逐位置定义，故不称为理论接受率。

**证据来源。** [方法中的 Position 接受与残差公式](DRAFT_METHODS_HISTORY.md)、[阶段 E/F 与 D2 记录](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)、[D8 真实输入对照](FEATURE_D8_REAL_EVAL_2026-10-14.md)、本地[全长汇总](artifacts/stage_f_full3d/analysis/summary.json)、[逐次耗时表](artifacts/stage_f_full3d/analysis/test_samples.csv)与[BOS 耗时表](artifacts/stage_f_full3d/analysis/bos_samples.csv)。

## Target 解码成本动机

**2026-10-16 Profiling 资料补充。** 远端 `profiling/raw/p1_vfront_final.csv` 的五次原始运行现已读取并保存。RTX 4090、FP32、SDPA、B=1、自然 EOS、两次完整预热后，AR 中位时间为 66.02357 秒，Core 为 66.31129 秒；逐运行 AR/Core 占比中位数为 99.5653586%。Core 包含 Prefill、AR、token 后处理与 VQ 解码，不含输入压缩、模型加载、KV 分配和渲染。该结果解决了 10 月 8 日材料中 99.57% 原始来源未定位的问题。

独立的 `profiling/ar_breakdown/ar_phase_window_ctx2048_n128_r3/summary.json` 给出 Target forward 占 2048 后连续 128-token 窗口的 96.682793%，三个重复、seed=2401。这与 P1 采用不同范围，不相乘推导完整流程 forward 占比。P3 原始分段表还显示局部延迟从上下文 513 的 13.47747 ms/token 增至 8192 的 30.54179 ms/token；P7 局部导出表的横坐标均为 8192，绘图采用 P3 的 `context_length_end`。

完整表、图、原始附件与复算脚本见 Obsidian [GaussianGPT Profiling 数据汇报](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/GaussianGPT Profiling 数据汇报.md>)。本次复算覆盖 P1、P3/P4/P5 汇总、Adapter 阶段成本、阶段 E 摘要及阶段 F 全部 360 条样本；未运行新的性能实验。

**主张。** GaussianGPT 的 Position 与 Feature 交替生成；在 `P_i → F_i → P_(i+1)` 条件链中，后续 Position 需要此前 Feature。Target 自回归解码的实际等待成本足以支持研究低成本提案，但目前的分项计时只能解释部分模型调用成本。

**证据与口径。** 阶段 F 的 90 份 Target 真实补全平均耗时 46.07 秒，中位数 51.60 秒，平均新生成 3,050 token；AR 吞吐 67.56 token/s，含解码、导出与图像写入的全流程吞吐 66.23 token/s。另有早期 RTX 4090 单 seed 分项记录：1024/2048/4096 长度下，Target 双 token 验证调用为 16.84/16.76/17.66 ms，D8 双 Position 更新为 6.50/6.48/6.67 ms。该分项记录计算得到的 1.13–1.16 倍是模型项估算，不是完整生成实测。

**来源。** [架构说明](ARCHITECTURE.md)；[实验与 profiling 记录](EXPERIMENTS.md) 的 fine-grained profiling 段；远端 `/root/workspace/GaussianGPT/profiling/route_a/stage_detail_seed5_1024.json`、`stage_detail_seed5_2048.json`、`stage_detail_seed5_4096.json`；阶段 F 本地 [summary.json](artifacts/stage_f_full3d/analysis/summary.json) 与远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/analysis/summary.json`。

**可用表述。** “在所测 RTX 4090 单 chunk 设置中，Target 全长补全平均等待 46.07 秒；开发期分项数据也显示 Target 验证调用具有显著成本，支持研究提案解码的端到端收益。”

**边界。** 分项结果来自一个 seed、每个长度一次运行；单次事件统计不是重复运行之间的波动范围，且未覆盖全部 Python 调度。全长时间只对应本次 30 场景、256-token 前缀、FP32 和单个 20×20×20 latent chunk；不外推到多 chunk 或其他硬件。

## P/F 行为差异

**主张。** Position 表示稀疏空间位置，Feature 是该位置的 4096 类离散特征码。二者处于不同条件位置；当前 P-only 只提议 Position、由 Target 采样 Feature，而 PF 路线另行提议 Feature。因此二类接受率和对应计算成本必须分开报告。

**证据与口径。** 阶段 E 为五模式、FP32、64/256/1024-token 前缀及最多 512-token 续写。合并在线计数后的 P/F 接受率如下：

| 模式 | Position，已接受/已核验 | Feature，已接受/已核验 | 相对 Target 的速度比，来源组几何平均 |
| --- | ---: | ---: | ---: |
| P-only | 92.4408% | 不提议 Feature | 1.43773× |
| 轻量 PF | 92.6410% | 33.2582% | 1.71250× |
| PF-D8 | 92.5066% | 55.4712% | 1.46646× |
| GPT-D8 | 78.9677% | 55.5221% | 0.93815× |

阶段 F 的 90 份轻量 PF 全长样本中，Position 已核验接受率的逐样本均值为 91.720%，Feature 为 23.256%；另按提议数作分母时分别为 55.698% 和 21.329%。这是 90 个样本接受率的算术平均，不是按 token 合并的总接受率；阶段 E 的分母与生成窗口不同。

**来源。** [多场景执行记录](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)；远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/online.json`、`multiscene_summary.json`；[完整三维输出记录](FULL_3D_OUTPUT_2026-10-15.md) 与本地 [test_samples.csv](artifacts/stage_f_full3d/analysis/test_samples.csv)。

**可用表述。** “在阶段 E 的固定窗口测试中，Position 接受率约为 92.4%–92.6%，轻量 PF 的 Feature 接受率为 33.26%；阶段 F 的全长轻量 PF 样本均值分别为 91.72% 和 23.26%。”

**边界。** 阶段 E 是计数合并率；阶段 F 是逐样本比率均值，且完整汇总行未保留每类 token 的原始分母。不能把两个阶段的 Feature 数字直接比较，也不能由接受率单独推出画质。

## 阶段 E：正式多场景解码

**主张。** 轻量 PF 在本轮 30 个独立测试场景的固定窗口 FP32 解码中高于 Target；微平均和来源组等权结果方向一致。

**测试规模与设置。** 30 个场景、56 条唯一裁剪、63,328 个正文 token；64/256 前缀各 56 条，1024 前缀 32 条；三个 seed、五种模式，共 2,160/2,160 次评价。模式为 Target、P-only、轻量 PF、PF-D8、GPT-D8；temperature/top-p=0.9。短裁剪只参加可用的前缀档，不复制样本补齐。

| 模式 | 微平均速度比 | 来源组几何平均 | 来源组几何平均 95% 区间 |
| --- | ---: | ---: | ---: |
| Target | 1.00000× | 1.00000× | — |
| P-only | 1.43491× | 1.43773× | 此处未列区间 |
| 轻量 PF | 1.69675× | 1.71250× | [1.68044, 1.74402] |
| PF-D8 | 1.45756× | 1.46646× | 此处未列区间 |
| GPT-D8 | 0.94020× | 0.93815× | 此处未列区间 |

轻量 PF 在 30/30 个来源组快于 Target；GPT-D8 在 24/30 个来源组慢于 Target。1.69675× 是池化 token/时间比，1.71250× 是来源组等权几何平均，不可互换。区间仅随来源组重采样，不附加到微平均数值。

**来源。** [多场景执行记录](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md) 的阶段 E 完成段；[实验记录](EXPERIMENTS.md) 顶部阶段 E 汇总；远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/pipeline.json`、`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/online.json`、`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/multiscene_summary.json`。

**可用表述。** “轻量 PF 在 30 个本轮保留场景上的固定窗口 FP32 解码，微平均速度比为 1.69675×，来源组几何平均为 1.71250×（95% 区间 [1.68044, 1.74402]）；30/30 个来源组快于 Target。”

**边界。** 这是最长 512-token 续写的 token 解码比较，不是完整场景制作、解码和渲染时间。BF16 分块与逐 token 分布差异未达到本计划采用的 0.01 总变差容差，故没有正式 BF16 速度结论。独立性仅按本轮训练和选择划分定义，历史模型来源另行说明。

## 高 Feature 接受率不保证更快

**主张。** Feature 接受率较高本身不保证提案路线有较高总吞吐；提案计算、Target 调用、前缀处理及运行实现成本同样影响结果。

**正式多场景对照。** 阶段 E 的 PF-D8 Feature 接受率为 55.4712%，来源组速度比 1.46646×；轻量 PF 的 Feature 接受率为 33.2582%，速度比为 1.71250×。这是 30 场景同轮对照，显示高 Feature 接受率没有转化为更高的总速度。

**专项对照。** 一个真实场景、四个不同裁剪的 D8 Feature 评价中，D8 Feature + Position ranker 的 Feature 接受率为 53.2552%、速度比 1.3561×；同轮轻量 MLP 为 20.2967%、1.4483×。D8 Gamma 2/4/6 的 Feature 接受率为 54.3726%/53.6912%/53.6286%，速度比为 0.9696×/0.9127×/0.8141×；增加提案长度未改善总速度。三项都是单真实场景控制，不能替代阶段 E 结论。

| 专项控制 | 接受或速度结果 | 来源与限制 |
| --- | --- | --- |
| D4 联合 P/F | Gamma 2/4/6 速度比 0.6328×/0.4918×/0.4006×；Gamma 2 的 P/F 接受率 19.2623%/4.4105% | [D4 训练与评价](D4_JOINT_TRAINING_2026-10-14.md)；一个真实场景；D4 随机初始化、D8 使用 38 epoch 预训练，不能单独归因于层数 |
| D8 Feature 接法 | Feature 接受率高于轻量模型，但速度比 1.3561× 对 1.4483× | [D8 真实输入评价](FEATURE_D8_REAL_EVAL_2026-10-14.md)；一个场景、四裁剪、144 次评价 |
| D8 Gamma 2/4/6 | Feature 接受变化小，速度比随 Gamma 增大而下降 | [D8 Gamma 对照](FEATURE_D8_GAMMA_2026-10-14.md)；一个场景、四裁剪、144 次评价 |

**轻量 Feature 训练对照。** 直接 Target 训练 step3600 的 Feature 接受率为 32.9079%，旧 MLP 为 29.5130%；新旧速度比为 1.01183×，95% 区间 [0.98589, 1.03544]。续训至 step12000 后 Feature 接受率为 32.1659%，step3600 为 32.9079%；新旧速度比 0.99762×，区间 [0.97596, 1.01747]。训练集分布指标略有提高，但没有可确认的额外在线速度收益。

**来源。** [轻量 Feature 直接 Target 训练](FEATURE_TARGET_V2_2026-10-14.md)、[轻量 Feature 继续训练](FEATURE_TARGET_CONTINUE_2026-10-14.md)；两份记录所列 JSON 均在相应远端运行目录。

**可用表述。** “在本项目已测配置中，Feature 接受率与端到端速度没有单调对应关系；提案成本和整条解码路径决定净收益。”

**边界。** 该结论是这些已测实现与输入范围内的经验观察，不能泛化为所有模型或硬件的普遍规律。

## 阶段 D1：动态选择的结论范围

**主张。** 开发集场景/前缀分组未显示优于静态轻量 PF 的动作差异，因此本轮不启用动态 P-only/K2 选择。

**证据与口径。** 十个开发场景的分析中，29/29 个可评估的场景×前缀组合由轻量 PF 胜出；162 个裁剪/seed 配对里，PF 有 149 次最快，P-only 有 8 次、PF-D8 有 5 次。分组使用场景和前缀长度；分析没有记录逐轮 Position 置信度及历史动作收益。

**来源。** 远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/action_analysis.json`、`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/scheduler_selection.json`；[多场景执行记录](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md) 的阶段 C 与 D1 段。

**可用表述。** “在本次开发集按场景与前缀长度分组的分析范围内，未发现动态动作胜过静态轻量 PF 的证据，故正式比较沿用静态模式。”

**边界。** 这不是对所有上下文调度策略的否定；未观测的逐轮置信度、动作切换成本与其他场景分布仍不在这项分析范围内。

## 阶段 D2：真实来源适配

**主张。** 首对真实来源适配训练提高了固定前缀上的 Feature 分布重合度，但没有显示稳定的在线速度增益；这支持停止追加训练，不支持“分布匹配提高必然加速”的表述。

**证据与口径。** A 组仅用旧数据，B 组按旧/真实来源 70/30 混合；两组都从 step12000 继续 3,600 步，并在十个开发来源组完成 810/810 次配对评价。旧/A/B 的来源组等权速度比分别为 1.6869×/1.6782×/1.6897×。B/旧为 1.00168×，95% 区间 [0.99219, 1.01034]；B/A 为 1.00688×，[0.99368, 1.02163]，区间均跨过 1。固定前缀重合度旧/A/B 汇总值为 25.2181%/25.2824%/26.8020%；另按来源组内位置数加权、来源组间等权计算，B 相对 A 增加 1.5151 个百分点，95% 区间 [1.2440, 1.8037]，十个来源组均提高。

**来源。** [多场景研究计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md) 的阶段 D2 记录；远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_d2_adaptation/stage_c/dev_pair_seed42001/online.json`、`multiscene_summary.json`、`adaptation_comparison.json` 与 `development_decision.json`。

**可用表述。** “在本轮十个开发来源组中，混合来源适配改善了固定前缀 Feature 重合度，但速度比的区间仍覆盖 1，因此未确认额外在线加速，也未追加训练 seed。”

**边界。** 这是首对训练的开发集结果，不是阶段 E 的独立测试结果；固定前缀分布重合度与在线完整生成时间是不同测量，不互相替代。

## 阶段 F：360 份全长输出

**主张。** 三种模式均完成真实场景全长生成及 Gaussian/多视角输出；真实补全中 PF 相对 Target 的端到端等待较短。PF 相对 P-only 的增量仍不确定。

**规模与条件。** 真实前缀补全为 30 场景×3 seed×3 模式，共 270 份；BOS 为 30 seed×3 模式，共 90 份；合计 360 份。真实前缀 256 token、正文上限 16,000、FP32、temperature/top-p=0.9、单个 20×20×20 latent chunk。360 份均自然 EOS，失败数与截断数均为零；每份均导出 Gaussian NPZ 与六张 384×384 PNG。

| 模式 | 平均完整流程时间 | 中位数 | 平均新生成 token | AR 吞吐 | 全流程吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target | 46.07 秒 | 51.60 秒 | 3,050 | 67.56 token/s | 66.23 token/s |
| P-only | 31.37 秒 | 32.13 秒 | 2,987 | 98.01 token/s | 95.26 token/s |
| 轻量 PF | 29.52 秒 | 31.08 秒 | 3,087 | 107.94 token/s | 104.59 token/s |

| 对照 | 场景等权完整时间比 | 95% 场景重采样区间 | 快于参考的场景 | 全样本总时间比 |
| --- | ---: | ---: | ---: | ---: |
| P-only / Target | 1.467× | [1.266, 1.731] | 25/30 | 1.469× |
| 轻量 PF / Target | 1.569× | [1.318, 1.852] | 27/30 | 1.561× |
| 轻量 PF / P-only | 1.070× | [0.912, 1.252] | 18/30 | 1.063× |

“完整时间比”定义为参考模式耗时/提议模式耗时，数值大于 1 表示提议模式完成该样本更快；它不按输出 token 数归一化。各模式输出长度并不完全相等，故同时给出新生成 token 数、AR 吞吐和全流程吞吐。全流程时间不含模型加载、预热、参考真实 chunk 的解码和渲染。

90 份 PF 样本的 P/F 已核验接受率逐样本均值为 91.720%/23.256%；按提议数作分母为 55.698%/21.329%。此处仍是接受统计，不是质量指标。

**BOS 补充。** 90 份 BOS 输出均自然 EOS。PF/Target 完整时间比为 1.791× [1.588, 2.023]，PF/P-only 为 1.220× [1.094, 1.365]。BOS 是无条件轨迹，不替代真实场景主结果。

**来源。** 本地 [阶段 F 汇总 JSON](artifacts/stage_f_full3d/analysis/summary.json)、[逐样本 CSV](artifacts/stage_f_full3d/analysis/test_samples.csv)、[分场景 CSV](artifacts/stage_f_full3d/analysis/test_scenes.csv)、[BOS CSV](artifacts/stage_f_full3d/analysis/bos_samples.csv)、[流程状态 JSON](artifacts/stage_f_full3d/pipeline.json)；远端对应目录 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/analysis/`。本地视频：[真实场景六视角并排视频](artifacts/stage_f_full3d/analysis/test_completion_30scenes_seed51701_6views.mp4)、[BOS 六视角并排视频](artifacts/stage_f_full3d/analysis/bos_30trajectories_6views.mp4)。说明页：[完整三维输出记录](FULL_3D_OUTPUT_2026-10-15.md)。

**可用表述。** “在 30 个场景的完整自然结束补全中，轻量 PF/Target 场景等权完整时间比为 1.569×，95% 区间 [1.318, 1.852]；轻量 PF/P-only 为 1.070× [0.912, 1.252]，该区间跨过 1，额外收益尚不确定。”

**边界。** 端到端结果仅覆盖单 chunk 与指定设置；等待时间比较允许各方法按自然 EOS 得到不同长度。阶段 E 固定窗口速度不得替代此处的完整时间比。

阶段 F 的分项时间和 AR 占比见上文[理论接受率、实测接受率与净速度](#理论接受率实测接受率与净速度)。这里的分项解释整体速度来自哪里，不改变场景配对时间比的统计方式。

## 三维质量现状

**已测量。** 270 份真实补全输出均成功、自然 EOS，未触及 16,000-token 上限；每种模式各 90 份。全 chunk 对一份真实 VQ 重建的全图 PSNR/SSIM 均值：Target 11.41 dB/0.7302，P-only 11.67 dB/0.7383，轻量 PF 11.46 dB/0.7321。轻量 PF 的 PSNR 范围为 6.94–20.12 dB，SSIM 为 0.606–0.942，显示样本差异明显。平均 Gaussian 数分别为 218,986、212,133、222,091；尺度上限裁切比例为 8.74%、8.84%、8.63%，平均不透明度为 0.343/0.345/0.350，alpha 覆盖率为 31.93%/31.15%/32.25%。这些数值用于描述本批输出与参考重建的差异。

**解释边界。** 全图指标包含给定前缀与随机生成区域；随机生成后缀存在多种可能结果。因此 PSNR/SSIM 接近不能作为生成质量等效的证据，也不能由此推出任一方法更优。

**资料可用性与未测项目。** 远端已有 30 个场景参考目录，每个目录含六视角 PNG 与 `metrics.json`；270 份真实补全输出各含六视角 PNG、Gaussian NPZ 和 `tokens.pt`。`gaussiangpt` 环境可导入 `lpips` 与 `torchmetrics`，但当前没有 LPIPS 数值，也未保存可见区域遮罩。全视图 LPIPS 会混合已知前缀和随机续写，不能单独解释为生成质量；还需先明确可见区域的遮罩与比较方式。多 chunk 连续性、接缝和完整房间延展仍没有结果。

**远端 Gaussian 数值审阅。** 对真实补全 `test_completion/samples/` 的 270 份与 BOS `bos_generation/samples/` 的 90 份 `gaussians.npz` 逐个只读载入。所有文件均可打开，`means`、`scales`、`opacities`、`quats`、`colors` 五类数组均未发现 NaN 或无穷值。四元数长度相对 1 的最大绝对误差为 (1.8\times10^{-7})；尺度在保存范围 (10^{-4})–(0.05)，不透明度与颜色位于 [0,1]。这说明已保存数组在数值上可用，不代表几何或渲染质量已达到目标。

| 样本组 | 模式 | NPZ 数 | Gaussian 均值 | 尺度裁剪比例 | Gaussian 中心坐标全局范围，x/y/z |
| --- | --- | ---: | ---: | ---: | --- |
| 真实补全 | Target | 90 | 218,986 | 8.74% | 最小 [-0.179, -0.206, -0.149]；最大 [4.320, 4.220, 4.023] |
| 真实补全 | P-only | 90 | 212,133 | 8.84% | 最小 [-0.180, -0.176, -0.104]；最大 [4.293, 4.248, 3.900] |
| 真实补全 | 轻量 PF | 90 | 222,091 | 8.63% | 最小 [-0.153, -0.183, -0.031]；最大 [4.292, 4.238, 3.919] |
| BOS | Target | 30 | 278,141 | 8.85% | 最小 [-0.151, -0.162, -0.101]；最大 [4.285, 4.251, 3.722] |
| BOS | P-only | 30 | 280,097 | 8.35% | 最小 [-0.125, -0.135, -0.058]；最大 [4.273, 4.240, 3.654] |
| BOS | 轻量 PF | 30 | 262,640 | 9.28% | 最小 [-0.156, -0.135, -0.021]；最大 [4.268, 4.234, 4.396] |

输出记录将坐标描述为边长 4 米的局部 chunk，渲染相机以边长的一半作为观察中心。默认 VQ 配置的体素边长为 0.025 米、latent stride 为 8；解码位置为 `(latent_index × stride + coord_offset) × grid_size`。训练偏移由连续坐标减去四舍五入后的体素坐标得到，目标通常位于约 `[-0.5,0.5]` 个体素；推理端的 `coord_offset` 使用恒等映射，未施加相同范围限制。因此 `[0,4]` 是名义网格范围，不是 Gaussian 中心硬边界；解码实现允许越界，但这不证明每个越界值都符合训练分布。远端 30 份同场景参考重建中有 25 份中心坐标超出 `[0,4]`；参考中心全局最小值 `[-0.1846, -0.1550, 0.0159]` 米、最大值 `[4.1703, 4.1810, 3.3373]` 米。生成输出的中心包围范围均在至少一个轴向超出对应参考样本；随机续写会改变空间范围，这项对照不能单独作为质量判据。尺度裁切比例来自生成时裁切前的质量记录；NPZ 中记录的是裁切后的尺度。

两张本地预览分别展示真实场景首个场景 seed 51701 与 BOS 首条轨迹 seed 51801；单图只供样本观察，不能据此估计整组出现频率或判断整体优劣。真实补全的 90 个场景×seed 组均有三个模式；读取全部 270 个 `tokens.pt` 后确认每组的前 256 个 token 完全一致。此结果确认条件输入相同，不是图像质量结果。本地交接副本没有逐样本 Gaussian NPZ 和独立 PNG；远端对应路径为 `stage_f_full3d/test_completion/samples/` 与 `stage_f_full3d/test_completion/references/`。参考和生成视图已存在，但没有可见区域遮罩；完整图像 LPIPS 会受随机续写影响，本轮未计算。BOS 没有成对的场景参考。

**来源。** 本地[阶段 F 汇总 JSON](artifacts/stage_f_full3d/analysis/summary.json)、[逐样本 CSV](artifacts/stage_f_full3d/analysis/test_samples.csv)、[真实场景预览](artifacts/stage_f_full3d/analysis/test_completion_30scenes_seed51701_6views.preview.png)、[BOS 预览](artifacts/stage_f_full3d/analysis/bos_30trajectories_6views.preview.png)、[完整三维输出记录](FULL_3D_OUTPUT_2026-10-15.md)；远端原始数组位于 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/test_completion/samples/` 与 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/bos_generation/samples/`。

## 产物定位

阶段 E 原始正式 JSON 以远端目录为准：

- `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/pipeline.json`
- `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/online.json`
- `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/multiscene_summary.json`

本地说明页：[P/F 多场景研究执行记录](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。阶段 E 原始逐次结果在本地交接副本中没有同名 JSON 文件。

阶段 F 的 JSON、CSV、视频在本地 `docs/project/artifacts/stage_f_full3d/`，并有远端副本 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/`。列表见上方“来源”项；本地说明页：[完整三维输出记录](FULL_3D_OUTPUT_2026-10-15.md)。
