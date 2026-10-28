# PFSpec 面向 CVPR 的正文重组说明

更新日期：2026-10-27。当前正文为 [PAPER_MAIN.md](PAPER_MAIN.md)。本轮依据用户确认的主张顺序，直接改写该正文；此前“摘要保持原文”的历史安排由本轮统一改写授权取代。

## 1. 完成的修改与贡献定位

正文现按 Abstract → Introduction → Related Work → Method → 机制分析 → Experiments → Discussion → 附录组织。摘要、引言、相关工作与两项贡献已经完整重写；方法开头和实验顺序使用同一主线。详细轮次推导保留于附录 A，其他批次、宽备用与动态选择结果保留于附录 B。项目执行过程和写作安排不再混入论文主文。

**核心主张：利用 GaussianGPT 的 Position–Feature 条件结构，以轻量类型化提案减少主路径串行调用，并以单备用条件状态预计算减少固定参考流程中的部分恢复工作。**

| 内容 | 当前定位 | 已有证据与边界 |
|---|---|---|
| Position Ranker + 条件 Feature MLP | 主要方法贡献：具体 P/F 低成本提案与验证安排 | Position-only、Ranker 对照、状态输入消融和完整速度支持其实际价值；不主张 MLP 架构普遍领先 |
| Position Rescue | 第二项具体机制贡献，与主要提案设计有主次关系 | 固定 W1 下的状态复用、轨迹前提及约 1.8% 重复解码增量；完整请求新增收益仍不稳定 |
| CUDA Graph 与共享执行实现 | 实现贡献和性能归因组成 | 3.240487× 为整个共享优化 Target 相对原实现；缺少只改变 CUDA Graph 的独立对照，不能全部归给单项技术 |
| 接受、推进、恢复概率和成本分析 | 支持机制理解与正确性 | 标准投机采样及有限请求恒等关系有既有基础；局部成本分解不等于新的通用最优理论 |

两项贡献的正式文本位于 Introduction 末尾。类型化提案与恢复预计算共同利用 P/F 条件链，但当前没有独立证据证明二者具有超出各自增量的协同效应，因此正文不声称这类额外收益。

## 2. 源码核对带来的实质性调整

**Position 提案为确定性点提案。** Ranker 选取最高分合法候选，评分不作为归一化提案概率；备用按同一评分及同分次序取下一个候选。因此主接受概率为 Target 的主候选质量，非主点备用的“拒绝且命中”联合概率为其 Target 质量。正文给出明确公式，避免将排序分数解释成残差概率。

**Feature MLP 预测隐藏状态残差。** 已有模型读取前一 Position 的 pre-finalnorm Target 状态、已生成 Feature 和候选 Position，预测状态残差，再复用 Target 归一化层及 Feature 输出头生成分布。Related Work 因而明确承认与 EAGLE 隐藏状态预测的联系，不能仅以“离散 Feature 与连续 feature 不同”划分创新。

**恢复依赖对应固定 W1 轮次。** W1 在恢复实际 Position 后使用 Target 分布生成 Feature，并保存状态供下一轮使用。Rescue 命中分支复用对应计算。正文不把这一串行安排概括为所有投机解码算法都必须支付的独立前向。

**树分支与随机流前提分别列出。** 单备用父索引为 [-1, 0, 1, 2, 0]，命中提交 [0, 4]；备用只覆盖第一 Position。轨迹保持要求主分支和备用条件状态等价、正确的逻辑位置和注意力、真实路径提交、相同 Feature policy 与采样算法、相同随机数消费顺序及图输出缓冲有效期。已测一致性与数学前提分别说明。

主要源码入口：

- [恢复解码器](../../../../patches/spec_next_20261022/position_rescue_decoder.py)
- [备用树运行时](../../../../patches/spec_next_20261022/position_rescue_runtime.py)
- [共享动态 Target 图](../../../../patches/feature_tree_20261021/reference/runtime/dynamic_target_graph.py)
- [共同执行比较入口](../../../../patches/pf_comparison_20261025/compare.py)
- [同预算 Feature 模型](../../../../patches/round2_20261017/b1_feature_models.py)

## 3. 已有实验足以回答什么

最新资料已经包含 480 次共同后端主比较、360 次结构与输入消融，以及 960 次 Rescue 重复解码。本轮没有将这些已完成实验重新列为待执行任务。

| 已有证据 | 正文采用的结论 |
|---|---|
| Position-only/共享优化 Target 1.395117×；W1/共享优化 Target 1.551487× | 在位置投机上增加 Feature 路径具有整体增量，不能全部归给单个预测器 |
| 移除 Target 状态/同预算完整模型 0.893863×，区间不含 1 | Target 状态复用具有较明确的速度证据 |
| 移除候选 Position 1.018197×；移除历史 Feature 1.004918×，区间均含 1 | 两个输入的独立速度贡献尚不明确 |
| 同预算单步 Transformer/残差 MLP 三种训练种子为 1.064218×、1.046379×、1.016082× | 当前不支持 MLP 普遍更优；一个种子的区间不含 1，不能忽略这一不利证据 |
| W2 推进 +4.37%、完整每轮成本 +8.68%、吞吐 0.960329× | 已测扩宽未获净收益；范围限三个开发来源 |
| 四轮 Rescue/W1 1.018092×，来源区间 [1.014818, 1.021327] | 小幅解码增量具有重复证据，不能扩大成同幅完整请求收益 |

共同执行的 Medusa、Hydra 和 EAGLE 结果只评价当前适配版本。训练、监督、输入及参数量尚不完全相同；同预算单步 Transformer 对照也不等同于完整 EAGLE。正文保留这两种比较的用途差异。

## 4. 本轮从原始记录复算的结果

复算使用本机 Python 读取 JSONL，按来源等权的配对对数比汇总；没有运行模型、GPU、远端命令或新增训练。下表区间沿用既有统计报告，本轮没有重新抽样生成区间。

| 原始记录 | 本轮复算结果 |
|---|---|
| [240 次归因解码](../spec_next_20261022/raw/paper_attribution_run01/records.jsonl) | 优化 Target/原 Target 3.240487015；W1/优化 Target 1.557471234；Rescue/W1 1.016787710；PFSpec/优化 Target 1.583617609；PFSpec/原 Target 5.131692298 |
| [480 次主比较](../pf_comparison_20261025/main_run01/records.jsonl) | PFSpec/优化 Target 1.575052887；W1/优化 Target 1.551486609；W1/Rescue 的 60 对输出及语义接受计数一致；恢复 1,080→723，差额逐对等于命中数，合计 357 |
| [360 次结构比较](../pf_comparison_20261025/mechanism_run01/records.jsonl) | 输入消融与三种训练种子的 Transformer/MLP 点值均与报告相符 |
| 四轮重复：[01](../rescue_repeat_20261022/raw/repeat_01/records.jsonl)、[02](../rescue_repeat_20261022/raw/repeat_02/records.jsonl)、[03](../rescue_repeat_20261022/raw/repeat_03/records.jsonl)、[04](../rescue_repeat_20261022/raw/repeat_04/records.jsonl) | 960 条记录；Rescue/W1 1.018091922；480 对输出及语义接受计数一致；恢复 8,400→5,492，差额逐对等于命中，合计 2,908 |
| [60 次完整请求](../w1_e2e_20261021/rescue_paper_run01/records.jsonl) | PFSpec/原 Target 请求加速比 4.307861721；Rescue/W1 1.001847901；W1 与 Rescue 的 20 对输出相同；原 Target/W1/Rescue 合计 token 为 10,240/9,837/9,837 |

1.58×、5.13×、约 1.8% 和 4.31× 的分母及批次明确分开。摘要保留 1.58×、约 1.8% 和 4.31×，移出 5.13× 与置信区间；完整归因和全部主要区间放入实验。4.31× 对应从已准备前缀开始、单 Chunk、512-token 上限的完整实现请求，包含执行优化及实际输出长度差异。

## 5. 先行工作与仍存在的科学风险

| 最近的先行工作 | 对本文定位的影响 |
|---|---|
| Medusa、Hydra、EAGLE | 多头起草、候选条件化和隐藏状态预测均有先例；本文需要以 P/F 的具体计算安排及实测说明价值 |
| SpecInfer、SpecTr | 树验证和选中已计算候选后的条件状态复用已存在；Rescue 的贡献范围限定于固定 W1 的恢复机制、条件与证据 |
| Sequoia、OPT-Tree、Learning to Draft | 成本选择、树构造和吞吐优化已有直接研究；本文的成本分解承担解释作用 |
| SpecBranch、Speculative Speculative Decoding | 拒绝分支并行和验证期间预备计算已有相关设计；正文明确 Rescue 预计算的是备用位置的 Target 条件状态 |
| FlashMesh、XSpecMesh | 三维生成投机加速已有近邻工作；Gaussian P/F 的条件结构与保持目标采样规则需要说清，不能仅凭应用对象不同主张广泛方法创新 |

文献链接随正文对应段落保留。2026 年文献的公开身份此前已按官方页面核对：FlashMesh 为 CVPR 2026，XSpecMesh 为 ICML 2026，Learning to Draft、SpecBranch、Speculative Speculative Decoding 为 ICLR 2026。FlashMesh 的具体描述依据公开预印本；XSpecMesh 使用官方出版页面支持的多头、验证与重采样概述，未因出版 PDF 获取受限而补写其接受规则或分布证明结论。

本次改写能减少贡献错位、数字混用和防御性叙述，不能增加方法本身的新颖性。CVPR 竞争力仍主要取决于审稿人是否认可：类型化计算安排在 3D Gaussian 自回归生成中提供了足够具体、有效且可解释的方法价值。单 Target、单设备、有上限请求，以及适配对照的训练差异仍构成实际限制；不将这些限制通过措辞掩盖。

## 6. 修改对象与未改动范围

实际修改：本目录 PAPER_MAIN.md；新增本说明；更新项目 .codex/AI_START_HERE.md 的最新入口。运行源码、模型权重、原始实验、既有统计报告和生成状态快照均未改动。本轮未重新运行性能实验，未生成新的三维产物。

## 7. 本轮评审反馈后的叙事收紧

依据用户补充的评审意见，进一步调整 Abstract、Introduction、两项贡献及 Related Work 第 2.3 节。主线收紧为：合法位置候选与已有 Target 状态提供低成本提案机会 → P/F 短路径产生有效推进 → 分析该流程剩余的位置恢复开销。引言加入 FlashMesh、XSpecMesh 的直接三维先行工作；组件介绍置于观察之后，删除“主要方法贡献”“边界明确的补充机制”等评价自身地位的句子。摘要保留 1.58× 和 4.31× 两项主要结果，Rescue 的约 1.8% 移至引言与实验，原始结果均保留。

对反馈中的两项判断作了证据区分。第一，各 Feature 输入的独立速度证据不足，不等于类型化短路径无效；当前主张明确为合法位置排序与 Target 状态复用的具体计算安排。第二，2026-10-27 直接读取 SpecTr 的 NeurIPS 出版 PDF，第 4 节、第 5 页明确讨论：选中的 token 属于候选集合时，可使用已计算的下一条件分布继续抽样而无需再次调用 Target。因此保留该直接先例，同时避免将其归为隐藏状态与 KV 的具体恢复机制。依据见 [SpecTr 原文](https://proceedings.neurips.cc/paper_files/paper/2023/file/6034a661584af6c28fd97a6f23e56c0a-Paper-Conference.pdf#page=5)。

未采用“不同于同质 token 序列的权衡”这一更强表述：现有实验没有直接比较这类序列结构，无法据此建立一般性差异。本轮没有新增方法、实验结果或性能复测；正文其余方法、数学分析与实验数据保持。

## 8. 正式论文稿整理

根据用户要求，PAPER_MAIN.md 已由证据说明式正文整理为完整中文论文稿，包含 Abstract、Introduction、Related Work、Method、正确性与计算收益分析、Experiments、Discussion、Conclusion 和附录 A–C。当前主张集中于：位置排序承担候选生成，神经预测从上一位置 Target 状态直接估计下一候选位置状态，以短 P/F 路径减少部分逐 token 草稿状态计算；Rescue 沿该路径的剩余恢复工作引出。

主文删除了评价自身贡献地位、反复声明不主张什么和逐批次汇报的文字；相关工作直接说明方法联系与具体差异。实验按位置投机起点、Feature 净增量、状态与成本分析、恢复收益和完整请求组织。表中数值采用适合论文阅读的精度，精确数值、批次设置、训练对照、完整计数和原始来源移入附录 C；详细推导与负向结果保留于附录 A、B。

此轮加强了问题动机和机制表达，没有增加未经测量的结果或首次、普遍最优等结论。候选输入消融、同预算 Transformer 结果、适配方法范围及完整请求长度差异仍在相应结果旁明确说明。共享 Ranker 的实验目的保留：评价不同 Feature 路径能否保留并扩大位置投机收益。

本轮未运行实验，未变动算法、权重或原始记录。后续排版及最终训练细节补充可基于这份正式主稿完成；本文件和 NOVELTY_ARGUMENT.md 继续保留研究判断与证据边界，避免将这些讨论再次搬入论文主文。

## 9. 方法定义与自然 EOS 三维证据补全

用户确定继续完善现有主线，不因项目页面变化改变研究节奏，也不将新模块作为当前前提。本轮完成以下内容：

- Method 3.2 补入位置候选的因果差值统计与扫描顺序过滤、当前规则的至多六候选上界、32 维历史特征和梯度提升树排序；正文明确该模型的二分类评分仅用于选择点提案。
- Method 3.2 与附录 C.6 给出 1,024 维状态输入、三层残差变换、2,864,384 个可训练参数、共享 4,194,304 个 Feature 输出权重元素，以及约 5.505M 线性乘加。候选构造、树评分和神经读出分别说明，不由乘加数替代执行时间。
- 附录 C.6 补齐主模型的 400/50 条训练与验证轨迹、418,021/46,893 状态行、已有初始化及直接 Target 训练阶段、CE 与采样策略 TV 目标、优化器和验证选择。此处轨迹数量不等于独立来源场景数。
- Related Work 2.2 加入此前已直接阅读原文的 P-EAGLE 与 DFlash，明确并行起草及 Target 状态辅助块预测的先例；主实验仍只评价实际已测适配方法。
- 新增第 5.7 节、表 5、附录 C.7，整合阶段 F 的全长补全与 BOS 生成，并调整摘要、引言和 Discussion 中的覆盖范围。
- 新增图 1 的方法示意、图 3 的自然 EOS 请求时间比；图 2 与补充图 S1 直接引用已有六视角预览。原始三维图像未改动。图源位于 figures/，可编辑 SVG 和 PNG 均已保存。

**恢复使用的全长证据。** 阶段 F 已有 270 次真实前缀补全和 90 次 BOS 请求，均自然 EOS、无截断。它沿用 step12000 基础轻量 P/F 路径，执行后端较早，未包含 Rescue。主稿此前未纳入该批完整三维证据；此前对研究完成度的概述遗漏了这部分，当前入口已补充说明。基础 P/F 的补全请求时间比为 1.569348978，BOS 为 1.790705485；对应区间沿用既有统计。该批不能作为当前完整 PFSpec 的全长 Rescue 结果。

**本轮复算。** 从两份 CSV 读取全部 360 行，确认模式配对完整、自然 EOS 与完成标记一致、无截断，且五项分时之和等于完整时间。从补全的 30 个来源、BOS 的 30 个种子重新汇总四项相对 Target 的几何平均点值，与既有 summary.json 在 1e-12 内一致。没有重新抽样区间、运行模型、进行 GPU 测量或生成新的三维场景。新增统计图采用既有点值和区间，定性图沿用预定首个输入。

## 10. 下一项证据的最小范围

已有 360 次全长基础 P/F 生成不再列为待重复工作。当前主要缺口收紧为：**完整 PFSpec 的 Rescue 及共享执行后端，在自然 EOS 请求中的增量是否仍成立。** 若取得可用运行环境，可用预先选定的少量来源、原有前缀和同一组种子，配对运行共享优化 Target、W1 与单备用 PFSpec；沿用既有完整请求入口，保留实际长度、EOS、完整时间和三维输出。来源选择不依赖此前加速比。该任务不新增训练或改变算法。

对草稿状态安排的进一步归因，先利用现有源码分清必需状态推进与当前适配器的额外调用。若论文仍需独立的时间因果证据，再增加一项限定对照；当前总时间差不全部归给状态推进。空间输入、并行双 Feature、概率 Position 提案暂保留为研究备选，未纳入本轮方法或主实验。

本机是资料与源码副本，未发现完整模型和 GPU 工程。已向用户询问可用 GPU 工程的连接入口；本轮独立完成方法与全长材料整理，未发起新远端运行。源码、权重、原始 CSV/JSON、既有 PNG/MP4 及生成状态快照保持。

## 11. 正式稿引用与内部资料分离

正式稿及附录已移除项目源码、实验日志、数据表和内部笔记的路径引用；学术论文链接、实验设置、统计数值、证明条件与限制保持。附录中的内部版本号及记录字段说明改为方法与统计口径说明。两张三维定性图以原图副本纳入论文 figures/ 目录，原始图像保持不变；论文通过图号引用，图片路径只用于排版载入。

以下索引仅供内部修稿使用，不属于论文参考文献。

**附录 B. 其他批次与负向结果**

- [较早记录](../spec_next_20261022/RESULTS.md)
- [备用数量记录](../position_widths_20261022/RESULTS.md)
- [新来源结果](../recovery_value_20261023/RESULTS.md)
- [执行分析](../recovery_value_execution_20261023/RESULTS.md)
- [概率与成本模型结果](../recovery_value_evidence_20261023/RESULTS.md)
- [成本选择](../recovery_model_20261021/RESULTS.md)
- [候选枚举](../v3_review_20261023/POOL_ENUMERATION.md)

**C.1 分布与轨迹保持的详细条件**

- [理论笔记](../paper_design_20261021/THEORY_NOTES.md)

**C.2 主比较的完整范围**

- [同后端实验](../pf_comparison_20261025/RESULTS.md)
- [主比较设置](../pf_comparison_20261025/main_run01/plan.json)
- [原始记录](../pf_comparison_20261025/main_run01/records.jsonl)

**C.3 结构、输入与训练预算**

- [结构比较设置](../pf_comparison_20261025/mechanism_run01/plan.json)
- [原始记录](../pf_comparison_20261025/mechanism_run01/records.jsonl)
- [结构与输入结果](../pf_comparison_20261025/RESULTS.md)
- [位置消融](../../NEXT_ROUND_RESULTS_2026-10-17.md)

**C.4 宽度、较重草稿与恢复重复测量**

- [宽度对照](../feature_tree_20261021/RESULTS.md)
- [阶段 E 证据](../../PAPER_EVIDENCE.md)
- [逐对机制统计](../pf_comparison_20261025/main_run01/decode_analysis.json)
- [重复测量](../rescue_repeat_20261022/RESULTS_REVIEW_20261023.md)

**C.5 执行实现与完整请求**

- [归因说明](../spec_next_20261022/PAPER_NEXT.md)
- [原始记录](../spec_next_20261022/raw/paper_attribution_run01/records.jsonl)
- [完整请求说明](../spec_next_20261022/PAPER_NEXT.md)
- [请求原始记录](../w1_e2e_20261021/rescue_paper_run01/records.jsonl)

**C.6 候选构造、预测器与训练细节**

- [候选生成器](../web_chatgpt_handoff_20261019/GaussianGPT_Web_Handoff_2026-10-19/source/motivation_20261016/eval_lean_candidate_to_draft.py)
- [在线因果特征](../position_transition_20261020/source_before/gpt_position_spec_decode_smoke.py)
- [排序器训练入口](../../../../.codex/transfer/research_evidence_20261016/source/eval_lean_ranker_holdout.py)
- [CPU 树评分](../position_transition_20261020/source_before/bench_fast_hgb.py)
- [历史评分计时](../../../../.codex/transfer/research_evidence_20261016/fast_ranker_summary.json)
- [状态预测器](../position_transition_20261020/source_before/train_feature_state_distill.py)
- [直接 Target 训练](../../../../patches/feature_target_v2/feature_target_v2.py)
- [训练数据组成](../../../../patches/feature_target_v2/results/data_plan.json)
- [前三阶段设置](../../../../patches/feature_target_v2/results/training_plan.json)
- [续训设置](../../../../patches/feature_target_v2/continuation_results/training_plan.json)
- [已选权重记录](../../../../patches/feature_target_v2/continuation_results/selected.json)

**C.7 全长单 Chunk 输出与定性素材**

- [全长实验说明](../../FULL_3D_OUTPUT_2026-10-15.md)
- [完整汇总](../stage_f_full3d/analysis/summary.json)
- [270 行补全记录](../stage_f_full3d/analysis/test_samples.csv)
- [90 行 BOS 记录](../stage_f_full3d/analysis/bos_samples.csv)
- [全部补全视频](../stage_f_full3d/analysis/test_completion_30scenes_seed51701_6views.mp4)
- [全部 BOS 视频](../stage_f_full3d/analysis/bos_30trajectories_6views.mp4)

论文图片与原图对应：

- [figures/natural_eos_completion.png](figures/natural_eos_completion.png) ← [原图](../stage_f_full3d/analysis/test_completion_30scenes_seed51701_6views.preview.png)
- [figures/natural_eos_bos.png](figures/natural_eos_bos.png) ← [原图](../stage_f_full3d/analysis/bos_30trajectories_6views.preview.png)

本轮只调整论文、内部索引及图片组织，没有重新运行实验。当前无可用 GPU，完整 PFSpec 的自然 EOS 补测未执行。

## 12. 英文 CVPR Overleaf 版本包

已依据正式中文稿组织完整英文主文与独立补充材料，使用 CVPR 2027 官方模板的匿名评审格式。英文主文保留核心问题、类型化短路径、Position Rescue、正确性条件、全部主要实验及适用范围；补充材料包含完整推导、其他实验、训练细节和自然终止三维结果。论文只引用学术文献，18 条 BibTeX 信息依据原始论文或出版页面整理，图片随包提供。

新增文件位于 overleaf/，其中 main.tex 与 supplementary.tex 可分别编译；可上传的包为 PFSpec_CVPR2027_Overleaf_EN.zip。预览为 PFSpec_CVPR2027_Main_EN.pdf 与 PFSpec_CVPR2027_Supplementary_EN.pdf。主文内容 7 页，参考文献及 LLM 使用说明 1 页；补充材料 8 页。已用本机既有 Tectonic 编译、逐页查看，并确认没有缺失引用或版心溢出。Overleaf 的 pdfLaTeX 排版可能出现少量换行差异。

当前保留匿名作者，LLM 使用说明描述本会话已知的论文准备用途；提交前仍由作者确认完整表述。此轮未改变算法或实验数值，没有运行 GPU 实验。当前完整 PFSpec 的自然 EOS 增量仍未测量。中文稿与原始实验资料保持原样。

## 13. 条件信息分工与完整三维证据的主文整合

中英文 Introduction 已围绕 P/F 能够廉价取得的条件信息、位置排序与直接状态预测的计算分工，以及有效推进相对完整执行成本的关系重组。Contribution 仍保留两项，将类型化短路径作为核心设计，恢复复用沿同一解码过程引出。Method 3.1 增加设计目标，明确位置历史用于提案，两类 Target 分布仍读取完整条件历史；提前选出第二位置不代表其 Target 分布与中间 Feature 独立。

PSNR/SSIM 从附录描述整合为主文 5.7 的表 6，与自然 EOS 的请求时间和六视角图共同呈现。三个方法的六个均值已与既有自然终止汇总逐项对应。表题与正文说明其为相对于真实裁剪 VQ 重建的参考图像统计，没有质量等价检验；BOS 无配对真实场景。附录 C.7 保留评价范围、其他输出统计和指标限制，并指向主文表 6。

已有消融与成本对照支持这套安排在已测配置下有效，无法单独证明其不可替代，或证明 MLP 优于所有轻量预测器；正文保留这些比较范围。基础 P/F 全长实验与当前含 Rescue 的完整版本继续分开。当前无可用 GPU，未新增自然 EOS 测量或训练。

Overleaf 源码、主文及补充 PDF 和 ZIP 已同步。英文主文内容与图表 8 页，文献及 LLM 使用说明另 1 页；补充材料 8 页。两份 PDF 编译成功并逐页查看，没有缺失引用、公式或表格溢出。算法、权重、原始实验和图像素材未改动。

## 14. 末页留白与结果表排版

根据用户反馈，取消英文主文结论与 LLM 使用说明之间的强制分页，将同批自然 EOS 请求时间和 PSNR/SSIM 合并为表 5，删除原独立表 6；附录 C.7 与中文主稿的表号及数据布局同步。表中明确时间与 token 数为均值，BOS 没有配对质量参考，质量均值不构成等价检验。全部实验数值保持。

精简英文引言和结论的重复表述，以及 LLM 使用说明的措辞；研究主张、已知限制和工具使用范围保持。主文与文献自然接排，当前本机编译总计 8 页，原第 8 页的大面积留白及额外第 9 页已消除。没有改变正文和文献字号、模板边距或图像尺寸。补充材料仍为 8 页。

已查看主文全部页面，核对表 5 的均值、区间、质量指标及附录指向；未发现缺失引用或版心溢出。PDF、Overleaf ZIP 和 README 已同步。Overleaf 的 pdfLaTeX 可能有少量换行差异。源码、模型和原始实验资料未改动。
