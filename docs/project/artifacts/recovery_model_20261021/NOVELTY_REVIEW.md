# 恢复成本模型的新颖性核读

**结论。** `R=2+2·1(P1∩F1)-1(E1)-1(E2)` 与按拒绝路径计入的恢复成本，可直接嵌入有限状态的 Markov reward / 半 Markov 决策过程。忽略 EOS 时，它是一个两 token 结构块上的接受前缀计数；加入 EOS 后，多出的项是终止路径少生成一个 token 的计数。一般数学框架、接受长度与实测成本的比值，已有明确前例。当前证据不支持把该等式、标准拒绝采样或“考虑恢复成本”单独表述为新理论。

仍可研究的范围较窄：把 Position/Feature 交替结构中不同拒绝分支的真实恢复调用、KV 操作和图复用成本显式纳入动作评估，并验证它是否比接受长度或通用吞吐测量更能预测、改善未参与选择的来源表现。这是待实验支持的候选贡献，不是已确认的新颖性结论。

## 公式与已有目标的关系

项目的无 EOS 正常轮次关系为

\[
R=2+2\mathbf 1_{A_1\cap A_F}.
\]

含 EOS 时为

\[
R=2+2\mathbf 1_{A_1\cap A_F}-\mathbf 1_{E_1}-\mathbf 1_{E_2}.
\]

令状态 `s` 包含已提交历史、下一 token 类型、Target KV、提案器状态、剩余预算及设备图状态；动作 `a` 选择一对、W1 或 W2 等既有提案方式；结果 `ω` 是本轮各候选接受、拒绝或 EOS 的实际路径。令 `r(s,a,ω)` 为该路径提交的 token 数，`c(s,a,ω)` 为实际墙钟成本。局部效率可以写作

\[
g(s,a)=\frac{\mathbb E[r(s,a,\omega)]}{\mathbb E[c(s,a,\omega)]},
\]

完整请求则是标准的有限状态期望成本问题：

\[
V(s)=\min_a\mathbb E[c(s,a,\omega)+V(s')\mid s,a].
\]

项目中的 `E1/E2` 只改变终止分支产生的 token 数；P1、Feature、P2 的拒绝与接受改变后续状态和恢复调用时间。把这几个条件事件列开，能更忠实地复算当前解码器的成本，但不会让其脱离普通状态转移奖励/成本模型。对于任意确定的 token 序列，每生成一个 token 记一分，就能得到相同的 `R`；把相邻 P/F 两 token 合并成一个结构块，只改变计数单位。若保留部分块和 EOS 状态，单 token 展开与 pair 展开等价。

需要区分“广义上能表达”与“某篇论文已经逐字写出同一式子”。本轮选定原文中未发现这条具体 P/F 公式，但这个事实不能证明数学贡献新颖。现有目标都可通过状态展开、不同 token 权重或转移相关成本表示同类问题。

| 工作 | 原文已有机制与目标 | 对本模型的含义 |
|---|---|---|
| [Leviathan 等，Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/html/2211.17192) | 按 `min(1,p(y)/q(y))` 接受草稿，拒绝后从归一化正残差抽样；推导预期接受数与速度比。 | W1 的分布保持是既有拒绝采样在异构提案上的实例化，不宜列为新定理。 |
| [SpecInfer](https://arxiv.org/html/2305.09781) | 用树注意力并行评分整棵候选树，以 multi-step speculative sampling 保持目标分布；论文定理证明采样正确性，并比较可验证 token 数。 | 通用树验证、精确采样和每轮有效 token 数已有直接先例。项目可继续比较类型条件路径与设备执行细节。 |
| [Sequoia](https://arxiv.org/html/2402.12374) | 动态规划求 `G(n,d)`（预期生成 token 数），硬件测量 `t(n)`（目标模型验证 n 个 token 的归一化耗时）与 `c`（每个草稿 token 的归一化成本），选择最大化 `G(n,d)/(t(n)+d·c)` 的树大小和深度。 | “进展除以实测成本”及按硬件选择推测规模已存在。项目的恢复时间可成为更细的 `c`，但需证明分支条件建模产生额外预测或净收益。 |
| [OPT-Tree](https://arxiv.org/html/2406.17276) | 每个输入上下文在节点预算内选择树，最大化预期最长接受路径 `E(A)`；草稿路径概率用于估计节点被接受的机会。 | 上下文相关的接受长度优化已有先例。pair 权重可按两个 token 折算，不能仅以交替类型命名宣称新目标。 |
| [Traversal Verification](https://arxiv.org/html/2505.12398) | 以从叶到根的序列级接受做树验证；被拒父节点下的候选不立即全部丢弃，并证明输出分布与目标一致。 | 候选利用率和拒绝后的有效路径已有新验证方案。本项目的恢复项是执行端真实 Target 补算成本，需和这类验证规则分开比较。 |
| [Dynamic Delayed Tree Expansion](https://arxiv.org/html/2602.16994) | 定义 block efficiency `E[τ+1]`；树枝分配考虑草稿、Target pass 的时间成本；用轻量 MLP 根据根节点特征及模型/硬件时间参数选择单路径深度、分支位置和分支规模。 | 这是对“状态相关推进效率 + 系统成本 + 学习式动作选择”最接近的原文。恢复路径作为细化成本仍可能有价值，但只有在对照其通用 block-efficiency 选择后带来留出收益，才形成清楚差异。 |
| [PosS: Position Specialist Generates Better Draft for Speculative Decoding](https://arxiv.org/html/2506.03566) | 对草稿深度位置 i 统计 `P(接受 i | 前 i−1 个均接受)`，并训练不同 specialist 处理逐步增大的隐藏状态偏差。 | 论文中的 position 指草稿序列偏移量，不指 GaussianGPT 的空间 Position 类型。它显示按局部条件接受分析已有先例；项目需把 P/F 语义类型差别讲清楚。 |
| [Spec-LLaVA](https://arxiv.org/html/2509.11961) | 根据草稿置信度动态扩展或裁剪候选树，采用 leaf-to-root 验证。目标是视觉输入条件下的文本生成。 | 动态树和多模态条件生成已有近作，但其输出仍是普通文本 token；原文没有建立 P/F 交替的 token 级结构收益或分支恢复成本模型。 |
| [On Speculative Decoding for Multimodal Large Language Models](https://arxiv.org/html/2404.08856) | 在 LLaVA 上用语言模型草稿；草稿可只读取文本，Target 仍使用图像条件进行验证，也比较带图像适配器的草稿。 | 多模态输入条件下应用投机解码已有原始研究。此类设置中的视觉与文本是输入上下文，和输出端交错的 Position/Feature 类型序列不同。 |
| [Is Multimodal Speculative Decoding Ready for Diffusion-Based Parallel Drafting? A Survey and Empirical Diagnosis](https://arxiv.org/html/2608.20743) | 2026 年综述及实证诊断按 drafter 并行度区分 L0 顺序、L1 多位置、L2 块级并行；将树分支和验证方法视作另一维度，并总结视觉、视频、语音、动作等方向。 | 该分类提示应把本项目定位为类型交替序列上的顺序/多候选草稿与验证执行协同，不要称为块并行草稿。该文属于综述与实证工作，具体新方法仍需追读它列出的原始论文。 |

## 论文主张应收窄到哪里

1. 将精确拒绝采样、残差分布及条件概率扩展作为已知方法的正确实现，不写成新的通用抽样定理。
2. 将 `R` 写成特定 P/F 解码状态机的计数恒等式。它解释 P1 与 F1 联合接受带来第二个完整 pair，P2 接受主要改变恢复成本；局部结果不等同全局最优调度理论。
3. 将 `c_w` 描述为现有树/投机算法成本目标的路径条件化版本，显式测量非 EOS 拒绝后的真实恢复调用、Target 时间、KV 和图创建/复用。只写成本模型的精度范围，不称其为新优化准则。
4. 将 W1/W2 描述为特定 Position/Feature 交替生成器上的方法适配。交替语法和轻量 Ranker/MLP 的组合有系统价值，但仅凭任务结构不同无法证明算法新颖。
5. 若实现动作选择器，候选新贡献应表述为“类型和恢复路径感知的提案预算选择”，并与 Sequoia 的硬件实测目标、OPT-Tree 的输入相关接受长度、延迟分支工作的 block-efficiency 选择直接比较。没有留出来源上的净收益时，保留为分析模型。

## 可复现区分实验

在未参与模型选择的来源上，从相同已提交前缀分别重放一对、W1、W2 及可比树宽动作；每种动作使用原来的精确验证和残差规则。策略在读取本轮候选前选定，避免按已观察到的候选筛选后仍沿用原提案概率。每条路径保存 P1、F1、P2 的接受状态，E1/E2、实际提交 token 与完整 pair 数，非 EOS 恢复 Target 调用及耗时，Target 验证、提案、采样、KV、图首次创建/复用的分项成本，剩余预算和终止类型。

在开发来源上拟合或配置三种排序：

- 仅使用 `E[A]` 的接受长度排序，对应 OPT-Tree 的主要目标；
- 期望 token 数除以实测草稿/Target pass 成本，对应 Sequoia 类目标；
- 本文按 P/F/EOS 路径计算 `E[R]` 与 `E[C]`，并用状态转移预测后续成本。

在来源留出集上比较动作排序准确度和实际吞吐收益，计入选择器运行时间。另做两项消融：将不同拒绝分支的恢复成本替换为共同平均值；将 pair 计数改回逐 token 计数。若两项变化不影响留出动作排序或外层时间，成本细分和 pair 公式提供解释性整理，不能作为新方法效果。除固定预算外，另报自然 EOS 结果，并将图冷启动与已创建图复用分别统计。按来源等权汇总，来源 bootstrap 区间与共同满预算子集分别列出。

这一实验支持的判据很直接：类型/恢复信息必须在通用接受长度或 token-per-cost 预测之外，稳定改善未见来源上的动作选择，并且外层总时间出现净下降。否则应把贡献保持在结构化生成中的成本分析与运行时适配。

## 检索范围与缺口

检索截止 **2026-10-21**。本轮复用 [Academic route 记录](/Users/lief/.codex/skills/research-router/records/routes/2026-10-21/gaussiangpt-ranker-w1-literature-20261021.json) 与 `/tmp/gaussiangpt_academic_raw_20261021/`，沿 `research-router → academic-local → academic-evidence` 路径，仅对问题相关原文补读，没有重跑泛检索。正文段落来自原论文公开 arXiv HTML；本轮重点补读 Sequoia、OPT-Tree、SpecInfer、Traversal Verification、Dynamic Delayed Tree Expansion、PosS、Spec-LLaVA、Gagrani 等人的多模态投机解码，以及 2026 综述与实证诊断。

这是一份有界比较，不覆盖 2025–2026 所有树选择、恢复采样、约束生成、语音/动作解码或多模态提案方法；综述列出的 DDTree、TAPS、AASD 等原文没有在本轮逐篇阅读。Spec-LLaVA 正文主张其为首个 VLM speculative decoding 方法；Gagrani 等人 2024 已在 LLaVA 上研究同类应用，因此本报告只依据两者各自实际方法段落，不沿用其优先性表述。没有检索到完全相同的 P/F 公式，不构成排除性结论。
