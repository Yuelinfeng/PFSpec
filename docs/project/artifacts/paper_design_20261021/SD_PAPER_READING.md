# 通用投机解码正文研读

日期：2026-10-23

范围：独立阅读六篇通用投机解码论文，覆盖精确残差抽样、树注意力、树结构选择、特征级起草和近期遍历验证。主 Agent 另行阅读 SpecTr、Grouped Speculative Decoding、SpecMER，本文件只记录与它们交叉处的必要事实，不重复其正文研读。

证据规则：

- 详细结论来自论文正文或附录的章节与 PDF 页码；每段结论后给出正文 URL。
- 会议或期刊身份单独给出出版来源。正文所读的 arXiv 版本与出版版本可能有版本差异，不能把两者混成同一版本。
- 六篇正文 PDF 均在 2026-10-23 由本机临时资料读取。Sequoia 使用的 PDF 首页标为 arXiv v3、2025-07-05；其会议身份由 NeurIPS DOI 单独确认。
- 摘要只用于定位论文，下面的详细判断没有把摘要当作正文证据。

## 一、先给主线判断

当前 GaussianGPT 的恢复分支可以准确描述为：在主 Position 候选被拒绝后，沿用参考解码器的残差抽样；提前把一个或多个合法恢复位置作为目标树分支计算；若实际抽到被覆盖的位置，则复用该位置对应的 Feature 分布、hidden 和 KV。这个过程直接落在已有的残差抽样、树注意力、共享前缀 KV 和候选成员命中复用的组合上。文献资料支持“结构化生成实例与成本分析”的定位，暂时不足以支持“通用树验证或残差抽样首创”的定位。

当前最需要保留的较窄问题是：在固定参考抽样路径、验证前已经支付的廉价 Position 排序和有限验证预算下，哪些恢复条件状态值得预计算？这项问题仍需与 Sequoia、OPT-Tree 的树结构选择，以及 SpecInfer、Traversal Verification 的完整树验证策略做同候选、同成本比较。只要新机制只把备用位置加入 Target 树，操作层面就有明确重合；新颖性需要来自受限信息、P/F 条件状态和真实恢复成本的组合证据，不能从“换成三维 token”直接推出。

## 二、论文身份与正文来源

| 论文 | 身份 | 正文所读版本与 URL | 出版来源 |
|---|---|---|---|
| Fast Inference from Transformers via Speculative Decoding，Yaniv Leviathan、Matan Kalman、Yossi Matias | 已发表，ICML 2023 | arXiv HTML：<https://arxiv.org/html/2211.17192> | PMLR 页面，ICML 2023，pp. 19274–19286：<https://proceedings.mlr.press/v202/leviathan23a.html> |
| SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification，Xupeng Miao 等 | 已发表，ASPLOS 2024 | arXiv HTML：<https://arxiv.org/html/2305.09781> | ACM DOI，ASPLOS 2024，pp. 932–949：<https://doi.org/10.1145/3620666.3651335> |
| SEQUOIA: Scalable and Robust Speculative Decoding，Zhuoming Chen 等 | 已发表，NeurIPS 2024 | arXiv HTML：<https://arxiv.org/html/2402.12374> | NeurIPS DOI：<https://doi.org/10.52202/079017-4116> |
| OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure，Jikai Wang 等 | 已发表，TACL 2025 | arXiv HTML：<https://arxiv.org/html/2406.17276> | TACL DOI：<https://doi.org/10.1162/tacl_a_00735> |
| EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty，Yuhui Li、Fangyun Wei、Chao Zhang、Hongyang Zhang | 已发表，ICML 2024 | arXiv HTML：<https://arxiv.org/html/2401.15077> | PMLR 页面，ICML 2024，vol. 235，pp. 28935–28948：<https://proceedings.mlr.press/v235/li24bt.html> |
| Traversal Verification for Speculative Tree Decoding，Yepeng Weng 等 | 已发表，NeurIPS 2025 | arXiv HTML：<https://arxiv.org/html/2505.12398> | NeurIPS DOI：<https://doi.org/10.52202/085713-2611> |

## 三、逐篇正文研读

### 1. Leviathan et al.：精确残差抽样的起点

**问题、缺口与适用定位**

论文处理大型自回归模型逐 token 运行代价高的问题，提出小模型或廉价近似模型先起草、Target 一次并行验证的链式流程。正文覆盖了“如何在保留 Target 输出分布的情况下减少串行 Target 调用”；它没有研究分叉树、P/F 类型状态或拒绝位置的条件状态预计算。适用定位是精确投机抽样的基础论文，当前恢复分支的概率语义应以它为直接来源。（正文：§§2.2–2.3、PDF p.3；正文 URL：<https://arxiv.org/html/2211.17192>）

**假设与机制**

给定 Target 分布 p 和草稿分布 q，先按 q 抽样 x。接受概率为 min(1, p(x)/q(x))；拒绝后从 norm(max(0, p-q)) 抽样。多步 Algorithm 1 先自回归生成 gamma 个草稿 token，再一次取得 p1 到 p(gamma+1)，从前向后处理第一处拒绝。主路径接受时，后续已经算好的 Target 分布可以继续使用；主路径拒绝时，后续草稿状态被丢弃，改从残差分布产生一个 token。（正文：§2.3、Algorithm 1、PDF p.3；正文 URL：<https://arxiv.org/html/2211.17192>）

**理论实际内容**

正文 Theorem 3.5 把单 token 接受率写成 beta = sum_x min(p(x), q(x))，等价于 1 - D_LK(p,q)；在各步接受事件独立同分布、均值为 alpha 的简化条件下，单轮期望输出长度为 (1 - alpha^(gamma+1))/(1-alpha)。Theorem 3.8 在 Target 可并行运行 gamma+1 个位置、草稿单次成本比例为 c 的条件下给出墙钟收益表达式；Theorem 3.11 给出算术操作量变化。Appendix A.1 逐项相加证明接受项与残差项合成 p。这里真正被证明的是抽样分布保持和受假设约束的期望成本公式，树状态复用没有进入定理。（正文：§§3.1–3.6、PDF pp.3–5；Appendix A.1、PDF p.11；正文 URL：<https://arxiv.org/html/2211.17192>）

**实验如何回答主张**

§4.1 在单 TPU-v4、batch size 1 上用 T5-XXL 11B 作 Target，用 T5-small、T5-base、T5-large 作草稿，任务为 WMT 英德翻译和 CNN/Daily Mail 摘要，测量 temp=0 与 temp=1。T5-small 在英德任务上达到 3.4x（temp=0）和 2.6x（temp=1），摘要任务达到 3.1x 和 2.3x；这些结果与正文的 alpha、c 预测方向一致。§4.2 进一步在 97M GPT-like、137B LaMDA 和 T5 设置上统计 alpha，显示较小模型、bigram 等廉价近似也可能保留可用的重合概率。实验支持链式精确抽样和成本量纲，未回答树分支是否值得提前算。（正文：§§4.1–4.2、PDF pp.6–7；正文 URL：<https://arxiv.org/html/2211.17192>）

**局限**

期望 token 数的闭式依赖独立同分布接受率假设；墙钟公式依赖足够并发资源和稳定的成本比例；实测主体是 batch size 1 的 T5 任务。论文没有把共享 KV、树宽、分支提交和恢复前向纳入同一个状态成本模型。输出分布保持也不等于不同实现使用同一随机调用顺序后逐 token 相同。（正文：§§3.1、3.3–3.5、4.1、PDF pp.3–6；正文 URL：<https://arxiv.org/html/2211.17192>）

**对 GaussianGPT 新颖性的威胁与可学习的论证组织**

残差抽样、拒绝后的补采样和“正确性与效率分开证明”的组织方式都已建立。GaussianGPT 的恢复位置抽样必须明确引用此来源；“新残差规则”“无质量变化的树恢复抽样”均不能作为贡献。可借鉴的写法是先给 Algorithm 1，再给分布保持证明，随后把 alpha、草稿成本和并发假设分别进入理论与实验，避免用一个速度数字承担所有主张。对 GaussianGPT 的适用定位：它提供 W1 参考过程的抽样语义，不能替代 P/F 条件恢复状态的实测成本研究。（正文：§2.3、§§3.2–3.5、Appendix A.1、PDF pp.3–5、11；正文 URL：<https://arxiv.org/html/2211.17192>）

### 2. SpecInfer：树注意力、共享 KV 与多步验证

**问题、缺口与适用定位**

SpecInfer 针对单链起草带来的候选覆盖不足和共享前缀重复计算，构造 token tree，并在一次 Target 解码中并行计算整棵树。它同时处理 greedy 和随机抽样。正文覆盖了树构造、树注意力、KV 管理、精确多步验证和服务系统；没有专门研究“主位置拒绝后，某个恢复位置的 Feature 状态是否应提前算”这一类型语义。适用定位是我们五节点 Target 树和共享状态实现的直接通用参照。（正文：§§4.1–4.3、PDF pp.7–9；正文 URL：<https://arxiv.org/html/2305.09781>）

**假设与机制**

§4.1 把每个树节点表示的 token 序列送入原始序列注意力，得到与该序列相同的注意力输出。§4.2 用深度优先顺序复用共享 KV，并按树的父子关系建立 topology-aware causal mask，使同一 kernel 中的节点只读取自己的祖先。树可由单个 SSM 扩展，也可合并多个 SSM 的结果。§4.3 的 VerifyStochastic 对每个分支递归使用 Target/SSM 概率比和残差分布，形成 multi-step speculative sampling。（正文：§§3–4.3、PDF pp.4–9；正文 URL：<https://arxiv.org/html/2305.09781>）

**理论实际内容**

Theorem 4.2 陈述 Multi-step speculative sampling 输出分布等于增量解码的 Target 分布；Theorem 4.3 陈述其拒绝概率不高于直接从 Target 抽样后判断是否在树中的方法。正文给出定义、算法和定理陈述，证明细节指向引用资料，因而不能把它描述成本文内完整展开的全新证明。§4.2 对树 mask 的关键实际保证是：共享前缀的隐藏状态保持一致，非祖先节点不会进入当前节点的注意力范围。（正文：§4.1–4.3、Theorems 4.2–4.3、PDF pp.7–9；正文 URL：<https://arxiv.org/html/2305.09781>）

**实验如何回答主张**

§6.1 使用 LLaMA-7B、OPT-13B、OPT-30B、LLaMA-65B 及小 SSM，在两台各含四张 A10 的 AWS g5.12xlarge 上评估。§6.2 报告树投机相对增量解码在单节点多 GPU 上 1.5–2.5x、跨节点多 GPU 上 2.4–2.8x，并称所有输入生成序列与增量解码相同；相对序列投机再降低 1.2–1.5x 延迟。§6.3 的 CPU DRAM offload 设置相对 FlexGen 降低 2.6–3.5x 每 token 延迟。§6.4–§6.6 把宽度、树并行和采样分别拆开：宽度 1 到 5 时平均验证 token 增加；树并行在大 batch 下最高达到 1.8x；MSS 相对直接 Target 抽样在多个数据集上提高 1.2–1.3x 可验证 token 数。结果把树结构、执行复用和采样策略的作用分开，适合借鉴。（正文：§§6.1–6.6、PDF pp.11–13；正文 URL：<https://arxiv.org/html/2305.09781>）

**局限**

收益依赖可利用的 GPU 空闲计算资源；batch 增大后优势减弱。树节点的 KV 与注意力中间量带来额外显存和算力，系统收益依赖硬件、并行布局和 kernel。论文的树节点是语言 token 序列，未评价 P/F 交替语法、类型条件分布或恢复位置的单独成本。（正文：§5.2–5.3、§6.2–6.5、PDF pp.10–13；正文 URL：<https://arxiv.org/html/2305.09781>）

**对 GaussianGPT 新颖性的威胁与可学习的论证组织**

GaussianGPT 的 [F_prev, a, f, c, b] 五节点布局、共同前缀、树 mask、深度优先 KV 管理和只提交真实路径，均与 SpecInfer 的通用操作重合。备用 b 被抽到时复用 b 行 Feature 分布和 hidden，可以解释为树中某个已计算节点的成员命中；这个事实足以否定“恢复分支复用首次出现”的说法。可保留的差异是：b 的语义来自 P/F 类型链上的拒绝恢复事件，主提案和随机流保持 W1；这些差异需要同一个通用树实现的对照，适合写成结构化生成适配与恢复成本分解。可学习的组织方式是把算法拆成树构造、状态计算、验证抽样和服务成本，再用 §6.4–§6.6 一项一项对应消融。（正文：§4.1–4.3、§6.4–6.6、PDF pp.7–9、12–13；正文 URL：<https://arxiv.org/html/2305.09781>）

### 3. Sequoia：动态规划树与硬件感知预算

**问题、缺口与适用定位**

Sequoia 观察到独立序列树和固定满 k 叉树在树规模扩大后收益趋于饱和，且 SpecInfer、SpecTr 在低温时可能重复提出并拒绝草稿高概率 token。论文把树结构优化和抽样验证分成两个模块，目标分别是提高可生成 token 数、改善温度鲁棒性。它没有以 P/F 恢复状态为对象；对 GaussianGPT 的直接威胁来自“受预算约束的树结构选择”和“硬件成本进入树宽/深度决策”已经有完整先例。（正文：§§2.1–2.2、§§3.1–3.2、PDF pp.3–7；正文 URL：<https://arxiv.org/html/2402.12374>）

**假设与机制**

在 positional acceptance assumption 下，节点分数 f(v) 是从根到该节点各位置接受率的乘积，Proposition 3.4 给出 F(T)=sum_v f(v)。动态规划张量 T(n,b) 在节点数 n、根的第一分支宽度 b 和最大分支 B 下搜索 F(T) 最大的树。验证侧使用 draft 分布无放回抽样；若 draft 的非零概率 token 都已被拒绝，则在尚未抽过的 token 上使用均匀分布继续处理。硬件选择再用测得的 Target 时间 t(n) 和草稿单步成本 c，最大化 G(n,d)/(t(n)+d*c)。（正文：§3.1.1、Proposition 3.4、§3.2.1、PDF pp.4–6；§4.1、PDF pp.8–9；正文 URL：<https://arxiv.org/html/2402.12374>）

**理论实际内容**

Theorem 3.6 在接受率具有幂律上界的条件下，给出 Sequoia 树期望生成 token 数的渐近下界，量级为 b log(n)/log(log(n))。Theorem 3.7 宣称 Sequoia 验证同时满足 optimal transport 和 cover 两项性质；正文将 SpecInfer、SpecTr 放在前一项、top-k 抽样放在后一项。相关证明位于 Appendix F。这里的理论对象是位置接受率和抽样性质，不能直接推出真实硬件的全局最优；硬件优化仍依赖测量的 t(n)、c 与有限搜索范围。（正文：§3.1.2、Theorem 3.6、PDF p.6；§3.2.2、Theorem 3.7、PDF p.7；Appendix F、PDF pp.17–24；正文 URL：<https://arxiv.org/html/2402.12374>）

**实验如何回答主张**

§3.1.3 在 Pythia-2.8B 草稿、Pythia-12B Target 和 WikiText-103 上比较树规模，显示 Sequoia 的期望生成量随树规模大致按对数增长，其他结构趋于平稳。§3.2.3 在 WikiText-103 的温度 0.2、0.6、1.0 上比较拒绝率，Sequoia 的拒绝率下降更快，低温下优于 SpecInfer、SpecTr，高温下仍保持竞争力。§4.1 报告 Llama2-7B 在 A100 上最高 4.04x、Llama3-70B 在 L40 offload 上最高 9.5x，最低延迟 0.60 秒/token；硬件优化选出的树在本机通常为 64–128 个 token，在 offload 场景可到 768 个 token。结果确实把树规模、温度、硬件和端到端速度连接起来。（正文：§3.1.3、PDF p.6；§3.2.3、PDF p.7；§4.1、PDF pp.8–10；正文 URL：<https://arxiv.org/html/2402.12374>）

**局限**

树结构公式依赖位置接受率模型；幂律条件描述的是一类统计行为，不能覆盖任意 Target/草稿组合。端到端选择依赖预先测得的硬件时间和特定模型对，换硬件、batch、offload 路径或深度限制就需要重新测量。方法关注树的期望推进量和验证吞吐，没有给出 P/F 恢复事件的状态价值，也没有把“不预计算”作为 GaussianGPT 的恢复集合选择动作来研究。（正文：§3.1.2、§4.1、Appendix B、PDF pp.6、8–10、17；正文 URL：<https://arxiv.org/html/2402.12374>）

**对 GaussianGPT 新颖性的威胁与可学习的论证组织**

任何“按预计收益选择恢复集合”“在预算下选择最佳树宽/深度”“用硬件成本停止扩宽”的通用主张，都必须和 Sequoia 的动态规划及 G(n,d)/(t(n)+d*c) 对照。GaussianGPT 的单备用 b 可看作一个预设的树动作；若再按 m_b 排序和枚举数量，它与 Sequoia 的受限树选择在目标形式上相近。当前可保留的差异是恢复价值 m_b 指向具体 P/F 拒绝事件，目标包含避免额外 Feature Target 前向；这只构成待测比较维度。可学习的组织方式是先分离树构造理论、验证理论和硬件选择，再用三个独立实验回答各自主张，避免把一张总速度表当成算法证明。（正文：§§3.1–3.2、§4.1–4.2、PDF pp.4–10；正文 URL：<https://arxiv.org/html/2402.12374>）

### 4. OPT-Tree：按当前输入贪心构造树

**问题、缺口与适用定位**

OPT-Tree 针对固定 binary、EAGLE 等树结构随输入和模型对变化的问题，试图在每个解码步按当前 draft 分布构造树。它的目标是最大化有限节点数下的数学期望接受长度，再用阈值限制起草深度。适用定位是在线、输入依赖的树结构选择器，对 GaussianGPT 的恢复集合选择和“预计收益可用于动作决定”构成直接威胁。（正文：§§2–3、PDF pp.2–4；正文 URL：<https://arxiv.org/html/2406.17276>）

**假设与机制**

对每个节点定义路径概率 p_hat：沿父节点到该节点的 draft 概率连乘。论文把 E(A)=sum p_hat 作为实际接受长度 A 的近似目标。Algorithm 1 每轮在最后一层取 p_hat 最大的 n 个候选，继续起草；当 Esub(T,n)-E 不超过阈值 delta，或深度达到限制时停止，再保留 p_hat 最大的 n 个节点。Theorem 3.1 说明子树节点的 p_hat 不高于祖先；Theorem 3.2 说明逐轮增加起草步时 Esub 单调不减。两条定理支撑贪心过程的单调性和树合法性，未证明 E(A) 等于真实接受长度，也未证明全硬件代价下的全局最优。（正文：§3、Algorithm 1、Theorems 3.1–3.2、PDF pp.2–4；正文 URL：<https://arxiv.org/html/2406.17276>）

**理论实际内容**

论文理论的核心是 draft 路径概率乘积作为接受长度代理量，以及由祖先概率更大推出的节点选择闭包。实际最关键的前提出现在文字说明中：有效草稿模型下 p_hat 与真实接受概率正相关。该相关性属于经验前提，正文 §4.2 用约 8000 个解码步记录 E(A) 与 A 的二维频数图来支持，图示相关不能替代真实最优性证明。（正文：§3、§4.2、PDF pp.2–3、5–6；正文 URL：<https://arxiv.org/html/2406.17276>）

**实验如何回答主张**

§4.1 在 LLaMA-2-7B/13B/70B、Vicuna-33B 及对应小草稿和 EAGLE 草稿上，用 4090、L20 与四张 A100，主要以 temp=0 的 MT-Bench、GSM8K 比较 Binary、EAGLE 和 OPT-Tree。表 1–2 的结果显示 OPT-Tree 在多数组合上提高平均接受长度，最高约 3.2x 相对自回归解码。§4.2 用约 8000 步验证 E(A) 与真实 A 的正相关。§4.3 与 Sequoia 比树规模，强草稿下超过 500 节点仍可提高平均接受长度；§4.4 说明接受长度最大化的树未必速度最高，阈值需要吸收起草成本；§4.5 的 temp=1 结果显示随机抽样下收益下降但仍有加速。实验因此区分代理目标、真实接受长度和 token/s。（正文：§§4.1–4.5、PDF pp.4–8；正文 URL：<https://arxiv.org/html/2406.17276>）

**局限**

论文承认硬件环境影响 token/s，实验沿用 EAGLE 解码框架。E(A) 与 A 只有经验相关；目标是 greedy 构树，随机抽样下这种相关变弱。节点更深会增加 draft 开销，接受长度高的树可能失去速度收益。论文没有研究 P/F 条件分布、恢复调用是否可避免，也没有把一组候选的状态复制和 KV 提交成本展开到类型生成任务。（正文：§§4.4–4.6、Limitations、PDF pp.7–9；正文 URL：<https://arxiv.org/html/2406.17276>）

**对 GaussianGPT 新颖性的威胁与可学习的论证组织**

若 GaussianGPT 只提出“按廉价分数给候选排序，再枚举数量”，OPT-Tree 已提供相近的在线树构造逻辑；若只报告候选覆盖和接受长度，也会落入其评价范围。可保留的差异在于恢复集合的价值定义包含真实拒绝事件、额外恢复前向和状态提交成本，且主抽样过程保持 W1。该差异需要在同候选、同 Target 树、同硬件上比较。可学习的组织方式是先定义代理量，再给单调性定理，最后用 E(A)-A 相关图、阈值消融和 token/s 分别验证，清楚说明理论代理没有直接等同真实收益。（正文：§3、§§4.2–4.5、PDF pp.2–8；正文 URL：<https://arxiv.org/html/2406.17276>）

### 5. EAGLE：特征级起草与树结构复用

**问题、缺口与适用定位**

EAGLE 观察到 token 级起草的随机性会使后续预测不稳定，提出在 Target 的第二高层特征上自回归起草，同时把提前一个时间步的 token 作为输入。Target 的 Embedding 和 LM Head 可以共享，草稿阶段预测 feature，验证阶段仍由 Target 一次前向处理树。适用定位是 P/F 项目中“隐藏状态或特征能否支撑廉价起草”的强先例；它不等于 GaussianGPT 的类型条件恢复状态方法。（正文：§§3.1–3.3、PDF pp.4–5；正文 URL：<https://arxiv.org/html/2401.15077>）

**假设与机制**

§3.1 的 feature&shifted-token 输入使用 feature 序列和提前一个时间步的 token 序列预测下一 feature，再经共享 LM Head 得到 token。§3.2 用 Smooth L1 训练 feature 回归，并在训练时加入 U(-0.1,0.1) 的 feature 噪声以缓和误差累积。§3.3 用 tree attention 一次取得树节点的 Target 概率，再对每个节点递归应用 speculative sampling；Appendix B.1 给出树布局，B.2 的 multi-round speculative sampling 在每次拒绝后把 p 更新为 norm(max(0,p-draft))，继续处理下一候选。（正文：§§3.1–3.3、PDF pp.4–5；Appendix B.1–B.2、PDF pp.12–13；正文 URL：<https://arxiv.org/html/2401.15077>）

**理论实际内容**

正文保证 EAGLE 沿用精确 speculative sampling，因此输出分布保持 Target；算法细节和递归残差在 Appendix B.2 展开。论文没有在主文给出独立于既有抽样方法的通用正确性定理，理论价值主要来自 feature 起草与标准验证的组合。其“feature uncertainty”解释由消融数据支持，不能写成一个已证明的 P/F 条件充分性结论。（正文：§3.3、§§4.1–4.3、PDF pp.5–7；Appendix B.2、PDF p.13；正文 URL：<https://arxiv.org/html/2401.15077>）

**实验如何回答主张**

§4.1 在 Vicuna 7B/13B/33B、LLaMA2-Chat 7B/13B/70B、Mixtral 8x7B 上，用 MT-bench、HumanEval、GSM8K、Alpaca 评估。LLaMA2-Chat 13B 的 temp=0 速度为 3.01–3.76x，temp=1 为 2.66–2.89x；平均每次 Target 前向生成约 3.2–4.5 token。HumanEval 的加速较高，Mixtral 8x7B 约 1.50x。§4.3.1 的 tree attention 消融让平均接受长度增加约 0.62–0.75；不使用树仍有约 2.3–2.7x 加速，说明特征起草和树验证的作用可分开。§4.3.2 显示 feature、token、shifted-token 输入之间的差异，提前一个时间步的 token 对随机性更有帮助；§4.4 显示 batch 增大后单请求加速下降，但最高吞吐约为自回归的 1.97–1.99x。实验支持特征起草、树注意力和输入消融，未评价 P/F 恢复位置的预计算收益。（正文：§§4.1–4.4、PDF pp.5–9；正文 URL：<https://arxiv.org/html/2401.15077>）

**局限**

EAGLE 主要测量语言模型延迟，草稿特征误差会沿自回归链传播；树注意力增加每次前向处理 token 数和显存压力。batch 增大后收益下降，MoE 的验证阶段还可能需要访问更多专家。Feature 与 token 的接口依赖目标模型内部表示，迁移到 GaussianGPT 的 FeatureState 需要独立验证，不能从名称相似推出条件分布相同。（正文：§§3.2、4.1、4.3.2–4.4、PDF pp.4–9；正文 URL：<https://arxiv.org/html/2401.15077>）

**对 GaussianGPT 新颖性的威胁与可学习的论证组织**

EAGLE 已把 Target hidden/feature、共享 LM Head、自回归 draft、树注意力和多轮残差串成完整流程。GaussianGPT 若把 Feature MLP 称作首个 feature-level speculation，或把共享 hidden/KV 计算称作新机制，均会遇到直接先例。可保留的定位是：GaussianGPT 的 Feature 条件由交替 P/F 语法决定，备用 b 代表拒绝后真实 Position 的恢复状态，研究重点是恢复状态的预计算和端到端成本；这与 EAGLE 的 feature draft 目标不同，仍需用去掉类型输入、去掉 hidden、统一 draft 器等对照证明。可学习的组织方式是机制图、输入消融、树消融、batch 结果和质量保证分开呈现。（正文：§§3.1–3.3、§§4.1–4.4、PDF pp.4–9、12–13；正文 URL：<https://arxiv.org/html/2401.15077>）

### 6. Traversal Verification：优先记录的近期验证工作

**问题、缺口与适用定位**

Traversal Verification 关注 token-level 树验证的两个问题：逐 token 接受没有利用序列联合概率，父节点拒绝会使其子节点丢弃。论文把验证顺序改为叶到根，并允许先处理完整后缀，再逐层决定父节点。适用定位是近期的验证算法和理论最优性参照；它与 GaussianGPT 的恢复分支相交在“候选树已有、验证顺序和残差转移可改变”，同时清楚表明仅做一个额外恢复节点无法承载通用验证算法贡献。（正文：§§2–3.3、PDF pp.2–7；正文 URL：<https://arxiv.org/html/2505.12398>）

**假设与机制**

Definition 3.1 把每条链上的子节点按 draft 分布自回归生成；Definition 3.2 要求验证器输出一条已接受前缀和一个额外 token，联合分布与 Target 下一步一致。Traversal Verification 先从叶到根计算序列级接受概率，再按递归残差把概率质量传给同层其他候选；Appendix F 将其解释为 sequence-level RRSw。该顺序让被拒绝父节点的其他子树仍有机会参与验证。（正文：§3.1–3.2、Algorithm 3、PDF pp.4–6；Appendix F、PDF p.18；正文 URL：<https://arxiv.org/html/2505.12398>）

**理论实际内容**

Theorem 3.3 证明树验证有效，也就是输出序列分布保持 Target。Theorem 3.4 在采样树退化为单链时，Traversal Verification 的期望接受 token 数等于 Block Verification，且不低于任意满足定义的有效链验证器。正文解释损失保持来自对子树规模的归纳；单链最优性来自给父节点加入伪子节点，把拒绝概率纳入 sequence-level RRSw。Appendix A 展开损失保持证明，Appendix B 展开单链最优性；这比仅引用标准抽样公式更接近完整的验证器理论，但最优性范围明确限于单链和定义内的有效验证器。（正文：§3.3、Theorems 3.3–3.4、PDF pp.6–7；Appendices A–B、PDF pp.13–17；正文 URL：<https://arxiv.org/html/2505.12398>）

**实验如何回答主张**

§4.1 使用 Spec-Bench 六类任务、Llama3.2-1B-Instruct 草稿与 Llama3.1-8B-Instruct Target，也补充 Llama-68M/Llama2-7B，在单 RTX A6000 上做三次独立运行。§4.2 的 Llama3 结果中，链、二叉树和 EAGLE 稀疏树的平均接受长度分别提升约 2.8%、2.2%、2.4%，token/s 提升约 2.5%、1.7%、2.1%；Llama-68M/Llama2-7B 组合提升约 5.7%、4.9%、3.8%，吞吐提升约 4.8%、3.7%、3.0%。§4.3 把链深度设为 2、4、6、8，把二叉树扩到深度 5，观察规模增大时优势上升；§4.4 在温度 0.2–1.0 上显示温度越高，Traversal 与 token-level 的差距越明显。Appendices C–D 还报告质量与统计口径，表明作者把理论无损与实测质量分开处理。（正文：§§4.1–4.4、PDF pp.7–9；Appendices C–D、PDF pp.17–18；正文 URL：<https://arxiv.org/html/2505.12398>）

**局限**

叶到根流程需要整棵树的 Target 概率，并可能引入额外遍历开销；正文承认接受长度收益高于真实吞吐收益。单链最优性不能直接推出多分支树的全局最优。方法研究的是验证器，依赖的树仍来自既有草稿器；它没有解决如何从廉价 P/F 信息选择哪些恢复位置，也没有研究图驻留、KV 搬移和 typed output head 的端到端代价。（正文：§4.2、§4.3、Appendix G、PDF pp.8–10、19；正文 URL：<https://arxiv.org/html/2505.12398>）

**对 GaussianGPT 新颖性的威胁与可学习的论证组织**

Traversal Verification 直接威胁“恢复分支让树验证更充分”“残差处理保证分布且提升接受长度”这类宽泛表述；它已把树级验证、序列级残差和单链最优性写成算法与定理。GaussianGPT 的 b 分支仍可定位为固定 W1 抽样过程中的额外恢复状态预计算，不改变接受路径，不做叶到根验证，也不追求树验证器最优性。可学习的组织方式是先给验证器定义，再给损失保持和最优性定理，最后用深度、树规模、温度和吞吐四组实验逐一回应主张。对 GaussianGPT 的比较必须明确它属于“状态计算安排”，Traversal 属于“验证抽样规则”。（正文：§§3.1–3.3、§§4.2–4.4、PDF pp.4–10；正文 URL：<https://arxiv.org/html/2505.12398>）

## 四、跨论文重合矩阵与当前定位

| GaussianGPT 操作 | 已有直接参照 | 重合程度 | 论文写法 |
|---|---|---|---|
| 拒绝后从 norm(max(0,p-q)) 抽样 | Leviathan；SpecInfer；EAGLE Appendix B | 直接重合 | 作为参考抽样语义，给出 P/F 条件下的实例化 |
| 主路径与备用 b 共享前缀，在一次 Target 前向中计算 | SpecInfer tree attention、DFS KV、topology-aware mask；EAGLE tree attention | 直接重合 | 描述为五节点 typed tree 和状态提交规则，不能称树注意力首创 |
| 抽到 b 后复用 b 的 Feature 分布、hidden、KV | 通用树节点命中后的候选状态复用；SpecTr 的成员命中复用下一条件分布 | 操作重合，语义有 P/F 特化 | 说明 b 是恢复事件对应的树成员，研究其调用节省与状态正确性 |
| 按预计收益选择候选和数量 | Sequoia 动态规划与硬件感知树选择；OPT-Tree 在线贪心树 | 目标层面重合 | 只有在恢复价值、信息时序和完整成本上给出可比新证据，才可保留选择问题 |
| 通过更换验证规则提高接受长度 | Sequoia 无放回抽样；Traversal Verification 叶到根验证 | 研究问题相邻，算法边界不同 | 当前沿用 W1 验证路径，不把验证器优化写成贡献 |
| 利用 Feature/hidden 作廉价预测 | EAGLE feature-level draft、Medusa 系列 | 机制层面重合 | 强调 GaussianGPT 的 P/F 条件链和恢复状态价值，补同资源输入消融 |

当前适用定位：

1. 已有实现可作为结构化三维生成上的恢复预计算实例，展示固定参考路径、类型条件状态、恢复命中和完整成本分解。
2. 标准抽样、树注意力、共享 KV、树结构选择和 feature-level draft 均有强先例；论文相关工作应主动承认这些继承关系。
3. 受限恢复集合选择仍是候选研究问题，现有六篇已覆盖其相邻工具。若选择器未在新状态上带来真实净收益，保留固定单备用和成本分析即可。

## 五、对论证组织的可复用方法

**先给自然方案的失败点，再限定问题。** Leviathan 先说明串行 Target 的成本，再给精确残差；Sequoia 先展示固定树在规模或温度上的问题，再把树构造和验证拆开；Traversal 先指出 token-level 验证浪费，再定义有效验证器。GaussianGPT 可沿用“拒绝位置需要新的 P/F 条件状态 → 现有树能计算该状态 → 固定参考抽样下选择预计算集合”的顺序。

**一个主张对应一组对象和一个实验。** SpecInfer 把树宽、树并行、MSS 分在 §6.4–§6.6；EAGLE 把 tree attention、输入、训练数据、batch 分开；OPT-Tree 把代理量相关性、树规模、阈值和温度分开。GaussianGPT 需要分别报告恢复命中质量、避免的 Target 恢复调用、额外树计算、KV 提交代价和完整请求墙钟。

**区分概率正确性、接受长度和实际速度。** 六篇论文都把三者放在不同位置：Leviathan 与 SpecInfer 的分布保持来自抽样规则；Sequoia、OPT-Tree 的树目标依赖接受率或路径概率；EAGLE 和 Traversal 的速度还受 tree attention、batch、硬件和遍历开销影响。GaussianGPT 的同输出、接受事件、恢复调用减少和端到端速度需要分别记录。

**把理论前提写入结论。** 独立同分布接受率、位置接受率、幂律条件、E(A) 与真实 A 的相关、足够并发资源、单链最优性都只在各自范围内成立。当前理论笔记里的 T1–T7 应按对应前提使用，不能把标准数学推论扩写为通用全局最优结论。

## 六、未读项与明确失败

- Dynamic Delayed Tree Expansion，<https://arxiv.org/html/2602.16994>：只取得 2026 arXiv 预印本线索和元数据，未读正文，不计入六篇证据。
- SpecTr、Grouped Speculative Decoding、SpecMER：由主 Agent 独立阅读，本文件只保留重合提示。
- 曾把 <https://arxiv.org/html/2305.16265> 误当作 SpecTr；读取后实际标题为 UNITE，标题与 ID 不符，相关内容全部弃用。已核对 SpecTr 的实际 ID 为 2310.15141，正文入口为 <https://arxiv.org/html/2310.15141>，不在本文件展开。
- ACM SpecInfer 页面返回 HTTP 403；采用 arXiv 正文和 DOI 元数据。
- MIT TACL OPT-Tree 页面返回 HTTP 403；采用 arXiv 正文和 DOI 元数据。
- Crossref 的部分标题检索返回 HTTP 429；改用精确 DOI、OpenAlex、arXiv 和会议页面。
- arXiv PDF 首次下载遇到 SSL 错误；重试成功，正文缓存读取日期为 2026-10-23。
- Research Router 的更新查询返回 cooldown；未把它当作论文证据，也未继续等待网站。

## 七、实际检索记录

### 成功检索词

- speculative decoding residual sampling Leviathan ICML 2023
- SpecInfer tree speculative inference verification
- Sequoia scalable robust speculative decoding
- OPT-Tree adaptive draft tree structure
- EAGLE speculative sampling feature uncertainty
- Traversal Verification speculative decoding
- speculative decoding token tree traversal verification
- speculative decoding dynamic tree expansion verification
- optimal tree speculative decoding theory sampling coupling
- speculative decoding tree expansion delayed verification
- 各论文标题配合 OpenAlex、Crossref、Google Scholar 的精确检索

### 成功原文 URL

- Leviathan 正文：<https://arxiv.org/html/2211.17192>；出版页：<https://proceedings.mlr.press/v202/leviathan23a.html>
- SpecInfer 正文：<https://arxiv.org/html/2305.09781>；出版 DOI：<https://doi.org/10.1145/3620666.3651335>
- Sequoia 正文：<https://arxiv.org/html/2402.12374>；出版 DOI：<https://doi.org/10.52202/079017-4116>
- OPT-Tree 正文：<https://arxiv.org/html/2406.17276>；出版 DOI：<https://doi.org/10.1162/tacl_a_00735>
- EAGLE 正文：<https://arxiv.org/html/2401.15077>；出版页：<https://proceedings.mlr.press/v235/li24bt.html>
- Traversal Verification 正文：<https://arxiv.org/html/2505.12398>；出版 DOI：<https://doi.org/10.52202/085713-2611>

本文件只新增于 docs/project/artifacts/paper_design_20261021/SD_PAPER_READING.md；论文临时 PDF 和网页缓存未纳入项目目录。

### 稳定原文位置

主代理完成整合后，将本轮已读六篇PDF从临时位置移入同一个Skill资料目录，未额外复制：

| 论文 | 本轮所读PDF |
|---|---|
| Leviathan | [05_Leviathan.pdf](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/05_Leviathan.pdf) |
| SpecInfer | [06_SpecInfer.pdf](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/06_SpecInfer.pdf) |
| Sequoia | [07_Sequoia.pdf](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/07_Sequoia.pdf) |
| OPT-Tree | [08_OPT_Tree.pdf](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/08_OPT_Tree.pdf) |
| EAGLE | [09_EAGLE.pdf](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/09_EAGLE.pdf) |
| Traversal Verification | [10_Traversal_Verification.pdf](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/10_Traversal_Verification.pdf) |
