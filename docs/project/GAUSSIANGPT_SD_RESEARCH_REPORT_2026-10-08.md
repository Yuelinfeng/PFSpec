# GaussianGPT 推理加速阶段性研究汇报（截至 2026-10-17）

## 从朴素投机解码到 P/F 分工及真实场景评价

整理日期：2026-10-17。范围覆盖 S2—S13 历史研究、P/F 多场景正式评价、Feature 候选成本分析，以及 Medusa/Hydra/EAGLE 启发的第二轮结构比较。早期路线按实际记录呈现；新增证据的来源与限制见各章链接。

### 摘要

GaussianGPT 把三维 Gaussian 场景编码为 Position 与 Feature 交替的离散序列。自回归生成需要频繁执行 Target 前向，投机解码由较便宜的草稿提出候选，再由 Target 验证并完成采样。早期完整 P/F 草稿和较大神经草稿多次出现“接受率提高、吞吐下降”；Position-only 的结构化查表在 BOS 轨迹上取得过 1.36149×，但该 runner 没有真实场景前缀。

随后，项目选定 Position Ranker、step12000 FeatureStateDraft 和静态 K2。阶段 E 在 30 个本轮正式测试场景上完成 2,160 次 FP32 固定窗口评价；轻量 PF 的微平均速度比为 1.69675×，来源组几何平均为 1.71250×，95% 区间 [1.68044, 1.74402]。阶段 F 将同一方法接入 VQ 解码、Gaussian 导出和六视角渲染；真实补全的场景等权完整时间比为 1.56935×，区间 [1.31784, 1.85226]。PF 相对 P-only 的完整时间比为 1.06968× [0.91223, 1.25177]，这项额外收益仍不确定。

10 月 17 日的候选成本研究显示，Feature 路径在相同开发前缀的固定窗口比较中较 P-only 快 1.171× [1.130, 1.211]；同前缀单轮推进效率比为 1.141× [1.085, 1.195]。Feature 收益随候选位置间隔与上下文而变，当前生成前特征还不能可靠预测净耗时，因此没有启用动态选择。Medusa/Hydra/EAGLE 启发的四种 Feature 结构也未超过旧轻量 PF：首训练种子中最快的 B1-E 为旧 PF 的 0.927× [0.905, 0.948]，B1-E 仅实现单步机制对照，尚未复现完整 EAGLE 架构。

当前结果支持保留 step12000 静态 PF-K2。固定窗口 Token 加速已有多场景结果；完整单 chunk 耗时相对 Target 也更短，额外胜过 P-only 的幅度尚未确定。全图 PSNR/SSIM 受随机续写影响，未建立可见补全区域质量、多 chunk 连续性或画质等效结论。

本文按研究问题组织章节，实验发生顺序见第三章。早期 P-only 探索与 S5 完整 P/F 负对照交错进行，章节次序不代表工作时间线。

## 第一章　前置知识：GaussianGPT 与投机解码

### 1.1 GaussianGPT 怎样生成三维场景

3D Gaussian Splatting 使用一组带位置、形状、透明度和外观参数的 Gaussian 表示场景。GaussianGPT 先用稀疏 VQ-VAE 将场景压缩成稀疏潜在网格，再把非空位置和对应的离散特征码排成序列。VQ-VAE 的作用是把复杂的连续场景参数转成较短的离散表示，并提供从离散表示返回场景的解码器。

```text
已有 Gaussian 场景 → 稀疏 VQ-VAE 编码 → P/F token 序列 → GPT 训练

生成条件或 BOS → GPT 自回归生成 P/F 序列
              → VQ-VAE 解码 → Gaussian 场景 → 渲染与质量评价
```

当前研究配置使用一个 Position token 和一个 Feature token 表示一个非空 latent：

\[
\mathrm{BOS},P_1,F_1,P_2,F_2,\ldots,\mathrm{EOS}.
\]

P 指定离散空间位置，F 指定该位置的特征码。一个 pair 通常对应一个非空 latent，不能直接理解为一个最终 Gaussian。P 与 F 的输出数量接近各一半；“P 接受率约 90%”描述候选是否被接受，与 token 数量占比无关。

| 对象 | 本项目配置 | 对推理的影响 |
| --- | --- | --- |
| Position | 20³ 网格，整体 ID 0—7999 | 有位置顺序及合法集合约束 |
| Feature | 4096 个码，整体 ID 8000—12095 | 在刚生成的 P 和全部历史条件下预测 |
| EOS / BOS | EOS=12096，BOS 复用该 ID | EOS 出现在 Position 预测位置，随后不再生成 F |
| PAD | 12097 | 不参与正常生成 |
| Target | 24 层、宽度 1024、16 个注意力头 | 上下文长度 16384 |
| D8 | 8 层、宽度 512、8 个注意力头 | 保持词表、序列约定和上下文能力一致 |

模型区分 P/F 的合法词表与输出计算，3D RoPE 将三维位置关系用于注意力中的位置编码。实现还依赖 token 元数据来恢复位置和执行位置限制。因此，修改生成顺序时必须同时维护模型输入、位置和缓存状态。[项目架构](ARCHITECTURE.md)给出更完整的数据链路；论文基础见 [GaussianGPT](https://arxiv.org/html/2603.26661v2)。

### 1.2 为什么普通自回归慢

普通自回归每生成一个 token，就执行一次 Target 前向。KV cache 保存此前 token 的注意力键和值，减少重复计算；它仍需按顺序生成下一个 token。生成 N 对 P/F 通常需要约 2N 次单 token Target 调用，长序列会反复支付模型执行和调度成本。

投机解码利用另一个较便宜的分布 q 提出多个候选，再让 Target 在一次因果前向中计算这些候选位置的条件分布 p。因果掩码保证每个位置只读取之前的 token；候选全部提供后，多行计算可以同时执行，但各行的条件历史仍然不同。

### 1.3 接受和拒绝后的采样为什么能保持分布

在同一历史下，先从 q 采样候选 y，以概率

\[
a(y)=\min\left(1,\frac{p(y)}{q(y)}\right)
\]

接受。若拒绝，则从

\[
r(x)=\frac{[p(x)-q(x)]_+}{\sum_z[p(z)-q(z)]_+}
\]

采样最终 token。接受贡献为 \(\min(p(x),q(x))\)，拒绝分支贡献为 \([p(x)-q(x)]_+\)，相加等于 p(x)。当 p=q 时拒绝概率为零，无需构造分母为零的残差分布。

这里 p、q 必须是实际采样使用的最终概率，包含各自的合法集合、temperature、top-k、top-p。拿原始 softmax 与经过过滤的候选混用，会破坏上述关系。分布一致也不要求同一 seed 下输出逐 token 一致，因为不同路径消耗随机数的顺序不同。

### 1.4 经典速度公式与适用条件

若候选长度为 γ，各位置接受率都近似为 α，单 token Draft 成本为 Target 的 c 倍，且一次整块验证约等于一次 Target 单步，则标准含额外 token 的模型给出：

\[
S_{\mathrm{SD}}\approx
\frac{1+\alpha+\cdots+\alpha^\gamma}{1+\gamma c}
=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+\gamma c)}.
\]

当 α→1，得到 \((\gamma+1)/(1+\gamma c)\)。公式假定验证和额外 token 的计算满足相应实现条件，且忽略长序列以外的边界影响；它不能直接预测本项目完整 wall-clock。出处为 [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/html/2211.17192v2)。第五章进一步区分完整 P/F 与当前单 P 算法的上限。

## 第二章　研究切入点：瓶颈在哪里

### 2.1 论文给出的任务耗时

GaussianGPT 论文 Appendix B、Table 5 在单张 A100 上统计 100 个样本，关闭 resampling，报告每个 chunk 的平均 wall-clock：

| 任务 | GaussianGPT | L3DG |
| --- | ---: | ---: |
| 无条件生成 | 78.1 ± 15.8 s | 23.8 ± 0.07 s |
| 25% context completion | 54.5 ± 13.8 s | 24.0 ± 0.18 s |
| 50% context completion | 38.9 ± 12.4 s | 24.1 ± 0.14 s |

已有上下文越多，需要生成的内容越少，GaussianGPT 的耗时相应下降。Appendix C 另报告约 12 m × 12 m 场景在 GH200 上平均需要约 6000 s；它包含大场景生成流程，不能与单 A100 的 chunk 数字直接比较。论文已经将 SD 列为未来加速方向，本研究的重点应落在 P/F 特有的接受行为、生成调度与提案成本上。[论文与附录](https://arxiv.org/html/2603.26661v2)

### 2.2 本项目能追溯到的 AR 分项

本轮找到本机导出的 CSV、元数据和原图，元数据标记上下文约 2048 token。下表保留 CSV 的每 token 统计；其列名为 `median_ms_per_token`，不能扩展为所有场景的平均值。

| 分项 | 每 token 时间 | 占 AR 窗口时间 |
| --- | ---: | ---: |
| Target forward | 13.17638 ms | 96.68279% |
| Sampling | 0.35757 ms | 2.62368% |
| 分项之外的剩余时间 | 0.09452 ms | 0.69353% |
| 合计 | 13.62846 ms | 100% |

![AR 窗口耗时构成：Target forward 占 96.68%](figures/gaussiangpt_sd_2026-10-08/ar_latency_breakdown.png)

图 1：由[本地 CSV 副本](figures/gaussiangpt_sd_2026-10-08/ar_breakdown.csv)重绘。forward 是原统计的阶段时间，可能包含阶段内同步；不能将同步时间再作为独立分项相加。原始远端 summary 的路径和资料范围见图表目录的[来源说明](figures/gaussiangpt_sd_2026-10-08/SOURCES.md)。导出件没有完整的 GPU、权重和 seed 配置，本轮未重新运行该测量。

2026-10-16 已读取远端 P1 原始五次运行：AR/Core 逐运行占比中位数为 **99.5653586%≈99.57%**。Core 包含 Prefill、AR、token 后处理与 VQ 解码，不含渲染。96.68% 来自独立的 2048 上下文窗口，与 P1 的统计范围不同；不将两个占比相乘，96.28% 不列为完整流程实测结果。

原元数据的缓存位置读取消融中，两项时间之差与另列的节省时间不一致，故本报告没有转用该消融注释。top-p 改为 greedy 的对照也改变了采样目标，不纳入同分布加速收益。

### 2.3 为什么选择 SD

该窗口中，单独消除采样及剩余开销，即使理想化为零，速度上限也只有约 \(1/0.9668279=1.0343\) 倍。大幅收益需要减少 forward 总成本。SD 可以用一次多行 Target 执行替代多次串行调用，同时通过概率接受机制保留 Target 采样分布。

研究目标因此明确为：**降低完成同一生成任务所需的 Target 串行执行成本，同时保持指定的 Target 采样规则。** 接受率、草稿成本、验证成本和拒绝后的额外调用需要共同优化。论文的 A100/GH200 时间与项目的 RTX 4090 解码结果分别解释，不直接换算绝对秒数。

## 第三章　总体方案与真实实验经过

### 3.1 修改的层次

研究集中在 GPT 的 Token 提案、Target 验证和缓存更新。VQ-VAE、场景表示及 Gaussian 解码沿用原系统。当前方法的主链路是“Ranker 提出 P1/P2，FeatureStateDraft 提出 F1，Target 批量验证，按类型完成接受或残差采样，再提交真实 Token 并更新缓存”。

| 研究方向 | 核心问题 | 主要方法 | 截至本次汇报的判断 |
| --- | --- | --- | --- |
| 神经 Feature 草稿 | Feature 分布与在线接受、计算成本如何共同影响吞吐 | Draft-A、D4/D8、FeatureStateDraft、B0/B1 结构比较 | step12000 轻量结构进入主方法；首轮 B0/B1 新结构均慢于旧 PF |
| 低成本 Position 提案 | 如何减少串行 Target 前向而控制提案成本 | D8 Position-only、结构化查表、Position Ranker | 查表是历史 BOS 轨迹结果；真实场景评价采用 Ranker |
| P/F 联合投机 | 如何让 Feature 接受转化为更多已提交 Token | 静态 PF-K2、不同 Feature 头及状态结构 | PF-K2 在阶段 E/F 快于 Target；相对 P-only 的完整流程增益区间仍跨 1 |
| 候选使用与条件选择 | 哪些上下文适合运行 Feature Draft 或追加候选 | A1–A3 成本测量、Feature 条件收益分析 | 有上下文差异；耗时预测不足以支持在线选择，暂保留静态 PF-K2 |

### 3.2 按实际记录排序的工作概览

| 阶段 | 记录时间 | 完成内容及研究意义 |
| --- | --- | --- |
| S2—S4 | 至 10-03 | 建立多种 Draft 尺寸；开展 Feature 蒸馏、类型诊断和温度对照 |
| S5—S7 | 10-04 | 完整 P/F 多对投机及短训对照多为负结果；探索输出头、F 旁路和真实联合解码 |
| S8—S11 | 10-04—10-06 | 实现结构化 Position 提案并与 D8 比较；提案成本会抵消接受收益 |
| S12—S13 | 10-06—10-07 | Position policy-KL、Draft 轨迹蒸馏及真实联合评价；S13.2 均值为 0.9104× |
| Draft-A v2、D4/D8 与真实输入评价 | 10-13—10-14 | 分别评价随机初始化顺序 Draft、D4 联合草稿、D8 与轻量 Feature 的真实输入表现 |
| 多场景阶段 E/F | 10-14—10-16 | 完成 30 场景固定窗口评价及 360 份自然 EOS 完整输出，确认 step12000 静态 PF-K2 为当前主方法 |
| 证据审阅 | 10-16 | 将源码、训练记录、逐次结果和阶段 E/F 统计对应起来，并复算已有数据 |
| 候选成本与 Feature 条件分析 | 10-17 | 完成 A1–A3 与 B1 分析；暂不训练独立 P2 Head，也未实现动态调度 |
| Medusa/Hydra/EAGLE 机制比较 | 10-17 | 完成 A0/A1/E0、四种 Feature 结构三训练种子、首种子在线比较及 A2 P1 短测；未形成替代主方法 |

早期 P-only 探索先于 S5 完整 P/F 负对照；本表按阶段汇总，不把章节顺序当作实验时序。完整历史材料见[阶段工作汇报](STAGE_REPORT_2026-10-08.md)、[第二轮执行记录](ROUND2_EXECUTION_2026-10-17.md)和[研究证据审阅](RESEARCH_EVIDENCE_AUDIT_2026-10-16.md)。

## 第四章　完整 P/F 投机的负结果与关键发现

### 4.1 草稿怎样建立

S2 的 canonical D8 学习完整 P/F 序列，配置为 38 epochs、每批 4 个 scene、梯度累积 4、bf16，优化器按参数分组使用 Muon 与 AdamW。D8 在 Position-only 中只负责 P 候选，并不代表它只接受过 P 训练。

S3 在 token 交叉熵之外加入 Feature KL，并用 Target 生成的轨迹训练 Draft。S5 的完整 P/F 路径让 Draft 连续生成 K 对候选，再由 Target 验证。若前部候选被拒，后部候选所依赖的历史随之失效，需要舍弃对应结果。Draft 猜出的 F 还会进入后续 P 的条件历史，这使类型差异影响整个候选块。

### 4.2 S5 的实际负收益

以下为 128-token 记录；速度比定义为路线 tokens/s 除以同次纯 Target tokens/s。

| K：候选 pair 数 | temperature | Target tok/s | 完整 P/F 路线 tok/s | 速度比 | 记录的总体 token 接受率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 49.408 | 26.353 | 0.533× | 0.487 |
| 2 | 1.0 | 50.691 | 28.924 | 0.571× | 0.398 |
| 4 | 1.0 | 49.675 | 21.553 | 0.434× | 0.213 |
| 2 | 0.1 | 未在本表来源单列 | 未在本表来源单列 | 0.795× | 0.630 |
| 4 | 0.1 | 未在本表来源单列 | 未在本表来源单列 | 0.891× | 0.563 |

以上接受率是该日志的总体统计，不能当作 αP 或 αF。现有汇总也没有给出足够的逐轮候选数与接受前缀长度，本文不反推“平均每轮推进长度”。降低 temperature 后仍未达到 1×，且 Target 分布已经改变。来源为[实验记录](EXPERIMENTS.md)及[阶段报告第 6 节](STAGE_REPORT_2026-10-08.md)。

这组结果说明当前草稿、块长度与实现没有抵消额外成本。它不能推出完整 P/F 投机在数学上不可行：在正确的候选历史上逐步计算 p、q，标准概率接受仍然适用。

### 4.3 P 与 F 的证据必须分开解释

早期 D8、seed=5、1024-token 诊断中，P 实际接受率为 97.27%，Feature overlap 为 0.3876。256-step Target 轨迹蒸馏使同长度的 F overlap 达到 0.7037，P 接受率为 98.24%，但成本公式估计仅从 1.1407× 变为 1.1424×。当 F 仍由 Target 生成时，提高 qF 匹配不会自动减少一次 Target 调用。

| 指标 | 数学或运行含义 | 本报告怎样使用 |
| --- | --- | --- |
| 分布 overlap | 同一历史下 \(\sum_x\min(p(x),q(x))\) | 在使用相同最终采样概率时等于该状态的期望接受概率 |
| F 旁路接受 | 提出 F 并统计，但最终 F 仍从 Target 采样 | 诊断草稿，不表示候选 F 已进入输出 |
| 真实联合 F 接受 | P 被接受后，实际接受并提交候选 F | 评价联合路线，分母为 P 已接受事件数 |
| 成对接受率 | P、F 均被接受的 pair 数 / pair 提议数 | 比较完整 pair 的通过能力 |

S13b 真实联合结果给出更直接的类型差异：P 接受 364/384=94.79%；在这些 P 接受事件中，F 接受 154/364=42.31%；完整 pair 接受 154/384=40.10%。高 P 接受无法独自保证整对通过。

### 4.4 连续接受长度的解释

定义 αP 为当前 P 被接受的概率，αF 为同一 pair 中 P 已接受条件下 F 被接受的概率，则同一统计集合内

\[
\rho=\Pr(P,F\text{均接受})=\alpha_P\alpha_F.
\]

若进一步近似认为沿候选块的分类型条件接受率固定，则从 P 开始：

\[
A_{2r}\approx\rho^r,\qquad
A_{2r+1}\approx\alpha_P\rho^r,
\]

其中 \(A_k\) 是前 k 个候选全被接受的概率。第一式中的跨 pair 幂次需要额外的稳定性近似；不能仅凭汇总接受率就认定真实长块满足该模型。

例如取 S13b 的汇总比例作说明，ρ≈0.401，则连续 2/4 个 pair 全通过的近似概率约为 16.08%/2.59%。它们是解释性推算，S13b 并未据此测出 K=2/4 的连续接受曲线。后续应直接记录 \(A_k\)，避免统一平均接受率掩盖 P/F 的交替差异。

## 第五章　两条路线的算法设计与理论上限

本章的路线 A/B 指 S2—S13 的早期研究路径；当前 PF-K2 的代码与结果见第六章 6.10—6.14。

### 5.1 路线 A：改善 Feature 草稿

这条路线保留完整 P/F 候选，优化 Draft 的分布质量。训练可概括为 token 交叉熵加分类型蒸馏项：

\[
\mathcal L=\mathcal L_{CE}
+\lambda_F\mathcal L_{KL,F}
+\lambda_P\mathcal L_{KL,P}.
\]

该式描述损失组成；各阶段实际计算的 raw/policy 分布和权重需分别读取配置。S3 重点增加 Feature KL；S12 引入 Position policy-KL；S13 改用 Draft 自生成历史，让 Target 在 Draft 更可能访问的前缀上提供训练目标。

实际路径为：采集轨迹 → 在相同历史上得到 Target/Draft logits → 按类型计算损失 → 选择权重 → 评价真实解码。只看验证 KL 不足以选出更快的草稿，还要看 αF、ρ、连续接受长度和每轮成本。S13b 在 512-step 训练预算内选中 step128 权重，其真实联合仍为 0.9104×，说明当前配方没有解决成本与接受的共同约束。

### 5.2 路线 B：单 Position 投机怎样保留 F 条件

早期 Position-only 路线每轮提出一个 P，并将前一轮尚未消费的 F 与候选 P 一起输入 Target：

```text
已提交历史，末尾 F 尚待 Target 消费
    → proposer 提出 P̂
    → Target 一次消费 [上一个 F 或 BOS, P̂]
    → 第一行给出真实 pP，第二行给出 pF(· | P̂)
    → P̂ 被接受：复用第二行采 F
    → P̂ 被拒绝：采最终 P，退回候选缓存，再计算该 P 对应的 F
    → 提交 [最终 P, Target F]，进入下一轮
```

下一轮 P 的 Target 分布依赖已提交的真实 F。单 P 方案已经通过这种调度处理条件依赖；一次猜多个非连续 P、忽略中间 F 的办法不属于这条历史实现。离线 K=4 位置命中评价也不等于在线一次验证四个 P。

### 5.3 两种 P 提案的区别

**D8 提案。** Draft 消费 BOS 得到第一轮 qP；每完成一对真实 P/F，Draft 一次消费 `[最终P,F]`，第二行提供下一轮 qP。候选 P 尚未提前写入 Draft 缓存，因此 P 被拒后 Draft 可以直接消费最终 pair。第一行 qF 可用于诊断，在 P-only 中不决定输出。

**结构化提案。** 从历史 P 提取相邻 ID 差值 gap，以最近至多 5 个 gap 和最后 P 的低 1 位选择计数表。计数被转换为候选位置概率 qP，无需 Draft Transformer 前向。F 仍由 Target 采样，下一轮 Target 继续读取真实 F。

查表 q 不读取 F 并不妨碍概率接受公式成立：q 可以是较粗的提议分布，只要已知实际 q，Target p 仍按完整历史计算。遗漏信息可能降低接受率，其影响应通过成本与速度衡量。

### 5.4 统一成本符号

以下先讨论 AR token 生成，忽略 EOS、初始化和末轮截断。设一对 P/F 的普通生成成本为

\[
B=T_P+T_F.
\]

\(D_K\) 是提出 K 对候选的总成本，\(V_K\) 是整块 Target 验证成本，H 是未归入模型前向的采样及调度成本。当前单 P 方案使用 D、V、C：D 为每轮提案或 Draft 更新的摊销成本，V 为 Target 双 token 验证，C 为 P 被拒后重新获得 F 分布的 Target 单步成本。

**“理论上限”需要同时说明：每轮最多提交多少 token，以及每轮执行成本假设。** 以下上限均属于指定成本模型，不是硬件无关的全系统速度承诺。

### 5.5 按 K 个 (P,F) 对投机的上限

若每轮最多提交 K 对，全部接受，不额外提交块外 token，则

\[
S_{PF,\mathrm{all}}(K)=\frac{KB}{D_K+V_K+H_K}
\leq\frac{KB}{V_K}.
\]

再假定 P、F 单步耗时相同，\(B=2T\)，且 \(V_K=T\)、草稿和其他成本为零，可得理想调用次数上限 **2K×**。若采用经典 SD 的额外 token 机制，γ=2K，且该额外 token 无需额外 Target 前向，则全接受时上限为 **(2K+1)×**。

| 候选 pair 数 K | 仅提交块内 pair，理想上限 | 经典含一个免费额外 token，理想上限 |
| ---: | ---: | ---: |
| 1 | 2× | 3× |
| 2 | 4× | 5× |
| 4 | 8× | 9× |
| 8 | 16× | 17× |

免费额外 token 是公式条件。某些实现会为拒绝恢复或全接受后的额外输出再运行一次模型；这些调用必须加到分母。一次推进奇数个 token 还会改变下一轮起始类型，不能在长期模型中始终假定从 P 开始。

在经典“拒绝后补一个 token、全接受后加一个 token”的规则下，令 \(A_k\) 表示前 k 个候选全通过，期望推进量严格写为

\[
G=1+\sum_{k=1}^{2K}A_k.
\]

采用第四章固定分类型条件率的近似，从 P 开始有

\[
G_P\approx1+\alpha_P(1+\alpha_F)
\frac{1-\rho^K}{1-\rho}.
\]

取 S13b 比例作纯理论示例：K=1/2/4 时 G 约为 2.349/2.890/3.194 token，即使零草稿、一次验证等于一次单步，也远低于全接受的 3/5/9。此处借用单 pair 的汇总率解释机制，未证明多 pair 实际达到该期望值。

历史 Target 合成块计时还有 K=1/2/4 约 22.76/23.26/24.76 ms。由 \(4\times22.76/24.76\approx3.677\) 得到的数字，比较的是以 K=1 块为单位的前向摊销效率。微测量采用短缓存及 3/5/9 行输入，未计真实候选生成和拒绝，故不作为完整解码上限，也不与其他实验的单步时间拼接。

### 5.6 单 Position-only 算法的上限

每轮正常提交一对 P/F，平均成本近似为

\[
T_{PO}=D+V+(1-\alpha_P)C+H,
\qquad S_{PO}\approx\frac{B}{T_{PO}}.
\]

全接受并保留 D8 成本时，\(S\leq B/(D+V)\)；再令提议成本为零，得到 \(S\leq B/V\)。若进一步 \(B=2T,V=T\)，理想上限为 **2×**。Position-only 实现每轮只提出一个 P，没有经典多候选块的额外推进能力。

历史阶段计时给出以下硬件成本模型估计。D 对应 D8 双 token 更新；零提议成本列只用于说明移除这项成本后的空间，查表实际仍有成本。

| 记录输出长度 | B：普通两步 ms | D：D8 更新 ms | V：双行验证 ms | C：拒绝重算 ms | αP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 26.7618 | 6.5024 | 16.8440 | 13.2853 | 498/512 |
| 2048 | 26.7277 | 6.4756 | 16.7586 | 13.1039 | 1000/1024 |
| 4096 | 28.7837 | 6.6688 | 17.6637 | 15.0392 | 1985/2048 |

| 记录输出长度 | D8、全接受、H=0 | D8、记录的 αP、H=0 | 零提议成本、全接受、H=0 | 零提议成本、αP=90%、H=0 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1.1463× | 1.1287× | 1.5888× | 1.4727× |
| 2048 | 1.1504× | 1.1354× | 1.5949× | 1.4792× |
| 4096 | 1.1829× | 1.1609× | 1.6295× | 1.5017× |

这组计算解释了为什么 D8 接受率接近 97% 仍只有有限收益：验证约 16.8—17.7 ms，Draft 更新还需约 6.5—6.7 ms。以该计时为条件，零提议成本的全接受上限约为 **1.59—1.63×**，D8 全接受上限约 **1.15—1.18×**。实际采样、调度及较低接受率还会消耗时间。这些数据来自同一阶段的三种长度计时记录，不能作为 S9 的精确性能预测。

### 5.7 从 AR 速度映射到完整生成流程

若 AR 在普通完整流程中的耗时占比为 f，其他部分不变，则

\[
S_{\mathrm{total}}=\frac{1}{(1-f)+f/S_{AR}}.
\]

例如令 f=99.57%、\(S_{AR}=2\)，则理论总速度为 1.9914×。2026-10-16 已找到 P1 对 f 的原始计时依据；AR 达到 2× 仍是假设，因此 1.9914× 不列为项目实测结果。96.68% 是 AR 内部独立窗口的 forward 占比，不能代替 f；若速度模型已计入完整 AR 时间，也不能再次乘 96.68% 作为额外加速。

## 第六章　关键实现、实验结果与产物

6.1—6.9 节整理 S2—S13 的 P-only 与早期联合实验；它们保留方法形成过程。当前选定系统及后续多场景、成本和结构结果见 6.10—6.14 节。

### 6.1 两条 P-only 路线的共同执行流程

以下伪代码对应已阅读的远端主引擎，省略性能记录和诊断字段：

```text
创建状态，output 为空
D8 路线消费 BOS 得到 qP；查表路线初始化位置历史
重复直到到达输出长度上限：
    从已提交历史构造实际 qP，采候选 P̂
    保存 Target 原长度 start
    Target 消费 [前一 F 或 BOS, P̂]，取得 pP 与 pF_given_P̂
    按 min(1,pP/qP) 接受；拒绝则从正残差概率采最终 P
    更新 proposer 的已提交位置历史，输出最终 P
    若 P 为 EOS 或已达长度上限：停止
    若 P 被拒：Target 长度退至 start+1，清掉旧 token 元数据
               再消费最终 P，取得对应 pF
    从 Target pF 采 F 并输出
    D8 路线消费 [最终P,F]，保留下一轮 qP
    查表路线仅更新历史和 token 元数据
返回 tokens 与统计
```

接受 P 时，Target 的第二行已经算出正确的 F 条件分布，无需再为 F 前向。拒绝时必须重新取得最终 P 对应的 F 分布，不能复用旧候选的第二行。

### 6.2 查表怎样从数据变成概率

建表语料为 Target 轨迹：32 条训练、4 条验证，每条 2048 sparse token。只用训练部分计数，共 32736 个训练 transition；验证为 4092 个 transition。对已提交位置 \(p_0,\ldots,p_j\)，定义 \(g_j=p_{j+1}-p_j\)。查询键使用已经出现的至多五个 gap，以及 \(p_j\mathbin{\&}1\)。

例如 P 历史 `[10,13,14,15,18,19]` 的键为 `((3,1,1,3,1),1)`。每个键保存“下一个 gap → 次数”，同时建立更短历史、按低位分组和全局计数。查询从最长可用历史逐级寻找非空行，没有对多行做权重混合。

选中行若为 `gap1:40, gap3:10`，last=19，则候选 P=20/22 的原始概率为 0.8/0.2。先排除非正间隔及词表之外的位置，再归一化。若所选行全部无效，直接尝试 last+1，再尝试 EOS，最后才对候选集合给均匀概率；此时不会重新查询更短历史。

正式 proposer 将原始概率转成 `log(max(q_raw,tiny))`，再执行 temperature、top-k、top-p，返回**实际采候选的最终 q**。接受函数也读取这份 q。首位置使用 last=-1 的通用规则；导出模型没有使用另行统计的首位置频次。普通 gap 行不直接统计 EOS，但拒绝残差中仍可采到 Target 的 EOS。

代码字段 `morton_low_bits` 的实际计算是 Position ID 按位与，没有调用 Morton 编码或交错 xyz 位。复现应遵循表达式，避免根据字段名另造空间编码。

### 6.3 缓存、词表及 EOS

完成 i 对输出后，output 长度为 2i；Target 已消费 `BOS…P_i`，末尾 F 等下一轮消费，Target 缓存长度也为 2i；D8 已消费到 F_i，长度为 2i+1。两条缓存差一来自调度。

拒绝且仍需生成 F 时，Target 长度退至 `start+1`，保留正确的上一 F/BOS，舍去候选 P。K/V 内存不必清零，逻辑长度排除了旧内容；token 元数据对应范围设为 -1，以免位置限制读到旧候选。

EOS 或 P 后达到长度上限时立即停止，无需补 F。终止瞬间的局部缓存不一定满足完整 pair 的关系；当前 API 不把这些缓存返回用于续写。未来增加续写能力时需要单独处理这条分支。

P 紧凑集合是 8000 个位置加 EOS，最后一项局部 index=8000，整体 ID=12096；F 集合是 4096 个 ID。采样后必须映射回整体 ID。temperature=0 使用 argmax；temperature>0 时按共享函数执行温度、top-k 和 top-p，其中 top-p 保留越过累计阈值的首项。

查表当前跳过 Draft 前向，但 runner 仍加载 D8 权重并分配相关缓存、维护元数据，因此不能将它描述成完全没有 Draft 显存占用。其查询还会重新扫描完整位置历史，h=5 仅限制表键长度，整个流程尚未变成恒定成本。

### 6.4 S8：离线命中与真实引擎接入

| 评价项 | 已有结果 | 解释 |
| --- | ---: | --- |
| h=5、bits=1 表项数 | 1325 | 来自已有轨迹计数 |
| 验证 mode top-1 | 3480/4092=85.04% | 选最大计数 gap 的离线命中 |
| K=4 连续命中均值 | 2.57969 | 离线位置预测能力，未转为在线多 P 验证 |
| 四个位置全命中比例 | 54.48% | 与真实概率接受率定义不同 |
| 纯 CPU 查询 | 约 0.7186 μs/query | 不含完整引擎的设备及概率处理 |
| 正式接口 256-token 运行 | 1.39922×；P=115/128 | 单 seed、短运行，证明接入与收益可能性 |

Python/tensor/vectorized 三种实现对同一输入的概率输出差为零；对应短运行速度为 1.36503/1.33324/1.33043×。该生成引擎每次只查一条，批量接口没有发挥其微测量优势。Python lookup 在一份 256-token 运行中仅占总时间约 0.63%，因此继续优化查表之前应先确定新的主要成本。

### 6.5 S9：早期 Position-only 路线的主要对照

记录设置为 RTX 4090、相同 Target/VQ-VAE/canonical D8、temperature=top-p=0.9、warmup=128、Python lookup，seed=0/1/2，四个请求长度。每条路线共 12 个结果。速度比保留源记录的逐运行比值聚合；“±”为历史汇总 std。

| 请求 token 数 | 查表速度比 | 查表 P 接受率 | D8 速度比 | D8 P 接受率 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.38617 ± 0.06483× | 93.750% | 1.04733 ± 0.02501× | 91.406% |
| 1024 | 1.37055 ± 0.01901× | 90.951% | 1.02437 ± 0.04618× | 87.565% |
| 2048 | 1.33807 ± 0.00462× | 90.527% | 1.02051 ± 0.03819× | 88.046% |
| 4096 | 1.35117 ± 0.00499× | 91.097% | 1.05654 ± 0.03299× | 88.323% |

![S9 速度对照：四个长度中结构化查表均值高于 D8](figures/gaussiangpt_sd_2026-10-08/s9_speedup_by_length.png)

图 2：由[历史汇总转录 CSV](figures/gaussiangpt_sd_2026-10-08/s9_summary.csv)重绘，点与误差线表示均值及记录的 std，非置信区间。D8 的后两个长度包含 seed2 在 1753 token 提前 EOS 的非等长运行。

查表 12/12 单元均大于 1×且完成请求长度，速度比总体均值 1.36149×；D8 均值 1.03719×，最低单元 0.96349×。目录名虽含 `multiscene`，实际模式为 `eos_seed_trajectory`，当前 runner 从 BOS 出发，没有独立场景或 prompt 输入。该结果支持不同随机轨迹和长度上的阶段判断。

“效果最好的两组”应具体表述为：**D8 Position-only 是主要神经草稿参照，结构化 Position 查表是 S9 历史矩阵中表现最明确的低成本路线。** 现有证据不足以对所有方法、独立场景和完整流程作统一排名。

### 6.6 S7—S13 神经草稿与混合策略的历史结果

| 阶段 | 完成的尝试 | 关键结果及判断 |
| --- | --- | --- |
| S7 | F-aware 128-step、旁路 F、真实联合 | 旁路 F 为 60/119；真实联合 F 为 51/121，仍有额外 Draft 探测成本 |
| S10 | eager 查表→D8 | 无回退也运行 D8，速度约 1.0137×，低于同阶段查表 1.3302× |
| S11 | lazy D8 回放 | 不回退为 1.37455×；较多回退时 P=127/128、速度降至 1.04909× |
| S12 | P/F KL 权重短训练与矩阵 | 7 个短训、24 个解码结果，速度差随 seed/长度改变方向 |
| S13a | canonical D8 的自生成历史蒸馏 | P 合并接受 91.41%→95.31%，F 旁路 46.72%→33.61% |
| S13b | 更长轨迹、λF=0.6、λP=0.3 | 512-step 预算，选 step128；旁路 F=34.26%，无稳定优势 |
| S13.2 | S13b 实际提交 P/F 候选 | P=94.79%，条件 F=42.31%，成对=40.10%，速度均值 0.9104× |

S13a 使用 8 条训练、2 条验证、每条 512-token Draft 轨迹，训练 128 step；S13b 从 S13a 所选权重继续，用 16/4 条、每条 1024-token 新轨迹训练。不同阶段的语料不同，不能把验证 KL 的下降直接解释为同一验证集上的改善。

S13.2 三个 seed 的真实速度为 0.8929/0.9206/0.9177×，均慢于纯 Target。S7 早期联合表的参考路线名称在旧记录中有歧义，本文不将其倍率用于精确跨阶段排名。S12 的部分 4096 请求长度不足且 EOS 标记不明确，同样不作为严格等工作量结论。

### 6.7 早期结果总表：每种数字能支持什么

| 方法与证据 | P 接受率 | F 指标 | 平均推进长度 | 实际速度结果 |
| --- | --- | --- | --- | --- |
| 普通 Target AR | 不适用 | 不适用 | 通常每次调用 1 token | 1×，各运行单独计时 |
| S5 完整 P/F，K=1/2/4 | 未分类型 | 总体接受 0.487/0.398/0.213 | 原始逐轮汇总待补 | 0.533/0.571/0.434× |
| S9 D8 P-only | 按长度均值 87.57%—91.41% | F 直接从 Target 采样 | 正常每轮输出 2；拒绝增加调用 | 12 个比值均值 1.03719× |
| S9 结构化 P-only | 按长度均值 90.53%—93.75% | F 直接从 Target 采样 | 同上 | 12 个比值均值 1.36149× |
| S13b 真实联合 | 94.79% | 条件 F=42.31% | 原始逐轮汇总待补 | 三种子均值 0.9104× |

不同阶段的长度、权重、采样参数和 runner 不完全一致。总表展示研究结果及定义，不能代替一份统一条件的最终性能榜单。

### 6.8 数值、状态、质量与计时范围

**运行与状态。** 早期路线及当前 K2 都有已记录的短运行、缓存状态和 EOS 行为验证。阶段 E 的分块/逐 Token 最大总变差为 Target 3.2708e-6、D8 3.5863e-6，包含缓存回退分支；这些有限样本结果支持已测数值路径，不覆盖所有可能上下文。

**概率语义。** 项目已有接受/残差采样的理论推导、小词表抽样与真实引擎数值记录。联合 50,000 次抽样最大绝对差为 0.00284。该证据支持对应配置的概率处理，不替代对每条长序列状态路径的穷尽证明。

**场景质量。** 阶段 F 已完成 360 份自然 EOS 输出、Gaussian 导出和六视角渲染，并保存全图 PSNR/SSIM。全图指标包含随机续写区域，尚无可见补全区域质量、LPIPS 或多 chunk 连续性结果；现有资料不能支持画质等效结论。

**计时。** S9 的历史 P-only runner 曾使用不同前向同步和输出计算范围，故其 1.36149× 仅作为历史 BOS 轨迹结果。阶段 E 与阶段 F 使用各自定义明确的当前路径，并分别报告固定窗口 Token 速度和完整单 chunk 时间，二者不可直接混算。各范围、时间组成和复算状态见 6.11—6.12 节。

### 6.9 实现入口和已有产物

下表路径均相对远端根目录 `/root/workspace/GaussianGPT`，源码依据来自此前授权的 JupyterLab 页面阅读；本轮未修改远端文件。行号和完整函数解释见[两条 P-only 实现汇报](POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)。该批源码阅读没有改动远端文件。

| 对象 | 相对路径 | 作用 |
| --- | --- | --- |
| 共同生成引擎 | `model/gpt/position_selective_speculative.py` | 双行验证、概率采样、缓存与停止条件 |
| 提案接口 | `model/gpt/draft_proposal.py` | D8/结构化 proposer 与状态 |
| 查表适配 | `model/gpt/structured_position_adapter.py` | 选表、合法位置、计数转概率 |
| 表构建 | `scripts/structured_position_lookup.py` | 读取轨迹、计数、离线评价与导出 |
| 对照入口 | `scripts/run_structured_position_e2e.py` | 用环境参数选择 D8 或 structured |
| 类型与位置处理 | `model/gpt/nanochat_gpt.py` | 位置顺序、3D RoPE、类型输出 |
| Target / VQ-VAE | `checkpoints/gpt_vfront.ckpt`、`checkpoints/vqvae_vfront.ckpt` | 主要模型权重 |
| canonical D8 | `profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt` | 主要神经草稿 |
| 结构化表 | `profiling/s8_structured_position/h5_b1/structured_zlow3_model.json` | 实际 h=5、bits=1，以内容为准 |
| S9 汇总 | `profiling/s9_multiscene/phase4_matrix_summary.json`、同目录 `phase4_matrix.csv` | 24 个路线结果的历史汇总 |
| S13 联合结果 | `profiling/s13_draft_prefix/e2e/joint_s13b_seed{0,1,2}.json` | 真实联合三种子结果 |
| 既有视觉产物 | `profiling/s8_structured_position/visual4096/orbit/gif/scene_000000__idx0000.gif` | 单轨迹渲染产物，当前未重新评价 |

runner 用 `STRUCTURED_PROPOSAL_MODE=structured` 或 `d8` 切换，结构化默认使用 Python lookup。复现还需要固定模型、采样参数、warmup、输出长度、EOS 和计时方式；不能只切换方法名就认为比较条件一致。以上为历史资产索引，供理解早期 P-only 实现与复现范围。

### 6.10 当前方法的代码、训练记录与输出对应

远端主工程为 `/root/workspace/GaussianGPT`。下表把当前运行入口与本地保留的源码副本、训练依据和结果文件对应起来；远端结果根目录写在对应行中。

| 功能 | 源码入口 | 训练或运行记录 | 结果及用途 |
| --- | --- | --- | --- |
| Position 候选 | [`draft_a/eval_lean_candidate_to_draft.py`](../../.codex/transfer/research_evidence_20261016/source/eval_lean_candidate_to_draft.py)、[`gpt_position_spec_decode_smoke.py`](artifacts/round2_20261017/source_snapshot/project/gpt_position_spec_decode_smoke.py) | Ranker 开发留出训练记录和速度分项 | Position Ranker 输出确定的单点候选；分类评分用于挑选位置，候选 q 为点分布 |
| FeatureStateDraft | [`draft_a/train_feature_state_distill.py`](../../.codex/transfer/research_evidence_20261016/source/train_feature_state_distill.py)、[`feature_target_v2.py`](../../.codex/transfer/research_evidence_20261016/source/draft_a/feature_target_v2.py)、[`feature_target_continue.py`](../../.codex/transfer/research_evidence_20261016/source/draft_a/feature_target_continue.py) | 400 条训练轨迹、50 条验证轨迹；训练 12,000 步后选定 `draft_a/runs/feature_target_continue_20261014/selected.pt` | 2,864,384 个可训练参数；验证重合度从 26.4062% 到 26.7549%，新增在线吞吐收益区间跨 1 |
| 统一 PF-K2 验证 | [`draft_a/pf_feature_unified.py`](../../.codex/transfer/research_evidence_20261016/source/draft_a/pf_feature_unified.py)、[`bench_gaussiangpt_pf_block2_cow.py`](artifacts/round2_20261017/source_snapshot/project/bench_gaussiangpt_pf_block2_cow.py) | 多场景入口 [`draft_a/pf_multiscene/run_final_evaluation.py`](../../.codex/transfer/research_evidence_20261016/source/draft_a/pf_multiscene/run_final_evaluation.py)；方法配置在 `stage_e_final/method_selection.json` | 负责 Ranker、MLP、Target 四行验证、接受/残差采样和缓存恢复；阶段 E 结果位于 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/` |
| 完整三维输出 | [`draft_a/pf_multiscene/full_3d_eval.py`](../../.codex/transfer/research_evidence_20261016/source/draft_a/pf_multiscene/full_3d_eval.py) | 30 场景三种子真实补全及 30 组 BOS 种子 | `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/` 保存 Token、Gaussian NPZ、PNG、逐样本 CSV 与汇总 |
| 候选边际成本 | 本地[`run_cost_study.py`](../../patches/candidate_cost_20261017/run_cost_study.py)与[`instrumented_decode.py`](../../docs/project/artifacts/candidate_cost_20261017/instrumented_decode.py) | `runs.jsonl`、`matched_rounds.jsonl`、同步子集及交错运行记录 | `/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/`；详细解释见[A1–A3 成本报告](artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md) |
| Feature 条件收益 | 本地[`collect_feature_value.py`](../../patches/feature_value_20261017/collect_feature_value.py)与[`analyze_feature_value.py`](../../patches/feature_value_20261017/analyze_feature_value.py) | `paths.jsonl`、`pairs.jsonl`、`analysis.json` | `/root/autodl-tmp/gaussiangpt_feature_value_20261017/`；详细解释见[Feature 条件收益报告](artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md) |
| 第二轮结构比较 | 本地[`b1_feature_models.py`](../../patches/round2_20261017/b1_feature_models.py)、[`b1_train.py`](../../patches/round2_20261017/b1_train.py)与[`b1_online.py`](../../patches/round2_20261017/b1_online.py)；实际运行源码副本位于 `artifacts/round2_20261017/source_snapshot/` | A0/A1/E0 诊断、三种子训练摘要、392 条首种子在线比较及 A2 P1 短测 | 逐条资料、训练记录、日志和限制见[第二轮执行记录](ROUND2_EXECUTION_2026-10-17.md)及其[产物目录](artifacts/round2_20261017/) |

阶段 E 与阶段 F 的复算汇总和 CSV 副本见[研究证据审阅](RESEARCH_EVIDENCE_AUDIT_2026-10-16.md)及[`research_evidence_20261016` 产物](artifacts/research_evidence_20261016/)。远端权重没有复制到本地文档包；表中模型路径均保留为远端记录。

### 6.11 阶段 E：固定窗口多场景 Token 速度

阶段 E 使用 FP32、temperature/top-p=0.9、64/256/1024 Token 前缀、最多 512 个新增 Token。30 个本轮正式测试场景产生 56 条唯一裁剪，共有 2,160 条记录：五种模式、三个 seed、全部可用前缀组合。本轮开发、适配和正式测试来源组没有交集；Target/D8 预训练来源 ID 记录不完整，因此此处的独立范围仅指本轮数据划分。计时包含前缀计算、缓存、提案、Target 验证、拒绝处理和采样，不含 VQ 解码、Gaussian 导出与渲染。

| 模式 | P 接受/已验证 | F 接受/已验证 | Token/s | 微平均速度比 | 来源组几何平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target | — | — | 70.66882 | 1.00000× | 1.00000× |
| P-only | 92.44076% | — | 101.40326 | 1.43491× | 1.43773× |
| 轻量 PF-K2 | 92.64095% | 33.25815% | 119.90716 | **1.69675×** | **1.71250×** |
| PF-D8 | 92.50658% | 55.47118% | 103.00437 | 1.45756× | 1.46646× |
| GPT-D8 | 78.96773% | 55.52214% | 66.44308 | 0.94020× | 0.93815× |

轻量 PF-K2 的来源组 95% 区间为 [1.68044, 1.74402]，30/30 来源组快于 Target。该区间对应来源组几何平均，不附到 1.69675× 微平均值。PF-D8 与 GPT-D8 的 Feature 接受率高于轻量 PF，整体速度仍更低；接受率不能单独代表净速度。

原始逐次数据为远端 `stage_e_final/fp32_test_full/online.json` 和 `multiscene_summary.json`；本地副本见[`online.json`](../../.codex/transfer/research_evidence_20261016/runs/stage_e_final/fp32_test_full/online.json)与[`multiscene_summary.json`](../../.codex/transfer/research_evidence_20261016/runs/stage_e_final/fp32_test_full/multiscene_summary.json)，重算表见[`stage_e_modes.csv`](artifacts/research_evidence_20261016/stage_e_modes.csv)与[`verification.json`](artifacts/research_evidence_20261016/verification.json)。2,160 行复算与原汇总一致，最大绝对差为 0，无重复的裁剪、前缀、seed 与模式组合。区间沿用原汇总文件中的记录值。

### 6.12 阶段 F：完整单 chunk 输出

阶段 F 以 256 Token 真实前缀、最多 16,000 个正文 Token、FP32 和 temperature/top-p=0.9 生成。真实补全覆盖 30 个场景、每场景三个 seed 与三种模式，共 270 份；BOS 另有 90 份。所有 360 份输出均自然 EOS，均完成 Token 保存、Gaussian NPZ 导出和六视角 PNG 渲染。

| 真实补全模式 | 完整时间均值 | 平均新增正文 Token | 场景等权时间比（Target/模式） | 95% 场景重采样区间 |
| --- | ---: | ---: | ---: | --- |
| Target | 46.06736 秒 | 3,050.18 | 1.00000× | — |
| P-only | 31.36916 秒 | 2,987.31 | 1.46713× | [1.26583, 1.73137] |
| 轻量 PF-K2 | 29.52030 秒 | 3,086.49 | **1.56935×** | **[1.31784, 1.85226]** |

该比值先在同场景、同 seed 下计算，再在场景内对三个 seed 取几何平均，最后对 30 个场景等权。PF-K2/P-only 为 1.06968× [0.91223, 1.25177]，区间跨过 1；因此阶段 F 支持 PF-K2 快于 Target，尚不足以确认它完整流程上稳定快于 P-only。全样本总耗时比、阶段 E 固定窗口速度与阶段 F 场景等权完整时间属于不同统计口径。

完整时间包括 AR、VQ 解码、Gaussian 导出、六视角渲染和 PNG 保存，不计模型加载、预热与参考重建。全图 PSNR/SSIM 均值为 Target 11.40815 dB/0.73016、P-only 11.66826 dB/0.73827、PF-K2 11.45588 dB/0.73214；随机续写区域也参与这些数值。局部补全质量与多 chunk 连续性仍待单独评价。逐样本表为[真实补全 CSV](artifacts/stage_f_full3d/analysis/test_samples.csv)、[分场景 CSV](artifacts/stage_f_full3d/analysis/test_scenes.csv)和[BOS CSV](artifacts/stage_f_full3d/analysis/bos_samples.csv)；预览见[阶段 F 输出目录](artifacts/stage_f_full3d/analysis/)及[完整三维输出记录](FULL_3D_OUTPUT_2026-10-15.md)。

### 6.13 候选成本 A1–A3 与 Feature 条件收益

A1–A3 使用十个开发来源、28 个来源与前缀组合、两个运行 seed，每次最多新增 256 Token。56 次 PF、56 次 P-only 和 56 次带分项记录用于固定窗口比较。另在 56 个相同上下文上各跑八个 seed 的 PF/P-only 单轮对照，共 896 次。固定窗口 PF/P-only 来源等权速度比为 **1.171× [1.130, 1.211]**；同前缀单轮推进效率比为 **1.141× [1.085, 1.195]**。单轮比值描述这一轮动作，不替代长期生成结论。

P2 被拒绝 113 次，非 EOS 回退补算 113 次；含一次前缀调用在内的 PF Target 调用为 5,875 次。P2 拒绝后的整段处理占带分项复测时间的 1.337%。假设把整段处理耗时完全删除且其他路径不变，时间相减估计为 1.0136×。这个估计省略新方法成本、输出分布变化及其余路径变化，只描述当前观测范围。它不支持训练独立 P2 Head 的优先级。

Feature 条件收益报告沿用这些低干扰速度数据，另补采 5,078 个含完整 P1/F1/P2 候选的轮次标签，并重放 448 次相同前缀单轮试验。28 个真实起始前缀与 28 个 PF 中间前缀各占一半。首个 P 候选间隔为 1 的分组，PF/P-only 为 1.190× [1.117, 1.261]；间隔大于 1 的分组为 0.970× [0.934, 1.015]。来源留一预测中，仅用生成前特征估计 Feature 接受概率的 R² 为 0.264；增加 Feature 草稿统计后为 0.509，但草稿统计需先支付运行成本。额外时间预测 R² 为 −2.695，误差高于均值参照。当前数据显示收益依上下文而变，未形成可用的在线选择规则。原始报告、分析表、重放数据和脚本分别位于[Feature 条件收益产物目录](artifacts/feature_value_20261017/)与[成本产物目录](artifacts/candidate_cost_20261017/)。

本文中 A1–A3 指候选边际成本研究；Feature 收益条件分析亦简称 B1。它与下节 Medusa/Hydra/EAGLE 结构实验中的 B1-M、B1-H、B1-E 使用不同命名。

### 6.14 Medusa/Hydra/EAGLE 启发的第二轮结构比较

第二轮复核了实际 P1 输入边界、Rank 与 Position ID 对应、隐藏状态归一化及 EAGLE 保存精度。Medusa/Hydra 的既有 P1 头从最后已处理 Position 状态预测下一 Position，在线 Target 还会处理待提交的历史 Feature；448 个固定真实上下文上的重合度分别为 1.566% 和 1.407%。十个缓存与整段样本的 Target 分布最大总变差为 0.000785。EAGLE 权重以 FP32 保存，训练使用 BF16 AMP；16,360 行复算的完整 softmax 重合度约 3.705%，top-p=0.9 后约 2.776%。这些数字分别描述输入条件、状态一致性和草稿质量，不能混作在线速度。

在共享 Position Ranker、K2 验证循环、训练轨迹和采样条件下比较 B0、B1-M、B1-H、B1-E。四种结构各用三个训练 seed；完整在线速度只对首训练 seed 进行了 392 条 FP32 开发比较。

| 方法 | 可训练参数 | Feature 接受/已验证 | 微平均 Token/s | 相对 Target 来源组速度比 [95% 区间] | 相对旧 PF |
| --- | ---: | ---: | ---: | --- | --- |
| Target | — | — | 70.414 | 1.000× | 0.616× [0.594, 0.642] |
| 旧 step12000 PF | 2,864,384 | 34.42% | 113.959 | 1.624× [1.557, 1.682] | 1.000× |
| B0，同结构重训 | 2,864,384 | 16.88% | 101.562 | 1.442× [1.401, 1.475] | 0.888× [0.859, 0.912] |
| B1-M，独立 Feature 头 | 3,087,872 | 9.40% | 97.563 | 1.386× [1.363, 1.404] | 0.853× [0.827, 0.881] |
| B1-H，条件 Feature 头 | 4,177,664 | 9.00% | 98.541 | 1.399× [1.366, 1.429] | 0.861× [0.833, 0.887] |
| B1-E，单步特征机制 | 1,317,632 | 26.94% | 106.025 | 1.506× [1.450, 1.558] | 0.927× [0.905, 0.948] |

首训练 seed 下，四个新结构均在 10/10 开发来源慢于旧 PF。B1-E 是新结构中最快者，Feature 接受率及固定上下文重合度也高于另外三种结构；其前向约 0.828 ms/次，旧 MLP 约 0.470 ms/次。模型大小不能替代实际 GPU 成本。其余两个训练 seed 只完成 448 个固定真实 Feature 上下文的离线重合比较，没有重复完整在线吞吐，因此训练 seed 间的速度稳定性未知。

B1-E 使用已知历史 Feature 与候选 Position 的两 Token 因果块，再经共享 Target embedding 与 Feature 输出头；它没有跨轮草稿缓存，完整 EAGLE 架构尚未复现。A2 把历史 Feature embedding 输入 P1 头后，448 个上下文重合度由 1.610% 增至 1.682%；三来源、每法三次的短测中，条件头吞吐低于 Target，暂不扩大独立 P1 训练或进入完整三维确认。训练记录见[三种子摘要](artifacts/round2_20261017/B1_training_summary_20261018.json)、[在线汇总](artifacts/round2_20261017/b1_online/20261018/summary.json)与[在线日志](artifacts/round2_20261017/b1_online.log)；源码副本、逐条结果和完整限制见[第二轮执行记录](ROUND2_EXECUTION_2026-10-17.md)及[`round2_20261017` 证据目录](artifacts/round2_20261017/)。

## 第七章　下一步方向与阶段结论

### 7.1 当前选择及仍需补足的证据

当前主方法保留 Position Ranker + step12000 FeatureStateDraft + 静态 PF-K2。阶段 E 支持其在 30 个场景的固定窗口 Token 生成中快于 Target；阶段 F 支持其完整单 chunk 时间快于 Target。相对 P-only 的完整时间区间跨过 1，论文表述应保留这一差异。

10 月 17 日成本测量显示，P2 拒绝处理的可节省空间有限；Feature 条件分析又显示上下文能影响净收益，但可用的生成前信息尚不能预测完整时间。当前不启用 P2 独立头或在线切换。第二轮新 Feature 结构的首 seed 吞吐均低于旧 PF，B1-E 仍缺跨训练 seed 在线速度和新来源确认；不据离线重合度推进正式方法替换。

仍待开展的工作限于能改变研究结论的部分：对真实补全可见区域补充质量指标；评价多 chunk 边界连续性；如未来要选用新 Feature 结构，再在独立来源补齐完整在线速度和三维输出。无需重复已完成的阶段 E/F、A1–A3 或首轮第二轮比较。

### 7.2 本阶段形成的认识

第一，减少 Target 串行调用是有效方向。早期 S5 完整 P/F 草稿、D8 联合解码及部分更大 Draft 受到 Feature 拒绝或提案成本影响；阶段 E/F 则证明低成本 PF-K2 在当前设置下有明确的 Target 相对速度收益。

第二，Position 与 Feature 的接受和计算价值不同。阶段 E 的 PF-D8、GPT-D8 有更高 Feature 接受率，却没有更高总体速度；第二轮 B1-E 也展示较高 Feature 接受与较低 Feature 前向成本之外的实际代价。应同时给出接受分母、推进 Token、Target 调用和整段耗时。

第三，固定 PF-K2 是现有数据支持的选择。它在阶段 E 的来源组几何平均为 1.71250×，阶段 F 的场景等权完整时间比为 1.56935×。这些数字对应不同计时范围；完整 PF 相对 P-only 的额外耗时收益还没有被确认。

第四，接受概率差异尚不足以推出动态策略。Feature 利益确有上下文变化，生成前的耗时预测质量较弱，P2 分支的成本空间也较小。保留静态配置有更直接的实测支持。

第五，代码、训练选择、逐次运行和输出资产现在有对应索引。关键入口为[`研究证据审阅`](RESEARCH_EVIDENCE_AUDIT_2026-10-16.md)、[候选成本 A1–A3](artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md)、[Feature 条件收益分析](artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md)及[第二轮结构比较](ROUND2_EXECUTION_2026-10-17.md)。早期结构化查表 1.36149× 保留为 BOS 轨迹历史结果，不替代真实前缀和完整场景证据。

本汇报不把完整单 chunk 的速度结果外推到多 chunk，也不把全图 PSNR/SSIM 解释为可见补全区域的质量结论。

---

### 资料与阅读入口

- [GaussianGPT 原论文与附录](https://arxiv.org/html/2603.26661v2)：模型背景和任务耗时。
- [经典 Speculative Decoding 论文](https://arxiv.org/html/2211.17192v2)：接受机制与理想速度公式。
- [阶段工作汇报](STAGE_REPORT_2026-10-08.md)：完整 S2—S13 时间线、实验数据及资产索引。
- [研究证据审阅](RESEARCH_EVIDENCE_AUDIT_2026-10-16.md)：源码、训练权重选择、阶段 E/F 逐次结果和复算依据。
- [候选成本 A1–A3](artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md)与[Feature 条件收益分析](artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md)：10 月 17 日的提案成本、同前缀效益和预测能力。
- [第二轮执行记录](ROUND2_EXECUTION_2026-10-17.md)：Medusa/Hydra/EAGLE 启发的结构比较、训练摘要、在线测量和限制。
- [D8 与结构化 Position 实现汇报](POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)：已阅读源码的函数、状态及入口说明。
- [论文证据索引](PAPER_EVIDENCE.md)：结果的统计口径、可表述范围与新增成本分析。
- [实验记录](EXPERIMENTS.md)：历史配置、结果与阶段问题。
- [图表来源与重绘说明](figures/gaussiangpt_sd_2026-10-08/SOURCES.md)：本地 CSV、图像和计算范围。

本次汇报整合已有研究和实验产物；未新增模型训练、GPU 性能测量或场景渲染。
