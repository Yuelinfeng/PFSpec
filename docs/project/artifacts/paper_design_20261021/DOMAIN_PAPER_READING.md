# 补充正文研读：理论论文与结构化生成论文

日期：2026-10-23。本记录由主代理直接阅读原文。学习重点是问题如何导出机制、证明究竟覆盖什么、实验如何验证主张。会议论文可以采用较简单的数学工具；说服力取决于问题识别、机制必要性和证据对应，不能用定理数量衡量。

## 原文与身份

| 编号 | 论文标题 | 作者 | 年月与身份 | 原文支撑 | 链接 |
|---|---|---|---|---|---|
| 1 | SpecTr: Fast Speculative Decoding via Optimal Transport | Ziteng Sun、Ananda Theertha Suresh、Jae Hun Ro、Ahmad Beirami、Himanshu Jain、Felix Yu | NeurIPS 2023；所读arXiv v2为2024-01 | §3–6、Algorithm 2、Theorem 1；§4明确候选成员可复用下一条件分布；PDF p.4–9 | [Official PDF](https://arxiv.org/pdf/2310.15141) / [Page](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html) / [Local PDF](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/01_SpecTr.pdf) |
| 2 | Grouped Speculative Decoding for Autoregressive Image Generation | Junhyuk So、Juncheol Shin、Hyunho Kook、Eunhyeok Park | ICCV 2025，2025-10 | §3.1–3.2诊断，§4 Theorem 1，§4.1动态聚类，§5质量与速度；出版页15378–15382，对应PDF p.4–8 | [Official PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/So_Grouped_Speculative_Decoding_for_Autoregressive_Image_Generation_ICCV_2025_paper.pdf) / [Page](https://openaccess.thecvf.com/content/ICCV2025/html/So_Grouped_Speculative_Decoding_for_Autoregressive_Image_Generation_ICCV_2025_paper.html) / [Local PDF](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/02_Grouped_SD.pdf) |
| 3 | SpecMER: Fast Protein Generation with K-mer Guided Speculative Decoding | Thomas A. Walton、Darin Tsui、Aryan Musharaf、Amirali Aghazadeh | NeurIPS 2025；会议身份由官方页与PDF页脚确认，发现页给出的2026年份未采用 | §3.1–3.3、Proposition 4.4，PDF p.5–6；§4.2–4.3与局限p.7–10 | [Official PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/18ab6eac1334e336a54fc81cad539bf5-Paper-Conference.pdf) / [Page](https://proceedings.neurips.cc/paper_files/paper/2025/hash/18ab6eac1334e336a54fc81cad539bf5-Abstract-Conference.html) / [Local PDF](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/03_SpecMER.pdf) |
| 4 | Coupling without Communication and Drafter-Invariant Speculative Decoding | Majid Daliri、Christopher Musco、Ananda Theertha Suresh | ISIT 2025，2025-06；作为相关理论工作，不替代ICML／NeurIPS学习样本 | §1、§3 Theorem 3、§5；共享随机性固定Target输出，比较耦合碰撞概率 | [Official PDF](https://arxiv.org/pdf/2408.07978) / [DOI](https://doi.org/10.1109/ISIT63088.2025.11195663) / [Local PDF](/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/04_Drafter_Invariant.pdf) |

前三篇PDF的对应定理页另经页面渲染阅读，避免只依据文本抽取中的公式残缺。第四篇使用公开HTML正文与Crossref会议元数据，并补读PDF p.1、3、10；未逐页审阅全文。其PDF v4首页说明Gumbel耦合相应结果另有更早工作，故本记录不把该结果的首创权归给此文。

## 1. SpecTr：问题扩展必须带来新的数学障碍

### 作者的问题链

§3用三个问题组织理论：单候选投机与统计抽样是什么关系、是否最优、多候选是否能增加并行收益。§4把单token的选择写成保持边缘分布的耦合，再将多个草稿写成带membership cost的离散最优传输。这里的难点在于同时满足最终样本分布正确和高候选覆盖概率。

§4给出具体失败例：把同一接受规则分别应用于多个候选，会增加某些输出的概率，破坏Target边缘分布。这使后面的新选择算法具有必要性。§5指出精确线性规划的变量随候选数指数增长；§6给出可计算的K-SEQ及有效传输计划、近似保证。§7再把单步结果递归用于完整自回归采样。§8区分block efficiency与实际墙钟，明确硬件模型是理想化的。

### 理论实际内容

Theorem 1在相应参数条件下证明K-SEQ保持有效耦合，并达到最优接受概率的常数比例；计算复杂度另给出。近似保证针对所定义的候选分布与membership目标，不能直接转成所有硬件上的速度保证。

### 对我们的直接影响

§4 PDF p.5原文："when Y ∈ {X1, . . . , Xk}, we could sample an extra token ... without calling ... since we have already computed the conditional probabilities"。它已经包含候选成员命中后复用下一条件分布的思想。

我们的单备用固定主候选与原残差抽样，额外预计算恢复位置；这个限制可能让选择问题更简单，也保留W1随机轨迹。但这种限制本身不证明创新。需与SpecTr的候选构造、耦合改变和目标逐项比较，解释为何固定参考抽样对当前问题有实际价值。不要写成首个拒绝恢复复用方法。

### 可借鉴的组织

先给无效的自然算法及反例，再定义必须满足的约束；证明与计算复杂度分别回答“合法”和“可用”；最后回到原任务。我们的理论应同样给出会改变抽样的反例，以及会造成成本选择失败的反例。SpecTr还保留了未改善结果的prefix-tree扩展（§7、Appendix D），说明负结果可以限定方法选择。

## 2. GSD：诊断、简单定理与机制形成一致解释

### 作者的问题链

引言之后单设§3诊断视觉token：低top-1概率、高总变差、不同视觉区域、token替换与解码上下文。§3.2引用标准关系α=1−TV(p,q)，用真实分布和玩具例说明“多个视觉上合理的选择”仍可带来较低接受率。§4把词表划为组，按组概率执行接受，Theorem 1用三角不等式证明分组总变差不会增大。§4.1又通过不完整图像解码说明静态嵌入聚类不足，提出依赖上下文概率的聚类。§5用聚类对照和质量—计算曲线支持机制。

### 应学习的地方

一条很简单的不等式能够承担设计解释，因为作者先给出了它要解决的具体分布现象。数学深度与公式长度没有直接关系。我们也应先提供P/F条件链中可复核的差异，再判断什么理论能解释差异；不能先命名一个框架，再寻找填充公式。

### 不应继承的主张

Theorem 1证明接受概率不降低；它不证明token级Target分布保持，也不保证真实速度必然提高。§5将本文方法放在放宽精确抽样的类别，质量依赖FID、CLIP等实测。将接受概率提高直接解释成墙钟提高，仍需要成本前提。

一个直接例子：p=(.9,.1)、q=(.1,.9)，两个token并为一组后组概率均为1，所有提案均接受，输出仍为q。这个例子不反驳组接受率定理，说明它的结论范围不含token级精确性。我们若保留Target分布，就不能把Feature码簇直接视作可互换输出，也不能把图像论文的视觉冗余叙事直接迁到Gaussian离散码。

## 3. SpecMER：领域先验必须进入可辨认的操作

### 作者的问题链

从高通量蛋白质筛选需求出发，指出通用草稿没有利用生物局部模式。§3用来自同源序列对齐的k-mer频率为批量草稿评分，选出一个候选再验证。§3.3用“至少一个候选可接受”的事件与误排序损失解释接受率；§4用错配蛋白质来源的k-mer和减少对齐数据的消融，判断领域信息是否真的起作用。

### 理论与证据边界

Proposition 4.4在对应候选接受事件与随机性设定下写出E[A*]=1−(1−α)^m−ε。它是接受事件分解，不能单独证明所选候选仍服从原草稿分布。§3.1先选择最高评分序列、再用原草稿模型评分；若将这种选择迁到我们的精确采样，需要计算选择后的真实提案分布，或另给有效耦合。只重算原模型似然并不足够。

论文将top-p与分布变化联系起来（§2.1），我们的参考policy应明确定义为采用相同温度、约束和top-p后的分布。对这个最终policy实施正确的接受与残差抽样，top-p本身不必破坏相对该policy的精确性。不能把发表论文里的每一句叙事都视作数学结论。

### 对我们的启示

Position Ranker是否利用了生成结构，要靠替换信息、打乱候选排序、统一提案器等对照说明。领域特征只出现在输入名词中不够。SpecMER的错配先验实验提供了可借鉴的思路，但本项目没有因此取得额外质量收益；论文的生物质量结果不能作为Gaussian质量结论。

## 4. Drafter-Invariant：固定随机输出已有相关理论

§5让Target与草稿使用共享随机噪声，从而使固定种子下的Target输出与所用草稿无关。§3给出Gumbel耦合的碰撞概率及其相对另一耦合的改进。

我们的备用方案保留同一个W1的主提案、实际q和随机顺序，因此是在同一参考解码器上保持轨迹；它没有证明替换任意草稿后输出也相同。这比Drafter-Invariant的目标更窄。正文应使用“相对固定参考解码器的轨迹保持”，避免将这种性质写成全新的随机性理论。

## 跨论文归纳：具体可采用的写作方法

1. 引言在应用价值后尽快转向一个明确、带限制的研究问题。
2. 先呈现自然方案为何不足，反例与真实诊断各承担自己的作用。
3. 数学定义须与真实操作对应：候选如何产生、何时可用、哪项概率是实际提案。
4. 定理只承诺它真正证明的量；速度、有限精度与质量另给证据。
5. 用消融验证设计依据，用负结果说明适用边界；主表承担实用性。
6. 接近工作在方法与信息条件上明确比较，避免只列论文名称。

这些组织方式来自公开论文正文。本轮没有获取匿名审稿过程，不能声称掌握审稿人的统一偏好或录用原因。

## 资料范围

研究路径为research-router → academic-local → academic-evidence。公开检索使用OpenAlex、Crossref与Google Scholar，正式方法结论来自上述正文；元数据负责定位和会议身份。List-Level Distribution Coupling（NeurIPS 2025）已由元数据发现，正文未在本轮读完，只作为后续理论比较项。2026近期相关工作并未穷尽，不做首次或唯一声明。
