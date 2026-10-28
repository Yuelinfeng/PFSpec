# 下一步投机解码方法选择

2026-10-21。主比较统一为候选完整解码吞吐 / 同轮原 Target 吞吐，包含执行优化与投机方法的合计收益。当前MLP已在这一口径超过2倍；后续研究目标是继续超过现有MLP。新候选尚无本项目在线速度，不能引用论文倍率替代。

## 当前依据

同轮120次的MLP、EAGLE v3、EAGLE v4 KV复用、Medusa LowRank相对原Target分别为4.637、2.986、2.888、2.454倍；参见[主表](main_run01/RESULTS.md)。此前轨迹中，MLP Position接受率92.53%、Feature36.97%；EAGLE Feature20.97%，Medusa P1拒绝99.24%。[轨迹分析](TRACE_SCREEN.md)支持优先改善Feature提案，并保留强Position Ranker。

## 候选比较

| 顺序 | 方法 | 可利用的机制 | 本项目限制与下一动作 |
|---|---|---|---|
| 1 | [Sequoia](https://arxiv.org/html/2402.12374)式Feature分支 | 不放回提案与精确残差采样，多个Feature候选共享前缀，一次Target验证各分支 | 不需先训新模型；现有线性Target图不支持树注意力和分支KV。先用W2取得各Feature条件下的P2分布与真实耗时，收益成立后再测W4。 |
| 2 | [UniVer](https://arxiv.org/html/2605.04543v1)，2026-05 | 用前缀接受概率协调同层分支与后续深度的验证 | 适合在树验证可运行后比较；须实现条件概率分配与选择，不能用简单“任一候选命中”代替精确随机验证。尚未核读作者源码，论文保证不能直接外推到本项目改写。 |
| 3 | [DFlash](https://arxiv.org/html/2602.06036v2)，2026-02/05 | 单次前向并行给出短块草稿，多层Target特征注入Draft KV | 需新草稿模型和P/F训练数据，多层特征读取还会增加Target成本。先验证短块q、类型掩码和单调Position支持；保留此前暂缓MultiTap的状态，不自动开始该训练。 |
| 4 | [CaDDTree](https://arxiv.org/html/2606.01813v1)，2026-06 | 按预计提交量与实际验证时延联合选择树规模 | 论文针对块扩散提供的分解q，以q构造接受量近似，部分结论依赖成本曲线条件。当前MLP只有单Feature提案，不能直接沿用其最优性结论。先测静态W1/W2/W4真实成本，再借鉴预算选择。 |
| 5 | [Window TV训练](https://arxiv.org/html/2609.24150v1)，2026-09 | 优化窗口内连续接受概率的乘积，而非只对单步分布匹配 | 现有MLP已有CE+2TV；窗口只有一个可训练Feature时，WTV退化为单步TV加常数，对固定Position存活率也主要表现为权重变化。需要先证明跨窗口目标与现有训练存在有效差别，再决定是否训练。 |

首选判断来自本项目成本结构，属于待验证的适配方案。Sequoia与UniVer提供不同验证方法，CaDDTree提供预算选择思路；它们需要分别验证，当前没有把三者组合视作现成算法。

## 暂缓的路线

官方[EAGLE-3](https://arxiv.org/html/2503.01840v2)采用多层特征融合与训练时递推。项目中的`eagle_v3_g3`是项目版本名，不能等同于作者EAGLE-3。现有版本的低Feature接受率不能否定完整官方方法；新适配需要多层状态、训练和类型化采样，短期投入高于复用MLP扩树。

[Speculative Streaming作者实现](https://github.com/apple-aiml-research/ml-speculative-streaming/tree/edb52c221634244850e821bc983707c6f3c799ab)增加多流权重、嵌入与注意力方式；已读生成入口的验证使用argmax相等比较，不能直接保留本项目0.9/0.9随机目标分布。暂缓直接移植。

[REST](https://arxiv.org/abs/2311.08252)与[SAM Decoding](https://aclanthology.org/2025.acl-long.595/)使用重复后缀检索。当前生成后缀中类型化P/F窗口2/4/8均无重复，普通动态完整token检索优先级低。仅Feature历史重复可以提供廉价候选，但当前只有重复比例，没有Target概率质量或接受收益，不能由此承诺速度。

## 首项可执行实验

保留Ranker与当前MLP权重，从单Feature候选W1扩展到两个不重复Feature候选W2。每个Feature候选必须拥有自己的P2候选与Target分布；树节点只读取共享历史和本分支祖先。序列位置按树深度递增，稀疏Position token按各自路径遵守单调规则；提交后只保留真实路径KV。首次图创建、候选采样、树mask、分支KV整理、恢复与EOS均计入完整解码时间。

先用小词表分布验证接受与残差，用真实同输入逐分支普通Target前向验证树形Target概率；允许合理浮点容差，但须单独记录top-p支持集变化。再用少量开发来源、多种子、512-token上限比较W1/W2，保留提前EOS。只有真实每token时间下降且分布验证成立时，再测W4与独立来源；若新增节点与KV开销抵消收益，则停止扩大树。主表一直除以同轮原Target，同时用候选/当前MLP判断是否带来新收益。

既有480行CPU残差结果W1/W2/W4为36.44%/46.11%/55.58%，[原始解释](../pf2x_breakthrough_20261021/RESIDUAL_FEASIBILITY.md)仍适用。它没有分支P2与新Target耗时，下一项应测这些缺失量，不重复同一模拟。

## 作者代码读取范围

三名原gpt-6-luna/max分别完成论文、源码和统计。实际研究路径为research-router → academic-local / academic-evidence，以及github-local。论文正文覆盖Sequoia、EAGLE-3、DFlash、UniVer、CaDDTree、WTV和REST；Streaming论文正文未取得，相关判断来自作者源码。未安装依赖、训练新权重或改变生产入口。

| 仓库与版本 | 已读实现入口 | 可复用范围 |
|---|---|---|
| [Sequoia 0ff3ff7](https://github.com/Infini-AI-Lab/Sequoia/tree/0ff3ff71475ca148f2dfb7545e0de5caf5c67b39) | Tree/SpecTree.py、Tree/SpecInferTree.py、Engine/Engine.py、Engine/Llama_model.py及测试 | 树mask、节点位置、KV路径与拒绝采样；该版本图入口属于Draft侧，Target树图需自行实现。 |
| [DFlash 07ebd93](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a) | dflash/model.py、cli.py、pyproject.toml | 普通DFlash包含temperature/top-p下的p/q与残差采样；当前源码另含DFlash2，须区分。 |
| [EAGLE cb7e084](https://github.com/SafeAILab/EAGLE/tree/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b) | model/cnets.py、traineagle3/cnets.py、ea_model.py、kv_cache.py、utils.py及依赖 | 多层状态条件化、Draft树展开；现有聊天模型权重不能直接用于GaussianGPT。 |
| [Streaming edb52c2](https://github.com/apple-aiml-research/ml-speculative-streaming/tree/edb52c221634244850e821bc983707c6f3c799ab) | modeling_qwen3_ss.py、token_generator.py、attention_mask.py、tree_decoding.py及配置 | 多流注意力设计；Target结构和随机验证均需适配。 |
| [suffixspec 49948aa](https://github.com/joelvarun/suffixspec/tree/49948aaf779d035c0ef98691332d255af9f9c57f) | index.py、drafter.py、decode.py及依赖 | 非上述论文作者实现，仅作低成本检索参考；仓库解码器只做greedy序列回放。 |

五份临时仓库副本已清理。匿名GitHub搜索部分为空或限流，随后使用站点限定发现与浅层Git读取；选定源码均可读。现有证据足以决定首项实验，尚不能证明任何新候选快于MLP。
