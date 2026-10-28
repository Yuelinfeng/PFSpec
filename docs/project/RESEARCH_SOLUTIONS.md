# 面向 GaussianGPT 当前瓶颈的论文检索与新方案

## 2026-10-28 作者训练流程与草稿数据衔接

[作者训练流程还原](AUTHOR_TRAINING_RECONSTRUCTION_2026-10-28.md)已用官方源码、论文原文和实际 checkpoint 训练状态完成对照。原流程是场景 Gaussian 拟合、VQ-VAE、多变体离线编码及随机空间块的 P/F 交叉熵训练。现有完整 D8 可以复用这条序列训练链路，轻量 Feature Draft 需要额外的同历史 Target 状态及分布监督。主要未具备条件为作者原始预处理语料和完整历史配置；单卡超参数、显存与收益尚未测量。本轮只完成研究与元数据读取，不重复已有固定轨迹的长训练。

## 2026-10-23 概率误差改善尚未转为决策收益

恢复价值策略已用浅树软概率标签与完整验证/提交/单步恢复操作成本作新开发验证。主概率在有备用验证轮中均值低估约.132；相同动作数量时更换候选少命中18次。新价值相对固定单备用.993432，对概率规则区间跨1。已通过CPU与GPU逐轮动作对照，不能以代码选中和执行不一致解释这批不足。下一研究焦点是可用信息与决策收益目标、主概率校准及首次图费用；未训练新神经头、未启动新来源确认。见[结果与限制](artifacts/recovery_value_evidence_20261023/RESULTS.md)。

## 2026-10-23 先分清实现开销与策略效果

正式动态路径现只处理选择需要的信息，空池直接跳过，线性参数预读并共享恢复成本计算。相同输入下原/新选择202个实际状态动作完全相同，去掉额外执行开销后新/旧动态解码1.058868、完整请求1.046277，说明旧负结果混入实现成本。同入口对照尚无动态优于固定单备用的证据；保留恢复价值问题，继续研究必须先量化完整动作费用及可预测的决策差异。成本探针现区分预热前与实际计时状态，旧模型不重拟合。详见[实现维护结果](artifacts/recovery_value_execution_20261023/RESULTS.md)。

## 2026-10-23 恢复价值关键证据实验完成

[完整结果](artifacts/recovery_value_20261023/RESULTS.md)。20新来源组、64/128前缀、512上限，旧10来源开发、模型8/2留源，参数固定后两进程800解码及400完整三维请求完成。value/单备用解码0.932238 [0.928435,0.936197]，完整请求0.941442 [0.933740,0.947926]；相对概率成本和同数量Ranker区间跨1，未得到来源外强对照优势。单备用/W1来源等权解码1.013946、完整请求1.008566；完整请求累计129.007296→128.991790秒，仅省0.015506秒，累计比1.000120 [0.982643,1.014994]，不概括为稳定端到端净收益。

1200记录、960输出/语义/调用对照、600跨轮与400跨阶段对照通过；400NPZ与2400PNG有效，无空白视图，Gaussian/像素有极小差异。解码Luna独立复算通过；其最终E2E任务额度用尽无交付，主代理从原始记录直接另算四项点值/区间，与主统计完全相同，未将未交付工作计为证据。神经权重和生产入口保留。RTX4090结束2MiB/0%，无计算进程，远端HEAD031d916/4181记录、原权重大小及821/822服务保持。原始记录在本机artifacts/recovery_value_20261023、源码patches/recovery_value_20261023，远端/root/autodl-tmp/gaussiangpt_recovery_value_20261023；matrix_progress为complete，不重复启动。当前动态价值选择只作为负结果和边界，v4正文仍以固定恢复预计算为主；单模型、单卡、有限chunk与512上限不变。

## 2026-10-23 融合版v4采用与证据更新

后续正式写作使用[v4入口](artifacts/paper_v4_20261023/PAPER_MAIN.md)与[审阅意见](artifacts/paper_v4_20261023/REVIEW.md)，承接用户730行融合稿和169行审计；原大纲及v3保留来源地位。中心为固定参考抽样下的条件恢复计算预计算，动态分配仍是设想。补明任意k子集可行及随机集合前提，更新四轮重复、完整请求和两类39状态证据。现有单备用解码新增收益约1.81%，完整三维请求新增收益尚不确定；总解码与总请求倍率分别按原测量范围报告。本轮无GPU、训练或运行源码变更。XSpecMesh官方身份已确认为ICML2026，Learning to Draft为ICLR2026且明确以吞吐为目标；下方旧身份状态保留历史，出版PDF版本差异仍待核对。

## 2026-10-23 理论优先的论文框架

用户要求对整个工作进行学术整理。完整[算法论文大纲](artifacts/paper_design_20261021/PAPER_OUTLINE.md)已按问题、文献、方法、理论、反例和实验组织；[理论正文](artifacts/paper_design_20261021/THEORY_NOTES.md)2026-10-23节为当前依据。通用投机、mesh和其他结构化生成正文研读分别见同目录SD_PAPER_READING.md、MESH_PAPER_READING.md、DOMAIN_PAPER_READING.md，独立审阅THEORY_REVIEW.md。

当前恢复预计算保持参考W1抽样与主提案，只决定提前计算的条件状态。SpecTr等已有候选命中复用，通用树与成本选择也有前例。集合选择与误差界属于本轮理论设计，尚无新增策略实施或净收益。论文完成条件仍包括最接近方法的同条件区别、来源外预测与第二模型／表示。CUDA Graph仅作执行支撑，既有速度和负结果保留原范围。本轮只修改静态文档，无远端或GPU操作。

本轮整合已完成14篇正文材料，明确FlashMesh（CVPR 2026）的层级预测与匹配验证、XSpecMesh（录用身份未确认）的多头阈值规则，与Gaussian P/F固定policy的区别。整体大纲另有压缩提案的主概率加权TV风险分析；真实信息损失尚未识别，不能凭训练平台期断言输入能力上限。所有新策略和迁移均以待实施状态记录。

## 2026-10-22 已测新候选：备用继续执行优于固定Feature混合

归一Ranker正类评分在39开发前缀重合度.684075，低于原点提案.689961；保守点提案混合最高仅.692011，不扩大GPU矩阵。历史Target Feature policy以.25混入MLP虽有平均接受改善，三十来源接受差额与净速度区间均跨零增益，暂不采用。

纯备用Position将W1四节点扩为五节点，在P1残差实际抽到备用值时用第五行Feature输出、hidden与[0,4]缓存路径继续，保留点提案alpha。三十来源新种子确认备用/W1=1.017481 [1.002716,1.026950]，388次恢复被省去，60/60输出与接受计数相同。该路线是少量额外计算换恢复成本，未改善原接受率。组合固定混合与备用尚无超过W1的可靠净收益。保留纯备用作为候选，生产默认不替换；文献全面新颖性与三维端到端结果尚未新增。数学解释、负结果与代码见[本轮记录](artifacts/spec_next_20261022/RESULTS.md)。

## 2026-10-21 恢复预算候选已完成开发验证

同前缀十来源39状态、八种子、三个动作、三次计时重复共2808行，另480行分项诊断已完成。恢复事件成本模型相对固定W1的局部进展率0.987458× [0.953654,1.020480]，与直接分别预测R/C的39个选择完全相同；直接比值模型1.010234× [0.984362,1.043486]，不支持稳定收益。成本MAE仅由1.3352降至1.3324ms，未兑现独立决策价值。保留静态W1，不继续本轮在线选择、外部测试或同数据调参。相关工作扩大到九篇原始论文加一综述，通用比值目标、学习式预算与状态奖励已有研究，pair/EOS公式作为特定解码器计数表达。论文优先保留类型化提案与执行协同，恢复分项仅作解释与负结果。见[完整报告](artifacts/recovery_model_20261021/RESULTS.md)与[相关工作](artifacts/recovery_model_20261021/NOVELTY_REVIEW.md)。

## 2026-10-21 论文主线与理论建模

新增[研究故事设计](artifacts/paper_design_20261021/RESEARCH_STORY.md)与[理论草稿](artifacts/paper_design_20261021/THEORY_NOTES.md)。当前最有依据的主线是类型条件提案、完整P/F对的提交与恢复模型，以及对应设备执行。正常轮的提交量由P1与Feature联合接受决定；P2接受省恢复调用。EOS、短预算和候选缺失单列。恢复感知提案预算选择仅为后续候选，当前W1为静态线性路径，W2仍无净收益。六篇正文比对显示标准抽样、随机树验证和硬件成本选树已有公开方法；新贡献需要同一前缀的成本预测与来源外在线收益来支持。本轮未启动新训练或GPU评价。

2026-10-21 用户将主比较改为原Target：已另做同轮六配置、10开发来源、64/256前缀、seed44101、512token上限，共120次。来源等权MLP/EAGLEv3/EAGLEv4复用/Medusa对原Target为4.6365/2.9861/2.8876/2.4537倍，包含执行优化与投机解码合计收益；Target执行优化单独为3.1527倍。原/优化Target20/20输出计数相同，四候选新旧主轮80/80相同，独立CPU复算10项通过。首项研究建议为保留Ranker的Feature W2分支真实验证成本，再考虑W4、UniVer及成本预算；DFlash需新模型，尚未训练。普通完整token后缀在已有生成后缀中零重复，优先级降低。详见[本轮结果](artifacts/original_target_next_20261021/RESULTS.md)与[方法筛选](artifacts/original_target_next_20261021/NEXT_METHODS.md)。原Target口径已超过2倍；下一项需要证明比现有MLP进一步提速。

2026-10-21 三种提案侧专用图实测完成：10开发来源、64/256前缀、seed44101、512token、180次；MLP/EAGLE v3/EAGLE v4 KV复用/Medusa LowRank的提案图新增倍率为1.034724/1.157529/1.175119/1.020445，相对同优化Target由1.439277/0.830013/0.788927/0.761582变为1.489254/0.960764/0.927083/0.777153。2倍未达，MLP仍最快。四候选各20/20输出和计数一致，Target调用、图形状与缓存事件一致，独立复算9/9通过。首次图创建进入同步外层墙钟；三臂均无新增图的18对、9来源子集中MLP为1.516174，EAGLE v3为0.974637且区间跨1，v4与Medusa为0.938478/0.781818。EAGLE仅T1递推与Feature读出头图化，T>1仍普通执行。无训练、生产默认和权重保留，GPU已释放。详见[完整结果](artifacts/proposal_graphs_20261021/RESULTS.md)。

2026-10-21 重测已有投机方法：Target采用显式BMM+CUDA Graph后，当前已有EAGLE/Medusa/Hydra/D8没有出现新优势。10开发来源主比较MLP1.451×、Ranker-only1.371×，其余0.492–0.899×，均相对同优化Target。双方无新增图的辅助子集也没有反超。3来源同条件原/优化配对的接受计数保持不变，Target提速2.804×、EAGLE v3γ3整条路径提速1.683×，故算法相对收益下降。此结论只覆盖项目已有适配与权重；不能推广为全部论文实现均无效。LFQ BOS重放不一致，暂不参与排序。详见[实测结果](artifacts/target_methods_20261021/RESULTS.md)，勿重复同一轮或自动追加旧模型训练。

2026-10-19 原 MLP 长续训结论：step12000 → 24000 → 48000 的在线 Feature 接受率为 34.03% → 33.78% → 35.60%。48000 相对 12000 的来源等权速度为 1.0021× [0.9850,1.0189]，接受率差区间包含零；58 组相同生成长度的描述性子集结论一致。完整验证重合度缓慢增加 0.7004 个百分点，未出现在线突增。当前证据不支持靠追加相同训练获得突破，保留原 step12000 PF 作为已完成的 Motivation。下一步完整方案需提出可单独验证的方法假设，并同时衡量接受率与实际成本。详见 [续训记录](FEATURE_MLP_EXTEND_2026-10-19.md)。

2026-10-19 Feature EAGLE v3 结论：真实/Ranker 条件混合、主要 Feature KL 与两处 Feature 递推，使 γ=3 在线 F 接受率从 21.59% 到 35.14%，吞吐相对 v2 为 1.129×。它对 Ranker-only 仅为 1.009×，来源间方向不一致；对旧 PF 为 0.844×。γ=5 的 F2 尚无在线提升证据。当前支持训练目标和条件数据改善预测，不支持更长草稿或替换旧 PF；后续应把接受改善与草稿历史重建、状态更新成本一起衡量。见 [v3 执行记录](FEATURE_EAGLE_V3_2026-10-19.md)。

2026-10-19 EAGLE v2 实测：Joint P/F 和 Ranker + Feature-only 的相同 1L-256 因果结构已经实现并训练。108 次开发运行中，B γ=3 对 Target 为 1.213×，对 Ranker-only 为 0.904×；旧 PF 对 Target 为 1.622×。A γ=3 为 0.739×，其 P 接受率 2.214%，大量 Position 恢复计算抵消分块收益。奇数窗口和隐藏状态递推已经接通，当前尚未证明它们带来额外速度。训练数据规模与自然 EOS 覆盖不足，保留旧 PF，后续决定应依据 [实现与结果](EAGLE_PF_V2_2026-10-19.md)，避免将 γ=1 的 Ranker 收益归给 Feature EAGLE。

2026-10-17 成本路线 B1 结论：Feature 的条件收益具有可测差异，见 [分析报告](artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md)。同前缀中，Position 候选间隔为 1 的组 PF/P-only 为 1.190×，间隔大于 1 的组为 0.970×，后者尚不能确认更快或更慢。生成前信息能预测部分接受概率与推进量，固定模型对额外时间的预测却弱于均值参照。下一项可考察已有 Position 候选特征对应的低成本 Feature 使用条件，先扩大开发前缀与重复测量，再考虑在线对照。草稿熵需要支付 Draft 成本，Target 概率仅作为诊断标签。保留固定 PF-K2，暂不增加 P₂ 容量；B1 没有识别训练条件差异的因果作用，不能直接作为扩大 Feature 训练的依据。没有实现动态算法。

2026-10-17 候选成本结论：以“哪些候选值得预测、预测多深”为下一阶段问题。已完成 [A1–A3 成本测量](artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md)，没有增加 Head。现用 PF 相对 P-only 的固定窗口加速为 1.171×，相同前缀即时效率比为 1.141×；Feature 路径有净收益。P2 拒绝整段处理仅占 1.337% 时间，免费删除的当前路径估计为 1.0136×，单纯提高 P2 接受率优先级较低；若能替换已有 Ranker 并减少提案成本，需要另算收益。初始与 PF 中间前缀的即时效率比为 1.059×、1.208×，存在上下文差异，但不能直接据此选择动态策略，也没有证明差异原因。保留现用 PF；A4、B 阶段因素对照与 C/D 调度未开展。该测量不能区分 embedding、历史范围和训练条件谁是 Feature 预测瓶颈。

2026-10-17 第二轮首批完成：[路线与实施计划](MEDUSA_HYDRA_EAGLE_ROUND2_REVIEW_AND_PLAN_2026-10-17.md)已完成首批诊断、三种子 Feature 训练、392 次在线比较、56 次分项测量及 A2 输入对照，见 [第二轮执行记录](ROUND2_EXECUTION_2026-10-17.md)。旧 PF 本轮仍最快，为 1.624×；B1-E 为 1.506×。简单增加历史 Feature embedding 未解决 P1 的低重合度，不能将条件不足视作唯一原因。当前继续使用旧 PF，保留 B1-E 的单步机制结果；P2 条件化先依据可减少的补算选择，树、递归与新来源正式确认尚未开展。提案质量的三训练种子差异已记录，在线速度仍限首训练种子。

2026-10-14 最新规划：D8、轻量P/F和D4实验已完成，下一阶段采用 [P/F多场景研究计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。以独立场景、P-only/PF/D8 Feature同条件对照、实际提案成本和完整三维输出补齐证据；动态P-only/K2选择与真实来源适配按开发结果开展。现有状态复用具有相关研究依据，方法贡献需要落在P/F条件关系与可测收益上。当前已能实际提出Feature，下文较早的“Feature不参与提案”仅描述当时实现。此次更新只形成计划，未启动新实验。

2026-10-08 阅读提示：当前暂缓新增实验，完整阶段结论见 [阶段工作汇报](STAGE_REPORT_2026-10-08.md)。下文保留历史研究候选；文中的建议速度与资源投入不作为本项目实测结果，也不自动触发后续任务。

2026-10-07 状态入口：[实验记录](EXPERIMENTS.md#s132-s13b-权重的真实联合-pf-评测)记载 S13b 的真实联合 P/F 结果；[联合 P/F 草稿训练计划](JOINT_DRAFT_TRAINING_PLAN.md#17-s13-结果后的-s14-更新方案参考研究报告)已依据结果调整 S14。项目已有 S8.4 结构化 Position proposal 的多 seed 单场景速度数据。下文较早的 Position-only 与 shadow 结论按各自日期和适用路径理解。

更新时间：2026-10-04。本文把当前本地记录的现象、论文线索和可以在 Autodl 上落地的实验路线放在一起。论文结论来自公开论文或官方页面；对 GaussianGPT 的适用性是基于当前项目结构的推断，不能替代实测。

候选方案的正确性前提、优先级筛选和停止条件见 [`LOGIC_VALIDATION.md`](LOGIC_VALIDATION.md)。下文方案顺序是待验证路线，不代表已经确认当前实现存在对应缺陷。

## 1. 当前问题的精确定义

Autodl 当前实现的设计说明确认：draft 只提出 position，feature 始终由 target 采样；draft 的 feature logits 只用于 overlap 诊断。记录中 position 实际接受率约 0.922、position overlap 约 0.938、feature overlap 约 0.382，端到端 speedup 约 1.06–1.15×，且存在运行波动。因此 feature overlap 不能解释当前 position-only decoder 的直接 rejection；问题应按三层拆开：

1. **proposal 层**：position proposal 已较准，feature logits 当前不参与 accept/reject；
2. **训练目标层**：只有改善 position proposal 的训练目标才直接关联当前 acceptance；feature distillation 的作用可能是共享参数带来的间接影响；
3. **系统层**：D8 forward、target 两 token block、position rejection 重算、KV cache 和 Python/CUDA 同步开销可能吃掉理论收益。

因此不能只用“整体 token accuracy”判断方案。至少需要分别报告 position expected acceptance、position overlap、feature overlap（仅作诊断）、draft/target/cache/sampling wall-clock，以及最终渲染质量。

## 2. 相关论文与可迁移启示

| 方向 | 论文 | 关键启示 | 对 GaussianGPT 的关系 |
| --- | --- | --- | --- |
| 基础 speculative decoding | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 小模型提出、目标模型并行验证，并用概率修正保持目标分布 | 当前实现的理论基线；必须保留 token-level correction |
| 蒸馏 draft | [DistillSpec](https://arxiv.org/abs/2310.08461) / [Google 介绍](https://research.google/pubs/distillspec-improving-speculative-decoding-via-knowledge-distillation/) | 用 white-box teacher 分布训练更匹配的 draft | 可用于 position-logit 对齐；feature KL 对当前接受率只有可能的共享参数间接影响 |
| 在线蒸馏 | [Online Speculative Decoding](https://arxiv.org/abs/2310.07177) | 用实际 query/rollout 持续更新 draft，减少训练分布和推理分布偏移 | 可用 target-sampled feature 构造真实 position proposal 前缀；当前不存在 rejected-feature replay |
| 自身多头 | [Medusa](https://arxiv.org/abs/2401.10774) / [官方实现](https://github.com/FasterDecoding/Medusa) | 在 target 上增加多个 future-token heads，用 tree attention 一次验证多个候选 | 有机会减少 D8 成本，但 GaussianGPT 的中间 feature 由 target 采样，future position head 需显式处理该条件 |
| 动态 draft tree | [EAGLE-2](https://arxiv.org/abs/2406.16858) | 用上下文相关的 confidence 动态选择树结构，不固定 draft 深度/分支 | 可参考 position candidate budget；跨 pair 深度受 target-sampled feature 限制 |
| 动态深度 | [Dynamic Depth Decoding](https://arxiv.org/abs/2409.00142) | draft 长度应随当前上下文变化，而不是固定 block size | 可研究 position proposal 是否值得启动；不能直接用 feature uncertainty 控制当前拒绝 |
| 温度与蒸馏 | [Temperature-Centric Investigation](https://arxiv.org/abs/2410.10141) | 温度改变 acceptance 和 KD 效果；训练温度应匹配部署温度 | 若优化 position proposal，应在当前 temperature=0.9/top-p=0.9 policy 下校准 |
| 直接优化 acceptance | [LK Losses](https://arxiv.org/abs/2602.23881) | KL 是 proxy；有限容量 draft 可用直接 acceptance 目标，论文报告平均接受长度提升 | 可作为 position-token loss 的设计参考；语言模型结果不能直接外推 |
| 低成本候选修复 | [Calibrated Speculative Decoding](https://aclanthology.org/2026.acl-long.1369/) | 用历史 rejection memory 和概率守门处理 proposal | 对当前离散 position proposal 的适用性未验证；不能把它解释成 feature rejection 修复 |
| 3D 层级 token | [G3PT](https://arxiv.org/abs/2409.06322) | cross-scale VQ + coarse-to-fine AR，避免对无序 3D 强行规定单一顺序 | 提示把位置/feature 交替序列改成空间层级序列 |
| 3D 多尺度 AR | [SAR3D](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_SAR3D_Autoregressive_3D_Object_Generation_and_Understanding_via_Multi-scale_3D_CVPR_2025_paper.html) | 预测下一尺度而非单个 token，减少生成步数 | 高风险长期路线；可先在小场景做 multi-scale VQ prototype |
| GaussianGPT 原论文 | [GaussianGPT](https://arxiv.org/abs/2603.26661) / [代码](https://github.com/lupin4/gaussiangpt) | 稀疏 3D VQ-VAE、3D RoPE、位置/特征 token 自回归生成 | 保持现有 representation 作为 baseline，避免把加速问题与重构问题混在一起 |

## 3. 优先推荐方案

### 方案 0：核验 position-only exact correction 与 cache（立即做）

Autodl 文件头说明当前 decoder 只 speculates position；feature 一直从 target 采样。统计结构也只有 position accept/reject，feature 侧只有 observation/overlap。因此 `feature overlap≈0.38` 不是 feature acceptance，也不会在现有算法中导致整对回滚。记录的 position accept rate 约为 0.92、position overlap 约为 0.94；两个统计仍要按各自定义区分。

需要逐行确认下面的 position-only 流程与实际实现相符：

```text
draft proposes P_i from Q_D(P_i | prefix)
target verifies P_i using P_T(P_i | prefix)
if accepted: sample F_i from target feature logits already returned by the block
if rejected: sample P_i from normalized [P_T - Q_D]+, rerun target on corrected P_i,
             then sample F_i from the corrected target state
```

如果这个路径和小分布 self-test 都成立，就不做 pair partial acceptance；现有实现没有 feature proposal 可供 partial accept。重点检查最终 policy 分布、`[P_T-Q_D]+` residual、position cache 回滚/回写、target feature sampling、draft cache 是否同步真实提交的 `[P_i,F_i]`、EOS 与零概率项。

### 方案 A：Position-only acceptance-aware/on-policy distillation（瓶颈确认后做）

若 profiling 显示 position rejection 或 draft 分布失配值得优化，在现有 S3 训练基础上，把 acceptance surrogate 加在 position-token 行，并优先用 teacher rollout 中真实 target-sampled feature 产生的上下文训练。对 target position 分布 `p` 和 draft position 分布 `q`，可先使用：

```text
expected_position_acceptance = sum_v min(p_v, q_v) = 1 - TV(p, q)
L_position_accept = 1 - expected_position_acceptance
```

建议实验矩阵：

```text
position CE
position CE + KL
position CE + KL + L_position_accept
position CE + L_position_accept
position CE + L_position_accept + temperature calibration
```

每个版本固定 teacher、draft、seed、temperature、top-p，先比较 position `sum min(p,q)`、实际 position acceptance、draft forward 时间和 wall-clock。LK Losses 的 acceptance 目标可作为 loss 设计参考，但语言模型上的结果不能直接外推到 3D position token。

### 方案 B：position 候选预算与动态验证（中风险，成本验证后再决定）

当前只能在已知前缀下提出 position 候选；target 采样出对应 feature 后，才能形成下一轮 position 的真实前缀。因此多步树不能简单跨过 feature token。可先评估同一 position 步上的有限候选方案：

- 按 position logits 的 entropy/置信度决定是否值得启动 proposal；
- 任何多候选分支都保留自己的 coordinate/3D RoPE/cache 路径，并按 exact tree sampler 做概率修正；
- 目标是最大化提交 position 数除以总 wall-clock，而不是最大化候选通过率；
- 只有成本模型证明可省 target work，才扩成候选树。

这和 EAGLE-2 的 context-aware tree 有概念联系，但 GaussianGPT 的 target-sampled feature 限制了连续 position 深度。先保持模型权重不变，并确认 3D RoPE、cache 分支和 target correction 成本。

### 方案 C：target 内置 position/feature 多头（中高风险，可能消除 draft 成本）

参考 Medusa，在 teacher 的隐藏状态上加多个轻量预测头：

1. `position_head[k]` 预测未来第 `k` 个位置 token；
2. `feature_head[k]` 预测对应 feature token，或让 feature head 读取候选位置 embedding；
3. 用 tree attention 一次验证候选 pair；
4. 先做冻结 teacher 的 head-only 训练，再做联合微调。

优点是共享 target 主干，不需要每轮启动 D8；但当前每个 position 后都要从 target 采样 feature，预测第二个未来 position 的 head 必须处理这个中间 feature 条件，不能直接照搬连续文本 token 的 future heads。需要 on-policy multi-head training，并先解决 target feature 与 future position 的条件关系，不能把“两 pair head”当成直接移植。

### 方案 D：层级/多尺度 tokenization（高风险，长期创新线）

G3PT 和 SAR3D 的共同启示是，3D 的困难部分不是“Transformer 不够大”，而是单一线性序列对空间无序和多尺度结构的表达不自然。可设计：

```text
coarse occupancy / position tokens
        → coarse feature tokens
        → residual feature codebooks
        → optional appearance refinement
```

第一阶段只生成稀疏占用和粗几何，feature 细节在局部块内并行或小树生成。这样可能同时降低序列长度并提高 draft 对 coarse token 的一致性，但需要重新训练 VQ-VAE、重做 serialization、重新评估渲染质量，不能作为短期 speedup patch。

### 方案 E：检查 feature codebook 粒度（较高成本的表示诊断）

GaussianGPT 原论文的 codebook 消融显示，较大的 feature codebook 不一定带来更好的使用率；如果当前使用的 4096 codebook 让 feature 分布过于碎片化，低 feature overlap 可能来自词表粒度，而非 Transformer 容量。由于当前解码器不 draft feature，这个消融不会直接提高 feature acceptance；只有它能改善 position proposal 的上下文或目标场景质量时才与当前问题相关。若作为表示研究，记录 codebook usage/perplexity、feature overlap、render PSNR/SSIM/LPIPS 和场景结构指标。

每种 codebook 都需要匹配的 VQ-VAE、tokenization 和 GPT 训练/评估，不能把不同 codebook 的 token 直接混用，因此它不是短期低成本实验。

### 方案 F：latent/render-aware feature distillation（中风险）

当前 S3 主要对离散 feature logits 做 CE/KL。可以参考 feature-level partial alignment 与近期 3D latent 学习方向，在 feature-token 行增加低秩 adapter 或中间 latent 对齐损失，并用固定视角 render loss 约束蒸馏结果。由于当前 q_F 不是 proposal，这条路线属于质量或间接共享表示实验，不是直接提升 speculative acceptance 的方案。

这条路线必须单独报告 position acceptance 与渲染质量：render-aware loss 可能提升 PSNR/LPIPS，却不改善 position proposal。最终生成仍使用 target exact correction。

## 4. 建议的实验顺序

### E0：先审计 exact correction，再拆分瓶颈

已读到 Autodl 文件头与统计结构，确认设计为 position-only proposal。下一步逐行核验 position accept/reject、residual correction 和 target/draft cache 回写，再用小规模随机分布 self-test 验证 exactness。当前无需实现 feature partial acceptance；如果将来改为 draft feature，才重新评审条件 feature correction。

### E1：确定 position proposal 与系统耗时的关系

对现有 teacher/D8 运行固定多 seed profiling，记录 position entropy、top-1 agreement、position `sum min(p,q)`、实际 position accept/reject、draft forward、target 2-token block、position rejection 后的 target rerun、cache、sampling 和同步耗时。feature overlap 保留为 q_F 诊断字段，不计入 acceptance；若 position acceptance 已高而 draft/target/cache 成本主导，就优先处理这些 wall-clock 项。

### E2：Acceptance-aware loss + calibration

只改 position-token 训练目标，不改 decoder。验收门槛：position expected acceptance、实际 position acceptance 和端到端 wall-clock 有一致收益；若只改善 q_F overlap，或 acceptance 上升而 wall-clock 不升，停止扩大 loss 复杂度。

### E3：动态 `(depth, width)` tree

只有单 proposal 与 D4/D6/profile 对照后仍有可优化余量，才比较单位置多候选和 entropy-aware candidate budget。记录 accepted positions/verification、目标 forward 次数、峰值显存、cache rebuild 和端到端 tok/s；不假设可以直接跨过 target-sampled feature 扩展多步深度。

### E4：评估 target-native position head 的条件可行性（延期）

未来位置 head 必须处理两位置之间由 target 采样的 feature 条件。先证明 head 能读取正确 target feature 前缀且有可行的 tree/cache 验证，再决定是否训练；目前不直接做 two-pair head。

### E5：小规模 hierarchical tokenizer

只选短序列和少量场景，比较 token 数、VQ 重构、render loss 和 AR 串行生成步数。跨 tokenizer 的 token acceptance 不能直接和当前模型比较；只有质量达标且串行步数明显下降，再扩大数据和模型。

## 5. 评估原则

需要分开两种研究目标：

- **Exact route**：当前只对 position proposal 按 target/draft 最终 policy 分布修正；改变 proposal 形式仍需对应的 exact correction。Feature-neighbor rescue 不属于当前解码器的接受路径。
- **Quality-preserving route**：允许语义/渲染近似，使用 codebook 距离、Gaussian 参数误差、render PSNR/SSIM/LPIPS 和场景结构指标评估，不能再宣称 lossless。

当前项目应先验证 position correction 与真实性能瓶颈，再决定是否投入 position acceptance loss 或候选树。relaxed semantic gating 作为独立近似实验；否则 acceptance 上升可能只是改变了生成分布，而不是加速同一个 GaussianGPT。

## 6. 结论

合理顺序是：核验当前 position sampler 的 exactness 与 cache；做多 seed 的 draft/target/rejection 成本拆分；然后比较 D4/D6 与 position-only distillation。Feature distillation、动态多位置树和 target-native heads 都不作为当前默认路线，只有证据显示它们能影响 `q_P` 或降低总时间时才继续。Hierarchical tokenization 仍是长期表示方向。

## 7. 补充检索方向

这些方向暂不列入第一批实现，但值得在主线实验后继续跟踪：

- [FSPAD](https://arxiv.org/abs/2408.15562)：feature sampling + partial alignment distillation，适合设计 feature-only 中间表示对齐；
- [VSD](https://arxiv.org/abs/2602.05774)：sequence/path-level acceptance 可启发未来将 feature 纳入 proposal 的新 decoder，不是当前 position-only acceptance 的直接修复；
- [SpecDec++](https://arxiv.org/abs/2405.19715) 与 [SVIP](https://arxiv.org/abs/2411.18462)：动态 draft 长度的思想可参考，但 target-sampled feature 限制了跨 pair 的 draft 深度；
- [SEQUOIA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html)：按硬件/预算用动态规划选择 token tree topology；
- [VAT](https://arxiv.org/abs/2412.02202)、[OAT](https://openaccess.thecvf.com/content/ICCV2025/html/Deng_Efficient_Autoregressive_Shape_Generation_via_Octree-Based_Adaptive_Tokenization_ICCV2025_paper.html)、[Can3Tok](https://iliyan.com/publications/Can3Tok/)：从 latent query、adaptive octree 和 scene-level 3DGS VAE 角度减少跨场景 feature 漂移；
- [Gaussian-JEPA](https://arxiv.org/abs/2608.15651) 与 [VAR-3D](https://arxiv.org/abs/2602.13818)：探索连续 latent/渲染监督对 Gaussian feature 稳定性和两阶段 objective mismatch 的影响；
- [PARD](https://arxiv.org/abs/2504.18583) 与 [DeepSeek-V3 MTP](https://arxiv.org/abs/2412.19437)：把现有 draft 改造成并行或 sequential multi-token predictor，减少逐 token draft 成本。

## 8. 全 token 结构化投机的实测边界（2026-10-04）

当前已在远端 RTX 4090 完成一个 full-token structured speculative negative control：每轮 draft 直接提出多个 [Position, Feature] pair，Target 统一验证并回滚 cache。K=1/2/4 在 temperature=1.0 下的 wall-clock speedup 分别为 0.533x、0.571x、0.434x，token acceptance 分别为 0.487、0.398、0.213；降低到 temperature=0.1 后，K=2/K=4 的 speedup 为 0.795x/0.891x，acceptance 为 0.630/0.563。所有测试的 sparse token stream 和 cache invariant 均通过，但还需要单独的分布级 exactness 检验。

这不是对 speculative decoding 范式的否定，而是对“直接复制文本式连续多 token draft”的边界测量。GaussianGPT 的条件链是 P_i -> Target-sampled F_i -> P_{i+1}，后续 Position 依赖 Target 才知道的中间 Feature；draft 在一个 block 内无法获得该条件，因此 pair 数越大越容易失配，同时引入额外 draft forward 和被拒绝 token 的 Target 计算。研究路线应保持在 F-aware/on-policy 草稿训练、position-level adaptive budget、以及 target-native/shared-trunk verification；不要继续把单纯扩大 full-token draft 当作默认优化方向。

此前出现的 `3.67x` 属于 Target block 的 pair-level 计算摊销：K=1 为约 `22.76 ms/pair`，K=4 的 block 约 `24.76 ms`，所以 `4*22.76/24.76≈3.67x`。它证明 GPU 上的 block 并行前向有效，但不包含 draft、acceptance、rollback、cache 和输出序列条件依赖；不能与后续完整 structured route 的 `0.533–0.891x` 端到端 wall-clock 结果混用。

## 9. 选择性 LM head 与 compact sampling 的可迁移性（2026-10-04）

这次实现属于 speculative decoding 的**验证/采样路径优化**，不是新的 draft 接受算法。GaussianGPT 的词表由 Position `0..7999`、Feature `8000..12095` 和 EOS `12096` 组成，实际每一行只允许其中一类 token。原始实现仍为完整词表做线性投影并在 sampling 前保留大量 `-inf` 项；selective LM head 与 compact support sampling 利用了这个结构约束，把无效计算和归一化工作去掉，同时保留 target/draft policy、top-p 截断、Position residual correction 和 cache 逻辑。

它与投机解码论文中的“提高 acceptance、增加 draft 深度或减少 target forward”属于同一系统，但优化层级不同：这里不减少 `target_verify_2` 调用，也不把 Feature 加入 proposal，因此不会绕过 `P_i -> Target-sampled F_i -> P_{i+1}` 的结构依赖。它可以作为任何 position-only、structured 或 target-native speculative route 的实现级优化，但不应被报告为新的 speculation 范式。

实测端到端收益为 128/1024/2048/4096 token 分别约 5.80%、6.63%、6.71% 和 8.38%（4096 仅两个 seed）；其中相对 selective-head-only endpoint，compact sampling 在 128 token 上只增加约 1.51% 吞吐。收益规模与 profiling 中 target 双 token forward 约 16.8–17.7 ms/对的主成本一致：sampling 压缩是低风险的小优化，不能替代减少 target verification 或共享 trunk 的研究路线。

进入 canonical branch 前的最低条件是：

- 全词表与 compact support 的 policy 分布在 `top_p=0.9` 和 `1.0`、Position/EOS 与 Feature 两类上数值等价；
- selected LM head 与完整 head 的合法 logits 误差保持在实现精度范围，mask 外不出现有限值；
- rejection correction、EOS、稀疏 token grammar、target/draft cache invariant 和短序列 engine smoke 全部通过；
- 在同一 fair harness 中补齐多 seed、长序列重复和质量/渲染回归，再决定是否把它作为默认推理路径。

当前证据满足前三项以及初步端到端对比，但 runner 仍是远端临时脚本，4096 只有两个公平的固定长度 seed，且没有固定随机数消耗顺序下的逐 token identity 证明。额外 seed7/8/9 的长序列探针出现单侧提前 EOS，已从固定长度吞吐均值排除并作为鲁棒性记录。因此当前裁决是：保留为 inference optimization candidate，记录为已验证的低风险 patch；下一阶段的主要研究预算仍放在 F-aware draft、position-level adaptive proposal budget 和 target-native/shared-trunk verification。

## 10. Feature-aware draft 与极低成本 micro-draft（2026-10-04）

当前 draft 的 `C_D` 约为 6.5–6.7 ms/双 Position，Target verification 约为 16.8–17.7 ms/对。Position acceptance 已较高，而 feature overlap 明显低于 position overlap；因此下一轮 draft 研究应同时回答两个问题：Feature 学习能否改善真实部署前缀下的 `q_P`，以及能否用几乎零成本的 proposal 保留足够的 Position acceptance。

### 10.1 Feature-aware auxiliary draft（低风险优先）

第一阶段不把 Feature 直接加入 acceptance，而是在共享 draft trunk 上增加 Feature 辅助目标。训练样本来自 Target rollout 的真实 `[P_i, F_i]`，优先使用 Target-sampled Feature 作为下一个 Position 的条件，避免只在 draft 自己生成的错误 Feature 上训练。可从以下损失开始：

```text
L = L_position_CE
  + lambda_F * L_feature_CE
  + lambda_P_KL * KL(P_T^P || Q_D^P)
  + lambda_F_KL * KL(P_T^F || Q_D^F)
```

部署时仍只让 draft 提出 Position，Target 继续采样 Feature；Feature head 的价值由真实 `q_P` overlap、Position acceptance、draft forward 时间和端到端 tok/s 判断。若 Feature loss 只提高 `q_F` 而不改善 `q_P` 或 wall-clock，应停止扩大 Feature 分支。若共享 trunk 的 Feature 表征能改善下一位置预测，再考虑 scheduled sampling 或轻量 hidden-state distillation，以缩小 teacher-forcing 与部署前缀的差异。

### 10.2 n-gram/lookup micro-draft（最低成本基线）

先建立一个只预测 Position 的查表草稿，作为 `C_D` 的下界。原始 token-id n-gram 在 8000 个 Position 和 4096 个 Feature 上会很稀疏，因此 key 应保留结构：最近一到两组 `[Position, Feature]`，必要时加入 coarse spatial bucket；按最长匹配到短阶 backoff，再回退到 unigram 或 D8。查表只产生 `Q_D^P`，现有 Target verify、residual correction、EOS 和 cache 逻辑全部不变，所以 exactness 边界清晰。

用当前 profiling 的近似成本，n-gram 路线满足下式就可能优于 D8：

```text
C_ngram + (1 - alpha_ngram) * C_reject
    < C_D8 + (1 - alpha_D8) * C_reject
```

取 `C_D8≈6.6 ms`、`C_reject≈13–15 ms`、`alpha_D8≈0.95–0.98`，忽略很小的 CPU lookup 和同步成本时，micro-draft 的 Position acceptance 粗略达到约 0.50–0.55 就有机会抵消 draft 计算节省。这个阈值应以真实 GPU wall-clock 重新测量，不能只用公式替代端到端 benchmark。

### 10.3 实验顺序与进入条件

1. 先做 unigram/bigram/trigram + backoff 的 Position-only lookup，记录 hit rate、lookup latency、Position overlap、acceptance、rejection correction 次数和 tok/s。
2. 再训练 Feature-aware auxiliary draft，固定解码器和采样 policy，比较 `lambda_F=0`、小权重和中等权重，确认 Feature loss 是否能传导到 `q_P`。
3. 若 n-gram acceptance 足够高，测试 `n-gram → D8 fallback` cascade；只有高置信度查表命中时才跳过 D8，低置信度样本回退到神经 draft。
4. 只有 Feature-aware draft 或 cascade 已经证明能降低总 wall-clock，才考虑让 Feature 进入联合 proposal；那会重新引入 pair acceptance、条件 Feature correction 和 cache exactness 的完整验证。

这条路线把“提高草稿质量”和“压低草稿成本”分开测量：Feature-aware 训练负责提高条件匹配，micro-draft 负责建立低成本上界，cascade 再判断两者是否能够组合。它比单纯扩大 D8 更容易解释每一毫秒的来源，也更符合当前 `P_i → Target-sampled F_i → P_{i+1}` 的结构。

### 10.4 论文调研后的可迁移结论

三类方法对当前目标的启示可以明确分工：

- [DistillSpec](https://arxiv.org/abs/2310.08461) 的关键不是普通 CE，而是 draft on-policy rollout 与针对 decoding policy 的 divergence。GaussianGPT 应让 Target 采样的 Feature 进入训练前缀，并在 `temperature=0.9/top-p=0.9` 的部署 policy 下比较 forward-KL、reverse-KL 和 acceptance surrogate。
- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177) 说明错误 proposal 本身就是训练样本：Target rejection 后的真实分布可以回放给 draft。对应到 GaussianGPT，应优先缓存被拒 Position 前缀及其 Target Feature logits，而不是只用静态 tokenized scene 做 teacher-forcing。
- [FSPAD](https://arxiv.org/abs/2408.15562) 直接处理 feature-level alignment 与 logit confidence 的冲突。它用带目标 token 信息的 Feature Sampling 构造 draft 输入，并用 Partial Alignment Distillation 将“用于下一步 draft 状态的 feature”和“用于 logits 的 feature”分开。GaussianGPT 的最小迁移版本是共享主干、增加独立 `h_feature`/`h_logit` 投影，再以 Target-sampled Feature embedding 作为条件；这比把 Feature logits 直接当作第二个 acceptance token 风险更低。

因此下一轮不应只增加 `lambda_F * CE(F)`。应做一个最小 FSPAD-like 消融：固定 draft 参数量和 Position head，比较 `(a)` 静态 teacher-forcing，`(b)` Target-sampled on-policy prefix，`(c)` `(b)+KL`，`(d)` `(c)+separate feature/logit projection`。只有当 `q_F` policy overlap、真实 pair overlap 和端到端成本同时改善，才进入联合 `[P,F]` proposal 与 K>1 block 验证。

### 10.5 Feature shadow acceptance 的实现进展（2026-10-04）

远端原型已把 Feature 从“只有 overlap 诊断”推进到独立的 proposal/ratio acceptance shadow。`feature_shadow=True` 时，在 Position 接受条件下从 Draft 的 `q_F` 采样并用 Target `p_F` 做接受判定；但最终仍提交 Target Feature，因而默认输出分布和 Position-only cache 路径不变。更关键的是，`q_F` 直接复用必需的 `draft_update_2[:,0,:]`，去除了先前约 5 ms/event 的额外 `draft_feature_probe`。

F-aware checkpoint 的 256-token 运行得到 Position alpha=0.92969、Feature shadow alpha=0.50420、pair shadow alpha=0.46875、Feature overlap=0.48695；两次 baseline/selective 吞吐分别为 72.182/73.980 和 73.248/74.514 tokens/s，wall-clock ratio 为 1.02490x 与 1.01728x。小分布 self-test 的理论 overlap=0.55、经验接受率=0.55172，源码和 cache invariant 均通过。该结果证明测量和共享 q_F 路径正确，但仍不是联合无损解码的性能结论。

下一步若要把 shadow 变成真实联合 acceptance，需要在 Target verify 前显式提出 Feature，使 Draft cache 停在 `[H_i,P_i]`；当 Feature 被拒绝时从 `[p_F-q_F]_+` residual 修正并回滚/重写后续 Draft 状态。这个路径会引入新的 Draft 状态机和额外 exactness 证明，不能把 retrospective shadow 的统计直接当作实际 pair acceptance 或 3.67x 收益。

### 10.6 真实联合 Feature acceptance 的首轮实测（2026-10-04）

已把 10.5 的 shadow 机制扩展成真实 proposal/acceptance 原型：Draft 在 Target verify 前先提出 `P_i` 和 `F_i`，Target 同一 block 验证两类 policy；Position 和 Feature 都使用 residual correction，Position rejection 与 Feature rejection 分别回滚/重采样。F-aware checkpoint 的两次 256-token 运行均为 Position `121/128`、Feature `51/121`、pair `51/128`，Feature overlap `0.41386`。baseline/joint 的 tok/s 为 `72.725/66.804` 和 `73.150/65.109`，对应 `0.9186x` 与 `0.8901x`。

这说明当前 Feature draft 的质量和成本同时不达标：Feature 条件分布的接受率约 42%，而每轮额外 `draft_feature_probe` 又增加了一次 Draft forward。真实联合 route 因而适合作为 exactness/cache reference，不适合作为当前加速实现。研究优先级应转为：先用共享 trunk、合并 q_F 计算或 n-gram/micro-draft 把 `C_F-probe` 压到近零，再考虑恢复联合 acceptance；在此之前继续扩大 K 或直接宣称 3.67x 都没有依据。

### 10.7 近零成本结构化 Position micro-draft 的首轮证据（2026-10-04）

用户提供的方案可抽象为：用极低成本的结构化 `Q_D^P` 替代神经 draft，只提出 Position；Target 继续按真实条件采样 Feature，并由原有 Position residual correction 保证 exactness。它仍属于 speculative decoding 的 proposal 侧，只是把 draft 从 Transformer 换成查表模型，不能把查表命中率直接当作最终 acceptance。

首轮使用真实 Target trajectory corpus（32 train、4 val、每条 2048 sparse token，交错 `[P,F]`）建立 gap 查表。key 是最近 `h` 个 Position gap 加当前 Position 的低位 Morton/Z-order 特征，预测下一个 gap，并按最长上下文到全局模式 backoff。当前最好配置为 `h=5`、低 1 位特征：验证 teacher-prefix mode top-1 `0.85044`，K=4 连续命中平均 `2.57969`，四个 mode 全部命中 `0.54485`，表项数 1325，CPU/Python 查表均值 `0.7186 μs/query`。`h=4,bits=1` 为 `0.84677/2.56857/0.53373`，表项 728；`h=6,bits=1` 没有给出稳定收益。普通三元上下文对照（`h=2,bits=0`）只有 `0.78153` top-1 和 `2.13913` 个 K=4 连续命中，说明结构特征与较长历史确实利用了 Position 序列规律。

从成本角度，查表项本身小于 1 μs，而现有 profiling 中 Target 双 token verification 约 `16.8–17.7 ms/pair`；因此 `C_micro` 的查表部分可以近似忽略，但这不是完整的 `C_D` 测量。Python dictionary、GPU/CPU 同步、Position support 映射、cache 提交和 rejection correction 仍需在真实 engine 中计时。不能用离线 K=4 命中代理推出 `3.67x`：后者只是 Target block 的理想 pair-level 摊销，且当前 gap 表还没有进入 exact acceptance 路径。

工程上下一步必须保留计数并归一化为真正的 `Q_D^P`，显式处理合法 gap、BOS/边界/EOS 和 token-type mask，再实现独立 adapter。验证顺序为：小词表 `P/Q` residual correction 分布测试 → Position rejection 后 Target cache 前缀测试 → 256-token engine smoke → 1024/2048/4096 的多 seed fair wall-clock。只有 `sum_v min(P_T,Q_D)`、实际 Position acceptance 和端到端 tok/s 同时成立，才把该查表模型作为 D8 的可替代 draft 或 cascade 分支。

早期探索中曾出现 `83.04%` top-1，但固定脚本重跑未复现；后续报告采用上述可复现的 `85.04%`（h=5,bits=1）和对应代理指标，并把早期数字视为未确认结果。artifact 位于远端 `profiling/s8_structured_position/h5_b1/`，没有把生成文件提交到本地仓库。

### 10.8 结构化 Position proposal 的归一化与 residual correction 验证（2026-10-04）

补做的 proposal exactness 前置实验将 h=5、低 1 位的 gap 表归一化为 `Q_D^P`。在 4 条验证轨迹的 4092 个 transition 上，`Q_D^P` 与验证 corpus 条件经验分布的平均 overlap=`0.84520`，真实 gap 的平均 proposal 概率=`0.78636`；它们只是 teacher-prefix/corpus proxy，不是 Target logits acceptance。100,000 次 synthetic residual correction 的经验 acceptance=`0.44762` 对理论 overlap=`0.45`，输出 L1=`0.00340`；一个 82 次观测的真实 context 上，经验 acceptance=`0.86612` 对理论 overlap=`0.86697`，L1=`0.00324`。这一步确认了 `Q_D^P` 归一化、ratio acceptance 和 residual correction 的数学实现，可以进入 adapter；仍不能外推到 3.67x 或端到端 speedup。结果位于远端 `profiling/s8_structured_position/h5_b1/structured_exactness.json`，下一步是用 Target neural logits 重算 overlap 并验证 cache rollback。

### 10.9 近零成本结构化 Position draft 的真实 Target 路径结果（2026-10-05）

首个端到端原型已经完成：StructuredPositionGapDraft 从 h=5、低 1 位结构化 gap 表生成 Q_D^P，不执行 Draft Transformer；PositionSelectiveSpeculativeEngine 仍负责 Target 两 token 验证、Position ratio acceptance、residual correction、rollback 和真实 Target Feature 采样。256 token、128 pair、seed=1 的固定长度运行中，Target AR 为 71.225 tok/s，结构化 adapter 为 97.354 tok/s，wall-clock speedup=1.36685x。Position acceptance 为 115/128=0.89844，Position overlap=0.89454，cache invariant 通过 128/128 次。

该结果把 S8 的离线可预测性推进到真实 Target neural logits 下的 acceptance 和吞吐证据，也说明降低 C_D 比继续扩大神经 draft 更直接。但成本仍不是字面上的零：engine 内 adapter lookup/support 映射均值约 122.6 μs/query，且当前只提出 Position，Feature acceptance 和 joint pair path 仍关闭。它证明的是 Position-only、exact-correction、negligible-cost proposal 可以带来正收益，不能宣称已经实现理论 3.67x；后者只描述理想 Target block 摊销。

当前裁决：保留该 adapter 作为 D8 的可选 micro-draft 分支，默认路径暂不替换；先把 runner 中的 draft-forward hook 固化为正式接口，随后在 1024/2048/4096 token 和多 seed 上复测。若长序列中 alpha_P 稳定且 table/support/synchronization 成本仍远低于 Draft Transformer，再评估 cascade（高置信度命中查表、低置信度回退 D8）；Feature-aware training 继续作为独立的联合 acceptance 研究线。

### 10.10 结构化 Position draft 的长序列 scaling 结论（2026-10-05）

同一 h=5、低 1 位表模型在 256/1024/2048/4096 token 端点上的实际 speedup 分别为 1.36685x、1.34220x、1.34804x、1.35691x；Position acceptance 为 0.89844、0.89648、0.90332、0.90820，Target neural logits 下的 Position overlap 为 0.89454、0.89961、0.90455、0.90903。每个长度的 Target/Draft cache invariant 均通过，Feature 仍由 Target 采样。

因此当前最强结论是：negligible-cost structured Position proposal 在现有 GaussianGPT 条件链上已经有可重复的单 seed 长度收益，且没有出现随 context 增长而恶化的 acceptance 或 rollback 问题。收益上限仍受 Target verification、sampling 和 host lookup 限制，所以它是约 1.35x 的可用 micro-draft 基线，不是理想 Target block 的 3.67x。

进入默认路径前的要求改为：扩展到多个 scene split；确认 lookup/support mapping 在 GPU 实现下不会随长度线性放大；做 table hit→D8 fallback 的 cascade 对照；并加入输出 token 分布和渲染质量回归。Feature-aware draft/joint acceptance 继续单独推进，不把本 Position-only 结果解释为 Feature 对齐已经解决。

### 10.11 多 seed 结果图与稳定性判断（2026-10-05）

S8.4 的 12 个 endpoint 表明，近零成本结构化 Position proposal 的真实 wall-clock 收益在同一场景/checkpoint 上具有跨 seed 和长度的稳定性：speedup 均值为 1.396x、1.391x、1.360x、1.365x（256/1024/2048/4096），Position acceptance 均值为 0.938、0.910、0.905、0.911。长序列没有出现 acceptance 崩溃，说明当前 h=5 结构化上下文至少能保持稳定的 Position proposal 质量。

结果图同时暴露出收益瓶颈：lookup 成本由约 133 μs/query 逐步升至 253 μs/query，结构化吞吐随长度从约 100 tok/s 降至约 91 tok/s。因此下一步应把 host dictionary/support mapping 迁移到 GPU 或 fused kernel，再用相同的多 seed grid 复测；否则继续优化表的预测质量可能被 host overhead 抵消。该结果仍然不能解释为 Feature 对齐、联合 acceptance 或理想 Target block 的 3.67x，后续需要多场景和 Feature-aware proposal 的独立证据。
### 10.12 S13 实测后的蒸馏路线更新（2026-10-07）

S13b 的三种子真实联合结果为 P 接受 `0.9479`、条件 F 接受 `0.4231`、成对接受 `0.4010`，平均吞吐约为纯 Target 的 `0.9104x`。这与 10.6 的判断一致：F 的接受能力和提出成本抵消了 P 的高接受率。与此同时，S8.4 已在同一场景的 12 个 seed/长度组合中测到结构化 Position proposal 约 `1.36–1.40x` 的 speedup。当前应扩展这条 P-only 路线到多个场景，再用它作为神经 Draft 的性能参照。

S13 已使用 Draft 自生成前缀和 Target 分布标签，后续训练需要补上两项。第一，采集真实 verification 第一次拒绝后的 `D_reject`，包括被拒候选、Target residual 采样结果，以及之后 2–8 个位置的真实状态，参考 [Draft-OPD](https://arxiv.org/abs/2605.29343)。第二，按 P/F 类型、Target 熵、Target/Draft 分布差距和实测接受率挑选可学样本，参考 [AdaSPEC](https://arxiv.org/abs/2510.19779)。[DistillSpec](https://arxiv.org/abs/2310.08461) 的 on-policy 分布蒸馏继续保留，但权重选择改用接受长度与完整 wall-clock utility，不再单看验证 Feature KL。

若下一轮目标是联合 Draft，P/F 在同一共享模型、同一次训练中共同更新，按类型分别统计并加权损失；无需拆成两套独立模型。若下一轮目标是 P-only，F 由 Target 采样，神经 Draft 先优化 P 目标，F 辅助损失必须作为独立消融。该结论是依据当前小样本结果形成的研究建议，不代表本轮已确认场景重建质量变化。执行步骤见 [S14 更新方案](JOINT_DRAFT_TRAINING_PLAN.md#17-s13-结果后的-s14-更新方案参考研究报告)。


## 2026-10-20 PF双候选的有限真实诊断

既有step12000 PF在10开发来源上完成480条4096维真实分布与两模式各48条分支P2条件；新增训练0，整体193.622秒。第二候选β为15.2197%/14.3099%，有利事件每轮点估计8.5644%/5.7112%；来源区间不包含上下文和分支抽样误差。2-token与4-token在有限配对上下文上耗时接近，配对只有2唯一上下文、1来源。当前没有得到减少实际Target调用的明确实施机制，不进入惰性W2完整实现，继续由原网页审阅下一方案。来源覆盖、KV、概率权重、计时和限制见 [R1成本分析](artifacts/pf2x_loop_20261019/round01/cost_probe_review_20261020/FEASIBILITY.md)。本轮没有同轮Target-only、独立来源或全长三维评价，不形成新的2×结论。


## 2026-10-20 PF-TypedVerify有限微测

原ChromeRound02方案尝试按P/F类型合并Target输出计算，沿用原权重、FP32、采样与W1规则。候选仅临时绑定在隔离进程中的同一模型实例，生产文件未改。真实4来源8上下文、1/2/4token各96对；688次数值对比通过，hidden/RoPE误差0，GPU有限logits最大差1.93119e-5。四token配对分组/原耗时中位0.995125，仅降约0.49%；一/二token中位耗时略增，四token全部时长均值无改善。65.529621秒后完成，未满足继续完整解码评价的条件。当前只否定该具体实现的扩展价值，不推出所有类型相关优化均无益；新增训练0，无新端到端速度与2×结论。原始配对和源码见artifacts/pf2x_loop_20261019/round02/，下一轮须回原网页选择。


## 2026-10-21 Attention与图重放的执行路线

现有MLP长续训和近期空间/状态预测没有可靠额外提速，本轮改从Target实际前向成本寻找空间。Attention局部实验显示，显式BMM在短上下文普通执行较慢，进入CUDA Graph后可显著减少该形状的时长；长上下文改善更大。下一步先完成真实Target的logits、pre-finalnorm hidden、概率、缓存与配对前向测量，再对原Target/原PF/新PF/同优化Target做完整解码。若仍未达2×，才用已保留的480行p/q估计多分支残差候选的真实新增推进与可承担成本。现阶段没有新完整解码倍率，不能据此替换原方法。方案和全部实际结果见artifacts/pf2x_breakthrough_20261021/。

## 2026-10-21 独立解码确认：主要收益来自执行路径

独立的30个场景、两种前缀长度和单seed组成60组配对。候选PF相对原Target为`4.519564 [4.228452, 4.720021]`，相对原PF为`2.613919 [2.506082, 2.672991]`；它相对采用相同Attention与CUDA Graph执行路径的Target为`1.442534 [1.372647, 1.495168]`。因此，超过2倍的数值由执行实现改进与PF解码共同产生，PF相对同优化Target的已测增益约1.44倍。原/候选方法在60组配对中的生成输出和计数一致。

独立确认的240次运行使用512-token上限。另有3个场景的自然EOS补测，共3组配对、12次解码，均由EOS结束且无预算截断；候选PF相对原Target、原PF、同优化Target的来源等权速度比分别为`3.630147 [2.791982, 4.246229]`、`2.209333 [1.987015, 2.373086]`、`1.273811 [1.019382, 1.443080]`。8项审计通过，旧/新Target与旧/新PF分别3/3组输出和计数相同。三个来源带来的区间较宽，结果作为小样本补充，不替代30场景的512-token矩阵。

另完成一个候选PF自然EOS样例的三维有效性验证：153个Position与153个Feature token得到11,286个Gaussian，六个`384×384`视图均为有限数值并保存，alpha coverage为`0.050615`。单个chunk的图像非空但结构稀疏，当前只确认解码和渲染链路可用，没有重建质量提升证据。旧`full3d_validity_run01`中1:1数量假设的失败记录保留；`run02`的VQ路径具有三级、总因子8的上采样，Gaussian/Position数量没有已证实的固定比例。完整自然EOS数据见[natural EOS目录](artifacts/pf2x_breakthrough_20261021/natural_eos_run01/)，三维数据见[run02元数据](artifacts/pf2x_breakthrough_20261021/full3d_validity_run02/metadata.json)与[视图0](artifacts/pf2x_breakthrough_20261021/full3d_validity_run02/views/view_00.png)。


## 2026-10-21 真实 Feature W2：接受率增加未形成可靠净收益

原Target口径的2倍已由既有执行优化加投机解码达到。进一步使用现有Ranker+12k MLP实现精确无放回双候选和树Target验证，完成两轮各72次；最终W2/原Target=4.769174，W2/当前MLP=1.012835且来源区间含1，W2/W1=0.960329且三个开发场景均低于1。Feature任一接受率增加7.14个百分点，Target调用只减少1.37%，额外树计算仍占成本。

当前停止扩宽，不自动推进W4或新训练。单Feature加宽尚未证明能突破当前速度；今后若转向长多步或成本自适应提案，需要先说明如何减少实际Target工作，并测完整成本，不能仅凭更高接受率启动。DFlash、WTV、官方EAGLE-3适配与此前暂停的MultiTap均未在本轮开始。W1的约5.47%执行增益尚需独立场景确认，不能算作双分支算法增益。见[完整结果](artifacts/feature_tree_20261021/RESULTS.md)。
