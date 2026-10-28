# GaussianGPT 项目交接入口

## 2026-10-28 5090 多 Chunk 采样分布核验

扩展和空列重采样的 8 个重建前缀完成 80 组概率对照，最大总变差距离 2.0862×10⁻⁶，top-p 保留集合相同；启动流重放及结束符分支完成。结果支持采样规则的分布保持，但原始调用前缀未保存，不能据此证明整场所有路径完全一致。范围、数值与程序路径见 [采样分布核验](../docs/project/MULTICHUNK_DISTRIBUTION_5090_2026-10-28.md)。

## 2026-10-28 作者训练流程研究完成

已读取官方源码 `031d916`、论文 v2 与 RTX 4090D 上两份官方 checkpoint 的训练元数据。当前 GPT 保存于 epoch99/50300 次更新，VQ-VAE 保存于 epoch19/10080 次更新；GPT 实际采用 Muon+AdamW 分组学习率，当前默认配置与历史记录存在差别。数据、空间块、P/F 监督及完整 D8/轻量 Feature Draft 的衔接方案见 [作者训练流程还原](../docs/project/AUTHOR_TRAINING_RECONSTRUCTION_2026-10-28.md)。作者预处理数据尚无公开发布证据；本轮没有训练、GPU 前向或性能测量。

## 2026-10-28 RTX 5090 多 Chunk 场景对比完成

用户指定 12 米 × 12 米场景，原 Target、W1、Rescue 均完成，场景调度验证和产物读取通过，不重复启动。三者请求时间为 9668.703、2364.214、2444.627 秒；一个 seed 的输出数量与空列数不同，不能当作等工作量加速。W1 与 Rescue 的完整 token 和列长度一致，Target 的生成轨迹不同。参数、结果和自动汇总重试记录见 [多 Chunk 运行记录](../docs/project/MULTICHUNK_5090_2026-10-28.md)。

## 2026-10-28 RTX 4090D 多 Chunk 阶段帧复现完成

AutoDL 项目 `/root/autodl-tmp/GaussianGPT` 已按 CSDN 思路生成 4×4 场景、10 张阶段帧和合成 GIF。默认采样参数得到 4/16 个非空列；阶段导出前后的 token 序列和列长度相同。文章未列列占用数量，不能确认作者运行是否同样稀疏。远端源码变更、产物路径和归因依据见 [AutoDL 复现记录](../docs/project/GAUSSIANGPT_AUTODL_2026-10-28.md)。

## 2026-10-28 RTX 5090 迁移环境

迁移后的主工程为 `/root/workspace/GaussianGPT`，使用原 `gaussiangpt` Conda 环境。5090 所需的 MinkowskiEngine、gsplat 和 PyTorch3D GPU 扩展已重新编译，FlashAttention 更新为 2.8.3；原扩展文件保存在 `/root/workspace/gaussiangpt_5090_env_20261028/original_extensions/`。Target、W1、Rescue 的普通执行与 CUDA Graph 小规模输出对照通过，VQ 三维解码与六视角渲染通过。环境版本、操作记录及验证范围见 [5090 迁移记录](../docs/project/GAUSSIANGPT_5090_2026-10-28.md)。没有训练或新的性能比较。

## 2026-10-27 方法与自然 EOS 三维证据补全

继续沿用现有 PFSpec 主线。当前 [PAPER_MAIN.md](../docs/project/artifacts/paper_v4_20261023/PAPER_MAIN.md) 的 Method 3.2、附录 C.6 已补齐合法候选规则、32 维历史输入、150 轮梯度提升树、Feature 残差网络及主训练配置；新增方法图与自然 EOS 结果图。

此前完成度讨论遗漏了已有阶段 F：270 次补全加 90 次 BOS，360 次均自然 EOS；基础轻量 P/F 的请求时间比为 1.569349×、1.790705×。现已整合到正文 5.7、表 5、附录 C.7，并直接复用两张六视角预览和两份完整视频。该批采用较早执行后端、没有 Rescue，不能记作当前完整 PFSpec 的全长结果；30 来源也不与其他批次累加。两份 CSV 的点值与完成状态已复算，没有新 GPU 或三维生成。

本轮图文修改见 [修改说明第 9–10 节](../docs/project/artifacts/paper_v4_20261023/CVPR_REVISION_NOTES_20261027.md)。现阶段只考虑补齐当前共享执行后端与单备用 Rescue 的自然 EOS 增量；不重复已有 360 次全长基础路径，不自动增加模型、训练或空间模块。GPU 运行入口已向用户询问，未发起远端新实验。以下内容按日期保留历史状态。

## 2026-10-27 面向 CVPR 的统一正文改写

当前 PAPER_MAIN.md 已整理为正式中文论文稿，按“结构信息减少草稿状态计算 → 位置投机与 Feature 路径净增量 → 剩余恢复预计算”组织。主文包含完整摘要至结论，详细推导、负向结果和实验范围移入附录 A–C；共享 Ranker 的实验目的、主要消融及适用范围均保留。修改说明第 8 节记录本轮写作调整。没有新增实验、算法、权重或原始数据变动。以下段落记录此前各轮工作；章节结构及写作状态以本段和修改说明第 8 节为准。

创新性机制对照见 [NOVELTY_ARGUMENT.md](../docs/project/artifacts/paper_v4_20261023/NOVELTY_ARGUMENT.md)。作者已明确共享 Position Ranker 的实验目的：先取得位置投机的有效加速起点，再评价 W1 与 EAGLE 式 Feature 路径能否保留并扩大收益。正文第 5.2 节及论证文件已同步：Position-only 为 1.395117×，W1 为 1.551487×，相对位置路径增加约 11.2%；当前 EAGLE 适配为 0.966838×，其完整安排未保留位置投机收益。“保障性”只作为已测经验起点，不表示任意附加草稿的速度保证；共享机制也不保证实际生成轨迹相同。状态推进、训练、缓存和轮次安排的归因限制仍保留。Medusa/Hydra 不属于共享 Ranker 的增量组。引言未再改写，没有新增实验、模块或运行源码修改。

后续评审反馈已整合：摘要与引言突出“合法位置排序 + 已有 Target 状态”的低成本 P/F 路径，Rescue 沿剩余恢复开销引出，摘要仅保留 1.58× 和 4.31×。引言加入三维先行工作；Related Work 根据 SpecTr 出版 PDF 第 4 节、第 5 页保留条件分布复用先例，并与隐藏状态及 KV 恢复区分。详细反馈处理见修改说明第 7 节。没有新增实验或运行源码变动。

用户确认以 P/F 类型化低成本提案为主贡献、固定 W1 下的恢复预计算为第二贡献，并授权统一改写。当前 [PAPER_MAIN.md](../docs/project/artifacts/paper_v4_20261023/PAPER_MAIN.md) 已按 Abstract、Introduction、Related Work、Method、机制分析、Experiments、Discussion 与附录组织；下方“摘要逐字保留”和旧章节编号属于历史状态。主要变更与证据范围见 [修改说明](../docs/project/artifacts/paper_v4_20261023/CVPR_REVISION_NOTES_20261027.md)。

正文明确 Ranker 的确定性点提案、Feature MLP 的隐藏状态残差预测、共享 Target 输出头，以及 Rescue 只覆盖固定轮次中的第一 Position 恢复。CUDA Graph 属于共享执行实现；1.58×、5.13×、独立重复约 1.8% 和完整请求 4.31× 分别解释。已有 480 次主比较和 360 次结构比较均已使用，不重复安排；同预算结果不支持 MLP 普遍更优，候选 Position 和历史 Feature 的独立速度收益尚不明确。

本轮只读源码与实验记录，并从原始 JSONL 复算主要点值、480 对重复输出及恢复差额、60 对新种子机制记录和 20 对完整请求。相关点值与报告一致，恢复差额逐对等于备用命中。未进行 GPU、SSH、训练或新性能测量，运行源码、权重、原始记录和 PROJECT_STATUS.md 均保持。

## 2026-10-26 Introduction、Related Work及方法分析写作

当前正文已将第1节写为Introduction，并在Method前加入Related Work中文初稿。 本轮依据用户进一步修订的版本，以低成本主路径推进和恢复条件计算两类需求组织叙述。引言将约1.8%标明为四轮重复；新增LTD、SpecBranch、SSD官方来源及与Rescue的计算对象区别，具体速度比较仍限已测适配配置。引言围绕推进与完整成本的矛盾、P/F类型化预测和恢复状态复用；相关工作引用原文，承认树验证、隐藏状态起草、候选命中复用及成本感知选择的已有基础。具体机制结论限原文已读范围，未把通用方法的全部移植列为已完成。

用户确定两项具体分析重点：P/F条件接受与W1推进、两类Position恢复的联合刻画；Rescue在固定参考抽样下的条件状态复用。当前[正文](../docs/project/artifacts/paper_v4_20261023/PAPER_MAIN.md)第2节为正式Method，第3节为Performance Analysis。式A为 \(E[N_r\mid s]=2+2j-e\)，式B为 \(E[K_{\rm rec}^{W1}\mid s]=(1-w)+j(1-\beta_{P_2})-e\)，合法备用的式C减去 \(p_P(b\mid h)\)。三式仅用于完整候选、非EOS提案、预算至少四token的适用轮；EOS项覆盖两个残差终止事件。位置恢复调用与总Target计算、W1分布保持与Rescue耦合轨迹保持分别论证。

当前rounds在BOS启动后递增。第3.6节明确 \(N=N_{\rm boot}+\sum N_r\)，区分 \(G^{request}=N/R\)、\(G^{loop}=(N-N_{\rm boot})/R\)；原Target记录的rounds=0不用于轮次分解，直接比较N/T。Rescue成本条件为完整记账下的局部充要关系，不是独立速度预测或全请求最优结论。有限精度边界保留。状态s的概率分析不条件于未使用随机结果；T_init明确包括BOS启动生成与其他循环外成本。

第6节按RQ1–RQ3组织，并将通用方法比较、结构输入消融、W1/W2成本和Rescue配对逐一对应。W1/W2推进+4.37%、摊入完整轮次成本+8.68%、吞吐0.960329；阶段E PF-D8与轻量MLP分列批次。Feature-only未完成；5.13同轮归因与旧确认、四轮重复、新种子Rescue增量分列。摘要逐字保留。

本次只改正文和既有说明，无新GPU、训练、SSH、运行源码、权重或原始测量变动；不新增动态选择或抽象预算模型。

## 2026-10-25 同执行条件的方法与结构比较完成

用户授权开始已有通用提案器及等预算消融的直接比较。新入口为`patches/pf_comparison_20261025/compare.py`，远端`/root/autodl-tmp/gaussiangpt_pf_comparison_20261025/source`。`validation_run02`的24方法、两种前缀、48组普通/图执行输出与动作对照全部通过；首次失败为B1-E在前向内创建固定遮罩，现仅将该常量提前准备，原/新输出精确相同，旧权重与源码未改。原失败记录留在`validation_run01`。

`main_run01`的30既有来源、64/256前缀、seed57101、512token、8方法480次全部完成。PFSpec/优化Target为1.575053 [1.487180,1.644946]；当前EAGLE v3适配版0.966838，Medusa低秩0.780244，Hydra条件0.719455。Rescue/W1为1.015189 [1.004406,1.022387]，60/60输出与接受相同，少357次恢复且等于命中，额外4015备用节点、净输入增加3658。SSH断开后原任务仍完整运行，没有补跑。

`mechanism_run01`的10开发来源、256token、18配置360次全部完成；B0/B1分列三种训练seed，四种输入消融复用原类，所有Feature模型使用同一四节点W1循环。无Target状态/同预算完整版0.893863 [0.832795,0.962154]；去候选Position和历史Feature均无明确独立速度收益。B1-E/B0分别1.064/1.046/1.016，仅中间种子区间不含1，不能声称MLP普遍更优。当前完整结果、原始记录与适用范围见[本轮报告](../docs/project/artifacts/pf_comparison_20261025/RESULTS.md)，正文第6节已更新。模型训练来源、参数量及原生控制流仍有差异，结论限当前适配版本。

已通过用户提供的SSH方式连接；凭据不保存到项目。远端根为`/root/autodl-tmp/gaussiangpt_pf_comparison_20261025`。结束RTX4090为2MiB/0%、无计算进程，主工程HEAD031d916、4181条既有工作区记录、三权重大小及修改时间和821/822服务保持。没有安装、新训练、生产修改或新增三维输出，摘要原文保留。两个矩阵均complete，不重复启动。

## 2026-10-25 摘要定稿与正文研究目标

用户确认摘要文字定稿，后续围绕方法有效性、P/F结构贡献和性能机制推进，采样分布保持单独论证。当前写作入口仍为[正文](../docs/project/artifacts/paper_v4_20261023/PAPER_MAIN.md)：第7节保存用户定稿原文，第6节已整合三项问题的正文初稿与已有数据。1.583618是相对同优化Target的结果；5.131692包含执行优化；四轮Rescue为1.018092；完整请求4.307862保留512上限与长度差异。当时待完成项为把已有适配通用提案器及等预算消融接入相同执行路径比较，现已于2026-10-25完成；不由历史训练与缓存差异推出普遍优势。现有证据更支持Target状态复用，候选Position输入的独立速度贡献尚不明确，显式XYZ打分无可测增量。本次只改既有两份Markdown，没有新训练、GPU、SSH、性能测量或状态快照更新；不重复已完成归因与备用数量实验，不启动动态选择训练。

## 2026-10-23 新概率模型与真实动作成本开发验证完成

恢复价值证据的新一轮已完成：原8/2来源浅树模型、33状态515动作1545重复、七方式两进程280条解码与12条在线诊断。新价值/固定单备用0.993432 [0.990375,0.996144]，对概率规则1.000437区间跨1，仍无强对照新增收益；没有新增来源外或完整三维请求。9540组CPU输入、2166次在线决策及158次备用ID核验通过，不能将不足归为选中与执行不一致。较小备用MSE没有带来更好候选选择，主概率校准与首次图费用仍有缺口。计时465.552秒+在线40.463秒，成本/配对88.085秒，任务均完成，不重复启动。GPU已空闲，原权重与服务保留。入口[本轮结果](../docs/project/artifacts/recovery_value_evidence_20261023/RESULTS.md)。

## 2026-10-23 动态恢复选择的实现维护完成

已按用户要求先核验实现再解释异常结果，规则写入AGENTS.md。原动态路径的冗余前缀、诊断快照、深拷贝及空池回调已移出正式执行；快速选择器保留原动作和数值语义，成本特征改用实际计时状态。三个既有开发来源七臂两进程168解码、84完整请求及12逐轮诊断完成，输出/调用/实际备用ID均通过。新/旧动态解码1.058868、完整请求1.046277；同入口动态/固定单备用0.990502/0.992057，仍无算法新增收益证据。旧20来源结果只描述旧版实现，不能据此否定研究方向。本轮为已有来源上的实现核验。主矩阵567.770秒、在线诊断39.661秒，GPU已空闲，原工程、权重与服务保留；不要重复启动。完整报告见[实现核验](../docs/project/artifacts/recovery_value_execution_20261023/RESULTS.md)。

## 2026-10-23 恢复价值关键证据实验完成

[完整结果](../docs/project/artifacts/recovery_value_20261023/RESULTS.md)。20新来源组、64/128前缀、512上限，旧10来源开发、模型8/2留源，参数固定后两进程800解码及400完整三维请求完成。value/单备用解码0.932238 [0.928435,0.936197]，完整请求0.941442 [0.933740,0.947926]；相对概率成本和同数量Ranker区间跨1，未得到来源外强对照优势。单备用/W1来源等权解码1.013946、完整请求1.008566；完整请求累计129.007296→128.991790秒，仅省0.015506秒，累计比1.000120 [0.982643,1.014994]，不概括为稳定端到端净收益。

1200记录、960输出/语义/调用对照、600跨轮与400跨阶段对照通过；400NPZ与2400PNG有效，无空白视图，Gaussian/像素有极小差异。解码Luna独立复算通过；其最终E2E任务额度用尽无交付，主代理从原始记录直接另算四项点值/区间，与主统计完全相同，未将未交付工作计为证据。神经权重和生产入口保留。RTX4090结束2MiB/0%，无计算进程，远端HEAD031d916/4181记录、原权重大小及821/822服务保持。原始记录在本机artifacts/recovery_value_20261023、源码patches/recovery_value_20261023，远端/root/autodl-tmp/gaussiangpt_recovery_value_20261023；matrix_progress为complete，不重复启动。当前动态价值选择只作为负结果和边界，v4正文仍以固定恢复预计算为主；单模型、单卡、有限chunk与512上限不变。

## 2026-10-23 融合版v4写作入口

采用用户v4的单一科学问题与章节结构，当前写作先读docs/project/artifacts/paper_v4_20261023/PAPER_MAIN.md及REVIEW.md。Downloads原文、旧PAPER_OUTLINE与v3记录保留为来源，不再用旧大纲顶部的“主入口”作为当前选择。v4需补全任意k子集可行等前提，并加入四轮1.018092、同轮分解5.131692及完整请求4.307862等已有证据；新增完整请求收益相对W1为1.001848、区间跨1，不能写成已稳定提高。两种39状态分析分列，当前不训练价值模型。XSpecMesh官方出版身份已确认ICML2026；下方未确认状态保留为历史。只新增两份静态文档与入口，无GPU、SSH、训练、运行源码或测量变化，生成状态快照不重写。

## 2026-10-23 v3材料与四轮重复复核完成

用户提供Downloads中的v3蓝图和CPU核验脚本/JSON，已完整阅读、精确算术重放及三篇最近方法原文对照。当前入口docs/project/artifacts/v3_review_20261023/DECISION.md、THEORY_REVIEW.md、NOVELTY_EQUIVALENCE.md及EVIDENCE_AVAILABILITY.csv。v3中k>1未实现/未测的状态已过时：position_widths_20261022六臂360次及spec_next/PAPER_NEXT.md归因/端到端均已完成，不重跑。完整历史与固定Target使p对σ(H)可测，低成本边界应限定压缩输入与计算预算；在线选择器有效性仍未证明。旧完整解码未保存逐轮候选/p/纠正事件/恢复毫秒，39固定前缀只有有限离线用途。 已完成同池枚举：k1仅2/39上下文有概率质量增加，来源等权+0.6764个百分点；k2..4无增加。当前不训练价值模型，先读POOL_ENUMERATION.md。

此前四轮960条再复算，单备用/W1=1.0180919，来源区间约[1.0148,1.0213]；480/480输出语义相同，少2908恢复，28/30来源正，首次图子集较慢。新解释rescue_repeat_20261022/RESULTS_REVIEW_20261023.md与analysis_review_run02，旧243条顺序提示由分析器对实际prompt轮换及已旋转来源的判读造成，已处理并保留旧产物；新差异0，倒置配对仍被识别。今天无新GPU/训练/部署或远端源码修改，GPU2MiB/0%、原服务与权重大小保持。Downloads原件保留。

## 2026-10-23 理论优先的算法论文整理

用户要求阅读投机解码和mesh原论文，从问题建模到理论、机制、章节与证据完成学术整理。主入口为docs/project/artifacts/paper_design_20261021/PAPER_OUTLINE.md，计划WORK_PLAN.md，正文笔记SD_PAPER_READING.md、MESH_PAPER_READING.md、DOMAIN_PAPER_READING.md，独立审阅THEORY_REVIEW.md。THEORY_NOTES.md顶部2026-10-23节为当前数学依据，RESEARCH_STORY.md已有同日入口。

主问题为固定参考抽样过程下的恢复条件状态预计算，CUDA Graph留在执行实现。SpecTr候选成员复用、通用树验证、成本选树与Drafter-Invariant都有前例，不声称首次、唯一或一般最优。新增集合选择与估计误差分析尚未实施策略；现有固定W1／单备用与2026-10-22证据保持。当前大纲完成不代表新颖性与迁移已经解决。只有静态文档和公共文献资料变化，无SSH、GPU、训练、部署或运行源码改动，PROJECT_STATUS.md保留原时间。

本轮文献与整合已经完成，两个文献子Agent及理论子Agent均已关闭。共14篇正文材料；FlashMesh为CVPR 2026，XSpecMesh录用身份未确认。当前整体模型分类型条件提案q与固定参考恢复集合B两层，前者有主位置概率加权TV风险与条件差异界，后者有轨迹保持、恢复测度、受限选择与误差界；均不宣称联合全局最优。新增选择、主概率加权训练与mesh迁移尚未实施。

## 2026-10-22 论文证明、贡献分解与单备用端到端完成

用户同意开始论文下一步。原Target/优化Target/W1/单备用四配置30来源×64/256×seed51101×512token共240次完成，远端spec_next/paper_attribution_run01，入口source_paper_next/evaluate_attribution.py。直接调用旧source_run02和feature_tree/source_run04，不改权重或训练。同轮分解3.240487×1.557471×1.016788=5.131692，单备用/优化Target1.583618。原/优化Target60/60输出计数相同、旧三配置180/180输出counters相同，命中388省388调用。详见docs/project/artifacts/spec_next_20261022/PAPER_NEXT.md，不重复派发。

前10来源20输入target/w1/rescue端到端60次完成，w1_e2e/rescue_paper_run01。全量备用/原Target4.307862×，共同满512的19配对4.185458×；备用/W1全量1.001848区间跨1，无稳定新增端到端收益。20/20tokens相同，60NPZ/360PNG有效。Gaussian行序变化，坐标一一对应最大距离5.977205e-7、最大字段相对L2 3.048302e-7，PNG变动通道最高0.00881619%，非逐位相同。理论新节在既有THEORY_NOTES.md，当前故事更新RESEARCH_STORY.md。主代理完成实现、原始记录复算与产物读取；本轮子代理均关闭且未交付，不作为证据。累计GPU外层1005.96秒<1200，GPU2MiB/0%、无计算进程，HEAD031d916/4181记录、原权重/Ranker/821/822服务保持。无新训练、自然全长或绝对场景质量结论，任务完成。

## 2026-10-22 用户要求复跑旧5.056×配置，已完成

直接运行既有spec_next/source_run02，不用widths扩展版；30相同来源×64/256×seed51101×512token×target/w1/rescue共180次complete。60份prompt相同，180/180输出、EOS、长度、全部counters和执行顺序与旧轮相同；Python/Torch/GPU版本相同。备用/原Target5.090087 [4.593280,5.421318]，旧值5.056076，本轮+0.672674%；备用/W1=1.015750 [1.001101,1.024756]。恢复1181→793、命中388相同。唯一结果目录远端spec_next/reproduce_5056_run01，已完整下载六文件且逐字节与远端相同，原结果和权重服务保留。完整命令见本机docs/project/artifacts/spec_next_20261022/REPRODUCE_5056.md。运行器605.295049秒，外层607.18秒；GPU2MiB/0%、无计算进程，临时SSH关闭。原source_run02不改，主工程HEAD031d916/默认4181记录保持。运行器、既有统计程序、主代理直接原始记录复算相同。任务完成，不再运行GPU或重复派发。

## 2026-10-22 备用1–4比较完成，保留单备用

30既有来源、64/256、新seed53101、512token上限、原Target/W1/备用1..4六臂360次全部完成；无新训练。单备用/原Target=4.905899，备用2/3/4=4.869986/4.834639/4.813895，相对单备用=.992680/.985475/.981246；暖态也无可靠扩宽收益。四臂各60/60输出/接受计数与W1相同，恢复1051→709/656/647/647，各排名新增命中342/53/9/0。首图计入，54满512组单备用/原Target5.176860为子集，不替代主值。旧5.056076保留历史；四进程重复1.0180919仍为单备用增量独立证据。

数值run03和同图成本widths_costs均8条passed，实际一个来源两个前缀，构造位置仅路径数值；成本记录300次T1实际图前向，不使用旧非图恢复成本推算速度。原始完整记录、两套独立CPU复算与RESULTS.md在docs/project/artifacts/position_widths_20261022。GPU累计1008.297937秒<1200，全部任务退出，2MiB/0%，HEAD031d916/4181记录、原权重与服务保留。原source_run02和主工程不替换；实测本机patches/spec_next_20261022，远端同根source_widths。本轮完成，不重复既定矩阵、不自动扩大或继续训练。下方运行中/SSH故障记录保留历史。

## 2026-10-22 备用数量主测已在连接恢复后启动

连接恢复后确认此前widths_main未派发，GPU空闲；已用四次测试实际墙钟103.312694秒，沿用原1200秒预算。首次正式启动30来源×64/256×seed53101×512token×原Target/W1/备用1..4，内部950秒、外层980秒，不重复启动。完整数值run03已下载，8组passed，实际一个来源两个前缀。主测当前运行，先读position_widths_20261022/EXECUTION_STATUS.md最新节；四个运行文件与本机逐字节相同，原HEAD/权重/服务保留。下方SSH故障时状态已由此条更新。

## 2026-10-22 四次重复已完成，正在比较备用数量

本轮接续更新：备用1..4实现、20次短解码与8组GPU分支数值已完成；六臂512token主测因SSH掉线未取得结果，是否真正派发也未确认。原远端spec_next目录source_widths/widths_numeric_run03保留；本机数值目录只有部分复制，先读position_widths_20261022/EXECUTION_STATUS.md的结束状态。先恢复读取widths_main/pipeline与进程，不重复启动已有任务；本轮GPU调用保守上限1185秒，没有安排自动续跑。当前最新已完成原Target倍率仍5.056×，单备用四轮重复新增确认约1.01809×。不得写成备用扩宽已取得新提速。

原rescue_repeat四批已全部complete，960条/480配对/120唯一case，独立复算备用/W1=1.0180919，95%来源区间约[1.0148,1.0213]。480/480输出与接受计数一致，恢复差额2908等于备用命中；各进程均有正收益，首图请求仍较慢。资料已完整下载至原目录，不重跑。下方旧标题保留启动时状态。

本会话扩展原patches/spec_next_20261022，测试固定备用1..4，不训练、不增加Feature分支或动态策略。新GPU累计预算1200秒；远端原主工程/root/workspace/GaussianGPT与feature_tree的source_run04依赖保留，本轮实测源码使用既有spec_next根目录source_widths。先读docs/project/artifacts/position_widths_20261022/PLAN.md与EXECUTION_STATUS.md。20次短解码完成，各备用与W1输出/接受计数相同；数值、成本与六臂512token主测仍待完成。原5.056×不乘新比值，生产默认及权重保留。

## 2026-10-22 用户要求多次确认备用增益，四次重复正在执行

已开始固定四进程串行重复，同30来源×64/256×新seed52101/52102×W1/备用，每次240解码，总960/480重复配对/120唯一case。方法顺序交替，首图来源循环偏移0/8/15/23；算法与计时沿用spec_next的source_run02不改。远端/root/autodl-tmp/gaussiangpt_rescue_repeat_20261022，suite.json记录PID和进度（初始suite PID89035）；不要重复启动。SSH复用/tmp/ggpt_rescue_repeat_20261022.sock。本机patches/rescue_repeat_20261022及docs/project/artifacts/rescue_repeat_20261022，先读其中PLAN.md和EXECUTION_STATUS.md，全部四次到齐后按来源聚类统计，并由两个独立程序复算；旧结果不混入新主值。无原Target重测/新训练/三维端到端测试，原服务和权重保留。

## 2026-10-22 接受率与备用分支实验完成

十开发来源240次、三十来源组合240次、纯备用新种子180次及相应数值/解码对照全部完成。可用新候选为纯备用Position：三十来源60配对，相对同轮W1=1.017481 [1.002716,1.026950]、相对原Target=5.056076 [4.553989,5.386374]；60/60输出和alpha计数相同，恢复1181→793，388次差额等于备用命中。29/30来源更快，冷来源可能变慢，首图成本计入。Feature历史policy混合的实际接受33.2333%→34.5925%，但来源差额区间跨0，组合/W1=1.004189且区间跨1，不替换现用W1。直接归一Ranker概率及保守混合诊断也无足够新收益，不重复已完成矩阵。

本机patches/spec_next_20261022与docs/project/artifacts/spec_next_20261022；远端同名根目录source_run02为开发/纯备用实测版、source_run03为快速混合/组合实测版。查看[最终结果](../docs/project/artifacts/spec_next_20261022/RESULTS.md)与EXECUTION_STATUS.md。三十来源此前用于旧方法，追加纯备用在组合结果读出后决定；可选source_scope标签差异另有记录，不改原plan。全部GPU任务退出，GPU2MiB/0%、原HEAD031d916/4181工作区记录、权重与服务保留。无新训练、生产默认或三维端到端测量，原4.966/4.335范围不变。原三名Luna/max完成实现与独立复算，可继续复用。

## 2026-10-21 同前缀恢复成本验证结束，保留静态 W1

十来源39状态、8种子、K1/W1/W2、3次计时重复，2808行主测与480行独立分项诊断完成。来源留出：恢复事件成本/W1=0.987458 [0.953654,1.020480]，与分别拟合R/C在39/39状态选择相同；直接R/C=1.010234 [0.984362,1.043486]，没有稳定新收益。未启动本轮外部测试或在线选择，不再用相同开发记录调参。最终统计只用dev_analysis_v2.json（旧版重采样随机数问题已处理，旧文件保留）。32项最终GPU对照、两矩阵独立计数复算与480/480诊断输出对照通过，主代理独立评分八模型相符。九篇原始论文加一综述表明通用进展/成本与动态选树已有研究，不把pair公式或一般成本比值作为新理论。详见[完成报告](../docs/project/artifacts/recovery_model_20261021/RESULTS.md)。本机源码patches/recovery_model_20261021，远端同名根目录source_run03为实测版、source_final_run04为完整交接版；GPU已释放，生产、权重和服务保留。原4.966×与4.335×范围不变，无新端到端结果。

## 2026-10-21 论文理论与方法设计已整理

用户希望完善论文故事。已形成[研究主线与投稿定位](../docs/project/artifacts/paper_design_20261021/RESEARCH_STORY.md)、[理论草稿](../docs/project/artifacts/paper_design_20261021/THEORY_NOTES.md)及CPU计数/代数记录。建议主线为异构P/F提案、完整对进展/恢复成本与类型化执行协同。正常轮R=2+2I(P1与F1接受)，P2接受影响恢复成本；含EOS时减去相应终止事件。W1/W2各18条的提交与Target调用恒等式通过；自然W1整批EOS计数也相容。标准残差抽样与硬件感知选树已有文献，尚不主张首次或全局最优。恢复感知预算选择器仅为建议，未实现、训练或测量。研究路线读取六篇相关正文，近作范围仍不完整。本轮复用三名Luna/max，无GPU或远端操作，无新速度结果；原生产代码和权重保留。

## 2026-10-21 W1 端到端实测已完成

4.966× 对应同一 W1 实现已补足 VQ、六视角渲染、Gaussian NPZ 与 PNG 保存，42/42 次完成。原 18 个 512-token 输入的来源等权端到端倍率为 4.334555× [3.899431,4.576120]，同轮 token 生成倍率 4.874125×；另三个测试场景自然 EOS 为 4.869038× [3.361697,6.610150]，token 吞吐 3.559610×，W1 总 token 数少 31.35%，不要把自然结束倍率当成相同工作量提速。两组分别统计，权重加载和输入编码在计时外，首次请求建图计时、随后复用。42 份 NPZ、252 张 PNG 实物读取通过；固定组两臂各18/18、自然组各3/3与旧输出一致，Luna/max独立复算一致。源码 patches/w1_e2e_20261021，远端同名根目录 source_run01/main_run01；完整结果见 [端到端报告](../docs/project/artifacts/w1_e2e_20261021/RESULTS.md)。GPU工作已退出，原生产、权重和服务保留。无需重复已完成矩阵。

## 2026-10-21 Feature W2 已实测，停止扩宽

4.966× 对应 W1 的完整说明见 [Ranker+MLP W1 单分支图执行方案](../docs/project/artifacts/feature_tree_20261021/W1_SOLUTION.md)：包含算法流程、执行组成、主比较口径和数值范围。本次仅整理文档，没有新增实测。

用户同意后已实现Feature W1/W2精确树验证，复用原三名Luna/max。两轮各72次、3开发来源、64/256前缀、三种子、512token，全部完成。最终MLP/W1/W2相对原Target为4.708736/4.966189/4.769174；W2/MLP=1.012835 [0.987925,1.036427]，W2/W1=0.960329 [0.938755,0.988246]，无可靠净提升，未运行W4或新训练。W1/MLP=1.054675是执行收益，尚无独立场景确认，不计作双分支创新。各轮W1/MLP18/18输出计数一致，跨轮72/72一致；最终1742语义项与12概率审计通过，440逐元素allclose诊断保留。先读[最终结果](../docs/project/artifacts/feature_tree_20261021/RESULTS.md)与[执行状态](../docs/project/artifacts/feature_tree_20261021/EXECUTION_STATUS.md)，不要重复已完成矩阵。源码patches/feature_tree_20261021，远端同名根目录source_run04为最终已测版，source_final_run05为补足审计/描述的交接版。GPU已释放，HEAD031d916/4181工作区记录与原服务保留。MultiTap继续暂停，DFlash/WTV/官方EAGLE-3新训练未开始。

## 2026-10-21 原Target主参照与新方法筛选完成

用户最新要求以原Target作为主分母。新同轮120次已完成：MLP4.6365倍、EAGLEv3 2.9861倍、v4复用2.8876倍、Medusa2.4537倍，均为执行优化加投机收益；原Target口径2倍已达，MLP仍最快。原/优化Target20/20输出计数相同，四候选跨轮80/80相同，独立CPU复算10项通过。源码`patches/original_target_next_20261021/`，远端`/root/autodl-tmp/gaussiangpt_original_target_next_20261021/source_run01/`，结果`main_run01/`。先读[本轮结果](../docs/project/artifacts/original_target_next_20261021/RESULTS.md)与[新方法筛选](../docs/project/artifacts/original_target_next_20261021/NEXT_METHODS.md)，不重复已完成测量。下一项优先复用Ranker+MLP测Feature W2分支真实Target成本与精确验证，再考虑W4、UniVer与成本预算；DFlash/WTV/官方EAGLE-3需要进一步适配，新训练尚未开始。此前MultiTap暂停状态保留。原三名Luna/max已完成本轮论文、源码与统计，继续复用。

## 2026-10-21 三种提案侧专用图实测已完成

2026-10-21 三种提案侧专用图实测完成：10开发来源、64/256前缀、seed44101、512token、180次；MLP/EAGLE v3/EAGLE v4 KV复用/Medusa LowRank的提案图新增倍率为1.034724/1.157529/1.175119/1.020445，相对同优化Target由1.439277/0.830013/0.788927/0.761582变为1.489254/0.960764/0.927083/0.777153。2倍未达，MLP仍最快。四候选各20/20输出和计数一致，Target调用、图形状与缓存事件一致，独立复算9/9通过。首次图创建进入同步外层墙钟；三臂均无新增图的18对、9来源子集中MLP为1.516174，EAGLE v3为0.974637且区间跨1，v4与Medusa为0.938478/0.781818。EAGLE仅T1递推与Feature读出头图化，T>1仍普通执行。无训练、生产默认和权重保留，GPU已释放。详见[完整结果](../docs/project/artifacts/proposal_graphs_20261021/RESULTS.md)。 本机源码patches/proposal_graphs_20261021，远端/root/autodl-tmp/gaussiangpt_proposal_graphs_20261021/source_final_run01，正式数据main_run01。先读docs/project/artifacts/proposal_graphs_20261021/RESULTS.md与EXECUTION_STATUS.md，不重复已完成测量。原三名Luna/max复用；结束HEAD031d916、4181条工作区记录、GPU2MiB/0%，原服务保留。

## 2026-10-21 同优化Target下的旧方法再评价主比较已完成

用户要求在Graph+BMM优化Target上实际重测EAGLE、Medusa及此前主要方法。11配置、10开发来源、64/256前缀、seed43101、512token、220次完成：MLP/同优化Target1.450584× [1.370351,1.523298]，Ranker-only1.371252×，EAGLE v3g3 0.817906×，Medusa0.725175×，其余均低于1。没有候选超过MLP，算法2×未达；不追加同一训练。3来源132次普通/优化配对11法各6/6输出计数一致。主测8项独立复算通过，单执行下输出配对项不适用；共同满预算与无新增图子集结论保持。LFQ纯BOS18次完成但重放一致Target3/3、MLP2/3、LFQ0/3，暂不报告LFQ有效速度。同历史数值复核run02也已完成：213次Target图前向、203次草稿状态匹配，最大Target policy TV0.012678、BF16草稿q TV0.021485，概率支持集变化，BOS路径不满足此前数值范围、不排名；具体算子未定位。本轮所有GPU任务结束，最后2MiB/0%，读docs/project/artifacts/target_methods_20261021/00_READ_ME_FIRST.md及EXECUTION_STATUS.md继续，不重跑主比较。三原Luna/max继续复用，原生产及权重保留。此前4.52×是对原慢Target，当前主倍率使用同优化Target。

## 2026-10-21 Ranker+MLP执行优化已完成独立实测

三个gpt-6-luna/max完成成本、解码与运行实现，主代理整合。显式BMM+动态CUDA Graph在30独立场景、prefix64/256、60配对条件、240次中：新PF/原Target=4.519564× [4.228452,4.720021]，新PF/原PF=2.613919×，新PF/同优化Target=1.442534×。固定原Target参照下2×已达，主要为执行收益，不能称同优化Target下算法2×。两种PF、两种Target各自60/60同种子输出及计数一致。首次图创建计入解码，runtime跨请求复用，短冷请求可能低于2×。另外3独立场景12次自然EOS全部完成，新PF/原Target=3.630147×、新PF/原PF=2.209333×；单chunk VQ解码11286Gaussian及六视图有限且非空。生产源码、原权重、采样设置保留，优化尚未设为生产默认。完整结果与10份隔离源码清单见docs/project/artifacts/pf2x_breakthrough_20261021/00_READ_ME_FIRST.md，源码在patches/pf2x_breakthrough_20261021。远端/root/autodl-tmp/gaussiangpt_pf2x_breakthrough_20261021，主工程/root/workspace/GaussianGPT；登录已恢复，复用/tmp/ggpt_pf2x_20261021.sock，无密码写入项目。最后GPU 2MiB、0%，原Jupyter/TensorBoard保留。旧失败证据保留，不重复已完成测量或无收益训练。

## 2026-10-21 Position-conditioned Two-Step已止于G2

G0/G1/G2完成。2400条真实Ranker条件含213条P1拒绝，教师概率重算、18组零残差重放及保存权重CPU验证通过。B/C各双种子各1200更新共4800次；A/B/C的Val普通来源等权重合度35.246255%/33.969741%/35.254536%，C对A仅+0.008281个百分点、对B+1.284795个百分点，未满足8pp/4pp且两个Val来源下降。G3–G5未执行，不能宣称新在线接受率或加速比。narrow C提案增量0.465364ms，G1+G2保守历时328.781896秒，已释放GPU。原生产与权重保留。详见docs/project/artifacts/position_transition_20261020/DECISION.md、EXECUTION_STATUS.md、WEB_SUBMISSION.md；不重复运行或动用剩余额度。实际远端/root/autodl-tmp/gaussiangpt_position_transition_20261020，SSH复用/tmp/ggpt_transition_active.sock。用户暂缓MultiTap，新计划先回原网页讨论。

## 2026-10-20 用户要求的N2空间模型完整解码测速已完成

180次、60完整配对、同轮原Target/Ranker+12k MLP/S3空间模型实测结束，无新训练。旧PF与S3合并吞吐123.7590/119.3687 token/s，原Target73.4555；来源等权加速比1.681580/1.635601。S3/旧PF=0.972657 [0.952074,0.993289]，58组共同满512为0.961998 [0.945646,0.977927]。Feature接受率34.5251%→35.8153%，没有额外速度收益，不扩大当前S3。480行CUDA数值和两长度重放通过，180实际输出独立复算通过，12份生产源码逐字节保留。唯一任务936.744秒结束，GPU已释放。完整结果见docs/project/artifacts/spatial_online_20261020/00_READ_ME_FIRST.md与WEB_SUBMISSION.md，不重复运行。下方N2仅CPU结论为较早阶段，已由用户新要求补足实测。

## 2026-10-20 N2空间embedding CPU实验已完成

8个3585参数模型、1200次更新及480条已有记录的留源筛选完成，独立逐行复算通过。Tune中空间模型相对旧q提高0.879037个百分点，相对最佳对照提高0.707884个百分点，一个来源略降，未达预设继续条件；Hold普通重合度旧q36.878741%、空间37.672828%、固定简单混合38.198092%。停止当前模型，GPU=0，无在线接受率或新加速结论。真实原始输入、全部权重/q/训练记录、技术失败记录及最终复算见docs/project/artifacts/spatial_embedding_20261020/00_READ_ME_FIRST.md。实际源码在patches/spatial_embedding_20261020，生产及原权重保留。不要重复拟合或自动扩大，后续提交状态以该目录WEB_SUBMISSION.md为准。


## 2026-10-20 用户要求的12k/48k新实测已完成

120次新种子、512-token、10开发来源的Ranker+MLP对照已完成。48k/12k来源等权速度0.994623×，95%区间[0.980059,1.007992]；两者F接受率35.230457%/35.252955%。共同满512的58配对也未改善。没有48k更快的证据，本次继续保留12k速度参照，48k权重保留。结果见docs/project/artifacts/feature_ckpt_compare_20261020/RESULTS.md，原始记录在raw/；不要重复启动。原源码/权重未改，GPU已释放。当前有效SSH复用/tmp/gaussiangpt_research_loop_20261020_reconnect.sock。

## 2026-10-20 Round03止于CPU筛选

480条已有p/q的来源6/2/2分析已完成。308参数空间探针留出重合度仅提高0.108377个百分点，简单邻居混合提高1.319351个百分点，未达到预设3个百分点。本轮不进入网络训练与在线评价，GPU=0。首次汇总保存故障与同设置重放均保留，概率/参数/逐行结果完全一致，主代理复算通过。先读round03/EXECUTION_STATUS.md与00_READ_ME_FIRST.md；结果包59文件23515260字节已通过原Chrome对话上传发送，网页正在制定下一轮完整方案；先读round03/WEB_SUBMISSION.md，勿重复发送、拟合或追加GPU。原三个Luna/max继续复用，当前未取得下一轮方案。

## 2026-10-20 Round02有限微测已完成并提交

PF-TypedVerify当前分组输出实现完成真实GPU微测，65.529621秒、736前向、728恢复、288配对，688数值比较通过。四token配对中位仅降0.4875%，一/二token略慢，未进入完整解码评价。原生产与权重未改。实际结果包33文件295741字节已约10:21通过电脑Chrome原对话上传发送，网页正制定Round03；先读 `docs/project/artifacts/pf2x_loop_20261019/round02/EXECUTION_STATUS.md` 与 `WEB_SUBMISSION.md`。不要重复训练、R1/R2实验或上传，等网页完整任务书保存后复用三名Luna/max开展下一轮。当前无新2×结果。

## 2026-10-19 用户授权PF分离的2×研究循环

当前192000步训练与既定评价完成后，用户允许开放实验，要求每轮由电脑上Chrome的GaussianGPT网页对话先制定详细计划，再由gpt-6-luna/max子代理执行、打包上传结果、继续迭代至有真实2×证据。此前设置过15分钟心跳（`gaussiangpt-2`）；本次实测结束时其自动化配置文件已不存在，未重新创建。用户明确禁止用内置浏览器交流，并单独批准本任务使用现有 `collaboration.spawn_agent` 接口，最多3个并行子代理。最新执行状态、网页地址、上传包和下一动作见 [研究循环记录](../docs/project/PF_2X_RESEARCH_LOOP_2026-10-19.md)。先读取该记录和远端训练状态，不重复启动任务。

## 2026-10-20 PF2X R1有限诊断完成

原训练与评价结束并向原Chrome网页提交最终包后，网页正式选择step12000。有限诊断全部完成，193.622秒、0新增训练，480条真实分布、两模式各48分支。2/4-token成本接近；配对仅2唯一上下文/1来源，没有同轮Target-only或新完整解码速度结论。先读首轮 `EXECUTION_STATUS.md` 和 `cost_probe_review_20261020/FEASIBILITY.md`；完成Luna复算后上传原网页获取下一轮计划，勿重复采集。

## 2026-10-19 用户要求继续原MLP至96000 / 192000

用户已明确要求从精确48000继续至96000、192000。新独立目录 `/root/autodl-tmp/gaussiangpt_feature_extend192_20261019/`，训练与评价全部完成；2026-10-20 03:43确认为 `complete / analysis_complete`，186次曲线和300次五节点对照均完成。精确96000、192000权重及根目录别名均已以CPU确认元数据。追加144,000次更新，原数据与CE+2TV不变，学习率从1e-6缓降至1e-7，96000处不重启。每1200步保存完整验证与权重，完成后自动做186次曲线及300次五节点同轮在线评价。设置、实际阶段和接续方式见 [新续训记录](../docs/project/FEATURE_MLP_EXTEND192_2026-10-19.md)。96000/192000固定验证重合度为27.5596481%/27.6413286%，在线F接受率35.7257%/35.2191%；相对12k来源速度约0.999倍，区间跨1，没有可靠额外提速。完整依据见 `docs/project/artifacts/feature_extend192_20261019/FINAL_RESULTS.md`。先读取新目录 `pipeline.json`，避免重复启动。下文前一轮“不追加”是当时建议，已由用户最新训练要求取代。

## 2026-10-19 原 MLP 长续训与在线对照已完成

原 step12000 已追加 36,000 次更新；186 次中间曲线和 180 次主要在线对照全部完成。十个开发来源上，精确 step12000/24000/48000 的 Feature 接受率为 34.0312%/33.7806%/35.6000%，吞吐为 116.5768/116.0411/116.7613 token/s。step48000/12000 来源等权速度比 1.0021× [0.9850,1.0189]，Feature 差 +1.5688 个百分点 [-1.0329,4.3738]。没有突然跃升或可靠额外提速的证据，保留原 step12000 PF，不追加相同训练。详情、长度分析及限制见 [MLP 长续训记录](../docs/project/FEATURE_MLP_EXTEND_2026-10-19.md)。远端独立目录为 `/root/autodl-tmp/gaussiangpt_feature_extend_20261019/`，状态为 `complete / analysis_complete`，精确 `24000.pt`、`48000.pt` 均保留，不要重复启动。

## 2026-10-19 Feature EAGLE v4 性能诊断完成

已实现可选 Draft KV 复用并完成 1,024-token 三种子对照。每次输出的 Draft 输入减少 99.18%；v4 相对 v3 的合并吞吐下降 1.22%，中位吞吐下降 1.28%；旧 PF 仍最快。未新增训练，缓存复用默认关闭。实现、接受计数、分项时长和限制见 [v4 执行记录](../docs/project/FEATURE_EAGLE_V4_2026-10-19.md)，网页审阅包为 [feature_eagle_v4_review.zip](../docs/project/artifacts/feature_eagle_v4_review.zip)。完整逐次数据位于远端 `/root/autodl-tmp/gaussiangpt_feature_eagle_v4_20261019/final_before/` 与 `final_after/`。

## 2026-10-19 γ=5 定向尝试完成

2026-10-19 γ=5 定向尝试完成：在 v3 上增加 F2 KL 权重与双 Feature 递推比例，训练 1,200 次、选择 step900。144 次开发运行中新 γ=5 F1/F2 为 543/2159、136/492，吞吐 87.788 token/s；原 v3 γ=5 为 87.237，原 v3 γ=3 为 97.434。全部达到相同生成长度的 21 组中，新/原 γ=5 为 0.989×，没有可靠额外提速，保留原 v3 γ=3 和旧 PF。九项测试与真实分块、贪心、计时重放验证通过。详见 [γ=5 定向记录](../docs/project/FEATURE_EAGLE_GAMMA5_2026-10-19.md)。 新候选在远端 `/root/autodl-tmp/gaussiangpt_feature_eagle_gamma5_20261019/training/selected.pt`，仅供研究，不替换 v3。不要按验证重合度的小幅增加自动续训。

这份文件是新 AI 接手项目时的第一入口，用于说明项目范围、已有证据和当前任务。

## 2026-10-19 Feature EAGLE v3 已完成

用户 v3 计划已执行：完成 T1/T2/T3、真实/Ranker 各半条件蒸馏、两处 Feature 递推、108 次在线比较和 13 次三维导出。γ=3 的 F 接受率从 21.59% 到 35.14%，同轮吞吐从 86.382 到 97.545 token/s，即 1.129×；旧 PF 为 115.628 token/s。γ=5 的 F2 为 30/116，尚无连续在线 F2 提升证据。主训练 1,600 次更新选 step1400，两处 Feature 阶段 800 次更新选 step500。研究权重在远端 `/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`，旧 PF 权重保留。九项测试通过。方法、前缀状态、同历史比较和计时限制见 [v3 执行记录](../docs/project/FEATURE_EAGLE_V3_2026-10-19.md)。不要重复训练或把条件诊断计数混作在线接受率。

## 2026-10-19 EAGLE v2 已实现、训练并完成开发比较

用户最新 v2 设计对应的 Joint P/F 与 Ranker + Feature-only 路线已在远端主工程接入；本地源码在 `patches/eagle_pf_v2/`。两者共享 1L-256 因果结构和 Target 验证，各完成 4,000 次更新与 100 次短 rollout。108 次开发运行中 B γ=3 / Target 为 1.213×，A γ=3 为 0.739×；Ranker-only 为 1.342×，旧 PF 为 1.622×。新 EAGLE 尚无额外速度收益。六项核心测试通过，三条路径各完成一个自然 EOS 的单 chunk 三维输出。数据来自 19 个场景的裁剪，缺少自然 EOS 训练。入口、训练预算、采样状态与性能限制见 [EAGLE v2 实现记录](../docs/project/EAGLE_PF_V2_2026-10-19.md)。运行产物在远端 `/root/autodl-tmp/gaussiangpt_eagle_pf_v2_20261019/`；原有 PF 权重保留。先读取该记录的最终结果，再决定后续训练，避免重复运行已完成实验。

## 2026-10-17 Gap 条件选择路线已完成阶段决策

G0–G5 已结束。D1 的 10 个新开发来源完成 180 次 512-token 和 90 次 2,048-token 在线评测；预设主指标 Gap/PF-K2=1.018× [0.990,1.048]，未达到 1.02 且区间下界高于 1.00 的进入条件。因此没有启动 G6 的 T 来源确认或 G7 完整三维评测。当前保留固定 PF-K2，Gap 条件选择尚未证明持续快于固定 PF-K2。详情见 [G5 报告](../docs/project/artifacts/gap_selection_20261017/G5_report.md)、[最终阶段结论](../docs/project/artifacts/gap_selection_20261017/G8_final_decision.md)、原始记录 `docs/project/artifacts/gap_selection_20261017/g5_raw/`。T 来源仍保留为未测量数据。

## 2026-10-17 成本路线 B1 已完成

最新结果见 [Feature 条件收益报告](../docs/project/artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md)。沿用 A1–A3 的 896 次相同前缀计时，补采 56 条轨迹、5,078 个候选轮次标签及 448 次单轮重放，输出与原记录一致。Feature 收益具有上下文差异：首个 Position 候选间隔为 1 / 大于 1 时，PF/P-only 为 1.190× / 0.970×，后一组区间跨过 1。来源留一验证中，生成前特征预测 Feature 接受概率的 R² 为 0.264；加入草稿统计后为 0.509，但必须先支付 Draft 成本。额外时间预测弱于均值参照，尚不能宣称动态选择更快。保留 PF-K2，P₂ Head 训练仍暂停；B2、K3/K4 和在线调度尚未启动。本处 B1 是成本路线分析，与此前 B1-M/H/E 模型编号不同。远端 /root/autodl-tmp/gaussiangpt_feature_value_20261017/；不要重复运行已完成补采。

## 2026-10-17 候选成本 A1–A3 已完成

最新工作转向不同候选的边际计算价值，见 [A1–A3 成本报告](../docs/project/artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md)。已完成 168 次主比较、896 次相同前缀单轮、8 次同步补算运行及 64 次少量计时点复测与交错对照，没有新训练。固定窗口 PF/P-only 为 1.171× [1.130,1.211]；相同前缀即时推进效率比为 1.141× [1.085,1.195]。P2 拒绝整段处理占时间 1.337%，免费消除并保持其他行为的乐观估计为 1.0136×，不作为任意新方法的全局上限。当前保留 Feature Draft，暂缓仅提高 P2 接受率的独立 Head 训练。A4、B 阶段训练和 C/D 调度尚未启动。远端结果在 /root/autodl-tmp/gaussiangpt_candidate_cost_20261017/，原工程与权重保持原样；不要重复运行已完成测量。

## 2026-10-17 第二轮执行入口

用户已授权执行 [Medusa/Hydra/EAGLE 第二轮计划](../docs/project/MEDUSA_HYDRA_EAGLE_ROUND2_REVIEW_AND_PLAN_2026-10-17.md)。最新阶段、实测与限制见 [第二轮执行记录](../docs/project/ROUND2_EXECUTION_2026-10-17.md)。新增产物位于远端 /root/autodl-tmp/gaussiangpt_round2_20261017/，先读取 progress.json、A1_position_diagnostics.json、E0_rows.json 与 b1_online/20261018/summary.json，按文件实际完成状态接续。原工程和前轮结果保持原样，避免重复训练或测速。

## 目录资料说明（2026-10-15）

远端 `/root/workspace/GaussianGPT` 的文件级用途、保留与清理建议、存放归类和判断依据见 [目录与文件整理说明](../docs/project/GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html)。该单文件文档支持路径搜索与目录筛选，包含隐藏项和 Git 内部资料。扫描时间为台北时间 2026-10-15 22:50；建议尚未执行，远端文件、Git 暂存内容和实验任务均未被改动。

## 研究与资产导航（2026-10-17）

- [文献资料收集](../docs/project/LITERATURE_COLLECTION_2026-10-16.md)：32 篇原文、版本与附件、部分作者代码，以及已有项目资料副本；按用户要求供后续模型独立判断，本次收集未新增方法评价或实验建议。
- [研究工作与真实证据对应](../docs/project/RESEARCH_EVIDENCE_AUDIT_2026-10-16.md)：2026-10-16 重新读取远端源码、训练记录与阶段 E/F 原始结果，含方法对应表、K2 实际流程、参数来源及逐次结果复算；完整研究汇报已纳入主要结论。
- [项目资产地图](../docs/project/PROJECT_MAP.md)：本地交接区、远端工程、外部数据和多场景结果的位置，以及主方法的代码、权重和正式结果入口。
- [Draft 方法历史](../docs/project/DRAFT_METHODS_HISTORY.md)：按研究问题串联方法、实验定位、当前主路线和停止原因。
- [论文证据索引](../docs/project/PAPER_EVIDENCE.md)：将论文主张对应到阶段 E/F 的统计口径、原始 JSON/CSV、可表述范围和仍缺资料。
- [多场景研究执行记录](../docs/project/PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)：阶段 A–F 的详细设置、数据来源和执行结果。
- [完整三维输出记录](../docs/project/FULL_3D_OUTPUT_2026-10-15.md)：阶段 F 全长输出、端到端时间、全图质量指标及视频入口。

阶段 F 已完成生成、统计及视频制作。远端 `stage_f_full3d/pipeline.json` 当前为 `complete / analysis_complete`；真实补全 270/270、BOS 90/90 均成功且自然结束。详细数据见 [阶段 F 完整输出记录](../docs/project/FULL_3D_OUTPUT_2026-10-15.md)，本地统计与视频在 `docs/project/artifacts/stage_f_full3d/analysis/`。

## 当前状态（2026-10-16）

Profiling 数据已整理为 Obsidian [数据汇报](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/GaussianGPT Profiling 数据汇报.md>)，附原始 CSV/JSON、复算表及六张图。远端 P1 原始五次运行已找到：AR/Core 逐运行占比中位数为 99.5653586%，约 99.57%，不含渲染；AR 窗口的 96.6828% 属于另一统计范围，不相乘。P7 局部延迟表把 32 个横坐标都写成 8192，报告使用 P3 原始 `context_length_end`。本次仅整理已有记录，未运行新的 GPU 实验；完整依据见 [论文证据索引](../docs/project/PAPER_EVIDENCE.md)。

用户已授权执行 [P/F 多场景研究计划](../docs/project/PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。阶段 A/B 已完成：6,813 个场景 JSON 已读取，划分为 10/30/20 个开发/测试/适配场景组；十个开发场景产出 19 条唯一裁剪。完整 FP32 开发评价已完成 810/810 条。轻量 PF 达到 1.6882×，P/F 接受率 91.78%/33.40%；PF-D8 的 F 接受率为 55.49%、速度 1.4434×；GPT-D8 为 0.9737×。D1 按场景与前缀分组后，29/29 组合均由轻量 PF 胜出，暂不实现动态调度。

BF16 短测显示 Target/D8 分块与逐 token 的最大分布总变差为 0.0282/0.0632，高于 0.01 容差，未运行完整 BF16 比较。真实来源适配首对训练已完成：20 个适配场景中 19 个重建成功，形成 36 条唯一裁剪、42,236 个真实 token；1 个原始 OBJ 索引越界，保留来源与失败日志。原始 Target 已完成 288/288 条真实前缀续写，共生成 554,705 个 token，45 条自然 EOS（15.625%），包括真实前缀后共 600,740 个正文 token。

D2 首对 810/810 次评价已完成：旧/A/B 的在线 F 接受率为 33.40%/32.92%/34.00%，来源组等权速度比为 1.6869×/1.6782×/1.6897×。B/旧速度比 1.00168，95% 区间 [0.99219,1.01034]，额外速度收益不可靠；B/A 同样跨过 1。固定前缀重合度 B 为 26.8020%、A 为 25.2824%，10 个来源组均改善，但在线收益尚不充分。按原计划停止追加适配训练，主方法保留旧 step12000 轻量 PF；保留 A/B 全部证据，测试不再选模型。

阶段 E 已完成：30 个正式测试场景全部成功，56 条唯一裁剪、63,328 个正文 token；64/256 前缀各有 56 条、1024 前缀有 32 条，三个 seed、五模式共 2,160/2,160 次 FP32 评价。状态 `stage_e_final/pipeline.json` 为 complete / ready_for_full_3d_stage。轻量 PF 的微平均速度比为 1.69675×，来源组几何平均为 1.71250×，95% 区间 [1.68044,1.74402]，30 个来源组均快于 Target，P/F 接受率 92.64095%/33.25815%。P-only 为 1.43773×，PF-D8 为 1.46646×，GPT-D8 为 0.93815×、24/30 来源组慢于 Target；此处三者均为来源组几何平均。正式结果与开发方向一致，方法继续使用旧 step12000 轻量 PF。

远端主工程 `/root/workspace/GaussianGPT`，结果根目录 `/root/autodl-tmp/pf_multiscene_pilot_20261015/`。阶段 E 逐次结果、汇总及图像在 `stage_e_final/fp32_test_full/`，裁剪为 `stage_e_final/final_test_crops.pt`。不要重复运行已完成训练或正式比较。

阶段 F 的两个开发场景三模式共 6/6 完整输出成功，均自然 EOS，Gaussian 与六视角渲染数值有效。完整生成共同正文上限为 16,000 token，真实前缀为 256，使用 T、P-only、轻量 PF 三模式。坐标为单个 20×20×20 latent chunk 局部范围；尚未实施多 chunk 扩展。试运行显示输出长度会影响总等待：第一场景 T/P/PF 总秒为 23.26/19.99/26.61，正文 token 为 1,892/2,350/3,442；第二场景为 25.71/16.06/18.18 秒，正文为 2,088/1,858/2,266。不能直接将固定窗口提速当作每条完整输出的等待时间改善。

阶段 F 使用 T、P-only、轻量 PF，在 30 个真实场景、三个 seed 下完成 270 份全长输出；30 个 BOS seed 的 90 份输出也全部结束。真实补全中，轻量 PF/Target 场景等权速度比为 1.569×，95% 场景重采样区间 [1.318,1.852]；P-only/Target 为 1.467× [1.266,1.731]。轻量 PF/P-only 为 1.070× [0.912,1.252]，区间跨过 1，额外收益尚不确定。轻量 PF 的 Position 与 Feature 接受/已核验分别为 91.720% 和 23.256%。

全部 360 份输出均自然结束并成功导出 Gaussian 与六视角图。输出包含各样本时间、长度、接受率和全图 PSNR/SSIM；指标限制与完整统计见 [阶段 F 记录](../docs/project/FULL_3D_OUTPUT_2026-10-15.md)。远端已有 30 份参考重建及全部生成 PNG/NPZ/token。默认 VQ 解码含无区间裁剪的连续坐标偏移；参考重建本身有 25/30 个场景的 Gaussian 中心落在名义 `[0,4]` 之外。三个模式在 90 个场景×seed 组的 256-token 前缀完全一致。可见区域图像质量和 LPIPS 尚无结果；本轮只覆盖单个 20×20×20 latent chunk，多 chunk 接缝与连续性未测。

D2 证据仍在 `stage_d2_adaptation/stage_c/dev_pair_seed42001/`，D1 为 `stage_c/action_analysis.json` 与 `scheduler_selection.json`。详细正式比较见 [多场景执行记录](../docs/project/PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。

## 当前任务（2026-10-14）

### 2026-10-14 P/F多场景研究规划

用户同意以P/F分工、低成本Feature提案和实际计算收益为主线，并要求完成规划。已形成 [多场景研究计划](../docs/project/PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)，本轮仅修改文档，新增实验尚未启动。顺序为两个开发场景的数据与渲染试运行、10场景五模式比较、按证据开展P-only/K2动态选择或真实来源适配、至少30独立场景正式评价、完整三维输出。训练场景首批20个，仅在适配阶段启用。按原始来源划分，测试不参与方法选择；FP32与可用BF16分别比较。现有LFQ K3与FeatureStateDraft不同，不能直接拼入调度消融。教师继续使用原始Target，D4保留历史结果。执行前核对远端当前资产与进程，旧GPU空闲记录不代表实时状态；先完成两个场景测量，再更新制作耗时和磁盘预算。静态文档规划不刷新生成状态页。

### 2026-10-14 D4 联合 P/F 首轮已完成

用户授权执行后，8600步训练、120次验证集选模、216次真实前缀对照、10组分项计时及训练前后成本测量全部完成。选中step7400，`selected.pt`与四阶段末尾权重均在 `draft_a/runs/d4_joint_20261014/`。D4 Gamma2/4/6速度比为0.6328×/0.4918×/0.4006×，P接受率19.2623%/20.5253%/20.2518%，F为4.4105%/4.1906%/4.0446%。同轮D8 Gamma2为0.9670×，轻量K2为1.4630×。D4提案耗时约D8的61.74%–65.69%，但每轮接受仅约0.13–0.14 token，未取得加速收益。累计35986829个预测位置、17992103个F位置；训练及阶段测量875.0720秒，验证F最终7.8693%。所有数值、恢复、计数复现与本地结果复核通过，GPU空闲。读取 `pipeline.json`、`real_summary.json`、`results_validation.json`，不要重复启动。当前D4为随机初始化，D8有38-epoch预训练，且只有一个真实场景，不能单独归因于层数。详见 [D4执行记录](../docs/project/D4_JOINT_TRAINING_2026-10-14.md)；此前 [计划](../docs/project/D4_JOINT_TRAINING_PLAN_2026-10-14.md) 的“尚未启动”描述仅表示规划时状态。

### 2026-10-14 D8 联合 P/F Gamma 2、4、6

用户要求将 D8 联合 P/F 的 Gamma 提升至 4 和 6。同轮 Target 与 Gamma2/4/6 共 144 次真实前缀评价已完成，Feature 接受率为 54.3726%/53.6912%/53.6286%，Position 为 86.4263%/86.8075%/88.5089%，速度比为 0.9696×/0.9127×/0.8141×。每轮接受长度增至 1.2098/1.7871/2.1229，但较长提案吞吐下降，三个种子组和四份裁剪的速度顺序均为 Gamma2 > Gamma4 > Gamma6。模型权重未修改，数值与九组短验证通过，Gamma2/Target 的 72 次计数复现上一轮，GPU 空闲。只有一个真实场景，结果限于当前 FP32 token 解码。远端入口 `draft_a/feature_d8_gamma_eval.py`，结果目录 `draft_a/runs/feature_d8_gamma_eval_20261014/`，读取 `summary.json`，无需重复运行。见 [Gamma 对照记录](../docs/project/FEATURE_D8_GAMMA_2026-10-14.md)。

### 2026-10-14 高接受率 D8 的真实输入评价

用户要求此前高 Feature 接受率 D8 直接作为草稿，实际权重为 `draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt`，8 层、512 维、37617680 参数。一个真实场景、四个裁剪、三档前缀及三个种子组的 144 次正式运行已完成。D8 Feature 加旧 ranker 的 P/F 接受率为 86.4366%/53.2552%，速度 1.3561×；D8 联合 P/F Gamma2 为 86.4263%/54.3726%、0.9595×。同轮 step12000 MLP 为 85.0586%/20.2967%、1.4483×；D8 两种接法的吞吐都低于 MLP。4420 个相同真实 Feature 前缀上的 D8 重合度为 52.3789%。数值与缓存验证通过，正式权重保留，GPU 空闲。入口为 `draft_a/feature_d8_real_eval.py`，目录 `draft_a/runs/feature_d8_real_eval_20261014/`；读取 `summary.json`，无需重复运行。历史 54.5466% 来自 Teacher 前缀的 GPT 联合解码，和本轮真实输入分开记录。详见 [D8 真实输入评价记录](../docs/project/FEATURE_D8_REAL_EVAL_2026-10-14.md)。

### 2026-10-14 真实 VFront 数据评价

用户要求用真实数据测量当前两版 P/F Draft，正式 108 次运行及全部固定前缀评价已完成。入口为远端 `draft_a/feature_real_eval.py`，结果目录 `draft_a/runs/feature_real_eval_20261014/`。实际资产只有一个 3D-FRONT 场景；五个裁剪中有一个重复，使用四个不同裁剪。真实前缀 64/256/1024 token、三个种子组、每次最多续写 512 token。step3600 P/F 接受率为 85.6598%/21.9531%，速度 1.4811×；step12000 为 85.0586%/20.2967%，速度 1.4654×，本轮未见总体额外收益。4420 个固定真实 Feature 前缀上的重合度为 13.3945%/13.4599%。采用共用外层计时，包含缓存释放及输入输出验证，不能和上一轮绝对速度直接拼接。原始权重保留，GPU 空闲；只有一个场景，不作跨场景置信区间或确定退化结论。详见 [真实 VFront 评价记录](../docs/project/FEATURE_REAL_EVAL_2026-10-14.md)。

### 2026-10-14 轻量 Feature Draft 增加训练次数

用户授权从直接 Target 版本 step3600 增加训练次数，观察持续收益。新增 8400 步及 108 次在线对照已完成，选定 step12000；远端目录为 `draft_a/runs/feature_target_continue_20261014/`，入口为 `draft_a/feature_target_continue.py`。完整验证重合度从 26.4062% 升至 26.7549%，最后 1200 步仅增加约 0.0058 个百分点。全部 50 条测试轨迹相同前缀的重合度从 26.5251% 升至 26.8958%，增幅区间 [0.2230,0.5276] 个百分点。在线 Feature 接受率为 32.1659%，同轮 step3600 为 32.9079%；速度分别为 1.7363×、1.7404×，新/旧区间 [0.97596,1.01747]，没有稳定额外收益的证据。教师始终是原始 Target，复用上一轮数据，正式权重保留，GPU 空闲。接手读取 `pipeline.json`、`history.json`、`online_summary.json` 与 `fixed_test_overlap.json`，无需重复启动。详见 [继续训练记录](../docs/project/FEATURE_TARGET_CONTINUE_2026-10-14.md)。

### 2026-10-14 轻量 Feature Draft 直接 Target 训练

用户明确要求沿用旧低成本 Feature Draft，把近期 GPT 完整分布训练与续训的有效设置迁入该结构，教师始终使用原始 Target。不要再次使用训练后的 GPT 作为本版本教师。完整版本已完成，远端 `draft_a/runs/feature_target_full_v2_20261014/` 的 `pipeline.json` 为 complete，入口为 `draft_a/feature_target_v2.py`。详见 [直接 Target 训练记录](../docs/project/FEATURE_TARGET_V2_2026-10-14.md)。从旧 `FeatureStateDraft` 权重开始，400 条训练与 50 条验证轨迹的全部可用 F 行，原始温度 1.0 完整分布 CE、低学习率续训、正式采样策略 TV 三阶段各 1200 步。训练 418021 个唯一 F 行、7524378 个累计 F 行；验证 46893 行，重合度从 26.2341% 升至 26.4062%，选择 step3600。108 次在线运行中，新 F 接受率 32.9079%，旧 MLP 29.5130%，差值区间 [0.8673,5.5180] 个百分点。新速度 1.7294×，同轮旧 1.7092×，新/旧速度区间 [0.98589,1.03544]，额外提速尚不稳定。新权重为目录内 `selected.pt`，正式权重保留。此前两教师各三 seed 的抽样迁移结果保留为历史，不能描述为本版本的完成结果。

### 2026-10-13 更新

用户要求继续联合候选的独立测试和在线解码评价，已完成。见 [联合候选独立评价](../docs/project/PF_JOINT_EVALUATION_2026-10-13.md)。F1 测试按轨迹重合度增加 0.1002 个百分点，区间包含零；两个新 F2 前缀来源各 8 条，按轨迹增幅为 0.2089、0.1305 个百分点。24 seed 五组在线对照中，父 K2 为 1.7307×，联合 K2 为 1.6940×，父 K3 为 1.6610×，联合 K3 为 1.6067×。联合 K3 的 F2 条件接受虽提高，但到达比例下降，速度下降约 3.27%。不替换父权重，测试不再选模。评价已退出，GPU 空闲。远端结果位于 `profiling/feature_draft/pf_joint_recursive_20261013/independent_evaluation/`。Draft-A v2 顺序任务已于北京时间 19:39:50 完成，其在线速度比最高 0.874×，属于另一条路线。下文 CPU 阶段未读测试集与作业运行状态为历史记录。

用户进一步授权在保留语料进程的条件下开展 F1/F2 联合递归训练。CPU 两组同预算对照已完成，见 [联合递归训练记录](../docs/project/PF_JOINT_RECURSIVE_2026-10-13.md)。两组仅比较 F2 梯度是否进入 F1 状态，固定各 4 轮、每轮每层 30000 个位置。joint 选中第 2 轮，F2 开发重合度从 24.4702% 升至 24.9143%，F1 按轨迹从 29.6134% 变为 29.5934%；detached 保留父模型。成对候选已实际载入验证，正式权重保留，测试集未读取。结束时语料进度 388/500，原两个作业继续运行。最新结果以该记录及远端 `profiling/feature_draft/pf_joint_recursive_20261013/cpu_comparison/comparison.json` 为准。

用户已授权先用 CPU 分析 LFQ Feature Draft，并准备两个输出头的重合度训练候选。见 [Feature Draft 接受率实验](../docs/project/LFQ_FEATURE_ACCEPTANCE_2026-10-13.md)。本任务保留当前 Draft-A v2 作业，正式 GPU 训练尚未启动，也没有安排自动启动任务。完整验证集覆盖 16 条轨迹、6138 个位置，当前联合重合度为 28.9468%。

用户随后同意先完成 LFQ 对照，再根据结果决定 F1/F2 联合递归训练。CPU 三组对照已经完成，训练预算为最多 8 轮、每轮 30000 个位置，数据比例为旧离线/新在线 40%/60%。原目标与新增监督各运行 3 轮，最佳仍为父模型；轨迹均衡运行 4 轮，最佳第 1 轮仅增加约 0.0163 个百分点，尚无明显收益。产物目录为远端 `profiling/feature_draft/lfq_head_acceptance_20261013/cpu_full_comparison`，读取 `comparison.json` 与 `trajectory_difference.json`。正式权重未替换；后续联合递归实验已完成，结果见上方最新记录。本轮不占用 GPU，不读取保留测试集。

用户已授权扩大 Draft-A Teacher rollout 至 500 条，并训练和评价 Draft-A v2。远端主工程 `/root/workspace/GaussianGPT` 已连接；生成进程 368154、顺序任务 368943 已启动。先读 [Draft-A v2 实验记录](../docs/project/DRAFT_A_ROLLOUT_V2_2026-10-13.md)，再读取远端 `draft_a/runs/rollout500_20261013/pipeline_status.json` 与 `generation_progress.json` 获取实时状态。运行目录中的 5000 步训练、overlap 与在线解码评价按顺序执行；不要重复启动同一任务。

原语料实际为 43 条、171112 token、34 条自然 EOS，旧的 9 条统计已过时。2026-10-13 的最新授权优先于下文历史暂缓记录。本机仍为文档与源码副本工作区，完整运行资产位于远端。

用户最新要求在本对话规划官方 Target 的 P/F 独立评测。已形成 [Target 实践评测计划](../docs/project/TARGET_EVALUATION_PLAN_2026-10-08.md)，顺序为运行对象确认、P0 数值验证、P2 小规模生成统计、P1 真实验证数据、扩大 P2、P3 Draft 对比；涉及三维质量结论时补充 Q1 解码对照。当前仅完成计划，未启动实验。后续依据最新授权实施，不把旧的暂缓记录理解为永久取消该方向。

用户要求暂缓继续实验，先整理此前工作并形成阶段汇报。已整理的成果见 [阶段工作汇报](../docs/project/STAGE_REPORT_2026-10-08.md)。本次文档工作没有运行新增训练、性能测量或渲染，也没有操作远端作业。

用户随后授权通过 Computer Use 查看电脑，已阅读远端 JupyterLab 的两条 P-only 实现，成果见 [D8 与结构化 Position 实现汇报](../docs/project/POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)。它包含候选概率、验证调度、缓存、EOS、函数位置与当前运行入口的计时限制。

已按用户提供的七章研究逻辑形成 [完整研究汇报](../docs/project/GAUSSIANGPT_SD_RESEARCH_REPORT_2026-10-08.md)，包含前置知识、瓶颈、两条路线、理论速度上限、算法实现、实验与后续方向。2026-10-16 已找到 P1 原始五次运行，99.57% 现已确认为所测 Core 中的 AR 占比；完整研究汇报现已纳入阶段 E/F、10 月 17 日候选成本与 Feature 收益分析，以及第二轮结构比较的主要结果。

后续会话先读完整研究汇报，将前两份作为历史与实现补充，并结合用户最新指示。远端文件列表已出现 S14 评价/overlap 目录，本轮未阅读其内容；旧材料中“S14 尚未实施”仅反映此前本地记录范围。不要仅根据旧文档的下一步条目启动实验。

## 三分钟阅读顺序

1. 先读本文件和完整研究汇报，确认当前任务与已完成范围；按需查阅阶段工作汇报与两条 P-only 实现汇报。
2. 读 [`PROJECT_STATUS.md`](../docs/project/PROJECT_STATUS.md)，查看自动生成的本地 Git、环境、权重和测速快照；它不覆盖远端或早期参考副本。
3. 读 [`ARCHITECTURE.md`](../docs/project/ARCHITECTURE.md)，理解 VQ-VAE、tokenization、GPT 和渲染链路。
4. 读 [`EXPERIMENTS.md`](../docs/project/EXPERIMENTS.md)，了解 S2 小模型、S3 蒸馏和 position-selective speculative decoding 的证据。
5. 需要快速了解所有 draft 方法的历史、接入方式和端到端结果时，读 [`DRAFT_METHODS_HISTORY.md`](../docs/project/DRAFT_METHODS_HISTORY.md)。
6. 需要面向训练初学者理解联合 P/F 草稿时，读 [`JOINT_PF_DRAFT_TRAINING_GUIDE.md`](../docs/project/JOINT_PF_DRAFT_TRAINING_GUIDE.md)。
7. 要寻找新方案时读 [`RESEARCH_SOLUTIONS.md`](../docs/project/RESEARCH_SOLUTIONS.md)，再读 [`LOGIC_VALIDATION.md`](../docs/project/LOGIC_VALIDATION.md)，查看方法适用条件、淘汰理由和停止标准。
8. 需要运行或维护时读 [`OPERATIONS.md`](../docs/project/OPERATIONS.md)。
9. 用户恢复实验后，先读 [`EXPERIMENTS.md` 的 S13.2](../docs/project/EXPERIMENTS.md#s132-s13b-权重的真实联合-pf-评测2026-10-07) 和 [`JOINT_DRAFT_TRAINING_PLAN.md` 第 17 节](../docs/project/JOINT_DRAFT_TRAINING_PLAN.md#17-s13-结果后的-s14-更新方案参考研究报告)。先确认远端对象与独立场景输入，再评价现有 P-only 路线，按实际缺口决定是否需要神经草稿训练。

## 项目一句话

GaussianGPT 把 3D Gaussian Splatting 场景离散成带几何结构的 token 序列，再用自回归 GPT 生成和补全场景，最后通过 VQ-VAE 解码并渲染回 Gaussian 场景。

## 当前工作区的边界

- 原版主干仍以 GitHub 仓库 [`Yuelinfeng/GaussianGPT_SD`](https://github.com/Yuelinfeng/GaussianGPT_SD) 为参照；当前本地文档目录没有源码 Git 历史，不能把本地 Git 状态当作 Autodl checkout 的状态。
- 本地新增的重点是 S2 draft 模型配置、S3 feature distillation / on-policy distillation / alignment diagnostic，以及 position-selective speculative decoding 原型和测速脚本。
- 文档、状态脚本和 `.codex/` 配置属于项目交接基础设施，不改变模型训练逻辑。

当前这个 macOS 目录是交接文档工作区。`.codex/remote-source/` 保存一份被 Git 忽略的早期模型、配置及脚本参考；它没有覆盖后续 proposer、cascade 和 S13 训练模块。`patches/` 保存局部推理补丁和联合块原型。完整运行版本、checkpoint 和原始性能产物仍以 Autodl 主工程为准。

状态脚本只扫描根目录中的运行资产，因此根目录缺少源码和权重符合当前布局。参考副本不能替代远端版本；本地 Git 尚无提交，也不能说明 Autodl checkout 的提交历史。

## 固定入口命令

```bash
# 查看当前状态（会覆盖生成文件 docs/project/PROJECT_STATUS.md）
python .codex/scripts/update_status.py

# 数学/采样逻辑快速自检
python scripts/selftest_position_selective_math.py

# 验证核心 Python 文件语法
python -m py_compile model/gpt/position_selective_speculative.py \
  scripts/profile_position_selective_sd.py scripts/run_e2e_fair.py \
  scripts/run_e2e_wallclock.py scripts/selftest_position_selective_math.py
```

## 交接时必须说明的事情

新 AI 在提出实现方案前，应明确写出：

- 使用的是哪个 teacher / VQ-VAE / draft checkpoint；
- 数据集、坐标系、token 上限和采样温度；
- 是在复现原版链路、训练 S2/S3，还是修改 position-selective 解码；
- 结果来自哪个 JSON、日志或 checkpoint，并说明其可读性与证据范围；
- 运行后是否刷新了 `PROJECT_STATUS.md`，以及是否有新的风险或未验证假设。

## 状态维护规则

代码、配置、权重、性能结果或环境变更后运行状态脚本。静态文档记录长期有效的架构和实验解释，生成状态文件记录本地易变事实。仅整理静态文档时保留生成页原时间。损坏或未完成的 checkpoint 不作为实验输入，先标明用途和可用性。
