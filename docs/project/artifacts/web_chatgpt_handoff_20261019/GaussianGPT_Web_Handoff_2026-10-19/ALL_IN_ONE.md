# GaussianGPT 完整研究交接正文

资料截止：2026-10-19。先阅读任务提示，再按证据索引核对。

# GaussianGPT 网页端研究交接包

资料截止：2026-10-19。用途：让网页端 ChatGPT 理解已完成研究，并协助设计下一阶段完整方法。

用户已确定：将当前最好的 Position Ranker + MLP 作为完成的 Motivation；下一步讨论论文级完整方案。接手时请沿用这个定位，保留对证据强度和新颖性的独立判断。

## 使用方法

上传这个 ZIP，将 [01_WEB_CHATGPT_PROMPT.md](01_WEB_CHATGPT_PROMPT.md) 的内容作为第一条消息。网页环境若不能展开 ZIP，请解压后先上传 [ALL_IN_ONE.md](ALL_IN_ONE.md)，再按问题补充 evidence 与 source 中的文件。不要声称已读取未能打开的附件。

## 阅读顺序

1. [01_WEB_CHATGPT_PROMPT.md](01_WEB_CHATGPT_PROMPT.md)：任务与期望输出。
2. [02_COMPLETED_WORK.md](02_COMPLETED_WORK.md)：已完成工作、关键结果与研究结论。
3. [03_CURRENT_METHOD.md](03_CURRENT_METHOD.md)：当前方法、输入状态、接受采样与源码对应。
4. [04_NEXT_RESEARCH.md](04_NEXT_RESEARCH.md)：下一步研究骨架、假设与实验顺序。
5. [05_EVIDENCE_MAP.md](05_EVIDENCE_MAP.md)：数字与文件的对应关系。
6. [06_RELATED_WORK.md](06_RELATED_WORK.md)：相关工作与贡献边界。
7. [07_LIMITATIONS.md](07_LIMITATIONS.md)：数据、统计、质量和环境限制。

ALL_IN_ONE.md 合并上述正文，便于一次读取；它与分篇文件内容重复，无需重复阅读。

## 包内资料的使用边界

evidence/reports 是已有报告的原文引用副本，保留各自日期和原始措辞。历史段落可能已被后续实验更新。其原工程链接未全部迁入本包，请使用 05_EVIDENCE_MAP.md 与 source_inventory.csv 定位已附文件。新整理正文负责说明当前状态。

evidence/data 包含精选的既有统计与逐次结果，source 包含相关源码快照。它们供阅读、复算和方法审阅使用；整个包不是可直接启动训练的完整工程。完整权重、数据集、全部渲染文件和所有历史日志未收入。未包含服务器账号密码、登录命令和私钥。

本次打包没有连接服务器，没有运行训练、GPU 测速、渲染或安装操作。远端状态仅依据既有记录，未在本轮重新确认。


---

# 给网页端 ChatGPT 的任务

你将接手 GaussianGPT-specific speculative decoding 的研究设计。先读取本包 00–07 正文，或读取 ALL_IN_ONE.md；再按需要读取证据与源码。每个关键判断请给出依据，区分“项目已经实测”“文献已有方法”“准备验证的假设”。

## 用户目标

我们希望将当前最好的 Position Ranker + MLP 作为已经完成的 Motivation，下一步探讨整个完整方案，将方法做到论文级别的复杂。请把研究重点转向完整方法的科学问题、结构、训练、验证、成本与贡献。已有轻量 PF 保留为最强现有参照。

## 已知状态

- 当前最快路线：Position Ranker + step12000 FeatureStateDraft + 静态 PF-K2。
- 阶段 E：30 个来源、五模式、2,160 次 FP32 固定窗口运行。轻量 PF 来源组几何平均提速 1.71250×，95% 区间 [1.68044,1.74402]；微平均为 1.69675×。
- 阶段 F：270 份真实补全、90 份 BOS 输出，均自然 EOS，并完成 Gaussian 导出与六视角图。真实补全 PF/Target 场景等权完整时间比 1.56935× [1.31784,1.85226]。PF/P-only 为 1.06968× [0.91223,1.25177]，额外完整时间收益仍不确定。
- 普通 D8/D4、直接 Position Head、Feature EAGLE、更长候选链、Gap 选择和 Draft KV 复用均已有实验，细节见包内正文。请避免把这些工作重新列作尚未开始的任务。
- 最近讨论提出“空间条件 Draft + P/F 非对称候选树 + 实际时延预算 + 候选历史训练”。这是候选研究方向，尚未实现，也没有新颖性和速度保证。请独立评估，允许替换或简化。

## 请依次产出

1. 用简短文字复述已完成工作与最有价值的三条 Motivation；指出会影响方案判断的证据缺口。
2. 提出一个统一的研究问题，给出一个主方案及一个有明确取舍的备选方案。每个模块说明输入、处理、输出、额外成本和移除后的行为。
3. 写清 P/F 交替、末尾 Feature 待 Target 消费、预测状态递推、分支状态、3D RoPE、EOS 和拒绝恢复的完整流程。严格区分已提交状态与提案状态。
4. 对随机采样给出可采用的候选生成与验证算法。若使用树、无放回抽样、候选裁剪或不同深度，说明实际提案概率如何计算、条件如何变化，以及保持 Target 分布所需的论证。不能把确定性的 Top-k 排名分数直接当作抽样概率。
5. 说明训练数据来源、同历史 Target 标签、分布监督、递推过程、来源级划分、选模与预算匹配。区分新设计与 EAGLE-3 已有的多步训练思想。
6. 给出与 EAGLE、Medusa/Hydra、Sequoia、FlashMesh/XSpecMesh 及近期随机树验证方法的逐点比较。把原创主张、借用组件和工程实现分开。能联网时读原文；不能联网时明确依据限于本包。
7. 将方案拆成能够逐步否定或支持假设的实验，列出低成本首轮、完整消融、独立来源确认和完整三维评价。用实际输出量与总时间共同判定，报告不利结果。成功标准中的数值请说明依据，别把随意阈值写成理论保证。
8. 最后给出具体到文件职责的实现计划，说明先做什么、依据什么决定继续或停止，以及需要从远端补取哪些最少资料。

## 约束与写作

默认简体中文，直接、克制。无需重写论文正文，也不要将提案写成已完成成果。不要为了模块数量增加成本。保留原 Target、VQ 与随机采样语义；如要改变它们，单独提出新问题并说明影响。不可通过几何规则直接改写最终输出后仍宣称保持原 Target 分布。跨轮次速度只用于各自范围，不拼成同轮排名。

网页端只负责分析与设计。本包不提供执行凭据，不授权自动连接服务器或启动训练。遇到缺失资料，先用现有事实推进，再列出确实影响决策的缺口。


---

# 已完成工作与当前结论

## 1. 当前研究定位

用户已将 Position Ranker + MLP 定位为完成的 Motivation。当前方法有正式多场景解码结果与完整单 chunk 三维输出证据；后续研究目标是形成具有统一问题、方法与验证的完整方案。研究方法仍需证明相对最强现有轻量 PF 的额外价值。

GaussianGPT 先将三维 Gaussian 场景编码为稀疏离散网格，按 Position、Feature 交替序列生成，再经 VQ-VAE 解码为 Gaussian 并渲染。Position 是空间位置，Feature 是该位置的 4096 类离散码。它们的提案成本、条件关系和接受行为应分别分析。

## 2. 工作总览

| 已完成工作 | 证据支持的结论 | 限制 |
| --- | --- | --- |
| 环境、数据、Target/VQ 与输出链路接通，性能剖析 | 自回归生成占主要等待时间；已经有真实三维输出 | 环境记录来自历史时点，本轮未重连 |
| 独立 Draft-A v2、预训练 D8 与适配、随机 D4 | 较高分布匹配或更小网络没有自动形成速度收益 | 初始化、数据、预算不同，不能单独归因于层数 |
| 结构化 Position 候选、Ranker 快速评分 | 避免额外 Position Transformer，形成低成本提案 | 当前 Ranker 只输出一个确定性位置候选 |
| Feature 状态 MLP、Target 分布监督、续训 | 形成当前最快 PF；step12000 按验证指标保留 | 续训本身的额外速度区间跨过 1 |
| LFQ 与多处 Feature、较长候选链 | 后续接受率改善会受到到达量和递推成本影响 | 结果限于相应历史设置 |
| 多场景开发、真实来源适配、正式阶段 E | 轻量 PF 优于所测 D8 接法；适配的额外速度不可靠 | 本轮来源分离不代表 Target 预训练从未接触 |
| 阶段 F 完整三维生成 | 360 份单 chunk 输出均自然结束并导出 | 未证明多 chunk 连续性或严格画质等效 |
| 10 月 17 日 Head 对照与成本分析 | P2 单独改进空间小；Feature 收益依上下文变化 | 离线标签、在线接受与真实速度各有统计范围 |
| Gap 条件选择 | 新开发来源上没有可靠超过固定 PF-K2 | 未启用独立 T 来源与新完整三维评价 |
| 10 月 19 日 EAGLE v2/v3、γ=5、v4 | 新模型、条件训练与缓存复用均已实际执行 | 旧 PF 仍最快；新路线尚无正式大规模确认 |

## 3. 当前方法的正式证据：阶段 E

30 个测试来源、56 个唯一裁剪、63,328 个输入正文 token；前缀 64/256/1024，最多续写 512 个新 token，三个 seed base、五模式，共 2,160 次。每模式 432 次。FP32，温度与 top-p 均为 0.9，RTX 4090。

| 模式 | token/s，所有运行合并 | 微平均速度比 | 来源组几何平均速度比 | P 接受率 | F 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target | 70.66882 | 1.00000 | 1.00000 | — | — |
| P-only | 101.40326 | 1.43491 | 1.43773 | 92.44076% | — |
| 轻量 PF | 119.90716 | 1.69675 | 1.71250 | 92.64095% | 33.25815% |
| PF-D8 | 103.00437 | 1.45756 | 1.46646 | 92.50658% | 55.47118% |
| GPT-D8 | 66.44308 | 0.94020 | 0.93815 | 78.96773% | 55.52214% |

轻量 PF 的来源组几何平均 95% 区间为 [1.68044,1.74402]，30/30 来源更快。该区间对应 1.71250，不对应微平均的 1.69675。接受率按 accepted/verified 合并计数。2,160 行的吞吐与接受统计曾从原始结果复算，和原汇总一致；本次打包另核对行数与主要聚合值。

时间包含前缀、缓存、提案、Target 验证、拒绝处理与采样，不含模型加载、数据准备、VQ 解码和渲染。BF16 开发时分块与逐 token 分布差异较大，未用于正式结果。

## 4. 完整三维证据：阶段 F

真实补全：30 场景 × 3 seed × 3 模式 = 270 份。BOS：30 seed × 3 模式 = 90 份。全部自然 EOS，全部有 token、Gaussian NPZ 与六视角 PNG，总计 2,160 张图。当前范围是单个 20×20×20 latent chunk。

| 真实补全 | 完整时间均值 | 平均新正文 token | 场景等权 Target 时间/方法时间 | 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| Target | 46.06736 秒 | 3050.18 | 1.00000 | — |
| P-only | 31.36916 秒 | 2987.31 | 1.46713 | [1.26583,1.73137] |
| 轻量 PF | 29.52030 秒 | 3086.49 | 1.56935 | [1.31784,1.85226] |

PF/P-only 为 1.06968 [0.91223,1.25177]，额外完整时间收益仍不确定。完整时间包括 AR、VQ 解码、Gaussian 导出、渲染、六张 PNG 写入；排除模型加载、预热与参考重建。各方法采样轨迹和输出长度不同，需要同时报告长度与总时间。

全图 PSNR/SSIM 仅是与参考 VQ 重建的描述性比较，包含随机续写区域。现有数据不足以声称严格质量等效；缺少可见区域质量、LPIPS 和多 chunk 接缝评价。

## 5. 早期对照与 10 月 17 日的实质进展

- 独立 Draft-A v2：随机初始化 8 层、512 维、约 37.62M 参数；500 条 Target 轨迹，训练 5,000 次更新并选 step3750。K2 在 BOS/context1024 为 0.85358×/0.87403×。它与预训练 D8 是不同模型来源。
- 预训练 D8：已有 38 epoch/step50502 权重，经三段 Feature 适配至 step3600。PF-D8 采用 Ranker P 和 D8 F；GPT-D8 连续提出 P/F，同权重的不同接法需分开命名。
- 随机 D4：4 层、256 维、约 9.37M 参数，选 step7400；一个真实场景 Gamma2/4/6 为 0.63281×/0.49179×/0.40065×。
- 较长 GPT-D8 草稿：Gamma2/4/6 为 0.96959×/0.91266×/0.81413×，接受长度增加未抵消成本。
- LFQ 双 Feature 联合训练曾改善第二处条件指标，但 K2/K3 吞吐均低于对应父模型；应保留为接受、到达和成本分离的历史证据。
- 真实来源适配 D2：B/旧 PF 为 1.00168 [0.99219,1.01034]；固定前缀拟合改善，在线额外速度不可靠。
- 第二轮 B0/B1-M/H/E：四结构各三训练 seed，只有首 seed 做完整在线比较，共 392 次。旧 PF 为 1.624×，B1-E 为 1.506×；B1-E/旧 PF 为 0.927 [0.905,0.948]。另外两 seed 的离线结果不等于多 seed 在线速度。
- 简单增加末尾 F embedding 的独立 Position Head 未解决低重合度。使用额外 Target 前向得到的状态会改变输入和成本，不能归为免费改善。
- A1–A3 成本：PF/P-only 固定窗口 1.17142 [1.12954,1.21093]；相同前缀即时效率比 1.14086 [1.08501,1.19496]。P2 拒绝整段处理仅占 1.3374%，在其余行为不变且该段免费消失时约为 1.01356×，此数字不约束任意新方法。
- Feature 条件价值：5,078 个完整候选轮次标签。候选间隔为 1/大于 1，PF/P-only 为 1.190 [1.117,1.261]/0.970 [0.934,1.015]。生成前接受概率预测 R²=0.264，加入 Draft 统计后 0.509；额外时间预测 R²=-2.695，尚不适合作为可靠净收益选择器。
- Gap 选择 G0–G5 已完成：10 个新开发来源，180 次 512-token 与 90 次 2048-token。长窗口 Gap/PF=1.018 [0.990,1.048]，没有满足预设继续条件，停止在 D1；30 个 T 来源未用于该路线评价。

## 6. 10 月 19 日：已实际完成的 EAGLE 工作

### EAGLE v2

Joint P/F 与 Ranker + Feature-only 两条路线均接入 1L-256、4 头因果 Draft，完成各 4,000 次主训练更新和 100 次短 rollout。108 次开发运行中，Feature-only γ=3 为 1.213×，Joint γ=3 为 0.739×，Ranker-only 为 1.342×，旧 PF 为 1.622×。Joint 的 P 接受率仅 2.214%，多数后续 Feature 尚未到达。三条路径各完成一次自然 EOS 三维输出。六项核心测试通过。

### Feature EAGLE v3

真实 P 与 Ranker P 条件各半；Target 在完全相同的候选历史上提供 Feature 分布。区分末尾 F 已消费与待消费状态，进行 T1/T2/T3 诊断与两处 Feature 递推训练。主阶段 1,600 次更新选 step1400；递推阶段 800 次更新选 step500。

在本轮相同的三开发来源、108 次短运行中，v2 γ=3 的 F 接受率 21.59% → v3 的 35.14%，吞吐 86.382 → 97.545 token/s，为 1.129×。旧 PF 同轮 115.628 token/s。v3/Ranker-only 仅 1.009×，来源方向不一致；v3/旧 PF=0.844×。完成 13 次三维导出、78 张图，但规模不足以支持新方法的正式质量结论。九项核心测试通过。

### γ=5 定向训练

增加 F2 KL 权重与递推比例，训练 1,200 次选 step900。144 次开发运行中，新 γ=5 为 87.788 token/s，原 γ=5 为 87.237，原 γ=3 为 97.434。主汇总受到提前 EOS 影响；全部达到相同长度的 21 组中，新/原 γ=5=0.989×。新 F2 提案 2,368 个，仅 492 个到达接受判定，占 20.78%；额外计算大多不能兑现。没有可靠额外速度收益，未替换原 v3 或旧 PF。

### Feature EAGLE v4

实现可选持久 Draft KV，无新训练。一个场景、三个 seed、每次 1,024 个新 token，v3/v4 输出与接受计数相同。Draft 输入从诊断运行的 345,971 降至 2,847，减少 99.18%；普通运行合并吞吐从 96.60 降至 95.42 token/s，下降 1.22%；中位数从 96.39 降至 95.15。分项中历史重放节省被提交状态追加成本抵消。旧 PF 同轮合并 117.03 token/s，仍最快。缓存复用默认关闭。

## 7. 可用的 Motivation 结论

1. P/F 分工与低成本提案已得到当前问题设定下的正式速度支持。
2. 接受率、平均接受长度、Draft 输入量的改善，都需要通过实际提交量与总时延兑现。
3. 后续研究应同时处理 Feature 不确定性、候选到达量、空间条件和验证成本。

这些结论支持进入完整方案设计。它们没有证明 Position 本身必然低熵、Feature 必然高熵，也没有证明某个新增模块一定有效。


---

# 当前最强方法：实现与状态

## 1. 运行对象

真实运行主工程：远端 `/root/workspace/GaussianGPT`。本机 `/Users/lief/Documents/ChatGPT/GaussianGPT` 是文档与源码快照交接区，缺少完整运行工程、模型权重与数据。当前包不提供直接执行环境。

Target：`/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt`。
VQ：`/root/workspace/GaussianGPT/checkpoints/vqvae_vfront.ckpt`。
Ranker：远端工程 `profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib`。
最佳 Feature MLP：远端工程 `draft_a/runs/feature_target_continue_20261014/selected.pt`，选择步数 12000。

研究 EAGLE v3：`/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`。γ=5 候选在独立同名实验目录下保存。它们尚未替代最佳 MLP。

## 2. Position Ranker

FrozenLean 从最近两个 gap、当前网格相位、历史 gap 统计等产生少量合法 Position 候选；HGB/Numba 快速评分选一个候选。位置合法性沿用 Target 的扫描顺序与位置限制。

它是确定性单点提案，qP 对选中位置的概率为 1。HGB 的分类评分仅用于排序，不能直接当作 Position 词表上的抽样概率。当前实现也没有完整的 Position 候选概率分布。历史快速评分约 0.08041 ms/次，该值不含整个解码轮次成本。

## 3. FeatureStateDraft

输入为：上一个已提交 Position 的 Target 最后层归一化前隐藏状态 1024 维、上一个实际已提交 Feature 的 128 维 embedding、当前候选 Position 的 128 维 embedding。将隐藏状态转换为 256 维，再与两个 embedding 拼成 512 维输入，经残差 MLP 预测 1024 维状态，沿用原 Target 的最终归一化、4096 类 Feature head 和 softcap=15，得到 qF。

可训练参数 2,864,384。共享 Target Feature head 有 4,194,304 个元素，不计入 Draft 可训练参数。MLP 自身没有独立 KV。一次预测的线性计算约 5,505,024 次乘加，未包含采样和调度。历史单场景 CUDA event 前向约 0.480/0.498 ms，不能视为正式多场景的完整提案成本。

训练使用 400 条 Target 轨迹、418,021 行；验证 50 条轨迹、46,893 行。标签是 Target 在当前 Position 条件下的完整 Feature 分布。step3600 → step12000 的验证重合度 26.4062% → 26.7549%；对应续训的在线额外速度区间跨过 1。

旧 MLP 读取归一化前状态，EAGLE v2/v3 读取最终 RMSNorm 后状态，不能混用。也不能以当前真实 Feature 或未来 Target 隐藏状态作为免费的 Draft 输入。

## 4. 静态 PF-K2 的真实含义

K2 指最多两个 Position 提案。普通一轮实际候选为 P1、F1、P2，共三个 token。Target 待消费输入为 `[末尾实际F, P1, F1, P2]`，一次取得 P1、F1、P2 与额外 F2 的四行条件分布。它与其他实验的 γ、普通 K2 token 数不能直接混称。

一轮开始时，输出已经包含完整 P/F 对；Target KV 通常消费到上一 P，末尾 F 已存在于输出但尚待 Target 消费。MLP 读取上一 P 的真实状态、实际末尾 F、候选 P1。Ranker 用自己的位置历史提出 P1/P2，当前 P2 提案器不读取 F1；这个依赖选择允许表达为合法提案，性能仍由 Target 条件分布验证。

1. 构造候选 P1、F1、P2；所有候选都属于临时提案状态。
2. Target 消费末尾 F 和候选序列，得到对应条件分布。
3. 顺序验证 P1、F1、P2，首次拒绝后舍弃其后候选。
4. P1 拒绝：按剩余 Target 概率抽位置，补算实际位置条件下的 F。
5. F1 拒绝：保留已接受 P1 的 Target 状态，按剩余概率抽 F，舍弃 P2。
6. P2 拒绝：保留 P1/F1，另取实际位置及其 F；全部接受则使用已有额外 F2 分布。
7. 提交真实输出与匹配的缓存长度。预算、候选不足时退到较短提案或 Target。EOS 按合法 Position/EOS 步骤处理，立即结束。

在不涉及 EOS、边界与特殊退化的普通轮次，实际提交量为 `C = 2 + 2 * I(P1 与 F1 均接受)`。因此 F1 接受直接影响推进量；P2 接受主要减少恢复前向，单独提高它的收益有限。成本分析已经对此测量。

## 5. 接受采样

同一历史和合法词表上使用实际最终分布 p、q：候选 y 从 q 产生，按 `min(1,p(y)/q(y))` 接受。拒绝后从归一化的 `[p-q]+` 抽样。Position 单点提案的接受概率退化为 p(P候选)。Feature 的 q 经过实际温度与 top-p 处理，不能用原始 logits 或其他历史上的 q 代替。

固定历史下的一步期望接受率为 `sum min(p,q) = 1 - TV(p,q)`。在线 accepted/verified、固定历史重合度、整条路径到达概率是不同量；不能将不同历史的 P/F 平均接受率直接相乘。

树形扩展需要为每个分支维护自己的历史、Target 可见前缀、Draft 状态与位置编码。不同 Feature 历史即使得到同一个后续 Position，也不能直接合并其 Target KV。3D RoPE 和增量状态必须对应实际历史。

## 6. 源码入口与快照时间

- `source/motivation_20261016/draft_a/pf_feature_unified.py`：最佳 PF 主循环、MLP/GPT adapter。
- `source/motivation_20261016/train_feature_state_distill.py`：FeatureStateDraft 定义。
- `source/motivation_20261016/draft_a/feature_target_v2.py`、`feature_target_continue.py`：训练来源、监督和续训。
- `source/motivation_20261016/eval_lean_candidate_to_draft.py`、`gpt_position_spec_decode_smoke.py`、`bench_fast_hgb.py`：Position 候选与快速 Ranker。
- `source/eagle_20261019/speculative/pf_draft.py`、`pf_distill.py`、`pf_engine.py`：EAGLE 网络、条件蒸馏与可选缓存路径。
- `source/eagle_20261019/test_pf_eagle.py`：已有状态、采样、EOS 与增量一致性测试。

前者来自 10 月 16 日真实源码核对，后者来自本地 10 月 19 日镜像。它们分别对应历史方法与新研究接口，不保证是远端此刻的全部最新源码；缺失依赖应从主工程补取，不能假定本包可单独运行。


---

# 下一阶段：完整方法设计与验证

## 1. 已决定与待论证的部分

已决定：现有 Position Ranker + MLP 作为完成的 Motivation，保留其代码、权重和正式证据。下一步研究完整方法，目标是科学问题、机制、训练、系统实现与实验共同成立。

待论证：最近讨论的“空间条件 Draft + P/F 非对称候选树 + 实际时延预算 + 候选历史训练”是主候选。空间模块、树宽和深度都没有新增实测。本包不把它指定为必须实现的唯一架构，接手模型应依据相关工作和可验证收益独立判断。

## 2. 统一问题与目标

问题：对于空间结构与外观特征交替生成的序列，如何利用空间条件，将有限的推测计算分配给能提高实际生成效率的候选？

统一目标可写为：

`效率 = E[每轮实际提交的新token数] / E[Draft + Target验证 + 采样 + 状态维护的完整时间]`。

这是研究与评价目标，当前没有可微且可靠的时延预测器。预算策略应先使用测得的硬件成本曲线；不要直接沿用已有时间预测失败的模型。目标中的实际提交量包含接受、替代与额外 Target token，需要计入 EOS 与长度边界。

## 3. 建议主方案的四个部分

### A. 空间条件 Feature Draft

保留当前低成本 Position 先验。Feature Draft 输入已有 Target 状态、待消费的真实末尾 F、当前候选 P，以及已生成空间邻域的 Feature 表示。候选坐标作为查询，读取已经提交的三维邻域；训练与推理都不能读取未来真实体素或当前真实 F。

Feature 编号本身没有天然数值距离，邻域表示应使用对应向量、坐标差与有效性信息。先比较小型邻域聚合与同容量序列模型，再决定是否需要更重的注意力结构。位置在已提交空间记忆与分支临时记忆中的归属要明确，拒绝后恢复到正确状态。

假设 H1：已生成空间邻域提供便宜而有效的 Feature 条件，使分布改善能够覆盖读取与计算成本。该假设需要与参数量相近、无空间输入的对照分离。

### B. P/F 非对称候选树

Position 少量候选，Feature 若干备选，每条 Feature 分支再继续其对应 Position。例如：

```text
已提交历史 → P1
              ├─ F1a → P2a → …
              ├─ F1b → P2b → …
              └─ F1c → P2c → …
```

假设 H2：在已有 P 提案较可靠、F 易成为首次拒绝点的设置中，把有限节点用于 F 宽度可能比单纯延长候选链更有效。已有 γ=5 结果为研究动机，不能证明树必然有效。多个分支的收益不是各自接受率相加，也不能直接使用独立重复试验公式推算。

首先可在固定小树形状下使用已有严格随机验证方法，减少同时改变采样和预算策略的复杂性。树形状由采样前可用信息选择，或在完整论证下进行自适应扩展；按抽到的 token 身份裁剪时，需重新分析实际概率。保持原 Target 分布是方法要求。

### C. 实际时延预算

在目标硬件上测量不同节点数、深度、上下文长度的 Draft、Target 验证、采样与状态维护时间。使用“期望实际提交量/总时间”选择宽度与深度，保留静态小树和当前 PF-K2 作为退化路径。

假设 H3：P/F 类型、位置候选统计、Feature 分布与上下文长度联合提供比仅按 gap 更有用的预算依据。先验证简单的分组成本表是否足够；复杂预测网络、强化学习和在线更新均不预设为必要组件。

预算估计只能使用决定时已经可得的信息。Target 当前验证概率可作为离线诊断标签，不能伪装成免费输入。Draft 熵等信号需要先支付产生它们的成本。

### D. 面向候选历史的训练

真实历史与当前模型产生的候选历史混合；Target 在完全相同的 Position/Feature 条件下给出监督。覆盖末尾 F 已消费/待消费、不同候选 P、Feature 分支与较短递推，不借用其他历史的真实 F 标签。

先以分布蒸馏为主，保留必要的真实 CE；隐藏状态回归是否有利需单独评价。EAGLE-3 已有多步训练思想，单独增加 rollout 不足以作为新的主要贡献。可研究训练样本如何对应有效分支与节点到达，避免大量优化不会被使用的深层候选；相应权重和梯度处理需明确，不能凭经验保证最终速度。

## 4. 下一步实际工作顺序

| 阶段 | 输入与操作 | 必须产出的结果 | 继续或停止依据 |
| --- | --- | --- | --- |
| N0 方法论证 | 本包、相关论文、现有源码 | 统一问题、模块依赖、随机采样流程、与现有工作的差异表 | 能解释新增结构的必要性和分布要求；核心主张不依赖未得信息 |
| N1 小树可行性与收益空间 | 现有 Ranker + MLP，同前缀受控重放，小节点数/深度 | 候选覆盖、真实推进、Target 额外成本、完整轮次时间；明确哪部分使用理想信息 | 先判断在实测成本下是否存在净收益空间；若仅理想信息下有效，需要说明实现差距 |
| N2 最小完整方法 | 固定小树、严格验证、已有 Draft | 状态流程、拒绝恢复、EOS、数值一致性、普通运行速度 | 接口正确且具备相对现用 PF 的可测价值，或有明确模型瓶颈可改善 |
| N3 空间条件与训练 | 来源隔离的数据、同历史 Target 标签、预算匹配 | 空间输入/容量/条件数据/递推训练的独立消融 | 确认是哪一项带来覆盖或效率收益，避免把额外训练次数混入机制结论 |
| N4 预算联合选择 | 独立开发来源、硬件成本表 | 静态与动态、对称与非对称、等节点/等时延比较 | 真实总时间优于对应静态方案；不以标签拟合优度代替在线收益 |
| N5 论文级确认 | 新的未参与选择的来源，多个训练 seed 与采样 seed | 固定窗口、长序列、自然 EOS、完整单 chunk 三维与适用性评价 | 来源级区间、长度/质量/成本共同支持主张；不利场景完整报告 |

N1 不需要先训练新大模型。N0 应先写清可执行方法，N1/N2 负责验证其计算基础；后续训练随证据展开。任何阶段发现方法无净收益，先分析原因，再决定改动或停止，不自动续训。

## 5. 必要实验矩阵

- 参照：Target、P-only、当前最佳 PF；适用时加入相同数据和预算的 EAGLE 风格 Draft、普通树算法。
- 结构：单链、普通对称树、P/F 非对称树；同节点数之外同时比较相近完整时间。
- 条件：原 MLP、同容量无空间输入、加入空间邻域；序列邻域与三维邻域区分。
- 训练：真实条件、候选条件、短递推；训练更新次数和监督位置数匹配。
- 预算：静态配置、仅 gap、更多生成前信号；计入策略自身成本。
- 输出：P/F 提案、到达、接受、首次拒绝位置、实际提交量、Target 调用及输入数、Draft/采样/缓存/恢复时延。
- 泛化：来源级隔离、多个训练 seed、多个采样 seed、长前缀、温度与 top-p 变化。第二个 Target 或其他结构化三维模型有助于说明适用性，是否纳入由论文主张范围和资源决定。
- 三维：固定 token 预算与自然 EOS 同时报告；可见区域质量、必要的感知/几何指标与失败样例。多 chunk 接缝属于扩展任务，开始前需重新确定方法边界。

## 6. 实现职责建议与不可省略的边界

空间记忆、候选树构造、树形 Target 验证和预算策略宜分开实现，保持现有 PF 路径可比较。具体文件名可在 N0 后决定；现有 `speculative/pf_draft.py` 负责神经 Draft，`pf_engine.py` 负责状态和验证，`pf_distill.py` 负责训练条件与标签，评价入口负责统一计时和统计。

应明确保存每个节点的父节点、token 类型、实际提案概率、坐标、可见祖先和状态长度。节点之间只能读取祖先与已提交历史。额外 Target 隐藏状态提取、邻域维护、接受后状态追加都计入成本。不要沿用 v4 “输入减少即提速”的推断。

暂不建议直接开始大规模新训练。先完成方案论证、小树收益测量与最小完整实现；已有 γ=5、Gap 和单纯缓存复用无可靠额外速度的结果需要进入设计依据。


---

# 主张、证据与位置

以下链接均指向包内文件。逐次数据用于独立复算；已保存的历史验证记录不等于本次重新运行 GPU 实验。原始报告以引文副本保留。

| 编号 | 内容 | 首选证据 | 补充 |
| --- | --- | --- | --- |
| E01 | 当前最佳方法与早期方法来源 | [真实证据对应](evidence/reports/RESEARCH_EVIDENCE_AUDIT_2026-10-16.md) | [方法历史](evidence/reports/DRAFT_METHODS_HISTORY.md) |
| E02 | 阶段 E 五模式数字 | [五模式 CSV](evidence/data/official_ef/stage_e_modes.csv) | [2160 行记录](evidence/data/official_ef/online.json)、[来源组汇总](evidence/data/official_ef/multiscene_summary.json) |
| E03 | 阶段 F 自然 EOS、时间与质量范围 | [完整摘要](evidence/data/stage_f/summary.json) | [270 行真实补全](evidence/data/stage_f/test_samples.csv)、[90 行 BOS](evidence/data/stage_f/bos_samples.csv) |
| E04 | 已有 E/F 复算与数值验证 | [10月16日复算](evidence/data/official_ef/verification.json) | [正式数值记录](evidence/data/official_ef/numerical_validation.json) |
| E05 | MLP step12000 来源 | [训练选择](evidence/data/mlp_training/selected.json) | [续训历史](evidence/data/mlp_training/history.json)、[新旧在线结果](evidence/data/mlp_training/online_summary.json) |
| E06 | 第二轮 Head 与输入条件对照 | [执行记录](evidence/reports/ROUND2_EXECUTION_2026-10-17.md) | [首seed在线摘要](evidence/data/round2/summary.json)、[逐次记录](evidence/data/round2/records.jsonl) |
| E07 | 三训练 seed 的范围 | [seed20261018](evidence/data/round2/B1_training_summary_20261018.json) | [seed20261019](evidence/data/round2/B1_training_summary_20261019.json)、[seed20261020](evidence/data/round2/B1_training_summary_20261020.json)；后两者没有完整在线速度 |
| E08 | 候选成本 A1–A3 | [报告](evidence/reports/CANDIDATE_COST_REPORT.md) | [分析 JSON](evidence/data/candidate_cost/analysis.json) |
| E09 | Feature 条件收益 | [报告](evidence/reports/FEATURE_VALUE_REPORT.md) | [分析 JSON](evidence/data/feature_value/analysis.json)、[输出验证](evidence/data/feature_value/verification.json) |
| E10 | Gap 新开发来源结果 | [G5 报告](evidence/reports/G5_report.md) | [摘要](evidence/data/gap/G5_summary.json)、[512逐次](evidence/data/gap/online_512.jsonl)、[2048逐次](evidence/data/gap/long_2048.jsonl) |
| E11 | EAGLE v2 实现、训练与比较 | [报告](evidence/reports/EAGLE_PF_V2_2026-10-19.md) | [108次记录](evidence/data/eagle_v2/records.json)、[摘要](evidence/data/eagle_v2/summary.json)、[训练来源](evidence/data/eagle_v2/data_manifest.json) |
| E12 | v3 条件蒸馏与递推 | [报告](evidence/reports/FEATURE_EAGLE_V3_2026-10-19.md) | [在线摘要](evidence/data/eagle_v3/online_summary.json)、[108次记录](evidence/data/eagle_v3/online_records.json)、[权重与选模](evidence/data/eagle_v3/selected_weights.json) |
| E13 | γ=5 定向训练无可靠额外收益 | [报告](evidence/reports/FEATURE_EAGLE_GAMMA5_2026-10-19.md) | [144次记录](evidence/data/gamma5/records.json)、[摘要](evidence/data/gamma5/summary.json)、[同长度分析](evidence/data/gamma5/analysis.json) |
| E14 | v4 缓存复用 | [报告](evidence/reports/FEATURE_EAGLE_V4_2026-10-19.md) | [原审阅说明](evidence/data/eagle_v4/REVIEW.md)、[54行CSV](evidence/data/eagle_v4/metrics.csv)；before/after完整JSON未收入 |
| E15 | 实际源码与时点 | [最佳PF主循环](source/motivation_20261016/draft_a/pf_feature_unified.py) | [EAGLE主循环](source/eagle_20261019/speculative/pf_engine.py)、[完整来源清单](source_inventory.csv) |

候选成本和 Feature 价值只附分析与验证摘要，未包含全部逐轮大日志。原报告中的远端路径可用于后续最少补取；当前包不能独立复算这两项的每一个中间值。

下一步方案没有对应“已完成”证据，其状态明确为研究提议。算法细节、新颖性和最终收益均需另行论证。


---

# 相关工作与贡献边界

本节提供阅读入口和设计边界，未进行穷尽式新颖性论证。项目已有 32 篇文献资料收藏；本包仅引用与当前方案最直接相关的论文，不复制整个 PDF 收藏。网页端若能联网，请阅读所用版本原文并核实更新；无法联网时明确这一限制。

| 论文 | 原始入口 | 已有内容与本项目关系 |
| --- | --- | --- |
| GaussianGPT | https://arxiv.org/abs/2603.26661 | 稀疏离散三维场景、Position/Feature 序列、3D RoPE 与 Gaussian 输出；本项目保留原 Target 与 VQ |
| Fast Inference from Transformers via Speculative Decoding | https://proceedings.mlr.press/v202/leviathan23a.html | 概率接受与剩余概率抽样，保持 Target 分布的基础 |
| EAGLE | https://proceedings.mlr.press/v235/li24bt.html | 使用 Target 特征的 Draft；不能把状态复用本身当作新的贡献 |
| EAGLE-2 | https://aclanthology.org/2024.emnlp-main.422/ | 上下文相关候选树；本项目需要另论证 P/F 结构与随机采样 |
| EAGLE-3 | https://arxiv.org/html/2503.01840 | 多层特征与训练中的多步模拟；本项目已有 v3 条件训练尝试 |
| Medusa / Hydra | https://proceedings.mlr.press/v235/cai24b.html / https://arxiv.org/abs/2402.05109 | 多预测头与条件化草稿，相关小型结构已在第二轮对照 |
| Sequoia | https://arxiv.org/html/2402.12374 | 候选树结构优化、随机验证与硬件成本；仅添加 P/F 类型标签的贡献强度需要谨慎评价 |
| XSpecMesh | https://proceedings.mlr.press/v306/chen26dn.html | 三维 Mesh 的多头提案、验证与蒸馏；质量保持措辞不自动等于本项目要求的随机分布保持，需逐算法比较 |
| FlashMesh | https://arxiv.org/html/2511.15618 | 利用 Mesh 层级与结构先验组织推测生成；已有三维结构化方法，需明确 Gaussian P/F 的额外研究内容 |
| DistillSpec | https://arxiv.org/abs/2310.08461 | Draft 分布蒸馏；分布监督本身属于借用方法 |
| RheoSampling | https://arxiv.org/html/2609.21827 | 近期研究随机候选、树扩展与裁剪概率的相互影响；本项目尚未实现或复现 |

潜在原创内容应集中于：空间条件如何改善低成本 Feature 提案；P/F 条件关系如何决定候选宽度与深度；如何在保持原 Target 分布的前提下，将实际时延用于结构化候选预算。上述主张均待验证。

借用组件应明确署名，例如标准拒绝采样、已有树验证算法、多步候选历史训练。源码融合、批量计算和缓存优化可以是必要实现，但需要测量证明其作用。新增模块数量不能作为贡献强度的替代指标。


---

# 已知限制、容易误读的地方与待补资料

## 1. 统计范围

阶段 E 是固定窗口 token 解码；阶段 F 是自然结束的单 chunk 完整流程。微平均吞吐、来源组几何平均与场景完整时间比不能互换。不同轮次的绝对 token/s 受前缀、来源、seed、预算与实现影响，不能跨轮拼接排名。

同一个随机 seed 在不同提案器中会以不同顺序消耗随机数，输出内容和长度可能不同。相同 seed 不要求不同算法产生相同序列；同算法的计时开关与等价状态实现可以做逐 token 对齐。

固定历史重合度、诊断条件采样、路径到达后接受率、实际推进量必须分别报告。训练 seed 和采样 seed 是不同重复单位。v3 的较高接受率和 v4 的较少输入均没有超过旧 PF；原始记录中的单例与多来源统计应按范围引用。

## 2. 数据与独立性

阶段 E 的 30 个测试来源与本轮开发/适配来源分离。历史 Target/D8 预训练来源清单不完整，无法宣称对所有模型均为从未见过的数据。EAGLE 的少量开发来源已经反复使用，不能变成新方法的最终确认数据。

Gap 路线单独保留的 T 来源尚未评价，只能按对应清单讨论；不要与已用过的阶段 E 三十来源混称。真正启动新方法确认前，应重新核对最新来源使用记录，现有包不保证远端此刻仍未使用。

## 3. 质量与数值

现有完整三维输出均为单个 20³ latent chunk。全图 PSNR/SSIM 包含随机续写部分，不能证明画质等效。没有完整的可见区域、LPIPS、多 chunk 接缝与长程连续性结果。

标准接受采样的分布结论依赖同历史、实际概率、正确缓存与合法词表。FP32 的分块/逐 token 有小数值差异；既有有限样本与测试不能代替算法论证。BF16 未用于正式阶段 E。

## 4. 工程与包内覆盖

真实工程与权重在远端，本地为资料与源码镜像。10 月 16 日记录的远端 Git HEAD 为 031d916，存在大量既有未提交工作，该 HEAD 不能唯一标识全部实测源码。后续 EAGLE 镜像来自 10 月 19 日。本次没有重连核对当前远端状态。

包内源码不是完整可执行工程，缺失的导入、训练数据、权重与 CUDA 扩展均不应擅自补造。源码与报告包含历史标识和原作者措辞，按原文引用保留。报告中的工程相对链接不保证可在包内打开，已收入文件统一见 05 和 source_inventory.csv。

包内没有训练权重、完整 3D 数据集、全部 360 份 Gaussian、完整图像与视频，也没有全部历史逐轮日志。v4 附有原审阅包中的 54 行 CSV 和原说明，完整 before/after JSON 仍在远端。

## 5. 真正实施前最少需补取的资料

- 确认远端实际源码、现有进程、环境与 GPU 使用情况；保留并使用主工程及原权重。
- 核对最佳 MLP 与 Ranker 配置、Target/VQ 路径、场景划分及来源使用情况。
- 若分析 v4 的算子成本，补取 final_before/final_after 的 profiles.json 与普通运行 records.json，避免由同步分项推断普通吞吐差异。
- 若实现树形验证，取得 Target attention、3D RoPE、KV 操作与实际采样处理的完整依赖；本包只附相关快照。
- 确定新确认来源、训练预算和主张范围后再组织正式评价。网页端当前任务是方案设计，不自动触发远端操作。
