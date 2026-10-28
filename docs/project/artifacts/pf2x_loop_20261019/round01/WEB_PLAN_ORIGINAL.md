# GaussianGPT PF 分离真实 2×：Round 01 方案与执行任务

版本：Round 01（仅制定方案；以用户上传的 2026-10-19 交接包为依据）

## 0. 阅读确认和执行边界

已解压并阅读 `GaussianGPT_PF2X_Round00_2026-10-19.zip` 的根目录 `00_CURRENT_TASK.md`，随后核对 `latest_reports/`、`mlp48_results/`、`mlp192_running/`、`prior_handoff/` 的方法及证据，以及 `source_current/`。包共 149 个文件，包含旧 PF / Target 的历史源码镜像，而非当前远端工程的完整依赖。`mlp192_running/pipeline.json` 在 **2026-10-19 15:55:28 UTC 的快照**仍为 `running/train`，`training/progress.json` 为 step 62750；step 96000 / 192000 **尚未取得结果**，不能据此推断当前服务器实时状态。

现在只允许无 GPU 的源码确认、数学设计、可读代码准备。只有主代理确认现有训练 **及随后的 186+300 次评价**全部完成、`pipeline.json = complete / analysis_complete` 且无竞争 GPU 任务后，才开始 Round 01 的新 GPU 测量。禁止重复启动正在运行的训练或覆盖原 Target / Ranker / 12k/24k/48k/96k/192k 权重。

## 1. 2× 正式口径

- **主比较**：同 RTX 4090、同原 Target（`checkpoints/gpt_vfront.ckpt`）、同 FP32、同 temperature=0.9 / top-p=0.9、batch=1、原 4D 空间 RoPE 和合法 P/EOS/F mask，按同来源、裁剪、前缀、采样 seed 和最大新 Token 预算配对。`S_s = TPS_{method,s}/TPS_{target,s}`（来源级每个方法各自生成 Token 数 / 外层同步 wall 秒）。主要估计是 `exp(mean_s log S_s)`；来源聚类 bootstrap 10,000 次，报 95% CI 与逐来源结果。合并吞吐作为补充而非替代，不能用迥异长度的不同轮次数据直接比较。
- **计入**：cache 分配、前缀 prefill、Ranker、Feature 草稿、实际提案采样、Target verification、残差抽样、EOS/长度控制、拒绝回退、Target 修正、上下文与状态提交、必要数据搬运与同步。**另列**：模型加载、预热、数据集准备、VQ 解码、导出、渲染。
- **固定窗口**：开发阶段建议 256/512 新 Token，独立验证阶段主报告 512，辅报 2048；完整保留每条实际输出长度、EOS / budget-capped 状态。不要在实际短输出中填充虚构 Token 计算速度；使用实际 Token/时间与配对长度分布说明，预先标明完整预算子集分析仅用于描述。
- **自然 EOS + 3D**：仅在固定窗口候选通过后于未参与选择的来源开展完整单 20³ latent chunk、自然 EOS、三维有效性与端到端 wall；单列 Token 解码、VQ/NPZ/六视角 PNG 耗时。自然 EOS 场景间 Token 长度可能不同，要同时报告实际 Tokens、EOS 率、可见区域质量与失败输出；不能把固定窗口 2× 等同于 3D 端到端 2×。
- **成功定义**：开发阶段 `S>=2.0` 是候选；**正式有效 2×** 要求预留确认来源上来源等权点估计及 95% CI **下界均 >2.0**，并满足正确性、自然 EOS、3D 有效性范围。非参数来源 bootstrap，不把同一来源多个 seed 当作独立来源；记录来源使用历史并防止确认集被选模污染。

## 2. 已有事实、成本缺口与数量级

- 阶段 E（30 正式来源、2160 行、FP32、预算512）：Target 70.6688 token/s；PF-K2 step12000 为119.9072；PF/Target 来源等权 **1.71250× [1.68044,1.74402]**，而合并比为1.69675。P 接受（已验证）92.64%、F 33.26%。固定窗口同一 Target 的 2× 合并吞吐**参考值**141.34 token/s，需当前 PF 吞吐再提升约17.9%，或在推进不变时总时间减少约15.2%；新一轮必须重测 Target，这不是绝对判定线。
- 阶段 F：90 真实补全每模式、自然 EOS、三维输出齐全；PF/Target **完整时间来源等权1.56935× [1.31784,1.85226]**，不是阶段 E 的1.7125。
- 当前 PF-K2 提案为 `[P1,F1,P2]`，实际 Target 一轮消费 `[上轮待消费F,P1,F1,P2]` 并取得后继F2条件分布。非退化普通轮次 `C=2+2*I(P1,F1连续接受)`；因此第一个 F 的拒绝截断后续推进。候选成本试验 F1 拒绝约57.6%轮次，Target 验证约17.5ms/轮、MLP前向约0.5ms/轮；P2 拒绝整段只占约1.34%，无需再专项优化。
- 如果**只提高F接受率且每轮时间、P1接受率不变**：正式阶段 E 的PF平均 `211346/81067=2.607` token/轮，21.74ms/轮；要同时间达到历史2×参考需平均约3.073 token/轮，对应F条件接受率粗略需要**约58%**（从33%提高约25百分点）。这是**简化非EOS成本模型的门槛推导**，不是可实现性预测。若新增候选使每轮更慢，所需推进更多。
- 12k/24k/48k：Feature接受34.03%/33.78%/35.60%，吞吐116.58/116.04/116.76；48k相对12k速度来源等权1.0021× [0.9850,1.0189]。96k/192k未知且正在训练，不重做同损失续训。
- 已排除优先项：EAGLE v3/v4，γ5-F2长链，单独Draft-KV复用，原Gap选择，P2-only Head。D8-F接受率约55%仍明显慢，说明接受率不能取代成本测量。
- 仍未知：第二个**同位置**Feature候选在残差分布下的真实补救概率；分支验证实际Target输入/调用/缓存成本；近邻空间条件在同MLP成本下的在线收益。现有包只含相关统计与部分源码，不含完整p/q 4096维路径；需在原服务器、当前训练完成后少量收集。

## 3. 主方案：PF-Branch2（首轮先做廉价精确双Feature候选）

### 3.1 保留P/F分工和第一候选路径

冻结 Target、Ranker 和已有 Feature MLP；每轮先由 Ranker 提出 P1 与 P2，MLP在真实已提交前缀及候选 P1 上产生`q1(F1)`。原样进行一次 Target 4-token验证 `[pending_F,P1,F1a,P2]`。P1拒绝分支完全复用现有代码。

当 P1 接受、第一次 F1a 被拒绝时，**不要立即从残差采最终F**：将条件 Target `p(F|历史,P1)`与第一次 q1 映射为 `r1 = normalize(max(p-q1,0))`，首先以不重复候选 `q2(f) = q1(f) / (1-q1(F1a)) (f≠F1a)` 抽取 F1b（分母为0时降级回单候选）；另设**同一次MLP前向的宽支持q2**作为小型对照：保留原始Feature logits，按相同温度但top-p=1.0生成第二候选q2并可排除F1a、重新归一化。这样测试能否覆盖top-p=0.9截断后留在Target残差中的概率质量。两种q2必须完整记录其实际分布；不能借用Target的p作为q2，也不允许再跑第二次MLP前向。在 `r1` 下做标准精确接受；失败则采样 `normalize(max(r1-q2,0))` 生成最终 F，结束本轮。接受第二候选时，提交 F1b，回退被拒绝候选对应 Target KV，从正确的 `P1` 前缀以 `tf([F1b,P2])` 验证 P2 并生成 bonus F2，P2拒绝沿用原精确残差恢复。第二候选 q2 和 residual 必须使用**同一条件下、实际top-p/温度后的FP32分布**。这个双候选修复是主方案的**低改动可运行探针**，额外 Target 前向只发生在第二候选接受后；完全无需新增训练与树形attention。

### 3.2 分布正确性依据

对任意合法 Target 分布 `p` 和已知提案 `q`：`y~q`，以 `min(1,p(y)/q(y))` 接受；拒绝的总质量为 `Z=Σ[p-q]+`，后续恰好从 `r=[p-q]+/Z` 采样，最终输出分布为 `min(p,q)+[p-q]+ =p`。将第二个合法 `q2` 的**同一算法**用于残差 `r1`，则第二阶段条件输出仍为 `r1`，复合结果仍严格为原 `p`。`q2`可根据已抽出的F1a选择，但不得使用生成时不可得的Target p作为**提案分布**，或把 `q1` 误用为实际 q2。边界需覆盖零概率、分母为零、EOS仅在P槽、token偏移8000、prefix待消费末尾F、P2概率在选中F后的条件历史、3D RoPE、KV rollback。

第一版不用“分支概率相加”“独立重复接受率”近似证明。**单路径惰性（lazy）第二候选本身即可精确抽样**，只在后续 Target 状态提交上比旧路径增加工作。

### 3.3 如惰性双候选显示收益上限，再考虑同一轮分支并行验证

用实际 p/q 路径和 Profile 测得（i）第二候选救回的F比例；（ii）由其触发的额外 `tf([F1b,P2])` 成本。只有当“即使改为一次共享Target验证的保守收益空间”仍有望达到2×，才允许第二阶段设计并行双分支Target验证。现有 `nanochat_gpt.py:CausalSelfAttention.forward` 的 KV 推理接口只支持普通因果路径，没有可直接使用的树形mask；不能直接对 `[F1a,P2a,F1b,P2b]` 做一般因果 attention，否则两分支互相泄漏历史。若确需树形改造，必须处理 branch-only ancestor attention、每行4D RoPE、per-branch单调Position mask、各branch KV的提交/回滚、slot logits，完成数值对照后再测速度。树形分支是**后续有条件实施**，不在Round01直接大改Target。

**可证伪假设 H1**：相同P1条件下的第二个低成本Feature候选，能够显著提高F的有效救回概率，并在计入额外Target条件前向、采样与状态维护后提高普通Token吞吐；若不能，停止该路线。

## 4. 备选：空间条件轻量Feature MLP（不与主方案同时大规模训练）

如果双候选的额外 Target 成本不合算，则针对当前最大的F拒绝瓶颈，在原 `FeatureStateDraft` 的低成本结构上增加**只读已提交3D邻域**条件：候选P1查询已提交P/F的最多8个空间最近邻，输入真实已提交Feature embedding、相对坐标、valid mask；禁止使用候选位置未来真实F或未接受的分支。聚合用均值/小MLP，控制参数和单步耗时，保留原 `prev_state / prev_feature / current_position`、Target冻结head、softcap和policy。先做等参数量的无邻域对照，训练使用原400/50来源和同条件Target分布 CE+2TV；第一个试验建议最多2,000次更新、相同训练位置数、验证每200步，额外时间以实测记；与12k或五节点最优者同条件在线比较。

**可证伪假设 H2**：空间邻域信息提高F1条件 `1-TV(p,q)`，其收益足以覆盖邻域读取与额外MLP成本。若只有固定历史改善、在线F1/真实速度不提高，停止；不得仅凭验证loss选为新主方法。

## 5. Round01步骤、源码与预算

### Gate 0：当前96k/192k全部完成（正式实验的前置条件）

主代理读取远端 `...feature_extend192.../pipeline.json`、`training/progress.json`、`evaluation/progress.json`、`evaluation/summary.json`、`analysis/`，确认 status `complete/analysis_complete`，186次在线曲线 + 300次五节点对照齐全；核对每个精确权重的step元数据、source-group CI、实际Token长度、EOS、吞吐。再决定旧12k或其中某个新权重作为Branch2主对照，**不得用训练期step62750估算**。

### Gate 1：零GPU源码与概率单元测试（训练0步，GPU 0）

1. 将远端正在使用的 `draft_a/pf_feature_unified.py` 与包内历史快照的 `decode/_decode`、`MLPAdapter.propose` 逐函数核对；检查 Ranker的`choose_position`、`fork_position_state`、Target `CaptureLastBlock`、`truncate`、`residual_sample`和`policy`（已有依赖 `bench_gaussiangpt_pf_block2_cow.py`、`draft_a/pf_ranker_eval.py`、`gpt_position_spec_decode_smoke.py`）。新功能**默认关闭**，不改现用PF路径。
2. 新建可选 `draft_a/pf_multicandidate.py`，暴露 `sample_second_feature(q1, q2_base, first_id, rng)` 和 `accept_residual_second(p, q1, q2, second_id, rng)`；仅在W2实验模式下允许`MLPAdapter.propose`附带返回当前同一次前向的原始Feature logits，以产生宽支持q2（不改变W1默认返回接口），返回实际接受/残差分布；`q1`、`q2`均可显式记录。若实现来源最新接口不同，以远端实际函数为准。
3. 纯CPU离散小词表穷举和统计测试：`q`零项、窄top-p和宽支持q2、候选重复排除、`q(y1)=1`、p/q支持集不同、合法的消失项、二次拒绝；同时在4096维随机分布测归一化、有限值和采样统计，容差与样本量显式记录。不得声称有限MC证明严格分布，主要依据为代数推导。

**Gate1输出**：`source_map.md`、`distribution_proof.md`、`tests/probability.json`、只含本轮源代码的 `changes.patch`。失败则不做正式新GPU测量。

### Gate 2：受控空间分析与惰性W2在线试验（当前GPU训练及评价结束后）

1. 修改 `draft_a/pf_feature_unified.py` 的 `decode/_decode(..., feature_width=1)`，复用 `MLPAdapter.propose` 一次 `q1`；仅在现有 `F1拒绝` 分支调用多候选函数。若第二候选接受，按选中实际 F1b 恢复 `P1` Target KV，调用 `tf([F1b,P2])`；如P2再拒绝用现有`point_residual`及后续补算。准确记录 `f1_second_proposed / verified / accepted`、首次及补救接受、`branch_extra_target_calls`、cache回退、真实推进和分支耗时。`feature_width=1` 输出应与原PF逐Token对齐；不要改变原`MLPAdapter`及原训练权重。
2. 新建 `draft_a/eval_pf2x_round01.py`，从现有 `source_current/feature_extend192_20261019/evaluate_extension.py` 复用开发裁剪、FP32环境、交错测量、外层GPU同步、JSONL记录和来源cluster bootstrap，不得直接触发其尚在运行的pipeline或覆盖原`evaluation/`。训练评价完成前可写但不得执行GPU新测量。
3. 先在既有开发10来源上每来源取2个前缀(64/256)，每前缀最多24个到达P1的实际在线条件位置，合计**至多480个F上下文**，以冻结Target提取相同上下文下的4096维 `pF`、MLP的 `q1F`；离线复算W1、W2-不重复和W2-宽支持两种第二候选策略的残差救回概率（按来源汇总，不把Target的`pF`作为推理时的候选策略输入）。JSONL/NPZ只保存少量必须证据，并有source/path/seed和q-policy元数据。该分析是有代价的诊断，必须与普通吞吐分开计时。
4. GPU普通试验矩阵：**3个既有开发来源 × 2个前缀 × 2个seed × 3路径(Target、PF-W1、PF-W2-lazy) × 最多256新Token = 36次解码**。交错运行、预热单列。成功进入更大测量的预设门槛：W2-lazy来源级点估计快于W1且没有明显来源系统性回退；同时结合实际多候选覆盖和分支验证成本计算可达到2×的成本上限。如果覆盖明显不足或增量forward使收益抵消，停止W2路线，不用更多seed硬凑收益。
5. 若小试验正向，再用**10个既有开发来源 × 2前缀 × 3seed × 3路径 × 512新Token = 180次普通解码**复测。先选定一种W2候选策略，再比较PF-W1/W2、Target的配对来源几何平均和95%区间；不将开发结果称为独立证实。采样和接受日志必须留完整输出Token及EOS。
6. 只在实际Target前向成本与W2可救回长度显示并行分支仍有胜算时，写树形设计说明和最低依赖/风险，不在Round01无条件实现。若可行，需回本网页确认Round02的正式修改范围。

**Gate2资源预算**：新增神经训练 **0更新**；GPU在已有任务完成后进行，先以**≤1 GPU小时**完成480上下文、36次普通试验和代表性低干扰Profile；若正向再批准追加**≤2 GPU小时**用于180次开发运行和数值核对。时间是可停上限而非预计实际消耗，超过需要重新评估；不得为等训练排队另起服务器。Host/CUDA event/Profile只用于定位，不与非插装普通wall秒混算。

### Gate 3：决策和自然输出

Round01默认**不使用**未参与选模的来源`G5_source_manifest.json/confirmation_t`，也不进行新的全量3D导出。若10来源开发结果能说明PF-W2具有足够显著净时间收益且有实现2×的合理空间，网页对话审核后再批准使用独立来源以及自然EOS/3D确认。若W2作用仅增加接受率而吞吐不增，转到备选空间条件MLP的**下一轮**定向训练，不自行把两路并行实验铺开。

## 6. 五节点MLP结果后的调整

只比较同轮精确12k/24k/48k/96k/192k的在线结果、来源CI、F条件分布重合、真实吞吐、EOS和生成长度；96000/192000可能无收益。若某个新节点在开发来源上的速度相对12k有可信增益且没有破坏分布或质量范围，Gate2 W1对照替换为该节点（仍保留12k为历史锚点），同时重新测Target。若新节点只提高固定验证重合度而在线速度无可信增益，继续用step12000 W1，不把96k/192k多GPU投入当作优势。若F接受提高但速度不变，优先量化其新增推进与成本对冲，再判断双候选所需覆盖门槛；若F接受下降则无需因训练长而采用它。结果若已经达到2×，仍按独立来源和完整输出协议确认后才能结束研究循环。

## 7. 最多三个Luna/max子代理（由Codex主代理分配）

| 角色 | 输入 | 输出 | 可验收结果 | 严禁事项 |
|---|---|---|---|---|
| L1：概率与源代码对齐 | `00_CURRENT_TASK.md`、现有PF源码、原policy/sampler和Target条件logits | `source_map.md`、`distribution_proof.md`、`pf_multicandidate.py`、CPU数值单元测试 | 连续残差接受代数正确、4096dim支持/零概率测试、功能开关默认W1 | 改Target分布、用Ranker得分当qP、动训练/GPU |
| L2：PF-W2 解码路径 | L1接口、`pf_feature_unified.decode/_decode`、cache依赖 | 解码改动、分支状态转移图、关键数值/输出回归测试 | W1逐token一致；W2的P1拒绝/P1接受F1首拒二接或二拒/P2拒绝/EOS正确，缓存条件无错 | 抢GPU、改原权重、未过Gate就重写树形attention |
| L3：评估与网页材料 | `evaluate_extension.py` 源码、来源使用记录、Stage E/F数据 | 独立`eval_pf2x_round01.py`、统计脚本、结果ZIP打包脚本/格式 | 普通wall与profile分开、同轮Target及W1/W2、来源CI、输出与EOS、180运行矩阵完整性 | 把未完成96000/192000作为结果、动当前pipeline、提前使用确认集 |

三名代理都按**gpt-6-luna/max**运行，最多三并行；主代理统一决定修改范围、安排互不冲突文件、审阅Merge、守护当前任务和GPU、执行最终性能试验。不要让多个代理同时修改`pf_feature_unified.py`。

## 8. 必须交付网页端的结果ZIP

文件名建议 `GaussianGPT_PF2X_Round01_Review.zip`，根目录**必须含** `00_READ_ME_FIRST.md`（状态、实际执行与未执行任务、是否已完成现有192k评价、下一轮问题）。

```
GaussianGPT_PF2X_Round01_Review/
  00_READ_ME_FIRST.md
  PLAN_AND_DEVIATIONS.md
  SOURCE_MAP.md
  METHOD_AND_CORRECTNESS.md
  environment.json
  checkpoints.json
  source_split_manifest.json
  results/
    experiment_plan.json
    paired_records.jsonl
    per_source.csv
    summary.json
    timing_components.json
    acceptance_by_depth.json
    numerical_verification.json
    output_checks.json
    p_q_diagnostic_summary.json
  code/
    changes.patch
    relevant_source/   # 只含本轮小范围源码，必要依赖片段
  logs/
    run_commands.txt
    concise_run.log
  DECISION.md
```

在仍未运行Gate2时，标记结果文件`not_run`并附原因，不制造空结果、虚构时间或声称完成。ZIP不包含私钥、登录资料、原始3D数据集或大权重。每个结果数字给出具体生成脚本、源记录行数、方法配置、路径与生成预算；完整 token 输出可以轻量JSONL压缩存放。`DECISION.md`明确本轮实际速度、接受与推进、总成本、停止/进入备选/继续分支树的依据，以及下一轮恳请网页端审阅的**一个**关键决策。

## 9. 必需最小远端补充材料

本包历史源码是2026-10-16 / 10-19的快照，无法保证远端实时路径与函数完全一样；实施前只需取得：①当前远端`draft_a/pf_feature_unified.py`与`bench_gaussiangpt_pf_block2_cow.py`（尤其`CaptureLastBlock/truncate/decide/residual_sample`）；②`draft_a/pf_ranker_eval.py`、`gpt_position_spec_decode_smoke.py` 的选择和状态接口；③当前 Target `model/gpt/nanochat_gpt.py`、`model/gpt/flash_attention.py`、KV Cache实现（**只有树形阶段才需改**）；④当前长续训最终`pipeline.json`、五节点原始`evaluation/records.jsonl`和`summary.json`、数据/来源分组清单。无需复制大权重或全部工程到网页端；Codex可在已授权的远端主工程内核对。

## 10. 研究结论边界

Round01的目的不是保证2×，而是以最少新增计算判断PF分离的第一个Feature瓶颈能否被**第二个精确合法的Feature候选**经济地突破，并留下真实可复算的结果。若它无净收益，明确停止，不以更多seed、训练或图表掩盖负面证据。后续研究路线必须由网页端收到本轮压缩包后重新确认。

拒绝任何不必要的工程化设计与流程，包括 PR/CI、工程门禁、可信基线建立、反复复核等。始终围绕全局目标，核对真实依据，合并相关任务，持续推进实质进展
