# GaussianGPT PF2X — Round03 证据驱动的 Feature Proposal 改进与真实 2× 验证计划

**方案状态：网页端审阅批准用任务书；Round03 尚未执行。** 2026-10-20。执行主体：Codex 主代理 + 最多 3 名既有 gpt-6-luna/max 子代理，使用现有 RTX 4090、原 Target、现有环境；不额外购买算力。不转用 Round02 未用预算。

## 0. 首先确认 Round02 结束

已按顺序审阅 `DECISION.md`、`RESULTS.md`、`CORRECTNESS_REVIEW.md`、`TIMING_REVIEW.md`，并核对 `micro_run01/paired_target.jsonl`、`micro_run01/probe_rows.jsonl`、`changes.patch`。当前 PF-TypedVerify 的 1/2/4 Token Forward 分组/原始 Host 配对比值中位数为 **1.004082 / 1.005851 / 0.995125**，四 Token 仅约 0.4875% 的中位降时，且四 Token 算术平均计时反而变慢。实验有 8 条真实上下文、4 来源、288 配对，688 条数值比较符合容差，实际 GPU 总墙钟 65.529621 s。**依原停止条件结束 Round02**；不合并分组快速路径，不进入原先 Gate C/D。未有同轮 Target-only/自然 EOS/三维对照，不声称取得 2×。

仍使用冻结正式 W1：`/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`（step12000），冻结 Target `/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt` 与现有 Ranker/VQ。12k→192k 继续训练没有可靠在线收益；不重跑旧 EAGLE、γ5、KV、W2、旧 Gap 和 Round02 输出投影。

## 1. 方法选择及其跨轮净收益机制

### 1.1 主线：PF-GeoResidual（低成本、因果空间邻域条件化 Feature 提案）

保留 **当前 P Ranker + W1 K=2、单分支 Target Verification**。只替换 `MLPAdapter.propose()` 返回的 Feature 提案分布 `q(F_1 | context, P_1)`，不修改 Target/Ranker/P2 策略。新方案使用当前 MLP 的 `base_logits`，增加一个非常小的邻域条件残差 `δ(F | N(P_1), Δxyz)`，先计算候选 Feature logits，再沿用既有 FP32 温度 0.9、top-p 0.9 的 policy：

`q_geo = Policy_0.9,0.9(base_logits + gate(context, N) · delta_geo(N,P1))`。

- `N(P1)` 仅从**已确认输出的完整 `(P,F)` 对**寻找位置 `P1` 的空间邻居，默认六面邻域（±x/±y/±z），至多 6 个；所用位置坐标来自当前 `utils/pos_tokens.py` 的 20³ 映射，不能把 CausalRanker 的 xyz/xzy **排序规则**误当坐标规则。
- 当前前缀末尾已生成的真实 F（虽可能尚未写入 Target KV）属于合法已知历史，可使用；候选 F1、未来 P2/F2、Target 当前 P1 后的隐藏特征均不可使用。拒绝恢复从实际 `prefix` 重建，不能读取未接受的候选分支。
- 初始实现上限：Feature 邻居嵌入 32 维、邻居位移嵌入/编码、池化 32 维、`32→4096` 的低秩 logit correction、零初始化残差门；不增加第二 Target 调用、第二 Draft Transformer 或新 KV。建议不改原 FeatureStateDraft 权重，只训新增小残差模块。
- 真实收益机制是 `q_geo` 与 Target `p_F` 的总变差距离降低，提升 F1 首次接受；**F1 被接受时普通 W1 轮次可输出 4 个 Token，F1 拒绝时通常输出 2 个**，因此在同等输出预算下减少 W1 轮次与 Target 验证次数。P2 拒绝后虽仍输出完整 P/F 对，却可能引发额外 Target 修正调用；这项成本必须单列。
- 正确性由原 Target 的精确拒绝采样保证：使用实际 `p_F,q_geo`，接受率 `min(1,p_F(f)/q_geo(f))`；拒绝从归一化 `[p_F-q_geo]_+` 抽样。任何对提案分布的优化只影响性能，不得改变最终 Target 分布。实现必须支持零残差、稀疏支持集、EOS、不同采样流。

### 1.2 不增加网络成本的先行备选：q-Cal

对原 MLP `raw_logits` 做**提案端**温度/top-p 或极少量静态偏置校准，保留 Target policy 0.9/0.9 不变。此处调整的仅是 q，接受、残差使用真正改变后的 q；不得偷偷更改 Target temperature/top-p。校准如果只带来 1–2 个百分点提升，仅作为消融/可解释负结果，不能据此大规模在线跑 2×。

### 1.3 Round03 之前的额外零 GPU 探索，仅是研发线索

利用 R1 留存的 `results/pq/vectors.npz` 和 480 条 `contexts.jsonl`，独立离线复算得到基础分布重合度均值 **36.4396%**（不是实际抽样接受率 35.4132%）。在相同 480 条开发样本内：

- 只做温度/top-p 网格，最高约 **37.4125%**（Tq=0.75、top-pq=0.90），提升 **0.97 个百分点**；
- 将原 q 与已提交六面邻居的 Feature 经验分布简单线性混合，最佳约 **37.5891%**（混合权重 0.10），提升 **1.15 个百分点**；
- 对本批候选 P1，按 20³ 坐标核算，六面已知邻居数为 1/2/3 个的上下文数量分别是 115/300/65。至少一个邻居的覆盖是 480/480；但 **“存在邻居”不证明其信息足以显著提高接受率**。

以上是**样本内研发观察**，未拆分训练/保留来源、未得到统计区间，也未测在线速度。它们说明“固定校准/简单邻居复制”的收益很小，**Round03 不能把这两种简单配方直接宣称为 2× 主创新**。GeoResidual 的额外假设是：学到不同空间方向与当前条件相关的**可变残差**，有可能优于简单混合；该假设尚无实测支持，必须通过 Gate 决定是否投入训练。

## 2. 到真实 2× 的量化门槛（先估计，再同轮实测）

主测加速定义：`S = T_target_only / T_candidate`，Target/W1/候选使用相同 GPU、Target 权重、FP32（matmul highest、TF32 off）、batch=1、温度/top-p 0.9/0.9、来源、prompt、budget；单次实际 wall 从进入生成（包括本次 cache 分配、前缀 prefill、Ranker、Draft、Target 验证/修正/采样）到 CUDA 同步，排除权重加载/预热并单列。三维解码和渲染另报。

历史 Stage E **来源等权速度比 1.7125×**，95% 来源区间 [1.6804,1.7440]。若总工作与其它成本不变，要达到 2×，需要新 PF 平均耗时至少下降 `1 - 1.7125/2 = 14.375%`。这是历史参考门槛；**不得把 R1/R2 不同批次的 Target 吞吐拼成新加速比**。

R1 的 60 次 W1 真实运行，共 3781 Token、1471 投机轮次、1186 次 F1 验证、420 次接受、1811 次 Target 调用。真实平均推进 `L0=3781/1471≈2.5704 Token/轮`，`Pr(本轮验证 F1)=1186/1471≈0.8063`，实际条件 F1 接受率 `420/1186=35.4132%`。在普通完整轮的简化近似下：

`E[L] ≈ 2 + 2·Pr(F1 进入验证且接受) = 2 + 2·v·α_F`，其中 `v≈0.8063`。它正确反映 **P2 接受与否主要改变 Target 修正成本，而不直接决定该轮生成 4 个还是 2 个 Token**。

仅用于筛选的“跨批次固定成本假设”：`L_req≈L0·(2/1.7125)≈3.002`，于是 `α_F,req≈62.1%`。用 R1 插装 Profile 的单轮 `20.28ms Target + 2.29ms 其它` 作参考，新增 Draft 耗时 0.25/0.50/1.00ms/轮时，所需条件 F1 接受率约为 **64.2%/66.3%/70.4%**。这些数值**不是本轮同条件的严格下界**：新的 q 会改变接受路径、P2 修正调用、早 EOS 和 Ranker 行为，且历史 Stage E 与 R1 不同来源。正式执行时必须用同轮 W1 的完整调用分布和时间作 trace replay / 反事实成本估计，再以实测为准。

**第一轮真正的推进门槛**：不能只看 CE/KL 下降；未计入额外成本时，想仅通过提升首 F 接受率冲击 2×，需要从当前约 35% 朝 60%+ 取得大幅变化。没有达到这个量级时，可以得到论文级机制/小幅实测收益，但**不得声称已经证明能够超过 2×**。

## 3. 文件/函数责任（以远端实际源码优先）

| 文件（均相对于 `/root/workspace/GaussianGPT/`） | 函数/接口 | 具体改动 |
|---|---|---|
| `draft_a/pf_feature_unified.py` | `MLPAdapter.propose`, `decode`, `_decode` | 不改变现有 W1 策略；给 Adapter 增加可选 `feature_variant="baseline"/"geo"` 与一个只读的已提交 P/F 邻居视图；默认与 step12000 逐 Token 结果保持一致。只有 Gate 通过才修改生产可选入口。|
| `train_feature_state_distill.py`, `train_feature_state_distill_continue.py` | `FeatureStateDraft.forward`, `loader_for`, `epoch` | **只复用现有 `FeatureStateDraft` 及旧蒸馏数据格式，冻结其权重**；新增空间残差训练不回写旧脚本/旧权重。旧训练输入为 `prev_state`、`prev_feature`、`position`，并无空间邻居条件。|
| `patches/pf2x_round01/pf_round03_feature.py`（新） | `CommittedNeighbors`, `GeoResidual`, `apply_feature_policy` | 解析实际历史 P/F、构造 20³ 六面已提交邻居、门控低秩修正、q 的策略归一与采样；CPU 小测试可独立加载。|
| `patches/pf2x_round01/pf_round03_offline.py`（新） | `load_r1`, `split_by_scene`, `score_policy`, `estimate_break_even` | 使用已有 NPZ/JSONL 做留源离线校准、特征有用性分析、分布重合率/按源 bootstrap、反事实成本门槛；无 GPU。|
| `patches/pf2x_round01/pf_round03_train.py`（新） | `build_geo_rows`, `train`, `validate` | 复用已有 state-pairs / 原教师 probs、已存在 S3 训练/验证数据，或在可用的已缓存 teacher 条件中补邻域字段；训练仅新的残差模块。权重不覆盖旧 ckpt。|
| `patches/pf2x_round01/pf_round03_eval.py`（新） | `run_method`, `paired_eval`, `source_bootstrap`, `validate_3d` | 三法同轮 `target`, `w1`, `geo`，同来源与长度/EOS、逐次输出/计时、Target 计数和跨来源区间。调用原加载与场景辅助，不重复实现 Target 解码。|
| `draft_a/pf_ranker_eval.py`, `draft_a/feature_acceptance_eval.py` | `typed_kinds`, `validate_sequence`, `policy`, `draw`, `residual` | 尽量复用而不改，确保精确 Target 拒绝采样和 P/F/EOS 对齐。|
| `model/gpt/nanochat_gpt.py`, `model/gpt/engine.py` | `GPT.forward`, `KVCache`, 4D RoPE | **本轮不改**；Round02 typed 分组结果不合入。|

远端开工前须读取最新实际文件并将“源码快照与当前运行源码 SHA256/函数签名差异”写入 `code_inventory.json`；若不存在现成状态对数据、原训练入口或场景评测加载函数，只补最少的相关文件/索引，不推断路径，不建立重复项目结构。

## 4. Gate 0：零 GPU 能力与数据检查（立即执行）

1. 保留 step12000、当前 Target、Ranker 与原 W1 的 hash 和实际运行参数；列出训练语料 S3、`train/val` 状态 pair、已有 teacher probs 和对应 scene 来源，记录缺失/可用情况。确认旧训练没有六面已提交邻域：其 FeatureStateDraft 三输入接口可由已上传源码直接确认。
2. 用 `results/pq/vectors.npz`、`contexts.jsonl` 复现上述 480 条 q/p 与邻域统计；**按 scene ID 切分**，例如预先按稳定排序取 6 个拟合来源、2 个调参来源、2 个留源诊断来源（源划分先写 manifest）。不得把同一来源的 64/256 / 不同 seed 拆到两边。
3. 对 `q-Cal` 的 Tq/top-pq 网格和空间邻居简单混合，只在拟合/调参来源选择，**只在留源诊断来源做一次评价**。报告 `α=Σmin(p,q)`、按源提升分布、EOS 前置条件、q 的 CPU计算量；对高熵与低熵拆分，判断是否存在可学习空间信号。
4. 再建立一个**低参数、跨源受控的离线空间残差上限探针**（比如冻结 12k 输出、最多几百参数的方向门控；不得用目标概率 p 作为推理输入），比较不含邻域的同参数控制头。若只有小样本，报告不确定性，不能以训练集 KL 下降作为有效依据。
5. 计算带 0/0.25/0.5/1.0ms 额外成本的筛选阈值，并核对当前 W1 的真实 Target 调用类别。可用已有 W1 trace 估计调用成本，但是反事实回放应显示是推算，不是 GPU 测量。

**Gate 0 停止**：如果留源真实重合度提升不到 **+3 个百分点**，或增益全部来自简单混合且空间条件头没有额外收益，则**不开展空间网络训练、不启动新的在线 GPU 实验**；将负结果交回网页端，重新选择更有力的机制。此门槛仅用于避免再次为 1% 改进付出大量训练，远低于 2× 所需幅度，不是“达成 2×”条件。

## 5. Gate 1：真实训练数据与训练/成本（仅 Gate 0 通过）

### 数据与监督

- 优先使用已存在 S3 `teacher_trajectory_corpus.pt` 派生的 `train.pt`/`val.pt`（`input_ids` 历史可恢复完整合法 P/F）、配对 `train_state_pairs.pt`/`val_state_pairs.pt`、`frozen_feature_head.pt`。原格式字段需在服务器核对；若原始 `input_ids` 的窗口只有 64 Token、邻居跨窗口不可见，**记录局限并从已有 teacher 轨迹恢复前缀**，不能填造未知邻居。
- 先按原始 scene/source 或 trajectory 身份进行 **源级 train/val 隔离**；如果 corpus 仅保留行而丢失来源关联，不能声称独立来源训练验证，必须在既有轨迹元信息上补可追溯关系，否则不开展可信的泛化宣称。
- 为每条 Teacher-Forcing `(h_prev,F_prev,P_candidate)` 建立已确认历史邻域 `(neighbor_P,neighbor_F,Δxyz)`；Teacher p 是该候选 P 下 Target 计算得到的 4096 类、经过 **原 Target policy(T=.9,top-p=.9)** 的分布。若 P_candidate 来自 Ranker 非真实位置，必须在该位置的**真实条件上下文**重新运行 Target 给 p，不可拿其他位置的 Feature 真值当标签。
- R1 的 480 条开发 p/q 可做误差定位，但不要一边在这些来源训练、一边声称它们是独立在线测试。可在开发范围内选训练，记录身份；最终 confirmation 集完全不参与训练/调参。

### 模型与损失

- 冻结原 step12000 `FeatureStateDraft`、Feature 4096 Head、Target 和 Ranker。
- 只训练 `GeoResidual` 的 32维邻域特征、低秩修正及门控；**输出修正 4096 维 logits，不直接复制某个邻居 token 作为确定性输出**。门控零初始化保证关闭时回退原 W1。
- 监督主指标为实际 policy 后的 Feature 重合度 `α_F = Σmin(p_F,q_F)`；训练可使用数值稳定的 `KL(p_F||softmax(logits_geo/.9))` 加入重合度辅助项，使用 Top-p 实际剪枝后的分布在**每次验证**重算指标。不得仅据 CE/KL 降低选 checkpoint。
- 训练分两段，每段上限 **2,000 optimizer steps**，总新增不超过 **4,000 steps**。第 2 段必须经过第一段的留源重合度与单提案延迟审查；每 250 步保存一次训练/验证重合度、KL、有效 support、MLP/邻域延迟、来源级结果。不要为了减少损失一路训满预算。

### 新 GPU 使用上限

本轮**新增独立授权总 GPU wall ≤ 10,800 s（3 小时）**，不得转入历史剩余额度：Gate 1 数据和短模型训练 ≤3,600s，Gate 2 开发同轮对照 ≤2,400s，Gate 3 独立/完整输出 ≤4,200s，另允许 Gate 0 后统一加载/形状与性能核查 ≤600s。所有额度合计 10,800s，包含载入/预热/训练/验证/生成/同步/导出，不得漏记。超时不补跑，保留未完成状态。若 CPU Gate 0 失败，**GPU=0**。

## 6. Gate 2：最小真实在线验证（仅通过 Gate 1）

### GPU 形状/正确性小试，最多 600 秒

- 同 seed 16 Token 的正常 W1 / 新包装但 `gate=0` 两条轨迹，要求输出 Token、Position/Feature 提案接受计数、Target 输入长度与调用计数**完全相同**；分布变更后不要求与原 W1 同 seed 输出一样，但要求采样后验分布正确。
- 邻居表更新与回退：真实输出才入表，P/F 配对必须完整；P1 拒绝、F1 拒绝、P2 拒绝、EOS、末尾 2-token 窗口均单测；4D `(x,y,z,slot)` 与 Target 真实历史一致，单调 Position 排序 xyz/xzy 两种配置不能污染三维邻接。
- 随机 p/q 穷举 + 4096维数值、zero-support/zero-residual/极端 top-p、重复 neighbor/缺邻域、CUDA FP32/bf16 Draft、`q` 精确传入 residual；确保 Target-only 模型概率未改。输出合法 `P:0..7999, F:8000..12095, EOS:12096`；EOS 只能发生在 P 槽，且在完整 P/F 对后。随机抽样方法只要求概率分布正确，不要求不同方法随机数流逐 Token 相同。
- 额外 Draft 成本在包含邻域查询、张量分配、logit correction、policy 的实际路径测量。若额外成本 > **0.5ms/轮** 且没有接近上述筛选收益的证据，先停止新方案；绝不只按核心矩阵乘法计时。

### 开发配对评价：10 开发来源 × 64/256 前缀 × 2 seeds × 3 方法 × 256 预算 = 120 次

- 统一原 Target-only、原 W1（step12000）与 PF-GeoResidual；**同一评价时段重测三者**，GPU FP32 Target、相同采样 T/p、同一真实 prompt、batch1、配对轮换运行顺序，16 Token 独立预热；同时保存输出 token 序列、EOS、实际 Token 数、Target 调用与输入 Token 总量、Draft 前向、真实 wall、cache/prefill/采样/恢复时间（必要时另作 profile，主吞吐不逐步同步）。
- 对来源等权速度比 `S_W1=T_target/T_w1` 与 `S_geo=T_target/T_geo`、`T_w1/T_geo` 进行 bootstrap（按来源聚类；两个前缀/seed 嵌套在来源下），同时报告合并吞吐作为辅助；EOS 提前结束保留真实长度、另分全预算与 EOS 组。
- 在线 Feature 接受率分母是实际验证数，不是 Teacher-Forcing overlap；报告 `Pr(F验证)`, `F接受/验证`, `4-token rounds`, `Target calls / 1000 emitted tokens`, `Target verify ms`, `P2修正次数`, `Draft增量ms/round`。

**Gate 2 停止**：如 `T_W1/T_Geo` 来源等权区间上界 ≤1，或真实目标加速明显下降，结束本轮；如仅 CE/KL 或离线 α 上升、但在线推进和 Target 调用未改善，同样结束。若点估计仍远低于 2×，则**不能因为少量正向提升直接投入全部独立集**；须由主代理按当前相对 2× 缺口和上界确认是否有继续可验证空间，并把决定写入 `GATE_DECISION.json`。若开发中观察到 ≥2× 只是候选成功，仍需 Gate3独立确认。

## 7. Gate 3：独立确认、长度/EOS 和三维（只在 Gate 2 有明确净收益空间时）

### 来源真实性

- 历史 `G5_source_manifest.json` 记录过 30 个 `confirmation_t` 候选，但尚未测量不等于独立于**所有**先前研究；必须在远端核对当前 manifest、真实路径、是否已用于本方案拟合/选择、与旧 Stage E/F/G5 的交集。历史 `final_test` 30 来源已在 Stage E/F 中使用，**不得称为全新**。
- 如果无法确认至少 20 个未参与本方案拟合和方法选择的可用来源，只能做标注清楚的“补充多场景复核”，不能宣传成完全独立确认。冻结来源清单及 hash 后开始评价，不按结果删场景。

### 固定长度同轮主表

- 预期 30 个已核实 confirmation 候选 × 前缀 64/256 × 2 seed × Target/W1/Geo × **512** 新 Token = 最多 **360 次**。若实际可用独立来源少于 30，报告逐项失败与样本减少，至少 20 个来源方可按预定统计判据声称独立确认；计时预算不够时明确停止而不是中途换来源/种子。
- 固定窗口速度要同时提供全部样本口径、真实输出相同预算到顶子集及 EOS 子集；对于不同生成长度，不能用短输出的较小 wall time 直接宣称速度优势。统计来源等权加速，来源级 bootstrap 95% 区间；主判断 `S_geo≥2.00` 且其来源重采样 95% 下界 >2.00，完整真实计时；若点估计达到但下界不足，报告“候选达标、未确认”。

### 完整自然 EOS 和三维

- 预先固定最多 8 个留源、64/256 前缀、1 seed、三方法，最多 **4096** 新 Token，即最多 **48 次**；报告自然 EOS 的实际终止位置，输出末尾必须是 `12096`；EOS 计数元字段不能代替实际末尾检查。超过预算仍未 EOS 标为 censored，不填补或“自动自然结束”。
- 记录每条实际输出长度、完整耗时、缓存/前缀/草稿/验证/恢复/采样、严格的 P/F 交替与 Position rank 单调、VQ 解码、3D Gaussian 参数有限、可视化/质量指标（若可用）及场景失败。多样性/质量对比应在相同采样设置下按来源统计，不能要求不同随机采样轨迹逐 Token 相等。**固定窗口 ≥2× 不等于完整自然 EOS ≥2×**；分别报告。

## 8. 可执行命令：已存在路径与待实现入口严格区分

> 下列 `pf_round03_*` 是本轮必须实现的 **CLI 契约**，尚未在原工程中存在。开发 Agent 必须先完成 `--help`、实际远端源码路径与 manifest 解析，禁止将示例输出或命令写成已运行结果。

```bash
cd /root/workspace/GaussianGPT
PY=/root/miniconda3/envs/gaussiangpt/bin/python
ROOT=/root/autodl-tmp/gaussiangpt_pf2x_round03
R1=/root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/run01
W1=/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt
# Gate 0: CPU-only, 读取既有 R1 文件；不得在运行时隐式初始化 CUDA
CUDA_VISIBLE_DEVICES='' "$PY" patches/pf2x_round01/pf_round03_offline.py \
  --contexts "$R1/collection/contexts.jsonl" \
  --vectors "$R1/collection/vectors.npz" \
  --out "$ROOT/gate0" --split-unit scene --seed 32003
# 只有通过 Gate 0 才有：
"$PY" patches/pf2x_round01/pf_round03_train.py \
  --project-root . --data-dir profiling/feature_draft/full_data \
  --states-dir profiling/feature_draft/full_states \
  --base-checkpoint "$W1" \
  --out "$ROOT/gate1" --max-steps 2000 --seed 32003
# 该 states-dir 是既有脚本默认约定的示例，必须在 Gate0 检查真实路径；不存在不得自动创建伪数据。
"$PY" patches/pf2x_round01/pf_round03_eval.py \
  --methods target w1 geo --source-manifest "$ROOT/development_sources.json" \
  --prefixes 64 256 --seeds 35731 35732 --budget 256 \
  --w1-checkpoint "$W1" --geo-checkpoint "$ROOT/gate1/selected.pt" \
  --output "$ROOT/gate2"
# Gate 3: confirmation_t 的真实可用路径与泄漏审计通过后设置（缺失不得运行）
"$PY" patches/pf2x_round01/pf_round03_eval.py \
  --methods target w1 geo --source-manifest "$ROOT/confirmed_heldout_sources.json" \
  --prefixes 64 256 --seeds 47131 47132 --budget 512 \
  --w1-checkpoint "$W1" --geo-checkpoint "$ROOT/gate1/selected.pt" \
  --output "$ROOT/gate3_fixed"
"$PY" patches/pf2x_round01/pf_round03_eval.py \
  --methods target w1 geo --source-manifest "$ROOT/confirmed_eos_sources.json" \
  --prefixes 64 256 --seeds 47133 --budget 4096 --natural-eos --validate-3d \
  --w1-checkpoint "$W1" --geo-checkpoint "$ROOT/gate1/selected.pt" \
  --output "$ROOT/gate3_eos3d"
```

训练若旧数据路径不同，保持真正的已存在文件并更新 `commands.jsonl`；不能默默改随机源集或者重新生成全部教师标签。主流程的超时限制从程序入口算起，计时记录 `time.perf_counter` 与 `torch.cuda.synchronize`，单阶段超时保留已完成记录，未完成项目写成 `not_run`。

## 9. 3 名 gpt-6-luna/max 子代理边界

| Luna | 唯一主要写入范围 | 责任及交付 |
|---|---|---|
| Luna-1：分布/数学 | `pf_round03_offline.py`、`PROBABILITY.md`、`THRESHOLD.md` | R1 p/q 复算、留源统计、最小预测增益/成本阈值、采样和残差/EOS 证明；**不修改生产解码**。|
| Luna-2：Feature 模型和接入 | `pf_round03_feature.py`、`pf_round03_train.py`、Gate2通过时 `draft_a/pf_feature_unified.py` | 邻域特征、冻结基线、训练、正确性与真实延迟；**唯一拥有生产 PF 修改权限**。|
| Luna-3：评测与审阅证据 | `pf_round03_eval.py`、manifest/测量脚本、`results/` | 同轮三法、长度/EOS、质量、来源区间、逐次结果、审阅包；**不改模型/采样算法**。|

Codex 主代理统一决策 Gate、模型/场景源身份确认、GPU 计时总账和最终 ZIP；不让三个子代理竞争同一 GPU。源码冲突由主代理在最后统一合并，不引入 PR/CI、可信基线建立、重复审核或工程门禁。

## 10. 停止与交付格式

即时停止：Gate0 实际增益不足、训练数据与来源身份无法核实、非法提案概率/采样偏差、默认 W1 路径输出不等价、P/F/EOS/4D RoPE/KV 失配、新增 Draft 成本超过收益空间、开发线上未见实际净速度提升、总 GPU 预算到顶、或独立测试未支持 2×。**负结果同样必须打包回本对话，不擅自切换其它大型模型或自启动下一轮。**

最终交付 `GaussianGPT_PF2X_Round03_Review.zip`：

```text
00_READ_ME_FIRST.md                # 状态、实际完成的门和未执行阶段
DECISION.md                         # 继续/停止，以及与 2x 的差距
METHOD_AND_PROOF.md                 # q 的改动、精确采样、P/F/EOS/KV/RoPE
THRESHOLD.md                        # F 接受率与 Draft 成本的同轮收益要求
SPLITS_AND_LEAKAGE.md              # 源集合、完整 ID 与哈希、是否曾被旧 Stage 使用
COST_AND_TIMING.md                 # 真正 wall、完整调用计数/长度/分项成本
QUALITY_AND_EOS.md                 # 不同预算和自然 EOS/三维结果或 not_run
commands.jsonl                     # 实际运行命令及退出码；命令示例不充数
budget.json                        # 每个 Gate 与累计 GPU wall
results/offline/{rows.jsonl,summary.json}
results/train/{config.json,history.json,selected.json}     # Gate1 才存在
results/development/{runs.jsonl,source_summary.json}       # Gate2 才存在
results/confirmation/{runs.jsonl,source_summary.json}      # Gate3 才存在
results/natural_eos_3d/{runs.jsonl,scenes.json}            # Gate3 才存在
code/changes.patch
code/source_before/              # 至少变动文件原文
code/source_after/               # 至少变动文件修改后原文
```

留存完整逐次 Token 序列和明确 EOS 末尾 ID、同轮每方法分项耗时、Target 调用/输入数与 Seed。ZIP 不含 checkpoint 二进制或数据集大体积副本，只含路径/哈希/结构化元信息。完整权重仍保存在既有服务器。

## 11. 本轮核心研究判据

论文方法问题：**与继续训练相同输入的 MLP 相比，显式提供已确认的三维邻域信息，能否实质提高 Feature Proposal 与 Target policy 的重合度，并以极低新增 Draft 成本换取跨轮 Target 调用减少？**

- 若真能达到稳定 2×，提交源级统计、实际解码完整 wall、独立自然 EOS/三维证据。
- 若有小幅收益但不足 2×，只将其称为成本受限的结构条件化改进，不包装为达标。
- 若 Gate0/1 无实质增益，尽早停止并回到机制设计，不重复 192k 训练、R1 分支、Round02 投影、EAGLE、γ5 或缓存重写。

**Round02 已结束；Round03 方案待 Codex 执行，本文无新在线性能与 2× 声称。**

拒绝任何不必要的工程化设计与流程，包括 PR/CI、工程门禁、可信基线建立、反复复核等。始终围绕全局目标，核对真实依据，合并相关任务，持续推进实质进展
