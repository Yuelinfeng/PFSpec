# GaussianGPT：空间条件 Feature Draft 与 P/F 非对称候选树——分阶段实验执行计划

日期：2026-10-20  
性质：**拟执行研究方案，不代表已完成实验**。依据 `GaussianGPT_Web_Handoff_2026-10-19.zip` 中 `03_CURRENT_METHOD.md`、`04_NEXT_RESEARCH.md`、`07_LIMITATIONS.md` 及对应成本、Feature、EAGLE/γ=5 报告。必须在远端实际执行前重新核对当前工程与数据划分。

## 0. 唯一目标、假设与停止原则

**目标**：在完全冻结 GaussianGPT Target 采样策略、保持准确 P/F 序列与合法空间状态的前提下，检验空间条件 Feature 预测及浅层 P/F 非对称候选树是否可以在相同/相近验证预算下，比 **现用 Position Ranker + FeatureStateDraft MLP + PF-K2** 提供可靠的新增在线吞吐，并最终兑现为单 chunk 真实三维生成时间收益。

- H-Space：显式查询**已提交**空间邻域，在相同模型规模和监督预算下，比简单扩大序列历史能更高效地改善候选 Feature 的分布重合度及连续接受价值。
- H-Width：在当前 Position 提案可靠、Feature 更易截断候选链的情况下，将有限候选节点从后续深度转移到首个 Feature 宽度，可能在真实验证成本下提高每轮净推进效率。
- H-Interaction：空间条件与多 Feature 分支可互补，但不预设一定协同；必须测量交互作用。

三项均为**可证伪假设**。旧 PF-K2 是强基线，N1/N2 失败就停止相关路线，不要求合并两项。

### 历史事实用于设计，而非新的性能承诺

- 正式阶段 E：轻量 PF 来源等权 vs Target 为 1.71250×；PF-D8 为 1.46646×（口径相同），说明增加 Feature 预测精度与容量不会必然净加速。
- A1–A3：同前缀 PF/P-only 局部推进效率 1.141×；P₂ 拒绝整段处理占 1.337%，免费删除的静态时间估计 1.0136×，仅限制该已测路径的补算优化。
- EAGLE γ=5：F₂ 提案到达率 492/2368=20.78%；比 γ=3 更慢，支持对浅层宽度进行先验探测，**不证明树会更快**。
- Gap-Gated G5：2,048-token Gap/PF=1.018× [0.990,1.048]，未达到开发来源继续条件。因而暂不训练动态调度网络。
- 现有完整三维单 chunk PF/P-only 比较区间跨过 1，任何新算法需重新独立确认。

## 1. 冻结边界与数据分层（N0）

### 1.1 完整性契约

1. **冻结** Target `gpt_vfront.ckpt`、VQ `vqvae_vfront.ckpt`、空间 Ranker 权重及当前最佳 MLP `selected.pt`（选择步 12000）。记录 SHA256、Git diff、CUDA/PyTorch、RTX 4090、FP32、CUDA 状态；不覆盖任何旧输出。
2. 固定实际策略：temperature 0.9、top-p 0.9、Position/Feature 分离合法词表、扫描顺序、EOS 仅合法 Position 时可出现、softcap 15 与 Target 原处理次序。
3. **Pending F 条件**：PF-K2 轮首末尾 F 已提交但未消费进 Target KV；对应旧 Target 状态为上一个 Position 的 hidden。空间记忆可包含已确认的 pending F，但 Draft 不可读本轮尚未接受的真实 F。
4. 真实 PF-K2 一轮：候选 `P1,F1,P2`，Target 输入 `[pending F,P1,F1,P2]`；无 EOS 的常规提交数 `2+2*I(P1,F1 连续接受)`。K2 指两个 Position 候选，不能与标准草稿 γ=2 混用。
5. 3D 位置映射按原工程核对：`x=id//400, y=(id//20)%20, z=id%20`；序列化 ID 邻近 ≠ 一般欧氏几何邻近。
6. 分支必须独立保存其祖先 Token、实际提案机制及其概率、Token 类型、P/F slot、3D RoPE、Target 可见历史、临时/已提交空间记忆及 KV 状态。**不同 Feature 祖先即使 P2 ID 相同，也不可合并 Target KV。**
7. 所有性能含草稿生成、邻域查询、分支构造、Target 验证、采样/残差、EOS、同步、缓存分配/写入/回退/提交、必要额外前向；正式窗口计时额外含 prefill，并单独给出纯 decode 时间。模型装载等项目沿用现有 E/F 协议独立披露。
8. 历史上传包是源码快照，非运行时完整主工程。开始前从远端核实实际模型/权重/依赖、受污染来源与当前运行状态；不直接假设包内源码可执行。

### 1.2 数据划分

- **D0：探索/审计**，使用已反复查看的开发前缀，允许检查 Target 概率和做离线 oracle，但**不能**据此报告泛化收益。
- **D1：新开发来源**，建议至少 10–15 个独立来源，分别用于预注册门槛与一次确定设计；训练源不得重合。所有超参/筛选用 D0，D1 主要做是否继续判断。若 D1 被重复用于挑选结构，必须更新其性质为开发集，并另设新确认来源。
- **T：最后独立确认**，建议 30 个此前未参与该方法训练、设计和选择的来源，封存直至设计冻结。历史 Gap T 清单需重新核实是否仍未被后续工作使用。
- 以场景**来源**为统计独立单位；scene crop、前缀、训练种子、采样种子均不得当作新的独立场景。Target 预训练来源 ID 不全，论文需保留此限制。

**N0 交付**：`baseline_contract.md`、`source_split_manifest.csv`、`baseline_hashes.json`、`sampler_cache_contract.md`、`reproduction_smoke.json`。N0 失败则停止后续实验。

## 2. N1：先做浅层树的覆盖—成本探测（不训练）

### 2.1 需要回答的两个独立问题

**概率问题**：已有 q_F 的前几名 Feature 候选，在 Target 的实际 `p_T(F1|h,P1)` 下覆盖多少质量？增加宽度与增加深度相比，哪一个在相同预算下带来更多**可延续的候选历史**？

**成本问题**：这些额外分支及其后续 Target 验证，是否真的能在单次批量前向中低成本处理？如果验证器不能高效实现树形 Mask，任何理论候选覆盖都不能称为实际收益。

### 2.2 预注册最小矩阵

所有配置固定同一 Ranker、原 Feature MLP、Target 与采样策略；不要一边新增树一边训练 Head：

| ID | 候选结构 | 关键控制 |
|---|---|---|
| T0 | 现用 PF-K2 单链 | 强基线，端点对照 |
| T1 | P1 → 两个 Feature 备选，分别衔接其 P2 | 首 Feature 宽度 b=2 |
| T2 | P1 → 三个 Feature 备选，分别衔接其 P2 | 首 Feature 宽度 b=3 |
| T3 | 等新增节点数的更深单链 | 宽度 vs 深度，严格对齐节点预算 |
| T4 | 等节点数/近似耗时的普通树 | 验证 P/F 非对称而非“树天然更好” |

**两套公平口径都要做**：
(1) 同候选验证节点预算；(2) 相同或接近的 Target 验证 wall-time。树的逻辑节点数、模型实际输入长度和 GPU 延迟不一定一致。

### 2.3 Feature 多候选的严格概率边界

在同一已接受 P1 历史下，若按 q_F 排出不同 Feature ID `f1,...,fb`，可考虑一个**待实现并验证**的精确顺序点提案方案：

- 首次目标分布 `p^(1)=p_T(F1|h,P1)`；各提案 `q^(j)=δ_fj`。
- 第 j 个候选以 `p^(j)(fj)` 被接受（当前剩余分布下的点提案接受概率），否则把剩余分布更新为对其他 Feature 重新归一化的 `p^(j+1)`；全部拒绝后从最终剩余分布抽 F。
- 当所有 `fj` 合法、唯一、在构造树前已确定时，选中某条候选分支的概率恰为 `Σ_j p_T(fj|h,P1)`；整体 Feature 输出仍服从 `p_T`，但**必须有实际正确的后续树验证和独立分支历史**。
- 该多点方案不同于现行从完整 q_F 随机抽取一次再按 `min(1,p/q)` 接受的方案。它将 q_F 用于排序，而不是将原有随机草稿接受率与 Top-k 覆盖率混为一谈。必须进行小词表严格枚举、重复 ID、EOS/掩码以及完整 P/F 序列验证，不能把上式当作代码正确性的替代品。
- 不能直接采用“若任何候选与 Target argmax 一致就通过”的确定性树校验来声称保持目标 top-p 随机分布。

因此 N1 至少报告以下三种**不可互换**的量：
1. `coverage_b = Σ p_T(fj)`，即指定候选集合在 Target Feature 分布下的覆盖质量；
2. 严格多候选算法中到达特定分支的概率及其后续 Prefix 生存概率；
3. 在线实际 `accepted / reached`、每次 Target 验证平均提交 Token 数、总时间。

对于 B 个验证 Token 的完整树，若只有候选 Feature 本身覆盖而没有各自后继 P2/F2 条件结果，则不能沿用现有 2/4 提交公式，也不能据此预测树加速。

### 2.4 操作步骤

N1a（不训练，离线概率）：从 D0 冻结前缀取合法 P1；用旧 MLP 得到 q_F，Target 计算**完全一致**历史下的 p_F；用 b=1,2,3,4 生成候选集合，记录 top-k 的 p mass、重复、EOS/非法、q 熵、空间 gap、P1 接受概率和每条候选对应的后继 Position；计算带 P1 到达因子的覆盖收益。额外加入 oracle `Target-topk` 作为**仅分析的乐观参照**，绝不参与在线决策。

N1b（硬件微基准）：只实测验证器：普通 PF block、节点数相近的序列块、树形 Mask、2/3 Feature 子节点、后续 P2 条件计算。长度分层 64/256/1024 以及不同验证节点数；记录 host、CUDA event、kernel launch、有效 token-input、显存峰值、回退/复制、额外目标前向，并轮换执行次序。准确测量树验证结构比理论 FLOPs 更重要。

N1c（正确执行的最小树）：只有验证器的成本合理，才实现严格随机多候选和后续分支验证；先运行小词表穷举、有限分布 TV、P/F 语法、逐层回退和多历史 cache 测试；再在 D0 执行短窗口在线吞吐。**先用单轮追踪估计覆盖收益，再以普通运行测试真实速度**。

### 2.5 N1 决策门槛

- **立即停止**：目标概率/实际提案分布不一致，随机抽样验证不正确，树形 KV 或 3D RoPE 回退不正确。
- **停止加宽**：b=3 对 b=2 的带到达权重增量很小，却增加显著 Target / cache 成本；不能为了更高 coverage 强行做 b=4 或深树。
- **继续 N3**：至少一种 b=2/3 结构在 D0 普通运行中展示明确的正向净速度趋势，且在单轮端与成本模型相符，并能够以固定树算法做到精确随机采样。探索阶段可先用 **≥约 3%** 的净吞吐提升作为实际意义筛选线（项目内部约定，不是统计证明）；若未达到且乐观成本推算也无空间，停止树方向。
- **若离线 oracle 上界仍无明显收益**，结束树路线；但要明确是指定验证模型及给定预算下的局部乐观参照，不是所有树算法的数学全局上限。

**交付**：`tree_probability_probe.jsonl`、`tree_cost_surface.csv`、`exact_tree_sampler_test.json`、`N1_decision.md`。

## 3. N2：空间条件信息诊断（先不建树）

### 3.1 只允许哪些信息

对候选位置 P1 建立 `M_committed`：所有已经**接受/残差正式提交**的 `(Position,Feature)`；包括确认但尚未消费进 Target KV 的 pending F；不包含草稿候选 Feature 的真实标签、不包含数据集未来 GT 空间信息。

令候选坐标 `r(P1)`，以其为中心查询 `r=1` 范围的前置邻域（可选按 6 个轴邻居或最多 26 个邻居），另记录 2 个半径档位的可用邻居数量、方向、相对位置、相对 Token 顺序、是否碰到边界和候选语义。**先统计已提交邻居的可用率**，不直接训练空间注意力模型。

真实 GaussianGPT Feature 为 4096 类离散码，码本 ID 不具有连续数值距离；使用冻结的 Target/VQ Feature embedding 或已有特征向量表示，不能对 ID 直接求几何平均。

### 3.2 四组严格消融

相同 Position Ranker，强旧 MLP 作为推理上限对比：

| 实验 | Feature 输入 | 额外条件 | 要隔离的因素 |
|---|---|---|---|
| S0 | 最佳旧 MLP | 无改动 | 真正强基线 |
| S1 | 同结构重新训练的 MLP | 仅已知历史状态 | 控制新数据和重新训练带来的改善 |
| S2 | 与 S1 参数/FLOPs 接近 | 额外最近 k 个**序列历史 Feature** | 控制“只是多读历史” |
| S3 | 与 S2 参数/FLOPs 接近 | 额外候选 P1 周围已提交**三维邻居 Feature** | 测定空间检索特异价值 |
| S4 | 与 S3 完全相同 | 打乱邻居坐标或匹配邻居数的非空间置换 | 检验真实相对几何排列是否发挥作用 |

建议 S3 首版采用极简归一化邻域聚合（邻居 embedding + 相对坐标 + occupancy mask，平均/注意力池化后残差接入），不要直接加一个多层 Cross-Attention。S2/S3/S4 用同样的张量长度、输入维度/参数、监督样本和训练更新，必要时显式匹配参数和 FLOPs 并另报新增访存。

训练模式首轮**只用现有真实 Ranker P1 候选条件蒸馏**；不得同时改变训练历史来源与空间结构，否则无法归因。若旧 MLP 原始数据与新数据不同，S1 是必须的同数据基线。

### 3.3 三层指标

**预测层**：相同真实历史下 `TV(p_T,q)`、Overlap、KL、q 熵、Top-k 在 p_T 下覆盖、来源分组、位置边界与已提交邻居数；保留类型 mask、softcap、top-p/temp 一致。建议主诊断为 overlap，不能只用 CE 或 q top1 accuracy。

**到达层**：`α_P = p_T(P1|h)`，`s2 = α_P×overlap_F`；根据当前 PF 逻辑估计连续推进增益。对所有样本和可到达样本分别统计，不用受接受条件筛选样本制造偏差。

**性能层**：完整空间检索、状态维护、Draft 前向、Target 批量验证和缓存成本；通过 PF-K2 真实在线运行比较 S0/S1/S2/S3/S4，报告是否增加预填/每轮预算。每轮速度与长期生成分开统计。

**必要的代价检查**：如果空间邻域查询要求 CPU<->GPU 交换、频繁 Python 字典、动态 scatter/gather 或额外 GPU 同步，必须计入；优先固定稀疏索引表或可增量更新 GPU memory，使用 CPU 快速查找也必须实测 host 开销。

### 3.4 N2 继续条件

- S3 必须先优于 S1，**还要优于 S2 和 S4**，否则无法把改善归因于“候选三维邻域结构”。
- 优先要求独立来源配对的连续接受价值有稳定改善，且在线净吞吐优于最好同预算无空间对照；若 overlap 有改善但在线收益不足，停止空间模块或简化其开销。
- 如果 6/26 邻居常不可用、收益仅由特定场景决定，应报告适用范围；不扩大到更复杂空间 Transformer。

**交付**：`neighbor_availability.jsonl`、`spatial_labels.jsonl`、`spatial_control_train.json`、`S0_to_S4_overlap.csv`、`spatial_cost.csv`、`N2_decision.md`。

## 4. N3：最小正确在线方法（只实现已经值得做的支线）

三种合法方向：

- N1 有效、N2 无效 → **保持旧 MLP + 固定宽度 b=2 的严格树**。
- N1 无效、N2 有效 → **保留旧 PF-K2 + 轻量空间条件 Feature Draft**。
- N1/N2 均有效 → **联合版本**，先做静态 b=2 并测试两模块交互，不引入动态预算网络。

### 4.1 建议接口分离

1. `candidate_position.py`：调用原 Ranker，仅输出候选/合法性、原分数排序；不得把 HGB score 当归一化提案概率。
2. `committed_spatial_memory.py`：维护已提交 Position/Feature；`lookup_committed(P1)` 只读；临时分支 memory 另行 copy-on-write，拒绝时 rollback。
3. `feature_draft.py`：输入冻结历史状态、pending F、P1、可选邻域，输出经过目标策略处理的规范化 q_F 与可追踪候选序列。
4. `candidate_tree.py`：生成固定形状/祖先索引，登记每个节点的 Type/Position/Feature/父节点/提案概率/路径 RoPE。
5. `target_tree_verify.py`：树形注意力可见性、缓存、条件概率与严格残差流程。
6. `benchmark.py`：收集简洁 trace，不改变普通测速路径。

这是**职责划分建议**，不是未经核实的主工程真实文件路径；可从当前 `pf_engine.py`、`pf_draft.py`、原 `pf_feature_unified.py` 的受控实验副本渐进适配。

### 4.2 必须通过的正确性测试（阻断式）

- 小词表：包括多点 Feature 分支、EOS、两轮 P/F 序列的穷举概率；将输出 TV 与 Reference Target 检查在浮点容忍范围内。
- 实际 Tensor：p/q 都在相同历史、词表 mask、softcap、temperature=0.9、top-p=0.9 下；对重复/非法/剩余候选空集正确处理。
- Target logits：树形批量验证与逐路径 full-prefix 重算比较概率；指定浮点容忍范围并报告最大偏差。3D RoPE 的 Position ID、P/F slot 必须完全一致。
- Branch cache：Feature 分支间隔离、被接受路径精确提交、拒绝后把缓存和空间记忆恢复至真实生成历史；不能复用非祖先其他分支。
- EOS/不足预算：EOS 仅合法 Position 位置，产生后不额外延长或使用残差历史；短于预期树长度时自动退化到验证过的 PF/P-only。
- 分布 vs 随机种子：对相同算法开/关计时、优化前后数学等价实现要求分支与输出一致；不同采样策略同随机种子**不要求**生成完全相同轨迹。
- 发生任一错误则暂停性能实验。

## 5. N4：训练因素、树因素和联合交互（只做已通过门槛的组合）

### 5.1 不可混淆的实验因素

**训练因素**（固定结构/参数/优化预算）：
A. 真实序列条件监督；B. 实际 Ranker 候选历史监督；C. 候选历史 + 短 rollout；每步监督取同历史合法 Target 分布。EAGLE v3 已显示候选历史训练有可能改善预测，因此新增研究需证明其对当前空间 Draft/分支树的额外贡献。

**结构因素**（同权重或配对训练、固定评估）：
- 单链 vs b=2 vs b=3；
- 同节点数（新增验证节点）vs 同实测验证时间；
- 三维邻域 vs 等容量序列邻域 vs 打乱空间；
- 仅在两条支线都有效时，做 2×2 对照：旧/空间 Draft × 单链/树。观测交互项，可能负交互而非必然协同。

**禁止同轮改变**：网络输入、参数量、训练样本、学习率/step、候选分支数、采样纠错、Target block 实现。若不可避免，拆成逐项实验。

### 5.2 样本构造/训练监督契约

每条训练记录至少有 `{source_id, prefix_hash, prefix_len, pending_F, previous_P, cached_hidden_kind, ranker_P1, candidate_F_path, token_type, position_3d, committed_neighbor_ids, target_logit_policy, q_policy, parent_id, reach_weight}`。

- `cached_hidden_kind` 要明确旧 MLP 的 pre-RMSNorm vs EAGLE v3 的 post-RMSNorm，任何转换不允许静默混用。
- 对真实提交和临时候选分支的邻域可见性分别设置 Mask；训练时的教师分布必须使用该节点**真实祖先路径**前缀重算，不能从另一分支挪用标签。
- 模型选择优先使用真实策略下的 overlap / 带到达权重的连续推进指标，并以普通吞吐作最后仲裁；不能直接只选最低 teacher-forced KL。
- 若尝试到达加权 loss，先明确定义采样分布和权重是否受当前 θ 影响；控制梯度截断、权重范围、有效样本量并与无权重严格对照。不要把 heuristics 说成理论最优。

## 6. N5：独立确认与完整三维

### 6.1 固定测试矩阵

- 最小必要方法：Target、P-only、最佳固定 PF-K2、最终候选方法，以及能证明机制来源的 1–2 个消融版本。官方样式 EAGLE/Medusa 若未做公平工程适配，列为 style-only，不宣称直接击败完整算法。
- Token：来源级分层；前缀 64/256/1024；固定窗口 256/512；2,048 长窗口及自然 EOS 子集；每来源 ≥3 个采样种子，多个训练种子需按可用资源独立测。方法顺序轮换；固定窗口和自然 EOS 各自报吞吐、实际 Token、时间、EOS 比率。避免事后仅按全部达到长度筛选作为主结果。
- 完整 3D：单 chunk 真实补全与 BOS 两组；场景 × seed × 方法配对；AR-only 时间和端到端（含 VQ 解码、Gaussian 输出、渲染及必要写入）分别报告。多 chunk/跨块接缝另设研究范围，不偷换输出规模。
- 质量：P/F 合法率、Cache 和采样正确性、可见区域与全部图像分开评估，报告 PSNR/SSIM 和可用的感知/几何指标及可视化失败样例；由于生成随机，即使理论分布保持，也不要求逐样本相同质量。
- 统计：以来源为单位、以固定对照为参照的配对比与来源级 bootstrap 95% CI；报告微平均 Token/s 与来源组几何平均二者，但不混用；训练与采样 seed 分开报告，不把轮次当独立来源。

### 6.2 继续/停止阈值（建议预先登记）

- N1/N2 探索门槛：至少一种改动在 D0 上展现 ≥约 3% 净吞吐提升的趋势，并有机制指标与成本证据；若未达到，可以据成本线修改一次**明确机制**，不得无限阈值搜索。
- D1 新开发来源：建议“相对最佳固定 PF 的来源级速度比点估计 ≥1.03，95% 区间下界 >1.00”，还要相对于最强容量/预算对照有清楚优势，作为启用保留 T 的**内部推进门槛**。该 3% 为项目实际意义阈值，并非统计定律；可在测试前按 GPU 噪声/测量预算确定并锁定。
- 最终 T：优先报告速度比与 CI、异常来源和场景方向；如果 CI 覆盖 1 或只优于 P-only 而不优于固定 PF，不宣称稳定超过 PF。
- 完整三维：若纯 Token 改善未兑现成端到端时间收益，结论限制为 Token 层；若质量或采样正确性失败，不能宣传 quality-preserving exact acceleration。
- 如果 N1/N2 都不成立，则结束树+空间训练路线，返回重新寻找结构性瓶颈，不因论文叙事需要而强行增加模块。

## 7. 实验指标字典（所有阶段保持一致）

| 维度 | 记录字段及解释 |
|---|---|
| 候选生成 | `n_proposed_P1,F1_branches,P2_by_branch,F2; q_policy; candidate_set; draft_ms` |
| 概率质量 | `target_p_P1; feature_p_mass_topk; overlap_q; P1_weighted_mass`；这些可用离线 Target oracle，但不能作为在线免费决策输入 |
| 到达与接受 | `n_proposed,n_reached,n_accepted` 各类型/深度/分支记录，`conditional_accept=accepted/reached` 与 `reach= reached/proposed` 分开 |
| 有效推进 | `n_committed_body_tokens`, `n_verified_tokens`, `extra_target_tokens`, `n_rounds`, `EOS`；PF-K2 与树均按**实际提交**而非 nominal γ |
| 验证实际成本 | `target_calls,target_input_tokens,mask_nodes,verify_host_ms,verify_cuda_ms,branch_cache_ms,rollback_ms,sampling_ms`，不同计时基准不得机械相加 |
| 性能 | `decode_only_ms`, `prefill_ms`, `total_window_ms`, `actual_new_tokens`, `tokens_per_s`, `peak_memory`, `3d_end_to_end_ms` |
| 正确性 | `max_TV_tree_vs_prefix`, `small_vocab_sequence_TV`, `cache_state_diff`, `valid_PF/EOS`, `branch_restore_pass` |
| 数据 | `source_id,scene_id,prefix_len,train_seed,sample_seed,checkpoint_hash,method_order` |

**记号规范**：Coverage 是固定候选集合的 Target 质量；Overlap 是 p/q 分布重合；Reached 是候选链实际到达验证位置；Accepted 是到达条件下接受；Committed 是轮次真实推进；Speed 是端到端计算结果。任何两个都不可直接用一个替代另一个。

## 8. 时间/资源执行原则与成果验收

1. **第一批（审计）**：N0 契约和小词表采样用例，不训练，不改官方代码，不触及未用测试来源。
2. **第二批（两个独立诊断）**：N1 离线树 coverage + Target tree kernel 形状成本；N2 已提交邻域可用率及 S1/S2/S3 等容量小样本训练。并行思考，但单独保存参数、来源与结果。
3. **第三批（仅走存活支线）**：N3 最小在线正确算法，再在 D0 上跑正式低干扰吞吐；没有净收益就停止。
4. **第四批（条件成立才展开）**：N4 候选条件数据训练、结构与训练因素消融；首先完成强 PF 同数据重训对照。
5. **第五批（冻结后确认）**：D1 一次决策、T 一次最终独立评测与单 chunk 端到端。结果不满足条件时停止，不追认微小加速为新颖方法。

**最终验收**：要支撑论文主要贡献，至少同时获得：(i) 一个有明确机制且正确的结构化候选/Feature 预测方法；(ii) 对比强固定 PF 的可复现在线净收益；(iii) 可以解释该收益的概率与计算证据；(iv) 冻结来源上的三维端到端及适用边界。即使最终负面，也要保留所有反例和可复现记录。

## 9. 论文贡献与实验映射

- Claim A「空间条件改善低成本 Feature 可预测性」→ S1/S2/S3/S4，同容量同训练，按邻域可用率和 Source 分层。
- Claim B「异构树改善给定 Target 验证预算的可延续候选覆盖」→ T0/T1/T2/T3/T4，同节点/等验证时间 + 严格多候选验证；没有真实加速则只能称候选覆盖分析。
- Claim C「联合机制带来稳定净速度」→ 2×2 联合消融、独立 D1/T 及自然 EOS/完整 3D；未实现则不可写。

**不属于新理论贡献**：接受率与 TV 等价、期望推进与总成本决定吞吐、标准残差采样的正确性。可以作为分析工具。真正需要辨认的新性质是**P/F 类型约束和空间已确认邻域是否改变了有限预算下的候选覆盖及实际计算收益**。
