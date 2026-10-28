# GaussianGPT PF2X — Round 01 修订执行计划（实施前审阅版）

**版本**：Round01-R1，2026-10-20  
**依据**：`GaussianGPT_PF2X_Round01_Preimplementation_Review.zip` 中的 `00_PREIMPLEMENTATION_REVIEW.md`、`REVIEW_BEFORE_IMPLEMENTATION.md`、`WEB_PLAN_ORIGINAL.md`、11 份 `source_current/` 源码只读副本，以及上轮已确认的历史正式 PF 性能证据。  
**状态**：方案审阅；**尚未修改代码、尚未运行新增 GPU 实验、尚未取得 96k/192k 最终结果**。  
**本次决策**：**取消 Round01 原计划中直接实现 `PF-W2-lazy` 完整解码路径和 36/180 次 W2 在线比较的默认安排**；优先完成零 GPU 概率及调度验证，待已有 192k 训练及 **486 次评价全部完成**后，开展有限 p/q 覆盖和 Target 分支成本诊断；仅当能证明有实现真实净加速的空间时，提出下一轮并行分支验证设计，由网页端审批后再实施。

---

## 1. 当前状态与证据边界

审阅包共 **14 个文件**：两份实施前审阅说明、一份旧计划和 11 个当前源码只读快照。审阅包明确：2026-10-20 00:28（台北时间）读到训练 step **82,850**，step 82,800 的验证重合度 **27.5337087%**；五节点在线评价尚未开始。此数字**只表示训练时点和离线验证指标**，不是 96k/192k 的在线 Feature 接受率、吞吐或已达成加速。网页端不具备查看服务器当前进程的实时权限；后续应以实际提交的 `pipeline.json`、`training/progress.json`、`evaluation/progress.json`、最终原始逐次记录及摘要判定运行结束。

历史正式固定窗口 PF 加速约 **1.7125×**（来源等权测量）；完整自然 EOS 三维生成的端到端加速约 **1.5693×**。这两个指标不能混为一谈。12k/24k/48k 的在线 Feature 接受率分别为 34.03%/33.78%/35.60%，吞吐约 116.58/116.04/116.76 token/s；48k 没有可靠的额外收益。五节点最终比较到来前，不预选 96k 或 192k 为最好权重。

当前包未包含最终五节点评价、完整 p/q 4096 维在线样本、可运行 Target 权重和新的 GPU 性能结果。下面所有性能公式属于**决策模型或上界**，不是观测到的加速。

## 2. 关键问题：第二次残差接受并不等于减少 Target 验证

### 2.1 用真实源码定义原 W1 的状态

对 `source_current/draft_a/pf_feature_unified.py`：

- `_decode()` 位于 **245** 行；第 **278–285** 行 `tf(...)` 负责 Target Forward 和调用计数；第 **314–320** 行 `rewind(...)` 同时处理 Target/Adapter 状态。
- 第 **347–350** 行要求每轮开始 `TargetKV.get_pos() == len(prefix)-1`：完整输出末尾的 Feature 尚未处理。
- 第 **368–376** 行使用 `choose_position()` 与 `fork_position_state()` 预提议 P₁、P₂。
- 第 **416–422** 行，`MLPAdapter.propose()` 给出完整 4096 类 `q(F₁)`，Target 一次消费 **`[pending_F,P₁,F₁a,P₂]`**。
- 第 **445–453** 行，首个 Feature 拒绝后直接从 `(p−q)_+` 的归一化残差抽取真实 F₁*，`rewind(before+2)`，提交输出 `[P₁,F₁*]`，下一轮接着执行完整的 PF-K2。
- 第 **455–479** 行，若 P₁、F₁a 连续接受，则还会验证 P₂ 并从 `logits[3]` 得到下一 Feature；这就是一次验证可以推进四个输出 Token 的关键。

`source_current/bench_gaussiangpt_pf_block2_cow.py:61–68` 的 `truncate()` 修改 KV 长度与缓存 token 尾部；`model/gpt/nanochat_gpt.py:587–609` 根据缓存内的实际 token 重建稀疏 4D RoPE 索引。不能复用错误 Feature 分支上的 P₂ logits。

### 2.2 统一记号、初始缓存和轨迹假设

设当前已输出完整一对 `…P₀,F₀`，`F₀` 暂未写入 Target KV。将此时 KV 长度记为 **n**，缓存内容以 `…P₀` 结束。初始状态：

- `prefix = …,P₀,F₀`，`KV = [前缀到P₀]`，`get_pos=n=len(prefix)-1`。
- Ranker 可基于 Position 历史提出 P₁、P₂；MLP 在真实历史和候选 P₁ 下产生 F₁a。
- 以下主轨迹限定 **P₁ 接受、F₁a 拒绝、不提前 EOS**。为直观比较采用一条后续 P₂/F₂/P₃ 接受的有利示例；这是**构造的条件轨迹**，并非真实 GPU 运行或声称两套随机方法具有相同 seed 输出。
- `F₁*` 是 W1 的直接残差抽样；`F₁b` 是 W2 第二阶段接受后产生的实际 Feature。两者在给定相同已接受前缀下具有相同**条件分布**，不要求逐 Token 相等。

### 2.3 W1：第一 Feature 拒绝后的两轮

| 时点 | Target 前向 / 操作 | KV 长度（新增可用状态） | 新增可见输出 | Target 调用累计 |
|---|---|---|---|---:|
| 开始 | 已有前缀 | n，止于 P₀ | 无 | 0 |
| 第 1 轮验证 | `tf([F₀,P₁,F₁a,P₂])` | 暂达 n+4 | 尚未提交 | 1 |
| F₁a 拒绝 | 采样 `F₁*∼normalize((p_F−q₁)_+)`；`rewind(n+2)` | **n+2**，止于 P₁ | **P₁,F₁*** | 1 |
| 第 2 轮验证 | Ranker 提议 P₂/P₃、MLP 提议 F₂a；`tf([F₁*,P₂,F₂a,P₃])` | 暂达 n+6 | 待验 | **2** |
| P₂、F₂a、P₃ 都接受 | 由末尾 logits 直接抽取 F₃ | **n+6**，止于 P₃ | **P₂,F₂a,P₃,F₃** | **2** |

在这个**6 输出 Token**的示例里，第一、第二轮合计 **2 次 Target Forward、8 个 Target 输入 Token**。下一轮待处理的是 F₃。如果第二轮 F₂a 拒绝或者 P₂/P₃ 拒绝，实际推进和修复调用会不同；不能将这一示例充作平均速度。

### 2.4 原提议的惰性 W2：首 F 拒绝后第二候选接受

| 时点 | Target 前向 / 操作 | KV 长度（新增可用状态） | 新增可见输出 | Target 调用累计 |
|---|---|---|---|---:|
| 开始 | 已有前缀 | n，止于 P₀ | 无 | 0 |
| 第 1 次验证 | `tf([F₀,P₁,F₁a,P₂])` | 暂达 n+4 | 尚未提交 | 1 |
| F₁a 拒绝 | 计算 `r₁=normalize((p_F−q₁)_+)`，以实际 q₂ 提议 F₁b 并按 `r₁/q₂` 接受 | 回退到 **n+2**，止于 P₁ | 暂定 P₁,F₁b | 1 |
| 惰性补算 | `tf([F₁b,P₂])`，用**F₁b 分支的** Target logits 验证 P₂ | 暂达 n+4 | 如 P₂ 接受，生成 F₂ | **2** |
| 提交补算结果 | 保留 P₂，F₂ 暂未消费 | **n+4**，止于 P₂ | **P₁,F₁b,P₂,F₂** | 2 |
| 下一轮（预算足够） | `tf([F₂,P₃,F₃a,P₄])` | 暂达 n+8 | 视后续接受结果而定 | **3** |

按**相同 6 个可见输出 Token**的预算作对照：在已经输出 `P₁,F₁b,P₂,F₂` 后，只剩一对 P₃/F₃，按现有近预算回退，W2 还需 `tf([F₂,P₃])`（P₃ 接受时），因此在相应理想路径中为 **3 次 Forward：4+2+2 输入**。与上面的 W1 **2 次 Forward：4+4 输入**对比，W2 没有减少 Target 调用。

若不限预算，W2 后续的四 Token 验证会把某些工作移到第三次 Forward；仍然没有由“第二 Feature 接受”本身产生的确定调用节省。对实际随机轨迹须按同前缀、同生成预算和真实耗时进行配对评价，不能强求不同采样算法逐 token 对齐。

### 2.5 特别检查三个边界

1. **第二候选拒绝**：W2 仍从第二次残差抽一个实际 Feature；若直接进入下一 W1 轮，它与原 W1 一样已得到一个合法 Feature，但增加了第二提案及残差计算。不存在附加已验证后缀。
2. **第二候选接受但 P₂ 拒绝**：需要从 `F₁b` 条件下的 Target P₂ 分布做 Point Residual；若抽到 EOS，必须立即结束，不处理 Feature。否则恢复真正 P₂ 并得到对应 F₂。此路径还可能增加补算开销，不能以“F₁b 接受”代替 P₂ 接受。
3. **只有补算短块更便宜**：与其为 W2 引入第二 Feature 抽样，应该先对照一个更简单的 **`W1-short` 反事实调度**：直接残差采样 F₁* 后，用 `tf([F₁*,P₂])` 产生 Target F₂，而不是下一轮四 Token PF-K2。二者输出分布均合法。若短块更快，收益来源是**调度窗口缩短**，而不是第二候选的接受率；且仍需考虑失去后续 F₂/P₃ 一轮接受的机会成本。

**审阅决定：惰性 W2 没有独立、明确的加速机制。Round01 不实现完整 W2-lazy，更不以二次 Feature 接受数作为上线准则。**

## 3. 概率正确性与速度正确性必须分开

对同一已确定前缀下合法概率分布 `p,q₁`，先抽 `a∼q₁`，以 `min(1,p(a)/q₁(a))` 接受，拒绝后残差为：

`r₁(f) = [p(f)−q₁(f)]₊ / Σ_j[p(j)−q₁(j)]₊`。

在拒绝情形下，可以取一个依赖已抽到的 `a` 的合法分布 `q₂(·|a)`，再抽 `b∼q₂(·|a)`，以 `min(1,r₁(b)/q₂(b|a))` 接受；若拒绝，从 `r₂∝[r₁−q₂]₊` 抽最终 F。在条件给定 `a` 时，第二阶段输出仍服从 `r₁`，因此合成后最终实际 F 仍服从原 Target 的 `p`。此数学推导不意味着已有 CPU 数值测试通过；数值测试尚待 Gate 1 完成。

首候选被拒绝时，其自身在 `r₁` 中的正概率为 0；所以 `q₂` 排除 `a` 是合法选择，但**合法不等于更快**。

- **W1**：第一次 F 拒绝，立刻得到一个来自 r₁ 的最终真实 F；下一轮可继续完整 PF-K2。
- **W2-lazy**：第一次 F 拒绝，经过 q₂ 的第二次接受/拒绝，再得到**同样来自 r₁ 的最终真实 F**；没有多一个已验证 Token。多出来的是候选采样、概率处理，以及可能的短块 Target Forward。
- **潜在有用的 W2**：必须在真正缩减串行 Target 调用、或者将 **F₁b 后的 P₂（及可能的更多后缀）在同一次 Target 并行验证中完成**时，才有可能抵消分支成本。并行分支能否做到、代价多大，目前未知。

不能把在 F₁a 分支上得到的 `logits[2]` 当作 F₁b 分支下的 P₂ 分布。原 `MLPAdapter.propose()`（`pf_feature_unified.py:127–153`）只返回相同前向的实际 top-p=0.9、温度0.9后的 q₁；“宽支持 q₂”需要拿到原始 Feature logits，但 Round01 的 CPU 阶段无需修改此接口。

## 4. 真实 2× 目标及收益空间计算

### 4.1 统一计时口径

固定窗口主指标：同一 Target 权重、设备、FP32 Target、采样 temperature=0.9/top-p=0.9、同来源与同前缀、相同新 Token 预算，**包含**缓存初始化、前缀预填、Position Ranker、Feature 提案、Target 验证、拒绝/残差采样、恢复、最终同步。排除权重加载、预热、VQ 解码和渲染（分别报告）。EOS/输出长度单独保存。主加速按来源等权几何平均配对速度比，来源重采样 95% 区间；完整自然 EOS 与三维有效性单独确认。

历史正式来源等权速度比约 **1.7125×**，向 2× 前进，在**同口径、假设 Target 耗时不变**时 PF 总耗时需要降低 `1−1.7125/2=14.375%`。**历史合并吞吐** 119.91 vs 70.67 token/s 则对应需吞吐提高 `(2×70.67/119.91−1)≈17.9%`、时间减少约 **15.2%**。二者来自不同汇总口径，**不可交叉用作同一个停止阈值**；新测试必须重测 Target 并以配对结果判定。

### 4.2 两类互相独立的收益证据

**证据 A：Feature 分布可救回概率（不是吞吐）**。

令 `A₁=Σ_f min(p(f),q₁(f))` 为首次 F 候选的理论接受概率，`R₁=1−A₁` 为拒绝概率。在首候选 `a` 被拒绝的条件下，其权重与 `[q₁(a)−p(a)]₊` 成比例。第二候选条件接受覆盖定义为：

`β(a)=Σ_f min(r₁(f), q₂(f|a))`。

由此，在已经具备合法 P₁ 条件、首次 F 拒绝之后的期望二次接受概率为：

`β̄ = Σ_a [q₁(a)−p(a)]₊ β(a) / R₁`，当 `R₁>0`。

可同时估计两种 q₂：①从 q₁ 排除第一次 a 后重归一化；②从**同一次 MLP 的原始 logits**算温度 0.9、top-p=1.0 的分布，再排除 a 后重归一化。**不能用 Target p 构造推理时实际 q₂**。`β̄` 是条件理论概率，不是额外输出 Token 或减少 Target 调用的比例。

**证据 B：新增验证的实际经济性（能否减少总耗时）**。

若以后预先提出 `F₁a/F₁b` 并尝试一次并行 Target 验证，则真正可能受益的事件必须同时满足：P₁ 接受、F₁a 拒绝、F₁b 被残差接受，以及 **F₁b 条件下 P₂ 可被正确验证并有后续推进**。所需统计至少包括 `p(P₂|…P₁,F₁b)` 和正确分支 KV 的处理开销。

非常宽松的上界：假定每次上述有利事件最多**节省一轮串行 Target 验证**，且分支验证完全免费，可用 `E[有利事件比例] × 现有每轮 Target 验证平均耗时` 计算**过度乐观的候选时间节省**。这是上界，不是预期速度；若连这样的上界都不足以填补距离 2× 的时间缺口，应立即停止该分支路线。若上界足够，仍须进一步扣除并行分支 token、mask、KV、MLP/Ranker、GPU launch、残差及同步成本，判断净节省是否为正。不得用该上界直接宣称预计达到 2×。

最重要的比较对象不是 W2-lazy 相对 W1 的“二次接受率”，而是 **W1 完整 K2、W1-short 自适应短窗口、以及真正预验证多个候选后缀的分支方案** 在相同输出预算下的总 Token 推进和真实 Target 时间。

## 5. Round01 修订任务（按 Gate 执行）

### Gate 0：当前持续任务隔离（立即有效，不占新 GPU）

1. 现有 192k 训练和 **486 次既定评价** 是唯一当前许可的 GPU 工作；不得重启、抢占、覆盖其 checkpoint、log 或 output。
2. 主代理只读观察 `pipeline.json`、`training/progress.json`、`evaluation/progress.json`。仅当训练完成、五节点评价文件齐全、分析结束时才解锁 Gate 2。
3. 对所有后续实验记录 Target 权重、Ranker、MLP 权重文件路径及 step 元数据、运行机器、注意力后端、FP32、temperature/top-p、source split；选择最佳五节点以**实际在线速度、来源配对统计与正确性**为依据，不凭较高 step 或离线验证重合度选权重。

**停止条件**：已有流程未 `complete` / 仍在运行 → 仅允许纯 CPU 工作，不启动任何新增 Target、MLP 或性能 GPU 进程。

### Gate 1：零 GPU 概率与调度验证（立刻可做；训练 0 步，GPU 0 小时）

**任务 1A：只读源码对应**

- 逐行对照 `draft_a/pf_feature_unified.py`：`MLPAdapter.propose`、`decode/_decode`、`tf`、`rewind`、`point_accept`、`feature_accept`、`point_residual`、`finish_round`。
- 对照 `bench_gaussiangpt_pf_block2_cow.py`：`CaptureLastBlock.run`、`truncate`、`residual_sample`、`fork_position_state`；`draft_a/pf_ranker_eval.py`：`position_state`、`typed_kinds`、`policy`、`cache_for`；`gpt_position_spec_decode_smoke.py`：`choose_position`；`model/gpt/nanochat_gpt.py`：`_build_sparse_rope_indices`；`model/gpt/engine.py`：`KVCache`。
- 输出 `SOURCE_MAP.md`（具体行号 + 调用关系 + 待核对差异）。远端实际源码如不同于只读快照，首先记录差异，不默默按旧行号修改。

**任务 1B：独立 CPU 数学模块，不改生产解码**

- 新建 `research/pf2x_round01/probability_probe.py`，只包含 NumPy 或纯 Python 函数：`normalize_nonnegative`、`residual_distribution(p,q)`、`conditional_second_q(q1,first,mode,raw_logits=None)`、`second_overlap(r1,q2)`、`enumerate_two_stage_distribution(p,q1,q2_policy)`；不能调用 CUDA，也不能以 Target p 生成 q₂。
- 对 2、3、4、8 类小词表精确枚举第一候选、第一次接受、第一次拒绝、二次接受、二次拒绝，核对最终输出概率等于原 p；要求误差量化并写入测试报告。特别覆盖 `q1(a)=1`、`p(a)=0`、支持集不交、排除重复、零残差（不应在实际拒绝条件下出现）、宽/窄支持、极小概率和零项。
- 对 4096 维随机合法分布做**CPU** 数值稳定性与 Monte Carlo 辅助测试；Monte Carlo 只能作为数值佐证，正确性仍以代数和穷举为主。保存随机种子、样本量、误差和异常路径。
- 输出 `PROBABILITY_PROOF.md`、`tests/probability_results.json` 与测试脚本；不新建生产 `pf_multicandidate.py`，不修改 `MLPAdapter.propose()`。

**任务 1C：同长度调度模拟器（不运行 Target）**

- 新建 `research/pf2x_round01/schedule_trace.py`，仅模拟事件、输出数、KV 已提交槽位、Target Forward 次数与输入长度，至少覆盖：P₁拒绝、F₁首拒且二接、F₁首拒且二拒、P₂拒绝、全部接受、预算尾部 K1 回退、合法 EOS。
- 比较四个调度：`W1-K2`；`W1-short-after-F-reject`；`W2-lazy-short`；`W2-sample-only-then-W1`。相同**实际生成 Token 数**为对照边界，必要时只比较预算尾部合理 K1 回退的轨迹；不以不同随机序列的 token 字面一致为前提。
- 在 `SCHEDULE_TRACES.md` 中附至少两个 6/8 Token 的例子：逐次 `[input]`、缓存 `get_pos`、新增输出、累计 Forward 以及适用事件条件，标明哪条是 W1 的源码路径、哪条是假设调度。
- 给出一条明确判据：**如果仅增加一次合法 Feature 抽样而不验证额外后缀，新增独立输出 Token 数为 0；若从四 Token 改成两 Token 提速，其增益应归于窗口调度而非第二候选。**

**Gate 1 产物**：`SOURCE_MAP.md`、`PROBABILITY_PROOF.md`、`SCHEDULE_TRACES.md`、`tests/probability_results.json`、`research/pf2x_round01/*.py` 的真实差异。**不输出 W2 解码器实现，不跑 GPU**。

**停止条件**：若数学枚举失败、无法正确保持 prefix/KV 对齐或远端源码与快照差异影响机制，暂停进一步修改，先提交阻塞问题；不允许通过跑 GPU 绕过正确性问题。若确认懒式 W2 没有独立推进收益，则将 W2-lazy 标记为不实施（即使概率测试完全通过）。

### Gate 2：五节点任务完成后的有限 p/q 与分支成本诊断（默认总计 ≤1 GPU 小时；训练 0 步）

**任务 2A：结算五节点，确定唯一开发参照**

- 必须看到最终 12k、24k、48k、96k、192k 的确切 checkpoint step、186 次在线曲线评价和 300 次五节点对照（合计 486），以及 `pipeline.json`、`evaluation/records.jsonl`、`evaluation/summary.json`、来源划分清单。
- 基于开发来源的**同轮实际吞吐和来源配对区间**选 W1 参照。若无可信改进，选已有 12k；较高 Feature overlap 但速度未提高，不视为更优。保存结果到 `CHECKPOINT_SELECTION.md`。独立确认来源不得用于选择。

**任务 2B：有限在线条件 p/q 提取——只作研究诊断**

- 基于远端真正使用的 `pf_feature_unified.py`，新建**独立只读诊断入口** `research/pf2x_round01/collect_pq.py`；可选择少量共享函数但**不改生产 `decode/_decode`、不改 Target 采样法则和模型参数**。
- 使用已有开发来源 **10 个场景 × 2 前缀（64/256）× 每前缀最多 24 个合格 F₁ 条件 = 至多 480 条**。按单个场景、prefix、seed 和出现轮次保存位置，记录真实 Target `p_F`、实际 MLP `q₁`、为构造宽支持 q₂ 所需的同次 MLP raw logits、P₁ 被验证接受的标记、真正的 Target 策略 `T=0.9, top_p=0.9`，以及对应 P/F token ID。不可使用未来真实 Feature 作为额外条件。
- 离线分别计算 `A₁=Σ min(p,q₁)`、`R₁`、`β̄`、二次候选排除重复的合法性、两种 q₂ 的覆盖、来源级均值/离散度。计算二次拒绝概率；**不要把 `β̄` 直接换算成 Token/Forward 或速度**。
- 为估计“能否提前验证 P₂”的乐观条件，对至多 **96 条** 有代表性的二次分支，单独从正确 P₁ KV 状态计算 `p(P₂|F₁b)` 与必要目标隐藏状态；必须记录这是额外 Target 条件计算，不能复用首条 F₁a 分支 logits，也不能混进普通推理耗时。
- 主要保存轻量数值摘要和来源统计；如需要保留完整4096维 p/q，可用独立压缩 NPZ，只包含上述采样点、seed、来源映射与策略元数据，不包含模型权重与私人数据。若没有足够真实 F₁ 首拒事件，报告实际样本数，不能补虚拟观测凑满480。

**任务 2C：Target 成本曲线，不实现分支解码**

- 新建 `research/pf2x_round01/measure_target_shapes.py`（只用于研究测时），在**同一合法前缀/条件 KV**下分别计量：`tf([pending_F,P₁,F₁a,P₂])`（4-token 主块）、`tf([F₁*,P₂])`（2-token 修复短块）、`tf([F₁*,P₂,F₂a,P₃])`（4-token 下一轮完整块）。两 Token 与四 Token 的前缀/分支依赖必须真实、合法；每次测量从正确 KV 快照/回放恢复，相关恢复设置成本不得漏报。
- 分开报告单次 Target Forward 设备耗时、Host wall、KV 建立/回放/截短、MLP/Ranker 调用、GPU launch及采样开销，完整普通 Decode 总 wall 为后续上线唯一速度依据；不同插装方式的计时不能相加或与正式吞吐互相替代。
- 不实现 tree mask，不修改 `model/gpt/nanochat_gpt.py`、`flash_attention.py` 或 `engine.KVCache`。若仅作可行性探针测试批处理两个分支，必须标记为 **成本下界/代理实验**，不可假装它等价于共享前缀的正确 Tree Attention。

**任务 2D：收益空间判定，不运行新的完整 W2**

- `analysis/FEASIBILITY.md` 同时报告：首次 F 接受与拒绝、二次残差候选 `β̄`、在二次接受条件下 P₂ 接受的估计、预计有利事件/轮的区间、W1 每轮真实 Target 与非Target成本、2/4-token 目标前向成本分布、KV/branch代价、来源差异。
- 首先使用上述**乐观、忽略分支成本**的节约上界检验距离真实 2× 的差距；若上界本身不足则停止。如果上界可能足够，再代入**有根据的非零**分支额外时间，判断可否存在净收益；无可测 branch 成本时明确标记“成本未知，不能批准完整实现”。
- 单独判断 `W1-short` 作为窗口调度策略是否有研究价值：比较两 Token 节约与放弃 F₂/P₃ 一次验证的期望推进损失，而不是把短块节约归因于第二候选。

**Gate 2 资源**：新增训练 **0 步**；实验总上限 **1 GPU 小时**（包含抽样诊断、必要 warmup、Target 条件 Forward 与 Profile，不含当前持续训练自身）；若无法在预算内完成，则按 2B→2C→2D 的信息收益优先级报告真实完成情况，不追加资源。预留测试来源/独立确认集暂不使用。

### Gate 3：网页端审阅决策——禁止自动跳入完整 Branch2

按下面优先顺序做出**一个**下一轮建议：

1. **停止双候选分支路线**：若二次覆盖太弱，或即使“分支成本=0、每个有利事件省一轮 Target”也填不平 2× 时间缺口；或者正确实现所需分支成本估计使净节省≤0。交付负面结论，下一轮优先考虑空间条件轻量 Feature MLP（须另行审批训练）。
2. **研究 W1 的短/长窗口自适应**：若两 Token Target 块的实测成本明显较低，并且扣除失去长块机会后仍有可信收益。该路线应**直接使用 W1 的精确残差抽样**，不要携带不必要的第二 Feature 候选。下一轮再批准真正在线实现与同预算 Target/W1 对照。
3. **有条件批准分支并行设计**：只有当真实 p/q 覆盖、P₂ 条件概率与非零分支成本支持较大正向净收益，且有实际希望填平 2× 缺口时，下一轮才设计/实现带正确分支祖先注意力、4D RoPE、Position 约束、KV commit/rollback 和 Target 验证的并行路径。**本轮只写风险与最小实现方案，不改模型 Target 主干。**

**Round01 默认不含**：新的 W2-lazy 生产路径、GPU 正式 W2 在线矩阵（旧计划的 36/180 次取消）、新 Draft 训练、EAGLE 重训、直接树形 Attention 改造、全量三维导出和独立确认来源测试。收到 Gate 2 结果 ZIP、经网页端明确确认下一轮计划前，主代理不得自行跳过 Gate 3。

## 6. 最多三名 Luna/max 子代理分工

| 代理 | 执行内容 | 唯一写入范围与必交产物 | 明确禁止 |
|---|---|---|---|
| **Luna-1：概率与代数（gpt-6-luna/max）** | `p/q/r₁/q₂` 推导、支持集、数值例子、穷举与 CPU 4096 维实验 | `research/pf2x_round01/probability_probe.py`、`PROBABILITY_PROOF.md`、`tests/probability_results.json` | GPU、训练、修改 Target 或生产 decode |
| **Luna-2：调度/KV 只读审阅（gpt-6-luna/max）** | 按源码模拟 W1/W1-short/W2-lazy/W2-sample-only，同输出长度逐轮 trace；规划真实条件 Target shape 诊断 | `schedule_trace.py`、`SOURCE_MAP.md`、`SCHEDULE_TRACES.md`、`measure_target_shapes.py`（先编写，Gate 2 前不得运行 GPU） | 生产 W2 解码、Tree Attention、绕过状态检查 |
| **Luna-3：采样、成本分析与网页审阅材料（gpt-6-luna/max）** | 设计 480 条 p/q 限量提取、96 条分支 pP₂ 诊断、统计区间与成本上界；五节点完成后执行经批准的诊断、压缩证据 | `collect_pq.py`、`FEASIBILITY.md`、`CHECKPOINT_SELECTION.md`、`results/*`、Review ZIP | 新训练、新 W2 在线跑分、提前动确认来源、干扰现有 486 次评价 |

主代理只负责远端实际状态与文件核对、任务分配、处理接口冲突、Gate 开关、最终 ZIP 整合。**同一生产源码不允许多人修改；本轮原则上无人修改生产解码器。**

## 7. 收到 96k / 192k 结果后的分叉

| 五节点结果 | 下一步 |
|---|---|
| 96k/192k 的离线重合度更高，但在线接受/吞吐无可信增益 | 保留 12k（或上一同轮已确认最快权重）作为 W1，Gate 2 只做 p/q 与分支成本诊断，不再同配方训练 |
| 某节点在来源配对吞吐上有可靠提升，且正确性/EOS无回退 | 选该节点为唯一开发 W1 参照；保留12k历史锚点，使用新实测速率重算 2× 缺口 |
| Feature 接受率增加但吞吐未提高 | 量化有效推进、验证次数和 MLP/Target 额外成本；不因接受率单点更好就扩大 Draft |
| 五节点全部表现近似或某些退化 | 不继续 384k 同配方训练；转向结构/提案组织诊断 |
| 已正式测得 2× 以上点估计 | 只视为候选达标；下一轮首先安排独立来源区间、自然 EOS 完整输出、三维有效性验证，而不是宣布达成 |
| 评价仍在运行或不完整 | 停留 Gate 1；**不启动 Gate 2** |

## 8. 网页端审阅 ZIP 格式（必需根目录文件）

建议文件名：`GaussianGPT_PF2X_Round01_CostProbe_Review.zip`

```text
GaussianGPT_PF2X_Round01_CostProbe_Review/
  00_READ_ME_FIRST.md                 # 状态、完成/未完成、训练评价是否确实结束、下一轮待决策问题
  DECISION.md                          # 停止Branch2 / 研究W1-short / 申请并行Branch2设计（三选一）
  SOURCE_MAP.md                        # 实际源码路径、函数行号和必要差异
  PROBABILITY_PROOF.md                 # 精确残差、二次q2、支持集的依据
  SCHEDULE_TRACES.md                   # 相同生成长度的逐轮Token/KV/Forward表
  CHECKPOINT_SELECTION.md             # 五节点结果结算；如任务未完成标记 not_run
  FEASIBILITY.md                       # p/q覆盖、Target形状成本、2×收益上界和未知项
  metadata/
    run_state.json                     # 远端pipeline、评价完成状态、完成时间/证据位置
    hardware_and_policy.json           # 同硬件/注意力后端/精度/温度/top-p
    sources_and_checkpoints.json       # 开发来源、权重/step、确认来源未使用
  results/
    probability_results.json           # 纯CPU概率验证
    schedule_traces.jsonl              # 静态状态机轨迹
    pq_contexts_summary.csv            # 如Gate2已完成；真实来源和样本数
    pq_coverage_by_source.csv          # 如Gate2已完成；A1/R1/β与分布
    target_shape_timings.csv           # 如Gate2已完成；2/4-token及KV成本
    feasibility_bounds.json            # 上界计算含假设/数据口径
    profile_components.json            # GPU Profile与Host计时分开
  code/
    changes.patch                      # 仅本轮实际改动；默认没有生产解码改动
    research/                          # 新增小型CPU/诊断脚本
  logs/
    commands.txt                       # 实际执行命令
    concise_run.log                    # 摘要和失败信息
```

每个结果表必须写清 method、stage、场景、前缀、seed、样本数、权重step、budget、采样策略、脚本名及源文件路径。**未运行的 Gate 不应提供看似正式的数据；填写 `not_run` 和原因即可。** 保留关键数值原始记录，ZIP 不包含模型大权重、隐私配置或整个仓库。上传网页端后，我可以独立核对样本分组、概率代数、成本分项与 Stop/Go 判据。

## 9. 最小补充材料与执行权限

本审阅包已包含 `pf_feature_unified.py`、`pf_ranker_eval.py`、`bench_gaussiangpt_pf_block2_cow.py`、`gpt_position_spec_decode_smoke.py`、`feature_acceptance_eval.py`、`collect_feature_teacher_from_s3.py`、`model/gpt/engine.py`、`nanochat_gpt.py`、`flash_attention.py`、`rope.py`、`rope_config.py`。**零GPU Gate 1 不需要任何额外源码**。

Gate 2 执行前最小补充：

- 实际远端当前版本与这 11 份快照是否相同；如已变更，只需相关 `git diff`/函数全文。
- 当前仍在运行的 `feature_extend192` 的真实完整状态文件及最终 12k/24k/48k/96k/192k 逐次在线评价、来源分组与权重 step 元数据。
- 用于选择开发场景的固定 source manifest 及现有评价环境入口；如 `evaluate_extension.py` 未在此只读包中，Codex 只需从当前服务器补取该**一个脚本**及其实际调用入口，不必上传整个工程。
- Gate 2 在远端本来可读取的 Target、Ranker、MLP 权重路径，无需上传网页端。

如进入下一轮真正并行树形验证，再申请完整当前 Attention/KV 推理入口、额外 batch/branch mask 依赖与正确性测试；**Round01 不要求也不修改这些复杂接口**。

## 10. 本次正式授权建议

**立即允许**：Gate 1 零GPU概率证明、枚举测试、同生成长度状态轨迹模拟、现有源码接口确认与小型诊断脚本准备。  
**条件允许**：现有 192k 训练和 486 次评价全部完成并提交最终状态文件后，按 ≤1 GPU 小时执行 Gate 2 的 480 条以内 p/q 覆盖、96 条以内真实分支条件概率和 Target 2/4 Token 成本诊断。  
**本轮不批准**：直接实现或上线 W2-lazy、跑 W2 与 W1 的完整在线矩阵、写树形 Target 验证、追加 Feature 模型训练。  
**下一轮网页决策**：收到完整 ZIP 后，在“停止 Branch2 / 尝试 W1 自适应短窗口 / 正式设计并行多分支 Target 验证”三者中选择一个，再对 Codex Luna/max 发布详细执行指令。

**研究原则**：真实 2× 必须以原 Target、同硬件/精度/采样、包含缓存/前缀/提案/验证/恢复/采样的实测端到端解码时间来确认，并单独验证自然 EOS 和三维有效性；不能把训练运行状态、离线重合度、第二 Feature 理论接受或成本上界作为已取得的速度结果。

拒绝任何不必要的工程化设计与流程，包括 PR/CI、工程门禁、可信基线建立、反复复核等。始终围绕全局目标，核对真实依据，合并相关任务，持续推进实质进展
