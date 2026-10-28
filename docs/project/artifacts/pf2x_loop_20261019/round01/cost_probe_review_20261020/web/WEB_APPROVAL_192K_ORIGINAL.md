# GaussianGPT PF2X — Round01 五节点确认与有限真实诊断执行批复

日期：2026-10-20。依据：`GaussianGPT_PF2X_192k_FinalResults_20261020.zip` 中 `00_READ_ME_FIRST.md`、`feature_extend192/FINAL_RESULTS.md`、`FINAL_EVIDENCE_REVIEW.md`、486 条 `evaluation/records.jsonl`、`round01/`、`research_code/` 与 `source_current/`。

## 1. 正式决定与证据

**批准唯一开发 Feature 权重：精确 step12000**，路径：
`/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`。

同轮五节点 12k/24k/48k/96k/192k 的 Feature 接受率为 34.0312% / 33.7806% / 35.6000% / 35.7257% / 35.2191%，合并吞吐为 121.7966 / 121.1858 / 121.8011 / 121.6328 / 121.6911 token/s。96k、192k 相对 12k 的来源等权速度分别为 0.99894× [0.98029,1.01723]、0.99895× [0.98063,1.01620]。五节点 300 条完整比较、31 节点 186 条中间曲线已结束；本轮无 Target-only 对照，不能拼接历史 Target 宣称新的 2×。

新增训练 **0**；不实现惰性 W2 解码器、不实现树形 Attention、不恢复旧 36/180 在线矩阵、不修改生产 PF 解码器、不占用独立确认来源。

## 2. R1 采集参数（批准）

- 硬件：既有 RTX 4090；Target、Feature MLP 均 FP32；batch=1；temperature=0.9；top-p=0.9。
- 来源：沿用 10 个现有开发场景（`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt`），prefix=64/256；seed_base=34701,34702,34703；每个 W1 解码预算为 **64** 个新 Token。
- 一次 `collect_pq.py` W1 采集，10×2×3=60 次正常生成，逐来源×prefix 最多保存 24 个已接受 P1 的 Feature 分布，总计 **不超过 480 条 p/q**。不足如实保存；`vectors.npz` 保存 4096 维真实 p、q1 和同次原始 MLP logits。
- `q2=narrow`：从 q1（top-p 0.9）排除首候选并归一；`q2=wide`：从同次 MLP raw logits 构造 temperature 0.9、top-p 1.0 的分布，排除首候选并归一。两种 q2 不得使用 Target p 构造提案。
- 分支：**narrow 最多 48 条，wide 最多 48 条，共最多 96 条独立 Target P2 条件**。均选来自真实 P1 已接受、F1a 实际拒绝且 q2 有合法支持的记录。按 10 来源×2 prefix 尽量平衡；不足不得补造。尽可能记录两组共有 sample_id 便于成对比较；现有 `_balanced` 按模式各自选择，无法保证两组完全相同，因此实际分析必须保留两种模式各自的来源/前缀覆盖，不得将非配对数据当作配对结果。
- 2/4-token 配对成本：**两模式合计最多 4 个已有真实上下文**，建议 narrow 2、wide 2；每个上下文每个形状各预热 1 次，之后 8 组交替配对。计入 4×(2+2×8)=**72 次额外 Target 形状计时前向**、最多 **76 次缓存截短**，并与独立分支条件前向、prefill 及其他 Target 工作分列。注意 `measure_target_shapes.py` 默认 `--pair-contexts=4` 是**每次调用**的默认值，两次执行必须显式 `--pair-contexts 2`，不得各自默认 4。没有合法 P3 的上下文如实记录缺测，不偷偷替换。
- 预算：所有加载、初始化、预热、普通/采集 16-token 同 seed 重放、W1 采集、narrow 与 wide 分支、配对计时、同步与分析所需 GPU 工作，**合计不超过 3600 秒 GPU 占用墙钟**。不允许每个命令重新计时 1 小时。以主代理统一记录的开始时间/剩余额度管理，防止进入下一个无法在限额内完成的阶段。分析优先使用 CPU。仅有合成 NPZ CPU 通过不等于 GPU 接口通过。

## 3. 执行顺序和明确的阻断条件

1. **接口预检（先于正式采集）**：在远端实际源码环境读取 `draft_a.feature_target_v2.HEAD/RANKER` 等历史权重指向，确认 Target、VQ、Ranker、step12000 Feature 与其 HEAD 路径相互匹配；验证 EOS=12096、F 词表 4096、xyz、4D RoPE / KV 语义。先用脚本 `--show-plan` 检查绝对路径、现有 10 来源、输出目录和样本配额。
2. **16-token 同 seed 普通/采集回放**：由 `collect_pq.py` 内置 `_trajectory_parity` 执行，要求输出 Token 逐项相同，`target_calls`、`target_input_tokens`、Position/Feature 提案、验证、接受、轮次及输出长度均相同。任何不一致立即停止，不进入 p/q 采集；只修诊断包装导致的问题，不修改生产解码逻辑。
3. **真实 W1 p/q**：只执行一次 `collect_pq.py --execute --budget 64 --max-per-source-prefix 24 --seed-bases 34701,34702,34703 --feature-step 12000`。统一保留 `metadata.json`、`trajectory_parity.json`、`contexts.jsonl`、`vectors.npz`、`w1_runs.jsonl`、`w1_profiles.jsonl`、`progress.json` 和实际调用日志。每格不足报告 missing，不强行补齐。
4. **真实分支 + Target 成本**：调用 `measure_target_shapes.py` 两次，分别 `--q2-mode narrow --max-contexts 48 --pair-contexts 2 --pair-repeats 8` 与 `--q2-mode wide --max-contexts 48 --pair-contexts 2 --pair-repeats 8`，共用同一 p/q 目录；输出目录分开。每次需使用同一 step12000、Target、VQ、Ranker、Feature HEAD。核对 KV 截短到处理完真实 P1，分支 Feature 是 q2 所抽取，P2 分布由该分支真实 Target Forward 计算，不复用 F1a logits；同步后计 CPU wall 与 CUDA event，分开统计前缀、形状、恢复、Ranker/MLP 与总成本。
5. **分析**：`analyze_cost_probe.py` 分别读取 narrow/wide 目录，各自输出完整 JSON，再在 `FEASIBILITY.md` 中并排汇总来源/前缀覆盖、首拒触发率、q2 合法支持率、残差重合度、按首拒候选加权的第二接受事件、真实 P2 条件事件、2/4-token paired cost、全部额外调用及失效/缺测。来源 CI 仅重采样来源，不包含上下文和分支抽样误差；所谓“省一轮 Target、分支成本为零”只能标明**乐观点估计而非严格上界/停止依据**。
6. **返回网页审阅**：任何新方案的实际实现均等待本次有限诊断交付及新一轮网页方案确认。不得根据局部 β 或 Feature 接受率直接宣称更快，更不能宣称已达到 2×。

### 预算不足时的停止顺序

优先保留通过同 seed 验证的 W1 p/q 数据；若 GPU 剩余额度不足，则先停止尚未开始的分支模式或配对测量并报告具体缺口，不能把未完成的测量标为 0 或填入预估值。若已执行 narrow 而 wide 无法完成，标记为单模式探索、不进行两个模式间有效性比较。不得通过粗暴中止导致丢失已经收集但尚未写盘的 p/q 样本；必要时只对诊断脚本增加有限的安全落盘/退出处理。

## 4. 源码及调用（保持当前已有目录）

复用远端/本机对应的 `patches/pf2x_round01/` 与包内 `research_code/`：

- `collect_pq.py` 的 `_trajectory_parity`、`_Probe`、`_instrument`、`_run`。
- `measure_target_shapes.py` 的 `_balanced`、`_position_tail`、`_shape_schedule`、`_execute_shape_schedule`、`_run`。
- `analyze_cost_probe.py` 的 `conditional_second_q`、`beta_bar`、`analyze`。

实际路径从现有 `feature_extend192/final_raw/evaluate_extension.py` 和已部署 `draft_a.feature_target_v2` 读取；已知：PROJECT=`/root/workspace/GaussianGPT`；CROPS=`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt`；Feature=`/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`。Target/VQ/RANKER/HEAD 应使用当前工程中已存在的实际配置文件或常量解析，**不得根据旧报告猜测缺失路径**。

现有分析脚本 `analyze_cost_probe.py` 每次接收**一个** measurement 目录，所以 narrow/wide 分别分析，再汇总；不能把只分析一个目录的结果误写为双模式汇总。

## 5. 交付

压缩成 `GaussianGPT_PF2X_Round01_CostProbe_Review.zip`，至少包括：

- `00_READ_ME_FIRST.md`、`DECISION.md`：完成状态、权重、采集结果、下一轮建议与未完成项。
- `results/pq/metadata.json`、`trajectory_parity.json`、`contexts.jsonl`、`vectors.npz`、`w1_runs.jsonl`、`w1_profiles.jsonl`；须保留实际 EOS token 与全部输出长度。
- `results/narrow/`、`results/wide/`：`metadata.json`、`branch_rows.jsonl`、`shape_timings.jsonl`、`progress.json`、必要 NPZ 与实际计数。
- `analysis/narrow.json`、`analysis/wide.json`、`FEASIBILITY.md`：P2 有利事件权重、支持缺失、Target 成本、来源汇总及不确定性边界。
- `commands.log`、`budget.json`、`source/changes.patch`：实际命令、每段 GPU 墙钟、剩余额度、诊断源码与真实差异。

**停止条件**：接口/同 seed 一致性失败立即停止；运行总预算触及 3600 秒即停止新增 GPU 工作；采样不足、无 q2 支持、不可生成合法 P3 时保留缺失事实；即使成本诊断看起来积极，也不能越过网页确认开始惰性 W2 或树形验证。若真实分支覆盖很低或分支额外 Target 成本明显不可能抵消减少的验证轮次，下一轮优先研究 W1-short / 调度或结构条件 Feature Draft，而不是盲目扩大分支。

## 6. 目标不变

正式 2× 按**相同实际任务、硬件、精度、采样、前缀和生成预算**下的原 Target 与 PF 总时间比计算；缓存/前缀/提案/验证/恢复/采样及同步在计时内，权重加载和预热另列。来源等权速度比及其 95% 区间、独立来源复现、自然 EOS、3D 有效性分开报告。**本次只是限额诊断，不是新的正式加速评测。**
