# Action probe 独立复核

## 结论

10 组完整请求的生成 token、EOS 标记和关键计数逐项一致。原始轮次文件中找到 29 个操作探针：446 个命名动作、每个动作 3 次同步计时，共 1338 次测量。每个探针都记录状态恢复标记；linear runtime 在恢复前后的序列化状态完全相同。`operations_summary.json` 的 446 个动作中位数、各 read bucket 汇总和 43 个操作收益区间均可从原始轮次记录复算得到。

这些数据支持同一已缓存位置上的操作成本对比。它们不等于完整解码器收益；forced miss、EOS 和接受长度范围都带有明确假设。

## 核对范围

读取了 `action_collect_run01/plan.json`、`progress.json`、`records.jsonl`、`rounds.jsonl`，`action_control_run01/plan.json`、`records.jsonl`，以及 `operations_summary.json`。未运行 GPU、远端命令或新来源数据，也未修改源码和原始记录。

## 请求配对与逐轮覆盖

两边各有 10 条记录，按 `pair_id` 全部一一对应：5 个 source、prefix 64/256、seed 59101、budget 256；source、context、crop 路径与 crop seed 等输入标识字段逐项相同。10/10 输出 token 序列完全相同，均生成 256 token，均未到 EOS。`target_calls`、`target_input_tokens`、`recovery_calls`、`tree_verify_calls`、accepted lengths 以及 P/F 计数等关键字段逐项一致。

collect 的 `rounds.jsonl` 有 937 行，与记录汇总的 `round_observer_events=937` 一致。记录汇总的 `stats.rounds` 合计 944，7 个请求的最后 round index 未写入观察流。读取本地 `adaptive_decoder.py` 可见，`finish_round` 只写 `eligible_for_value_observation` 为真的轮次；该值要求 first、second 候选均存在且剩余预算至少为 4。7 个缺项的最后一轮都只剩 2 个 token，因此属于观察条件排除，不是任意缺行。29 个带完整操作探针的事件均在已保存轮次中。

## 操作配对与状态

29 个探针分布在 read bucket 128/256/512，数量为 7/5/17；共含 43 个备用候选上下文对。16 个上下文有 1 个备用，12 个有 2 个，1 个有 3 个。动作构成为：216 个常规接受路径、72 个 P1 拒绝后 EOS 路径、72 个 miss 恢复路径、43 个 W1 恢复到备用 token 路径、43 个 rescue 命中路径。446 个动作各记录 3 个重复。

每个 miss token 都由探针标记为显式构造；它在已记录的合法 Position 范围内，高于先前位置 rank，且不等于 P1、EOS 或任一备用。每个 k0 miss 与对应 k1 miss 对相同 token 做恢复；每个 k0 “恢复到备用”也用该备用 token，k1 命中则提交 `[0,4]` 且不调用恢复前向。72 个 EOS 动作都提交 `[0]`，没有恢复前向。

29/29 探针报告 cache 和 runtime 记账已恢复，且 `runtime_state_before` 与 `runtime_state_after_restore` 完全相等，linear runtime 字段也完全相等。所有动作的 cache 起点、提交后位置和动作结束位置都符合记录路径长度与恢复调用数。快照范围均为 cache 起点加 5。JSONL 没有保存 K/V 原始张量，因此无法独立逐元素复算 K/V 恢复；这里的恢复依据是保存的恢复标记、位置范围和完整 runtime 状态相等。

## 图准备与温态计时

W1 形状在探针前已存在 28/29 次；唯一冷标记对应 read bucket 256。每个探针有 3 条 prewarm 操作，共 87 条。prewarm 新建了 7 个图事件：tree 4 节点/bucket 256 一个，rescue 5 节点/bucket 128、256、512 各一个，linear T1 Feature 图三个（bucket 256、512、512）。在 1338 次正式重复中，tree、rescue 和 linear 均没有新增图事件。prewarm 事件与测量重复分开记录；临时探针图在完成后清理，最终 runtime 记账恢复到开始值。

因此，`operations_summary.json` 的中位数是温态动作成本，不含这些图首次准备时间。它也不应被用来推断首次请求的完整耗时。

## 汇总复算

从每个动作的三个 `repeat_wall_ms` 重新取中位数，446/446 项与 `operations_summary.json` 完全一致。按 bucket 重算 common extra、miss extra 与 hit saved，所有中位数、均值、最小值和最大值均一致：

| Read bucket | 不同探针状态 | 备用上下文对 | common extra 中位数 | miss extra 中位数 | hit saved 中位数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7 | 10 | 0.1747 ms | 0.1688 ms | 3.2138 ms |
| 256 | 5 | 6 | 0.1872 ms | 0.1897 ms | 3.2585 ms |
| 512 | 17 | 27 | 0.2244 ms | 0.2204 ms | 3.3213 ms |

汇总中的 `n=30/18/81` 对 common extra 表示备用上下文对乘以三个 accepted commit 长度。miss 与 hit 的每个备用上下文中位数也重复三次以对齐这三个长度；所以它们的 n 不是独立状态数，也不是新增计时重复。真实独立备用上下文对数为 10/6/27。三个重复的同步计时仍有波动，例如 bucket 512 的 hit saved 最大值达到 6.6283 ms，不能把 n 当作置信区间样本数。

43 个 `operation_gain_lower_ms` / `operation_gain_upper_ms` 均按汇总保存的 policy mass 与动作中位数复算，公式结果无差异。区间改变 common-accept commit 长度 2/3/4；miss 使用显式构造的合法 token，并按 token-independent recovery cost 合并。汇总说明也指出 Feature 接受概率未在该探针中观测。

## 解释边界

每次动作计时是同步墙钟，覆盖指定的 verify、commit 与可选单 token Target recovery。它不含 Ranker、MLP proposal、policy 构造、sampling、prefill 或整段 decode 成本。该实验的 EOS 路径是合成操作路径；两组完整请求都没有 EOS。强制 miss token用于测量分支代价，不是来自当前随机残差的真实抽样。

因此，命中备用时约 3.2–3.3 ms 的动作成本差只能说明所测缓存状态与所测形状下省掉了恢复前向的操作成本；完整策略价值还取决于真实命中概率、图首次准备、其他计算与全请求时间。collect 外层请求包含同步诊断，不能与不含诊断的 control 外层墙钟比较。


## Prefix 512 扩展与汇总复算

新增的 `action_collect_1024_run01` / `action_control_1024_run01` 各有 2 条配对请求：两个既有 source、prefix 512、seed 59102、budget 128。两条输出 token 序列、EOS 标记和关键计数均相同，未出现 EOS。新 collect 轮次文件有 101 行，其中 4 个可执行树动作的探针都位于 read bucket 1024；合计 7 个备用候选上下文对、69 个动作、207 次重复计时。动作配对 token、miss 路径、`[0,4]` 命中路径和状态恢复均与原 29 个探针采用相同判据，未发现差异。

新探针的 W1 图在进入探针前均已存在。prewarm 新建一个 5 节点 rescue 图和一个 bucket 1024 的 T1 Feature 图；207 次正式重复均无新图事件。4 个探针的 cache 和 runtime 状态都恢复到记录的起始状态。

对合并后的 `operations_summary_all.json` 独立重算，33 个状态里 29 个原状态的汇总字段保持不变；515 个动作中位数、4 个 bucket 的统计及 50 个候选收益区间公式全部与原始重复计时一致。新增 bucket 1024 有 4 个状态、7 个候选对：common extra 中位数 0.2865 ms，miss extra 中位数 0.2916 ms，hit saved 中位数 3.4433 ms。汇总的 n=21 是 7 个候选对分别按三个 accepted commit 长度扩展后的数目，不代表 21 个独立状态。

`action_validation_all.json` 的 12 个成对语义、33 个探针、515 个动作、1545 次计时和状态恢复标记都与原始记录复算相符。旧、新请求的完整输出没有用来比较外层速度；collect 含同步诊断，control 不含。
