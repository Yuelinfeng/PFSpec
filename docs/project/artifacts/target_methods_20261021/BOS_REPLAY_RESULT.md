# LFQ BOS 顺序重放数值结果

日期：2026-10-21。此次只重放一条 LFQ BOS 轨迹，用来比较原 Target 与图路径在相同输入和缓存历史下的输出。结果显示，短上下文 T4 图前向与原 Target 出现明显数值差异；现有 prefix 微测的结论不能推广到 BOS 短上下文。LFQ 暂不纳入方法排序。

## 结果

使用 seed `44101`、BOS prompt `[12096]`、预算 512。原 Target 重跑结果与 `lfq_bos_run01` 保存的原版 token 序列逐位相同，长度均为 512。该原序列与优化版的首个 token 差异位于 index 147，`9645 → 11885`。

探针记录并重放了 213 次 Target 前向和 156 次缓存截短。图路径沿相同 token、logit 类型、缓存起止位置和截短顺序执行；213 次调用均实际重放图，包含 T1 9 次、T2 1 次、T4 203 次，未发生回退或停用图。203 次 Feature Draft 调用的 `h_prev` 都与对应原 Target 调用的 hidden 行逐值相同，候选 q 使用相同输入、模型权重与 BF16 autocast 重算。

| 项目 | 测得结果 |
| --- | ---: |
| Target policy 最大 TV | 0.0126781 |
| Target pre-finalnorm hidden 最大逐行 L2 相对差 | 0.00132106 |
| 有限 logits 支持集一致 | 是 |
| temperature/top-p 后 policy 支持集一致 | 否 |
| BF16 Feature Draft q 最大 TV | 0.0214849 |
| BF16 q 支持集一致 | 53 / 203 次 |

首个 Target policy TV 超过 `1e-5` 的位置是 call 2、row 1、绝对位置 3，输入类型为 Feature，TV 为 `0.000438221`。该调用是从缓存位置 2 开始的 T4 前向，整次调用的最大 policy TV 为 `0.000501543`，最大 hidden 行 L2 相对差为 `0.000654258`。首个非零 TV 出现在 BOS 位置 0，数值仅 `1.30e-8`。

首个 Draft q 差异是 draft 1，对应 Target call 3、row 0、绝对位置 5；q TV 为 `0.00359588`，top-p 支持集已经不同。整个轨迹的最大 q TV 是 `0.0214849`，出现在 draft 115、源位置 301。最大 Target policy TV 出现在 call 200、T4 位置 482；最大 hidden 行 L2 相对差出现在 call 109、T4 位置 270。

## 与 prefix 微测的适用范围

此前 `runtime_verify_run02` 使用 511 个 prefix token，在位置 510 至 520 验证 T1、T2、T3、T4、T6 等形状。该轮 T4 最大 policy TV 为 `8.91e-7`，pre-finalnorm hidden 最大逐行 L2 相对差为 `1.56e-6`，输出支持集一致。

BOS 重放从位置 0 开始；第一、第二次 T1 前向与原 Target 很接近，首个明显差异随位置 2 的 T4 调用出现，且当时 read bucket 仅为 8。两次测量覆盖了不同上下文长度和 read bucket。prefix 长上下文的数值结果不能代表 BOS 起始的小上下文行为。BOS 这次单 seed 轨迹的差异也不能据此推断所有上下文的误差范围。

## 探针可信度与解释边界

原路径重跑与既有原版输出逐位一致；重放期间逐次比对了输入 token、logit 类型、缓存位置和截短位置。所有 Target 调用均有实际图重放，且 Draft hidden 的 203 次来源匹配全部精确。这些证据不支持由探针记录、位置对齐、缓存截短或 hidden 配对造成大误差。

原 Target 使用 FP32 的 SDPA fallback；图路径在同一 FP32 Target trunk 上使用显式 attention 计算。原缓存请求长度为 521，图缓存池分配长度为 1024；原 SDPA 只读取当前 `end_pos` 之前的键值，图路径使用 read bucket 并以 query 绝对位置遮住未来键值。当前证据没有显示物理缓存长度造成了观察到的差异。偏差最早随短上下文 T4 图前向出现，但本次试验没有逐层替换计算路径，不能把原因精确归到某一层或某个 attention 运算。

图路径的 Target policy 差异会传到 BF16 Feature Draft q。q 的支持集在 203 次中有 150 次发生变化。这证明同一参考 hidden 与图 hidden 的差异足以改变 Draft 分布；单条轨迹无法隔离 BF16 对误差的贡献，也无法证明最终 token 分歧由哪一次接受、拒绝或残差抽样决策引起。

结合 `lfq_bos_run01` 的三组配对结果（LFQ token parity 为 0/3），本轮保留 LFQ 不排名。此次是数值诊断，没有重新评估输出质量，也没有比较速度。

## 产物

- 探针摘要：`bos_replay_verify_run02/metadata.json`
- 完整 Target 与 Draft 原始张量：远端 `/root/autodl-tmp/gaussiangpt_target_methods_20261021/bos_replay_verify_run02/raw_outputs.pt`（约 151 MB）
- 保存原始序列与三 seed 差异：`lfq_bos_run01/records.jsonl`、`lfq_bos_run01/DIAG_LFQ_BOS.md`
- prefix 长上下文对照：`runtime_verify_run02/metadata.json`
