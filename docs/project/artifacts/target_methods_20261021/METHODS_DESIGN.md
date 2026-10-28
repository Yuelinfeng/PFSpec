# 同优化 Target 方法适配

日期：2026-10-21。此文件描述 `patches/target_methods_20261021/method_adapters.py` 提供给 `evaluate_methods.py` 的统一入口。适配层重用现有 Target、权重、采样和接受/拒绝代码；候选执行只在 Target 上打开动态图和显式 Attention 上下文，draft 模型继续使用原来的 Attention 实现。

## 入口和返回值

`build_context(project, device="cuda")` 加载一次原始 Target wrapper，并把 `wrapper.transformer` 作为 `context["target"]` 暴露给 runner。Ranker 和 Position 排序也只加载一次。`load_method(name, context)` 返回带 `metadata` 的 `MethodAdapter`。`decode_method(adapter, target, prompt, budget, seed, runtime=None)` 返回 `(stats, output_suffix)`；EOS token 如被生成，会保留在 suffix 最后。

旧解码器没有返回 token suffix 的方法，由本文件按原函数源码生成只增加 suffix 返回值的拷贝。统计与采样循环保持旧实现。动态执行时，适配层只把对应 Target KV cache 工厂送入 `runtime.cache_for`，并在当前调用内恢复原引用。

## 当前可运行的 12 个方法

| ID | 既有入口和权重 | 候选规则与 Target hidden | Target 最大 query T |
| --- | --- | --- | ---: |
| `target` | `draft_a.pf_ranker_eval.decode(k=0)` | Target 自采样；不消费 hidden | 1 |
| `ranker_only` | `draft_a.pf_ranker_eval.decode(k=1)`，共享 FastHistRanker | Ranker 提议离散 Position，Target 生成 Feature | 2 |
| `pf_mlp` | `draft_a.pf_feature_unified.decode`；`draft_a/runs/feature_target_continue_20261014/selected.pt` | 共享 Ranker + MLP Feature q；Feature MLP 输入 Target pre-finalnorm block state | 4 |
| `eagle_v3_g3` | `speculative.pf_engine.decode`；`/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt` | Feature Draft + Position Ranker，γ=3；Draft 读取 Target post-finalnorm hidden | 4 |
| `eagle_v3_g5` | `speculative.pf_engine.decode`；同 `eagle_v3_g3` 的 v3 权重 | 固定 v3 权重，只把 γ 改为 5；Draft 读取 Target post-finalnorm hidden | 6 |
| `eagle_v4_reuse_g3` | EAGLE v3 权重，`reuse_draft_cache=True` | 与 v3 γ=3 同一提案和采样规则；额外复用 Draft KV | 4 |
| `eagle_v2_joint_g3` | `speculative.pf_engine.decode`；`/root/autodl-tmp/gaussiangpt_eagle_pf_v2_20261019/training_extended/joint/selected.pt` | Joint P/F Draft，γ=3；Draft 读取 Target post-finalnorm hidden | 4 |
| `medusa_lowrank` | `/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare.py:decode_heads`；`step5_method_compare/medusa_lowrank/medusa_linear_heads_best.pt` | 独立 P1/F1/P2 低秩头共享现有 hook 的 RMSNorm(last-block output)；头间不消费已提议 token embedding | 4 |
| `hydra_conditional` | `/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare.py:decode_heads`；`step5_method_compare/hydra_width177/medusa_conditional_heads_best.pt` | 条件 P1→F1→P2 头；输入沿用现有 Medusa hook 的 RMSNorm(last-block output) | 4 |
| `d8_pf` | `draft_a.pf_feature_unified.decode`；`draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt` | 8 层 D8 只提议 Feature，Position 仍由共享 Ranker 提议；D8 使用自己的 Target 缓存状态 | 4 |
| `d8_joint` | `draft_a.feature_acceptance_eval.decode(k=2)`；同一 D8 权重 | Joint P/F 自回归 Draft，沿用原全词表 logits 与精确接受/残差采样 | 3 |
| `lfq_scratch_pf_bos` | `bench_lfq_scratch_pf_cow.block2_decode`；`profiling/feature_draft/lfq_scratch_30epoch/lfq_scratch_conditional_best.pt` | 只接受 `[EOS]`，直接使用原 BOS 解码器；FeatureStateDraft 输入 Target pre-finalnorm state，Ranker 提议 Position | 4 |

EAGLE Target cache 的工厂是 `speculative.pf_engine.target_cache`；PF MLP 与 D8-PF 通过 `draft_a.pf_ranker_eval.cache_for`；D8 Joint 通过 `draft_a.feature_acceptance_eval.cache_for`；Medusa/Hydra 通过本次调用创建的 `make_cache` 闭包。每个工厂只把当前 Target 路由到动态图 cache pool，草稿模型的 cache 仍由原 factory 创建。

EAGLE `return_hidden=True` 需要 Target post-finalnorm hidden。动态图直接返回这一状态，同时另把 pre-finalnorm block state 发布给 MLP hook。Medusa/Hydra 的旧 hook 对 last-block output 执行 `F.rms_norm` 并写入 `hook.value`；适配层传给 runtime 的 callable 对动态图发布的 pre-finalnorm state 执行同样变换。普通前向仍由原 hook 处理。EAGLE 的 returned hidden、Medusa/Hydra 的 hook hidden 与 MLP 的 pre-finalnorm state 分开处理。Medusa/Hydra 与 LFQ 的原解码器从返回列表中省略 EOS sentinel；适配层只在旧 stats 表明自然结束时补上 EOS，并保留 `raw_body_tokens`、`raw_eos_count` 与原始 `eos` 计数。

γ 深度对照使用一份 EAGLE v3 权重，因此 `eagle_v3_g3` 与 `eagle_v3_g5` 的模型权重相同。单独训练的 γ=5 权重保留在原位置，本轮不混入深度对照。

各 decoder 仍使用自己的 RNG 布局、EOS 规则、q/p 分布、接受规则、拒绝残差和 cache truncate。Target 优化不触碰草稿模型的 Attention 上下文。runner 负责同 seed 的普通/动态图输出及计数对照，并保存每次 runtime 的实际图调用和回退原因。

## 输入范围限制

LFQ scratch 的 `block2_decode` 从 `[BOS]` 自行开始，签名没有真实 `prompt` 输入。`lfq_scratch_pf_bos` 只在 `prompt=[EOS]` 时调用它，复用其原解码循环与概率参数，并把 Target cache 导入同一 runtime pool。该入口原始返回列表只含 P/F token；自然 EOS 通过 `stats["eos"]` 标记。适配层在返回 suffix 时补入 EOS sentinel，另存原始 P/F token 数与 EOS 计数。BOS-only 的少量 seed 结果只能描述此设置，不能代表场景前缀表现。

LFQ 在非空真实前缀下仍未接入。若要比较 prefix64/256，需要另写并逐项核验新的状态机；本轮保留旧权重与既有解码入口。

当前集合也不含 D8 以外的旧大模型草稿、EAGLE v1 标签或相近结构的多个长训节点。现有记录中没有能接入同一完整前缀解码的已确认 EAGLE v1 入口。D8-PF 与 D8 Joint 是同一 8 层权重下两种不同解码路线，保留为两种方法。

## 未知和适用范围

当前只完成入口适配与 CPU 语法验证，不代表 12 个方法都已在本 runner 实际加载，也不代表同优化 Target 下的速度结果。动态图复用范围由 `DynamicTargetGraph` 的 batch、query T、KV 容量桶与 read bucket 决定；未走图的 Target 调用会记入 fallback。Medusa/Hydra 继续使用旧 hook 对 last-block output 执行的 RMSNorm，动态图分支按相同运算生成 hook value。LFQ BOS-only、D8 及这些头方法的同优化速度和排序仍未知。
