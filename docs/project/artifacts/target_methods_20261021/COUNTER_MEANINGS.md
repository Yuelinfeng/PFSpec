# Target 方法计数口径

日期：2026-10-21。本文把 `method_adapters.py` 接入的十二种方法，与原解码函数写入 `stats` 的字段对应起来。计数用于解释每次运行做了哪些工作；耗时和完整输出长度仍决定实际速度。

## 通用字段

| 字段 | 含义 |
| --- | --- |
| `generated_tokens` | 适配器返回的规范 suffix 长度，包含自然生成的 EOS。`records.jsonl` 中 `output_tokens` 的长度与此相同。 |
| `proposed` | 已从草稿分布或确定性 Ranker 取得的候选 token 数。 |
| `verified` | 进入 Target 接受判断的候选 token 数；顺序验证遇到首个拒绝后，后续候选不再计入此项。 |
| `accepted` | 被 Target 接受的候选 token 数。Target 自己采样的 bonus、拒绝后的残差样本和 fallback 采样不计入。 |
| `unverified` | `proposed - verified`，表示本轮曾经提出但没走到 Target 接受判断的候选。 |
| `acceptance_verified` | `accepted / verified`；分母为实际进入接受判断的候选。分母为零时值为 `null`。 |
| `acceptance_proposed` | `accepted / proposed`；分母还包含首个拒绝之后留下的未验证候选。 |
| `target_calls` | Target `forward` 函数的调用次数，包含 prefill、候选验证和恢复调用。一次调用可以输入多个 token。 |
| `target_input_tokens` | 所有 Target 调用输入宽度之和；比 `target_calls` 更接近 Target 实际处理的 token 槽数，仍不表示 FLOPs 或耗时。 |
| `draft_calls` | 对该方法草稿模型的调用次数，具体含义随模型与缓存策略而变。它不等于候选数。 |
| `draft_input_tokens` | 草稿模型处理的输入 token 槽数，仅部分实现提供；不同结构下每个槽的计算量不同。 |
| `ranker_calls` | Position Ranker 被调用的次数。一次调用可能产生一个 Position 提案，也可能用于预演、状态更新或恢复。 |
| `rounds` | 解码器自行定义的候选循环次数；各方法的循环边界不同。 |

`accepted_lengths` 保存每轮接受的候选数，不含 Target bonus、残差或 fallback 产生的 token。带有 `round_generated_lengths` 的实现另存每轮实际输出长度，两者不能互换。

对于具有 `position`、`feature`、`eos` 三类统计的解码器，原代码逐类累加，满足：

```text
proposed = position_proposed + feature_proposed + eos_proposed
verified = position_verified + feature_verified + eos_verified
accepted = position_accepted + feature_accepted + eos_accepted
```

`position_acceptance_verified` 与 `feature_acceptance_verified` 分别按各自的 `*_accepted / *_verified` 计算。它们描述已验证候选的局部结果，不能单独代表一轮能生成多少 token。

## 各方法的字段对应

### Target、Ranker-only、PF-MLP 与 D8-PF

这四种方法使用 `draft_a.pf_ranker_eval.decode` 或 `draft_a.pf_feature_unified.decode` 的 P/F 计数结构。聚合式按 P、F、EOS 三类求和。

`target` 使用 `k=0`，没有草稿候选，因此 `proposed`、`verified`、`accepted` 都为零，接受率为空值。输出由 Target 逐 token 采样。

`ranker_only` 使用 `k=1`。Ranker 提出 Position；Target 生成 Feature。`position_*` 记录 Ranker 候选，`feature_*` 为零。`position_fallbacks` 记录 Ranker 无法给出下一位置时改走 Target 采样的次数，它不属于拒绝。

`pf_mlp` 与 `d8_pf` 使用宽度最多为三项的 P1/F1/P2 候选块。`position_*` 汇总 P1、P2，`feature_*` 记录 F1。`proposal_width=3` 是每个完整块的最大候选数；预算边界、拒绝和 EOS 会缩短实际块。

`pf_mlp` 的 `draft_calls` 和 `draft_input_tokens` 记录 Feature MLP 的调用及输入位置；Position 仍由 Ranker 提议。`d8_pf` 的字段名相同，草稿部分来自八层 D8 GPT。D8 的前缀状态、增量状态和缓存追赶都会贡献 `draft_input_tokens`，所以它不能和 MLP 的输入 token 数当作相同计算量。

### D8-Joint

`d8_joint` 使用 `draft_a.feature_acceptance_eval.decode(k=2)`。它每轮最多提出两个 token，类型按 P/F 交替，并额外区分 EOS。此处 `k=2` 的宽度为两个候选；PF-MLP 的 `k=2` 则对应 P1/F1/P2、最多三个候选。

总量按通用三类公式计算。`target_input_tokens` 累加 Target 每次 forward 的输入宽度。`draft_calls` 只数调用次数，原函数没有提供 `draft_input_tokens`，不能用它估计草稿 token 槽数。

### EAGLE v2、v3 γ=3、v3 γ=5 与 v4 Draft-cache 复用

EAGLE 的 `speculative.pf_engine.decode` 提供 `position_*`、`feature_*`，未提供聚合式 `proposed`、`verified`、`accepted`。本轮的派生总数为：

```text
proposed = position_proposed + feature_proposed
verified = position_verified + feature_verified
accepted = position_accepted + feature_accepted
accepted_per_target_call = accepted / target_calls
```

EAGLE 把 Position 槽的 EOS sentinel 也记入 `position_*`；它没有单独的 `eos_proposed` 计数。`reject_position`、`reject_feature` 记录首个被拒绝提案的类别，`reject_by_slot` 进一步按 P1/F1/P2 等槽位统计。`bonus_position` 和 `bonus_feature` 记录完整接受后由 Target 采样的额外 token。

`target_calls` 含 prefill、验证和恢复 forward；`verify_calls` 只计验证阶段的 Target 调用，`recovery_calls` 只计恢复调用。字段 `progress_per_verification` 定义为 `generated_tokens / verify_calls`，不等同于 `generated_tokens / target_calls`。

`draft_calls` 和 `draft_input_tokens` 会计入草稿状态预填、逐步提案及缓存追赶；v4 的 `draft_commit_calls`、`draft_commit_tokens` 单列持久 Draft KV 追赶工作。缓存复用可改变草稿调用与输入量，同时保持提案接受统计相同。

### Medusa LowRank 与 Hydra Conditional

`step5_method_compare.decode_heads` 返回稀疏的 `Counter`。Counter 中缺失的计数字段按零解释。适配器记录的阶段字段可派生为：

```text
proposed = p1_checked + f1_proposed + p2_proposed
verified = p1_checked + f1_checked + p2_checked
accepted = p1_accepted + f1_accepted + p2_accepted
accepted_per_target_call = accepted / target_calls
```

`p1_checked` 是每轮 P1 进入 Target 判断的次数；每轮可进入短 P1 分支或长 P1/F1/P2 分支。`f1_proposed` 与 `p2_proposed` 只在长分支产生；P1 拒绝后，F1/P2 即使已由头部给出，也不会计入 `f1_checked`、`p2_checked`。F1 拒绝后，P2 不会进入接受判断。

`long_blocks` 计长 P1/F1/P2 提案块，`k1_steps` 计短 P1 分支。`head_forward` 不是神经网络 kernel 数：Medusa 一次原始多头调用写一次；Hydra 也每次生成 P1 写一次，后续条件 F1/P2 头调用不另增该值。它们不适合跨头结构比较模型成本。

Target 调用数可由 `forward_prefill + forward_single + forward_pair + forward_block3 + forward_block4` 求得。若需要输入 token 槽数，可按前缀长度 `L = len(prompt) - 1` 计算：

```text
target_input_tokens = L * forward_prefill + forward_single
                    + 2 * forward_pair + 3 * forward_block3
                    + 4 * forward_block4
```

这里的宽度来自 `decode_heads` 实际送入 Target 的 `[pending_f]`、`[pending_f, p1]`、三 token EOS 块或四 token 完整块。

### LFQ Scratch PF BOS

`lfq_scratch_pf_bos` 仅调用 `bench_lfq_scratch_pf_cow.block2_decode`，输入限制为 `[EOS]`。该方法仅在lfq_bos_run01中实测。18次记录的原/图重放未一致，补充同历史数值复核也测出支持集变化，计数字段可读但不能据此给出已验证速度结论。

原函数用下列字段记录两级 Position、Feature 候选：

```text
proposed = position_1_proposals + feature_proposals + position_2_proposals
verified = position_1_proposals + feature_checked + position_2_checked
accepted = position_1_accepted + feature_accepted + position_2_accepted
accepted_per_target_call = accepted / target_calls
```

P1 Ranker 没有候选时走 `position_fallback`，该次 Target 采样不属于提案。`feature_checked` 只在 P1 接受后增加，`position_2_checked` 只在 P1 与 F1 都接受后增加。

LFQ 提供 `position_1_mass_sum`、`feature_alpha_sum` 和 `position_2_mass_sum`。平均期望接受概率分别是 `position_1_mass_sum / position_1_proposals`、`feature_alpha_sum / feature_checked`、`position_2_mass_sum / position_2_checked`。这些值来自候选分布；实际抽样结果分别按 `position_1_accepted / position_1_proposals`、`feature_accepted / feature_checked`、`position_2_accepted / position_2_checked` 计算。

LFQ 没有显式 `target_calls`。可从 `forward_single + forward_pair + forward_block4` 得到调用次数，从 `forward_single + 2 * forward_pair + 4 * forward_block4` 得到输入 token 槽数。`eos` 是原 Counter 的自然终止次数；适配器据此把 EOS sentinel 加入规范 suffix，并保存 `raw_body_tokens` 与 `raw_eos_count`。

## 哪些计数不可直接互比

- `accepted` 只有在明确统一候选总数公式后才可横比。通用 P/F 解码器把 EOS 单独记作 `eos_*`；EAGLE 把它放在 Position 槽；Medusa/Hydra 用 P1/P2 字段；LFQ 的 EOS 是终止事件计数，未作为候选分类。因此跨方法总接受数应使用上面的派生公式。
- `feature_acceptance_verified`、EAGLE 的 `feature_accepted / feature_verified`、Medusa/Hydra 的 `f1_accepted / f1_checked`、LFQ 的 `feature_accepted / feature_checked` 都是条件接受率，但进入分母的路径受更早 Position 接受与否影响。它们适合解释对应阶段，不能单独代表完整解码收益。
- `position_acceptance_verified` 也受提案规则影响。Ranker 的 Position 通常是点质量提案；EAGLE Joint、D8-Joint、Medusa 与 Hydra 使用神经分布提出 Position。即使名称相同，q 分布与拒绝残差路线不同。
- `rounds`、`accepted_lengths`、`draft_calls`、`head_forward` 和 `target_calls` 的循环边界或调用粒度不同。一次 Target 调用可验证一至多个候选；一次 `head_forward` 可生成多个头输出；Draft KV 的预填与追赶也会增加调用量。
- `accepted / target_calls` 是接受候选数对 Target forward 次数的比例。它不计 bonus、残差、fallback 与 Target-only 输出；不同 query 宽度也被当作一次调用。可同时报告 `generated_tokens / target_calls` 与 Target 输入槽数，最终速度仍以外层同步解码时间为准。
- `draft_input_tokens` 有助于比较同一模型及相同缓存策略内的输入量。MLP、八层 D8、EAGLE 的状态网络和 Medusa/Hydra 头部计算不同，输入槽数不能当作统一 FLOPs。

动态图运行字段属于 Target 执行层：`runtime_delta.calls.graph_forward_Tn` 记录每种 query 宽度的图执行次数，`original_forward` 记录走原始 Target `forward` 的调用，`fallbacks` 按原因计数。它们与解码器 `stats.target_calls` 是不同层的观测；`new_graph_events` 和 `new_cache_events` 还记录图准备及 KV 容量。优化版的外层时间包含首次图准备，跨方法解释时要一并查看。

## 已有记录与限制

`all_methods_smoke_run01` 覆盖十一种方法，单一开发场景、prefix64/256、一个 seed、32-token 预算、original/optimized 两种执行，共 44 条记录；LFQ BOS 不在其中。审计状态为 passed，44 条输出都达到 32 token，未出现自然 EOS。44 条记录的通用聚合计数、EAGLE typed totals、Medusa/Hydra 派生分支数关系均成立。这个小样本用于确认字段可读和公式关系，不用于方法速度排序，也没有实际覆盖 EOS 补入分支。

### paired_pilot_run01 实测计数

该轮有 132 条记录：11 种方法、6 个提示（3 个开发场景 × prefix64/256）、原始与优化执行各一份，预算 256、一个 seed。整轮审计通过，每种方法的 6 组配对都保持输出 token 和计数相同。计数表按每种方法 6 条记录汇总后再计算比例；接受率的分子、分母及每种方法的字段来源见 [counter_summary.json](paired_pilot_run01/counter_summary.json)，由纯 CPU 脚本 [counter_summary.py](../../../../patches/target_methods_20261021/counter_summary.py) 从 records 复算。该样本来自开发场景，LFQ BOS 未纳入。

表中 `全部接受` 写作 `accepted/verified（accepted/proposed）`；`生成/Target` 为生成 token 数除 Target forward 次数，`Target槽/生成` 为输入槽数除生成 token 数。`Draft/生成` 对 EAGLE、MLP、D8 表示草稿调用数；Medusa/Hydra 列的是 head forward 数。`Draft槽/生成` 对应原计数字段，缺字段时写作“无字段”，不估算。

| 方法 | 分类型接受率（accepted/verified） | 全部接受率（accepted/verified；accepted/proposed） | 生成/Target | Target槽/生成 | Draft 或 head /生成 | Draft槽/生成 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Target | 无候选 | — | 1.00 | 1.66 | 0.00 | 0.00 |
| Ranker-only | P 89.8% | 89.8%；89.8% | 1.80 | 1.68 | Draft 0.00 | 0.00 |
| PF-MLP | P 88.5%；F 35.3% | 67.3%；49.0% | 2.25 | 2.21 | Draft 0.38 | 0.38 |
| D8-PF | P 90.8%；F 57.2% | 78.3%；63.2% | 2.63 | 1.99 | Draft 0.33 | 1.65 |
| D8-Joint | P 68.8%；F 62.7% | 65.7%；55.4% | 2.09 | 2.05 | Draft 1.15 | 无字段 |
| EAGLE v2 Joint γ3 | P 1.3%；F 0.0% | 1.3%；0.4% | 1.00 | 3.12 | Draft 1.99 | 144.50 |
| EAGLE v3 γ3 | P 94.0%；F 24.9% | 64.1%；46.6% | 2.28 | 2.27 | Draft 1.61 | 117.35 |
| EAGLE v3 γ5 | P 92.1%；F 22.4% | 59.4%；28.1% | 2.25 | 3.06 | Draft 2.39 | 115.25 |
| EAGLE v4 reuse γ3 | P 94.0%；F 24.9% | 64.1%；46.6% | 2.28 | 2.27 | Draft 1.65 | 3.45 |
| Medusa LowRank | P1 0.5%；F1 25.0% (4 verified)；P2 0% (1 verified) | 0.6%；0.2% | 1.00 | 3.11 | Head 0.50 | 无字段 |
| Hydra Conditional | P1 1.2%；F1 12.5% (8 verified)；P2 0% (1 verified) | 1.3%；0.4% | 1.00 | 3.11 | Head 0.50 | 无字段 |

EAGLE v3 γ3 的 Target 槽数是按 Position/Feature 提案、验证调用和恢复调用派生；它与动态图记录的 query 宽度总和一致。EAGLE v4 reuse γ3 与 v3 γ3 的候选数及接受率相同，Draft 输入槽数从 117.35 降至 3.45/生成 token，但计数下降本身不能推出相同幅度的耗时变化。

这轮原始与优化执行下，相对各自 Target 的端到端速度比也有变化：Target 执行优化为 2.80× [2.12, 3.22]；EAGLE v3 γ3 从 1.36× 变为 0.81×，γ5 从 1.20× 变为 0.64×，v4 reuse γ3 从 1.35× 变为 0.79×，v2 Joint γ3 从 0.75× 变为 0.55×；Medusa LowRank 从 0.84× 变为 0.78×，Hydra Conditional 从 0.83× 变为 0.70×。这些是 3 个开发场景、每法 6 组配对的来源等权几何均值，不是独立来源结果。

计数能解释变化方向，不能分摊实际耗时。EAGLE v3 γ3 和 v4 reuse γ3 每生成 token 有约 2.28 个 Target 输入槽；前者还处理约 117 个 Draft 输入槽。γ5 的 Target 槽数升至 3.06，F 接受率约 22%。v2 Joint 的总接受率仅 1.3%，生成量几乎等于 Target 调用数。Medusa 与 Hydra 的 P1 接受率分别约 0.5% 和 1.2%，整块候选很少被接受，仍约每生成一个 token 调用一次 Target，同时处理约 3.1 个 Target 输入槽。Target 执行优化提高了 Target 路径的速度后，这些方法仍保留的候选生成、草稿计算、拒绝判断和较宽验证块对总时长更显眼；本轮没有单独测出它们各自占用的时间比例，不能据上述槽数推算 CPU 或 GPU 耗时份额。

## 十来源主比较计数

optimized_main_run01共有220条记录，仅优化执行。全部计数按20个提示汇总，接受率随场景、前缀、随机数及输出历史变化；不能把这里与256-token小试的接受率差解释为运行优化改变了模型能力。相同条件的旧/新接受计数一致证据来自paired_pilot_run01。

| 方法 | Feature接受/实际验证 | 生成token/Target调用 | Target输入槽/生成token |
| --- | ---: | ---: | ---: |
| pf_mlp | 1272/3518=36.16% | 2.423 | 1.848 |
| eagle_v3_g3 | 709/4008=17.69% | 2.094 | 2.079 |
| eagle_v3_g5 | 937/4534=20.67% | 2.213 | 2.799 |
| eagle_v4_reuse_g3 | 709/4008=17.69% | 2.094 | 2.079 |
| medusa_lowrank | 5/45=11.11% | 1.003 | 2.818 |
| hydra_conditional | 1/18=5.56% | 1.000 | 2.822 |
| d8_pf | 1797/2976=60.38% | 2.704 | 1.651 |

D8-PF在此样本Feature接受率高于MLP、Target调用也更少，实际速度仍低于MLP，说明接受率与调用数不能代替总时长。EAGLE v3/v4的候选与接受计数完全相同；v4节省Draft输入量，但总速度更低。Medusa/Hydra各有一次自然EOS，raw_body_tokens=4、raw_eos_count=1、规范suffix长度5，已实际覆盖补入EOS的输出路径；其余19条各512token。派生计数和字段见[主测计数](optimized_main_run01/counter_summary.json)。本轮没有分项耗时测量，不据此分配CPU/GPU时间比例。

## 源码依据

- 适配入口、方法 ID 与 EOS 输出规范化：`patches/target_methods_20261021/method_adapters.py`。
- P-only、Ranker + MLP、D8-PF 计数：远端 `/root/workspace/GaussianGPT/draft_a/pf_ranker_eval.py:decode`、`draft_a/pf_feature_unified.py:decode`。
- P/F/EOS 聚合类型、分母和 D8-Joint 计数：远端 `/root/workspace/GaussianGPT/draft_a/feature_acceptance_eval.py:KINDS`、`_kind`、`_add_rates`、`decode`。
- EAGLE typed counts、cache commit 与 bonus：远端 `/root/workspace/GaussianGPT/speculative/pf_engine.py:decode`。
- Medusa/Hydra P1/F1/P2 与 Target forward 分支：远端 `/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare.py:decode_heads`。
- LFQ 两级 P/F/P 计数：远端 `/root/workspace/GaussianGPT/bench_lfq_scratch_pf_cow.py:block2_decode`。
- Target 图调用与 fallback 计数：`patches/target_methods_20261021/runtime/dynamic_target_graph.py:DynamicTargetGraph.stats`。
- 当前样本与公式复算来源：`docs/project/artifacts/target_methods_20261021/all_methods_smoke_run01/{plan.json,records.jsonl,audit.json}`。
- 配对试跑的汇总与输入记录：`docs/project/artifacts/target_methods_20261021/paired_pilot_run01/{counter_summary.json,records.jsonl,audit.json,summary.json}`。
