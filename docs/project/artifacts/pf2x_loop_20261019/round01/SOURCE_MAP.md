# Round01 调度源码映射

## 使用范围

本轮调度分析使用只读副本 [`source_current/`](source_current/) 最初的11份源码。主代理之后另加入 `bench_fast_hgb.py` 和 `eval_lean_ranker_cpu.py`，目录当前共13份；本文已有路径和行号仍对应最初11份源码，没有随新增文件调整。本轮只读取本地副本，没有访问远端工程；远端当前文件是否与副本完全一致，仍待远端任务结束后按获准范围确认。

## 解码与调用关系

| 功能 | 文件与行号 | 实际行为 |
| --- | --- | --- |
| Feature MLP 提议 | `source_current/draft_a/pf_feature_unified.py:112-160` | `MLPAdapter.propose()` 在 `127-153` 行调用 Feature 网络，按温度 0.9、top-p 0.9 构造完整 4096 项 `q` 并只返回 `q`。`rewind()` 在 `155-157` 行为空操作，因为该适配器没有独立 KV。 |
| 调度入口与计数 | `source_current/draft_a/pf_feature_unified.py:237-285` | `decode()` 进入 `_decode()`；局部 `tf()` 在 `278-285` 行累加 Target Forward 次数与输入 token 数，并按 KV 起点设置 Position/Feature 类型。 |
| 轮次起始状态 | `source_current/draft_a/pf_feature_unified.py:333-376` | 当前完整前缀进入循环；`348-349` 行要求 `TargetKV.get_pos() == len(prefix)-1`。只有首个 Position 提议存在且剩余预算至少 4 token 时，才复制 Ranker 状态并预提议第二个 Position。 |
| K2 目标块 | `source_current/draft_a/pf_feature_unified.py:413-423` | `MLPAdapter.propose()` 产生 `q`，从 `q` 抽 `F1a`；Target 一次输入 `[pending_F,P1,F1a,P2]`。这是四个输入 token 的 Target Forward。 |
| Position 接受 | `source_current/draft_a/pf_feature_unified.py:295-297,424-442` | `P1` 依据 Target 分布作点提议接受；接受后把预览 Ranker 状态作为当前状态。 |
| Feature 接受与残差 | `source_current/draft_a/pf_feature_unified.py:299-304,443-453` | 首个 Feature 被拒绝后从 `(p_F-q)_+` 抽一个最终 Feature，回退到 `before+2`，提交 `[P1,F1*]`。该轮的 P2 预览状态在 `450` 行由 `drop_eos()` 清除。 |
| 第二 Position 接受/拒绝 | `source_current/draft_a/pf_feature_unified.py:455-480` | `F1a` 接受后才判断 `P2`。接受时由已有 logits 抽 Target Feature bonus；拒绝时回退到 `before+3`，非 EOS 时以正确的 `P2` 再做一次单 token Target Forward 来抽 Feature。 |
| Target 调用和隐藏状态 | `source_current/bench_gaussiangpt_pf_block2_cow.py:35-58` | `CaptureLastBlock.run()` 把输入送入 GPT，返回 logits 与最后一层 block state；MLP 使用已提交 Position 对应的隐藏状态。 |
| KV 截短及残差 | `source_current/bench_gaussiangpt_pf_block2_cow.py:61-91` | `truncate()` 设定新的 `cache_seqlens` 并清掉 token cache 尾段；Feature 与 Position 的残差采样分别见 `71-86` 行。KV 张量本身不在此函数逐槽清零，后续输入会覆盖保留长度后的槽位。 |
| Ranker 状态复制 | `source_current/bench_gaussiangpt_pf_block2_cow.py:103-137` | `fork_position_state()` 要求源 Ranker 有 pending 提议，复制预测器及可变计数，再供预览 P1 后的 P2。 |
| Position 候选状态 | `source_current/gpt_position_spec_decode_smoke.py:34-112,122-145` | `CausalRanker.propose()` 设置 pending，`finalize()` 接受实际 Position，`drop_eos()` 清掉 pending；`choose_position()` 从 Lean 候选中选一个 Position。 |
| 前缀回放和类型 | `source_current/draft_a/pf_ranker_eval.py:29-40` | `position_state()` 从前缀中的 Position 重建 Ranker 状态；`typed_kinds(start,count)` 依据绝对 KV 起点的奇偶序列给每个输入位置标记其要预测的类型。 |
| KV 建立及策略 | `source_current/draft_a/feature_acceptance_eval.py:61-88` | `cache_for()` 建立 batch=1、FP32 的 `KVCache`；`policy()` 调用温度 0.9、top-p 0.9 的 Target 策略；`residual()` 使用非负差值归一化。`pf_ranker_eval.py:11` 从该模块导入这些函数。 |
| 稀疏 RoPE 输入 | `source_current/model/gpt/nanochat_gpt.py:542-609` | `_build_sparse_rope_indices()` 根据缓存中的 token 和本次输入重建 Position/Feature 槽位；稀疏布局会把实际 KV 前缀与当前输入拼接后再生成索引。 |
| KV 位置与 token | `source_current/model/gpt/engine.py:10-96` | `KVCache.get_pos()` 读取 `cache_seqlens`；`store_tokens()` 写入 token cache，`get_tokens()` 返回给 RoPE 重建使用。 |
| EOS 序列约束 | `source_current/draft_a/pf_ranker_eval.py:15-26` 及 `pf_feature_unified.py:399-410,428-437,473-480` | EOS 只作为完整 P/F 对后的终止符；若 Position 分布抽到 EOS，循环清除 pending 状态并停止，不再读取 Feature。 |

调用顺序可概括为：

```text
decode -> _decode -> choose_position -> CausalRanker.propose
                       |-> MLPAdapter.propose -> q(F1a)
                       |-> tf -> CaptureLastBlock.run -> GPT / KVCache
                       |-> 接受或残差采样 -> rewind -> truncate
                       `-> Ranker finalize 或 drop_eos
```

## prefix 与 KV 的边界

令可见前缀长度为 `L`。源码要求 `get_pos=L-1`：最后一个可见 token 已在 `prefix` 中，尚未作为下一次 Target 输入消费。这里的 `get_pos` 是已消费 KV 槽数量，不是可见 token 数，也不是最后一个 token 的零起始下标。

例：初始可见序列为 `[BOS,P0,F0]`，`get_pos=2`，KV 只含 `[BOS,P0]`。K2 输入 `[F0,P1,F1a,P2]` 后，试算 KV 暂到 6。若 F1a 被拒绝，截短到 4，再把 `[P1,F1*]` 加入可见前缀，状态重新满足 `get_pos=len(prefix)-1`。`F1*` 是可见 token，留给下一次 Target 调用处理。

EOS 同样是只追加到可见前缀末尾的特殊终止状态，Target KV 不消费 EOS；停止前仍满足长度关系。状态模拟在每次提交后同时断言 `get_pos == len(prefix)-1` 和 KV token 序列等于 `prefix[:-1]`。

## 网页计划中需要按源码理解的事项

1. 原方案中的 W1 F 拒绝路径符合当前代码：回退到 `before+2`，提交一个由 Feature 残差分布产生的实际 Feature，再进入下一轮。
2. 当前源码在该路径清除预览的 P2 pending 状态。W1-short 与 W2-lazy-short 若要接着做 `[F1*,P2]`，需由调度重新提议 P2；也可以另行设计保留预览状态的调度，但那属于源码当前路径以外的状态处理。本轮模拟采用重新提议，并计入 Ranker 调用。
3. `MLPAdapter.propose()` 只返回 top-p 处理后的 `q`，没有把网络原始 logits 交给调用方。由 `q` 排除首候选并重新归一化的 q2 可以从现有返回值构造；基于同次原始 logits 的宽支持 q2 暂时没有可用接口。
4. 第二候选被残差接受时，它成为当前唯一的实际 Feature，替代第一次拒绝后的残差 Feature。没有任何新 Position/Feature 后缀因这次抽样而得到 Target 验证，所以第二次接受本身新增真实输出 token 数为 0。
5. 两 token Target 窗口需要单独的调度选择。现有 `_decode()` 在 F1 拒绝后继续循环；剩余预算至少 4 时会再次选择 K2。任何 W1-short 或 W2-lazy-short 轨迹都是调度假设，不是现有实现路径。
6. 本轮没有运行 Target、GPU 或任何正式速度评价。模拟中的候选符号和事件条件用于逐步记账，不能作为真实采样、吞吐或概率结果。

