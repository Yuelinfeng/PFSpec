# Round01 调度事件轨迹

## 模拟边界

[`schedule_trace.py`](../../../../../patches/pf2x_round01/schedule_trace.py) 以标准库模拟源码控制流、可见输出、Target 输入、KV 截短和 Ranker pending 状态。它只验证构造路径的控制流与缓存记账；不验证模型 token-ID、实际 Ranker 数值单调性、4D RoPE 数值或采样概率。它不调用 Target、不载入模型、不访问 GPU，也不测时。结果在 [`results/schedule_traces.jsonl`](results/schedule_traces.jsonl)，每行是一条包含逐事件状态的静态轨迹。

轨迹以完整前缀 `[BOS,P0,F0]` 起步；其 `get_pos=2`，KV 只含 `[BOS,P0]`。每次提交后都满足 `get_pos=len(prefix)-1`。`@R1`、`@R2` 是提议轮次标识：首轮候选为 `P1@R1`、`P2@R1`，下一轮同一 Position 槽的新候选会标作 `P2@R2`，之后可见 `P3@R2`。因此被清除的 `P2@R1` 与重新提出的 `P2@R2` 是两个事件符号。Feature 候选也带相同轮次标识，例如 `F1a@R1`、`F1b@R1`、`F2a@R2`。这些标签仅用于区分构造路径，不代表模型 token-ID，也不表示 Ranker 的实际数值排序。Target 输入候选块时，KV 会短暂进入试算状态；拒绝分支再按源码截短。可见末尾 Feature 仍未处理。终止 EOS 也不会进入 KV，且不会再读取 Feature。

目标事件路径覆盖 P1 拒绝、F1 首提议拒绝后第二候选接受和拒绝、P2 拒绝、全接受、尾部 K1 接受与拒绝、P1 残差 EOS、P2 残差 EOS。44 条轨迹中，36 条在非 EOS 条件下恰好生成预算长度；8 条以 EOS 提前结束；20 条经过预算尾部 K1。所有行都记录事件分支，因此这些计数代表模拟覆盖范围，不表示分支出现率。

## 同预算例子：6 个实际输出 token

条件为首轮 `P1@R1` 接受、`F1a@R1` 拒绝；W2 的具体分支取第二候选接受，后续 Target 提议均接受。首轮预览的 `P2@R1` 在 F1 拒绝时被清除；下轮新提议记为 `P2@R2`，后续依次标作 `P3@R2`、`P4@R3`。这些是事件标签，不是模型 token ID。

| 调度 | 逐次 Target 输入与 get_pos | 总 Target 输入 | 新输出数 | Ranker 选择次数 |
| --- | --- | ---: | ---: | ---: |
| W1-K2（源码路径） | `[F0,P1@R1,F1a@R1,P2@R1]`：2→6；截短到 4。随后 `[F1_RESIDUAL@R1,P2@R2,F2a@R2,P3@R2]`：4→8。 | 8，2 次 Forward | 6 | 4 |
| W1-short-after-F-reject（假设调度） | `[F0,P1@R1,F1a@R1,P2@R1]`：2→6；截短到 4。随后 `[F1_RESIDUAL@R1,P2@R2]`：4→6；预算剩 2，再 `[F2_TARGET@R2,P3@R3]`：6→8。 | 8，3 次 Forward | 6 | 4 |
| W2-lazy-short（假设调度） | 首轮同上；q2 接受 `F1b@R1` 后不增加输出。随后 `[F1b@R1,P2@R2]`：4→6；再 `[F2_TARGET@R2,P3@R3]`：6→8。 | 8，3 次 Forward | 6 | 4 |
| W2-sample-only-then-W1（假设调度） | 首轮同上；q2 接受 `F1b@R1` 后不增加输出。随后 `[F1b@R1,P2@R2,F2a@R2,P3@R2]`：4→8。 | 8，2 次 Forward | 6 | 4 |

首个 K2 输入四个 token 后，`P1@R1` 与 `F1a@R1` 的事件处理提交 `[P1@R1,F1_RESIDUAL@R1]`，然后 Target cache 从试算位置 6 截短至 4。W1-K2 下一块因此从可见末尾的 `F1_RESIDUAL@R1` 起步。W2-lazy-short 的短块从 q2 最终产生的单个 Feature 起步，另行执行 Target 前向来验证新提议的 `P2@R2`；Feature 二次接受没有独立增加输出 token。

若 W2 第二候选被拒绝，第二阶段从残差分布产生一个最终 Feature，标签记为 `F1_RESIDUAL_2@R1`。预算 6 的 W2-lazy-short 输入长度仍为 `[4,2,2]`，W2-sample-only-then-W1 仍为 `[4,4]`；输入 token 数、输出数与 Target 调用数不变。

## 同预算例子：8 个实际输出 token

仍采用首轮 `P1@R1` 接受、`F1a@R1` 拒绝；后续分支均接受。预算为 8 时，前缀条件相同，最后一轮在 K2 或尾部 K1 之间切换。首轮预览的 `P2@R1` 清除后，下轮新候选标作 `P2@R2`，该轮的第二个 Position 候选为 `P3@R2`。

| 调度 | 逐次 Target 输入与 get_pos | 输入长度 | 新输出数 | Ranker 选择次数 |
| --- | --- | --- | ---: | ---: |
| W1-K2（源码路径） | `[F0,P1@R1,F1a@R1,P2@R1]`：2→6；截短到 4。随后 `[F1_RESIDUAL@R1,P2@R2,F2a@R2,P3@R2]`：4→8；尾部 `[F3_TARGET_BONUS@R2,P4@R3]`：8→10。 | `[4,4,2]`，3 次 Forward | 8 | 5 |
| W1-short-after-F-reject（假设调度） | 首轮同上；短块 `[F1_RESIDUAL@R1,P2@R2]`：4→6；之后 `[F2_TARGET@R2,P3@R3,F3a@R3,P4@R3]`：6→10。 | `[4,2,4]`，3 次 Forward | 8 | 5 |
| W2-lazy-short（假设调度） | 首轮同上；q2 产生 `F1b@R1`，短块 `[F1b@R1,P2@R2]`：4→6；之后 `[F2_TARGET@R2,P3@R3,F3a@R3,P4@R3]`：6→10。 | `[4,2,4]`，3 次 Forward | 8 | 5 |
| W2-sample-only-then-W1（假设调度） | 首轮同上；q2 产生 `F1b@R1`，下一 W1 K2 块 `[F1b@R1,P2@R2,F2a@R2,P3@R2]`：4→8；尾部 `[F3_TARGET_BONUS@R2,P4@R3]`：8→10。 | `[4,4,2]`，3 次 Forward | 8 | 5 |

第二候选拒绝时，`F1b@R1` 对应 `F1_RESIDUAL_2@R1`，上述输入长度与输出数不变。该表只给出构造事件路径的计数，不比较不同算法的 token 字面结果，也不推断耗时。

## 其他拒绝与 EOS 分支

- `P1` 拒绝：四 token K2 试算后，源码回退到 `before+1`。非 EOS 残差 Position 再经一次单 token Target 前向抽 Feature；若残差为 EOS，则在完整旧前缀后停止。
- `P2` 拒绝：源码回退到 `before+3`，保留已接受的 P1/F1 分支；非 EOS 残差 Position 需要一次单 token Target 前向产生 Feature。
- 尾部 K1：预算剩 2 时，源码的第二 Position 候选为空，走 `[pending_F,P1]` 两 token 前向。Position 拒绝后，还需要 `[P1_RESIDUAL]` 一个 token 的 Target 前向；本模拟记录两种情况。
- 合法 EOS：模拟了 P1 与 P2 的 Target Position 分布产生 EOS。它计入 `generated_tokens`，但不计入内容 token；停止时 KV 只保留已处理上下文，不再推进到 EOS，也不产生后续 Feature。

## 结果解释

第二次 Feature 抽样无论被接受还是拒绝，最终都只提交一个 Feature。若只增加 q2 抽样及其接受判断、后面不增加任何 Target 验证，独立新增输出 token 数为 0。W2-sample-only-then-W1 的 6-token 例子仍是 2 次 Target Forward、8 个输入 token，和该条件下的 W1-K2 相同。

W1-short 与 W2-lazy-short 的 6-token 例子输入长度为 `[4,2,2]`，Target Forward 次数为 3；同路径 W1-K2 是 `[4,4]`，2 次。8-token 例子两种窗口次序都为 3 次 Forward、10 个输入 token，长度分别为 `[4,4,2]` 与 `[4,2,4]`。短窗口可能改变设备计算成本，结果文件没有 GPU 耗时，因此当前结果只反映调度窗口变化。

本模拟中，W1-K2 的接受与拒绝分支对应 `pf_feature_unified.py` 的当前路径。W1-short、W2-lazy-short、W2-sample-only-then-W1 都是待讨论的调度假设，不是生产解码实现。W1 在 F1 拒绝后会清除预览 P2 的 pending 状态；本模拟的两种 short 路径随后重新选择 P2 并计入 Ranker 调用。q2 使用从现有 `q1` 排除首候选并重归一化的可用方式；未模拟需要 MLP 原始 logits 的宽支持提案。

脚本断言每次提交后 `get_pos == len(prefix)-1`，并逐个比较 KV token 符号与 `prefix[:-1]`。`get_pos` 表示已消费的 KV 槽数；可见前缀长度还包含一个尚未消费的末尾 token，因此两者相差 1。EOS 是特殊末尾状态：它计入可见输出预算，不进入 KV，也不再触发后续 Feature。

生成命令：

```bash
python patches/pf2x_round01/schedule_trace.py
```

