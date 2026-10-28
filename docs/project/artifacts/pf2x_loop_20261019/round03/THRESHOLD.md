# R1 调度计数与成本筛选

本页依据本地 R1 原始记录复算，未调用 Target 或模型。计算脚本为 [`threshold_cpu.py`](threshold_cpu.py)，结构化结果为 [`results/threshold.json`](results/threshold.json)。

## 已保存记录中的计数

输入为 R1 `w1_runs.jsonl`、`w1_profiles.jsonl`、`contexts.jsonl`：60 次运行、24,481 条分项记录、480 条上下文；覆盖 10 个开发场景，每场景 48 条上下文和 6 次运行。输出 3,781 个 token、1,471 个调度轮次，Target 总计 1,811 次前向、处理 15,664 个输入 token。

| 分支或计数 | 次数 | 说明 |
|---|---:|---|
| P1 提议 / 验证 / 接受 | 1,471 / 1,471 / 1,228 | 每轮先提议并验证一个 P1；尾轮只走这条分支。 |
| 四输入验证 | 1,421 | 每次输入 `[pending_f, P1, F1, P2]`，共 5,684 个输入 token。 |
| 两输入尾轮验证 | 50 | 输入 `[pending_f, P1]`，不提议 F1 或 P2，共 100 个输入 token。 |
| F1 验证 / 接受 / 拒绝 | 1,186 / 420 / 766 | 仅 P1 通过的分支继续验证 F1。条件接受率为 420/1,186 = **35.4132%**。 |
| P2 验证 / 接受 / 拒绝 | 420 / 382 / 38 | 仅 F1 通过的分支继续验证 P2；P2 拒绝时仍需 Target 补算一个 Feature。 |
| 一输入补算 | 280 | 拒绝分支按实际调用记录，共 280 个输入 token。 |
| 提示词 prefill | 60 | 30 个 64-token 前缀和 30 个 256-token 前缀，合计 9,600 个输入 token。 |
| Draft 前向 / Ranker 位置提议 | 1,421 / 2,892 | Profile 中 Ranker 辅助操作另计，见结果 JSON。 |

Target 总量由四输入验证 5,684、两输入尾轮验证 100、一输入补算 280、提示词 prefill 9,600 相加，得到 15,664 个输入 token。前向次数相应为 1,421 + 50 + 280 + 60 = **1,811**。原始运行统计与 Profile 汇总逐项相符。Profile 的 Ranker 阶段记录位置提议 2,892 次，另有前缀重放 60、状态分叉与提交各 1,421、清除待处理状态 767、提交位置 704；`ranker_calls` 只计位置提议调用。

## 完整轮、尾轮与 EOS

1,421 次四输入验证中有 1,420 个普通完整轮，另外一个轮次以 EOS 结束。50 个两输入尾轮各产出两个 token，共 100 个。普通完整轮合计产出 3,680 个 token，平均 **2.59155 token/轮**。这些轮次中 F1 接受 420 次，每次比两 token 的起始推进多产出两个 token，因此按普通完整轮计数得到 `2 + 2 × 420/1,420 = 2.59155`。这只描述当前记录中普通完整轮的计数关系。

唯一 EOS 运行来自场景 `83c802b6-a1c0-4753-b38a-490bafe0ddd3`、256-token 前缀、seed 39053。整次输出 5 个 token；末轮使用四输入验证，P1 拒绝后残差采到 EOS，只输出一个末尾 token。EOS ID 为 12096。它不带 Feature，不能按普通两/四 token 轮计入公式。全部 1,471 轮混合后的平均为 `3,781/1,471 = 2.57036 token/轮`，与普通完整轮均值不同。

实现依据 [`round02/source_before/draft_a/pf_feature_unified.py`](../round02/source_before/draft_a/pf_feature_unified.py) 的 `_decode`：源码 336–345 行为每条运行建立 Target KV、以 `prompt[:-1]` 做初始 prefill 并重放 Ranker 前缀状态；Round02 与 Round03 的该文件逐字节相同。R1 前缀为 64 或 256，进入循环时 Ranker 状态已存在，因此未走空状态的首次 P/F 引导分支。开轮处有 `cache.get_pos() == len(prefix) - 1` 条件（347–349 行）：`prefix` 含有可见但尚未送入 Target 的末尾 token，KV 已消费部分在逻辑上对应 `prefix[:-1]`。这解释了为什么 Target 输入从 `pending_f` 开始，不能把可见 token 数直接当成已消费 KV 数。

完整分支先提议 P1，并在预算足够时推进 Ranker 临时状态来提议 P2；随后产生 F1，再用一次四输入 Target 前向验证 `[pending_f, P1, F1, P2]`。P1 拒绝时从 P 分布残差抽取实际位置并恢复 KV；非 EOS 时追加一次单 token Target 前向以取得后续 Feature。P1 接受而 F1 拒绝时，按 Feature 残差采样并恢复 KV，不需要额外 Target 前向。F1 通过时继续判断 P2；P2 拒绝会恢复 KV并补算一个 Feature，普通输出长度仍为四。若剩余预算只有两个 token，代码只验证 `[pending_f, P1]`，之后直接生成 Feature，因此该短尾不具备 F1 验证。初始 prefill、Ranker 前缀重放和每轮生成统计是不同阶段；`round_generated_lengths` 不包含任何 prefix0 引导输出。

Target KV 在进入每轮时保存已处理的输入位置；`prefix` 则包括目前可见的最后一个待处理 token。拒绝后源码按实际采样结果调用截短，并在必要时用 Target 计算下一步分布，所以可见输出长度和 KV 已处理长度会在一步内错开。EOS 是特殊末尾状态：`round02/source_before/draft_a/pf_ranker_eval.py` 的 `validate_sequence`（15–26 行）要求 EOS 是唯一末尾标记，且它前面已有完整 P/F 对；本轮真实 EOS 位于被验证的 Position 槽，不生成配对 Feature。

## 2× 成本筛选参考

全轮平均和 F1 覆盖率用于与 Round03 任务书的筛选公式一致：

```text
L0 = 3781 / 1471 = 2.5703603
v  = 1186 / 1471 = 0.8062542
L_req(0) = L0 × (2 / 1.7125) = 3.0018806
L_req(c) = L_req(0) × (1 + c / 22.57)
alpha_req(c) = [L_req(c) - 2] / (2 × v)
```

其中 `1.7125` 是历史 Stage E 的来源等权速度比参考；`22.57 ms/轮` 是 R1 Profile 中 20.28 ms Target 加 2.29 ms 其它成本的参考。表中的“零成本参照推进量”固定取 `c=0`；计入新增成本后，所需平均推进量按 `L_req(c)` 上升：

| 每轮新增 Draft 成本 | 零成本参照推进量 `L_req(0)` | 计入成本后的所需推进量 `L_req(c)` | 所需 F1 条件接受率 |
|---:|---:|---:|---:|
| 0 ms | 3.00188 | 3.00188 | 62.1318% |
| 0.25 ms | 3.00188 | 3.03513 | 64.1939% |
| 0.5 ms | 3.00188 | 3.06838 | 66.2559% |
| 1.0 ms | 3.00188 | 3.13488 | 70.3800% |

这些是跨批次成本假设下的 CPU 估算。历史 1.7125 不是与 R1 同轮取得的对照；新提案会改变 P1/F1/P2 分支比例、补算次数、EOS 位置和各轮长度。以上数字没有逐条重放新策略，也不能作为严格速度上限。实际收益只能由同轮完整计时确定；当前结果不支持 2× 结论。

## 复算

在仓库根目录运行：

```sh
python3 docs/project/artifacts/pf2x_loop_20261019/round03/threshold_cpu.py
```

脚本仅使用 Python 标准库，读取已有 JSONL 和 `source_manifest.json`，以断言核对实际计数后写入 `results/threshold.json`。不加载模型，也不访问 GPU、网络或远端。
