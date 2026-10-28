# Round02 测量准备

本文件记录本地可复核的样本、微测次序和后续评价范围。当前没有运行 GPU，`results/expected_measurements.json` 只保存测量方案、已存在的样本标识和空结果栏位；它不含本轮新测量值。完整评价入口暂缓，等微测出现实际收益后再安排。

## 源码与可执行边界

R1 原始记录来自 `round01/cost_probe_raw/run01/collection/contexts.jsonl`，共 480 行。R1 源码镜像 `round01/source_current/` 有 13 个文件；Round02 更新镜像 `round02/source_before/` 现有 16 个文件，包含 Target、统一解码、Ranker、缓存辅助、4D RoPE、位置 token 工具、场景加载相关文件。权重、裁剪数据和运行环境不在这份源码目录内；主代理已确认后续微测使用既有完整工程环境。

两份本地 `nanochat_gpt.py` 都仍使用逐行稀疏输出计算：每一行分别建 token ID、`index_select` 权重、调用 `F.linear`，之后执行同一 softcap 与槽位遮罩；目前镜像里没有分组输出头参数或实现。Round02 计划中的 grouped 模式属于后续实现项，微测必须等该实现可调用后再开始。现有解码器按 `cache.get_pos() == len(prefix)-1` 要求缓存只包含已处理 token，末尾待处理 Feature token 仍可见。`typed_kinds(start, count)` 依绝对位置奇偶决定每行预测 Position 或 Feature。Target 的稀疏 4D RoPE 会从实际 token cache 读取位置历史；只设置缓存长度、没有真实 token 历史，不足以复现原坐标。

## 微测样本与覆盖

选择 4 个 R1 开发来源，各取初始前缀类别 64、256 的一条真实记录，共 8 条。每条均有四个真实 Target 输入 token；8 条记录的 P1 均已接受，F1 接受与拒绝各 4 条；F1 接受的 4 条中，P2 接受和拒绝各 2 条，F1 拒绝的 4 条均有残差 Feature。测量只重放固定 token，不调用抽样，因此这些分支标记只描述 R1 已存上下文，不是微测输出结论。`prefix_group` 表示原始 prompt 类别；样本经过先前生成后，实际缓存保留长度会高于该类别。

完整 `sample_id` 值保存在 `results/expected_measurements.json`。下表拆列显示同一标识里的来源、前缀类别、`seed_base`、`seed` 和轮次。

| 来源 ID | 前缀类 | `seed_base` | `seed` | R1 轮次 | `context_prefix` 长度 | KV 保留长度 | 四个 Target 输入 ID |
|---|---:|---:|---:|---:|---:|---:|---|
| `003ac11d-2abc-44f8-9836-4354e7dfa543` | 64 | 34702 | 35790 | 6 | 81 | 80 | `[11723, 2423, 9418, 2424]` |
| `003ac11d-2abc-44f8-9836-4354e7dfa543` | 256 | 34702 | 39054 | 0 | 257 | 256 | `[8395, 3686, 10483, 3687]` |
| `154620a2-e63e-474b-b950-b7cc287b0e43` | 64 | 34702 | 35790 | 0 | 65 | 64 | `[8646, 103, 9926, 105]` |
| `154620a2-e63e-474b-b950-b7cc287b0e43` | 256 | 34703 | 39055 | 0 | 257 | 256 | `[11726, 489, 11215, 490]` |
| `22f7de3c-4349-4dcc-9495-dde7bf690278` | 64 | 34703 | 35791 | 0 | 65 | 64 | `[11732, 405, 10748, 406]` |
| `22f7de3c-4349-4dcc-9495-dde7bf690278` | 256 | 34703 | 39055 | 0 | 257 | 256 | `[10569, 1502, 9748, 1522]` |
| `9a21209b-f836-4d36-b106-04495545481a` | 64 | 34703 | 35791 | 8 | 83 | 82 | `[11788, 284, 12084, 285]` |
| `9a21209b-f836-4d36-b106-04495545481a` | 256 | 34703 | 39055 | 0 | 257 | 256 | `[10824, 933, 11800, 943]` |

每条样本满足 `target_inputs[0] == context_prefix[-1]`。先用精确的 `context_prefix[:-1]` 建 Target KV，起始 keep 为 `len(context_prefix)-1`；最后一个可见 Feature 尚未进入 KV。每次短前向后用现有 `truncate(cache, keep, old_pos)` 返回同一 keep，记录还原耗时。缓存有效 token 前缀须保留原始 token ID，便于 RoPE 重建坐标；不能用填入 `cache_seqlens` 的方式替代前缀。

微测三个形状的输入、预测行如下。`logit_types` 表示每行所预测的 token 类别，所以 1-token 的输入是 Feature，第一行输出仍预测 P1。

| 输入行数 | 固定输入 | Target 输出行分别预测 |
|---:|---|---|
| 1 | `[pending F]` | P1 |
| 2 | `[pending F, P1]` | P1、F1 |
| 4 | `[pending F, P1, F1 draft, P2]` | P1、F1、P2、Target bonus F |

针对每个样本和形状，original 与 grouped 使用同一输入 ID、同一 keep、同一 Target 权重和 FP32。每种模式各预热 2 次，之后交错 12 对；每对都运行两种模式，首个模式按样本、形状和对次序轮换。总量为 24 个样本-形状单元、96 次预热短前向、576 次配对短前向，即 672 次短前向与 672 次 KV 还原。8 次真实长前缀预填充另计，总 Target 前向数为 680 次。此方案不做新 q2 条件采样或其他抽样。所选记录没有 P1 拒绝或自然 EOS；P1 拒绝后的奇数 start、EOS 路径、额外数值样本或独立 KV 数值核验均不包含在上述 680 次调用内。若加入任何一项，需将新增 Target 调用、缓存还原次数和耗时另列，不并入基础微测数。

Host 计时在同步后的 Target 调用前后取时，CUDA Event 只计该调用在当前 stream 上的执行；两项各自汇总，不相加。预填充、模型载入、预热、KV 还原分别记录。另记一次“前向加还原”的循环耗时，用来估算真实测量循环成本，但不并入前向主指标。原始与 grouped 的 logits、各行有效词表、hidden state 差值从已有前向结果计算，不增加 Target 调用；任何额外数值样本或缓存恢复调用都需另行计数。

按形状输出 Host 与 CUDA Event 的中位数、同一上下文配对比、来源内中位数及来源等权汇总。4 个来源的样本量只适合筛选微小实现收益，不能支撑完整解码速度结论。B阶段总用时仍受 900 秒新授权上限约束，包含加载、预热、前缀预填充、计时和数值核验；Round02 全部阶段的新授权共 7200 秒，按 900+1800+4500 秒分段使用，不挪用上一轮额度。计划软参考为 4-token Target 块 Host 成对中位数至少降时 8%，或逐阶段数据能说明较大同步成本已消除；这一条只决定是否继续工程验证，不证明达到 2×。

## 达到 2× 所需的节省幅度

历史来源等权速度比为 1.7125×，到 2.00× 需 PF 总时间再降约 14.375%。R1 插装分项估计 Target 占 PF 时间约 89.8%；若其他阶段不变，Target 总段需降约 16.01%。这是规模估算，完整评价必须用同轮 Target 与 W1 实测重算。

计划里的 8% 软参考针对 4-token Target 块 Host 成对中位数的降时。即使假设它覆盖了全部 Target 时间，对 PF 总时间的估算影响也只有约 7.184%，历史速度比约到 1.845×。4-token 块只是 Target 工作的一部分，实际端到端收益不能从此算式推出。只有微测显示真实净收益后，才进入完整解码接入。

## 同轮完整评价矩阵

C阶段按同一 GPU、Target/Feature FP32、同轮 Target 权重、同一采样设定比较四种方法；W1 使用现有 step12000 MLP，temperature/top-p 为 0.9/0.9：

| 方法 | 说明 |
|---|---|
| `target` | Target-only，现有输出头 |
| `w1` | 现有 W1，现有输出头 |
| `pf_typed` | W1，分组类型感知输出头 |
| `target_typed` | Target-only，使用同一分组输出头以分开估计通用输出计算节省 |

主评价为 10 个开发来源 × 前缀 64/256 × 种子 35711/35712 × 4 方法 × 256 新 token，共 160 次运行；每方法先预热 16 token，运行次序轮换。若自然 EOS 提前发生，保留真实输出长度，不补生成。记录完整 wall time、实际输出数、`target_calls`、`target_input_tokens`、Ranker/MLP 调用数和 EOS。以来源为重采样单位汇总来源等权速度比和区间。额外 `target_typed` 的 10 来源 × 两前缀 × 两种子共 40 次运行，与主表分开报告。

D阶段固定窗口主表使用 30 个 `confirmation_t` 候选来源、前缀 64/256、种子 47111/47112、`target`/`w1`/`pf_typed`、每次 512 新 token，共 360 次运行。自然 EOS 与三维范围另列：从 `confirmation_t` 原清单顺序预先取前 8 个来源、两种前缀、种子 47111、三个方法，最多 4096 新 token，共至多 48 次。达到长度上限仍无 EOS 时标注 `censored=true` 并报告实际长度，不写成自然 EOS。

自然 EOS/三维记录须包含结束时间、实际长度、末尾 EOS ID 12096、Position rank 单调合法性、VQ 解码、Gaussian 参数有限性、scene exporter 结果；可用的历史几何或渲染指标另行并列。完整生成耗时与固定窗口耗时分开汇总，输出长度不相同的自然完成运行不作为等工作量的速度比较。

## 本地来源清单与使用范围

本地可供后续 D阶段优先考虑的是 `docs/project/artifacts/gap_selection_20261017/G5_source_manifest.json` 的 `confirmation_t` 30 个 scene ID，逐项列入 `results/expected_measurements.json`。`G5_source_audit.json` 记录该集合与 D1、此前 60 个已分配来源没有交集；`G5_report.md` 记录实验在 D1 后停止，T 来源没有参与那次速度测量。它们尚未用于 Round02 方法选择。审计同时指出 Target 训练来源清单不含完整 scene ID，所以与预训练来源是否重合仍未知。此处只确认了本地清单记录，没有读取远端裁剪文件或权重；正式运行前仍由主代理在实际工程环境核对样本可用性。

`docs/project/artifacts/gap_selection_20261017/source_inventory/scene_splits.json` 另有 30 个 `final_test` scene ID，与开发集合交集为 0。`docs/project/artifacts/stage_f_full3d/analysis/test_scenes.csv` 含同一 30 个 scene 的 90 行历史结果，覆盖 `target_only`、`p_only`、`step12000`。这批来源已经参与 Stage E/F 和后续历史工作，适合复核旧结果，不应称作本轮全新来源。G5 的 10 个 D1 来源也已用于那次方法比较，不作为独立确认集。

Round02 计划未要求此时实现完整评价 CLI；微测若未证明有用，就保留负结果，不扩展工程工作量。
