# confirmation_run01 独立数据复算

## 数据范围

直接读取本目录的 `plan.json`、`prompts.json`、`pipeline.json`、`records.jsonl` 与 `summary.json`。计划列出 30 个独立来源、prefix 64/256、seed 50101、512 token 上限，共 60 个 prompt、240 条记录。流水线标记 complete，实际也是 240/240 条、60/60 配对，每对含 `target`、`w1`、`fast_reuse`、`combined` 各一条。source、prefix、seed 与 prompt 表逐对一致；生成长度等于 token 列表长度，240 条都记录 `sequence_validated=true`。

计划标记来源范围为 `independent`，输入 crop 路径指向 `stage_e_final/final_test_crops.pt`。计时读取逐臂的 `outer_wall_seconds` 和生成 token 数，使用每条记录的 token 吞吐比；没有用输出长度比代替速度比。模型权重加载在计时外，图建立、缓存分配与重置、prefill、树验证、恢复、抽样和图捕获均计入外层耗时。

来源等权倍率先在每个来源内对配对吞吐比取 log 平均，再对 30 个来源等权平均并取指数。95% 区间按来源重采样 10,000 次，seed 402101。独立重算的来源倍率与本轮 `summary.json` 完全一致。

## 速度结果

| 比较 | 来源等权 GM | 来源重采样 95% 区间 | 配对数 / 来源数 |
|---|---:|---:|---:|
| W1 / Original Target | 5.0480× | [4.8019, 5.2347] | 60 / 30 |
| fast_reuse / Original Target | 4.9994× | [4.7506, 5.2103] | 60 / 30 |
| combined / Original Target | 5.0691× | [4.7708, 5.3073] | 60 / 30 |
| fast_reuse / W1 | 0.9904× | [0.9625, 1.0153] | 60 / 30 |
| combined / W1 | 1.0042× | [0.9752, 1.0308] | 60 / 30 |
| combined / fast_reuse | 1.0139× | [1.0009, 1.0226] | 60 / 30 |

`fast_reuse/W1` 与 `combined/W1` 的全样本区间都跨 1。`combined/fast_reuse` 的下界略高于 1，点估计约快 1.39%；同一 token 输出使这组成对差值可以直接衡量备用 Position 分支的耗时影响。该幅度较小，且 `combined/W1` 的整体区间仍跨 1。

| 方法 / W1 | 全部配对（60） | 四臂满 512（52） | 四臂无新增图（56） | 满 512 且无新增图（51） |
|---|---:|---:|---:|---:|
| fast_reuse | 0.9904× [0.9625, 1.0153] | 0.9962× [0.9764, 1.0161] | 0.9943× [0.9685, 1.0178] | 0.9939× [0.9743, 1.0139] |
| combined | 1.0042× [0.9752, 1.0308] | 1.0109× [0.9913, 1.0306] | 1.0124× [0.9854, 1.0371] | 1.0127× [0.9926, 1.0332] |

相同四种筛选直接比较 `combined/fast_reuse`：全部 60 对为 1.0139× [1.0009, 1.0226]；满 512 的 52 对为 1.0148× [1.0047, 1.0215]；无新增图的 56 对为 1.0182× [1.0146, 1.0218]；满 512 且无新增图的 51 对为 1.0189× [1.0153, 1.0224]。子集区间按来源等权重采样计算。各方法对 Original Target 的逐来源倍率保存在两份分析 JSON 中。

## Feature 接受率

此处 Feature 接受率按 `feature_accepted / feature_verified` 计算，分母是实际验证过的 Feature 候选数；没有用树验证轮数作分母。

| 方法 | Feature accepted | Feature verified | 微平均 |
|---|---:|---:|---:|
| W1 | 3,542 | 10,658 | 33.2333% |
| fast_reuse | 3,544 | 10,245 | 34.5925% |
| combined | 3,544 | 10,245 | 34.5925% |

按每个来源分别计算 accepted/verified 后等权平均，W1 为 33.6449%，fast_reuse 与 combined 均为 34.5947%，差值 +0.9497 个百分点。30 个来源中 20 个上升、10 个下降；来源重采样 95% 区间为 [−1.6660, +3.5001] 个百分点，涵盖 0。接受率提高尚未带来清楚的 fast_reuse/W1 速度提升。

每来源差值如下，单位为百分点：

| 来源 ID 前缀 | fast_reuse − W1 |
|---|---:|
| 1af21af5 | +6.90 |
| 1be3ff59 | −17.67 |
| 1ecf4f03 | +0.07 |
| 2aabe005 | +3.01 |
| 44858e7f | −2.80 |
| 4fa56715 | −5.61 |
| 5a0f02be | +9.27 |
| 702a3f08 | +8.00 |
| 78f54b4e | +4.45 |
| 876631f3 | −4.78 |
| 888f5f5c | +5.61 |
| 89d895eb | +6.66 |
| a33765d2 | −1.76 |
| a38c6892 | +1.19 |
| a791d489 | −5.59 |
| a855d99f | +11.36 |
| abbcc79a | +1.44 |
| aee07a67 | −10.98 |
| b7f55c1e | +1.69 |
| b9d74420 | +0.20 |
| c28a2210 | −0.53 |
| c69e0c9d | +0.03 |
| cbbaa3ef | +5.27 |
| d091a071 | +7.23 |
| d0e0f18e | +15.15 |
| ebb55c6a | +1.95 |
| f081f259 | +0.11 |
| f2469055 | −9.60 |
| f32ae7af | +10.48 |
| f5696e0a | −12.26 |

## combined 与 fast_reuse 的逐对结果

两臂在 60/60 配对中的 token 序列、生成长度、EOS、`accepted_lengths`、树验证次数、Position 接受计数、Feature 尝试和接受计数均一致。逐项不同的共享计数是资源成本字段：`target_calls`、`target_input_tokens`、`target_tree_nodes`、`recovery_calls`；备用 Position 专属计数也只在 combined 中出现。

`fast_reuse recovery_calls − combined recovery_calls` 等于 combined 的 `position_rescue_hits`，60/60 配对均成立，合计减少 380 次恢复（1,134 → 754），备用命中 380 次。Target calls 同样减少 380 次（12,455 → 12,075）。对应成本是额外 4,143 个 Target tree nodes 和 3,763 个 Target 输入 token。combined 记录 4,143 次备用节点尝试、命中 380 次，命中率约 9.17%。这些备用命中没有计入 Feature 接受数。

## 自然 EOS、图运行与解释边界

Original Target 有 4 条自然 EOS，输出长度为 123、407、511、295；W1 有 1 条，长度为 131；fast_reuse 与 combined 各有 3 条，长度为 111、269、43。各 EOS 输出尾部为计划中的 EOS id 12096。四臂共同满 512 的配对为 52/60；主速度表保留自然结束数据，另列满预算子集。

四臂共同无新增图配对为 56/60、覆盖 29 个来源；其中同时满 512 的有 51 对。无新增图子集按 `runtime_created`、正值 `graph_setup_ms` 和各 `*_delta.new_graph_events` 排除首次建立或新增图形状的行，包含 `rescue_runtime_delta`。三个 speculative 方法都有图回放；各自记录 60 次 `unsupported_query_tokens` 回退。原始首次建立成本仍包含在全部配对的外层计时中。

收到本轮 240 条结果并读出倍率后，主代理决定追加一组独立的纯备用 Position 对照：30 个来源、prefix 64/256、seed 51101、`target/w1/rescue` 共 180 条。该组不并入本轮统计，也不据本轮结果调整权重。后续目录的计划文字若沿用运行器默认来源标签，应另注明实际 crop 来自 `stage_e_final/final_test_crops.pt` 与这 30 个独立来源；原始 plan 文件保持原样。

## 分析文件

- `confirmation_analysis_target.json`：四臂以 Original Target 为分母的来源等权倍率、bootstrap 区间、逐来源倍率与输出长度对照。
- `confirmation_analysis_w1.json`：四臂以 W1 为分母的同类统计。
- 本文的 Feature 接受来源差值与 `combined/fast_reuse` 直接比率由原始 240 条记录独立重算。
