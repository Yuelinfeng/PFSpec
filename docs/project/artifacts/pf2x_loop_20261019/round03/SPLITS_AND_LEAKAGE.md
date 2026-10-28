# Round03 来源划分与既有使用范围

本轮 R1 CPU 汇总只使用十个既有开发场景。完整 ID 和 6/2/2 分区见 [`source_manifest.json`](source_manifest.json)；seed 32003 已记录，字典序分区不依赖随机抽样。

| 分区 | 场景数 | R1 上下文行 | R1 运行记录 |
|---|---:|---:|---:|
| fit | 6 | 288 | 36 |
| tune | 2 | 96 | 12 |
| hold | 2 | 96 | 12 |

R1 `contexts.jsonl` 的 480 条记录均标为 `development`，每个场景 48 条；R1 六十次运行每个场景六条。fit 是前六个排序 ID，tune 是接下来的两个，hold 为最后两个。拆分单位是场景，因而同一场景的不同前缀、seed 和上下文不会落到不同分区。

## 历史用途与解释范围

这些十个场景等同本地 `gap_selection_20261017/source_inventory/scene_splits.json` 的 development 集。Round03 原任务书在划分前已报告基于同一批 480 条上下文的 q 校准与六面邻居混合结果。因此本次 fit/tune/hold 是之后这条 CPU 分析内固定的留源诊断；hold 仍是有既往探索记录的开发来源，不可称为全新、从未用于选择的独立来源。hold 只做一次诊断汇总，不扩展至其它场景。

本地旧正式来源清单可区分为：

- `source_inventory/scene_splits.json` 记录 10 个 development、30 个 final_test、20 个 adaptation_pool 场景。R1 十场景与 development 完全一致，与 final_test、adaptation_pool 均无交集。
- `stage_f_full3d/analysis/test_scenes.csv` 有 90 行、30 个不同场景，ID 与 final_test 的 30 个场景完全一致；R1 与该集合无交集。`research_evidence_20261016/stage_e_modes.csv` 是方法汇总表，不含可用于逐场景交集计算的 ID，因此本地资料不足以逐个核对 Stage E 来源。
- `gap_selection_20261017/G5_source_manifest.json` 保留 10 个 D1 来源和 30 个 `confirmation_t` 候选。G5 来源审计中 D1 与 T 不重叠，所选场景也没有落入此前已分配的 development/final_test/adaptation 集；两组与 R1 及旧 Stage F 场景均无交集。G5 最终记录写明在进入 T 来源阶段前停止，故该批候选未形成 G5 T 来源评价结果。这些记录不能证明它们从未影响其它研究中的方案选择；可用 Target 训练场景清单也不完整，预训练来源交集未知。

Round03 使用的 R1 集合不包括 G5 的 D1 或 `confirmation_t`。本页不将 T 候选称为绝对新来源，也不据本地列表判断远端场景文件当前可用。若未来实际进入来源确认，主代理需在当时核对真实清单与文件可用性；本轮 CPU 计算不读取这些场景数据。

## 既有空间条件路线

本地 `round03/source_before/feature_spatial_delta.py` 显示，早先的位移条件模块接收 `h(Pprev)`、`Fprev`、`Pprev`、当前候选 `Pcur`，并拼接两位置的 xyz 与 `Δxyz`；它没有六面已提交邻居的 Feature 汇总。对应本地 `prior_spatial_summary.json` 记录 3,020 条验证行，起始重合率 32.373623%；完成六个 epoch 时重合率降至 32.217491%，最佳轮为 epoch 0。故项目已有一条位移条件化尝试，且结果未改善。Round03 提出的六面历史邻居信息与之不同；目前还没有它能提高留源重合率或在线速度的结果。

## 数据边界

本次阈值脚本只从 R1 `w1_runs.jsonl`、`w1_profiles.jsonl`、`contexts.jsonl` 读取。划分统计仅验证每个分区的场景数、上下文数和运行记录数，不重新抽样来源、不加载 Target 或训练资料。hold 的身份不改变其开发来源历史，也不能支持 2× 或独立确认结论。
