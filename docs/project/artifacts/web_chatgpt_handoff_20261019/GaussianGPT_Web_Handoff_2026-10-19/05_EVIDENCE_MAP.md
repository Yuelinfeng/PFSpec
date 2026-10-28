# 主张、证据与位置

以下链接均指向包内文件。逐次数据用于独立复算；已保存的历史验证记录不等于本次重新运行 GPU 实验。原始报告以引文副本保留。

| 编号 | 内容 | 首选证据 | 补充 |
| --- | --- | --- | --- |
| E01 | 当前最佳方法与早期方法来源 | [真实证据对应](evidence/reports/RESEARCH_EVIDENCE_AUDIT_2026-10-16.md) | [方法历史](evidence/reports/DRAFT_METHODS_HISTORY.md) |
| E02 | 阶段 E 五模式数字 | [五模式 CSV](evidence/data/official_ef/stage_e_modes.csv) | [2160 行记录](evidence/data/official_ef/online.json)、[来源组汇总](evidence/data/official_ef/multiscene_summary.json) |
| E03 | 阶段 F 自然 EOS、时间与质量范围 | [完整摘要](evidence/data/stage_f/summary.json) | [270 行真实补全](evidence/data/stage_f/test_samples.csv)、[90 行 BOS](evidence/data/stage_f/bos_samples.csv) |
| E04 | 已有 E/F 复算与数值验证 | [10月16日复算](evidence/data/official_ef/verification.json) | [正式数值记录](evidence/data/official_ef/numerical_validation.json) |
| E05 | MLP step12000 来源 | [训练选择](evidence/data/mlp_training/selected.json) | [续训历史](evidence/data/mlp_training/history.json)、[新旧在线结果](evidence/data/mlp_training/online_summary.json) |
| E06 | 第二轮 Head 与输入条件对照 | [执行记录](evidence/reports/ROUND2_EXECUTION_2026-10-17.md) | [首seed在线摘要](evidence/data/round2/summary.json)、[逐次记录](evidence/data/round2/records.jsonl) |
| E07 | 三训练 seed 的范围 | [seed20261018](evidence/data/round2/B1_training_summary_20261018.json) | [seed20261019](evidence/data/round2/B1_training_summary_20261019.json)、[seed20261020](evidence/data/round2/B1_training_summary_20261020.json)；后两者没有完整在线速度 |
| E08 | 候选成本 A1–A3 | [报告](evidence/reports/CANDIDATE_COST_REPORT.md) | [分析 JSON](evidence/data/candidate_cost/analysis.json) |
| E09 | Feature 条件收益 | [报告](evidence/reports/FEATURE_VALUE_REPORT.md) | [分析 JSON](evidence/data/feature_value/analysis.json)、[输出验证](evidence/data/feature_value/verification.json) |
| E10 | Gap 新开发来源结果 | [G5 报告](evidence/reports/G5_report.md) | [摘要](evidence/data/gap/G5_summary.json)、[512逐次](evidence/data/gap/online_512.jsonl)、[2048逐次](evidence/data/gap/long_2048.jsonl) |
| E11 | EAGLE v2 实现、训练与比较 | [报告](evidence/reports/EAGLE_PF_V2_2026-10-19.md) | [108次记录](evidence/data/eagle_v2/records.json)、[摘要](evidence/data/eagle_v2/summary.json)、[训练来源](evidence/data/eagle_v2/data_manifest.json) |
| E12 | v3 条件蒸馏与递推 | [报告](evidence/reports/FEATURE_EAGLE_V3_2026-10-19.md) | [在线摘要](evidence/data/eagle_v3/online_summary.json)、[108次记录](evidence/data/eagle_v3/online_records.json)、[权重与选模](evidence/data/eagle_v3/selected_weights.json) |
| E13 | γ=5 定向训练无可靠额外收益 | [报告](evidence/reports/FEATURE_EAGLE_GAMMA5_2026-10-19.md) | [144次记录](evidence/data/gamma5/records.json)、[摘要](evidence/data/gamma5/summary.json)、[同长度分析](evidence/data/gamma5/analysis.json) |
| E14 | v4 缓存复用 | [报告](evidence/reports/FEATURE_EAGLE_V4_2026-10-19.md) | [原审阅说明](evidence/data/eagle_v4/REVIEW.md)、[54行CSV](evidence/data/eagle_v4/metrics.csv)；before/after完整JSON未收入 |
| E15 | 实际源码与时点 | [最佳PF主循环](source/motivation_20261016/draft_a/pf_feature_unified.py) | [EAGLE主循环](source/eagle_20261019/speculative/pf_engine.py)、[完整来源清单](source_inventory.csv) |

候选成本和 Feature 价值只附分析与验证摘要，未包含全部逐轮大日志。原报告中的远端路径可用于后续最少补取；当前包不能独立复算这两项的每一个中间值。

下一步方案没有对应“已完成”证据，其状态明确为研究提议。算法细节、新颖性和最终收益均需另行论证。
