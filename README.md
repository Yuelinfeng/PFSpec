# GaussianGPT 本地项目交接区

这里保存 GaussianGPT 的项目理解、实验状态和 AI 交接文档。当前目录包含文档、补丁及 `.codex/remote-source/` 早期源码参考；完整运行工程、权重和原始性能产物仍以 Autodl 为准。

当前工作状态（2026-10-08）：暂缓新增实验，完成阶段性整理。详细汇报见 [阶段工作汇报](docs/project/STAGE_REPORT_2026-10-08.md)，覆盖 S2—S13、技术演进、实验总表、资产位置与遗留问题。

面向研究汇报，优先阅读新的 [GaussianGPT 推理加速阶段性研究汇报](docs/project/GAUSSIANGPT_SD_RESEARCH_REPORT_2026-10-08.md)。正文按七章串联背景、瓶颈、完整 P/F 负结果、两条优化路线、算法实现与下一步，纳入两类投机的理论上限及两张已有数据重绘图。AR 内部 96.68% 有导出数据依据；99.57% 的完整流程占比仍待原始计时支持。

两个主要 P-only 方法的实际算法见 [D8 与结构化 Position 实现汇报](docs/project/POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)：已通过 Computer Use 阅读远端源码，说明候选概率、Target 双 token 验证、拒绝后采样、缓存和调用入口。

新 AI 请先读：

1. [`.codex/AI_START_HERE.md`](.codex/AI_START_HERE.md)：交接入口、当前任务和阅读顺序；
2. [`docs/project/PROJECT_STATUS.md`](docs/project/PROJECT_STATUS.md)：自动生成的当前状态快照；
3. [`docs/project/ARCHITECTURE.md`](docs/project/ARCHITECTURE.md)：原版数据流和本地创新线；
4. [`docs/project/EXPERIMENTS.md`](docs/project/EXPERIMENTS.md)：S2、S3 和 speculative decoding 证据；
5. [`docs/project/RESEARCH_SOLUTIONS.md`](docs/project/RESEARCH_SOLUTIONS.md)：论文检索结果和候选路线；
6. [`docs/project/LOGIC_VALIDATION.md`](docs/project/LOGIC_VALIDATION.md)：方案逻辑审查、淘汰结论和验证门槛；
7. [`docs/project/OPERATIONS.md`](docs/project/OPERATIONS.md)：运行、复现和维护规则。

历史实验记录已覆盖 S13a/b 训练和 S13b 真实联合评价。[联合 P/F 草稿训练计划](docs/project/JOINT_DRAFT_TRAINING_PLAN.md)前 16 节保留此前方案，第 17 节记录结果后的 S14 建议。本轮远端文件列表已出现 S14 评价与 overlap 目录，内容尚未纳入本地汇总；不要沿用旧材料中“S14 尚未实施”作为远端当前事实。新增实验仍暂缓。

同步源码或实验产物到本目录后，运行：

```bash
python .codex/scripts/update_status.py
```

状态刷新脚本只更新 `docs/project/PROJECT_STATUS.md`，不会覆盖架构和实验说明。
