# V3 记录字段可用性

## 结论

完整解码资料足以按来源、前缀、seed和方法复算输出、EOS、整次命中数、调用数、图事件与请求耗时。现存记录没有把每轮Ranker候选、Target概率、真实纠正位置、命中和恢复时长串成一条事件记录。离线诊断保存了固定前缀的候选分数与Target概率向量；它不能替代完整生成期间的逐轮资料。

## 新增宽度实验

[RESULTS.md](../position_widths_20261022/RESULTS.md)对应的`widths_main/records.jsonl`共有360行：30来源、prefix 64/256、seed 53101，Target、W1和备用宽度1至4各60行。每次rescue行记录备用节点、被拒绝、命中、恢复调用的整次总数，并按备用排名汇总命中；有完整输出token、EOS、逐轮接受长度、每请求图事件及首次建图耗时。没有每轮备用候选ID、Ranker分数、Target概率、实际纠正位置或命中索引。`accepted_lengths`和`round_generated_lengths`的逐轮顺序无法与备用判断对应。

宽度1至4全量解码记录显示，命中数为342、395、404、404；恢复调用为709、656、647、647。W1恢复为1051次。该证据能回答整次解码的宽度收益和成本，不能回答每次拒绝时各候选的精确质量。

`offline_mass.json`有39条固定上下文汇总：source、prefix、Ranker point、拒绝质量，以及每个宽度的备用ID、可用数量、质量和条件命中率。它不包含逐轮p向量。其输入[39上下文诊断](../spec_next_20261022/raw/position_overlap_run01.json)保存每行完整12098维`target_policy`、Ranker `ids/scores`和候选数1至4，因此可按固定候选ID查概率。样本为10来源、prefix64/256/512各10条、prefix1024九条，无在线seed和实际解码事件。

`widths_numeric_run03`另有2个定点上下文、1来源、prefix64/256及宽度1至4，共8行数值对照。保存六项候选顺序，其中仅一项是自然候选且带Ranker分数，其他位置为构造项；保存逐节点概率差异、支持集、hidden与KV差异摘要，但没有完整p向量、在线seed或完整KV数组。该组支持图分支数值比较，不能代表自然生成中的备用命中。

`widths_costs/width_records.jsonl`只有8行，覆盖一个实测来源的prefix64/256和宽度1至4。记录有单个实际Ranker候选的分数及Target候选概率；额外位置为构造候选，分数缺失。文件还含固定路径比较、新图事件、普通T1恢复时间和每类30个暖态计时样本。case_selection列出两个候选来源，但实际8条width_records只含一个来源。它是两前缀成本诊断，不能扩展为完整请求统计。

## 其他解码和数值资料

[confirmation_rescue_run01](../spec_next_20261022/raw/confirmation_rescue_run01/records.jsonl)含180行，rescue为60行、30来源、prefix64/256、seed51101；[repeat_01](../rescue_repeat_20261022/raw/repeat_01/records.jsonl)含240行，rescue为120行、30来源、两个新seed。两者都含请求身份、完整输出、EOS、整次备用命中/拒绝、恢复调用、图事件和同步外层耗时；都没有轮级候选/概率/纠正事件/恢复耗时。

[五节点数值验证](../spec_next_20261022/raw/candidates_verify_run01/numeric_report.json)是13个固定状态、8来源、seed48101的节点对照，覆盖prefix64五条、prefix256八条。它有候选数和数值误差标量，不含完整p向量或生成中的命中事件。该资料不代表完整30来源解码覆盖。

[PAPER_NEXT.md](../spec_next_20261022/PAPER_NEXT.md)的归因运行另有240行完整解码记录，覆盖30来源、prefix64/256、seed51101和target/优化Target/W1/rescue四种方法；每行保存整次输出、计数、EOS、图与耗时。它补充同轮性能归因，但没有增加逐轮候选和命中映射。端到端结果覆盖20输入、10来源、三臂共60次，另有60份Gaussian文件与360张PNG审阅结果；该证据支持请求耗时与所测输出保持分析，记录粒度仍为请求级。

源码在每轮运行中能取得Ranker ids/scores、Target分布、实际纠正P并判断命中。当前评测没有启用可选的`backup_diagnostics`；即使启用，源码只存首选与备用候选对应的概率，不存完整词表向量、实际纠正P、命中标记或恢复时间。因此现有缺项代表本次没有落盘这些字段，不能据此判断计算过程没有产生它们。

逐字段路径、行数、覆盖和限制见[EVIDENCE_AVAILABILITY.csv](EVIDENCE_AVAILABILITY.csv)。本次只读本机资料并新增本目录这两个说明文件；未启动GPU或访问远端。
