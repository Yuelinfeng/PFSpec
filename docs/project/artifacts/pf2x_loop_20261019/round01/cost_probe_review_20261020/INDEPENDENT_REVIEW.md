# PF-2× cost_probe run01 独立复算

本报告只读使用 `cost_probe_raw/run01/` 的 JSON、JSONL 与原始概率向量，以及同目录下复制的分析脚本。复算概率时用 Python 标准库解析 `vectors.npz` 中三组 `float32 (480, 4096)` 数组；未安装依赖，也未调用 GPU、远端或网页。

## 阶段和样本计数

`pipeline.json` 的 collect、narrow/wide measure、narrow/wide analyze 五阶段均为 `complete`。由起止时间复算总时长为 193.622 秒。采样阶段有 60 次 W1 解码，10 个开发来源 × 2 个前缀 × 3 个 seed_base，每次预算 64 tokens；得到 480 个唯一上下文，20 个来源×前缀格各 24 个，无采样短缺。上下文都来自真实接受的 P1；其中 F1a 实际接受 171 条、拒绝 309 条。480 条是每格限额抽样，不等于 1,186 个实际接受 P1 状态的全集。

`w1_runs.jsonl` 的 60 行中，`output_tokens` 长度均等于 `generated_tokens`。59 次生成满 64 tokens，1 次在 5 tokens 后输出 token 12096 并记录 `eos=true`；总输出为 3,781 tokens。该条 `eos_proposed`、`eos_verified`、`eos_accepted` 都为零，EOS 专项计数与终止标记的关系无法由现有记录确定。W1 原始总计为 1,421 个 P1 提议、1,186 接受、235 拒绝；接受 P1 后有 420 个 F1a 接受、766 个 F1a 拒绝。Feature、Position 分项的 proposed/verified/accepted 相加与总计吻合。

每个 q2 模式从 480 个上下文中各选 48 个合法的 P1 接受、F1a 拒绝分支，覆盖全部 20 格：12 格各取 2 条，8 格各取 3 条。两模式都无 q2 不支持上下文。分支候选 `q2_candidate_probability` 与从原始概率向量复算的最大差为 narrow 0、wide 3.5×10⁻¹⁸；每条分支 P2 概率也与同一 `branch_2` Target 计时记录完全一致。

## 概率复算

`narrow` 以该上下文实际 q1 为 q2 底分布；`wide` 以同次 MLP 原始 logits 经 temperature 0.9 softmax 为底分布，使用完整词表且没有再做 top-p 截断。两者都将首候选 F1a 的质量置零后重新归一化。wide 没有用 Target 分布构造 q2。测量脚本再从该 q2 实际抽样 F1b，并用 Target 在 `[F1b, P2]` 分支上的前向结果读取这个分支自己的 `P(P2)`；它没有沿用 F1a 分支的 P2 概率。

从原始向量重算的均值与两个分析 JSON 相符：A1=0.364396、R1=0.635604。`β̄` 按首次拒绝候选的 `δ(a)=(q1(a)−p(a))₊` 加权，narrow 为 0.152197，wide 为 0.143099；对应第二候选接受失败质量为 0.847803、0.856901。两种 q2 的 R1 加权非法质量均为零。

| q2 模式 | `P(P2 接受 | F1b 接受)` | `P(F1b 与 P2 均接受 | 首次拒绝)` | H：每轮有利事件概率及来源95%区间 |
| --- | ---: | ---: | ---: |
| narrow | 0.867837 | 0.176487 | 0.085644 [0.036622, 0.139545] |
| wide | 0.936961 | 0.127776 | 0.057112 [0.009886, 0.110844] |

H 包含 P1 接受、F1a 拒绝、F1b 接受、P2 接受四个条件。逐分支重算使用 `α₂=min(r1[b]/q2[b|a],1)`，再乘该分支真实的 `P(P2)`。分析摘要中的 `P(P2 接受 | F1b 接受)`按 20 个来源×前缀格的分支均值合并；每格分支数为 2 或 3，因此它不是把 48 行直接拼起来计算的比例。H 也先在每个格内用 `P1 接受概率 × R1 加权合法质量 × mean(α₂·P(P2))` 计算，再对每个来源的两个前缀取平均，最后对 10 个来源等权平均。

P1 接受且 F1a 拒绝的估计率按 20 格等权为 0.527017；按各格 P1 提议数加权为 0.532976。H 使用每个来源的两个前缀等权，再对来源等权，因此应用时需沿用相同的来源和前缀权重口径。

来源区间从上述 10 个来源均值有放回抽取 2,000 次，种子 20261020。区间只反映来源间变化，不含每格 24 个上下文的抽样误差，也不含每格 2–3 个分支的有限样本误差。虽然来源和前缀覆盖齐全，H 仍是小分支样本的点估计；不少格的分支联合权重为零。

## 成本与可作结论

W1 每轮平均生成 2.591346 tokens。`main_4` 的 Target host wall 均值为 narrow 18.230 ms、wide 17.527 ms；诊断 `branch_2` 本身均值为 17.560 ms、17.015 ms。若假设每个 H 事件省去一次 `main_4`，且分支计算、KV 恢复及调度成本全为零，则 H × `main_4` ÷ 每轮 tokens 得到约 0.602 ms/token（narrow）和 0.386 ms/token（wide）。这只是有限样本下的乐观点算例，既不是实测端到端节省，也不是严格上界；分支运行和概率估计的不确定性都未被它覆盖。

W1 采样本身没有插入额外 Target 前向；两种测量的 Target 调用都标为独立诊断。`analysis/narrow.json`、`analysis/wide.json` 的 feasibility 均为 `pending`，原因是未提供同口径配对 W1 与 Target 吞吐率。当前资料没有同轮 Target-only 对照，不能从这些数据推出相对 Target 的速度比或宣称达到 2×。下一步若要作速度判断，需补同窗口、同来源权重、同计时范围的 Target-only 与 W1 实测，并把分支及 KV 维护成本纳入实际解码流程。

主要依据：`cost_probe_raw/run01/pipeline.json`、`collection/metadata.json`、`collection/contexts.jsonl`、`collection/w1_runs.jsonl`、`collection/vectors.npz`、`measurement/{narrow,wide}/branch_rows.jsonl`、`measurement/{narrow,wide}/shape_timings.jsonl`、`analysis/{narrow,wide}.json`；概率与加权口径见 `source/analyze_cost_probe.py:73–102,174–199,638–689,774–859,1030–1073`，分支采样及自身 P2 读取见 `source/measure_target_shapes.py:51–83,489–506,643–669`，W1 样本保存见 `source/collect_pq.py:113–166,653–730`。
