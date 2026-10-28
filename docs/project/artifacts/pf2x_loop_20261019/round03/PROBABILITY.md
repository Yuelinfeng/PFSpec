# Round03 离线概率筛选

本次工具只读取 R1 的有限开发记录，在 CPU 上比较 Feature 提案分布与 Target 概率分布的重合度。它不会加载 Target 权重，也不会运行解码或 GPU 任务。全部480条记录此前已向网页计划提供；网页计划报告过样本内温度0.75、top-p 0.90，以及邻域混合权重0.10的观察值。因此，来源切分只隔离本脚本里的拟合和调参流程，不能代表此前未见来源。

输入位于 `docs/project/artifacts/pf2x_loop_20261019/round01/cost_probe_raw/run01/collection/`：`contexts.jsonl` 有480行，`vectors.npz` 保存 `p_f`、`q1`、`raw_logits` 三个 `(480,4096)` 数组，另有60条 `w1_runs.jsonl` 和24,481条 `w1_profiles.jsonl`。脚本会核对行数、向量索引、样本 ID、每个来源48行、矩阵形状以及10个来源清单。原数组以 float32 保存；计算时转为 float64，保留每个存储值，并逐行报告原始概率和误差。计算重合度前按各行自身总和归一化，不设正概率截断值。

来源切分严格读取 `round03/source_manifest.json`：按完整 scene ID 字典序排序，前6个来源拟合、随后2个调参、末2个来源只做一次诊断，共288/96/96行。清单记录 seed 32003；这个划分不使用随机顺序。网页计划已看过全部来源与样本内对照结果，所以末两源的区间只能说明本次拟合/调参流程的变化，无法代表完整研究过程的独立复核。

评分采用每行 `α = Σ min(p_i, q_i)`，并同时给出按行平均与按来源等权平均。来源区间用来源为单位、seed 32003 的有放回重采样，共10,000次。末两源的区间取值很粗，文档和结果会同时列出两源逐源数值与来源数。

提案校准只读取 `raw_logits`，遍历7个温度与5个 top-p 值；每个候选在两个调参来源上选择。简单邻域法先按行总和归一化 `q1`，再与已提交六面邻居 Feature 的经验分布按权重线性混合，遍历7个权重；没有可用邻居的行保持 `q1`。原始 `q1` 数值单独保留。温度与 top-p 规则遵循累计质量首次严格超过阈值时保留越界项。

空间条件探针使用 [pf_round03_feature.py](/Users/lief/Documents/ChatGPT/GaussianGPT/patches/pf2x_round01/pf_round03_feature.py) 的 `DirectionalResidualProbe`：固定 rank=4、neighbor_groups=8、logit_buckets=16、seed=32003，在6个拟合来源用 Target 的 `p_f` 作训练标签，训练100步。预测只读取 MLP 原始 logits、候选位置和已提交邻居 Feature。另拟合一个相同参数量、不读取邻居 ID 的 context-only 控制头。两种头各自在调参来源挑选残差倍率；只有这些选择完成后，才对末两源进行一次评分。输出保存全部行最终 `q`、源矩阵原值和拟合参数，便于复算逐行结果。

高熵与低熵分组使用 `q1` 熵，切分点取6个拟合来源的中位数。结果同时列出调参源与诊断源的分组规模和重合度，不根据诊断结果再次调参。每个选定方法会记录480行批处理的 CPU 经过时间和每行毫秒数；邻居查询单列，另记录用 T=0.9/top-p=0.9 从 `raw_logits` 重建 `q` 与保存的 `q1` 的最大绝对差。这里的时间来自 CPU/Python 批处理，不代表 GPU 在线新增 Draft 延迟。停止建议按预先给出的判据计算：空间头相对 `q1` 的末两源等权提升不足0.03，或提升没有同时超过简单混合与等参数控制头时，建议停止空间训练准备；否则仅提交主代理审议。

成本筛选逐次汇总60条 R1 W1 记录中的输出 Token、轮数、已验证 Feature 与接受 Feature，并按旧分析脚本的阶段定义汇总 Target / 其它 Host 时间。参考式为 `L_req = L0 × (2 / 1.7125) × (1 + extra_ms / (T_target + T_other))`、`α_req = (L_req − 2) / (2v)`，其中 `L0` 是 R1 W1 每轮平均输出数，`v` 是每轮已验证 Feature 数。它复用更早批次的1.7125速度比和本批 R1 成本数据，只供筛选：P2、EOS、Ranker、接受路径变化均未重放，因此不能作为严格成本界或速度结果。

本机标准库自测命令：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 patches/pf2x_round01/pf_round03_offline.py --self-test
```

真实 R1 CPU 分析命令（需要 NumPy；本机没有运行此命令）：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 patches/pf2x_round01/pf_round03_offline.py \
  --collection-dir docs/project/artifacts/pf2x_loop_20261019/round01/cost_probe_raw/run01/collection \
  --split-manifest docs/project/artifacts/pf2x_loop_20261019/round03/source_manifest.json \
  --out docs/project/artifacts/pf2x_loop_20261019/round03/results/offline
```

真实运行会生成 `summary.json`、逐样本 `rows.jsonl`、`q_outputs.npz` 和 `fit_parameters.npz`。当前 `results/offline/selftest.json` 只包含合成标准库自测结果；它不代表480行实算。该工具没有 Target-only 同轮对照，也没有运行候选解码，任何离线重合度都不能换算成真实速度比。

真实运行曾在最终写入 summary 时因熵分组计数成为 NumPy `int64` 而失败；逐样本记录、概率数组和拟合参数已先写出。计数现显式转成 Python `int`。自测在 NumPy 可用时会执行相同的 NumPy 布尔计数表达式并验证 JSON 编码；当前本机没有 NumPy，因此该专项部分会标为跳过。本次没有重跑480行数据，也没有改动模型、网格、seed、来源切分、尺度或停止条件。
