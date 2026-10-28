## 2026-10-20 03:48更新

原192000步与486次评价已全部结束。完整合成NPZ输入现已在既有远端Python3.10.21/NumPy2.2.6、CUDA不可见的CPU环境通过，见 `results/npz_cpu_validation.json`。本记录下文保留先前准备时的状态；真实模型/CUDA/KV及性能仍未验证，先等网页接收最终结果并确认开发参照。首个合成输入的KV时间字段已按真实格式改为wall_seconds，分析器与测量器没有为此修改。

# PF-2X 诊断入口准备记录

状态：本机完成入口与纯 CPU 数学自测；尚未启动模型、CUDA 或远端任务。当前远端训练和既定评价仍在进行，诊断入口保持未运行。未来诊断总预算不超过 1 GPU 小时。

## 本轮实现

仅新增本记录和三个诊断入口。入口要求运行者显式提供工程目录、数据文件、Target/VQ 权重、Position 排名器、Feature 权重、Feature 步数标签、输出目录；不会替用户挑选 Feature 权重。

`collect_pq.py` 复用 `draft_a.pf_feature_unified.decode`。临时包裹现有 MLP、Target 调用、策略抽样与接受决策，保留同一次 MLP 计算的原始 logits、实际 `q1`、Target 的 `p_F`、真实接受记录和各阶段用时。普通 W1 的采样与额外分支诊断分开记录。只保存 Target 实际接受 P1 后的 F 上下文，每个来源/前缀最多 24 条，最多 10 个既有开发来源、64/256 两种前缀，共 480 条。默认 seed bases 沿用现有评价入口的 34701、34702、34703，实际运行 seed、轮次、来源路径和样本不足数量都会保存。

策略固定为 FP32 Target 与 FP32 Feature MLP，`temperature=0.9`、`top_p=0.9`。Target 的 `p_F` 从真实 W1 四 Token 调用的 Feature 位置 logits 计算；`q1` 与原始 MLP logits 来自同一次模型调用。所有源行计数保存在 `metadata.json`，采样上下文保存在 `contexts.jsonl`，4096 维向量保存在压缩 `vectors.npz`，W1 各操作记录保存在 `w1_profiles.jsonl`。每个正式采集 run 的输出 Token 和完整计数写入 `w1_runs.jsonl`。

正式采集前，入口会用同一来源、前缀、seed 和不超过 16 Token 的短预算，先跑普通 W1，再跑带采集包装的 W1；比较完整输出 Token 及 Position/Feature 提案、接受、调用次数、轮次和输入长度。结果写入 `trajectory_parity.json`；不一致时停止采集并保留失败记录。该成对短回放的 Target 调用与正式采集行分开统计。每个正式采集 run 本身不增加 Target 前向。

`measure_target_shapes.py` 只挑选真实 P1 接受、F1 拒绝且残差 Token 已保存的上下文。对每条记录先重放原 W1 四 Token 输入取得 P1 隐状态，再恢复到处理完 P1 的 KV 位置。短块、后续四 Token 块与第二分支都从相符的位置继续；第二分支使用自己的 Target logits，不复用首个 F 分支结果。分支输入为 q2 抽出的不同 Feature 与 P2；P2 接受概率取自该次分支 Target 前向。单次调用最多 96 条分支，`--q2-mode` 显式选窄支持或宽支持。没有合法 q2 支持的真实样本会计为未入选，不补造样本。

`analyze_cost_probe.py` 只在 CPU 上读取采样和计时文件。它对两种 q2 分别计算 `A1`、`R1`、`β(a)`、按首次拒绝候选分布加权的 `β̄` 及来源/前缀离散度。若第二候选 `b` 由 q2 抽出，第二次接受用 `min(r1[b], q2[b|a]) / q2[b|a]` 加权；联合 P2 成功事件还乘以该分支的 `p(P2)`。设 `L(c)` 为上下文 `c` 的首拒候选中有合法第二候选支持的概率质量，则全体首拒触发率为 `P(P1接受) × E[R1(c)]`，每轮合法事件率单独按 `P(P1接受) × E[R1(c) × L(c)]` 计算。首拒候选无第二候选支持时贡献零；合法分支采样的平均联合成功率只乘合法事件率。输出同时列出 q2 无支持质量的普通上下文均值和按 `R1` 加权的首拒事件比例。完整第二阶段的 Feature 分布仍按 `r1` 统计，不把接受子集当作全体 `r1`。

来源区间只重采样来源级别的代入均值，没有包含上下文抽样与分支抽样误差。合法分支缺样的来源/前缀单元保持缺失，不补成零；汇总均值会标出覆盖数量和缺失单元，只对已有单元计算。无合法首拒候选支持的单元事件率明确记零。有限分支样本给出的只是估计值。

成本结果分列 2/4 Token Target 前向的 CUDA 时间和主机墙钟时间、KV 分配/前缀重放/恢复用时、非 Target 操作及整段诊断墙钟时间。`measure_target_shapes.py` 在 `branch_4` 完成后，对最先选出的最多 4 个上下文做形状预热与配对重复；`--pair-contexts` 可设为 1–20，默认 4，`--pair-repeats` 可设为 1–16，默认 8。每个入选上下文对 2/4 Token 输入各预热一次，再按交替顺序做 8 组配对；起始形状随上下文序号轮换。短输入固定为 `[F1残差,P2]`，长输入固定为 `[F1残差,P2,F2草稿,P3]`，长输入只复用此前生成的 F2 草稿和 P3。若原位置策略没有合法 P3，该上下文记为不可用，不另选上下文补足。

配对开始时先把 `branch_4` 缓存截到 `keep=before_context+2`。每个前向都从 keep 开始，2 Token 调用到 `n+4`、4 Token 调用到 `n+6`，随后立即恢复 keep。预热记录与配对前向使用不同记录类型；配对记录含 `sample_id`、`pair_id`、顺序、固定输入、设备时间、主机墙钟时间、KV 到达位置和恢复用时。分析器只对同一上下文的 2/4 配对计算差值与比例，预热、配对和原有单次形状均值各自汇总；KV 恢复时间单独汇总，不并入前向配对值。配对只测条件前向，不代表端到端加速。

默认新增最多 72 次 Target 前向：4 个上下文各 2 次预热及 8 组双向配对；另有最多 76 次 KV 截短：每个配对上下文从 `branch_4` 回到 keep 一次，再为每次前向恢复一次。分支条件总上限仍为 96，不增加 q2 抽样或 `branch_rows`。入口按实际调用记录前缀预填、原形状、预热、配对及全部 Target 前向数，也累计实际 KV 截短数；每上下文的 outer wall 记录采用这些计数，不用固定常数推算。R1 所有任务共用 1 GPU 小时；未来主代理应按剩余总额度设置整个进程截止时间，不能让各子命令各自重新取得一小时。W1 外围计时包含入口包装和同步，仍标记为诊断计时。若提供配对 W1 与 Target token/s 及来源说明，分析器只给出“每个有利事件恰好节省一次 Target 验证、分支成本为零”的乐观点估计，不代表真实收益，也不能据此决定停止。两种速率必须采用相同固定窗口、聚合方式和计时口径；来源说明文字本身不能证明口径相同，当前结果会标注尚未确认。

## 源码接口依据与待核对项

入口按当前只读快照使用以下真实符号：

- `draft_a/pf_feature_unified.py`：`MLPAdapter.propose`、`decode`、`_StageProfile.run`；W1 四 Token Target 输入和 KV 截断位置。
- `draft_a/pf_ranker_eval.py`：`policy`、`typed_kinds`、`position_state`、`cache_for`、`draw`。
- `gpt_position_spec_decode_smoke.py`：`choose_position`。
- `bench_gaussiangpt_pf_block2_cow.py`：`CaptureLastBlock.run`、`truncate`、`residual_sample`、`fork_position_state`、`load_feature_model`。
- `collect_feature_teacher_from_s3.py`：`policy_batch`。
- `bench_fast_hgb.py`：`FastHistRanker(clf)` 与 `predict_proba(x)`。
- 远端 `draft_a/corpus_v2.py:load_target` 展示了 `GaussianVQVAE.load_from_checkpoint`、`GaussianGPT.load_from_checkpoint` 和 `_orig_mod` 的现行加载方式；本机 `patches/draft_a_rollout_v2/corpus_v2.py` 是一份旧 patch 副本，只作接口参照。原入口还核实 EOS、P/F 词表大小、xyz 排列与稀疏 Position 约束。两个诊断加载器使用显式权重路径，并保留这些数值和排列要求。

当前 `bench_fast_hgb.py` 顶层导入 `eval_lean_ranker_cpu.build_examples`；这两个当前文件快照现已由主代理取得。`FastHistRanker` 还需要环境中的 Numba、NumPy、scikit-learn 与 joblib。GaussianGPT/VQ 类、CUDA、PyTorch 与实际远端依赖组合尚未在本机加载验证。后续首次运行需以远端现有环境验证这组真实接口；本轮不安装、不连接远端，也不调用 CUDA。

## 未来调用示例

以下路径和值是占位符，需由未来操作者依据已完成的既定评价明确传入。示例不选择特定 Feature 步数或权重文件。

```sh
python patches/pf2x_round01/collect_pq.py \
  --project-root /path/to/GaussianGPT \
  --crops /path/to/existing_dev_crops.pt \
  --target-checkpoint /path/to/selected_target.ckpt \
  --vqvae-checkpoint /path/to/selected_vqvae.ckpt \
  --ranker /path/to/position_ranker.joblib \
  --feature-checkpoint /path/to/selected_feature.pt \
  --feature-head /path/to/feature_head.pt \
  --feature-step STEP_LABEL \
  --output-dir /path/to/research/pf2x_round01/pq \
  --show-plan
```

配对形状的默认计划为 `--pair-contexts 4 --pair-repeats 8`；可先在本机运行 `python -B patches/pf2x_round01/measure_target_shapes.py --cpu-test` 验证调度和模拟缓存位置，无需模型、NumPy 或 CUDA。

确认参数和输出位置后，未来执行时将 `--show-plan` 换为 `--execute`。分支测量同样要求提供上述真实模型路径、采集目录及 `--q2-mode narrow` 或 `--q2-mode wide`；若两个 q2 模式都采集，两个调用的 `--max-contexts` 总和必须不超过 96。分析入口需同时给出采集目录、分支测量目录、最终 Feature 权重路径、步数标签和一个尚不存在的输出 JSON 路径。所有输出目录在执行前都必须不存在，避免覆盖旧记录。

## 本机验证与未运行项

已实际运行三个入口的 `--help`、纯 Python 语法解析、`measure_target_shapes.py --cpu-test` 和 `analyze_cost_probe.py --self-test`。配对调度模拟验证了 4 个上下文、8 次预热、32 组配对、72 次前向和 76 次截短；每组各有一次 2/4 Token 调用且顺序交替，所有调用复用固定输入并从 keep 到达预期 KV 位置后恢复。概率自测覆盖两阶段输出分布、首次拒绝候选加权、q2 接受权重、联合 P2 权重、合法与无支持分支混合、零 `R1` 行，以及 `2^-60` 正概率；262 组穷举分布最大绝对误差为 `1.39e-16`。CLI 帮助与自测采用 `python -B`，未写入字节码缓存。当前本机 Python 缺 NumPy，完整 NPZ 输入分析尚未运行；不安装依赖，留待远端既有 CPU 环境验证。短轨迹同 seed 对照与形状配对流程均尚未在 GPU 环境运行。

未运行 `--execute`、真实采集、Target 形状测量、模型加载、CUDA 或远端命令。当前训练/既定 486 次评价未完成；未来仍须在总计 1 GPU 小时内按采集、分支测量、分析的顺序执行，并保存实际样本不足事实和运行耗时。
