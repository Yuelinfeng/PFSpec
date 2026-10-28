# GaussianGPT 研究工作与真实证据对应

日期：2026-10-16。本文供汇报第二、三、四部分选材使用，依据本轮重新连接 Autodl 后读取的源码、训练记录、原始测量与已有输出。当前汇报正文未改动；没有新增训练、GPU 测速或渲染。

当前证据支持采用 Position Ranker + step12000 FeatureStateDraft + 静态 K2。阶段 E 的轻量 PF 微平均速度比为 **1.69675×**；阶段 F 的真实前缀完整补全时间比为 **1.56935×**。两项数字使用不同计时范围和汇总方式，须分别说明。

## 1. 位置、版本与证据等级

本文用 `G` 表示远端 `/root/workspace/GaussianGPT`，用 `R` 表示远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015`。表内 `G/…`、`R/…` 均由这两个绝对路径展开。

远端 Git HEAD 为 `031d916`，当前源码包含既有未提交工作。前轮定位统计为 4,176 项 Git 状态条目，因此 HEAD 不能单独标识本轮所读源码。研究记录主要在本机 `/Users/lief/Documents/ChatGPT/GaussianGPT/docs/project`；远端没有同名文档目录。本轮没有改动远端源码、权重、数据或 Git 内容。

本轮读取的源码与 JSON/CSV 副本保存在本机 `.codex/transfer/research_evidence_20261016/`，该位置沿用已有忽略规则。`source/` 对应 `G`，`runs/` 对应 `R`；未下载权重。核心来源清单为该目录的 `remote_inventory.json`，补充源码另存于 `additional_sources.tar.gz`。权重来源、参数计数、场景来源组和输出数量另存于 `weight_and_output_metadata.json`；场景划分为 `scene_splits.json`。

本轮可直接审阅的复算结果：

- [验证记录](artifacts/research_evidence_20261016/verification.json)
- [阶段 E 五模式数值](artifacts/research_evidence_20261016/stage_e_modes.csv)
- [阶段 F 模式与完整时间](artifacts/research_evidence_20261016/stage_f_modes.csv)

下文的“源码确认”表示读到实际调用或计算；“原始记录复算”表示从逐次记录重新汇总；历史单场景和 Teacher 轨迹结果始终保留其输入范围。

## 2. 工作—证据对应表

| 已完成工作 | 核心实现及权重 | 原始证据 | 可支持的结论 |
| --- | --- | --- | --- |
| 独立 Draft-A v2 | `G/draft_a/train_v2.py`、`benchmark_v2.py`；`G/draft_a/runs/rollout500_20261013/training/best.pt` | 同运行目录 `training/complete.json`、`training/validation.jsonl`、`benchmark_v2.json`、`result_summary.json` | 500 条 Teacher 轨迹后完成 5,000 步 CE 训练；所测在线配置的最佳速度比 0.87403×，未取得净加速 |
| GPT-D8 联合 P/F | `G/draft_a/feature_reuse.py`、`feature_adapt.py`、`feature_adapt_continue.py`、`feature_acceptance_eval.py`；`feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt` | `G/draft_a/runs/feature_acceptance_20261014/` 内各阶段 `plan.json`；`R/stage_e_final/fp32_test_full/online.json` | 38 epoch 预训练 D8 经 Feature 分布训练后接入联合验证；与独立随机初始化的 Draft-A v2 分开记录 |
| D4 容量对照 | `G/draft_a/d4_joint_train.py`、`d4_joint_eval.py`；`d4_joint_20261014/selected.pt` | `G/draft_a/runs/d4_joint_20261014/training_plan.json`、`selected.json`、`real_online.json`、`real_summary.json` | 随机初始化 D4 完成 8,600 步，验证选择 step7400；真实输入 Gamma2 为 0.63281×，不能把差异单独归因于层数 |
| Position Ranker | `G/eval_lean_candidate_to_draft.py`、`gpt_position_spec_decode_smoke.py`、`bench_fast_hgb.py`；`G/profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib` | `R/stage_e_final/fp32_test_full/online.json`；历史成本为 `G/profiling/target_eval/results/fast_ranker_96001_96012/summary.json` | 无额外 Position Transformer；阶段 E P-only 微平均 1.43491×，历史快速评分平均约 0.08041 ms/次 |
| 轻量 Feature MLP | `G/train_feature_state_distill.py`、`draft_a/feature_target_v2.py`、`feature_target_continue.py`；`feature_target_continue_20261014/selected.pt` | 对应运行目录 `data_plan.json`、`training_plan.json`、`history.json`、`selected.json`、`online_summary.json` | 2,864,384 个可训练参数；依据完整验证集选 step12000，续训相对 step3600 的在线额外收益尚不确定 |
| 联合 P/F K2 | `G/draft_a/pf_feature_unified.py`，由 `G/draft_a/pf_multiscene/feature_real_eval_multiscene.py` 调用 | `R/stage_e_final/method_selection.json`、`fp32_test_full/online.json`、`multiscene_summary.json` | 低成本 P/F 提案与 Target 验证结合，阶段 E 轻量 PF 微平均 1.69675× |
| 完整三维输出 | `G/draft_a/pf_multiscene/full_3d_eval.py`、`run_full_3d_pipeline.py`、`analyze_full_3d.py` | `R/stage_f_full3d/analysis/summary.json`、两份逐样本 CSV、`test_completion/samples/`、`bos_generation/samples/` | 270 份真实补全和 90 份 BOS 输出完成，均有 token、Gaussian NPZ 与六视角 PNG；真实补全时间比 1.56935× |

权重简写均位于 `G/draft_a/runs/`。表内“完整三维”仅指本批单个 `20×20×20` latent chunk 的完整自然结束生成。

## 3. 汇报第二部分：前期方法探索

### 3.1 研究顺序应怎样讲

现有记录包含并行和交错的路线。2026-10-13 的 LFQ、F1/F2 递归候选与 Draft-A v2 是不同工作；2026-10-14 的 Feature 直接 Target 训练从既有 `state_distill_mixed_overlap` 权重出发，Position ranker 也沿用既有权重；随后开展 D8 接法、提案长度和 D4 对照。2026-10-15 至 16 才通过多场景开发选择、阶段 E 与阶段 F 扩大证据范围。

因此可按研究问题组织第二部分：统一 GPT 草稿的接受率与成本限制、Position 结构化提案、Feature 低成本预测、开发集方法选择。日期与实际调用关系分别陈述，避免把这些方法写成依次替代的单线过程。早期更广的 S2/S3、LFQ、联合候选导航见 [方法历史](DRAFT_METHODS_HISTORY.md)；本轮重点核实附件点名的研究工作。

### 3.2 独立 Draft-A v2 的实证

`train_v2.py:19–25, 32–110` 使用 8 层、512 维、8 头 `NanoGPTGaussianModel`，共 37,617,680 参数，从随机权重开始。数据为原始 Target 产生的 500 条、1,024,144 个正文 token；按完整轨迹分为 400/50/50。`training/complete.json` 的步数为 5,000，`training/validation.jsonl` 中最小 CE 为 3.622643，位于 step3750。

`benchmark_v2.py` 使用标准顺序草稿、Target 分块验证、概率接受和拒绝后的剩余概率采样。`result_summary.json.decoding` 的 `context=1024,k=2` 为 0.8740255×；空上下文 K2 为 0.8535775×。范围为两个上下文长度、每配置三个 seed、最多 256-token 续写，模型加载不计时。这里的 K2 表示普通草稿的两个提案 token，与最终 PF 的 K2 含义不同。

同文件的固定前缀测试重合度为 Position 93.1700%、Feature 10.5304%。这是每类 1,600 个位置的分布统计，不能替换在线接受率。该实验说明在已测设置中，学会 Position 仍不足以抵消普通草稿和联合生成的成本。

### 3.3 GPT-D8 的真实训练链

本轮用 CPU 读取 `pretrained_d8.pt` 元数据，确认其 `origin` 为：

`G/profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt`

元数据为 epoch37、global step50502、8 层、512 维、8 头、37,617,680 参数。`feature_reuse.py:20–32` 将既有权重转成当前接口，随后三个实际训练计划形成以下链条：

`pretrained_d8.pt → adapt_d8/step_1200.pt → adapt_d8_continue_1200/step_2400.pt → adapt_d8_acceptance/step_3600.pt`

上述路径均相对 `G/draft_a/runs/feature_acceptance_20261014/`。第一段由 `feature_adapt.py` 执行，后两段由 `feature_adapt_continue.py` 执行；监督为原始 Target 完整 Feature 分布及 P/EOS 硬标签，最后 1,200 步增加权重 2 的采样策略 TV。TV 是两分布概率差绝对值之和的一半。正式阶段 E 明确载入最后的 step3600。

`PF-D8` 和 `GPT-D8` 使用同一份 D8 权重。前者的 Position 来自 ranker，仅 Feature 使用 D8；后者让 D8 连续产生 P/F 提案。实际接法见 `feature_real_eval_multiscene.py:225–243, 279–328, 347–363`。二者的结构与在线状态维护不同，不能合并命名。

### 3.4 支持方法选择的关键开发结果

| 开发对照 | 同轮结果 | 依据与解释范围 |
| --- | --- | --- |
| 真实输入：轻量 PF / PF-D8 / GPT-D8 | 速度比 1.44832× / 1.35606× / 0.95954×；F 接受率 20.2967% / 53.2552% / 54.3726% | `G/draft_a/runs/feature_d8_real_eval_20261014/summary.json.overall`；一个场景、四个不同裁剪，不能据此单独推断多场景泛化 |
| GPT-D8 Gamma2/4/6 | 0.96959× / 0.91266× / 0.81413× | `feature_d8_gamma_eval_20261014/summary.json.overall`；同场景同轮比较表明延长提案未取得额外净收益 |
| 随机初始化 D4 Gamma2/4/6 | 0.63281× / 0.49179× / 0.40065×；同轮轻量 PF 1.46297× | `d4_joint_20261014/real_summary.json.overall`；D4 为 4 层、256 维、9,371,656 参数；与预训练 D8 的训练起点不同 |
| 轻量 Feature 直接 Target 训练 | 新/旧速度比 1.01183，95% 区间 [0.98589,1.03544] | `feature_target_full_v2_20261014/online_summary.json.new_over_old`；F 接受率改善，额外速度收益尚不明确 |
| step3600 继续到 step12000 | 新/旧速度比 0.99762，95% 区间 [0.97596,1.01747] | `feature_target_continue_20261014/online_summary.json.new_over_old`；12 条既有 Teacher 测试轨迹，不能把续训写成速度显著提高 |
| 多场景开发集动态选择 | 29/29 个场景×前缀组合均由 step12000 轻量 PF 最快；未实现动态调度 | `R/stage_c/scheduler_selection.json`；只覆盖场景和前缀长度两类信号 |
| 真实来源适配 | B/旧速度比 1.00168，95% 区间 [0.99219,1.01034]，停止增加训练 | `R/stage_d2_adaptation/stage_c/dev_pair_seed42001/adaptation_comparison.json`、`development_decision.json`；固定前缀指标改善没有变成可确认的在线收益 |

前三行各有自己的同轮 Target，绝对吞吐不可跨轮拼接。后续阶段 E 支持相同的方法选择方向，但方法选择本身已有开发证据。

## 4. 汇报第三部分：最终方法的实际实现

### 4.1 Position Ranker：候选、特征与输出分布

`eval_lean_candidate_to_draft.py:47–93` 的 `FrozenLean` 根据最近两个 gap、当前位置在边长 20 网格中的相位，以及在线累计的 gap 频数产生小候选集。它组合相位条件众数、gap=1、历史众数与少量高频候选；稀疏统计和特定相位会增加候选。`ids_from_gaps` 依据 Target 的扫描顺序映射剔除非法位置。

`gpt_position_spec_decode_smoke.py:34–112` 的 `CausalRanker` 为每个候选构造 32 个特征：gap、空间索引、固定尺度的进度、最近四个 gap、候选数量、若干历史众数标记、频数与概率、三 gap 条件统计和最近出现间隔。信息均来自已提交的 Position 历史；不读取下一真实位置或待生成 Feature。进度项在线使用 `i/511`，不会用最终生成长度。

训练入口 `eval_lean_ranker_holdout.py:106–116` 使用 `HistGradientBoostingClassifier`，150 轮、最多 15 个叶子、学习率 0.05。标签表示候选是否等于 Target 已采样的下一 Position。`bench_fast_hgb.py:50–99` 将数值分裂树转成 Numba 评分；`choose_position:126–140` 取评分最大的候选。

**评分只用于选择候选。实际 Position 提案是选中位置上的点分布。** 因而其接受概率是 Target 对该位置的概率；拒绝后将该位置概率置零，再归一化采样。不能把 ranker 的二分类评分直接当作整个 Position 词表的提案分布。

成本证据为历史 12 seed、1,024-token 对照：`fast_ranker_96001_96012/summary.json.summaries.learned_fast` 保存 2,273 次评分，共 0.182772 秒，即 **0.080410 ms/次评分**；相同调用数的 sklearn 为 1.125827 ms/次。该时间仅含评分调用，未含候选构造、状态复制与 Target 验证，也不代表阶段 E 的逐轮总成本。

### 4.2 Feature MLP：真实输入、参数与训练

`train_feature_state_distill.py:25–68` 定义 `FeatureStateDraft`。实际选定权重的隐藏维度为 1,024；输入是上一 P 的 Target 最后一个 Transformer block 隐藏状态、上一已提交 F 的 128 维 embedding、当前候选 P 的 128 维 embedding。隐藏状态经 LayerNorm 和 `1024→256` 线性层后，与两个 embedding 拼接成 512 维输入；残差网络为 `512→512→512→1024`，中间使用 GELU，输出加到上一隐藏状态上。

| 可训练部分 | 参数个数 |
| --- | ---: |
| 输入 LayerNorm | 2,048 |
| 1024→256 线性层 | 262,400 |
| 4096×128 Feature embedding | 524,288 |
| 8000×128 Position embedding | 1,024,000 |
| 拼接后 LayerNorm | 1,024 |
| 残差网络三层 | 262,656 + 262,656 + 525,312 |
| 合计 | **2,864,384** |

模型沿用 Target 的最终归一化和 `4096×1024` Feature 输出权重，再做 softcap=15。共享输出权重有 4,194,304 个元素，不计入上述可训练参数，也未保存在学生的参数文件中。这里共享的是原始 Target Feature head；没有为在线提案额外运行完整 Target Transformer。

`feature_target_v2.py:38–105` 从完整因果前缀提取上一 P 状态，并以原始 Target 在当前 P 后的完整 4,096 类分布作标签；第一个 Feature 由 Target 直接生成，不进入该状态模型监督行。训练有 400 条轨迹、418,021 行，验证有 50 条、46,893 行。`objective:126–133` 为原始温度 1.0 完整分布 CE，加权重 0 或 2 的 temperature/top-p=0.9 策略 TV。

`feature_target_v2.py:183–235` 按完整验证集位置加权重合度选权重。三个 1,200 步阶段得到 step3600，随后 `feature_target_continue.py:38–51` 再训练 8,400 步；最终验证重合度从 26.4062% 到 26.7549%，`selected.json.step=12000`。在线新/旧速度区间跨过 1，因此采用 step12000 的依据应写成验证指标选择和开发保留，不能写成继续训练已证明额外提速。

在线输入由 `pf_feature_unified.py:127–153, 341–360, 440–452` 确认：`h_prev` 总是来自已提交 P；发生 F 拒绝后保留 P1 的隐藏状态，并使用新采样的已提交 F。没有读取当前真实 Feature 标签。

按各线性层维度计算，一次 Feature 预测约含 **5,505,024 次乘加**，其中共享输出头占 4,194,304 次；该推算不包含归一化、激活、softmax、访存和 CPU/GPU 调度。

已有前向时间证据也已找到：`G/draft_a/runs/d4_joint_20261014/stage_profile.json.rows` 的两个 `model=mlp` 记录使用同一份 step12000 权重，载入路径见 `d4_joint_eval.py:16, 277–280`。seed24720、128-token 短测中，64 前缀有 48 次 Feature 前向，CUDA event 均值 **0.48043 ms/次**，主机计时均值 0.44904 ms/次；1024 前缀的 57 次分别为 **0.49814 ms/次**和 0.46655 ms/次。两种计时观察范围不同，不相加；这里只计 Feature 前向，不含随后独立执行的采样策略与抽样。它们来自历史单场景短测，不能作为阶段 E 各场景的完整提案成本。

### 4.3 K2 的提案—验证流程

正式入口为 `run_final_evaluation.py:99–127 → feature_real_eval_multiscene.py:347–363 → pf_feature_unified.decode`。`P-only` 则调用 `pf_ranker_eval.decode(k=1)`；`GPT-D8` 调用 `feature_acceptance_eval.decode(gamma=2)`，应区分三条路径。

统一 PF 循环 `pf_feature_unified.py:347–480` 的输入是已提交的完整 P/F 前缀。缓存已消费到上一 P，末尾 F 尚待处理。

1. Ranker 提出 P1，复制位置历史并假定 P1 提交，再提出 P2；MLP 从已提交状态预测 F1 的分布并采样 F1。
2. Target 一次处理 `[上一F, P1, F1, P2]`，四行输出依次给出 P1、F1、P2 和额外 F2 的条件分布。K2 表示最多两个 Position 提案，中间有一个 Feature 提案，实际提案宽度为三个 token。
3. 按 P1、F1、P2 顺序验证。P 点提案使用 `p(P)`；F 提案使用 `min(1,p(F)/q(F))`。
4. P1 拒绝时移除候选位置对应的缓存，按 Target 剩余概率重新采样位置，再由 Target 采样其 F；F1 拒绝时保留 P1，按归一化 `max(pF-qF,0)` 采样 F，并丢弃后续 P2。
5. P2 拒绝时保留 P1/F1，重新采样 P2 对应位置及其 F；三个提案均接受时，直接由已取得的 Target 分布采样额外 F2。
6. 候选不足或剩余预算不足四个 token 时退到单 Position 提案；没有合法候选时由 Target 直接生成。EOS 只在完整 P/F 对结束后出现，遇到 EOS 立即结束。

缓存回退由 `bench_gaussiangpt_pf_block2_cow.py:62–93` 和统一循环 `314–320, 425–478` 实现：同时更新长度和 token 缓存，并按拒绝位置保留已接受部分。MLP 自身没有 KV Cache，GPT Feature 接法维护独立缓存。3D RoPE 继续由原始 `model/gpt/nanochat_gpt.py:543–611` 根据缓存中的实际 Position token 构造坐标，Position 合法性按同一 `sparse_position_rank_map` 维护。

阶段 E 保存的 FP32 分块/逐 token 最大总变差为 Target `3.2708e-6`、D8 `3.5863e-6`，包含缓存回退验证；另有各前缀的计时开关输出一致记录。它们是已有数值证据，本轮没有重新运行 GPU 数值实验，也不把有限样本验证描述成完整分布证明。

## 5. 汇报第四部分：阶段 E 正式结果

`R/stage_e_final/pipeline.json` 为 `complete / ready_for_full_3d_stage`。`crop_inventory.json` 记录 30 场景、56 个唯一裁剪、63,328 个正文 token；裁剪实物为 `final_test_crops.pt`。开发/正式测试/适配分别有 10/30/20 个来源组，本轮读取划分文件及 6,813 行场景目录，确认三组之间没有来源组交集；该独立性限于本轮划分，历史 Target/D8 的预训练来源另计。

`method_selection.json` 确定 FP32、temperature/top-p=0.9、前缀 64/256/1024、最多续写 512 token、seed base 34701/34702/34703、静态 K2 和旧 step12000。实际 seed 还按裁剪与前缀派生，见评价脚本 `334–340`。64/256 前缀各有 56 个可用裁剪，1024 前缀有 32 个，因此总数为 `(56+56+32)×3×5=2,160`；每模式 432 次。

| 模式 | P 接受率 | F 接受率 | token/s | 微平均速度比 | 来源组几何平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target | — | — | 70.66882 | 1.00000× | 1.00000× |
| P-only | 92.44076% | — | 101.40326 | 1.43491× | 1.43773× |
| 轻量 PF | 92.64095% | 33.25815% | 119.90716 | **1.69675×** | **1.71250×** |
| PF-D8 | 92.50658% | 55.47118% | 103.00437 | 1.45756× | 1.46646× |
| GPT-D8 | 78.96773% | 55.52214% | 66.44308 | 0.94020× | 0.93815× |

来源为 `fp32_test_full/online.json.rows`；对应保存字段位于 `multiscene_summary.json.overall.<mode>`。接受率先合并 accepted/verified 计数；token/s 为 token 总数/总秒数，微平均速度比再除以 Target 吞吐。来源组速度比先在组内汇总吞吐，再对组间速度比取几何平均。轻量 PF 的来源组 95% 区间为 [1.68044,1.74402]，30/30 来源组快于 Target；GPT-D8 有 24/30 来源组慢于 Target。该区间不能附到 1.69675× 的微平均结果上。

本轮重新汇总 2,160 行，五模式吞吐、两个速度比和两类接受率均与原汇总一致，最大绝对差为 0；没有重复的裁剪×前缀×seed base×模式记录。阶段 E 区间沿用原文件，没有再次抽样计算。计时包括前缀、缓存、提案、验证、拒绝处理和采样，不含模型加载、数据准备、VQ 解码和渲染。

BF16 不在正式结果中。现有开发记录显示其分块/逐 token 分布差异未满足本计划使用的 0.01 容差，不能补写 BF16 正式速度或把 FP32 结论外推。

## 6. 阶段 F：从 token 到 Gaussian 与六视角图

`full_3d_eval.py:118–155` 设定 256-token 真实前缀、16,000 正文 token 上限、FP32、temperature/top-p=0.9。真实补全为 30 场景×3 seed（51701–51703）×3 模式，共 270 份；BOS 为 30 seed（51801–51830）×3 模式，共 90 份。

`full_3d_eval.py:219–253` 将最终 token 交给 VQ-VAE，输出 Gaussian NPZ 并保存六张 PNG。`gaussian_tensors:65–83` 将 P 转成局部三维坐标、F 减去 8000 后交给 VQ 解码，得到中心、尺度、不透明度、四元数和颜色；尺度保存前裁切到 `[1e-4,0.05]`，颜色与不透明度处理到 `[0,1]`。`render_views:86–95` 调用 `gsplat.rasterization`。因此最终输出已连接完整 VQ 解码和渲染链路。

本轮远端数量核对：真实补全有 270 份 `tokens.pt`、270 份 `gaussians.npz`、1,620 张 PNG；BOS 有 90、90、540。两份逐样本 CSV 共有 360 行，三模式组合齐全，状态均完成、自然 EOS，截断数为零。文件数量确认与保存指标相互对应；本轮未重新载入全部 NPZ 或逐张评价 PNG。

| 真实补全模式 | 完整时间均值 | 平均新正文 token | 相对 Target 的场景等权时间比 | 95% 场景重采样区间 |
| --- | ---: | ---: | ---: | --- |
| Target | 46.06736 秒 | 3,050.18 | 1.00000× | — |
| P-only | 31.36916 秒 | 2,987.31 | 1.46713× | [1.26583,1.73137] |
| 轻量 PF | 29.52030 秒 | 3,086.49 | **1.56935×** | **[1.31784,1.85226]** |

来源为 `R/stage_f_full3d/analysis/test_samples.csv` 和 `summary.json.test_completion`。时间比按同场景同 seed 的 Target 时间/候选时间计算，场景内三个 seed 取几何平均，再让 30 场景等权。PF/Target 的全样本总时间比另为 1.56053×，不能替换场景等权数字；PF/P-only 为 1.06968× [0.91223,1.25177]，额外完整时间收益仍不确定。BOS 的 PF/Target 为 1.79071× [1.58758,2.02307]，用于补充观察。

本轮从两份 CSV 重算 P-only/Target、PF/Target、PF/P-only 的时间比，并按原算法和 seed 重做每组 10,000 次来源组重采样；点估计及区间均复现。每行完整时间也等于 AR、VQ 解码、导出、渲染与 PNG 保存五部分之和。完整时间不含模型加载、预热和参考重建；各模式自然 EOS 得到的输出长度不同，所以同时保留 token 数和吞吐。

同一 CSV 复算的全图 PSNR/SSIM 均值为 Target 11.40815 dB/0.73016、P-only 11.66826 dB/0.73827、轻量 PF 11.45588 dB/0.73214。它们只描述与一份参考 VQ 重建的差异，包含随机续写区域；不能据此宣称画质等效或某方法更优。单个局部 chunk 的结果也不能证明多 chunk 接缝和连续性。

代表性输出沿用现有材料，无需再次生成：

- [真实补全六视角预览](artifacts/stage_f_full3d/analysis/test_completion_30scenes_seed51701_6views.preview.png)及[30 场景对照视频](artifacts/stage_f_full3d/analysis/test_completion_30scenes_seed51701_6views.mp4)。
- [BOS 六视角预览](artifacts/stage_f_full3d/analysis/bos_30trajectories_6views.preview.png)及[30 条轨迹视频](artifacts/stage_f_full3d/analysis/bos_30trajectories_6views.mp4)。
- 第一份真实场景 seed51701 的原始 token、Gaussian 和独立视图在 `R/stage_f_full3d/test_completion/samples/1af21af5-a873-492c-b201-c8d3eb4888b4_seed_51701/{target_only,p_only,step12000}/`。

## 7. 已核实结论与剩余限制

汇报第二部分可以说明：普通 GPT Draft 的 Position 匹配和更高 Feature 接受率，均没有自动带来更高净吞吐；已有开发对照支持低成本分工。第三部分可以直接解释 Position 点提案、共享 Target Feature head 的状态 MLP、三个 token 提案与四行 Target 验证，以及逐位置拒绝处理。第四部分分别采用阶段 E 的 1.69675× 与阶段 F 的 1.56935×，并保留各自范围。

仍需保留的限制为：早期实验中有单场景和重复使用的 Teacher 测试轨迹；D4 与 D8 的训练起点不同；阶段 E 仅 FP32 固定窗口；阶段 F 仅单 chunk；PF 相对 P-only 的完整时间额外收益区间跨过 1；可见区域质量、LPIPS 和多 chunk 连续性尚无结果。Feature 前向成本有历史短测，正式多场景测量没有保留逐次分项延迟。

本轮仅新增静态证据整理和既有结果复算，未改变模型行为或运行环境。`PROJECT_STATUS.md` 保留原生成时间，当前汇报正文也保持原样。
