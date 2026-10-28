# 同一 Target 图下的候选方法清单

日期：2026-10-21。权重存在性与文件大小来自远端 `/root/workspace/GaussianGPT` 及 `/root/autodl-tmp` 的只读查询。未启动 GPU、训练或历史测量入口。

本轮候选运行器以同一 Target 和同一输入组比较 `original` 与 `optimized` 两种执行；优化执行使用 `patches/target_methods_20261021/runtime/dynamic_target_graph.py`。旧评测入口本身没有接入动态图，新 `method_adapters.py` 已将主 11 法连到共同 Target runtime；主11法的3来源配对132次与10来源完整计时220次已结束，详见RESULTS.md。LFQ另走BOS-only18次，原/图重放未一致，追加数值复核测得支持集变化，暂不排名。

## 本轮运行器候选

| runner ID | 实际差异与建议配置 | 权重路径与大小 | 可追溯入口 | 当前 Target 图状态与历史限制 |
| --- | --- | --- | --- | --- |
| `target` | 普通自回归 Target；所有速度比的共同参照 | `/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt`，3,421,157,765 B | 新 runner 的共同 Target loader | 同一 checkpoint 分别走 `original` 与 `optimized`；Target 自身不是草稿方法。 |
| `pf_mlp` | 已用的 Ranker + Feature MLP；作为既有 PF 参照 | `/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`，11,466,893 B；共享 Position Ranker 见 `ranker_only` | `draft_a/pf_ranker_eval.py:decode` | 适配器将 Target 前向连到共同图运行器；配对运行已完成，见RESULTS.md。 |
| `ranker_only` | Position Ranker 提案，不运行 Feature Draft | `/root/workspace/GaussianGPT/profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib`，310,544 B | `draft_a/pf_ranker_eval.py:decode`；旧对照入口 `step5_method_compare.py:p_only` | 适配器将 Target 前向连到共同图运行器；配对运行已完成，见RESULTS.md。 |
| `eagle_v2_joint_g3` | EAGLE P/F 联合提案，γ=3 | `/root/autodl-tmp/gaussiangpt_eagle_pf_v2_20261019/training_extended/joint/selected.pt`，6,296,728 B | `eval_pf_draft.py` → `speculative/pf_engine.py:decode(mode="joint", gamma=3)` | 历史 γ=3 速度约为原 Target 的 0.739×，Position 接受率较低；这是训练版本与提案结构的实测差异。 |
| `eagle_v3_g3` | EAGLE Feature-only，γ=3 | `/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`，6,297,944 B | `eval_pf_draft.py --feature-only --feature-checkpoint … --gammas 3`；解码函数为 `speculative/pf_engine.py:decode(mode="feature")` | 历史单轮为原 Target 的约 1.367×，仍慢于当时旧 PF；训练条件和新 Target 图都需单独报告。 |
| `eagle_v3_g5` | 与 `eagle_v3_g3` 同一权重，γ=5；比较提案深度 | 同上，6,297,944 B | 同上，`--gammas 5` | γ 是推理配置，权重不变；不应算作新训练版本。 |
| `eagle_v4_reuse_g3` | v3 Feature-only γ=3，并启用 Draft KV 复用 | 与 v3 相同权重，6,297,944 B | `eval_pf_draft.py --feature-only --feature-checkpoint … --gammas 3 --reuse-draft-cache` | v4 是缓存执行选项，不是新权重。旧测试减少了 99.18% Draft 输入量，但总吞吐下降约 1.22%；需用新 Target 图重测组合效果。 |
| `medusa_lowrank` | 三个低秩独立预测头，共用 Target 隐状态，产生 P1/F1/P2 候选 | `/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare/medusa_lowrank/medusa_linear_heads_best.pt`，11,457,377 B | `/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare.py:decode_heads(head_kind="independent")` | 历史首个 Position 接受率约 0.78%，深层候选到达少。旧路径的 P1 不读取尚未处理的 pending Feature；新运行保留该模型语义。 |
| `hydra_conditional` | 条件头按 P1→F1→P2 顺序提案 | `/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare/hydra_width177/medusa_conditional_heads_best.pt`，11,470,867 B | 同上 `decode_heads(head_kind="conditional")` | 历史首个 Position 接受率约 0.46%，深层候选到达少。P1 同样未读取 pending Feature；本轮只比较整条适配路径。 |
| `d8_pf` | 8 层 Target 形状的 D8 Feature 提案，配既有 Ranker | `/root/workspace/GaussianGPT/draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt`，150,501,947 B；Ranker 路径见 `ranker_only` | `draft_a/feature_d8_real_eval.py` → `draft_a/feature_real_eval.py:run` | 单一重建场景历史结果为原 Target 的 1.356×、旧 PF 的 0.936×。样本只有一个场景；旧入口无本轮动态图。 |
| `d8_joint` | 同一 D8 权重同时提 P/F，γ=2 | 与 `d8_pf` 相同的 150,501,947 B 权重 | `draft_a/feature_d8_real_eval.py` → `draft_a/feature_real_eval.py:run(joint=True)`；验证解码见 `draft_a/feature_acceptance_eval.py` | 历史单场景速度为原 Target 的 0.960×。与 D8-PF 是不同提案机制，共用同一权重。 |

当前主比较使用 11 个运行器 ID：`target`、`pf_mlp`、`ranker_only`、`eagle_v3_g3`、`eagle_v3_g5`、`eagle_v4_reuse_g3`、`eagle_v2_joint_g3`、`medusa_lowrank`、`hydra_conditional`、`d8_pf`、`d8_joint`。Target 本身是每个执行臂的共同速度参照，不是草稿候选。

## 已有但暂未列入本轮 runner 的路线

| 路线 | 已有权重或代码 | 是否为独立提案方法 | 当前不优先纳入的原因 |
| --- | --- | --- | --- |
| 项目中未确认标作 EAGLE v1 的模型 | `/root/workspace/GaussianGPT/profiling/feature_draft/full_train/feature_draft_best.pt`，20,896,981 B；配置为两层 `FeatureDraft`，离线入口 `eval_feature_draft_policy.py` | 是旧 FeatureDraft 权重；项目资料没有证据把它称为正式 EAGLE v1，也未确认现成的精确在线解码入口 | 不将名称相近当作版本证据；若要重测，先选定其解码语义与合法残差路径。 |
| EAGLE v2 Feature-only | `/root/autodl-tmp/gaussiangpt_eagle_pf_v2_20261019/training_extended/feature/selected.pt`，6,296,728 B；与 v2 joint 分开的训练权重 | 与 EAGLE v2 Joint 共享 Draft 架构和训练批次系列，但解码分支不同 | 本轮 runner 当前列出 v2 Joint 与 v3 Feature；可作为下一候选，不与 v3 训练变体混算。 |
| Medusa Aligned | `/root/workspace/GaussianGPT/profiling/baselines_5way/b3_medusa_aligned/train_40seq/medusa_aligned_heads_best.pt`，25,170,605 B；入口 `bench_gaussiangpt_medusa_aligned.py:medusa_aligned_decode` | 是 Medusa 头部与 Target 状态对齐方式的不同训练方案 | 本轮先覆盖 step5 的低秩 Medusa 与 Hydra；这组是另一训练集与权重。 |
| Medusa Conditional / Transition | `profiling/baselines_5way/b3_medusa_conditional/train_40seq_v3/medusa_conditional_heads_best.pt`，9,463,443 B；`profiling/baselines_5way/b3_medusa_transition/train_40seq_v4/medusa_conditional_heads_best.pt`，11,549,395 B | 同一条件头机制的训练版本，不应当作两种新采样算法 | 代表权重已有；需要限制候选数量，避免多组训练版本挤掉真正不同的解码机制。 |
| Medusa 预特征 P1 | `profiling/baselines_5way/medusa_pre_feature_p1/train_40seq/medusa_pre_feature_p1_best.pt`，60,985,096 B | 预测条件结构不同，属于可比较的旧路线 | 需要确认与当前 Target 图 adapter 的接口和额外 Target 状态成本后再加入。 |
| LFQ on-policy warmstart | `profiling/feature_draft/lfq_new_onpolicy_warmstart/lfq_scratch_conditional_best.pt`，10,851,517 B | 是 LFQ 训练来源变体，解码方法与 scratch 权重相同 | 可作为后续 LFQ 权重对照；本轮 BOS-only 小比较以 scratch 权重为代表。 |
| LFQ 单步 Feature，BOS-only 补充对照 | `/root/workspace/GaussianGPT/profiling/feature_draft/lfq_scratch_30epoch/lfq_scratch_conditional_best.pt`，10,851,453 B；入口 `bench_lfq_paired_ab.py` → `bench_lfq_scratch_pf_cow.py:block2_decode` | LFQ 条件 Feature 提案，已有权重与旧解码入口 | 旧函数没有 prompt 参数，只从 BOS 开始。新 runner 支持 `--prefixes 0 --sources 1`，并单列 `target`、`pf_mlp`、LFQ 三法，original/optimized 两执行、三种子、512-token 预算。此项不含场景上下文，不能用于来源泛化结论。 |
| LFQ 递归 P/F | `profiling/feature_draft/pf_joint_recursive_20261013/cpu_comparison/joint/pf3_depth2_best.pt`，9,791,691 B；入口 `evaluate_pf_joint_recursive.py`，加载 `--joint-dir` | 额外提出递归 Feature 候选，属于独立的延伸解码方法 | 历史 K2/K3 GPU 比较较慢；若纳入，需用 runner 统一 Target 与计时，不复用旧速度作结论。 |
| 标准 D8 Position 模型 | `profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt`，1,056,118,155 B；入口 `profiling/s2/s2_5d2a_speculative_throughput_bench.py` | 与 D8 Feature/PF 权重不同的独立 Position 草稿模型 | 文件大且旧结果仅比 Position-only 快约 1.06–1.15×；当前代表集已有 D8-PF 与 D8 Joint。 |
| B1-E 连续 Feature | `/root/autodl-tmp/gaussiangpt_round2_20261017/b1_training/20261020/b1_e/selected.pt`，5,278,399 B；入口 `/root/autodl-tmp/gaussiangpt_round2_20261017/b1_online.py` | 单步连续 Feature 预测机制 | 历史曾达原 Target 约 1.506×，但慢于旧 PF；可在主方法复测后作为成本较低的补充。 |

Medusa/Hydra 的另有多组权重位于 `profiling/baselines_5way/`。它们是同类头部的训练版本；推荐先用 step5 的 Medusa low-rank 与 Hydra conditional 两个代表，避免把训练版本误报为不同算法。LFQ 的 BOS-only 补测与主 11 法分开解释；LFQ 递归 P/F、B1-E 与标准 D8 Position 则是真正不同的候选机制，但有旧速度或接入成本限制。

## 证据边界

远端查询逐项确认了本表所有指定权重文件非空，并记录了其字节数；入口文件和函数名来自远端源码及现有项目记录。表内历史速度来自已有报告，主要覆盖单场景或小样本，不可与本轮多来源结果合并。此次没有载入权重、执行解码、训练模型或改动远端项目。

