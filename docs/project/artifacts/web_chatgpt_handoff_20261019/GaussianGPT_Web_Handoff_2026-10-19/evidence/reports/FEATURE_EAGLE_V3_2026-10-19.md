# Feature EAGLE v3：条件蒸馏与两处 Feature 递推

本轮依据用户提供的 `GaussianGPT_FeatureEAGLE_%E6%8E%A5%E5%8F%97%E7%8E%87%E6%8F%90%E5%8D%87%E4%BF%AE%E6%94%B9%E8%AE%A1%E5%88%92_v3.md` 执行。保留 1L-256、4 头的 Feature-only 模型、既有 Position Ranker、Target、精确接受采样、VQ 和渲染。改动集中在条件数据、Feature 分布监督和短递推训练。

主工程为远端 `/root/workspace/GaussianGPT`，Python 使用既有 `/root/miniconda3/envs/gaussiangpt/bin/python`。Target 为 `checkpoints/gpt_vfront.ckpt`，VQ 为 `checkpoints/vqvae_vfront.ckpt`。起始 Feature 权重是 v2 的 `training_extended/feature/selected.pt`。独立运行目录为 `/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/`。没有安装依赖、训练 Joint 或改写旧 PF 权重。

## T1/T2/T3 的实际含义

诊断使用十个已有开发来源，各取一个裁剪，前缀为 64/128/256 token。T1 使用真实 P 和真实 F，T2 使用同一历史上的 Ranker P，T3 继续沿 Ranker P / 当前 Draft 采样 F 递推到第二个 F。T2/T3 每个前缀取四个随机种子。Target 标签始终来自完全相同的候选 token 历史。

T2/T3 额外区分两种前缀状态：末尾 F 已由 Target 消费；末尾 F 已存在于输出中但尚待 Target 消费。后一种对应在线多数轮次的起点，Draft 要先根据真实的前一 P 状态和实际末尾 F 预测状态，再消费新的 P。训练也覆盖这个过程。

重合度使用 softcap=15、Feature 词表、温度 0.9 和 top-p=0.9 后的最终分布。KL 正反两个方向使用温度 0.9 的密集分布，未做 top-p；其数值不能直接解释为接受率。真实 CE 只在真实 P/F 样本计算。每个 F 另有条件抽样接受数；候选路径的到达计数在首次 P/F 拒绝处停止。这些条件诊断与后面的连续在线计数分别报告。

旧 PF 按其原接口使用最终 RMSNorm 之前的 Target P 状态、实际末尾 F 和候选 P。新 EAGLE 使用最终 RMSNorm 之后的状态。两者不能混用输入状态。旧 PF 仅比较相同条件下的 F1；它原有路径没有 F2 草稿。

## 训练前诊断

最终诊断入口为远端 `diagnostic_before_final/`，本地摘要见 [训练前诊断](artifacts/feature_eagle_v3_20261019/diagnostic_before_summary.json)。同前缀 Ranker P 与真实 P 的相同率为 73.3%。

| 条件 | v2 重合度 | v2 正向密集 KL | 旧 PF 重合度 |
| --- | ---: | ---: | ---: |
| T1，真实 P | 23.57% | 2.715 | 27.70% |
| T2，末尾 F 已消费 | 22.79% | 2.709 | 31.36% |
| T2，末尾 F 待消费 | 28.83% | 2.076 | 31.36% |
| T3 第二个 F，末尾 F 已消费 | 30.93% | 1.869 | — |
| T3 第二个 F，末尾 F 待消费 | 31.96% | 1.722 | — |

Target 状态经原 Feature 输出头得到的 logits 与直接返回的同位置 Feature logits 一致；全局 Feature ID 和状态时刻也经过实际因果模块验证。当前证据支持改善分布拟合；它没有支持“Ranker P 必然导致明显下降”或“第二个 F 必然更差”。F2 的条件历史与 F1 不同，数值不能用于独立断言深度的因果影响。

## 训练方法

继续使用 v2 的 19 个真实场景、36 个裁剪，按 16/3 个来源划分训练和验证。主数据取 32 至 512 token、间隔 16 token 的 31 种前缀，共 2,220 个条件样本：训练为 955 个真实条件加 955 个 Ranker 条件，验证为 155 加 155。来源清单见 [条件数据摘要](artifacts/feature_eagle_v3_20261019/conditioned_summary.json)。Ranker 条件没有借用其他位置的真实 F 标签；其 Feature 标签全部来自 Target 在当前候选上的密集概率。没有人为添加 EOS。

损失为 `KL + 0.1*真实F的CE + 0.1*(after_P与after_F的SmoothL1)`。末尾 F 待消费时，还对该实际 F 的预测状态提供回归监督。真实 CE 仅作用于有对应标签的样本。主数据初始验证损失量级中 KL 约 2.52、加权 CE 约 0.75、加权状态损失约 0.02，因此分布蒸馏是主要信号。

首批较稀疏的前缀覆盖完成 1,600 次更新；末批训练重合度为 71%，验证为 26.8%，故没有直接延长同一批训练。随后增加每个场景的候选前缀覆盖。`single/` 保留首批试验；主结果使用 `single_dense/` 和 `rollout/`。

单步训练使用真实/Ranker 混合批次。两处 Feature 阶段每 200 次更新，用当前模型从训练来源重新生成至多 128 条 Ranker P / Draft F 轨迹，并让 Target 重新给出同序列标签。离散 token 不求梯度，预测状态在相邻节点间停止梯度传播；当前节点正常更新。训练损失同时保留单步批次和两处 Feature 批次。验证用独立来源固定候选轨迹，不参与参数更新。

权重按最终采样分布重合度选择。单步使用混合验证重合度；两处 Feature 阶段使用单步重合度的 0.5，加固定验证轨迹 F1/F2 重合度各 0.25。原始损失、重合度、选择依据与权重均保存。

主单步阶段完成 1,600 次更新，batch=8，学习率 `1e-4`，选用第 1,400 步；验证重合度从 23.45% 到 27.69%，密集 KL 从 2.524 到 2.216。两处 Feature 阶段完成 800 次更新，学习率 `3e-5`，选用第 500 步。两阶段 AdamW 各自初始化，梯度范数上限 1.0、权重衰减 0.01。主训练总计使用 32,000 个 Feature 分布监督位置，含重复抽样；这不等于 32,000 条独立样本。

相对单步选用权重，两处 Feature 阶段的单步验证重合度为 27.69%→28.19%；固定两处 Feature 验证轨迹上的 F1 为 29.30%→29.45%，F2 为 30.96%→32.70%。该阶段有小幅改善，尚需由独立于训练更新的开发诊断和在线结果判断实际价值。选用权重及完整指标见 [权重记录](artifacts/feature_eagle_v3_20261019/selected_weights.json)。

## 训练后同历史比较

将 v2 在 v3 实际生成的候选历史上重新计算，得到下表；T3 的两版模型消费完全相同的前缀和候选 token。其 v2 F2 数值与训练前表不同，原因是候选 F1 已变化。不能混用两张表的 F2 数值判断训练收益。

| 条件 | v2 重合度 | v3 重合度 | 密集 KL，v2→v3 | v3 条件采样接受数 |
| --- | ---: | ---: | ---: | ---: |
| T1，真实 P，F1 | 23.57% | 27.87% | 2.715→2.190 | 4/30 |
| T2，末尾 F 已消费，F1 | 22.79% | 27.31% | 2.709→2.274 | 31/120 |
| T2，末尾 F 待消费，F1 | 28.83% | 35.61% | 2.076→1.530 | 52/120 |
| T3，末尾 F 已消费，F2 | 30.02% | 34.40% | 1.912→1.532 | 34/120 |
| T3，末尾 F 待消费，F2 | 29.98% | 35.66% | 1.850→1.390 | 37/120 |

T3 的 F1 与同状态的 T2 一致。旧 PF 在相同 T1/T2 条件的重合度仍为 27.70%/31.36%。v3 的待消费 F 情形已有分布改善，已消费 F 情形仍低于旧 PF。T1 真实样本 CE 从 6.948 降至 6.446。

条件采样在所有诊断节点执行，用来观察给定条件下的接受事件。它没有要求前序 Position 已接受，因此不能冒充连续在线接受率。路径采样则在首次拒绝处停止：v3 待消费 F 的 T3 中，F1 为 35/88，F2 为 11/24；F2 到达数很少。每个来源只有三个原始前缀，重复抽样也不增加独立来源数量。完整正反向 KL、条件计数和路径计数见 [训练后摘要](artifacts/feature_eagle_v3_20261019/diagnostic_after_summary.json) 与 [逐节点数据](artifacts/feature_eagle_v3_20261019/diagnostic_after_records.json)。

## 连续在线结果

三个既有开发来源、64/256-token 前缀、种子 34701/34702，每次最多续写 128 token，九条路径各 12 次，共 108 次主运行。主比较不插入逐阶段同步。所有方法本轮均生成 1,536 token，吞吐使用总 token / 总时间；预填、缓存分配、草稿历史重建、恢复和采样全部计入。原始数据见 [逐次记录](artifacts/feature_eagle_v3_20261019/online_records.json) 与 [汇总](artifacts/feature_eagle_v3_20261019/online_summary.json)。

| 方法 | F1 接受/验证 | F2 接受/验证 | 总 F 接受率 | token/s | 对 Target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target | — | — | — | 71.336 | 1.000× |
| Ranker-only，γ=1 | — | — | — | 96.712 | 1.356× |
| v2 γ=3 | 117/542 | — | 21.59% | 86.382 | 1.211× |
| v3 单步权重，γ=3 | 152/544 | — | 27.94% | 93.230 | 1.307× |
| v3 两处 Feature 权重，γ=3 | 182/518 | — | 35.14% | 97.545 | 1.367× |
| v2 γ=5 | 98/575 | 23/83 | 18.39% | 79.225 | 1.111× |
| v3 γ=5 | 136/520 | 30/116 | 26.10% | 83.827 | 1.175× |
| v3 γ=4 | 129/528 | 27/109 | 24.49% | 84.275 | 1.181× |
| 旧 PF | 184/498 | — | 36.95% | 115.628 | 1.621× |

γ=3 的 F 接受率增加 13.55 个百分点，吞吐为 v2 的 1.129×；三个来源的速度比均高于 1，分别约 1.121/1.165/1.102。单步权重到两处 Feature 权重的比较包含额外训练更新，未单独分离递推样本和更新次数的贡献。

v3 γ=3 / Ranker-only 仅为 1.009×，三个来源分别为 1.055/1.000/0.978，当前不能声称稳定优于只提 Position。v3 γ=3 / 旧 PF 为 0.844×，仍未超过旧 PF。保留 v3 γ=3 作为本轮研究权重，旧 PF 继续保留。

γ=5 的 F1 从 17.04% 到 26.15%，F2 从 27.71% 到 25.86%。F2 分母只有 83/116，且两版轨迹不同，当前没有连续在线 F2 提升的证据。相同历史的分布重合改善没有自动转化为更长窗口的净收益；本轮不推荐 γ=5。

v3 γ=3 的 P 接受为 678/768，v2 为 644/768；Ranker 参数未改变，差异来自实际到达历史。首次拒绝分别为 P1=71、F1=336、P2=19；v2 对应 110/425/14。每次候选验证的总生成推进量从 2.356 到 2.608；仅已接受候选数从 1.167 到 1.460，后者不包含替代 token 或额外 Target token。Target 调用从 788 减至 691，旧 PF 同轮也是 691 次。

一个 64-token 续写的同步分项样本如下，单位毫秒，只比较采用相同计时方式的新旧 EAGLE：

| 阶段 | v2 γ=3 | v3 γ=3 |
| --- | ---: | ---: |
| Ranker | 8.745 | 7.381 |
| Draft 历史重建 | 24.193 | 21.090 |
| 消费 P | 22.971 | 19.981 |
| 消费 F | 42.670 | 37.778 |
| Feature 分类 | 2.184 | 1.927 |
| Target 验证 | 456.846 | 396.621 |
| Target 恢复 | 91.195 | 52.885 |

分项是整次运行各阶段的累计时间，轮次数和轨迹已经不同，不能解释为单步算子本身变快。旧 PF 沿用原有 host/CUDA event 分项，另存在 [分项记录](artifacts/feature_eagle_v3_20261019/online_profiles.json)，不与上述同步值直接混算。所有九条路径的分项与普通重放输出一致。

## 三维输出和端到端时间

两个已有开发场景、256-token 真实前缀、各一个种子，分别生成至自然 EOS。第二个场景采用相反的方法顺序。八次运行均完成 Gaussian 解码、有限数值验证、NPZ 与六视角 PNG，共 48 张图。下表每格为“完整 body token / 含导出总秒数”：

| 场景 | Target | v2 γ=3 | v3 γ=3 | 旧 PF |
| --- | ---: | ---: | ---: | ---: |
| `003ac11d…`，51701 | 1,892 / 24.502 | 1,878 / 20.051 | 2,904 / 28.936 | 3,442 / 27.366 |
| `030345aa…`，51702 | 2,690 / 34.766 | 2,558 / 29.579 | 2,170 / 22.351 | 2,698 / 20.818 |

同 seed 的不同提案器消耗随机数的顺序不同，生成长度和几何内容也不同。这些结果支持自然结束和完整导出，不能直接据两次总耗时宣布等工作量的端到端速度。完整数值见 [场景 0](artifacts/feature_eagle_v3_20261019/full3d_scene0.json) 与 [场景 1](artifacts/feature_eagle_v3_20261019/full3d_scene1.json)。首视角已查看，场景 1 可辨认墙面、地面和家具；没有进行独立质量指标评价。

为减少长度差异，再在场景 0、种子 51701 上使用相同 1,024 个新 token 预算。五条路径都得到 1,280 个完整 P/F body token，没有提前 EOS；因此这里是等长度的部分 chunk 输出。计时从准备好的 token 开始，包含自回归、VQ、Gaussian NPZ 与六张 PNG 写入，模型加载不计入。Gaussian 数量仍因具体位置和特征不同而变化。

| 方法 | AR 秒数 | 含导出总秒数 | Gaussian 数量 | Target 总秒数/方法总秒数 |
| --- | ---: | ---: | ---: | ---: |
| Target | 14.473 | 15.043 | 108,456 | 1.000× |
| v2 γ=3 | 11.449 | 11.951 | 115,003 | 1.259× |
| v3 γ=3 | 9.258 | 9.748 | 109,725 | 1.543× |
| v3 γ=5 | 11.383 | 11.865 | 110,356 | 1.268× |
| 旧 PF | 8.402 | 8.842 | 99,974 | 1.701× |

本单例 v3 γ=3 的端到端时间相对 v2 为 1.226×，旧 PF 仍更快。该组只有一个前缀和一个种子，保留首次渲染开销，不作为跨场景稳定收益。原始数据见 [等预算三维记录](artifacts/feature_eagle_v3_20261019/fixed_budget_3d.json)。本轮三维导出共 13 次、78 张 PNG；完整数组、tokens 和图像保存在远端对应输出目录，没有提交到 Git。

## 源码与复现入口

本地源码仍位于 `patches/eagle_pf_v2/`，与 v2 共用工程。

| 文件 | 本轮改动 |
| --- | --- |
| `speculative/pf_distill.py` | 条件标签、配对诊断、Feature 蒸馏和短递推辅助函数 |
| `train_pf_draft.py` | 在原训练入口增加 `--feature-v3 diagnose/prepare/train` |
| `speculative/pf_engine.py` | 仅新增 F1/F2 提案、验证、接受计数和首次拒绝槽位 |
| `eval_pf_draft.py` | 新旧 Feature 权重同轮比较，按深度汇总，保留旧 PF 路径 |
| `generate_pf_scene.py` | 接受指定 Feature 权重、旧 PF 与不同场景索引 |
| `test_pf_eagle.py` | 增加密集 KL、无效硬标签、递推与增量一致性、两种 Target 状态的验证 |

Target 核心、Feature Draft 结构及 Ranker 源码本轮没有修改。

```bash
python -m unittest test_pf_eagle -v
python train_pf_draft.py --feature-v3 diagnose --data <v2状态目录> \
  --init <Feature权重> --output <新诊断目录>
python train_pf_draft.py --feature-v3 prepare --data <v2状态目录> \
  --init <v2 Feature权重> --output <新条件数据目录> --v3-anchors <偶数前缀长度>
python train_pf_draft.py --feature-v3 train --data <条件数据目录> \
  --init <Feature权重> --output <新训练目录> --steps 1600 \
  --lr 0.0001 --hidden-weight 0.1 --validate-every 200
python train_pf_draft.py --feature-v3 train --v3-stage rollout \
  --data <条件数据目录> --init <单步选用权重> --output <新递推目录> \
  --steps 800 --lr 0.00003 --hidden-weight 0.1 --validate-every 100
python eval_pf_draft.py --training <v2训练目录> --feature-only \
  --feature-checkpoint <v3权重> --reference-feature <v2权重> \
  --single-feature <单步权重> --output <新在线目录> \
  --scenes 3 --prefixes 64 256 --seeds 34701 34702 --budget 128 --old-pf
python generate_pf_scene.py --training <v2训练目录> \
  --feature-checkpoint <v3权重> --reference-feature <v2权重> \
  --modes target feature_v2 feature feature_g5 old_pf \
  --prefix 256 --body-limit 1280 --output <新等预算目录>
```

九项核心测试通过；新旧 EAGLE、Target、Ranker-only 和旧 PF 的分项重放均复现普通输出。v2 γ=3/5、Ranker-only、Target、旧 PF 的本轮输出也逐 token 复现 v2 保存的同条件记录，各 12 次。训练/验证来源没有重合，十个诊断开发来源与这 19 个来源没有重合。

最终结论：本轮完成定向训练，γ=3 Feature 接受率和相对 v2 的真实速度均改善，保留 `rollout/selected.pt` 作为本轮研究权重；γ=5 没有更高净收益，旧 PF 仍更快。数据来源少、自然 EOS 未被监督、单一训练种子，以及开发来源已在历史工作中使用，仍限制独立泛化结论。继续扩大同一数据的训练步数缺少直接依据。
