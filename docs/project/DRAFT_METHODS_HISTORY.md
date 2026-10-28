# GaussianGPT Draft 方法历史记录

更新时间：2026-10-16

## 适用范围与阅读导航

本文梳理 GaussianGPT 的 Draft 方法演进、实验适用范围及当前方法定位。第 2–10 节保留早期方法说明和当时的小规模结果；这些数据不能替代 2026-10-15 至 2026-10-16 的独立多场景与完整三维流程结果，也不与其合并统计。

截至 2026-10-16，当前主路线为旧 step12000 轻量 PF + 原始 Target。阶段 E 的固定窗口比较支持该组合相对 Target 的提速；阶段 F 的完整流程比较也显示其快于 Target，但 PF 相对 P-only 的增量尚不能确认。完整设置、统计口径和质量限制见下方第 12 节及对应实验记录。

建议阅读顺序：先看 [项目交接入口](../../.codex/AI_START_HERE.md) 获取当前状态，再读 [多场景研究计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md) 与 [完整三维输出记录](FULL_3D_OUTPUT_2026-10-15.md) 了解正式证据；查逐项方法结果时读 [实验记录](EXPERIMENTS.md)，查训练与输入边界时读 [轻量 Feature 训练记录](FEATURE_TARGET_V2_2026-10-14.md)、[续训记录](FEATURE_TARGET_CONTINUE_2026-10-14.md)、[D4 记录](D4_JOINT_TRAINING_2026-10-14.md) 和 [D8 真实输入记录](FEATURE_D8_REAL_EVAL_2026-10-14.md)。本文的第 2–10 节用于回顾方法机制，不替代这些逐次结果。

远端源码、checkpoint 和逐次性能产物位于 Autodl 工作区；本地工作区以交接文档和实验摘要为主。

## 1. 共同问题定义

GaussianGPT 的 token 流不是普通文本序列，而是交错的：

```text
[Position_0, Feature_0, Position_1, Feature_1, ...]
```

在当前解码器中，条件关系是：

```text
P_i → Target-sampled F_i → P_{i+1}
```

因此当前 speculative route 只让 Draft 提出 Position。Target 在同一个验证 block 中计算 Position 分布并采样 Feature。Position 被拒绝时，使用 Target/Draft 的 residual correction，再回滚并重建对应的 Target cache；Feature 始终来自 Target。

对 Position proposal，Target policy 和 Draft policy 分别记为 `p_T^P` 与 `q_D^P`，理论接受概率为：

```text
alpha_P = sum_v min(p_T^P(v), q_D^P(v))
```

拒绝后从 `[p_T^P-q_D^P]_+` 归一化分布采样。所有方法都必须保持以下不变量：token grammar、BOS/EOS、3D RoPE、token-type support、Target/Draft cache 前缀和最终输出分布。

除非特别说明，端到端实验使用固定 Teacher/VQ-VAE、RTX 4090、CUDA、`temperature=0.9`、`top_p=0.9`，并在相同长度和相同 warmup 下比较 Target AR 与 speculative 路径。

## 2. S2：独立小型 Transformer draft

### 思想

用容量更小的 GPT 近似 Teacher 的下一个 Position 分布，以 Draft forward 换取 Target verification block 的摊销。Draft 和 Teacher 保持相同词表、token grammar、3D RoPE 和上下文长度，只缩小 Transformer 容量：

| 配置 | Embedding | Layers / Heads | 用途 |
|---|---:|---:|---|
| D4 | 256 | 4 / 4 | 最小容量 smoke |
| D6 | 384 | 6 / 6 | 中间容量对照 |
| D8 | 512 | 8 / 8 | 当前主要 draft |
| D12 | 768 | 12 / 12 | 接近 Teacher 的容量对照 |

### 接入方式

加载 D4/D6/D8/D12 checkpoint，调用现有 `PositionSelectiveSpeculativeEngine` 的 Draft update。每轮 Draft 只产生 `Q_D^P`，Target 负责验证 Position、采样 Feature、执行 residual correction 和 cache rollback。

当前主要可复用 checkpoint：

```text
profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt
```

### 测试和结果

在 256-token 的端到端测试中，D8 的 Position acceptance 约为 0.922，selective 路径约为 1.06×；另一组更严格的公平复测约为 1.15×。在 1024/2048/4096 的阶段 profiling 中，D8 双 Position forward 约为 6.5–6.7 ms，对比 Target 双 token verification 的 16.8–17.7 ms。

Position acceptance 已经较高，但 Draft forward 仍然是每轮固定成本。因此 D8 证明了 Position-only speculative decoding 可以工作，却没有达到理想 Target block 的收益上限。

### 当前结论

D8 是神经 draft 的 canonical 对照。D4、D6、D12 主要用于容量—接受率—成本对照；在没有完整 wall-clock 证据前，不能因为模型更大就默认它更快。

## 3. Position-selective speculative decoding

### 思想

不直接预测完整 `[Position, Feature]` pair，而是利用 GaussianGPT 的条件结构，只 speculative Position，把 Feature 留给 Target。这样避免 Draft 在不知道 Target Feature 的情况下提前预测下一个 Position。

### 接入方式

```text
Draft:      q_D^P(P_i | H_i)
Target:     p_T^P(P_i | H_i), p_T^F(F_i | H_i, P_i)
Correction: Position ratio acceptance + residual correction
Commit:     corrected Position + Target Feature
```

被接受的 Position 和 Target Feature 共同写入两侧 cache；被拒绝时，错误 Position 对应的 cache 必须丢弃，再以 corrected Position 重新取得 Feature 条件分布。

### 测试和结果

数学自检通过：理论 overlap=0.55，经验接受率=0.55172。端到端 D8 测试的 Position acceptance 约为 92%–97%，wall-clock 约为 1.06–1.15×，具体数值随 seed、对照脚本和 CUDA 状态变化。

### 当前结论

这是当前正确性已经最清楚的 speculative 算法路径，也是后续所有 Position proposer 的共同验证接口。它的瓶颈是 Target verification 和 Draft forward，而不是 Feature rejection，因为当前 Feature 没有进入 Draft acceptance。

## 4. S3：Feature-aware / on-policy distillation draft

### 思想

让 Draft 不只学习离线 teacher-forced token，还学习部署时会遇到的 Target-sampled Feature 条件前缀。训练中保留 Position CE，并增加 Feature CE/KL；on-policy 版本先由固定 Target 采样 trajectory corpus，再训练 D8 student。

典型损失为：

```text
L = L_position_CE
  + lambda_F * L_feature_CE
  + lambda_P_KL * KL(P_T^P || Q_D^P)
  + lambda_F_KL * KL(P_T^F || Q_D^F)
```

### 接入方式

训练完成后只替换 Draft checkpoint，解码仍使用 Position-only engine：Draft 提出 Position，Target 采样 Feature。Feature logits 在这一阶段主要用于诊断，不直接触发 Feature acceptance。

### 测试和结果

在相同 Teacher/VQ-VAE、D8 容量、seed=5、1024/2048 token 条件下：

| Draft | Position acceptance | Position overlap | Feature overlap | 公式预测 speedup |
|---|---:|---:|---:|---:|
| D8 canonical | 97.27% | 0.9720 | 0.3876 | 1.1407× |
| 256-step on-policy F-aware | 98.24% | 0.9862 | 0.7037 | 1.1424× |
| 256-step on-policy F-aware，2048 token | 97.85% | 0.9835 | 0.6107 | 1.1529× |

单 seed 的 256-token wrapper 曾得到 1.1369× wall-clock，但后续即时双 seed 对照中，canonical 与 F-aware 的均值约为 1.0420× 和 1.0494×，差异仍在运行噪声范围。短跑 `lambda_F=0.3` checkpoint 的 Feature overlap 只有约 0.35–0.39，不能作为有效联合 Feature draft。

### 当前结论

F-aware distillation 确实改善了 `q_F`，并小幅改善 `q_P`，但当前 decoder 不使用 Draft Feature，因此不能把 Feature overlap 的提高直接等同于加速。它是未来联合 `[Position, Feature]` proposal 的训练基础，不是已经证明的独立加速方案。

## 5. Feature shadow acceptance

### 思想

把 Draft 的 Feature logits 从“只有 overlap 诊断”推进到“可以估计 Feature acceptance”，但不改变最终输出。实现复用已有 `draft_update_2`：已提交 `[P_i,F_i]` 的第 0 行提供 `q_F(F_i | H_i,P_i)`，第 1 行继续提供下一 Position。

### 接入方式

在 Position 已接受后，从 Draft `q_F` 采样一个 shadow Feature，按 Target/Draft ratio 计算 Feature acceptance 和 pair acceptance；最终仍写入 Target 采样的 Feature，所以 Target/Draft cache 提交顺序保持不变。

### 测试和结果

使用 F-aware checkpoint、256 token、128 pair：

| 指标 | 结果 |
|---|---:|
| Position acceptance | 119/128 = 92.97% |
| Feature shadow acceptance | 60/119 = 50.42% |
| Pair shadow acceptance | 60/128 = 46.88% |
| wall-clock | 1.017–1.025× |

### 当前结论

该路径证明了 Feature proposal 的测量和共享 `q_F` 计算可以实现，同时避免额外约 5 ms/event 的 Feature probe。它仍然不是联合解码，因为 shadow Feature 没有写入最终输出。

## 6. 真实联合 `[Position, Feature]` draft

### 思想

在 Target verification 前同时提出 Position 和 Feature，由 Target 分别验证两类分布；Feature 被拒绝时执行 Feature residual correction，Position 被拒绝时回滚并重建后续 Feature 条件。

### 接入方式

联合路径必须处理三条状态分支：

```text
P accepted, F accepted  → 提交 Draft pair
P accepted, F rejected  → 保留 P，使用 Feature residual correction
P rejected              → 回滚 P/F，按剩余概率采样 P，再取得对应的 F
```

这要求新的 Draft 状态机、联合 cache 前缀验证和 `[P,F]` 分布 exactness 测试。

### 测试和结果

两次 256-token、128-pair 运行的统计完全一致：

| 指标 | 结果 |
|---|---:|
| Position acceptance | 121/128 = 94.53% |
| Feature acceptance | 51/121 = 42.15% |
| Pair acceptance | 51/128 = 39.84% |
| Target / joint | 72.725/66.804 tok/s；73.150/65.109 tok/s |
| wall-clock speedup | 0.9186×；0.8901× |

cache invariant、`py_compile` 和 50,000 次小词表 joint Monte Carlo 均通过，但额外 `draft_feature_probe` 的成本超过了联合 acceptance 的收益。

### 当前结论

联合路径在机制和数学上成立，在当前 Draft 质量和成本下却是负收益。它暂时作为 exactness/cache reference，不进入默认推理路径。

## 7. Full-token structured speculative negative control

### 思想

直接把文本 speculative decoding 的连续多 token 方式迁移到 GaussianGPT：Draft 每轮提出多个完整 `[P,F]` pair，Target 一次性验证 K 个 pair。

### 接入方式

使用 cached Transformer 生成 K 个完整 pair，Target 统一计算 block logits，并实现 token grammar、cache 截断、rollback 和最终 invariant 验证。

### 测试和结果

| K | temperature | wall-clock speedup | token acceptance |
|---:|---:|---:|---:|
| 1 | 1.0 | 0.533× | 0.487 |
| 2 | 1.0 | 0.571× | 0.398 |
| 4 | 1.0 | 0.434× | 0.213 |
| 2 | 0.1 | 0.795× | 0.630 |
| 4 | 0.1 | 0.891× | 0.563 |

所有测试都通过 sparse token stream 和 cache invariant，但 K 增大后接受率和速度都恶化。

### 当前结论

失败原因是结构性条件依赖：`P_{i+1}` 依赖 Target 才知道的 `F_i`。这不是简单增加 Draft 层数或 pair 数可以解决的问题，因此该路线保留为负对照。

## 8. S8：近零成本结构化 Position micro-draft

### 思想

用 Target trajectory corpus 中的 Position 序列规律替代神经 Draft。当前最佳表模型使用最近 5 个 Position gap 和当前 Position 的低 1 bit Morton/Z-order 特征，并通过最长上下文匹配和 backoff 生成合法的 `Q_D^P`。

它只提出 Position，Target 仍采样 Feature，因此完全复用 Position-only engine 的 ratio acceptance、residual correction、rollback 和 cache 逻辑。

### 接入方式

```text
structured_position_adapter(context)
        ↓
Q_D^P over legal Position/EOS support
        ↓
existing PositionSelectiveSpeculativeEngine
        ↓
Target verification + correction + Target Feature sampling
```

adapter 不运行 Draft Transformer，只维护 token/position cache 元数据。Counter 在初始化时预计算为 table，运行时只做 context 编码、查表和 support 映射。

### 离线测试

`h=5, feature_bits=1` 的结果：

- teacher-prefix top-1：85.04%；
- K=4 连续命中平均：2.58 个；
- 四个位置全部命中：54.48%；
- 表项数：1325；
- 纯 Python 查表：约 0.72 μs/query。

这些是 corpus proxy，不是 Target neural acceptance。归一化 proposal 与 residual correction 的 synthetic exactness 也通过：理论 overlap=0.45 时，100,000 次经验接受率为 0.44762，输出 L1=0.00340。

### 端到端测试

多 seed、同一场景、同一 checkpoint 的 12 个固定长度 endpoint：

| 目标 token 数 | speedup 均值 ± std | Position acceptance 均值 ± std | lookup 均值 |
|---:|---:|---:|---:|
| 256 | 1.396 ± 0.031× | 0.938 ± 0.040 | 133 μs/query |
| 1024 | 1.391 ± 0.035× | 0.910 ± 0.012 | 168 μs/query |
| 2048 | 1.360 ± 0.013× | 0.905 ± 0.005 | 206 μs/query |
| 4096 | 1.365 ± 0.011× | 0.911 ± 0.003 | 253 μs/query |

所有长度都生成完整目标长度，cache invariant 全部通过。已经生成了 `multiseed_scaling.png` 和 PDF 结果图。

### 当前结论

这是目前实际效果最好的 Draft 路线：在不运行 Draft Transformer 的情况下取得约 1.36–1.40× 的稳定 Position-only wall-clock 收益。它仍有三个边界：只覆盖同一场景、Feature 没有进入 proposal、host lookup/support mapping 会随长度增加。下一步应先 GPU/fused 化，再做跨场景验证和 `structured → D8` cascade。

## 9. Selective LM head 与 compact sampling：配套优化

这两个改动严格来说不是新的 Draft，而是利用 token-type support 优化 Target 验证和采样：Position 行只计算 Position/EOS support，Feature 行只计算 Feature support，并在采样和 residual correction 时压缩无效词表位置。

在固定 Teacher/D8/Target 条件下，完整路径 speedup 为：

| 目标长度 | speedup |
|---:|---:|
| 128 | 1.058× |
| 1024 | 1.066× |
| 2048 | 1.067× |
| 4096 | 1.084× |

它不减少 `target_verify_2` 调用次数，也不改变 acceptance 条件，但可以叠加到 D8、structured 或未来 target-native 路径。

## 10. 3.67× 理论结果的边界

此前的 3.67× 来自 Target K=4 block 的 pair-level 计算摊销：

```text
K=1: 22.76 ms/pair
K=4: 24.76 ms/block = 6.19 ms/pair
3.67× = 22.76 / 6.19
```

它只说明四个 Target pair 可以在一个 block 中并行计算，没有计入 Draft 生成、proposal 分布失配、acceptance、residual correction、rollback、sampling、同步和最终输出长度。因此它不是当前任一 Draft 的端到端 speedup，也不能与 full-token route 的 0.434–0.891× 或 structured Position route 的 1.36–1.40× 混用。

## 11. 当前统一裁决

截至 2026-10-16，当前主路线为旧 step12000 轻量 PF + 原始 Target。阶段 E 的独立多场景 FP32 比较和阶段 F 的全长真实场景输出均支持该组合相对 Target 提速；阶段 F 中 PF 相对 P-only 的区间仍跨过 1，因此不把增量收益写成已确认结果。方法选择、计时边界和质量限制详见第 12 节。

早期神经 D8、D4、查表 Position-only、联合 `[P,F]`、LFQ 和 cascade 实验仍保留为各自条件下的历史证据。它们的短窗口速度、接受率或固定前缀指标不与阶段 E/F 合并；未在相同场景、精度、提案结构和计时范围下比较的数字，不用于跨方法排名。

长期架构背景见 [`ARCHITECTURE.md`](ARCHITECTURE.md)，逐项实验记录见 [`EXPERIMENTS.md`](EXPERIMENTS.md)，采样与缓存路径边界见 [`LOGIC_VALIDATION.md`](LOGIC_VALIDATION.md)。

## 12. S9–S13 之后的研究进展与方法定位（截至 2026-10-16）

### 阶段 E：独立多场景固定窗口评价

阶段 E 使用 30 个正式测试场景、56 条唯一裁剪、三个 seed、五种模式，完成 2,160/2,160 次 FP32 评价。旧 step12000 轻量 PF 的微平均速度比为 **1.69675×**，来源组几何平均为 **1.71250×**，95% 区间 **[1.68044, 1.74402]**；P/F accepted/verified 为 **92.64095%/33.25815%**。微平均按全体 token 与耗时汇总，来源组几何平均让每个来源组等权，二者分别保留。

同一阶段的来源组几何平均为 P-only **1.43773×**、PF-D8 **1.46646×**、GPT-D8 **0.93815×**。这些是固定窗口 token 解码数据；阶段 E 未测完整三维生成耗时，也不单独证明渲染质量相同。测试模型和设置在读取方法结果前已定，结果没有用于重新选模。详见 [多场景研究计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md) 与 [实验记录](EXPERIMENTS.md)。

### 阶段 F：全长单 chunk 输出

阶段 F 完成 270 份真实前缀补全和 90 份 BOS 输出，全部成功自然 EOS，并保存 Gaussian NPZ 与六视角图像。真实补全从准备好的 256-token 前缀计时至 NPZ 与 PNG 写出，覆盖单个 20×20×20 latent chunk。PF/Target 端到端场景等权速度比为 **1.569× [1.318, 1.852]**；P-only/Target 为 **1.467× [1.266, 1.731]**；PF/P-only 为 **1.070× [0.912, 1.252]**。最后一区间跨过 1，现有证据不足以确认 PF 相对 P-only 的额外收益。

全 chunk PSNR/SSIM 约 11.41–11.67 dB/0.7302–0.7383，仅用于描述与真实 VQ 重建的差异。远端已有配对参考视图和逐样本生成图像；三个模式在 90 个场景×seed 组的 256-token 条件前缀完全一致。默认 VQ 解码不限制连续坐标偏移，参考重建自身也有 25/30 个场景的 Gaussian 中心超出名义 `[0,4]` 米范围；这解释了边界现象为何可能出现，但不能证明全部生成几何正常。可见区域图像质量、LPIPS 和多 chunk 接缝仍未测量，不能据全图指标推断质量等同。详见 [完整三维输出记录](FULL_3D_OUTPUT_2026-10-15.md)。

### 其他路线与停止原因

| 路线 | 已有证据与适用范围 | 当前定位 |
|---|---|---|
| S8–S9 查表 Position-only | 单一场景的固定随机轨迹矩阵覆盖 3 seed × 4 长度，12 个单元速度均大于 1×，均值 1.36149×；输入不是独立场景。 | 保留为低成本 Position proposer 的历史结果。它不等同于阶段 E 的 P-only 实现，不能直接把两组数值当作同一路线的扩场景复测。见 [S9 记录](EXPERIMENTS.md)。 |
| P-only 与轻量 PF | 小规模真实输入试验覆盖一个重建场景；step12000 轻量 Feature + Position ranker 为 1.4483×，D8 Feature + 同一 ranker 为 1.3561×，D8 联合 P/F 为 0.9595×。D8 的 Feature 接受较高，但额外前向后总吞吐更低。阶段 E 与 F 的正式结果另列于上文。 | 当前主方法使用旧 step12000 轻量 PF。PF 在阶段 F 相对 P-only 的增量仍不确定。小样本细节见 [D8 真实输入记录](FEATURE_D8_REAL_EVAL_2026-10-14.md) 与 [真实前缀记录](FEATURE_REAL_EVAL_2026-10-14.md)。 |
| 轻量 Feature 续训与直接 Target 监督 | 12 条旧测试轨迹上的 step3600/step12000 在线速度比为 1.7294×/1.7363×，配对区间包含持平；同一批前缀的 Feature 分布重合度有所上升。单一真实场景续训对照中，step12000 为 1.4654×，step3600 为 1.4811×，方向不一致。 | 保留 step12000 作为已预先选定的主方法权重；续训的分布匹配变化没有形成稳定的额外速度证据，不追加这一训练配方。见 [直接 Target 训练](FEATURE_TARGET_V2_2026-10-14.md) 与 [继续训练](FEATURE_TARGET_CONTINUE_2026-10-14.md)。 |
| D8 与 D4 神经草稿 | D8 联合 Gamma2/4/6 的单场景速度比为 0.9696×/0.9127×/0.8141×；D4 随机初始化联合 P/F 的单场景结果为 0.6328×/0.4918×/0.4006×。D4 与 D8 的训练起点不同。正式阶段 E 的 PF-D8 与 GPT-D8 数字见上文，属于另一批多场景比较。 | D8 保留为神经草稿及特征质量比较项；D4 当前配置不采用。单场景 D4 结果不能把差异单独归因于层数。见 [Gamma 对照](FEATURE_D8_GAMMA_2026-10-14.md) 与 [D4 训练评价](D4_JOINT_TRAINING_2026-10-14.md)。 |
| 真实来源适配与动态调度 | D2 的混合来源训练提高了固定真实前缀 Feature 重合度；在线速度比 B/旧为 1.00168× [0.99219, 1.01034]，B/A 为 1.00688× [0.99368, 1.02163]。D1 的 29/29 场景/前缀组合由轻量 PF 胜出，但没有记录逐轮置信度与历史动作收益。 | D2 不追加训练 seed；D1 不实现动态选择，保留旧 step12000 轻量 PF 与静态 K2。完整条件见 [多场景研究计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。 |
| S13b 真实联合 `[P,F]` | 三个 seed、256-token 评价的 P、条件 F、成对接受率为 0.9479/0.4231/0.4010，平均速度比为 0.9104×，三次均慢于 Target。 | 不按 S13b 配方单纯增加训练步数；Feature 提议成本未被接受收益抵消，保留为机制对照。后续转向多场景 P-only/PF 测量。见 [实验记录](EXPERIMENTS.md)。 |
| Structured→D8 cascade | eager 与 lazy 版本均完成单场景 256-token 路径；高置信阈值的 lazy 版本触发 61 次 D8 回退，速度比为 1.04909×，低于本轮无回退结构化路径的 1.37455×。 | 仅保留为状态流与回退路径参照；没有跨 seed、跨场景的额外吞吐证据。见 [实验记录](EXPERIMENTS.md)。 |
| LFQ F1/F2 联合候选 | 一组 24-seed BOS 1024-token 测试中，父/联合 K2 为 1.7307×/1.6940×，父/联合 K3 为 1.6610×/1.6067×。联合候选的部分 F2 条件指标提高，没有转成更高吞吐。 | 保留为不同模型和不同解码规则下的探索，不与 FeatureStateDraft 的 PF 结果拼接，也不替换当前权重。见 [LFQ 联合候选评价](PF_JOINT_EVALUATION_2026-10-13.md)。 |
| Medusa、EAGLE 与 Route-A | Medusa/EAGLE 是文献比较方向；本项目记录未完成对应的 GaussianGPT 多场景方法评价。Route-A 收录早期 S3 训练、全 token 负对照、选择性 LM head 和采样压缩等 profiling，不是阶段 E/F 的同条件参赛路线。 | 均作为方法启发或早期实验保留，不宣称已有当前主路线级别的独立场景证据。相关文献与适用讨论见 [研究方案汇总](RESEARCH_SOLUTIONS.md)，早期测量见 [实验记录](EXPERIMENTS.md)。 |
| Draft-A v2 | 500 条 Target 轨迹训练后，普通 Draft-A 的两 token 提议在 256-token 窗口约为 0.8536×–0.8740×；更长提议更慢。 | 未取得加速收益，保留为普通神经草稿的负向历史结果。见 [Draft-A 记录](DRAFT_A_ROLLOUT_V2_2026-10-13.md)。 |

本节把小样本与正式阶段分开列出。所有旧结果都限定在各自的场景、精度、前缀和计时范围内；阶段 E 的窗口速度、阶段 F 的完整等待时间、静态前缀分布指标与在线接受率不互相替代。
