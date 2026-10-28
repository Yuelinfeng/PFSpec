# GaussianGPT PF2X — Round 02：P/F 类型感知验证快速路径与同轮 2× 判定

**状态：执行方案，等待 Codex 主代理按 Gate 顺序实施；截至本文件生成，没有运行新的 GPU 实验，也没有取得 2×。**  
**依据：** `GaussianGPT_PF2X_Round01_CostProbe_Review.zip`（64 文件）、Round01 历史论文级阶段 E（来源等权 1.7125×）、既有 192k 五节点评价完成报告。  
**版本：** 2026-10-20，Round02 计划。**不得调用上一轮剩余 GPU 额度作为本轮授权。**

## 0. R1 审阅与研究决策

已按顺序阅读 `00_READ_ME_FIRST.md`、`DECISION.md`、`FEASIBILITY.md`、`INDEPENDENT_REVIEW.md`、`TIMING_REVIEW.md`，并检查 `results/pq/vectors.npz`（三组 480×4096 float32）、`contexts.jsonl`、`w1_runs.jsonl`、`w1_profiles.jsonl`、narrow/wide `branch_rows.jsonl` 和 `shape_timings.jsonl`、两份 `analysis/*.json`、`source/changes.patch` 及 `source_current` 中的 PF/Target 代码。

- 60 次 W1，3,781 个实际新 Token，1 次自然 EOS（输出 5 Token，末尾 ID=12096），其余 59 次达到 64-Token 预算；4.8 百条真实 p/q，Feature 接受 420/1186=35.4132%，Position 接受 1610/1891=85.1401%。
- 480 个上下文覆盖 10 开发来源×前缀 64/256；第二候选各 48 条，两个模式共有同一组 48 个 `sample_id`，但提案抽样不同。narrow/wide 首拒后的二阶段接受重合度 β 为 15.2197%/14.3099%；联同 P₂ 条件事件的每轮 H 点估计 8.5644%/5.7112%，来源区间未包含上下文/分支抽样误差。**H 不是跨轮节省 Target 调用的实测概率，也不是严格上界。**
- 2/4-token 配对各模式 16+16 次计时，实际仅 **2 个唯一上下文、1 个来源**：narrow Host 17.3557/18.0881ms，wide 16.9467/17.4622ms。短前向不免费，不可外推到完整解码。R1 全流程 193.622 秒，新增训练 0，生产模型和解码器未修改；16-Token 同 seed 包装/普通轨迹一致。
- W1 插装 Profile：1471 轮平均产出 2.5913 Token/轮；平均 Target/轮 Host 约 20.28ms，非 Target/轮约 2.29ms（分项计时，**不可当成无插装总时间的精确分解**）。这表明减少 Target 调用的成本或减少一次调用的工作，更可能改变总体速度。
- 为辅助方向判断，另在 R1 的 **开发** p/q 上做了本地 CPU 离线试算（非正式实验）：以已提交邻居的 Feature 经验分布与 MLP q 混合，4 最近邻、混合系数 0.1 时平均重合度从 0.36440 到约 0.37446；只对 q 做 1.2 次幂再归一化为约 0.37109。两者均是探索性的局部约 +0.7～1.0 个百分点，不证明独立泛化或实际提速，更不足以单独成为冲击 2× 的主路线。

**决定：** 不实现惰性 W2、不直接实现树形 Attention、不重复 narrow/wide 有利事件、不重跑 12k→192k 续训、不重复 EAGLE/γ=5/持久 Draft KV/旧 Gap 规则、不开展未经证明的 W1-short。**选用现有 Position Ranker + step12000 MLP + W1-K2 的 P/F 分离框架，只对 Target Verification 的 P/F 类型特殊性做可关闭的快速路径。**若快速路径的实际节省不足，以数值结论结束 Round02，下一轮再评估真正更强的 Feature 提案，而不是为了 2× 扩大工程复杂度。

## 1. 真正的收益机制、目标和统计口径

### 1.1 主方案：PF-TypedVerify（P/F 类型感知的结构化验证）

现有一个 W1 四 Token 验证块通常为：

```
KV 已处理前缀 [..., P_(i-1)]，F_(i-1) 尚未处理
输入到同一次 Target Forward:
    [F_(i-1), P_i, F_i(draft), P_(i+1)]
预测/验证位置:
    logits[0] -> P_i (8000 位置类别 + EOS)
    logits[1] -> F_i (4096 Feature 类别)
    logits[2] -> P_(i+1) (位置 + EOS)
    logits[3] -> F_(i+1) (Target bonus)
```

**不改变** P_i/F_i/P_(i+1) 提案、采样、拒绝及残差法则；不增加新的 Target Forward；不牺牲 W1 的 4-Token 同轮推进。减时途径仅为：

1. 在 Target 的稀疏输出层，将两个 Position row、两个 Feature row **按类型分组投影**，复用各自冻结权重的视图/索引，避免当前 `GPT.forward()` 在每一个 row 中单独新建 Token ID、`index_select`、执行 `F.linear`、再向完整 logits 张量拷贝的重复操作。对 2-Token 尾窗同理；缓存允许时可复用一次性的索引，但不能缓存任何上下文相关概率。
2. 若第 1 步测试确实能省时，再测试 **同一验证块的设备端概率/接受判断合并**：将 P₁、F₁、P₂ 的合法掩码、softcap、temperature/top-p、接受比较在 GPU 上完成，尽量减少不必要的 GPU↔CPU 同步。必须保持按首个拒绝停止的精确分布及 EOS 规则；该步可独立开关，避免在第 1 步失败时仍重写解码器。
3. 如果上述两步有效，在同一输出长度下维持/减少 Target 调用次数和状态维护次数，降低 **每次既有验证块** 的真实时间。**这是计算成本机制，不是接受率提升或新草稿长度带来的收益。**

此方案的研究定位是 **PF 专用验证器的结构化实现**，配合现有 Ranker/MLP 才是完整 PF 方法。单纯 GPU Kernel 优化不能被包装成概率或预测理论创新；论文必须区分方法性结果与验证器系统优化的贡献。

### 1.2 必须计算的达标差距

正式速度：在同一来源、前缀、精度、采样与预算，计入 **KV 初始化、前缀 Prefill、Ranker、MLP、完整 Target 验证、回退、残差及同步、实际 Token 生成**：

`S_source = T_original_Target / T_PF`，主指标为 **来源等权的速度比几何平均**，以**来源**为重采样单位报告 95% 区间；每来源先对前缀/种子按预注册权重聚合时间比再计算来源等权几何平均。额外报告合并 Token/s、实际长度分布、完整 EOS 及三维有效性；**不把诊断中 101.87 token/s 的插装速度与历史 Target 混算**。

目标 `S >= 2.00×`；正式达标还要求独立来源的 95% 区间**下界超过 2.00×**，自然 EOS/三维有效性和正确性无无法解释的退化。点估计达到 2.0 但 CI 下界不足时记为“候选接近/点估计达到”，**不称可靠达到**。

历史正式来源等权 `S=1.7125×`；若保持 Target 端不变、输出数相同，PF 需要相对历史 W1 **约 16.79% 吞吐提升或 14.375% 总时间下降**（`1-1.7125/2`），但本轮采用**同批 Target/W1**重新计算实际差距。以 R1 插装构成为近似，若只有 Target 段变化，目标需要约 `14.375% / 89.8% ≈ 16.0%` 的 Target 段减时。这仅是规模判断，非实测优化预期。

**对照公平性：** 主表比较原始不改动 Target-only、原 W1、PF-TypedVerify；额外提供 Target-only 也使用同等类型感知输出头的**系统优化消融**，用于区分 PF 算法增益与通用内核增益。不能为了冲击 2× 对 Target-only 故意关闭其原有成熟快速路径。

## 2. 精确文件与函数范围

所有变更限定在已有 `/root/workspace/GaussianGPT/`，以现有 `patches/pf2x_round01/` 和 `draft_a/` 为基础；不另建完整平行生产系统。

| 文件 | 现有入口/拟增函数 | 必做事项 |
|---|---|---|
| `model/gpt/nanochat_gpt.py` | `GPT.forward()` 的 `logit_types` 路径；新增私有 `_project_typed_logits_grouped(...)` | 新参数 `typed_projection="original"|"grouped"` **默认 original**；分组投影时仍生成与原路径同型合法 logits，并调用同一 slot mask / monotonic Position mask。不得改 Transformer、RoPE、原训练 API、权重值。若 GPU 测试无收益立即回退，不永久加 API。 |
| `model/gpt/nanochat_gpt.py` | `_apply_slot_logits_mask()`、`_apply_monotonic_position_mask()`、`_get_rope()` | 只读核对语义；不能因优化省略 mask、EOS 或修改 4D RoPE。 |
| `draft_a/pf_feature_unified.py` | `decode()` / `_decode()` / `tf()`、`_StageProfile`、`CaptureLastBlock` 调用 | 通过可选 `verify_fastpath` 切换目标前向输出模式；原始 `False` 保持完全不变。仅在分组路径有效后探索批量 GPU 接受；保留阶段级 Target 调用数、输入数、KV 开始/结束/截短位置、输出 Token/自然 EOS。 |
| `bench_gaussiangpt_pf_block2_cow.py` | `CaptureLastBlock.run()`、`truncate()` | 保持 Hidden State 捕获位置（末块后、最终 norm 前）、原始索引不变；如增加可选参数必须不影响原 W1。 |
| `draft_a/pf_ranker_eval.py` | `typed_kinds()`、`policy()`、`validate_sequence()` | 复用唯一的原有 policy；不调整 Ranker 提案、不创建第二套采样策略。 |
| `patches/pf2x_round01/pf_typed_verify_probe.py` | 新增轻量诊断脚本 | 从现有 R1 `contexts.jsonl` 选真实上下文，warmup，交替配对 original/grouped 的 1/2/4 Token Target forward，分别输出 Host、CUDA Event 与端到端一致性。前缀完整时按真实 KV/slot 构造，而不是随意伪造四 Token 列表。 |
| `patches/pf2x_round01/eval_pf2x_round02.py` | 新增轻量评价入口 | 调用原 Target-only、原 W1、候选 W1，控制模型生命周期、预热、配对顺序、计时边界、实际输出/EOS、来源级汇总与 VQ 三维验证入口。尽量复用 `draft_a/feature_acceptance_eval.py`、`draft_a/pf_ranker_eval.py` 的现有加载/场景辅助逻辑。 |
| `patches/pf2x_round01/test_pf_typed_verify.py` | 新增 CPU/必要 GPU 测试 | 对比所有 P/F slot、前缀位移、EOS、拒绝、residual、4D RoPE、KV 截短恢复；验证所调用输出分布的精确性与概率正确性。 |

**最小源码确认点：** 当前上传包含上述主要文件的只读镜像，但没有远端项目所有训练/数据依赖及最新场景清单。Codex 先读远端实际版本（`source_current` 仅是交接快照）；需确认 `utils/pos_tokens.py`、`draft_a/eval_overlap.py`、`model/gaussian_gpt.py` 的加载流程，以及冻结的正式测试来源清单。缺少哪一个仅补相应文件或 manifest，**不用上传完整仓库/权重**。

## 3. 数学、KV 与 RoPE 的不可变约束

1. 在相同 prefix 和同一 Target 权重下，原与 grouped 输出 logits 应经同一 `15*tanh(logits/15)` softcap、槽位 mask、Position 单调 rank 约束及 `.9/.9` policy 后满足浮点容忍下的同分布。Position 合法词表 `0..7999` 加 EOS `12096`；Feature 合法词表 `8000..12095`；不能让 Feature 输出 EOS，也不能把 ID offset 混淆。
2. 分组只影响矩阵乘法调度：对当前验证窗口的 Position rows 与 Feature rows 分别投影，再按绝对槽位放回原始行。`typed_kinds(cache.get_pos(),len(tokens))` 是唯一输入类型依据；不可硬编码所有窗口都从 Position 开始。
3. 进入 Target 前：`cache.get_pos() == len(prefix)-1`。对 `before=cache.get_pos()` 的四 Token 窗口，接受 P₁、拒绝 F₁ 时，`truncate(cache,before+2,before+4)` 后 KV 对应 `prefix[:-1]`；P₁ 拒绝和 P₂ 拒绝分别按原路径保留 `before+1`、`before+3`；EOS 终止不继续写入 Feature。写 Cache 时 `token_cache`、每层 KV 的有效前缀、`cache_seqlens` 与可见末尾待处理 Token 必须保持原语义。
4. RoPE 使用 Target 的空间坐标 `oned_to_threed_indices(position_token,20)` 与 slot 维度；共享同一 P 的 F 继承相同空间坐标，拒绝回退后的下一次调用必须按实际修正 P/F 重建，而不是复用被拒分支的 RoPE。
5. P₁ 用确定性 Ranker 提案分布 `δ_P`，接受率 `p_P(P₁)`；F₁ 提案 `q`，接受 `min(1,p_F(F₁)/q(F₁))`，拒绝 residual `(p_F-q)_+ / Σ(p_F-q)_+`；P₂ 的拒绝与 EOS 分支完全沿用原路径。**优化任何阶段不得改变这个随机过程。**
6. GPU 侧合并判断若改变随机数调用顺序，跨实现同 seed 逐 Token **不保证**相同，但必须用固定外部均匀随机流进行逐分支校验，及多样本分布/TV/接受率统计；仅变动线性代数调度且沿用采样调用顺序的分组头，要求在足够远离采样边界的固定测试轨迹上逐 Token 对齐。
7. **不更换推理精度**：Target FP32，MLP 使用所选 step12000 当前正式配置（R1 为 `old_mlp_fp32`），温度/top-p=0.9/0.9；若生产标准不同，先报告实际配置并对齐，不能默默从 FP32 改 BF16 获得加速。

## 4. Gate 执行、资源预算及退出标准

### Gate A：仅 CPU / 静态正确性，GPU 0

- Luna-1 对照 `nanochat_gpt.py` 的原始 row-by-row `logit_types` 投影，写 grouped 行/词表/EOS 逆映射规范及 `test_pf_typed_verify.py` 的 CPU logits 代数测试。与训练/整模型无关的索引测试用小随机矩阵即可。
- Luna-2 制作 **只在启用时运行的** `typed_projection="grouped"` 轻量实现或对比原型；不修改默认分支、Ranker 与 MLP。不引入 tree mask、新自定义 CUDA kernel、`torch.compile` 或 CUDA Graph。
- Luna-3 基于现有 R1 计数重算需要节省的百分比，准备同轮评价及输出文件结构，不启动 GPU。
- **停止：** 新路径无法用当前 API 保持 logits/mask/EOS/KV 的原语义，则不进入 GPU，直接报告失败。

### Gate B：微观 Target 验证成本，最多 900 秒 GPU（含加载/预热）

- 选择至少 8 个真实 P/F 上下文、尽量覆盖至少 4 个开发来源、prefix 64/256；若不可用则实际记录。原/grouped 交替配对，每上下文每形状 1、2、4 Token 预热 2 次、计时 12 组。**同一上下文及形状固定输入；每次运行后按相同 keep 位置恢复 KV。** Host 和 CUDA event 各记，不相加；统计每来源中位数和同上下文的相对时间比。
- 验证同一 Prefix 上完整 Target p/q 分布及模型 Hidden State 的数值接近；核对 4D RoPE、EOS、P1/F1/P2/bonus、P1/P2 拒绝、尾窗 K1。
- **继续条件：** 4-token Target block 的真实 paired Host 计时在大多数来源取得有意义的稳定下降，且完整 W1 端到端的预测收益空间仍能接近差额。参考软阈值：分组投影单独提供 **≥8%** 的 4-token 块中位降时，或更低但逐阶段证据显示可消除较大同步成本。阈值是工程决策条件，**不是 2× 的证明**。
- **立即停止：** 概率偏差显著、EOS/位置掩码失效、KV/RoPE 不一致；或配对结果显示优化没有净收益且无可信的进一步削减路径。停止时不进入全解码实施，仅输出负结果。

### Gate C：按证明有用的最小改动完成解码接入，最多 1800 秒 GPU（含加载/预热）

- 仅在 Gate B 通过时接入 `verify_fastpath=True`，保留 `False` 原路径。同一 GPU 实测 original Target-only / 原 W1 / 候选 W1，另有 matched-fastpath Target-only 消融。先进行 16/64-token 闭环正确性测试；验证 `target_calls`、`target_input_tokens`、Ranker/MLP 调用数与原 W1 **应相同**（边界 EOS 或数值扰动须解释），否则属于算法变更而不再是原计划。
- 开发评价：**10 个原开发来源 × prefix 64/256 × 2 seed（35711,35712） × 4 方法 × 256 新 Token = 160 次运行**；每方法先 16-token 预热，轮换运行顺序。提前 EOS 不能用另一个方法继续生成填齐，也不能以短输出较快冒充相同长度速度。
- 如果原 W1 相对同轮 Target 仍在历史约 1.7× 量级，则 Gate C 候选需在开发来源上至少显示**接近或达到 2× 的可靠趋势**，且 Target 验证减时能够解释其总体收益，才有资格进入 Gate D。若候选的来源等权点估计 **<1.95×**、没有稳定改善或任何质量/正确性问题，**停止正式评测，不扩大 GPU**。若 1.95～2.00×，只记录潜力，不称成功，可提交网页重新选路线；若 >2.00×，继续独立来源。

### Gate D：独立来源、自然 EOS 和三维有效性，最多 4500 秒 GPU，严格以 Gate C 通过为前提

- 源清单由现有**独立、未用于这轮调参**的测试 split 加载；先读 manifest 锁定来源与随机种子，防止开发结果泄漏。目标 **至少 20 个独立来源（若正式历史 30 来源仍可用，则沿用 30）× prefix 64/256 × 2 seed（47111,47112）× 3 方法（原 Target-only / W1 / PF-TypedVerify）× 512 Token**。20 来源时是 240 次运行；30 来源为 360 次。额外 matched-kernel Target 消融做 10 来源分层子集，不混入主表。
- 自然 EOS：至少 8 个独立来源 × 两种前缀 × 1 seed × 3 方法，最大安全预算 4096 Token（若 EOS 未到，记 `censored=true` 而非自然 EOS）；记录完成时间、实际生成长度、真实最后一个 Token 是否 12096、位置单调合法性、VQ 解码/3D 导出成功与显式失败原因。全长和固定窗口结果分开，不按不同生成长度的总时间直接比较加速。
- 完整三维质量：在前述完整输出上记录 VQ 解码成功、Gaussian 参数合法/有限、scene exporter 成功；如果可用同历史流程的定量质量指标则执行，不能以“有 3D 文件”代替几何/视觉质量无损证明。精确目标分布推导与有限随机样本质量检查分别报告。
- **成功：** 独立来源 `S` 来源等权几何平均 ≥2，**来源 bootstrap 95% 下界 >2**；完整 EOS、三维有效性及采样正确性无无法解释退化。**停止/未达标：** 下界 ≤2、独立来源低于目标、任何正确性问题、长度差异导致无法比较，或 GPU 超预算；真实负结果完整打包，不追加新的自选实验。

**Round02 新授权总 GPU 上限：900+1800+4500=7200 秒（2 小时），包括模型加载、warmup、计时、短回放、正式开发及独立测试和自然 EOS。** 若完整自然 EOS 和 3D 评测会超时，按固定窗口主评测优先执行，完整 EOS/质量记为未完成，**不能宣称最终可靠 2×**。该预算是本轮新额度；前一轮未使用的剩余额度不纳入。

### 优先级与不做事项

先验证微观收益，再做完整解码，再做独立来源；各 Gate 不预支后续预算。任何 Gate 超时则留下截至当时的已完成数据，并停机。不能通过不断调开关、挑场景、改变精度/采样、忽略 EOS 来凑 2×。

## 5. 实际执行命令（新脚本 CLI 在实现时保持一致）

环境：`cd /root/workspace/GaussianGPT`，Python `/root/miniconda3/envs/gaussiangpt/bin/python`，RTX 4090。目标 `checkpoints/gpt_vfront.ckpt`、VQ `checkpoints/vqvae_vfront.ckpt`、Ranker `profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib`、Feature MLP `draft_a/runs/feature_target_continue_20261014/selected.pt`。开发数据 `.../pf_multiscene_pilot_20261015/stage_c/dev_crops.pt`；独立来源目录/manifest **须由现有工程确认**，不可伪造路径。

```bash
# Gate A：CPU 规范与单元测试，无 GPU
python patches/pf2x_round01/test_pf_typed_verify.py --mode cpu

# Gate B：原/新 Target 验证路径，保持相同缓存位置、相同 token/hidden
/root/miniconda3/envs/gaussiangpt/bin/python \
  patches/pf2x_round01/pf_typed_verify_probe.py \
  --target checkpoints/gpt_vfront.ckpt \
  --vqvae checkpoints/vqvae_vfront.ckpt \
  --contexts /root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/run01/collection/contexts.jsonl \
  --context-count 8 --min-sources 4 --shapes 1 2 4 \
  --warmups 2 --paired-repeats 12 \
  --output /root/autodl-tmp/gaussiangpt_pf2x_round02/gate_b

# Gate C：4 方法的同轮开发比较
/root/miniconda3/envs/gaussiangpt/bin/python \
  patches/pf2x_round01/eval_pf2x_round02.py \
  --split development \
  --crops /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt \
  --target checkpoints/gpt_vfront.ckpt \
  --vqvae checkpoints/vqvae_vfront.ckpt \
  --ranker profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib \
  --feature draft_a/runs/feature_target_continue_20261014/selected.pt \
  --methods target w1 pf_typed target_typed \
  --scenes 10 --prefixes 64 256 --seeds 35711 35712 \
  --budget 256 --output /root/autodl-tmp/gaussiangpt_pf2x_round02/gate_c

# Gate D：仅在 Gate C 达到进入条件后；实际独立 split 路径在 Gate A 确定
# 运行前由主代理将下列变量设为真实且冻结的测试 manifest 绝对路径
# export PF2X_HELDOUT_MANIFEST=/absolute/path/to/frozen_test_manifest.json
/root/miniconda3/envs/gaussiangpt/bin/python \
  patches/pf2x_round01/eval_pf2x_round02.py \
  --split test --source-manifest "$PF2X_HELDOUT_MANIFEST" \
  --target checkpoints/gpt_vfront.ckpt --vqvae checkpoints/vqvae_vfront.ckpt \
  --ranker profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib \
  --feature draft_a/runs/feature_target_continue_20261014/selected.pt \
  --methods target w1 pf_typed --scenes 20 \
  --prefixes 64 256 --seeds 47111 47112 --budget 512 \
  --output /root/autodl-tmp/gaussiangpt_pf2x_round02/gate_d_window

# 自然 EOS 与 3D 有效性：同一个 eval 脚本的专用模式
/root/miniconda3/envs/gaussiangpt/bin/python \
  patches/pf2x_round01/eval_pf2x_round02.py \
  --split test --source-manifest "$PF2X_HELDOUT_MANIFEST" \
  --target checkpoints/gpt_vfront.ckpt --vqvae checkpoints/vqvae_vfront.ckpt \
  --ranker profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib \
  --feature draft_a/runs/feature_target_continue_20261014/selected.pt \
  --methods target w1 pf_typed --scenes 8 \
  --prefixes 64 256 --seeds 47113 --budget 4096 \
  --natural-eos --validate-3d \
  --output /root/autodl-tmp/gaussiangpt_pf2x_round02/gate_d_eos
```

**注意：** 需要先真实设置并检查 `PF2X_HELDOUT_MANIFEST`；未发现独立测试 manifest 时不得执行 Gate D。`pf_typed_verify_probe.py`、`eval_pf2x_round02.py`、`test_pf_typed_verify.py` 是本轮待新增文件，上述参数定义是交付契约而非已存在命令；主代理必须实现 `--help` 和实际输出目录，不得将未实现命令写成已执行。可按时间预算在 Gate D 中减少样本数，必须先报告实际清单、删减规则及因此降低的结论级别。

## 6. 三名 Luna/max 子代理

- **Luna-1 — 数学/模型语义与正确性：** 负责 grouped row 的词表映射、softcap/mask/EOS/单调性等价规范，以及 CPU+GPU 错误轨迹/分布测试；不改生产解码循环。交付 `PROBABILITY_AND_TYPED_LOGITS.md`、`test_pf_typed_verify.py`、`tests.json`。
- **Luna-2 — 验证器与目标模块：** **唯一**负责修改 `model/gpt/nanochat_gpt.py` 及原 `pf_feature_unified.py` 的条件分支，默认保持旧路径，独立提交 `changes.patch`；负责 microbenchmark 实现与 KV/Hidden/RoPE 验证。不得引入树形 Attention、多候选、增加训练，不能在 Gate B 未通过时推进解码接入。交付 `IMPLEMENTATION.md`、probe 结果与差分。
- **Luna-3 — 测试/统计/交接：** 实现评价 CLI，核实正式来源清单与 Token 预算，执行获准 Gate，来源级 bootstrap、固定窗口/自然 EOS/3D 三类表严格分开；负责 ZIP/JSONL/metrics 证据。交付 `EVAL.md`、`RESULTS.md`、`DECISION.md`。

主代理核对远端当前文件与配置，持有 Gate 决策权；每通过一个 Gate 才启动下一个。开发 Agent 可在 CPU 并行分析，但不要三个子代理同时改同一源码或发起独立 GPU 实验。

## 7. 交付：网页端审阅压缩包

名称：`GaussianGPT_PF2X_Round02_Review.zip`，建议目录：

```text
00_READ_ME_FIRST.md
DECISION.md                  # 完成/失败 Gate、实际2×结论、未完成事项
RESULTS.md                   # 主表，来源区间，必要失败结果
CORRECTNESS.md               # P/F/EOS/KV/RoPE/分布验证
TIMING_AND_CALLS.md          # Target 输入形状与调用数、Host/CUDA、节省归因
QUALITY_AND_EOS.md           # 固定预算/自然 EOS/三维有效性分别记录
methods.json                # 权重、模型版本、精度、temperature/top-p
budget.json                 # 含加载/预热/验证/导出的统一墙钟预算与实际时间
results/dev_runs.jsonl       # 每次：scene/prefix/seed/method/output IDs/秒/调用/进度/EOS
results/test_runs.jsonl      # 没进入Gate D则明确不存在，不创建空结果冒充
results/paired_target.jsonl  # 同一上下文与输入/恢复位置的 original/grouped 成对测量
results/source_summary.csv
results/source_bootstrap.json
results/eos_runs.jsonl
results/quality_runs.jsonl
source/changes.patch
source/test_pf_typed_verify.py
source/pf_typed_verify_probe.py
source/eval_pf2x_round02.py
logs/commands.log
logs/errors.log
```

每项结论给出原始数据文件及重算脚本/命令，报告所有未进入 Gate 的原因。每轮只打包对审阅必要的源码 diff、实际数值和脚本，不上传大型权重、渲染缓存或重复镜像。审阅时先看原始配对数据及 `source/changes.patch`，再决定后续研究路线。

## 8. 备选：更强条件 Feature Draft，但本轮不立即训练

只有验证器优化不能产生足够节省时，下一轮才考虑 **Spatial-Conditioned Lightweight Feature Distribution**。在已提交的 P/F Prefix 中构造空间邻域条件 `N(P_i)`，将邻居 Feature、位置距离及原 `h_prev` 输入小型 Feature 条件模块；输出 `q(F_i | h_prev,F_(i-1),P_i,N(P_i))`，保持独立 `Ranker(P)` 和精确 Target 接受/拒绝。它的真正净收益机制是 **增大 F₁ 分布重合度 → 提升四 Token 一次验证完成率 → 减少同等输出长度的 Target 调用数**；但 R1 开发 p/q 的简单邻居混合只取得约 1pp 重合度增益，因此它目前只是训练假设。不能仅以邻居检索或 CE 降低就认为能达到 2×。若进入下一轮，须首先提出每 Token Draft 额外开销和至少需要的 Feature 接受率门槛，再决定是否训练。

## 9. 不可宣称的结论

当前没有同批正式 Target-only，尚无 Round02 完整解码实测；**真实 2× 未取得**。R1 概率/分支证据不能推出 W2 带来端到端收益，2/4 Token 形状来自极少上下文，48 分支上的来源 CI 不含抽样误差。任何 Gate 结果都要保持原始实际长度和 EOS，不能借改变 Prefix、GPU、sampling、precision、模型权重或统计口径得到表面加速。

**本轮执行授权：Gate A 立即，Gate B 在 CPU 通过后，Gate C/D 只按结果进入；总 GPU ≤7200 秒，新训练 0。未通过时明确结束，返回本对话重新决策。**
