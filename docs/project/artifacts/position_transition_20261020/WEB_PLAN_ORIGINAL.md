# GaussianGPT PF2X｜下一轮：Position-Conditioned Two-Step Transition

**状态：方案设计，尚未实施。** 适用于现有 GaussianGPT/Ranker/step12000 Feature MLP；本轮不续训 12k→192k、不重做 EAGLE、N2-S3、LFQ、树、KV 缓存、旧 Gap 与 Round02。原 Target、Position Ranker、生产 PF-K2 均冻结。新实验需 Codex 主代理独立批准 GPU 调用，三名原 gpt-6-luna/max 只在本计划边界内执行。

## 1. 问题与可证伪假设

当前源代码 `draft_a/pf_feature_unified.py`：`MLPAdapter.propose(prefix, p1, h_prev, rng)` 的输入为 **上一个已确认 Position P0 的 Target 最后块 Hidden** `h0=h_prev`、末尾**尚未处理**的 Feature `F0=prefix[-1]` 与 Ranker 提案 `P1=p1`。旧 MLP 一次映射输出4096类 Feature 提案。Target 在 `decode/_decode` 的验证调用 `tf([F0,P1,F1_draft,P2])` 中则依次得到 `hF`（处理 F0 后）与 `hP`（处理 P1 后），其中 `states[0]=hF`、`states[1]=hP`，而 `p_F1 = policy(logits[1])[8000:12096]`。这两个状态是已有验证计算的副产物，不是 Draft 在提案前能获取的免费输入。

**核心假设 H**：当前限制并非再训练不足，而是旧 MLP 将 `F0→P1` 两个 Target 条件变换合成一个直接分类任务。将其分解成两个轻量、受教师中间态监督的状态更新，在相同输入、近似相同参数/成本下，可改善 **相同 Ranker 条件** 下 post-policy Feature 分布的重合率，进而减少相同实际输出长度所需 Target 串行验证。若改善不足或延迟过高，即否定此特定假设并停止。

**重要限定**：学生 C 没有获取新外部信息，所有输入仍是 `h0、F0、P1` 和已知位置；分阶段方式改变的是**归纳偏置与监督**。不把中间 Hidden 当成推理时额外可访问数据。过去 EAGLE Feature Hidden 预测没有净加速，因而本次严格控制模型小、无 Draft KV、只有两次低秩更新，并与同输入容量对照比较。

## 2. 三个正式候选和一个教师参照

- **A / W1**：冻结原 `Ranker + 12k MLP`，完全不变。
- **B / Direct-Residual**：保持 A 的原 MLP logits；使用同一时刻输入 `h0、F0、P1、合法位置/4D 坐标信息`，以一个轻量残差头修正 logits。作为**参数量与训练额外效果对照**。
- **C / TwoStep-Residual（主研究）**：保持 A 的 MLP logits。先以 `F0` 更新压缩隐状态，再以 `P1` 更新，预测 `hF/hP` 的教师压缩表示，最后输出 4096 类低秩 logits 残差。**无新增 Target Forward，无独立 KV，无树结构**。
- **T / 教师（仅分析，不计加速）**：Target 实际验证中捕获的 `hF、hP、p_F1`；严禁在 B/C 在线提案时使用真实 `hF、hP`。

建议状态宽度 `d=128`、更新低秩 `r=64`、Feature/Position Token embedding 经**冻结训练集内确定的投影**缩为64维、logits 修正秩 `r_z=32`；B 隐层维数通过脚本计算调至 C 的参数数目±10%。若 CPU/GPU 微测表明大于允许成本，再统一缩到 `d=64, r=32, r_z=16`，但只能在训练前做一次固定选择，不能根据在线速度临时调参。

两阶段概念式：

\[
\begin{aligned}
  u_0 &= R_h(h_0),\\
  \hat u_F &= u_0 + g_F(u_0, E(F_0), c_F),\\
  \hat u_P &= \hat u_F + g_P(\hat u_F,E(P_1),\Delta xyz, c_P),\\
  z_C(f) &= z_{\rm old}(f)+[U_{4096\times r_z}v(\hat u_P,u_0)]_f,\\
  q_C &= \operatorname{policy}_{T=0.9,\,top_p=0.9}(z_C).
\end{aligned}
\]

注意旧 `z_old` 已 softcap；**加修正后直接做一次 temperature/top-p，不再二次 softcap**。`R_h` 仅用训练来源 Hidden 拟合并冻结；`uF*=R_h(hF), uP*=R_h(hP)` 仅在训练监督使用。训练主损失为冻结 Target **softmax 前的已 softcap logits 在温度0.9下的 dense distribution** 与学生 dense softmax 的 KL；辅以小权重（建议 `0.05/0.05` 起始固定）对 `uF/uP` 的正规化 Smooth-L1，以及 `1e-4` 的残差平方惩罚。Tune/线上分数必须使用实际 top-p 后的 `sum(min(p,q))`、按 Ranker 接受概率加权分数和实际时间。不能用 CE/KL 降低取代接受率。

## 3. 理论推进/成本收益门槛（只用于筛选，不当成实测加速）

最近 N2-S3 同轮旧 PF/Target 来源等权 **1.68158×**；到2×还需要候选相对旧 PF **1.18936×**，即其他条件相同时减少约 **15.92%** 解码时间。R1 开发诊断中 `L≈2.57 token/轮`、`alpha_F≈0.354`、Position 接受约0.93、Target 20.28ms/轮、其余2.29ms/轮。以条件简化近似

\[
L_{new}\approx2.57+2(0.93)(\alpha_{F,new}-0.354),\quad
\frac{S_{new}}{S_{W1}}\approx\frac{L_{new}}{2.57}\cdot\frac{22.57}{22.57+c_{extra}}.
\]

对应需要的 Feature 接受率筛选级估算：新增成本0/0.25/0.50 ms/轮时，约 **61.6% / 63.4% / 65.2%**。这些来自**不同批次与简化推进假设**，绝非统计置信下界。真实 Gate 使用同轮完整解码时间、实际 EOS、Position/P2 恢复及 CUDA 成本。早期改善不到 +8pp 就不值得继续；即使+8pp仍不足以说明可达到2×。

## 4. 数据与条件对齐

1. 先用远端真实来源清单核对：**Train ≥16 来源、Val ≥4 来源、既有开发10来源**分别隔离；只从 `train/val` 收集训练监督。测试是否有真正未用作本方案开发的 `confirmation_t`，与过去 Stage E/F/G5、历史训练来源按 Scene ID 与相关字段去重。不能未经证实标为“新场景”。
2. 使用原 W1 真实 Ranker 提案采集，包括被接受和被拒绝的 P1；仅在原本会调用 `adapter.propose()` 的普通完整窗口采集，避免把短窗口和回退分支错误当同分布。教师采样仍 T=0.9、top-p=0.9。
3. 每条记录包含 `scene_id、prefix_hash、prefix_len、rng_seed、P0、F0、P1、ranker可观测状态、h0、hF、hP、target_logits_F、old_logits_F、p_F、q_old、p_P1、P1实际接受标记、RoPE/位置索引、EOS/窗口类型`。`hF/hP` 必须来自同一次现有 `tf([F0,P1,F1_draft,P2])` Forward 的 `states[0]/states[1]`；即使 P1 最终拒绝也可将该候选条件作为教师数据。不得用验证后的 `F1` 或 Target `hP` 作为学生在线输入。对 P1 接受的条件误差可依实际 Position 接受概率加权；记录原始全量，不只截取接受样本。
4. 默认上限：Train 16来源×100行=1600；Val 4来源×100行=400；既有开发10来源×40行=400；**最多2400条**，不足如实报告，不复制填充。可沿用现有真实训练/验证轨迹；按来源分组，不能同 Scene 的 Token 切到 Train/Val 两边。以前 R1 的480条不含成对 `hF/hP`，不能假定其可直接用于本任务训练。
5. 数据采集进程只增加旁路 Hook/记录器。`CaptureLastBlock.run` 当前抓取的是**最终 norm 前、最后 Transformer Block 的输出**；与学生监督必须使用同样定义。对真实 Target FinalNorm/Feature lm_head 的重算应复原 `logits[1]`（原有 softcap/typed mask/Policy 顺序）；判定教师索引和缓存长度正确后才训练。

## 5. 分阶段执行（新预算；按 Gate 停止）

| Gate | 任务 | 资源上限（累计不跨 Gate 挪用） | 放行与停止条件 |
|---|---|---:|---|
| **G0 只读** | 阅读远端实际代码与来源Manifest；确认 `h0/hF/hP`、F0/P1 ID、Target typed logits、4D RoPE、P1接受、保存格式；搭建单一学生接口和 B/C 参数表 | CPU ≤20分钟；GPU 0 | 任意状态时序或来源隔离无法核实，即停止并索取最小源码/Manifest |
| **G1 教师数据与成本预探针** | 16-token 原 W1/采集重放；上限2400条 Ranker P1 条件教师行；小幅微测 B/C 的额外单轮 GPU 成本 | GPU ≤900秒（含加载/预热/写盘） | 学生输入泄漏、原采样序列/计数改变、教师 p 重算不符立即停止；主学生额外成本目标 ≤0.25ms/轮，>0.5ms/轮停止或仅训练一个预先固定的更窄模型 |
| **G2 受控训练** | A 冻结；B/C 两种头各用 seeds=17、29；AdamW lr=3e-4, batch=64, 每配置每seed最多1200次更新，选配置仅看 Val 来源的真实 post-policy overlap 与校准 | GPU ≤2400秒 | C 相对 A **Val来源等权接受重合至少+8pp**、相对同参数 B 至少 **+4pp**、Val来源均非负；不满足不进入在线；若训练前成本>0.5ms/轮，停止 |
| **G3 在线小矩阵** | 3既有开发来源×2前缀(64/256)×2新seed×4法(Target/W1/B/C)×256 Token，共最多48次；完整计时 | GPU ≤900秒 | C/W1 实际配对吞吐未达 **1.05×** 或 C 平均每轮接受提升与新增成本不符即停止；不得以 CE 改善替代 |
| **G4 开发完整同轮** | 10既有开发来源×2前缀×2seed×4法×512 Token，最多160次，六种或完全配平法序，记录完整长度/EOS与同满预算子集 | GPU ≤1500秒 | C/W1 来源等权未达 **1.08×** 或 C/Target 未达 **1.90×** 即停止；不临时改变模型结构或训练 |
| **G5 保留确认** | 经审查确认未参与本轮模型选参的来源，至少20来源；相同前缀/种子/生成条件配对 Target/W1/C，512 Token；另做未截断自然 EOS 输出、VQ/3D有效性 | GPU ≤1500秒 | 正式2×：来源等权2×且以来源重采样95%区间下界>2×，实际生成长度/EOS与3D有效性无无法解释的退化；否则如实报告未达成 |

**合计 GPU 上限7200秒（2小时），CPU只读准备上限20分钟。** 包含加载、预热、采集、训练、短重放、完整测量、保存；不是每个代理单独得到该额度；前一 Gate 未通过不自动允许使用后续额度。本轮没有调用任何 GPU。若运行时间接近上限，只保留完整的配对组，不用不完整组计算配对速度。

## 6. 最小代码接口及正确性

优先在既有补丁工作区（如 `patches/pf_position_transition/`）新增，不覆盖正式 `draft_a/pf_feature_unified.py`、Target 本体及权重：

- `capture_transition_teacher.py`：调用现有 `bench_gaussiangpt_pf_block2_cow.CaptureLastBlock.run()` 或旁路 Hook，采集验证时的 `states[0]/states[1]`、`logits[0]/logits[1]`、`h_prev` 和上述来源元信息；采集版对照原 W1 的 16-token 同种子 Token/P/F接受/Target调用一致。***若有完整 hidden snapshot 增大 CPU copy，采集时不计入正式性能评测，而在推理测速时不打开采集器。***
- `transition_model.py`：单一模型文件实现 `DirectResidual`、`TwoStepResidual`，统一接口 `proposal_logits(z_old, h_prev, pending_f_id, p1_id, position_info) -> logits_new`；只对 Feature `8000:12096`；模型 checkpoint 附 projection、source manifest、dtype、损失与版本。
- `train_transition.py`：按来源训练、验证；保存最佳 Val 真实 post-policy overlap、P1概率加权 overlap、密集 KL、辅助态误差。选择只看 Val；训练前冻结超参和种子。
- `transition_adapter.py`：最小封装当前 `MLPAdapter.propose()`，先运行原 MLP 一次得到**已softcap** logits，再调用 B/C 预测残差一次，执行唯一 `.9/.9` Feature policy；不修改 Target、Ranker、EOS/Position Mask。
- `eval_transition.py`：Target-only / W1 / B / C 配对完整计时，记录真实 Token、Target 调用、验证输入量、P1/F1/P2接受、KV rewind、采样、自然 EOS、场景来源与3D导出状态。

**严格正确性边界**：

- 单步 P/F 位置严格交替；Feature ID `8000..12095`、EOS `12096` 只属于 Position；Prefix BOS、padding及非法位置沿用原码。
- Target 始终提供真实 p；学生 q 完整4096维且归一、非负；接受 `min(1,p(x)/q(x))`，拒绝用归一化 `[p-q]_+`，沿用真实 RNG 与 Top-p 规则，不用 greedy 匹配偷换。
- 位置候选 `P1` 的空间 RoPE 坐标由现有合法位置映射获得；学生不得取 Target 对 P1 的未来 KV/Hidden。缓存长度与原 `_decode()` 的 `rewind()` 完全一致；P1/P2拒绝、EOS、末尾不足4 Token、prefix0/64/256 均覆盖。
- 先用 **残差全零的 B/C** 同种子16与64 Token 比对原 W1：实际 Token、候选/接受计数、Target输入长度和KV恢复轨迹应全一致。再测试非零学生的分布归一、离线数值与在线推理一致。若因不同浮点算子导致非零版本与旧 W1 的输出不同，这是预期，不得误判为质量恶化。
- CPU/GPU 带采集 profile 仅定位，正式总速度采用外围 CUDA 同步包围的 **完整实际解码 wall**，包括 cache分配、前缀预填、ranker、old MLP、学生、Feature policy、Target verify、回退和收尾；模型加载/预热单列。FP32 `highest`、TF32关闭、T/top-p均0.9、batch1、同GPU。

## 7. 子代理职责（最多三名）

1. **Luna-1：状态与概率**。唯一负责教师索引、Target hidden 重算、数据来源/泄漏检查和概率/KV正确性证明；交付 `ALIGNMENT.md`、source manifest和教师数据清单。
2. **Luna-2：唯一模型开发者**。独占 `transition_model.py / train_transition.py / transition_adapter.py`；冻结原12k MLP，训练 B/C，评估模型规模/实际增量耗时，交付源码、训练曲线和权重标识。
3. **Luna-3：评价与独立复算**。独占 `eval_transition.py`、来源统计、EOS/3D结果和Web审阅包；不得提前根据开发结果更换确认来源。

主代理负责文件合并、统一计时、Gate批准与停止，三名代理不得并行分别跑大矩阵，不得擅自延长 GPU 或训练预算。

## 8. 网页端可审阅交付

`GaussianGPT_PF2X_PositionTransition_Review.zip`：

- `00_READ_ME_FIRST.md`：Gate进度、实际 GPU CPU 消耗、权重及环境、是否停止。
- `DECISION.md`：A/B/C 的条件 overlap、额外 Draft成本、来源级在线速度、是否满足 Gate、未完成项。
- `ALIGNMENT.md`：`h0,hF,hP,p_F` 时间索引及 Target/GPT 代码对照、4D RoPE/KV边界。
- `source_manifest.csv`：Train/Val/Dev/Confirmation、去重键和污染风险。
- `training/*.jsonl`：每步损失、来源等权 post-policy overlap、选定权重与超参；不必打包 Target 大权重。
- `results/records.jsonl`、`results/source_speed.csv`、`results/paired_summary.json`：每个种子/来源/前缀/方法完整时长与输出、接受/验证、Target调用与EOS、共同满预算子集。
- `results/quality.json`：自然EOS长度和3D可解码性，未执行时明确 `not_run`。
- `source/changes.patch`、必要新增脚本、复现命令及错误日志。

**启动前需要的最小远端核对材料**：现行 `pf_feature_unified.py`、`bench_gaussiangpt_pf_block2_cow.py`、`collect_feature_teacher_from_s3.py`、Target最后块/最终norm相关源码、现有 train/val 与开发来源Manifest及权重路径；远端真实权重不上传。若与既有源码快照有差异，先在 `ALIGNMENT.md` 标明再调整适配器。

## 8.1 Codex 需实现并锁定的 CLI（示意命令，不是现成可运行脚本）

以下参数必须在 **G0** 确认当前服务器来源 manifest、loader 名称及权重路径后一次确定。脚本是本轮新增文件，Codex 先实现参数接口再运行；不得把占位路径当作已有路径。

```bash
# G1：旁路教师采集，不改原W1路径
python patches/pf_position_transition/capture_transition_teacher.py \
  --target checkpoints/gpt_vfront.ckpt \
  --feature draft_a/runs/feature_target_continue_20261014/selected.pt \
  --source-manifest <VERIFIED_SPLIT_MANIFEST> \
  --train-rows 1600 --val-rows 400 --dev-rows 400 \
  --temperature 0.9 --top-p 0.9 \
  --out patches/pf_position_transition/results/teacher

# G2：Direct与TwoStep分别两seed，统一冻结训练/Val与参数容量
for method in direct two_step; do
  for seed in 17 29; do
    python patches/pf_position_transition/train_transition.py \
      --method "$method" --seed "$seed" --updates 1200 \
      --batch-size 64 --lr 0.0003 \
      --teacher patches/pf_position_transition/results/teacher \
      --out "patches/pf_position_transition/results/${method}_${seed}"
  done
done

# G3/G4：同轮Target/W1/B/C，所有模型固定，以相同prefix/seed配对
python patches/pf_position_transition/eval_transition.py \
  --methods target w1 direct two_step \
  --source-manifest <VERIFIED_DEV_MANIFEST> \
  --prefixes 64 256 --seeds 39101 39102 \
  --budget 512 --temperature 0.9 --top-p 0.9 \
  --out patches/pf_position_transition/results/online
```

G3 使用上述接口中的开发来源前三个及 `--budget 256`，G4 使用开发全部10来源和 `--budget 512`；两者种子固定，正式训练完成后才可安排。每个 Gateway 由统一主进程监控绝对 elapsed 和真实任务数；如实际命令/路径更改，必须在交付里同时记录“计划与实际”。G5 的确认来源另设 manifest，先做去重，不自动从 `<VERIFIED_DEV_MANIFEST>` 派生。

## 9. 研究决策

- 若 C 明显好于 B 且额外成本很小，则位置条件两步状态转移是可归因的设计贡献，再追求2×。
- 若 B≈C，说明收益主要来自额外参数/训练或 logit 校正，而非双阶段机制；不把它写作显著结构创新。
- 若两者不胜 A，说明现有 `h0,F0,P1` 信息或教师监督仍不足，停止该支路；下一轮再考虑真正不同的信息来源（例如仅从已提交 Target KV 读出部分 Position-Conditioned Attention），需另立成本诊断，不能直接增大模型。

**现阶段没有执行任何新实验，没有取得2×。**
