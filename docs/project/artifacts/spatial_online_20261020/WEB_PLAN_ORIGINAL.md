# GaussianGPT：N2-S3 GPU 真实解码比较执行计划

**版本**：2026-10-20 / 限定实验（不含新的训练）  
**状态**：批准源码只读适配与有界在线比较；尚未执行本轮 GPU。  
**目标**：在同一 RTX 4090、原始 Target、相同 FP32 与采样条件下，测量已训练 S3 seed29 相对原 PF-K2(step12000) 与 Target-only 的真实完整解码表现。暂停 N3、LFQ、候选树、额外训练和调参。

## 1. 模型/数据冻结

三方法：
- **T** 原 Target-only：原始 `checkpoints/gpt_vfront.ckpt`，不使用 Ranker/Feature Draft。
- **W1** 原 PF-K2：同 Target、现行 Position Ranker、12k Feature MLP `/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`；保持现有位置提案、拒绝修正和缓存逻辑。
- **S3** W1 + N2 S3 seed29：额外加载且仅加载已有 `model_S3_seed_29.npz` 和 `condition_inputs.npz`；Position Ranker、MLP、Target、提案长度与验证路径均同 W1。

N2 交接包路径：`GaussianGPT_PF2X_N2_CPU_Review/raw/run01/experiment02/{model_S3_seed_29.npz,condition_inputs.npz,q_outputs.npz}`，以及 `source_final/{run_experiment.py,prepare_data.py}`。远端已报结果目录 `/root/autodl-tmp/gaussiangpt_spatial_embedding_20261020/run01/experiment02`；Codex 首先核实这些文件实际存在，缺少时从已有归档复制，不重新训练或 PCA 拟合。

**固定 N2 数学配置**：Feature local vocab 4096；Feature global ID `8000..12095`；EOS/BOS `12096`；20³ Position；PCA32 表 `pca_embedding32`；8 维 logits 统计使用 `stats_fit_mean/stats_fit_scale`；六面邻居固定方向 `(-x,+x,-y,+y,-z,+z)`；每槽 32 维 embedding + 3 相对坐标/19 + 1 年龄/历史对数 = 36 维；Mask 后两层 32 维 GELU、单头 AttentionPool、`context_out`、PCA32 类别点积除 `sqrt(32)`；已训练的残差倍率 `1.0`。参数 key 为 `slot_in__{weight,bias}`、`slot_hidden__{weight,bias}`、`attention__{weight,bias}`、`context_out__{weight,bias}`，共 3585 参数。**不可重算 PCA、重标定统计、改倍率、换 seed。**

**概率**：Target 和 Feature 都采用 temperature=`0.9`、top-p=`0.9`，FP32，`torch.set_float32_matmul_precision('highest')`，`torch.backends.cuda.matmul.allow_tf32=False`，`torch.backends.cudnn.allow_tf32=False`，禁用 BF16 autocast。原 Feature MLP 的 `raw_logits` 已有原定 softcap，不再二次 softcap。令旧 logits `ℓ_f`、冻结 S3 残差 `δ_f`，采样 q_S3 = `policy(ℓ+1.0·δ; 0.9,0.9)`，合法类仅 Feature 4096 项。Target p 继续使用原有 policy；按 `min(1,p_f/q_f)` 接受，拒绝后用 `[p−q]_+` 归一的残差抽样。Position EOS、单调合法 mask、4D spatial RoPE、KV 与 Ranker 恢复一概不变。

## 2. 最小只读源码适配与唯一开发入口

先只读查证现行服务器中的：
- `draft_a/pf_feature_unified.py`: `MLPAdapter.__init__/prepare/propose/rewind/close`，`decode/_decode`，具体 `adapter.propose(prefix,p1,h_prev,rng)` 的调用时机及计时范围。
- `draft_a/pf_ranker_eval.py`: `policy/position_state/validate_sequence/cache_for`。
- `bench_gaussiangpt_pf_block2_cow.py`: `CaptureLastBlock/truncate/residual_sample`，Target-only 匹配实现。
- N2 `source_final/run_experiment.py`: `_torch_model/_logit_q_statistics/_standardize_fit_stats/_feature_slots/_evaluate_model`；`source_final/prepare_data.py`: `committed_pairs/derive_context_features`。
- 现用多来源 Target/W1 测评脚本或其最近正式运行入口：来源 ID → 场景 → 固定 64/256 Token 真实前缀的转换，以及 Target-only 的真实采样边界。若远端当前版与归档不同，保留差异并以当前实际源码为准，不擅自重构。

**仅新增** `patches/pf2x_n2_online/{spatial_adapter.py,eval_n2_s3_online.py,validate_n2_online.py}`；尽量直接调用现有 `decode`，不改写生产 PF 模块或 Target、Ranker、Feature 权重。若现行入口无法注入 adapter，允许在独立实验副本中进行最小接口适配，保存补丁；所有旧行为保持默认，禁止全局猴子补丁影响三方法。

`SpatialMLPAdapter` 应沿用 `MLPAdapter` 的一次 Feature Forward 和 `h_prev`，但需在获取**已经 softcap 的旧 logits、进入 policy 前**加入 S3 残差。不要在原 `MLPAdapter.propose` 已输出的 q 上二次 softmax，也不要调用原 `propose` 造成重复 MLP Forward。一次提案只能调用一次原 MLP 和一次 S3 小网络。在线逐次构造 6 个槽、8 维统计和 q；所有检索、CPU/GPU 复制、归一化、S3 前向、采样与状态维护必须在解码计时内。禁止读取当前 F1 的 Target p、未来 Feature、尚未验收的 Token 或离线真实答案。

缓存按已知完整 P/F 历史维护：当前被采样且写入输出前缀的 pending Feature 虽可能尚未进入 Target KV，但属于已知历史；不能把“尚未提交到 KV”误认为“尚未生成”，也不能把 `p1` 或 `f1` 候选当成**已确认的邻居 Feature**。构建邻居时严格复现 N2 `derive_context_features` 的已知历史语义；如以增量索引优化查询，应在 Target 拒绝后回滚 Index，先从朴素可审计版本起步，不在本轮根据速度临时更换算法。

## 3. 前置一致性检查（通过才能正式测速）

1. **文件与配置**：校验所有输入路径、shape/dtype、3585 参数、PCA32/fit statistics 与 N2 原始工件；检查实际 Target Feature offset、词表、位置编码和数据集对应一致。确认当前 W1 无 BF16 路径启用。
2. **480 条保留条件 CUDA vs CPU**：从 N2 `prepared_data.npz` 加载已提交历史 + `raw_logits`；CUDA S3 模型逐批生成残差，经一次真实 online policy 得到 q；按 `vector_index/sample_id` 对齐保存的 `q_S3_seed_29_{fit,tune,hold}`。首先比对逐槽位置、local Feature ID、mask、相对坐标、年龄、PCA 值、8 维标准化统计、残差 logits 与 q。逐行验证非负、有限、sum 1。参考阈值：最终 q 的最大绝对差不超过 1e-5、逐行 TV 不超过 1e-4；因 CPU/GPU top-p 边界舍入导致例外时必须逐项定位，不能放宽阈值后静默继续。
3. **Zero-residual 重放**：adapter 照常进行邻居构建和 S3 推理，但将残差 multiplier 测试性设为 0，最终 q 必须直接复用旧 MLP 的原 policy；保持同一个 CUDA RNG 与提案顺序。至少覆盖前缀 64/256 各一例（建议包括有邻居和无邻居情形），同 seed 16 新 Token，对比 W1 与 gate=0 的逐 Token 输出、每类 proposal/verify/accept 计数、Target 调用、ranker 调用、回退及 EOS；任一不一致立即停止。**Zero gate 只用于测试**，正式 S3 固定 gate=1.0。
4. **非零残差最小烟测**：16 Token 的 S3 推理完成、合法 P/F 交替、Prefix/EOS 合法、无未来条件读取、Reject 后 cache `get_pos` 与 `len(prefix)-1` 一致；Target/MLP/Ranker 权重 hash 或参数摘要不变。W1 与 S3 不要求随机 Token 完全一致，只需精确接受校正正确。

原 Target 采用同一 `vanilla`/Target-only 采样路径，严禁为建立比较人为增加 Target 额外阶段。先预热三方法及长短前缀，**普通计时不插入逐阶段 `torch.cuda.synchronize()`**，在开始和完成边界同步；分项计时只做非正式参考，不能混进主要吞吐数据。

## 4. 180 次正式比较

**硬件**：同一 RTX 4090；同一进程中同时加载同一 Target、Ranker、MLP 和冻结 S3，避免不同权重或环境。Target-only 不使用 Ranker/Feature 模型参与解码；加载成本计入整轮 1800 秒总预算，但不计入每个方法的纯解码秒数，二者分别报告。

**来源**：N2/R1 `source_manifest.json` 记录的既有 10 个开发来源，不称独立测试。 **Prefix**：实际真实 P/F Token 64 / 256（另含 BOS）；严格复用同一来源/前缀采样方式。**新种子**：预定 `38101,38102,38103`；若与历史重复，必须在运行前一次性更换并写入 manifest，不能事后因结果改变。**生成预算**：最多 512 个新 Token，自然 EOS 时提前停止。**配对组数**：10 来源×2 前缀×3 seed=60；每组三方法 Target-only、W1、S3 各一次，总 180 次。

同一组方法轮换以消除运行顺序偏差：种子 index 0 顺序 T/W1/S3，index 1 W1/S3/T，index 2 S3/T/W1；每个运行前检查硬件与精度设置不变。所有单组输出必须按来源/prefix/seed 保留三种真实 token 序列、EOS、实际输出长度及计时；不同方法可由于分布合法抽样而产生不同 Token，不要求同 seed 输出相同。

**主解码 wall**：从每次实际 `decode()` 开始到 GPU 同步完成；包含 KV 分配、真实前缀 prefill、Ranker、原 MLP、S3 邻居检索/特征构建/推理、q policy、所有 Token 抽样、Target Verification、拒绝恢复、缓存/状态维护、EOS 与完整输出；模型从磁盘加载、预热单独计时并记录。若当前 `decode()` 自身已经计时且完整，不重复遗漏其前后额外 S3 工作；同轮三方法使用一致的边界。

**记录**：每次 `source_id,prefix_tokens,seed,method,weight_paths_and_hash,elapsed_wall_s,new_tokens,tok_per_s,terminated_by_eos,last_token_id,eos_12096_present,full_budget,position_proposed/verified/accepted,feature_proposed/verified/accepted,target_calls,target_input_tokens,rounds,mean_generated_per_round,cache_recoveries,ranker_calls,mlp_calls,s3_calls,neighbor_lookup_ms,device_info,precision`；不同路径无意义的统计标为 null。保留输出 `tokens.jsonl`，不要只给 EOS metadata。

**统计**：预先定义来源等权的配对速度比 `T/W1`、`T/S3`、`S3/W1`（以 tok/s 比值计算，并注明 output length 可能不同）；每来源先跨前缀与 seed 聚合配对 log speed ratio，再对来源取等权几何均值；固定 10 来源 cluster bootstrap、10,000 次，报告 95% 区间、种子及来源逐个结果。再单列三方法**共同输出满 512 Token**的组，按相同输出长度用 wall 时间比计算；报告组数/来源覆盖，对 EOS/长度不同的组分表解释。另报告总 Token/总 Wall 合并吞吐，但它与来源等权指标分开，绝不拼接历史 Target。仅在有真正独立来源的后续评价才谈正式 2× 确认；这次 10 开发来源只能提出开发阶段速度证据。

**自然 EOS/质量边界**：记录 EOS 是否真实在输出末尾 `12096`、正常长度、合法 P/F 与 Position 单调序；不得把 EOS 子计数替代实际输出。此次不自动追加自然 EOS 全生成、独立来源或 3D 质量测量，也不声称这些验证已经完成。

## 5. 总时间预算、停止条件与 Luna 分工

**单一 monotonic 墙钟时限 1800 秒**，从开始模型加载到结果保存与结束，不分配每代理独立额度。建议阶段上限（可提前完成，不转成新实验）：
- 0–250 秒：权重/数据加载、N2 参数恢复、CUDA 480 条 q 对照。
- 250–400 秒：同种子 16-token gate=0 重放、非零烟测、预热。
- 400–1750 秒：正式 180 次（每组三方法完成后写盘）；剩余时间不足完成下一个完整三方法配对组时，停止新增运行并保留完整配对组。
- 1750–1800 秒：指标校验、输出文件 flush、结果打包；如果总时间即将耗尽，优先安全终止与保全原始结果。

任何以下情形立即停止，不改超参数补跑：GPU/CUDA 或精度不可保持，checkpoint 映射不一致，480 行概率差超门槛且无法解释，gate=0 输出/计数不一致，出现未知未来信息泄漏、EOS/PF/KV 非法，崩溃或数据无法按完整配对保存，1800 秒硬期限到达。若仅一两个样本故障，可以保存失败记录而不是事后选择性删除；是否允许补运行须另行审批。不得把未跑完的 180 次称为完整矩阵。

**Luna/max 3 人（原有代理）**：
- Luna-1：**只读映射 + 正确性**。对照 CPU 模型、PCA32、统计特征与 480 条 CUDA q；gate=0 重放、P/F/EOS/KV 语义，输出 `CUDA_PARITY.md` 与差异分析，不拥有生产解码文件。
- Luna-2：**唯一代码作者**。独立 `spatial_adapter.py`，必要的最小调用适配；确保一次 MLP 前向、一次 S3 前向，完整成本被计入，保留原生产文件不变；输出 `IMPLEMENTATION.md` 与 `changes.patch`。
- Luna-3：**计时/评价**。负责 `eval_n2_s3_online.py`、固定配对次序、完整 1800 秒预算与逐行数据、来源 bootstrap、EOS 解析、报告；不得修改训练或 S3 logits 公式。

Codex 主代理统一管理 GPU 独占与是否进入正式阶段；原代理可并行只读准备，但 GPU 必须串行使用，预先锁定参数。无需任何新训练，不启动其他研究方向。

## 6. 交付

提交 `GaussianGPT_PF2X_N2_S3_GPU_Review.zip`，包含：
- `00_READ_ME_FIRST.md`：完成状态、运行时长、是否完成完整 180 次、使用的权重/seed/参数。
- `DECISION.md`：S3 对 W1 是否更快、相对同轮 Target 的开发阶段速度比、局限与下一步，不得由小幅变化夸大结论。
- `CUDA_PARITY.md`、`parity_480.json`、`replay_zero_gate.jsonl`、`IMPLEMENTATION.md`。
- `results/records.jsonl`（含所有 180 次或完整配对组）、`results/tokens.jsonl`（全部实际输出 Token）、`results/summary.json`、`results/by_source.csv`、`results/paired_full_budget.csv`、`results/eos_length.csv`、`results/bootstrap.json`、`results/timing_budget.json`。
- `code/spatial_adapter.py`、`code/eval_n2_s3_online.py`、`code/validate_n2_online.py`、`code/changes.patch`、复现命令/执行日志；静态的旧 checkpoint 不必复制入包，用真实路径及 SHA256 记录。

**复现命令约定**（下列是要实现的新实验 CLI，不是假装既有脚本已经支持这些参数）：

```bash
cd /root/workspace/GaussianGPT
/root/miniconda3/envs/gaussiangpt/bin/python \
  patches/pf2x_n2_online/eval_n2_s3_online.py \
  --gpt checkpoints/gpt_vfront.ckpt \
  --feature-checkpoint draft_a/runs/feature_target_continue_20261014/selected.pt \
  --n2-artifacts /root/autodl-tmp/gaussiangpt_spatial_embedding_20261020/run01/experiment02 \
  --sources-manifest docs/project/artifacts/pf2x_loop_20261019/round03/source_manifest.json \
  --prefixes 64 256 --seeds 38101 38102 38103 \
  --budget 512 --methods target w1 s3 \
  --precision fp32-highest --temperature 0.9 --top-p 0.9 \
  --require-cuda-parity --require-zero-gate-replay \
  --total-wall-budget-seconds 1800 \
  --output /root/autodl-tmp/gaussiangpt_n2_s3_online_20261020
```

Ranker、VQ、数据来源的现行路径由现有匹配评测入口读取并写入 manifest；若路径无法从远端现行代码解析，应在只读适配阶段要求最小补充（当前评测的加载入口及其模型/来源参数），不要猜测不存在的 checkpoint。

**判读**：本轮即使看到 2× 点估计，也只能称为旧开发来源上的候选结果；若 S3 比 W1 快但区间包含 1，则说明证据不足；若 S3 不快，则止步，不扩大当前 S3 的训练或临时调倍率。**GPU 不因剩余时间而自动启动其他实验。**
