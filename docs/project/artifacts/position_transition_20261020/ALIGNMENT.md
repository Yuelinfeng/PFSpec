# Position-Conditioned Two-Step Transition：源码对齐

## 状态定义

旧 MLP 在 `propose(prefix, p1, h_prev, rng)` 中接收上一个已提交 Position 的末块状态、待处理 Feature `prefix[-1]` 和 Ranker 提出的 `P1`。`h_prev` 由已有 Target 计算产生；prompt 预填结束时，解码器已处理到最后一个已提交 Position，末尾 Feature 仍待 Target 处理。[`pf_feature_unified.py:127-153`](source_before/draft_a/pf_feature_unified.py) [`pf_feature_unified.py:278-289`](source_before/draft_a/pf_feature_unified.py)

常规完整窗口把 `[F0, P1, F1_draft, P2]` 送入一次 Target 前向。`CaptureLastBlock.run` 返回末 Transformer 块输出，处于最终 norm 之前；因此 `states[0]=hF` 是处理 `F0` 后的状态，`states[1]=hP` 是处理 `P1` 后的状态。`logits[0]` 给出 `P1` 的 Target 分布，`logits[1]` 给出在候选 `P1` 条件下的 Feature 分布。[`bench_gaussiangpt_pf_block2_cow.py:35-58`](source_before/bench_gaussiangpt_pf_block2_cow.py) [`pf_feature_unified.py:413-445`](source_before/draft_a/pf_feature_unified.py)

Target 先抽取 `P1` 分布并以 `p(P1)` 做点提案接受；只有通过后，旧解码器才进入 `p_F1` 接受分支。采集器在同一已执行前向处保存 `hF/hP/logits[1]`，所以被拒绝的 `P1` 也留下候选条件的教师行。该行用于分析和监督时应保留 `p_p1`，不能当作实际生成轨迹中无条件出现的 `F1`。[`pf_feature_unified.py:424-445`](source_before/draft_a/pf_feature_unified.py) [`bench_gaussiangpt_pf_block2_cow.py:88-91`](source_before/bench_gaussiangpt_pf_block2_cow.py)

## 概率、词表与空间位置

Target 对稀疏类型输出先做 softcap 15，再应用 P/F slot 约束和推理时 Position 单调约束。P1 行支持 Position `0..7999` 与 EOS `12096`；Feature 行支持 `8000..12095`。Feature logits 的 `[8000:12096]` 已 softcap，保存后不再套一次。旧 MLP 的 `FeatureStateDraft.forward` 也在输出 logits 前按同一个 15 做 softcap；`MLPAdapter` 随后用温度 0.9、top-p 0.9 得到 `q_old`。[`nanochat_gpt.py:1056-1098`](source_before/model/gpt/nanochat_gpt.py) [`train_feature_state_distill.py:56-68`](source_before/train_feature_state_distill.py) [`pf_feature_unified.py:137-153`](source_before/draft_a/pf_feature_unified.py)

`p_p1` 是现有 `decide(float(p[p1]), rng, device)` 收到的值；`p_f_policy` 由真实 `logits[1,8000:12096]` 经相同 `.9/.9` 策略重算。采集器还从已捕获的 `hP` 批量计算最终 norm 与完整 `lm_head`，比较完整词表头和稀疏 Feature 行的策略差异；此步骤只重算输出头，没有再走 Target Transformer。[`pf_feature_unified.py:295-308`](source_before/draft_a/pf_feature_unified.py) [`nanochat_gpt.py:1054-1098`](source_before/model/gpt/nanochat_gpt.py)

真实 checkpoint 配置为 `rope_basis=position`、`dense_chunks=false`、`chunk_shape=[20,20,20]`、`chunk_order=xyz`、`learned_pos_embed=false`；`rope_config.py` 对该组合给出 4 维 RoPE。稀疏 Position ID 按 `x*400+y*20+z` 解成坐标，第四维来自 P/F token slot；BOS 使用单独坐标。RoPE 建造过程从 KV 中读取此前真实 token，按当前位置生成坐标；Target 缓存截断保留指定前缀，后缀槽位会清空。[`checkpoint_metadata.json`](checkpoint_metadata.json) [`rope_config.py:27-44`](source_before/rope_config.py) [`nanochat_gpt.py:542-609`](source_before/model/gpt/nanochat_gpt.py) [`bench_gaussiangpt_pf_block2_cow.py:61-72`](source_before/bench_gaussiangpt_pf_block2_cow.py)

## 来源和采集范围

`source_manifest.json` 指定 Train 16 个来源、Val 4 个来源、Dev 10 个来源及 Confirmation 20 个来源。G1 使用 Train/Val/Dev，分别最多 100/100/40 行；Confirmation 的 `max_rows=0`，本轮不载入。每条 crop 依 manifest 的 `crops_path` 与 `crop_index` 读取，并核对 scene ID。每来源用真实 64-token prompt、512-token 解码预算和 `collection_seed=39111`；不足行数照实记录，不补造样本。[`source_manifest.json`](source_manifest.json)

采集只记录旧 PF-K2 原本会进入的四 token 完整窗口：需要 Ranker 给出 P1、P2，且本轮还剩至少四个 token 预算。采集保留 P1 接受与拒绝的候选。短窗口、Position 回退与预算不足四 token 的末轮不纳入教师行。来源清单标明其中部分曾用于 D2、旧训练、Gap D1 或开发；旧 MLP 训练记录缺场景 ID，无法证实其训练来源与本轮来源完全分离。[`EXECUTION_STATUS.md`](EXECUTION_STATUS.md) [`WEB_PLAN_ORIGINAL.md`](WEB_PLAN_ORIGINAL.md)

## 采集实现及尚未运行的项目

新增 [`capture_transition_teacher.py`](../../../../patches/pf_position_transition/capture_transition_teacher.py)，公开 `load_runtime`、`collect_teacher` 和 `save_token_embeddings`。运行时复用一次 Target、step-12000 Feature MLP、原 Position Ranker 和原 PF-K2 解码器；采集包装器只复制已算出的隐藏状态与 logits，不改采样调用。脚本先做同 seed 的 16-token 原 MLP 与采集路径对照，核对输出 token、计数及 `rewind` 轨迹；再采集来源行、保存 `teacher.pt` 与原 Target `transformer.wte.weight` 的 CPU 副本。

本机已完成脚本 AST 编译、导入、CLI help 和 manifest split 数核对：16/4/10/20，计划行上限 1600/400/400。本机没有运行 Target、CUDA、来源解码或 900 秒 G1。实际 16-token 重放、完整头策略差、源行计数与目标 embedding shape 等证据仍需主进程在统一时间限内取得。此采集器不提供在线速度结论。
