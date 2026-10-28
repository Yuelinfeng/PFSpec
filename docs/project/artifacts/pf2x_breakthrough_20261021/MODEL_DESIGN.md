# PF 2×突破候选设计

## 结论

先评估 Target 注意力实现，再决定是否进入多分支残差解码。注意力路线保留原 Target 与原 PF 的采样和接受过程，直接问一个具体问题：四 Token 验证能否相对单 Token Target 前向更省时。它不需要提高接受率。若此路能带来净收益却仍未到 2×，再测量批次 2 的真实残差分支；批次 4 只作成本上限探针。

当前旧 PF 来源等权速度为 1.68158×，10 个开发来源的完整 512-token 对照中，PF 吞吐 123.7590 token/s、Target 吞吐 73.4555 token/s。旧 PF 每单位输出耗时约为 Target 的 59.47%；到 2×需降至 50%。因此 PF 路径总用时还要减少约 15.92%，等价于不增加用时的情况下，平均推进量需增加约 18.94%。这些数字说明接受率小幅上扬无法单独完成目标。

## 方向一：按 Target 块形状选择注意力实现

Target-only 每次通常处理一个输入 Token，PF-K2 一次会验证四个输入 Token。PyTorch 的 SDPA 默认调度、math SDPA 和显式 BMM+softmax 在这两种形状上的相对速度可能不同。实验实现仅在短解码块内替换注意力计算；长度大于 4 的预填继续走现有路径。采样、Target 权重、Ranker、Feature MLP、温度、top-p 与 KV 回退均保持原样，因此目标是降低成本，接受率预期不变。

已有微基准出现上下文长度交叉：T=4、缓存长度 64/256 时 default 约 57 μs、explicit_bmm 约 89 μs；长度 768 时分别约 116/96 μs；长度 2048 时约 279/89 μs。短前缀阶段候选较慢，长上下文阶段较快，因此单一微基准均值无法回答完整输出的净效果。首次完整解码采用 3 个来源、前缀 256、一个种子、512 Token，并固定 explicit_bmm；不根据单次输出临时切换实现。

这一方向只有在 PF 的 T=4 路径相对 Target 的 T=1 路径明显变快时，才会提高 PF/Target 比值。若两种路径同比例变快，绝对吞吐会上升，但相对倍率仍接近 1.68×。完整对照运行 `default`、`math`、`explicit_bmm` 中预先选定的一种实现，并比对真实完整 Target 与 PF 总耗时、输出 Token 和计数。比较 Target logits 之外，还需比对进入 Feature MLP 的 pre-final-norm hidden state。

实现入口为 [attention_override.py](../../../../patches/pf2x_breakthrough_20261021/attention_override.py)，解码器评估由 [evaluate_runtime.py](../../../../patches/pf2x_breakthrough_20261021/evaluate_runtime.py) 承载，注意力微基准由 [bench_sdpa_candidates.py](../../../../patches/pf2x_breakthrough_20261021/attention/bench_sdpa_candidates.py) 承载。Target 源码参考使用已保存的 [nanochat_gpt.py](../position_transition_20261020/source_before/model/gpt/nanochat_gpt.py)，实验不改远端生产实现。

## 方向二：P1 后的多 Feature 残差分支

现有 Target 前向已经得到 P1、F1、P2 和 F2 所需结果；若 F1 草稿被拒绝，后续 P2/F2 结果依赖错误的 F1，当前实现会丢弃这段结果。可以在同一个 Target 调用中并行验证多种 F1 候选：各批次共享同一段已处理 KV 前缀和 P1，Feature Token 各不相同，Position Ranker 为每支给出同一个 P2 提案。Target 输出每支条件下的 P2 分布与 F2 分布。最后只沿被精确采样选中的分支继续。

候选分支必须服从残差采样，不能把“任意一支通过”直接当作接受。令 Target 的 Feature 分布为 `p`，首提案分布为 `q1`。首提案拒绝后，残差分布为 `r1=(p-q1)+/R1`。第二提案 `q2` 只可依赖已经抽出的候选历史，不能读取 Target 的 `p` 后再构造。给定候选 `b~q2`，其接受质量为 `min(r1(b),q2(b))`；拒绝后从 `(r1-q2)+` 归一化抽样。于是“第二提案接受质量 + 最后残差质量”逐项还原 `r1`。更多分支可按同一递推继续：每步接受 `min(rj,qj+1)`，余量进入下一步。每个分支的 Target P2 接受与拒绝仍按该分支自己的 Target 分布处理；没有选中的分支 logits 不可改变结果。

R1 的 480 条真实 p/q 条件给出首拒后的第二提案接受率 β=15.22%；计入 P1 和该支 P2 接受后，新增连续推进事件点估计为 8.56%/轮。单次第二提案因此不能承担 2×目标。作一个仅供安排成本探针的外推：若后续每个残差阶段也能保持 β=15.22%，且 P2 通过概率与首个第二分支相同，则第三、第四阶段各再贡献约 7.26、6.16 个百分点；三支合计约 21.98 个百分点。按 2 Token/事件估算，平均每轮可多推进约 0.44 Token，约为当前 W1 平均 2.48 Token 的 17.7%。这还未扣批次计算、KV 复制和采样调度成本，略低于 2×需要的 18.94%推进增幅。β 在更深残差上尚无数据，不能把这个外推当作性能结论。

先实现批次 2，输入是共享前缀、首候选、第二残差候选和共同 P2 提案；输出是两支 Target 条件分布、每支 KV 状态及精确选中的分支。批次 4 仅在批次 2 前向成本有余量时测量。代码应放在研究补丁目录，例如 `tree_residual_decode.py`，复用已保存的解码参考 [bench_gaussiangpt_pf_block2_cow.py](../position_transition_20261020/source_before/bench_gaussiangpt_pf_block2_cow.py) 的采样与恢复逻辑；不要接入生产 Target 或默认解码路径。

## 既有失败为何没有覆盖这两条路线

Position Transition 双阶段模型学习的是 Position 状态变化。它相对旧 12k MLP 的 Val overlap 仅增加 0.008281 个百分点，提案额外耗时 0.465364 ms；它没有验证多 Feature 候选分支，也没有测量注意力形状差异。

空间 S3 在旧 PF 上把 Feature 接受率从 34.5251%升至 35.8153%，但在线速度降为旧 PF 的 0.972657×。这说明每轮增加的小模型和查询耗时会抵消接受收益；S3 既没有批次分支状态，也没有替换 T=1/4 注意力实现。

PF-TypedVerify 把输出组织方式改成分组执行，四 Token 配对中位耗时只下降 0.4875%，没有取得完整解码收益。它不包含两条不同 F1 输入的 Target 行，也没有使用批次维度复制 KV。

R1 的二候选概率证明覆盖精确残差权重，真实第二分支采样也覆盖了 48 个条件；但 Target 形状计时仅有两个不同上下文、一个来源，分支调用仍是串行 batch=1。它没有测试一次调用内的多分支 KV、稀疏 RoPE 或批次 attention。这些结果支持保留严格采样规则，并提醒必须先测批次成本。

## 最小实验矩阵

1. 注意力微基准：固定 FP32、TF32 关闭，覆盖缓存长度 64/256/768/2048、查询长度 1/2/4，比较 default/math/explicit_bmm 的设备与主机时长；另比对完整 mask 下的 Target logits、pre-final-norm hidden 与合法 Token 支持。
2. 首个完整解码：3 个已有来源、前缀 256、一个种子、512 个新 Token，交错运行 original_target、original_pf、candidate_pf、candidate_target。候选实现固定 explicit_bmm；各次保留完整输出 Token、Target/draft 调用数、接受数、KV 恢复数及同步外围耗时。
3. 只有首轮显示 candidate_pf 对同实现 Target 或原始 Target 有可复核净收益时，才扩大到 10 来源×2 前缀×3 种子×512 Token。最终同时报告来源等权速度比、每个来源、自然 EOS、共同满长度和原始逐次计时。
4. 若注意力路线仍不足以到 2×，才用既有 480 条 p/q 条件对残差阶段做离线分布统计；再以 Target 批次 1/2/4 比对单支、双支和四支的 KV/attention 成本。批次 2 的时延增量需低于其真实新增推进收益；否则停止树形路线。批次 4 还须显示足以弥合剩余倍率缺口的净收益，才进入完整解码。

## 可复用资料与未知项

可复用的主要资料在 [R1 诊断包](../pf2x_loop_20261019/round01/cost_probe_review_20261020/)：`results/pq/vectors.npz` 含 480 条真实 Target/Draft 4096 维分布，`results/pq/contexts.jsonl` 与 `w1_profiles.jsonl` 含前缀及实际调用数据，`results/narrow/branch_rows.jsonl` 和 `branch_states.npz` 含第二候选条件与 P2 结果；`PROBABILITY_PROOF.md` 给出离散残差递推的 CPU 精确计算。原 Feature 权重为远端 `/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`，模型接口、Position Ranker 和完整 PF-K2 参考的本地只读副本在 [position_transition source_before](../position_transition_20261020/source_before/)；近期结果分别在 `position_transition_20261020/`、`spatial_online_20261020/`、`pf2x_loop_20261019/round02/`。

尚未知的关键事项包括：显式 BMM 对真实 Target T=1 与 PF T=4 的差异收益；多分支 cache 是否能在现有 KVCache、稀疏 RoPE 和 typed mask 下维持逐行一致；批次 2/4 的显存与同步成本；第三、第四残差阶段的真实 β 和分支 P2 接受率；以及新解码器在未用于模型选择的来源上的速度与完整输出有效性。当前只有候选设计和历史证据，没有本轮线上速度或 2×结果。


## 完成后的结果入口

本设计后的独立确认、自然EOS与三维有效性已经完成，最终实现、数值、计时范围、原始结果与适用限制见[00_READ_ME_FIRST.md](00_READ_ME_FIRST.md)。固定原Target参照的新PF为4.519564×，相对原PF为2.613919×，同优化Target下为1.442534×。下方或上方较早阶段的未运行描述保留设计时状态，不代表当前完成状态。
