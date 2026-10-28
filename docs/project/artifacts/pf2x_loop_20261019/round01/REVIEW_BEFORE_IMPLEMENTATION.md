# 首轮实施前审阅：精确采样成立，串行第二候选的成本机制待论证

2026-10-20，台北时间。审阅者为用户指定的gpt-6-luna/max子代理 `/root/pf2x_evidence_review`；主代理依据当前源码核对。仅阅读源码与方案，未执行新GPU任务，也没有首轮性能结果。

## 主要结论

连续残差采样的代数成立。对合法分布p和q1，首候选按标准接受规则处理；拒绝后目标分布为 `r1=normalize((p-q1)+)`。第二候选若从实际q2采样，以 `min(1,r1(y)/q2(y))` 接受，再从 `normalize((r1-q2)+)` 处理拒绝，第二阶段输出分布仍为r1，合成后输出分布仍为p。q2可以依赖首候选，但采样与接受必须使用同一个实际分布。原首候选确实被拒绝时，其残差目标质量为0，因此排除它不会丢失r1的正质量。

当前最需要网页重新论证的是串行第二候选的速度机制。现有W1在首F拒绝后，已经从精确残差抽到实际F，然后在下一轮用 `[实际F,P2,F2草稿,P3]` 继续完整PF-K2验证。计划中的惰性W2在第二F接受后，先额外执行 `[F1b,P2]`，以获得P2及后继F2分布。这个短前向可能把原本可以在下一轮与P3共同执行的工作拆开，当前看不到减少Target调用的机制。

该判断来自调度与源码分析，还没有计时结果。不能据此声称W2已经失败，也不能用第二候选接受数增加证明速度改善。需要先说明，相同实际输出长度下有哪些前向被合并、跳过或缩短，以及节省如何超过额外前向、采样和状态维护。

## 与当前源码对应

- `source_current/draft_a/pf_feature_unified.py:127` 的 `MLPAdapter.propose()` 只返回同次前向经temperature=0.9、top-p=0.9处理的q1；宽支持q2需要保留这次前向的原始logits。
- 同文件 `_decode()` 从第245行开始。第421–422行形成并验证 `[pending_f,p1,f1,p2]`；第445–453行处理首F拒绝，将KV保留到 `before+2`，追加p1和残差F，清除待处理P2，再进入下一轮。
- P2的Ranker候选可以复用：Position状态不依赖Feature值。但原 `logits[2]` 以F1a为条件；改成F1b后，该Target条件分布不能复用。
- 若沿用计划中的短前向，保留 `before+2` 后输入 `[F1b,P2]` 的状态关系合理。P2拒绝时还需丢弃P2缓存槽、保留F1b，再处理真实Position。EOS出现在Position槽时不得继续生成Feature。
- `source_current/bench_gaussiangpt_pf_block2_cow.py:61` 的 `truncate()` 会缩短KV长度并清除token尾部；`source_current/model/gpt/nanochat_gpt.py:587` 附近按缓存token重建空间RoPE条件。新分支必须重算对应Target条件，不能使用旧分支logits。
- `bench_gaussiangpt_pf_block2_cow.py` 实际位于工程根目录。它与统一解码入口同时存在；正式测速需要明确调用路径，不能在实现和测速时混用两套入口。

## 已补充的依赖

初次审阅使用六个现用源码文件。审阅指出policy与KV allocator定义缺失，随后只读补取 `draft_a/feature_acceptance_eval.py`、`collect_feature_teacher_from_s3.py`、`model/gpt/engine.py`、`model/gpt/rope.py`、`model/gpt/rope_config.py`，现在共11个文件。未改动远端文件；这些是当前源码副本，并非本轮代码差异。

## 请网页端决定

请依据上述真实调度回答一个问题：首轮是否应取消惰性W2的完整解码实现，先只做零GPU概率验证，以及原训练结束后的有限p/q覆盖与Target分支成本诊断，再凭明确的收益空间决定分支并行验证？

如果仍建议实现惰性W2，请给出与现行W1对齐的两轮或多轮具体token/KV调用轨迹，指出实际节省的工作及成立条件。若需要改变总体实施方向，请提供新的详细任务边界与停止条件。当前192k训练与486次评价未完成；所有新增GPU测量继续等待，不提前使用确认来源。
