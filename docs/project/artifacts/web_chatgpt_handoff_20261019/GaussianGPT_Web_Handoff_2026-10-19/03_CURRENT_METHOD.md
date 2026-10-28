# 当前最强方法：实现与状态

## 1. 运行对象

真实运行主工程：远端 `/root/workspace/GaussianGPT`。本机 `/Users/lief/Documents/ChatGPT/GaussianGPT` 是文档与源码快照交接区，缺少完整运行工程、模型权重与数据。当前包不提供直接执行环境。

Target：`/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt`。
VQ：`/root/workspace/GaussianGPT/checkpoints/vqvae_vfront.ckpt`。
Ranker：远端工程 `profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib`。
最佳 Feature MLP：远端工程 `draft_a/runs/feature_target_continue_20261014/selected.pt`，选择步数 12000。

研究 EAGLE v3：`/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`。γ=5 候选在独立同名实验目录下保存。它们尚未替代最佳 MLP。

## 2. Position Ranker

FrozenLean 从最近两个 gap、当前网格相位、历史 gap 统计等产生少量合法 Position 候选；HGB/Numba 快速评分选一个候选。位置合法性沿用 Target 的扫描顺序与位置限制。

它是确定性单点提案，qP 对选中位置的概率为 1。HGB 的分类评分仅用于排序，不能直接当作 Position 词表上的抽样概率。当前实现也没有完整的 Position 候选概率分布。历史快速评分约 0.08041 ms/次，该值不含整个解码轮次成本。

## 3. FeatureStateDraft

输入为：上一个已提交 Position 的 Target 最后层归一化前隐藏状态 1024 维、上一个实际已提交 Feature 的 128 维 embedding、当前候选 Position 的 128 维 embedding。将隐藏状态转换为 256 维，再与两个 embedding 拼成 512 维输入，经残差 MLP 预测 1024 维状态，沿用原 Target 的最终归一化、4096 类 Feature head 和 softcap=15，得到 qF。

可训练参数 2,864,384。共享 Target Feature head 有 4,194,304 个元素，不计入 Draft 可训练参数。MLP 自身没有独立 KV。一次预测的线性计算约 5,505,024 次乘加，未包含采样和调度。历史单场景 CUDA event 前向约 0.480/0.498 ms，不能视为正式多场景的完整提案成本。

训练使用 400 条 Target 轨迹、418,021 行；验证 50 条轨迹、46,893 行。标签是 Target 在当前 Position 条件下的完整 Feature 分布。step3600 → step12000 的验证重合度 26.4062% → 26.7549%；对应续训的在线额外速度区间跨过 1。

旧 MLP 读取归一化前状态，EAGLE v2/v3 读取最终 RMSNorm 后状态，不能混用。也不能以当前真实 Feature 或未来 Target 隐藏状态作为免费的 Draft 输入。

## 4. 静态 PF-K2 的真实含义

K2 指最多两个 Position 提案。普通一轮实际候选为 P1、F1、P2，共三个 token。Target 待消费输入为 `[末尾实际F, P1, F1, P2]`，一次取得 P1、F1、P2 与额外 F2 的四行条件分布。它与其他实验的 γ、普通 K2 token 数不能直接混称。

一轮开始时，输出已经包含完整 P/F 对；Target KV 通常消费到上一 P，末尾 F 已存在于输出但尚待 Target 消费。MLP 读取上一 P 的真实状态、实际末尾 F、候选 P1。Ranker 用自己的位置历史提出 P1/P2，当前 P2 提案器不读取 F1；这个依赖选择允许表达为合法提案，性能仍由 Target 条件分布验证。

1. 构造候选 P1、F1、P2；所有候选都属于临时提案状态。
2. Target 消费末尾 F 和候选序列，得到对应条件分布。
3. 顺序验证 P1、F1、P2，首次拒绝后舍弃其后候选。
4. P1 拒绝：按剩余 Target 概率抽位置，补算实际位置条件下的 F。
5. F1 拒绝：保留已接受 P1 的 Target 状态，按剩余概率抽 F，舍弃 P2。
6. P2 拒绝：保留 P1/F1，另取实际位置及其 F；全部接受则使用已有额外 F2 分布。
7. 提交真实输出与匹配的缓存长度。预算、候选不足时退到较短提案或 Target。EOS 按合法 Position/EOS 步骤处理，立即结束。

在不涉及 EOS、边界与特殊退化的普通轮次，实际提交量为 `C = 2 + 2 * I(P1 与 F1 均接受)`。因此 F1 接受直接影响推进量；P2 接受主要减少恢复前向，单独提高它的收益有限。成本分析已经对此测量。

## 5. 接受采样

同一历史和合法词表上使用实际最终分布 p、q：候选 y 从 q 产生，按 `min(1,p(y)/q(y))` 接受。拒绝后从归一化的 `[p-q]+` 抽样。Position 单点提案的接受概率退化为 p(P候选)。Feature 的 q 经过实际温度与 top-p 处理，不能用原始 logits 或其他历史上的 q 代替。

固定历史下的一步期望接受率为 `sum min(p,q) = 1 - TV(p,q)`。在线 accepted/verified、固定历史重合度、整条路径到达概率是不同量；不能将不同历史的 P/F 平均接受率直接相乘。

树形扩展需要为每个分支维护自己的历史、Target 可见前缀、Draft 状态与位置编码。不同 Feature 历史即使得到同一个后续 Position，也不能直接合并其 Target KV。3D RoPE 和增量状态必须对应实际历史。

## 6. 源码入口与快照时间

- `source/motivation_20261016/draft_a/pf_feature_unified.py`：最佳 PF 主循环、MLP/GPT adapter。
- `source/motivation_20261016/train_feature_state_distill.py`：FeatureStateDraft 定义。
- `source/motivation_20261016/draft_a/feature_target_v2.py`、`feature_target_continue.py`：训练来源、监督和续训。
- `source/motivation_20261016/eval_lean_candidate_to_draft.py`、`gpt_position_spec_decode_smoke.py`、`bench_fast_hgb.py`：Position 候选与快速 Ranker。
- `source/eagle_20261019/speculative/pf_draft.py`、`pf_distill.py`、`pf_engine.py`：EAGLE 网络、条件蒸馏与可选缓存路径。
- `source/eagle_20261019/test_pf_eagle.py`：已有状态、采样、EOS 与增量一致性测试。

前者来自 10 月 16 日真实源码核对，后者来自本地 10 月 19 日镜像。它们分别对应历史方法与新研究接口，不保证是远端此刻的全部最新源码；缺失依赖应从主工程补取，不能假定本包可单独运行。
