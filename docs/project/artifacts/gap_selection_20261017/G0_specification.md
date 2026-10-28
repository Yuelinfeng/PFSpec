# G0：组件、数据与采样路径

**状态：完成。** 本阶段核实了运行输入、Position 提案规则、B1 间隔口径和两个解码路径。未训练新模型，未改动远端主工程源码。

## 固定运行输入

运行位置为 AutoDL 主工程 `/root/workspace/GaussianGPT`，实验脚本与结果在 `/root/autodl-tmp/gaussiangpt_gap_selection_20261017/`。主工程提交为 `031d9160622f73dae69324a25d656ce4eb086524`；开始本轮时工作区已有 4,176 条状态记录，本轮没有在主工程目录写入文件。

设备为 RTX 4090，Python 3.10.21、PyTorch 2.8.0+cu129、CUDA 12.9。实际 Target 生成使用 FP32、temperature 0.9、top-p 0.9。各权重的 SHA256、文件大小和修改时间记录在 [g0_asset_inventory.json](g0_asset_inventory.json)。

| 用途 | 远端文件 |
|---|---|
| GaussianGPT Target | `/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt` |
| VQ 解码器 | `/root/workspace/GaussianGPT/checkpoints/vqvae_vfront.ckpt` |
| Position Ranker | `/root/workspace/GaussianGPT/profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib` |
| 轻量 Feature MLP | `/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt` |
| Feature MLP 公共输出层 | `/root/workspace/GaussianGPT/profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt` |

## Position 候选与 gap

真实调用链为 `CausalRanker.propose` → `ids_from_gaps` 生成合法候选 → `choose_position` 评分。候选集合按 ID 排序；单候选直接采用，多候选使用 `FastHistRanker.predict_proba` 后取 `numpy.argmax`。这一过程不抽样；固定前缀和模型输入会得到相同的首个候选 P₁。Target 随机数只用于 Target 采样及接受／残差抽样。

B1 使用的 gap 是序列化 Position ID 之差：

```text
gap = candidate_position_id - previous_position_id
```

这不是三维欧氏距离。`utils/pos_tokens.py` 将 ID 解码为 `x=id//400, y=(id//20)%20, z=id%20`；gap 为 1 表示下一个序列化 ID，跨越 z 边界时不等价于空间中的单位距离。

从 B1 的 `paths.jsonl` 对 5,122 个候选 gap 全部重算，结果逐项一致；前 100 条抽样也全部一致。其中 3,970 条 gap 为 1，1,152 条大于 1，没有非正间隔。B1 中 5,078 条记录含完整 P₁/F₁/P₂ 候选，这一数量用于 Feature 标签，不等于 gap 标签总数。逐条抽样和完整计数保存在 [g0_static_audit.json](g0_static_audit.json)。

## P-only 与 PF-K2 的数据流

两条路径从同一份已提交前缀开始，并调用相同 Position Ranker 生成 P₁。策略判断放在 P₁ 生成后、Feature MLP 和本轮 Target 验证前。P-only 跳过 Feature MLP，Target 验证首个 Position 后按 Target 分布生成 Feature。PF-K2 使用 MLP 的 Feature 分布提议 F₁，并由 Target 批量验证已有 Feature、P₁、F₁、P₂；接受、拒绝、残差抽样和缓存回退仍沿用原解码器代码。

Feature MLP 没有独立 KV Cache。它读取当前已提交的 Target 隐状态 `h_prev`、前缀末尾 Feature 和提议 P₁。Ranker 预览态只在实际 Position 提交后保留；拒绝后会清除未提交的预览态。Target KV 在拒绝时按原路径截断，再对残差 Token 继续前向。

首版分流规则已落在隔离模块 [gap_gate.py](../../../../patches/gap_selection_20261017/gap_gate.py)：合法候选且剩余预算不少于 4 时，gap=1 使用 PF-K2，gap>1 使用 P-only。BOS 尚无已提交 Position、无合法候选、非正间隔、候选越界或剩余预算不足时，沿用 P-only 解码器的处理路径；无候选时由 Target 采样位置，EOS 仍由 Target 概率及既有停止逻辑处理。

## G0 结论与范围

间隔选择依赖已提交前缀及确定性 P₁，因此本轮的策略本身不会读取当前轮 Target 概率或接受结果。B1 的 56 个固定前缀、10 个来源属于已反复使用的开发数据，不能作为新来源确认集。

G0 已完成：运行资产已记录、Ranker 确定性已由实现确认、B1 gap 定义已逐项复核、特殊情况的回退路径已列明。真实模型上的长期速度收益仍待 G2/G3；本阶段没有据此声称新方法提速。
