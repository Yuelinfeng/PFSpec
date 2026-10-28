"""Build readable reports from measured round-two artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/project/artifacts/round2_20261017'


def main():
    a=json.loads((DATA/'A1_position_diagnostics.json').read_text())
    state=json.loads((DATA/'A0_state_alignment.json').read_text())
    ranks=json.loads((DATA/'A0_rank_order.json').read_text())
    samples=json.loads((DATA/'A0_ten_samples.json').read_text())
    text='''# A0：源码与状态核验

日期：2026-10-17。主工程：/root/workspace/GaussianGPT。新增实验：/root/autodl-tmp/gaussiangpt_round2_20261017。

## 结论

Medusa/Hydra 的 Position 由自身头产生。当前训练和在线均从最后一个 Position 的状态跨过历史 Feature 预测下一 Position，标签偏移相符。源码中“Feature 锚点”的部分注释与实际取样不同。低接受率不能仅归因于 Feature 网络形式。

## 数值证据

'''
    text+=f'- 位置 ID 与 rank 完全一致：{ranks["id_rank_identical"]}；位置数 {ranks["position_count"]}。\n'
    text+=f'- 已测上下文中，最后块 RMS 归一化状态与 lm_head 输入最大绝对差为 {state["max_normalized_vs_lm_input_abs"]:.8g}；两种状态输入 Hydra 后 q 的最大总变差为 {state["max_hydra_q_tv"]:.8g}。\n'
    text+=f'- 十个缓存/整段对照样本中，状态最大绝对差为 {max(x["cached_vs_full_hidden_max_abs"] for x in samples):.8g}，Target 分布最大总变差为 {max(x["cached_vs_full_target_tv"] for x in samples):.8g}。这些样本来自同一来源，不能替代全部缓存分支验证。\n'
    text+='''- 整段前向不会自动应用位置递增限制；计算离线 Target 分布时显式添加相同限制。首次遗漏该操作的诊断副本保存在 initial_unmasked_diagnostic，不用于正式统计。
- 现有验证循环使用提案时的完整 q、接受概率及正残差，并处理 P/F 类型、EOS 和缓存回退。本轮复用该循环；七条路径短生成另行验证。没有重新证明全部已有分支。

## 调用关系

```mermaid
flowchart LR
  A[真实 P/F 前缀] --> B[处理至末尾 Position]
  B --> C[Position 隐藏状态]
  C --> D[Medusa 自身 P1 头]
  C --> E[Hydra 自身 P1 头]
  D --> F[F1 与 P2 候选]
  E --> F
  G[待处理历史 Feature] --> H[Target 批量验证]
  F --> H
  H --> I[接受或残差采样并更新缓存]
```

## 资产

source_snapshot 保存实际运行源码；git_status.txt 和 git_diff.patch.gz 保存原工作区状态与差异。previous_evidence 含旧比较的逐条 JSON、训练记录及汇总。权重保留远端原路径，没有复制进文档包。

十个样本的已处理前缀、待处理 Feature、标签位置、草稿/Target top-5 和重合度见 A0_ten_samples.json。原训练代码、在线代码与行号对应见上一级路线评审文档。
'''
    (DATA/'A0_source_audit.md').write_text(text)
    text=f'''# A1：Position 条件与提案质量

日期：2026-10-17。已完成 {a['real_contexts']} 个真实上下文、576 个训练域上下文和 64 个验证域上下文。真实数据来自原 10 个开发来源、28 个可用来源/前缀组合，每个组合取末尾 16 个 Position 锚点。完整抽样位置见 A1_position_diagnostics.json。

## 固定上下文重合度

| 方法 | 真实前缀 | 训练域 | 验证域 |
| --- | ---: | ---: | ---: |
'''
    names={'medusa_current':'Medusa 当前状态','hydra_current':'Hydra 当前状态',
        'medusa_after_feature_diagnostic':'Medusa 改用 Feature 后状态，仅诊断',
        'hydra_after_feature_diagnostic':'Hydra 改用 Feature 后状态，仅诊断','ranker':'现有 Ranker'}
    for name,label in names.items():
        vals=[a['summaries'][s][name]['overlap']*100 for s in ('real','train','validation')]
        text+='| '+label+' | '+' | '.join(f'{v:.2f}%' for v in vals)+' |\n'
    text+='''
## 解释

位置支持集和状态归一化在已测范围内一致；这两项未解释低 P1 接受率。训练域与真实域差异较大，训练拟合能力和来源变化仍可能同时影响结果。

已处理历史 Feature 的状态带来明显更高重合度，支持继续测试历史 Feature 条件。该操作同时改变输入分布并增加 Target 前向；此时 Target 已能直接给出下一 Position 概率，不能将上述数字当成低成本草稿的收益。下一项 A2 使用已知 Feature 的 embedding，评价无需额外 Target 前向的条件提案。

这里记录固定上下文的分布重合度，不能替代在线接受率。Ranker 行采用点分布，且抽样集中在指定前缀附近；其 75% 左右的离线统计不与旧在线约 90% 的接受率直接比较。改用 Feature 后状态也不属于训练充分对照。

每行另含真实下一 P 的概率、top-1/top-5、熵、EOS、非法概率质量和 rank 间隔。当前训练/验证标签来自 Medusa 的完整轨迹划分。
'''
    split_path=DATA/'A0_trace_split_alignment.json'
    if split_path.exists():
        split=json.loads(split_path.read_text())
        text+=f'\n另已读取 Hydra 原轨迹加载逻辑，40 条序列及顺序一致：{split["sequence_order_identical"]}；使用相同划分种子 20261010，因此原训练/验证归属相符。详见 A0_trace_split_alignment.json。\n'
    else:
        text+='\nHydra 原训练的轨迹顺序仍需单独对应，避免将不同划分混作同一验证集。\n'
    (DATA/'A1_position_report.md').write_text(text)
    e=json.loads((DATA/'E0_rows.json').read_text())
    rows=e['rows']
    mean=lambda key:sum(r[key] for r in rows)/len(rows)
    text=f'''# E0：已有 EAGLE 的状态与精度核验

日期：2026-10-17。复用 e46_hprev_dual_20ep/last.pt 与原八条验证轨迹；FP32 重算 {len(rows):,} 行，无新增训练。

## 结果

- 保存张量类型：{e['storage_dtypes']}。参数加载、草稿输出与共享输出头运算均为 FP32。BF16 是原训练 AMP 设置，先前“BF16 checkpoint”的称呼不准确。
- 本次完整 softmax 平均重合度：{mean('overlap_softmax')*100:.4f}%。
- 同一批状态应用 temperature=0.9、top-p=0.9 后：{mean('overlap_top_p_09')*100:.4f}%。
- TF32 matmul：{e['tf32_matmul']}；matmul precision：{e['matmul_precision']}。

## token 边界

hidden_prev_raw 来自同一 token 的倒数第二个 Transformer 块，与“前一个 token 的最后状态”含义不同。实际监督为 source_index=t，已知 token x[t+1]，预测归一化状态 h[t+1]，再通过共享头预测 x[t+2]。

代码中的 P2F/F2P 标记隐藏状态从哪个 token 跨到哪个 token。P2F=1 对应提案 Position；F2P=2 对应提案 Feature。下表按实际提案类型命名。

| 实际提案 | source index 区间 | 行数 | softmax 重合度 | top-p=0.9 重合度 |
| --- | --- | ---: | ---: | ---: |
'''
    for key,g in e['groups'].items():
        typ,bucket=key.split(':');label='Position' if typ=='1' else 'Feature'
        text+=f'| {label} | {bucket} | {g["count"]} | {g["overlap_softmax"]*100:.3f}% | {g["overlap_top_p_09"]*100:.3f}% |\n'
    text+='''
## 范围与后续

已有策略等价与缓存数值一致性记录仍作为历史正确性证据；本次测的是 teacher-forced 提案质量，没有新增 EAGLE 原权重在线速度结果。表中 source index 为轨迹位置，不能称为独立在线前缀长度实验。

原 EAGLE 的草稿 RoPE 使用一维递增索引；Target 隐藏状态已包含原三维位置处理。这里确认实际实现，没有证明一维草稿位置方式导致性能下降。首个有效转移样本数量有限，不能据此评价任意长前缀下新建缓存的首步性能。

B1-E 另行实现共享 Position 的小型单步特征预测，使用两 token 因果块读取历史 Feature 与候选 Position；它覆盖特征预测机制，不代表完整 EAGLE 复现。递归、动态树和多层融合暂未加入。
'''
    (DATA/'E0_alignment_report.md').write_text(text)


if __name__=='__main__':main()
