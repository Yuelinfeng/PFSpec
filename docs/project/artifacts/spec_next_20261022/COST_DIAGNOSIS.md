# 恢复成本与提案分布诊断

## 结论

当前 W1 的恢复前向有可见成本，已记录的频率表明其空间有限。将另一批诊断的恢复事件中位耗时乘以历史长解码调用数，全部恢复约占 W1 外层耗时的 3.34%；其中树内 P1 拒绝约对应 2.69%。后一项还是每次拒绝都命中备用位置、且不付新增节点成本的乐观估算。跨样本中位数无法给出严格的时间上界。

短期最值得测的是同一轮内的 `rescue` 与 `reuse`。前者只在 Target 实际抽中的 P1 恰好等于预设备用值时省掉一次恢复前向；后者将上一处真实 Target 的完整 Feature 概率分布混入 MLP 提案。39 个上下文的分布重叠给出小幅改善信号，该初始诊断不提供生成速度证据。完整解码的实测结论另见 RESULTS.md。

## 数据与计算范围

所有数值均由本机已保存 JSONL/JSON 重新计算，未运行模型或 GPU。主要输入为：

| 输入 | 用途 |
|---|---|
| `docs/project/artifacts/recovery_model_20261021/dev_run01/records.jsonl` | 初始单轮阶段接受、恢复频率 |
| `docs/project/artifacts/recovery_model_20261021/components_run01/records.jsonl` | 恢复前向和树验证的事件耗时 |
| `docs/project/artifacts/feature_tree_20261021/main_run02/records.jsonl` | 完整解码 W1/W2 的吞吐、调用数和树接受深度 |
| `docs/project/artifacts/spec_next_20261022/raw/position_overlap_run01.json` | Target Position 分布与候选集合重叠 |
| `docs/project/artifacts/spec_next_20261022/raw/feature_reuse_overlap_run01.json` | MLP 与上一处 Target Feature 分布混合后的重叠 |

### 恢复事件的乐观成本估计

`main_run02` 的 W1 完成 9,216 个输出 token，外层耗时 27.010 秒，合并吞吐 341.21 token/秒。3,436 次树验证中，199 次 P1 未接受，占 5.79%；P1 接受后，1,157/3,237 次 Feature 尝试至少有一支被接受，之后 1,110/1,157 次 P2 接受。W1 另有 15 次 K1 尾部步骤。

W1 共记录 247 次恢复调用。另一个单轮计时样本中，暖态恢复事件中位数为 3.648 毫秒，树验证事件中位数为 4.538 毫秒。把该中位数套到长解码的调用数，恢复工作约为 901 毫秒、即完整耗时的 3.34%。这是跨样本估算，事件时间包含设备事件计时的限制，不能视作精确可省 wall time。

5 节点 Position 备用树只可能覆盖树内 P1 拒绝。若 199 次都成功用备用位置继续，在该中位数假设下可省约 726 毫秒、占 W1 外层耗时 2.69%。平均到每次树验证，额外节点、图建立与状态提交合计约需低于 0.211 毫秒才有机会打平；实际命中仅覆盖部分拒绝事件。K1 尾部不含备用 Position 分支，不能计入可节省量。

对初始上下文的 39 行数据，Position 点提案平均接受率为 0.690；Ranker 正类评分归一为 q 后，单次提案的分布重合度为 0.684；候选集合的 Target 总质量为 0.793。候选中的次高分位置额外覆盖平均 Target 质量 0.0908。若错误地把这个初始上下文数字当作完整解码命中率，单轮 break-even 预算约为 0.331 毫秒；同一中位数假设套到长解码 P1 拒绝频率后为 0.211 毫秒/次树验证，两个值均只能帮助安排实验。

W1 与 W2 的同轮长解码也说明接受率上升不能单独代表速度提升：W2 外层耗时比 W1 多 5.53%，Target 树节点多 43.99%，Target 输入 token 多 36.23%；Feature 分支命中率由 35.74%升至 42.89%，恢复调用却从 247 增至 338。额外验证工作抵消了接受率带来的 token 增量。

### 上一处 Target Feature 分布混入 MLP 提案

当前项目历史中未找到“每条解码路径保留上一处完整 Target Feature 概率分布，并将它作为未来 MLP 提案分布一部分”的既有在线测试。既有 Position 转移方法保存离散 Feature ID 或邻近位置；W1 用 MLP 分布直接提案。新代码 `patches/spec_next_20261022/feature_reuse_decoder.py` 在同一解码路径保存上一处 Target Feature policy，再按权重混入当前 MLP 的 q；每个候选仍由当前 Target 分布做精确验证与残差抽样。因此这项实验检验的是提案分布的变化，不能据分布重叠直接推断生成速度。

在 39 个上下文、10 个来源上，`q=(1-λ)·q_MLP+λ·pF_prev` 的来源等权结果如下：

| λ | Target 与 q 重叠 | P1 接受后的联合覆盖质量 |
|---:|---:|---:|
| 0.00 | 0.2562 | 0.1817 |
| 0.25 | 0.2785 | 0.1906 |
| 0.50 | 0.2852 | 0.1904 |
| 0.75 | 0.2808 | 0.1849 |
| 1.00 | 0.2665 | 0.1742 |

λ=0.25 时，按 P1 接受质量加权的覆盖增加 0.00891；10 个来源中 7 个上升、3 个下降。在这些相同初始前缀上，按每轮进展公式折算为约 0.01781 个额外 token。该值不能直接搬到完整生成轨迹，也不将跨样本的 P/F 平均概率相乘。实际混合还需承担向量相加与缓存管理成本，速度以完整解码为准。

### 延长验证链的成本条件

W1 在 3,436 个树验证轮次中，1,110 次走到 P2 接受，比例为 32.31%。若增加一对 Feature/Position 输出，在到达后总能接受并额外生成两 token，则按 W1 的 token 速度换算，每轮验证新增 Target 计算的平均预算约需低于 1.894 毫秒。若新增输出在可到达轮次中的接受率为 `s`，筛选条件变为 `1.894·s` 毫秒/轮；还应扣除图执行和缓存提交成本。

W2 已提供一个成本警示：Target 树规模增加约 44%，整体速度却下降。继续延长链只有在实测新增 token 收益覆盖每轮树增量时才值得。

## 建议的实测与判断条件

1. 对 `target`、`w1`、`rescue`、`reuse` 使用相同来源、prefix、seed 和 512 token 预算；各方法独立计时，首次图成本计入外层耗时。
2. 除来源等权吞吐倍率外，同时看共同满预算配对、无新增图配对、输出长度、EOS、恢复调用、Target 输入 token 和树节点数。计数相同不能替代 token 序列与合法性的核验。
3. `rescue` 应单独报告备用命中数/备用尝试数，并与 P1 拒绝次数对照。初始上下文的 9.08%备用质量只用于提出实验，不用于结果汇报。
4. `reuse` 应报告混合分布的实际 Feature 接受、恢复次数与新增向量成本。分布 overlap 上升但外层速度没有改善时，不扩大 λ 或增加训练。
5. 若后续测试 F2/P3，记录额外候选在所有轮次上产生的 Target 输入 token 与图成本，再和 `1.894·s` 毫秒/轮的乐观估算比较。

## 可复算脚本

- `patches/spec_next_20261022/cost_diagnosis.py`：汇总历史单轮、部件计时、长解码和两份分布重叠记录。
- `patches/spec_next_20261022/summarize_runs.py`：按任意 arm 组合计算同 pair 吞吐、来源等权 log-GM、来源 bootstrap、满预算与无新增图子集；图事件扫描会遍历所有 `*_delta` 字段，包括 `rescue_runtime_delta`。

```bash
python patches/spec_next_20261022/cost_diagnosis.py
python patches/spec_next_20261022/summarize_runs.py --overlap-json docs/project/artifacts/spec_next_20261022/raw/feature_reuse_overlap_run01.json
python patches/spec_next_20261022/summarize_runs.py --run <run-dir> --reference target
```

本文件保留主测前的成本推算，不包含三维端到端测量。所有成本门槛来自历史 W1 的测量或 39 个上下文的重叠诊断，结果应作为排序依据，不能当作新方法的速度结论。
