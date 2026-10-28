"""Relate K2 committed tokens and Target calls to measured branch counts."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/project/artifacts/round2_20261017'


def main():
    path=DATA/'b1_online/20261018/records.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    methods=['old_pf','b0','b1_m','b1_h','b1_e'];result={}
    for name in methods:
        rr=[r for r in rows if r['method']==name];totals=collections.Counter();differences=[]
        for r in rr:
            s=r['stats'];ended=int(s['eos'])
            predicted_tokens=2*s['rounds']+2*s['feature_accepted']-ended
            predicted_calls=1+s['rounds']+s['position_verified']-s['position_accepted']+s['position_fallbacks']-ended
            if predicted_tokens!=r['generated_tokens'] or predicted_calls!=s['target_calls']:
                differences.append(dict(scene=r['scene'],prefix=r['prefix'],seed=r['seed'],
                    predicted_tokens=predicted_tokens,actual_tokens=r['generated_tokens'],predicted_calls=predicted_calls,actual_calls=s['target_calls']))
            totals.update(dict(rounds=s['rounds'],tokens=r['generated_tokens'],target_calls=s['target_calls'],
                feature_accepted=s['feature_accepted'],p2_rejected=sum(x==2 for x in s['accepted_lengths']),
                p2_accepted=sum(x==3 for x in s['accepted_lengths']),eos=ended))
        if not rr:continue
        result[name]=dict(runs=len(rr),**totals,mean_committed_tokens=totals['tokens']/totals['rounds'],
            feature_accepted_per_round=totals['feature_accepted']/totals['rounds'],
            p2_rejection_calls_upper_fraction=totals['p2_rejected']/totals['target_calls'],identity_differences=differences)
    (DATA/'B1_k2_accounting.json').write_text(json.dumps(result,indent=2))
    text='''# K2 的推进量与调用成本

该分析对应本项目共享 Ranker 的 K2 实现，提案为 P1/F1/P2。它描述实际控制流，没有提出新的采样定理。下文适用于已有真实前缀、不含 BOS 启动的路径。

## Feature 的收益来自哪里

在不遇到 EOS 的普通轮次中，P1 或 F1 未接受时提交一对 P/F；P1 与 F1 都接受时，还会提交下一对 P/F。此时 P2 即使被拒绝，也会通过 Target 残差采样及补算完成下一对。

因此每轮提交量为 C=2+2·I(P1 和 F1 均接受)。若 θ 是每轮 F1 被接受的概率，则 E[C]=2+2θ。这里的 θ 包含能否到达 F1，和 accepted/verified 的 Feature 条件接受率含义不同。

对当前有限窗口的实际记录，可逐条核对：

- 新增 token 数 = 2×轮数 + 2×Feature 接受数 − EOS 指示量。
- Target 调用数 = 1 次前缀计算 + 轮数 + Position 未接受数 + Position 回退数 − EOS 指示量。

EOS 在 token 数中计为一个。第二式使用本实现“Position 提案不含 EOS”的前提；不能直接迁到任意多候选算法。

## Position 的第二步作用

P2 接受率主要影响补算次数。普通轮次中，提高 P2 接受率不会继续增加 C；它减少 P2 未接受后的 Target 调用。因此条件 P2 模型需要以节省的补算抵消自身成本。

下面把所有 P2 未接受都视为可消除，给出最多可减少的调用次数。EOS 轮次可能根本没有补算，因此该数字是上限。Target 各次前向长度不同，调用比例也不等同于时间比例。

| 方法 | 测量数 | 平均每轮提交 token | 每轮 F1 接受量 θ | P2 未接受数 | 占 Target 调用数的上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for name,d in result.items():
        text+=f'| {name} | {d["runs"]} | {d["mean_committed_tokens"]:.3f} | {d["feature_accepted_per_round"]:.3f} | {d["p2_rejected"]} | {d["p2_rejection_calls_upper_fraction"]*100:.2f}% |\n'
    text+='''
## 速度判断

以完整同步轮次耗时 T 计，单 token 成本约为 E[T]/E[C]。与每轮约提交两个 token 的 P-only 比较，Feature 路径需要满足 E[T_PF] / E[T_P-only] < 1+θ。该式用于解释已测收益；两条路径的上下文分布和 EOS 不同，不能只把两个全局均值代入后称为因果证明。

该关系给出后续选择依据：先提高低成本 F1 的有效到达与接受，再比较其额外耗时；P2 条件化先估计可减少的补算；树和更长递归还要计入节点利用率及更大的验证块成本。参数量、接受率和实际时间分别报告。
'''
    (DATA/'B1_k2_cost_analysis.md').write_text(text)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
