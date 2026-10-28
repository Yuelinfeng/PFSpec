"""Summarize the controlled Feature experiment without mixing metric scopes."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/project/artifacts/round2_20261017'


def ratio(d):
    return f'{d["geomean"]:.3f}× [{d["ci95"][0]:.3f}, {d["ci95"][1]:.3f}]'


def main():
    d=json.loads((DATA/'b1_online/20261018/summary.json').read_text())
    text=f'''# B0/B1：共享 Position 的 Feature 机制比较

日期：2026-10-17。首训练种子完成 {d['rows']} 次 FP32 开发评测。来源等权区间按 10 个来源重采样；它描述首训练种子下的来源差异，未包含训练种子波动。

## 在线结果

| 方法 | 微平均 token/s | 相对 Target 来源等权几何平均 [95% 区间] | 相对旧 PF | Feature 接受/已验证 |
| --- | ---: | --- | --- | ---: |
'''
    for name,v in d['summary'].items():
        c=v['counts'];accept='—' if not c.get('feature_verified') else f'{100*c["feature_accepted"]/c["feature_verified"]:.2f}%'
        rt='1.000×' if name=='target' else ratio(v['comparisons']['target'])
        rp='1.000×' if name=='old_pf' else ratio(v['comparisons']['old_pf'])
        text+=f'| {name} | {v["micro_tokens_per_second"]:.3f} | {rt} | {rp} | {accept} |\n'
    e=d['summary']['b1_e']['comparisons']['b0']
    text+=f'''
旧 PF 是本轮七路径中吞吐最高的方法，四种新结构均在 10/10 来源慢于旧 PF。新结构中 B1-E 的首种子吞吐最高，相对同预算 B0 为 {ratio(e)}。该结果属于指定训练和运行条件，不能推断完整 EAGLE 优于 MLP。

B1-M 与 B1-H 的首种子结果没有显示候选 Position 条件带来明显优势。参数量、参数化方式和预算有限，不据此宣称 Position 条件无用。旧 step12000 的训练来源与预算不同，四个新模型又统一取消隐藏状态辅助损失，因此它与 B0 的差异也不能单独归因于训练时长。

## 实际比较范围

原 10 个开发来源各取第一条裁剪；64/256/1024 可用前缀，两个运行种子，最多 256 新 token，temperature/top-p=0.9。七种方法各 56 条。方法次序按上下文轮换，缓存分配与前缀计算计时；模型加载和等量短 warmup 不计入。部分运行自然 EOS，输出长度不同，逐条记录同时保留长度和时间。

各结构使用相同历史信息、Position Ranker、K2 验证和采样条件；B1-M 明确不读候选 Position。B1-E 使用已知历史 Feature 与候选 Position 的两 token 因果块，读取共享 Target embedding 和输出头；没有维护跨轮草稿 KV。可训练参数分别为 B0 2.864M、B1-M 3.088M、B1-H 4.178M、B1-E 1.318M，参数量未严格匹配。共享固定参数也另行记录。

训练为同一 40 条 Target 轨迹、36/4 完整轨迹划分、16,348/2,044 行、10 轮、AdamW lr=1e-4/wd=0.01。统一使用 temperature=0.9 的教师分布交叉熵，选择验证重合度最高权重。原 Medusa/Hydra 的整序列实验与本轮 Feature 机制实验分别解释。

## 分项成本

'''
    pp=DATA/'b1_profile/20261018/records.jsonl'
    if pp.exists():
        rows=[json.loads(s) for s in pp.read_text().splitlines() if s.strip()]
        by=collections.defaultdict(list)
        for r in rows:
            if 'stage_profile' in r['stats']:by[r['method']].append(r)
        text+='| 方法 | 分项运行数 | Feature 前向 CUDA event ms/次 | Feature 前向 host ms/次 |\n| --- | ---: | ---: | ---: |\n'
        costs={}
        for name,rr in by.items():
            events=[r['stats']['stage_profile']['feature_forward'] for r in rr]
            calls=sum(e['calls'] for e in events)
            gpu=1000*sum(e['cuda_events']['seconds'] for e in events)/calls
            host=1000*sum(e['host_seconds'] for e in events)/calls
            text+=f'| {name} | {len(rr)} | {gpu:.3f} | {host:.3f} |\n'
            costs[name]=dict(runs=len(rr),calls=calls,gpu_event_ms_per_call=gpu,host_ms_per_call=host)
        (DATA/'B1_component_costs.json').write_text(json.dumps(costs,indent=2))
        text+='\n分项计时来自独立的小规模运行，不能用其吞吐替代上表无分项记录的主速度。CUDA event 与 host 区间分别解释，不直接相加。\n'
    else:text+='分项记录尚未同步。\n'
    fp=DATA/'B1_fixed_contexts.json'
    text+='\n## 训练种子与固定上下文\n\n'
    if fp.exists():
        fd=json.loads(fp.read_text());text+='| 结构 | seed 20261018 | seed 20261019 | seed 20261020 |\n| --- | ---: | ---: | ---: |\n'
        for name in ('b0','b1_m','b1_h','b1_e'):
            values=[fd['summary'][f'{seed}:{name}']['source_equal_overlap'] for seed in (20261018,20261019,20261020)]
            text+='| '+name+' | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |\n'
        text+='\n该表对 448 个相同真实 Feature 上下文按来源等权，候选 Position 为真实序列中的下一位置。额外两种训练种子仅补充离线提案质量，未重跑完整在线速度；它们不能证明上表吞吐优势跨训练种子稳定。\n'
    else:text+='两个额外训练种子及固定上下文结果仍待完成；首种子表不包含该不确定性。\n'
    text+='''
## 可解释的计算收益

逐条 K2 计数关系和 P2 可减少调用的上限见 [推进量与调用成本](B1_k2_cost_analysis.md)。当前控制流中，Feature 被接受增加每轮推进；P2 被接受主要减少补算。后续工作优先依据完整路径成本选择，不能仅按接受率排序。

## 后续范围

本轮没有新模型替换旧 PF。B1-E 保留为小预算特征机制候选；A2 单独评价历史 Feature 对 P1 的作用。尚未实施候选树、跨轮特征递归、多层融合及新方案正式三维确认。旧模型的正式结果继续有效，开发数据不用于宣称三维质量保持。
'''
    (DATA/'B1_mechanism_report.md').write_text(text)


if __name__=='__main__':main()
