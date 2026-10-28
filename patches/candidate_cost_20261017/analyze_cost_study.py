"""Summarize observed branch costs and immediate same-prefix action value."""
import collections
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def divide(a,b):
    return a/b if b else 0.


def summarize_intervals(values):
    values=np.array(values,dtype=float)
    rng=np.random.default_rng(20261021)
    draws=rng.choice(values,(10000,len(values)),replace=True)
    means=np.exp(np.mean(np.log(draws),axis=1))
    return dict(geomean=float(np.exp(np.log(values).mean())),
                ci95=np.quantile(means,[.025,.975]).tolist(),sources=len(values),
                sources_above_one=int((values>1).sum()))


def main():
    dest=ROOT/'full'
    progress=json.loads((dest/'progress.json').read_text())
    assert progress.get('status')=='complete',progress
    rows=read_rows(dest/'runs.jsonl')
    matched=read_rows(dest/'matched_rounds.jsonl')
    sync=read_rows(dest/'sync_p2.jsonl')
    scenes=sorted({r['scene'] for r in rows})
    by_method={m:[r for r in rows if r['method']==m] for m in ('pf_plain','p_only_plain','pf_profile')}
    def tps(rr,seconds='seconds'):
        return sum(r['generated_tokens'] for r in rr)/sum(r[seconds] for r in rr)
    speed={m:dict(runs=len(rr),tokens=sum(r['generated_tokens'] for r in rr),
                  seconds=sum(r['seconds'] for r in rr),tokens_per_second=tps(rr)) for m,rr in by_method.items()}
    for m,ref in [('pf_plain','p_only_plain'),('pf_profile','pf_plain')]:
        ratios=[tps([r for r in by_method[m] if r['scene']==s])/tps([r for r in by_method[ref] if r['scene']==s]) for s in scenes]
        speed[m]['relative_to_'+ref]=summarize_intervals(ratios)
    prof=by_method['pf_profile']
    for run in prof:
        st=run['stats']
        assert run['generated_tokens']==2*st['rounds']+2*st['feature_accepted']-int(st['eos'])
        assert st['target_calls']==1+st['rounds']+st['position_verified']-st['position_accepted']+st['position_fallbacks']-int(st['eos'])
    rounds=[]
    for run in prof:
        for rr in run['stats']['round_profile']['rounds']:
            rounds.append(dict(**rr,scene=run['scene'],prefix=run['prefix'],seed=run['seed']))
    def operations(rr,stage=None,operation=None):
        return [o for r in rr for o in r['operations'] if (stage is None or o['stage']==stage) and (operation is None or o['operation']==operation)]
    def sums(ops):
        return dict(calls=len(ops),host_seconds=sum(o['host_seconds'] for o in ops),
                    gpu_seconds=sum(o.get('gpu_seconds',0) for o in ops),
                    input_tokens=sum(o.get('input_tokens',0) for o in ops))
    branches={}
    for name in sorted({r['branch'] for r in rounds}):
        rr=[r for r in rounds if r['branch']==name]
        branches[name]=dict(rounds=len(rr),tokens=sum(r['generated_tokens'] for r in rr),
            round_share=len(rr)/len(rounds),mean_tokens=np.mean([r['generated_tokens'] for r in rr]).item(),
            host_seconds=sum(r['host_seconds'] for r in rr),gpu_seconds=sum(r['gpu_seconds'] for r in rr),
            mean_host_ms=np.mean([r['host_seconds'] for r in rr]).item()*1000,
            mean_gpu_ms=np.mean([r['gpu_seconds'] for r in rr]).item()*1000,
            target_calls=sum(r['counts']['target_calls'] for r in rr),eos=sum(r['eos'] for r in rr),
            stages={stage:sums(operations(rr,stage=stage)) for stage in ('feature_forward','ranker','target_verify','target_context','sampling','cache_maintenance')})
    repair=[o for r in rounds for o in r['regions'] if o['name']=='p2_repair']
    repair_non_eos=[r for r in repair if r['non_eos']]
    tf_repair=operations(rounds,operation='p2_recompute')
    assert len(repair_non_eos)==len(tf_repair)
    total_profile_seconds=sum(r['stats']['seconds'] for r in prof)
    repair_wall=sum(r['host_seconds'] for r in repair_non_eos)
    all_repair_wall=sum(r['host_seconds'] for r in repair)
    per_source={}
    for scene in scenes:
        rr=[r for r in rounds if r['scene']==scene]
        run_time=sum(r['stats']['seconds'] for r in prof if r['scene']==scene)
        regs=[o for r in rr for o in r['regions'] if o['name']=='p2_repair' and o['non_eos']]
        wall=sum(o['host_seconds'] for o in regs)
        per_source[scene]=dict(repair_seconds=wall,total_seconds=run_time,
            share=wall/run_time,optimistic_speed=run_time/(run_time-wall))
    p2=dict(rejected_rounds=len(repair),non_eos_recompute_rounds=len(repair_non_eos),
        target_calls=sum(r['stats']['target_calls'] for r in prof),
        forward=sums(tf_repair),non_eos_repair_host_seconds=repair_wall,
        all_repair_host_seconds=all_repair_wall,
        non_eos_repair_gpu_seconds=sum(r['gpu_seconds'] for r in repair_non_eos),
        all_repair_gpu_seconds=sum(r['gpu_seconds'] for r in repair),
        profile_decode_seconds=total_profile_seconds,
        non_eos_repair_host_share=repair_wall/total_profile_seconds,
        forward_gpu_interval_share=sum(r['gpu_seconds'] for r in tf_repair)/total_profile_seconds,
        optimistic_remove_entire_non_eos_repair_speed=total_profile_seconds/(total_profile_seconds-repair_wall),
        optimistic_by_source=summarize_intervals([r['optimistic_speed'] for r in per_source.values()]),
        per_source=per_source,
        scope='Local time subtraction on observed paths; excludes any new proposal cost and future trajectory changes')
    p2_proposals=[]
    for row in rounds:
        proposals=[o for o in row['operations'] if o['stage']=='ranker' and o['operation']=='position_proposal']
        if row['counts']['feature_proposed']:
            assert len(proposals)==2
            p2_proposals.append(proposals[1])
    p2['existing_p2_proposal']=sums(p2_proposals)
    p2['existing_p2_proposal']['host_share']=sum(o['host_seconds'] for o in p2_proposals)/total_profile_seconds
    sync_rounds=[rr for row in sync for rr in row['stats']['round_profile']['rounds']]
    sync_ops=operations(sync_rounds,operation='p2_recompute')
    assert all(o['synchronized'] for o in sync_ops)
    p2['synchronized_subset']=dict(runs=len(sync),forward=sums(sync_ops))
    light_progress=json.loads((dest/'repair_light_progress.json').read_text())
    assert light_progress.get('status')=='complete'
    light=read_rows(dest/'repair_light.jsonl')
    region_runs=[r for r in light if r['method']=='regions']
    regions=[o for run in region_runs for row in run['stats']['round_profile']['rounds']
             for o in row['regions'] if o['name']=='p2_repair' and o['non_eos']]
    light_total=sum(run['stats']['seconds'] for run in region_runs)
    light_repair=sum(o['host_seconds'] for o in regions)
    controls=[r for r in light if r['method']=='plain']
    control_keys={(r['scene'],r['prefix'],r['seed']) for r in controls}
    light_control=[r for r in region_runs if (r['scene'],r['prefix'],r['seed']) in control_keys]
    assert len(region_runs)==56 and len(controls)==8
    p2['light']=dict(runs=len(region_runs),control_runs=len(controls),
        total_decode_seconds=light_total,non_eos_repair_host_seconds=light_repair,
        non_eos_repair_gpu_seconds=sum(o['gpu_seconds'] for o in regions),
        host_share=light_repair/light_total,
        optimistic_speed=light_total/(light_total-light_repair),
        endpoint_observer_throughput_ratio=tps(light_control)/tps(controls),
        matched_outputs_equal=True)
    p2['light']['optimistic_added_cost_ms_per_p2_proposal']=light_repair/len(p2_proposals)*1000
    matched_summary={}
    def matched_speed(rr):
        return sum(r['generated_tokens'] for r in rr)/sum(r['round']['host_seconds'] for r in rr)
    for origin in ('all','initial','pf_middle'):
        selected=matched if origin=='all' else [r for r in matched if r['origin']==origin]
        methods={}
        for mode in ('p_only','pf'):
            rr=[r for r in selected if r['mode']==mode]
            methods[mode]=dict(rounds=len(rr),tokens=sum(r['generated_tokens'] for r in rr),
                mean_tokens=np.mean([r['generated_tokens'] for r in rr]).item(),
                mean_host_ms=np.mean([r['round']['host_seconds'] for r in rr]).item()*1000,
                mean_gpu_ms=np.mean([r['round']['gpu_seconds'] for r in rr]).item()*1000,
                tokens_per_second=matched_speed(rr),
                branch_counts=dict(collections.Counter(r['round']['branch'] for r in rr)))
        ratios=[]
        for scene in scenes:
            a=[r for r in selected if r['scene']==scene and r['mode']=='pf']
            b=[r for r in selected if r['scene']==scene and r['mode']=='p_only']
            if a and b:ratios.append(matched_speed(a)/matched_speed(b))
        matched_summary[origin]=dict(methods=methods,relative=summarize_intervals(ratios),
            extra_tokens_per_round=methods['pf']['mean_tokens']-methods['p_only']['mean_tokens'],
            extra_host_ms_per_round=methods['pf']['mean_host_ms']-methods['p_only']['mean_host_ms'],
            time_ratio=methods['pf']['mean_host_ms']/methods['p_only']['mean_host_ms'],
            advancement_ratio=methods['pf']['mean_tokens']/methods['p_only']['mean_tokens'])
    counted=collections.Counter()
    for r in prof:
        for k in ('target_calls','rounds','feature_accepted','feature_verified','position_accepted','position_verified'):
            counted[k]+=r['stats'][k]
    for row in rounds+[r['round'] for r in matched]:
        assert row['host_seconds']>0 and row['gpu_seconds']>=0
        assert math.isfinite(row['host_seconds']) and math.isfinite(row['gpu_seconds'])
        c=row['counts']
        assert row['generated_tokens']==2+2*c['feature_accepted']-int(row['eos'])
        assert c['target_calls']==1+c['position_verified']-c['position_accepted']+c['position_fallbacks']-int(row['eos'])
    previous=read_rows(Path('/root/autodl-tmp/gaussiangpt_round2_20261017/b1_online/20261018/records.jsonl'))
    previous=[r for r in previous if r['method']=='old_pf']
    previous_map={(r['scene'],r['prefix'],r['seed']):r for r in previous}
    equality=[]
    for row in prof:
        other=previous_map[(row['scene'],row['prefix'],row['seed'])]
        equality.append(all(row['stats'][k]==other['stats'][k] for k in counted))
    validation=dict(profile_vs_plain_outputs_equal=True,
        previous_counts_equal_runs=sum(equality),previous_runs=len(equality),
        unique_run_keys=len({(r['scene'],r['prefix'],r['seed'],r['method']) for r in rows}),
        paired_keys=len({(r['context_id'],r['trial'],r['mode']) for r in matched}),
        round_tokens_equal=True,round_target_calls_equal=True)
    assert all(equality)
    assert len(rows)==168 and len(matched)==896 and len(prof)==56
    assert len(repair)==113 and counted['target_calls']==5875
    theta=counted['feature_accepted']/counted['rounds']
    costs=dict(counts=dict(counted),joint_acceptance_per_round=theta,
        feature_acceptance_given_verified=counted['feature_accepted']/counted['feature_verified'],
        round_host_seconds=sum(r['host_seconds'] for r in rounds),
        round_gpu_seconds=sum(r['gpu_seconds'] for r in rounds),
        stages={stage:sums(operations(rounds,stage=stage)) for stage in ('feature_forward','ranker','target_verify','target_context','sampling','cache_maintenance')})
    result=dict(status='complete',speed=speed,branches=branches,p2=p2,
                matched=matched_summary,costs=costs,validation=validation)
    (dest/'analysis.json').write_text(json.dumps(result,indent=2))
    make_report(dest,result)
    make_figure(dest,result)
    print(json.dumps(result,indent=2))


def make_report(dest,r):
    p2=r['p2'];m=r['matched']['all'];sp=r['speed']['pf_plain']['relative_to_p_only_plain']
    def ratio(x):return f"{x['geomean']:.3f}× [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    lines=['# GaussianGPT 候选的边际计算价值：A1–A3',
        '', '本轮只测量现有 step12000 轻量 PF 与 P-only，没有训练新 Head。', '',
        '## 结论与选择', '',
        f"同批不加分项计时的固定窗口 token 生成中，PF/P-only 来源等权速度比为 **{ratio(sp)}**。",
        f"P₂ 非 EOS 拒绝后的整段处理占少量计时点复测总时长 **{p2['light']['host_share']*100:.3f}%**；把这段处理全部免费删除、其余行为保持原样，时间相减估计为 **{p2['light']['optimistic_speed']:.4f}×**。这包含残差采样、KV 回退、Target 补算、状态保存和 Feature 采样，范围大于单独补算前向。",
        f"相同前缀的单轮对照中，PF/P-only 来源等权即时推进效率比为 **{ratio(m['relative'])}**。该结果只描述当前一轮，不能代替长期策略比较。",
        '', '当前选择：保留现用 Feature Draft，暂缓以提高 P₂ 接受率为主要目的的独立 Head 训练。若能减少已有提案成本或与 Feature 共享计算，另行比较净时间；本轮未证明新的动态方法已经成立。', '',
        '## 对象与测量范围', '',
        '- 原十个开发来源，每个来源首条裁剪，64/256/1024 可用前缀，共 28 个来源与前缀组合。',
        '- 两个运行种子、每次最多 256 个新 token，FP32、temperature/top-p=0.9；使用原 Target、Ranker、step12000。',
        '- 56 次普通 PF、56 次普通 P-only、56 次带分支计时的 PF，方法顺序轮换。',
        '- 56 个相同上下文：28 个真实初始前缀与各自一条 PF 运行的中间前缀；每个上下文 8 个种子，两种策略，共 896 次单轮。',
        '- 单轮对照复用同一解码循环，P-only 停止第二位置提案且不运行 Feature Draft。缓存分配、前缀计算与历史恢复在单轮计时之外；两种策略都从准备完毕的状态开始。',
        '- 固定窗口计时包含缓存分配、前缀计算、提案、验证、拒绝处理与释放；模型加载、预热和 JSON 保存在计时之外。',
        '', '## A1：Feature 的边际价值', '',
        '| 相同上下文单轮 | P-only | PF-K2 |', '| --- | ---: | ---: |',
        f"| 平均推进 token | {m['methods']['p_only']['mean_tokens']:.3f} | {m['methods']['pf']['mean_tokens']:.3f} |",
        f"| 平均耗时，ms | {m['methods']['p_only']['mean_host_ms']:.3f} | {m['methods']['pf']['mean_host_ms']:.3f} |",
        f"| 推进效率，token/s | {m['methods']['p_only']['tokens_per_second']:.2f} | {m['methods']['pf']['tokens_per_second']:.2f} |", '',
        f"平均额外推进 {m['extra_tokens_per_round']:.3f} token，额外耗时 {m['extra_host_ms_per_round']:.3f} ms。推进量比为 {m['advancement_ratio']:.3f}，耗时比为 {m['time_ratio']:.3f}。收益取决于两者的关系。",
        '', '| 上下文来源 | PF/P-only 即时推进效率比，来源等权 [95% 区间] |', '| --- | --- |',
        f"| 真实初始前缀 | {ratio(r['matched']['initial']['relative'])} |",
        f"| PF 运行中间前缀 | {ratio(r['matched']['pf_middle']['relative'])} |", '',
        '两类上下文的收益不同，但前缀内容、位置和历史长度也不同，不能据此确定差异原因或直接选定动态策略。相同上下文对照比较的是整个 PF 动作，包含额外位置提案与更长 Target 验证，无法单独归因于 MLP 这一组件。', '',
        f"完整 PF 记录中，F₁ 接受/已验证为 {r['costs']['feature_acceptance_given_verified']*100:.2f}%，每轮 P₁、F₁ 连续接受概率为 {r['costs']['joint_acceptance_per_round']*100:.2f}%。普通轮次推进量遵循 2+2×连续接受指示量。", '',
        'Feature 被拒绝时，保留 P₁ 的 Target 状态，残差生成 F₁ 并回退缓存；该分支没有新增 Target 补算。P₁ 被拒绝时，需要补算实际 Position；P₂ 被拒绝且结果非 EOS 时，也需要补算实际 Position。F₁ 连续接受后，额外推进的 P₂/F₂ 来自已算验证结果或拒绝后的补算。', '',
        '## A2：P₂ 补算成本', '',
        f"P₂ 拒绝 {p2['rejected_rounds']} 次，非 EOS 实际补算 {p2['non_eos_recompute_rounds']} 次；完整 PF Target 调用 {p2['target_calls']} 次。原 113 次属于拒绝计数，实际补算次数另列。", '',
        '| 时间范围 | 累计 host 时间，s | 累计 CUDA event 区间，s |', '| --- | ---: | ---: |',
        f"| 仅 Target 补算前向 | {p2['forward']['host_seconds']:.6f} | {p2['forward']['gpu_seconds']:.6f} |",
        f"| 非 EOS 拒绝整段处理 | {p2['non_eos_repair_host_seconds']:.6f} | {p2['non_eos_repair_gpu_seconds']:.6f} |", '',
        f"仅补算前向的 CUDA event 区间占总时长 {p2['forward_gpu_interval_share']*100:.3f}%；整段处理的 host 时间占比为 {p2['non_eos_repair_host_share']*100:.3f}%。前向 host 时长未逐次同步，不单独解释成其完整等待时间。",
        f"额外 {p2['synchronized_subset']['runs']} 次运行在补算前后同步；其中 {p2['synchronized_subset']['forward']['calls']} 次前向的同步 host 时间累计 {p2['synchronized_subset']['forward']['host_seconds']:.6f} s，CUDA event 区间 {p2['synchronized_subset']['forward']['gpu_seconds']:.6f} s。同步运行仅用于佐证成本量级。", '',
        f"另对全部 56 条 PF 路径复测，只记录每轮及 P₂ 整段处理的起止。整段 host 时间累计 {p2['light']['non_eos_repair_host_seconds']:.6f} s，总时长 {p2['light']['total_decode_seconds']:.6f} s，占比 {p2['light']['host_share']*100:.3f}%；对应乐观时间相减为 {p2['light']['optimistic_speed']:.4f}×。全部输出与普通运行一致；8 组交错普通运行中，少量计时点/普通运行吞吐比为 {p2['light']['endpoint_observer_throughput_ratio']:.3f}×。", '',
        f"现有 P₂ Ranker 提案共 {p2['existing_p2_proposal']['calls']} 次，CPU 函数时长累计 {p2['existing_p2_proposal']['host_seconds']:.6f} s，占分项计时总时长 {p2['existing_p2_proposal']['host_share']*100:.3f}%。新 P₂ 模块若替换当前 Ranker，还应比较新旧提案成本；仅凭拒绝补算占比，无法排除更便宜的 P₂ 提案机制。", '',
        f"将可免费删除的处理时间粗略摊到全部 {p2['existing_p2_proposal']['calls']} 次 P₂ 提案，每次约 {p2['light']['optimistic_added_cost_ms_per_p2_proposal']:.3f} ms。若新增模块每轮都运行，它需要先覆盖自身成本，且真实模型无法保证消除全部拒绝。该数字只用于当前路径的成本量级判断。", '',
        '免费删除整段处理的数字是已观测路径上的乐观时间相减结果。它省略新提案成本，并假设前缀、输出、接受事件及其他操作都保持原样；不能解释为任意条件 P₂ 方法的全局加速上限。CUDA event 与 host 时间分别使用，不相加。', '',
        '## A3：各分支推进效率', '',
        '| 分支 | 轮数 | 平均推进 token | 平均 host 耗时，ms | Target 调用/轮 |', '| --- | ---: | ---: | ---: | ---: |']
    names=dict(p1_reject='P₁ 拒绝',f1_reject='F₁ 拒绝',p2_reject='P₂ 拒绝',all_accept='全部接受',k1_accept='单位置接受',k1_reject='单位置拒绝',fallback='无位置提案')
    for name,b in r['branches'].items():
        lines.append(f"| {names[name]} | {b['rounds']} | {b['mean_tokens']:.3f} | {b['mean_host_ms']:.3f} | {b['target_calls']/b['rounds']:.3f} |")
    lines += ['', '| 分支 | Feature 前向 | Target 验证 | Target 状态与补算 | 采样 | KV 操作 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in ('p1_reject','f1_reject','p2_reject','all_accept'):
        b=r['branches'][name]
        values=[b['stages'][stage]['gpu_seconds']*1000/b['rounds'] for stage in
                ('feature_forward','target_verify','target_context','sampling','cache_maintenance')]
        lines.append('| '+names[name]+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', '第二表单位为每轮平均 CUDA event 区间，ms。各项与 host 分支耗时使用不同时间范围，不相加。']
    overhead=r['speed']['pf_profile']['relative_to_pf_plain']
    lines += ['', '分支均值来自不同的可到达上下文。接受组与拒绝组的差值属于描述统计，不能直接解释成接受事件的因果节省。', '',
        '## 验证与限制', '',
        f"56 组 PF 普通运行与分项运行的输出逐 token 一致；与上一轮同种子记录的关键计数在 {r['validation']['previous_counts_equal_runs']}/{r['validation']['previous_runs']} 组一致。轮次输出量、Target 调用量与完整记录相符，EOS 短验证通过。",
        f"带分项计时 PF / 普通 PF 的来源等权吞吐比为 {ratio(overhead)}。分项计时增加了开销，正式速度采用普通运行；分支成本占比属于带计时运行。单轮对照仅记录轮次起止，避免逐操作计时影响两种策略的比较。",
        'CUDA event 记录同一流上的时间区间，可能包含 CPU 提交间隙，不等于纯 kernel 活跃时间。P₂ 整段结束前已有采样取值同步；其 host 区间包含该段实际等待。',
        '来源区间按十个开发场景重采样，只反映当前权重和运行设置。中间前缀由 PF 产生；有限重复和上下文抽样不能证明任意长期策略更优。本轮没有新增正式测试或三维质量结论。', '',
        '## 文件与下一步', '',
        '实现位于 patches/candidate_cost_20261017/；逐轮记录、相同上下文、同步子集及 analysis.json 位于 docs/project/artifacts/candidate_cost_20261017/full/。远端目录为 /root/autodl-tmp/gaussiangpt_candidate_cost_20261017/。',
        '原 Target、VQ、Ranker、Feature 权重与正式解码器保持原样。A4 深度成本、B 阶段训练和 C/D 调度均未启动；下一步依据实测收益空间选择。', '',
        '![候选成本与推进量](candidate_cost_summary.png)', '']
    (dest/'CANDIDATE_COST_REPORT.md').write_text('\n'.join(lines))


def make_figure(dest,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    m=r['matched']['all']['methods']
    modes=['p_only','pf'];labels=['P-only','PF-K2']
    ax[0].bar(labels,[m[a]['tokens_per_second'] for a in modes],color=['#9aa7b8','#3276ac'],width=.55)
    ax[0].set_ylabel('Tokens / second');ax[0].set_title('Same-prefix immediate advancement')
    for i,a in enumerate(modes):
        ax[0].text(i,m[a]['tokens_per_second']*1.02,f"{m[a]['mean_tokens']:.2f} tokens\n{m[a]['mean_host_ms']:.2f} ms",ha='center',fontsize=9)
    ax[0].set_ylim(0,max(m[a]['tokens_per_second'] for a in modes)*1.3)
    p=r['p2'];values=[p['forward_gpu_interval_share']*100,p['light']['host_share']*100]
    ax[1].bar(['Recompute forward\nCUDA interval','Entire non-EOS repair\nhost elapsed'],values,color=['#6d9fc1','#d5955e'],width=.55)
    ax[1].set_ylabel('% of instrumented decode time');ax[1].set_title('Observed P2 repair cost (separate clocks)')
    for i,v in enumerate(values):ax[1].text(i,v*1.02,f'{v:.3f}%',ha='center')
    ax[1].set_ylim(0,max(values)*1.3)
    fig.savefig(dest/'candidate_cost_summary.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
