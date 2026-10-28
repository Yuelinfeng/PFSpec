"""Recompute policy scores from raw warm observations and exported choices."""
import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
import random
import statistics as st


def source_interval(logs):
    values = list(logs.values())
    rng = random.Random(20261021)
    draws = sorted(math.exp(st.mean(rng.choices(values,k=len(values)))) for _ in range(10000))
    return dict(ratio=math.exp(st.mean(values)),source_interval95=[draws[249],draws[9749]],
                per_source={k:math.exp(v) for k,v in logs.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    raw=[json.loads(s) for s in (args.root/'dev_run01/records.jsonl').read_text().splitlines()]
    comp=[json.loads(s) for s in (args.root/'components_run01/records.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for r in raw:
        if r['cold_status']=='warm':
            groups[(r['source_id'],r['context_id'],r['seed'],r['action_id'])].append(r)
    observations={k:dict(R=v[0]['R'],C=st.median(x['round_wall_ms'] for x in v)) for k,v in groups.items()}
    actions=('K1','W1','W2')
    contexts=sorted({(k[0],k[1]) for k in observations})
    def score(choices):
        totals=defaultdict(lambda:dict(R=0.,C=0.))
        for source,context in contexts:
            action=choices.get(source+'::'+context,'W1')
            seeds=set.intersection(*({k[2] for k in observations if k[0]==source and k[1]==context and k[3]==a} for a in actions))
            for seed in seeds:
                row=observations[(source,context,seed,action)]
                totals[source]['R']+=row['R']/len(seeds)
                totals[source]['C']+=row['C']/len(seeds)
        return {s:v['R']/v['C'] for s,v in totals.items()}
    w1=score({})
    fixed={a:source_interval({s:math.log(v/w1[s]) for s,v in score({s+'::'+c:a for s,c in contexts}).items()}) for a in actions}
    analyzed=json.loads(args.analysis.read_text())['scope_results']['single_round']['thermal_modes']['warm']
    policy={}
    for name,value in analyzed['policies'].items():
        rates=score(value['choices'])
        detail=source_interval({s:math.log(v/w1[s]) for s,v in rates.items()})
        detail['reported_point_equal']=abs(detail['ratio']-value['relative_to_fixed_W1']['geomean_speedup'])<1e-12
        policy[name]=detail
    raw0={(r['context_id'],r['seed'],r['action_id']):r for r in raw if r['repeat_index']==0}
    equality=sum(r['actual_tokens']==raw0[(r['context_id'],r['seed'],r['action_id'])]['actual_tokens'] for r in comp)
    components={}
    for action in actions:
        local=[r for r in comp if r['action_id']==action and r['cold_status']=='warm']
        names=sorted(set().union(*(r['component_device_ms'] for r in local)))
        summary={name:dict(mean_all_ms=st.mean(r['component_device_ms'].get(name,0.) for r in local),
                           conditional_median_ms=st.median(r['component_device_ms'][name] for r in local if name in r['component_device_ms']),
                           observed_rows=sum(name in r['component_device_ms'] for r in local)) for name in names}
        components[action]=dict(warm_rows=len(local),recovery_calls=sum(r['recovery_calls'] for r in local),events=summary)
    costs={}
    for action in actions:
        local=[v for k,v in observations.items() if k[3]==action]
        costs[action]=dict(observations=len(local),mean_R=st.mean(r['R'] for r in local),mean_C_ms=st.mean(r['C'] for r in local))
    recovery_choices=analyzed['policies']['recovery_event_cost']['choices']
    separate_choices=analyzed['policies']['full_separate_R_C']['choices']
    report=dict(fixed_actions=fixed,policies=policy,point_scores_all_match=all(v['reported_point_equal'] for v in policy.values()),
                recovery_and_separate_identical_choices=sum(recovery_choices[k]==v for k,v in separate_choices.items()),
                contexts=len(contexts),sources=len(w1),warm_unique_observations=len(observations),action_means=costs,
                components=dict(output_pairs=len(comp),output_equal_pairs=equality,by_action=components),
                eos_raw_rows=sum(r['eos'] for r in raw),cold_raw_rows=sum(r['cold_status']=='cold' for r in raw),
                event_units='CUDA stream intervals including possible CPU submission gaps; separate instrumented run',
                interpretation='Local one-round progress per cost, excluding setup/prefill and online selection; no full-request speed claim')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.out),point_scores_all_match=report['point_scores_all_match'],fixed_actions={k:v['ratio'] for k,v in fixed.items()},components_output_equal=equality)))
    if not report['point_scores_all_match'] or equality!=len(comp):
        raise AssertionError('Evidence differs from primary output')


if __name__=='__main__':
    main()
