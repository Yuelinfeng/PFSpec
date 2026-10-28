"""Source-level paired analysis with repeated conditions kept together."""
from pathlib import Path
import argparse
import collections
import json
import math
import random
import statistics


def interval(values, seed=20261023):
    rng=random.Random(seed);values=list(values);n=len(values)
    draws=sorted(math.exp(statistics.fmean(values[rng.randrange(n)] for _ in range(n))) for _ in range(10000))
    return [draws[249],draws[9749]]


def analyze(paths, best_fixed='fixed1'):
    rows=[]
    for path in paths:
        progress=json.loads((path/'progress.json').read_text())
        if progress['status']!='complete' or progress['completed']!=progress['expected']:
            raise ValueError(f'Incomplete run: {path}')
        local=[json.loads(line) for line in (path/'records.jsonl').read_text().splitlines() if line]
        plan=json.loads((path/'plan.json').read_text())
        if len(local)!=progress['expected'] or {r['method'] for r in local}!=set(plan['methods']):
            raise ValueError(f'Rows differ from planned run: {path}')
        rows.extend(local)
    groups=collections.defaultdict(dict)
    for r in rows:
        key=(r['repeat_index'],r['source'],r['prefix'],r['seed'])
        if r['method'] in groups[key]:raise ValueError('Duplicate method row')
        groups[key][r['method']]=r
    methods=sorted({r['method'] for r in rows})
    if 'w1' not in methods:raise ValueError('The W1 reference arm is required')
    if best_fixed not in methods:raise ValueError('The declared best fixed arm is missing')
    if any(set(g)!=set(methods) for g in groups.values()):raise ValueError('Unpaired method group')
    semantic=['accepted','proposed','verified','position_accepted','feature_accepted','eos_accepted','accepted_lengths','round_generated_lengths']
    output_errors=[];semantic_errors=[];call_errors=[]
    for key,g in groups.items():
        reference=g['w1']
        for method,r in g.items():
            if r['output_tokens']!=reference['output_tokens']:output_errors.append([key,method])
            if any(k not in r['stats'] or k not in reference['stats'] or r['stats'][k]!=reference['stats'][k] for k in semantic):semantic_errors.append([key,method])
            if 'w1' in g:
                saved=reference['stats']['recovery_calls']-r['stats']['recovery_calls']
                hits=r['stats'].get('position_rescue_hits',0)
                if saved!=hits:call_errors.append([key,method,saved,hits])
                if reference['stats']['target_calls']-r['stats']['target_calls']!=hits:
                    call_errors.append([key,method,'target-call difference differs from hits'])
    comparisons={}
    for denominator in dict.fromkeys(['w1',best_fixed,'mass','ranker_same_k']):
        if denominator not in methods:continue
        for numerator in methods:
            if numerator==denominator:continue
            condition=collections.defaultdict(list);process=collections.defaultdict(list)
            for key,g in groups.items():
                ratio=g[denominator]['outer_wall_seconds']/g[numerator]['outer_wall_seconds']
                condition[key[1:]].append(math.log(ratio));process[key[0]].append((key[1],math.log(ratio)))
            source=collections.defaultdict(list)
            for key,logs in condition.items():source[key[0]].append(statistics.fmean(logs))
            means={s:statistics.fmean(v) for s,v in source.items()}
            by_process={}
            for run,vals in process.items():
                tmp=collections.defaultdict(list)
                for s,v in vals:tmp[s].append(v)
                by_process[str(run)]=math.exp(statistics.fmean(statistics.fmean(v) for v in tmp.values()))
            comparisons[numerator+'/'+denominator]=dict(ratio=math.exp(statistics.fmean(means.values())),
                source_interval95=interval(means.values()),source_count=len(source),unique_conditions=len(condition),
                paired_observations=len(groups),sources_faster=sum(x>0 for x in means.values()),
                source_ratios={s:math.exp(x) for s,x in means.items()},by_process=by_process)
    summary={m:dict(runs=sum(r['method']==m for r in rows),
                   generated_tokens=sum(r['generated_tokens'] for r in rows if r['method']==m),
                   recovery_calls=sum(r['stats']['recovery_calls'] for r in rows if r['method']==m),
                   rescue_hits=sum(r['stats'].get('position_rescue_hits',0) for r in rows if r['method']==m),
                   target_tree_nodes=sum(r['stats'].get('target_tree_nodes',0) for r in rows if r['method']==m)) for m in methods}
    return dict(status='passed' if not(output_errors or semantic_errors or call_errors) else 'failed',
                rows=len(rows),paired_groups=len(groups),methods=methods,best_fixed=best_fixed,method_summary=summary,
                comparisons=comparisons,output_errors=output_errors,semantic_errors=semantic_errors,call_errors=call_errors)


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--best-fixed',default='fixed1');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=analyze(a.runs,a.best_fixed);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','rows','paired_groups','method_summary']},indent=2))
    for name,row in result['comparisons'].items():
        if name.startswith('value/'):print(name,row['ratio'],row['source_interval95'])
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
