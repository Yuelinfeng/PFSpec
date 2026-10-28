"""Independently summarize complete timed decode records with standard Python."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random


def describe(rows):
    generated=sum(r['generated_tokens'] for r in rows)
    seconds=math.fsum(r['outer_wall_seconds'] for r in rows)
    counts={k:sum(r.get(k,0) or 0 for r in rows) for k in (
        'target_calls','target_input_tokens','draft_calls','draft_input_tokens',
        'ranker_calls','position_verified','position_accepted','feature_verified','feature_accepted')}
    return dict(runs=len(rows),tokens=generated,seconds=seconds,tps=generated/seconds if seconds else None,
                lengths=dict(sorted(Counter(r['generated_tokens'] for r in rows).items())),
                eos=sum(r['eos'] for r in rows),counts=counts,
                feature_acceptance=counts['feature_accepted']/counts['feature_verified'] if counts['feature_verified'] else None,
                position_acceptance=counts['position_accepted']/counts['position_verified'] if counts['position_verified'] else None)


def quantile(values,p):
    ordered=sorted(values); x=(len(ordered)-1)*p; i=int(x)
    return ordered[i]*(1-(x-i))+ordered[min(i+1,len(ordered)-1)]*(x-i)


def source_summary(ratios):
    if not ratios:return dict(source_count=0,geometric_mean=None,ci95=None)
    logs=[math.log(ratios[s]) for s in sorted(ratios)]
    rng=random.Random(381012)
    boot=[math.exp(math.fsum(rng.choice(logs) for _ in logs)/len(logs)) for _ in range(10000)]
    return dict(source_count=len(logs),geometric_mean=math.exp(math.fsum(logs)/len(logs)),ci95=[quantile(boot,.025),quantile(boot,.975)])


def main():
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('--run',type=Path,required=True); args=a.parse_args()
    rows=[json.loads(x) for x in (args.run/'records.jsonl').read_text().splitlines() if x.strip()]
    methods=sorted({r['method'] for r in rows}); sources=sorted({r['source'] for r in rows})
    errors=[]
    prompts=json.loads((args.run/'prompts.json').read_text())['contexts']
    prompt_by_id={r['context_id']:r['prompt'] for r in prompts}
    ranks=json.loads((args.run/'rank_map.json').read_text())['values']
    def require(ok,message):
        if not ok:errors.append(message)
    require(len(rows)==180,'Expected 180 complete records')
    require(len(methods)==3 and len(sources)==10,'Expected three methods and ten sources')
    groups=defaultdict(list)
    for r in rows:
        groups[r['pair_id']].append(r)
        output=r['output_tokens']; n=r['generated_tokens']; eos=r['eos']
        require(n==len(output) and 0<n<=512,f"Bad length {r['pair_id']} {r['method']}")
        require(eos==(output[-1]==12096),f"EOS flag differs {r['pair_id']} {r['method']}")
        require(12096 not in output[:-1],f"Nonterminal EOS {r['pair_id']} {r['method']}")
        require(eos or n==512,f"Unexplained short output {r['pair_id']} {r['method']}")
        require(math.isfinite(r['outer_wall_seconds']) and r['outer_wall_seconds']>0,'Nonpositive timed wall')
        body=output[:-1] if eos else output
        require(len(body)%2==0,'Incomplete PF pair')
        require(all(0<=x<8000 for x in body[::2]),'Invalid Position token')
        require(all(8000<=x<12096 for x in body[1::2]),'Invalid Feature token')
        prompt=prompt_by_id[r['prompt_context_id']]
        require(prompt[0]==12096 and len(prompt)==r['prefix']+1,'Invalid prompt context')
        positions=(prompt[1:]+body)[::2]
        require(all(ranks[b]>ranks[a] for a,b in zip(positions,positions[1:])), 'Position order differs')
        for kind in ['position','feature']:
            require(0<=r.get(kind+'_accepted',0)<=r.get(kind+'_verified',0)<=r.get(kind+'_proposed',0),'Invalid proposal count ordering')
        require(r['cache_recoveries']==r.get('verified',0)-r.get('accepted',0),'Recovery count differs from rejected proposals')
        if r['method'] in ('old_pf','spatial_s3'):
            require(r['mlp_calls']==r['draft_calls']==r['feature_proposed'],'Feature MLP call count differs')
            require(r['s3_calls']==(r['mlp_calls'] if r['method']=='spatial_s3' else 0),'Spatial call count differs')
        else:
            require(r['mlp_calls']==r['s3_calls']==r['draft_calls']==0,'Target-only used a Feature proposal')
    require(len(groups)==60,'Expected 60 paired contexts')
    for pid,g in groups.items():
        require(sorted(r['method'] for r in g)==methods,'Missing or duplicate method '+pid)
        require(len({(r['source'],r['prefix'],r['derived_seed'],r['budget']) for r in g})==1,'Paired settings differ '+pid)
    by_method={m:describe([r for r in rows if r['method']==m]) for m in methods}
    ratios={}
    for numerator in methods:
        for denominator in methods:
            if numerator==denominator:continue
            per_source={}; pooled_source={}
            for src in sources:
                num=describe([r for r in rows if r['source']==src and r['method']==numerator])
                den=describe([r for r in rows if r['source']==src and r['method']==denominator])
                require(num['runs']==6 and den['runs']==6,'Incomplete source cells')
                pooled_source[src]=num['tps']/den['tps']
                paired=[{r['method']:r for r in g} for g in groups.values() if len(g)==3 and g[0]['source']==src]
                per_source[src]=math.exp(math.fsum(math.log(g[numerator]['tokens_per_second']/g[denominator]['tokens_per_second']) for g in paired)/len(paired))
            vals=[per_source[s] for s in sources]
            geo=math.exp(math.fsum(map(math.log,vals))/len(vals))
            rng=random.Random(381012);boot=[]
            for _ in range(10000):
                boot.append(math.exp(math.fsum(math.log(rng.choice(vals)) for _ in vals)/len(vals)))
            common=[g for g in groups.values() if len(g)==3 and all(r['generated_tokens']==512 for r in g)]
            both=[g for g in groups.values() if all(any(r['method']==m and r['generated_tokens']==512 for r in g) for m in (numerator,denominator))]
            ratios[numerator+'/'+denominator]=dict(source_equal_geometric_mean=geo,source_bootstrap_95_ci=[quantile(boot,.025),quantile(boot,.975)],
                source_ratios=per_source,source_pooled_ratios=pooled_source,
                common_full512_source_ratios={src:math.exp(math.fsum(math.log(next(r['outer_wall_seconds'] for r in g if r['method']==denominator)/next(r['outer_wall_seconds'] for r in g if r['method']==numerator)) for g in common if g[0]['source']==src)/sum(g[0]['source']==src for g in common)) for src in sources if any(g[0]['source']==src for g in common)},
                all_three_full512_groups=len(common),pairwise_full512_groups=len(both),
                common_full512={m:describe([r for g in common for r in g if r['method']==m]) for m in (numerator,denominator)},
                pairwise_full512={m:describe([r for g in both for r in g if r['method']==m]) for m in (numerator,denominator)})
            entry=ratios[numerator+'/'+denominator]
            entry['common_full512_ratio']=source_summary(entry['common_full512_source_ratios'])
    result=dict(status='passed' if not errors else 'failed',errors=errors,rows=len(rows),paired_groups=len(groups),sources=len(sources),
                by_method=by_method,ratios=ratios,scope='New timed sampling on ten previously explored development sources; source intervals omit repeat-level timing uncertainty')
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
