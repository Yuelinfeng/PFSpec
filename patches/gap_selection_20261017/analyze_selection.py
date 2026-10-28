"""Summarize the preregistered development comparison and same-source intervals."""
import json
from pathlib import Path

import numpy as np

ROOT=Path('/root/autodl-tmp/gaussiangpt_gap_selection_20261017/full')


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def bootstrap(values,geometric=False):
    x=np.asarray(values,dtype=float);z=np.log(x) if geometric else x
    rng=np.random.default_rng(20261023)
    draws=z[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1)
    fn=np.exp if geometric else lambda a:a
    return dict(estimate=float(fn(z.mean())),ci95=fn(np.quantile(draws,[.025,.975])).tolist(),
                independent_scenes=len(x))


def main():
    progress=json.loads((ROOT/'progress.json').read_text())
    assert progress['status']=='complete',progress
    records=read(ROOT/'runs.jsonl');inputs=json.loads((ROOT/'inputs.json').read_text())
    methods=['p_only','pf','gap','random'];scenes=sorted({r['scene'] for r in records})
    assert len(records)==len(inputs['prompts'])*3*4
    assert len(scenes)==inputs['scenes']
    per_method={m:[] for m in methods};per_prompt={}
    for item in inputs['prompts']:
        rows={m:[] for m in methods}
        for r in records:
            if r['prompt_id']==inputs['prompts'].index(item):rows[r['method']].append(r)
        assert all(len(v)==3 for v in rows.values())
        item_stats={}
        for m in methods:
            tok=sum(r['generated_tokens'] for r in rows[m]);sec=sum(r['seconds'] for r in rows[m])
            item_stats[m]=dict(tokens=tok,seconds=sec,tokens_per_second=tok/sec)
        per_prompt[item['prompt_id']]=item_stats
    source_stats={}
    for scene in scenes:
        source_stats[scene]={}
        for m in methods:
            part=[r for r in records if r['scene']==scene and r['method']==m]
            tok=sum(r['generated_tokens'] for r in part);sec=sum(r['seconds'] for r in part)
            source_stats[scene][m]=dict(tokens=tok,seconds=sec,tokens_per_second=tok/sec)
        for m in ('pf','gap','random'):
            per_method[m].append(source_stats[scene][m]['tokens_per_second']/source_stats[scene]['p_only']['tokens_per_second'])
    speed={m:dict(ratio=bootstrap(per_method[m],True),scenes_above_one=int(sum(x>1 for x in per_method[m])))
           for m in ('pf','gap','random')}
    compare={}
    for m in ('gap','random'):
        ratios=[]
        for scene in scenes:
            ratios.append(source_stats[scene][m]['tokens_per_second']/source_stats[scene]['pf']['tokens_per_second'])
        compare[m+'_vs_pf']=dict(ratio=bootstrap(ratios,True),scenes_above_one=int(sum(x>1 for x in ratios)))
        compare[m+'_vs_p_only']=speed[m]
    selected={m:dict(eligible=0,requested_pf=0,executed_pf=0,executed_p_only=0,
                     branches={},eos=0,rounds=0) for m in ('pf','p_only','gap','random')}
    for r in records:
        out=selected[r['method']];st=r['stats'];sel=st['selection']
        out['rounds']+=st['rounds'];out['eos']+=int(st['eos'])
        out['eligible']+=sel['counts'].get('eligible',0)
        out['requested_pf']+=sel['counts'].get('requested_pf',0)
        out['executed_pf']+=sel['counts'].get('executed_pf',0)
        out['executed_p_only']+=sel['counts'].get('executed_p_only',0)
        for k,v in sel['branches'].items():out['branches'][k]=out['branches'].get(k,0)+v
    for m,out in selected.items():
        out['feature_round_share']=out['executed_pf']/max(1,out['eligible'])
        out['token_rate_microseconds']=inputs.get('unused')
    calibration=json.loads((ROOT/'random_probability.json').read_text())
    actual_random=selected['random']['feature_round_share'];actual_gap=selected['gap']['feature_round_share']
    micro=calibration['selector_microseconds']
    calibration.update(actual_gap_pf_fraction=actual_gap,actual_random_pf_fraction=actual_random,
        frequency_difference_pp=100*(actual_gap-actual_random),calibration_pf_fraction=calibration['probability'],
        selector_microseconds=micro)
    for m,out in selected.items():out.pop('token_rate_microseconds',None)
    results=dict(status='complete',protocol=dict(scenes=len(scenes),unique_crops=inputs['crops'],
        prompts=len(inputs['prompts']),seeds=3,methods=methods,token_budget=inputs['budget'],
        independent_unit='source scene',random_pf_probability=calibration['probability']),
        speed_vs_p_only=speed,paired_policy_comparisons=compare,selection=selected,
        random_frequency=calibration,source_speeds=source_stats,prompt_speeds=per_prompt,
        caveats=['Development data only; three seeds per crop and prefix.',
            'Random frequency set by a separate calibration seed before the policy comparison.',
            'Source intervals do not turn crops from one source into independent scenes.',
            'No full-scene rendering was performed in G1/G2.'])
    (ROOT/'analysis.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(dict(protocol=results['protocol'],speed=results['speed_vs_p_only'],
        comparisons=results['paired_policy_comparisons'],selection=results['selection'],
        random_frequency=calibration),indent=2))
    make_plot(results)


def make_plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['p_only','pf','gap','random'];labels=['P-only','Fixed PF-K2','Gap selection','Frequency-matched random']
    fig,ax=plt.subplots(figsize=(8.3,4.6));ax.axhline(1,color='gray',ls='--',lw=1)
    for i,m in enumerate(order):
        if m=='p_only':estimate=1.;lo=hi=1.
        else:
            d=result['speed_vs_p_only'][m]['ratio'];estimate=d['estimate'];lo,hi=d['ci95']
        ax.errorbar(estimate,i,xerr=[[estimate-lo],[hi-estimate]],fmt='o',capsize=4,color='#287d8e')
        ax.text(hi+.02,i,f'{estimate:.3f}× [{lo:.3f}, {hi:.3f}]',va='center',fontsize=9)
    ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(.8,max(1.5,max(result['speed_vs_p_only'][m]['ratio']['ci95'][1] for m in ('pf','gap','random'))+.32))
    ax.set_xlabel('Tokens per second relative to P-only')
    ax.set_title(f"Development online comparison · {result['protocol']['independent_unit']} bootstrap")
    fig.tight_layout();fig.savefig(ROOT/'gap_online_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
