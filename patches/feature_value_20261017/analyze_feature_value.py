"""Analyze matched-prefix value and source-held-out predictability."""
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_value_20261017')
PRIOR = Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/full')
PRE = ['body_tokens','previous_position','previous_x','previous_y','previous_z',
       'recent_gap_mean','recent_gap_std','recent_unit_gap_fraction',
       'p1_gap','p1_dx','p1_dy','p1_dz','candidate_count','invalid_count',
       'single_candidate','ranker_score','ranker_margin']
POST = ['q_entropy','q_top1','q_margin','q_support']


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def interval(values, geometric=False):
    values = np.asarray(values, dtype=float)
    transformed = np.log(values) if geometric else values
    rng = np.random.default_rng(20261021)
    draws = transformed[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    transform = np.exp if geometric else lambda x:x
    return dict(estimate=float(transform(transformed.mean())),
                ci95=transform(np.quantile(draws, [.025,.975])).tolist(), sources=len(values))


def source_mean(data, key):
    return interval([np.mean([r[key] for r in data if r['scene'] == scene])
                     for scene in sorted({r['scene'] for r in data})])


def value_summary(data):
    source_ratios = []
    by_source = []
    for scene in sorted({r['scene'] for r in data}):
        part = [r for r in data if r['scene'] == scene]
        n_pf, n_po = [sum(r[k] for r in part) for k in ('n_pf','n_po')]
        t_pf, t_po = [sum(r[k] for r in part) for k in ('t_pf','t_po')]
        ratio = (n_pf/t_pf)/(n_po/t_po)
        source_ratios.append(ratio)
        by_source.append(dict(scene=scene, contexts=len(part), ratio=ratio))
    return dict(contexts=len(data), speed=interval(source_ratios, True),
        source_values=by_source,
        a_f=source_mean(data,'a_f'), a_p=source_mean(data,'a_p'),
        joint=source_mean(data,'joint'),
        g_tokens=source_mean(data,'g_tokens'), c_ms=source_mean(data,'c_ms'),
        positive_measured_contexts=sum(r['ratio'] > 1 for r in data))


def held_out(data, features, target):
    x = np.array([[r['features'][k] for k in features] for r in data])
    y = np.array([r[target] for r in data])
    scenes = np.array([r['scene'] for r in data])
    pred, reference = np.zeros(len(data)), np.zeros(len(data))
    fold_rows = []
    for scene in sorted(set(scenes)):
        train, test = scenes != scene, scenes == scene
        train_counts = {s:int((scenes[train] == s).sum()) for s in set(scenes[train])}
        weights = np.array([1/train_counts[s] for s in scenes[train]])
        weights *= len(weights)/weights.sum()
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        model.fit(x[train], y[train], ridge__sample_weight=weights)
        pred[test] = model.predict(x[test])
        reference[test] = np.average(y[train], weights=weights)
        fold_rows.append(dict(scene=scene, n=int(test.sum()),
            mae=float(mean_absolute_error(y[test],pred[test])),
            reference_mae=float(mean_absolute_error(y[test],reference[test]))))
    weights = np.array([1/(scenes == s).sum() for s in scenes]); weights /= weights.sum()
    sse = float(np.sum(weights*(pred-y)**2))
    ref_sse = float(np.sum(weights*(reference-y)**2))
    return dict(features=features, target=target, folds=len(set(scenes)), samples=len(data),
        alpha=10., source_equal_r2=float(r2_score(y,pred,sample_weight=weights)),
        improvement_over_training_mean=1-sse/ref_sse,
        source_equal_mae=float(np.sum(weights*np.abs(y-pred))),
        reference_mae=float(np.sum(weights*np.abs(y-reference))),
        pearson=float(np.corrcoef(y,pred)[0,1]), per_source=fold_rows,
        per_row_predictions=pred.tolist())


def main():
    dest = ROOT/'full'
    assert json.loads((dest/'progress.json').read_text())['status'] == 'complete'
    paths = rows(dest/'paths.jsonl')
    pairs = rows(dest/'pairs.jsonl')
    timings = rows(PRIOR/'matched_rounds.jsonl')
    contexts = json.loads((PRIOR/'matched_contexts.json').read_text())
    population = []
    for run in paths:
        for rr in run['rounds']:
            if 'labels' not in rr:
                continue
            assert all(0 <= rr['labels'][k] <= 1.00001 for k in ('a_f','a_p','joint'))
            population.append(dict(scene=run['scene'], prefix=run['prefix'], seed=run['seed'],
                features=dict(rr['pre_features'], **rr['post_features']),
                **rr['labels'], reached=int(rr['counts']['feature_verified'] > 0),
                accepted=int(rr['counts']['feature_accepted'] > 0)))
    matched = []
    for ci, context in enumerate(contexts):
        labelled = [r['rounds'][0] for r in pairs if r['context_id'] == ci]
        assert len(labelled) == 8
        first = labelled[0]
        assert all(r['pre_features'] == first['pre_features'] for r in labelled)
        assert all(abs(r['labels']['a_f']-first['labels']['a_f']) < 1e-6 for r in labelled)
        part = [r for r in timings if r['context_id'] == ci]
        assert len(part) == 16
        item = dict(context_id=ci, scene=context['scene'], origin=context['origin'],
                    prefix=context['prefix'], features=dict(first['pre_features'], **first['post_features']),
                    **first['labels'])
        for mode, suffix in [('pf','pf'),('p_only','po')]:
            rr = [r for r in part if r['mode'] == mode]
            item['n_'+suffix] = float(np.mean([r['generated_tokens'] for r in rr]))
            item['t_'+suffix] = float(np.mean([r['round']['host_seconds'] for r in rr]))
        item.update(g_tokens=item['n_pf']-item['n_po'],
                    c_ms=1000*(item['t_pf']-item['t_po']),
                    ratio=(item['n_pf']/item['t_pf'])/(item['n_po']/item['t_po']))
        matched.append(item)
    assert len(population) == 5078 and len(paths) == 56 and len(pairs) == 448
    strata = {}
    for label, test in [
        ('initial',lambda r:r['origin']=='initial'),
        ('pf_middle',lambda r:r['origin']=='pf_middle'),
        ('length_lt256',lambda r:r['features']['body_tokens']<256),
        ('length_256_1023',lambda r:256<=r['features']['body_tokens']<1024),
        ('length_ge1024',lambda r:r['features']['body_tokens']>=1024),
        ('p1_unit_gap',lambda r:r['features']['p1_gap']==1),
        ('p1_larger_gap',lambda r:r['features']['p1_gap']>1)]:
        part = [r for r in matched if test(r)]
        if part:
            strata[label] = value_summary(part)
    for key in ['a_f','q_entropy','ranker_score']:
        values = [r[key] if key=='a_f' else r['features'][key] for r in matched]
        edges = np.quantile(values, [.25,.5,.75]).tolist()
        for index in range(4):
            part = [r for r,v in zip(matched,values) if np.searchsorted(edges,v,side='right')==index]
            if part:
                strata[key+'_q'+str(index+1)] = dict(edges=edges, **value_summary(part))
    initial_by_scene = {r['scene']:r['ratio'] for r in strata['initial']['source_values']}
    middle_by_scene = {r['scene']:r['ratio'] for r in strata['pf_middle']['source_values']}
    origin_contrast = interval([middle_by_scene[s]/initial_by_scene[s]
                               for s in sorted(initial_by_scene)], True)
    calibration = {}
    for key in ['a_f','joint']:
        values = [r[key] for r in population]
        edges = np.quantile(values,[.25,.5,.75]).tolist()
        bins = []
        for index in range(4):
            part = [r for r in population if np.searchsorted(edges,r[key],side='right')==index]
            reached = sum(r['reached'] for r in part)
            bins.append(dict(n=len(part),sources=len({r['scene'] for r in part}),
                predicted_joint=float(np.mean([r['joint'] for r in part])),
                actual_joint=float(np.mean([r['accepted'] for r in part])),
                mean_a_f=float(np.mean([r['a_f'] for r in part])),
                reached_feature=reached,
                actual_conditional=sum(r['accepted'] for r in part)/reached if reached else None))
        calibration[key] = dict(edges=edges,bins=bins)
    predictability = dict(
        af_pre=held_out(population,PRE,'a_f'),
        af_post=held_out(population,PRE+POST,'a_f'),
        g_pre=held_out(matched,PRE,'g_tokens'),
        c_pre=held_out(matched,PRE,'c_ms'))
    correlations = {k:float(np.corrcoef([r['features'][k] for r in population],
                                        [r['a_f'] for r in population])[0,1])
                    for k in ['body_tokens','p1_gap','ranker_score','q_entropy','q_top1']}
    env_a=json.loads((dest/'environment_start.json').read_text())
    env_b=json.loads((dest/'environment_end.json').read_text())
    assert env_a['git_head']==env_b['git_head'] and env_a['git_status']==env_b['git_status']
    assert env_a['assets']==env_b['assets']
    result=dict(scope='development only; one-step matched prefixes; no online selector',
        validation=dict(paths_equal=len(paths),pairs_equal=len(pairs),labelled_rounds=len(population),
            total_rounds=sum(len(r['rounds']) for r in paths),git_unchanged=True,assets_unchanged=True,
            matched_eos=sum(int(r['rounds'][0]['eos']) for r in pairs)),
        population=dict(a_f=source_mean(population,'a_f'),a_p=source_mean(population,'a_p'),
            joint=source_mean(population,'joint'),actual_joint=source_mean(population,'accepted'),
            a_f_quantiles=np.quantile([r['a_f'] for r in population],[0,.1,.25,.5,.75,.9,1]).tolist(),
            correlations=correlations, calibration=calibration),
        matched=value_summary(matched),strata=strata,predictability=predictability,
        middle_vs_initial_speed_ratio=origin_contrast,
        matched_contexts=matched)
    (dest/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['validation','matched']},indent=2))
    for name,model in predictability.items():
        print(name,{k:round(model[k],4) for k in ['source_equal_r2','improvement_over_training_mean','source_equal_mae','reference_mae','pearson']})
    plot(result,dest)


def plot(result,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(14,4.2))
    names=['all','initial','pf_middle','p1_unit_gap','p1_larger_gap']
    labels=['All prefixes','Initial prefix','PF middle','P1 gap = 1','P1 gap > 1']
    for i,key in enumerate(names):
        item=(result['matched'] if key=='all' else result['strata'][key])['speed']
        lo,hi=item['ci95'];x=item['estimate']
        axes[0].errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color='#287d8e',capsize=4)
    axes[0].set_yticks(range(len(names)),labels);axes[0].invert_yaxis()
    axes[0].axvline(1,color='gray',ls='--');axes[0].set_xlabel('PF / P-only throughput')
    axes[0].set_title('Same prefixes; 95% source intervals')
    contexts=result['matched_contexts']
    for origin,color,label in [('initial','#287d8e','Initial'),('pf_middle','#d18b32','PF middle')]:
        part=[r for r in contexts if r['origin']==origin]
        axes[1].scatter([r['a_f'] for r in part],[r['g_tokens'] for r in part],c=color,label=label,alpha=.75)
    axes[1].set(xlabel='Exact Feature overlap (diagnostic)',ylabel='Extra tokens / round',title='56 contexts × 8 trials / method')
    axes[1].legend(frameon=False)
    bins=result['population']['calibration']['joint']['bins']
    xx=np.arange(4);axes[2].bar(xx-.17,[r['predicted_joint'] for r in bins],.34,label='Expected',color='#287d8e')
    axes[2].bar(xx+.17,[r['actual_joint'] for r in bins],.34,label='Observed',color='#d18b32')
    axes[2].set(xticks=xx,xticklabels=['Q1','Q2','Q3','Q4'],ylabel='P1 + F1 accepted',title='5,078 proposal rounds; diagnostic bins')
    axes[2].legend(frameon=False)
    fig.tight_layout();fig.savefig(dest/'feature_value_summary.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    main()
