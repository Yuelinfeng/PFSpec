"""Summarize paired Feature mechanisms without treating training seeds as scenes."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def analyze(args):
    root=args.root
    rows=[json.loads(x) for x in (root/'online_main/records.jsonl').read_text().splitlines()]
    if len(rows)!=420:
        raise ValueError('Expected 420 complete observations')
    sources=sorted({r['source'] for r in rows})
    methods=list(dict.fromkeys(r['method'] for r in rows))
    spec=importlib.util.spec_from_file_location('single_step_stats',args.compare)
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    result=dict(sources=len(sources),runs=len(rows),methods={},paired={})
    for method in methods:
        own=[r for r in rows if r['method']==method]
        counts={k:sum(r['counts'].get(k,0) for r in own) for k in
                ('feature_proposed','feature_verified','feature_accepted')}
        result['methods'][method]=dict(**counts,
            acceptance=counts['feature_accepted']/counts['feature_verified'],
            accepted_per_proposal=counts['feature_accepted']/counts['feature_proposed'],
            tokens_per_second=sum(r['generated_tokens'] for r in own)/sum(r['seconds'] for r in own))
    per_source_logs={s:[] for s in sources}
    counts_by_method={method:{source:{
        key:sum(r['counts'].get(key,0) for r in rows if r['method']==method and r['source']==source)
        for key in ('feature_accepted','feature_verified')}
        for source in sources} for method in methods}
    def acceptance(method,draw):
        count=counts_by_method[method]
        return sum(count[s]['feature_accepted'] for s in draw)/sum(count[s]['feature_verified'] for s in draw)
    for seed in (20261018,20261019,20261020):
        m,t=f'mlp:s{seed}',f'transformer:s{seed}'
        own=[r for r in rows if r['method'] in (m,t)]
        summary=compare.paired_summary(own,[m,t],[m])['methods'][t]['comparisons'][m]
        indexed={(r['source'],r['prefix'],r['method']):r for r in own}
        for source in sources:
            for prefix in (64,256):
                a,b=indexed[(source,prefix,t)],indexed[(source,prefix,m)]
                per_source_logs[source].append(math.log((a['generated_tokens']/a['seconds']) /
                                                        (b['generated_tokens']/b['seconds'])))
        validation={}
        for kind in ('mlp','transformer'):
            dest=root/'train_01'/str(seed)/kind
            selected=json.loads((dest/'selected.json').read_text())
            history=json.loads((dest/'history.json').read_text())
            validation[kind]=dict(selected=selected,initial=history[0],final=history[-1])
        rng=random.Random(281026)
        differences=[]
        for _ in range(4000):
            draw=rng.choices(sources,k=len(sources))
            differences.append(100*(acceptance(t,draw)-acceptance(m,draw)))
        differences.sort()
        result['paired'][str(seed)]=dict(throughput=summary,validation=validation,
            acceptance_difference_pp=100*(result['methods'][t]['acceptance']-result['methods'][m]['acceptance']),
            acceptance_ci95_pp=[differences[99],differences[3899]])
    means=[statistics.mean(per_source_logs[s]) for s in sources]
    rng=random.Random(281026)
    draws=sorted(math.exp(statistics.mean(rng.choices(means,k=len(means)))) for _ in range(4000))
    result['combined_throughput']=dict(ratio=math.exp(statistics.mean(means)),
        ci95=[draws[99],draws[3899]],sources=len(sources),
        unit='scene; both prefixes and all three training seeds retained within scene')
    write(root/'analysis.json',result)


def latency(args):
    import torch
    from feature_model import load_model,make_model
    from mlp_medusa_graph import ProposalForwardGraph
    root=args.root
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assets={k:v.cuda() for k,v in torch.load(root/'fixed_assets.pt',map_location='cpu',weights_only=True).items()}
    data_plan=json.loads((root/'train_01/20261018/mlp/data_plan.json').read_text())
    sample=torch.load(data_plan['splits']['val']['records'][0]['path'],weights_only=True)
    inputs=tuple(sample[k][:1].cuda() for k in ('previous_state','previous_feature','position'))
    models,graphs,result={},{},{}
    with torch.inference_mode():
        original=torch.load(args.original,map_location='cpu',weights_only=True)
        model=make_model('mlp',assets);model.load_state_dict(original['model_state'],strict=True)
        models['pf_w1']=model.cuda().float().eval().requires_grad_(False)
        for seed in (20261018,20261019,20261020):
            for kind in ('mlp','transformer'):
                value=torch.load(root/'train_01'/str(seed)/kind/'selected.pt',map_location='cpu',weights_only=True)
                models[f'{kind}:s{seed}']=load_model(value,assets).eval().requires_grad_(False)
        for name,model in models.items():
            graph=ProposalForwardGraph(model,'forward',method_kind='mlp')
            with graph.context():
                for _ in range(20):model(*inputs)
            graphs[name]=graph;result[name]=dict(interval_us=[])
        names=list(models)
        for repeat in range(7):
            for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                with graphs[name].context():
                    begin.record()
                    for _ in range(200):models[name](*inputs)
                    end.record()
                end.synchronize()
                result[name]['interval_us'].append(begin.elapsed_time(end)*1000/200)
        for name in names:
            result[name]['median_us']=statistics.median(result[name]['interval_us'])
            result[name]['graph']=graphs[name].stats();graphs[name].close()
    write(root/'latency.json',dict(models=result,
        scope='batch1 FP32 graph forward including input copies and possible host submission gaps; policy and Target excluded'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('analyze','latency'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--compare',type=Path)
    p.add_argument('--original',type=Path)
    args=p.parse_args();(analyze if args.phase=='analyze' else latency)(args)


if __name__=='__main__':main()
