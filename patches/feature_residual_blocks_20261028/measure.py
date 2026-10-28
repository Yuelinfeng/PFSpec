"""Measure proposal forward latency and summarize Feature acceptance by scene."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def latency(args):
    import torch
    from block_model import from_checkpoint
    from draft_a.feature_target_v2 import HEAD
    from mlp_medusa_graph import ProposalForwardGraph
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    head = torch.load(HEAD, map_location='cpu', weights_only=True)['weight']
    sample = torch.load(args.sample, map_location='cpu', weights_only=True)
    inputs = tuple(sample[k][:1].cuda() for k in ('previous_state','previous_feature','position'))
    paths = {'pf_w1': args.original}
    paths.update({f'blocks{n}': args.training_root/f'add{n}'/'selected.pt' for n in (0,3,5)})
    models, graphs, result = {}, {}, {}
    with torch.inference_mode():
        for name, path in paths.items():
            value = torch.load(path, map_location='cpu', weights_only=True)
            if name == 'pf_w1':
                from train_feature_state_distill import FeatureStateDraft
                cfg = value['model_config']
                keys = ('position_vocab','feature_vocab','d_model','emb_dim','mlp_dim','target_softcap')
                model = FeatureStateDraft(head, **{k:cfg[k] for k in keys})
                model.load_state_dict(value['model_state'], strict=True)
                model = model.cuda().float()
            else:
                model = from_checkpoint(value, head)
            model = model.eval().requires_grad_(False)
            graph = ProposalForwardGraph(model, 'forward', method_kind='mlp')
            with graph.context():
                for _ in range(20):
                    model(*inputs)
            models[name], graphs[name] = model, graph
            result[name] = dict(checkpoint=str(path), gpu_forward_us=[])
        # Rotate order to reduce effects from clock and temperature drift.
        names = list(paths)
        for repeat in range(7):
            for name in names[repeat % len(names):] + names[:repeat % len(names)]:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                with graphs[name].context():
                    start.record()
                    for _ in range(300):
                        models[name](*inputs)
                    end.record()
                end.synchronize()
                result[name]['gpu_forward_us'].append(start.elapsed_time(end)*1000/300)
        for name in names:
            result[name]['median_gpu_forward_us'] = statistics.median(result[name]['gpu_forward_us'])
            result[name]['runtime'] = graphs[name].stats()
            graphs[name].close()
    write(args.output, dict(scope='batch1 FP32 graph forward including input copies; CUDA-event intervals may include host submission gaps; excludes policy, RNG and Target',
                            gpu=torch.cuda.get_device_name(), models=result))


def summarize(args):
    rows = [json.loads(line) for line in (args.online/'records.jsonl').read_text().splitlines()]
    methods = list(dict.fromkeys(r['method'] for r in rows))
    sources = sorted({r['source'] for r in rows})
    counts = {m: {} for m in methods}
    result = {}
    for method in methods:
        for source in sources:
            own = [r for r in rows if r['method']==method and r['source']==source]
            if len(own) != 2:
                raise ValueError('Expected two prefixes per scene')
            counts[method][source] = {
                k: sum(r['counts'].get(k, 0) for r in own)
                for k in ('feature_proposed','feature_verified','feature_accepted')}
        total = {k:sum(s[k] for s in counts[method].values()) for k in
                 ('feature_proposed','feature_verified','feature_accepted')}
        result[method] = dict(**total,
            conditional_acceptance=total['feature_accepted']/total['feature_verified'],
            accepted_per_proposal=total['feature_accepted']/total['feature_proposed'],
            comparisons={})
    def rate(method, source_sample):
        accepted = sum(counts[method][s]['feature_accepted'] for s in source_sample)
        verified = sum(counts[method][s]['feature_verified'] for s in source_sample)
        return accepted/verified
    for method in methods:
        for reference in ('pf_w1','blocks0'):
            if reference==method or reference not in methods:
                continue
            rng = random.Random(281026)
            differences=[]
            for _ in range(4000):
                draw = rng.choices(sources,k=len(sources))
                differences.append(100*(rate(method,draw)-rate(reference,draw)))
            differences.sort()
            result[method]['comparisons'][reference] = dict(
                acceptance_difference_pp=100*(rate(method,sources)-rate(reference,sources)),
                ci95_pp=[differences[99],differences[3899]],
                sources_higher=sum(rate(method,[s])>rate(reference,[s]) for s in sources))
            own={r['case_id']:r for r in rows if r['method']==method}
            ref={r['case_id']:r for r in rows if r['method']==reference}
            result[method]['comparisons'][reference].update(
                matching_outputs=sum(own[k]['output_tokens']==ref[k]['output_tokens'] for k in own),
                total_pairs=len(own),
                matching_feature_counts=sum(all(own[k]['counts'].get(c)==ref[k]['counts'].get(c)
                                                for c in ('feature_proposed','feature_verified','feature_accepted'))
                                            for k in own))
    spec=importlib.util.spec_from_file_location('block_compare_summary',args.compare)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    throughput=module.paired_summary(rows,methods,[m for m in ('pf_w1','blocks0') if m in methods])
    validation={}
    for n in (0,3,5):
        root=args.training_root/f'add{n}'
        history=json.loads((root/'history.json').read_text())
        validation[f'blocks{n}']=dict(selected=json.loads((root/'selected.json').read_text()),
                                     initial=history[0], final=history[-1])
    write(args.output,dict(sources=len(sources),runs=len(rows),acceptance=result,
                           throughput=throughput,validation=validation,
                           uncertainty='paired scene resampling; histories may differ across models'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('latency')
    a.add_argument('--training-root',type=Path,required=True)
    a.add_argument('--original',type=Path,required=True)
    a.add_argument('--sample',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('summarize')
    a.add_argument('--training-root',type=Path,required=True)
    a.add_argument('--online',type=Path,required=True)
    a.add_argument('--compare',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    (latency if args.action=='latency' else summarize)(args)


if __name__=='__main__':
    main()
