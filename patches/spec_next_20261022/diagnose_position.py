"""Measure point and normalized-score proposal overlap at identical histories."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'source', 'crops', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path[:0] = [str(args.source), str(args.source/'reference'),
                    str(args.source/'reference/runtime'), str(args.project)]
    os.chdir(args.project)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context
    from dynamic_target_graph import DynamicTargetGraph
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
    from draft_a.pf_feature_unified import _shared
    from draft_a.pf_ranker_eval import cache_for
    from draft_a.learning_eval import _fp32_runtime
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = build_context(args.project, device='cuda')
    core, shared = ctx['target'], _shared()
    runtime = DynamicTargetGraph(core, cache_capacity_buckets=(2048,))
    records = torch.load(args.crops, map_location='cpu', weights_only=True)
    scenes = sorted({str(r['scene']) for r in records})[:10]
    rows = []
    with ExitStack() as stack:
        for context in (threadpool_limits(limits=1), torch.inference_mode(),
                        _fp32_runtime(torch), torch.autocast('cuda', enabled=False),
                        runtime.attention_context()):
            stack.enter_context(context)
        for scene in scenes:
            for prefix in (64, 256, 512, 1024):
                record = next((r for r in records if str(r['scene']) == scene
                               and len(r['tokens']) >= prefix + 4), None)
                if record is None:
                    continue
                prompt = [ctx['eos']] + list(map(int, record['tokens'][:prefix]))
                state = shared.position_state(prompt, ctx['ranks'])
                candidate = state.propose(prompt[-2])
                cache = runtime.cache_for(core, len(prompt) + 8, factory=cache_for)
                capture = CaptureLastBlock(core)
                try:
                    capture.run(core, cache, prompt[:-1],
                                shared.typed_kinds(0, len(prompt)-1), ctx['device'])
                    logits, _ = capture.run(core, cache, prompt[-1:],
                                            shared.typed_kinds(prefix, 1), ctx['device'])
                    target = shared.policy(logits[-1]).float().cpu().tolist()
                finally:
                    capture.close()
                row = dict(source=scene, prefix=prefix, crop_path=record.get('path'),
                           crop_seed=record.get('crop_seed'), target_policy=target)
                if candidate is None:
                    row.update(ids=[], scores=[], q=[], point_acceptance=0.,
                               probability_overlap=0., candidate_mass=0.)
                else:
                    ids, features, _, invalid = candidate
                    scores = ([1.] if len(ids) == 1 else
                              ctx['ranker'].predict_proba(features)[:, 1].tolist())
                    weights = np.asarray(scores, dtype=np.float64)
                    if not np.isfinite(weights).all() or (weights < 0).any():
                        raise ValueError('Ranker emitted invalid scores')
                    if weights.sum() <= 0:
                        weights[:] = 1.
                    q = (weights / weights.sum()).tolist()
                    point = int(ids[int(np.argmax(scores))])
                    row.update(ids=list(map(int, ids)), scores=scores, q=q,
                               point=point, invalid=int(invalid),
                               point_acceptance=target[point],
                               probability_overlap=sum(min(target[int(t)], v)
                                                       for t, v in zip(ids, q)),
                               candidate_mass=sum(target[int(t)] for t in ids))
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items()
                                  if k not in ('target_policy',)}, ensure_ascii=False), flush=True)
    by_source = {}
    for scene in scenes:
        group = [r for r in rows if r['source'] == scene]
        by_source[scene] = {k: sum(r[k] for r in group)/len(group) for k in
                           ('point_acceptance', 'probability_overlap', 'candidate_mass')}
    result = dict(rows=rows, by_source=by_source, contexts=len(rows),
                  source_mean={k: sum(v[k] for v in by_source.values())/len(by_source)
                               for k in ('point_acceptance', 'probability_overlap', 'candidate_mass')},
                  definition='Exact overlap at identical teacher prefixes, no fitted parameters; not online speed.',
                  device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  source=str(args.source), project=str(args.project), crops=str(args.crops))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('contexts','source_mean')}))


if __name__ == '__main__':
    main()
