"""Compare MLP Feature overlap with an earlier observed Target Feature policy."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'source', 'crops', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path[:0] = [str(args.source), str(args.source/'reference'),
                    str(args.source/'reference/runtime'), str(args.project)]
    os.chdir(args.project)
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context, load_method
    from dynamic_target_graph import DynamicTargetGraph
    from proposal_adapters import build_proposal
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
    from draft_a.pf_feature_unified import _shared
    from draft_a.pf_ranker_eval import cache_for
    from draft_a.learning_eval import _fp32_runtime
    from gpt_position_spec_decode_smoke import choose_position
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = build_context(args.project, device='cuda')
    core, shared = ctx['target'], _shared()
    adapter = load_method('pf_mlp', ctx)
    runtime = DynamicTargetGraph(core, cache_capacity_buckets=(2048,))
    proposal = build_proposal(adapter, verify=False)
    records = torch.load(args.crops, map_location='cpu', weights_only=True)
    scenes = sorted({str(r['scene']) for r in records})[:10]
    rows = []
    with ExitStack() as stack:
        for context in (threadpool_limits(limits=1), torch.inference_mode(),
                        _fp32_runtime(torch), torch.autocast('cuda', enabled=False),
                        runtime.attention_context(), proposal.context()):
            stack.enter_context(context)
        for scene in scenes:
            for prefix in (64, 256, 512, 1024):
                record = next((r for r in records if str(r['scene']) == scene
                               and len(r['tokens']) >= prefix + 4), None)
                if record is None:
                    continue
                prompt = [ctx['eos']] + list(map(int, record['tokens'][:prefix]))
                state = shared.position_state(prompt, ctx['ranks'])
                first = choose_position(ctx['ranker'], state, prompt[-2])
                if first is None:
                    continue
                p1 = int(first[0])
                cache = runtime.cache_for(core, len(prompt)+8, factory=cache_for)
                capture = CaptureLastBlock(core)
                try:
                    old_logits, old_states = capture.run(
                        core, cache, prompt[:-1], shared.typed_kinds(0, len(prompt)-1), ctx['device'])
                    previous = shared.policy(old_logits[-1])[8000:12096].float().clone()
                    adapter.model.prepare(prompt, len(prompt)+8)
                    rng = torch.Generator(device='cuda').manual_seed(47201)
                    q = adapter.model.propose(prompt, p1, old_states[-1], rng).float().clone()
                    logits, _ = capture.run(core, cache, [prompt[-1], p1],
                                            shared.typed_kinds(prefix, 2), ctx['device'])
                    p1_mass = float(shared.policy(logits[0])[p1].item())
                    current = shared.policy(logits[1])[8000:12096].float().clone()
                    overlaps = {}
                    for weight in (0., .25, .5, .75, 1.):
                        mix = (1.-weight)*q + weight*previous
                        overlaps[str(weight)] = float(torch.minimum(current, mix).sum().item())
                    rows.append(dict(source=scene, prefix=prefix, p1=p1, previous_p=prompt[-2],
                                     p1_mass=p1_mass, overlaps=overlaps,
                                     q=q.cpu().tolist(), previous_policy=previous.cpu().tolist(),
                                     target_policy=current.cpu().tolist()))
                    print(json.dumps(dict(source=scene, prefix=prefix, p1_mass=p1_mass,
                                          overlaps=overlaps)), flush=True)
                finally:
                    capture.close()
                    adapter.model.close()
    proposal.close()
    by_source = {}
    for scene in scenes:
        group = [r for r in rows if r['source'] == scene]
        by_source[scene] = {
            w: dict(overlap=sum(r['overlaps'][w] for r in group)/len(group),
                    joint=sum(r['p1_mass']*r['overlaps'][w] for r in group)/len(group))
            for w in rows[0]['overlaps']}
    means = {w: {k:sum(v[w][k] for v in by_source.values())/len(by_source)
                 for k in ('overlap','joint')} for w in rows[0]['overlaps']}
    result = dict(rows=rows, contexts=len(rows), by_source=by_source, source_mean=means,
                  definition='Only prior committed Position Target Feature policy is reused; same current history and candidate. No training or online speed.',
                  project=str(args.project), source=str(args.source), crops=str(args.crops))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(contexts=len(rows), source_mean=means)))


if __name__ == '__main__':
    main()
