"""Compare the single-round probe with the measured decoder on a real GPU."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--crops', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path[:0] = [str(args.source), str(args.source/'reference'),
                    str(args.source/'reference/runtime'), str(args.project)]
    os.chdir(args.project)
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context, load_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from tree_decoder import decode
    from proposal_adapters import build_proposal
    from probe_decoder import decode_probe
    from draft_a.pf_ranker_eval import cache_for
    from draft_a.learning_eval import _fp32_runtime
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = build_context(args.project, device='cuda')
    core = ctx['target']
    adapter = load_method('pf_mlp', ctx)
    runtime = DynamicTargetGraph(core, cache_capacity_buckets=(2048,))
    tree = TargetTreeRuntime(core, runtime, graph=True, max_width=4)
    proposal = build_proposal(adapter, verify=False)
    factory = lambda core, length: runtime.cache_for(core, length, factory=cache_for)
    raw = torch.load(args.crops, map_location='cpu', weights_only=True)
    record = next(r for r in raw if len(r['tokens']) >= 260)
    rows = []
    with ExitStack() as stack:
        for context in (threadpool_limits(limits=1), torch.inference_mode(),
                        _fp32_runtime(torch), torch.autocast('cuda', enabled=False),
                        proposal.context(), runtime.attention_context()):
            stack.enter_context(context)
        for prefix_len in (64, 256):
            prompt = [12096] + list(map(int, record['tokens'][:prefix_len]))
            for seed in (46101, 46102):
                for action, width, budget in (('K1', 1, 2), ('W1', 1, 4), ('W2', 2, 4)):
                    old_stats, old_out = decode(core, adapter.model, ctx['ranker'], ctx['ranks'],
                        prompt, budget, seed, tree_runtime=tree, width=width, cache_factory=factory)
                    for diagnostics in ('off', 'components'):
                        stats, suffix = decode_probe(core, adapter.model, ctx['ranker'], ctx['ranks'],
                            prompt, seed, action_id=action, action_budget=budget,
                            tree_runtime=tree, cache_factory=factory, runtime=runtime,
                            max_rounds=1, diagnostics=diagnostics)
                        r = stats['generated_tokens']
                        row = dict(prefix=prefix_len, seed=seed, action=action,
                                   diagnostics=diagnostics, R=r, output_tokens=list(map(int,suffix)),
                                   original_first_R=old_stats['round_generated_lengths'][0],
                                   output_equal=list(suffix)==list(old_out[:r]),
                                   progress_equal=r==old_stats['round_generated_lengths'][0],
                                   accepted_equal=stats['accepted_lengths']==old_stats['accepted_lengths'][:1],
                                   rounds_equal=stats['rounds']==1)
                        rows.append(row)
                for width in (1, 2):
                    old_stats, old_out = decode(core, adapter.model, ctx['ranker'], ctx['ranks'],
                        prompt, 16, seed, tree_runtime=tree, width=width, cache_factory=factory)
                    stats, suffix = decode_probe(core, adapter.model, ctx['ranker'], ctx['ranks'],
                        prompt, seed, action_id=f'W{width}', action_budget=16,
                        tree_runtime=tree, cache_factory=factory, runtime=runtime,
                        max_rounds=None, diagnostics='off')
                    names = ('target_calls','draft_calls','ranker_calls','rounds','k1_steps',
                             'tree_verify_calls','feature_branch_hits','recovery_calls',
                             'accepted_lengths','round_generated_lengths')
                    rows.append(dict(prefix=prefix_len, seed=seed, action=f'W{width}', scope='full16',
                                     output_equal=list(suffix)==list(old_out),
                                     counts_equal=all(stats[k]==old_stats[k] for k in names)))
    proposal.close()
    passed = all(all(v for k,v in row.items() if k.endswith('_equal')) for row in rows)
    result = dict(passed=passed, observations=len(rows), rows=rows,
                  source=str(args.source), project=str(args.project), scene=record.get('scene'),
                  device=torch.cuda.get_device_name(), torch=torch.__version__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=passed, observations=len(rows), output=str(args.out))))
    if not passed:
        raise AssertionError('Probe differs from measured decoder')


if __name__ == '__main__':
    main()
