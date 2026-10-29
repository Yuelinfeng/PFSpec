#!/usr/bin/env python3
"""Compare ordinary and graph execution for W1, W2 and Rescue."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repro', type=Path, default=Path('/root/autodl-tmp/pfspec_repro_4090d_20261028'))
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=128)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    repro = args.repro.resolve()
    project, tree, candidate = repro/'project', repro/'tree', repro/'candidate'
    sys.path[:0] = [str(candidate), str(repro/'deps'), str(tree), str(tree/'reference'),
                    str(tree/'reference/runtime'), str(project)]
    os.chdir(project)
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context, load_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from position_rescue_runtime import PositionRescueRuntime
    from tree_decoder import decode as decode_tree
    from position_rescue_decoder import decode as decode_rescue
    from proposal_adapters import build_proposal
    from draft_a.pf_ranker_eval import cache_for
    from draft_a.learning_eval import _fp32_runtime

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = torch.load(args.crops, map_location='cpu', weights_only=True)
    record = next(r for r in records if len(r['tokens']) >= 260)
    tokens = record['tokens'].tolist()
    fields = ('target_calls', 'target_input_tokens', 'draft_calls', 'ranker_calls', 'rounds',
              'accepted_lengths', 'round_generated_lengths', 'position_proposed', 'position_verified',
              'position_accepted', 'feature_proposed', 'feature_verified', 'feature_accepted',
              'tree_verify_calls', 'candidate_tree_nodes', 'target_tree_nodes', 'recovery_calls',
              'position_backup_nodes', 'position_backup_rejections', 'position_rescue_hits')
    result = []
    with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast('cuda', enabled=False):
        context = build_context(project, device='cuda')
        target = context['target']
        adapter = load_method('pf_mlp', context)
        for method in ('w1', 'w2', 'rescue'):
            for length in (64, 256):
                runs = []
                for optimized in (False, True):
                    runtime = DynamicTargetGraph(target, cache_capacity_buckets=(2048,))
                    tree_runtime = TargetTreeRuntime(target, runtime, graph=optimized, max_width=4)
                    rescue_runtime = PositionRescueRuntime(target, runtime, graph=optimized) if method == 'rescue' else None
                    proposal = build_proposal(adapter, verify=False) if optimized else None
                    factory = lambda core, size: runtime.cache_for(core, size, factory=cache_for)
                    decoder = decode_rescue if rescue_runtime else decode_tree
                    kwargs = dict(tree_runtime=tree_runtime, width=2 if method == 'w2' else 1, cache_factory=factory)
                    if rescue_runtime:
                        kwargs['rescue_runtime'] = rescue_runtime
                    with (proposal.context() if proposal else nullcontext()), runtime.attention_context(enabled=optimized):
                        stats, output = decoder(target, adapter.model, context['ranker'], context['ranks'],
                                                [context['eos']] + tokens[:length], args.budget, 68101, **kwargs)
                    torch.cuda.synchronize()
                    runs.append((stats, output))
                    if proposal:
                        proposal.close()
                    del runtime, tree_runtime, rescue_runtime, proposal
                    gc.collect()
                a, b = runs
                different = [field for field in fields if a[0].get(field) != b[0].get(field)]
                row = dict(method=method, prefix=length, tokens=len(a[1]),
                           output_equal=a[1] == b[1], differing_counts=different)
                result.append(row)
                print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps({'source': str(record.get('scene') or record.get('source')),
                                   'budget': args.budget, 'rows': result,
                                   'passed': all(r['output_equal'] and not r['differing_counts'] for r in result)}, indent=2) + '\n')
    if not all(r['output_equal'] and not r['differing_counts'] for r in result):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
