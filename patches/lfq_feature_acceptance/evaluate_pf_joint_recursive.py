#!/usr/bin/env python3
"""Independent fixed-prefix evaluation and alternating K2/K3 GPU timing."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from joblib import load
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset

from bench_lfq_paired_ab import aggregate, paired_speed
from bench_pf3_joint_refresh_ab import jit_stage2_check
from collect_lfq_onpolicy_sequences import fingerprint
from collect_pf3_depth2_refresh import Rows, COLUMNS
from compare_lfq_sequences import evaluate, paired_stats
from pf3_depth2_model import load_base, load_depth2
from train_lfq_heads import sequence_ids
from train_lfq_scratch_draft import load_data
from train_pf3_depth2 import eval_model
from train_pf_joint_recursive import difference, write_json


def metrics_pair(old_values, new_values, ids):
    ids = np.asarray(ids)
    old_values = np.asarray(old_values); new_values = np.asarray(new_values)
    old_seq = {str(int(s)): float(old_values[ids == s].mean()) for s in np.unique(ids)}
    new_seq = {str(int(s)): float(new_values[ids == s].mean()) for s in np.unique(ids)}
    return {"parent_row_overlap": float(old_values.mean()), "joint_row_overlap": float(new_values.mean()),
            "parent_sequence_overlap": float(np.mean(list(old_seq.values()))),
            "joint_sequence_overlap": float(np.mean(list(new_seq.values()))),
            "parent_per_sequence": old_seq, "joint_per_sequence": new_seq,
            "paired_rows": paired_stats(new_values - old_values, ids),
            "paired_sequences": difference(new_seq, old_seq, 164039)}


def summarize(records, methods):
    sets = {name: sorted([r for r in records if r['method'] == name], key=lambda r: r['seed']) for name in methods}
    aggs = {}
    for name, rows in sets.items():
        a = aggregate(rows); stats = [r['stats'] for r in rows]
        sums = {key: sum(s.get(key, 0) for s in stats) for key in
                ('feature_checked', 'feature_accepted', 'feature_alpha_sum', 'feature_1_checked',
                 'feature_1_accepted', 'feature_1_alpha_sum', 'feature_2_checked', 'feature_2_accepted',
                 'feature_2_alpha_sum', 'feature_2_proposals')}
        f1 = sums['feature_1_checked'] or sums['feature_checked']
        f1ok = sums['feature_1_accepted'] if sums['feature_1_checked'] else sums['feature_accepted']
        f1alpha = sums['feature_1_alpha_sum'] if sums['feature_1_checked'] else sums['feature_alpha_sum']
        f2 = sums['feature_2_checked']
        calls = sum(s.get('forward_single', 0) + s.get('forward_pair', 0) +
                    sum(s.get(f'forward_block{k}', 0) for k in (4, 6, 8)) for s in stats)
        a.update(f1_checked=f1, f1_accept=f1ok / f1 if f1 else None,
                 f1_ealpha=f1alpha / f1 if f1 else None, f2_checked=f2,
                 f2_accept=sums['feature_2_accepted'] / f2 if f2 else None,
                 f2_ealpha=sums['feature_2_alpha_sum'] / f2 if f2 else None,
                 f2_reach_per_f1=f2 / f1 if f1 else None,
                 joint_success_per_f1=sums['feature_2_accepted'] / f1 if f1 else None,
                 f2_proposed=sums['feature_2_proposals'],
                 f2_not_evaluated=sums['feature_2_proposals'] - f2,
                 target_calls=calls, tokens_per_target_call=a['tokens'] / calls if calls else None)
        aggs[name] = a
    pairs = [('parent_k2', 'joint_k2'), ('parent_k3', 'joint_k3'),
             ('parent_k2', 'parent_k3'), ('joint_k2', 'joint_k3')]
    pairs += [('vanilla', name) for name in methods if name != 'vanilla']
    return {'aggregates': aggs,
            'paired': {new + '_vs_' + old: paired_speed(sets[old], sets[new]) for old, new in pairs}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=24)
    p.add_argument('--seed-start', type=int, default=480101)
    p.add_argument('--collection-seeds', type=int, default=8)
    p.add_argument('--collection-seed-start', type=int, default=481101)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--parent-base', type=Path, default=Path('profiling/feature_draft/lfq_new_onpolicy_warmstart/lfq_scratch_conditional_best.pt'))
    p.add_argument('--parent-depth2', type=Path, default=Path('profiling/feature_draft/pf3_joint_refresh_train/pf3_depth2_best.pt'))
    p.add_argument('--joint-dir', type=Path, default=Path('profiling/feature_draft/pf_joint_recursive_20261013/cpu_comparison/joint'))
    a = p.parse_args()
    if a.out_dir.exists() or min(a.seeds, a.collection_seeds) < 2 or a.max_tokens % 2 or not 8 <= a.max_tokens <= 1024:
        p.error('Use a new output directory and valid sample sizes')
    if not torch.cuda.is_available():
        p.error('CUDA is required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    joint_base = a.joint_dir / 'lfq_scratch_conditional_best.pt'
    joint_second = a.joint_dir / 'pf3_depth2_best.pt'
    models = {}
    for name, first, second in [('parent', a.parent_base, a.parent_depth2), ('joint', joint_base, joint_second)]:
        base = load_base(first, device)
        d2 = load_depth2(second, base, device, fingerprint(first))
        models[name] = (base, d2)
    online = Path('profiling/feature_draft/lfq_new_onpolicy')
    f1_ds = load_data(online, 'test')
    f1_ids = sequence_ids(online, 'test', len(f1_ds))
    if f1_ids is None:
        raise ValueError('F1 test needs sequence IDs')
    excluded = set()
    for root in [online, Path('profiling/feature_draft/state_generation_512')]:
        for split in ['train', 'val']:
            obj = torch.load(root / 'states' / f'{split}_state_pairs.pt', map_location='cpu', weights_only=True)
            if 'sequence_id' in obj:
                excluded.update(int(x) for x in obj['sequence_id'].unique())
    for name in ['pf3_depth2_data', 'pf3_refresh_stale', 'pf3_refresh_trained']:
        for split in ['train', 'val']:
            file = Path('profiling/feature_draft') / name / f'{split}.pt'
            if file.exists():
                obj = torch.load(file, map_location='cpu', weights_only=True)
                excluded.update(int(x) for x in obj['sequence_id'].unique())
    independent = set(range(a.seed_start, a.seed_start + a.seeds)) | set(range(a.collection_seed_start, a.collection_seed_start + a.collection_seeds))
    if excluded & (independent | set(int(x) for x in f1_ids.unique())):
        raise ValueError('Evaluation seed appears in training or development data')
    del obj
    a.out_dir.mkdir(parents=True)
    f1 = evaluate(f1_ds, models['parent'][0], models['joint'][0], device, 64)
    test = {'f1': metrics_pair(f1['old_overlap'], f1['new_overlap'], f1_ids.numpy()),
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            'policy': {'temperature': .9, 'top_p': .9}, 'f2': {},
            'caution': 'Selected weights are fixed. Different seeds are independent trajectories, not independent scenes. Offline F1 data has no trajectory IDs.'}
    write_json(a.out_dir / 'test_overlap.json', test)
    print('F1_TEST', json.dumps(test['f1']), flush=True)
    import gpt_position_spec_decode_smoke as dec
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE
    from bench_fast_hgb import FastHistRanker
    from profiling.s3.s3_onpolicy_feature_distill_train import make_kv_cache
    from bench_lfq_scratch_pf_cow import block2_decode
    from pf3_depth2_decode import blockk_decode, audit_prefix_math
    from bench_pf_position_joint import compare_parity
    from bench_lfq_production_ab import make_optimized, audit_policy
    audit_prefix_math()
    vq = GaussianVQVAE.load_from_checkpoint('checkpoints/vqvae_vfront.ckpt', map_location='cpu').eval()
    teacher = GaussianGPT.load_from_checkpoint('checkpoints/gpt_vfront.ckpt', map_location='cpu', vqvae=vq).eval()
    teacher.gpt.to(device).eval().requires_grad_(False)
    full = getattr(teacher.gpt, '_orig_mod', teacher.gpt); target = full.transformer
    eos = int(full.eos_token_id); vocab = int(full.position_vocab_size)
    if (eos, vocab) != (12096, 8000):
        raise ValueError('Unexpected Target vocabulary')
    ranks = target.sparse_position_rank_map.detach().cpu().long()
    ranker = FastHistRanker(load('profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib'))
    def cache(): return make_kv_cache(target, 1 + a.max_tokens + 8, device)
    common = dict(eos=eos, temp=.9, top_p=.9, device=device, p_vocab=vocab, ranks=ranks, max_tokens=a.max_tokens)
    compiled = {}; audits = {}
    val = load_data(online, 'val')
    for name, (base, d2) in models.items():
        d2_jit, audit2 = jit_stage2_check(base, d2, device, Path('profiling/feature_draft/pf3_refresh_trained'))
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            sample = (torch.zeros(1, 1024, device=device), torch.tensor([0], device=device), torch.tensor([10], device=device))
            base_jit = make_optimized(base, 'jit', sample)
        audit1 = audit_policy(base, base_jit, val, device=device, n=64, tv_tol=1e-5, abs_tol=1e-5)
        compiled[name] = (base_jit, d2_jit); audits[name] = {'f1': audit1, 'f2': audit2}
    del f1_ds, val
    with torch.inference_mode(), threadpool_limits(limits=1, user_api='openmp'):
        for proposer in ['parent', 'joint']:
            collected = Rows(); groups = []
            for seed in range(a.collection_seed_start, a.collection_seed_start + a.collection_seeds):
                before = len(collected.cols['p1'])
                blockk_decode(target, cache, ranker, compiled[proposer][0], seed=seed, max_k=3,
                              depth2_model=compiled[proposer][1], depth2_observer=collected,
                              draft_temp=.9, draft_top_p=.9, **common)
                n = len(collected.cols['p1']) - before; groups.extend([seed] * n)
                print('F2_COLLECT', proposer, seed, 'ROWS', n, flush=True)
            tensors = [torch.stack(collected.cols[k]) if k in ('hprev', 'teacher_probs') else
                       torch.tensor(collected.cols[k], dtype=torch.long) for k in COLUMNS]
            ds = TensorDataset(*tensors)
            results = [eval_model(d2, base, DataLoader(ds, batch_size=64), device, compare_stale=False)[0]
                       for base, d2 in models.values()]
            test['f2'][proposer + '_prefixes'] = metrics_pair(*results, groups)
            write_json(a.out_dir / 'test_overlap.json', test)
            print('F2_TEST', proposer, json.dumps(test['f2'][proposer + '_prefixes']), flush=True)
        methods = ['vanilla', 'parent_k2', 'joint_k2', 'parent_k3', 'joint_k3']
        def run(method, seed, limit=None):
            kw = common if limit is None else dict(common, max_tokens=limit)
            if method == 'vanilla': return dec.vanilla(target, cache, seed=seed, **kw)
            name, k = method.split('_'); base, d2 = compiled[name]
            if k == 'k2': return block2_decode(target, cache, ranker, base, seed=seed, draft_temp=.9, draft_top_p=.9, **kw)
            return blockk_decode(target, cache, ranker, base, seed=seed, max_k=3, depth2_model=d2,
                                 draft_temp=.9, draft_top_p=.9, **kw)
        for name in models:
            ref = run(name + '_k2', 990501, 128)
            got = blockk_decode(target, cache, ranker, compiled[name][0], seed=990501, max_k=2,
                               draft_temp=.9, draft_top_p=.9, **dict(common, max_tokens=128))
            compare_parity(ref, got)
            print('K2_PARITY_PASS', name, flush=True)
        for i, method in enumerate(methods): run(method, 880501 + i, 32)
        records = []
        with (a.out_dir / 'records.jsonl').open('w') as output:
            for i in range(a.seeds):
                seed = a.seed_start + i; rotation = i % len(methods)
                for method in methods[rotation:] + methods[:rotation]:
                    torch.cuda.synchronize(); start = time.perf_counter()
                    seq, stats = run(method, seed)
                    torch.cuda.synchronize(); seconds = time.perf_counter() - start
                    positions = seq[::2]
                    if not seq or len(seq) % 2 or len(seq) > a.max_tokens or any(int(ranks[y]) <= int(ranks[x]) for x, y in zip(positions, positions[1:])):
                        raise AssertionError('Invalid output sequence')
                    checked = stats.get('feature_checked', 0); pp = stats.get('position_1_proposals', 0)
                    rec = dict(seed=seed, method=method, tokens=len(seq), wall_s=seconds, tok_per_s=len(seq) / seconds,
                               f_accept=stats.get('feature_accepted', 0) / checked if checked else 0.,
                               f_ealpha=stats.get('feature_alpha_sum', 0) / checked if checked else 0.,
                               p1_accept=stats.get('position_1_accepted', 0) / pp if pp else 0.,
                               stats={k: float(v) for k, v in stats.items()})
                    records.append(rec); output.write(json.dumps(rec) + '\n'); output.flush()
                    print('DECODE', seed, method, 'TOK_S', rec['tok_per_s'], flush=True)
                progress = {'completed_seeds': i + 1, 'seeds': a.seeds, 'last_seed': seed}
                write_json(a.out_dir / 'progress.json', progress)
        result = {**summarize(records, methods), 'records': records, 'audits': audits,
                  'config': test['config'], 'policy': test['policy'], 'test_overlap_file': str(a.out_dir / 'test_overlap.json'),
                  'caution': 'One loaded Target; alternating method order; timing includes cache setup, draft, verification and sampling. Same seed can follow different sampled trajectories. No rendering timing.'}
        write_json(a.out_dir / 'summary.json', result)
        print('COMPLETE', json.dumps(result['aggregates']), flush=True)


if __name__ == '__main__': main()
