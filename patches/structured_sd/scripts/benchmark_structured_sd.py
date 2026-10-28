#!/usr/bin/env python3
"""Validate structured SD against serial inference, then measure complete decodes.

Run with --validate-only first if desired. Normal execution always validates;
there is no unchecked timing mode. Outputs include actual lengths, checkpoints,
policy, model dtype, source hashes, GPU, per-seed paired speedups and call counts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.profile_position_selective_sd import load_models
from model.gpt.position_selective_speculative import (
    PositionSelectiveSpeculativeEngine, make_kv_cache, sampling_probs,
    sample_from_probs, unwrap_compiled,
)
from model.gpt.structured_speculative import StructuredSpeculativeEngine


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-checkpoint', default='checkpoints/gpt_vfront.ckpt')
    p.add_argument('--draft-checkpoint', default='profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt')
    p.add_argument('--vqvae-checkpoint', default='checkpoints/vqvae_vfront.ckpt')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--warmup-tokens', type=int, default=32)
    p.add_argument('--validation-tokens', type=int, default=24)
    p.add_argument('--seeds', type=int, nargs='+', default=[5, 6, 7])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--k-pairs', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--temperature', type=float, default=.9)
    p.add_argument('--top-p', type=float, default=.9)
    p.add_argument('--top-k', type=int, default=None)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--output-json', required=True)
    return p.parse_args()


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(clean(report), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


@torch.inference_mode()
def ar_generate(outer, *, max_tokens, temperature, top_k, top_p, seed):
    outer = unwrap_compiled(outer)
    model = outer.transformer
    device = model.get_device()
    cache = make_kv_cache(model, max_tokens)
    rng = torch.Generator(device=device).manual_seed(seed)
    eos = int(outer.eos_token_id)
    current = torch.tensor([[eos]], dtype=torch.long, device=device)
    output = []
    while len(output) < max_tokens:
        logits = model(current, kv_cache=cache)[:, -1, :]
        current = sample_from_probs(sampling_probs(logits, temperature, top_k, top_p), rng)
        token = int(current.item())
        output.append(token)
        if token == eos:
            break
    return torch.tensor([output], device=device), {'generated_tokens': len(output),
            'terminated_on_eos': output[-1] == eos, 'target_calls': len(output)}


@torch.inference_mode()
def validate(target, draft, args):
    """Real checkpoint tests; all expensive cache replays stay outside timing."""
    result = {'passed': False, 'block_equivalence': [], 'greedy_equality': [],
              'rollback_audits': [], 'max_logit_error': 0., 'max_policy_tv': 0.,
              'max_kv_error': 0.}
    n = args.validation_tokens
    seq, _ = ar_generate(target.gpt, max_tokens=n, temperature=.9, top_k=None, top_p=.9, seed=5)
    eos = int(unwrap_compiled(target.gpt).eos_token_id)
    history = [eos] + seq[0].tolist()
    if history[-1] == eos:
        history.pop()
    for name, outer in [('target', target.gpt), ('draft', draft.gpt)]:
        model = unwrap_compiled(outer).transformer
        device = model.get_device()
        ids = torch.tensor([history], dtype=torch.long, device=device)
        reference = make_kv_cache(model, n+2)
        rows = [model(ids[:, i:i+1], kv_cache=reference) for i in range(ids.shape[1])]
        serial = torch.cat(rows, dim=1)
        for size in (2, 4, 5, 8, 9):
            cache = make_kv_cache(model, n+2)
            blocks = [model(ids[:, i:i+size], kv_cache=cache) for i in range(0, ids.shape[1], size)]
            actual = torch.cat(blocks, dim=1)
            if not torch.equal(torch.isfinite(actual), torch.isfinite(serial)):
                raise AssertionError(f'{name} block {size}: structural masks differ')
            mask = torch.isfinite(serial)
            err = float((actual[mask]-serial[mask]).abs().max())
            p = sampling_probs(serial[0], .9, None, .9)
            q = sampling_probs(actual[0], .9, None, .9)
            tv = float((p-q).abs().sum(-1).max()/2)
            # Numeric thresholds are declared before measurement. Any support
            # mismatch or excessive policy error prevents performance results.
            if err > .002 or tv > .001:
                raise AssertionError(f'{name} block {size}: logit_error={err}, policy_tv={tv}')
            result['max_logit_error'] = max(result['max_logit_error'], err)
            result['max_policy_tv'] = max(result['max_policy_tv'], tv)
            result['block_equivalence'].append(dict(model=name, block=size, logit_error=err, policy_tv=tv))
            print('BLOCK_EQ', name, size, err, tv, flush=True)

    for k in sorted(set(args.k_pairs)):
        kwargs = dict(max_tokens=n, temperature=0., top_k=None, top_p=None, seed=5)
        expected, _ = ar_generate(target.gpt, **kwargs)
        engine = StructuredSpeculativeEngine(target.gpt, draft.gpt, k_pairs=k, check_invariants=True)
        actual = engine.generate(**kwargs)
        if not torch.equal(actual.tokens, expected):
            raise AssertionError(f'greedy output mismatch for K={k}')
        result['greedy_equality'].append(dict(k=k, tokens=int(expected.shape[1])))
        print('GREEDY_EQ', k, expected.shape[1], flush=True)

    # Replay committed history with serial forwards, compare actual K/V and
    # metadata after every round, including Feature rejection and bonus cases.
    audit_count = 0
    def audit(output, tc, dc):
        nonlocal audit_count
        for outer, cache in [(target.gpt, tc), (draft.gpt, dc)]:
            model = unwrap_compiled(outer).transformer
            full = torch.tensor([[eos]+output], dtype=torch.long, device=model.get_device())
            ref = make_kv_cache(model, n+2)
            count = cache.get_pos()
            for i in range(count):
                model(full[:, i:i+1], kv_cache=ref)
            if not torch.equal(cache.get_tokens(count), ref.get_tokens(count)):
                raise AssertionError('rollback token metadata mismatch')
            for name in ('k_cache', 'v_cache'):
                a = getattr(cache, name)[:, :, :count]
                b = getattr(ref, name)[:, :, :count]
                err = float((a-b).abs().max()) if count else 0.
                result['max_kv_error'] = max(result['max_kv_error'], err)
                torch.testing.assert_close(a, b, atol=.002, rtol=.002)
        audit_count += 1
    for seed in args.seeds:
        engine = StructuredSpeculativeEngine(target.gpt, draft.gpt, k_pairs=2,
                   check_invariants=True, audit_callback=audit)
        r = engine.generate(max_tokens=n, temperature=.9, top_k=None, top_p=.9, seed=seed)
        result['rollback_audits'].append(dict(seed=seed, stats=r.stats.as_dict()))
        print('ROLLBACK_EQ', seed, r.stats.position_rejects, r.stats.feature_rejects, flush=True)
    result['audit_rounds'] = audit_count
    result['passed'] = True
    return result


@torch.inference_mode()
def main():
    args = parse_args()
    if min(args.max_tokens, args.warmup_tokens, args.validation_tokens, args.repeats) < 1:
        raise ValueError('token counts and repeats must be positive')
    report = {'config': vars(args), 'status': 'loading', 'runs': []}
    report['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        'model/gpt/structured_speculative.py', 'model/gpt/position_selective_speculative.py',
        'model/gpt/nanochat_gpt.py', 'scripts/benchmark_structured_sd.py']}
    report['checkpoints'] = {p: {'size': Path(p).stat().st_size, 'mtime_ns': Path(p).stat().st_mtime_ns}
        for p in (args.target_checkpoint, args.draft_checkpoint, args.vqvae_checkpoint)}
    target, draft = load_models(args)
    outer = unwrap_compiled(target.gpt)
    report['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if torch.device(args.device).type == 'cuda' else 'cpu',
        'dtype': str(next(outer.transformer.parameters()).dtype),
        'rope_basis': outer.gpt_config.rope_basis,
        'hard_constraints': outer.gpt_config.sparse_position_hard_constraints}
    try:
        report['validation'] = validate(target, draft, args)
    except Exception as error:
        report.update(status='validation_failed', error=repr(error))
        save(args.output_json, report)
        raise
    report['status'] = 'validated'
    save(args.output_json, report)
    if args.validate_only:
        print('VALIDATION PASS', args.output_json, flush=True)
        return

    engines = {'position_only': PositionSelectiveSpeculativeEngine(target.gpt, draft.gpt)}
    engines.update({f'structured_k{k}': StructuredSpeculativeEngine(target.gpt, draft.gpt,
                      k_pairs=k, record_blocks=False) for k in args.k_pairs})
    methods = ['target_ar'] + list(engines)
    def run(method, seed, count):
        kwargs = dict(max_tokens=count, temperature=args.temperature, top_k=args.top_k,
                      top_p=args.top_p, seed=seed)
        if method == 'target_ar':
            return ar_generate(target.gpt, **kwargs)
        r = engines[method].generate(**kwargs)
        return r.tokens, r.stats.as_dict()

    for method in methods:
        run(method, 100005, args.warmup_tokens)
    print('WARMUP COMPLETE', flush=True)
    for seed_idx, seed in enumerate(args.seeds):
        for repeat in range(args.repeats):
            order = methods if repeat % 2 == 0 else list(reversed(methods))
            shift = seed_idx % len(order)
            order = order[shift:] + order[:shift]
            group = []
            for method in order:
                sync(args.device)
                start = perf_counter()
                tokens, stats = run(method, seed, args.max_tokens)
                sync(args.device)
                ms = (perf_counter()-start)*1000
                count = int(tokens.shape[1])
                row = dict(seed=seed, repeat=repeat, method=method, elapsed_ms=ms,
                           generated_tokens=count, tokens_per_s=count*1000/ms, stats=stats,
                           token_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest())
                group.append(row)
                print('RUN', seed, repeat, method, count, round(ms, 3), round(row['tokens_per_s'], 3), flush=True)
            base = next(r['tokens_per_s'] for r in group if r['method']=='target_ar')
            selective = next(r['tokens_per_s'] for r in group if r['method']=='position_only')
            for r in group:
                r['speedup_vs_target'] = r['tokens_per_s']/base
                r['speedup_vs_position_only'] = r['tokens_per_s']/selective
            report['runs'].extend(group)
            report['status'] = 'running'
            save(args.output_json, report)
    report['summary'] = {}
    for method in methods:
        rows = [r for r in report['runs'] if r['method']==method]
        report['summary'][method] = {name: {'mean': statistics.mean(r[name] for r in rows),
              'median': statistics.median(r[name] for r in rows),
              'min': min(r[name] for r in rows), 'max': max(r[name] for r in rows)}
              for name in ('tokens_per_s', 'speedup_vs_target', 'speedup_vs_position_only')}
    report['status'] = 'complete'
    save(args.output_json, report)
    print('COMPLETE', json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
