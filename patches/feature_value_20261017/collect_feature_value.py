"""Collect diagnostic labels without modifying the production decoder or RNG."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from joblib import load

PROJECT = Path('/root/workspace/GaussianGPT')
PRIOR = Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017')
ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_value_20261017')
sys.path[:0] = [str(PROJECT), str(PRIOR), '/root/autodl-tmp/gaussiangpt_next_round_20261017']
import step5_method_compare as prior
import bench_gaussiangpt_medusa_linear as medusa
from bench_gaussiangpt_pf_block2_cow import load_feature_model
from draft_a import pf_feature_unified as original
from branch_probe import RoundTimer, build_decoder
from run_cost_study import environment, write_json, append


class RecordingRanker:
    def __init__(self, core):
        self.core = core
        self.scores = None

    def predict_proba(self, features):
        result = self.core.predict_proba(features)
        self.scores = result[:, 1].copy()
        return result


def context_features(prefix):
    positions = np.asarray(prefix[1::2], dtype=np.int64)
    gaps = np.diff(positions[-9:])
    prev = int(positions[-1])
    return dict(body_tokens=len(prefix)-1, previous_position=prev,
                previous_x=prev // 400, previous_y=prev // 20 % 20,
                previous_z=prev % 20, history_pairs=len(positions),
                recent_gap_mean=float(gaps.mean()) if len(gaps) else 0.,
                recent_gap_std=float(gaps.std()) if len(gaps) else 0.,
                recent_unit_gap_fraction=float((gaps == 1).mean()) if len(gaps) else 0.)


class LabelTimer(RoundTimer):
    def begin_round(self, prefix, length, stats):
        super().begin_round(prefix, length, stats)
        self.current['pre_features'] = context_features(prefix)
        self.proposals = 0

    def run(self, stage, fn, *args, **kwargs):
        if kwargs.get('operation') == 'position_proposal':
            ranker = args[0]
            ranker.scores = None
            result = super().run(stage, fn, *args, **kwargs)
            self.proposals += 1
            if result is not None and self.proposals == 1:
                p1, candidates, invalid = result
                previous = int(args[-1])
                scores = ranker.scores
                ordered = np.sort(scores) if scores is not None else None
                self.current['pre_features'].update(
                    p1_gap=p1-previous, p1_x=p1 // 400, p1_y=p1 // 20 % 20,
                    p1_z=p1 % 20, p1_dx=p1 // 400-previous // 400,
                    p1_dy=p1 // 20 % 20-previous // 20 % 20,
                    p1_dz=p1 % 20-previous % 20,
                    candidate_count=int(candidates), invalid_count=int(invalid),
                    single_candidate=int(candidates == 1),
                    ranker_score=float(ordered[-1]) if ordered is not None else 0.,
                    ranker_margin=float(ordered[-1]-ordered[-2]) if ordered is not None else 0.)
                self.current['p1'] = p1
            return result
        return super().run(stage, fn, *args, **kwargs)

    def feature(self, q):
        values = q.detach().cpu().numpy().astype(np.float64)
        nonzero = values[values > 0]
        top = np.sort(values)[-2:]
        self.current['post_features'] = dict(q_entropy=float(-(nonzero*np.log(nonzero)).sum()),
            q_top1=float(top[-1]), q_margin=float(top[-1]-top[-2]),
            q_support=int(len(nonzero)))

    def labels(self, logits, q, p1, p2, f_local):
        policy = original._shared().policy
        p_position = policy(logits[0])
        p_feature = policy(logits[1])[original.P_VOCAB:original.EOS]
        p_next = policy(logits[2])
        a_p = float(p_position[p1])
        a_f = float(torch.minimum(p_feature, q).sum())
        self.current['labels'] = dict(a_p=a_p, a_f=a_f, joint=a_p*a_f,
            sampled_feature_alpha=min(1., float(p_feature[f_local]/q[f_local].clamp_min(1e-30))),
            eos1=float(p_position[original.EOS]), eos2_sampled=float(p_next[original.EOS]),
            p2_probability_sampled=float(p_next[p2]))


def observed_decoder():
    path = ROOT/'base_observed_decode.py'
    build_decoder(path)
    source = path.read_text().replace('RoundTimer(profile, torch, device, sync_p2, detail)',
                                     'LabelTimer(profile, torch, device, sync_p2, detail)')
    old = 'q = adapter.propose(prefix, p1, h_prev, rng)'
    assert source.count(old) == 1
    source = source.replace(old, old+'\n            timer.feature(q)')
    old = 'logits, states = tf([pending_f, p1, f1, p2], "target_verify")'
    assert source.count(old) == 1
    source = source.replace(old, old+'\n            timer.labels(logits, q, p1, p2, f_local)')
    destination = ROOT/'labelled_decode.py'
    destination.write_text(source)
    namespace = dict(vars(original), LabelTimer=LabelTimer)
    exec(compile(source, str(destination), 'exec'), namespace)
    return namespace['observed_decode']


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    dest = ROOT/('smoke' if args.smoke else 'full')
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest/'environment_start.json', environment())
    fn = observed_decoder()
    target, device, pv, eos, ranks, _ = medusa.target_loader(SimpleNamespace(
        gpt=PROJECT/'checkpoints/gpt_vfront.ckpt', vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    ranker = RecordingRanker(prior.FastHistRanker(load(prior.RANKER)))
    feature = load_feature_model(prior.FEATURE, prior.FEATURE_HEAD, target, device, pv, eos)[0].float().eval().requires_grad_(False)
    records = torch.load(prior.RECORDS, map_location='cpu', weights_only=True)
    first = {}
    for item in records:
        first.setdefault(item['scene'], item)
    paths = [json.loads(s) for s in (PRIOR/'full/runs.jsonl').read_text().splitlines()]
    paths = [r for r in paths if r['method'] == 'pf_plain']
    contexts = json.loads((PRIOR/'full/matched_contexts.json').read_text())
    pairs = [json.loads(s) for s in (PRIOR/'full/matched_rounds.jsonl').read_text().splitlines()]
    pairs = [r for r in pairs if r['mode'] == 'pf']
    if args.smoke:
        paths, pairs = paths[:1], pairs[:2]
    fields = ['target_calls','generated_tokens','rounds','position_accepted',
              'position_verified','feature_accepted','feature_verified']

    def run(prompt, budget, seed, max_rounds=None):
        with torch.autocast(device_type='cuda', enabled=False):
            return fn(target, original.MLPAdapter(feature), ranker, ranks,
                      prompt, budget, seed, True, max_rounds=max_rounds, detail=False)

    for group, rows in [('paths', paths), ('pairs', pairs)]:
        path = dest/(group+'.jsonl')
        previous = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        done = {r['row_id'] for r in previous}
        for i, row in enumerate(rows):
            if i in done:
                continue
            if group == 'paths':
                prompt = [eos]+first[row['scene']]['tokens'][:row['prefix']].tolist()
                stats, output = run(prompt, 256, row['actual_seed'])
                assert all(stats[k] == row['stats'][k] for k in fields)
                metadata = {k: row[k] for k in ('scene','prefix','seed','actual_seed')}
            else:
                prompt = contexts[row['context_id']]['prompt']
                stats, output = run(prompt, 8, row['seed'], 1)
                metadata = {k: row[k] for k in ('scene','prefix','seed','context_id','origin','trial')}
                observed = stats['round_profile']['rounds'][0]
                assert observed['branch'] == row['round']['branch']
                assert observed['counts'] == row['round']['counts']
            assert output == row['output'], (group, i)
            append(path, dict(row_id=i, **metadata, output_equal=True,
                output=output, rounds=stats['round_profile']['rounds']))
            write_json(dest/'progress.json', dict(stage=group, done=i+1, total=len(rows)))
            if i % 8 == 0:
                print(group, i+1, len(rows), flush=True)
    write_json(dest/'environment_end.json', environment())
    write_json(dest/'progress.json', dict(status='complete', paths=len(paths), pairs=len(pairs),
        speed_source=str(PRIOR/'full/matched_rounds.jsonl'), no_head_training=True))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
