"""Read existing models and measure fixed-context Position proposals."""
import collections
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from joblib import load

PROJECT = Path('/root/workspace/GaussianGPT')
OLD = Path('/root/autodl-tmp/gaussiangpt_next_round_20261017')
OUT = Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
sys.path[:0] = [str(PROJECT), str(OLD)]
import step5_method_compare as old
import bench_gaussiangpt_medusa_linear as medusa
from gaussiangpt_medusa_conditional_v3.medusa_conditional_heads import ConditionalHeads, TokenSpec, sampling_probs
from draft_a.pf_ranker_eval import position_state
from gpt_position_spec_decode_smoke import choose_position


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def capture_assets():
    files = [PROJECT / x for x in [
        'bench_gaussiangpt_medusa_linear.py', 'bench_gaussiangpt_b1_two_model.py',
        'bench_gaussiangpt_pf_block2_cow.py', 'gpt_position_spec_decode_smoke.py',
        'gaussiangpt_medusa_conditional_v3/medusa_conditional_heads.py',
        'train_feature_state_distill.py', 'draft_a/pf_feature_unified.py',
        'model/gpt/nanochat_gpt.py', 'model/gpt/engine.py',
        'profiling/eagle/eagle_chain.py', 'profiling/eagle/e33_train_chain.py',
        'profiling/eagle/e48_acceptance_audit.py', 'profiling/eagle/e22_teacher_trajectory_collector.py']]
    files += [OLD / x for x in ['step5_method_compare.py', 'stage5_medusa_lowrank.py',
        'step5_train_budget_heads.py', 'step5_train_feature_same_traces.py', 'train_hydra_shared_position.py']]
    for p in files:
        if p.exists():
            dest = OUT / 'source_snapshot' / ('project' if PROJECT in p.parents else 'previous_round') / p.relative_to(PROJECT if PROJECT in p.parents else OLD)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    for name, cmd in [('git_diff.patch', ['git', 'diff', 'HEAD']), ('git_status.txt', ['git', 'status', '--short'])]:
        (OUT / name).write_text(subprocess.run(cmd, cwd=PROJECT, text=True, capture_output=True).stdout)
    evidence = OLD / 'step5_method_compare'
    for p in evidence.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.jsonl', '.md', '.txt', '.log'):
            dest = OUT / 'previous_evidence' / p.relative_to(evidence)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    save('environment.json', dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), cwd=os.getcwd(), git_head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True, capture_output=True).stdout.strip(),
        matmul_precision=torch.get_float32_matmul_precision(), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        target=str(PROJECT / 'checkpoints/gpt_vfront.ckpt'), vqvae=str(PROJECT / 'checkpoints/vqvae_vfront.ckpt'),
        ranker=str(old.RANKER), medusa=str(old.LOWRANK), hydra=str(old.HYDRA), feature=str(old.FEATURE)))


def metric(p, q, truth, prev, ranks, eos):
    q = q.float()
    legal = torch.zeros_like(q, dtype=torch.bool)
    legal[:len(ranks)] = ranks.to(q.device) > ranks[prev]
    legal[eos] = True
    ids = torch.topk(q, 5).indices.tolist()
    return dict(overlap=float(torch.minimum(p, q).sum()), q_true=float(q[truth]),
        top1_hit=int(ids[0] == truth), top5_hit=int(truth in ids),
        entropy=float(-(q * q.clamp_min(1e-30).log()).sum()), eos_probability=float(q[eos]),
        illegal_mass=float(q[~legal].sum()), sum_q=float(q.sum()),
        true_rank_gap=int(ranks[truth] - ranks[prev]),
        q_top5=[(i, float(q[i])) for i in ids])


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'A1_position_diagnostics.json').exists():
        raise FileExistsError('Completed diagnostics already exist')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    capture_assets()
    started = time.perf_counter()
    target, dev, pv, eos, ranks, make_cache = medusa.target_loader(SimpleNamespace(
        gpt=PROJECT / 'checkpoints/gpt_vfront.ckpt', vqvae=PROJECT / 'checkpoints/vqvae_vfront.ckpt'))
    linear = old.load_medusa_heads(old.LOWRANK, target, pv, eos)
    hydra = ConditionalHeads(target, TokenSpec(pv, eos-pv, pv, eos, eos+2), width=177).to(dev)
    hydra.load_state_dict(torch.load(old.HYDRA, map_location='cpu', weights_only=False)['heads'])
    hydra.eval().requires_grad_(False)
    ranker = old.FastHistRanker(load(old.RANKER))
    hook = medusa.LastBlockHook(target)
    captured = []
    handle = target.lm_head.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
    rank_fact = dict(position_count=pv, id_rank_identical=bool(torch.equal(ranks[:pv], torch.arange(pv))),
        strictly_increasing=bool(torch.all(ranks[1:pv] > ranks[:pv-1])))
    save('A0_rank_order.json', rank_fact)
    data = torch.load(old.RECORDS, map_location='cpu', weights_only=True)
    first = {}
    for r in data:
        first.setdefault(r['scene'], r)
    cases = []
    for scene, r in sorted(first.items()):
        body = r['tokens'].long().flatten()
        for prefix in (64, 256, 1024):
            if len(body) > prefix+1:
                anchors = list(range(max(1, prefix-31), prefix, 2))
                cases.append(('real', scene, body[:prefix+2], anchors, prefix))
    trace_files = [PROJECT / 'profiling/baselines_5way/b3_medusa_linear' / n / 'train_token_traces.pt' for n in ('pipeline_train', 'train_32x1024')]
    traces = medusa.read_trace_files(trace_files)
    perm = torch.randperm(len(traces), generator=torch.Generator().manual_seed(20261010)).tolist()
    val = set(perm[:max(1, round(len(traces)*.1))])
    for i, (key, body) in enumerate(traces):
        body = body.long().flatten()
        cand = torch.arange(1, len(body)-2, 2)
        indices = torch.linspace(0, len(cand)-1, min(16, len(cand))).long()
        cases.append(('validation' if i in val else 'train', key, body, cand[indices].tolist(), len(body)))
    rows, state_rows, samples = [], [], []
    for ci, (domain, scene, body, anchors, prefix) in enumerate(cases):
        full = torch.cat((torch.tensor([eos]), body)).to(dev)
        captured.clear()
        logits = target(full[None])[0]
        h = hook.value[0]
        head_h = captured[-1][0]
        state_rows.append(dict(domain=domain, scene=scene, prefix=prefix,
            normalized_vs_lm_input_max_abs=float((h-head_h).abs().max())))
        for t in anchors:
            prev, pending, truth = map(int, full[t:t+3])
            if not (0<=prev<pv and pv<=pending<eos and 0<=truth<pv):
                continue
            target_logits = logits[t+1].clone()
            target_logits[:pv].masked_fill_(ranks.to(dev) <= ranks[prev], -torch.inf)
            p = medusa.policy(target_logits, .9, .9)
            raw = linear.raw(h[t:t+1])[0][0]
            q = linear.distribution(raw, 'position', prev_p=prev, ranks=ranks, temp=.9, top_p=.9)
            qh = hydra.global_probs(sampling_probs(hydra.p1_logits(h[t:t+1], torch.tensor([prev], device=dev)), .9, .9), 'P')[0]
            q_lm = hydra.global_probs(sampling_probs(hydra.p1_logits(head_h[t:t+1], torch.tensor([prev], device=dev)), .9, .9), 'P')[0]
            later = linear.raw(h[t+1:t+2])[0][0]
            q_later = linear.distribution(later, 'position', prev_p=prev, ranks=ranks, temp=.9, top_p=.9)
            h_later = hydra.global_probs(sampling_probs(hydra.p1_logits(h[t+1:t+2], torch.tensor([prev], device=dev)), .9, .9), 'P')[0]
            state = position_state(full[:t+2].tolist(), ranks)
            selected = choose_position(ranker, state, prev)
            qr = torch.zeros_like(q)
            if selected is not None:
                qr[selected[0]] = 1
            qs = dict(medusa_current=q, hydra_current=qh, medusa_after_feature_diagnostic=q_later, hydra_after_feature_diagnostic=h_later)
            if selected is not None:
                qs['ranker'] = qr
            row = dict(domain=domain, scene=scene, requested_prefix=prefix, anchor_index=t,
                prev_position=prev, pending_feature=pending, true_next_position=truth,
                processed_tokens=t+1, true_label_index=t+2,
                hydra_norm_q_tv=float((qh-q_lm).abs().sum()/2),
                methods={name:metric(p, qq, truth, prev, ranks, eos) for name,qq in qs.items()})
            rows.append(row)
            if domain=='real' and len(samples)<10:
                cache = make_cache(t+10)
                st = collections.Counter()
                out_cache, hc = hook.forward(target, cache, full[:t+1].tolist(), ['position' if j%2==0 else 'feature' for j in range(t+1)], 0, st, 'prefill')
                hc_last = hc[-1].clone()
                out_next, hn = hook.forward(target, cache, [pending], ['position'], t+1, st, 'feature')
                pc = medusa.policy(out_next[0], .9, .9)
                sample = dict(row, processed_tail=full[max(0,t-7):t+1].tolist(),
                    cached_vs_full_hidden_max_abs=float((hc_last-h[t]).abs().max()),
                    cached_vs_full_target_tv=float((pc-p).abs().sum()/2),
                    target_top5=[(int(i),float(p[i])) for i in p.topk(5).indices],
                    cache_position_after_pending=int(cache.get_pos()))
                samples.append(sample)
                del cache
        print(json.dumps(dict(done=ci+1,total=len(cases),domain=domain,rows=len(rows),seconds=time.perf_counter()-started)),flush=True)
        del logits, h, head_h
    handle.remove()
    hook.close()
    summaries = {}
    for domain in ('real','train','validation'):
        rr = [r for r in rows if r['domain']==domain]
        summaries[domain] = {}
        for name in rr[0]['methods']:
            values = [r['methods'][name] for r in rr if name in r['methods']]
            summaries[domain][name] = dict(count=len(values), **{k:sum(v[k] for v in values)/len(values) for k in ('overlap','q_true','top1_hit','top5_hit','entropy','eos_probability','illegal_mass')})
    save('A1_position_diagnostics.json', dict(status='complete', seconds=time.perf_counter()-started,
        real_contexts=sum(r['domain']=='real' for r in rows), rows=rows, summaries=summaries))
    save('A0_state_alignment.json', dict(cases=state_rows,
        max_normalized_vs_lm_input_abs=max(r['normalized_vs_lm_input_max_abs'] for r in state_rows),
        max_hydra_q_tv=max(r['hydra_norm_q_tv'] for r in rows)))
    save('A0_ten_samples.json', samples)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()
