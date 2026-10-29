#!/usr/bin/env python3
"""B1-Hybrid A/B: canonical D8 unified vs Position Ranker + same D8 Feature.

Draft: ONE standard, autoregressive, native GaussianGPT D8; proposes P,F,P
with independent KV. Target: frozen 24-layer native GaussianGPT, Block4
causal verification. Uses exact min(1,p/q) acceptance and (p-q)+ residual;
Target and D8 use identical sparse type/rank constraints, but separate caches.

EOS support: P1=EOS -> single-token verification, P2=EOS -> Block3 verification.
Near the token cap: one-pair Block2 verification. No frozen files edited.

Requires repository-root support scripts:
  bench_gaussiangpt_pf_block2_cow.py (Frozen),
  gpt_position_spec_decode_smoke.py,
  profiling/s3/s3_onpolicy_feature_distill_train.py

--self-test runs fake causal CPU target/draft with no model checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

D8_DEFAULT = ('profiling/frozen/S2_3c_d8_canonical_final_031d916/checkpoint/'
              'd8_canonical_val_best_epoch37_step50502.ckpt')


def policy(logits, temperature, top_p):
    # Exact same finite-precision top-p rule as Frozen v1.
    from eval_lean_candidate_to_draft import target_policy
    return target_policy(logits, temp=temperature, top_p=top_p)


def draw(p, generator):
    return int(torch.multinomial(p, 1, generator=generator).item())


def acceptance(p, q, token, generator):
    qv = float(q[token].item())
    if not qv > 0:
        raise AssertionError('Sampled proposal has q(token)=0')
    alpha = min(1.0, float(p[token].item()) / qv)
    return bool(float(torch.rand((), device=p.device, generator=generator).item()) < alpha)


def residual(p, q, generator):
    # Same full-vocabulary vector p/q at the same causal, typed prefix.
    positive = (p.float() - q.float()).clamp_min(0)
    z = float(positive.sum().item())
    if not z > 1e-9:
        raise AssertionError('Rejected proposal has no positive residual')
    return draw(positive / z, generator)


def rewind(cache, keep, old):
    if int(cache.get_pos()) != old or not 0 <= keep <= old:
        raise AssertionError(f'KV rewind mismatch: {keep=} {old=} actual={cache.get_pos()}')
    cache.cache_seqlens.fill_(keep)
    cache.token_cache[:, keep:old].fill_(-1)
    if int(cache.get_pos()) != keep:
        raise AssertionError('KV rewind failed')


def assert_pos(token, previous, p_vocab, ranks):
    if not (0 <= token < p_vocab and int(ranks[token]) > int(ranks[previous])):
        raise AssertionError(f'Illegal/nonmonotone Position {previous}->{token}')


def assert_feat(token, p_vocab, eos):
    if not p_vocab <= token < eos:
        raise AssertionError(f'Illegal Feature ID {token}')


@torch.inference_mode()
def b1_decode(target, draft, make_target_cache, make_draft_cache,
              *, eos, p_vocab, ranks, max_tokens, seed, device,
              target_temp=0.9, target_top_p=0.9,
              draft_temp=0.9, draft_top_p=0.9):
    """Return complete pairs, Counter. No use of Ours' specialized drafts.

    Loop invariant: both KV caches process [BOS, ..., P_(i-1)] and current
    F_(i-1) is pending. Target and Draft share committed token prefix.
    Three D8 forwards construct q(P1), q(F1|P1), q(P2|P1,F1).
    """
    if not (max_tokens >= 2 and max_tokens % 2 == 0):
        raise ValueError('max_tokens must be positive and even')
    if not (target_temp > 0 and draft_temp > 0 and 0 < target_top_p <= 1 and 0 < draft_top_p <= 1):
        raise ValueError('Invalid policy parameters')
    target_cache = make_target_cache()
    draft_cache = make_draft_cache()
    tg = torch.Generator(device=device).manual_seed(seed)
    # Independent draft sampling stream: no accidental reuse of target draw.
    dg = torch.Generator(device=device).manual_seed(seed + 1000003)
    st = Counter()
    output = []

    def forward(model, cache, tokens, kinds, prefix, counter_name):
        if len(tokens) != len(kinds):
            raise AssertionError('logit_types mismatch')
        if int(cache.get_pos()) != prefix:
            raise AssertionError(f'{counter_name}: cache at {cache.get_pos()} expected {prefix}')
        x = torch.as_tensor([tokens], device=device, dtype=torch.long)
        out = model(x, kv_cache=cache, logit_types=kinds)[0]
        if out.shape[0] != len(kinds) or int(cache.get_pos()) != prefix + len(tokens):
            raise AssertionError(f'{counter_name}: model/cache shape or advancement mismatch')
        st[counter_name] += 1
        return out

    def tp(logits):
        return policy(logits, target_temp, target_top_p)

    def qp(logits):
        return policy(logits, draft_temp, draft_top_p)

    # Bootstrap: identical Target-only first P/F pair; process same tokens in D8.
    lt = forward(target, target_cache, [eos], ['position'], 0, 'forward_single')
    p0 = draw(tp(lt[0]), tg)
    if p0 == eos:
        st['eos'] += 1
        return output, st
    if not 0 <= p0 < p_vocab:
        raise AssertionError('Invalid initial Position')
    lt = forward(target, target_cache, [p0], ['feature'], 1, 'forward_single')
    f0 = draw(tp(lt[0]), tg)
    assert_feat(f0, p_vocab, eos)
    output.extend((p0, f0))
    if len(output) == max_tokens:
        return output, st
    forward(draft, draft_cache, [eos, p0], ['position', 'feature'], 0, 'draft_prefill')
    prev_p, pending_f = p0, f0

    def finish_pair(p, f):
        assert_pos(p, prev_p, p_vocab, ranks)
        assert_feat(f, p_vocab, eos)
        output.extend((p, f))

    while len(output) < max_tokens:
        before = int(target_cache.get_pos())
        if before != int(draft_cache.get_pos()):
            raise AssertionError('Draft/Target committed prefix mismatch')
        # q1 = D8(next Position | committed prefix incl. pending previous F).
        ld = forward(draft, draft_cache, [pending_f], ['position'], before, 'draft_single')
        q1 = qp(ld[0])
        p1_proposal = draw(q1, dg)
        st['p1_checked'] += 1

        # Boundary or proposed EOS: exact K1 speculative verification.
        # EOS cannot be fed into a future Feature slot in a Block2.
        k1 = len(output) + 2 == max_tokens or p1_proposal == eos
        if k1:
            st['k1_steps'] += 1
            if p1_proposal == eos:
                l = forward(target, target_cache, [pending_f], ['position'], before, 'forward_single')
            else:
                l = forward(target, target_cache, [pending_f, p1_proposal],
                            ['position', 'feature'], before, 'forward_pair')
            p1dist = tp(l[0])
            st['p1_overlap_sum'] += float(torch.minimum(p1dist, q1).sum().item())
            accepted1 = acceptance(p1dist, q1, p1_proposal, tg)
            if accepted1:
                st['p1_accepted'] += 1
                chosen = p1_proposal
            else:
                st['p1_rejected'] += 1
                chosen = residual(p1dist, q1, tg)
            if chosen == eos:
                # If verifying [pendingF, proposedP] and rejected -> undo invalid P.
                if p1_proposal != eos:
                    rewind(target_cache, before + 1, before + 2)
                st['eos'] += 1
                break
            assert_pos(chosen, prev_p, p_vocab, ranks)
            if accepted1 and p1_proposal != eos:
                fdist = tp(l[1])
            else:
                if p1_proposal != eos:
                    rewind(target_cache, before + 1, before + 2)
                fl = forward(target, target_cache, [chosen], ['feature'], before + 1,
                             'forward_single')
                fdist = tp(fl[0])
            f1 = draw(fdist, tg)
            assert_feat(f1, p_vocab, eos)
            finish_pair(chosen, f1)
            prev_p, pending_f = chosen, f1
            if len(output) != max_tokens:
                # Proposed EOS was rejected: repair D8 cache to the same prefix.
                forward(draft, draft_cache, [chosen], ['feature'], before + 1,
                        'draft_repair')
            continue

        assert_pos(p1_proposal, prev_p, p_vocab, ranks)
        ld = forward(draft, draft_cache, [p1_proposal], ['feature'], before + 1, 'draft_single')
        qf = qp(ld[0])
        f1_proposal = draw(qf, dg)
        assert_feat(f1_proposal, p_vocab, eos)
        ld = forward(draft, draft_cache, [f1_proposal], ['position'], before + 2, 'draft_single')
        q2 = qp(ld[0])
        p2_proposal = draw(q2, dg)

        # EOS in third proposal: cannot feed EOS into F2, so verify Block3.
        p2_eos = p2_proposal == eos
        if not p2_eos:
            assert_pos(p2_proposal, p1_proposal, p_vocab, ranks)
        tgt_tokens = [pending_f, p1_proposal, f1_proposal]
        tgt_kinds = ['position', 'feature', 'position']
        if not p2_eos:
            tgt_tokens.append(p2_proposal)
            tgt_kinds.append('feature')
        out = forward(target, target_cache, tgt_tokens, tgt_kinds, before,
                      'forward_block3' if p2_eos else 'forward_block4')
        after = before + len(tgt_tokens)
        st['long_blocks'] += 1
        st['f1_proposed'] += 1
        st['p2_proposed'] += 1

        p1dist = tp(out[0]); st['p1_overlap_sum'] += float(torch.minimum(p1dist, q1).sum().item())
        if not acceptance(p1dist, q1, p1_proposal, tg):
            st['p1_rejected'] += 1
            corrected_p = residual(p1dist, q1, tg)
            rewind(target_cache, before + 1, after)
            rewind(draft_cache, before + 1, before + 3)
            if corrected_p == eos:
                st['eos'] += 1
                break
            assert_pos(corrected_p, prev_p, p_vocab, ranks)
            lf = forward(target, target_cache, [corrected_p], ['feature'], before + 1,
                         'forward_single')
            f_correct = draw(tp(lf[0]), tg)
            finish_pair(corrected_p, f_correct)
            prev_p, pending_f = corrected_p, f_correct
            if len(output) < max_tokens:
                forward(draft, draft_cache, [corrected_p], ['feature'], before + 1,
                        'draft_repair')
            continue

        st['p1_accepted'] += 1
        st['f1_checked'] += 1
        fd = tp(out[1]); st['f1_overlap_sum'] += float(torch.minimum(fd, qf).sum().item())
        if not acceptance(fd, qf, f1_proposal, tg):
            st['f1_rejected'] += 1
            f_correct = residual(fd, qf, tg)
            rewind(target_cache, before + 2, after)
            rewind(draft_cache, before + 2, before + 3)
            assert_feat(f_correct, p_vocab, eos)
            finish_pair(p1_proposal, f_correct)
            prev_p, pending_f = p1_proposal, f_correct
            continue

        st['f1_accepted'] += 1
        st['p2_checked'] += 1
        p2dist = tp(out[2]); st['p2_overlap_sum'] += float(torch.minimum(p2dist, q2).sum().item())
        accepted2 = acceptance(p2dist, q2, p2_proposal, tg)
        if accepted2:
            st['p2_accepted'] += 1
            chosen_p2 = p2_proposal
        else:
            st['p2_rejected'] += 1
            chosen_p2 = residual(p2dist, q2, tg)
        if chosen_p2 == eos:
            if p2_eos:
                # Prefix of Target block3 and Draft cache are already correct.
                finish_pair(p1_proposal, f1_proposal)
                st['eos'] += 1
                break
            # P2 row's logits are valid even if the proposal is not EOS.
            rewind(target_cache, before + 3, after)
            finish_pair(p1_proposal, f1_proposal)
            st['eos'] += 1
            break

        assert_pos(chosen_p2, p1_proposal, p_vocab, ranks)
        if accepted2 and not p2_eos:
            f2dist = tp(out[3])
        else:
            if not p2_eos:
                rewind(target_cache, before + 3, after)
            f2out = forward(target, target_cache, [chosen_p2], ['feature'], before + 3,
                            'forward_single')
            f2dist = tp(f2out[0])
        f2 = draw(f2dist, tg)
        finish_pair(p1_proposal, f1_proposal)
        # 2nd position must be strictly monotone relative to P1.
        assert_pos(chosen_p2, p1_proposal, p_vocab, ranks)
        assert_feat(f2, p_vocab, eos)
        output.extend((chosen_p2, f2))
        prev_p, pending_f = chosen_p2, f2
        if len(output) < max_tokens:
            forward(draft, draft_cache, [chosen_p2], ['feature'], before + 3,
                    'draft_sync_p2')

    if len(output) > max_tokens or len(output) % 2:
        raise AssertionError('Generated malformed P/F output')
    for x, y in zip(output[::2], output[2::2]):
        assert_pos(y, x, p_vocab, ranks)
    return output, st


@torch.inference_mode()
def b1_hybrid_decode(target, draft, make_target_cache, make_draft_cache, ranker,
                     *, eos, p_vocab, ranks, max_tokens, seed, device,
                     target_temp=0.9, target_top_p=0.9,
                     draft_temp=0.9, draft_top_p=0.9):
    """Hybrid ablation: SAME canonical D8 provides Feature q; Lean/Numba ranker provides P.

    P1 and P2 are point-mass Position proposals; F1 is stochastic D8 proposal.
    The D8 is *not* given Target hidden states. It keeps an independent causal KV.
    Its q_F is computed with one 2-token incremental call [pending_F, proposed_P1]
    and synchronized with exactly the Target-committed tokens after decisions.
    No B1-D8 Position proposal is sampled/used in this ablation.
    """
    import gpt_position_spec_decode_smoke as dec
    from bench_gaussiangpt_pf_block2_cow import fork_position_state

    if not (max_tokens >= 2 and max_tokens % 2 == 0):
        raise ValueError('max_tokens must be even >=2')
    if not (target_temp > 0 and draft_temp > 0 and
            0 < target_top_p <= 1 and 0 < draft_top_p <= 1):
        raise ValueError('Invalid sampling policy')

    tcache = make_target_cache()
    dcache = make_draft_cache()
    tg = torch.Generator(device=device).manual_seed(seed)
    dg = torch.Generator(device=device).manual_seed(seed + 1000003)
    st = Counter()
    out = []

    def fwd(model, cache, tokens, kinds, prefix, key):
        if len(tokens) != len(kinds) or int(cache.get_pos()) != prefix:
            raise AssertionError(f'{key}: invalid input/cache prefix at {cache.get_pos()}, wanted {prefix}')
        x = torch.as_tensor([tokens], device=device, dtype=torch.long)
        logits = model(x, kv_cache=cache, logit_types=kinds)[0]
        if logits.shape[0] != len(tokens) or int(cache.get_pos()) != prefix+len(tokens):
            raise AssertionError(f'{key}: cache advance/shape mismatch')
        st[key] += 1
        st[key+'_tokens'] += len(tokens)
        return logits

    def tp(logits): return policy(logits, target_temp, target_top_p)
    def qp(logits): return policy(logits, draft_temp, draft_top_p)
    def sample_t(logits): return draw(tp(logits), tg)

    # Bootstrap from frozen Target policy, then prefill D8 with the same prefix.
    p0 = sample_t(fwd(target, tcache, [eos], ['position'], 0, 'forward_single')[0])
    if p0 == eos:
        st['eos'] += 1
        return out, st
    if not 0 <= p0 < p_vocab:
        raise AssertionError('Illegal initial P')
    f0 = sample_t(fwd(target, tcache, [p0], ['feature'], 1, 'forward_single')[0])
    assert_feat(f0, p_vocab, eos)
    out.extend((p0, f0))
    if len(out) == max_tokens:
        return out, st
    fwd(draft, dcache, [eos,p0], ['position','feature'], 0, 'draft_prefill')
    online = dec.CausalRanker(rank_map=ranks, p_vocab=p_vocab)
    prev_p, pending_f = p0, f0

    while len(out) < max_tokens:
        before = int(tcache.get_pos())
        if before != int(dcache.get_pos()):
            raise AssertionError(f'Hybrid cache prefix mismatch: T={before} D={dcache.get_pos()}')
        first = dec.choose_position(ranker, online, prev_p)
        shadow = None
        second = None
        can_block4 = first is not None and len(out)+4 <= max_tokens
        if can_block4:
            p1 = int(first[0]); assert_pos(p1, prev_p, p_vocab, ranks)
            shadow = fork_position_state(online)
            shadow.finalize(p1)
            second = dec.choose_position(ranker, shadow, p1)
            can_block4 = second is not None

        if not can_block4:
            # K1 Position-only, with independent D8 cache synchronized on commit.
            st['k1_steps'] += 1
            if first is None:
                plogits = fwd(target, tcache, [pending_f], ['position'], before,
                              'forward_single')[0]
                chosen = sample_t(plogits)
                st['position_fallback'] += 1
                if chosen == eos:
                    online.drop_eos(); st['eos'] += 1
                    break
                assert_pos(chosen, prev_p, p_vocab, ranks)
                fdist = tp(fwd(target, tcache, [chosen], ['feature'], before+1,
                               'forward_single')[0])
            else:
                p1 = int(first[0]); assert_pos(p1, prev_p, p_vocab, ranks)
                v = fwd(target, tcache, [pending_f,p1], ['position','feature'], before,
                        'forward_pair')
                p = tp(v[0])
                st['p1_checked'] += 1
                st['p1_overlap_sum'] += float(p[p1].item())
                if float(torch.rand((), device=device, generator=tg).item()) < float(p[p1].item()):
                    chosen = p1
                    st['p1_accepted'] += 1
                    fdist = tp(v[1])
                else:
                    st['p1_rejected'] += 1
                    corrected = p.clone(); corrected[p1] = 0
                    corrected /= corrected.sum()
                    chosen = draw(corrected, tg)
                    rewind(tcache, before+1, before+2)
                    if chosen == eos:
                        online.drop_eos(); st['eos'] += 1
                        break
                    assert_pos(chosen, prev_p, p_vocab, ranks)
                    fdist = tp(fwd(target, tcache, [chosen], ['feature'], before+1,
                                   'forward_single')[0])
            f = draw(fdist, tg); assert_feat(f, p_vocab, eos)
            out.extend((chosen,f))
            online.finalize(chosen)
            if len(out) < max_tokens:
                # This is mandatory: D8 next Feature must observe the actual P.
                fwd(draft, dcache, [pending_f,chosen], ['position','feature'], before,
                    'draft_sync_k1')
            prev_p,pending_f = chosen,f
            continue

        p1,p2 = int(first[0]),int(second[0])
        assert_pos(p2,p1,p_vocab,ranks)
        st['long_blocks'] += 1
        st['p1_proposed'] += 1
        st['p2_proposed'] += 1

        # The same D8 q_F used for sampling is used for alpha and residual.
        # Row0 (Position logits) is intentionally unused.
        dl = fwd(draft, dcache, [pending_f,p1], ['position','feature'], before,
                 'draft_feature_pair')
        qf = qp(dl[1])
        f1 = draw(qf, dg)
        assert_feat(f1, p_vocab, eos)
        st['f1_proposed'] += 1

        v = fwd(target, tcache, [pending_f,p1,f1,p2],
                ['position','feature','position','feature'],before,'forward_block4')
        p1d = tp(v[0]); alpha1 = float(p1d[p1].item())
        st['p1_checked'] += 1; st['p1_overlap_sum'] += alpha1
        if float(torch.rand((),device=device,generator=tg).item()) >= alpha1:
            st['p1_rejected'] += 1
            r = p1d.clone(); r[p1] = 0; r /= r.sum()
            fixedp = draw(r,tg)
            rewind(tcache,before+1,before+4)
            rewind(dcache,before+1,before+2)
            if fixedp == eos:
                online.drop_eos(); st['eos'] += 1
                break
            assert_pos(fixedp,prev_p,p_vocab,ranks)
            f_corr = sample_t(fwd(target,tcache,[fixedp],['feature'],before+1,
                                  'forward_single')[0])
            assert_feat(f_corr,p_vocab,eos)
            out.extend((fixedp,f_corr))
            online.finalize(fixedp)
            if len(out) < max_tokens:
                fwd(draft,dcache,[fixedp],['feature'],before+1,'draft_repair')
            prev_p,pending_f = fixedp,f_corr
            continue

        st['p1_accepted'] += 1
        online = shadow  # P1 committed and P2 is pending in COW state.
        pd = tp(v[1])
        st['f1_checked'] += 1
        st['f1_overlap_sum'] += float(torch.minimum(pd,qf).sum().item())
        if not acceptance(pd,qf,f1,tg):
            st['f1_rejected'] += 1
            fixedf = residual(pd,qf,tg)
            assert_feat(fixedf,p_vocab,eos)
            rewind(tcache,before+2,before+4)
            # D8 already processed exactly [pending_f,p1], no rollback.
            out.extend((p1,fixedf))
            online.drop_eos()
            prev_p,pending_f = p1,fixedf
            continue

        st['f1_accepted'] += 1
        p2d = tp(v[2]); alpha2 = float(p2d[p2].item())
        st['p2_checked'] += 1; st['p2_overlap_sum'] += alpha2
        if float(torch.rand((),device=device,generator=tg).item()) < alpha2:
            st['p2_accepted'] += 1
            chosen2 = p2
            fd2 = tp(v[3])
        else:
            st['p2_rejected'] += 1
            r = p2d.clone(); r[p2] = 0; r /= r.sum()
            chosen2 = draw(r,tg)
            rewind(tcache,before+3,before+4)
            if chosen2 == eos:
                out.extend((p1,f1))
                online.drop_eos(); st['eos'] += 1
                break
            assert_pos(chosen2,p1,p_vocab,ranks)
            fd2 = tp(fwd(target,tcache,[chosen2],['feature'],before+3,
                           'forward_single')[0])
        f2 = draw(fd2,tg); assert_feat(f2,p_vocab,eos)
        out.extend((p1,f1,chosen2,f2))
        online.finalize(chosen2)
        if len(out) < max_tokens:
            # Process accepted Feature and the actual P2, not the proposed P2.
            fwd(draft,dcache,[f1,chosen2],['position','feature'],before+2,
                'draft_sync_p2')
        prev_p,pending_f = chosen2,f2

    if len(out)%2 or len(out)>max_tokens:
        raise AssertionError('Malformed P/F sequence')
    for x,y in zip(out[::2],out[2::2]):
        assert_pos(y,x,p_vocab,ranks)
    return out,st


def check_model_compatibility(t, d):
    p_vocab = int(t.position_vocab_size)
    eos = int(t.eos_token_id)
    if (p_vocab, eos) != (int(d.position_vocab_size), int(d.eos_token_id)):
        raise AssertionError('Target/D8 vocab and EOS mismatch')
    if bool(t.shared_vocab) or bool(d.shared_vocab):
        raise AssertionError('Require disjoint sparse Position/Feature vocab')
    if not torch.equal(t.transformer.sparse_position_rank_map,
                       d.transformer.sparse_position_rank_map):
        raise AssertionError('Target/D8 rank map mismatch')
    return p_vocab, eos, t.transformer.sparse_position_rank_map.detach().cpu().long()



def unified_acceptance_counts(method, stats):
    """Normalize printed acceptance counters without altering frozen raw stats."""
    if method == 'pf_cow':
        return (int(stats.get('position_1_accepted',0)),int(stats.get('position_1_proposals',0)),
                int(stats.get('feature_accepted',0)),int(stats.get('feature_checked',0)),
                int(stats.get('position_2_accepted',0)),int(stats.get('position_2_checked',0)))
    if method == 'position_only':
        return (int(stats.get('accepted',0)),int(stats.get('proposal',0)),0,0,0,0)
    return (int(stats.get('p1_accepted',0)),int(stats.get('p1_checked',0)),
            int(stats.get('f1_accepted',0)),int(stats.get('f1_checked',0)),
            int(stats.get('p2_accepted',0)),int(stats.get('p2_checked',0)))


def aggregate(rows, method):
    arr = [r for r in rows if r['method'] == method]
    tokens = sum(x['tokens'] for x in arr)
    seconds = sum(x['ar_seconds'] for x in arr)
    c = Counter()
    for r in arr: c.update(r.get('normalized_acceptance',{}))
    return {
        'normalized_acceptance':dict(c),
        'runs':len(arr), 'tokens': tokens, 'ar_seconds':seconds,
        'tokens_per_second':tokens/seconds if seconds>0 else 0,
        'eos':sum(int(x['ended_by_eos']) for x in arr),
        'cap_hit':sum(int(x['cap_hit']) for x in arr),
        'target_forwards':sum(sum(x['stats'].get(k,0) for k in
           ('forward_single','forward_pair','forward_block3','forward_block4')) for x in arr),
        'draft_forwards':sum(sum(x['stats'].get(k,0) for k in
           ('draft_single','draft_prefill','draft_sync_p2','draft_repair','draft_feature_pair','draft_sync_k1')) for x in arr),
        'target_input_rows':sum(sum(int(x['stats'].get(k,0))*sz for k,sz in
            (('forward_single',1),('forward_pair',2),('forward_block3',3),('forward_block4',4))) for x in arr),
        'hybrid_d8_input_rows':sum(sum(x['stats'].get(k+'_tokens',0) for k in
           ('draft_prefill','draft_feature_pair','draft_sync_k1','draft_sync_p2','draft_repair')) for x in arr),
    }


def benchmark(args):
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for real B1 benchmark')
    if args.max_tokens < 4 or args.max_tokens % 2 or args.max_tokens > 16000:
        raise ValueError('max-tokens must be even 4..16000')
    if args.seeds < 1:
        raise ValueError('seeds must be >=1')
    if args.output_dir.exists() and (args.output_dir/'summary.json').exists():
        raise FileExistsError('Refusing to overwrite completed benchmark')
    torch.set_float32_matmul_precision('highest')
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE
    from profiling.s3.s3_onpolicy_feature_distill_train import make_kv_cache
    import gpt_position_spec_decode_smoke as dec
    import bench_gaussiangpt_pf_block2_cow as cow
    from bench_fast_hgb import FastHistRanker
    from joblib import load
    from threadpoolctl import threadpool_limits

    dev = torch.device('cuda:0')
    vq = GaussianVQVAE.load_from_checkpoint(str(args.vqvae), map_location='cpu').eval()
    teacher = GaussianGPT.load_from_checkpoint(str(args.gpt), map_location='cpu', vqvae=vq, strict=True).eval()
    d8 = GaussianGPT.load_from_checkpoint(str(args.d8), map_location='cpu', vqvae=vq, strict=True).eval()
    teacher.gpt.to(dev).eval().requires_grad_(False)
    d8.gpt.to(dev).eval().requires_grad_(False)
    model_t = getattr(teacher.gpt, '_orig_mod', teacher.gpt)
    model_d = getattr(d8.gpt, '_orig_mod', d8.gpt)
    p_vocab,eos,ranks = check_model_compatibility(model_t,model_d)
    target = model_t.transformer
    draft = model_d.transformer
    print('B1: Native D8+Target strict loads passed; P/F rank maps identical',flush=True)
    def make_t(): return make_kv_cache(target, 1+args.max_tokens+16,dev)
    def make_d(): return make_kv_cache(draft, 1+args.max_tokens+16,dev)
    kwargs = dict(eos=eos,p_vocab=p_vocab,ranks=ranks,device=dev,
                  max_tokens=args.max_tokens,temp=0.9,top_p=0.9)
    ranker = feature = None
    if any(m in args.methods for m in ('position_only','pf_cow','b1_hybrid')):
        ranker = FastHistRanker(load(args.ranker))
    if 'pf_cow' in args.methods:
        feature, dt, dp = cow.load_feature_model(args.feature_checkpoint,args.feature_head,
                                                  target,dev,p_vocab,eos)
        print(f'PF checkpoint student policy=({dt:g},{dp:g}); runtime=(0.9,0.8)',flush=True)
    def one(method,seed,n):
        common = dict(kwargs,max_tokens=n)
        if method=='vanilla':
            return dec.vanilla(target,make_t,seed=seed,**common)
        if method=='position_only':
            return dec.speculative(target,make_t,ranker,seed=seed,**common)
        if method=='pf_cow':
            return cow.block2_decode(target,make_t,ranker,feature,seed=seed,
                                     draft_temp=0.9,draft_top_p=0.8,**common)
        if method=='b1_hybrid':
            return b1_hybrid_decode(target,draft,make_t,make_d,ranker,
                                    eos=eos,p_vocab=p_vocab,ranks=ranks,
                                    max_tokens=n,seed=seed,device=dev,
                                    target_temp=0.9,target_top_p=0.9,
                                    draft_temp=args.draft_temp,
                                    draft_top_p=args.draft_top_p)
        if method=='b1_d8':
            return b1_decode(target,draft,make_t,make_d,eos=eos,p_vocab=p_vocab,
                             ranks=ranks,max_tokens=n,seed=seed,device=dev,
                             target_temp=0.9,target_top_p=0.9,
                             draft_temp=args.draft_temp,draft_top_p=args.draft_top_p)
        raise ValueError(method)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows,traces=[],{}
    with torch.inference_mode(), threadpool_limits(limits=1,user_api='openmp'):
        for j,m in enumerate(args.methods):
            one(m,args.seed_start+999+j,min(32,args.max_tokens))
        torch.cuda.synchronize()
        for i in range(args.seeds):
            seed=args.seed_start+i
            order=args.methods if i%2==0 else list(reversed(args.methods))
            for m in order:
                torch.cuda.synchronize()
                start=time.perf_counter()
                seq,st=one(m,seed,args.max_tokens)
                torch.cuda.synchronize()
                seconds=time.perf_counter()-start
                if len(seq)%2 or len(seq)>args.max_tokens:
                    raise AssertionError('Malformed output length')
                for x,y in zip(seq[::2],seq[1::2]):
                    assert 0<=x<p_vocab
                    assert_feat(y,p_vocab,eos)
                for a,b in zip(seq[::2],seq[2::2]):
                    assert_pos(b,a,p_vocab,ranks)
                eof=bool(st['eos'])
                cap=not eof and len(seq)==args.max_tokens
                if not (eof or cap): raise AssertionError('Neither EOS nor cap')
                row=dict(seed=seed,method=m,tokens=len(seq),pairs=len(seq)//2,
                         ar_seconds=seconds,tokens_per_second=len(seq)/seconds,
                         ended_by_eos=eof,cap_hit=cap,
                         stats={k:float(v) for k,v in st.items()},
                         normalized_acceptance=dict(zip(('p1_accepted','p1_checked',
                              'f1_accepted','f1_checked','p2_accepted','p2_checked'),
                              unified_acceptance_counts(m,st))))
                rows.append(row)
                if args.save_token_traces:
                    traces[f'{seed}_{m}']=torch.tensor(seq,dtype=torch.int16)
                a1,n1,af,nf,a2,n2=unified_acceptance_counts(m,st)
                print(f'seed={seed} {m:14s} tok={len(seq):5d} AR={seconds:8.3f}s '
                      f'tok/s={len(seq)/seconds:8.2f} EOS={int(eof)} '
                      f'TBlock4={st["forward_block4"]} TBlock3={st["forward_block3"]} '
                      f'TBlock2={st["forward_pair"]} TBlock1={st["forward_single"]} '
                      f'P1={a1}/{n1} '
                      f'F1={af}/{nf} '
                      f'P2={a2}/{n2} '
                      f'D8calls={sum(int(st.get(k,0)) for k in ("draft_prefill","draft_single","draft_feature_pair","draft_repair","draft_sync_p2","draft_sync_k1"))}',flush=True)
                (args.output_dir/'partial_results.json').write_text(json.dumps({'runs':rows},indent=2)+'\n')
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    config['target_sha256']=hashlib.sha256(args.gpt.read_bytes()).hexdigest()
    config['d8_sha256']=hashlib.sha256(args.d8.read_bytes()).hexdigest()
    if ranker is not None:
        config['ranker_sha256']=hashlib.sha256(args.ranker.read_bytes()).hexdigest()
    summary=dict(label='B1 D8 vs Ranker+D8 Hybrid vs Frozen PF, exact verification; same-session controls',
                 target_policy=dict(temperature=0.9,top_p=0.9),
                 d8_proposal_policy=dict(temperature=args.draft_temp,top_p=args.draft_top_p),
                 config=config,runs=rows,aggregate={m:aggregate(rows,m) for m in args.methods},
                 notes=['B1 uses native D8 for all P/F/P proposals. Hybrid uses the SAME D8 only for F1 and frozen Ranker for P1/P2; PF+COW uses frozen feature-state student.',
                        'Rejection uses exact (p-q)+ residual; EOS and cap handled at complete P/F pair boundaries.',
                        'Frozen PF is measured in same process for fair comparison.',
                        'No equality of outputs by same RNG seed is expected.'])
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.save_token_traces:
        torch.save(dict(traces=traces,runs=rows,eos=eos),args.output_dir/'token_traces.pt')
    for m in args.methods:
        print(f'AGGREGATE {m:14s}: {summary["aggregate"][m]["tokens_per_second"]:.3f} tok/s',flush=True)
    print('SAVED',args.output_dir/'summary.json',flush=True)


def self_test():
    """Causal fake target/draft, forced path coverage, exact residual identity."""
    from unittest.mock import patch
    import torch.nn as nn
    P=16; E=20
    class Cache:
        def __init__(self):
            self.cache_seqlens=torch.zeros(1,dtype=torch.int32)
            self.token_cache=torch.full((1,128),-1,dtype=torch.long)
        def get_pos(self): return int(self.cache_seqlens[0].item())
    class Fake(nn.Module):
        def __init__(self, is_draft, *, draft_eos=None, bos_eos=False):
            super().__init__()
            self.is_draft=is_draft
            self.draft_eos=draft_eos
            self.bos_eos=bos_eos
            self.last_cache=None
        def forward(self, idx, *, kv_cache, logit_types):
            start=kv_cache.get_pos(); n=idx.size(1)
            assert len(logit_types)==n
            before=kv_cache.token_cache[0,:start].tolist()
            incoming=idx[0].tolist()
            logits=torch.full((1,n,E+1),float('-inf'))
            for row,(t,kind) in enumerate(zip(incoming,logit_types)):
                hist=before+incoming[:row+1]
                if kind=='position':
                    positions=[x for x in hist if 0<=x<P]
                    prev=positions[-1] if positions else None
                    if prev is None:
                        logits[0,row,E if self.bos_eos else 0]=0.
                    elif prev>=P-1:
                        logits[0,row,E]=0.
                    elif self.is_draft and (self.draft_eos=='p1' or
                                           (self.draft_eos=='p2' and len(positions)>=2)):
                        logits[0,row,E]=0.
                    else:
                        nextp=prev+1
                        logits[0,row,nextp]=1.2 if not self.is_draft else 0.2
                        if nextp+1<P:
                            logits[0,row,nextp+1]=.2 if not self.is_draft else 1.2
                            logits[0,row,E]=-.1
                        else:
                            logits[0,row,E]=.4
                elif kind=='feature':
                    logits[0,row,P:E]=torch.tensor([1.2,.3,.0,-.5]) if not self.is_draft else torch.tensor([.0,1.2,.5,-.5])
                else: raise AssertionError(kind)
            kv_cache.token_cache[:,start:start+n]=idx
            kv_cache.cache_seqlens.add_(n)
            self.last_cache=kv_cache
            return logits
    # Ensure residual mixture exactly recovers target distribution on toy vectors.
    p=torch.tensor([.6,.3,.1],dtype=torch.float64)
    q=torch.tensor([.2,.5,.3],dtype=torch.float64)
    mass=torch.minimum(p,q)
    rem=(p-q).clamp_min(0)
    torch.testing.assert_close(mass+rem,p,atol=1e-12,rtol=0)
    assert float(rem.sum())>0
    scenarios={'p1_reject':[False], 'f1_reject':[True,False],
               'p2_reject':[True,True,False], 'all_accept':[True,True,True]}
    for label,decisions in scenarios.items():
        target=Fake(False);draft=Fake(True)
        it=iter(decisions)
        def choose(_p,_q,_t,_r): return next(it,True)
        with patch(__name__+'.acceptance',side_effect=choose):
            seq,stats=b1_decode(target,draft,Cache,Cache,eos=E,p_vocab=P,
                  ranks=list(range(P)),max_tokens=6,seed=59,device=torch.device('cpu'),
                  target_temp=1.,target_top_p=1.,draft_temp=1.,draft_top_p=1.)
        assert len(seq)%2==0 and len(seq)<=6,(label,seq,stats)
        assert stats['forward_block4'] >= 1,(label,seq,stats)
        assert all(b>a for a,b in zip(seq[::2],seq[2::2]))
        if label=='p1_reject': assert stats['p1_rejected']>=1
        if label=='f1_reject': assert stats['f1_rejected']>=1
        if label=='p2_reject': assert stats['p2_rejected']>=1
        if label=='all_accept': assert stats['p2_accepted']>=1
        if len(seq)==6:
            # Target and draft share all committed processed tokens (last F pending).
            valid=[E]+seq[:-1]
            for mod in (target,draft):
                cache=mod.last_cache
                # Draft may omit the final accepted P at the cap (no more proposals).
                # Its processed prefix must never contain an incorrect proposal.
                observed=cache.token_cache[0,:cache.get_pos()].tolist()
                assert observed==valid[:cache.get_pos()],(label,mod.is_draft,cache.get_pos(),valid,observed)
                if not mod.is_draft:
                    assert cache.get_pos()==len(valid),(label,cache.get_pos(),len(valid))
        print('PASS:',label,'tokens=',seq,'stats=',dict(stats),flush=True)
    # All EOS branches, including P1 directly, P2 bonus EOS and rejection to EOS.
    eos_cases=[
        ('bos_eos', dict(bos_eos=True), [], None, 0),
        ('p1_proposed_eos',dict(draft_eos='p1'),[True],None, 2),
        ('p1_proposed_eos_rejected',dict(draft_eos='p1'),[False],None, None),
        ('p2_proposed_eos',dict(draft_eos='p2'),[True,True,True],None, 4),
        ('p2_proposed_eos_rejected',dict(draft_eos='p2'),[True,True,False],None, None),
        ('p1_rejected_to_eos',dict(),[False], 'eos',2),
        ('p2_rejected_to_eos',dict(),[True,True,False],'eos',4),
    ]
    for label,opts,decisions,repair,expected in eos_cases:
        target=Fake(False,bos_eos=opts.get('bos_eos',False))
        draft=Fake(True,draft_eos=opts.get('draft_eos'))
        it=iter(decisions)
        def choose(_p,_q,_t,_r): return next(it,True)
        def sample_residual(p,q,g):
            if repair=='eos': return E
            return residual_original(p,q,g)
        residual_original=residual
        with patch(__name__+'.acceptance',side_effect=choose), \
             patch(__name__+'.residual',side_effect=sample_residual):
            seq,stats=b1_decode(target,draft,Cache,Cache,eos=E,p_vocab=P,
                      ranks=list(range(P)),max_tokens=6,seed=59,device=torch.device('cpu'),
                      target_temp=1.,target_top_p=1.,draft_temp=1.,draft_top_p=1.)
        if expected is not None:
            assert len(seq)==expected,(label,seq,stats)
        else:
            assert len(seq)>=4 and stats['p1_rejected']+stats['p2_rejected']>=1,(label,seq,stats)
        assert all(b>a for a,b in zip(seq[::2],seq[2::2])),(label,seq)
        if label not in ('p1_proposed_eos_rejected','p2_proposed_eos_rejected'):
            assert stats['eos']==1,(label,stats)
        print('PASS:',label,'tokens=',seq,'EOS=',stats['eos'],flush=True)
    target=Fake(False);draft=Fake(True)
    seq,st=b1_decode(target,draft,Cache,Cache,eos=E,p_vocab=P,ranks=list(range(P)),
                    max_tokens=4,seed=59,device=torch.device('cpu'),
                    target_temp=1.,target_top_p=1.,draft_temp=1.,draft_top_p=1.)
    assert st['k1_steps']==1 and st['forward_block4']==0,(seq,st)
    print('PASS: cap_last_pair_K1',seq,flush=True)
    print('PASS: stochastic proposal/residual law, 4 full Block4 branches, EOS and K1',flush=True)
    print('SCOPE: CPU fake causal transformer only; real CUDA Block4/KV needs Smoke',flush=True)


def hybrid_self_test():
    """12 synthetic causal checks, separate Target/D8 cache and conditional Feature q."""
    import copy
    from unittest.mock import patch
    import torch.nn as nn
    import gpt_position_spec_decode_smoke as dec

    P=16; E=20
    class Cache:
        def __init__(self):
            self.cache_seqlens = torch.zeros(1,dtype=torch.int32)
            self.token_cache = torch.full((1,128),-1,dtype=torch.long)
        def get_pos(self): return int(self.cache_seqlens[0].item())

    class Fake(nn.Module):
        def __init__(self,is_draft=False,bos_eos=False):
            super().__init__();self.is_draft=is_draft;self.bos_eos=bos_eos
            self.last_cache=None
        def forward(self, idx, *, kv_cache, logit_types):
            start=kv_cache.get_pos();incoming=idx[0].tolist();kinds=logit_types
            assert len(incoming)==len(kinds)
            old=kv_cache.token_cache[0,:start].tolist()
            logits=torch.full((1,len(incoming),E+1),float('-inf'))
            for i,(tok,kind) in enumerate(zip(incoming,kinds)):
                history=old+incoming[:i+1]
                positions=[x for x in history if 0<=x<P]
                if kind=='position':
                    if not positions:
                        logits[0,i,E if self.bos_eos else 0]=0
                    else:
                        prev=positions[-1]
                        if prev>=P-1:
                            logits[0,i,E]=0
                        else:
                            logits[0,i,prev+1]=1.0
                            if prev+2<P: logits[0,i,prev+2]=0.0
                            logits[0,i,E]=-2.5
                elif kind=='feature':
                    logits[0,i,P:E]=(torch.tensor([1.2,.3,0.,-.5])
                                        if not self.is_draft else torch.tensor([0.,1.2,.3,-.5]))
                else: raise AssertionError(kind)
            kv_cache.token_cache[:,start:start+len(incoming)]=idx
            kv_cache.cache_seqlens.add_(len(incoming))
            self.last_cache=kv_cache
            return logits

    class FakeRanker:
        def __init__(self,rank_map,p_vocab):
            self.pending=None
            self.saved=[]
        def finalize(self,p):
            if self.pending is None: raise AssertionError('Ranker finalize without proposal')
            self.saved.append(p);self.pending=None
        def drop_eos(self): self.pending=None

    def choose(_ranker,state,prev):
        if state.pending is not None: raise AssertionError('Ranker has uncommitted pending')
        state.pending=prev
        if prev+1>=P: return None
        return prev+1,1,0

    scenarios={
        'p1_reject': ([.999],[]),
        'f1_reject': ([0.],[False]),
        'p2_reject': ([0.,.999],[True]),
        'all_accept': ([0.,0.],[True]),
    }
    for name,(draws,features) in scenarios.items():
        t=Fake();d=Fake(is_draft=True)
        it_rand=iter(draws);it_feat=iter(features)
        def rand(*_args,**_kwargs): return torch.tensor(next(it_rand,0.))
        def feat(*_args,**_kwargs): return next(it_feat,True)
        with patch('gpt_position_spec_decode_smoke.CausalRanker',FakeRanker),\
             patch('gpt_position_spec_decode_smoke.choose_position',side_effect=choose),\
             patch('bench_gaussiangpt_pf_block2_cow.fork_position_state',side_effect=copy.deepcopy),\
             patch('torch.rand',side_effect=rand),\
             patch(__name__+'.acceptance',side_effect=feat):
            output,st=b1_hybrid_decode(t,d,Cache,Cache,object(),eos=E,
                 p_vocab=P,ranks=list(range(P)),max_tokens=6,seed=29,
                 device=torch.device('cpu'),target_temp=1.,target_top_p=1.,
                 draft_temp=1.,draft_top_p=1.)
        assert len(output)%2==0 and len(output)<=6,(name,output,st)
        assert st['forward_block4']>=1,(name,dict(st))
        assert st['draft_feature_pair']>=1,(name,dict(st))
        assert all(y>x for x,y in zip(output[::2],output[2::2])),(name,output)
        expected_key = {'p1_reject':'p1_rejected','f1_reject':'f1_rejected','p2_reject':'p2_rejected','all_accept':'p2_accepted'}[name]
        assert st[expected_key]>=1,(name,st)
        # Each cache contains only actually confirmed tokens (possibly shorter at cap).
        correct=[E]+output[:-1]
        for mod in (t,d):
            cache=mod.last_cache
            observed=cache.token_cache[0,:cache.get_pos()].tolist()
            assert observed==correct[:cache.get_pos()],(name,mod.is_draft,observed,correct)
        print('PASS hybrid:',name,'tokens=',output,'stats=',dict(st),flush=True)

    # No candidate -> Target-only fallback, with D8 sync and EOS handling.
    def no_guess(_ranker,state,prev):
        if state.pending is not None: raise AssertionError('Uncommitted fake proposal')
        state.pending=prev
        return None
    with patch('gpt_position_spec_decode_smoke.CausalRanker',FakeRanker),\
         patch('gpt_position_spec_decode_smoke.choose_position',side_effect=no_guess):
        t=Fake();d=Fake(True)
        out,st=b1_hybrid_decode(t,d,Cache,Cache,object(),eos=E,p_vocab=P,
            ranks=list(range(P)),max_tokens=6,seed=11,device=torch.device('cpu'),
            target_temp=1.,target_top_p=1.,draft_temp=1.,draft_top_p=1.)
    assert st['position_fallback']>0 and st['forward_block4']==0 and st['draft_sync_k1']>=1,(out,st)
    print('PASS hybrid: no_legal_candidate_fallback')

    with patch('gpt_position_spec_decode_smoke.CausalRanker',FakeRanker),\
         patch('gpt_position_spec_decode_smoke.choose_position',side_effect=choose):
        t=Fake(bos_eos=True);d=Fake(True)
        out,st=b1_hybrid_decode(t,d,Cache,Cache,object(),eos=E,p_vocab=P,
            ranks=list(range(P)),max_tokens=6,seed=11,device=torch.device('cpu'),
            target_temp=1.,target_top_p=1.,draft_temp=1.,draft_top_p=1.)
    assert not out and st['eos']==1
    print('PASS hybrid: BOS_EOS')

    with patch('gpt_position_spec_decode_smoke.CausalRanker',FakeRanker),\
         patch('gpt_position_spec_decode_smoke.choose_position',side_effect=choose):
        t=Fake();d=Fake(True)
        out,st=b1_hybrid_decode(t,d,Cache,Cache,object(),eos=E,p_vocab=P,
            ranks=list(range(P)),max_tokens=4,seed=11,device=torch.device('cpu'),
            target_temp=1.,target_top_p=1.,draft_temp=1.,draft_top_p=1.)
    assert st['k1_steps']==1 and st['forward_block4']==0 and len(out)==4,(out,st)
    print('PASS hybrid: cap_last_pair_K1')

    # Rejection repaired into EOS: no dangling Position/Feature and no invalid
    # proposal survives in the processed KV prefix.
    for which,draws,expected_tokens in [
        ('p1_repair_eos',[.999],2),
        ('p2_repair_eos',[0.,.999],4),
    ]:
        t=Fake();d=Fake(True)
        rand_iter=iter(draws)
        def rr(*args,**kwargs): return torch.tensor(next(rand_iter,0.))
        original_draw=draw
        draw_calls=[0]
        def draw_eos_on_repair(prob,gen):
            draw_calls[0]+=1
            if draw_calls[0]==4:
                assert float(prob[E])>0,(which,prob[E])
                return E
            return original_draw(prob,gen)
        with patch('gpt_position_spec_decode_smoke.CausalRanker',FakeRanker),\
             patch('gpt_position_spec_decode_smoke.choose_position',side_effect=choose),\
             patch('bench_gaussiangpt_pf_block2_cow.fork_position_state',side_effect=copy.deepcopy),\
             patch('torch.rand',side_effect=rr),\
             patch(__name__+'.acceptance',return_value=True),\
             patch(__name__+'.draw',side_effect=draw_eos_on_repair):
            out,st=b1_hybrid_decode(t,d,Cache,Cache,object(),eos=E,p_vocab=P,
                ranks=list(range(P)),max_tokens=6,seed=29,device=torch.device('cpu'),
                target_temp=1.,target_top_p=1.,draft_temp=1.,draft_top_p=1.)
        assert len(out)==expected_tokens and st['eos']==1,(which,out,st)
        assert st['p1_rejected' if which.startswith('p1') else 'p2_rejected']>=1
        print('PASS hybrid:',which,'tokens=',out)
    print('PASS hybrid: exact qF + Position delta, COW ranker state, dual KV and fallback branches',flush=True)
    print('SCOPE: CPU fake causal Target/D8 only; real CUDA Smoke still required',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--gpt',type=Path,default=Path('checkpoints/gpt_vfront.ckpt'))
    ap.add_argument('--vqvae',type=Path,default=Path('checkpoints/vqvae_vfront.ckpt'))
    ap.add_argument('--d8',type=Path,default=Path(D8_DEFAULT))
    ap.add_argument('--ranker',type=Path,default=Path('profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib'))
    ap.add_argument('--feature-checkpoint',type=Path,default=Path('profiling/feature_draft/state_distill_mixed_overlap/feature_state_draft_best.pt'))
    ap.add_argument('--feature-head',type=Path,default=Path('profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt'))
    ap.add_argument('--methods',nargs='+',choices=['vanilla','position_only','pf_cow','b1_d8','b1_hybrid'],
                    default=['vanilla','b1_d8','b1_hybrid','position_only','pf_cow'])
    ap.add_argument('--seed-start',type=int,default=88201)
    ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--max-tokens',type=int,default=128)
    ap.add_argument('--draft-temp',type=float,default=0.9)
    ap.add_argument('--draft-top-p',type=float,default=0.9)
    ap.add_argument('--output-dir',type=Path,default=Path('profiling/baselines_5way/b1_hybrid_smoke'))
    ap.add_argument('--save-token-traces',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        hybrid_self_test();return
    benchmark(args)


if __name__=='__main__':
    main()
