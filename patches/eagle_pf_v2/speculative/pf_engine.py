"""One causal Target verifier for joint and Ranker/Feature P/F proposals."""
import time

import torch

from model.gpt.engine import KVCache
from speculative.pf_draft import TokenSpec


def policy(logits, temperature=0.9, top_p=0.9):
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling parameters")
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    scores = logits.float() / temperature
    if top_p < 1:
        ordered, indices = scores.sort(descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scores = scores.scatter(-1, indices, ordered.masked_fill(remove, -torch.inf))
    return scores.softmax(-1)


def draw(p, rng):
    return int(torch.multinomial(p, 1, generator=rng).item())


def accept(p, q, token, rng, greedy=False):
    if greedy:
        actual = int(p.argmax())
        return actual == token, actual
    if not q[token] > 0:
        raise ValueError("proposed token has zero proposal probability")
    alpha = (p[token] / q[token]).clamp(max=1)
    if torch.rand((), device=p.device, generator=rng) < alpha:
        return True, token
    residual = (p-q).clamp_min(0)
    mass = residual.sum()
    if not mass > 0:
        raise ArithmeticError("rejection has no positive residual probability")
    return False, draw(residual/mass, rng)


def validate_tokens(tokens, spec, ranks):
    if not tokens or tokens[0] != spec.eos:
        raise ValueError("prefix must begin with the model's BOS sentinel")
    last_rank = -1
    for i, token in enumerate(tokens[1:], 1):
        if token == spec.eos:
            if i % 2 != 1 or i != len(tokens)-1:
                raise ValueError("EOS must terminate a Position slot")
        elif i % 2:
            if not 0 <= token < spec.positions or int(ranks[token]) <= last_rank:
                raise ValueError("Position must be legal and strictly increasing")
            last_rank = int(ranks[token])
        elif not spec.offset <= token < spec.offset+spec.features:
            raise ValueError("invalid Feature token")


class PositionRanker:
    """The existing deterministic Ranker defines q_P as a point mass."""
    def __init__(self, model, ranks):
        self.model, self.ranks = model, ranks
        self.state = None
        self.previous = None

    def initialize(self, prefix):
        from draft_a.pf_ranker_eval import position_state
        self.previous = prefix[-2] if len(prefix) > 1 else None
        self.state = position_state(prefix, self.ranks) if self.previous is not None else None

    def propose(self, state, previous, eos):
        from gpt_position_spec_decode_smoke import choose_position
        if state is None:
            return eos, None, None
        out = choose_position(self.model, state, previous)
        token = eos if out is None else int(out[0])
        return token, state, self.committed(state, token, eos)

    @staticmethod
    def committed(pending, token, eos):
        if pending is None:
            return None
        if token == eos:
            return pending
        from bench_gaussiangpt_pf_block2_cow import fork_position_state
        result = fork_position_state(pending)
        result.finalize(token)
        return result

    def after_target_position(self, prefix):
        """A Target recovery/bonus may introduce a Position absent from the trial."""
        self.initialize(prefix)


def target_cache(core, length):
    cfg = core.config
    weight = next(core.parameters())
    return KVCache(1, cfg.n_kv_head, length, cfg.n_embd//cfg.n_head, cfg.n_layer,
                   weight.device, weight.dtype)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode(core, draft, prompt, max_new_tokens, *, mode="feature", gamma=3,
           ranker=None, seed=42, temperature=0.9, top_p=0.9, draft_context=0,
           profile=False, trace=False, reuse_draft_cache=False):
    if mode not in {"feature", "joint", "target"} or gamma < 1:
        raise ValueError("invalid mode or proposal length")
    if mode != "target" and (draft is None or draft.config.mode != mode):
        raise ValueError("Draft mode differs from requested decoding mode")
    if mode == "feature" and ranker is None:
        raise ValueError("Feature-only decoding requires a Position Ranker")
    device = next(core.parameters()).device
    spec = TokenSpec.from_core(core)
    ranks = core.sparse_position_rank_map
    prefix = list(map(int, prompt))
    validate_tokens(prefix, spec, ranks)
    if max_new_tokens < 0 or len(prefix)+max_new_tokens > core.config.sequence_len:
        raise ValueError("requested generation exceeds Target context")
    if draft_context < 0:
        raise ValueError("draft_context must be nonnegative")
    if reuse_draft_cache and draft_context:
        raise ValueError("persistent Draft cache requires full-history state")
    stats = dict(mode=mode, gamma=gamma, seed=seed, rounds=0, target_calls=0,
                 verify_calls=0, recovery_calls=0, ranker_calls=0, draft_calls=0,
                 draft_input_tokens=0, draft_commit_calls=0, draft_commit_tokens=0,
                 position_proposed=0, feature_proposed=0,
                 position_verified=0, feature_verified=0, position_accepted=0,
                 feature_accepted=0, bonus_position=0, bonus_feature=0,
                 reject_position=0, reject_feature=0, accepted_lengths=[],
                 feature_by_depth={}, reject_by_slot={},
                 timings={k: 0.0 for k in ("ranker", "draft_reset", "draft_P", "draft_F",
                                            "draft_head", "feature_sample", "target_policy",
                                            "acceptance", "draft_commit",
                                            "verify", "recovery", "prefill")})
    traces = []
    rng = torch.Generator(device=device).manual_seed(seed)
    synchronize(device)
    started = time.perf_counter()
    initial = len(prefix)
    if max_new_tokens == 0 or (len(prefix) > 1 and prefix[-1] == spec.eos):
        return dict(stats, seconds=0.0, generated_tokens=0, eos=prefix[-1] == spec.eos), []
    cache = target_cache(core, initial+max_new_tokens)
    draft_cache = None
    draft_cache_pos = 0
    # Only real, consumed Target states enter this array.
    history = torch.empty(1, initial+max_new_tokens, core.config.n_embd, device=device)
    cached_logits = None

    def measured(name, function):
        if profile:
            synchronize(device)
        t = time.perf_counter()
        result = function()
        if profile:
            synchronize(device)
            stats["timings"][name] += time.perf_counter()-t
        return result

    def append_actual_cache_tokens(cache_obj, from_pos, to_pos, committed_tokens):
        ids = torch.tensor([committed_tokens[from_pos+1:to_pos+1]], device=device, dtype=torch.long)
        tokens_per_latent = core.sparse_tokens_per_latent
        first_position = 1+(from_pos//tokens_per_latent)*tokens_per_latent
        suffix = [spec.eos]+committed_tokens[first_position:to_pos+1]
        suffix_ids = torch.tensor([suffix], device=device, dtype=torch.long)
        local_start = 1+from_pos+1-first_position
        local_stop = 1+to_pos+1-first_position
        actual_coords = core._build_sparse_position_indices(suffix_ids)[:, local_start:local_stop]
        actual_previous_hidden = history[:, from_pos:to_pos]
        draft(actual_previous_hidden, ids, actual_coords, cache_obj)
        stats["draft_calls"] += 1
        stats["draft_input_tokens"] += ids.shape[1]
        stats["draft_commit_calls"] += 1
        stats["draft_commit_tokens"] += ids.shape[1]

    def target_forward(tokens, phase):
        begin = int(cache.get_pos())
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        kinds = ["position" if (begin+j) % 2 == 0 else "feature" for j in range(len(tokens))]
        logits, hidden = measured(phase, lambda: core(ids, kv_cache=cache,
                                                     logit_types=kinds, return_hidden=True))
        history[:, begin:begin+len(tokens)] = hidden
        stats["target_calls"] += 1
        if phase == "verify":
            stats["verify_calls"] += 1
        if phase == "recovery":
            stats["recovery_calls"] += 1
        return logits[0]

    logits = target_forward(prefix, "prefill")
    cached_logits = logits[-1]
    if mode == "feature":
        measured("ranker", lambda: ranker.initialize(prefix))

    while len(prefix)-initial < max_new_tokens:
        if prefix[-1] == spec.eos and len(prefix) > 1:
            break
        remaining = max_new_tokens-(len(prefix)-initial)
        # Position recovery and even-window bonuses leave a Feature to generate.
        if mode == "target" or len(prefix) % 2 == 0:
            consumed = int(cache.get_pos())
            if consumed < len(prefix):
                cached_logits = target_forward(prefix[consumed:], "recovery")[-1]
            token = measured("target_policy", lambda: draw(policy(cached_logits, temperature, top_p), rng))
            prefix.append(token)
            if mode == "feature" and token != spec.eos and (len(prefix)-1) % 2 == 1:
                measured("ranker", lambda: ranker.after_target_position(prefix))
            continue

        stats["rounds"] += 1
        base, consumed = len(prefix), int(cache.get_pos())
        if consumed not in {base-1, base}:
            raise RuntimeError("Target consumed length must differ by at most one pending token")
        tokens_tensor = torch.tensor([prefix], device=device)
        coords = core._build_sparse_position_indices(tokens_tensor)
        # gamma=1 in B uses no Feature Draft computation.
        use_draft = mode == "joint" or min(gamma, remaining) > 1
        dc = None
        state = None
        draft_checkpoint = None
        new_draft_cache = False
        if use_draft:
            if reuse_draft_cache:
                if draft_cache is None:
                    draft_cache = draft.new_cache(initial+max_new_tokens+gamma)
                    new_draft_cache = True
                dc = draft_cache
                draft_checkpoint = draft_cache_pos
                desired_draft_pos = consumed-1
                if not new_draft_cache and draft_checkpoint > desired_draft_pos:
                    dc.cache_seqlens.fill_(desired_draft_pos)
                    draft_cache_pos = desired_draft_pos
                    draft_checkpoint = desired_draft_pos
                elif not new_draft_cache and draft_checkpoint < desired_draft_pos:
                    measured("draft_commit", lambda: append_actual_cache_tokens(
                        dc, draft_checkpoint, desired_draft_pos, prefix))
                    draft_cache_pos = desired_draft_pos
                    draft_checkpoint = desired_draft_pos
            else:
                dc = draft.new_cache(base+gamma)
            start = max(1, consumed-draft_context) if draft_context else 1
            if (not reuse_draft_cache or new_draft_cache) and consumed > start:
                def reset():
                    return draft(history[:, start-1:consumed-1], tokens_tensor[:, start:consumed],
                                 coords[:, start:consumed], dc)
                measured("draft_reset", reset)
                stats["draft_calls"] += 1
                stats["draft_input_tokens"] += consumed-start
                if reuse_draft_cache:
                    draft_cache_pos = consumed-1
            state = history[:, consumed-1:consumed]
            if consumed < base:
                pending = prefix[-1]
                kind = "P" if pending < spec.positions else "F"
                state = measured("draft_"+kind, lambda: draft(state, tokens_tensor[:, -1:], coords[:, -1:], dc))
                stats["draft_calls"] += 1
                stats["draft_input_tokens"] += 1
        candidates, qs, pending_states = [], [], {}
        trial = list(prefix)
        working = ranker.state if mode == "feature" else None
        previous = prefix[-2] if base > 1 else None
        for j in range(min(gamma, remaining)):
            kind = "P" if j % 2 == 0 else "F"
            if kind == "P" and mode == "feature":
                token, pending, working = measured("ranker", lambda: ranker.propose(working, previous, spec.eos))
                pending_states[j] = pending
                stats["ranker_calls"] += 1
                q = torch.zeros(spec.vocab, device=device)
                q[token] = 1
            else:
                local = measured("draft_head", lambda: draft.logits(state[0, -1], kind))
                def sample_draft_token():
                    if kind == "P":
                        if previous is not None:
                            local[:spec.positions].masked_fill_(ranks <= ranks[previous], -torch.inf)
                        ids = torch.cat((torch.arange(spec.positions, device=device),
                                         torch.tensor([spec.eos], device=device)))
                    else:
                        ids = torch.arange(spec.offset, spec.offset+spec.features, device=device)
                    proposal = torch.zeros(spec.vocab, device=device)
                    proposal[ids] = policy(local, temperature, top_p)
                    return proposal, draw(proposal, rng)
                if kind == "F":
                    q, token = measured("feature_sample", sample_draft_token)
                else:
                    q, token = sample_draft_token()
            candidates.append(token)
            qs.append(q)
            stats[("position" if kind == "P" else "feature")+"_proposed"] += 1
            if kind == "F":
                depth = str(j//2+1)
                counts = stats["feature_by_depth"].setdefault(depth, dict(proposed=0, verified=0, accepted=0))
                counts["proposed"] += 1
            trial.append(token)
            if token == spec.eos:
                break
            if kind == "P":
                previous = token
            if j+1 < min(gamma, remaining):
                token_tensor = torch.tensor([[token]], device=device)
                trial_coords = core._build_sparse_position_indices(torch.tensor([trial], device=device))[:, -1:]
                state = measured("draft_"+kind, lambda: draft(state, token_tensor, trial_coords, dc))
                stats["draft_calls"] += 1
                stats["draft_input_tokens"] += 1

        pending_count = base-consumed
        verify = target_forward(prefix[consumed:]+candidates, "verify")
        accepted, rejected = 0, False
        round_trace = dict(before=base, consumed_before=consumed, proposals=candidates,
                           rejection=None, bonus_kind=None)
        for j, (token, q) in enumerate(zip(candidates, qs)):
            kind = "position" if j % 2 == 0 else "feature"
            row = pending_count+j-1
            target_logits = cached_logits if row < 0 else verify[row]
            p = measured("target_policy", lambda: policy(target_logits, temperature, top_p))
            ok, actual = measured("acceptance", lambda: accept(p, q, token, rng, greedy=temperature == 0))
            stats[kind+"_verified"] += 1
            if kind == "feature":
                stats["feature_by_depth"][str(j//2+1)]["verified"] += 1
            prefix.append(actual)
            if mode == "feature" and kind == "position":
                ranker.state = ranker.committed(pending_states[j], actual, spec.eos)
                ranker.previous = actual
                if ranker.state is None and actual != spec.eos:
                    measured("ranker", lambda: ranker.after_target_position(prefix))
            if ok:
                accepted += 1
                stats[kind+"_accepted"] += 1
                if kind == "feature":
                    stats["feature_by_depth"][str(j//2+1)]["accepted"] += 1
            else:
                rejected = True
                stats["reject_"+kind] += 1
                slot = ("P" if kind == "position" else "F")+str(j//2+1)
                stats["reject_by_slot"][slot] = stats["reject_by_slot"].get(slot, 0)+1
                round_trace["rejection"] = kind
                break
            if actual == spec.eos:
                break
        keep = base+accepted
        cache.cache_seqlens.fill_(keep)
        cached_logits = verify[pending_count+accepted-1] if pending_count+accepted else cached_logits
        stats["accepted_lengths"].append(accepted)
        if not rejected and prefix[-1] != spec.eos and len(prefix)-initial < max_new_tokens:
            next_token = measured("target_policy", lambda: draw(policy(cached_logits, temperature, top_p), rng))
            kind = "position" if len(prefix) % 2 else "feature"
            prefix.append(next_token)
            stats["bonus_"+kind] += 1
            round_trace["bonus_kind"] = kind
            if mode == "feature" and kind == "position" and next_token != spec.eos:
                measured("ranker", lambda: ranker.after_target_position(prefix))
        if reuse_draft_cache and use_draft and prefix[-1] != spec.eos:
            committed_draft_pos = keep-1
            if committed_draft_pos < draft_checkpoint:
                raise RuntimeError("Target rollback moved before the persistent Draft cache checkpoint")
            if committed_draft_pos > draft_checkpoint:
                def commit_actual_tokens():
                    dc.cache_seqlens.fill_(draft_checkpoint)
                    append_actual_cache_tokens(dc, draft_checkpoint, committed_draft_pos, prefix)
                measured("draft_commit", commit_actual_tokens)
            draft_cache_pos = committed_draft_pos
        round_trace.update(accepted=accepted, after=len(prefix), consumed_after=int(cache.get_pos()))
        if trace:
            traces.append(round_trace)

    synchronize(device)
    stats.update(seconds=time.perf_counter()-started, generated_tokens=len(prefix)-initial,
                 eos=len(prefix)>1 and prefix[-1] == spec.eos, draft_context=draft_context)
    stats["tokens_per_second"] = stats["generated_tokens"]/stats["seconds"]
    stats["progress_per_verification"] = stats["generated_tokens"]/stats["verify_calls"] if stats["verify_calls"] else None
    validate_tokens(prefix, spec, ranks)
    if trace:
        stats["trace"] = traces
    return stats, prefix[initial:]
