"""Evaluate the selected GPT Feature draft with the existing causal Position ranker."""

import argparse
import json
from pathlib import Path
import time

from draft_a.feature_acceptance_eval import (
    EOS, KINDS, PREFIXES, _require, _ratio, _add_rates, _test_records,
    _prompts, _online_summary, _online_totals, _bootstrap, _online_measure,
    cache_for, policy, draw, residual, verify_blocks, toy_test,
)


def validate_sequence(prompt, output, ranks):
    _require(prompt[0] == EOS, "Missing BOS")
    body = prompt[1:] + output
    if EOS in body:
        _require(body[-1] == EOS and body.count(EOS) == 1, "Invalid EOS location")
        body = body[:-1]
        _require(len(body) % 2 == 0, "EOS must follow a complete pair")
    positions, features = body[::2], body[1::2]
    _require(all(0 <= p < 8000 for p in positions), "Invalid Position")
    _require(all(8000 <= f < EOS for f in features), "Invalid Feature")
    _require(all(ranks[b] > ranks[a] for a, b in zip(positions, positions[1:])),
             "Position ranks must increase")


def position_state(prefix, ranks):
    from gpt_position_spec_decode_smoke import CausalRanker
    state = CausalRanker(rank_map=ranks, p_vocab=8000)
    positions = prefix[1::2]
    for previous, actual in zip(positions, positions[1:]):
        state.propose(previous)
        state.finalize(actual)
    return state


def typed_kinds(start, count):
    return ["position" if (start + j) % 2 == 0 else "feature" for j in range(count)]


def decode(target, draft, ranker, ranks, prompt, budget, k, seed):
    """K Position point proposals, K-1 Feature samples, and a Target Feature bonus."""
    import torch
    from gpt_position_spec_decode_smoke import choose_position
    from bench_gaussiangpt_pf_block2_cow import fork_position_state, truncate

    _require(budget >= 2 and budget % 2 == 0 and k >= 0, "Need an even token budget")
    validate_sequence(prompt, [], ranks)
    _require((len(prompt) - 1) % 2 == 0, "Prompt must end at a complete pair")
    rng = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    length = len(prompt)
    tc = cache_for(target, length + budget + 2 * k + 2)
    dc = cache_for(draft, length + budget + 2 * k + 2) if k > 1 and draft is not None else None
    stats = dict(target_calls=0, draft_calls=0, ranker_calls=0,
                 target_input_tokens=0, draft_input_tokens=0, proposed=0,
                 verified=0, accepted=0, rounds=0, accepted_lengths=[], position_fallbacks=0)
    for kind in KINDS:
        for key in ("proposed", "verified", "accepted"):
            stats[kind + "_" + key] = 0

    def forward(core, tokens, cache, which):
        stats[which + "_calls"] += 1
        stats[which + "_input_tokens"] += len(tokens)
        kinds = typed_kinds(cache.get_pos(), len(tokens)) if which == "target" else ["feature"] * len(tokens)
        return core(torch.tensor([tokens], device="cuda"), kv_cache=cache, logit_types=kinds)[0]

    def tf(tokens):
        return forward(target, tokens, tc, "target")

    def df(tokens):
        return forward(draft, tokens, dc, "draft")

    def rewind(cache, keep):
        old = int(cache.get_pos())
        if old > keep:
            truncate(cache, keep, old)

    def commit(pending, actual):
        state = fork_position_state(pending)
        state.finalize(actual)
        return state

    prefix = list(prompt)
    if length > 1:
        tf(prompt[:-1])
        if k > 1 and draft is not None:
            df(prompt[:-1])
    online = position_state(prefix, ranks) if k and length > 1 else None

    while len(prefix) - length < budget:
        _require(tc.get_pos() == len(prefix) - 1, "Target KV position differs")
        remaining = budget - (len(prefix) - length)
        if not k or online is None:
            token = draw(policy(tf(prefix[-1:])[-1]), rng)
            prefix.append(token)
            if token == EOS:
                break
            if k and len(prefix) % 2 == 1:
                online = position_state(prefix, ranks)
            continue

        _require(len(prefix) % 2 == 1 and remaining % 2 == 0, "Round must start at a complete pair")
        stats["rounds"] += 1
        previous = prefix[-2]
        pending_states, positions = [], []
        working = online
        for _ in range(min(k, remaining // 2)):
            stats["ranker_calls"] += 1
            proposal = choose_position(ranker, working, previous)
            if proposal is None:
                break
            proposed = proposal[0]
            pending_states.append(working)
            positions.append(proposed)
            working = commit(working, proposed)
            previous = proposed

        if not positions:
            stats["position_fallbacks"] += 1
            token = draw(policy(tf(prefix[-1:])[-1]), rng)
            prefix.append(token)
            if token == EOS:
                online.drop_eos()
                break
            online.finalize(token)
            prefix.append(draw(policy(tf([token])[-1]), rng))
            stats["accepted_lengths"].append(0)
            continue

        base = len(prefix)
        trial, candidates, qs = list(prefix), [], []
        for j, proposed in enumerate(positions):
            candidates.append(proposed)
            qs.append(None)
            trial.append(proposed)
            if j < len(positions) - 1:
                _require(dc.get_pos() < len(trial), "Draft KV position differs")
                q = policy(df(trial[int(dc.get_pos()):])[-1])
                feature = draw(q, rng)
                candidates.append(feature)
                qs.append(q)
                trial.append(feature)

        logits = tf(prefix[-1:] + candidates)
        stats["proposed"] += len(candidates)
        stats["position_proposed"] += len(positions)
        stats["feature_proposed"] += len(positions) - 1
        accepted, rejected = 0, False
        for j, (token, q) in enumerate(zip(candidates, qs)):
            kind = "position" if j % 2 == 0 else "feature"
            p = policy(logits[j])
            alpha = float(p[token]) if q is None else min(1.0, float(p[token] / q[token]))
            stats["verified"] += 1
            stats[kind + "_verified"] += 1
            if float(torch.rand((), device="cuda", generator=rng)) < alpha:
                actual = token
                accepted += 1
                stats["accepted"] += 1
                stats[kind + "_accepted"] += 1
            else:
                if q is None:
                    r = p.clone()
                    r[token] = 0
                    _require(float(r.sum()) > 0, "Zero Position residual")
                    actual = draw(r / r.sum(), rng)
                else:
                    actual = draw(residual(p, q), rng)
                rejected = True
            prefix.append(actual)
            if kind == "position":
                if actual == EOS:
                    pending_states[j // 2].drop_eos()
                else:
                    online = commit(pending_states[j // 2], actual)
            if rejected:
                rewind(tc, len(prefix) - 1)
                if dc is not None:
                    rewind(dc, len(prefix) - 1)
                if kind == "position" and actual != EOS:
                    prefix.append(draw(policy(tf([actual])[-1]), rng))
                break

        stats["accepted_lengths"].append(accepted)
        if prefix[-1] == EOS:
            break
        if not rejected:
            _require(accepted == len(candidates), "Accepted width differs")
            prefix.append(draw(policy(logits[len(candidates)]), rng))
        _require(len(prefix) - length <= budget, "Token budget exceeded")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = prefix[length:]
    validate_sequence(prompt, output, ranks)
    for key in ("proposed", "verified", "accepted"):
        _require(stats[key] == sum(stats[kind + "_" + key] for kind in KINDS), "Typed totals differ")
    stats.update(generated_tokens=len(output), seconds=elapsed,
                 tokens_per_second=_ratio(len(output), elapsed), eos=bool(output and output[-1] == EOS),
                 k=k, proposal_width=2 * k - 1 if k else 0, seed=seed,
                 prompt_tokens=length - 1,
                 mean_accepted_length=_ratio(sum(stats["accepted_lengths"]), stats["rounds"]))
    return _add_rates(stats)


def validate_models(target, draft, ranker, ranks, record, seed):
    import torch
    import pickle
    from gpt_position_spec_decode_smoke import choose_position
    from bench_gaussiangpt_pf_block2_cow import fork_position_state

    seq = torch.cat([torch.tensor([EOS]), record["tokens"]])
    result = {"toy": toy_test(), "blocks": {}, "typed_feature_max_tv": 0.0,
              "target_feature_trials": [], "terminal_eos": []}
    for name, core in (("target", target), ("draft", draft)):
        result["blocks"][name] = verify_blocks(core, seq)
        count = min(129, len(seq))
        full_cache, typed_cache = cache_for(core, count + 4), cache_for(core, count + 4)
        tokens = seq[:count].reshape(1, -1).cuda()
        full = core(tokens, kv_cache=full_cache)[0]
        typed = core(tokens, kv_cache=typed_cache, logit_types=["feature"] * count)[0]
        for row in range(1, count, 2):
            tv = float((policy(full[row]) - policy(typed[row])).abs().sum() / 2)
            _require(tv < 0.01, "Feature-only head probabilities differ")
            result["typed_feature_max_tv"] = max(result["typed_feature_max_tv"], tv)
    for prefix, prompt in _prompts(record):
        if prefix:
            state = position_state(prompt, ranks)
            proposal = choose_position(ranker, state, prompt[-2])
            if proposal is not None:
                original = pickle.dumps(state)
                child = fork_position_state(state)
                child.finalize(proposal[0])
                choose_position(ranker, child, proposal[0])
                _require(pickle.dumps(state) == original, "Preview changed the original ranker state")
        for k in (2, 3, 4):
            matched = decode(target, target, ranker, ranks, prompt, 64, k, seed + k + prefix)
            _require(matched["feature_verified"] > 0 and matched["feature_acceptance_verified"] > 0.95,
                     "Target Feature draft acceptance must exceed 95 percent")
            result["target_feature_trials"].append(matched)
    last = max(range(8000), key=lambda p: ranks[p])
    for k in (2, 3, 4):
        terminal = decode(target, draft, ranker, ranks, [EOS, last, 8000], 8, k, seed + k)
        _require(terminal["eos"] and terminal["generated_tokens"] == 1, "Terminal EOS must end generation")
        result["terminal_eos"].append(terminal)
    result["status"] = "passed"
    return result


def run(args):
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    from draft_a import learning_eval as evaluation
    from draft_a.feature_reuse import make_model

    started = time.perf_counter()
    corpus, train, overlap, analysis = evaluation._helpers()
    records, expected = _test_records(args.source_root.resolve(), args.scenes, analysis)
    result = dict(status="running", checkpoint=str(args.checkpoint.resolve()),
                  ranker=str(args.ranker.resolve()), source_root=str(args.source_root.resolve()),
                  selected_test_files=[r["name"] for r in records], expected_test_files=expected,
                  split="test", target_precision="FP32", draft_precision="FP32",
                  policy=dict(evaluation.POLICY),
                  parameters=dict(k=args.k, tokens=args.tokens, scenes=args.scenes, seed=args.seed,
                                  prefixes=list(PREFIXES)),
                  k_definition="Position proposals per round; maximum total proposal width 2K-1",
                  position_proposal="Existing FastHistRanker deterministic point distribution",
                  feature_proposal="Selected GPT; Feature head only; reads P/F causal context",
                  acceptance_definition="accepted/verified; verification stops at first rejection",
                  proposal_ratio_definition="accepted/proposed; includes unused later proposals",
                  timings={}, online=dict(scene_count=len(records), rows=[],
                      scope="cache allocation, prefill, ranker history, proposals, verification, residual sampling; excludes loading and numerical validation"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["timings"]["elapsed_seconds"] = time.perf_counter() - started
        corpus.write_json(args.output, result)

    try:
        with torch.inference_mode(), evaluation._fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
            target = corpus.load_target().float().eval().requires_grad_(False)
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            draft = make_model(state["config"]).float().eval().requires_grad_(False)
            draft.load_state_dict(state["model"], strict=True)
            result.update(checkpoint_step=state.get("step"), model_config=state["config"])
            del state
            ranker = FastHistRanker(load(args.ranker))
            ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
            _require(draft.transformer.sparse_position_rank_map.cpu().tolist() == ranks, "Model rank orders differ")
            result["position_rank_identity"] = ranks == list(range(8000))
            save()
            longest = max(records, key=lambda r: len(r["tokens"]))
            context = len(longest["tokens"]) + args.tokens + 12
            evaluation._model_ready(target, context, torch)
            evaluation._model_ready(draft, context, torch)
            stage = time.perf_counter()
            result["online"]["validation"] = validate_models(target.transformer, draft.transformer, ranker, ranks, longest, args.seed)
            result["timings"]["validation_seconds"] = time.perf_counter() - stage
            save()
            stage = time.perf_counter()
            for prefix, prompt in _prompts(records[0]):
                for k in (0, *args.k):
                    decode(target.transformer, draft.transformer, ranker, ranks, prompt, 32, k, args.seed + k)
            result["timings"]["warmup_seconds"] = time.perf_counter() - stage
            save()
            for index, record in enumerate(records):
                for prefix, prompt in _prompts(record):
                    seed = (args.seed + index * 1009 + prefix * 17) % (2 ** 63 - 1)
                    # Rotate timed order to reduce a fixed order effect.
                    order = [0, *args.k]
                    shift = (index + prefix // 64) % len(order)
                    order = order[shift:] + order[:shift]
                    for k in order:
                        row = decode(target.transformer, draft.transformer, ranker, ranks, prompt, args.tokens, k, seed)
                        row.update(name=record["name"], split="test", mode="draft" if k else "target_only",
                                   requested_tokens=args.tokens)
                        result["online"]["rows"].append(row)
                        save()
                        print(json.dumps(dict(event="online", name=record["name"], prefix=prefix, k=k,
                                              position=row["position_acceptance_verified"],
                                              feature=row["feature_acceptance_verified"], tps=row["tokens_per_second"])), flush=True)
            rows = result["online"]["rows"]
            result["online"]["summary"] = _online_summary(rows, args.seed)
            by_k = []
            for k in args.k:
                selected = [r for r in rows if r["k"] == k]
                reference = [r for r in rows if r["k"] == 0]
                totals = _online_totals(selected)
                totals.update(k=k, proposal_width=2 * k - 1,
                              speedup_vs_target_only=totals["tokens_per_second"] / _online_totals(reference)["tokens_per_second"],
                              bootstrap=_bootstrap(selected, _online_measure, args.seed + k))
                def measure(sample):
                    a = _online_totals([r for r in sample if r["k"] == k])
                    b = _online_totals([r for r in sample if r["k"] == 0])
                    return {"speedup_vs_target_only": a["tokens_per_second"] / b["tokens_per_second"]}
                totals["paired_speedup_bootstrap"] = _bootstrap(selected + reference, measure, args.seed + k)
                by_k.append(totals)
            result["by_k"] = by_k
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="failed", error=str(error))
        save()
        raise
    print(json.dumps(dict(status=result["status"], output=str(args.output), by_k=result["by_k"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("draft_a/runs/rollout500_20261013"))
    parser.add_argument("--ranker", type=Path, default=Path("profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=24685)
    args = parser.parse_args()
    _require(1 <= args.scenes <= 50 and args.tokens >= 2 and args.tokens % 2 == 0, "Invalid scene count or budget")
    _require(set(args.k) <= {2, 3, 4} and len(args.k) == len(set(args.k)), "K must be unique among 2/3/4")
    run(args)


if __name__ == "__main__":
    main()
