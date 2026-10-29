"""Compare the existing state MLP and current GPT under matched decoding settings."""

import argparse
import json
from pathlib import Path
import time

from draft_a.pf_ranker_eval import (
    EOS, KINDS, PREFIXES, _require, _ratio, _add_rates, _test_records,
    _prompts, cache_for, policy, draw, validate_sequence, position_state,
    typed_kinds, decode as gpt_decode,
)


def mlp_decode(target, feature, ranker, ranks, prompt, budget, seed, bf16=False):
    """Original K=2 COW algorithm with causal prefix state replay."""
    import torch
    from gpt_position_spec_decode_smoke import choose_position
    from bench_gaussiangpt_pf_block2_cow import (
        CaptureLastBlock, fork_position_state, truncate, residual_point,
        residual_sample, decide,
    )
    from collect_feature_teacher_from_s3 import policy_batch

    _require(budget >= 2 and budget % 2 == 0, "Need an even token budget")
    validate_sequence(prompt, [], ranks)
    _require((len(prompt) - 1) % 2 == 0, "Prompt must end at a complete pair")
    rng = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    length = len(prompt)
    cache = cache_for(target, length + budget + 6)
    capture = CaptureLastBlock(target)
    stats = dict(target_calls=0, draft_calls=0, ranker_calls=0,
                 target_input_tokens=0, draft_input_tokens=0, rounds=0,
                 proposed=0, verified=0, accepted=0, accepted_lengths=[],
                 position_fallbacks=0)
    for kind in KINDS:
        for key in ("proposed", "verified", "accepted"):
            stats[kind + "_" + key] = 0

    def tf(tokens):
        stats["target_calls"] += 1
        stats["target_input_tokens"] += len(tokens)
        return capture.run(target, cache, tokens,
                           typed_kinds(int(cache.get_pos()), len(tokens)), torch.device("cuda"))

    def count(kind, field, n=1):
        stats[field] += n
        stats[kind + "_" + field] += n

    prefix = list(prompt)
    h_prev = None
    online = position_state(prefix, ranks) if length > 1 else None
    try:
        if length > 1:
            _, h = tf(prompt[:-1])
            h_prev = h[-1].detach().clone()
        while len(prefix) - length < budget:
            _require(int(cache.get_pos()) == len(prefix) - 1, "Target KV position differs")
            if online is None:
                logits, _ = tf(prefix[-1:])
                p0 = draw(policy(logits[-1]), rng)
                prefix.append(p0)
                if p0 == EOS:
                    break
                logits, h = tf([p0])
                prefix.append(draw(policy(logits[-1]), rng))
                h_prev = h[-1].detach().clone()
                online = position_state(prefix, ranks)
                continue

            stats["rounds"] += 1
            before = int(cache.get_pos())
            previous, pending_f = prefix[-2], prefix[-1]
            stats["ranker_calls"] += 1
            first = choose_position(ranker, online, previous)
            second = shadow = None
            if first is not None and budget - (len(prefix) - length) >= 4:
                shadow = fork_position_state(online)
                shadow.finalize(first[0])
                stats["ranker_calls"] += 1
                second = choose_position(ranker, shadow, first[0])

            if second is None:
                accepted = 0
                if first is None:
                    stats["position_fallbacks"] += 1
                    logits, _ = tf([pending_f])
                    actual = draw(policy(logits[-1]), rng)
                else:
                    proposed = first[0]
                    count("position", "proposed")
                    logits, h = tf([pending_f, proposed])
                    p = policy(logits[0])
                    count("position", "verified")
                    if decide(float(p[proposed]), rng, torch.device("cuda")):
                        count("position", "accepted")
                        accepted = 1
                        actual = proposed
                        h_prev = h[1].detach().clone()
                        final_f = draw(policy(logits[1]), rng)
                    else:
                        actual = residual_point(p, proposed, rng)
                        truncate(cache, before + 1, before + 2)
                prefix.append(actual)
                if actual == EOS:
                    online.drop_eos()
                    stats["accepted_lengths"].append(accepted)
                    break
                if not accepted:
                    logits, h = tf([actual])
                    h_prev = h[-1].detach().clone()
                    final_f = draw(policy(logits[-1]), rng)
                prefix.append(final_f)
                online.finalize(actual)
                stats["accepted_lengths"].append(accepted)
                continue

            p1, p2 = first[0], second[0]
            count("position", "proposed", 2)
            count("feature", "proposed")
            stats["draft_calls"] += 1
            stats["draft_input_tokens"] += 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                _, feature_logits = feature(
                    h_prev.unsqueeze(0),
                    torch.tensor([pending_f - 8000], device="cuda"),
                    torch.tensor([p1], device="cuda"),
                )
            q = policy_batch(feature_logits.float(), 0.9, 0.9)[0]
            f_local = draw(q, rng)
            f1 = f_local + 8000
            logits, h = tf([pending_f, p1, f1, p2])
            count("position", "verified")
            p = policy(logits[0])
            if not decide(float(p[p1]), rng, torch.device("cuda")):
                actual = residual_point(p, p1, rng)
                truncate(cache, before + 1, before + 4)
                prefix.append(actual)
                stats["accepted_lengths"].append(0)
                if actual == EOS:
                    online.drop_eos()
                    break
                logits, h = tf([actual])
                h_prev = h[-1].detach().clone()
                prefix.append(draw(policy(logits[-1]), rng))
                online.finalize(actual)
                continue

            count("position", "accepted")
            online = shadow
            h_p1 = h[1].detach().clone()
            p_f = policy(logits[1])[8000:EOS]
            count("feature", "verified")
            alpha = min(1.0, float(p_f[f_local] / q[f_local].clamp_min(1e-30)))
            if not decide(alpha, rng, torch.device("cuda")):
                actual_f = residual_sample(p_f, q, rng) + 8000
                truncate(cache, before + 2, before + 4)
                prefix.extend([p1, actual_f])
                online.drop_eos()
                h_prev = h_p1
                stats["accepted_lengths"].append(1)
                continue

            count("feature", "accepted")
            prefix.extend([p1, f1])
            count("position", "verified")
            p = policy(logits[2])
            if decide(float(p[p2]), rng, torch.device("cuda")):
                count("position", "accepted")
                actual = p2
                h_prev = h[3].detach().clone()
                final_f = draw(policy(logits[3]), rng)
                accepted = 3
            else:
                actual = residual_point(p, p2, rng)
                truncate(cache, before + 3, before + 4)
                accepted = 2
                if actual != EOS:
                    logits, h = tf([actual])
                    h_prev = h[-1].detach().clone()
                    final_f = draw(policy(logits[-1]), rng)
            prefix.append(actual)
            stats["accepted_lengths"].append(accepted)
            if actual == EOS:
                online.drop_eos()
                break
            prefix.append(final_f)
            online.finalize(actual)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        capture.close()
    output = prefix[length:]
    validate_sequence(prompt, output, ranks)
    _require(len(output) <= budget, "Output exceeds token budget")
    for key in ("proposed", "verified", "accepted"):
        _require(stats[key] == sum(stats[kind + "_" + key] for kind in KINDS), "Typed totals differ")
    stats.update(generated_tokens=len(output), seconds=elapsed,
                 tokens_per_second=_ratio(len(output), elapsed),
                 eos=bool(output and output[-1] == EOS), k=2, proposal_width=3,
                 seed=seed, prompt_tokens=length - 1,
                 mean_accepted_length=_ratio(sum(stats["accepted_lengths"]), stats["rounds"]))
    return _add_rates(stats), output


def validate(target, feature, ranker, ranks, record, seed):
    import torch
    from bench_gaussiangpt_pf_block2_cow import block2_decode, CaptureLastBlock
    from draft_a.feature_acceptance_eval import toy_test

    result = dict(toy=toy_test(), original_algorithm_matches=[], prefix_state_max_abs=0.0,
                  prefix_state_max_relative_l2=0.0, prefix_feature_max_tv=0.0,
                  terminal_eos=[], first_rejection_counts="accepted <= verified <= proposed")
    common = dict(eos=EOS, temp=0.9, top_p=0.9, device=torch.device("cuda"),
                  p_vocab=8000, ranks=ranks, max_tokens=64, draft_temp=0.9, draft_top_p=0.9)
    for trial in (seed, seed + 1, seed + 2):
        output, old_stats = block2_decode(target, lambda: cache_for(target, 80), ranker, feature,
                                         seed=trial, **common)
        row, adapted = mlp_decode(target, feature, ranker, ranks, [EOS], 64, trial, True)
        _require(output == adapted, "Adapted algorithm output differs from original")
        _require(row["feature_accepted"] == old_stats["feature_accepted"] and
                 row["feature_verified"] == old_stats["feature_checked"], "Original Feature counts differ")
        _require(row["position_accepted"] == old_stats["position_1_accepted"] + old_stats["position_2_accepted"],
                 "Original Position accepted count differs")
        result["original_algorithm_matches"].append(dict(seed=trial, tokens=len(output), status="passed"))
    for prefix, prompt in _prompts(record):
        if not prefix:
            continue
        a, b = CaptureLastBlock(target), CaptureLastBlock(target)
        ca, cb = cache_for(target, len(prompt) + 4), cache_for(target, len(prompt) + 4)
        try:
            _, full = a.run(target, ca, prompt[:-1], typed_kinds(0, len(prompt) - 1), torch.device("cuda"))
            reference = None
            for token in prompt[:-1]:
                _, states = b.run(target, cb, [token], typed_kinds(int(cb.get_pos()), 1), torch.device("cuda"))
                reference = states[-1]
            difference = float((full[-1] - reference).abs().max())
            relative = float((full[-1] - reference).norm() / reference.norm().clamp_min(1e-12))
            _require(relative < 1e-3, f"Prefix relative state difference {relative}")
            state = position_state(prompt, ranks)
            from gpt_position_spec_decode_smoke import choose_position
            proposal = choose_position(ranker, state, prompt[-2])
            if proposal is not None:
                f = torch.tensor([prompt[-1] - 8000], device="cuda")
                p = torch.tensor([proposal[0]], device="cuda")
                from collect_feature_teacher_from_s3 import policy_batch
                _, logits_full = feature(full[-1:].float(), f, p)
                _, logits_reference = feature(reference.unsqueeze(0).float(), f, p)
                tv = float((policy_batch(logits_full, .9, .9) -
                            policy_batch(logits_reference, .9, .9)).abs().sum() / 2)
                _require(tv < .01, f"Prefix Feature distribution TV {tv}")
                result["prefix_feature_max_tv"] = max(result["prefix_feature_max_tv"], tv)
            result["prefix_state_max_relative_l2"] = max(result["prefix_state_max_relative_l2"], relative)
            result["prefix_state_max_abs"] = max(result["prefix_state_max_abs"], difference)
        finally:
            a.close()
            b.close()
    last = max(range(8000), key=lambda p: ranks[p])
    for bf16 in (False, True):
        row, output = mlp_decode(target, feature, ranker, ranks, [EOS, last, 8000], 8, seed, bf16)
        _require(output == [EOS] and row["verified"] == 0, "Terminal fallback must emit EOS")
        result["terminal_eos"].append(dict(bf16=bf16, status="passed"))
    result["status"] = "passed"
    return result


def run(args):
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from draft_a import learning_eval as evaluation
    from draft_a.feature_reuse import make_model

    started = time.perf_counter()
    corpus, _, _, analysis = evaluation._helpers()
    records, expected = _test_records(args.source_root.resolve(), args.scenes, analysis)
    variants = ["target_only", "old_mlp_fp32", "old_mlp_bf16", "current_gpt_fp32"]
    result = dict(status="running", checkpoint=str(args.checkpoint.resolve()),
                  old_checkpoint=str(args.old_checkpoint.resolve()), ranker=str(args.ranker.resolve()),
                  source_root=str(args.source_root.resolve()),
                  selected_test_files=[r["name"] for r in records], expected_test_files=expected,
                  target_precision="FP32", policy=dict(evaluation.POLICY),
                  parameters=dict(tokens=args.tokens, scenes=args.scenes, seed=args.seed, k=2,
                                  prefixes=list(PREFIXES), variants=variants),
                  acceptance_definition="accepted/verified; only reached proposals enter verified",
                  proposal_ratio_definition="accepted/proposed; includes unused later proposals",
                  timings={}, online=dict(rows=[], scene_count=len(records),
                    scope="cache allocation, prefix processing, ranker history, proposals, verification, residual sampling; excludes loading and numerical validation"))
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
            result["checkpoint_step"] = state.get("step")
            del state
            old, _, _ = load_feature_model(args.old_checkpoint, args.old_head, target.transformer,
                                          torch.device("cuda"), 8000, EOS)
            result["model_parameters"] = dict(current=sum(p.numel() for p in draft.parameters()),
                                             old=sum(p.numel() for p in old.parameters()),
                                             old_shared_feature_head=old.frozen_head.numel())
            ranker = FastHistRanker(load(args.ranker))
            ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
            _require(draft.transformer.sparse_position_rank_map.cpu().tolist() == ranks, "Model rank orders differ")
            result["online"]["validation"] = validate(target.transformer, old, ranker, ranks, records[0], args.seed)
            save()

            def decode_variant(variant, prompt, budget, seed):
                if variant.startswith("old_mlp"):
                    row, _ = mlp_decode(target.transformer, old, ranker, ranks, prompt, budget, seed,
                                        variant == "old_mlp_bf16")
                else:
                    row = gpt_decode(target.transformer, draft.transformer, ranker, ranks, prompt, budget,
                                     0 if variant == "target_only" else 2, seed)
                row.update(variant=variant, mode="target_only" if variant == "target_only" else "draft",
                           feature_precision="BF16 autocast" if variant == "old_mlp_bf16" else "FP32",
                           target_precision="FP32", temperature=0.9, draft_top_p=0.9, target_top_p=0.9)
                return row

            for _, prompt in _prompts(records[0]):
                for variant in variants:
                    decode_variant(variant, prompt, 32, args.seed)
            for index, record in enumerate(records):
                for prefix, prompt in _prompts(record):
                    seed = (args.seed + index * 1009 + prefix * 17) % (2 ** 63 - 1)
                    shift = (index + prefix // 64) % len(variants)
                    order = variants[shift:] + variants[:shift]
                    for variant in order:
                        row = decode_variant(variant, prompt, args.tokens, seed)
                        row.update(name=record["name"], split="test", requested_tokens=args.tokens)
                        result["online"]["rows"].append(row)
                        save()
                        print(json.dumps(dict(event="online", name=record["name"], prefix=prefix,
                                              variant=variant, P=row["position_acceptance_verified"],
                                              F=row["feature_acceptance_verified"], tps=row["tokens_per_second"])), flush=True)
            _require(len(result["online"]["rows"]) == len(records) * 3 * len(variants), "Run count differs")
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="failed", error=str(error))
        save()
        raise
    print(json.dumps(dict(status=result["status"], output=str(args.output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt"))
    parser.add_argument("--source-root", type=Path, default=Path("draft_a/runs/rollout500_20261013"))
    parser.add_argument("--ranker", type=Path, default=Path("profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib"))
    parser.add_argument("--old-checkpoint", type=Path, default=Path("profiling/feature_draft/state_distill_mixed_overlap/feature_state_draft_best.pt"))
    parser.add_argument("--old-head", type=Path, default=Path("profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=24685)
    args = parser.parse_args()
    _require(1 <= args.scenes <= 50 and args.tokens >= 2 and args.tokens % 2 == 0, "Invalid scene count or token budget")
    run(args)


if __name__ == "__main__":
    main()
