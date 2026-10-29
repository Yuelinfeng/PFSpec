"""Independent test and online acceptance for one already-selected checkpoint.

Both models use FP32 and temperature0.9/top_p0.9. Online implementation follows
draft_a.benchmark_v2.decode; later proposals after the first rejection are not verified.
Run as draft_a.feature_acceptance_eval; no training or model selection occurs.
"""

import argparse
import json
import math
from pathlib import Path
import random
import time


EOS = 12096
KINDS = ("position", "feature", "eos")
PREFIXES = (0, 64, 256)
BOOTSTRAP_DRAWS = 2000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ratio(a, b):
    return a / b if b else None


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _bootstrap(rows, measure, seed, draws=BOOTSTRAP_DRAWS):
    """Resample scenes; all runs belonging to a scene stay together."""
    scenes = {}
    for row in rows:
        scenes.setdefault(row["name"], []).append(row)
    groups = list(scenes.values())
    values = {key: [] for key in measure(rows)}
    rng = random.Random(seed)
    for _ in range(draws if groups else 0):
        sample = [row for _ in groups for row in groups[rng.randrange(len(groups))]]
        for key, value in measure(sample).items():
            if value is not None:
                _require(math.isfinite(value), "Nonfinite bootstrap statistic")
                values[key].append(value)
    return {"method": "scene resampling, percentile 95% intervals", "scene_count": len(groups),
            "resamples": draws if groups else 0, "seed": seed,
            "ci95": {key: [_percentile(v, 0.025), _percentile(v, 0.975)] for key, v in values.items()},
            "valid_resamples": {key: len(v) for key, v in values.items()}}


def cache_for(core, length):
    import torch
    from model.gpt.engine import KVCache

    cfg = core.config
    _require(next(core.parameters()).dtype == torch.float32, "Online model must be FP32")
    return KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=length,
                   head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer,
                   device="cuda", dtype=torch.float32)


def policy(logits):
    from draft_a.eval_overlap import sampling_probs
    return sampling_probs(logits, None, 0.9, 0.9)


def draw(probabilities, rng):
    import torch
    return int(torch.multinomial(probabilities, 1, generator=rng))


def residual(p, q):
    import torch
    probabilities = (p - q).clamp_min(0)
    mass = probabilities.sum()
    _require(bool(torch.isfinite(mass)) and float(mass) > 0,
             "Rejected proposal has no residual mass")
    return probabilities / mass


def _kind(token, slot):
    if token == EOS:
        _require(slot % 2 == 0, "EOS may only finish a complete position/feature pair")
        return "eos"
    kind = "position" if slot % 2 == 0 else "feature"
    _require((0 <= token < 8000) if kind == "position" else (8000 <= token < EOS),
             "Proposal token type differs from slot")
    return kind


def validate_output(prompt, output):
    _require(prompt and prompt[0] == EOS, "Prompt must start with BOS (EOS id)")
    full = prompt[1:] + output
    if EOS in full:
        _require(full[-1] == EOS and full.count(EOS) == 1, "EOS must occur only at the end")
        full = full[:-1]
        _require(len(full) % 2 == 0, "EOS must follow a complete pair")
    positions, features = full[::2], full[1::2]
    _require(all(0 <= p < 8000 for p in positions), "Invalid Position output")
    _require(all(8000 <= f < EOS for f in features), "Invalid Feature output")
    _require(all(b > a for a, b in zip(positions, positions[1:])), "Positions must increase")


def _add_rates(stats):
    for kind in ("", *KINDS):
        prefix = kind + "_" if kind else ""
        proposed, verified, accepted = (stats[prefix + key] for key in ("proposed", "verified", "accepted"))
        _require(0 <= accepted <= verified <= proposed, "Invalid proposal/verification counts")
        stats[prefix + "acceptance_verified"] = _ratio(accepted, verified)
        stats[prefix + "acceptance_proposed"] = _ratio(accepted, proposed)
        stats[prefix + "unverified"] = proposed - verified
    stats["acceptance"] = stats["acceptance_verified"]
    return stats


def decode(target, draft, prompt, budget, k, seed):
    """Exact speculative decode, with typed counts only up to first rejection."""
    import torch

    _require(budget > 0 and k >= 0, "Budget must be positive and gamma nonnegative")
    validate_output(prompt, [])
    rng = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    length = len(prompt)
    tc = cache_for(target, length + budget + k + 2)
    dc = cache_for(draft, length + budget + k + 2) if k else None
    stats = dict(target_calls=0, draft_calls=0, proposed=0, verified=0, accepted=0,
                 rounds=0, accepted_lengths=[], target_input_tokens=0)
    for kind in KINDS:
        for key in ("proposed", "verified", "accepted"):
            stats[kind + "_" + key] = 0

    def forward(core, tokens, cache, which):
        stats[which + "_calls"] += 1
        if which == "target":
            stats["target_input_tokens"] += len(tokens)
        logits = core(torch.tensor([tokens], device="cuda"), kv_cache=cache)[0]
        _require(logits.dtype == torch.float32, "Online logits must be FP32")
        return logits

    def tf(tokens):
        return forward(target, tokens, tc, "target")

    def df(tokens):
        return forward(draft, tokens, dc, "draft")

    if length > 1:
        tf(prompt[:-1])
        if k:
            df(prompt[:-1])
    prefix = list(prompt)
    while len(prefix) - length < budget:
        _require(tc.get_pos() == len(prefix) - 1, "Target cache position differs")
        if not k:
            token = draw(policy(tf(prefix[-1:])[-1]), rng)
            prefix.append(token)
            if token == EOS:
                break
            continue
        _require(dc.get_pos() == len(prefix) - 1, "Draft cache position differs")
        stats["rounds"] += 1
        candidates, qs = [], []
        token = prefix[-1]
        for _ in range(min(k, budget - (len(prefix) - length))):
            q = policy(df([token])[-1])
            token = draw(q, rng)
            candidates.append(token)
            qs.append(q)
            if token == EOS:
                break
        logits = tf(prefix[-1:] + candidates)
        accepted, rejected, base = 0, False, len(prefix)
        stats["proposed"] += len(candidates)
        for j, token in enumerate(candidates):
            stats[_kind(token, base + j - 1) + "_proposed"] += 1
        for j, (token, q) in enumerate(zip(candidates, qs)):
            p = policy(logits[j])
            kind = _kind(token, base + j - 1)
            stats["verified"] += 1
            stats[kind + "_verified"] += 1
            if float(torch.rand((), device="cuda", generator=rng)) < min(1.0, float(p[token] / q[token])):
                prefix.append(token)
                accepted += 1
                stats["accepted"] += 1
                stats[kind + "_accepted"] += 1
                if token == EOS:
                    break
            else:
                prefix.append(draw(residual(p, q), rng))
                tc.cache_seqlens.fill_(base + j)
                dc.cache_seqlens.fill_(base + j)
                rejected = True
                break
        stats["accepted_lengths"].append(accepted)
        if prefix[-1] == EOS or len(prefix) - length >= budget:
            break
        if not rejected:
            _require(accepted == len(candidates), "Accepted block length differs")
            bonus = draw(policy(logits[len(candidates)]), rng)
            df(candidates[-1:])
            prefix.append(bonus)
            if bonus == EOS:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = prefix[length:]
    validate_output(prompt, output)
    _require(len(output) <= budget, "Generated output exceeds requested budget")
    for key in ("proposed", "verified", "accepted"):
        _require(stats[key] == sum(stats[kind + "_" + key] for kind in KINDS), "Typed totals differ")
    stats.update(generated_tokens=len(output), seconds=elapsed,
                 tokens_per_second=_ratio(len(output), elapsed), eos=bool(output and output[-1] == EOS),
                 k=k, seed=seed, prompt_tokens=length - 1,
                 mean_accepted_length=_ratio(sum(stats["accepted_lengths"]), stats["rounds"]))
    return _add_rates(stats)


def verify_blocks(core, seq):
    """Compare FP32 tokenwise, blockwise and rewound-cache sampling probabilities."""
    import torch

    seq = seq[:129].reshape(1, -1).cuda()
    _require(seq.shape[1] > 0, "Empty block-validation sequence")
    step_cache = cache_for(core, seq.shape[1] + 17)
    refs = []
    for t in range(seq.shape[1]):
        row = core(seq[:, t:t + 1], kv_cache=step_cache)[0, -1]
        _require(row.dtype == torch.float32, "Block-validation logits must be FP32")
        refs.append(policy(row).cpu())
    maximum = 0.0
    for width in (2, 4, 8, 16):
        cached = cache_for(core, seq.shape[1] + 17)
        for start in range(0, seq.shape[1], width):
            logits = core(seq[:, start:start + width], kv_cache=cached)[0]
            _require(logits.dtype == torch.float32, "Block-validation logits must be FP32")
            for j, row in enumerate(logits):
                tv = float((policy(row).cpu() - refs[start + j]).abs().sum() / 2)
                _require(tv < 0.01, f"Block probabilities differ: {width}/{start + j}/{tv}")
                maximum = max(maximum, tv)
        if seq.shape[1] > 7:
            cached.cache_seqlens.fill_(7)
            logits = core(seq[:, 7:15], kv_cache=cached)[0]
            _require(logits.dtype == torch.float32, "Rewound-cache logits must be FP32")
            for j, row in enumerate(logits):
                tv = float((policy(row).cpu() - refs[7 + j]).abs().sum() / 2)
                _require(tv < 0.01, f"Rewound-cache probabilities differ: {j}/{tv}")
                maximum = max(maximum, tv)
    return {"max_total_variation": maximum, "tolerance": 0.01,
            "dtype": "FP32", "widths": [2, 4, 8, 16], "rewind_tested": seq.shape[1] > 7}


def toy_test():
    """The reference residual identity and a seeded CPU empirical test."""
    p, q = [0.05, 0.25, 0.0, 0.7], [0.4, 0.1, 0.5, 0.0]
    weights = [max(a - b, 0.0) for a, b in zip(p, q)]
    r = [value / math.fsum(weights) for value in weights]
    accepted_mass = math.fsum(min(a, b) for a, b in zip(p, q))
    expected = [min(a, b) + (1 - accepted_mass) * c for a, b, c in zip(p, q, r)]
    _require(all(math.isclose(a, b, abs_tol=1e-12, rel_tol=0) for a, b in zip(expected, p)),
             "Residual sampling identity differs")
    rng = random.Random(34)

    def sample(probabilities):
        uniform, cumulative = rng.random(), 0.0
        for index, value in enumerate(probabilities):
            cumulative += value
            if uniform < cumulative:
                return index
        return len(probabilities) - 1

    counts, accepted = [0] * len(p), 0
    for _ in range(100000):
        token = sample(q)
        if rng.random() < min(1.0, p[token] / q[token]):
            accepted += 1
        else:
            token = sample(r)
        counts[token] += 1
    empirical = [n / 100000 for n in counts]
    _require(max(abs(a - b) for a, b in zip(empirical, p)) < 0.007, "Toy empirical probabilities differ")
    return {"status": "passed", "expected": p, "empirical": empirical,
            "acceptance": accepted / 100000, "exact_acceptance": accepted_mass, "seed": 34}


def _test_records(source_root, max_scenes, analysis):
    plan = json.loads((source_root / "plan.json").read_text())
    entries = plan["entries"]
    names = [entry["name"] for entry in entries]
    _require(len(set(names)) == len(names), "Source plan repeats a filename")
    tests = [entry for entry in entries if entry["split"] == "test"]
    _require(len(tests) == 50, "Source plan must contain exactly 50 test scenes")
    for entry in tests:
        name = entry["name"]
        _require(Path(name).name == name and name.endswith(".pt"), "Test filenames must be plain .pt names")
    selected = tests[:max_scenes]
    records = analysis.read_records(source_root, selected)
    _require([r["name"] for r in records] == [e["name"] for e in selected], "Test record order differs")
    _require(all(r["split"] == "test" for r in records), "Only test records may be evaluated")
    return records, [e["name"] for e in tests]


def _test_summary(rows, evaluation):
    # Existing reducer has two named slots; only its arithmetic is reused.
    return evaluation._summarize_types([dict(row, split="val") for row in rows])["val"]


def _offline_measure(rows, evaluation):
    fields = ("pooled_overlap", "equal_trajectory_mean_overlap", "top1_match", "pentropy", "qentropy",
              "token_weighted_ce", "target_token_weighted_ce", "target_token_weighted_raw_entropy")
    return {kind + "/" + field: values[field]
            for kind, values in _test_summary(rows, evaluation).items() for field in fields}


def _offline(target, draft, records, source_root, output, evaluation, train, overlap, seed):
    import torch

    root = output.parent / "teacher_cache_test"
    per_sequence, timings = [], []
    reused = created = 0
    started = time.perf_counter()
    for record in records:
        seq = train.sequence(record)
        meta = evaluation._meta(record, seq, overlap)
        path = root / record["name"]
        torch.cuda.synchronize()
        cache_started = time.perf_counter()
        if path.exists():
            cache = torch.load(path, map_location="cpu", weights_only=True)
            _require(cache.get("source_root") == str(source_root) and
                     isinstance(cache.get("tokens"), torch.Tensor) and
                     torch.equal(cache["tokens"], record["tokens"]), "Test cache source/tokens differ")
            reused += 1
        else:
            cache = evaluation._teacher_labels(target, record, seq, meta, overlap, torch)
            cache.update(source_root=str(source_root), tokens=record["tokens"].clone().cpu())
            evaluation._valid_cache(cache, meta, torch)
            evaluation._save_labels(path, cache, torch)
            created += 1
        evaluation._valid_cache(cache, meta, torch)
        torch.cuda.synchronize()
        cache_seconds = time.perf_counter() - cache_started
        draft_started = time.perf_counter()
        per_sequence.extend(evaluation._student_sequence(draft, seq, cache, overlap, torch))
        torch.cuda.synchronize()
        timings.append(dict(name=record["name"], cache_seconds=cache_seconds,
                            draft_seconds=time.perf_counter() - draft_started))
    return {"split": "test", "scene_count": len(records), "expected_scene_count": 50,
            "complete": len(records) == 50, "scope": evaluation.SCOPE,
            "policy": dict(evaluation.POLICY), "entropy_units": "nats",
            "teacher_cache": str(root), "reused_caches": reused, "created_caches": created,
            "seconds_before_bootstrap": time.perf_counter() - started,
            "summary": _test_summary(per_sequence, evaluation), "per_sequence": per_sequence,
            "scene_timings": timings,
            "bootstrap": _bootstrap(per_sequence, lambda r: _offline_measure(r, evaluation), seed)}


def _online_totals(rows):
    fields = ("target_calls", "draft_calls", "target_input_tokens", "generated_tokens", "rounds",
              "proposed", "verified", "accepted", "seconds",
              *(kind + "_" + key for kind in KINDS for key in ("proposed", "verified", "accepted")))
    result = {field: math.fsum(row.get(field, 0) for row in rows) if field == "seconds"
              else sum(row.get(field, 0) for row in rows) for field in fields}
    result.update(scene_count=len({row["name"] for row in rows}), run_count=len(rows),
                  eos_runs=sum(bool(row.get("eos")) for row in rows),
                  tokens_per_second=_ratio(result["generated_tokens"], result["seconds"]))
    return _add_rates(result)


def _online_measure(rows):
    stats = _online_totals(rows)
    fields = ("tokens_per_second", "acceptance_verified", "acceptance_proposed",
              *(kind + "_acceptance_" + denominator for kind in KINDS for denominator in ("verified", "proposed")))
    return {field: stats[field] for field in fields}


def _online_summary(rows, seed):
    result = {"overall": {}, "groups": []}
    for mode in ("draft", "target_only"):
        selected = [row for row in rows if row["mode"] == mode]
        result["overall"][mode] = dict(_online_totals(selected),
                                      bootstrap=_bootstrap(selected, _online_measure, seed))
    keys = sorted({(row["mode"], row["prompt_tokens"], row["k"]) for row in rows})
    for index, (mode, prefix, gamma) in enumerate(keys):
        selected = [r for r in rows if (r["mode"], r["prompt_tokens"], r["k"]) == (mode, prefix, gamma)]
        reference = [r for r in rows if r["mode"] == "target_only" and r["prompt_tokens"] == prefix]
        names = {r["name"] for r in selected}
        reference = [r for r in reference if r["name"] in names]
        tps = _online_totals(reference)["tokens_per_second"]
        totals = _online_totals(selected)
        group = dict(mode=mode, prompt_tokens=prefix, gamma=gamma, **totals,
                     speedup_vs_target_only=_ratio(totals["tokens_per_second"], tps) if tps else None,
                     bootstrap=_bootstrap(selected, _online_measure, seed + index + 1))
        if mode == "draft":
            def speedup(sample):
                speculative = _online_totals([r for r in sample if r["mode"] == "draft"])
                direct = _online_totals([r for r in sample if r["mode"] == "target_only"])
                value = _ratio(speculative["tokens_per_second"], direct["tokens_per_second"]) if direct["tokens_per_second"] else None
                return {"speedup_vs_target_only": value}
            group["paired_speedup_bootstrap"] = _bootstrap(selected + reference, speedup, seed + index + 1)
        result["groups"].append(group)
    return result


def _prompts(record):
    body = record["tokens"].tolist()
    return [(prefix, [EOS] + body[:prefix]) for prefix in PREFIXES if prefix <= len(body)]


def _online(target, draft, records, args, result, save):
    import torch
    from draft_a.benchmark_v2 import decode as target_only_decode
    from draft_a import learning_eval as evaluation

    if not records:
        return {"scene_count": 0, "rows": [], "skipped": "online-scenes=0"}
    longest = max(records, key=lambda r: len(r["tokens"]))
    seq = torch.cat([torch.tensor([EOS]), longest["tokens"]])
    started = time.perf_counter()
    validation = {"verify_blocks": {"target": verify_blocks(target.transformer, seq),
                                     "draft": verify_blocks(draft.transformer, seq)},
                  "target_target": [], "terminal_eos": []}
    result["online"] = {"validation": validation, "rows": [], "scene_count": len(records),
                        "requested_prefix_tokens": list(PREFIXES), "unavailable_prefixes": [],
                        "scope": "cache allocation, prefill, proposals, verification and residual sampling; model loading and output validation excluded"}
    for prefix, prompt in _prompts(records[0]):
        for gamma in args.gamma:
            matched = decode(target.transformer, target.transformer, prompt, min(args.tokens, 32), gamma, args.seed + gamma)
            matched.update(name=records[0]["name"], requested_prefix_tokens=prefix)
            validation["target_target"].append(matched)
            save()
            _require(matched["acceptance_verified"] is not None and matched["acceptance_verified"] > 0.95,
                     "Target/Target verified acceptance must exceed 0.95")
    for gamma in args.gamma:
        terminal = decode(target.transformer, target.transformer, [EOS, 7999, 8000], 8, gamma, args.seed + 100 + gamma)
        validation["terminal_eos"].append(terminal)
        save()
        _require(terminal["eos"] and terminal["generated_tokens"] == 1 and terminal["eos_verified"] == 1,
                 "Terminal EOS must be verified separately")
    result["timings"]["online_validation_seconds"] = time.perf_counter() - started
    warmup_started = time.perf_counter()
    target_only_decode(target.transformer, draft.transformer, [EOS], 16, 0, args.seed)
    for gamma in args.gamma:
        decode(target.transformer, draft.transformer, [EOS], 16, gamma, args.seed + gamma)
    result["timings"]["online_warmup_seconds"] = time.perf_counter() - warmup_started
    for index, record in enumerate(records):
        prompts = _prompts(record)
        result["online"]["unavailable_prefixes"].extend(
            dict(name=record["name"], prefix_tokens=n, body_tokens=len(record["tokens"]))
            for n in PREFIXES if n not in [p for p, _ in prompts])
        for prefix, prompt in prompts:
            context = len(prompt) + args.tokens + max(args.gamma) + 2
            evaluation._model_ready(target, context, torch)
            evaluation._model_ready(draft, context, torch)
            seed = (args.seed + index * 1009 + prefix * 17) % (2 ** 63 - 1)
            direct = target_only_decode(target.transformer, draft.transformer, prompt, args.tokens, 0, seed)
            for kind in KINDS:
                for key in ("proposed", "verified", "accepted"):
                    direct[kind + "_" + key] = 0
            _add_rates(direct)
            direct.update(name=record["name"], split="test", mode="target_only", requested_tokens=args.tokens)
            result["online"]["rows"].append(direct)
            save()
            for gamma in args.gamma:
                row = decode(target.transformer, draft.transformer, prompt, args.tokens, gamma, seed)
                row.update(name=record["name"], split="test", mode="draft", requested_tokens=args.tokens)
                result["online"]["rows"].append(row)
                save()
                print(json.dumps(dict(event="online", name=record["name"], prefix_tokens=prefix,
                                      gamma=gamma, position=row["position_acceptance_verified"],
                                      feature=row["feature_acceptance_verified"], tokens_per_second=row["tokens_per_second"])), flush=True)
    result["online"]["summary"] = _online_summary(result["online"]["rows"], args.seed)
    return result["online"]


def run(args):
    import torch
    from draft_a import learning_eval as evaluation
    from draft_a.feature_reuse import make_model

    started = time.perf_counter()
    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    corpus, train, overlap, analysis = evaluation._helpers()
    records, expected = _test_records(source_root, args.max_scenes, analysis)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", checkpoint=str(checkpoint), source_root=str(source_root),
                  output=str(output), split="test", model_selection="one already-selected checkpoint; no search or ranking",
                  expected_test_files=expected, selected_test_files=[r["name"] for r in records],
                  target_precision="FP32", draft_precision="FP32", policy=dict(evaluation.POLICY),
                  parameters=dict(tokens=args.tokens, max_scenes=args.max_scenes, online_scenes=args.online_scenes,
                                  gamma=args.gamma, seed=args.seed, skip_offline=args.skip_offline,
                                  prefixes=list(PREFIXES), bootstrap_resamples=BOOTSTRAP_DRAWS),
                  acceptance_definition="accepted/verified; only proposals reached before or at first rejection are verified",
                  proposal_ratio_definition="accepted/proposed; includes unused later proposals in denominator",
                  timings={}, toy_test=toy_test(), offline={"skipped": True} if args.skip_offline else None,
                  online=None)

    def save():
        result["timings"]["total_elapsed_seconds"] = time.perf_counter() - started
        corpus.write_json(output, result)

    target = draft = state = None
    try:
        with evaluation._fp32_runtime(torch):
            load_started = time.perf_counter()
            target = corpus.load_target().float().eval().requires_grad_(False)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            draft = make_model(state["config"]).float()
            draft.load_state_dict(state["model"], strict=True)
            draft.eval().requires_grad_(False)
            result.update(checkpoint_step=state.get("step"), model_config=state["config"],
                          model_parameters={"target": sum(p.numel() for p in target.parameters()),
                                            "draft": sum(p.numel() for p in draft.parameters())})
            state = None
            torch.cuda.synchronize()
            result["timings"]["model_loading_seconds"] = time.perf_counter() - load_started
            save()
            if not args.skip_offline:
                stage = time.perf_counter()
                result["offline"] = _offline(target, draft, records, source_root, output,
                                              evaluation, train, overlap, args.seed)
                result["timings"]["offline_seconds"] = time.perf_counter() - stage
                save()
            stage = time.perf_counter()
            result["online"] = _online(target, draft, records[:args.online_scenes], args, result, save)
            result["timings"]["online_seconds"] = time.perf_counter() - stage
            target = draft = state = None
        result["status"] = "complete"
        save()
    except Exception as error:
        target = draft = state = None
        result.update(status="failed", error=str(error))
        save()
        raise
    return result


def self_test():
    toy = toy_test()
    stats = dict(proposed=4, verified=1, accepted=0,
                 position_proposed=2, position_verified=1, position_accepted=0,
                 feature_proposed=2, feature_verified=0, feature_accepted=0,
                 eos_proposed=0, eos_verified=0, eos_accepted=0)
    _add_rates(stats)
    _require(stats["feature_acceptance_verified"] is None and stats["feature_unverified"] == 2,
             "Unreached Feature proposals must not enter verified denominator")
    _require(_kind(EOS, 0) == "eos", "EOS must be separate from Position")
    validate_output([EOS], [1, 8000, 2, 8001, EOS])
    rows = [dict(name="a", value=1), dict(name="a", value=2), dict(name="b", value=3)]
    ci = _bootstrap(rows, lambda sample: {"mean": sum(r["value"] for r in sample) / len(sample)}, 34, 100)
    _require(ci["scene_count"] == 2 and ci["resamples"] == 100, "Bootstrap must resample scenes")
    _require(_bootstrap([], lambda _: {"missing": None}, 34)["ci95"]["missing"] == [None, None],
             "Empty intervals must be null")
    return dict(self_test="passed", toy_test=toy, file_writes=0,
                coverage=["residual identity and empirical probabilities", "first rejection denominators",
                          "EOS separation", "scene bootstrap", "empty intervals"],
                cuda_validation="verify_blocks and Target/Target trials run during online evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--checkpoint", type=Path, help="Final selected checkpoint; never use test results to select it")
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--max-scenes", type=int, default=50)
    parser.add_argument("--online-scenes", type=int, default=12, help="Use 0 for offline only")
    parser.add_argument("--gamma", nargs="+", default=["2,4"], help="Comma-separated or space-separated positive integers")
    parser.add_argument("--seed", type=int, default=24681)
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, allow_nan=False), flush=True)
        return
    if args.source_root is None or args.output is None:
        parser.error("--source-root and --output are required with --checkpoint")
    try:
        args.gamma = [int(value) for item in args.gamma for value in item.split(",")]
    except ValueError:
        parser.error("--gamma must contain integers")
    if (not args.gamma or min(args.gamma) <= 0 or len(set(args.gamma)) != len(args.gamma)
            or args.tokens <= 0 or not 1 <= args.max_scenes <= 50
            or not 0 <= args.online_scenes <= args.max_scenes or not 0 <= args.seed < 2 ** 63 - 1024):
        parser.error("Require positive tokens/gamma, 1..50 max-scenes, 0..max-scenes online-scenes and a nonnegative seed")
    result = run(args)
    print(json.dumps(dict(status=result["status"], output=result["output"],
                          offline_summary=result["offline"].get("summary"),
                          online_summary=result["online"].get("summary")), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
