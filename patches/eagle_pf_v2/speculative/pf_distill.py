"""Feature-conditioned labels, paired probes and short recurrent distillation."""
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from speculative.pf_engine import PositionRanker, accept, draw, policy, target_cache, validate_tokens
from speculative.pf_draft import TokenSpec


def runtime(core):
    from joblib import load
    from bench_fast_hgb import FastHistRanker
    from draft_a.feature_target_v2 import RANKER
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    return FastHistRanker(load(RANKER)), core.sparse_position_rank_map.cpu().tolist()


def read_draft(core, path):
    from speculative.pf_draft import DraftConfig, PFDraft
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = PFDraft(core, DraftConfig(**saved["config"])).to(next(core.parameters()).device)
    model.load_state_dict(saved["state_dict"])
    if model.config.mode != "feature":
        raise ValueError("Feature distillation requires the Feature-only model")
    return model.eval()


def feature_scores(core, hidden):
    offset = core.sparse_feature_token_offset
    weight = core.lm_head.weight[offset:offset+core.sparse_feature_vocab_size]
    return 15*torch.tanh(F.linear(hidden, weight).float()/15)


def distribution_terms(logits, teacher, labels=None):
    """Dense temperature KL for learning; final top-p overlap for reporting."""
    log_q, log_p = F.log_softmax(logits/0.9, -1), F.log_softmax(teacher.detach()/0.9, -1)
    p = log_p.exp()
    kl = (p*(log_p-log_q)).sum(-1).mean()
    reverse = (log_q.exp()*(log_q-log_p)).sum(-1).mean()
    with torch.no_grad():
        overlap = torch.minimum(policy(teacher.detach(), .9, .9), policy(logits.detach(), .9, .9)).sum(-1).mean()
    ce = logits.sum()*0
    if labels is not None:
        valid = labels >= 0
        if valid.any():
            ce = F.cross_entropy(logits[valid], labels[valid])
    return dict(kl=kl, reverse_kl=reverse, overlap=overlap, ce=ce)


def teacher_path(core, prefix, candidates, raw_history=False):
    """Both outputs correspond to exactly the supplied candidate token history."""
    device = next(core.parameters()).device
    ids = torch.tensor([prefix+candidates], device=device)
    validate_tokens(ids[0].tolist(), TokenSpec.from_core(core), core.sparse_position_rank_map)
    cache = target_cache(core, ids.shape[1])
    raw = []
    handle = core.transformer.h[-1].register_forward_hook(lambda module, inputs, output: raw.append(output.detach())) if raw_history else None
    try:
        with torch.no_grad():
            kinds = ["position" if j % 2 == 0 else "feature" for j in range(len(prefix))]
            first_logits, past = core(ids[:, :len(prefix)], kv_cache=cache, logit_types=kinds, return_hidden=True)
            kinds = ["feature" if j % 2 == 0 else "position" for j in range(len(candidates))]
            logits, future = core(ids[:, len(prefix):], kv_cache=cache, logit_types=kinds, return_hidden=True)
            coordinates = core._build_sparse_position_indices(ids)
    finally:
        if handle is not None:
            handle.remove()
    row = dict(prefix=ids[:, :len(prefix)].cpu(), candidates=ids[:, len(prefix):].cpu(),
               history=past.detach().cpu(), target=future.detach().cpu(), coordinates=coordinates.cpu())
    if raw_history:
        row["history_raw"] = raw[0].cpu()
    return row, first_logits[0, -1], logits[0]


@torch.no_grad()
def propose_path(core, draft, prefix, history, ranker_model, ranks, depth, seed, pending=False):
    device = next(core.parameters()).device
    trial = list(prefix)
    ids = torch.tensor([prefix], device=device)
    coords = core._build_sparse_position_indices(ids)
    dc = draft.new_cache(len(prefix)+2*depth)
    past = history.to(device)
    consumed = len(prefix)-int(pending)
    if consumed > 1:
        draft(past[:, :consumed-1], ids[:, 1:consumed], coords[:, 1:consumed], dc)
    state = past[:, consumed-1:consumed]
    if pending:
        state = draft(state, ids[:, -1:], coords[:, -1:], dc)
    ranker = PositionRanker(ranker_model, ranks)
    ranker.initialize(prefix)
    working, previous = ranker.state, prefix[-2] if len(prefix) > 1 else None
    rng = torch.Generator(device=device).manual_seed(seed)
    for _ in range(depth):
        token, _, working = ranker.propose(working, previous, core.sparse_eos_token_id)
        if token == core.sparse_eos_token_id:
            break
        for kind in ("P", "F"):
            if kind == "F":
                token = draw(policy(draft.logits(state[0, -1], "F"), .9, .9), rng)+draft.spec.offset
            trial.append(token)
            c = core._build_sparse_position_indices(torch.tensor([trial], device=device))[:, -1:]
            state = draft(state, torch.tensor([[token]], device=device), c, dc)
            if kind == "P":
                previous = token
    return trial[len(prefix):]


def recurrent_predictions(draft, row, return_aux=False):
    """Detach between predicted nodes; each current node keeps its own gradient."""
    device = draft.hidden_in.weight.device
    prefix, candidates = row["prefix"].to(device), row["candidates"].to(device)
    history, coords = row["history"].to(device), row["coordinates"].to(device)
    ids = torch.cat((prefix, candidates), 1)
    sources = [history[:, :-1]]
    current = history[:, -1:]
    pending = torch.as_tensor(row.get("pending", False), device=device).expand(prefix.shape[0]).bool()
    pending_pred = None
    if pending.any():
        pending_pred = draft(history[:, :-1], prefix[:, 1:], coords[:, 1:prefix.shape[1]])[:, -1:]
        current = torch.where(pending[:, None, None], pending_pred.detach(), current)
    predictions = []
    for j in range(candidates.shape[1]):
        sources.append(current.detach())
        stop = prefix.shape[1]+j+1
        current = draft(torch.cat(sources, 1), ids[:, 1:stop], coords[:, 1:stop])[:, -1:]
        predictions.append(current)
    result = torch.cat(predictions, 1)
    return (result, pending_pred, pending) if return_aux else result


def conditioned_loss(core, draft, row, kl_weight=1., ce_weight=.1, hidden_weight=.1, f2_weight=1.):
    pred, pending_pred, pending = recurrent_predictions(draft, row, return_aux=True)
    device = pred.device
    target = row["target"].to(device)
    values = dict(hidden_after_P=F.smooth_l1_loss(pred[:, ::2], target[:, ::2]),
                  hidden_after_F=F.smooth_l1_loss(pred[:, 1::2], target[:, 1::2]))
    if pending_pred is not None:
        values["hidden_pending_F"] = F.smooth_l1_loss(pending_pred[pending], row["history"].to(device)[pending, -1:])
        values["hidden_after_F"] = .5*(values["hidden_after_F"]+values["hidden_pending_F"])
    logits = draft.logits(pred[:, ::2], "F")
    labels = row["labels"].to(device)
    teachers = feature_scores(core, target[:, ::2])
    values.update(distribution_terms(logits.flatten(0, 1), teachers.flatten(0, 1), labels.flatten()))
    depth_kl, depth_weights = [], []
    for depth in range(logits.shape[1]):
        term = distribution_terms(logits[:, depth], teachers[:, depth])["kl"]
        values[f"kl_F{depth+1}"] = term
        depth_kl.append(term)
        depth_weights.append(f2_weight if depth == 1 else 1.)
    values["weighted_kl"] = sum(w*k for w, k in zip(depth_weights, depth_kl))/sum(depth_weights)
    with torch.no_grad():
        for depth in range(logits.shape[1]):
            values[f"overlap_F{depth+1}"] = torch.minimum(policy(logits[:, depth].detach(), .9, .9), policy(teachers[:, depth], .9, .9)).sum(-1).mean()
    values["loss"] = kl_weight*values["weighted_kl"]+ce_weight*values["ce"]+hidden_weight*(values["hidden_after_P"]+values["hidden_after_F"])
    return values


def collate(rows):
    keys = ("prefix", "candidates", "history", "target", "coordinates", "labels")
    result = {key: torch.cat([r[key] for r in rows], 0) for key in keys}
    result["pending"] = torch.tensor([r.get("pending", False) for r in rows])
    return result


def groups(rows):
    result = defaultdict(list)
    for row in rows:
        result[(row["prefix"].shape[1], row["candidates"].shape[1])].append(row)
    return result


def sample_batch(grouped, randomizer, size):
    group = grouped[randomizer.choice(list(grouped))]
    by_kind = {kind: [r for r in group if r["kind"] == kind] for kind in ("real", "ranker")}
    if all(by_kind.values()):
        rows = [randomizer.choice(by_kind["real" if i % 2 == 0 else "ranker"]) for i in range(size)]
    else:
        rows = [randomizer.choice(group) for _ in range(size)]
    return collate(rows)


@torch.no_grad()
def probe(core, draft, row, before_logits, logits, seed, label, old=None, resample=False):
    pred = recurrent_predictions(draft, row)
    device = pred.device
    rng = torch.Generator(device=device).manual_seed(seed)
    candidates = row["candidates"][0].tolist()
    result, alive = [], not resample
    for j in range(0, len(candidates), 2):
        p_position = policy(before_logits if j == 0 else logits[j-1], .9, .9)
        position_ok = bool(torch.rand((), generator=rng, device=device) < p_position[candidates[j]])
        alive = alive and position_ok
        scores = draft.logits(pred[0, j], "F")
        teacher = logits[j, draft.spec.offset:draft.spec.offset+draft.spec.features]
        rebuilt = feature_scores(core, row["target"].to(device)[0, j])
        if not torch.allclose(rebuilt, teacher, atol=2e-5, rtol=2e-5):
            raise ArithmeticError("Target hidden/readout does not match typed Feature logits")
        p, q = policy(teacher, .9, .9), policy(scores, .9, .9)
        token = draw(q, rng) if resample or row["kind"] == "T1" else candidates[j+1]-draft.spec.offset
        accepted, _ = accept(p, q, token, rng)
        terms = distribution_terms(scores[None], teacher[None], row["labels"][:, j//2].to(device))
        entry = dict(model=label, kind=row["kind"], depth=j//2+1,
                     pending_feature=bool(row.get("pending", False)),
                     **{k: float(v) for k, v in terms.items()}, conditional_verified=1,
                     conditional_accepted=int(accepted), path_verified=int(alive),
                     path_accepted=int(alive and accepted), position_probability=float(p_position[candidates[j]]))
        result.append(entry)
        alive = alive and accepted
        if old is not None and j == 0:
            prefix = row["prefix"][0].tolist()
            # The old model consumes the last Target P state and the actual last F.
            _, old_logits = old(row["history_raw"].to(device)[0, -2][None],
                                torch.tensor([prefix[-1]-draft.spec.offset], device=device),
                                torch.tensor([candidates[0]], device=device))
            old_terms = distribution_terms(old_logits, teacher[None], row["labels"][:, 0].to(device))
            oq = policy(old_logits[0], .9, .9)
            ot = draw(oq, rng)
            ok, _ = accept(p, oq, ot, rng)
            result.append(dict(model="old_pf", kind=row["kind"], depth=1,
                               pending_feature=bool(row.get("pending", False)),
                               **{k: float(v) for k, v in old_terms.items()},
                               conditional_verified=1, conditional_accepted=int(ok),
                               path_verified=0, path_accepted=0, position_probability=float(p_position[candidates[0]])))
    return result


def summarize_probes(rows):
    result = {}
    for key in sorted({(r["model"], r["kind"], r["depth"], r["pending_feature"]) for r in rows}):
        selected = [r for r in rows if (r["model"],r["kind"],r["depth"],r["pending_feature"]) == key]
        item = {metric: sum(r[metric] for r in selected)/len(selected)
                for metric in ("kl", "reverse_kl", "overlap", "position_probability")}
        for kind in ("conditional", "path"):
            n, a = (sum(r[kind+"_"+field] for r in selected) for field in ("verified", "accepted"))
            item.update({kind+"_verified": n, kind+"_accepted": a, kind+"_acceptance": a/n if n else None})
        item["real_ce"] = sum(r["ce"] for r in selected)/len(selected) if key[1] == "T1" else None
        result["/".join(map(str, key))] = item
    return result


def diagnose(core, draft, args):
    from train_pf_draft import save_json
    ranker_model, ranks = runtime(core)
    records = torch.load(args.v3_crops, map_location="cpu", weights_only=False)
    selected, seen = [], set()
    for row in records:
        if row["scene"] not in seen and len(row["tokens"]) > max(args.v3_anchors)+4:
            seen.add(row["scene"])
            selected.append(row)
        if len(selected) >= args.v3_scenes:
            break
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from draft_a.feature_target_v2 import HEAD
    old, _, _ = load_feature_model(Path("draft_a/runs/feature_target_continue_20261014/selected.pt"), HEAD, core, "cuda", 8000, 12096)
    reference = read_draft(core, args.v3_reference) if args.v3_reference else None
    rows, matched, total = [], 0, 0
    for record in selected:
        for anchor in args.v3_anchors:
            prefix = [draft.spec.eos]+record["tokens"][:anchor].tolist()
            real = record["tokens"][anchor:anchor+2].tolist()
            real_row, before, logits = teacher_path(core, prefix, real, raw_history=True)
            real_row.update(labels=torch.tensor([[real[1]-draft.spec.offset]]), kind="T1")
            rows.extend(dict(r, scene=record["scene"], prefix_tokens=anchor) for r in probe(core, draft, real_row, before, logits, args.seed+anchor, args.v3_label, old))
            if reference is not None:
                rows.extend(dict(r, scene=record["scene"], prefix_tokens=anchor)
                            for r in probe(core, reference, real_row, before, logits, args.seed+anchor, "feature_v2_replay", resample=True))
            for repeat in range(args.v3_repeats):
                seed = args.seed+anchor+repeat*1009
                for pending in (False, True):
                    candidates = propose_path(core, draft, prefix, real_row["history"], ranker_model, ranks, 2, seed, pending=pending)
                    if not candidates:
                        continue
                    matched += candidates[0] == real[0]
                    total += 1
                    for name, path in (("T2", candidates[:2]), ("T3", candidates)):
                        row, before, logits = teacher_path(core, prefix, path, raw_history=name == "T2")
                        row.update(labels=torch.full((1, len(path)//2), -1), kind=name, pending=pending)
                        rows.extend(dict(r, scene=record["scene"], prefix_tokens=anchor, repeat=repeat)
                                    for r in probe(core, draft, row, before, logits, seed, args.v3_label, old if name == "T2" else None))
                        if reference is not None:
                            rows.extend(dict(r, scene=record["scene"], prefix_tokens=anchor, repeat=repeat)
                                        for r in probe(core, reference, row, before, logits, seed, "feature_v2_replay", resample=True))
            print(json.dumps(dict(diagnose=args.v3_label, scene=record["scene"], anchor=anchor)), flush=True)
    save_json(args.output/"diagnostic_records.json", rows)
    result = dict(sources=len(selected), anchors=args.v3_anchors, repeats=args.v3_repeats,
                  ranker_matches_real=matched, ranker_comparisons=total,
                  sampling=dict(temperature=.9, top_p=.9), dense_kl_top_p=None,
                  reference_replayed_on_current_candidates=str(args.v3_reference) if args.v3_reference else None,
                  scope="paired conditional probes; path counts stop at the first P/F rejection; old PF comparison is at F1 only",
                  totals=summarize_probes(rows))
    save_json(args.output/"diagnostic_summary.json", result)
    return result


def prepare(core, draft, args):
    from train_pf_draft import save_json
    ranker_model, ranks = runtime(core)
    manifest = json.loads((args.data/"manifest.json").read_text())
    rows, skipped = [], 0
    for entry in manifest["rows"]:
        original = torch.load(args.data/entry["path"], map_location="cpu", weights_only=True)
        for anchor in args.v3_anchors:
            if len(original["tokens"]) <= anchor+2:
                continue
            prefix = original["tokens"][:anchor+1].tolist()
            real = original["tokens"][anchor+1:anchor+3].tolist()
            row = dict(prefix=original["tokens"][:anchor+1][None], candidates=torch.tensor([real]),
                       history=original["hidden"][:anchor+1][None], target=original["hidden"][anchor+1:anchor+3][None],
                       coordinates=original["coordinates"][:anchor+3][None], labels=torch.tensor([[real[1]-draft.spec.offset]]),
                       kind="real", split=entry["split"], scene=entry["scene"])
            rows.append(row)
            candidates = propose_path(core, draft, prefix, row["history"], ranker_model, ranks, 1, args.seed+anchor+len(rows), pending=True)
            if not candidates:
                skipped += 1
                continue
            conditioned, _, _ = teacher_path(core, prefix, candidates)
            conditioned.update(labels=torch.full((1, 1), -1), kind="ranker", split=entry["split"], scene=entry["scene"], pending=True)
            rows.append(conditioned)
        print(json.dumps(dict(prepare=entry["path"], rows=len(rows))), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(rows, args.output/"conditioned.pt")
    summary = dict(source=str(args.data), initial=str(args.init), anchors=args.v3_anchors,
                   rows=len(rows), skipped_eos=skipped, train_scenes=manifest["train_scenes"], val_scenes=manifest["val_scenes"],
                   splits={split: {kind: sum(r["split"] == split and r["kind"] == kind for r in rows)
                                   for kind in ("real", "ranker")} for split in ("train", "val")})
    save_json(args.output/"conditioned_summary.json", summary)
    return summary


def train_conditioned(core, draft, args):
    from train_pf_draft import save_json
    ranker_model, ranks = runtime(core)
    all_rows = torch.load(args.data/"conditioned.pt", map_location="cpu", weights_only=False)
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]
    if {r["scene"] for r in train_rows} & {r["scene"] for r in val_rows}:
        raise ValueError("scene sources overlap across training and validation")
    train_groups, val_groups = groups(train_rows), groups(val_rows)
    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = dict(kl_weight=args.kl_weight, ce_weight=args.ce_weight, hidden_weight=args.hidden_weight,
                   f2_weight=args.v3_f2_weight)
    metadata = dict(initial=str(args.init), stage=args.v3_stage, steps=args.steps, lr=args.lr,
                    batch=args.v3_batch, seed=args.seed, data=str(args.data), weights=weights,
                    distribution="dense T=0.9 KL; final top-p=0.9 overlap for selection",
                    rollout_refresh=args.v3_refresh, rollout_pool=args.v3_pool,
                    rollout_fraction=args.v3_rollout_fraction,
                    train_scenes=sorted({r["scene"] for r in train_rows}), val_scenes=sorted({r["scene"] for r in val_rows}))
    save_json(args.output/"training_config.json", metadata)
    val_rollout_groups = None
    if args.v3_stage == "rollout":
        validation_pool = []
        originals = [r for r in val_rows if r["kind"] == "real"]
        vrng = random.Random(args.seed+811)
        for number in range(64):
            source = vrng.choice(originals)
            prefix = source["prefix"][0].tolist()
            candidates = propose_path(core, draft, prefix, source["history"], ranker_model, ranks, 2, args.seed+811+number, pending=True)
            if len(candidates) < 4:
                continue
            row, _, _ = teacher_path(core, prefix, candidates)
            row.update(labels=torch.full((1, 2), -1), kind="ranker", pending=True, scene=source["scene"])
            validation_pool.append(row)
        if not validation_pool:
            raise ValueError("no two-Feature validation examples")
        val_rollout_groups = groups(validation_pool)
        torch.save(validation_pool, args.output/"rollout_validation.pt")

    def evaluate():
        draft.eval()
        totals, count = defaultdict(float), 0
        with torch.no_grad():
            for group in val_groups.values():
                for start in range(0, len(group), args.v3_batch):
                    selected = group[start:start+args.v3_batch]
                    values = conditioned_loss(core, draft, collate(selected), **weights)
                    for key, value in values.items():
                        totals[key] += float(value)*len(selected)
                    count += len(selected)
        result = {key: value/count for key, value in totals.items()}
        if val_rollout_groups:
            totals, count = defaultdict(float), 0
            with torch.no_grad():
                for group in val_rollout_groups.values():
                    for start in range(0, len(group), args.v3_batch):
                        selected = group[start:start+args.v3_batch]
                        values = conditioned_loss(core, draft, collate(selected), **weights)
                        for key, value in values.items():
                            totals[key] += float(value)*len(selected)
                        count += len(selected)
            result.update({"rollout_"+key: value/count for key, value in totals.items()})
        result["selection_score"] = (.5*result["overlap"]+.5*(result["rollout_overlap_F1"]+args.v3_f2_weight*result["rollout_overlap_F2"])/(1+args.v3_f2_weight)
                                     if val_rollout_groups else result["overlap"])
        return result

    initial = evaluate()
    best, history, rollout_grouped = initial["selection_score"], [], None
    save_json(args.output/"initial_validation.json", initial)
    torch.save(dict(config=asdict(draft.config), state_dict=draft.state_dict(), step=0, validation=initial, metadata=metadata), args.output/"selected.pt")
    started = time.perf_counter()
    for step in range(1, args.steps+1):
        if args.v3_stage == "rollout" and (step-1) % args.v3_refresh == 0:
            draft.eval()
            pool = []
            originals = [r for r in train_rows if r["kind"] == "real"]
            for number in range(args.v3_pool):
                source = randomizer.choice(originals)
                prefix = source["prefix"][0].tolist()
                candidates = propose_path(core, draft, prefix, source["history"], ranker_model, ranks, 2, args.seed+step*1000+number, pending=True)
                if len(candidates) < 4:
                    continue
                row, _, _ = teacher_path(core, prefix, candidates)
                row.update(labels=torch.full((1, 2), -1), kind="ranker", split="train", scene=source["scene"], pending=True)
                pool.append(row)
            if not pool:
                raise ValueError("no two-Feature rollout examples")
            rollout_grouped = groups(pool)
            torch.save(pool, args.output/"rollout_pool_last.pt")
            print(json.dumps(dict(refresh=step, rows=len(pool))), flush=True)
        draft.train()
        batch = sample_batch(train_groups, randomizer, args.v3_batch)
        optimizer.zero_grad(set_to_none=True)
        values = conditioned_loss(core, draft, batch, **weights)
        value = values["loss"]
        if rollout_grouped:
            rollout_values = conditioned_loss(core, draft, sample_batch(rollout_grouped, randomizer, args.v3_batch), **weights)
            value = (1-args.v3_rollout_fraction)*value+args.v3_rollout_fraction*rollout_values["loss"]
        if not torch.isfinite(value):
            raise ArithmeticError("nonfinite Feature distillation loss")
        value.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.)
        optimizer.step()
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate()
            record = dict(step=step, seconds=time.perf_counter()-started,
                          train={k: float(v.detach()) for k,v in values.items()}, validation=metrics)
            if rollout_grouped:
                record["rollout"] = {k: float(v.detach()) for k,v in rollout_values.items()}
            history.append(record)
            saved = dict(config=asdict(draft.config), state_dict=draft.state_dict(), step=step, validation=metrics, metadata=metadata)
            torch.save(saved, args.output/"last.pt")
            if metrics["selection_score"] > best:
                best = metrics["selection_score"]
                torch.save(saved, args.output/"selected.pt")
            save_json(args.output/"history.json", history)
            print(json.dumps(record), flush=True)
    return dict(stage=args.v3_stage, initial_overlap=initial["overlap"], best_selection_score=best, steps=args.steps, seconds=time.perf_counter()-started)


def run_feature_v3(core, args):
    from train_pf_draft import save_json
    if args.init is None:
        raise ValueError("--init must name the existing Feature checkpoint")
    draft = read_draft(core, args.init)
    if args.feature_v3 == "diagnose":
        result = diagnose(core, draft, args)
    elif args.feature_v3 == "prepare":
        result = prepare(core, draft, args)
    else:
        result = train_conditioned(core, draft, args)
    save_json(args.output/"result.json", result)
