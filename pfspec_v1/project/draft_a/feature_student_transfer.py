"""Finite state-student training with GPT and Target distribution teachers."""

import argparse
import copy
import json
from pathlib import Path
import random
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def probabilities(logits, top_p=0.9):
    import torch
    p = (logits.float() / .9).softmax(-1)
    if top_p is None:
        return p
    ordered, ids = p.sort(dim=-1, descending=True)
    remove = ordered.cumsum(-1) > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    ordered = ordered.masked_fill(remove, 0)
    p = torch.zeros_like(p).scatter(-1, ids, ordered)
    return p / p.sum(-1, keepdim=True)


def rows_for(records, target, draft, count):
    import torch
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
    from draft_a.pf_ranker_eval import typed_kinds, cache_for

    gathered = {key: [] for key in ("previous_state", "previous_feature", "position", "target", "gpt", "scene", "depth")}
    capture = CaptureLastBlock(target)
    try:
        for scene, record in enumerate(records):
            body = record["tokens"].long().cuda()
            seq = torch.cat([torch.tensor([12096], device="cuda"), body])
            n_pairs = len(body) // 2
            require(n_pairs > 1, "Need at least two Position/Feature pairs")
            choices = torch.linspace(1, n_pairs - 1, min(count, n_pairs - 1), device="cuda").round().long().unique()
            indices = 2 * choices + 1
            cache = cache_for(target, len(seq) + 4)
            logits, states = capture.run(target, cache, seq.tolist(), typed_kinds(0, len(seq)), torch.device("cuda"))
            target_p = probabilities(logits.index_select(0, indices)[:, 8000:12096], None)
            previous = states.index_select(0, indices - 2).float().cpu()
            del logits, states, cache
            dc = cache_for(draft, len(seq) + 4)
            logits = draft(seq.reshape(1, -1), kv_cache=dc, logit_types=["feature"] * len(seq))[0]
            gpt_p = probabilities(logits.index_select(0, indices)[:, 8000:12096], None)
            require(bool(torch.isfinite(target_p).all()) and bool(torch.isfinite(gpt_p).all()), "Teacher probabilities must be finite")
            gathered["previous_state"].append(previous)
            gathered["previous_feature"].append((seq.index_select(0, indices - 1) - 8000).cpu())
            gathered["position"].append(seq.index_select(0, indices).cpu())
            gathered["target"].append(target_p.half().cpu())
            gathered["gpt"].append(gpt_p.half().cpu())
            gathered["scene"].append(torch.full((len(indices),), scene, dtype=torch.long))
            gathered["depth"].append(choices.cpu())
            del logits, dc, target_p, gpt_p
            if (scene + 1) % 10 == 0:
                print(json.dumps(dict(event="labels", split=record["split"], scenes=scene + 1)), flush=True)
    finally:
        capture.close()
    return {key: torch.cat(value) for key, value in gathered.items()}


def collect(args):
    import torch
    from draft_a import learning_eval as ev
    from draft_a.feature_reuse import make_model

    root = args.output
    source = json.loads((args.source_root / "plan.json").read_text())
    corpus, _, _, analysis = ev._helpers()
    names = {split: [e for e in source["entries"] if e["split"] == split] for split in ("train", "val")}
    require(len(names["train"]) == 400 and len(names["val"]) == 50, "Need original 400/50 train/val split")
    plan = dict(status="collecting", source_root=str(args.source_root.resolve()), checkpoint=str(args.checkpoint.resolve()),
                files={s: [e["name"] for e in entries] for s, entries in names.items()},
                samples_per_scene=args.samples, teacher_temperature=.9, teacher_top_p=None,
                label_storage="FP16 probabilities; normalize in FP32 at use", state_storage="FP32 Target previous Position state",
                input="previous Target Position state, previous Feature, proposed Position",
                scope="real train/validation causal histories; no test labels", splits={})
    save_json(root / "data_plan.json", plan)
    started = time.perf_counter()
    with ev._fp32_runtime(torch):
        target = corpus.load_target().float().eval().requires_grad_(False)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        draft = make_model(state["config"]).float().eval().requires_grad_(False)
        draft.load_state_dict(state["model"], strict=True)
        del state
        for split, entries in names.items():
            records = analysis.read_records(args.source_root, entries)
            rows = rows_for(records, target.transformer, draft.transformer, args.samples)
            torch.save(dict(rows=rows, names=[r["name"] for r in records], split=split,
                            source_root=str(args.source_root.resolve()), checkpoint=str(args.checkpoint.resolve())), root / f"{split}_labels.pt")
            plan["splits"][split] = dict(scenes=len(records), rows=len(rows["position"]))
            save_json(root / "data_plan.json", plan)
            del rows
    plan.update(status="complete", seconds=time.perf_counter() - started)
    save_json(root / "data_plan.json", plan)


def normalize(labels):
    return labels.float() / labels.float().sum(-1, keepdim=True)


def truncate_probabilities(p):
    import torch
    ordered, ids = p.sort(dim=-1, descending=True)
    remove = ordered.cumsum(-1) > .9
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    ordered.masked_fill_(remove, 0)
    out = torch.zeros_like(p).scatter(-1, ids, ordered)
    return out / out.sum(-1, keepdim=True)


def evaluate(model, rows, batch=256):
    import torch

    values = dict(target_overlap=0., gpt_overlap=0., target_ce=0., rows=0)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows["position"]), batch):
            end = start + batch
            _, logits = model(rows["previous_state"][start:end], rows["previous_feature"][start:end], rows["position"][start:end])
            q = probabilities(logits)
            p = normalize(rows["target"][start:end])
            g = normalize(rows["gpt"][start:end])
            ps = truncate_probabilities(p)
            gs = truncate_probabilities(g)
            values["target_overlap"] += float(torch.minimum(ps, q).sum())
            values["gpt_overlap"] += float(torch.minimum(gs, q).sum())
            values["target_ce"] += float(-(p * (logits.float() / .9).log_softmax(-1)).sum())
            values["rows"] += len(q)
    for key in ("target_overlap", "gpt_overlap", "target_ce"):
        values[key] /= values["rows"]
    return values


def train(args):
    import torch
    from train_feature_state_distill import FeatureStateDraft

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(6)
    data = {s: torch.load(args.output / f"{s}_labels.pt", map_location="cpu", weights_only=True) for s in ("train", "val")}
    require(not set(data["train"]["names"]) & set(data["val"]["names"]), "Training and validation must be disjoint")
    rows = {s: {k: v.cuda() for k, v in d["rows"].items()} for s, d in data.items()}
    head = torch.load(args.old_head, map_location="cpu", weights_only=True)["weight"]
    initial = torch.load(args.old_checkpoint, map_location="cpu", weights_only=True)
    cfg = initial["model_config"]
    summary = dict(status="training", objective="CE to teacher full temperature0.9 distribution + 3*(1-overlap to teacher nucleus distribution)",
                   train_rows=len(rows["train"]["position"]), val_rows=len(rows["val"]["position"]),
                   seeds=args.seeds, steps=args.steps, teachers=["gpt", "target"],
                   selection="maximum Target Feature overlap on all 50 validation scenes; fixed seed first entry for online runs",
                   initial_checkpoint=str(args.old_checkpoint.resolve()), jobs=[])
    save_json(args.output / "training.json", summary)
    started = time.perf_counter()
    for teacher in ("gpt", "target"):
        for seed in args.seeds:
            torch.manual_seed(seed)
            random.seed(seed)
            model = FeatureStateDraft(head, position_vocab=cfg["position_vocab"], feature_vocab=cfg["feature_vocab"],
                                      d_model=cfg["d_model"], emb_dim=cfg["emb_dim"], mlp_dim=cfg["mlp_dim"],
                                      target_softcap=cfg["target_softcap"]).cuda().float()
            model.load_state_dict(initial["model_state"], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
            rng = torch.Generator(device="cuda").manual_seed(seed)
            root = args.output / f"{teacher}_seed{seed}"
            root.mkdir(parents=True, exist_ok=True)
            job = dict(teacher=teacher, seed=seed, history=[], selected_step=0, best_overlap=-1., selected_checkpoint=str(root / "selected.pt"))
            summary["jobs"].append(job)
            total_steps = max(args.steps)
            targets = normalize(rows["train"][teacher])
            nucleus = truncate_probabilities(targets)
            for step in range(total_steps + 1):
                if step in args.steps:
                    validation = evaluate(model, rows["val"])
                    train_metrics = evaluate(model, {k: v[:2048] for k, v in rows["train"].items()})
                    record = dict(step=step, validation=validation, train_sample=train_metrics,
                                  teacher=teacher, seed=seed, seconds=time.perf_counter() - started)
                    job["history"].append(record)
                    if validation["target_overlap"] > job["best_overlap"]:
                        job.update(best_overlap=validation["target_overlap"], selected_step=step)
                        value = dict(initial, model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                     teacher=teacher, seed=seed, transfer_step=step,
                                     selected_validation=validation, student_temperature=.9, student_top_p=.9)
                        torch.save(value, root / "selected.pt")
                    save_json(args.output / "training.json", summary)
                    print(json.dumps(dict(event="training_eval", **record)), flush=True)
                if step == total_steps:
                    break
                model.train()
                ids = torch.randint(len(targets), (args.batch,), device="cuda", generator=rng)
                _, logits = model(rows["train"]["previous_state"][ids], rows["train"]["previous_feature"][ids], rows["train"]["position"][ids])
                logq = (logits.float() / .9).log_softmax(-1)
                ce = -(targets[ids] * logq).sum(-1).mean()
                overlap = torch.minimum(nucleus[ids], logq.exp()).sum(-1).mean()
                loss = ce + 3 * (1 - overlap)
                require(bool(torch.isfinite(loss)), "Nonfinite student loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                require(bool(torch.isfinite(norm)), "Nonfinite student gradient")
                optimizer.step()
            job["status"] = "complete"
            save_json(args.output / "training.json", summary)
            del model, optimizer, targets, nucleus
    summary.update(status="complete", seconds=time.perf_counter() - started)
    save_json(args.output / "training.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "train"])
    parser.add_argument("--output", type=Path, default=Path("draft_a/runs/feature_student_transfer_20261014"))
    parser.add_argument("--source-root", type=Path, default=Path("draft_a/runs/rollout500_20261013"))
    parser.add_argument("--checkpoint", type=Path, default=Path("draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt"))
    parser.add_argument("--old-checkpoint", type=Path, default=Path("profiling/feature_draft/state_distill_mixed_overlap/feature_state_draft_best.pt"))
    parser.add_argument("--old-head", type=Path, default=Path("profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[24685, 24686, 24687])
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 200, 600, 1200, 2400])
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (collect if args.mode == "collect" else train)(args)


if __name__ == "__main__":
    main()
