"""Train D4 against original Target labels using the agreed finite budget."""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
import traceback

from draft_a.d4_joint_cost import D8, SOURCE, LABELS, EVALUATION, ROOT, load_model, real_inputs, measure
from draft_a.feature_target_v2 import save

STAGES = [("initial", 5000, 3e-4, 0.0), ("full_ce", 1200, 3e-5, 0.0),
          ("low_lr", 1200, 1e-5, 0.0), ("policy_tv", 1200, 1e-5, 2.0)]
ENDS = [5000, 6200, 7400, 8600]


def atomic_state(path, state):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def model_state(model, config, step, tokens, features):
    return dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=config,
                step=step, trained_tokens=tokens, trained_feature_tokens=features)


def update(model, optimizer, batch, labels, rate, tv):
    import torch
    from draft_a.feature_adapt_continue import continuation_loss
    tokens = sum(len(r["seq"])-1 for r in batch)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    model.train()
    total = 0.0
    for record in batch:
        seq = record["seq"].cuda().unsqueeze(0)
        p = labels[record["name"]].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=seq[:, :-1])[0]
            loss = continuation_loss(logits, seq[0, 1:], p, tv)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss")
        weight = (len(record["seq"])-1)/tokens
        (loss*weight).backward()
        total += float(loss.detach())*weight
        del seq, logits, loss, p
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(norm)):
        raise ValueError("Nonfinite training gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total, tokens, sum(len(r["tokens"])//2 for r in batch), float(norm)


def validate_initial(model, config, records, labels, root, smoke):
    import torch
    from draft_a.corpus_v2 import load_target
    from draft_a.feature_acceptance_eval import decode, verify_blocks
    from draft_a.learning_eval import _fp32_runtime
    target = load_target().float().eval().requires_grad_(False)
    d8 = load_model(D8)
    real = real_inputs()[0]["seq"]
    if not torch.equal(model.transformer.sparse_position_rank_map, target.transformer.sparse_position_rank_map):
        raise ValueError("D4 and Target Position ordering differs")
    numerical = {}
    short = []
    with _fp32_runtime(torch):
        for name, item in [("target", target), ("d8", d8), ("d4", model)]:
            numerical[name] = verify_blocks(item.transformer, real)
        for prefix in ([64] if smoke else [64, 256, 1024]):
            for gamma in ([2] if smoke else [2, 4, 6]):
                prompt = real[:prefix+1].tolist()
                same = decode(target.transformer, target.transformer, prompt, 32, gamma, 24700)
                if same["accepted"] != same["verified"]:
                    raise ValueError("Target self proposals differ")
                a = decode(target.transformer, model.transformer, prompt, 32, gamma, 24700)
                b = decode(target.transformer, model.transformer, prompt, 32, gamma, 24700)
                if any(a[k] != b[k] for k in ["accepted", "verified", "proposed", "rounds", "generated_tokens"]):
                    raise ValueError("D4 repeated decoding counts differ")
                short.append(dict(prefix=prefix, gamma=gamma, status="passed", target_self_accepted=same["accepted"],
                                  target_self_verified=same["verified"]))
        measure({"d4": model, "d8": d8}, root, smoke)
    del target, d8
    gc.collect()
    torch.cuda.empty_cache()
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    # Keep this trial separate from formal initialization and optimizer updates.
    initial = model_state(model, config, 0, 0, 0)
    atomic_state(root / "initial.pt", initial)
    longest = max(records, key=lambda r: len(r["seq"]))
    batch = [longest, records[0]]
    first = update(model, optimizer, batch, labels, 3e-4, 0.0)
    smoke_state = dict(model_state(model, config, 1, first[1], first[2]), optimizer=optimizer.state_dict(),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    atomic_state(root / "short_restore.pt", smoke_state)
    second = update(model, optimizer, batch, labels, 1e-5, 2.0)
    expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    restored = torch.load(root / "short_restore.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["torch_rng"])
    torch.cuda.set_rng_state_all(restored["cuda_rng"])
    repeated = update(model, optimizer, batch, labels, 1e-5, 2.0)
    delta = 0.0
    for name, value in model.state_dict().items():
        actual = value.detach().cpu()
        torch.testing.assert_close(actual, expected[name], atol=1e-6, rtol=1e-6)
        if actual.is_floating_point():
            delta = max(delta, float((actual-expected[name]).abs().max()))
    if not math.isclose(second[0], repeated[0], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Restored training loss differs")
    original = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(original["model"], strict=True)
    save(root / "initial_validation.json", dict(models=numerical, short_runs=short,
         training=dict(status="passed", longest_name=longest["name"], longest_tokens=len(longest["seq"])-1,
                       first_loss=first[0], tv_loss=second[0], restored_loss=repeated[0], max_parameter_difference=delta)))
    del optimizer, smoke_state, expected, restored, original, initial
    gc.collect()
    torch.cuda.empty_cache()


def run(args):
    import torch
    from omegaconf import OmegaConf
    from draft_a.feature_distill import _read_training, load_training_labels
    from draft_a.feature_reuse import make_model, evaluate
    from draft_a.learning_eval import require_complete_cache
    if (args.root / "training_plan.json").exists() and not args.resume:
        raise FileExistsError("Training already has a plan; use resume for its saved state")
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training plan requires BF16 CUDA support")
    plan = json.loads((LABELS / "plan.json").read_text())
    require_complete_cache(SOURCE, EVALUATION)
    records = _read_training(plan)
    labels = load_training_labels(LABELS, records)
    config = OmegaConf.to_container(OmegaConf.load("conf/model/gpt_draft_d4.yaml"), resolve=True)
    size = config["gpt_size"]
    if (size["n_layer"], size["n_embd"], size["n_head"], size["n_kv_head"]) != (4, 256, 4, 4):
        raise ValueError("Expected the existing D4 architecture")
    if any(len(r["seq"])-1 > config["n_ctx"] for r in records):
        raise ValueError("Training trajectory exceeds D4 context")
    torch.manual_seed(24690)
    torch.cuda.manual_seed_all(24690)
    model = make_model(config)
    stages = [(n, 1, lr, tv) for n, _, lr, tv in STAGES] if args.smoke else STAGES
    metadata = dict(target="checkpoints/gpt_vfront.ckpt", seed=24690, initialization="random D4", config=config,
                    parameters=sum(p.numel() for p in model.parameters()), source=str(SOURCE), labels=str(LABELS),
                    evaluation=str(EVALUATION), train_files=[r["name"] for r in records], val_files=plan["val_files"],
                    train_body_tokens=sum(len(r["tokens"]) for r in records),
                    train_feature_slots=sum(len(r["tokens"])//2 for r in records),
                    stages=[dict(name=n, steps=s, lr=lr, tv_weight=tv) for n, s, lr, tv in stages],
                    trajectories_per_update=2, train_precision="BF16", evaluation_precision="FP32", smoke=args.smoke,
                    objective="all F raw Target distribution CE; P/EOS hard CE; last stage adds 2 times F policy TV",
                    selection="phase-end validation online Gamma4 throughput, not historical test or real inputs")
    if args.resume:
        previous = json.loads((args.root / "training_plan.json").read_text())
        if previous != metadata:
            raise ValueError("Resume plan differs from the saved run")
    else:
        save(args.root / "training_plan.json", metadata)
        save(args.root / "training_progress.json", dict(status="running", phase="initial_validation", step=0))
        validate_initial(model, config, records, labels, args.root, args.smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = random.Random(24690)
    order, cursor, tokens, features, start_step = [], 0, 0, 0, 0
    history = []
    if args.resume:
        state = torch.load(args.root / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        order, cursor = state["order"], state["cursor"]
        tokens, features, start_step = state["trained_tokens"], state["trained_feature_tokens"], state["step"]
        history = json.loads((args.root / "history.json").read_text())
        del state
    else:
        # Trials consumed RNG; formal training begins from the documented seed.
        torch.manual_seed(24690)
        torch.cuda.manual_seed_all(24690)
    started = time.monotonic()

    def snapshot(step, complete_state=False):
        state = model_state(model, config, step, tokens, features)
        if complete_state:
            state.update(optimizer=optimizer.state_dict(), python_rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(), order=order, cursor=cursor)
            atomic_state(args.root / "last.pt", state)
        else:
            atomic_state(args.root / f"step_{step:04d}.pt", state)

    def metrics(step):
        snapshot(step)
        torch_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        path = args.root / f"step_{step:04d}_metrics.json"
        evaluate(args.root / f"step_{step:04d}.pt", SOURCE, EVALUATION, path, 1 if args.smoke else None)
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        result = json.loads(path.read_text())
        row = dict(step=step, seconds=time.monotonic()-started, trained_tokens=tokens, trained_feature_tokens=features,
                   splits=result["splits"])
        history.append(row)
        save(args.root / "history.json", history)
        print(json.dumps(dict(event="metrics", step=step, val=result["splits"]["val"])), flush=True)

    if not args.resume:
        metrics(0)
    milestones = {0}
    offset = 0
    for index, (_, count, _, _) in enumerate(stages):
        local = {count} if args.smoke else ({1000, 2500, count} if index == 0 else {600, count})
        milestones.update(offset+s for s in local)
        offset += count
    total_steps = offset
    offset = 0
    for index, (name, count, initial_lr, tv) in enumerate(stages):
        for local in range(1, count+1):
            step = offset+local
            if step <= start_step:
                continue
            batch = []
            for _ in range(2):
                if cursor == len(order):
                    order = list(range(len(records)))
                    rng.shuffle(order)
                    cursor = 0
                batch.append(records[order[cursor]])
                cursor += 1
            warmup = min(200, count) if index == 0 else 0
            if local <= warmup:
                rate = initial_lr*local/warmup
            else:
                fraction = (local-warmup-1)/max(1, count-warmup-1)
                rate = initial_lr*(.1+.9*(1+math.cos(math.pi*fraction))/2)
            loss, nt, nf, norm = update(model, optimizer, batch, labels, rate, tv)
            tokens += nt
            features += nf
            if step == 1 or step % 50 == 0 or local == count:
                progress = dict(status="running", phase=name, step=step, expected=total_steps, loss=loss, lr=rate,
                                gradient_norm=norm, trained_tokens=tokens, trained_feature_tokens=features,
                                seconds=time.monotonic()-started)
                save(args.root / "training_progress.json", progress)
                print(json.dumps(dict(event="training", **progress)), flush=True)
            if step in milestones:
                metrics(step)
            if step % 500 == 0 or local == count:
                snapshot(step, True)
        offset += count
    if not args.smoke and (tokens != 35986829 or features != 17992103):
        raise ValueError("Completed training exposure differs from the planned 43 passes")
    save(args.root / "training_progress.json", dict(status="complete", step=total_steps, expected=total_steps,
         trained_tokens=tokens, trained_feature_tokens=features, seconds=time.monotonic()-started))
    print(json.dumps(dict(event="training_complete", step=total_steps, tokens=tokens, features=features)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.root.exists():
            save(args.root / "training_failure.json", dict(error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
