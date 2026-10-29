"""Train complete trajectories against shared cumulative token budgets."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from draft_a.analyze_v2 import read_records
from draft_a.train_v2 import make_model, sequence


SOURCE_ROOT = Path("draft_a/runs/rollout500_20261013")
STUDY_ROOT = Path("draft_a/runs/pf_learning_budget_20261013")


def write_json(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def save_state(path, state):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_history(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            phase = row["phase"]
            if phase in rows:
                raise ValueError(f"Duplicate phase in {path}: {phase}")
            json.dumps(row, allow_nan=False)
            rows[phase] = row
    return rows


def append_phase(path, rows, row):
    phase = row["phase"]
    if phase in rows:
        if rows[phase] != row:
            raise ValueError(f"Phase already has different results: {phase}")
        return
    encoded = json.dumps(row, allow_nan=False) + "\n"
    previous = path.read_text() if path.exists() else ""
    if previous and not previous.endswith("\n"):
        previous += "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(previous + encoded)
    temporary.replace(path)
    rows[phase] = row


def load_inputs(source_root, study_root, size, seed):
    plan = json.loads((study_root / "plan.json").read_text())
    if Path(plan["source_root"]).resolve() != source_root.resolve():
        raise ValueError("Source directory differs from the study plan")
    if seed not in plan["seeds"]:
        raise ValueError("Seed is absent from the study plan")
    expected = dict(lr=3e-4, weight_decay=0.01, gradient_clip=1.0,
                    trajectories_per_update=2)
    for key, value in expected.items():
        if plan[key] != value:
            raise ValueError(f"Unexpected training setting: {key}")
    budgets = plan["token_budgets"]
    if (not budgets or any(type(n) is not int or n <= 0 for n in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError("Token budgets must be strictly increasing positive integers")
    source = json.loads((source_root / "plan.json").read_text())
    entries = {entry["name"]: entry for entry in source["entries"]}
    if len(entries) != len(source["entries"]):
        raise ValueError("Source plan repeats a filename")
    subsets = plan["subsets"]
    for count in (100, 200, 400):
        names = subsets[str(count)]
        if len(names) != count or len(set(names)) != count:
            raise ValueError(f"Wrong subset size: {count}")
        if any(name not in entries or entries[name]["split"] != "train" for name in names):
            raise ValueError("Every training file must belong to the original train split")
    if not (set(subsets["100"]) <= set(subsets["200"]) <= set(subsets["400"])):
        raise ValueError("Training subsets must be nested")
    train_files = list(subsets[str(size)])
    val_files = list(plan["val_files"])
    train_eval_files = list(plan["train_eval_files"])
    original_val = {entry["name"] for entry in source["entries"] if entry["split"] == "val"}
    if len(val_files) != 50 or len(set(val_files)) != 50 or set(val_files) != original_val:
        raise ValueError("Evaluation must use the original 50 validation files")
    if len(train_eval_files) != 32 or len(set(train_eval_files)) != 32:
        raise ValueError("Training evaluation needs 32 distinct files")
    if not set(train_eval_files) <= set(subsets["100"]):
        raise ValueError("Training evaluation files must belong to the smallest subset")
    if set(train_files) & set(val_files):
        raise ValueError("Training and validation files overlap")
    for name in train_files + val_files:
        if Path(name).name != name:
            raise ValueError("Corpus filenames must be simple names")
    records = read_records(source_root, [entries[name] for name in train_files])
    if [record["name"] for record in records] != train_files:
        raise ValueError("Loaded training order differs from the plan")
    for record in records:
        record["seq"] = sequence(record)
    cfg_value = plan["config"]
    cfg = OmegaConf.create(cfg_value) if isinstance(cfg_value, dict) else OmegaConf.load(cfg_value)
    if (cfg.gpt_size.n_layer, cfg.gpt_size.n_embd, cfg.gpt_size.n_head) != (8, 512, 8):
        raise ValueError("Draft architecture must be 8 layers, width 512, 8 heads")
    if any(not 0 < len(record["seq"]) - 1 <= cfg.n_ctx for record in records):
        raise ValueError("A complete trajectory exceeds the model context")
    return plan, cfg, records, train_files, val_files, train_eval_files


def evaluate_phase(checkpoint, source_root, study_root, model, max_records=None):
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    try:
        from draft_a.learning_eval import evaluate_checkpoint

        options = {} if max_records is None else {"max_records": max_records}
        metrics = evaluate_checkpoint(checkpoint, source_root, study_root, **options)
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Evaluation must return a nonempty metrics dictionary")
        json.dumps(metrics, allow_nan=False)
        return metrics
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)
        model.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--size", type=int, choices=(100, 200, 400), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--eval-max-records", type=int)
    parser.add_argument("--objective", choices=("ce", "feature_kd"), default="ce")
    parser.add_argument("--evaluation-root", type=Path)
    args = parser.parse_args()
    if not args.smoke and (args.max_updates is not None or args.eval_max_records is not None):
        parser.error("Update and evaluation limits are available only with --smoke")
    if args.smoke:
        args.max_updates = 2 if args.max_updates is None else args.max_updates
        args.eval_max_records = 1 if args.eval_max_records is None else args.eval_max_records
        if args.max_updates <= 0 or args.eval_max_records <= 0:
            parser.error("Smoke limits must be positive")

    try:
        _run(args)
    except RuntimeError as error:
        output_root = args.study_root / "smoke" if args.smoke else args.study_root
        out = output_root / f"train_n{args.size}_seed{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        progress_path = out / "progress.json"
        value = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        value.update(status="failed", error=str(error), size=args.size,
                     seed=args.seed, smoke=args.smoke)
        write_json(progress_path, value)
        raise


def _run(args):

    started = time.monotonic()
    wall_offset = 0.0
    plan, cfg, train, train_files, val_files, train_eval_files = load_inputs(
        args.source_root, args.study_root, args.size, args.seed)
    evaluation_root = args.evaluation_root or args.study_root
    teacher_labels = None
    if args.objective == "feature_kd":
        from draft_a.feature_distill import load_training_labels, feature_loss
        if plan.get("objective") != args.objective or args.size != 400:
            raise ValueError("Feature distillation requires its 400-trajectory plan")
        teacher_labels = load_training_labels(args.study_root, train)
    budgets = list(plan["token_budgets"])
    output_root = args.study_root / "smoke" if args.smoke else args.study_root
    out = output_root / f"train_n{args.size}_seed{args.seed}"
    last_path = out / "last.pt"
    history_path = out / "validation.jsonl"
    run_options = dict(smoke=args.smoke, max_updates=args.max_updates,
                       eval_max_records=args.eval_max_records)
    if args.objective != "ce":
        run_options.update(objective=args.objective, evaluation_root=str(evaluation_root))
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(last_path)
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run already has output; use --resume: {out}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires CUDA with BF16 support")
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    model = make_model(cfg)
    if args.objective == "feature_kd":
        reference = torch.load(evaluation_root / f"train_n400_seed{args.seed}" / "phase_00.pt",
                               map_location="cpu", weights_only=True)
        if (reference["config"] != config or reference["train_files"] != train_files
                or reference["seed"] != args.seed):
            raise ValueError("The CE initial state uses different inputs")
        if any(not torch.equal(value.detach().cpu(), reference["model"][name])
               for name, value in model.state_dict().items()):
            raise ValueError("Initialization differs from the paired CE run")
        del reference
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = random.Random(args.seed + 1)
    order, cursor = [], 0
    step, trained_tokens, next_phase = 0, 0, 0
    trained_feature_tokens = 0
    last_update_tokens = 0
    training_seconds = 0.0
    pending = None
    rows = read_history(history_path)
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        required = dict(config=config, seed=args.seed, size=args.size,
                        train_files=train_files, val_files=val_files,
                        train_eval_files=train_eval_files, token_budgets=budgets,
                        run_options=run_options, lr=3e-4, weight_decay=0.01,
                        gradient_clip=1.0, trajectories_per_update=2)
        for key, value in required.items():
            if state.get(key) != value:
                raise ValueError(f"Resume setting differs from the saved run: {key}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step, trained_tokens = state["step"], state["trained_tokens"]
        if args.objective != "ce":
            trained_feature_tokens = state["trained_feature_tokens"]
        next_phase = state["next_phase"]
        last_update_tokens = state["last_update_tokens"]
        training_seconds = state["training_seconds"]
        order, cursor = state["order"], state["cursor"]
        if order and sorted(order) != list(range(len(train))):
            raise ValueError("Saved training order is invalid")
        if not 0 <= cursor <= len(order):
            raise ValueError("Saved training cursor is invalid")
        rng.setstate(state["python_rng"])
        random.setstate(state["python_global_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        pending = state["pending_phase"]
        preceding = set(range(pending["phase"]))
        if set(rows) not in (preceding, preceding | {pending["phase"]}):
            raise ValueError("Saved metrics do not form a complete sequence of phases")
        if any(row["trained_tokens"] > trained_tokens or phase > pending["phase"]
               for phase, row in rows.items()):
            raise ValueError("Saved metrics extend beyond the resume state")
        wall_offset = max([state["wall_seconds"]] + [row["wall_seconds"] for row in rows.values()])
        del state
    out.mkdir(parents=True, exist_ok=True)
    model.train()

    def wall_seconds():
        return wall_offset + time.monotonic() - started

    def progress(status, **details):
        value = dict(status=status, size=args.size, seed=args.seed, step=step,
                     trained_tokens=trained_tokens, target_tokens=trained_tokens,
                     training_seconds=training_seconds, wall_seconds=wall_seconds(),
                     max_token_budget=budgets[-1], next_phase=next_phase,
                     smoke=args.smoke, **details)
        if args.objective != "ce":
            value.update(objective=args.objective, trained_feature_tokens=trained_feature_tokens)
        write_json(out / "progress.json", value)
        return value

    def finish_pending(point):
        phase = point["phase"]
        checkpoint = out / f"phase_{phase:02d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if phase in rows:
            for key, value in point.items():
                if rows[phase].get(key) != value:
                    raise ValueError(f"Saved metrics differ from phase {phase}: {key}")
            metrics_path = out / f"phase_{phase:02d}_metrics.json"
            if not metrics_path.exists():
                write_json(metrics_path, rows[phase])
            progress("running", phase=phase, metrics=rows[phase]["metrics"])
            return
        progress("evaluating", phase=phase)
        eval_started = time.monotonic()
        metrics = evaluate_phase(checkpoint, args.source_root, evaluation_root,
                                 model, args.eval_max_records)
        row = dict(point, metrics=metrics, size=args.size, seed=args.seed,
                   training_seconds=training_seconds, wall_seconds=wall_seconds(),
                   evaluation_seconds=time.monotonic() - eval_started, smoke=args.smoke)
        append_phase(history_path, rows, row)
        write_json(out / f"phase_{phase:02d}_metrics.json", row)
        progress("running", phase=phase, metrics=metrics)
        print(json.dumps(dict(event="evaluation", **row), allow_nan=False), flush=True)

    def save_phase(phase, budget, smoke_endpoint=False):
        nonlocal next_phase, pending
        overshoot = 0 if budget is None else trained_tokens - budget
        if overshoot < 0 or (budget and overshoot >= last_update_tokens):
            raise ValueError("Evaluation point is not the first complete update reaching its budget")
        next_phase = phase + 1
        pending = dict(phase=phase, step=step, trained_tokens=trained_tokens,
                       target_tokens=trained_tokens, budget_target_tokens=budget,
                       overshoot=overshoot, last_update_tokens=last_update_tokens,
                       smoke_endpoint=smoke_endpoint)
        if args.objective != "ce":
            pending.update(trained_feature_tokens=trained_feature_tokens)
        point = dict(model=model.state_dict(), config=config, seed=args.seed, size=args.size,
                     train_files=train_files, val_files=val_files,
                     train_eval_files=train_eval_files, source_root=str(args.source_root),
                     token_budgets=budgets, **pending)
        if args.objective != "ce":
            point.update(objective=args.objective, distillation_temperature=1.0,
                         evaluation_root=str(evaluation_root))
        save_state(out / f"phase_{phase:02d}.pt", point)
        full = dict(point, optimizer=optimizer.state_dict(), lr=3e-4, weight_decay=0.01,
                    gradient_clip=1.0, trajectories_per_update=2, run_options=run_options,
                    python_rng=rng.getstate(), python_global_rng=random.getstate(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    order=list(order), cursor=cursor, next_phase=next_phase,
                    pending_phase=pending, training_seconds=training_seconds,
                    wall_seconds=wall_seconds())
        save_state(last_path, full)
        del full, point
        finish_pending(pending)

    print(json.dumps(dict(event="start", size=args.size, seed=args.seed,
                          parameters=sum(p.numel() for p in model.parameters()),
                          budgets=budgets, step=step, trained_tokens=trained_tokens,
                          effective_batch=2, train_precision="BF16", smoke=args.smoke)), flush=True)
    if pending is not None:
        finish_pending(pending)
    else:
        save_phase(0, 0)

    while True:
        while next_phase <= len(budgets) and trained_tokens >= budgets[next_phase - 1]:
            save_phase(next_phase, budgets[next_phase - 1])
        if trained_tokens >= budgets[-1]:
            break
        if args.smoke and step >= args.max_updates:
            if not any(row["step"] == step for row in rows.values()):
                save_phase(next_phase, None, smoke_endpoint=True)
            break
        torch.cuda.synchronize()
        update_started = time.monotonic()
        batch = []
        for _ in range(2):
            if cursor >= len(order):
                order = list(range(len(train)))
                rng.shuffle(order)
                cursor = 0
            batch.append(train[order[cursor]])
            cursor += 1
        num_tokens = sum(len(record["seq"]) - 1 for record in batch)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for record in batch:
            seq = record["seq"].unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.objective == "ce":
                    loss = model(input_ids=seq, labels=seq)
                else:
                    logits = model(input_ids=seq[:, :-1])[0]
                    probabilities = teacher_labels[record["name"]].cuda()
                    loss = feature_loss(logits, seq[0, 1:], probabilities)
                    del logits, probabilities
            if not bool(torch.isfinite(loss)):
                progress("failed", reason="nonfinite_loss", attempted_step=step + 1)
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            weight = (len(record["seq"]) - 1) / num_tokens
            (loss * weight).backward()
            total_loss += loss.item() * weight
            del loss, seq
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm = float(norm.item())
        if not math.isfinite(grad_norm):
            progress("failed", reason="nonfinite_gradient", attempted_step=step + 1)
            raise RuntimeError(f"Nonfinite gradient at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        training_seconds += time.monotonic() - update_started
        step += 1
        trained_tokens += num_tokens
        trained_feature_tokens += sum(len(record["tokens"]) // 2 for record in batch)
        last_update_tokens = num_tokens
        if step == 1 or step % 10 == 0 or args.smoke:
            loss_value = {"train_ce" if args.objective == "ce" else "train_objective_loss": total_loss}
            print(json.dumps(dict(event="update", step=step, **loss_value,
                                  objective=args.objective,
                                  finiteLoss=math.isfinite(total_loss), grad_norm=grad_norm,
                                  update_target_tokens=num_tokens, target_tokens=trained_tokens,
                                  training_seconds=training_seconds, wall_seconds=wall_seconds()),
                             allow_nan=False), flush=True)
            progress("running", **loss_value, finiteLoss=math.isfinite(total_loss),
                     grad_norm=grad_norm, last_update_tokens=last_update_tokens)

    completed = progress("smoke_complete" if args.smoke else "complete",
                         completed_phases=sorted(rows), token_budget_reached=trained_tokens >= budgets[-1])
    write_json(out / "complete.json", completed)
    print(json.dumps(dict(event="complete", **completed), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
