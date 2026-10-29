"""Continue full-distribution Feature training from a saved Draft checkpoint."""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import torch

POLICY = dict(
    temperature=0.9,
    top_p=0.9,
    support="4096 allowed Feature ids",
    slots="all Feature prediction slots",
    nucleus="descending logits; cumulative > top_p; shift right once; renormalize",
)


def policy_probs(logits, temperature=0.9, top_p=0.9):
    """Keep the first item crossing top_p and differentiate retained probabilities."""
    scaled = logits.float() / temperature
    with torch.no_grad():
        ordered, indices = scaled.detach().sort(dim=-1, descending=True)
        remove = ordered.softmax(dim=-1).cumsum(dim=-1) > top_p
        remove = torch.cat((torch.zeros_like(remove[..., :1]), remove[..., :-1]), dim=-1)
        remove = torch.zeros_like(remove).scatter(-1, indices, remove)
    return scaled.masked_fill(remove, -torch.inf).softmax(dim=-1)


def feature_policy_tv(feature_logits, target_probabilities):
    """Return one TV value per Feature slot from raw T=1 Target labels."""
    with torch.no_grad():
        p = policy_probs(target_probabilities.detach().float().log(),
                         POLICY["temperature"], POLICY["top_p"])
    q = policy_probs(feature_logits, POLICY["temperature"], POLICY["top_p"])
    return (p - q).abs().sum(dim=-1) * 0.5


def continuation_loss(logits, labels, probabilities, acceptance_weight=0.0):
    """Add summed Feature TV with the original all-token denominator."""
    from draft_a.feature_distill import FEATURE_END, FEATURE_START, feature_loss

    loss = feature_loss(logits, labels, probabilities)
    if acceptance_weight == 0.0:
        return loss
    feature_mask = (labels >= FEATURE_START) & (labels < FEATURE_END)
    selected = logits[feature_mask, FEATURE_START:FEATURE_END]
    tv_sum = feature_policy_tv(selected, probabilities).sum()
    return loss + acceptance_weight * tv_sum / labels.numel()


def self_test():
    """Exercise the actual torch objective on CPU without loading a model."""
    from draft_a.feature_distill import FEATURE_END, FEATURE_START, feature_loss
    try:
        from draft_a.eval_overlap import sampling_probs
    except ModuleNotFoundError as error:
        if error.name != "draft_a.eval_overlap":
            raise
        from eval_overlap import sampling_probs

    rng = torch.Generator(device="cpu").manual_seed(24686)
    width = FEATURE_END - FEATURE_START
    raw = torch.randn((4, width), generator=rng)
    targets = torch.randn((4, width), generator=rng).softmax(-1)
    cases = [*raw, *targets.log(), torch.zeros(width),
             torch.tensor([0.6, 0.25, 0.1, 0.05]).log(),
             torch.tensor([0.5, 0.25, 0.125, 0.125]).log(),
             torch.tensor([0.999, 0.0005, 0.0003, 0.0002]).log()]
    max_policy_delta = 0.0
    for row in cases:
        full = torch.full((FEATURE_END + 2,), -torch.inf)
        full[FEATURE_START:FEATURE_START + row.numel()] = row
        expected = sampling_probs(full, None, POLICY["temperature"], POLICY["top_p"])
        actual = policy_probs(full)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        if not torch.equal(actual > 0, expected > 0):
            raise AssertionError("Sampling support differs")
        max_policy_delta = max(max_policy_delta, float((actual - expected).abs().max()))

    # Exact cumulative equality retains the next item before removing later items.
    boundary_logits = torch.tensor([0.5, 0.25, 0.2, 0.05]).log()
    cutoff = float(boundary_logits.softmax(-1).cumsum(-1)[1])
    boundary = policy_probs(boundary_logits, 1.0, cutoff)
    torch.testing.assert_close(boundary, torch.tensor([0.5 / 0.95, 0.25 / 0.95, 0.2 / 0.95, 0.0]))
    crossing = policy_probs(torch.tensor([0.6, 0.25, 0.1, 0.05]).log(), 1.0, 0.9)
    torch.testing.assert_close(crossing, torch.tensor([0.6 / 0.95, 0.25 / 0.95, 0.1 / 0.95, 0.0]))

    raw.requires_grad_(True)
    targets.requires_grad_(True)
    tv = feature_policy_tv(raw, targets)
    p = policy_probs(targets.detach().log())
    q = policy_probs(raw)
    torch.testing.assert_close(tv, 1.0 - torch.minimum(p, q).sum(-1), atol=2e-6, rtol=2e-6)
    gradient, target_gradient = torch.autograd.grad(tv.sum(), (raw, targets), allow_unused=True)
    if not bool(torch.isfinite(gradient).all()) or not bool((gradient != 0).any()):
        raise AssertionError("TV gradient must be finite and nonzero")
    if target_gradient is not None:
        raise AssertionError("Target policy conversion must have no gradient")
    identical = feature_policy_tv(targets.detach().log(), targets)
    torch.testing.assert_close(identical, torch.zeros_like(identical), atol=2e-6, rtol=0)

    labels = torch.tensor([7, FEATURE_START + 2, 23, FEATURE_START + 17, FEATURE_END])
    logits = torch.randn((labels.numel(), FEATURE_END + 2), generator=rng, requires_grad=True)
    probabilities = targets.detach()[:2]
    original = feature_loss(logits, labels, probabilities)
    zero_weight = continuation_loss(logits, labels, probabilities, 0.0)
    if not torch.equal(original, zero_weight):
        raise AssertionError("Zero-weight loss differs from the original")
    original_gradient, = torch.autograd.grad(original, logits, retain_graph=True)
    zero_gradient, = torch.autograd.grad(zero_weight, logits, retain_graph=True)
    if not torch.equal(original_gradient, zero_gradient):
        raise AssertionError("Zero-weight gradient differs from the original")
    weight = 0.7
    weighted = continuation_loss(logits, labels, probabilities, weight)
    feature_mask = (labels >= FEATURE_START) & (labels < FEATURE_END)
    expected_tv = feature_policy_tv(logits[feature_mask, FEATURE_START:FEATURE_END], probabilities)
    torch.testing.assert_close(weighted, original + weight * expected_tv.sum() / labels.numel())
    weighted_gradient, = torch.autograd.grad(weighted, logits)
    if not torch.equal(weighted_gradient[~feature_mask], original_gradient[~feature_mask]):
        raise AssertionError("Position or EOS supervision differs")
    print(json.dumps(dict(
        success=True, device="cpu", sampling_cases=len(cases),
        max_policy_difference=max_policy_delta, zero_weight_exact=True,
        tv_equals_one_minus_overlap=True, finite_nonzero_q_gradient=True,
        target_conversion_without_gradient=True, identical_tv_max=float(identical.max()),
        all_feature_slots_and_all_token_denominator=True, position_eos_unchanged=True,
    )), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=24686)
    parser.add_argument("--acceptance-weight", type=float, default=0.0)
    parser.add_argument("--min-position", type=float, default=0.97)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("checkpoint", "output", "label_root", "evaluation_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for training")
    if not math.isfinite(args.acceptance_weight) or args.acceptance_weight < 0:
        parser.error("--acceptance-weight must be finite and nonnegative")
    if not math.isfinite(args.min_position) or not 0 <= args.min_position <= 1:
        parser.error("--min-position must be between 0 and 1")

    from draft_a.feature_reuse import make_model, evaluate
    from draft_a.feature_distill import _read_training, load_training_labels
    from draft_a.learning_train import write_json

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.steps <= 0 or args.lr <= 0:
        raise ValueError("steps and lr must be positive")

    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    plan = json.loads((args.label_root / "plan.json").read_text())
    records = _read_training(plan)
    probabilities = load_training_labels(args.label_root, records)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = initial["config"]
    start_step = int(initial.get("step", 0))
    model = make_model(config)
    model.load_state_dict(initial["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    del initial

    order, cursor = [], 0
    trained_tokens = 0
    feature_tokens = 0
    started = time.monotonic()
    history = []
    best = -1.0
    milestones = sorted(set([0, min(200, args.steps), min(600, args.steps), args.steps]))
    metadata = dict(
        initial_checkpoint=str(args.checkpoint),
        start_step=start_step,
        steps=args.steps,
        lr=args.lr,
        schedule="cosine to 0.1 times initial lr",
        seed=args.seed,
        label_root=str(args.label_root),
        source_root=plan["source_root"],
        train_files=[r["name"] for r in records],
        objective=("all F Target distributions, P and EOS hard labels"
                   if args.acceptance_weight == 0.0 else
                   "all F Target distribution CE plus acceptance_weight * summed F policy TV; "
                   "P and EOS hard CE; all terms divided by all target tokens"),
        acceptance_weight=args.acceptance_weight,
        min_position=args.min_position,
        policy=dict(POLICY),
        feature_temperature=1.0,
        train_precision="BF16",
        evaluation_precision="FP32",
        trajectories_per_update=2,
        parameters=sum(p.numel() for p in model.parameters()),
    )
    write_json(args.output / "plan.json", metadata)

    def measure(local_step):
        nonlocal best
        global_step = start_step + local_step
        path = args.output / f"step_{global_step:04d}.pt"
        torch.save(
            dict(
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                config=config,
                step=global_step,
                trained_tokens=trained_tokens,
                trained_feature_tokens=feature_tokens,
                **metadata,
            ),
            path,
        )
        result_path = args.output / f"step_{global_step:04d}_metrics.json"
        evaluate(path, Path(plan["source_root"]), args.evaluation_root, result_path, None)
        result = json.loads(result_path.read_text())
        feature = result["splits"]["val"]["feature"]["pooled_overlap"]
        position = result["splits"]["val"]["position"]["pooled_overlap"]
        row = dict(
            step=global_step,
            local_step=local_step,
            feature=feature,
            position=position,
            trained_tokens=trained_tokens,
            trained_feature_tokens=feature_tokens,
            seconds=time.monotonic() - started,
            checkpoint=str(path),
        )
        history.append(row)
        if position >= args.min_position and feature > best:
            best = feature
            write_json(args.output / "selected.json", row)
        write_json(args.output / "history.json", history)
        print(json.dumps(dict(event="measurement", **row)), flush=True)
        model.train()

    measure(0)
    model.train()
    for local_step in range(1, args.steps + 1):
        batch = []
        for _ in range(2):
            if cursor == len(order):
                order = list(range(len(records)))
                rng.shuffle(order)
                cursor = 0
            batch.append(records[order[cursor]])
            cursor += 1
        tokens = sum(len(record["seq"]) - 1 for record in batch)
        rate = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * (local_step - 1) / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for record in batch:
            sequence = record["seq"].cuda().unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=sequence[:, :-1])[0]
                loss = continuation_loss(
                    logits,
                    sequence[0, 1:],
                    probabilities[record["name"]].cuda(),
                    args.acceptance_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            weight = (len(record["seq"]) - 1) / tokens
            (loss * weight).backward()
            total += float(loss.detach()) * weight
            del sequence, logits, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite training gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trained_tokens += tokens
        feature_tokens += sum(len(record["tokens"]) // 2 for record in batch)
        if local_step == 1 or local_step % 50 == 0:
            write_json(
                args.output / "progress.json",
                dict(
                    status="training",
                    step=start_step + local_step,
                    local_step=local_step,
                    loss=total,
                    lr=rate,
                    trained_tokens=trained_tokens,
                    trained_feature_tokens=feature_tokens,
                    seconds=time.monotonic() - started,
                ),
            )
            print(
                json.dumps(
                    dict(
                        event="progress",
                        step=start_step + local_step,
                        local_step=local_step,
                        loss=total,
                        lr=rate,
                        trained_tokens=trained_tokens,
                        trained_feature_tokens=feature_tokens,
                    )
                ),
                flush=True,
            )
        if local_step in milestones:
            measure(local_step)
    selected_path = args.output / "selected.json"
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else None
    write_json(
        args.output / "progress.json",
        dict(
            status="complete",
            step=start_step + args.steps,
            trained_tokens=trained_tokens,
            trained_feature_tokens=feature_tokens,
            seconds=time.monotonic() - started,
            selected=selected,
        ),
    )
    del model, optimizer, probabilities
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
