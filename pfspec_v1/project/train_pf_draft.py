"""Collect real-scene Target states and train matched Joint/Feature P/F Drafts."""
import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from speculative.pf_draft import DraftConfig, PFDraft
from speculative.pf_engine import target_cache, validate_tokens


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def load_core():
    from draft_a.corpus_v2 import load_target
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wrapper = load_target()
    return wrapper, wrapper.transformer.float().eval().requires_grad_(False)


@torch.inference_mode()
def collect(core, crops, output, chunk=256):
    from speculative.pf_draft import TokenSpec
    spec = TokenSpec.from_core(core)
    records = torch.load(crops, map_location="cpu", weights_only=False)
    scenes = sorted({r["scene"] for r in records})
    if len(scenes) < 3:
        raise ValueError("need at least three independent scene sources")
    random.Random(20261019).shuffle(scenes)
    validation = set(scenes[:max(1, len(scenes)//5)])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, record in enumerate(records):
        body = record["tokens"].long()
        tokens = torch.cat((torch.tensor([spec.eos]), body,
                            torch.tensor([spec.eos]) if record.get("has_eos") else torch.empty(0, dtype=torch.long)))
        validate_tokens(tokens.tolist(), spec, core.sparse_position_rank_map)
        if len(tokens) > core.config.sequence_len:
            raise ValueError("real sequence exceeds Target context")
        cache = target_cache(core, len(tokens))
        parts = []
        device = next(core.parameters()).device
        for start in range(0, len(tokens), chunk):
            ids = tokens[start:start+chunk][None].to(device)
            kinds = ["position" if (start+j) % 2 == 0 else "feature" for j in range(ids.shape[1])]
            _, hidden = core(ids, kv_cache=cache, logit_types=kinds, return_hidden=True)
            parts.append(hidden[0].cpu())
        coords = core._build_sparse_position_indices(tokens[None].to(device))[0].cpu()
        path = output/f"scene_{number:03d}.pt"
        split = "val" if record["scene"] in validation else "train"
        torch.save(dict(tokens=tokens, hidden=torch.cat(parts), coordinates=coords,
                        scene=record["scene"], split=split), path)
        rows.append(dict(path=path.name, scene=record["scene"], split=split,
                         tokens=len(tokens), has_eos=bool(record.get("has_eos"))))
        print(json.dumps(dict(collect=number+1, total=len(records), tokens=len(tokens))), flush=True)
    manifest = dict(source=str(crops), rows=rows, train_scenes=sorted(set(scenes)-validation),
                    val_scenes=sorted(validation), target="checkpoints/gpt_vfront.ckpt",
                    vqvae="checkpoints/vqvae_vfront.ckpt", state="final RMSNorm output",
                    dtype="FP32", coordinate_order="xyz plus token slot")
    save_json(output/"manifest.json", manifest)
    return manifest


def windows(data_dir, split, length):
    manifest = json.loads((data_dir/"manifest.json").read_text())
    result = []
    for entry in manifest["rows"]:
        if entry["split"] != split:
            continue
        row = torch.load(data_dir/entry["path"], map_location="cpu", weights_only=True)
        # Every row aligns h_t, x_(t+1), h_(t+1), x_(t+2).
        for start in range(0, len(row["tokens"])-2, length):
            stop = min(start+length, len(row["tokens"])-2)
            if stop <= start:
                continue
            result.append(dict(source=row["hidden"][start:stop][None],
                               tokens=row["tokens"][start+1:stop+1][None],
                               target=row["hidden"][start+1:stop+1][None],
                               coordinates=row["coordinates"][start+1:stop+1][None],
                               next_tokens=row["tokens"][start+2:stop+2][None]))
    if not result:
        raise ValueError("empty training or validation split")
    return result


def train(core, args, mode):
    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    config = DraftConfig(mode=mode, width=args.width, layers=args.layers)
    draft = PFDraft(core, config).cuda()
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=0.01)
    train_rows, val_rows = windows(args.data, "train", args.window), windows(args.data, "val", args.window)
    out = args.output/mode
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(draft=draft.description(), steps=args.steps, seed=args.seed, lr=args.lr,
                    window=args.window, hidden_weight=args.hidden_weight, token_weight=args.token_weight,
                    data=str(args.data), train_windows=len(train_rows), val_windows=len(val_rows),
                    supervision="true-token CE and hidden regression", precision="FP32")
    save_json(out/"training_config.json", metadata)
    best, history, seen = float("inf"), [], 0
    start_time = time.perf_counter()

    def loss(row):
        return draft.losses(**{k: v.cuda() for k, v in row.items()},
                            ranks=core.sparse_position_rank_map,
                            hidden_weight=args.hidden_weight, token_weight=args.token_weight)

    for step in range(1, args.steps+1):
        draft.train()
        batch = randomizer.choice(train_rows)
        optimizer.zero_grad(set_to_none=True)
        values = loss(batch)
        if not torch.isfinite(values["loss"]):
            raise ArithmeticError("nonfinite Draft training loss")
        values["loss"].backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optimizer.step()
        seen += batch["tokens"].numel()
        if step % args.validate_every == 0 or step == args.steps:
            draft.eval()
            totals, count = {}, 0
            with torch.no_grad():
                for row in val_rows:
                    n = row["tokens"].numel()
                    for key, val in loss(row).items():
                        totals[key] = totals.get(key, 0.0)+float(val)*n
                    count += n
            metrics = {k: v/count for k, v in totals.items()}
            entry = dict(step=step, train={k: float(v.detach()) for k, v in values.items()},
                         validation=metrics, cumulative_tokens=seen, seconds=time.perf_counter()-start_time)
            history.append(entry)
            print(json.dumps(dict(mode=mode, **entry)), flush=True)
            checkpoint = dict(config=asdict(config), state_dict=draft.state_dict(),
                              step=step, validation=metrics, cumulative_tokens=seen,
                              metadata=metadata)
            torch.save(checkpoint, out/"last.pt")
            if metrics["loss"] < best:
                best = metrics["loss"]
                torch.save(checkpoint, out/"selected.pt")
            save_json(out/"history.json", history)
    if args.rollout_steps:
        # Each candidate prefix is queried again; original scene suffixes are never reused.
        from joblib import load
        from bench_fast_hgb import FastHistRanker
        from draft_a.feature_target_v2 import RANKER
        from speculative.pf_engine import PositionRanker, decode, policy
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
        ranker_model = FastHistRanker(load(RANKER))
        ranks = core.sparse_position_rank_map.cpu().tolist()
        manifest = json.loads((args.data/"manifest.json").read_text())
        originals = [torch.load(args.data/r["path"], map_location="cpu", weights_only=True)
                     for r in manifest["rows"] if r["split"] == "train"]
        rollout_history = []
        selected = torch.load(out/"selected.pt", map_location="cpu", weights_only=False)
        draft.load_state_dict(selected["state_dict"])
        optimizer = torch.optim.AdamW(draft.parameters(), lr=args.lr*0.1, weight_decay=0.01)
        for index in range(args.rollout_steps):
            original = randomizer.choice(originals)
            anchor = min(128, len(original["tokens"])-4)
            anchor -= anchor % 2
            prefix = original["tokens"][:anchor+1].tolist()
            draft.eval()
            stats, _ = decode(core, draft, prefix, 3, mode=mode, gamma=3,
                              ranker=PositionRanker(ranker_model, ranks),
                              seed=args.seed+index, trace=True)
            candidates = stats["trace"][0]["proposals"]
            all_tokens = torch.tensor([prefix+candidates], device="cuda")
            coords = core._build_sparse_position_indices(all_tokens)
            with torch.no_grad():
                tc = target_cache(core, len(prefix)+len(candidates))
                _, past_hidden = core(all_tokens[:, :len(prefix)], kv_cache=tc, return_hidden=True)
                true_logits, true_hidden = core(all_tokens[:, len(prefix):], kv_cache=tc, return_hidden=True)
            draft.train()
            optimizer.zero_grad(set_to_none=True)
            sources = [past_hidden[:, :-1]]
            current = past_hidden[:, -1:]
            hidden_losses, soft_losses = {"P": [], "F": []}, {"P": [], "F": []}
            for j, token in enumerate(candidates):
                if token == draft.spec.eos:
                    break
                sources.append(current)
                stop = len(prefix)+j+1
                pred = draft(torch.cat(sources, 1), all_tokens[:, 1:stop], coords[:, 1:stop])[:, -1:]
                current = pred
                consumed_kind = "P" if token < draft.spec.positions else "F"
                hidden_losses[consumed_kind].append(torch.nn.functional.smooth_l1_loss(pred, true_hidden[:, j:j+1]))
                next_kind = "F" if consumed_kind == "P" else "P"
                if mode == "feature" and next_kind == "P":
                    continue
                local_logits = draft.logits(pred[0, 0], next_kind)
                teacher = policy(true_logits[0, j], 0.9, 0.9)
                if next_kind == "F":
                    teacher = teacher[draft.spec.offset:draft.spec.offset+draft.spec.features]
                else:
                    ids = torch.cat((torch.arange(draft.spec.positions, device="cuda"),
                                     torch.tensor([draft.spec.eos], device="cuda")))
                    teacher = teacher[ids]
                    previous = int(all_tokens[0, stop-2])
                    local_logits[:draft.spec.positions] = local_logits[:draft.spec.positions].masked_fill(
                        core.sparse_position_rank_map <= core.sparse_position_rank_map[previous], -torch.inf)
                log_q = torch.log_softmax(local_logits/0.9, -1)
                soft_losses[next_kind].append(-(teacher*log_q.masked_fill(teacher == 0, 0)).sum())
            zero = draft.hidden_out.weight.sum()*0
            hidden_sum = sum((torch.stack(v).mean() if v else zero) for v in hidden_losses.values())
            soft_sum = sum((torch.stack(v).mean() if v else zero) for v in soft_losses.values())
            value = args.hidden_weight*hidden_sum + args.token_weight*soft_sum
            if not torch.isfinite(value):
                raise ArithmeticError("nonfinite rollout loss")
            value.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optimizer.step()
            entry = dict(step=index+1, loss=float(value.detach()), candidate_tokens=len(candidates),
                         source=original["scene"], proposal=candidates)
            rollout_history.append(entry)
            print(json.dumps(dict(mode=mode, rollout=entry)), flush=True)
        draft.eval()
        totals, count = {}, 0
        with torch.no_grad():
            for row in val_rows:
                n = row["tokens"].numel()
                for key, val in loss(row).items():
                    totals[key] = totals.get(key, 0.0)+float(val)*n
                count += n
        metrics = {k: v/count for k, v in totals.items()}
        checkpoint = dict(config=asdict(config), state_dict=draft.state_dict(),
                          step=args.steps, rollout_steps=args.rollout_steps, validation=metrics,
                          rollout_from_step=selected["step"], rollout_lr=args.lr*0.1,
                          cumulative_tokens=seen, metadata=metadata)
        torch.save(checkpoint, out/"rollout_last.pt")
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(checkpoint, out/"selected.pt")
        save_json(out/"rollout_history.json", dict(records=rollout_history, validation=metrics,
                  selected=metrics["loss"] <= best,
                  scope="actual candidate prefixes; recurrent predicted hidden; Target soft-label CE"))
    return dict(mode=mode, best_validation=best, steps=args.steps, cumulative_tokens=seen,
                rollout_steps=args.rollout_steps, seconds=time.perf_counter()-start_time, output=str(out))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect", type=Path, help="existing real-scene crop file")
    parser.add_argument("--mode", choices=["feature", "joint", "both"], default="both")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--token-weight", type=float, default=0.1)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--feature-v3", choices=["diagnose", "prepare", "train"])
    parser.add_argument("--init", type=Path)
    parser.add_argument("--kl-weight", type=float, default=1.)
    parser.add_argument("--ce-weight", type=float, default=.1)
    parser.add_argument("--v3-stage", choices=["single", "rollout"], default="single")
    parser.add_argument("--v3-batch", type=int, default=8)
    parser.add_argument("--v3-refresh", type=int, default=200)
    parser.add_argument("--v3-pool", type=int, default=128)
    parser.add_argument("--v3-f2-weight", type=float, default=1.)
    parser.add_argument("--v3-rollout-fraction", type=float, default=.5)
    parser.add_argument("--v3-anchors", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--v3-crops", type=Path, default=Path("/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt"))
    parser.add_argument("--v3-scenes", type=int, default=10)
    parser.add_argument("--v3-repeats", type=int, default=4)
    parser.add_argument("--v3-label", default="feature_v2")
    parser.add_argument("--v3-reference", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.window, args.validate_every) < 1:
        parser.error("steps, window and validation interval must be positive")
    wrapper, core = load_core()
    if args.feature_v3:
        if min(args.v3_anchors) < 2 or any(x % 2 for x in args.v3_anchors):
            parser.error("Feature conditioning anchors must end after a P/F pair")
        if min(args.v3_batch, args.v3_refresh, args.v3_pool, args.v3_scenes, args.v3_repeats) < 1:
            parser.error("Feature sample counts and refresh interval must be positive")
        if min(args.kl_weight, args.ce_weight, args.hidden_weight) < 0:
            parser.error("Feature loss weights must be nonnegative")
        if args.v3_f2_weight <= 0 or not 0 <= args.v3_rollout_fraction <= 1:
            parser.error("F2 weight must be positive and rollout fraction within [0, 1]")
        from speculative.pf_distill import run_feature_v3
        run_feature_v3(core, args)
        return
    if args.collect:
        collect(core, args.collect, args.data)
    result = [train(core, args, mode) for mode in (["feature", "joint"] if args.mode == "both" else [args.mode])]
    save_json(args.output/"training_summary.json", result)


if __name__ == "__main__":
    main()
