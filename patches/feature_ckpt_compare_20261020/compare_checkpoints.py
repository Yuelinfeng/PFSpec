"""Compare the original PF decoder with exact 12k and 48k Feature weights."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time


SEED_BASES = (48701, 48702, 48703)
PREFIXES = (64, 256)
BUDGET = 512
EXPECTED_RUNS = 120
COUNTER_FIELDS = (
    "target_calls", "draft_calls", "ranker_calls", "target_input_tokens",
    "draft_input_tokens", "rounds", "proposed", "verified", "accepted",
    "position_proposed", "position_verified", "position_accepted",
    "feature_proposed", "feature_verified", "feature_accepted",
    "position_fallbacks", "generated_tokens",
)
METHODS = ((12000, "checkpoint_12000"), (48000, "checkpoint_48000"))


class DeadlineReached(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def path_from_project(value, project):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def check_stop(deadline, stop_signal, stage):
    if stop_signal["name"]:
        raise InterruptedError("stop requested by " + stop_signal["name"] + " during " + stage)
    if time.monotonic() >= deadline:
        raise DeadlineReached("shared deadline reached during " + stage)


def crop_source(record):
    return record.get("scene") or Path(record["path"]).parents[1].name


def select_records(raw, validate):
    selected = {}
    for record in raw:
        validate(record)
        source = crop_source(record)
        if len(record["tokens"]) >= 256 and source not in selected:
            selected[source] = dict(record, scene=source)
    records = [selected[source] for source in sorted(selected)]
    if len(records) != 10:
        raise ValueError("Expected ten development sources with a crop of at least 256 tokens; got "
                         + str(len(records)))
    return records


def make_plan(records):
    plan = []
    for source_index, record in enumerate(records):
        for prefix_index, prefix in enumerate(PREFIXES):
            for seed_index, seed_base in enumerate(SEED_BASES):
                pair_id = f"{source_index:02d}-{prefix:03d}-{seed_base}"
                order = METHODS if (source_index + prefix_index + seed_index) % 2 == 0 else METHODS[::-1]
                seed = seed_base + int(record["crop_seed"]) * 1009 + prefix * 17
                for order_index, (step, method) in enumerate(order):
                    plan.append(dict(
                        pair_id=pair_id, source=record["scene"], source_index=source_index,
                        crop_seed=int(record["crop_seed"]), prefix=prefix,
                        prefix_index=prefix_index, seed_index=seed_index,
                        seed_base=seed_base, seed=seed, budget=BUDGET,
                        order_index=order_index, step=step, method=method,
                    ))
    if len(plan) != EXPECTED_RUNS:
        raise RuntimeError("Internal plan size differs from 120")
    return plan


def checkpoint_layout(torch, path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or "step" not in value or "model_state" not in value:
        raise ValueError("Feature checkpoint lacks step or model_state: " + str(path))
    config = value.get("model_config")
    state = value["model_state"]
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Feature checkpoint has unexpected model fields: " + str(path))
    layout = {key: tuple(tensor.shape) for key, tensor in state.items()}
    result = dict(step=int(value["step"]), model_config=config,
                  state_layout=layout, tensor_count=len(layout))
    del value
    return result


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(rows):
    if not rows:
        return dict(runs=0, generated_tokens=0, outer_wall_seconds=0.0,
                    pooled_tokens_per_second=None, feature_acceptance=None,
                    position_acceptance=None, mean_length=None,
                    eos_count=0, eos_rate=None, output_length_counts={})
    tokens = sum(int(row["generated_tokens"]) for row in rows)
    seconds = sum(float(row["outer_wall_seconds"]) for row in rows)
    f_verified = sum(int(row["feature_verified"]) for row in rows)
    f_accepted = sum(int(row["feature_accepted"]) for row in rows)
    p_verified = sum(int(row["position_verified"]) for row in rows)
    p_accepted = sum(int(row["position_accepted"]) for row in rows)
    lengths = {}
    for row in rows:
        key = str(int(row["generated_tokens"]))
        lengths[key] = lengths.get(key, 0) + 1
    eos_count = sum(bool(row["eos"]) for row in rows)
    return dict(
        runs=len(rows), generated_tokens=tokens, outer_wall_seconds=seconds,
        pooled_tokens_per_second=(tokens / seconds if seconds else None),
        feature_acceptance=(f_accepted / f_verified if f_verified else None),
        position_acceptance=(p_accepted / p_verified if p_verified else None),
        mean_length=tokens / len(rows), eos_count=eos_count,
        eos_rate=eos_count / len(rows), output_length_counts=lengths,
        feature_accepted=f_accepted, feature_verified=f_verified,
        position_accepted=p_accepted, position_verified=p_verified,
    )


def build_summary(rows, plan, status):
    by_step = {step: [row for row in rows if row["step"] == step]
               for step, _ in METHODS}
    paired = {}
    row_by_pair = {}
    for row in rows:
        row_by_pair.setdefault(row["pair_id"], {})[row["step"]] = row
    for pair_id, pair in row_by_pair.items():
        if 12000 in pair and 48000 in pair:
            paired[pair_id] = pair

    by_source = {}
    source_names = sorted({entry["source"] for entry in plan})
    per_source = {}
    for source in source_names:
        source_pairs = [pair for pair in paired.values()
                        if pair[12000]["source"] == source]
        a = [pair[12000] for pair in source_pairs]
        b = [pair[48000] for pair in source_pairs]
        source_a, source_b = describe(a), describe(b)
        ratio = (source_b["pooled_tokens_per_second"] / source_a["pooled_tokens_per_second"]
                 if source_a["pooled_tokens_per_second"] and source_b["pooled_tokens_per_second"] else None)
        by_source[source] = {"step_12000": source_a, "step_48000": source_b,
                             "speed_ratio_48000_over_12000": ratio,
                             "complete_pairs": len(source_pairs)}
        if ratio is not None and ratio > 0:
            per_source[source] = ratio

    geo_mean = (math.exp(sum(math.log(value) for value in per_source.values()) / len(per_source))
                if per_source else None)
    bootstrap = []
    if len(per_source) == 10:
        names = sorted(per_source)
        rng = random.Random(487031)
        for _ in range(10000):
            sampled = [per_source[rng.choice(names)] for _ in names]
            bootstrap.append(math.exp(sum(math.log(value) for value in sampled) / len(sampled)))
    full_pairs = [pair for pair in paired.values()
                  if pair[12000]["generated_tokens"] == BUDGET
                  and pair[48000]["generated_tokens"] == BUDGET]
    full_rows = {step: [pair[step] for pair in full_pairs] for step, _ in METHODS}
    return dict(
        status=status,
        run_count=len(rows), expected_run_count=EXPECTED_RUNS,
        paired_context_count=len(paired), expected_pair_count=60,
        by_checkpoint={str(step): describe(values) for step, values in by_step.items()},
        by_source=by_source,
        speed_48000_over_12000=dict(
            source_equal_geometric_mean=geo_mean,
            source_bootstrap_replicates=(len(bootstrap) if bootstrap else 0),
            source_bootstrap_95_ci=([quantile(bootstrap, .025), quantile(bootstrap, .975)]
                                    if bootstrap else None),
            per_source=per_source,
            basis="per-source pooled generated tokens divided by synchronized outer wall; source means weighted equally",
        ),
        common_full_512=dict(
            paired_contexts=len(full_pairs),
            step_12000=describe(full_rows[12000]),
            step_48000=describe(full_rows[48000]),
            interpretation="descriptive subset where both outputs reached 512 tokens",
        ),
        length_note="When generated lengths differ, total wall time alone is not a speed comparison; use pooled tokens/second and the common 512-token subset.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Remote GaussianGPT project directory")
    parser.add_argument("--crops", required=True, help="Existing development crops file")
    parser.add_argument("--checkpoint-12000", required=True, help="Exact step-12000 Feature weights")
    parser.add_argument("--checkpoint-48000", required=True, help="Exact step-48000 Feature weights")
    parser.add_argument("--out", required=True, help="New output directory; it must not exist")
    parser.add_argument("--deadline-seconds", type=float, default=1800,
                        help="One shared wall-clock limit including loading, warmup, and replay")
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    crops = path_from_project(args.crops, project)
    checkpoints = {
        12000: path_from_project(args.checkpoint_12000, project),
        48000: path_from_project(args.checkpoint_48000, project),
    }
    out = Path(args.out).expanduser().resolve()
    if args.deadline_seconds <= 0:
        parser.error("--deadline-seconds must be positive")
    if out.exists():
        parser.error("--out must name a new directory: " + str(out))
    out.mkdir(parents=True)

    started_wall = time.time()
    started_mono = time.monotonic()
    deadline = started_mono + args.deadline_seconds
    stop_signal = {"name": None}
    previous_handlers = {}

    def request_stop(signum, _frame):
        stop_signal["name"] = signal.Signals(signum).name

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, request_stop)

    pipeline = dict(
        status="running", project=str(project), crops=str(crops),
        checkpoints={str(step): str(path) for step, path in checkpoints.items()},
        output=str(out), started_at_unix=started_wall,
        deadline_unix=started_wall + args.deadline_seconds,
        deadline_seconds=args.deadline_seconds, completed_runs=0,
        expected_runs=EXPECTED_RUNS, current_stage="startup",
    )
    write_json(out / "pipeline.json", pipeline)
    write_json(out / "progress.json", dict(status="running", completed_runs=0,
                                            expected_runs=EXPECTED_RUNS, stage="startup"))
    rows = []
    plan = []
    status = "failed"

    def update(stage, current_status="running"):
        pipeline.update(status=current_status, current_stage=stage,
                        completed_runs=len(rows), elapsed_seconds=time.monotonic() - started_mono,
                        remaining_seconds=max(0.0, deadline - time.monotonic()))
        write_json(out / "pipeline.json", pipeline)
        write_json(out / "progress.json", dict(status=current_status,
                                                completed_runs=len(rows),
                                                expected_runs=EXPECTED_RUNS,
                                                stage=stage,
                                                remaining_seconds=max(0.0, deadline - time.monotonic())))

    try:
        check_stop(deadline, stop_signal, "imports")
        sys.path.insert(0, str(project))
        os.chdir(project)
        import torch
        from joblib import load
        from threadpoolctl import threadpool_limits
        from bench_fast_hgb import FastHistRanker
        from bench_gaussiangpt_pf_block2_cow import load_feature_model
        from draft_a.feature_target_v2 import HEAD, RANKER
        from draft_a.corpus_v2 import EOS, load_target, validate
        from draft_a.learning_eval import _fp32_runtime
        from draft_a.pf_feature_unified import MLPAdapter, decode

        if not project.is_dir() or not crops.is_file():
            raise FileNotFoundError("Project directory or crop file is missing")
        for path in checkpoints.values():
            if not path.is_file():
                raise FileNotFoundError("Feature checkpoint is missing: " + str(path))

        check_stop(deadline, stop_signal, "CPU checkpoint reading")
        layouts = {step: checkpoint_layout(torch, path) for step, path in checkpoints.items()}
        for step, layout in layouts.items():
            if layout["step"] != step:
                raise ValueError("Expected step " + str(step) + " weights, found " + str(layout["step"]))
        if (layouts[12000]["model_config"] != layouts[48000]["model_config"]
                or layouts[12000]["state_layout"] != layouts[48000]["state_layout"]):
            raise ValueError("The two Feature checkpoints have different model structures")
        checkpoint_info = {str(step): dict(step=item["step"], tensor_count=item["tensor_count"])
                           for step, item in layouts.items()}
        pipeline["checkpoint_cpu_read"] = checkpoint_info
        update("load_crop_sources")

        torch.set_num_threads(6)
        raw = torch.load(crops, map_location="cpu", weights_only=True)
        records = select_records(raw, validate)
        plan = make_plan(records)
        write_json(out / "plan.json", dict(
            crops=str(crops), sources=[dict(scene=r["scene"], crop_seed=int(r["crop_seed"]),
                                            path=str(r["path"])) for r in records],
            prefixes=list(PREFIXES), seed_bases=list(SEED_BASES), budget=BUDGET,
            checkpoints={str(step): str(path) for step, path in checkpoints.items()},
            planned_runs=EXPECTED_RUNS,
            order_rule="order alternates by source, prefix, and seed indices; each pair uses the same prompt and seed",
            timing="synchronized outer wall around original unprofiled decode, including prefill, proposal, verification, recovery, and sampling",
        ))
        update("load_original_target_and_ranker")

        check_stop(deadline, stop_signal, "load original Target and ranker")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        else:
            raise RuntimeError("CUDA is required for the requested online comparison")
        with torch.inference_mode(), _fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
            if (torch.get_float32_matmul_precision() != "highest"
                    or torch.backends.cuda.matmul.allow_tf32
                    or torch.backends.cudnn.allow_tf32):
                raise RuntimeError("FP32 highest precision with TF32 disabled was not established")
            target = load_target().float().eval().requires_grad_(False)
            ranker = FastHistRanker(load(RANKER))
            ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
            adapters = {}
            for step, _method in METHODS:
                check_stop(deadline, stop_signal, "load Feature step " + str(step))
                model, temperature, top_p = load_feature_model(
                    checkpoints[step], HEAD, target.transformer, torch.device("cuda"), 8000, 12096)
                if (temperature, top_p) != (.9, .9):
                    raise ValueError("Feature checkpoint policy metadata must be temperature=.9 and top_p=.9")
                adapters[step] = MLPAdapter(model)
            sample = records[0]
            warm_prompt = torch.cat([torch.tensor([EOS]), sample["tokens"][:64]]).tolist()

            def decode_once(step, prompt, budget, seed):
                check_stop(deadline, stop_signal, "decode step " + str(step))
                torch.cuda.synchronize()
                outer_started = time.perf_counter()
                result, tokens = decode(target.transformer, adapters[step], ranker,
                                        ranks, prompt, budget, seed)
                torch.cuda.synchronize()
                outer_wall = time.perf_counter() - outer_started
                result = dict(result)
                result["output_tokens"] = [int(token) for token in tokens]
                result["outer_wall_seconds"] = outer_wall
                return result

            verification = dict(
                warmups=[], same_checkpoint_replays=[], cross_checkpoint_replay_equal=None)
            update("warmup_32_tokens")
            for step, _method in METHODS:
                check_stop(deadline, stop_signal, "32-token warmup")
                row = decode_once(step, warm_prompt, 32, 48600)
                verification["warmups"].append(dict(step=step, generated_tokens=row["generated_tokens"],
                                                    target_calls=row["target_calls"],
                                                    feature_calls=row["draft_calls"],
                                                    ranker_calls=row["ranker_calls"]))
            update("same_seed_16_token_replay")
            replay_rows = {}
            replay_seed = 48616
            for step, _method in METHODS:
                first = decode_once(step, warm_prompt, 16, replay_seed)
                second = decode_once(step, warm_prompt, 16, replay_seed)
                equal = (first["output_tokens"] == second["output_tokens"]
                         and all(first.get(key) == second.get(key) for key in COUNTER_FIELDS))
                if not equal:
                    raise RuntimeError("Same-checkpoint same-seed replay changed output or counters at step "
                                       + str(step))
                replay_rows[step] = first
                verification["same_checkpoint_replays"].append(dict(
                    step=step, seed=replay_seed, output_tokens=first["output_tokens"],
                    counters={key: first.get(key) for key in COUNTER_FIELDS}, equal=True))
            verification["cross_checkpoint_replay_equal"] = dict(
                output_tokens=(replay_rows[12000]["output_tokens"] == replay_rows[48000]["output_tokens"]),
                counters=all(replay_rows[12000].get(key) == replay_rows[48000].get(key)
                             for key in COUNTER_FIELDS),
                interpretation="reported for context; checkpoint weights can change sampled output",
            )
            write_json(out / "verification.json", verification)
            pipeline["hardware_software"] = dict(
                python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, cuda_device=device_name,
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                autocast="disabled in original decoder", target_precision="FP32",
                head=str(HEAD), ranker=str(RANKER), decoder="draft_a.pf_feature_unified.decode",
            )
            pipeline["verification"] = dict(warmups_per_checkpoint=1,
                                             warmup_budget=32,
                                             repeated_same_seed_budget=16,
                                             replay_equal_for_each_checkpoint=True,
                                             cross_checkpoint_replay=verification["cross_checkpoint_replay_equal"])
            update("matrix")

            record_by_source = {record["scene"]: record for record in records}
            plan_by_pair = {}
            for entry in plan:
                plan_by_pair.setdefault(entry["pair_id"], []).append(entry)
            for pair_id, entries in plan_by_pair.items():
                for entry in entries:
                    check_stop(deadline, stop_signal, "matrix before " + pair_id)
                    record = record_by_source[entry["source"]]
                    prompt = torch.cat([torch.tensor([EOS]), record["tokens"][:entry["prefix"]]]).tolist()
                    row = decode_once(entry["step"], prompt, BUDGET, entry["seed"])
                    row.update(
                        pair_id=pair_id, source=entry["source"], source_index=entry["source_index"],
                        crop_seed=entry["crop_seed"], prefix=entry["prefix"],
                        seed_index=entry["seed_index"], seed_base=entry["seed_base"],
                        seed=entry["seed"], budget=BUDGET, step=entry["step"],
                        method=entry["method"], order_index=entry["order_index"],
                    )
                    append_jsonl(out / "records.jsonl", row)
                    rows.append(row)
                    update("matrix")
                    print(json.dumps(dict(pair_id=pair_id, source=entry["source"],
                                          prefix=entry["prefix"], step=entry["step"],
                                          generated_tokens=row["generated_tokens"],
                                          outer_wall_seconds=row["outer_wall_seconds"],
                                          complete_runs=len(rows)), ensure_ascii=False), flush=True)
                    check_stop(deadline, stop_signal, "matrix after " + pair_id)

        check_stop(deadline, stop_signal, "summary")
        status = "complete" if len(rows) == EXPECTED_RUNS else "partial"
        write_json(out / "summary.json", build_summary(rows, plan, status))
        check_stop(deadline, stop_signal, "finalization")
        update(status, status)
    except DeadlineReached as exc:
        status = "budget_overrun" if len(rows) == EXPECTED_RUNS else "partial_deadline"
        pipeline["stop_reason"] = str(exc)
        write_json(out / "summary.json", build_summary(rows, plan, status))
        update(status, status)
    except InterruptedError as exc:
        status = "partial_signal"
        pipeline["stop_reason"] = str(exc)
        write_json(out / "summary.json", build_summary(rows, plan, status))
        update(status, status)
    except Exception as exc:
        status = "failed" if not rows else "partial_failed"
        pipeline["error"] = dict(type=type(exc).__name__, message=str(exc))
        write_json(out / "summary.json", build_summary(rows, plan, status))
        update(status, status)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if status != "complete":
        raise SystemExit("Comparison ended with status " + status + "; see " + str(out / "pipeline.json"))


if __name__ == "__main__":
    main()
