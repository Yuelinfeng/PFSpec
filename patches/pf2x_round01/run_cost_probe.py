#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

LIMIT, MODES, HERE = 3600, ("narrow", "wide"), Path(__file__).resolve().parent
SAVE_RESERVE_SECONDS = 60.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(value, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = ("project_root", "crops", "target_checkpoint", "vqvae_checkpoint", "ranker",
                "feature_checkpoint", "feature_head", "feature_step", "output_root")
    optional = {"collection_budget", "seed_bases", "max_per_source_prefix", "branch_budgets",
                "pair_contexts", "deadline_unix"}
    if not isinstance(raw, dict) or set(raw) - set(required) - optional:
        raise ValueError("config has unknown fields or is not a JSON object")
    missing = [key for key in required if raw.get(key) in (None, "")]
    if missing:
        raise ValueError("missing config fields: " + ", ".join(missing))
    base = path.resolve().parent
    project = Path(raw["project_root"]).expanduser()
    project = (project if project.is_absolute() else base / project).resolve()
    output = Path(raw["output_root"]).expanduser()
    output = (output if output.is_absolute() else base / output).resolve()
    cfg = {"project_root": str(project), "output_root": str(output),
           "feature_step": str(raw["feature_step"])}
    for key in ("crops", "target_checkpoint", "vqvae_checkpoint", "ranker", "feature_checkpoint", "feature_head"):
        value = Path(raw[key]).expanduser()
        cfg[key] = str((value if value.is_absolute() else project / value).resolve())
    cfg["collection_budget"] = _int(raw.get("collection_budget", 64), "collection_budget", 4, 64)
    cfg["max_per_source_prefix"] = _int(raw.get("max_per_source_prefix", 24), "max_per_source_prefix", 1, 24)
    if cfg["collection_budget"] % 2:
        raise ValueError("collection_budget must be even")
    cfg["maximum_contexts"] = 20 * cfg["max_per_source_prefix"]
    seeds = raw.get("seed_bases", [34701, 34702, 34703])
    if isinstance(seeds, str):
        seeds = [int(x.strip()) for x in seeds.split(",") if x.strip()]
    if not isinstance(seeds, list) or not seeds or any(isinstance(x, bool) or not isinstance(x, int) for x in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seed_bases must be unique integers")
    cfg["seed_bases"] = ",".join(map(str, seeds))
    branches, pairs = raw.get("branch_budgets", {m: 48 for m in MODES}), raw.get("pair_contexts", {m: 2 for m in MODES})
    if not isinstance(branches, dict) or set(branches) != set(MODES) or not isinstance(pairs, dict) or set(pairs) != set(MODES):
        raise ValueError("branch_budgets and pair_contexts must define narrow and wide")
    cfg["branch_budgets"] = {m: _int(branches[m], f"{m} branch budget", 1, 96) for m in MODES}
    cfg["pair_contexts"] = {m: _int(pairs[m], f"{m} pair contexts", 1, 4) for m in MODES}
    if sum(cfg["branch_budgets"].values()) > 96 or sum(cfg["pair_contexts"].values()) > 4:
        raise ValueError("combined branch budget must be <=96 and paired contexts <=4")
    if any(cfg["pair_contexts"][m] > cfg["branch_budgets"][m] for m in MODES):
        raise ValueError("paired contexts cannot exceed their mode's branch budget")
    deadline = raw.get("deadline_unix")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError("deadline_unix must be a finite Unix timestamp")
    cfg["deadline_unix"] = float(deadline) if deadline is not None else None
    if output.exists():
        raise FileExistsError(f"output_root already exists: {output}")
    return cfg


def _steps(cfg: dict, deadline_unix: float) -> list[dict]:
    scripts = {n: HERE / n for n in ("collect_pq.py", "measure_target_shapes.py", "analyze_cost_probe.py")}
    if any(not p.is_file() for p in scripts.values()):
        raise FileNotFoundError("the three probe scripts must sit beside run_cost_probe.py")
    def make(name, artifact, path, script, args):
        return dict(name=name, artifact=artifact, path=str(path),
                    command=[sys.executable, str(scripts[script]), *map(str, args)])
    c = Path(cfg["output_root"]) / "collection"
    m = {mode: Path(cfg["output_root"]) / "measurement" / mode for mode in MODES}
    shared = ["--project-root", cfg["project_root"], "--target-checkpoint", cfg["target_checkpoint"],
              "--vqvae-checkpoint", cfg["vqvae_checkpoint"], "--ranker", cfg["ranker"],
              "--feature-checkpoint", cfg["feature_checkpoint"], "--feature-head", cfg["feature_head"],
              "--feature-step", cfg["feature_step"]]
    steps = [make("collect", "collection", c, "collect_pq.py",
                  [*shared[:2], "--crops", cfg["crops"], *shared[2:], "--output-dir", c,
                   "--seed-bases", cfg["seed_bases"], "--budget", cfg["collection_budget"],
                   "--max-per-source-prefix", cfg["max_per_source_prefix"],
                   "--sample-seed", 20261020, "--temperature", 0.9, "--top-p", 0.9,
                   "--deadline-unix", deadline_unix, "--execute"])]
    for mode in MODES:
        steps.append(make(f"measure_{mode}", f"measurement_{mode}", m[mode], "measure_target_shapes.py",
                          [*shared[:2], "--collection-dir", c, *shared[2:], "--output-dir", m[mode],
                           "--q2-mode", mode, "--max-contexts", cfg["branch_budgets"][mode],
                           "--pair-contexts", cfg["pair_contexts"][mode], "--pair-repeats", 8,
                           "--selection-seed", 20261021, "--branch-seed", 20261022 + MODES.index(mode),
                           "--execute"]))
    for mode in MODES:
        output = Path(cfg["output_root"]) / "analysis" / f"{mode}.json"
        steps.append(make(f"analyze_{mode}", f"analysis_{mode}", output, "analyze_cost_probe.py",
                          ["--collection-dir", c, "--measurement-dir", m[mode],
                           "--feature-checkpoint", cfg["feature_checkpoint"], "--feature-step",
                           cfg["feature_step"], "--output", output]))
    for i, step in enumerate(steps, 1):
        step["log"] = str(Path(cfg["output_root"]) / "logs" / f"{i:02d}_{step['name']}.log")
    return steps


def _save_json(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mark_collection_partial(path: Path, reason: str) -> None:
    for name in ("metadata.json", "progress.json"):
        target = path / name
        if not target.is_file():
            continue
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("status") == "running":
            value["status"] = "partial"
            value["stop_reason"] = reason
            _save_json(target, value)


def _run_process(step: dict, cwd: str, deadline: float) -> tuple[int | None, bool]:
    with Path(step["log"]).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(step["command"], cwd=cwd, stdout=log,
                                   stderr=subprocess.STDOUT)
        term_sent = False
        while True:
            returncode = process.poll()
            if returncode is not None:
                return returncode, False
            left = deadline - time.monotonic()
            if (step["name"] == "collect" and not term_sent
                    and left <= SAVE_RESERVE_SECONDS):
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                term_sent = True
            if left <= 0:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                return process.wait(), True
            time.sleep(min(0.2, left))


def _execute_steps(state: dict, steps: list[dict], cwd: str,
                   pipeline: Path, deadline: float) -> int:
    failed = False
    save = lambda: _save_json(pipeline, state)
    save()
    for step in steps:
        left = deadline - time.monotonic()
        step["remaining_budget_seconds"] = max(0.0, left)
        if left <= 0:
            step["status"], step["ended_at"], failed = "not_started_deadline", _now(), True
            state["artifacts"][step["artifact"]] = "not_started"
            save()
            break
        step["status"], step["started_at"] = "running", _now()
        state["artifacts"][step["artifact"]] = "running"
        save()
        try:
            returncode, timed_out = _run_process(step, cwd, deadline)
            step["returncode"] = returncode
            if timed_out:
                step["status"], failed = "timed_out", True
                state["artifacts"][step["artifact"]] = (
                    "partial" if Path(step["path"]).exists() else "incomplete")
                if step["name"] == "collect":
                    _mark_collection_partial(Path(step["path"]), "runner_deadline")
            else:
                failed = returncode != 0
                step["status"] = "failed" if failed else "complete"
                state["artifacts"][step["artifact"]] = (
                    ("partial" if Path(step["path"]).exists() else "failed")
                    if failed else "complete")
                if failed and step["name"] == "collect":
                    _mark_collection_partial(Path(step["path"]), "collector_nonzero")
        except OSError as error:
            step["status"], step["error"], failed = "failed", str(error), True
            state["artifacts"][step["artifact"]] = (
                "partial" if Path(step["path"]).exists() else "failed")
            if step["name"] == "collect":
                _mark_collection_partial(Path(step["path"]), "runner_error")
        step["ended_at"] = _now()
        step["remaining_budget_seconds"] = max(0.0, deadline - time.monotonic())
        state["remaining_budget_seconds"] = step["remaining_budget_seconds"]
        save()
        if failed:
            break
    if failed:
        state["status"] = "partial"
        for step in steps:
            if step["status"] == "pending":
                step["status"] = "not_run_after_failure"
    else:
        state["status"] = "complete"
    state["ended_at"] = _now()
    state["remaining_budget_seconds"] = max(0.0, deadline - time.monotonic())
    save()
    return 0 if state["status"] == "complete" else 1


def _cpu_test() -> int:
    with tempfile.TemporaryDirectory(prefix="pf2x_cost_probe_cpu_") as temporary:
        root = Path(temporary)
        collection = root / "collection"
        marker = root / "downstream_ran"
        collector = HERE / "collect_pq.py"
        command = [sys.executable, str(collector),
                   "--project-root", str(root), "--crops", str(root / "unused.crops"),
                   "--target-checkpoint", str(root / "unused.target"),
                   "--vqvae-checkpoint", str(root / "unused.vq"),
                   "--ranker", str(root / "unused.ranker"),
                   "--feature-checkpoint", str(root / "unused.feature"),
                   "--feature-head", str(root / "unused.head"),
                   "--feature-step", "cpu_test", "--output-dir", str(collection),
                   "--cpu-test"]
        downstream = [sys.executable, "-c",
                      "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('ran')"]
        steps = [
            dict(name="collect", artifact="collection", path=str(collection), command=command,
                 log=str(root / "collect.log"), status="pending", started_at=None,
                 ended_at=None, returncode=None, remaining_budget_seconds=None),
            dict(name="downstream", artifact="downstream", path=str(marker), command=downstream,
                 log=str(root / "downstream.log"), status="pending", started_at=None,
                 ended_at=None, returncode=None, remaining_budget_seconds=None),
        ]
        state = dict(status="running", remaining_budget_seconds=120.0,
                     artifacts={"collection": "not_started", "downstream": "not_started"},
                     steps=steps)
        deadline = time.monotonic() + 120.0
        result = _execute_steps(state, steps, str(root), root / "pipeline.json", deadline)
        runs = [json.loads(line) for line in (collection / "w1_runs.jsonl").read_text().splitlines()]
        contexts = [json.loads(line) for line in (collection / "contexts.jsonl").read_text().splitlines()]
        metadata = json.loads((collection / "metadata.json").read_text())
        if (result != 1 or state["status"] != "partial"
                or steps[0]["returncode"] != 2 or steps[0]["status"] != "failed"
                or state["artifacts"]["collection"] != "partial"
                or steps[1]["status"] != "not_run_after_failure"
                or marker.exists() or len(runs) != 1 or len(contexts) != 1
                or runs[0].get("output_tokens") != [12096, 3, 4]
                or runs[0].get("stats") != {"target_calls": 2, "rounds": 1}
                or contexts[0].get("sample_id") is None or metadata.get("status") != "partial"):
            raise AssertionError("partial collection persistence or pipeline stop failed")
    print(json.dumps(dict(status="cpu_test_passed", complete_w1_rows=1,
                          retained_contexts=1, downstream_started=False), ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded PF cost probe pipeline.")
    parser.add_argument("--config", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--show-plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--cpu-test", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_test:
        return _cpu_test()
    if args.config is None:
        parser.error("--config is required for --show-plan and --execute")
    cfg = _config(args.config)
    if args.show_plan:
        plan_start = time.time()
        plan_deadline = min(cfg["deadline_unix"] if cfg["deadline_unix"] is not None
                            else plan_start + LIMIT, plan_start + LIMIT)
        steps = _steps(cfg, plan_deadline)
        remaining = (min(LIMIT, max(0.0, cfg["deadline_unix"] - time.time()))
                     if cfg["deadline_unix"] is not None else LIMIT)
        print(json.dumps(dict(status="plan_only", wall_budget_seconds=remaining,
                              configured_deadline_unix=cfg["deadline_unix"],
                              collection_stop_reserve_seconds=SAVE_RESERVE_SECONDS,
                              maximum_contexts=cfg["maximum_contexts"],
                              branch_total=sum(cfg["branch_budgets"].values()),
                              paired_contexts=sum(cfg["pair_contexts"].values()),
                              paired_target_forwards=sum(cfg["pair_contexts"].values()) * 18,
                              paired_kv_truncations=sum(cfg["pair_contexts"].values()) * 19,
                              config=cfg, commands=steps), ensure_ascii=False, indent=2))
        return 0
    for key in ("project_root", "crops", "target_checkpoint", "vqvae_checkpoint", "ranker", "feature_checkpoint", "feature_head"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(cfg[key])
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    start_wall, start_mono = time.time(), time.monotonic()
    deadline_unix = min(cfg["deadline_unix"] if cfg["deadline_unix"] is not None
                        else start_wall + LIMIT, start_wall + LIMIT)
    deadline = start_mono + deadline_unix - start_wall
    steps = _steps(cfg, deadline_unix)
    for step in steps:
        step.update(status="pending", started_at=None, ended_at=None, returncode=None,
                    remaining_budget_seconds=None)
    remaining = max(0.0, deadline_unix - start_wall)
    state = dict(status="running", wall_budget_seconds=remaining, started_at=_now(),
                 deadline_unix=deadline_unix,
                 deadline_at=datetime.fromtimestamp(deadline_unix, timezone.utc).isoformat(),
                 ended_at=None, remaining_budget_seconds=remaining, config=cfg,
                 artifacts={s["artifact"]: "not_started" for s in steps}, steps=steps)
    pipeline = root / "pipeline.json"
    result = _execute_steps(state, steps, cfg["project_root"], pipeline, deadline)
    print(json.dumps(dict(status=state["status"], pipeline=str(pipeline),
                          remaining_budget_seconds=state["remaining_budget_seconds"]), ensure_ascii=False))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
