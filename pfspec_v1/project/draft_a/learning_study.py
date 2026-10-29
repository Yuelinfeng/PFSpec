"""Prepare and run a finite study of typed proposal learning curves."""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch

from draft_a.analyze_v2 import read_records
from draft_a.corpus_v2 import ROOT, write_json

STUDY = Path("draft_a/runs/pf_learning_budget_20261013")


def prepare(source_root, study_root):
    output = study_root / "plan.json"
    if output.exists():
        plan = json.loads(output.read_text())
        if Path(plan["source_root"]).resolve() != source_root.resolve():
            raise ValueError("Source directory differs from the saved study")
        return plan
    source = json.loads((source_root / "plan.json").read_text())
    entries = [r for r in source["entries"] if r["split"] == "train"]
    val_entries = [r for r in source["entries"] if r["split"] == "val"]
    records = read_records(source_root, entries)
    assert len(records) == 400 and len(val_entries) == 50
    entry_map = {r["name"]: r for r in entries}
    categories = {}
    for record in records:
        entry = entry_map[record["name"]]
        categories[record["name"]] = (
            str(entry["max_new_tokens"]), str(entry["temperature"]),
            str(entry["top_p"]), "old" if entry["origin"] != "new" else "new",
            str(record["has_eos"]), str(len(record["tokens"]) // 512),
        )
    frequencies = [collections.Counter(v[i] for v in categories.values()) for i in range(6)]
    counts = [collections.Counter() for _ in range(6)]
    remaining = list(categories)
    random.Random(20261013).shuffle(remaining)
    ordered = []
    for number in range(1, 401):
        def score(name):
            values = categories[name]
            total = 0.0
            for index, value in enumerate(values):
                expected = number * frequencies[index][value] / 400
                old = counts[index][value]
                total += ((old + 1 - expected) ** 2 - (old - expected) ** 2) / max(expected, 1)
            return total
        name = min(remaining, key=score)
        remaining.remove(name)
        ordered.append(name)
        for index, value in enumerate(categories[name]):
            counts[index][value] += 1
    subsets = {str(size): ordered[:size] for size in (100, 200, 400)}
    val_files = [r["name"] for r in val_entries]
    assert not set(ordered) & set(val_files)
    train_eval = ordered[:32]
    records_by_name = {r["name"]: r for r in records}
    descriptions = {}
    for size, names in subsets.items():
        descriptions[size] = dict(
            sequences=len(names), body_tokens=sum(len(records_by_name[n]["tokens"]) for n in names),
            natural_eos=sum(records_by_name[n]["has_eos"] for n in names),
            categories={key: dict(collections.Counter(categories[n][i] for n in names))
                        for i, key in enumerate(("requested_length", "temperature", "top_p", "origin", "eos", "length_bin_512"))},
        )
    tokens_one_pass = sum(len(r["tokens"]) + int(r["has_eos"]) for r in records)
    reference_tokens = 25 * tokens_one_pass
    budgets = [round(reference_tokens * factor) for factor in (0.1, 0.2, 0.5, 1, 1.5, 2)]
    seeds = [24681, 24682, 24683]
    jobs = [dict(size=size, seed=seed) for seed in seeds for size in (400, 100, 200)]
    plan = dict(
        version=1, source_root=str(source_root), study_root=str(study_root),
        created_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        subsets=subsets, val_files=val_files, train_eval_files=train_eval,
        subset_descriptions=descriptions, seeds=seeds, jobs=jobs,
        token_budgets=budgets, reference_tokens=reference_tokens,
        reference_explanation="25 complete passes through the 400 training trajectories; equivalent to the previous 5000 two-trajectory updates",
        samples_per_type=32, temperature=0.9, top_p=0.9,
        config="conf/model/gpt_draft_a.yaml", lr=0.0003, weight_decay=0.01,
        gradient_clip=1.0, trajectories_per_update=2,
        train_precision="BF16 autocast", evaluation_precision="FP32",
        budget_interpretation="Evaluate after the first complete update reaching a token budget; record actual tokens and final-update excess",
        selection="Fixed budget endpoints; no validation-driven early stopping",
        subset_selection="Deterministic greedy balance over length, sampling parameters, origin and natural EOS; nested file lists",
        evaluation="Shared validation50 and shared training32; EOS reported separately; teacher-prefix overlap is expected acceptance, not measured online acceptance",
        test_policy="Previously viewed test data is excluded; this study is development evidence",
        limits=["Three initialization seeds are descriptive evidence", "Generated trajectories do not establish physical scene independence", "Raw CE values alone do not establish relative task difficulty"],
    )
    write_json(output, plan)
    return plan


def summarize(study_root, plan):
    rows = []
    for job in plan["jobs"]:
        path = study_root / f"train_n{job['size']}_seed{job['seed']}" / "validation.jsonl"
        if path.exists():
            rows.extend({**json.loads(line), "size": job["size"], "seed": job["seed"]}
                        for line in path.read_text().splitlines() if line.strip())
    write_json(study_root / "curve_summary.json", dict(plan=str(study_root / "plan.json"), records=rows,
        scope="Development curves on shared teacher prefixes; no independent final test or online speed claim"))
    from draft_a.learning_report import build_report
    build_report(study_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--study-root", type=Path, default=STUDY)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.source_root, args.study_root)
    print(json.dumps(dict(stage="plan", subsets=plan["subset_descriptions"], budgets=plan["token_budgets"])), flush=True)
    if args.prepare_only:
        return
    status_path = args.study_root / "study_status.json"
    def status(stage, state="running", **details):
        write_json(status_path, dict(stage=stage, state=state, pid=os.getpid(), epoch=time.time(), **details))
    def execute(stage, arguments):
        status(stage, command=arguments)
        print(json.dumps(dict(stage=stage, command=arguments)), flush=True)
        with (args.study_root / f"{stage}.log").open("a") as log:
            completed = subprocess.run([sys.executable, "-u", "-m", *arguments], stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{stage} exited {completed.returncode}")
    shared = ["--source-root", str(args.source_root), "--study-root", str(args.study_root)]
    try:
        execute("teacher_cache", ["draft_a.learning_eval", "--prepare-cache", *shared])
        from draft_a.learning_eval import evaluate_checkpoint
        reference_file = args.study_root / "previous5000_metrics.json"
        if not reference_file.exists():
            status("previous5000_evaluation")
            write_json(reference_file, evaluate_checkpoint(args.source_root / "training/last.pt", args.source_root, args.study_root))
        completed_jobs = []
        for job in plan["jobs"]:
            name = f"train_n{job['size']}_seed{job['seed']}"
            out = args.study_root / name
            if not (out / "complete.json").exists():
                arguments = ["draft_a.learning_train", *shared, "--size", str(job["size"]), "--seed", str(job["seed"])]
                if (out / "last.pt").exists():
                    arguments.append("--resume")
                execute(name, arguments)
            completed_jobs.append(name)
            summarize(args.study_root, plan)
            status("curves", completed_jobs=completed_jobs, total_jobs=len(plan["jobs"]))
        summarize(args.study_root, plan)
        status("complete", "complete", completed_jobs=completed_jobs)
    except Exception as error:
        status("failed", "failed", error=str(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
