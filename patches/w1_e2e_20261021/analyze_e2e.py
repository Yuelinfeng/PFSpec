#!/usr/bin/env python3
"""Recompute paired request-time and token-throughput ratios from saved rows."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gm(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def interval(values, seed=402101):
    rng = random.Random(seed)
    values = list(values)
    samples = sorted(gm(rng.choices(values, k=len(values))) for _ in range(10000))
    return [samples[249], samples[9749]]


def analyze_multi(root):
    plan = json.loads((root / "plan.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    rows = read_rows(root / "records.jsonl")
    methods = plan["methods"]
    groups = defaultdict(dict)
    for row in rows:
        key = (row["cohort"], row["pair_id"])
        if row["method"] in groups[key]:
            raise ValueError("Repeated method for one input")
        groups[key][row["method"]] = row
        if (not all(row[k] for k in ("sequence_validated", "finite_gaussians", "finite_rgb_alpha"))
                or not math.isclose(sum(row["stages"].values()), row["total_seconds"],
                                    rel_tol=1e-12, abs_tol=1e-8)):
            raise ValueError("Invalid output or inconsistent stage time")
    complete = {k:v for k,v in groups.items() if set(v) == set(methods)}
    summary = dict(status=progress["status"], runs=len(rows),
        complete_pairs=len(complete), incomplete_pairs=[list(k) for k in groups if k not in complete],
        expected_runs=progress["expected"], methods=methods,
        uncertainty="Source-resampled descriptive interval, 10000 draws, seed402101.",
        scope="Only complete input groups enter paired statistics; request time and token throughput reported separately.",
        cohorts={})
    for cohort in sorted({k[0] for k in complete}):
        selected = [v for k,v in complete.items() if k[0] == cohort]
        totals = {}
        for method in methods:
            arm = [v[method] for v in selected]
            totals[method] = dict(runs=len(arm),
                generated_tokens=sum(r["generated_tokens"] for r in arm),
                eos_count=sum(r["eos"] for r in arm),
                total_seconds=sum(r["total_seconds"] for r in arm),
                stages_seconds={k:sum(r["stages"][k] for r in arm) for k in arm[0]["stages"]},
                runtime_creation_rows=sum(r["runtime_created"] for r in arm),
                recovery_calls=sum(r.get("counters", {}).get("recovery_calls", 0) for r in arm),
                rescue_hits=sum(r.get("stats", {}).get("position_rescue_hits", 0) for r in arm))
        comparisons = {}
        for method, reference in [(m,"target") for m in methods if m != "target"] + [("rescue","w1")]:
            if method not in methods or reference not in methods:
                continue
            scopes = {}
            for scope in ("all", "all_methods_full512"):
                pairs = selected if scope == "all" else [v for v in selected
                    if all(r["generated_tokens"] == 512 for r in v.values())]
                metrics = defaultdict(lambda: defaultdict(list))
                for v in pairs:
                    a, b = v[reference], v[method]
                    fields = dict(request_latency=a["total_seconds"]/b["total_seconds"],
                        generation_latency=a["stages"]["generation"]/b["stages"]["generation"],
                        generation_throughput=(b["generated_tokens"]/b["stages"]["generation"])/
                            (a["generated_tokens"]/a["stages"]["generation"]))
                    for field, value in fields.items():
                        metrics[field][a["source"]].append(value)
                result = {}
                for field, by_source in metrics.items():
                    values = {k:gm(v) for k,v in by_source.items()}
                    result[field] = dict(source_equal_gm=gm(values.values()),
                        source_interval95=interval(values.values()), by_source=values)
                scopes[scope] = dict(pairs=len(pairs), sources=len({v[reference]["source"] for v in pairs}),
                    metrics=result)
            comparisons[method+"/"+reference] = scopes
        parity = [v for v in selected if "w1" in v and "rescue" in v]
        summary["cohorts"][cohort] = dict(pairs=len(selected), sources=len({v["target"]["source"] for v in selected}),
            arms=totals, comparisons=comparisons,
            w1_rescue_parity=dict(pairs=len(parity),
                outputs_equal=sum(v["w1"]["output_tokens"] == v["rescue"]["output_tokens"] for v in parity),
                recovery_difference_equals_hits=sum(
                    v["w1"].get("counters",{}).get("recovery_calls",0)-
                    v["rescue"].get("counters",{}).get("recovery_calls",0) ==
                    v["rescue"].get("stats",{}).get("position_rescue_hits",0) for v in parity)))
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def analyze(root, previous=None, natural_previous=None):
    plan = json.loads((root / "plan.json").read_text())
    if set(plan.get("methods", [])) != {"target", "w1"}:
        return analyze_multi(root)
    progress = json.loads((root / "progress.json").read_text())
    rows = read_rows(root / "records.jsonl")
    if progress["status"] != "complete" or len(rows) != progress["expected"]:
        raise ValueError("The measurement is incomplete")
    groups = defaultdict(dict)
    for row in rows:
        key = (row["cohort"], row["pair_id"])
        if row["method"] in groups[key]:
            raise ValueError("Repeated arm for an input")
        groups[key][row["method"]] = row
        if not all(row[k] for k in ("sequence_validated", "finite_gaussians", "finite_rgb_alpha")):
            raise ValueError("Invalid output record")
        if not math.isclose(sum(row["stages"].values()), row["total_seconds"], rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("Stage time does not sum to total")
        if row["saved_views"] != 6 or row["alpha_coverage"] <= 0 or row["quality"]["gaussians"] <= 0:
            raise ValueError("Output views or Gaussians are empty")
    for arms in groups.values():
        if set(arms) != {"target", "w1"}:
            raise ValueError("Input lacks one arm")
        a, b = arms["target"], arms["w1"]
        for key in ("source", "seed", "prefix"):
            if a[key] != b[key]:
                raise ValueError("Arm input differs")
    summary = dict(status="complete", runs=len(rows), pairs=len(groups),
                   timing_validation="Every stage sum matches the saved outer wall time",
                   uncertainty="Source-resampled descriptive interval, 10000 draws, seed 402101; three sources per cohort give limited precision.", cohorts={})
    for cohort in sorted({r["cohort"] for r in rows}):
        selected = [r for r in rows if r["cohort"] == cohort]
        pairs = [(arms["target"], arms["w1"]) for (name, _), arms in groups.items() if name == cohort]
        ratios = defaultdict(lambda: defaultdict(list))
        pair_rows = []
        for a, b in pairs:
            fields = dict(total=a["total_seconds"]/b["total_seconds"],
                          compute=a["compute_seconds"]/b["compute_seconds"],
                          generation_latency=a["stages"]["generation"]/b["stages"]["generation"],
                          generation_throughput=(b["generated_tokens"]/b["stages"]["generation"])/(a["generated_tokens"]/a["stages"]["generation"]))
            for field, value in fields.items():
                ratios[field][a["source"]].append(value)
            pair_rows.append(dict(pair_id=a["pair_id"], source=a["source"], **fields,
                                  target_seconds=a["total_seconds"], w1_seconds=b["total_seconds"],
                                  target_tokens=a["generated_tokens"], w1_tokens=b["generated_tokens"]))
        arms = {}
        for method in ("target", "w1"):
            arm = [r for r in selected if r["method"] == method]
            stages = {key:sum(r["stages"][key] for r in arm) for key in arm[0]["stages"]}
            seconds = sum(r["total_seconds"] for r in arm)
            arms[method] = dict(runs=len(arm), total_seconds=seconds, mean_seconds=seconds/len(arm),
                                compute_seconds=sum(r["compute_seconds"] for r in arm), stages_seconds=stages,
                                stages_mean_seconds={k:v/len(arm) for k,v in stages.items()},
                                generated_tokens=sum(r["generated_tokens"] for r in arm),
                                generated_token_range=[min(r["generated_tokens"] for r in arm),max(r["generated_tokens"] for r in arm)],
                                eos_count=sum(r["eos"] for r in arm),
                                gaussian_count_range=[min(r["quality"]["gaussians"] for r in arm),max(r["quality"]["gaussians"] for r in arm)],
                                alpha_coverage_range=[min(r["alpha_coverage"] for r in arm),max(r["alpha_coverage"] for r in arm)],
                                runtime_creation_rows=sum(r["runtime_created"] for r in arm))
        metrics = {}
        for field, source_values in ratios.items():
            source_values = {source:gm(values) for source,values in source_values.items()}
            metrics[field] = dict(source_equal_gm=gm(source_values.values()),
                                  source_interval95=interval(source_values.values()), by_source=source_values)
        summary["cohorts"][cohort] = dict(pairs=len(pairs), sources=len(ratios["total"]), arms=arms, metrics=metrics,
                                          pooled_total_ratio=arms["target"]["total_seconds"]/arms["w1"]["total_seconds"], pairs_detail=pair_rows)
    comparisons = {}
    plan = json.loads((root / "plan.json").read_text())
    for label, old_path, cohort, names in (
        ("fixed512_vs_main_run02", previous, "fixed512", {"target":"target", "w1":"tree_w1"}),
        ("natural_vs_previous", natural_previous, "natural_eos", {"target":"original_target", "w1":"candidate_pf"}),
    ):
        if old_path is None:
            continue
        old_prompts = json.loads((old_path.parent / "prompts.json").read_text())["contexts"]
        new_prompts = next(c["prompts"] for c in plan["cohorts"] if c["name"] == cohort)
        if new_prompts != old_prompts:
            raise ValueError("Input prompts differ from the previous run")
        old = {(r["pair_id"],r["method"]):r for r in read_rows(old_path)}
        result = {}
        for method, name in names.items():
            arm = [r for r in rows if r["cohort"] == cohort and r["method"] == method]
            for row in arm:
                previous_row = old[(row["pair_id"], name)]
                if any(row[k] != previous_row[k] for k in ("source", "prefix", "seed")):
                    raise ValueError("Input metadata differs from the previous run")
            result[method] = dict(compared=len(arm), identical_outputs=sum(
                r["output_tokens"] == old[(r["pair_id"],name)]["output_tokens"] for r in arm))
        result["input_prompts_equal"] = True
        comparisons[label] = result
    summary["output_comparisons"] = comparisons
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--natural-previous", type=Path)
    args = parser.parse_args()
    value = analyze(args.root, args.previous, args.natural_previous)
    print(json.dumps({k:(c.get("metrics") or c.get("comparisons"))
                     for k,c in value["cohorts"].items()}, indent=2))
