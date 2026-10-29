"""从已有 JSON 生成 Feature student 转移学习报告，不加载模型或权重。

服务器用法：
    python -m draft_a.feature_transfer_report --root DIR --output-dir DIR

必需输入：data_plan.json、training.json。
可选输入：cost.json、online.json、validation_online.json、label_validation.json；缺失或运行中的
文件标为待完成。训练记录可尚未完成，曲线只使用三 seed 都有的 step。
输出：transfer_summary.json、transfer_report.md、training_curves.png/.pdf。
完整 test 结果另生成 online_comparison.png/.pdf，展示五种 variant 的
F accepted/verified 和绝对吞吐，各附场景重采样 95% 区间。
绘图使用 matplotlib Agg；其余汇总仅需 Python 标准库。不访问网络、不训练。

训练曲线为验证集 Target overlap 的三 seed 均值及样本标准差（ddof=1）。
选型读取生产脚本写入的 selected_step，不执行新选型。在线评测只使用
seed24685 的已选权重；训练三 seed 平均值与在线单 seed 结果分别展示。
成本读取未开启阶段计时的 seconds；host 与 GPU 时间分别列出，不能相加。
test 重采样以场景为单位，保留各场景全部 variant/prefix，2000 次。
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics


TEACHERS = ("gpt", "target")
METRICS = ("target_overlap", "gpt_overlap", "target_ce")
VARIANTS = ("old_mlp_fp32", "current_gpt_fp32", "gpt_student_fp32",
            "target_student_fp32", "target_only")
PREFIXES = (0, 64, 256)
STAGES = ("cache_allocate", "feature_forward", "feature_context",
          "cache_maintenance", "ranker", "target_verify", "target_context", "sampling")
BOOTSTRAP_DRAWS = 2000
SEED = 24685


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, integer=False, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{label} 需要有限数值")
    require(value >= 0 and (not positive or value > 0), f"{label} 数值范围无效")
    require(not integer or int(value) == value, f"{label} 需要整数")
    return int(value) if integer else float(value)


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} 顶层需要 JSON object")
    return value


def ratio(a, b):
    return a / b if b else None


def distribution(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def metric_values(row, location):
    source = row["validation"]
    result = {key: number(source[key], f"{location}.{key}") for key in METRICS}
    require(all(result[key] <= 1 for key in METRICS[:2]), f"{location} overlap 超过 1")
    require(number(source["rows"], f"{location}.rows", integer=True) == 3200,
            f"{location} 需要全部 3200 个验证位置")
    return result


def data_scope(plan, training):
    files = plan["files"]
    names = files["val"]
    require(isinstance(names, list) and len(names) == 50 and
            all(isinstance(name, str) for name in names) and len(set(names)) == 50,
            "data_plan 需要 50 个不同验证场景")
    require(plan["samples_per_scene"] == 64, "每个验证场景需要 64 个 F 位置")
    require(not set(files.get("train", [])).intersection(names), "train 与 val 场景重叠")
    observed = plan["splits"]["val"]
    require(observed["scenes"] == 50 and observed["rows"] == 3200,
            "data_plan 的验证位置数量与 50×64 不一致")
    require(training["val_rows"] == 3200, "training 的验证位置数量需要为 3200")
    train_names = files["train"]
    require(len(train_names) == 400 and len(set(train_names)) == 400 and
            plan["splits"]["train"]["scenes"] == 400 and plan["splits"]["train"]["rows"] == 25600 and
            training["train_rows"] == 25600, "训练采集数量与 400×64 不一致")
    return dict(split="val", scene_count=50, feature_positions_per_scene=64,
                row_count=3200, validation_names=names, train_scene_count=400, train_row_count=25600,
                weighting="按 F 位置取平均；每场景位置数相同",
                quality_variable="validation.target_overlap",
                quality_definition="同一真实 prefix 下 Target 与学生 top-p 采样分布的重合度",
                teacher_temperature=plan.get("teacher_temperature"),
                teacher_top_p=plan.get("teacher_top_p"),
                label_storage=plan.get("label_storage"),
                initial_checkpoint=training.get("initial_checkpoint"))


def summarize_training(plan, training):
    scope = data_scope(plan, training)
    seeds = training["seeds"]
    require(isinstance(seeds, list) and len(seeds) == 3 and
            all(type(seed) is int and seed >= 0 for seed in seeds) and len(set(seeds)) == 3,
            "训练需要三个不同的整数 seed")
    steps = training["steps"]
    require(isinstance(steps, list) and steps and steps[0] == 0 and
            all(type(step) is int and step >= 0 for step in steps) and steps == sorted(set(steps)),
            "训练 step 需要从 0 开始并递增")
    jobs, initials = {}, []
    for job in training["jobs"]:
        teacher, seed = job["teacher"], job["seed"]
        require(teacher in TEACHERS and seed in seeds, "job 的 teacher/seed 超出计划")
        require((teacher, seed) not in jobs, "teacher/seed job 重复")
        records, order = {}, []
        for row in job["history"]:
            step = number(row["step"], "history.step", integer=True)
            require(step in steps and step not in records, "history step 重复或超出计划")
            require(row.get("teacher", teacher) == teacher and row.get("seed", seed) == seed,
                    "history 的 teacher/seed 与 job 不一致")
            records[step] = dict(step=step, metrics=metric_values(row, f"{teacher}/{seed}/{step}"))
            order.append(step)
        require(order == sorted(order), "history step 顺序无效")
        selected = None
        if records:
            selected_step = number(job["selected_step"], "selected_step", integer=True)
            require(selected_step in records, "selected_step 没有对应验证记录")
            selected = records[selected_step]
            best = number(job["best_overlap"], "best_overlap")
            require(math.isclose(best, selected["metrics"]["target_overlap"], abs_tol=1e-8) and
                    math.isclose(best, max(r["metrics"]["target_overlap"] for r in records.values()), abs_tol=1e-8),
                    "已选记录与 best_overlap 不一致")
        initial = records.get(0)
        if initial:
            initials.append(initial["metrics"])
        jobs[teacher, seed] = dict(
            teacher=teacher, seed=seed, records=records, initial=initial,
            last=records[order[-1]] if order else None, selected=selected,
            selected_checkpoint=job.get("selected_checkpoint"),
            complete=order == steps and job.get("status") == "complete",
        )
    complete = training.get("status") == "complete"
    if complete:
        require(len(jobs) == 6 and all(j["complete"] for j in jobs.values()),
                "training 标记完成，但六个 job 的记录尚未齐全")
    teachers = {}
    for teacher in TEACHERS:
        group = [jobs[teacher, seed] for seed in seeds if (teacher, seed) in jobs]
        curve = []
        for step in steps:
            if len(group) == 3 and all(step in job["records"] for job in group):
                curve.append(dict(step=step, **distribution(
                    job["records"][step]["metrics"]["target_overlap"] for job in group)))
        summary = dict(seed_count=len(group), complete_seed_count=sum(j["complete"] for j in group),
                       curve=curve, missing_curve_steps=[s for s in steps if s not in {r["step"] for r in curve}],
                       jobs=[{key: job[key] for key in (
                           "seed", "initial", "last", "selected", "selected_checkpoint", "complete")}
                             for job in group])
        for endpoint in ("last", "selected"):
            observed = [j for j in group if j[endpoint] is not None]
            summary[endpoint] = dict(
                steps=[dict(seed=j["seed"], step=j[endpoint]["step"]) for j in observed],
                metrics={key: distribution(j[endpoint]["metrics"][key] for j in observed)
                         for key in METRICS},
                target_delta_from_initial=distribution(
                    j[endpoint]["metrics"]["target_overlap"] - j["initial"]["metrics"]["target_overlap"]
                    for j in observed if j["initial"] is not None),
            )
        teachers[teacher] = summary
    return dict(status="complete" if complete else "partial", source_status=training.get("status"),
                scope=scope, seeds=seeds, scheduled_steps=steps,
                initial=dict(source_records=len(initials), metrics={
                    key: distribution(row[key] for row in initials) for key in METRICS}),
                teachers=teachers, selection=training.get("selection"),
                std_definition="样本标准差，ddof=1；表示训练 seed 间差异")


def source_rows(data, split, variants, tokens=None):
    require(data.get("split") == split, f"需要 {split} 在线记录")
    require(data.get("validation", {}).get("status") == "passed", "输入验证尚未通过")
    parameters = data["parameters"]
    require(parameters["scenes"] == 12 and parameters["prefixes"] == list(PREFIXES),
            "在线评测需要 12 场景及 prefix0/64/256")
    require(parameters["seed"] == SEED and parameters["k"] == 2, "在线 seed/K 与计划不一致")
    require(data["policy"]["temperature"] == .9 and data["policy"]["top_p"] == .9,
            "在线采样策略需要 .9/.9")
    token_budget = number(parameters["tokens"], "parameters.tokens", integer=True, positive=True)
    if tokens is not None:
        require(token_budget == tokens, f"test 在线预算需要为 {tokens}")
    groups = {variant: {} for variant in variants}
    identities = set()
    for row in data["rows"]:
        # 当前生产脚本使用平铺字段，同时兼容带 main 的成本记录。
        main = row.get("main", row)
        variant = row.get("variant", main.get("variant"))
        require(variant in groups, "出现未知在线 variant")
        name = row.get("name", main.get("name"))
        require(isinstance(name, str) and bool(name), "在线行需要场景名")
        prefix = number(main["prompt_tokens"], "prompt_tokens", integer=True)
        seed = number(main["seed"], "seed", integer=True)
        require(prefix in PREFIXES and row.get("split", main.get("split")) == split,
                "在线行的 prefix/split 无效")
        key = (name, prefix, seed)
        require(key not in groups[variant], "在线行重复")
        require(main["k"] == (0 if variant == "target_only" else 2), "在线行的 K 无效")
        require(main.get("requested_tokens", token_budget) == token_budget, "在线行的预算不同")
        for field in ("target_precision", "feature_precision"):
            require(field not in main or main[field] == "FP32", "在线行需要 FP32")
        counts = {}
        for stem in ("", "position_", "feature_", "eos_"):
            a, v, p = [number(main[stem + field], stem + field, integer=True)
                       for field in ("accepted", "verified", "proposed")]
            require(a <= v <= p, "接受计数顺序无效")
            counts.update({stem + field: value for field, value in zip(("accepted", "verified", "proposed"), (a, v, p))})
        for field in ("accepted", "verified", "proposed"):
            require(counts[field] == sum(counts[k + "_" + field] for k in ("position", "feature", "eos")),
                    "分类型计数与总数不同")
        seconds = number(main["seconds"], "main.seconds", positive=True)
        generated = number(main["generated_tokens"], "generated_tokens", integer=True)
        require(generated <= token_budget, "产出超过预算")
        value = dict(name=name, prefix=prefix, seed=seed, variant=variant,
                     seconds=seconds, generated_tokens=generated, counts=counts, source=row, main=main)
        groups[variant][key] = value
        identities.add(key)
    names = {key[0] for key in identities}
    require(len(names) == 12 and len(identities) == 36 and
            all(set(rows) == identities for rows in groups.values()),
            "各 variant 需要相同的 36 个场景/prefix/seed 组合")
    require(all({p for n, p, _ in identities if n == name} == set(PREFIXES) for name in names),
            "场景的三个 prefix 不齐全")
    return groups, dict(split=split, scene_count=12, prompts_per_variant=36,
                        requested_tokens=token_budget, seed=SEED, prefixes=list(PREFIXES),
                        k=2, policy=data["policy"], validation_status="passed")


def pooled(rows):
    counts = {key: sum(row["counts"][key] for row in rows) for key in rows[0]["counts"]}
    seconds = sum(row["seconds"] for row in rows)
    generated = sum(row["generated_tokens"] for row in rows)
    measures = dict(tokens_per_second=ratio(generated, seconds))
    for stem, name in (("", "all"), ("position_", "position"), ("feature_", "feature")):
        measures[name + "_accepted_over_verified"] = ratio(counts[stem + "accepted"], counts[stem + "verified"])
        measures[name + "_accepted_over_proposed"] = ratio(counts[stem + "accepted"], counts[stem + "proposed"])
    return dict(run_count=len(rows), generated_tokens=generated, unprofiled_seconds=seconds,
                counts=counts, metrics=measures)


def online_measure(rows):
    grouped = {variant: [row for row in rows if row["variant"] == variant] for variant in VARIANTS}
    totals = {variant: pooled(group) for variant, group in grouped.items()}
    result = {}
    for variant, total in totals.items():
        for key, value in total["metrics"].items():
            result[f"{variant}.{key}"] = value
        acceptance = total["metrics"]["feature_accepted_over_verified"]
        old_acceptance = totals["old_mlp_fp32"]["metrics"]["feature_accepted_over_verified"]
        result[f"{variant}.feature_acceptance_delta_vs_old"] = (
            acceptance - old_acceptance if acceptance is not None else None)
        speed = total["metrics"]["tokens_per_second"]
        for reference in ("old_mlp_fp32", "target_only"):
            result[f"{variant}.speedup_vs_{reference}"] = ratio(speed, totals[reference]["metrics"]["tokens_per_second"])
    return result


def scene_bootstrap(rows, measure):
    # 服务器上优先复用原评测的场景重采样实现。
    try:
        from draft_a.feature_acceptance_eval import _bootstrap
    except ModuleNotFoundError as error:
        if error.name not in ("draft_a", "draft_a.feature_acceptance_eval"):
            raise
    else:
        return _bootstrap(rows, measure, SEED, draws=BOOTSTRAP_DRAWS)
    scenes = {}
    for row in rows:
        scenes.setdefault(row["name"], []).append(row)
    groups = list(scenes.values())
    samples = {key: [] for key in measure(rows)}
    rng = random.Random(SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [row for _ in groups for row in groups[rng.randrange(len(groups))]]
        for key, value in measure(sampled).items():
            if value is not None:
                require(math.isfinite(value), "重采样统计需要有限数值")
                samples[key].append(value)

    def percentile(values, fraction):
        values = sorted(values)
        index = (len(values) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return dict(method="scene resampling, percentile 95% intervals", scene_count=len(groups),
                resamples=BOOTSTRAP_DRAWS, seed=SEED,
                ci95={key: [percentile(values, .025), percentile(values, .975)] if values else [None, None]
                      for key, values in samples.items()},
                valid_resamples={key: len(values) for key, values in samples.items()})


def summarize_online(data, training, split):
    groups, scope = source_rows(data, split, VARIANTS, tokens=512 if split == "test" else None)
    weights = data["selected_weights"]
    for teacher in TEACHERS:
        selected = next((job for job in training["teachers"][teacher]["jobs"] if job["seed"] == SEED), None)
        require(selected is not None and selected["selected"] is not None,
                "在线权重没有对应的训练选型记录")
        declared = weights[teacher]
        require(declared["seed"] == SEED and declared["step"] == selected["selected"]["step"] and
                declared["path"] == selected["selected_checkpoint"], "在线权重与训练已选权重不同")
    rows = [row for group in groups.values() for row in group.values()]
    bootstrap = scene_bootstrap(rows, online_measure)
    points = online_measure(rows)
    variants = {}
    for variant, group in groups.items():
        total = pooled(list(group.values()))
        total["ci95"] = {key: bootstrap["ci95"][f"{variant}.{key}"] for key in total["metrics"]}
        total["paired_speed"] = {
            reference: dict(ratio=points[f"{variant}.speedup_vs_{reference}"],
                            ci95=bootstrap["ci95"][f"{variant}.speedup_vs_{reference}"])
            for reference in ("old_mlp_fp32", "target_only")
        }
        total["paired_feature_acceptance_delta"] = dict(
            difference=points[f"{variant}.feature_acceptance_delta_vs_old"],
            ci95=bootstrap["ci95"][f"{variant}.feature_acceptance_delta_vs_old"],
            reference="old_mlp_fp32",
        )
        total["by_prefix"] = {str(prefix): pooled([r for r in group.values() if r["prefix"] == prefix])
                              for prefix in PREFIXES}
        variants[variant] = total
    return dict(status="complete", scope=scope, selected_weights=weights, variants=variants,
                bootstrap=bootstrap, pairing="同名场景、同 prefix、同 seed；保留各 variant 的全部运行")


def summarize_cost(data):
    variants = VARIANTS[:2]
    groups, scope = source_rows(data, "test", variants)
    result = {}
    for variant, group in groups.items():
        rows = list(group.values())
        total = pooled(rows)
        stages = {stage: dict(calls=0, input_tokens=0, host_seconds=0.0,
                              cuda_events=dict(count=0, seconds=0.0), operations={}) for stage in STAGES}
        profiled_seconds, overheads, lengths = [], [], []
        for row in rows:
            source = row["source"]
            require(source.get("profile_output_matches") is True, "成本记录需标记同 seed 输出一致")
            seconds = number(source["profile_seconds"], "profile_seconds", positive=True)
            overhead = number(source["profile_overhead_ratio"], "profile_overhead_ratio", positive=True)
            require(math.isclose(overhead, seconds / row["seconds"], rel_tol=1e-8), "profile 耗时比值不同")
            profiled_seconds.append(seconds)
            overheads.append(overhead)
            main = row["main"]
            rounds = number(main["rounds"], "rounds", integer=True)
            observed = [number(v, "round_generated_lengths", integer=True, positive=True)
                        for v in main["round_generated_lengths"]]
            require(len(observed) == rounds and all(v <= 4 for v in observed), "逐轮产出记录无效")
            recorded_mean = main["mean_generated_per_round"]
            require(recorded_mean is None if not rounds else
                    math.isclose(number(recorded_mean, "mean_generated_per_round"), sum(observed) / rounds, rel_tol=1e-8),
                    "逐轮平均产出与记录不同")
            outside = row["generated_tokens"] - sum(observed)
            require(outside in ((0, 1, 2) if row["prefix"] == 0 else (0,)), "逐轮产出与 token 总数不同")
            lengths.extend(observed)
            profile = source.get("stage_profile", source.get("stats", {}).get("stage_profile"))
            require(isinstance(profile, dict) and set(profile) == set(STAGES), "阶段计时条目不齐全")
            for stage, entry in profile.items():
                dest = stages[stage]
                for key in ("calls", "input_tokens"):
                    dest[key] += number(entry[key], f"{stage}.{key}", integer=True)
                dest["host_seconds"] += number(entry["host_seconds"], f"{stage}.host_seconds")
                for key in ("count", "seconds"):
                    dest["cuda_events"][key] += number(entry["cuda_events"][key], f"{stage}.cuda_events.{key}", integer=key == "count")
                for operation, values in entry.get("operations", {}).items():
                    dest_operation = dest["operations"].setdefault(operation, dict(calls=0, input_tokens=0, context_tokens=0))
                    for key in dest_operation:
                        dest_operation[key] += number(values[key], f"{operation}.{key}", integer=True)
        profile_sum = sum(profiled_seconds)
        total.update(
            stage_profile=stages, profile_seconds=profile_sum,
            profile_overhead_ratio=ratio(profile_sum, total["unprofiled_seconds"]),
            per_run_profile_overhead_ratio=distribution(overheads),
            rounds=len(lengths), mean_generated_per_round=ratio(sum(lengths), len(lengths)),
            round_generated_length_histogram=dict(sorted(Counter(str(v) for v in lengths).items())),
            profile_output_matches=True,
        )
        result[variant] = total
    scope.update(unprofiled_runs=72, profiled_runs=72, reported_decodes=144)
    return dict(status="complete", scope=scope, variants=result,
                timing_definition="主速度来自未开启阶段计时的 seconds；host/GPU 时间分别展示，不能相加")


def build_summary(root):
    root = Path(root)
    training = summarize_training(read_json(root / "data_plan.json"), read_json(root / "training.json"))
    summary = dict(status="training_ready", root=str(root.resolve()), training=training, sources={})
    label_path = root / "label_validation.json"
    if label_path.is_file():
        labels = read_json(label_path)
        references = {}
        for split in ("train", "val"):
            raw = labels.get("splits", {}).get(split, {}).get("gpt_target_nucleus_overlap")
            if raw is not None:
                value = number(raw, f"label_validation.{split}.gpt_target_nucleus_overlap")
                require(value <= 1, "教师自身 overlap 超过 1")
                references[split] = dict(gpt_target_nucleus_overlap=value,
                                        rows=25600 if split == "train" else 3200)
        summary["label_validation"] = dict(source=str(label_path.resolve()),
                                          source_status=labels.get("status"), splits=references)
    for filename, key in (("cost.json", "cost"), ("online.json", "test_online"),
                          ("validation_online.json", "validation_online")):
        path = root / filename
        if not path.is_file():
            summary[key] = dict(status="pending", reason="输入文件尚未提供")
            summary["sources"][key] = dict(path=str(path.resolve()), exists=False)
            continue
        data = read_json(path)
        summary["sources"][key] = dict(path=str(path.resolve()), exists=True, status=data.get("status"))
        if data.get("status") != "complete":
            summary[key] = dict(status="pending", reason="输入尚未标记完成", observed_rows=len(data.get("rows", [])))
        elif key == "cost":
            summary[key] = summarize_cost(data)
        else:
            require(training["status"] == "complete", "在线报告需要六个训练 job 完成")
            summary[key] = summarize_online(data, training, "test" if key == "test_online" else "val")
    if training["status"] != "complete":
        summary["status"] = "partial"
    elif all(summary[key]["status"] == "complete" for key in ("cost", "test_online", "validation_online")):
        summary["status"] = "complete"
    return summary


def training_figure(training):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("绘图需要 matplotlib；本工具不会安装依赖。") from error
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        colors = {"gpt": "#0072B2", "target": "#D55E00"}
        for teacher in TEACHERS:
            curve = training["teachers"][teacher]["curve"]
            if not curve:
                continue
            steps = [row["step"] for row in curve]
            means = [100 * row["mean"] for row in curve]
            stds = [100 * row["std"] for row in curve]
            ax.plot(steps, means, color=colors[teacher], marker="o", linewidth=2,
                    label=f"{teacher.upper()} teacher: 3 seeds")
            ax.fill_between(steps, [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)], color=colors[teacher], alpha=.18)
        initial = training["initial"]["metrics"]["target_overlap"]["mean"]
        if initial is not None:
            ax.axhline(100 * initial, color="#666666", linestyle="--", linewidth=1.2,
                       label="Original MLP: step 0")
        title = "Feature student transfer: validation target_overlap"
        if training["status"] != "complete":
            title += " (partial records)"
        ax.set(title=title, xlabel="Training step", ylabel="validation.target_overlap (%)")
        ax.grid(axis="y", alpha=.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", frameon=False)
        if not any(training["teachers"][t]["curve"] for t in TEACHERS):
            ax.text(.5, .5, "Waiting for shared steps across all 3 seeds", transform=ax.transAxes,
                    ha="center", va="center")
        fig.text(.5, .035, "Mean +/- sample SD across training seeds; 50 validation scenes x 64 Feature positions.\n"
                 "Matched-prefix distribution overlap; checkpoint selection uses this validation set.",
                 ha="center", va="bottom", fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, .11, 1, 1))
        return fig


def online_figure(online):
    """仅使用完整 test 汇总，区间复用成对场景重采样结果。"""
    require(online["status"] == "complete" and online["scope"]["split"] == "test",
            "端到端图需要完整 test 结果")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("绘图需要 matplotlib；本工具不会安装依赖。") from error
    labels = ("Old MLP\nFP32", "Current GPT\nFP32", "GPT student\nFP32",
              "Target student\nFP32", "Target only\nFP32")
    colors = ("#7A7A7A", "#8064A2", "#0072B2", "#D55E00", "#009E73")
    panels = (("feature_accepted_over_verified", "Feature accepted / verified (%)", 100),
              ("tokens_per_second", "Throughput (token/s)", 1))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.1))
        for ax, (metric, ylabel, scale) in zip(axes, panels):
            values, intervals = [], []
            for variant in VARIANTS:
                value = online["variants"][variant]
                mean = value["metrics"][metric]
                interval = value["ci95"][metric]
                values.append(mean * scale if mean is not None else float("nan"))
                intervals.append([v * scale if v is not None else None for v in interval])
            ax.bar(range(len(VARIANTS)), values, color=colors, width=.65, alpha=.9)
            high = max([v for v in values if math.isfinite(v)] +
                       [v[1] for v in intervals if v[1] is not None] + [1.0])
            for index, (mean, interval) in enumerate(zip(values, intervals)):
                if not math.isfinite(mean):
                    ax.text(index, .035, "n/a", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", color="#555555")
                    continue
                low, upper = interval
                if low is not None and upper is not None:
                    # 直接绘制区间端点，保留可能未包含点估计的百分位区间。
                    ax.vlines(index, low, upper, color="#222222", linewidth=1.3)
                    ax.hlines((low, upper), index - .08, index + .08,
                              color="#222222", linewidth=1.3)
                label_y = max(mean, upper if upper is not None else mean) + .025 * high
                ax.text(index, label_y, f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(len(VARIANTS)), labels)
            ax.set_xlim(-.6, len(VARIANTS) - .4)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
            ax.set_ylim(0, 108 if scale == 100 else high * 1.16)
            if scale == 100:
                ax.set_yticks(range(0, 101, 20))
        axes[0].set_title("Feature proposal acceptance")
        axes[1].set_title("Unprofiled end-to-end throughput")
        scope = online["scope"]
        fig.suptitle("Unified K2 PF decoder: five variants on the same test scenes", fontsize=13)
        fig.text(.5, .025,
                 f"{scope['scene_count']} test scenes x 3 prefixes; {scope['requested_tokens']}-token budget; FP32; temperature/top-p 0.9/0.9.\n"
                 "95% intervals: paired scene bootstrap (2,000 draws). Student training seed: 24685.\n"
                 "Throughput uses unprofiled wall time. Target-only Feature proposal acceptance: n/a.",
                 ha="center", va="bottom", fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, .14, 1, .94))
        return fig


def value_text(value, scale=1, digits=3):
    return "—" if value is None else f"{value * scale:.{digits}f}"


def mean_text(value, scale=1):
    text = value_text(value["mean"], scale)
    return text + (" ± " + value_text(value["std"], scale) if value["std"] is not None else "")


def interval_text(value, scale=1):
    return f"[{value_text(value[0], scale)}, {value_text(value[1], scale)}]"


def markdown(summary):
    training = summary["training"]
    lines = ["# Feature student 转移学习结果", "", f"状态：`{summary['status']}`。", "",
             "标签采集范围：400 个 train 场景、50 个 val 场景，每场景 64 个 F 位置；分别为 25600 和 3200 个位置。",
             "训练离线评估使用 50 个 val 场景，每场景 64 个 F 位置，共 3200 个位置；按位置取平均。",
             "曲线与训练表只使用这批验证数据，未使用 test 指标。", "",
             "## 训练验证结果", "",
             "质量变量为 `validation.target_overlap`：同一真实 prefix 下 Target 与学生采样分布的重合度，即各 F 值概率的逐项较小值之和。",
             "validation 的 Target、GPT 教师及学生均使用 temperature 0.9、top-p 0.9；`gpt_overlap` 衡量与 GPT 教师的重合度。",
             "在线接受率另由独立解码中的 accepted/verified 计算，分母只包含实际到达的提案。", "",
             "训练目标为教师全分布 CE + 3×(1−overlap)。CE 使用 temperature 0.9 的教师全分布与学生 softmax；",
             "训练 overlap 项比较 top-p 0.9 的教师 nucleus 与未截取的学生 softmax。验证的 overlap 两侧均采用 top-p 0.9。",
             "`target_ce` 使用 Target 全分布计算交叉熵，单位为 nats。", "",
             "曲线为两种 teacher 各三 seed 的 `target_overlap` 均值 ± 样本标准差（ddof=1）。",
             "只绘制同一 teacher 三个 seed 都有记录的 step，未插值。标准差表示训练随机性，未表示场景置信区间。", "",
             "![训练验证曲线](training_curves.png)", "",
             "| 记录 | seed 数 | target_overlap (%) | gpt_overlap (%) | target_ce (nats) |", "",
             "|---|---:|---:|---:|---:|"]
    initial = training["initial"]
    lines.append("| 原 MLP stage0 / step 0 | 同一权重 | " + " | ".join(
        mean_text(initial["metrics"][key], 100 if key.endswith("overlap") else 1) for key in METRICS) + " |")
    for teacher in TEACHERS:
        group = training["teachers"][teacher]
        for endpoint, label in (("last", "最后记录"), ("selected", "已选记录")):
            value = group[endpoint]
            lines.append(f"| {teacher} teacher {label} | {value['metrics']['target_overlap']['n']} | " + " | ".join(
                mean_text(value["metrics"][key], 100 if key.endswith("overlap") else 1) for key in METRICS) + " |")
    lines.extend(["", f"原 MLP 初始值汇总 {initial['source_records']} 条 step 0 记录；这些记录来自相同初始权重和相同验证集，未增加独立样本数。"])
    references = summary.get("label_validation", {}).get("splits", {})
    if references:
        lines.extend(["", "标签的 CPU 验证记录另提供教师自身分布重合度，来源为 `label_validation.json`。",
                      "`gpt_target_nucleus_overlap` 衡量同一真实 prefix 上 GPT 与 Target 的 nucleus 分布重合度，作为 student 转移学习目标的参考。",
                      "此处直接报告分布概率的重合度；在线接受率另在下方解码结果中计算。", "",
                      "| split | F 位置数 | gpt_target_nucleus_overlap (%) |", "",
                      "|---|---:|---:|"])
        for split in ("val", "train"):
            if split in references:
                row = references[split]
                lines.append(f"| {split} | {row['rows']} | {value_text(row['gpt_target_nucleus_overlap'], 100)} |")
        lines.extend(["", "标签概率以 FP16 保存；使用前在 FP32 下重新归一。"])
    for teacher in TEACHERS:
        group = training["teachers"][teacher]
        last_steps = ", ".join(f"seed{x['seed']}: {x['step']}" for x in group["last"]["steps"]) or "尚无"
        selected_steps = ", ".join(f"seed{x['seed']}: {x['step']}" for x in group["selected"]["steps"]) or "尚无"
        lines.extend(["", f"{teacher} 最后 step：{last_steps}；已选 step：{selected_steps}。",
                      f"相对各 seed step 0 的 Target overlap 变化：最后记录 {mean_text(group['last']['target_delta_from_initial'], 100)} 个百分点；"
                      f"已选记录 {mean_text(group['selected']['target_delta_from_initial'], 100)} 个百分点。"])
        if group["missing_curve_steps"]:
            lines.append(f"尚无三 seed 齐全记录的 step：{group['missing_curve_steps']}。")
    lines.extend(["", "已选记录直接读取训练脚本的 selected_step，按同一验证集 Target overlap 选型。",
                  "同一批验证数据兼用于选型和展示，已选值存在选择带来的乐观偏差。",
                  "曲线提升支持当前验证位置上的分布匹配变化，未建立 test 泛化、在线接受率或速度结论。", "",
                  "## Adapter 成本", ""])
    cost = summary["cost"]
    if cost["status"] != "complete":
        lines.extend([f"待完成：{cost['reason']}。", ""])
    else:
        lines.extend(["每个 variant 为 12 个 test 场景 × 3 个 prefix，共 36 次主运行和 36 次 profile 运行；两个 variant 合计 144 次解码。",
                      "主速度仅使用未开启阶段计时的耗时；profile 输出一致性沿用输入 JSON 的验证标记。", "",
                      "| variant | 主耗时 (s) | 主速度 (token/s) | profile 耗时 (s) | profile/主耗时 | profile 增幅 (%) | 每轮产出 |", "",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for variant, value in cost["variants"].items():
            fields = [value["unprofiled_seconds"], value["metrics"]["tokens_per_second"],
                      value["profile_seconds"], value["profile_overhead_ratio"],
                      100 * (value["profile_overhead_ratio"] - 1), value["mean_generated_per_round"]]
            lines.append(f"| {variant} | " + " | ".join(value_text(v) for v in fields) + " |")
        lines.extend(["", "每轮产出用逐轮 token 数之和除以轮数；prefix0 的首次 Target P/F 未计入轮数。", "",
                      "阶段表为 36 次 profile 运行的各项时间总和。GPU 时间来自 CUDA events；host 时间来自 host_seconds。",
                      "两列分别解读，不能相加。GPU 列的 — 表示未记录 GPU event。缓存分配和 prefill 已包含在计时中。", "",
                      "| variant | stage | GPU (s) | host (s) | 输入 token 数 |", "",
                      "|---|---|---:|---:|---:|"])
        for variant, value in cost["variants"].items():
            for stage, row in value["stage_profile"].items():
                gpu = row["cuda_events"]["seconds"] if row["cuda_events"]["count"] else None
                lines.append(f"| {variant} | {stage} | {value_text(gpu, digits=6)} | {value_text(row['host_seconds'], digits=6)} | {row['input_tokens']} |")
        lines.extend(["", "GPT 的 context_and_forward 包含待处理 P/F 历史和提案 P 的同次计算；无法拆开的时间保留为一项。输入与历史 token 数保存在 summary 中。", ""])
    for key, title in (("test_online", "Test 在线结果"), ("validation_online", "Validation 在线结果")):
        lines.extend([f"## {title}", ""])
        online = summary[key]
        if online["status"] != "complete":
            lines.extend([f"待完成：{online['reason']}。", ""])
            continue
        scope = online["scope"]
        lines.extend([f"范围：{scope['scene_count']} 个 {scope['split']} 场景，三个 prefix，预算 {scope['requested_tokens']} token，FP32，.9/.9，K2。Target-only 使用 K0。",
                      "在线 student 固定使用训练 seed24685 的已选权重；上方训练表均值来自三个训练 seed，其余种子用于观察训练波动。",
                      "解码 seed 以 24685 为起点，按场景与 prefix 偏移；同一组合的 variant 使用相同解码 seed。", ""])
        if key == "validation_online":
            lines.extend(["此部分用于验证集在线比较，与 test 分开展示；它未替代 test 结果。", ""])
        else:
            lines.extend(["![同一 test 集五种 variant 的接受率与吞吐](online_comparison.png)", "",
                          "两面板分别显示 F accepted/verified 与绝对 token/s，均附场景重采样 95% 区间。",
                          "Target-only 没有 Feature 提案，图中 F 接受率标为 n/a。", ""])
        lines.extend(["| variant | 全部 accepted/verified (%) | 全部 accepted/proposed (%) | F accepted/verified (%) | F accepted/proposed (%) | token/s | 相对旧 MLP 速度 [95%区间] | 相对 Target 速度 [95%区间] |", "",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for variant, value in online["variants"].items():
            m = value["metrics"]
            fields = [value_text(m[key], 100) for key in ("all_accepted_over_verified", "all_accepted_over_proposed",
                                                         "feature_accepted_over_verified", "feature_accepted_over_proposed")]
            fields.append(value_text(m["tokens_per_second"]))
            for reference in ("old_mlp_fp32", "target_only"):
                speed = value["paired_speed"][reference]
                fields.append(value_text(speed["ratio"]) + " " + interval_text(speed["ci95"]))
            lines.append(f"| {variant} | " + " | ".join(fields) + " |")
        for variant in ("gpt_student_fp32", "target_student_fp32"):
            delta = online["variants"][variant]["paired_feature_acceptance_delta"]
            lines.extend(["", f"{variant} 相对旧 MLP 的 Feature 接受率差值："
                          f"{value_text(delta['difference'], 100)} 个百分点，"
                          f"场景成对重采样 95% 区间 {interval_text(delta['ci95'], 100)}。"])
        lines.extend(["", "接受率先相加计数，再计算比值；verified 只包含首次拒绝前实际参与验证的提案，proposed 包含未使用的后续提案。",
                      "速度为总生成 token 数除以总主耗时；成对速度比用相同场景/prefix/seed 的运行集合。",
                      "95% 区间按场景成对重采样 2000 次，保留该场景的全部 variant/prefix。三个 prefix 存在关联，独立场景数仍为 12。",
                      "总计、P/F 接受率区间和逐 prefix 结果保存在 transfer_summary.json；小幅变化与包含 1 的速度区间不支持可靠的速度方向判断。", ""])
    return "\n".join(lines).rstrip() + "\n"


def run(root, output_dir):
    summary = build_summary(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []

    def save_figure(fig, name):
        try:
            for extension in ("png", "pdf"):
                filename = f"{name}.{extension}"
                fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
                plot_files.append(filename)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    save_figure(training_figure(summary["training"]), "training_curves")
    if summary["test_online"]["status"] == "complete":
        save_figure(online_figure(summary["test_online"]), "online_comparison")
    summary["plots"] = plot_files
    report = markdown(summary)
    (output_dir / "transfer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "transfer_report.md").write_text(report, encoding="utf-8")
    files = [str((output_dir / name).resolve()) for name in
             ["transfer_summary.json", "transfer_report.md", *plot_files]]
    return dict(status=summary["status"], files=files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
