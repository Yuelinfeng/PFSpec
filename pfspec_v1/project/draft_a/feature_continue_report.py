"""根据完整评测 JSON 生成续训对比；仅读取评测和训练记录。"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys


DRAWS = 2000
SEED = 24687
KINDS = ("all", "feature", "position", "eos")
LABELS = {"all": "全部 token", "feature": "Feature", "position": "P", "eos": "EOS"}
COUNTS = tuple(
    f"{kind}_{field}"
    for kind in KINDS
    for field in ("accepted", "verified", "proposed")
)
FIELDS = COUNTS + ("generated_tokens", "seconds")
NOTES = [
    "固定 test50 使用已保存的 Target 轨迹中的固定前缀，overlap 与在线接受率分开报告。",
    "两个模型各自随机生成上下文会不同；同 seed 不等于相同轨迹。在线配对依据是同名场景，生成轨迹不相同。",
    "accepted/verified 仅包含到首次拒绝为止实际参与验证的提议；accepted/proposed 的分母包含未使用的后续提议。",
    "合并结果先相加 accepted、verified、proposed，再计算比值；速度为总生成 token 数除以总秒数。",
    "差值统一为新模型减旧模型；接受率和 overlap 的差值与区间使用百分点。",
    "95% 差值区间使用同场景跨模型成对重采样，每次保留该场景在当前组合中的全部运行；2000 次，seed=24687，采用百分位法。",
    "12 个在线场景的小幅变化不能声称显著。区间包含 0 时，当前样本无法可靠区分变化方向；区间不含 0 也只支持当前场景集合下的方向判断。",
    "同场景跨 prefix/Gamma 的运行存在关联；合并 Gamma 后仍只有原来的场景数，不能把运行次数当作独立场景数。",
    "本工具仅读取既有结果，不改变训练或选型，不用 test 结果选模型。history 仅展示 val 指标，不执行排名。",
    "速度范围沿用输入在线评测：包括缓存分配、prefill、提议、验证和剩余分布采样；不包含模型加载及输出验证。跨次运行的硬件负载会影响速度。",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须为数值")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError(f"{label} 数值无效")
    return int(value) if integer else value


def ratio(a, b):
    return a / b if b else None


def difference(old, new):
    return new - old if old is not None and new is not None else None


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def interval(values):
    return [percentile(values, 0.025), percentile(values, 0.975)] if values else [None, None]


def offline_data(data, tag):
    if data.get("status") != "complete" or data.get("split") != "test":
        raise ValueError(f"{tag} 需要完整的 test 评测")
    offline = data["offline"]
    expected = data["expected_test_files"]
    selected = data["selected_test_files"]
    if (len(expected) != 50 or len(set(expected)) != 50
            or len(selected) != 50 or set(expected) != set(selected)
            or offline.get("scene_count") != 50 or offline.get("complete") is not True):
        raise ValueError(f"{tag} 固定 test50 场景不完整")
    result = {kind: {} for kind in KINDS}
    for row in offline["per_sequence"]:
        kind, name = row["kind"], row["name"]
        if kind not in result or name not in expected or name in result[kind]:
            raise ValueError(f"{tag} 固定前缀场景条目重复或无效")
        count = number(row["sample_count"], "sample_count", integer=True)
        total = number(row["overlap_sum"], "overlap_sum")
        if total > count + 1e-6:
            raise ValueError(f"{tag} overlap_sum 超出样本数")
        result[kind][name] = (total, count)
    for kind in KINDS:
        if set(result[kind]) != set(expected):
            raise ValueError(f"{tag} {kind} 固定前缀场景不完整")
        pooled = ratio(sum(v[0] for v in result[kind].values()),
                       sum(v[1] for v in result[kind].values()))
        recorded = offline["summary"][kind]["pooled_overlap"]
        if ((pooled is None) != (recorded is None)
                or (pooled is not None and not math.isclose(pooled, recorded, abs_tol=1e-8))):
            raise ValueError(f"{tag} {kind} 逐场景 overlap 与 summary 不一致")
    return result


def offline_comparison(old, new):
    rows = []
    for kind in KINDS:
        names = sorted(old[kind])
        pairs = [(old[kind][name], new[kind][name]) for name in names]
        totals = [tuple(sum(pair[side][i] for pair in pairs) for i in (0, 1))
                  for side in (0, 1)]
        values = [ratio(*total) for total in totals]
        rng = random.Random(SEED)
        deltas = []
        for _ in range(DRAWS):
            sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            a, b = [ratio(sum(pair[side][0] for pair in sampled),
                          sum(pair[side][1] for pair in sampled)) for side in (0, 1)]
            delta = difference(a, b)
            if delta is not None:
                deltas.append(delta)
        models = []
        for side, source in enumerate((old, new)):
            means = [ratio(*source[kind][name]) for name in names]
            means = [value for value in means if value is not None]
            models.append({"overlap_sum": totals[side][0], "sample_count": totals[side][1],
                           "pooled_overlap": values[side],
                           "equal_scene_overlap": statistics.mean(means) if means else None})
        rows.append({"kind": kind, "scene_count": len(names), "old": models[0], "new": models[1],
                     "delta": difference(*values), "delta_ci95": interval(deltas),
                     "valid_draws": len(deltas),
                     "weighting": "sum(overlap_sum)/sum(sample_count)"})
    return rows


def online_data(data, tag):
    online = data["online"]
    if online.get("unavailable_prefixes"):
        raise ValueError(f"{tag} 存在无法评测的 prefix")
    rows = online["rows"]
    domains = set()
    result = []
    for row in rows:
        mode = row["mode"]
        if mode not in ("draft", "target_only"):
            raise ValueError(f"{tag} 存在未知在线模式 {mode}")
        gamma = number(row["k"], "k", integer=True)
        prefix = number(row["prompt_tokens"], "prompt_tokens", integer=True)
        if (mode == "draft" and gamma == 0) or (mode == "target_only" and gamma != 0):
            raise ValueError(f"{tag} mode 与 Gamma 不一致")
        key = (mode, prefix, gamma, row["name"])
        if key in domains:
            raise ValueError(f"{tag} 在线运行条目重复")
        domains.add(key)
        if row["name"] not in data["selected_test_files"]:
            raise ValueError(f"{tag} 在线场景超出 test50")
        values = []
        for kind in KINDS:
            stem = "" if kind == "all" else f"{kind}_"
            counts = [number(row[stem + field], stem + field, integer=True)
                      for field in ("accepted", "verified", "proposed")]
            if not counts[0] <= counts[1] <= counts[2]:
                raise ValueError(f"{tag} {kind} 接受计数顺序无效")
            values.extend(counts)
        for index in range(3):
            if values[index] != sum(values[start + index] for start in (3, 6, 9)):
                raise ValueError(f"{tag} F、P、EOS 计数之和与总计不一致")
        values.extend([number(row["generated_tokens"], "generated_tokens", integer=True),
                       number(row["seconds"], "seconds")])
        if values[-1] <= 0:
            raise ValueError(f"{tag} 在线耗时必须大于 0")
        result.append({"key": key, "name": row["name"], "mode": mode,
                       "prefix": prefix, "gamma": gamma, "seed": row.get("seed"),
                       "requested_tokens": row.get("requested_tokens"), "values": values})
    prefixes = sorted(data["parameters"]["prefixes"])
    gammas = sorted(data["parameters"]["gamma"])
    if not prefixes or not gammas or len(set(prefixes)) != len(prefixes) or len(set(gammas)) != len(gammas):
        raise ValueError(f"{tag} prefix/Gamma 列表为空或存在重复值")
    grouped = defaultdict(set)
    for row in result:
        grouped[row["mode"], row["prefix"], row["gamma"]].add(row["name"])
    count = number(online["scene_count"], "scene_count", integer=True)
    if count <= 0:
        raise ValueError(f"{tag} 在线场景数必须大于 0")
    reference = None
    expected_groups = [(mode, prefix, gamma) for prefix in prefixes
                       for mode, widths in (("target_only", [0]), ("draft", gammas))
                       for gamma in widths]
    if set(grouped) != set(expected_groups):
        raise ValueError(f"{tag} prefix/Gamma 组合不完整")
    for key in expected_groups:
        names = grouped[key]
        if len(names) != count or (reference is not None and names != reference):
            raise ValueError(f"{tag} 各在线组合的场景不一致")
        reference = names
    return result


def metrics(values):
    result = {}
    for i, kind in enumerate(KINDS):
        a, v, p = values[3 * i:3 * i + 3]
        result[kind + "_accepted_over_verified"] = ratio(a, v)
        result[kind + "_accepted_over_proposed"] = ratio(a, p)
    result["tokens_per_second"] = ratio(values[-2], values[-1])
    return result


def total_values(rows):
    return [sum(row["values"][i] for row in rows) for i in range(len(FIELDS))]


def summarize(rows, target_rows):
    totals = total_values(rows)
    target_totals = total_values(target_rows)
    result = {"run_count": len(rows), "scene_count": len({row["name"] for row in rows}),
              "counts": dict(zip(FIELDS, totals)), "metrics": metrics(totals),
              "target_only_tokens_per_second": metrics(target_totals)["tokens_per_second"],
              "below_30pct_runs": {}}
    result["speedup_vs_target_only"] = ratio(result["metrics"]["tokens_per_second"],
                                              result["target_only_tokens_per_second"])
    for kind in KINDS:
        result["below_30pct_runs"][kind] = {}
        for denominator in ("verified", "proposed"):
            key = f"{kind}_accepted_over_{denominator}"
            values = [metrics(row["values"])[key] for row in rows]
            result["below_30pct_runs"][kind][denominator] = {
                "count": sum(value is not None and value < 0.30 for value in values),
                "defined_runs": sum(value is not None for value in values),
                "undefined_runs": sum(value is None for value in values),
            }
    return result


def paired_online(old_rows, new_rows):
    scenes = sorted({row["name"] for row in old_rows})
    paired = []
    for name in scenes:
        paired.append((total_values([row for row in old_rows if row["name"] == name]),
                       total_values([row for row in new_rows if row["name"] == name])))
    values = {key: [] for key in metrics([0] * len(FIELDS))}
    rng = random.Random(SEED)
    for _ in range(DRAWS):
        sampled = [paired[rng.randrange(len(paired))] for _ in paired]
        results = [metrics([sum(pair[side][i] for pair in sampled)
                            for i in range(len(FIELDS))]) for side in (0, 1)]
        for key in values:
            delta = difference(results[0][key], results[1][key])
            if delta is not None:
                values[key].append(delta)
    old, new = metrics(total_values(old_rows)), metrics(total_values(new_rows))
    return {key: {"delta": difference(old[key], new[key]),
                  "delta_ci95": interval(draws), "valid_draws": len(draws)}
            for key, draws in values.items()}


def online_comparison(old, new):
    prefixes = sorted({row["prefix"] for row in old})
    gammas = sorted({row["gamma"] for row in old if row["mode"] == "draft"})
    combinations = ([(prefix, gamma) for prefix in prefixes for gamma in gammas]
                    + [(prefix, None) for prefix in prefixes]
                    + [(None, gamma) for gamma in gammas] + [(None, None)])
    groups = []
    for prefix, gamma in combinations:
        sides, targets = [], []
        for source in (old, new):
            sides.append([row for row in source if row["mode"] == "draft"
                          and (prefix is None or row["prefix"] == prefix)
                          and (gamma is None or row["gamma"] == gamma)])
            targets.append([row for row in source if row["mode"] == "target_only"
                            and (prefix is None or row["prefix"] == prefix)])
        a, b = [summarize(rows, target) for rows, target in zip(sides, targets)]
        groups.append({"prefix": prefix, "gamma": gamma, "old": a, "new": b,
                       "paired": paired_online(*sides),
                       "delta_counts": {field: b["counts"][field] - a["counts"][field]
                                        for field in FIELDS},
                       "delta_speedup_vs_target_only": difference(
                           a["speedup_vs_target_only"], b["speedup_vs_target_only"]),
                       "delta_below_30pct_runs": {
                           kind: {denominator: b["below_30pct_runs"][kind][denominator]["count"]
                                  - a["below_30pct_runs"][kind][denominator]["count"]
                                  for denominator in ("verified", "proposed")} for kind in KINDS}})
    return groups


def training_history(data):
    if not isinstance(data, list):
        raise ValueError("history 必须为列表")
    by_step = {}
    for row in data:
        step = number(row["step"], "step", integer=True)
        if step in by_step:
            raise ValueError("history 全局 step 重复")
        for key in ("feature", "position"):
            if number(row[key], key) > 1:
                raise ValueError("history overlap 超出 0 到 1")
        by_step[step] = row
    return {"steps": [{"step": step, "available": True, "record": by_step[step]}
                      for step in sorted(by_step)]}


def make_summary(old, new, history):
    a, b = offline_data(old, "旧模型"), offline_data(new, "新模型")
    if set(a["all"]) != set(b["all"]):
        raise ValueError("旧新模型固定 test50 场景不一致，无法进行配对比较")
    fixed_contexts = []
    for source in (old, new):
        fixed_contexts.append({(row["kind"], row["name"]): (row["context"], row["token_count"])
                               for row in source["offline"]["per_sequence"]})
    if fixed_contexts[0] != fixed_contexts[1]:
        raise ValueError("旧新固定前缀的 context 或 token_count 不一致，无法按同一输入比较")
    x, y = online_data(old, "旧模型"), online_data(new, "新模型")
    if {row["key"] for row in x} != {row["key"] for row in y}:
        raise ValueError("旧新模型在线场景、prefix 或 Gamma 不一致，无法进行配对比较")
    warnings = []
    settings = ("policy", "target_precision", "draft_precision", "source_root", "model_config")
    for key in settings:
        if old.get(key) != new.get(key):
            warnings.append(f"旧新 {key} 不一致，差值可能同时受到评测条件影响。")
    for key in ("tokens", "seed", "online_scenes"):
        if old["parameters"].get(key) != new["parameters"].get(key):
            warnings.append(f"旧新 parameters.{key} 不一致。")
    for key in ("policy", "scope"):
        if old["offline"].get(key) != new["offline"].get(key):
            warnings.append(f"旧新固定前缀 {key} 不一致。")
    other = {row["key"]: row for row in y}
    for key in ("seed", "requested_tokens"):
        if any(row[key] != other[row["key"]][key] for row in x):
            warnings.append(f"部分同场景在线运行的 {key} 不一致。")
    if any(a[kind][name][1] != b[kind][name][1] for kind in KINDS for name in a[kind]):
        warnings.append("固定前缀的部分场景样本数不同；各模型按自身 sample_count 加权，配对单位仍为场景。")
    records = training_history(history)
    if not records["steps"]:
        warnings.append("history 为空，未提供 val 记录。")
    return {"units": {"rates_and_rate_deltas": "fraction", "speed": "tokens/second",
                      "delta_direction": "new_minus_old", "markdown_rate_deltas": "percentage_points"},
            "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "paired_scene",
                           "method": "percentile", "confidence": 0.95},
            "notes": NOTES, "warnings": warnings,
            "models": {tag: {key: source.get(key) for key in (
                "checkpoint", "checkpoint_step", "policy", "parameters", "timings",
                "source_root", "target_precision", "draft_precision")}
                       for tag, source in (("old", old), ("new", new))},
            "fixed_test50": offline_comparison(a, b),
            "online": {"groups": online_comparison(x, y)}, "history": records}


def fmt(value, percent=False, signed=False):
    if value is None:
        return "无定义"
    value = value * 100 if percent else value
    return format(value, "+.4f" if signed else ".4f") + ("%" if percent else "")


def ci_text(values, percent=True):
    scale = 100 if percent else 1
    if any(value is None for value in values):
        return "无定义"
    return f"[{values[0] * scale:+.4f}, {values[1] * scale:+.4f}]"


def direction(values):
    if any(value is None for value in values):
        return "有效样本不足"
    if values[0] > 0:
        return "区间高于 0"
    if values[1] < 0:
        return "区间低于 0"
    return "区间包含 0"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows]


def group_name(group):
    prefix = "合并 prefix" if group["prefix"] is None else str(group["prefix"])
    gamma = "合并 Gamma" if group["gamma"] is None else str(group["gamma"])
    return prefix, gamma


def markdown(summary):
    lines = ["# GaussianGPT 续训结果对比", "", "差值方向：新模型 − 旧模型。本文只描述已经完成的评测。", ""]
    for tag, label in (("old", "旧模型"), ("new", "新模型")):
        model = summary["models"][tag]
        lines += [f"{label}：`{model['checkpoint']}`；全局 step={model['checkpoint_step']}。", ""]
    lines += ["## 比较口径", ""] + [f"- {note}" for note in summary["notes"]] + [""]
    if summary["warnings"]:
        lines += ["## 输入限制", ""] + [f"- {warning}" for warning in summary["warnings"]] + [""]
    lines += ["## 固定前缀 test50 overlap", "",
              "从 per_sequence 的 overlap_sum 与 sample_count 重算；每次抽取 50 个场景并允许重复，旧新使用同一组抽样场景。零样本条目保留场景身份，不贡献分子或分母。", ""]
    rows = []
    for row in summary["fixed_test50"]:
        if row["kind"] not in ("feature", "position"):
            continue
        rows.append([LABELS[row["kind"]], row["scene_count"],
                     f"{row['old']['sample_count']} / {row['new']['sample_count']}",
                     fmt(row["old"]["pooled_overlap"], True), fmt(row["new"]["pooled_overlap"], True),
                     fmt(row["delta"] * 100 if row["delta"] is not None else None, signed=True),
                     ci_text(row["delta_ci95"]), direction(row["delta_ci95"]), row["valid_draws"]])
    lines += table(["类型", "场景数", "旧/新样本数", "旧 overlap", "新 overlap", "差值（百分点）",
                    "配对差值 95% CI（百分点）", "方向", "有效抽样次数"], rows) + [""]
    lines += ["## 在线接受率", "",
              "保留各 prefix/Gamma 组合及合并结果。P 表示 Position token；V=accepted/verified，Q=accepted/proposed，差值与 CI 单位为百分点。完整类型、原始计数及抽样次数保存在 continuation_summary.json。", ""]
    groups = summary["online"]["groups"]
    for kind in ("feature", "position"):
        lines += [f"### {LABELS[kind]}", ""]
        rows = []
        for group in groups:
            cells = list(group_name(group))
            for denominator in ("verified", "proposed"):
                key = f"{kind}_accepted_over_{denominator}"
                paired = group["paired"][key]
                cells += [fmt(group["old"]["metrics"][key], True),
                          fmt(group["new"]["metrics"][key], True),
                          fmt(paired["delta"] * 100 if paired["delta"] is not None else None, signed=True),
                          ci_text(paired["delta_ci95"])]
            rows.append(cells)
        lines += table(["prefix", "Gamma", "旧 V", "新 V", "ΔV", "ΔV 配对 95% CI",
                        "旧 Q", "新 Q", "ΔQ", "ΔQ 配对 95% CI"], rows) + [""]
    lines += ["## Feature 低于 30% 的运行数", "",
              "严格小于 30%；按单次场景/prefix/Gamma 运行分别计数。表中为低于阈值的次数 / 有定义的次数；分母为 0 的比值不计入有定义次数。", ""]
    rows = []
    for group in groups:
        cells = list(group_name(group))
        for denominator in ("verified", "proposed"):
            for tag in ("old", "new"):
                count = group[tag]["below_30pct_runs"]["feature"][denominator]
                cells.append(f"{count['count']}/{count['defined_runs']}")
        rows.append(cells)
    lines += table(["prefix", "Gamma", "旧 V 次数", "新 V 次数", "旧 Q 次数", "新 Q 次数"], rows) + [""]
    lines += ["## 在线速度", "",
              "tokens/s 按总 token / 总秒数计算；倍速使用各模型本次同 prefix 集合的 Target-only 速度。配对差值区间仅反映场景重采样的不确定性。", ""]
    rows = []
    for group in groups:
        if group["prefix"] is None or group["gamma"] is None:
            continue
        a, b = group["old"], group["new"]
        paired = group["paired"]["tokens_per_second"]
        rows.append([*group_name(group),
                     fmt(a["metrics"]["tokens_per_second"]), fmt(b["metrics"]["tokens_per_second"]),
                     fmt(paired["delta"], signed=True), ci_text(paired["delta_ci95"], False),
                     fmt(a["speedup_vs_target_only"]), fmt(b["speedup_vs_target_only"])])
    lines += table(["prefix", "Gamma", "旧 tokens/s", "新 tokens/s", "差值 tokens/s", "配对差值 95% CI",
                    "旧倍速", "新倍速"], rows) + [""]
    lines += ["## 续训 val 记录", "", "按输入 history 中的全局 step 升序展示；仅使用已有记录，不从 test 推断或补造节点。", ""]
    rows = []
    for entry in summary["history"]["steps"]:
        row = entry["record"]
        rows.append([entry["step"], fmt(row["feature"], True) if row else "缺失",
                     fmt(row["position"], True) if row else "缺失"])
    lines += table(["全局 step", "val Feature overlap", "val P overlap"], rows) + [""]
    lines += ["## 训练进度曲线", "",
              "Validation overlap (expected single-step acceptance)：在已保存的 Target 轨迹前缀上衡量单步期望接受程度；在线生成上下文中的实际接受率另见前缀曲线。", ""]
    training_plot = summary["training_plot"]
    if training_plot["status"] == "created":
        lines += ["![实际 history 的 validation F/P overlap](training_progress.png)", "",
                  "PDF：training_progress.pdf。", ""]
    else:
        lines += [training_plot["reason"], ""]
    plot = summary["plot"]
    lines += ["## 前缀曲线", ""]
    if plot["status"] == "created":
        lines += ["两面板分别为 Feature 与 P 的 accepted/verified；颜色表示 Gamma，线型和标记表示模型；不叠加数值标签。", "",
                  "![旧新模型在线前缀曲线](continuation_trend.png)", "",
                  "PDF：continuation_trend.pdf。", ""]
    else:
        lines += [plot["reason"], ""]
    return "\n".join(lines)


def plot_modules():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise RuntimeError("当前 Python 缺少 matplotlib；可使用 --no-plot 生成文字与 JSON。") from exc
        raise
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    return plt, PercentFormatter


def plot_trend(summary, output_dir):
    plt, PercentFormatter = plot_modules()

    groups = [group for group in summary["online"]["groups"]
              if group["prefix"] is not None and group["gamma"] is not None]
    gammas = sorted({group["gamma"] for group in groups})
    prefixes = sorted({group["prefix"] for group in groups})
    colors = {gamma: plt.get_cmap("tab10")(i) for i, gamma in enumerate(gammas)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    try:
        for ax, kind, title in zip(axes, ("feature", "position"), ("Feature", "P (Position)")):
            values = []
            for gamma in gammas:
                rows = sorted([group for group in groups if group["gamma"] == gamma],
                              key=lambda group: group["prefix"])
                for tag, label, style, marker, size in (
                        ("old", "Old", "--", "o", 8), ("new", "New", "-", "x", 6)):
                    y = [group[tag]["metrics"][f"{kind}_accepted_over_verified"] for group in rows]
                    values.extend(value for value in y if value is not None)
                    ax.plot([group["prefix"] for group in rows],
                            [value if value is not None else float("nan") for value in y],
                            color=colors[gamma], linestyle=style, marker=marker, markersize=size,
                            markerfacecolor="none", linewidth=2, label=f"Gamma={gamma} / {label}")
            if kind == "feature":
                ax.axhline(0.30, color="0.55", linestyle=":", linewidth=1)
                values.append(0.30)
            if values:
                margin = max(0.015, (max(values) - min(values)) * 0.15)
                ax.set_ylim(max(0, min(values) - margin), min(1, max(values) + margin))
            ax.set_title(title)
            ax.set_xlabel("Prefix length (tokens)")
            ax.set_ylabel("Accepted / verified")
            ax.set_xticks(prefixes)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(gammas) * 2, frameon=False,
                   bbox_to_anchor=(0.5, 0.055))
        count = groups[0]["old"]["scene_count"]
        fig.suptitle(f"Continuation comparison: {count} scenes, model-specific generated contexts", fontsize=12)
        fig.text(0.5, 0.02, "Same seed does not imply the same trajectory; paired units are scenes.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.13, 1, 0.94))
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"continuation_trend.{extension}", dpi=200)
    finally:
        plt.close(fig)
    return {"status": "created", "files": ["continuation_trend.png", "continuation_trend.pdf"]}


def plot_training(summary, outdir):
    entries = sorted(summary["history"]["steps"], key=lambda entry: entry["step"])
    if not entries:
        return {"status": "skipped", "files": [], "reason": "history 为空，未生成训练进度图。"}
    plt, PercentFormatter = plot_modules()
    steps = [entry["step"] for entry in entries]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    try:
        series = []
        for ax, kind, title, color in zip(
                axes, ("feature", "position"), ("Feature", "P (Position)"), ("#1565c0", "#d97706")):
            values = [entry["record"][kind] for entry in entries]
            series.append(values)
            ax.plot(steps, values, "o-", color=color, linewidth=2, markersize=5)
            padding = max(0.005, (max(values) - min(values)) * 0.25)
            ax.set_ylim(max(0, min(values) - padding), min(1, max(values) + padding))
            ax.margins(x=0.10)
            ax.set_title(title)
            ax.set_xlabel("Global step")
            ax.set_ylabel("Validation overlap (%)")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Validation overlap (expected single-step acceptance)", fontsize=13)
        fig.text(0.5, 0.02, "Actual history points on saved Target trajectories; validation only.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, values in zip(axes, series):
            occupied = []
            for step, value in zip(steps, values):
                label = ax.annotate(f"{value * 100:.2f}%", (step, value), xytext=(0, 10),
                                    textcoords="offset points", ha="center", va="center", fontsize=8)
                chosen, best_score = (0, 10), float("inf")
                for dy in (12, -16, 26, -30, 40, -44, 54, -58, 68, -72):
                    for dx in (0, -24, 24, -48, 48):
                        label.set_position((dx, dy))
                        box = label.get_window_extent(renderer).expanded(1.08, 1.20)
                        outside = not (ax.bbox.contains(box.x0, box.y0)
                                       and ax.bbox.contains(box.x1, box.y1))
                        overlaps = sum(box.overlaps(other) for other in occupied)
                        score = 1000000 * outside + 10000 * overlaps + abs(dx) + abs(dy)
                        if score < best_score:
                            chosen, best_score = (dx, dy), score
                        if not outside and not overlaps:
                            break
                    if not outside and not overlaps:
                        break
                label.set_position(chosen)
                occupied.append(label.get_window_extent(renderer).expanded(1.08, 1.20))
        for extension in ("png", "pdf"):
            fig.savefig(outdir / f"training_progress.{extension}", dpi=200)
    finally:
        plt.close(fig)
    return {"status": "created", "files": ["training_progress.png", "training_progress.pdf"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new", "history", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true", help="仅生成文字与 JSON，无需 matplotlib")
    args = parser.parse_args(argv)
    try:
        summary = make_summary(read_json(args.old), read_json(args.new), read_json(args.history))
        summary["inputs"] = {key: str(getattr(args, key).resolve()) for key in ("old", "new", "history")}
        outputs = [args.output_dir / name for name in (
            "continuation_results.md", "continuation_summary.json", "continuation_trend.png", "continuation_trend.pdf",
            "training_progress.png", "training_progress.pdf")]
        if any(path.resolve() == source.resolve() for path in outputs
               for source in (args.old, args.new, args.history)):
            raise ValueError("输出文件不能覆盖输入文件")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.no_plot:
            summary["plot"] = {"status": "skipped", "files": [],
                               "reason": "本次指定 --no-plot，未生成图片；已有同名图片不代表本次结果。"}
            summary["training_plot"] = dict(summary["plot"])
        else:
            summary["plot"] = plot_trend(summary, args.output_dir)
            summary["training_plot"] = plot_training(summary, args.output_dir)
        files = (["continuation_results.md", "continuation_summary.json"]
                 + summary["plot"]["files"] + summary["training_plot"]["files"])
        summary["files"] = files
        (args.output_dir / "continuation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output_dir / "continuation_results.md").write_text(markdown(summary), encoding="utf-8")
        print(json.dumps({"files": [str((args.output_dir / name).resolve()) for name in files],
                          "warnings": summary["warnings"]}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"无法生成续训对比：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
