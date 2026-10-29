"""从本地实验 JSON 生成中文报告和图表；运行时仅依赖 Python 与 matplotlib。

用法：python pf_ranker_report.py --input results.json --output-dir report
统计以 online.rows 为准；已有 summary 和 by_k 提供可核对的区间。
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


KINDS = ("position", "feature", "eos")
COUNTS = ("accepted", "verified", "proposed")


def ratio(a, b):
    return a / b if a is not None and b else None


def number(value, field, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (integer and int(value) != value)):
        raise ValueError(f"{field} 需要有限的非负数值")
    return int(value) if integer else value


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("online.rows 需要包含测量记录")
    for index, row in enumerate(rows):
        label = f"online.rows[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} 需要为对象")
        if row.get("mode") not in ("draft", "target_only"):
            raise ValueError(f"{label}.mode 需要为 draft 或 target_only")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError(f"{label}.name 缺少场景名")
        for key in ("prompt_tokens", "generated_tokens"):
            number(row.get(key), f"{label}.{key}", True)
        if number(row.get("seconds"), f"{label}.seconds") <= 0:
            raise ValueError(f"{label}.seconds 需要大于零")
        for key in ("seed", "requested_tokens", "k", "proposal_width"):
            if row.get(key) is not None:
                number(row[key], f"{label}.{key}", True)
        for key in ("target_calls", "draft_calls", "ranker_calls", "target_input_tokens",
                    "draft_input_tokens", "rounds", "position_fallbacks"):
            if row.get(key) is not None:
                number(row[key], f"{label}.{key}", True)
        if row["mode"] == "draft":
            if not row.get("k"):
                raise ValueError(f"{label}.k 需要为正整数")
            for kind in KINDS[:2]:
                for key in COUNTS:
                    number(row.get(f"{kind}_{key}"), f"{label}.{kind}_{key}", True)
            for kind in ("", *KINDS):
                stem = kind + "_" if kind else ""
                keys = [stem + key for key in COUNTS]
                if any(key in row for key in keys):
                    values = [number(row.get(key), f"{label}.{key}", True) for key in keys]
                    if not values[0] <= values[1] <= values[2]:
                        raise ValueError(f"{label}.{stem} 计数需要满足 accepted ≤ verified ≤ proposed")
            for key in COUNTS:
                if key in row and all(f"{kind}_{key}" in row for kind in KINDS):
                    if row[key] != sum(row[f"{kind}_{key}"] for kind in KINDS):
                        raise ValueError(f"{label}.{key} 与分类型总数不符")
        if row.get("requested_tokens") is not None and row["generated_tokens"] > row["requested_tokens"]:
            raise ValueError(f"{label} 生成长度超出预算")


def totals(rows):
    result = {"run_count": len(rows), "scene_count": len({r["name"] for r in rows}),
              "prompt_count": len({(r["name"], r["prompt_tokens"]) for r in rows})}
    fields = ("generated_tokens", "seconds", "target_calls", "draft_calls", "ranker_calls",
              "target_input_tokens", "draft_input_tokens", "rounds", "position_fallbacks",
              *COUNTS, *(f"{kind}_{key}" for kind in KINDS for key in COUNTS))
    for key in fields:
        values = [r.get(key) for r in rows]
        result[key] = (math.fsum(values) if key == "seconds" else sum(values)) if (
            rows and all(value is not None for value in values)) else None
    result["tokens_per_second"] = ratio(result["generated_tokens"], result["seconds"])
    for kind in ("", *KINDS):
        stem = kind + "_" if kind else ""
        a, v, p = (result[stem + key] for key in COUNTS)
        result[stem + "acceptance_verified"] = ratio(a, v)
        result[stem + "acceptance_proposed"] = ratio(a, p)
        result[stem + "unverified"] = p - v if p is not None and v is not None else None
    result["eos_runs"] = sum(bool(r.get("eos")) for r in rows)
    return result


def identity(row):
    return row["name"], row["prompt_tokens"], row.get("seed"), row.get("requested_tokens")


def paired_speed(rows, direct):
    selected = defaultdict(list)
    references = defaultdict(list)
    for row in rows:
        selected[identity(row)].append(row)
    for row in direct:
        references[identity(row)].append(row)
    paired, matched = [], []
    ambiguous = 0
    for key, candidates in selected.items():
        if len(candidates) != 1:
            ambiguous += len(candidates)
            continue
        row = candidates[0]
        options = references.get(key, [])
        exact = [r for r in options if r.get("k") == row["k"]]
        options = exact or [r for r in options if r.get("k") in (None, 0)]
        if len(options) == 1:
            paired.append(row)
            matched.append(options[0])
        elif options:
            ambiguous += 1
    a, b = totals(paired), totals(matched)
    return {"matched_runs": len(paired), "draft_runs": len(rows),
            "unmatched_runs": len(rows) - len(paired), "ambiguous_runs": ambiguous,
            "draft": a, "target_only": b,
            "speedup_vs_target_only": ratio(a["tokens_per_second"], b["tokens_per_second"])}


def width_unit(data, requested):
    if requested != "auto":
        return requested
    params = data.get("parameters") or {}
    if (data.get("k_definition") or data.get("by_k") is not None or "k" in params
            or any("proposal_width" in r for r in data["online"]["rows"] if r["mode"] == "draft")):
        return "position"
    return "gamma" if "gamma" in params else "position"


def source_intervals(group, source, warnings):
    if not isinstance(source, dict):
        return
    keys = ("run_count", "generated_tokens", "seconds",
            *(f"{kind}_{key}" for kind in KINDS[:2] for key in COUNTS))
    if any(key in source and (group.get(key) is None or not math.isclose(
            source[key], group[key], rel_tol=1e-9, abs_tol=1e-9)) for key in keys):
        warnings.append(f"{group['label']} 的输入汇总与 rows 不符，未采用该汇总的区间。")
        return
    if not all(key in source for key in keys):
        warnings.append(f"{group['label']} 的输入汇总缺少计数或耗时，区间未采用。")
        return
    supplied = (source.get("bootstrap") or {}).get("ci95") or {}
    group["ci95"] = {key: value for key, value in supplied.items()
                       if key in ("tokens_per_second", "position_acceptance_verified",
                                  "position_acceptance_proposed", "feature_acceptance_verified",
                                  "feature_acceptance_proposed") and valid_interval(value)}
    pair = group["paired_speed"]
    speed = source.get("speedup_vs_target_only")
    interval = ((source.get("paired_speedup_bootstrap") or {}).get("ci95") or {}).get("speedup_vs_target_only")
    if (pair["unmatched_runs"] == 0 and speed is not None and pair["speedup_vs_target_only"] is not None
            and math.isclose(speed, pair["speedup_vs_target_only"], rel_tol=1e-9, abs_tol=1e-9)
            and valid_interval(interval)):
        group["speedup_ci95"] = interval
        group["interval_method"] = (source.get("paired_speedup_bootstrap") or {}).get("method")


def valid_interval(value):
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in value)
            and value[0] <= value[1])


def build_summary(data, source_path, unit="auto"):
    if not isinstance(data, dict) or not isinstance(data.get("online"), dict):
        raise ValueError("输入需要包含 online 对象")
    online = data["online"]
    rows = online.get("rows")
    validate_rows(rows)
    unit = width_unit(data, unit)
    direct = [r for r in rows if r["mode"] == "target_only"]
    drafts = [r for r in rows if r["mode"] == "draft"]
    warnings = []
    if not drafts:
        warnings.append("输入尚无 draft 测量记录。")
    if data.get("status") != "complete":
        warnings.append("JSON 未声明实验完成；本报告仅描述已记录数据。")
    if any(r.get("seed") is None or r.get("requested_tokens") is None for r in rows):
        warnings.append("部分行未记录 seed 或预算；配对要求双方对应字段一致，仍无法确认完整运行条件。")
    params = data.get("parameters") or {}
    ks = sorted({r["k"] for r in drafts})
    groups, by_k = [], []
    source_k = data.get("by_k") or online.get("by_k") or []
    source_groups = (online.get("summary") or {}).get("groups") or []

    def make_group(selected, k, prefix=None):
        width = 2 * k - 1 if unit == "position" else k
        label = f"{'K' if unit == 'position' else 'Gamma'}={k}"
        if prefix is not None:
            label += f"，前缀={prefix}"
        for row in selected:
            if row.get("proposal_width") is not None and row["proposal_width"] != width:
                raise ValueError(f"{label} 的 proposal_width 与宽度定义不符")
        group = dict(totals(selected), k=k, proposal_width=width, label=label,
                     paired_speed=paired_speed(selected, direct), ci95={}, speedup_ci95=None)
        if prefix is not None:
            group["prompt_tokens"] = prefix
            candidates = [r for r in source_groups if r.get("mode") == "draft"
                          and r.get("prompt_tokens") == prefix and r.get("k", r.get("gamma")) == k]
        else:
            candidates = [r for r in source_k if r.get("k") == k]
        if len(candidates) == 1:
            source_intervals(group, candidates[0], warnings)
        if group["paired_speed"]["unmatched_runs"]:
            warnings.append(f"{label} 存在缺失或重复配对，速度比仅使用可唯一配对的记录。")
        return group

    for k in ks:
        selected = [r for r in drafts if r["k"] == k]
        missing = {identity(r) for r in direct} - {identity(r) for r in selected}
        if missing:
            warnings.append(f"提案参数 {k} 缺少 {len(missing)} 个 direct 对应的 draft 记录。")
        by_k.append(make_group(selected, k))
        for prefix in sorted({r["prompt_tokens"] for r in selected}):
            groups.append(make_group([r for r in selected if r["prompt_tokens"] == prefix], k, prefix))
    scene_count = len({r["name"] for r in rows})
    prompt_count = len({(r["name"], r["prompt_tokens"]) for r in rows})
    if online.get("scene_count") is not None and online["scene_count"] != scene_count:
        warnings.append("online.scene_count 与已记录场景数不同，报告采用 rows 中的实际数量。")
    expected_ks = params.get("k") if unit == "position" else params.get("gamma")
    if isinstance(expected_ks, list) and set(expected_ks) != set(ks):
        warnings.append("已记录提案参数与 parameters 中的计划不同。")
    multiplicities = Counter((r["mode"], r.get("k", 0), identity(r)) for r in rows)
    if any(n > 1 for n in multiplicities.values()):
        warnings.append("发现重复运行标识；接收率保留全部测量记录，重复记录的速度配对需进一步确认。")
    expected_scenes = params.get("scenes", params.get("online_scenes"))
    prefixes = online.get("requested_prefix_tokens", params.get("prefixes"))
    if expected_scenes is not None and expected_scenes != scene_count:
        warnings.append("已记录场景数与计划不同。")
    if isinstance(prefixes, list) and expected_scenes is not None:
        if expected_scenes * len(prefixes) != prompt_count:
            warnings.append("已记录 prompt 数与计划不同；可能存在未运行或不可用前缀。")
    if online.get("unavailable_prefixes"):
        warnings.append("输入记录了不可用前缀，实际覆盖范围需按 rows 解读。")
    policy = data.get("policy") or online.get("policy") or {}
    metadata = {"scene_count": scene_count, "prompt_count": prompt_count, "run_count": len(rows),
                "prefix_tokens": sorted({r["prompt_tokens"] for r in rows}),
                "expected_scenes": expected_scenes, "expected_k": expected_ks,
                "target_precision": data.get("target_precision"),
                "draft_precision": data.get("draft_precision"),
                "temperature": policy.get("temperature", policy.get("temp")),
                "top_p": policy.get("top_p"), "status": data.get("status"),
                "split": sorted({str(r.get("split", data.get("split", "未记录"))) for r in rows}),
                "checkpoint_step": data.get("checkpoint_step"), "scope": online.get("scope"),
                "position_rank_identity": data.get("position_rank_identity"),
                "validation_status": (online.get("validation") or {}).get("status")}
    if metadata["target_precision"] != "FP32" or metadata["draft_precision"] != "FP32":
        warnings.append("目标实验要求 FP32；JSON 精度字段缺失或与要求不同。")
    if metadata["temperature"] is None or metadata["top_p"] is None:
        warnings.append("JSON 未完整记录 temperature 与 top_p。")
    return {"input": str(source_path), "width_unit": unit, "metadata": metadata,
            "definitions": {"acceptance_verified": "sum(accepted) / sum(verified)",
                            "acceptance_proposed": "sum(accepted) / sum(proposed)",
                            "throughput": "sum(generated_tokens) / sum(seconds)",
                            "speedup": "paired draft throughput / paired target_only throughput",
                            "pair_fields": ["name", "prompt_tokens", "seed", "requested_tokens"]},
            "by_k": by_k, "groups": groups, "target_only": totals(direct),
            "warnings": list(dict.fromkeys(warnings))}


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if value is not None else "—"


def fraction(group, kind, denominator):
    a, b = group[kind + "_accepted"], group[kind + "_" + denominator]
    value = group[kind + "_acceptance_" + denominator]
    return f"{a}/{b}（{value:.2%}）" if value is not None else f"{a if a is not None else '—'}/{b if b is not None else '—'}（无可用比例）"


def interval_text(value):
    return f"[{fmt(value[0])}, {fmt(value[1])}]" if value else "未提供"


def tables(groups, unit):
    label = "K" if unit == "position" else "Gamma"
    lines = [f"| {label} | 最大提案宽度 | 运行数 | P accepted/verified | P accepted/proposed | F accepted/verified | F accepted/proposed |",
             "|---|---|---|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g['k']} | {g['proposal_width']} | {g['run_count']} | "
                     + " | ".join(fraction(g, kind, den) for kind in KINDS[:2]
                                  for den in ("verified", "proposed")) + " |")
    lines += ["", f"| {label} | 总生成 token / 总秒数 | 全部 draft token/s | 配对 draft token/s | 配对 direct token/s | 速度比 | 配对数/运行数 | 速度比 95% 区间 |",
              "|---|---|---|---|---|---|---|---|"]
    for g in groups:
        p = g["paired_speed"]
        lines.append(f"| {g['k']} | {g['generated_tokens']} / {fmt(g['seconds'])} | {fmt(g['tokens_per_second'])} | "
                     f"{fmt(p['draft']['tokens_per_second'])} | {fmt(p['target_only']['tokens_per_second'])} | "
                     f"{fmt(p['speedup_vs_target_only'])} | {p['matched_runs']}/{p['draft_runs']} | "
                     f"{interval_text(g['speedup_ci95'])} |")
    return lines


def markdown(summary):
    m, unit = summary["metadata"], summary["width_unit"]
    precision = lambda value: str(value) if value is not None else "未记录（预期 FP32）"
    lines = ["# P/F 在线实验报告", ""]
    measured = [g for g in summary["by_k"] if g["paired_speed"]["speedup_vs_target_only"] is not None]
    if measured:
        best = max(measured, key=lambda g: g["paired_speed"]["speedup_vs_target_only"])
        lines += [f"整体吞吐最高：{best['label']}，配对速度比 {fmt(best['paired_speed']['speedup_vs_target_only'])}×，"
                  f"95% 场景重采样区间 {interval_text(best['speedup_ci95'])}。", ""]
    lines += [
             f"实际范围：{m['scene_count']} 场景 / {m['prompt_count']} 个 prompt / {m['run_count']} 次测量。",
             "prompt 按场景名与前缀长度去重；同一 prompt 的不同 K 或 seed 共享这一标识。",
             f"前缀长度：{', '.join(str(v) for v in m['prefix_tokens'])}；数据划分：{', '.join(m['split'])}。",
             f"Target 精度：{precision(m['target_precision'])}；Feature draft 精度：{precision(m['draft_precision'])}。",
             f"temperature：{fmt(m['temperature'])}；top_p：{fmt(m['top_p'])}；实验状态：{m['status'] or '未记录'}。",
             f"使用已有 step {m['checkpoint_step']} 权重。本轮只测试解码，未重新训练。",
             "Position 提案来自既有 FastHistRanker；GPT 草稿只提出 Feature，仍读取 P/F 因果上下文。",
             f"输入：`{summary['input']}`。", "", "## 统计口径", "",
             "P 表示 Position，F 表示 Feature。按类型先累加计数，再计算比例。表格同时列出实际分子、分母与接收率。",
             "verified 仅包括首次拒绝之前及拒绝位置的实际验证提案；拒绝之后的未使用提案保留在 proposed 中。",
             "accepted/verified 描述到达该类型提案时的条件接收率；accepted/proposed 描述全部提案的利用率。",
             "旧联合 Gamma 描述每轮 P/F 联合 token 提案宽度。新 K 描述 Position 提案数，最大总提案宽度为 2K−1；预算或可用 Position 不足时实际宽度更短。",
             "参考旧文件也记录 accepted/verified；参数宽度定义与接收率分母需要分别解读。",
             "EOS 单独计数；目标补充 token 与拒绝后的重新采样 token 均不计入 accepted。",
             "EOS 结束运行统计输出终止情况；EOS 提案计数只覆盖主动提出 EOS 的提案。本次 Position ranker 提出普通 Position，EOS 提案计数可为零。",
             "吞吐量 = 总生成 token / 总耗时；速度比 = 配对 draft 吞吐量 / 配对 target_only 吞吐量。",
             "速度配对要求场景、前缀、seed 和生成预算一致；优先采用同 K direct，其次采用共用 direct。重复或缺失配对不进入速度比。", ""]
    if unit == "position":
        lines += ["本报告按 Position K 解读；summary.groups.gamma 在本次实验中同样表示 K。",
                  "本次 PF 与 target_only 共用 typed sparse head。旧联合测试使用 fullhead，旧速度与本次速度存在实现差异；结论以本次配对为主。"]
    else:
        lines += ["当前输入识别为旧联合实验，表格按 Gamma 展示原有联合 token 宽度。",
                  "该输入用于结构参考，提供旧 fullhead 测量结果；本次 typed sparse head 实验的速度需由新 JSON 给出。"]
    lines += ["", "## 按提案参数汇总", "", *tables(summary["by_k"], unit), "", "## 前缀分组", ""]
    for prefix in m["prefix_tokens"]:
        selected = [g for g in summary["groups"] if g["prompt_tokens"] == prefix]
        lines += [f"### 前缀 {prefix}", "", *tables(selected, unit), ""]
    lines += ["## 图表", "", "![P/F 接收率与吞吐速度比](acceptance_speed.png)", "",
              "图中接收率对应累加后的比例；PDF 为同图的矢量版本。区间仅使用输入已有且与当前计数和配对速度一致的统计项。",
              "输入未提供可用区间时仅显示点值。速度图区间对应输入 paired_speedup_bootstrap.ci95；统计覆盖率沿用输入定义。",
              "接收率区间保留在 summary.json 的 ci95 字段。", "", "## 测量限制与设计疑点", "",
              "同一场景的多个前缀共享内容，统计推断宜按场景共同重采样；逐 token 独立抽样会低估不确定性。",
              "实际 verified 受早期拒绝筛选；接收率随 K 的变化同时受生成轨迹、到达位置和提案类型影响。",
              "各 K 共用同一权重，Feature 条件接收率的差异不能作为训练进步的证据。",
              "固定 seed 可以复现一次随机流；不同模式消耗随机数的顺序不同，生成轨迹和 EOS 长度可能不同。",
              "场景子集的代表性与单次运行的计时波动限制速度结论；短测仅用于运行验证。",
              "计时包含缓存分配、前缀处理、提案、验证和拒绝后采样；本次 PF 还包含 Position 状态处理。模型加载与数值验证耗时独立记录。",
              "跨实验速度比较还需确认硬件、计时范围、采样策略和输出长度一致；JSON 未记录的条件无法由报告确认。",
              "FP32 标签来自 JSON 精度字段，报告工具不加载模型，无法独立确认运行时张量精度。",
              "报告工具核对计数与预算，不执行模型或验证输出序列；缓存、首次拒绝与 Position 状态的正确性需由 eval 验证支持。", ""]
    if unit == "position":
        lines.append("当前 eval 轮换 direct 与各 K 的计时顺序以减少顺序影响；同一场景仍只有一组 seed，缺少多次独立计时。")
    else:
        lines.append("参考旧 eval 使用固定计时顺序，测量可能受顺序影响。")
    for g in summary["by_k"]:
        lines.append(f"{g['label']}：P 未实际验证 {g['position_unverified']}，F 未实际验证 {g['feature_unverified']}；"
                     f"EOS 接收/实际验证/提案 {g['eos_accepted']}/{g['eos_verified']}/{g['eos_proposed']}；"
                     f"EOS 结束运行 {g['eos_runs']}。")
    if summary["warnings"]:
        lines += ["", "### 数据提示", "", *(f"- {item}" for item in summary["warnings"])]
    return "\n".join(lines) + "\n"


def plot(summary, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("生成图表需要 matplotlib；请在已具备该依赖的环境运行") from error
    groups = summary["by_k"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    label = "K" if summary["width_unit"] == "position" else "Gamma"
    x = list(range(len(groups)))
    colors = ("#2166ac", "#67a9cf", "#b35806", "#f1a340")
    for offset, (kind, den) in enumerate((k, d) for k in KINDS[:2] for d in ("verified", "proposed")):
        points = [(i, g[f"{kind}_acceptance_{den}"]) for i, g in enumerate(groups)
                  if g[f"{kind}_acceptance_{den}"] is not None]
        bars = axes[0].bar([i + (offset - 1.5) * .18 for i, _ in points], [v for _, v in points],
                           width=.18, label=f"{'P' if kind == 'position' else 'F'} accepted/{den}", color=colors[offset])
        axes[0].bar_label(bars, labels=[f"{v:.1%}" for _, v in points],
                          padding=-13 if offset % 2 else 3, fontsize=8)
    axes[0].set(title="Acceptance: actual counts", ylabel="Acceptance fraction", ylim=(0, 1.12))
    if groups:
        axes[0].legend(fontsize=9, loc="lower left")
    for i, g in enumerate(groups):
        value = g["paired_speed"]["speedup_vs_target_only"]
        if value is None:
            continue
        ci = g["speedup_ci95"]
        axes[1].plot(i, value, "o", color="#2166ac")
        if ci:
            axes[1].vlines(i, ci[0], ci[1], color="#2166ac", linewidth=1.5)
            axes[1].hlines(ci, i - .07, i + .07, color="#2166ac")
        axes[1].annotate(f"{value:.3f}\n{g['paired_speed']['matched_runs']}/{g['run_count']} paired",
                         (i, value), xytext=(8, 6), textcoords="offset points", fontsize=9)
    axes[1].axhline(1, color="#666666", linestyle="--", linewidth=1)
    axes[1].set(title="Paired throughput ratio", ylabel="draft / target_only")
    axes[1].margins(x=.35, y=.3)
    for ax in axes:
        ax.set_xticks(x, [f"{label}={g['k']}\nwidth={g['proposal_width']}" for g in groups])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    m = summary["metadata"]
    fig.suptitle(f"{m['scene_count']} scenes / {m['prompt_count']} prompts | "
                 f"Target {m['target_precision'] or 'unknown'} / Draft {m['draft_precision'] or 'unknown'}", fontsize=12)
    fig.savefig(output_dir / "acceptance_speed.png", dpi=180)
    fig.savefig(output_dir / "acceptance_speed.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="实验 JSON 路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--width-unit", choices=("auto", "position", "gamma"), default="auto",
                        help="默认根据 JSON 识别新 K 或旧 Gamma；无定义字段时按新 K 解读")
    args = parser.parse_args()
    try:
        source, destination = args.input.resolve(), args.output_dir.resolve()
        if source in {destination / name for name in (
                "report.md", "summary.json", "acceptance_speed.png", "acceptance_speed.pdf")}:
            raise ValueError("输入文件与报告输出路径重合")
        def invalid_constant(value):
            raise ValueError(f"JSON 含非有限数值 {value}")
        data = json.loads(source.read_text(encoding="utf-8"), parse_constant=invalid_constant)
        summary = build_summary(data, source, args.width_unit)
        destination.mkdir(parents=True, exist_ok=True)
        plot(summary, destination)
        (destination / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (destination / "report.md").write_text(markdown(summary), encoding="utf-8")
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(1, f"报告生成失败：{error}\n")
    print(f"已生成报告：{destination / 'report.md'}")


if __name__ == "__main__":
    main()
