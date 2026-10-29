"""从同一会话的四种 PF 测量生成报告；仅需 Python 与 matplotlib。

python pf_matched_report.py --input results.json --output-dir DIR
也可通过 python -m draft_a.pf_matched_report 运行。
所有统计从 online.rows 重算，按场景共同重采样 2000 次。
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

if __package__:
    from . import pf_ranker_report as shared
    from .feature_acceptance_eval import _bootstrap, _online_totals
else:
    import pf_ranker_report as shared
    from feature_acceptance_eval import _bootstrap, _online_totals


VARIANTS = ("target_only", "old_mlp_fp32", "old_mlp_bf16", "current_gpt_fp32")
DRAFTS = ("old_mlp_fp32", "current_gpt_fp32", "old_mlp_bf16")
LABELS = {"target_only": "Target only FP32", "old_mlp_fp32": "旧 MLP FP32",
          "old_mlp_bf16": "旧 MLP BF16（精度参考）", "current_gpt_fp32": "当前 GPT FP32"}
PLOT_LABELS = {"old_mlp_fp32": "Old MLP FP32", "current_gpt_fp32": "Current GPT FP32",
               "old_mlp_bf16": "Old MLP BF16 (reference)"}
COLORS = {"old_mlp_fp32": "#2878b5", "current_gpt_fp32": "#db6b26", "old_mlp_bf16": "#888888"}
METRICS = ("tokens_per_second", "acceptance_verified", "acceptance_proposed",
           *(f"{kind}_acceptance_{den}" for kind in shared.KINDS for den in ("verified", "proposed")))
OUTPUTS = ("report.md", "summary.json", "matched_comparison.png", "matched_comparison.pdf")
DEFAULT_OUTPUT = Path("experiments/pf_learning_budget/pf_matched_20261014")


def clean_interval(value):
    return value if shared.valid_interval(value) else None


def bootstrap(rows, measure, seed):
    result = _bootstrap(rows, measure, seed, draws=2000)
    result["ci95"] = {key: clean_interval(value) for key, value in result["ci95"].items()}
    return result


def measure(rows):
    counts = _online_totals(rows)
    return {key: counts[key] for key in METRICS}


def policy_of(row, policy):
    local = row.get("policy") or {}
    temperature = row.get("temperature", local.get("temperature", local.get("temp",
                         policy.get("temperature", policy.get("temp")))))
    target_p = row.get("target_top_p", local.get("top_p", policy.get("top_p")))
    draft_p = row.get("draft_top_p", local.get("top_p", policy.get("top_p")))
    return temperature, target_p, draft_p


def prepare_rows(data):
    if not isinstance(data, dict) or not isinstance(data.get("online"), dict):
        raise ValueError("输入需要包含 online 对象")
    original = data["online"].get("rows")
    if not isinstance(original, list) or not original:
        raise ValueError("online.rows 需要包含测量记录")
    params, policy = data.get("parameters") or {}, data.get("policy") or {}
    rows, warnings = [], []
    for index, source in enumerate(original):
        if not isinstance(source, dict):
            raise ValueError(f"online.rows[{index}] 需要为对象")
        row = dict(source)
        variant = row.get("variant")
        if variant not in VARIANTS:
            raise ValueError(f"online.rows[{index}].variant 无法识别")
        expected_mode = "target_only" if variant == "target_only" else "draft"
        if row.setdefault("mode", expected_mode) != expected_mode:
            raise ValueError(f"online.rows[{index}] 的 mode 与 variant 不符")
        if row.get("k") != (0 if variant == "target_only" else 2):
            raise ValueError(f"online.rows[{index}] 需要 Target K=0、MLP/GPT K=2")
        shared.number(row.get("seed"), f"online.rows[{index}].seed", True)
        if "requested_tokens" not in row:
            row["requested_tokens"] = params.get("tokens")
            warnings.append("部分生成预算由 parameters.tokens 补充。")
        shared.number(row.get("requested_tokens"), f"online.rows[{index}].requested_tokens", True)
        for kind in ("", *shared.KINDS):
            stem = kind + "_" if kind else ""
            for field in shared.COUNTS:
                key = stem + field
                if variant == "target_only":
                    row.setdefault(key, 0)
                shared.number(row.get(key), f"online.rows[{index}].{key}", True)
            a, v, p = (row[stem + key] for key in shared.COUNTS)
            if not a <= v <= p:
                raise ValueError(f"online.rows[{index}].{stem} 需要 accepted ≤ verified ≤ proposed")
        for key in shared.COUNTS:
            if row[key] != sum(row[f"{kind}_{key}"] for kind in shared.KINDS):
                raise ValueError(f"online.rows[{index}].{key} 与分类型总数不符")
            if variant == "target_only" and row[key] != 0:
                raise ValueError("target_only 提案计数需要为零")
        expected_width = 0 if variant == "target_only" else 3
        if row.get("proposal_width", expected_width) != expected_width:
            raise ValueError(f"online.rows[{index}] 的 proposal_width 与 K 不符")
        row["_policy"] = policy_of(row, policy)
        for value in row["_policy"]:
            if value is not None:
                shared.number(value, f"online.rows[{index}].policy")
                if value <= 0:
                    raise ValueError("采样参数需要大于零")
        if any(p is not None and p > 1 for p in row["_policy"][1:]):
            raise ValueError("top_p 需要不大于 1")
        row["_session"] = row.get("session_id", row.get("session",
                              data.get("session_id", data.get("session", "single_input_session"))))
        if not isinstance(row["_session"], (str, int)):
            raise ValueError("session 标识需要为字符串或整数")
        rows.append(row)
    shared.validate_rows(rows)
    return rows, warnings


def pair_identity(row):
    return (*shared.identity(row), row["_session"], *row["_policy"])


def paired(rows, reference):
    selected, references = defaultdict(list), defaultdict(list)
    for row in rows:
        selected[pair_identity(row)].append(row)
    for row in reference:
        references[pair_identity(row)].append(row)
    pairs, ambiguous = [], 0
    for key, options in selected.items():
        candidates = references.get(key, [])
        if len(options) == 1 and len(candidates) == 1:
            pairs.append({"name": options[0]["name"], "left": options[0], "right": candidates[0]})
        elif len(options) > 1 or len(candidates) > 1:
            ambiguous += len(options)
    return pairs, ambiguous


def pair_measure(pairs):
    a = _online_totals([pair["left"] for pair in pairs])
    b = _online_totals([pair["right"] for pair in pairs])
    result = {"throughput_ratio": shared.ratio(a["tokens_per_second"], b["tokens_per_second"]),
              "reverse_throughput_ratio": shared.ratio(b["tokens_per_second"], a["tokens_per_second"])}
    for kind in shared.KINDS[:2]:
        key = kind + "_acceptance_verified"
        result[kind + "_acceptance_difference_pp"] = (
            100 * (a[key] - b[key]) if a[key] is not None and b[key] is not None else None)
    return result


def pair_summary(rows, reference, seed):
    pairs, ambiguous = paired(rows, reference)
    a = shared.totals([pair["left"] for pair in pairs])
    b = shared.totals([pair["right"] for pair in pairs])
    boot = bootstrap(pairs, pair_measure, seed)
    # 差值区间允许负数，比例区间沿用现有报告的数值要求。
    raw_ci = _bootstrap(pairs, pair_measure, seed, draws=2000)["ci95"] if pairs else {}
    for key in ("position_acceptance_difference_pp", "feature_acceptance_difference_pp"):
        value = raw_ci.get(key)
        boot["ci95"][key] = value if value and all(v is not None for v in value) else None
    return {"matched_runs": len(pairs), "selected_runs": len(rows),
            "unmatched_runs": len(rows) - len(pairs), "ambiguous_runs": ambiguous,
            "left": a, "right": b, **pair_measure(pairs), "bootstrap": boot}


def group_summary(rows, direct, seed):
    result = shared.totals(rows)
    result["bootstrap"] = bootstrap(rows, measure, seed)
    if rows and rows[0]["variant"] != "target_only":
        result["paired_speed"] = pair_summary(rows, direct, seed)
    else:
        result["paired_speed"] = None
    return result


def build_summary(data, source_path):
    rows, warnings = prepare_rows(data)
    params, policy = data.get("parameters") or {}, data.get("policy") or {}
    seed = shared.number(params.get("seed", 24685), "parameters.seed", True)
    variants = {variant: [r for r in rows if r["variant"] == variant] for variant in VARIANTS}
    direct = variants["target_only"]
    prefixes = sorted({r["prompt_tokens"] for r in rows})
    names = sorted({r["name"] for r in rows})
    sets = {variant: {pair_identity(r) for r in selected} for variant, selected in variants.items()}
    same_rows = bool(direct) and all(sets[v] == sets["target_only"] and
                                   len(sets[v]) == len(variants[v]) for v in VARIANTS)
    expected_prefixes = params.get("prefixes", [0, 64, 256])
    expected_scenes = shared.number(params.get("scenes", 12), "parameters.scenes", True)
    observed_pairs = {(r["name"], r["prompt_tokens"]) for r in direct}
    complete_coverage = (same_rows and len(names) == expected_scenes and
                         prefixes == sorted(expected_prefixes) and
                         observed_pairs == {(name, prefix) for name in names for prefix in expected_prefixes} and
                         len(direct) == expected_scenes * len(expected_prefixes))
    sampling = sorted({r["_policy"] for r in rows}, key=str)
    expected_sampling = all(all(v is not None and math.isclose(v, 0.9) for v in triplet)
                            for triplet in sampling)
    same_budget = params.get("tokens") is not None and all(
        r["requested_tokens"] == params["tokens"] for r in rows)
    if data.get("status") != "complete":
        warnings.append("实验尚未声明 complete；结论仅覆盖已记录的测量。")
    if not complete_coverage:
        warnings.append("四种实现的场景/前缀覆盖尚未齐全或存在重复；速度仅使用唯一有效配对。")
    if not expected_sampling:
        warnings.append("实际采样字段存在缺失或与 temperature=top_p=0.9 不符；不同策略分别配对。")
    if not same_budget:
        warnings.append("行预算与 parameters.tokens 存在差异或未完整记录。")
    if params.get("k", 2) != 2:
        warnings.append("parameters.k 与本报告 K=2 范围不符。")
    online = data["online"]
    validation = online.get("validation") or {}
    validation_status = validation.get("status") if isinstance(validation, dict) else None
    if validation_status not in ("passed", "complete"):
        warnings.append("online.validation 尚未声明通过。")
    overall = {v: group_summary(variants[v], direct, seed) for v in VARIANTS}
    primary = pair_summary(variants["current_gpt_fp32"], variants["old_mlp_fp32"], seed)
    precision = pair_summary(variants["old_mlp_bf16"], variants["old_mlp_fp32"], seed)
    by_prefix = []
    for prefix in prefixes:
        selected = {v: [r for r in variants[v] if r["prompt_tokens"] == prefix] for v in VARIANTS}
        by_prefix.append({"prompt_tokens": prefix,
                          "variants": {v: group_summary(selected[v], selected["target_only"], seed)
                                       for v in VARIANTS},
                          "primary_comparison": pair_summary(selected["current_gpt_fp32"],
                                                              selected["old_mlp_fp32"], seed),
                          "bf16_precision_reference": pair_summary(selected["old_mlp_bf16"],
                                                                    selected["old_mlp_fp32"], seed)})
    for label, result in [("主比较", primary), ("BF16 精度参考", precision)]:
        if result["unmatched_runs"]:
            warnings.append(f"{label}有 {result['unmatched_runs']} 次测量未进入配对。")
    for v in DRAFTS:
        pair = overall[v]["paired_speed"]
        if pair and pair["unmatched_runs"]:
            warnings.append(f"{LABELS[v]} 有 {pair['unmatched_runs']} 次测量缺少唯一 Target 配对。")
    return {"input": str(source_path), "status": data.get("status"), "parameters": params,
            "policy": policy, "metadata": {
                "scene_count": len(names), "scene_names": names, "run_count": len(rows),
                "prefixes": prefixes, "base_seed": seed,
                "row_seeds": sorted({r["seed"] for r in rows}), "sampling": sampling,
                "sessions": sorted({r["_session"] for r in rows}, key=str),
                "same_scene_prefix_seed_budget_policy": same_rows,
                "complete_coverage": complete_coverage,
                "matched_conditions": same_rows and same_budget and expected_sampling,
                "expected_runs": expected_scenes * len(expected_prefixes) * len(VARIANTS),
                "target_precision": data.get("target_precision"),
                "validation_status": validation_status, "timing_scope": online.get("scope"),
                "checkpoint_step": data.get("checkpoint_step")},
            "definitions": {"acceptance_verified": "sum(accepted) / sum(verified)",
                            "acceptance_proposed": "sum(accepted) / sum(proposed)",
                            "throughput": "sum(generated_tokens) / sum(seconds)",
                            "speedup": "paired variant throughput / paired target_only throughput",
                            "primary_ratio": "paired current_gpt_fp32 throughput / paired old_mlp_fp32 throughput",
                            "precision_ratio": "paired old_mlp_bf16 throughput / paired old_mlp_fp32 throughput",
                            "pair_fields": ["name", "prompt_tokens", "seed", "requested_tokens",
                                            "session", "temperature", "target_top_p", "draft_top_p"],
                            "intervals": "2000 scene resamples; keep all prefixes and both sides of each pair together; percentile 95%"},
            "overall": overall, "by_prefix": by_prefix, "primary_comparison": primary,
            "bf16_precision_reference": precision, "validation": validation,
            "source_metadata": {key: data.get(key) for key in (
                "checkpoint", "old_checkpoint", "ranker", "source_root", "model_parameters", "timings")},
            "warnings": list(dict.fromkeys(warnings))}


def fmt(value, digits=3):
    return shared.fmt(value, digits)


def interval(value, percent=False):
    if not value:
        return "无可用区间"
    return (f"[{value[0]:.1%}, {value[1]:.1%}]" if percent else
            f"[{fmt(value[0])}, {fmt(value[1])}]")


def rate_text(group, kind, denominator, with_ci=False):
    a, b = group[kind + "_accepted"], group[kind + "_" + denominator]
    value = group[kind + "_acceptance_" + denominator]
    text = f"{a}/{b}（{value:.1%}）" if value is not None else f"{a}/{b}（无可用比例）"
    if with_ci and value is not None:
        text += "；" + interval(group["bootstrap"]["ci95"].get(kind + "_acceptance_" + denominator), True)
    return text


def tables(groups):
    lines = ["| 实现 | 次数 | P accepted/verified；95% 区间 | F accepted/verified；95% 区间 | P accepted/proposed | F accepted/proposed | EOS accepted/verified/proposed |",
             "|---|---:|---|---|---|---|---|"]
    for variant in DRAFTS:
        group = groups[variant]
        eos = "/".join(str(group["eos_" + key]) for key in shared.COUNTS)
        lines.append(f"| {LABELS[variant]} | {group['run_count']} | " + " | ".join([
            rate_text(group, "position", "verified", True), rate_text(group, "feature", "verified", True),
            rate_text(group, "position", "proposed"), rate_text(group, "feature", "proposed"), eos]) + " |")
    lines += ["", "| 实现 | 生成 token | 总秒数 | token/s；95% 区间 | 配对实现 / Target token/s | 对 Target 速度比；95% 区间 | 配对数/次数 | EOS 结束次数 |",
              "|---|---:|---:|---|---|---|---|---:|"]
    for variant in VARIANTS:
        group = groups[variant]
        pair = group["paired_speed"]
        throughput = f"{fmt(group['tokens_per_second'])}；{interval(group['bootstrap']['ci95']['tokens_per_second'])}"
        paired_tps = f"{fmt(pair['left']['tokens_per_second'])} / {fmt(pair['right']['tokens_per_second'])}" if pair else "—"
        speed = (f"{fmt(pair['throughput_ratio'])}×；{interval(pair['bootstrap']['ci95']['throughput_ratio'])}"
                 if pair else "—")
        count = f"{pair['matched_runs']}/{group['run_count']}" if pair else "—"
        lines.append(f"| {LABELS[variant]} | {group['generated_tokens']} | {fmt(group['seconds'])} | "
                     f"{throughput} | {paired_tps} | {speed} | {count} | {group['eos_runs']} |")
    return lines


def comparison_text(pair, label):
    ci = pair["bootstrap"]["ci95"]
    lines = [f"{label}：吞吐比 {fmt(pair['throughput_ratio'])}×，95% 区间 {interval(ci['throughput_ratio'])}；"
             f"唯一配对 {pair['matched_runs']}/{pair['selected_runs']}。"]
    for kind, name in (("position", "P"), ("feature", "F")):
        key = kind + "_acceptance_difference_pp"
        lines.append(f"{name} accepted/verified 配对差值 {fmt(pair[key], 2)} 个百分点，"
                     f"95% 区间 {interval(ci[key])} 个百分点。")
    return lines


def validation_lines(value, path="online.validation"):
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            lines.extend(validation_lines(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            lines.extend(validation_lines(item, f"{path}[{index}]"))
    elif isinstance(value, (int, float, bool)) or path.endswith((".status", ".seed", ".tokens")):
        lines.append(f"- `{path}`：{value}")
    return lines


def markdown(summary):
    m, primary = summary["metadata"], summary["primary_comparison"]
    value = primary["throughput_ratio"]
    lines = ["# 同条件旧 MLP PF 与当前 GPT PF 对照", ""]
    if value is None:
        lines.append("当前记录尚无可用的旧 MLP FP32 / 当前 GPT FP32 唯一配对，主比较数值暂不可得。")
    else:
        direction = "更高" if value > 1 else "更低" if value < 1 else "相同"
        ci = primary["bootstrap"]["ci95"]["throughput_ratio"]
        lines.append(f"主比较：当前 GPT FP32 相对旧 MLP FP32 的吞吐比为 {fmt(value)}×，"
                     f"95% 场景重采样区间 {interval(ci)}，点估计吞吐{direction}。")
        if ci and ci[0] <= 1 <= ci[1]:
            lines.append("该区间包含 1，当前场景样本仍允许两者吞吐接近。")
        new = summary["overall"]["current_gpt_fp32"]
        old = summary["overall"]["old_mlp_fp32"]
        if new["feature_acceptance_verified"] is not None and old["feature_acceptance_verified"] is not None:
            gain = 100 * (new["feature_acceptance_verified"] - old["feature_acceptance_verified"])
            lines.append(f"当前 Feature 接收率提高 {gain:.2f} 个百分点；同精度吞吐比为 {value:.3f}×。")
        if new["target_calls"] is not None and old["target_calls"]:
            saved = 1 - new["target_calls"] / old["target_calls"]
            lines.append(f"当前 Target 调用数 {new['target_calls']}，旧方案 {old['target_calls']}，"
                         f"减少 {saved:.2%}；完整解码耗时为 {new['seconds']:.3f} 秒与 {old['seconds']:.3f} 秒。")
        lines.append("Feature 质量改善带来了更少的 Target 验证调用；当前独立 GPT 的计算与上下文维护成本仍高于旧状态 MLP。")
    if summary["status"] != "complete" or not m["complete_coverage"]:
        lines.append("实验记录尚未完整；以下数值仅描述当前已完成测量。")
    lines += ["", *comparison_text(primary, "当前 GPT FP32 / 旧 MLP FP32"),
              f"反向吞吐比（旧 MLP FP32 / 当前 GPT FP32）：{fmt(primary['reverse_throughput_ratio'])}×；"
              f"95% 区间 {interval(primary['bootstrap']['ci95']['reverse_throughput_ratio'])}。", "",
              "BF16 仅作为旧 MLP 的执行精度参考，主比较采用双方 FP32。",
              *comparison_text(summary["bf16_precision_reference"], "旧 MLP BF16 / 旧 MLP FP32"),
              "", "## 实验范围与运行条件", "",
              f"记录状态：{summary['status']}；{m['scene_count']} 个场景；{m['run_count']}/{m['expected_runs']} 次测量。",
              f"前缀 token 数：{', '.join(map(str, m['prefixes']))}；预算 {summary['parameters'].get('tokens')} token；"
              "Target K=0，三种 PF 实现 K=2，最大提案顺序为 P1/F1/P2。",
              f"基准 seed：{m['base_seed']}；各场景/前缀使用派生 seed，四种实现逐行共享对应 seed。",
              f"四种实现的场景/前缀/seed/预算/采样配对一致：{'是' if m['same_scene_prefix_seed_budget_policy'] else '尚未齐全'}；"
              f"完整覆盖：{'是' if m['complete_coverage'] else '尚未齐全'}。",
              "采样策略（temperature / Target top_p / draft top_p）：" +
              "；".join(" / ".join(fmt(v) for v in triplet) for triplet in m["sampling"]) + "。",
              "全部结果来自同一输入会话；配对还保留输入中的 session 标识。",
              "场景：" + "、".join(m["scene_names"]) + "。",
              f"Target 精度：{m['target_precision'] or '未记录'}；旧 MLP FP32、当前 GPT FP32 为主比较；"
              "旧 MLP BF16 使用 autocast，低精度运算会改变提案概率、接收和生成轨迹。",
              "旧 MLP 读取 Target 在上一个 Position 的状态、待处理 Feature 和下一 Position；"
              "非零前缀通过真实 Target 预处理恢复该状态，并重放 Position ranker 历史。",
              "当前 GPT 从自己的 KV 历史与 P/F 因果上下文生成 Feature；两者状态来源不同。",
              "两种 PF 共用 Position ranker 和 Target typed heads；Target-only 使用同一 Target typed 解码路径。",
              "计时从 CUDA 同步之后开始，到解码结束的 CUDA 同步为止；包含 cache 分配、前缀处理、"
              "ranker 历史、提案、Target 验证和拒绝后采样。模型加载、数值验证和输出序列验证另行处理。",
              f"JSON 计时范围：{m['timing_scope'] or '未记录'}。",
              "上下文包含 BOS、前缀、已生成 P/F 历史和当前提案；前缀 0/64/256 对应含 BOS 的 1/65/257 token。",
              "GPU 测量前对每种实现和三个前缀进行预热；正式计时轮换四种实现的顺序。",
              f"输入：`{summary['input']}`。", "", "## 统计口径", "",
              "所有汇总从 online.rows 独立计算，输入 summary 不参与计算。",
              "P/F/EOS 分开累加 accepted、verified、proposed。accepted/verified 表示实际到达提案的接收率；"
              "accepted/proposed 包含首次拒绝后未使用的提案。EOS 单独计数，Target 补充 token 和拒绝后采样不计入 accepted。",
              "吞吐为总生成 token / 总秒数；速度比为唯一配对两侧的总吞吐之比。EOS 会使输出长度不同，"
              "速度比按各自实际生成 token 数计算。重复记录和缺少对应记录的测量均不进入配对。",
              "区间采用 2000 次场景重采样及 2.5%/97.5% 分位数，同一场景全部前缀和配对两侧共同抽取；"
              "接收率每次先累加分子分母。零分母返回无可用比例及区间。",
              "", "## 总体汇总", "", *tables(summary["overall"]), "", "## 按前缀汇总", ""]
    for group in summary["by_prefix"]:
        lines += [f"### 前缀 {group['prompt_tokens']}", "", *tables(group["variants"]), "",
                  *comparison_text(group["primary_comparison"], "当前 GPT FP32 / 旧 MLP FP32"), "",
                  *comparison_text(group["bf16_precision_reference"], "旧 MLP BF16 / 旧 MLP FP32"), ""]
    lines += ["## 图表", "", "![接收率与配对加速比](matched_comparison.png)", "",
              "左面板展示 P/F accepted/verified，总体累加计数；右面板展示总体及各前缀相对 Target-only 的加速比。"
              "细线为 95% 场景重采样区间；BF16 灰色表示旧执行精度参考。PDF 提供同图的矢量版本。",
              "", "## 运行验证", "", f"online.validation 状态：{m['validation_status'] or '未记录'}。",
              "验证记录覆盖原 block2 在前缀 0 的三 seed / 64 token 输出逐 token 比对、"
              "前缀 64/256 的真实 Target 状态与逐 token 参考，以及 terminal EOS。",
              "前缀状态同时记录 rawAbs、relativeL2 与传播到旧 Feature 分布的 TV；"
              "数值尺度通过 relativeL2<1e-3 评估，Feature 分布要求 TV<0.01。",
              "报告仅呈现输入验证结果，不运行模型。", "", *validation_lines(summary["validation"]), "",
              "## 结论适用范围", "",
              "本次使用一个基准 seed、12 个场景，每个场景/前缀只有一次计时。"
              "场景重采样区间描述样本场景变化，未覆盖多基准 seed 或同一输入重复计时的波动。",
              "固定 seed 在不同实现中消耗随机数的顺序不同，生成轨迹、拒绝路径和 EOS 长度可以不同。",
              "Feature 条件接收率受前序 Position 接收筛选，接收率差值同时受生成轨迹影响；"
              "两种状态来源和模型规模的差异限制对单一架构因素的归因。",
              "BF16 与 FP32 的数值和速度差异单独报告；其结果不能替代同精度主比较。", ""]
    if summary["warnings"]:
        lines += ["### 数据提示", "", *(f"- {item}" for item in summary["warnings"]), ""]
    return "\n".join(lines) + "\n"


def draw_interval(ax, x, ci, color, width=.04):
    if ci:
        ax.vlines(x, ci[0], ci[1], color=color, linewidth=1.3)
        ax.hlines(ci, x - width, x + width, color=color, linewidth=1.3)


def plot(summary, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("生成图表需要 matplotlib；请使用已有该依赖的 Python 环境") from error
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained")
    for index, variant in enumerate(DRAFTS):
        group = summary["overall"][variant]
        for offset, (kind, tag) in enumerate((("position", "P"), ("feature", "F"))):
            value = group[kind + "_acceptance_verified"]
            x = index + (offset - .5) * .32
            if value is None:
                axes[0].text(x, .03, f"{tag}: N/A", ha="center", fontsize=9)
                continue
            ci = group["bootstrap"]["ci95"][kind + "_acceptance_verified"]
            axes[0].bar(x, value, width=.28, color=COLORS[variant],
                        alpha=1 if offset == 0 else .6, hatch="" if offset == 0 else "//",
                        label=f"{tag} accepted/verified" if index == 0 else None)
            draw_interval(axes[0], x, ci, "#333333")
            axes[0].text(x, max(value, ci[1] if ci else value) + .025,
                         f"{value:.1%}", ha="center", va="bottom", fontsize=10)
    axes[0].set_xticks(range(len(DRAFTS)), [PLOT_LABELS[v].replace(" (reference)", "\n(reference)") for v in DRAFTS])
    axes[0].set(title="P/F acceptance (all prefixes)", ylabel="accepted / verified", ylim=(0, 1.18))
    axes[0].legend(fontsize=9, loc="upper left")
    groups = [("All", summary["overall"]), *[(str(g["prompt_tokens"]), g["variants"]) for g in summary["by_prefix"]]]
    upper = 1.0
    for index, variant in enumerate(DRAFTS):
        for position, (_, variants) in enumerate(groups):
            pair = variants[variant]["paired_speed"]
            if not pair or pair["throughput_ratio"] is None:
                continue
            value = pair["throughput_ratio"]
            ci = pair["bootstrap"]["ci95"]["throughput_ratio"]
            x = position + (index - 1) * .23
            color = COLORS[variant]
            axes[1].plot(x, value, "o", color=color, markersize=6,
                         label=PLOT_LABELS[variant] if position == 0 else None)
            draw_interval(axes[1], x, ci, color)
            anchor = max(value, ci[1] if ci else value)
            upper = max(upper, anchor)
            axes[1].annotate(f"{value:.2f}x", (x, anchor), xytext=(0, 7 + index * 12),
                             textcoords="offset points", ha="center", color=color, fontsize=9,
                             bbox={"facecolor": "white", "edgecolor": "none", "pad": 1, "alpha": .8})
    axes[1].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(groups)), [name for name, _ in groups])
    axes[1].set(title="Paired speedup vs Target only", xlabel="Prefix tokens (All = pooled)",
                ylabel="variant throughput / Target throughput", ylim=(0, upper * 1.35))
    axes[1].set_xlim(-.6, len(groups) - .4)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=9, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    m, pair = summary["metadata"], summary["primary_comparison"]
    fig.suptitle(f"Matched PF: K=2 | {m['scene_count']} scenes | base seed {m['base_seed']} | "
                 f"temperature / top_p = 0.9 / 0.9 | status: {summary['status']}", fontsize=12)
    fig.supxlabel(f"95% scene intervals (2000 resamples). Main FP32 ratio: Current GPT / Old MLP = "
                  f"{fmt(pair['throughput_ratio'])}x {interval(pair['bootstrap']['ci95']['throughput_ratio'])}", fontsize=10)
    try:
        fig.savefig(output_dir / "matched_comparison.png", dpi=180)
        fig.savefig(output_dir / "matched_comparison.pdf")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="同一会话的实验 JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="报告输出目录")
    args = parser.parse_args()
    try:
        source, destination = args.input.resolve(), args.output_dir.resolve()
        if source in {(destination / name).resolve() for name in OUTPUTS}:
            raise ValueError("输入路径与报告输出路径重合")
        def invalid_constant(value):
            raise ValueError(f"JSON 含非有限数值 {value}")
        data = json.loads(source.read_text(encoding="utf-8"), parse_constant=invalid_constant)
        summary = build_summary(data, source)
        text = markdown(summary)
        destination.mkdir(parents=True, exist_ok=True)
        plot(summary, destination)
        (destination / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (destination / "report.md").write_text(text, encoding="utf-8")
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(1, f"报告生成失败：{error}\n")
    print(f"已生成报告：{destination / 'report.md'}")


if __name__ == "__main__":
    main()
