#!/usr/bin/env python3
"""独立复算已完成的同一 Target 方法对照记录。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys


BOOTSTRAP_SEED = 402101
BOOTSTRAP_DRAWS = 10000
DEFAULT_POSITION_VOCAB = 8000
DEFAULT_FEATURE_OFFSET = 8000


class AuditInputError(RuntimeError):
    pass


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AuditInputError(f"缺少输入文件：{path}") from error
    except json.JSONDecodeError as error:
        raise AuditInputError(f"JSON 无法读取：{path}:{error.lineno}:{error.colno}") from error


def read_records(path: Path):
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AuditInputError(
                        f"逐次记录格式错误：{path}:{line_no}:{error.colno}"
                    ) from error
                if not isinstance(row, dict):
                    raise AuditInputError(f"逐次记录第 {line_no} 行不是对象")
                rows.append(row)
    except FileNotFoundError as error:
        raise AuditInputError(f"缺少输入文件：{path}") from error
    return rows


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close_number(left, right, *, rel_tol=1e-12, abs_tol=1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not finite_number(left) or not finite_number(right):
        return False
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def quantile(values, probability):
    ordered = sorted(values)
    point = (len(ordered) - 1) * probability
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    if high == low:
        return ordered[low]
    return ordered[low] * (high - point) + ordered[high] * (point - low)


def compare_ratio(rows_by_pair, numerator, denominator, budget, full_only=False):
    source_logs = defaultdict(list)
    pair_count = 0
    for arms in rows_by_pair.values():
        if numerator not in arms or denominator not in arms:
            continue
        up, down = arms[numerator], arms[denominator]
        if full_only and (up["generated_tokens"] != budget or down["generated_tokens"] != budget):
            continue
        if min(up["generated_tokens"], down["generated_tokens"]) <= 0:
            continue
        up_rate = up["generated_tokens"] / up["outer_wall_seconds"]
        down_rate = down["generated_tokens"] / down["outer_wall_seconds"]
        source_logs[up["source"]].append(math.log(up_rate) - math.log(down_rate))
        pair_count += 1

    source_means = {
        source: math.fsum(values) / len(values)
        for source, values in source_logs.items()
    }
    source_names = sorted(source_means)
    result = {
        "pair_count": pair_count,
        "source_count": len(source_names),
        "source_equal_geometric_mean": None,
        "source_bootstrap_95_ci": None,
        "per_source": {},
    }
    if not source_names:
        return result

    result["source_equal_geometric_mean"] = math.exp(
        math.fsum(source_means.values()) / len(source_names)
    )
    result["per_source"] = {
        source: math.exp(source_means[source]) for source in source_names
    }
    result["source_bootstrap_seed"] = BOOTSTRAP_SEED
    result["source_bootstrap_replicates"] = BOOTSTRAP_DRAWS
    if len(source_names) > 1:
        rng = random.Random(BOOTSTRAP_SEED)
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled_means = [source_means[rng.choice(source_names)] for _ in source_names]
            draws.append(math.exp(math.fsum(sampled_means) / len(source_names)))
        result["source_bootstrap_95_ci"] = [
            quantile(draws, 0.025), quantile(draws, 0.975)
        ]
    return result


def expected_contrasts(methods, executions):
    names = {}

    def add(label, numerator, denominator):
        if numerator in {f"{execution}:{method}" for execution in executions for method in methods} and \
                denominator in {f"{execution}:{method}" for execution in executions for method in methods}:
            names[label] = (numerator, denominator)

    if "original" in executions and "optimized" in executions:
        for method in methods:
            add(f"execution_gain:{method}", f"optimized:{method}", f"original:{method}")
    for method in methods:
        if method == "target":
            continue
        for execution in executions:
            add(f"algorithm_gain:{execution}:{method}",
                f"{execution}:{method}", f"{execution}:target")
            if method != "pf_mlp" and "pf_mlp" in methods:
                add(f"versus_pf:{execution}:{method}",
                    f"{execution}:{method}", f"{execution}:pf_mlp")
    return names


def compare_summary_number(actual, expected, path):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: summary 项不是对象"]
        if set(actual) != set(expected):
            errors.append(f"{path}: 键集合不同")
        for key, value in expected.items():
            if key in actual:
                errors.extend(compare_summary_number(actual[key], value, f"{path}.{key}"))
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: 列表长度不同"]
        for index, (left, right) in enumerate(zip(actual, expected)):
            errors.extend(compare_summary_number(left, right, f"{path}[{index}]"))
        return errors
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not close_number(actual, expected):
            errors.append(f"{path}: 数值不同，summary={actual!r} raw={expected!r}")
        return errors
    if actual != expected:
        errors.append(f"{path}: summary={actual!r} raw={expected!r}")
    return errors


def validate_sequence(prompt, output, *, eos_id, position_vocab, feature_offset, ranks,
                      allow_empty_output=False):
    errors = []
    if not prompt or prompt[0] != eos_id:
        errors.append("prompt BOS sentinel differs from eos_id")
        return errors
    prefix = prompt[1:]
    if eos_id in prefix:
        errors.append("prompt contains an EOS after its BOS sentinel")
    if len(prefix) % 2:
        errors.append("prompt does not end on a complete P/F pair")
    if not isinstance(output, list) or (not output and not allow_empty_output):
        errors.append("generated output is empty or not a list")
        return errors
    if any(type(token) is not int for token in prompt + output):
        errors.append("prompt or output contains a non-integer token")
        return errors

    body = prefix + output
    if eos_id in body:
        if body[-1] != eos_id or body.count(eos_id) != 1:
            errors.append("EOS is duplicated or is not the last generated token")
        else:
            body = body[:-1]
    if len(body) % 2:
        errors.append("P/F body does not contain complete pairs")
    previous_rank = None
    for index, token in enumerate(body):
        if index % 2 == 0:
            if not 0 <= token < position_vocab:
                errors.append(f"Position token out of range at body index {index}: {token}")
            elif ranks is not None:
                rank = ranks[token]
                if previous_rank is not None and rank <= previous_rank:
                    errors.append(f"Position rank is not increasing at body index {index}")
                previous_rank = rank
        elif not feature_offset <= token < eos_id:
            errors.append(f"Feature token out of range at body index {index}: {token}")
    return errors


def check(name, passed, details=None):
    return {"name": name, "passed": bool(passed), "details": details or {}}


def load_inputs(run_dir: Path):
    plan = read_json(run_dir / "plan.json")
    prompts_doc = read_json(run_dir / "prompts.json")
    summary = read_json(run_dir / "summary.json")
    rows = read_records(run_dir / "records.jsonl")
    if summary.get("status") != "complete":
        raise AuditInputError("summary.status 不是 complete；仅审计完整运行目录")
    pipeline_path = run_dir / "pipeline.json"
    if pipeline_path.exists():
        pipeline = read_json(pipeline_path)
        if pipeline.get("status") != "complete":
            raise AuditInputError("pipeline.status 不是 complete；仅审计完整运行目录")
    else:
        pipeline = None
    return plan, prompts_doc, rows, summary, pipeline


def run_audit(run_dir: Path):
    plan, prompts_doc, rows, summary, pipeline = load_inputs(run_dir)
    prompts = prompts_doc.get("contexts") if isinstance(prompts_doc, dict) else prompts_doc
    if not isinstance(prompts, list):
        raise AuditInputError("prompts.json 缺少 contexts 列表")
    methods = plan.get("methods")
    executions = plan.get("executions")
    if not isinstance(methods, list) or not methods or not isinstance(executions, list) or not executions:
        raise AuditInputError("plan.json 缺少 methods 或 executions")
    if len(set(methods)) != len(methods) or len(set(executions)) != len(executions):
        raise AuditInputError("plan.json 的 methods 或 executions 含重复值")
    if "target" not in methods:
        raise AuditInputError("plan.methods 不含 target 参照臂")

    budget = int(plan.get("budget", 0))
    if budget < 2 or budget % 2:
        raise AuditInputError("plan.budget 必须是正偶数")
    expected_runs = len(prompts) * len(methods) * len(executions)
    if int(plan.get("prompt_count", -1)) != len(prompts):
        raise AuditInputError("plan.prompt_count 与 prompts.json 数量不符")
    if int(plan.get("expected_runs", -1)) != expected_runs:
        raise AuditInputError("plan.expected_runs 与组合数不符")
    if summary.get("expected_runs") != expected_runs or summary.get("completed_runs") != expected_runs:
        raise AuditInputError("summary 中的运行总数与完整组合数不符")
    if len(rows) != expected_runs:
        raise AuditInputError(f"records 数量为 {len(rows)}，预期 {expected_runs}")
    if pipeline is not None and int(pipeline.get("completed_runs", -1)) != expected_runs:
        raise AuditInputError("pipeline.completed_runs 与完整组合数不符")

    prompt_eos_id = prompts_doc.get("eos_id", -1) if isinstance(prompts_doc, dict) else -1
    eos_id = int(plan.get("eos_id", prompt_eos_id))
    if eos_id < 0:
        raise AuditInputError("plan 或 prompts 缺少 eos_id")
    if "eos_id" in prompts_doc and int(prompts_doc["eos_id"]) != eos_id:
        raise AuditInputError("plan.eos_id 与 prompts.eos_id 不符")
    position_vocab = int(plan.get("position_vocab_size", plan.get("p_vocab", DEFAULT_POSITION_VOCAB)))
    feature_offset = int(plan.get("feature_token_offset", plan.get("feature_offset", DEFAULT_FEATURE_OFFSET)))
    rank_values = plan.get("position_rank_map")
    if not isinstance(rank_values, list) or len(rank_values) < position_vocab:
        raise AuditInputError("plan.position_rank_map 缺失或长度不足，无法复算 Position 顺序")
    if any(type(value) is not int for value in rank_values[:position_vocab]):
        raise AuditInputError("position_rank_map 含非整数值")
    ranks = rank_values[:position_vocab]

    prompt_by_id = {}
    prompt_errors = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            prompt_errors.append("prompt entry is not an object")
            continue
        pair_id = str(prompt.get("pair_id", ""))
        if not pair_id or pair_id in prompt_by_id:
            prompt_errors.append(f"pair_id is empty or repeated: {pair_id!r}")
            continue
        tokens = prompt.get("prompt")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            prompt_errors.append(f"prompt token list is invalid for {pair_id}")
            continue
        prompt_by_id[pair_id] = prompt
        prompt_errors.extend(
            f"{pair_id}: {message}"
            for message in validate_sequence(
                tokens, [], eos_id=eos_id, position_vocab=position_vocab,
                feature_offset=feature_offset, ranks=ranks, allow_empty_output=True
            )
        )

    prompt_names = set(prompt_by_id)
    expected_sources = set(str(item) for item in plan.get("sources", []))
    prompt_sources = {str(item.get("source")) for item in prompt_by_id.values()}
    source_list_match = not expected_sources or expected_sources == prompt_sources
    rows_by_pair = defaultdict(dict)
    duplicate_arms = []
    unplanned_rows = []
    record_errors = []
    all_validated = True
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", ""))
        method = row.get("method")
        execution = row.get("execution")
        arm = row.get("arm")
        if pair_id not in prompt_by_id or method not in methods or execution not in executions:
            unplanned_rows.append({"index": index, "pair_id": pair_id, "arm": arm})
            continue
        expected_arm = f"{execution}:{method}"
        if arm != expected_arm:
            record_errors.append(f"record {index}: arm={arm!r}, expected {expected_arm!r}")
        if arm in rows_by_pair[pair_id]:
            duplicate_arms.append({"pair_id": pair_id, "arm": arm})
        rows_by_pair[pair_id][arm] = row
        prompt = prompt_by_id[pair_id]
        for key in ("source", "prefix", "seed"):
            if row.get(key) != prompt.get(key):
                record_errors.append(f"record {index}: {key} differs from prompt {pair_id}")
        output = row.get("output_tokens")
        generated = row.get("generated_tokens")
        seconds = row.get("outer_wall_seconds")
        eos = row.get("eos")
        if not isinstance(output, list) or not output or any(type(token) is not int for token in output):
            record_errors.append(f"record {index}: output_tokens invalid")
            output = [] if not isinstance(output, list) else output
        if type(generated) is not int or generated != len(output):
            record_errors.append(f"record {index}: generated_tokens differs from output length")
        if len(output) > budget:
            record_errors.append(f"record {index}: output exceeds plan budget")
        if not finite_number(seconds) or seconds <= 0:
            record_errors.append(f"record {index}: outer_wall_seconds must be finite and positive")
        if output and finite_number(seconds) and seconds > 0:
            if not close_number(row.get("tokens_per_second"), len(output) / seconds):
                record_errors.append(f"record {index}: tokens_per_second differs from raw length/time")
        if type(eos) is not bool or eos != bool(output and output[-1] == eos_id):
            record_errors.append(f"record {index}: eos flag differs from output")
        stats = row.get("stats")
        if not isinstance(stats, dict):
            record_errors.append(f"record {index}: stats missing or invalid")
        else:
            output_has_eos = bool(output and output[-1] == eos_id)
            if "eos" in stats and bool(stats["eos"]) != output_has_eos:
                record_errors.append(f"record {index}: stats.eos differs from normalized output")
            if "raw_eos_count" in stats:
                raw_eos_count = stats.get("raw_eos_count")
                raw_body_tokens = stats.get("raw_body_tokens")
                if type(raw_eos_count) is not int or raw_eos_count not in (0, 1):
                    record_errors.append(f"record {index}: raw_eos_count must be 0 or 1")
                elif type(raw_body_tokens) is not int or raw_body_tokens < 0:
                    record_errors.append(f"record {index}: raw_body_tokens is invalid")
                elif len(output) != raw_body_tokens + raw_eos_count:
                    record_errors.append(f"record {index}: normalized output length differs from raw EOS fields")
                if bool(raw_eos_count) != output_has_eos:
                    record_errors.append(f"record {index}: raw_eos_count differs from normalized output EOS")
        validated = row.get("sequence_validated") is True
        all_validated &= validated
        if not validated:
            record_errors.append(f"record {index}: runner did not record successful sequence validation")
        if output:
            record_errors.extend(
                f"record {index}: {message}"
                for message in validate_sequence(
                    prompt["prompt"], output, eos_id=eos_id,
                    position_vocab=position_vocab, feature_offset=feature_offset,
                    ranks=ranks
                )
            )
        counters = row.get("counters")
        if not isinstance(counters, dict):
            record_errors.append(f"record {index}: counters missing or invalid")

    expected_pair_ids = set(prompt_by_id)
    completeness_errors = []
    expected_arms = {f"{execution}:{method}" for execution in executions for method in methods}
    if expected_pair_ids != prompt_names:
        completeness_errors.append("prompt pair IDs are not unique")
    for pair_id in prompt_names:
        present = set(rows_by_pair.get(pair_id, {}))
        if present != expected_arms:
            completeness_errors.append(
                f"{pair_id}: arms missing={sorted(expected_arms-present)} extra={sorted(present-expected_arms)}"
            )

    parity_errors = []
    parity = {}
    if "original" in executions and "optimized" in executions:
        for method in methods:
            pairs = []
            for pair_id in sorted(prompt_names):
                arms = rows_by_pair.get(pair_id, {})
                old = arms.get(f"original:{method}")
                new = arms.get(f"optimized:{method}")
                if old is None or new is None:
                    continue
                output_equal = old.get("output_tokens") == new.get("output_tokens")
                counters_equal = old.get("counters") == new.get("counters")
                pairs.append({
                    "pair_id": pair_id,
                    "source": old.get("source"),
                    "output_equal": output_equal,
                    "counters_equal": counters_equal,
                })
                if not output_equal or not counters_equal:
                    parity_errors.append(f"{method}/{pair_id}: old/new output or counters differ")
            parity[method] = {
                "pair_count": len(pairs),
                "output_equal_count": sum(item["output_equal"] for item in pairs),
                "counters_equal_count": sum(item["counters_equal"] for item in pairs),
                "pairs": pairs,
            }
        summary_parity = summary.get("same_seed_parity", {})
        for method, value in parity.items():
            listed = summary_parity.get(method, {})
            for key in ("pair_count", "output_equal_count", "counters_equal_count"):
                if listed.get(key) != value[key]:
                    parity_errors.append(f"summary.same_seed_parity.{method}.{key} differs")

    calculated_arms = {}
    arm_errors = []
    for execution in executions:
        for method in methods:
            arm = f"{execution}:{method}"
            chosen = [row for row in rows if row.get("arm") == arm]
            total_tokens = sum(int(row["generated_tokens"]) for row in chosen)
            total_seconds = math.fsum(float(row["outer_wall_seconds"]) for row in chosen)
            lengths = Counter(str(int(row["generated_tokens"])) for row in chosen)
            calculated = {
                "runs": len(chosen),
                "generated_tokens": total_tokens,
                "outer_wall_seconds": total_seconds,
                "pooled_tokens_per_second": total_tokens / total_seconds if total_seconds else None,
                "eos_count": sum(bool(row.get("eos")) for row in chosen),
                "output_lengths": dict(sorted(lengths.items(), key=lambda item: int(item[0]))),
            }
            calculated_arms[arm] = calculated
            listed = summary.get("arms", {}).get(arm)
            if listed is None:
                arm_errors.append(f"summary.arms 缺少 {arm}")
            else:
                arm_errors.extend(compare_summary_number(listed, calculated, f"summary.arms.{arm}"))

    contrasts = {}
    ratio_errors = []
    contrast_defs = expected_contrasts(methods, executions)
    summary_contrasts = summary.get("contrasts", {})
    if set(summary_contrasts) != set(contrast_defs):
        ratio_errors.append(
            f"contrast 名称不同：summary={sorted(summary_contrasts)} raw={sorted(contrast_defs)}"
        )
    for label, (numerator, denominator) in contrast_defs.items():
        calculated = {
            "numerator": numerator,
            "denominator": denominator,
            "all_output_lengths": compare_ratio(rows_by_pair, numerator, denominator, budget),
            "common_full_budget": compare_ratio(rows_by_pair, numerator, denominator, budget, True),
        }
        contrasts[label] = calculated
        listed = summary_contrasts.get(label)
        if listed is None:
            ratio_errors.append(f"summary.contrasts 缺少 {label}")
        else:
            for key in ("numerator", "denominator", "all_output_lengths", "common_full_budget"):
                ratio_errors.extend(
                    compare_summary_number(listed.get(key), calculated[key], f"summary.contrasts.{label}.{key}")
                )
    family_counts = defaultdict(Counter)
    for label in contrast_defs:
        parts = label.split(":")
        family = parts[0]
        method = parts[1] if family == "execution_gain" else parts[2]
        family_counts[method][family] += 1
    if "original" in executions and "optimized" in executions and "pf_mlp" in methods:
        for method in methods:
            expected_families = 1 if method == "target" else 3 if method == "pf_mlp" else 5
            if sum(family_counts[method].values()) != expected_families:
                ratio_errors.append(
                    f"{method}: 比值类别数为 {sum(family_counts[method].values())}，预期 {expected_families}"
                )

    graph_results = {}
    graph_errors = []
    optimized_count = len(prompt_names)
    optimized_timing_text = str(plan.get("timing_scope", "")).lower()
    cold_setup_in_timing = "runtime creation" in optimized_timing_text and "graph creation" in optimized_timing_text
    if "optimized" in executions:
        final_runtime_stats = summary.get("dynamic_runtimes", {})
        for method in methods:
            arm = f"optimized:{method}"
            chosen = [row for row in rows if row.get("arm") == arm]
            calls = Counter()
            fallbacks = Counter()
            graph_events = []
            cache_events = []
            create_rows = []
            row_graph_deltas = []
            for row in chosen:
                if row.get("runtime_created") is True:
                    create_rows.append(row.get("pair_id"))
                delta = row.get("runtime_delta")
                if not isinstance(delta, dict):
                    graph_errors.append(f"{arm}/{row.get('pair_id')}: runtime_delta 缺失")
                    continue
                for key, value in (delta.get("calls") or {}).items():
                    if type(value) is not int or value < 0:
                        graph_errors.append(f"{arm}/{row.get('pair_id')}: calls.{key} 无效")
                    elif value:
                        calls[key] += value
                for key, value in (delta.get("fallbacks") or {}).items():
                    if type(value) is not int or value < 0:
                        graph_errors.append(f"{arm}/{row.get('pair_id')}: fallbacks.{key} 无效")
                    elif value:
                        fallbacks[key] += value
                new_graph_events = delta.get("new_graph_events") or []
                before = delta.get("graph_count_before")
                after = delta.get("graph_count_after")
                if type(before) is not int or type(after) is not int or after < before:
                    graph_errors.append(f"{arm}/{row.get('pair_id')}: graph_count 不递增")
                elif after - before != len(new_graph_events):
                    graph_errors.append(f"{arm}/{row.get('pair_id')}: graph 事件数与 graph_count 变化不符")
                row_graph_deltas.append({"pair_id": row.get("pair_id"), "before": before,
                                         "after": after, "new_graphs": len(new_graph_events)})
                graph_events.extend(new_graph_events)
                cache_events.extend(delta.get("new_cache_events") or [])

            graph_calls = calls.get("graph_forward", 0)
            typed_call_sum = sum(
                value for key, value in calls.items()
                if key.startswith("graph_forward_T") and key[len("graph_forward_T"):].isdigit()
            )
            if graph_calls <= 0:
                graph_errors.append(f"{arm}: 没有 graph_forward")
            if typed_call_sum != graph_calls:
                graph_errors.append(f"{arm}: graph_forward_Tn 合计与 graph_forward 不符")
            if calls.get("original_forward", 0) != sum(fallbacks.values()):
                graph_errors.append(f"{arm}: original_forward 与 fallbacks 总数不符")
            if len(create_rows) != 1:
                graph_errors.append(f"{arm}: runtime_created 次数为 {len(create_rows)}，预期 1")
            elif chosen and create_rows[0] != chosen[0].get("pair_id"):
                graph_errors.append(f"{arm}: 首次 runtime 创建不在该方法第一条运行")
            if len(graph_events) <= 0:
                graph_errors.append(f"{arm}: 没有新建图事件")
            unexpected_fallbacks = {
                reason: count for reason, count in fallbacks.items()
                if reason not in {"unsupported_query_tokens", "unsupported_window_for_full_block"}
            }
            if unexpected_fallbacks:
                graph_errors.append(f"{arm}: 存在未预期 fallback：{unexpected_fallbacks}")
            if fallbacks.get("unsupported_query_tokens", 0) > optimized_count:
                graph_errors.append(f"{arm}: 长查询 fallback 超过每次运行一次")
            if fallbacks.get("unsupported_window_for_full_block", 0) > 2 * optimized_count:
                graph_errors.append(f"{arm}: full-block 窗口 fallback 过多")

            final = final_runtime_stats.get(method)
            if not isinstance(final, dict):
                graph_errors.append(f"summary.dynamic_runtimes 缺少 {method}")
                final = {}
            if final.get("calls", {}) != dict(calls):
                graph_errors.append(f"{arm}: summary runtime calls 与逐次增量不符")
            if final.get("fallbacks", {}) != dict(fallbacks):
                graph_errors.append(f"{arm}: summary runtime fallbacks 与逐次增量不符")
            if final.get("graph_count") != len(graph_events):
                graph_errors.append(f"{arm}: summary graph_count 与逐次图事件不符")
            if len(final.get("graph_events", [])) != len(graph_events):
                graph_errors.append(f"{arm}: summary graph_events 数量与逐次图事件不符")
            if len(final.get("cache_events", [])) != len(cache_events):
                graph_errors.append(f"{arm}: summary cache_events 数量与逐次事件不符")
            for graph_index, event in enumerate(graph_events):
                for key in ("prepare_ms", "warmup_ms", "capture_ms", "memory_allocated_delta"):
                    value = event.get(key)
                    if not finite_number(value) or (key.endswith("_ms") and value < 0):
                        graph_errors.append(f"{arm}: graph event {graph_index} 的 {key} 无效")
                        break
            graph_results[method] = {
                "optimized_runs": len(chosen),
                "runtime_created_pair_ids": create_rows,
                "calls": dict(sorted(calls.items())),
                "fallbacks": dict(sorted(fallbacks.items())),
                "graph_count": len(graph_events),
                "cache_event_count": len(cache_events),
                "row_graph_deltas": row_graph_deltas,
                "cold_runtime_and_graph_creation_in_outer_timing": cold_setup_in_timing,
            }

    checks = [
        check("完整运行数量", len(rows) == expected_runs and summary.get("completed_runs") == expected_runs,
              {"prompts": len(prompts), "methods": len(methods), "executions": len(executions),
               "expected_runs": expected_runs, "records": len(rows)}),
        check("pair 与 arm 唯一齐全", not duplicate_arms and not unplanned_rows and not completeness_errors,
              {"duplicate_arms": duplicate_arms, "unplanned_rows": unplanned_rows,
               "errors": completeness_errors}),
        check("来源与 prompt 对应", source_list_match and not prompt_errors,
              {"plan_sources": sorted(expected_sources), "prompt_sources": sorted(prompt_sources),
               "errors": prompt_errors}),
        check("输出合法且完成运行器验证", not record_errors and all_validated,
              {"record_error_count": len(record_errors), "errors": record_errors[:100],
               "independent_rank_map_entries": len(ranks)}),
        check("old/new 输出与计数一致", not parity_errors,
              {"methods": parity, "errors": parity_errors}),
        check("各 arm 汇总匹配逐次记录", not arm_errors,
              {"arm_count": len(calculated_arms), "errors": arm_errors[:100]}),
        check("五类来源等权比值可复算", not ratio_errors,
              {"contrast_count": len(contrasts), "bootstrap_seed": BOOTSTRAP_SEED,
               "bootstrap_draws": BOOTSTRAP_DRAWS,
               "ratio_families_by_method": {method: dict(sorted(counts.items()))
                                            for method, counts in family_counts.items()},
               "contrasts": contrasts,
               "errors": ratio_errors[:100]}),
        check("Target 图调用与回退可复算", not graph_errors and cold_setup_in_timing,
              {"methods": graph_results, "cold_setup_in_timing_scope": cold_setup_in_timing,
               "errors": graph_errors[:100]}),
    ]
    failures = [item for item in checks if not item["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "run_dir": str(run_dir.resolve()),
        "audit_scope": "CPU-only recomputation from plan.json, prompts.json, records.jsonl, and summary.json; no runner or metrics module imported.",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                      "unit": "source", "quantile": "linear interpolation"},
        "run_shape": {"prompt_count": len(prompts), "method_count": len(methods),
                      "execution_count": len(executions), "record_count": len(rows),
                      "budget": budget, "sources": sorted(prompt_sources),
                      "methods": methods, "executions": executions},
        "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="包含 plan/prompts/records/summary 的已完成目录")
    parser.add_argument("--output", type=Path, help="审计 JSON 路径，默认 run_dir/audit.json")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    output = (args.output if args.output is not None else run_dir / "audit.json").resolve()
    if not run_dir.is_dir():
        parser.error(f"run_dir 不存在或不是目录：{run_dir}")
    if output.exists():
        parser.error(f"输出文件已存在，为保留旧产物而停止：{output}")
    try:
        report = run_audit(run_dir)
    except AuditInputError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        parser.error(f"输出文件已存在，为保留旧产物而停止：{output}")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "passed_checks": sum(item["passed"] for item in report["checks"]),
                      "check_count": len(report["checks"])}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
