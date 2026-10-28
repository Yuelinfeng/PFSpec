"""Run diagnostic selector parity checks through the normal experiment runner."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
from numbers import Real
from pathlib import Path
import sys
from types import SimpleNamespace


METHODS = ("old_value", "value_reference", "value")
PAIR_METHODS = (
    ("old_value", "value_reference"),
    ("old_value", "value"),
    ("value_reference", "value"),
)
TOLERANCE = 1e-12


def _safe_scalar(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("selector trace contains a non-finite number")
        return result
    return str(value)


def _canonical_trace(selection_module, model, round_context):
    normalized, candidates, truncated = selection_module._candidate_rows(
        round_context, model
    )
    context_fields = (
        "prefix_len", "remaining_budget", "cache_pos", "read_bucket",
        "graph_exists", "graph_cold", "previous_position_rank",
        "main_position", "main_position_rank", "main_score",
        "candidate_count", "position_vocab_size",
    )
    context = {
        key: _safe_scalar(normalized.get(key))
        for key in context_fields
    }
    graph_state = {}
    for key, item in normalized.get("graph_state", {}).items():
        graph_state[str(key)] = {
            field: _safe_scalar(item.get(field))
            for field in ("read_bucket", "graph_exists", "candidate_available")
            if field in item
        }
    context["graph_state"] = graph_state

    candidate_features = []
    for row in candidates:
        candidate_features.append({
            "token_id": int(row["token_id"]),
            "ranker_rank": int(row["ranker_rank"]),
            "ranker_score": _safe_scalar(row["ranker_score"]),
            "position_rank": _safe_scalar(row.get("position_rank")),
            "rank_gap": _safe_scalar(row.get("rank_gap")),
            "features": {
                key: float(value)
                for key, value in selection_module.candidate_feature_values(
                    normalized, row
                ).items()
            },
        })
    raw_pool = round_context.get("candidates", round_context.get("candidate_pool", ()))
    candidate_pool = []
    for row in raw_pool:
        candidate_pool.append({
            key: _safe_scalar(row.get(key))
            for key in (
                "token_id", "position_id", "ranker_rank", "rank",
                "ranker_score", "score", "position_rank", "rank_value",
                "rank_gap", "legal", "is_eos",
            ) if key in row
        })
    return {
        "round_index": _safe_scalar(round_context.get("round_index")),
        "context": context,
        "candidate_pool": candidate_pool,
        "candidate_features": candidate_features,
        "candidate_truncation": int(truncated),
        "has_backup_candidates": bool(candidates),
    }


def _differences(left, right, path="", tolerance=TOLERANCE, limit=20):
    result = []
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys, right_keys = set(left), set(right)
        if left_keys != right_keys:
            result.append({
                "field": path or "<root>",
                "left_only": sorted(left_keys - right_keys),
                "right_only": sorted(right_keys - left_keys),
            })
        for key in sorted(left_keys & right_keys):
            result.extend(_differences(
                left[key], right[key], f"{path}.{key}" if path else str(key),
                tolerance, limit,
            ))
            if len(result) >= limit:
                break
        return result[:limit]
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            result.append({"field": path, "left_length": len(left), "right_length": len(right)})
            return result
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result.extend(_differences(
                left_item, right_item, f"{path}[{index}]", tolerance, limit,
            ))
            if len(result) >= limit:
                break
        return result[:limit]
    numeric = (
        isinstance(left, Real) and not isinstance(left, bool) and
        isinstance(right, Real) and not isinstance(right, bool)
    )
    if numeric:
        delta = abs(float(left) - float(right))
        if delta > tolerance:
            result.append({"field": path, "left": left, "right": right, "abs_delta": delta})
    elif left != right:
        result.append({"field": path, "left": left, "right": right})
    return result


def _decision_comparison(left, right):
    left_row = left.as_dict()
    right_row = right.as_dict()
    differences = _differences(left_row, right_row)
    return {
        "passed": not differences,
        "fields_compared": sorted(set(left_row) | set(right_row)),
        "differences": differences,
    }


class _RecordingSelector:
    def __init__(self, method, selector, selection_module, traces, compiled_models):
        self.method = method
        self.selector = selector
        self.selection = selection_module
        self.traces = traces
        self.compiled_models = compiled_models

    def __call__(self, round_context):
        selected = [int(token) for token in self.selector(round_context)]
        snapshot = _canonical_trace(
            self.selection, self.selector.model, round_context
        )
        row = {
            "method": self.method,
            "call_index": len(self.traces[self.method]),
            **snapshot,
            "backup_ids": selected,
        }
        if self.method != "old_value":
            options = {
                "mode": self.selector.mode,
                "max_backups": self.selector.max_backups,
                "fixed_k": self.selector.fixed_k,
            }
            reference = self.selection.choose_backups(
                self.selector.model, round_context, **options,
            )
            compiled = self.selection.choose_backups_compiled(
                self.selector.model, round_context,
                compiled_models=self.compiled_models[id(self.selector.model)],
                **options,
            )
            row["reference_decision"] = reference.as_dict()
            row["compiled_decision"] = compiled.as_dict()
            row["reference_compiled_comparison"] = _decision_comparison(
                reference, compiled
            )
            row["actual_reference_comparison"] = _decision_comparison(
                self.selector.last_decision, reference
            )
            row["actual_compiled_comparison"] = _decision_comparison(
                self.selector.last_decision, compiled
            )
            row["returned_ids_reference_equal"] = selected == list(reference.backup_ids)
            row["returned_ids_compiled_equal"] = selected == list(compiled.backup_ids)
        self.traces[self.method].append(row)
        return selected

    def stats(self):
        result = dict(self.selector.stats())
        result["calls"] = int(result.get("selector_calls", 0))
        return result

    def __getattr__(self, name):
        return getattr(self.selector, name)


def _load_runner(runner_path):
    spec = importlib.util.spec_from_file_location(
        "recovery_value_experiment_runner", runner_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _method_trace_slice(row, traces, method):
    before = row.get("selector_stats_before") or {}
    after = row.get("selector_stats_after") or {}
    start = int(before.get("selector_calls", before.get("calls", 0)))
    end = int(after.get("selector_calls", after.get("calls", 0)))
    if end < start or end > len(traces[method]):
        raise ValueError(f"invalid selector call offsets for {method}: {start}:{end}")
    selected = traces[method][start:end]
    if len(selected) != end - start:
        raise ValueError(f"trace count differs from selector counters for {method}")
    for item in selected:
        item["scope"] = "formal_request"
        item["request"] = {
            "pair_id": row.get("pair_id"),
            "context_id": row.get("context_id"),
            "source": row.get("source"),
            "prefix": row.get("prefix"),
            "seed": row.get("seed"),
            "repeat_index": row.get("repeat_index"),
        }
    return selected


def _request_key(row):
    return (row.get("pair_id"), row.get("source"), row.get("prefix"), row.get("seed"))


def _match_rounds(left, right):
    left_rows = [item for item in left if item["has_backup_candidates"]]
    right_rows = [item for item in right if item["has_backup_candidates"]]
    errors = []
    if len(left_rows) != len(right_rows):
        errors.append({"issue": "nonempty_callback_count", "left": len(left_rows), "right": len(right_rows)})
    count = min(len(left_rows), len(right_rows))
    comparisons = []
    for index in range(count):
        left_item, right_item = left_rows[index], right_rows[index]
        input_diff = _differences(
            {key: left_item[key] for key in (
                "context", "candidate_pool", "candidate_features",
                "candidate_truncation",
            )},
            {key: right_item[key] for key in (
                "context", "candidate_pool", "candidate_features",
                "candidate_truncation",
            )},
        )
        action_equal = left_item["backup_ids"] == right_item["backup_ids"]
        comparisons.append({
            "round_index": left_item.get("round_index"),
            "inputs_equal_with_tolerance": not input_diff,
            "backup_ids_equal": action_equal,
            "input_differences": input_diff,
            "left_backup_ids": left_item["backup_ids"],
            "right_backup_ids": right_item["backup_ids"],
        })
        if input_diff or not action_equal:
            errors.append({
                "issue": "round_mismatch",
                "round_index": left_item.get("round_index"),
                "input_differences": input_diff,
                "left_backup_ids": left_item["backup_ids"],
                "right_backup_ids": right_item["backup_ids"],
            })
    return comparisons, errors


def _prepare_import_paths(source_dir):
    """Mirror run_experiment's module precedence before importing adapters."""
    project = Path("/root/workspace/GaussianGPT")
    source = Path("/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04")
    old = Path("/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_widths")
    paths = [
        source_dir, old, source, source / "reference",
        source / "reference" / "runtime", project,
    ]
    sys.path[:0] = [str(path) for path in paths]
    return paths


def _plan_errors(plan):
    errors = []
    expected_args = {
        "phase": "timing", "methods": ",".join(METHODS), "prefixes": "64,256",
        "sources": 2, "seeds": 1, "seed_start": 58101, "budget": 512,
    }
    actual_args = plan.get("args", {})
    for key, expected in expected_args.items():
        actual = actual_args.get(key)
        if actual != expected:
            errors.append({"issue": "plan_argument", "field": key,
                           "expected": expected, "actual": actual})
    expected_methods = list(METHODS)
    if plan.get("methods") != expected_methods:
        errors.append({"issue": "plan_methods", "expected": expected_methods,
                       "actual": plan.get("methods")})
    return errors


def _run(args):
    source_dir = Path(__file__).resolve().parent
    runner_path = source_dir / "run_experiment.py"
    if not runner_path.is_file():
        raise FileNotFoundError(runner_path)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)

    _prepare_import_paths(source_dir)
    selection_module = importlib.import_module("selection")
    adaptive_module = importlib.import_module("adaptive_decoder")
    traces = {method: [] for method in METHODS}
    compiled_models = {}
    factory_calls = []
    expected_factory_calls = {
        "old_value": ("value", "reference"),
        "value_reference": ("value", "reference"),
        "value": ("value", "compiled"),
    }
    original_make_selector = selection_module.make_selector

    def recording_make_selector(metadata, method, *, implementation="compiled"):
        call_index = len(factory_calls)
        if call_index >= len(METHODS):
            raise AssertionError("unexpected extra selector factory call")
        label = METHODS[call_index]
        expected_method, expected_implementation = expected_factory_calls[label]
        if (method, implementation) != (expected_method, expected_implementation):
            raise AssertionError(
                f"selector creation order mismatch for {label}: "
                f"got {(method, implementation)}"
            )
        selector = original_make_selector(
            metadata, method, implementation=implementation
        )
        compiled_models.setdefault(
            id(selector.model), selection_module._CompiledModels.build(selector.model)
        )
        factory_calls.append({
            "method": label,
            "selection_method": method,
            "implementation": implementation,
        })
        return _RecordingSelector(
            label, selector, selection_module, traces, compiled_models
        )

    selection_module.make_selector = recording_make_selector
    sentinel_calls = {"deepcopy": 0, "runtime_snapshot": 0}

    def forbidden_deepcopy(*_args, **_kwargs):
        sentinel_calls["deepcopy"] += 1
        raise AssertionError("new adaptive path called copy.deepcopy")

    def forbidden_runtime_snapshot(*_args, **_kwargs):
        sentinel_calls["runtime_snapshot"] += 1
        raise AssertionError("new adaptive path collected a runtime snapshot")

    adaptive_module.copy = SimpleNamespace(deepcopy=forbidden_deepcopy)
    adaptive_module._copy_graph_snapshot = forbidden_runtime_snapshot

    runner = _load_runner(runner_path)
    argv = [
        str(runner_path), "--phase", "timing",
        "--crops", str(args.crops.resolve()), "--out", str(out),
        "--model", str(args.model.resolve()), "--methods", ",".join(METHODS),
        "--prefixes", "64,256", "--sources", "2", "--seeds", "1",
        "--seed-start", "58101", "--budget", "512", "--repeat-index", "0",
        "--max-wall-seconds", str(args.max_wall_seconds),
        "--legacy-adaptive", str(args.legacy_adaptive.resolve()),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = argv
        runner.main()
    finally:
        sys.argv = previous_argv
        selection_module.make_selector = original_make_selector

    if factory_calls != [
        {"method": method, "selection_method": selector_method,
         "implementation": implementation}
        for method, (selector_method, implementation) in expected_factory_calls.items()
    ]:
        raise AssertionError(f"selector creation order was {factory_calls}")

    plan = json.loads((out / "plan.json").read_text())
    plan["correctness_trace_only"] = True
    plan["performance_timing_usable"] = False
    plan["selector_comparison_work_injected"] = True
    (out / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    progress = json.loads((out / "progress.json").read_text())
    records = [
        json.loads(line)
        for line in (out / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if progress.get("status") != "complete":
        raise RuntimeError(f"experiment runner did not complete: {progress}")
    if len(records) != 12 or progress.get("expected") != 12:
        raise RuntimeError(f"unexpected request count: {len(records)}")

    errors = []
    grouped = {}
    request_rows = []
    for row in records:
        method = row["method"]
        selected = _method_trace_slice(row, traces, method)
        before_stats = row.get("selector_stats_before") or {}
        after_stats = row.get("selector_stats_after") or {}
        selector_calls = int(after_stats.get(
            "selector_calls", after_stats.get("calls", 0)
        )) - int(before_stats.get(
            "selector_calls", before_stats.get("calls", 0)
        ))
        decoder_stats = row.get("stats", {})
        reported_calls = int(decoder_stats.get("backup_selector_calls", 0))
        empty_skips = int(
            decoder_stats.get("backup_selector_empty_candidate_skips", 0)
        )
        eligible_rounds = int(
            decoder_stats.get("backup_selector_eligible_rounds", 0)
        )
        if selector_calls != len(selected) or reported_calls != selector_calls:
            errors.append({
                "issue": "selector_call_accounting",
                "method": method,
                "request": _request_key(row),
                "trace_calls": len(selected),
                "selector_stats_delta": selector_calls,
                "decoder_reported_calls": reported_calls,
            })
        if method != "old_value" and eligible_rounds != selector_calls + empty_skips:
            errors.append({
                "issue": "eligible_round_accounting",
                "method": method,
                "request": _request_key(row),
                "eligible_rounds": eligible_rounds,
                "selector_calls_plus_empty_skips": selector_calls + empty_skips,
            })
        key = _request_key(row)
        grouped.setdefault(key, {})[method] = selected
        request_rows.append({
            "request": {
                "pair_id": row.get("pair_id"),
                "context_id": row.get("context_id"),
                "source": row.get("source"),
                "prefix": row.get("prefix"),
                "seed": row.get("seed"),
            },
            "method": method,
            "selector_calls_before": before_stats.get("selector_calls", before_stats.get("calls", 0)),
            "selector_calls_after": after_stats.get("selector_calls", after_stats.get("calls", 0)),
            "formal_trace_calls": len(selected),
            "decoder_reported_calls": reported_calls,
            "decoder_zero_actions": int(decoder_stats.get("backup_selector_zero_actions", 0)),
            "decoder_empty_candidate_skips": empty_skips,
            "decoder_eligible_rounds": eligible_rounds,
            "empty_candidate_callbacks": sum(
                not item["has_backup_candidates"] for item in selected
            ),
            "backup_ids_by_round": [
                {"round_index": item["round_index"], "ids": item["backup_ids"]}
                for item in selected
            ],
        })

    if len(grouped) != 4:
        errors.append({"issue": "unique_request_count", "expected": 4,
                       "actual": len(grouped)})
    sources = {row.get("source") for row in records}
    prefixes = {row.get("prefix") for row in records}
    if len(sources) != 2 or prefixes != {64, 256}:
        errors.append({"issue": "request_matrix", "sources": sorted(sources),
                       "prefixes": sorted(prefixes)})
    for method in METHODS:
        method_rows = [row for row in request_rows if row["method"] == method]
        if len(method_rows) != 4:
            errors.append({"issue": "method_request_count", "method": method,
                           "expected": 4, "actual": len(method_rows)})
    request_accounting = {}
    for row in request_rows:
        req = row["request"]
        key = (req.get("pair_id"), req.get("source"), req.get("prefix"), req.get("seed"))
        request_accounting[(key, row["method"])] = row
    pair_summary = []
    for key, methods in grouped.items():
        if set(methods) != set(METHODS):
            errors.append({"request": key, "issue": "method_set", "methods": sorted(methods)})
            continue
        old_accounting = request_accounting.get((key, "old_value"))
        if old_accounting is not None:
            for new_method in METHODS[1:]:
                new_accounting = request_accounting.get((key, new_method))
                if new_accounting is None:
                    continue
                if old_accounting["decoder_reported_calls"] != (
                    new_accounting["decoder_reported_calls"] +
                    new_accounting["decoder_empty_candidate_skips"]
                ):
                    errors.append({
                        "issue": "legacy_new_call_mapping", "request": key,
                        "new_method": new_method,
                        "old_calls": old_accounting["decoder_reported_calls"],
                        "new_calls_plus_skips": (
                            new_accounting["decoder_reported_calls"] +
                            new_accounting["decoder_empty_candidate_skips"]
                        ),
                    })
                if old_accounting["decoder_zero_actions"] != (
                    new_accounting["decoder_zero_actions"] +
                    new_accounting["decoder_empty_candidate_skips"]
                ):
                    errors.append({
                        "issue": "legacy_new_zero_action_mapping", "request": key,
                        "new_method": new_method,
                        "old_zero_actions": old_accounting["decoder_zero_actions"],
                        "new_zero_actions_plus_skips": (
                            new_accounting["decoder_zero_actions"] +
                            new_accounting["decoder_empty_candidate_skips"]
                        ),
                    })
        for left_method, right_method in PAIR_METHODS:
            comparisons, pair_errors = _match_rounds(
                methods[left_method], methods[right_method]
            )
            pair_summary.append({
                "request": key,
                "left_method": left_method,
                "right_method": right_method,
                "left_empty_candidate_callbacks": sum(
                    not item["has_backup_candidates"] for item in methods[left_method]
                ),
                "right_empty_candidate_callbacks": sum(
                    not item["has_backup_candidates"] for item in methods[right_method]
                ),
                "compared_nonempty_rounds": len(comparisons),
                "comparisons": comparisons,
            })
            errors.extend({"request": key, "left_method": left_method,
                           "right_method": right_method, **error}
                          for error in pair_errors)

    callback_parity = []
    for method in METHODS[1:]:
        for item in traces[method]:
            if item.get("scope") != "formal_request":
                continue
            for field in (
                "reference_compiled_comparison",
                "actual_reference_comparison",
                "actual_compiled_comparison",
            ):
                comparison = item[field]
                callback_parity.append({
                    "method": method,
                    "call_index": item["call_index"],
                    "request": item.get("request"),
                    "comparison": field,
                    "passed": comparison["passed"],
                    "differences": comparison["differences"],
                })
                if not comparison["passed"]:
                    errors.append({
                        "issue": "reference_compiled_decision_mismatch",
                        "method": method,
                        "call_index": item["call_index"],
                        "comparison": field,
                        "differences": comparison["differences"],
                    })
            for field in ("returned_ids_reference_equal", "returned_ids_compiled_equal"):
                if not item[field]:
                    errors.append({
                        "issue": "returned_action_mismatch",
                        "method": method,
                        "call_index": item["call_index"],
                        "request": item.get("request"),
                        "comparison": field,
                        "selected_ids": item["backup_ids"],
                        "reference_ids": item["reference_decision"]["backup_ids"],
                        "compiled_ids": item["compiled_decision"]["backup_ids"],
                    })

    if sentinel_calls != {"deepcopy": 0, "runtime_snapshot": 0}:
        errors.append({"issue": "lightweight_path_sentinel", "calls": sentinel_calls})
    expected_args = plan.get("args", {})
    errors.extend(_plan_errors(plan))

    all_traces = []
    for method in METHODS:
        for item in traces[method]:
            if "scope" not in item:
                item["scope"] = "prewarm"
            all_traces.append(item)
    trace_payload = {
        "schema": "online_execution_trace_v1",
        "diagnostic_only": True,
        "performance_timing_usable": False,
        "methods": list(METHODS),
        "selector_creation_order": factory_calls,
        "selection_tolerance": TOLERANCE,
        "traces": all_traces,
    }
    validation = {
        "schema": "online_execution_validation_v1",
        "status": "passed" if not errors else "failed",
        "diagnostic_only": True,
        "performance_timing_usable": False,
        "request_count": len(records),
        "unique_request_count": len(grouped),
        "methods": list(METHODS),
        "selector_creation_order": factory_calls,
        "formal_selector_calls_by_method": {
            method: sum(row["method"] == method and row["formal_trace_calls"]
                        for row in request_rows)
            for method in METHODS
        },
        "formal_empty_candidate_callbacks_by_method": {
            method: sum(row["method"] == method and row["empty_candidate_callbacks"]
                        for row in request_rows)
            for method in METHODS
        },
        "reference_compiled_callback_checks": len(callback_parity),
        "reference_compiled_callback_failures": sum(not row["passed"] for row in callback_parity),
        "compared_pairs": len(pair_summary),
        "compared_nonempty_rounds": sum(row["compared_nonempty_rounds"] for row in pair_summary),
        "lightweight_path_sentinel_calls": sentinel_calls,
        "selection_tolerance": TOLERANCE,
        "plan_args": expected_args,
        "errors": errors,
        "request_rows": request_rows,
        "pair_summary": pair_summary,
        "callback_parity": callback_parity,
        "scope": (
            "Correctness trace only. Selector reference/compiled comparison work is injected "
            "inside callbacks; saved wall times must not be interpreted as performance results."
        ),
    }
    (out / "traces.json").write_text(
        json.dumps(trace_payload, indent=2, allow_nan=False) + "\n"
    )
    (out / "validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": validation["status"],
        "requests": validation["request_count"],
        "pairs": validation["compared_pairs"],
        "rounds": validation["compared_nonempty_rounds"],
        "errors": len(errors),
    }, sort_keys=True))
    return 0 if not errors else 1


def _self_test():
    source_dir = Path(__file__).resolve().parent
    inserted_paths = _prepare_import_paths(source_dir)
    if sys.path[:len(inserted_paths)] != [str(path) for path in inserted_paths]:
        raise AssertionError("experiment import precedence differs from runner")
    selection_module = importlib.import_module("selection")
    model = {
        "schema_version": 1,
        "kind": "gaussiangpt_recovery_value_cpu_model",
        "config": {
            "max_backups": 1, "max_measured_backups": 1,
            "default_max_backups": 1, "max_enum_candidates": 8,
            "supported_k": [0, 1], "position_vocab_size": 8000,
            "eos_id": 12096,
        },
        "models": {
            name: {"feature_names": ["bias"], "means": {"bias": 0.0},
                   "scales": {"bias": 1.0}, "weights": {"bias": weight}}
            for name, weight in (
                ("residual_mass", 0.2), ("recovery_value_ms", 1.0),
                ("recovery_cost_ms", 2.0), ("shape_cost_ms", 0.0),
            )
        },
    }
    context = {
        "prefix_len": 64, "remaining_budget": 128, "cache_pos": 63,
        "read_bucket": 128, "graph_exists": True, "graph_cold": False,
        "position_vocab_size": 8000, "eos_id": 12096,
        "main_position": 100, "main_score": 0.9, "main_position_rank": 90,
        "previous_position_rank": 80,
        "graph_state": {
            "0": {"read_bucket": 128, "graph_exists": True,
                  "candidate_available": True},
            "1": {"read_bucket": 256, "graph_exists": True,
                  "candidate_available": True},
        },
        "candidates": [
            {"token_id": 100, "ranker_rank": 0, "ranker_score": 0.9,
             "position_rank": 90},
            {"token_id": 101, "ranker_rank": 1, "ranker_score": 0.7,
             "position_rank": 100},
        ],
    }
    traces = {method: [] for method in METHODS}
    selector_reference = selection_module.make_selector(
        model, "value", implementation="reference"
    )
    selector_compiled = selection_module.make_selector(
        model, "value", implementation="compiled"
    )
    compiled_models = {id(model): selection_module._CompiledModels.build(model)}
    reference_wrapper = _RecordingSelector(
        "value_reference", selector_reference, selection_module, traces, compiled_models
    )
    compiled_wrapper = _RecordingSelector(
        "value", selector_compiled, selection_module, traces, compiled_models
    )
    reference_ids = reference_wrapper(context)
    compiled_ids = compiled_wrapper(context)
    if reference_ids != compiled_ids:
        raise AssertionError("reference and compiled actions differ in self-test")
    if not traces["value_reference"][0]["reference_compiled_comparison"]["passed"]:
        raise AssertionError("reference and compiled decision fields differ in self-test")
    if not traces["value"][0]["actual_reference_comparison"]["passed"]:
        raise AssertionError("actual selector differs from its paired decision in self-test")
    old_raw_context = {**context, "round_index": 1}
    new_raw_context = dict(context)
    old_snapshot = _canonical_trace(selection_module, model, old_raw_context)
    new_snapshot = _canonical_trace(selection_module, model, new_raw_context)
    snapshot = new_snapshot
    if not snapshot["has_backup_candidates"] or len(snapshot["candidate_features"]) != 1:
        raise AssertionError("candidate trace was not captured in self-test")
    left = {**old_snapshot, "backup_ids": list(reference_ids)}
    right = {**new_snapshot, "backup_ids": list(compiled_ids)}
    comparisons, mismatches = _match_rounds([left], [right])
    if mismatches or not comparisons[0]["inputs_equal_with_tolerance"]:
        raise AssertionError("legacy round_index leaked into input parity matching")
    changed_candidate = {
        **snapshot,
        "candidate_pool": [dict(row) for row in snapshot["candidate_pool"]],
        "candidate_features": [dict(row) for row in snapshot["candidate_features"]],
        "backup_ids": list(compiled_ids),
    }
    changed_candidate["candidate_pool"][0]["token_id"] = 999
    changed_candidate["candidate_features"][0]["token_id"] = 999
    _, mismatches = _match_rounds([left], [changed_candidate])
    if not mismatches:
        raise AssertionError("candidate ID change was not detected")
    good_plan = {
        "args": {"phase": "timing", "methods": ",".join(METHODS),
                 "prefixes": "64,256", "sources": 2, "seeds": 1,
                 "seed_start": 58101, "budget": 512},
        "methods": list(METHODS),
    }
    if _plan_errors(good_plan):
        raise AssertionError("expected plan representation failed validation")
    bad_plan = {**good_plan, "args": {**good_plan["args"], "methods": list(METHODS)}}
    if not _plan_errors(bad_plan):
        raise AssertionError("plan method representation mismatch was not detected")
    return {"passed": True, "reference_compiled_action": reference_ids}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--legacy-adaptive", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=180)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), sort_keys=True))
        return 0
    missing = [name for name in ("crops", "out", "model", "legacy_adaptive")
               if getattr(args, name) is None]
    if missing:
        parser.error("required for a run: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
