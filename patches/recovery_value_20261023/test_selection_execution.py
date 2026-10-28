from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selection import (  # noqa: E402
    _CompiledModels,
    _candidate_rows,
    choose_backups,
    choose_backups_compiled,
    make_selector,
)
import fit_selection as fit_module  # noqa: E402


MODEL_PATH = ROOT / "docs/project/artifacts/recovery_value_20261023/selection_model.json"
RECORDS_PATH = ROOT / "docs/project/artifacts/recovery_value_20261023/dev_collect_run01/rounds.jsonl"
METHODS = {
    "value": "net_value",
    "mass": "mass_times_cost",
    "ranker_same_k": "ranker_same_k",
}


def _constant_linear(value: float) -> dict[str, object]:
    return {
        "feature_names": ["bias"],
        "means": {"bias": 0.0},
        "scales": {"bias": 1.0},
        "weights": {"bias": float(value)},
    }


def _synthetic_model(*, mass: float = 0.5, recovery_value: float = 2.0,
                     recovery_cost: float = 2.0, shape_cost: float = 1.0,
                     max_backups: int = 1, supported_k: list[int] | None = None,
                     include_supported_k: bool = True) -> dict[str, object]:
    config = {
        "position_vocab_size": 8000,
        "eos_id": 12096,
        "max_backups": max_backups,
        "max_measured_backups": max_backups,
        "max_enum_candidates": 8,
        "default_max_backups": max_backups,
    }
    if include_supported_k:
        config["supported_k"] = supported_k if supported_k is not None else list(range(max_backups + 1))
    return {
        "schema_version": 1,
        "kind": "gaussiangpt_recovery_value_cpu_model",
        "config": config,
        "models": {
            "residual_mass": _constant_linear(mass),
            "recovery_value_ms": _constant_linear(recovery_value),
            "recovery_cost_ms": _constant_linear(recovery_cost),
            "shape_cost_ms": _constant_linear(shape_cost),
        },
    }


def _synthetic_context(*, candidates: list[dict[str, object]] | None = None) -> dict[str, object]:
    if candidates is None:
        candidates = [
            {"token_id": 100, "ranker_rank": 0, "ranker_score": 0.9, "position_rank": 5},
            {"token_id": 300, "ranker_rank": 2, "ranker_score": 0.2, "position_rank": 11},
            {"token_id": 200, "ranker_rank": 2, "ranker_score": 0.2, "position_rank": 9},
        ]
    return {
        "prefix_len": 16,
        "remaining_budget": 64,
        "cache_pos": 16,
        "read_bucket": 32,
        "graph_exists": True,
        "graph_cold": False,
        "position_vocab_size": 8000,
        "eos_id": 12096,
        "main_position": 100,
        "main_score": 0.9,
        "main_position_rank": 5,
        "previous_position_rank": 3,
        "candidate_pool": candidates,
        "graph_state": {
            "0": {"read_bucket": 32, "graph_exists": True},
            "1": {"read_bucket": 64, "graph_exists": True, "candidate_available": True},
        },
    }


def _wide_context() -> dict[str, object]:
    candidates = [
        {"token_id": 100, "ranker_rank": 0, "ranker_score": 0.9, "position_rank": 5},
    ]
    for index, token in enumerate((201, 202, 203, 204, 205), start=1):
        candidates.append({
            "token_id": token,
            "ranker_rank": index,
            "ranker_score": 0.8 - 0.1 * index,
            "position_rank": 5 + index * 3,
        })
    context = _synthetic_context(candidates=candidates)
    context["graph_state"] = {
        str(count): {
            "read_bucket": 32 * (count + 1),
            "graph_exists": True,
            "candidate_available": True,
        }
        for count in range(5)
    }
    return context


class CompiledSelectionSyntheticTests(unittest.TestCase):
    def _assert_same(self, model, context, *, mode, max_backups=1, fixed_k=None):
        reference = choose_backups(
            model, context, mode=mode, max_backups=max_backups, fixed_k=fixed_k
        )
        compiled = choose_backups_compiled(
            model,
            context,
            mode=mode,
            max_backups=max_backups,
            fixed_k=fixed_k,
            compiled_models=_CompiledModels.build(model),
        )
        self.assertEqual(reference.backup_ids, compiled.backup_ids)
        self.assertEqual(reference.mode, compiled.mode)
        self.assertEqual(reference.evaluated_subsets, compiled.evaluated_subsets)
        self.assertEqual(reference.candidate_truncation, compiled.candidate_truncation)
        self.assertEqual(reference.ranker_prefix_matches, compiled.ranker_prefix_matches)
        for field in (
            "predicted_mass",
            "predicted_recovery_value_ms",
            "predicted_shape_cost_ms",
            "predicted_net_value_ms",
        ):
            self.assertTrue(math.isclose(
                getattr(reference, field), getattr(compiled, field),
                rel_tol=1e-12, abs_tol=1e-12,
            ), (field, getattr(reference, field), getattr(compiled, field)))
        return compiled

    def test_equal_scores_keep_ranker_order_and_empty_pool(self):
        model = _synthetic_model(recovery_value=2.0, shape_cost=1.0)
        context = _synthetic_context()
        result = self._assert_same(model, context, mode="net_value")
        self.assertEqual(result.backup_ids, (200,))

        empty_context = _synthetic_context(candidates=[
            {"token_id": 100, "ranker_rank": 0, "ranker_score": 0.9, "position_rank": 5},
            {"token_id": 12096, "ranker_rank": 1, "ranker_score": 0.1, "is_eos": True},
        ])
        empty_result = self._assert_same(model, empty_context, mode="net_value")
        self.assertEqual(empty_result.backup_ids, ())
        self.assertEqual(empty_result.evaluated_subsets, 1)
        self.assertEqual(empty_result.predicted_net_value_ms, 0.0)

    def test_mass_product_preserves_raw_mass_and_ms_units(self):
        model = _synthetic_model(mass=1.5, recovery_cost=2.0, shape_cost=0.0)
        result = self._assert_same(model, _synthetic_context(), mode="mass_times_cost")
        self.assertEqual(result.backup_ids, (200,))
        self.assertEqual(result.predicted_mass, 1.0)
        self.assertEqual(result.predicted_recovery_value_ms, 3.0)
        self.assertEqual(result.predicted_net_value_ms, 3.0)

    def test_backup_limits_supported_shapes_and_graph_availability(self):
        modes = (
            "net_value",
            "mass_times_cost",
            "ranker_same_k",
            "mass_topk",
            "value_topk",
            "ranker_topk",
        )
        for maximum in range(5):
            with self.subTest(max_backups=maximum):
                model = _synthetic_model(max_backups=4)
                for mode in modes:
                    self._assert_same(
                        model, _wide_context(), mode=mode, max_backups=maximum
                    )

        model = _synthetic_model(max_backups=4, include_supported_k=False)
        for mode in modes:
            self._assert_same(model, _wide_context(), mode=mode, max_backups=4)

        model = _synthetic_model(max_backups=4, supported_k=[0, 2, 3, 4])
        context = _wide_context()
        context["graph_state"].update({
            "2": {"candidate_available": False, "read_bucket": 128, "graph_exists": True},
            "3": {"candidate_available": True, "read_bucket": 256, "graph_exists": False},
            "4": {"candidate_available": False, "read_bucket": 512, "graph_exists": False},
        })
        for mode in modes:
            self._assert_same(model, context, mode=mode, max_backups=4)

    def test_fixed_k_success_and_failure_paths_match(self):
        model = _synthetic_model(max_backups=4, supported_k=[0, 2, 4])
        context = _wide_context()
        context["graph_state"].update({
            "2": {"candidate_available": True, "read_bucket": 128, "graph_exists": True},
            "4": {"candidate_available": False, "read_bucket": 512, "graph_exists": False},
        })
        for fixed_k in (0, 2):
            for mode in ("net_value", "mass_times_cost", "ranker_same_k", "ranker_topk"):
                self._assert_same(
                    model, context, mode=mode, max_backups=4, fixed_k=fixed_k
                )
        for fixed_k in (1, 3, 4):
            with self.subTest(fixed_k=fixed_k):
                errors = []
                for chooser in (choose_backups, choose_backups_compiled):
                    try:
                        chooser(
                            model,
                            context,
                            mode="net_value",
                            max_backups=4,
                            fixed_k=fixed_k,
                        )
                    except Exception as error:
                        errors.append((type(error), str(error)))
                    else:
                        errors.append(None)
                self.assertEqual(errors[0], errors[1])
                self.assertIsNotNone(errors[0])

    def test_threshold_unknown_features_and_missing_models(self):
        context = _synthetic_context()
        below = _synthetic_model(recovery_value=1.0, shape_cost=1.0 - 0.5e-12)
        above = _synthetic_model(recovery_value=1.0, shape_cost=1.0 - 2.0e-12)
        self.assertEqual(
            self._assert_same(below, context, mode="net_value").backup_ids, ()
        )
        self.assertEqual(
            self._assert_same(above, context, mode="net_value").backup_ids, (200,)
        )

        model = _synthetic_model(recovery_value=1.25, shape_cost=0.0)
        model["models"]["recovery_value_ms"] = {
            "feature_names": ["unseen_feature", "ranker_score", "bias"],
            "means": {"unseen_feature": 0.0, "ranker_score": 0.0, "bias": 0.0},
            "scales": {"unseen_feature": 1.0, "ranker_score": 1.0, "bias": 1.0},
            "weights": {"unseen_feature": 1000.0, "ranker_score": 0.0, "bias": 1.25},
        }
        self._assert_same(model, context, mode="net_value")

        for missing_name in (
            "residual_mass", "recovery_value_ms", "recovery_cost_ms", "shape_cost_ms"
        ):
            with self.subTest(missing_model=missing_name):
                model = _synthetic_model()
                model["models"][missing_name] = None
                for mode in ("net_value", "mass_times_cost", "ranker_same_k"):
                    self._assert_same(model, context, mode=mode)
        model = _synthetic_model()
        model["models"] = {}
        for mode in ("net_value", "mass_times_cost", "ranker_same_k"):
            self._assert_same(model, context, mode=mode)

    def test_zero_and_negative_shape_predictions(self):
        context = _synthetic_context()
        for shape_cost in (0.0, -0.25):
            with self.subTest(shape_cost=shape_cost):
                model = _synthetic_model(recovery_value=0.0, shape_cost=shape_cost)
                result = self._assert_same(model, context, mode="net_value")
                self.assertEqual(result.backup_ids, () if shape_cost == 0.0 else (200,))
                self.assertEqual(result.predicted_shape_cost_ms, shape_cost)

    def test_make_selector_default_and_stats_shape(self):
        model = _synthetic_model()
        compiled = make_selector(model, "value")
        reference = make_selector(model, "value", implementation="reference")
        self.assertEqual(compiled.implementation, "compiled")
        self.assertEqual(reference.implementation, "reference")
        self.assertEqual(compiled(_synthetic_context()), reference(_synthetic_context()))
        self.assertEqual(set(compiled.stats()), set(reference.stats()))
        self.assertEqual(compiled.stats()["selector_calls"], 1)
        self.assertEqual(reference.stats()["selector_calls"], 1)


class ShapeCostFeatureContextTests(unittest.TestCase):
    @staticmethod
    def _row(scope, context, labels):
        return {
            "source_id": "scene-A",
            "scene_id": "scene-A",
            "context_id": "ctx-01",
            "seed": 7,
            "round_id": 3,
            "scope": scope,
            "round_context": context,
            "labels": labels,
        }

    def test_probe_uses_postwarm_context_and_keeps_value_context(self):
        candidates = [
            {"token_id": 100, "ranker_rank": 0, "ranker_score": 0.9, "position_rank": 5},
            {"token_id": 200, "ranker_rank": 1, "ranker_score": 0.2, "position_rank": 12},
        ]
        cold_context = {
            "prefix_len": 64,
            "remaining_budget": 128,
            "cache_pos": 63,
            "read_bucket": 16,
            "graph_exists": False,
            "graph_cold": True,
            "position_vocab_size": 8000,
            "eos_id": 12096,
            "main_position": 100,
            "main_score": 0.9,
            "main_position_rank": 5,
            "previous_position_rank": 3,
            "candidates": candidates,
            "graph_state": {
                "0": {"read_bucket": 16, "graph_exists": False, "graph_enabled": True},
                "1": {"read_bucket": 32, "graph_exists": False, "graph_enabled": True},
            },
        }
        warmed_context = {
            **cold_context,
            "graph_exists": True,
            "graph_cold": False,
            "graph_state": {
                "0": {"read_bucket": 16, "graph_exists": True, "graph_enabled": True},
                "1": {"read_bucket": 32, "graph_exists": True, "graph_enabled": True},
            },
        }
        records = [
            self._row("single_round", cold_context, {
                "target_mass_by_token": [{"token_id": 200, "probability": 0.4}],
            }),
            self._row("shape_probe", cold_context, {
                "extra_shape_cost_ms_by_k": {"1": 0.25},
                "cost_probe_feature_context": warmed_context,
            }),
        ]
        value_samples, shape_samples, stats = fit_module._aggregate_records(
            records, position_vocab_size=8000, eos_id=12096
        )
        self.assertEqual(len(value_samples), 1)
        self.assertEqual(len(shape_samples), 1)
        self.assertFalse(value_samples[0]["context"]["graph_exists"])
        self.assertTrue(value_samples[0]["context"]["graph_cold"])
        self.assertTrue(shape_samples[0]["context"]["graph_exists"])
        self.assertFalse(shape_samples[0]["context"]["graph_cold"])
        self.assertTrue(shape_samples[0]["context"]["graph_state"]["1"]["graph_exists"])
        self.assertEqual(shape_samples[0]["shape_costs"], {1: 0.25})
        self.assertEqual(value_samples[0]["shape_costs"], {})
        self.assertEqual(stats["shape_probe_postwarm_context_rows"], 1)
        self.assertEqual(stats["shape_probe_legacy_context_rows"], 0)

    def test_old_development_rows_replay_with_legacy_context_note(self):
        if not RECORDS_PATH.exists():
            self.skipTest("preserved legacy development records are unavailable")
        rows = fit_module._records_from_jsonl(str(RECORDS_PATH))
        value_samples, shape_samples, stats = fit_module._aggregate_records(
            rows, position_vocab_size=8000, eos_id=12096
        )
        self.assertEqual(len(rows), 1908)
        self.assertEqual(stats["shape_probe_cost_context_rows"], 50)
        self.assertEqual(stats["shape_probe_postwarm_context_rows"], 0)
        self.assertEqual(stats["shape_probe_legacy_context_rows"], 50)
        self.assertIn("may predate shape-probe warmup", stats["shape_probe_context_note"])
        self.assertEqual(len(shape_samples), 50)

        sample = shape_samples[0]
        source_row = next(
            row for row in rows
            if row["identity"]["source_id"] == sample["identity"]["source_id"]
            and row["identity"]["context_id"] == sample["identity"]["context_id"]
            and row["identity"]["seed"] == sample["identity"]["seed"]
            and row["identity"]["round_id"] == sample["identity"]["round_id"]
            and (
                row["probe_kind"] == "shape_probe"
                or any(row["labels"].get(name) is not None for name in (
                    "extra_shape_cost_ms_by_k", "shape_cost_ms_by_k", "shape_costs"
                ))
            )
        )
        expected_context, _, _ = _candidate_rows(
            {"round_context": source_row["round_context"]},
            {"config": {
                "position_vocab_size": 8000,
                "eos_id": 12096,
                "max_enum_candidates": 8,
                "max_backups": 4,
                "max_measured_backups": 4,
            }},
        )
        self.assertEqual(sample["context"], expected_context)
        self.assertEqual(sample["measurement_context_source"], "legacy_round_context")


class DevelopmentRecordParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists() or not RECORDS_PATH.exists():
            raise unittest.SkipTest("the preserved development model and records are unavailable")
        cls.model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        cls.compiled_models = _CompiledModels.build(cls.model)
        cls.contexts = []
        with RECORDS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                cls.contexts.append(row["round_context"])

    def test_all_development_contexts_match_all_modes_and_report_cpu_time(self):
        self.assertEqual(len(self.contexts), 1908)
        summary = {"context_count": len(self.contexts), "modes": {}}
        for method, mode in METHODS.items():
            reference = make_selector(self.model, method, implementation="reference")
            compiled = make_selector(self.model, method, implementation="compiled")
            compiled._compiled_models = self.compiled_models
            mismatches = []
            for index, context in enumerate(self.contexts):
                reference_ids = reference(context)
                compiled_ids = compiled(context)
                reference_decision = reference.last_decision
                compiled_decision = compiled.last_decision
                if reference_ids != compiled_ids:
                    mismatches.append((index, reference_ids, compiled_ids))
                    continue
                for field in (
                    "predicted_mass",
                    "predicted_recovery_value_ms",
                    "predicted_shape_cost_ms",
                    "predicted_net_value_ms",
                ):
                    if not math.isclose(
                        getattr(reference_decision, field),
                        getattr(compiled_decision, field),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        mismatches.append((index, field))
                if reference_decision.evaluated_subsets != compiled_decision.evaluated_subsets:
                    mismatches.append((index, "evaluated_subsets"))
                if reference_decision.candidate_truncation != compiled_decision.candidate_truncation:
                    mismatches.append((index, "candidate_truncation"))
                if reference_decision.ranker_prefix_matches != compiled_decision.ranker_prefix_matches:
                    mismatches.append((index, "ranker_prefix_matches"))
            self.assertEqual(mismatches, [], f"{method}: {mismatches[:3]}")

            timings = {"reference_us": [], "compiled_us": []}
            for repeat in range(3):
                order = (
                    (("reference_us", reference), ("compiled_us", compiled))
                    if repeat % 2 == 0 else
                    (("compiled_us", compiled), ("reference_us", reference))
                )
                for key, selector in order:
                    started = time.perf_counter_ns()
                    for context in self.contexts:
                        selector(context)
                    elapsed = (time.perf_counter_ns() - started) / len(self.contexts) / 1000.0
                    timings[key].append(elapsed)
            reference_us = statistics.median(timings["reference_us"])
            compiled_us = statistics.median(timings["compiled_us"])
            summary["modes"][method] = {
                "actions_equal": True,
                "decision_fields_equal_within_1e-12": True,
                "reference_median_us_per_context": reference_us,
                "compiled_median_us_per_context": compiled_us,
                "compiled_over_reference": compiled_us / reference_us if reference_us else None,
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
