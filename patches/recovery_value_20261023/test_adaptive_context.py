"""CPU-only checks for lightweight adaptive-selector inputs."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PATCH_DIR))

import adaptive_decoder as decoder
import selection


class _FakeCache:
    max_seq_len = 512


class _FakeRuntime:
    graph_enabled = True
    backup_count = 4

    def __init__(self, graph_map=None, graph_groups=None):
        self._graphs = graph_map or {}
        self._graphs_by_count = graph_groups
        self._graph_events = []


def _make_contexts():
    tree = _FakeRuntime(graph_map={})
    rescue = _FakeRuntime(graph_groups={1: {}, 2: {}, 3: {}, 4: {}})
    cache = _FakeCache()
    graph_state = decoder._graph_shapes(tree, rescue, cache, 63, 2, cache.max_seq_len)
    decoder._mark_candidate_availability(graph_state, 2, True)
    candidates = [
        {"token_id": 100, "ranker_rank": 0, "ranker_score": 0.92,
         "position_rank": 200, "rank_gap": 10},
        {"token_id": 101, "ranker_rank": 1, "ranker_score": 0.71,
         "position_rank": 205, "rank_gap": 15},
        {"token_id": 102, "ranker_rank": 2, "ranker_score": 0.63,
         "position_rank": 212, "rank_gap": 22},
    ]
    light = decoder._light_selector_context(
        prefix_len=64,
        remaining_budget=448,
        cache_pos=63,
        previous_position_rank=190,
        main_position=100,
        main_position_rank=200,
        main_score=0.92,
        candidates=candidates,
        position_vocab_size=8000,
        eos_id=12096,
        graph_state=graph_state,
    )
    diagnostic = dict(
        prefix_tokens=tuple(range(64)),
        initial_prompt_tokens=tuple(range(64)),
        pending_feature=8123,
        device="cpu",
        labels={"target_mass_by_token": {"101": 1.0}, "actual_position": 101},
        prefix_len=64,
        remaining_budget=448,
        logical_cache_pos=63,
        cache_pos=63,
        previous_position=99,
        previous_position_rank=190,
        graph_state={key: dict(value) for key, value in graph_state.items()},
        position_vocab_size=8000,
        eos_id=12096,
        main_position=100,
        main_position_rank=200,
        main_score=0.92,
        candidates=[dict(row) for row in candidates],
    )
    return light, diagnostic, candidates


def _decision_model():
    return {
        "config": {
            "max_backups": 2,
            "max_measured_backups": 2,
            "max_enum_candidates": 8,
            "supported_k": [0, 1, 2],
        },
        "models": {
            "recovery_value_ms": {
                "feature_names": ["bias"],
                "means": {"bias": 0.0},
                "scales": {"bias": 1.0},
                "weights": {"bias": 1.0},
            },
            "shape_cost_ms": {
                "feature_names": ["bias", "backup_count"],
                "means": {"bias": 0.0, "backup_count": 0.0},
                "scales": {"bias": 1.0, "backup_count": 1.0},
                "weights": {"bias": 0.0, "backup_count": 0.0},
            },
        },
    }


class AdaptiveContextTests(unittest.TestCase):
    def test_light_path_avoids_observer_snapshot_and_keeps_selection_inputs(self):
        with patch.object(
            decoder, "_copy_graph_snapshot",
            side_effect=AssertionError("observer snapshot ran on the lightweight path"),
        ):
            self.assertIsNone(decoder._observer_runtime_snapshot(None, object(), object()))
            light, diagnostic, _ = _make_contexts()

        for forbidden in (
            "prefix_tokens", "initial_prompt_tokens", "pending_feature",
            "device", "labels", "runtime_before", "runtime_after",
        ):
            self.assertNotIn(forbidden, light)
        model = _decision_model()
        normalized_light, rows_light, trunc_light = selection._candidate_rows(light, model)
        normalized_diagnostic, rows_diagnostic, trunc_diagnostic = selection._candidate_rows(
            diagnostic, model
        )
        self.assertEqual(
            {key: value for key, value in normalized_light.items() if key != "graph_state"},
            {key: value for key, value in normalized_diagnostic.items() if key != "graph_state"},
        )
        self.assertEqual(set(normalized_light["graph_state"]), set(normalized_diagnostic["graph_state"]))
        for key in normalized_light["graph_state"]:
            for field in ("read_bucket", "graph_exists", "candidate_available"):
                self.assertEqual(
                    normalized_light["graph_state"][key].get(field),
                    normalized_diagnostic["graph_state"][key].get(field),
                )
        self.assertEqual(trunc_light, trunc_diagnostic)
        self.assertEqual(
            [selection.candidate_feature_values(normalized_light, row) for row in rows_light],
            [selection.candidate_feature_values(normalized_diagnostic, row) for row in rows_diagnostic],
        )
        light_decision = selection.choose_backups(model, light, max_backups=2)
        diagnostic_decision = selection.choose_backups(model, diagnostic, max_backups=2)
        self.assertEqual(light_decision.as_dict(), diagnostic_decision.as_dict())

    def test_fixed_zero_fixed_one_and_dynamic_actions(self):
        scores = (0.92, 0.71, 0.63)
        best_zero, fixed_zero = decoder._ranked_candidate_indices(scores, 0)
        best_one, fixed_one = decoder._ranked_candidate_indices(scores, 1)
        self.assertEqual(best_zero, best_one)
        self.assertEqual(fixed_zero, [])
        self.assertEqual(fixed_one, [1])
        self.assertEqual(decoder._validate_backup_count(0), 0)
        self.assertEqual(decoder._validate_backup_count(1), 1)
        self.assertEqual(decoder._validate_backup_count(4), 4)
        for invalid in (-1, 5, 0.5, True):
            with self.assertRaises(ValueError):
                decoder._validate_backup_count(invalid)

        light, _, candidates = _make_contexts()
        selected = selection.choose_backups(
            _decision_model(), light, max_backups=2
        ).backup_ids
        dynamic = decoder._select_backup_ids(selected, [row["token_id"] for row in candidates[1:]], 2)
        self.assertEqual(dynamic, [101, 102])
        self.assertEqual(decoder._select_backup_ids([], [101, 102], 2), [])

    def test_cost_features_use_warmed_graph_state_and_keep_pre_warm_state(self):
        _, source_context, candidates = _make_contexts()
        before = {key: dict(value) for key, value in source_context["graph_state"].items()}
        after = {key: dict(value) for key, value in before.items()}
        after["0"].update(read_bucket=128, graph_enabled=True, graph_exists=True)
        after["1"].update(read_bucket=256, graph_enabled=True, graph_exists=True)
        context = decoder._cost_probe_feature_context(source_context, candidates, after)
        self.assertFalse(before["0"]["graph_exists"])
        self.assertTrue(context["graph_state"]["0"]["graph_exists"])
        self.assertTrue(context["graph_state"]["1"]["graph_exists"])
        self.assertEqual(context["graph_state"]["1"]["read_bucket"], 256)
        self.assertEqual(context["read_bucket"], 128)


if __name__ == "__main__":
    unittest.main()
