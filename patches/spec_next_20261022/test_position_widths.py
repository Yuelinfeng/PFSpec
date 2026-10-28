"""CPU checks for variable-width Position rescue trees."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


HERE = Path(__file__).resolve().parent
TREE_DIR = HERE.parent / "feature_tree_20261021"
sys.path.insert(0, str(TREE_DIR))
sys.path.insert(0, str(HERE))

from position_rescue_runtime import (  # noqa: E402
    PositionRescueRuntime,
    TargetTreeRuntime,
    rescue_key_mask,
    rescue_layout,
)


class _Target:
    sparse_position_vocab_size = 128
    sparse_shared_vocab = True
    sparse_feature_vocab_size = 64
    sparse_tokens_per_latent = 2
    sparse_num_position_tokens = 1


class _LinearRuntime:
    def __init__(self, target):
        self.target = target

    def _trunk(self, *args):
        raise AssertionError("CPU tests must not enter the Target forward")

    def _make_weight_views(self, *args):
        raise AssertionError("CPU tests must not prepare Target weights")


class _Cache:
    def __init__(self, position=8):
        self.position = int(position)

    def get_pos(self):
        return self.position


class PositionWidthTests(unittest.TestCase):
    def test_masks_keep_sibling_paths_separate(self):
        before = 8
        for count in range(1, 5):
            parents, depths, kinds, paths = rescue_layout(count)
            mask = rescue_key_mask(before, 16, backup_count=count)
            self.assertEqual(len(parents), 4 + count)
            self.assertEqual(len(mask), 4 + count)
            self.assertEqual(paths[0], (0,))
            self.assertEqual(kinds[:4], ("position", "feature", "position", "feature"))
            for index in range(count):
                node = 4 + index
                self.assertEqual(parents[node], 0)
                self.assertEqual(depths[node], 1)
                self.assertEqual(paths[node], (0, node))
                self.assertTrue(mask[node][before])
                self.assertTrue(mask[node][before + node])
                for sibling in range(count):
                    if sibling != index:
                        self.assertFalse(mask[node][before + 4 + sibling])
                for branch_node in (1, 2, 3):
                    self.assertFalse(mask[node][before + branch_node])
            for node in range(4):
                for backup in range(count):
                    self.assertFalse(mask[node][before + 4 + backup])

    def test_graph_pools_switch_for_shrunk_candidate_sets(self):
        target = _Target()
        runtime = PositionRescueRuntime(
            target, _LinearRuntime(target), graph=False, backup_count=4
        )
        cache = _Cache()
        observed = []
        original_verify = TargetTreeRuntime.verify

        def fake_verify(instance, cache_arg, pending_f, p1, features, p2s):
            count = len(instance._backup_positions)
            observed.append((count, instance.NODE_COUNT, instance._graphs))
            instance._graphs.setdefault(("cpu", count), object())
            return (), ()

        TargetTreeRuntime.verify = fake_verify
        try:
            runtime.verify(cache, 9, 3, [5], [4], backup_positions=[10, 11, 12])
            runtime.verify(cache, 9, 3, [5], [4], backup_positions=[10])
            runtime.verify(cache, 9, 3, [5], [4], backup_positions=[10, 11, 12])
        finally:
            TargetTreeRuntime.verify = original_verify

        self.assertEqual([(item[0], item[1]) for item in observed],
                         [(3, 7), (1, 5), (3, 7)])
        stats = runtime.stats()
        self.assertEqual(stats["graph_count"], 2)
        self.assertEqual(stats["graph_counts_by_backup_count"], {
            "4": 0, "3": 1, "1": 1,
        })

    def test_legacy_argument_and_invalid_paths_keep_errors(self):
        target = _Target()
        runtime = PositionRescueRuntime(
            target, _LinearRuntime(target), graph=False, backup_count=4
        )
        cache = _Cache()
        original_verify = TargetTreeRuntime.verify

        def fake_verify(instance, *args):
            return (), ()

        TargetTreeRuntime.verify = fake_verify
        try:
            runtime.verify(cache, 9, 3, [5], [4], backup_position=10)
        finally:
            TargetTreeRuntime.verify = original_verify
        self.assertEqual(runtime.NODE_COUNT, 5)

        runtime._backup_positions = (10, 10)
        with self.assertRaisesRegex(ValueError, "distinct"):
            runtime._node_tokens(9, 3, [5], [4])
        runtime._backup_positions = (3,)
        with self.assertRaisesRegex(ValueError, "distinct"):
            runtime._node_tokens(9, 3, [5], [4])
        runtime._backup_positions = None

        parents, _depths, _kinds, _paths = rescue_layout(4)
        runtime._pending = {
            "cache": cache,
            "before": 8,
            "parents": parents,
            "node_tokens": tuple(range(8)),
        }
        saved_torch = sys.modules.get("torch")
        if saved_torch is None:
            sys.modules["torch"] = types.ModuleType("torch")
        try:
            with self.assertRaisesRegex(ValueError, "outside the verified tree"):
                TargetTreeRuntime.commit(runtime, cache, [0, 8])
            with self.assertRaisesRegex(ValueError, "root-to-node path"):
                TargetTreeRuntime.commit(runtime, cache, [0, 1, 7])
        finally:
            if saved_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = saved_torch


if __name__ == "__main__":
    unittest.main()
