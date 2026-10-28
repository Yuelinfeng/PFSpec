"""Exercise pairing and source weighting without a GPU or saved weights."""
import math
import unittest

from compare import expand_methods, make_prompts, paired_summary


def observation(case, source, method, tokens, seconds):
    return dict(case_id=case, source=source, method=method,
                generated_tokens=tokens, seconds=seconds, counts={})


class ComparisonTests(unittest.TestCase):
    def test_sources_have_equal_weight_despite_unequal_repetitions(self):
        rows = []
        for index in range(8):
            rows.extend([observation(f"a{index}", "a", "reference", 100, 1),
                         observation(f"a{index}", "a", "method", 100, .5)])
        rows.extend([observation("b", "b", "reference", 100, 1),
                     observation("b", "b", "method", 100, 2)])
        result = paired_summary(rows, ["reference", "method"], ["reference"], samples=100)
        ratio = result["methods"]["method"]["comparisons"]["reference"]
        self.assertAlmostEqual(ratio["source_equal_throughput_ratio"], 1)
        self.assertEqual(ratio["sources"], 2)

    def test_early_eos_changes_workload_not_only_elapsed_time(self):
        rows = [observation("case", "source", "reference", 100, 1),
                observation("case", "source", "method", 50, .5)]
        result = paired_summary(rows, ["reference", "method"], ["reference"], samples=100)
        self.assertEqual(result["methods"]["method"]["comparisons"]["reference"]
                         ["source_equal_throughput_ratio"], 1)

    def test_incomplete_and_duplicate_observations_are_not_silently_paired(self):
        a = observation("case", "source", "reference", 100, 1)
        b = observation("case", "source", "method", 100, 1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            paired_summary([a], ["reference", "method"], ["reference"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_summary([a, b, b], ["reference", "method"], ["reference"])

    def test_case_ids_cannot_pair_different_sources(self):
        rows = [observation("case", "a", "reference", 100, 1),
                observation("case", "b", "method", 100, 1)]
        with self.assertRaisesRegex(ValueError, "Mismatched source"):
            paired_summary(rows, ["reference", "method"], ["reference"])

    def test_training_seeds_remain_distinct_methods(self):
        self.assertEqual(expand_methods(["pf_w1", "b0", "b1_e"], [1, 2]),
                         ["pf_w1", "b0:s1", "b0:s2", "b1_e:s1", "b1_e:s2"])

    def test_same_crop_is_used_for_all_prefixes_and_seeds(self):
        records = [dict(scene="a", path="first", tokens=list(range(20))),
                   dict(scene="a", path="second", tokens=list(range(20, 40)))]
        prompts = make_prompts(records, [2, 4], 1, [1, 2], 12096)
        self.assertEqual(len(prompts), 4)
        self.assertEqual({p["crop_path"] for p in prompts}, {"first"})
        self.assertTrue(all(p["prompt"] == [12096] + list(range(p["prefix"])) for p in prompts))


if __name__ == "__main__":
    unittest.main()
