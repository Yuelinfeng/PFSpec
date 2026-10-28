#!/usr/bin/env python3
"""Enumerate two-stage residual sampling with only the Python standard library."""

from fractions import Fraction
from bisect import bisect_right
from itertools import product
import json
import math
from pathlib import Path
import platform
import random
import sys


SEED = 20261020
MC_SEED = 20261021
MC_SAMPLE_COUNT = 100000
FLOAT_TOLERANCE = 1e-11
OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "docs/project/artifacts/pf2x_loop_20261019/round01/results/probability_cpu.json"
)


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def exact_distribution(counts, total):
    return tuple(Fraction(value, total) for value in counts)


def exact_sum(values):
    return sum(values, Fraction(0, 1))


def exact_positive_difference(left, right):
    return tuple(max(a - b, Fraction(0, 1)) for a, b in zip(left, right))


def exact_residual(target, proposal):
    difference = exact_positive_difference(target, proposal)
    mass = exact_sum(difference)
    if mass == 0:
        return None, mass
    return tuple(value / mass for value in difference), mass


def exact_exclude_and_renormalize(base, excluded):
    kept = tuple(Fraction(0, 1) if i == excluded else value for i, value in enumerate(base))
    denominator = exact_sum(kept)
    if denominator == 0:
        return None, denominator
    return tuple(value / denominator for value in kept), denominator


def exact_speculative_output(target, proposal):
    """Sum every proposal, acceptance, and residual outcome exactly."""
    output = [Fraction(0, 1) for _ in target]
    residual, residual_mass = exact_residual(target, proposal)
    proposal_candidates = 0
    rejected_candidate_paths = 0
    rejection_probability = Fraction(0, 1)

    for candidate, proposal_mass in enumerate(proposal):
        if proposal_mass == 0:
            continue
        proposal_candidates += 1
        accepted_joint_mass = min(target[candidate], proposal_mass)
        output[candidate] += accepted_joint_mass
        rejected_mass = proposal_mass - accepted_joint_mass
        if rejected_mass == 0:
            continue
        if residual is None:
            raise AssertionError("A rejected proposal has no positive residual mass")
        rejected_candidate_paths += 1
        rejection_probability += rejected_mass
        for token, probability in enumerate(residual):
            output[token] += rejected_mass * probability

    return tuple(output), {
        "proposal_candidates": proposal_candidates,
        "rejected_candidate_paths": rejected_candidate_paths,
        "rejection_probability": rejection_probability,
        "residual_mass": residual_mass,
    }


def exact_conditional_q2(base, first_id):
    return exact_exclude_and_renormalize(base, first_id)[0]


def exact_two_stage(target, q1, q2_base, strategy):
    """Expand first-stage branches and the conditional second proposal."""
    positive_first, z1 = exact_residual(target, q1)
    rejected_first = exact_positive_difference(q1, target)
    rejected_probability = exact_sum(rejected_first)
    if z1 != rejected_probability:
        raise AssertionError("First-stage residual and rejection masses differ")

    output = [min(p_i, q_i) for p_i, q_i in zip(target, q1)]
    stats = {
        "first_proposal_candidates": sum(value > 0 for value in q1),
        "first_rejection_branches": 0,
        "second_proposal_candidates": 0,
        "second_rejected_candidate_paths": 0,
        "second_zero_residual_branches": 0,
        "zero_denominator_fallback_branches": 0,
        "first_residual_mass": z1,
        "first_rejection_probability": rejected_probability,
    }

    if z1 == 0:
        return tuple(output), stats

    for first_id, branch_mass in enumerate(rejected_first):
        if branch_mass == 0:
            continue
        stats["first_rejection_branches"] += 1
        base = q1 if strategy == "exclude_from_q1" else q2_base
        q2, denominator = exact_exclude_and_renormalize(base, first_id)
        if q2 is None:
            branch_output = positive_first
            stats["zero_denominator_fallback_branches"] += 1
        else:
            branch_output, second_stats = exact_speculative_output(positive_first, q2)
            stats["second_proposal_candidates"] += second_stats["proposal_candidates"]
            stats["second_rejected_candidate_paths"] += second_stats["rejected_candidate_paths"]
            if second_stats["rejection_probability"] == 0:
                stats["second_zero_residual_branches"] += 1
        for token, probability in enumerate(branch_output):
            output[token] += branch_mass * probability

    return tuple(output), stats


def exact_error(left, right):
    return max((abs(a - b) for a, b in zip(left, right)), default=Fraction(0, 1))


def exact_metric(value):
    return {"fraction": f"{value.numerator}/{value.denominator}", "float": float(value)}


def exact_distribution_summary(distribution):
    return {
        "sum": exact_metric(exact_sum(distribution)),
        "minimum": exact_metric(min(distribution)),
        "finite": all(math.isfinite(float(value)) for value in distribution),
    }


def run_small_exhaustive():
    classes = []
    total_pair_count = 0
    total_triple_count = 0
    total_first_branches = 0
    total_second_candidates = 0
    total_second_rejected_paths = 0
    total_fallbacks = 0
    total_zero_first = 0
    total_zero_second = 0
    total_point_mass_q1 = 0
    total_disjoint_support_pairs = 0
    max_error = Fraction(0, 1)
    max_norm_error = Fraction(0, 1)

    class_specs = (
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
        (8, 2),
    )

    for vocab_size, denominator in class_specs:
        distributions = tuple(
            exact_distribution(counts, denominator)
            for counts in compositions(denominator, vocab_size)
        )
        pair_count = 0
        triple_count = 0
        class_first_branches = 0
        class_second_candidates = 0
        class_second_rejected_paths = 0
        class_fallbacks = 0
        class_zero_first = 0
        class_zero_second = 0
        class_point_mass_q1 = 0
        class_disjoint_support_pairs = 0
        class_max_error = Fraction(0, 1)
        class_max_norm_error = Fraction(0, 1)

        for target, q1 in product(distributions, repeat=2):
            pair_count += 1
            if sum(value > 0 for value in q1) == 1:
                class_point_mass_q1 += 1
            if not any(p_i > 0 and q_i > 0 for p_i, q_i in zip(target, q1)):
                class_disjoint_support_pairs += 1
            if target == q1:
                class_zero_first += 1

            result, stats = exact_two_stage(target, q1, None, "exclude_from_q1")
            if result != target:
                raise AssertionError("Exact q1-without-repeat path differs from target")
            class_first_branches += stats["first_rejection_branches"]
            class_fallbacks += stats["zero_denominator_fallback_branches"]
            class_zero_second += stats["second_zero_residual_branches"]
            class_second_candidates += stats["second_proposal_candidates"]
            class_second_rejected_paths += stats["second_rejected_candidate_paths"]
            class_max_error = max(class_max_error, exact_error(result, target))
            class_max_norm_error = max(
                class_max_norm_error, abs(exact_sum(result) - Fraction(1, 1))
            )

        for target, q1, q2_base in product(distributions, repeat=3):
            triple_count += 1
            result, stats = exact_two_stage(
                target, q1, q2_base, "exclude_from_independent_base"
            )
            if result != target:
                raise AssertionError("Exact wide-support path differs from target")
            class_first_branches += stats["first_rejection_branches"]
            class_fallbacks += stats["zero_denominator_fallback_branches"]
            class_zero_second += stats["second_zero_residual_branches"]
            class_second_candidates += stats["second_proposal_candidates"]
            class_second_rejected_paths += stats["second_rejected_candidate_paths"]
            class_max_error = max(class_max_error, exact_error(result, target))
            class_max_norm_error = max(
                class_max_norm_error, abs(exact_sum(result) - Fraction(1, 1))
            )

        distributions_count = len(distributions)
        classes.append({
            "vocabulary_category_count": vocab_size,
            "integer_denominator": denominator,
            "integer_distributions": distributions_count,
            "q1_target_pairs": pair_count,
            "independent_q2_base_triples": triple_count,
            "strategies": [
                "q2=q1 conditioned on excluding the first candidate",
                "q2=q2_base conditioned on excluding the first candidate; q2_base is independently enumerated",
            ],
            "first_rejection_branches": class_first_branches,
            "second_candidate_ids_enumerated": class_second_candidates,
            "second_rejected_candidate_paths": class_second_rejected_paths,
            "zero_denominator_fallback_branches": class_fallbacks,
            "zero_first_residual_pairs": class_zero_first,
            "zero_second_residual_branches": class_zero_second,
            "q1_point_mass_pairs": class_point_mass_q1,
            "disjoint_target_q1_support_pairs": class_disjoint_support_pairs,
            "max_final_probability_error": exact_metric(class_max_error),
            "max_normalization_error": exact_metric(class_max_norm_error),
        })

        total_pair_count += pair_count
        total_triple_count += triple_count
        total_first_branches += class_first_branches
        total_second_candidates += class_second_candidates
        total_second_rejected_paths += class_second_rejected_paths
        total_fallbacks += class_fallbacks
        total_zero_first += class_zero_first
        total_zero_second += class_zero_second
        total_point_mass_q1 += class_point_mass_q1
        total_disjoint_support_pairs += class_disjoint_support_pairs
        max_error = max(max_error, class_max_error)
        max_norm_error = max(max_norm_error, class_max_norm_error)

    return {
        "arithmetic": "fractions.Fraction exact rational arithmetic",
        "small_vocabulary_configuration_count": len(classes),
        "vocabulary_category_counts_covered": sorted(
            {vocab_size for vocab_size, _ in class_specs}
        ),
        "small_vocabulary_configurations": classes,
        "q1_target_pair_count": total_pair_count,
        "independent_q2_base_triple_count": total_triple_count,
        "total_first_rejection_branches": total_first_branches,
        "total_second_candidate_ids_enumerated": total_second_candidates,
        "total_second_rejected_candidate_paths": total_second_rejected_paths,
        "zero_denominator_fallback_branches": total_fallbacks,
        "zero_first_residual_pairs": total_zero_first,
        "zero_second_residual_branches": total_zero_second,
        "q1_point_mass_pairs": total_point_mass_q1,
        "disjoint_target_q1_support_pairs": total_disjoint_support_pairs,
        "max_final_probability_error": exact_metric(max_error),
        "max_normalization_error": exact_metric(max_norm_error),
    }


def boundary_cases():
    half = Fraction(1, 2)
    zero = Fraction(0, 1)
    one = Fraction(1, 1)

    cases = {}

    target = (half, half)
    q1 = (one, zero)
    q2_base = (zero, one)
    result, stats = exact_two_stage(target, q1, q2_base, "exclude_from_q1")
    cases["q1_point_mass_and_zero_q2_denominator"] = {
        "result_matches_target": result == target,
        "target_support_ids": [0, 1],
        "q1_support_ids": [0],
        "q2_base_support_ids": [1],
        "first_residual_mass": exact_metric(stats["first_residual_mass"]),
        "fallback_branches": stats["zero_denominator_fallback_branches"],
    }

    target = (zero, one, zero)
    q1 = (one, zero, zero)
    q2_base = (zero, zero, one)
    result, stats = exact_two_stage(target, q1, q2_base, "exclude_from_independent_base")
    cases["disjoint_support_and_rejection_twice"] = {
        "result_matches_target": result == target,
        "target_support_ids": [1],
        "q1_support_ids": [0],
        "q2_base_support_ids": [2],
        "first_residual_mass": exact_metric(stats["first_residual_mass"]),
        "second_rejected_candidate_paths": stats["second_rejected_candidate_paths"],
    }

    target = (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))
    q1 = target
    q2_base = (one, zero, zero)
    result, stats = exact_two_stage(target, q1, q2_base, "exclude_from_independent_base")
    cases["zero_first_residual"] = {
        "result_matches_target": result == target,
        "first_residual_mass": exact_metric(stats["first_residual_mass"]),
        "second_stage_started": stats["first_rejection_branches"] > 0,
    }

    target = (half, half)
    q1 = (one, zero)
    q2_base = (zero, one)
    result, stats = exact_two_stage(target, q1, q2_base, "exclude_from_independent_base")
    cases["zero_second_residual"] = {
        "result_matches_target": result == target,
        "second_zero_residual_branches": stats["second_zero_residual_branches"],
    }

    epsilon = Fraction(1, 2**60)
    delta = Fraction(1, 2**61)
    target = (
        Fraction(3, 10) + epsilon / 2,
        Fraction(3, 10) + epsilon / 2,
        Fraction(2, 5) - epsilon,
    )
    q1 = (Fraction(3, 10), Fraction(3, 10), Fraction(2, 5))
    q2_base = (half + delta, half - delta, zero)
    result, stats = exact_two_stage(target, q1, q2_base, "exclude_from_independent_base")
    residual, residual_mass = exact_residual(target, q1)
    actual_q2, _ = exact_exclude_and_renormalize(q2_base, 2)
    second_residual, second_residual_mass = exact_residual(residual, actual_q2)
    cases["tiny_positive_residuals"] = {
        "result_matches_target": result == target,
        "first_residual_mass": exact_metric(residual_mass),
        "second_residual_mass": exact_metric(second_residual_mass),
        "first_residual_support": sum(value > 0 for value in residual),
        "second_residual_support": sum(value > 0 for value in second_residual or ()),
        "no_positive_mass_cutoff_used": True,
    }

    expected = {
        "q1_point_mass_and_zero_q2_denominator": True,
        "disjoint_support_and_rejection_twice": True,
        "zero_first_residual": True,
        "zero_second_residual": True,
        "tiny_positive_residuals": True,
    }
    for name, required in expected.items():
        if required and not cases[name]["result_matches_target"]:
            raise AssertionError(f"Boundary case failed: {name}")
    if cases["q1_point_mass_and_zero_q2_denominator"]["fallback_branches"] < 1:
        raise AssertionError("Point-mass fallback was not reached")
    if cases["disjoint_support_and_rejection_twice"]["second_rejected_candidate_paths"] < 1:
        raise AssertionError("Two-stage rejection path was not reached")
    if cases["zero_second_residual"]["second_zero_residual_branches"] < 1:
        raise AssertionError("Zero second residual was not reached")
    if cases["zero_first_residual"]["second_stage_started"]:
        raise AssertionError("Second stage ran with zero first residual mass")
    if residual_mass != epsilon or second_residual_mass != delta:
        raise AssertionError("Tiny residual values differ from the requested exact inputs")
    return cases


def float_sum(values):
    return math.fsum(values)


def float_softmax(logits, temperature):
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = float_sum(weights)
    return [value / total for value in weights]


def float_policy(logits, temperature, top_p):
    scaled = [value / temperature for value in logits]
    full = float_softmax(logits, temperature)
    order = sorted(range(len(logits)), key=lambda index: (-scaled[index], index))
    keep_count = len(order)
    if top_p < 1.0:
        cumulative = 0.0
        for rank, index in enumerate(order):
            cumulative += full[index]
            if cumulative > top_p:
                keep_count = rank + 1
                break
    keep = order[:keep_count]
    maximum = max(scaled[index] for index in keep)
    weights = [math.exp(scaled[index] - maximum) for index in keep]
    total = float_sum(weights)
    result = [0.0] * len(logits)
    for index, weight in zip(keep, weights):
        result[index] = weight / total
    return result


def float_normalize(values):
    mass = float_sum(values)
    if mass == 0.0:
        return None, mass
    return [value / mass for value in values], mass


def float_exclude_and_renormalize(base, excluded):
    kept = [0.0 if index == excluded else value for index, value in enumerate(base)]
    denominator = float_sum(kept)
    if denominator == 0.0:
        return None, denominator
    return [value / denominator for value in kept], denominator


def float_distribution_metrics(distribution):
    total = float_sum(distribution)
    return {
        "finite": all(math.isfinite(value) for value in distribution),
        "nonnegative": all(value >= 0.0 for value in distribution),
        "sum": total,
        "normalization_error": abs(total - 1.0),
        "support_size": sum(value > 0.0 for value in distribution),
    }


def float_two_stage(target, q1, q2_base, strategy):
    first_difference = [max(p_i - q_i, 0.0) for p_i, q_i in zip(target, q1)]
    first_residual, first_residual_mass = float_normalize(first_difference)
    first_rejection_weights = [max(q_i - p_i, 0.0) for p_i, q_i in zip(target, q1)]
    first_rejection_probability = float_sum(first_rejection_weights)
    first_accept = [min(p_i, q_i) for p_i, q_i in zip(target, q1)]
    output = list(first_accept)
    stage2_branch_count = 0
    second_candidate_ids = 0
    second_rejected_candidate_paths = 0
    second_zero_residual_branches = 0
    fallback_branches = 0
    max_branch_error = 0.0
    max_q2_norm_error = 0.0
    max_second_residual_norm_error = 0.0
    max_stage2_rejection_mass_difference = 0.0

    if first_residual_mass == 0.0:
        return output, {
            "first_residual_mass": first_residual_mass,
            "first_rejection_probability": first_rejection_probability,
            "first_residual_rejection_mass_difference": abs(
                first_residual_mass - first_rejection_probability
            ),
            "stage2_branch_count": 0,
            "second_candidate_ids_enumerated": 0,
            "second_rejected_candidate_paths": 0,
            "second_zero_residual_branches": 0,
            "zero_denominator_fallback_branches": 0,
            "max_stage2_branch_error": 0.0,
            "max_q2_normalization_error": 0.0,
            "max_second_residual_normalization_error": 0.0,
            "max_second_rejection_mass_difference": 0.0,
        }

    for first_id, branch_mass in enumerate(first_rejection_weights):
        if branch_mass == 0.0:
            continue
        stage2_branch_count += 1
        base = q1 if strategy == "exclude_from_q1" else q2_base
        q2, denominator = float_exclude_and_renormalize(base, first_id)
        if q2 is None:
            branch_output = first_residual
            fallback_branches += 1
        else:
            q2_metrics = float_distribution_metrics(q2)
            max_q2_norm_error = max(max_q2_norm_error, q2_metrics["normalization_error"])
            if not q2_metrics["finite"] or not q2_metrics["nonnegative"]:
                raise AssertionError("Nonfinite or negative q2 probability")
            second_candidate_ids += q2_metrics["support_size"]
            second_accept = [min(r_i, q_i) for r_i, q_i in zip(first_residual, q2)]
            second_difference = [max(r_i - q_i, 0.0) for r_i, q_i in zip(first_residual, q2)]
            second_residual_mass = float_sum(second_difference)
            reverse_difference_mass = float_sum(
                max(q_i - r_i, 0.0) for r_i, q_i in zip(first_residual, q2)
            )
            max_stage2_rejection_mass_difference = max(
                max_stage2_rejection_mass_difference,
                abs(second_residual_mass - reverse_difference_mass),
            )
            if second_residual_mass == 0.0:
                branch_output = second_accept
                second_zero_residual_branches += 1
            else:
                second_residual, _ = float_normalize(second_difference)
                if second_residual is None:
                    raise AssertionError("Positive second residual did not normalize")
                second_residual_metrics = float_distribution_metrics(second_residual)
                max_second_residual_norm_error = max(
                    max_second_residual_norm_error,
                    second_residual_metrics["normalization_error"],
                )
                second_reject_probability = reverse_difference_mass
                branch_output = [
                    accepted + second_reject_probability * residual
                    for accepted, residual in zip(second_accept, second_residual)
                ]
                second_rejected_candidate_paths += sum(
                    q_i > r_i for r_i, q_i in zip(first_residual, q2)
                )

        max_branch_error = max(
            max_branch_error,
            max(abs(actual - expected) for actual, expected in zip(branch_output, first_residual)),
        )
        for token, probability in enumerate(branch_output):
            output[token] += branch_mass * probability

    return output, {
        "first_residual_mass": first_residual_mass,
        "first_rejection_probability": first_rejection_probability,
        "first_residual_rejection_mass_difference": abs(
            first_residual_mass - first_rejection_probability
        ),
        "stage2_branch_count": stage2_branch_count,
        "second_candidate_ids_enumerated": second_candidate_ids,
        "second_rejected_candidate_paths": second_rejected_candidate_paths,
        "second_zero_residual_branches": second_zero_residual_branches,
        "zero_denominator_fallback_branches": fallback_branches,
        "max_stage2_branch_error": max_branch_error,
        "max_q2_normalization_error": max_q2_norm_error,
        "max_second_residual_normalization_error": max_second_residual_norm_error,
        "max_second_rejection_mass_difference": max_stage2_rejection_mass_difference,
    }


def float_cdf(distribution):
    result = []
    cumulative = 0.0
    for probability in distribution:
        cumulative += probability
        result.append(cumulative)
    if result:
        result[-1] = 1.0
    return result


def sample_cdf(cumulative, rng):
    index = bisect_right(cumulative, rng.random())
    return min(index, len(cumulative) - 1)


def float_second_stage_summary(first_residual, q2_base, rejection_weights):
    first_rejection_mass = float_sum(rejection_weights)
    accepted_joint = [0.0] * len(first_residual)
    branch_acceptance = [0.0] * len(first_residual)
    q2_denominators = [0.0] * len(first_residual)
    max_branch_mix_error = 0.0
    fallback_count = 0

    for first_id, branch_mass in enumerate(rejection_weights):
        if branch_mass == 0.0:
            continue
        denominator = float_sum(
            probability if token != first_id else 0.0
            for token, probability in enumerate(q2_base)
        )
        q2_denominators[first_id] = denominator
        if denominator == 0.0:
            fallback_count += 1
            continue

        accepted_mass = [
            min(first_residual[token], q2_base[token] / denominator)
            if token != first_id else 0.0
            for token in range(len(first_residual))
        ]
        beta = float_sum(accepted_mass)
        branch_acceptance[first_id] = beta
        rejected_mass = max(0.0, 1.0 - beta)
        residual_mass = float_sum(
            max(
                first_residual[token]
                - (q2_base[token] / denominator if token != first_id else 0.0),
                0.0,
            )
            for token in range(len(first_residual))
        )
        if residual_mass == 0.0:
            branch_output = accepted_mass
        else:
            branch_residual = [
                max(
                    first_residual[token]
                    - (q2_base[token] / denominator if token != first_id else 0.0),
                    0.0,
                ) / residual_mass
                for token in range(len(first_residual))
            ]
            branch_output = [
                accepted_mass[token] + rejected_mass * branch_residual[token]
                for token in range(len(first_residual))
            ]
        max_branch_mix_error = max(
            max_branch_mix_error,
            max(
                abs(actual - expected)
                for actual, expected in zip(branch_output, first_residual)
            ),
        )
        for token, probability in enumerate(accepted_mass):
            accepted_joint[token] += branch_mass * probability

    joint_second_acceptance = float_sum(
        branch_mass * beta
        for branch_mass, beta in zip(rejection_weights, branch_acceptance)
    )
    beta_bar = (
        joint_second_acceptance / first_rejection_mass
        if first_rejection_mass > 0.0 else 0.0
    )
    accepted_only = (
        [probability / joint_second_acceptance for probability in accepted_joint]
        if joint_second_acceptance > 0.0 else [0.0] * len(first_residual)
    )
    full_vector_tv_from_first_residual = 0.5 * float_sum(
        abs(actual - expected)
        for actual, expected in zip(accepted_only, first_residual)
    ) if joint_second_acceptance > 0.0 else 0.0

    most_weighted_id = max(
        range(len(rejection_weights)),
        key=lambda token: rejection_weights[token] * branch_acceptance[token],
    )
    branch_beta = branch_acceptance[most_weighted_id]
    denominator = q2_denominators[most_weighted_id]
    if branch_beta > 0.0 and denominator > 0.0:
        branch_accepted_only = [
            (min(first_residual[token], q2_base[token] / denominator)
             if token != most_weighted_id else 0.0) / branch_beta
            for token in range(len(first_residual))
        ]
        branch_tv = 0.5 * float_sum(
            abs(actual - expected)
            for actual, expected in zip(branch_accepted_only, first_residual)
        )
    else:
        branch_tv = 0.0

    return {
        "first_rejection_mass": first_rejection_mass,
        "joint_first_reject_second_accept_mass": joint_second_acceptance,
        "conditional_second_acceptance_beta_bar": beta_bar,
        "branch_beta_values": branch_acceptance,
        "q2_denominators": q2_denominators,
        "accepted_only_distribution": accepted_only,
        "accepted_only_total_variation_from_r1": full_vector_tv_from_first_residual,
        "largest_joint_branch": {
            "first_candidate_id": most_weighted_id,
            "first_rejection_joint_mass": rejection_weights[most_weighted_id],
            "beta_a": branch_beta,
            "joint_second_acceptance_mass": (
                rejection_weights[most_weighted_id] * branch_beta
            ),
            "accepted_only_total_variation_from_r1": branch_tv,
        },
        "maximum_full_second_stage_branch_error": max_branch_mix_error,
        "zero_q2_denominator_fallback_count": fallback_count,
    }


def run_4096_monte_carlo(target, q1, q2_base, first_residual, analytic):
    rng = random.Random(MC_SEED)
    q1_cdf = float_cdf(q1)
    q2_base_cdf = float_cdf(q2_base)
    residual_cdf = float_cdf(first_residual)
    rejection_weights = [
        max(q_i - p_i, 0.0) for p_i, q_i in zip(target, q1)
    ]
    n = MC_SAMPLE_COUNT
    first_reject_count = 0
    second_accept_count = 0
    second_reject_count = 0
    final_counts = [0] * len(target)
    accepted_second_counts = [0] * len(target)
    q2_draw_count = 0
    residual_draw_count = 0
    top_ids = sorted(range(len(target)), key=lambda token: (-target[token], token))[:16]
    top_lookup = {token: index for index, token in enumerate(top_ids)}
    final_bin_counts = [0] * (len(top_ids) + 1)
    accepted_bin_counts = [0] * (len(top_ids) + 1)

    for _ in range(n):
        first_id = sample_cdf(q1_cdf, rng)
        first_q = q1[first_id]
        first_accept_probability = min(1.0, target[first_id] / first_q)
        if rng.random() < first_accept_probability:
            output_id = first_id
        else:
            first_reject_count += 1
            denominator = analytic["q2_denominators"][first_id]
            if denominator == 0.0:
                output_id = sample_cdf(residual_cdf, rng)
            else:
                while True:
                    second_id = sample_cdf(q2_base_cdf, rng)
                    q2_draw_count += 1
                    if second_id != first_id:
                        break
                second_q = q2_base[second_id] / denominator
                second_accept_probability = min(
                    1.0, first_residual[second_id] / second_q
                ) if second_q > 0.0 else 1.0
                if rng.random() < second_accept_probability:
                    output_id = second_id
                    second_accept_count += 1
                    accepted_second_counts[second_id] += 1
                    accepted_bin_counts[top_lookup.get(second_id, len(top_ids))] += 1
                else:
                    second_reject_count += 1
                    while True:
                        residual_draw_id = sample_cdf(residual_cdf, rng)
                        residual_draw_count += 1
                        residual_probability = first_residual[residual_draw_id]
                        second_probability = (
                            q2_base[residual_draw_id] / denominator
                            if residual_draw_id != first_id else 0.0
                        )
                        residual_accept_probability = max(
                            0.0,
                            1.0 - second_probability / residual_probability,
                        ) if residual_probability > 0.0 else 0.0
                        if rng.random() < residual_accept_probability:
                            output_id = residual_draw_id
                            break
        final_counts[output_id] += 1
        final_bin_counts[top_lookup.get(output_id, len(top_ids))] += 1

    expected_first_reject = analytic["first_rejection_mass"]
    expected_second_accept = analytic["joint_first_reject_second_accept_mass"]
    expected_beta_bar = analytic["conditional_second_acceptance_beta_bar"]
    empirical_first_reject = first_reject_count / n
    empirical_joint_second_accept = second_accept_count / n
    empirical_beta_bar = (
        second_accept_count / first_reject_count if first_reject_count else 0.0
    )

    final_bins = []
    accepted_bins = []
    exact_accepted_only = analytic["accepted_only_distribution"]
    exact_accepted_only_mass = float_sum(exact_accepted_only)
    for bin_index, token in enumerate(top_ids):
        token_bin = [candidate for candidate in range(len(target)) if
                     top_lookup.get(candidate) == bin_index]
        expected_p = float_sum(target[candidate] for candidate in token_bin)
        observed_p = final_bin_counts[bin_index] / n
        final_bins.append({
            "target_token_id": token,
            "target_probability": expected_p,
            "observed_probability": observed_p,
            "absolute_error": abs(observed_p - expected_p),
            "standard_error": math.sqrt(expected_p * (1.0 - expected_p) / n),
        })

        expected_accepted = float_sum(
            exact_accepted_only[candidate] for candidate in token_bin
        ) if exact_accepted_only_mass > 0.0 else 0.0
        observed_accepted = (
            accepted_bin_counts[bin_index] / second_accept_count
            if second_accept_count else 0.0
        )
        accepted_bins.append({
            "target_token_id": token,
            "accepted_only_exact_probability": expected_accepted,
            "accepted_only_observed_probability": observed_accepted,
            "absolute_error": abs(observed_accepted - expected_accepted),
            "standard_error": (
                math.sqrt(expected_accepted * (1.0 - expected_accepted) / second_accept_count)
                if second_accept_count else 0.0
            ),
            "r1_probability": float_sum(first_residual[candidate] for candidate in token_bin),
        })

    rest_candidates = [token for token in range(len(target)) if token not in top_lookup]
    expected_rest_p = float_sum(target[token] for token in rest_candidates)
    observed_rest_p = final_bin_counts[-1] / n
    final_bins.append({
        "target_token_id": "other",
        "target_probability": expected_rest_p,
        "observed_probability": observed_rest_p,
        "absolute_error": abs(observed_rest_p - expected_rest_p),
        "standard_error": math.sqrt(expected_rest_p * (1.0 - expected_rest_p) / n),
    })
    expected_rest_accepted = float_sum(
        exact_accepted_only[token] for token in rest_candidates
    ) if exact_accepted_only_mass > 0.0 else 0.0
    observed_rest_accepted = (
        accepted_bin_counts[-1] / second_accept_count if second_accept_count else 0.0
    )
    accepted_bins.append({
        "target_token_id": "other",
        "accepted_only_exact_probability": expected_rest_accepted,
        "accepted_only_observed_probability": observed_rest_accepted,
        "absolute_error": abs(observed_rest_accepted - expected_rest_accepted),
        "standard_error": (
            math.sqrt(expected_rest_accepted * (1.0 - expected_rest_accepted) / second_accept_count)
            if second_accept_count else 0.0
        ),
        "r1_probability": float_sum(first_residual[token] for token in rest_candidates),
    })

    max_final_bin_error = max(item["absolute_error"] for item in final_bins)
    max_accepted_bin_error = max(item["absolute_error"] for item in accepted_bins)
    accepted_only_group_tv_from_r1 = 0.5 * float_sum(
        abs(item["accepted_only_exact_probability"] - item["r1_probability"])
        for item in accepted_bins
    )
    return {
        "method": "seeded path sampling for q1, accept/reject, q2 conditioned on excluding the first candidate, and the second residual",
        "dimension": len(target),
        "precision": "IEEE-754 binary64 Python float",
        "seed": MC_SEED,
        "sample_count": n,
        "first_rejection": {
            "observed_count": first_reject_count,
            "observed_probability": empirical_first_reject,
            "exact_probability_from_float_inputs": expected_first_reject,
            "absolute_error": abs(empirical_first_reject - expected_first_reject),
            "binomial_standard_error": math.sqrt(
                expected_first_reject * (1.0 - expected_first_reject) / n
            ),
        },
        "second_candidate_acceptance": {
            "observed_count": second_accept_count,
            "observed_joint_probability": empirical_joint_second_accept,
            "exact_joint_probability_from_float_inputs": expected_second_accept,
            "joint_probability_absolute_error": abs(
                empirical_joint_second_accept - expected_second_accept
            ),
            "conditional_observed_probability_given_first_rejection": empirical_beta_bar,
            "conditional_exact_beta_bar": expected_beta_bar,
            "conditional_probability_absolute_error": abs(
                empirical_beta_bar - expected_beta_bar
            ),
            "conditional_binomial_standard_error": (
                math.sqrt(expected_beta_bar * (1.0 - expected_beta_bar) / first_reject_count)
                if first_reject_count else 0.0
            ),
            "second_rejection_count": second_reject_count,
        },
        "draw_counts": {
            "q2_base_draws_including_excluded_first_candidate": q2_draw_count,
            "first_residual_draws_for_second_rejection_sampling": residual_draw_count,
        },
        "final_output_group_errors": {
            "groups": final_bins,
            "maximum_absolute_group_error": max_final_bin_error,
            "grouped_total_variation": 0.5 * float_sum(item["absolute_error"] for item in final_bins),
        },
        "accepted_only_f1b_group_errors": {
            "groups": accepted_bins,
            "maximum_absolute_group_error": max_accepted_bin_error,
            "grouped_total_variation_from_analytic_accepted_only_distribution": (
                0.5 * float_sum(item["absolute_error"] for item in accepted_bins)
            ),
            "analytic_accepted_only_grouped_total_variation_from_r1": accepted_only_group_tv_from_r1,
        },
        "sampled_final_output_token_counts": sum(final_counts),
        "sampled_accepted_f1b_token_counts": sum(accepted_second_counts),
    }


def synthetic_4096_distributions():
    rng = random.Random(SEED)
    target_logits = [rng.gauss(0.0, 1.0) for _ in range(4096)]
    draft_logits = [rng.gauss(0.0, 1.0) for _ in range(4096)]
    temperature = 0.9
    target = float_policy(target_logits, temperature, 0.9)
    q1 = float_policy(draft_logits, temperature, 0.9)
    q2_base = float_policy(draft_logits, temperature, 1.0)
    return target, q1, q2_base


def run_4096_float64(target, q1, q2_base):
    temperature = 0.9

    target_metrics = float_distribution_metrics(target)
    q1_metrics = float_distribution_metrics(q1)
    q2_base_metrics = float_distribution_metrics(q2_base)
    if not all(metrics["finite"] and metrics["nonnegative"] for metrics in
               (target_metrics, q1_metrics, q2_base_metrics)):
        raise AssertionError("A 4096-dimensional input distribution is invalid")

    strategy_results = {}
    max_final_error = 0.0
    max_final_norm_error = 0.0
    max_first_mass_difference = 0.0
    max_branch_error = 0.0
    max_q2_norm_error = 0.0
    max_second_residual_norm_error = 0.0
    max_second_rejection_mass_difference = 0.0
    total_branches = 0
    total_second_candidates = 0
    total_second_rejections = 0
    total_fallbacks = 0

    for strategy in ("exclude_from_q1", "exclude_from_independent_base"):
        result, stats = float_two_stage(target, q1, q2_base, strategy)
        result_metrics = float_distribution_metrics(result)
        final_error = max(abs(actual - expected) for actual, expected in zip(result, target))
        total_variation = 0.5 * float_sum(
            abs(actual - expected) for actual, expected in zip(result, target)
        )
        if not result_metrics["finite"] or not result_metrics["nonnegative"]:
            raise AssertionError("A 4096-dimensional output distribution is invalid")
        if final_error > FLOAT_TOLERANCE or result_metrics["normalization_error"] > FLOAT_TOLERANCE:
            raise AssertionError("A 4096-dimensional output differs beyond tolerance")
        strategy_results[strategy] = {
            **stats,
            "final_distribution": result_metrics,
            "maximum_absolute_probability_error": final_error,
            "total_variation_from_target": total_variation,
            "q2_base_origin": (
                "same synthetic draft logits at temperature 0.9, top-p=1.0"
                if strategy == "exclude_from_independent_base"
                else "q1 from those draft logits at temperature 0.9, top-p=0.9"
            ),
            "q2_base_uses_target_probabilities": False,
        }
        max_final_error = max(max_final_error, final_error)
        max_final_norm_error = max(max_final_norm_error, result_metrics["normalization_error"])
        max_first_mass_difference = max(
            max_first_mass_difference,
            stats["first_residual_rejection_mass_difference"],
        )
        max_branch_error = max(max_branch_error, stats["max_stage2_branch_error"])
        max_q2_norm_error = max(max_q2_norm_error, stats["max_q2_normalization_error"])
        max_second_residual_norm_error = max(
            max_second_residual_norm_error,
            stats["max_second_residual_normalization_error"],
        )
        max_second_rejection_mass_difference = max(
            max_second_rejection_mass_difference,
            stats["max_second_rejection_mass_difference"],
        )
        total_branches += stats["stage2_branch_count"]
        total_second_candidates += stats["second_candidate_ids_enumerated"]
        total_second_rejections += stats["second_rejected_candidate_paths"]
        total_fallbacks += stats["zero_denominator_fallback_branches"]

    intersection = sum(p_i > 0.0 and q_i > 0.0 for p_i, q_i in zip(target, q1))
    first_residual, _ = float_normalize(
        [max(p_i - q_i, 0.0) for p_i, q_i in zip(target, q1)]
    )
    rejection_weights = [
        max(q_i - p_i, 0.0) for p_i, q_i in zip(target, q1)
    ]
    stage2_summary = float_second_stage_summary(first_residual, q2_base, rejection_weights)
    return {
        "dimension": 4096,
        "precision": "IEEE-754 binary64 Python float",
        "seed": SEED,
        "temperature": temperature,
        "target_top_p": 0.9,
        "q1_top_p": 0.9,
        "wide_q2_base_top_p": 1.0,
        "q2_base_source": "independent of target p; generated from draft logits only",
        "input_distributions": {
            "target_p": target_metrics,
            "q1": q1_metrics,
            "wide_q2_base": q2_base_metrics,
        },
        "support_overlap_target_q1": {
            "intersection_size": intersection,
            "target_only_size": target_metrics["support_size"] - intersection,
            "q1_only_size": q1_metrics["support_size"] - intersection,
        },
        "second_stage_exact_float_expansion": {
            "beta_definition": "beta(a)=sum_b min(r1(b), q2(b|a))",
            "first_rejection_mass": stage2_summary["first_rejection_mass"],
            "joint_first_reject_second_accept_mass": stage2_summary[
                "joint_first_reject_second_accept_mass"
            ],
            "conditional_second_acceptance_beta_bar": stage2_summary[
                "conditional_second_acceptance_beta_bar"
            ],
            "accepted_only_total_variation_from_r1": stage2_summary[
                "accepted_only_total_variation_from_r1"
            ],
            "largest_joint_branch": stage2_summary["largest_joint_branch"],
            "maximum_full_second_stage_branch_error": stage2_summary[
                "maximum_full_second_stage_branch_error"
            ],
            "q2_base_uses_target_probabilities": False,
        },
        "strategies": strategy_results,
        "two_stage_branch_count": total_branches,
        "second_candidate_ids_enumerated": total_second_candidates,
        "second_rejected_candidate_paths": total_second_rejections,
        "zero_denominator_fallback_branches": total_fallbacks,
        "maximum_final_probability_error": max_final_error,
        "maximum_final_normalization_error": max_final_norm_error,
        "maximum_first_residual_rejection_mass_difference": max_first_mass_difference,
        "maximum_second_branch_error": max_branch_error,
        "maximum_q2_normalization_error": max_q2_norm_error,
        "maximum_second_residual_normalization_error": max_second_residual_norm_error,
        "maximum_second_rejection_mass_difference": max_second_rejection_mass_difference,
        "zero_thresholds_used_for_branch_decisions": True,
        "_monte_carlo_inputs": {
            "first_residual": first_residual,
            "q2_denominators": stage2_summary["q2_denominators"],
            "branch_beta_values": stage2_summary["branch_beta_values"],
            "accepted_only_distribution": stage2_summary["accepted_only_distribution"],
            "first_rejection_mass": stage2_summary["first_rejection_mass"],
            "joint_first_reject_second_accept_mass": stage2_summary[
                "joint_first_reject_second_accept_mass"
            ],
            "conditional_second_acceptance_beta_bar": stage2_summary[
                "conditional_second_acceptance_beta_bar"
            ],
        },
    }


def main():
    small = run_small_exhaustive()
    boundaries = boundary_cases()
    inputs = synthetic_4096_distributions()
    large = run_4096_float64(*inputs)
    monte_carlo_inputs = large.pop("_monte_carlo_inputs")
    monte_carlo = run_4096_monte_carlo(
        inputs[0],
        inputs[1],
        inputs[2],
        monte_carlo_inputs["first_residual"],
        monte_carlo_inputs,
    )
    result = {
        "status": "passed",
        "method": {
            "small_vocabulary": "Exact rational enumeration over 9 configurations with vocabulary category counts 2, 3, 4, and 8; each configuration expands every first candidate, second candidate, accept event, and residual event.",
            "large_vocabulary": "Full-vector probability expansion in binary64 plus an auxiliary seeded path-sampling run.",
            "branch_rule": "Use exact zero to select zero-mass fallback; no positive-mass threshold changes a distribution.",
            "formula": "First stage uses p and q1; after rejection, q2 is tested against normalized positive(p-q1).",
            "joint_mass_expansion": "For target t and proposal q, accepted token mass is min(t_i,q_i); rejected candidate mass is q_i-min(t_i,q_i), followed by residual normalize(max(t-q,0)).",
            "wide_q2_rule": "Condition an independently supplied q2_base on excluding the first candidate; q2_base is never built from p.",
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "small_exhaustive": small,
        "boundary_cases": boundaries,
        "float64_4096": large,
        "monte_carlo_4096": monte_carlo,
        "limits": [
            "Small-vocabulary results are exact rational probability expansions, not Monte Carlo estimates.",
            "The 4096-dimensional inputs are synthetic distributions from fixed-seed logits, not model outputs.",
            "The 4096-dimensional Monte Carlo run is an auxiliary sampling-loop check; exact enumeration and full-vector expansion remain the primary evidence.",
            "Monte Carlo errors reflect finite sampling and are not a stronger proof of the probability identity.",
            "Python binary64 results do not establish CUDA or GPU FP32 behavior.",
            "This script does not exercise Target KV state, RoPE, decoder control flow, or runtime performance.",
            "The nucleus policy mirrors the copied teacher helper; tie ordering is not stress-tested because the seeded logits are continuous random values.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(OUTPUT),
        "small_pairs": small["q1_target_pair_count"],
        "small_triples": small["independent_q2_base_triple_count"],
        "small_configurations": small["small_vocabulary_configuration_count"],
        "vocabulary_category_counts": small["vocabulary_category_counts_covered"],
        "small_max_error": small["max_final_probability_error"],
        "float64_max_error": large["maximum_final_probability_error"],
        "float64_max_norm_error": large["maximum_final_normalization_error"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
