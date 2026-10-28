#!/usr/bin/env python3
"""CPU analysis for the finite PF cost probe."""

from __future__ import annotations

import argparse
import bisect
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Callable, Iterable, Sequence


FEATURES = 4096
TEMPERATURE = 0.9


def normalize_nonnegative(values: Sequence[float]) -> list[float]:
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be a non-empty finite nonnegative vector")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probability mass must be positive")
    return [value / total for value in values]


def softmax_logits(logits: Sequence[float], temperature: float = TEMPERATURE) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    values = [float(value) / temperature for value in logits]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("logits must be a non-empty finite vector")
    high = max(values)
    exps = [math.exp(value - high) for value in values]
    return normalize_nonnegative(exps)


def nucleus_distribution(logits: Sequence[float], temperature: float,
                         top_p: float) -> list[float]:
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    full = softmax_logits(logits, temperature)
    order = sorted(range(len(full)), key=lambda index: (-float(logits[index]), index))
    retained = []
    cumulative = 0.0
    for token in order:
        retained.append(token)
        cumulative += full[token]
        if cumulative > top_p:
            break
    kept = set(retained)
    return normalize_nonnegative([full[index] if index in kept else 0.0
                                  for index in range(len(full))])


def residual_distribution(p: Sequence[float], q: Sequence[float]) -> list[float]:
    p = normalize_nonnegative(p)
    q = normalize_nonnegative(q)
    if len(p) != len(q):
        raise ValueError("p and q dimensions differ")
    residual = [max(left - right, 0.0) for left, right in zip(p, q)]
    mass = math.fsum(residual)
    if mass <= 0.0:
        raise ValueError("a rejected proposal requires positive residual mass")
    return [value / mass for value in residual]


def q2_base(q1: Sequence[float], mode: str,
            raw_logits: Sequence[float] | None = None) -> list[float]:
    if mode == "narrow":
        return normalize_nonnegative(q1)
    if mode == "wide":
        if raw_logits is None:
            raise ValueError("wide q2 requires same-call Feature logits")
        if len(q1) != len(raw_logits):
            raise ValueError("q1 and raw Feature logits dimensions differ")
        return softmax_logits(raw_logits, TEMPERATURE)
    raise ValueError("mode must be narrow or wide")


def conditional_second_q(q1: Sequence[float], first: int, mode: str,
                         raw_logits: Sequence[float] | None = None) -> list[float]:
    base = q2_base(q1, mode, raw_logits)
    if not 0 <= int(first) < len(base):
        raise ValueError("first candidate is outside the Feature vocabulary")
    base[int(first)] = 0.0
    if math.fsum(base) <= 0.0:
        raise ValueError("q2 has no support after excluding the first candidate")
    return normalize_nonnegative(base)


def second_overlap(r1: Sequence[float], q2: Sequence[float]) -> float:
    r1 = normalize_nonnegative(r1)
    q2 = normalize_nonnegative(q2)
    if len(r1) != len(q2):
        raise ValueError("r1 and q2 dimensions differ")
    return math.fsum(min(left, right) for left, right in zip(r1, q2))


def _float_units(values: Sequence[float]) -> tuple[list[int], int]:
    ratios = [float(value).as_integer_ratio() for value in values]
    powers = [denominator.bit_length() - 1 for _, denominator in ratios]
    common_power = max(powers, default=0)
    units = [numerator << (common_power - power)
             for (numerator, _), power in zip(ratios, powers)]
    return units, common_power


def _units_float(value: int, power: int) -> float:
    return float(Fraction(value, 1 << power))


def overlap_by_excluded(r1: Sequence[float], base: Sequence[float]) -> tuple[list[float], list[bool]]:
    """Compute overlap with base conditioned away from each possible first id."""
    r1 = normalize_nonnegative(r1)
    base = normalize_nonnegative(base)
    if len(r1) != len(base):
        raise ValueError("r1 and q2 base dimensions differ")
    r_units, r_power = _float_units(r1)
    base_units, base_power = _float_units(base)
    positive_base = []
    for index, mass in enumerate(base):
        if mass > 0.0:
            positive_base.append((r1[index] / mass, index))
    positive_base.sort(key=lambda row: row[0])
    ordered_ratios = [row[0] for row in positive_base]
    sorted_indices = [row[1] for row in positive_base]
    position_by_index = {index: position for position, index in enumerate(sorted_indices)}
    prefix_r = [0]
    prefix_s = [0]
    for index in sorted_indices:
        prefix_r.append(prefix_r[-1] + r_units[index])
        prefix_s.append(prefix_s[-1] + base_units[index])
    total_s_units = prefix_s[-1]
    total_base_units = sum(base_units)
    betas: list[float] = []
    legal: list[bool] = []
    for excluded in range(len(base)):
        remaining_units = total_base_units - base_units[excluded]
        if remaining_units <= 0:
            betas.append(0.0)
            legal.append(False)
            continue
        remaining = _units_float(remaining_units, base_power)
        scale = 1.0 / remaining
        if not math.isfinite(scale):
            covered = math.fsum(
                min(r1[index], base[index] / remaining)
                for index in range(len(base)) if index != excluded)
            betas.append(min(1.0, max(0.0, covered)))
            legal.append(True)
            continue
        split = bisect.bisect_right(ordered_ratios, scale)
        low_units = prefix_r[split]
        high_units = total_s_units - prefix_s[split]
        excluded_position = position_by_index.get(excluded)
        if excluded_position is not None:
            if excluded_position < split:
                low_units -= r_units[excluded]
            else:
                high_units -= base_units[excluded]
        covered = math.fsum((_units_float(low_units, r_power),
                             scale * _units_float(high_units, base_power)))
        betas.append(min(1.0, max(0.0, covered)))
        legal.append(True)
    return betas, legal


def beta_bar(p: Sequence[float], q1: Sequence[float], mode: str,
             raw_logits: Sequence[float] | None = None) -> dict:
    p = normalize_nonnegative(p)
    q1 = normalize_nonnegative(q1)
    if len(p) != len(q1):
        raise ValueError("p and q1 dimensions differ")
    overlap = math.fsum(min(left, right) for left, right in zip(p, q1))
    rejection_weights = [max(draft - target, 0.0) for target, draft in zip(p, q1)]
    observed_r1 = math.fsum(rejection_weights)
    r1_mass = math.fsum(max(target - draft, 0.0) for target, draft in zip(p, q1))
    if abs(observed_r1 - r1_mass) > 2e-10:
        raise ArithmeticError("first-rejection mass differs from the residual mass")
    if observed_r1 <= 0.0:
        return dict(a1=overlap, r1=0.0, beta_bar=None,
                    second_reject_probability=None, q2_legal_mass=None,
                    q2_invalid_mass=None, beta_by_first=None)
    r1 = residual_distribution(p, q1)
    base = q2_base(q1, mode, raw_logits)
    per_first, legal = overlap_by_excluded(r1, base)
    weighted = math.fsum(weight * value for weight, value in zip(rejection_weights, per_first)) / observed_r1
    legal_mass = math.fsum(weight for weight, is_legal in zip(rejection_weights, legal)
                           if is_legal) / observed_r1
    return dict(a1=overlap, r1=observed_r1, beta_bar=weighted,
                second_reject_probability=1.0 - weighted,
                q2_legal_mass=legal_mass, q2_invalid_mass=1.0 - legal_mass,
                beta_by_first=per_first)


def enumerate_two_stage_distribution(
    p: Sequence[float], q1: Sequence[float],
    q2_policy: Callable[[int], Sequence[float]],
) -> list[float]:
    """Enumerate the exact final-token law for first and second rejection."""
    p = normalize_nonnegative(p)
    q1 = normalize_nonnegative(q1)
    if len(p) != len(q1):
        raise ValueError("p and q1 dimensions differ")
    output = [0.0] * len(p)
    first_residual_mass = math.fsum(max(target - draft, 0.0)
                                    for target, draft in zip(p, q1))
    first_residual = (residual_distribution(p, q1) if first_residual_mass > 0.0
                      else None)
    for first, first_mass in enumerate(q1):
        if first_mass <= 0.0:
            continue
        alpha1 = min(1.0, p[first] / first_mass)
        output[first] += first_mass * alpha1
        reject1 = first_mass * (1.0 - alpha1)
        if reject1 <= 0.0:
            continue
        if first_residual is None:
            raise ArithmeticError("positive first rejection has no residual")
        q2 = normalize_nonnegative(q2_policy(first))
        if len(q2) != len(p):
            raise ValueError("q2 dimension differs from p")
        for second, second_mass in enumerate(q2):
            if second_mass <= 0.0:
                continue
            alpha2 = min(1.0, first_residual[second] / second_mass)
            accept2 = second_mass * alpha2
            output[second] += reject1 * accept2
            reject2 = second_mass * (1.0 - alpha2)
            if reject2 > 0.0:
                remainder = [max(left - right, 0.0)
                             for left, right in zip(first_residual, q2)]
                remainder_mass = math.fsum(remainder)
                if remainder_mass <= 0.0:
                    raise ArithmeticError("second rejection has no residual")
                residual2 = [value / remainder_mass for value in remainder]
                for token, mass in enumerate(residual2):
                    output[token] += reject1 * reject2 * mass
    return normalize_nonnegative(output)


def _distribution_error(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def _legal_event_trigger(p1_accept_probability: float,
                         residual_masses: Sequence[float],
                         legal_masses: Sequence[float]) -> float | None:
    if len(residual_masses) != len(legal_masses):
        raise ValueError("residual and legal-mass rows differ in length")
    if not residual_masses:
        return None
    if not 0.0 <= p1_accept_probability <= 1.0:
        raise ValueError("P1 acceptance probability is outside [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in residual_masses):
        raise ValueError("residual mass is outside [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in legal_masses):
        raise ValueError("legal q2 mass is outside [0, 1]")
    return p1_accept_probability * statistics.fmean(
        residual * legal for residual, legal in zip(residual_masses, legal_masses))


def self_test() -> dict:
    rng = random.Random(20261020)
    cases = [
        ([0.2, 0.3, 0.5], [0.5, 0.2, 0.3]),
        ([0.5, 0.5, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.3, 0.7], [1.0, 0.0, 0.0]),
        ([0.1, 0.2, 0.7], [0.1, 0.2, 0.7]),
        ([1e-14, 0.25, 0.75 - 1e-14], [0.4, 0.3, 0.3]),
    ]
    maximum_error = 0.0
    exact_cases = 0
    dimensions = (2, 3, 4, 8)
    for size in dimensions:
        for _ in range(32):
            p = normalize_nonnegative([rng.random() ** 2 for _ in range(size)])
            q1 = normalize_nonnegative([rng.random() ** 2 for _ in range(size)])
            raw = [rng.uniform(-8.0, 8.0) for _ in range(size)]
            for mode in ("narrow", "wide"):
                base = q2_base(q1, mode, raw)

                def policy(first: int, base=base) -> list[float]:
                    values = list(base)
                    values[first] = 0.0
                    return normalize_nonnegative(values)

                final = enumerate_two_stage_distribution(p, q1, policy)
                error = _distribution_error(final, p)
                maximum_error = max(maximum_error, error)
                exact_cases += 1
                residual = residual_distribution(p, q1) if math.fsum(
                    max(target - draft, 0.0) for target, draft in zip(p, q1)) > 0.0 else None
                if residual is not None:
                    overlap_values, legal_values = overlap_by_excluded(residual, base)
                    for excluded, is_legal in enumerate(legal_values):
                        if not is_legal:
                            continue
                        conditioned = list(base)
                        conditioned[excluded] = 0.0
                        direct = second_overlap(residual, conditioned)
                        if abs(overlap_values[excluded] - direct) > 2e-12:
                            raise AssertionError("excluded-candidate overlap differs from direct sum")
    for p, q1 in cases:
        raw = [math.log(max(value, 1e-16)) for value in q1]
        for mode in ("narrow", "wide"):
            base = q2_base(q1, mode, raw)
            rejection_candidates = [first for first in range(len(q1))
                                    if q1[first] > p[first]]
            invalid = []
            for first in rejection_candidates:
                values = list(base)
                values[first] = 0.0
                if math.fsum(values) <= 0.0:
                    invalid.append(first)
            if invalid or not rejection_candidates:
                continue

            def policy(first: int, base=base) -> list[float]:
                values = list(base)
                values[first] = 0.0
                return normalize_nonnegative(values)

            final = enumerate_two_stage_distribution(p, q1, policy)
            maximum_error = max(maximum_error, _distribution_error(final, p))
            exact_cases += 1
    if maximum_error > 2e-12:
        raise AssertionError(f"enumerated output law differs from p by {maximum_error}")

    p = [0.05, 0.15, 0.8]
    q1 = [0.5, 0.3, 0.2]
    raw = [1.0, -1.0, 0.5]
    metrics = beta_bar(p, q1, "wide", raw)
    if metrics["beta_bar"] is None or not 0.0 <= metrics["beta_bar"] <= 1.0:
        raise AssertionError("beta summary is outside [0, 1]")
    zero_reject = beta_bar([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], "narrow")
    zero_legal_mass = (1.0 - zero_reject["q2_invalid_mass"]
                       if zero_reject["q2_invalid_mass"] is not None else 0.0)
    zero_trigger = _legal_event_trigger(0.7, [zero_reject["r1"]], [zero_legal_mass])
    if zero_reject["r1"] != 0.0 or zero_trigger != 0.0:
        raise AssertionError("zero-R1 rows must contribute zero legal event mass")
    p_weighted = [0.3, 0.2, 0.5]
    q_weighted = [0.5, 0.4, 0.1]
    rejected_weights = [max(q - p, 0.0) for p, q in zip(p_weighted, q_weighted)]
    residual = residual_distribution(p_weighted, q_weighted)
    candidate_betas = [
        second_overlap(residual, conditional_second_q(q_weighted, first, "narrow"))
        if rejected_weights[first] > 0.0 else 0.0
        for first in range(len(q_weighted))]
    expected_beta_bar = (math.fsum(weight * value for weight, value
                                   in zip(rejected_weights, candidate_betas)) /
                         math.fsum(rejected_weights))
    measured_beta_bar = beta_bar(p_weighted, q_weighted, "narrow")["beta_bar"]
    if abs(expected_beta_bar - measured_beta_bar) > 1e-12:
        raise AssertionError("beta_bar did not use the first-rejection candidate distribution")
    tiny = 2.0 ** -60
    tiny_p = [2.0 ** -10, 2.0 ** -10, 1.0 - 2.0 ** -9]
    tiny_q = [2.0 ** -10 + tiny, 2.0 ** -10 - tiny, 1.0 - 2.0 ** -9]
    tiny_metrics = beta_bar(tiny_p, tiny_q, "narrow")
    if tiny_metrics["r1"] != tiny or tiny_metrics["beta_bar"] is None:
        raise AssertionError("2^-60 first-rejection mass was discarded")
    tiny_base = [1.0, tiny]
    tiny_q2 = conditional_second_q(tiny_base, 0, "narrow")
    tiny_overlap, tiny_legal = overlap_by_excluded([1.0, tiny], tiny_base)
    if not tiny_legal[0] or tiny_q2 != [0.0, 1.0] or tiny_overlap[0] != tiny:
        raise AssertionError("positive second-proposal support was lost after exclusion")
    tiny_final = enumerate_two_stage_distribution(
        tiny_p, tiny_q, lambda first: conditional_second_q(tiny_q, first, "narrow"))
    if _distribution_error(tiny_final, tiny_p) > 2e-12:
        raise AssertionError("two-stage enumeration lost a 2^-60 positive residual")
    mixed_trigger = _legal_event_trigger(0.6, [0.5, 0.25], [0.0, 1.0])
    if mixed_trigger is None or abs(mixed_trigger - 0.075) > 1e-15:
        raise AssertionError("unsupported first-reject mass did not contribute zero")
    weighted_invalid = 1.0 - math.fsum(
        residual * legal for residual, legal in zip([0.5, 0.25], [0.0, 1.0])) / math.fsum([0.5, 0.25])
    if abs(weighted_invalid - (2.0 / 3.0)) > 1e-15:
        raise AssertionError("unsupported-event fraction did not use R1 weights")
    if abs(mixed_trigger * 0.8 - 0.06) > 1e-15:
        raise AssertionError("mixed legal-support event weight differs")
    q2 = conditional_second_q(q1, 0, "wide", raw)
    if q2[0] != 0.0 or abs(math.fsum(q2) - 1.0) > 1e-12:
        raise AssertionError("q2 did not exclude the first candidate")
    r1 = [0.2, 0.5, 0.3]
    q2 = conditional_second_q([0.4, 0.3, 0.3], 0, "narrow")
    p2_accept = [0.1, 0.6, 0.3]
    accepted_weights = [min(r, q) for r, q in zip(r1, q2)]
    beta_expected = math.fsum(accepted_weights)
    beta_from_q2_draws = math.fsum(
        q * (weight / q if q > 0.0 else 0.0)
        for q, weight in zip(q2, accepted_weights))
    joint_expected = math.fsum(weight * p2 for weight, p2 in zip(accepted_weights, p2_accept))
    joint_from_q2_draws = math.fsum(
        q * (weight / q if q > 0.0 else 0.0) * p2
        for q, weight, p2 in zip(q2, accepted_weights, p2_accept))
    if abs(beta_expected - beta_from_q2_draws) > 1e-12:
        raise AssertionError("second-stage acceptance weighting differs from beta")
    if abs(joint_expected - joint_from_q2_draws) > 1e-12:
        raise AssertionError("joint second-stage / P2 weighting differs")
    if not math.isfinite(maximum_error):
        raise AssertionError("enumeration error is not finite")
    return dict(passed=True, seed=20261020, dimensions=list(dimensions),
                random_cases=exact_cases, maximum_absolute_error=maximum_error,
                checks=["exact two-stage output law", "wide and narrow q2",
                        "disjoint support", "single-token q1 support",
                        "zero residual rejection is unreachable", "first-token exclusion",
                        "first-rejection candidate weighting",
                        "second-acceptance importance weight", "joint P2 event weight",
                        "mixed legal and unsupported branches", "zero-R1 event contribution",
                        "2^-60 positive masses"])


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(values) if values else None


def _sd(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _source_bootstrap(source_values: dict[str, float], seed: int,
                      draws: int = 2000) -> dict:
    sources = sorted(source_values)
    if not sources:
        return dict(source_count=0, draws=0, seed=seed, ci95=None)
    rng = random.Random(seed)
    means = [statistics.fmean(source_values[rng.choice(sources)]
                              for _ in sources) for _ in range(draws)]
    return dict(source_count=len(sources), draws=draws, seed=seed,
                ci95=[_percentile(means, 0.025), _percentile(means, 0.975)])


def _profile_costs(rows: list[dict]) -> dict:
    by_run = {}
    for row in rows:
        key = (row.get("scene"), row.get("prefix"), row.get("seed_base"))
        item = by_run.setdefault(key, dict(scene=key[0], prefix=key[1], seed_base=key[2],
                                           setup_target_host=0.0, setup_target_cuda=0.0,
                                           round_target_host=0.0, round_target_cuda=0.0,
                                           round_non_target_host=0.0, round_non_target_cuda=0.0,
                                           rounds=row.get("rounds", 0), generated_tokens=row.get("generated_tokens", 0),
                                           outer_seconds=row.get("outer_seconds")))
        stage = row.get("stage", "")
        operation = row.get("operation", "")
        host = float(row.get("host_seconds", 0.0))
        cuda = float(row.get("cuda_seconds", 0.0))
        if operation in {"target_kv", "capture_open", "prefill", "prefix_replay"}:
            item["setup_target_host"] += host
            item["setup_target_cuda"] += cuda
        elif stage in {"target_verify", "target_context"}:
            item["round_target_host"] += host
            item["round_target_cuda"] += cuda
        else:
            item["round_non_target_host"] += host
            item["round_non_target_cuda"] += cuda
    run_rows = list(by_run.values())
    valid = [row for row in run_rows if row["rounds"] > 0]
    return dict(
        runs=len(run_rows),
        mean_rounds=_mean(row["rounds"] for row in run_rows),
        mean_generated_tokens_per_round=_mean(
            row["generated_tokens"] / row["rounds"] for row in valid),
        fixed_setup_target_host_seconds=_mean(row["setup_target_host"] for row in run_rows),
        fixed_setup_target_cuda_seconds=_mean(row["setup_target_cuda"] for row in run_rows),
        target_host_seconds_per_round=_mean(row["round_target_host"] / row["rounds"] for row in valid),
        target_cuda_seconds_per_round=_mean(row["round_target_cuda"] / row["rounds"] for row in valid),
        non_target_host_seconds_per_round=_mean(row["round_non_target_host"] / row["rounds"] for row in valid),
        non_target_cuda_seconds_per_round=_mean(row["round_non_target_cuda"] / row["rounds"] for row in valid),
        outer_seconds_mean=_mean(row["outer_seconds"] for row in run_rows),
        timing_note="operation sums are instrumented components; host and CUDA values are separate and are not added",
    )


def analyze(args) -> dict:
    import numpy as np

    collection = Path(args.collection_dir)
    measurement = Path(args.measurement_dir)
    metadata_path = collection / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    measurement_metadata_path = measurement / "metadata.json"
    measurement_metadata = (json.loads(measurement_metadata_path.read_text(encoding="utf-8"))
                            if measurement_metadata_path.is_file() else {})
    if str(metadata.get("feature_step")) != str(args.feature_step):
        raise ValueError("feature_step differs from collection metadata")
    if str(Path(metadata.get("feature_checkpoint", "")).resolve()) != str(Path(args.feature_checkpoint).resolve()):
        raise ValueError("feature_checkpoint differs from collection metadata")
    contexts = _read_jsonl(collection / "contexts.jsonl")
    with np.load(collection / "vectors.npz", allow_pickle=False) as arrays:
        p_vectors = arrays["p_f"].astype(np.float64)
        q_vectors = arrays["q1"].astype(np.float64)
        raw_logits = arrays["raw_logits"].astype(np.float64)
    if not (p_vectors.shape == q_vectors.shape == raw_logits.shape):
        raise ValueError("p, q1, and same-call Feature logits arrays differ in shape")
    if p_vectors.shape[1] != FEATURES or len(contexts) != p_vectors.shape[0]:
        raise ValueError("vector count or Feature vocabulary differs from the context table")

    metric_rows = []
    vectors_by_id = {}
    for row in contexts:
        index = int(row["vector_index"])
        if not 0 <= index < len(contexts):
            raise ValueError("vector_index is outside the saved arrays")
        p = normalize_nonnegative(p_vectors[index].tolist())
        q = normalize_nonnegative(q_vectors[index].tolist())
        raw = raw_logits[index].tolist()
        if row.get("policy", {}).get("temperature") != TEMPERATURE or row.get("policy", {}).get("top_p") != 0.9:
            raise ValueError("context policy is not the recorded 0.9/0.9 rule")
        if row.get("p1_accepted_actual") is not True:
            raise ValueError("saved F context did not follow an actual accepted P1")
        modes = {mode: beta_bar(p, q, mode, raw) for mode in ("narrow", "wide")}
        p1_probability = float(row["p1_accept_probability"])
        if not 0.0 <= p1_probability <= 1.0:
            raise ValueError("P1 acceptance probability is outside [0, 1]")
        item = dict(sample_id=row["sample_id"], scene=row["scene"], prefix=int(row["prefix"]),
                    p1_accept_probability=p1_probability,
                    p1_accepted_actual=row.get("p1_accepted_actual"),
                    f1_accepted_actual=row.get("f1_accepted_actual"),
                    a1=modes["narrow"]["a1"], r1=modes["narrow"]["r1"], modes=modes)
        metric_rows.append(item)
        vectors_by_id[row["sample_id"]] = (p, q, raw, row)

    branch_rows = _read_jsonl(measurement / "branch_rows.jsonl")
    measurement_timing_rows = _read_jsonl(measurement / "shape_timings.jsonl")
    shape_rows = [row for row in measurement_timing_rows
                  if row.get("record_type") == "target_forward"]
    prewarm_rows = [row for row in measurement_timing_rows
                    if row.get("record_type") == "shape_prewarm"]
    paired_rows = [row for row in measurement_timing_rows
                   if row.get("record_type") == "shape_pair_forward"]
    branch_metrics: dict[str, list[dict]] = {"narrow": [], "wide": []}
    for branch in branch_rows:
        sample_id = branch["sample_id"]
        if sample_id not in vectors_by_id:
            raise ValueError(f"branch row refers to missing sample {sample_id}")
        p, q, raw, context = vectors_by_id[sample_id]
        first = int(branch["first_reject_local_id"])
        second = int(branch["second_candidate_local_id"])
        mode = branch["q2_mode"]
        q2 = conditional_second_q(q, first, mode, raw)
        r1 = residual_distribution(p, q)
        if q2[second] <= 0.0:
            raise ValueError("sampled second candidate has zero q2 probability")
        accept_probability = min(1.0, r1[second] / q2[second])
        p2_probability = float(branch["p2_accept_probability"])
        if not 0.0 <= p2_probability <= 1.0:
            raise ValueError("P2 acceptance probability is outside [0, 1]")
        branch_metrics[mode].append(dict(scene=context["scene"], prefix=int(context["prefix"]),
                                         sample_id=sample_id,
                                         second_accept_probability=accept_probability,
                                         p2_accept_probability=p2_probability,
                                         joint_sample_weight=accept_probability * p2_probability))

    count_rows = metadata.get("by_source_prefix", [])
    counts_by_cell = {(row["scene"], int(row["prefix"])): row for row in count_rows}
    if not counts_by_cell:
        raise ValueError("collection metadata lacks per-source and per-prefix proposal counts")
    context_cells = {}
    for row in metric_rows:
        context_cells.setdefault((row["scene"], row["prefix"]), []).append(row)
    branch_cells: dict[str, dict[tuple[str, int], list[dict]]] = {
        "narrow": {}, "wide": {}}
    for mode in ("narrow", "wide"):
        for row in branch_metrics[mode]:
            branch_cells[mode].setdefault((row["scene"], row["prefix"]), []).append(row)

    result = dict(status="complete", feature_step=args.feature_step,
                  feature_checkpoint=str(Path(args.feature_checkpoint).resolve()),
                  collection_dir=str(collection.resolve()), measurement_dir=str(measurement.resolve()),
                  sample_count=len(contexts),
                  source_count=len(metadata.get("selected_sources", [])),
                  sampled_source_count=len({row["scene"] for row in contexts}),
                  collection_sample_shortfalls=metadata.get("sample_shortfalls", []),
                  measurement_sample_counts=dict(
                      selected_contexts=measurement_metadata.get("selected_contexts"),
                      unsupported_q2_contexts=measurement_metadata.get("unsupported_q2_contexts"),
                      maximum_contexts=measurement_metadata.get("max_contexts"),
                      q2_mode=measurement_metadata.get("q2_mode"),
                      paired_contexts_with_p3=measurement_metadata.get("paired_contexts_with_p3"),
                      paired_contexts_skipped_no_p3=measurement_metadata.get(
                          "paired_contexts_skipped_no_p3"),
                      all_diagnostic_target_calls=measurement_metadata.get(
                          "all_diagnostic_target_calls")),
                  measurement_call_counts=measurement_metadata.get("target_call_counts"),
                  measurement_kv_truncation_counts=measurement_metadata.get(
                      "kv_truncation_counts"),
                  branch_sample_count=len(branch_rows),
                  policy=dict(target_temperature=TEMPERATURE, target_top_p=0.9,
                              q1_temperature=TEMPERATURE, q1_top_p=0.9,
                              wide_q2_temperature=TEMPERATURE, wide_q2_top_p=1.0),
                  by_mode={}, source_summary={}, target_shapes={}, w1_profile=None,
                  feasibility=dict(status="pending", reason="paired Target and selected W1 rates were not supplied"))
    result["uncertainty_limits"] = [
        "source intervals resample source-level plug-in means only; they omit context and branch sampling error",
        "source and prefix cells without required samples remain missing and are excluded from available-cell means",
        "branch event estimates use finite conditional samples and are point estimates",
    ]

    source_names = sorted({row["scene"] for row in count_rows})
    all_cells = sorted(counts_by_cell)
    for mode in ("narrow", "wide"):
        by_cell = {}
        for key in all_cells:
            scene, prefix = key
            selected = context_cells.get(key, [])
            counts = counts_by_cell[key]
            r1_weights = [row["modes"][mode]["r1"] for row in selected]
            beta_values = [row["modes"][mode]["beta_bar"] for row in selected]
            valid = [(weight, beta) for weight, beta in zip(r1_weights, beta_values)
                     if beta is not None]
            beta_denominator = math.fsum(weight for weight, _ in valid)
            beta_mean = (math.fsum(weight * beta for weight, beta in valid) / beta_denominator
                         if beta_denominator > 0.0 else None)
            r1_mean = _mean(r1_weights)
            legal_masses = [
                (1.0 - row["modes"][mode]["q2_invalid_mass"]
                 if row["modes"][mode]["q2_invalid_mass"] is not None else 0.0)
                for row in selected]
            legal_r1_mean = (_mean(weight * legal for weight, legal
                                   in zip(r1_weights, legal_masses))
                             if selected else None)
            p1_probability_mean = counts.get("p1_probability_mean")
            trigger = (_legal_event_trigger(float(p1_probability_mean), r1_weights, legal_masses)
                       if p1_probability_mean is not None and selected else None)
            all_trigger = (float(p1_probability_mean) * r1_mean
                           if p1_probability_mean is not None and r1_mean is not None else None)
            invalid_r1_weighted = (
                1.0 - legal_r1_mean / r1_mean
                if legal_r1_mean is not None and r1_mean is not None and r1_mean > 0.0
                else None)
            beta_sd = _sd(beta_values)
            proposal_count = int(counts.get("p1_proposals", 0))
            collected = len(selected)
            by_cell[key] = dict(
                scene=scene, prefix=prefix, p1_proposals=proposal_count,
                p1_probability_mean=p1_probability_mean,
                p1_acceptance_actual_rate=counts.get("p1_acceptance_actual_rate"),
                p1_accepted_actual=counts.get("p1_accepted"),
                p1_rejected_actual=counts.get("p1_rejected"),
                f1_accepted_after_p1_actual=counts.get("f1_accepted"),
                f1_rejected_after_p1_actual=counts.get("f1_rejected"),
                accepted_p1_contexts_collected=collected,
                accepted_p1_contexts_attempted=counts.get("source_rows"),
                sample_shortfall=next((row for row in metadata.get("sample_shortfalls", [])
                                       if row.get("scene") == scene and
                                       int(row.get("prefix", -1)) == prefix), None),
                a1_mean=_mean(row["modes"][mode]["a1"] for row in selected),
                a1_sd=_sd(row["modes"][mode]["a1"] for row in selected),
                r1_mean_given_accepted_p1=r1_mean,
                r1_sd_given_accepted_p1=_sd(r1_weights),
                beta_mean_condition_weighted=beta_mean,
                beta_sd_across_contexts=beta_sd,
                second_reject_probability=(1.0 - beta_mean) if beta_mean is not None else None,
                q2_invalid_rejection_mass_mean=_mean(
                    row["modes"][mode]["q2_invalid_mass"] for row in selected),
                q2_invalid_rejection_event_fraction_r1_weighted=invalid_r1_weighted,
                legal_q2_r1_weighted_mean=legal_r1_mean,
                p1_and_f1_reject_probability=all_trigger,
                p1_and_legal_f1_reject_probability=trigger,
            )

        cells_with_contexts = [by_cell[key] for key in all_cells
                               if by_cell[key]["accepted_p1_contexts_collected"]]
        all_r1_weights = [row["modes"][mode]["r1"] for row in metric_rows]
        all_beta_values = [(row["modes"][mode]["r1"],
                            row["modes"][mode]["beta_bar"])
                           for row in metric_rows
                           if row["modes"][mode]["beta_bar"] is not None]
        denominator = math.fsum(weight for weight, _ in all_beta_values)
        weighted_beta = (math.fsum(weight * beta for weight, beta in all_beta_values) / denominator
                         if denominator > 0.0 else None)
        all_cell_triggers = [row["p1_and_f1_reject_probability"] for row in cells_with_contexts
                             if row["p1_and_f1_reject_probability"] is not None]
        legal_cell_triggers = [row["p1_and_legal_f1_reject_probability"]
                               for row in cells_with_contexts
                               if row["p1_and_legal_f1_reject_probability"] is not None]
        proposal_weight_total = math.fsum(row["p1_proposals"] for row in cells_with_contexts)
        proposal_weighted_all_trigger = (
            math.fsum(row["p1_proposals"] * row["p1_and_f1_reject_probability"]
                      for row in cells_with_contexts
                      if row["p1_and_f1_reject_probability"] is not None) /
            proposal_weight_total if proposal_weight_total > 0.0 else None)
        proposal_weighted_legal_trigger = (
            math.fsum(row["p1_proposals"] * row["p1_and_legal_f1_reject_probability"]
                      for row in cells_with_contexts
                      if row["p1_and_legal_f1_reject_probability"] is not None) /
            proposal_weight_total if proposal_weight_total > 0.0 else None)

        by_source = {}
        for source in source_names:
            rows = [by_cell[key] for key in all_cells if key[0] == source]
            all_trigger_rows = [row["p1_and_f1_reject_probability"] for row in rows
                                if row["p1_and_f1_reject_probability"] is not None]
            legal_trigger_rows = [row["p1_and_legal_f1_reject_probability"] for row in rows
                                  if row["p1_and_legal_f1_reject_probability"] is not None]
            missing_prefixes = [row["prefix"] for row in rows
                                if row["p1_and_legal_f1_reject_probability"] is None]
            by_source[source] = dict(
                prefixes=[row["prefix"] for row in rows],
                trigger_prefix_coverage_complete=not missing_prefixes,
                trigger_missing_prefixes=missing_prefixes,
                p1_proposals=sum(row["p1_proposals"] for row in rows),
                accepted_p1_contexts_collected=sum(
                    row["accepted_p1_contexts_collected"] for row in rows),
                p1_and_f1_reject_probability_mean=(
                    statistics.fmean(all_trigger_rows) if all_trigger_rows else None),
                p1_and_legal_f1_reject_probability_mean=(
                    statistics.fmean(legal_trigger_rows) if legal_trigger_rows else None),
                p1_and_legal_f1_reject_probability_sd=_sd(legal_trigger_rows),
                beta_mean_condition_weighted=(
                    statistics.fmean([row["beta_mean_condition_weighted"] for row in rows
                                      if row["beta_mean_condition_weighted"] is not None])
                    if any(row["beta_mean_condition_weighted"] is not None for row in rows)
                    else None),
            )
        result["by_mode"][mode] = dict(
            a1_mean=_mean(row["modes"][mode]["a1"] for row in metric_rows),
            r1_mean=_mean(row["modes"][mode]["r1"] for row in metric_rows),
            beta_mean_condition_weighted=weighted_beta,
            second_reject_probability=(1.0 - weighted_beta) if weighted_beta is not None else None,
            p1_and_f1_reject_probability_mean=_mean(all_cell_triggers),
            p1_and_f1_reject_probability_proposal_weighted=proposal_weighted_all_trigger,
            p1_and_legal_f1_reject_probability_mean=_mean(legal_cell_triggers),
            p1_and_legal_f1_reject_probability_proposal_weighted=proposal_weighted_legal_trigger,
            contexts_with_zero_first_rejection=sum(row["modes"][mode]["r1"] == 0.0 for row in metric_rows),
            q2_invalid_rejection_mass_mean=_mean(row["modes"][mode]["q2_invalid_mass"]
                                                  for row in metric_rows),
            q2_invalid_rejection_event_fraction_r1_weighted=(
                (math.fsum(row["modes"][mode]["r1"] *
                           row["modes"][mode]["q2_invalid_mass"]
                           for row in metric_rows
                           if row["modes"][mode]["q2_invalid_mass"] is not None) /
                 math.fsum(row["modes"][mode]["r1"] for row in metric_rows))
                if math.fsum(row["modes"][mode]["r1"] for row in metric_rows) > 0.0
                else None),
            source_prefix_cells=len(all_cells),
            source_prefix_cells_with_contexts=len(cells_with_contexts),
            trigger_cell_coverage=len(cells_with_contexts),
            trigger_cell_coverage_complete=len(cells_with_contexts) == len(all_cells),
            trigger_missing_cells=[f"{scene}|{prefix}" for scene, prefix in all_cells
                                   if not by_cell[(scene, prefix)]["accepted_p1_contexts_collected"]],
            by_source_prefix={f"{scene}|{prefix}": row for (scene, prefix), row in by_cell.items()},
            by_source=by_source)

        branch_by_cell = {}
        for key in all_cells:
            rows = branch_cells[mode].get(key, [])
            if not rows:
                trigger = by_cell[key]["p1_and_legal_f1_reject_probability"]
                if trigger == 0.0:
                    branch_by_cell[key] = dict(
                        scene=key[0], prefix=key[1], branch_rows=0,
                        mean_second_accept_probability=0.0,
                        mean_second_accept_and_p2_probability=0.0,
                        p2_probability_given_second_accept=None,
                        second_accept_and_p2_probability_given_first_reject=0.0,
                        favorable_event_per_round=0.0,
                        status="zero_legal_trigger")
                continue
            second_mass = math.fsum(row["second_accept_probability"] for row in rows)
            joint_mass = math.fsum(row["joint_sample_weight"] for row in rows)
            r1_mean_cell = by_cell[key]["r1_mean_given_accepted_p1"]
            legal_r1_mean_cell = by_cell[key]["legal_q2_r1_weighted_mean"]
            legal_rejection_fraction = (
                legal_r1_mean_cell / r1_mean_cell
                if r1_mean_cell is not None and r1_mean_cell > 0.0 and
                legal_r1_mean_cell is not None else None)
            joint_given_first_reject = (
                legal_rejection_fraction * statistics.fmean(
                    row["joint_sample_weight"] for row in rows)
                if legal_rejection_fraction is not None else None)
            branch_by_cell[key] = dict(
                scene=key[0], prefix=key[1], branch_rows=len(rows),
                mean_second_accept_probability=statistics.fmean(
                    row["second_accept_probability"] for row in rows),
                mean_second_accept_and_p2_probability=statistics.fmean(
                    row["joint_sample_weight"] for row in rows),
                p2_probability_given_second_accept=(
                    joint_mass / second_mass if second_mass > 0.0 else None),
                legal_fraction_of_first_rejects=legal_rejection_fraction,
                second_accept_and_p2_probability_given_first_reject=(
                    joint_given_first_reject),
                favorable_event_per_round=(
                    by_cell[key]["p1_and_legal_f1_reject_probability"] *
                    statistics.fmean(row["joint_sample_weight"] for row in rows)
                    if by_cell[key]["p1_and_legal_f1_reject_probability"] is not None else None),
                status="estimated_from_sample",
            )
        event_by_source = {}
        for source in source_names:
            values = [row["favorable_event_per_round"] for key, row in branch_by_cell.items()
                      if key[0] == source and row["favorable_event_per_round"] is not None]
            if values:
                event_by_source[source] = statistics.fmean(values)
        covered_cells = [row for row in branch_by_cell.values()]
        if covered_cells:
            second_mass = math.fsum(row["mean_second_accept_probability"] for row in covered_cells)
            joint_mass = math.fsum(row["mean_second_accept_and_p2_probability"]
                                   for row in covered_cells)
            source_coverage_complete = len(event_by_source) == len(source_names)
            missing_cells = [f"{scene}|{prefix}" for scene, prefix in all_cells
                             if (scene, prefix) not in branch_by_cell]
            result["by_mode"][mode]["p2_after_second_accept"] = dict(
                branch_rows=sum(row["branch_rows"] for row in covered_cells),
                covered_source_prefix_cells=len(covered_cells),
                expected_source_prefix_cells=len(all_cells),
                p2_probability_given_second_accept=(
                    joint_mass / second_mass if second_mass > 0.0 else None),
                second_accept_and_p2_probability_given_first_reject=(
                    _mean(row.get("second_accept_and_p2_probability_given_first_reject")
                          for row in covered_cells)),
                favorable_event_per_round_mean=_mean(event_by_source.values()),
                favorable_event_per_round_ci95=_source_bootstrap(
                    event_by_source, args.bootstrap_seed),
                source_coverage=len(event_by_source),
                source_coverage_complete=source_coverage_complete,
                source_prefix_coverage_complete=not missing_cells,
                missing_source_prefix_cells=missing_cells,
                by_source_prefix={f"{scene}|{prefix}": row
                                  for (scene, prefix), row in branch_by_cell.items()},
                weighting_note=(
                    "first-reject candidate a follows delta(a)=(q1(a)-p(a))_+/R1; q2 candidate b "
                    "uses min(r1[b], q2[b|a]) / q2[b|a], and P2 joint weights also multiply p(P2)"),
                interpretation_note=(
                    "branch samples are conditional on legal q2 support; event rates multiply by "
                    "the R1-weighted legal mass. Unsupported first-reject events contribute zero."),
                interval_note=(
                    "source resampling interval covers source plug-in means only; it does not include "
                    "context or branch sampling error. Missing legal cells stay absent."),
            )
        else:
            result["by_mode"][mode]["p2_after_second_accept"] = dict(
                branch_rows=0, status="pending_no_branch_rows")

    for source in source_names:
        rows = [row for key, row in counts_by_cell.items() if key[0] == source]
        result["source_summary"][source] = dict(
            prefixes=sorted(int(row["prefix"]) for row in rows),
            four_token_proposals=sum(int(row.get("p1_proposals", 0)) for row in rows),
            p1_accepted_actual=sum(int(row.get("p1_accepted", 0)) for row in rows),
            p1_rejected_actual=sum(int(row.get("p1_rejected", 0)) for row in rows),
            f1_accepted_after_p1_actual=sum(int(row.get("f1_accepted", 0)) for row in rows),
            f1_rejected_after_p1_actual=sum(int(row.get("f1_rejected", 0)) for row in rows),
            saved_accepted_p1_contexts=sum(len(context_cells.get((source, int(row["prefix"])), []))
                                           for row in rows),
        )

    shape_groups = {}
    for row in shape_rows:
        shape = row["shape"]
        shape_groups.setdefault(shape, []).append(row)
    for shape, rows in sorted(shape_groups.items()):
        result["target_shapes"][shape] = dict(
            runs=len(rows), input_tokens=rows[0].get("input_tokens"),
            cuda_seconds_mean=_mean(row.get("cuda_seconds") for row in rows),
            cuda_seconds_sd=_sd(row.get("cuda_seconds") for row in rows),
            host_wall_seconds_mean=_mean(row.get("wall_seconds") for row in rows),
            host_wall_seconds_sd=_sd(row.get("wall_seconds") for row in rows),
            by_source={source: _mean(row.get("wall_seconds") for row in rows
                                     if row.get("scene") == source)
                       for source in sorted({row.get("scene") for row in rows})},
        )
    prewarm_by_shape = {}
    for row in prewarm_rows:
        prewarm_by_shape.setdefault(row["shape"], []).append(row)
    result["shape_prewarm"] = {
        shape: dict(
            calls=len(rows),
            cuda_seconds_mean=_mean(row.get("cuda_seconds") for row in rows),
            host_wall_seconds_mean=_mean(row.get("host_wall_seconds") for row in rows),
            distinct_contexts=len({row.get("sample_id") for row in rows}),
        )
        for shape, rows in sorted(prewarm_by_shape.items())
    }

    pair_groups = {}
    fixed_inputs: dict[tuple[str, str], set[tuple[int, ...]]] = {}
    for row in paired_rows:
        pair_groups.setdefault(row.get("pair_id"), []).append(row)
        fixed_inputs.setdefault((row.get("sample_id"), row["shape"]), set()).add(
            tuple(int(token) for token in row["input_ids"]))
    complete_pairs = []
    incomplete_pair_ids = []
    for pair_id, rows in sorted(pair_groups.items(), key=lambda item: str(item[0])):
        by_shape = {row["shape"]: row for row in rows}
        if (len(rows) != 2 or len(by_shape) != 2 or
                set(by_shape) != {"short_2", "next_4"}):
            incomplete_pair_ids.append(pair_id)
            continue
        short = by_shape["short_2"]
        next_row = by_shape["next_4"]
        if (short.get("sample_id") != next_row.get("sample_id") or
                short.get("kv_position_before") != next_row.get("kv_position_before") or
                short.get("kv_position_after_restore") != short.get("kv_position_before") or
                next_row.get("kv_position_after_restore") != next_row.get("kv_position_before") or
                len(fixed_inputs.get((short.get("sample_id"), "short_2"), set())) != 1 or
                len(fixed_inputs.get((short.get("sample_id"), "next_4"), set())) != 1):
            incomplete_pair_ids.append(pair_id)
            continue

        def ratio(numerator, denominator):
            numerator = float(numerator)
            denominator = float(denominator)
            return numerator / denominator if denominator > 0.0 else None

        short_host = float(short["host_wall_seconds"])
        next_host = float(next_row["host_wall_seconds"])
        short_cuda = float(short["cuda_seconds"])
        next_cuda = float(next_row["cuda_seconds"])
        ordered = sorted(rows, key=lambda row: int(row["order"]))
        if (ordered[0].get("pair_start_shape") != ordered[0]["shape"] or
                [int(row["pair_order"]) for row in ordered] != [0, 1]):
            incomplete_pair_ids.append(pair_id)
            continue
        complete_pairs.append(dict(
            pair_id=pair_id, sample_id=short["sample_id"],
            pair_start_shape=ordered[0].get("pair_start_shape"),
            observed_order=[row["shape"] for row in ordered],
            device=short.get("device"), kv_keep=short["kv_position_before"],
            fixed_input_ids=dict(short_2=short["input_ids"],
                                 next_4=next_row["input_ids"]),
            short_cuda_seconds=short_cuda, next_cuda_seconds=next_cuda,
            next_minus_short_cuda_seconds=next_cuda - short_cuda,
            next_over_short_cuda_ratio=ratio(next_cuda, short_cuda),
            short_host_wall_seconds=short_host, next_host_wall_seconds=next_host,
            next_minus_short_host_wall_seconds=next_host - short_host,
            next_over_short_host_wall_ratio=ratio(next_host, short_host),
            paired_restore_cuda_seconds=math.fsum(
                float(row["restore_cuda_seconds"]) for row in rows),
            paired_restore_host_wall_seconds=math.fsum(
                float(row["restore_host_wall_seconds"]) for row in rows),
            fixed_input_sequences=(
                len(fixed_inputs[(short["sample_id"], "short_2")]) == 1 and
                len(fixed_inputs[(short["sample_id"], "next_4")]) == 1),
        ))
    by_sample_pairs = {}
    for row in complete_pairs:
        by_sample_pairs.setdefault(row["sample_id"], []).append(row)
    result["shape_pair_comparison"] = dict(
        planned_pair_ids=len(pair_groups), complete_pairs=len(complete_pairs),
        incomplete_pair_ids=incomplete_pair_ids,
        paired_contexts=len(by_sample_pairs),
        short_then_next_pairs=sum(row["pair_start_shape"] == "short_2"
                                  for row in complete_pairs),
        next_then_short_pairs=sum(row["pair_start_shape"] == "next_4"
                                  for row in complete_pairs),
        host_next_minus_short_seconds_mean=_mean(
            row["next_minus_short_host_wall_seconds"] for row in complete_pairs),
        host_next_minus_short_seconds_sd=_sd(
            row["next_minus_short_host_wall_seconds"] for row in complete_pairs),
        host_next_over_short_ratio_mean=_mean(
            row["next_over_short_host_wall_ratio"] for row in complete_pairs),
        cuda_next_minus_short_seconds_mean=_mean(
            row["next_minus_short_cuda_seconds"] for row in complete_pairs),
        cuda_next_minus_short_seconds_sd=_sd(
            row["next_minus_short_cuda_seconds"] for row in complete_pairs),
        cuda_next_over_short_ratio_mean=_mean(
            row["next_over_short_cuda_ratio"] for row in complete_pairs),
        by_sample_id={sample_id: dict(
            pairs=len(rows),
            host_next_minus_short_seconds_mean=_mean(
                row["next_minus_short_host_wall_seconds"] for row in rows),
            host_next_over_short_ratio_mean=_mean(
                row["next_over_short_host_wall_ratio"] for row in rows),
            cuda_next_minus_short_seconds_mean=_mean(
                row["next_minus_short_cuda_seconds"] for row in rows),
            cuda_next_over_short_ratio_mean=_mean(
                row["next_over_short_cuda_ratio"] for row in rows),
        ) for sample_id, rows in by_sample_pairs.items()},
        pairs=complete_pairs,
        interpretation=(
            "same-context paired Target forward times only; KV restoration is reported separately and "
            "these measurements do not establish end-to-end acceleration"),
    )

    kv_operation_rows = [row for row in measurement_timing_rows
                         if row.get("record_type") == "kv_operation"]
    timing_restore_groups = {}
    for row in kv_operation_rows:
        timing_kind = row.get("timing_kind")
        if timing_kind in {"shape_pair_setup", "shape_prewarm", "shape_pair_forward"}:
            timing_restore_groups.setdefault(timing_kind, []).append(row)
    result["timing_restore_costs"] = {
        timing_kind: dict(
            calls=len(rows),
            cuda_seconds_mean=_mean(row.get("cuda_seconds") for row in rows),
            host_wall_seconds_mean=_mean(row.get("wall_seconds") for row in rows),
        )
        for timing_kind, rows in sorted(timing_restore_groups.items())
    }
    result["kv_operations"] = {}
    for row in kv_operation_rows:
        result["kv_operations"].setdefault(row["operation"], []).append(row)
    result["kv_operations"] = {
        operation: dict(calls=len(rows), host_wall_seconds_mean=_mean(row.get("wall_seconds") for row in rows),
                        cuda_seconds_mean=_mean(row.get("cuda_seconds") for row in rows))
        for operation, rows in result["kv_operations"].items()
    }
    result["w1_profile"] = _profile_costs(_read_jsonl(collection / "w1_profiles.jsonl"))

    if args.w1_tokens_per_second is not None:
        if args.target_tokens_per_second is None or not args.speed_evidence:
            raise ValueError("both token rates and --speed-evidence are required")
        if args.w1_tokens_per_second <= 0.0 or args.target_tokens_per_second <= 0.0:
            raise ValueError("token rates must be positive")
        for mode in ("narrow", "wide"):
            branch = result["by_mode"][mode]["p2_after_second_accept"]
            if (branch.get("source_prefix_coverage_complete") is not True or
                    result["by_mode"][mode].get("trigger_cell_coverage_complete") is not True):
                result["feasibility"][mode] = dict(
                    status="pending_incomplete_source_prefix_coverage",
                    reason="some required source/prefix cells lack accepted-context or branch samples",
                    not_a_stop_decision=True)
                continue
            shape4 = result["target_shapes"].get("main_4", {})
            event_rate = branch.get("favorable_event_per_round_mean")
            outputs_per_round = result["w1_profile"].get("mean_generated_tokens_per_round")
            forward_wall = shape4.get("host_wall_seconds_mean")
            if None in (event_rate, outputs_per_round, forward_wall) or outputs_per_round <= 0.0:
                result["feasibility"][mode] = dict(status="pending", reason="missing measured costs or event rate")
                continue
            required = max(0.0, 1.0 / args.w1_tokens_per_second -
                           0.5 / args.target_tokens_per_second)
            optimistic_estimate = event_rate * forward_wall / outputs_per_round
            result["feasibility"][mode] = dict(
                status="estimated_rate_definitions_unconfirmed",
                exploratory_comparison=("estimated_optimistic_estimate_below_gap"
                                        if optimistic_estimate < required
                                        else "estimated_optimistic_estimate_reaches_gap"),
                current_w1_over_target_speed= args.w1_tokens_per_second / args.target_tokens_per_second,
                required_seconds_saved_per_token=required,
                estimated_optimistic_seconds_saved_per_token=optimistic_estimate,
                source_rates=dict(w1_tokens_per_second=args.w1_tokens_per_second,
                                  target_tokens_per_second=args.target_tokens_per_second,
                                  evidence=args.speed_evidence),
                rate_window_aggregation_timing_confirmed=False,
                rate_comparison_note=(
                    "both rates require the same fixed window, aggregation, and timing definition; "
                    "the evidence path alone does not establish matching definitions"),
                assumption="one favorable event saves one serialized Target verification and branch work costs zero",
                interpretation=(
                    "optimistic finite-sample point comparison only; branch runtime is unmeasured and "
                    "this value does not decide whether work should stop"),
            )
        mode_statuses = [result["feasibility"].get(mode, {}).get("status")
                         for mode in ("narrow", "wide")]
        if any(status == "estimated_rate_definitions_unconfirmed" for status in mode_statuses):
            result["feasibility"]["status"] = "estimated_point_comparison_unconfirmed"
        elif any(status == "pending_incomplete_source_prefix_coverage"
                 for status in mode_statuses):
            result["feasibility"]["status"] = "pending_required_source_prefix_coverage"
        else:
            result["feasibility"]["status"] = "pending_required_measurements"
        result["feasibility"]["not_a_stop_decision"] = True

    short2 = result["target_shapes"].get("short_2", {})
    next4 = result["target_shapes"].get("next_4", {})
    residual_rows = [row for row in shape_rows if row.get("shape") == "short_2"
                     and row.get("p2_eos_probability") is not None]
    loss_rows = [row for row in residual_rows if row.get("next_round_generated_tokens") is not None]
    if loss_rows:
        result["w1_short"] = dict(
            short2_host_wall_seconds_mean=short2.get("host_wall_seconds_mean"),
            next4_host_wall_seconds_mean=next4.get("host_wall_seconds_mean"),
            expected_short_output_tokens_mean=_mean(2.0 - float(row["p2_eos_probability"])
                                                      for row in loss_rows),
            observed_w1_next_round_output_tokens_mean=_mean(
                row["next_round_generated_tokens"] for row in loss_rows),
            expected_progress_loss_tokens_mean=_mean(
                float(row["next_round_generated_tokens"]) -
                (2.0 - float(row["p2_eos_probability"])) for row in loss_rows),
            paired_rows=len(loss_rows),
            note="descriptive matched W1 trajectory comparison; shape times do not establish end-to-end speed",
        )
    else:
        result["w1_short"] = dict(status="pending", reason="no paired next-round rows")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="run the CPU probability and weighting checks")
    parser.add_argument("--collection-dir", type=Path,
                        help="directory created by collect_pq.py")
    parser.add_argument("--measurement-dir", type=Path,
                        help="directory created by measure_target_shapes.py")
    parser.add_argument("--feature-checkpoint", type=Path)
    parser.add_argument("--feature-step")
    parser.add_argument("--w1-tokens-per-second", type=float)
    parser.add_argument("--target-tokens-per-second", type=float)
    parser.add_argument("--speed-evidence",
                        help="source paths for the paired W1 and Target rates")
    parser.add_argument("--bootstrap-seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
        return 0
    required = ("collection_dir", "measurement_dir", "feature_checkpoint", "feature_step", "output")
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser.error("data analysis requires " + ", ".join("--" + name for name in missing))
    if args.output.exists():
        parser.error("output path already exists")
    if args.target_tokens_per_second is not None and args.w1_tokens_per_second is None:
        parser.error("--target-tokens-per-second requires --w1-tokens-per-second")
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=result["status"], samples=result["sample_count"],
                          branches=result["branch_sample_count"], output=str(args.output)),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
