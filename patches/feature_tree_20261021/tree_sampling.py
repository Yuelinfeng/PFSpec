"""Exact small-vocabulary laws for repeated Feature proposals.

The executable self-test uses only the Python standard library. Runtime tensor
helpers live here too, but import PyTorch only when a tensor input is supplied.
"""
from __future__ import annotations

import json
import math
from typing import Iterable, Sequence


class ZeroResidualMass(ValueError):
    """Raised when a rejection would leave no target residual distribution."""


def normalize(values: Iterable[float]) -> tuple[float, ...]:
    items = tuple(float(value) for value in values)
    if not items:
        raise ValueError("distribution cannot be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in items):
        raise ValueError("distribution must contain finite nonnegative values")
    mass = math.fsum(items)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("distribution must have positive finite mass")
    return tuple(value / mass for value in items)


def proposal_after_exclusions(
    q: Sequence[float], excluded: Iterable[int]
) -> tuple[float, ...]:
    """Return q conditioned on unselected ids, or uniform over them if empty."""
    base = normalize(q)
    excluded_set = {int(value) for value in excluded}
    if any(value < 0 or value >= len(base) for value in excluded_set):
        raise IndexError("excluded id is outside the vocabulary")
    available = [0.0 if index in excluded_set else value
                 for index, value in enumerate(base)]
    mass = math.fsum(available)
    if mass > 0.0:
        return tuple(value / mass for value in available)
    count = sum(index not in excluded_set for index in range(len(base)))
    if count == 0:
        raise ValueError("no unselected token remains")
    return tuple(0.0 if index in excluded_set else 1.0 / count
                 for index in range(len(base)))


def positive_residual(target: Sequence[float], proposal: Sequence[float]) -> tuple[float, ...]:
    """Return normalized (target - proposal)+; zero mass is an explicit error."""
    p = normalize(target)
    q = normalize(proposal)
    if len(p) != len(q):
        raise ValueError("target and proposal vocabularies differ")
    positive = tuple(max(left - right, 0.0) for left, right in zip(p, q))
    mass = math.fsum(positive)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ZeroResidualMass("rejection has no positive target residual")
    return tuple(value / mass for value in positive)


def acceptance_probability(
    residual: Sequence[float], proposal: Sequence[float], token: int
) -> float:
    r = tuple(float(value) for value in residual)
    q = tuple(float(value) for value in proposal)
    token = int(token)
    if len(r) != len(q) or not 0 <= token < len(r):
        raise ValueError("invalid residual, proposal, or token id")
    if any(not math.isfinite(value) or value < 0.0 for value in r + q):
        raise ValueError("probabilities must be finite and nonnegative")
    if q[token] <= 0.0:
        raise ValueError("sampled token has zero proposal probability")
    return min(1.0, r[token] / q[token])


def draw_distribution(probabilities: Sequence[float], rng) -> int:
    probs = normalize(probabilities)
    value = float(rng.random())
    if not 0.0 <= value < 1.0:
        raise ValueError("rng.random() must return a value in [0, 1)")
    cumulative = 0.0
    last_positive = None
    for index, probability in enumerate(probs):
        if probability > 0.0:
            last_positive = index
        cumulative += probability
        if value < cumulative:
            return index
    if last_positive is None:
        raise ValueError("distribution has no positive entry")
    return last_positive


def sample_repeated_proposals(
    target: Sequence[float],
    proposal: Sequence[float],
    width: int,
    rng,
) -> tuple[int, dict]:
    """Sample exactly from target using q candidates without replacement.

    Each candidate uses q conditioned on earlier candidate ids. Rejected
    candidates update r <- normalize((r - q_j)+). After the last rejection,
    the result is sampled from the remaining r.
    """
    p = normalize(target)
    q = normalize(proposal)
    if len(p) != len(q):
        raise ValueError("target and proposal vocabularies differ")
    width = int(width)
    if width < 1 or width > len(q):
        raise ValueError("width must fit the number of unique candidate ids")
    residual = p
    excluded: set[int] = set()
    tried = []
    for branch in range(width):
        qj = proposal_after_exclusions(q, excluded)
        token = draw_distribution(qj, rng)
        alpha = acceptance_probability(residual, qj, token)
        tried.append({"token": token, "proposal_probability": qj[token],
                      "acceptance_probability": alpha})
        if float(rng.random()) < alpha:
            return token, {"accepted_branch": branch, "tried": tried,
                           "all_rejected": False}
        residual = positive_residual(residual, qj)
        excluded.add(token)
    token = draw_distribution(residual, rng)
    return token, {"accepted_branch": None, "tried": tried,
                   "all_rejected": True, "final_residual": list(residual)}


def enumerate_output_law(
    target: Sequence[float], proposal: Sequence[float], width: int
) -> tuple[float, ...]:
    """Enumerate all proposal/rejection histories for a small vocabulary."""
    p = normalize(target)
    q = normalize(proposal)
    if len(p) != len(q):
        raise ValueError("target and proposal vocabularies differ")
    width = int(width)
    if width < 1 or width > len(q):
        raise ValueError("width must fit the number of unique candidate ids")
    output = [0.0] * len(p)

    def visit(residual, excluded, branch, path_mass):
        if branch == width or path_mass <= 0.0:
            for token, probability in enumerate(residual):
                output[token] += path_mass * probability
            return
        qj = proposal_after_exclusions(q, excluded)
        for candidate, candidate_mass in enumerate(qj):
            if candidate_mass <= 0.0:
                continue
            alpha = acceptance_probability(residual, qj, candidate)
            accepted_mass = path_mass * candidate_mass * alpha
            output[candidate] += accepted_mass
            rejected_mass = path_mass * candidate_mass * (1.0 - alpha)
            if rejected_mass > 0.0:
                next_residual = positive_residual(residual, qj)
                visit(next_residual, excluded | {candidate}, branch + 1,
                      rejected_mass)

    visit(p, set(), 0, 1.0)
    return tuple(output)


def point_residual(target: Sequence[float], proposed: int) -> tuple[float, ...]:
    """Residual for a point proposal; a certain target point cannot be rejected."""
    p = normalize(target)
    proposed = int(proposed)
    if not 0 <= proposed < len(p):
        raise IndexError("proposed id is outside the vocabulary")
    residual = list(p)
    residual[proposed] = 0.0
    mass = math.fsum(residual)
    if mass <= 0.0:
        raise ZeroResidualMass("rejected point proposal has no residual support")
    return tuple(value / mass for value in residual)


def enumerate_point_law(target: Sequence[float], proposed: int) -> tuple[float, ...]:
    """Enumerate output probabilities for exact point-proposal verification."""
    p = normalize(target)
    proposed = int(proposed)
    if not 0 <= proposed < len(p):
        raise IndexError("proposed id is outside the vocabulary")
    output = [0.0] * len(p)
    output[proposed] += p[proposed]
    rejection_mass = 1.0 - p[proposed]
    if rejection_mass > 0.0:
        residual = point_residual(p, proposed)
        for token, probability in enumerate(residual):
            output[token] += rejection_mass * probability
    return tuple(output)


def classify_token(token: int, *, p_vocab: int, eos: int) -> str:
    token = int(token)
    p_vocab = int(p_vocab)
    eos = int(eos)
    if token == eos:
        return "eos"
    if 0 <= token < p_vocab:
        return "position"
    if p_vocab <= token < eos:
        return "feature"
    raise ValueError("token id is outside the typed vocabulary")


def _is_torch_tensor(value) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def torch_proposals_without_replacement(q, width: int, rng):
    """Return (candidate ids, conditional q_j tensors) for runtime decoding."""
    if not _is_torch_tensor(q):
        raise TypeError("q must be a PyTorch Tensor")
    import torch

    if q.ndim != 1 or not q.is_floating_point():
        raise ValueError("q must be a one-dimensional floating tensor")
    width = int(width)
    if width < 1 or width > q.numel():
        raise ValueError("width must fit the number of unique candidate ids")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("q must contain finite nonnegative probabilities")
    if not bool(q.sum() > 0):
        raise ValueError("q must have positive mass")

    candidates = []
    distributions = []
    excluded = []
    for branch in range(width):
        if branch == 0:
            qj = q
        else:
            qj = q.clone()
            if excluded:
                qj[torch.tensor(excluded, dtype=torch.long, device=q.device)] = 0.0
            mass = qj.sum()
            if bool(mass > 0):
                qj = qj / mass
            else:
                qj = torch.ones_like(q)
                if excluded:
                    qj[torch.tensor(excluded, dtype=torch.long, device=q.device)] = 0.0
                remaining = int(q.numel()) - len(excluded)
                if remaining <= 0:
                    raise ValueError("no unselected token remains")
                qj = qj / float(remaining)
        token = int(torch.multinomial(qj, 1, generator=rng).item())
        candidates.append(token)
        distributions.append(qj)
        excluded.append(token)
    return candidates, distributions


def torch_acceptance_probability(residual, proposal, token: int) -> float:
    if not _is_torch_tensor(residual) or not _is_torch_tensor(proposal):
        raise TypeError("residual and proposal must be PyTorch Tensors")
    token = int(token)
    if residual.ndim != 1 or proposal.shape != residual.shape:
        raise ValueError("residual and proposal must have the same one-dimensional shape")
    q_value = float(proposal[token].item())
    if not math.isfinite(q_value) or q_value <= 0.0:
        raise ValueError("sampled token has zero proposal probability")
    r_value = float(residual[token].item())
    if not math.isfinite(r_value) or r_value < 0.0:
        raise ValueError("residual probability is invalid")
    return min(1.0, r_value / q_value)


def torch_positive_residual(residual, proposal):
    if not _is_torch_tensor(residual) or not _is_torch_tensor(proposal):
        raise TypeError("residual and proposal must be PyTorch Tensors")
    import torch

    if residual.ndim != 1 or proposal.shape != residual.shape:
        raise ValueError("residual and proposal must have the same one-dimensional shape")
    raw = torch.clamp(residual.float() - proposal.float(), min=0.0)
    mass = float(raw.sum().item())
    if not math.isfinite(mass) or mass <= 0.0:
        raise ZeroResidualMass("rejection has no positive target residual")
    return raw / mass


def torch_point_residual(target, proposed: int):
    if not _is_torch_tensor(target) or target.ndim != 1:
        raise TypeError("target must be a one-dimensional PyTorch Tensor")
    import torch

    proposed = int(proposed)
    raw = target.float().clone()
    raw[proposed] = 0.0
    mass = float(raw.sum().item())
    if not math.isfinite(mass) or mass <= 0.0:
        raise ZeroResidualMass("rejected point proposal has no residual support")
    return raw / mass


class _ScriptedRandom:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def random(self):
        if self.index >= len(self.values):
            raise AssertionError("scripted random stream exhausted")
        value = self.values[self.index]
        self.index += 1
        return value


def run_sampling_law_tests() -> dict:
    cases = []

    ordinary_target = (0.55, 0.30, 0.15)
    ordinary_proposal = (0.50, 0.35, 0.15)
    for width in (1, 2, 3):
        observed = enumerate_output_law(ordinary_target, ordinary_proposal, width)
        error = max(abs(left - right)
                    for left, right in zip(observed, normalize(ordinary_target)))
        cases.append({"name": f"ordinary_width_{width}", "max_abs_error": error,
                      "passed": error <= 1e-12})

    support_target = (0.10, 0.20, 0.30, 0.40)
    support_proposal = (0.75, 0.25, 0.0, 0.0)
    exhausted = proposal_after_exclusions(support_proposal, {0, 1})
    exhausted_ok = exhausted == (0.0, 0.0, 0.5, 0.5)
    observed = enumerate_output_law(support_target, support_proposal, 3)
    error = max(abs(left - right)
                for left, right in zip(observed, support_target))
    cases.append({"name": "support_exhaustion_uniform_unselected",
                  "fallback_distribution": list(exhausted),
                  "max_abs_error": error,
                  "passed": exhausted_ok and error <= 1e-12})

    all_rejected_rng = _ScriptedRandom((0.5, 0.2, 0.9, 0.2, 0.0))
    token, result = sample_repeated_proposals(
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2, all_rejected_rng
    )
    all_rejected_ok = (token == 0 and result["all_rejected"]
                       and result["accepted_branch"] is None
                       and result["final_residual"] == [1.0, 0.0, 0.0])
    cases.append({"name": "all_candidates_rejected_then_final_residual",
                  "result_token": token, "tried": result["tried"],
                  "final_residual": result.get("final_residual"),
                  "passed": all_rejected_ok})

    point_target = (0.20, 0.30, 0.50)
    point_observed = enumerate_point_law(point_target, proposed=1)
    point_error = max(abs(left - right)
                      for left, right in zip(point_observed, point_target))
    eos_ok = classify_token(2, p_vocab=2, eos=2) == "eos"
    feature_ok = classify_token(1, p_vocab=2, eos=5) == "position"
    feature_id_ok = classify_token(4, p_vocab=2, eos=5) == "feature"
    cases.append({"name": "point_proposal_and_eos_position_event",
                  "max_abs_error": point_error,
                  "eos_classification": "eos",
                  "feature_classification": "feature",
                  "passed": point_error <= 1e-12 and eos_ok and feature_ok and feature_id_ok})

    zero_residual_raised = False
    try:
        positive_residual((0.5, 0.5), (0.5, 0.5))
    except ZeroResidualMass:
        zero_residual_raised = True
    impossible_point_raised = False
    try:
        point_residual((0.0, 1.0), proposed=1)
    except ZeroResidualMass:
        impossible_point_raised = True
    cases.append({"name": "zero_residual_is_explicit",
                  "positive_difference_raises": zero_residual_raised,
                  "certain_point_rejection_raises": impossible_point_raised,
                  "passed": zero_residual_raised and impossible_point_raised})

    eos_is_terminal = classify_token(5, p_vocab=2, eos=5) == "eos"
    highest_feature = classify_token(4, p_vocab=2, eos=5) == "feature"
    feature_global_ids = [2 + local_id for local_id in range(3)]
    candidates_exclude_eos = all(token < 5 for token in feature_global_ids)
    cases.append({"name": "eos_is_not_a_feature_token",
                  "eos_classification": "eos",
                  "highest_feature_classification": "feature",
                  "feature_candidate_ids": feature_global_ids,
                  "passed": eos_is_terminal and highest_feature
                  and candidates_exclude_eos})

    torch_case = {"name": "torch_width1_proposal_rng_order",
                  "passed": True,
                  "skipped": False}
    try:
        import torch
    except ImportError:
        torch_case.update(skipped=True,
                          note="PyTorch is unavailable; no package was installed")
    else:
        q = torch.tensor([0.10, 0.30, 0.60], dtype=torch.float32)
        actual_rng = torch.Generator(device="cpu").manual_seed(8421)
        reference_rng = torch.Generator(device="cpu").manual_seed(8421)
        actual_ids, actual_q = torch_proposals_without_replacement(
            q, 1, actual_rng
        )
        reference_id = int(torch.multinomial(q, 1, generator=reference_rng).item())
        same_id = (actual_ids == [reference_id]
                   and len(actual_q) == 1
                   and torch.equal(actual_q[0], q))
        same_state = torch.equal(actual_rng.get_state(), reference_rng.get_state())
        torch_case.update(actual_id=actual_ids[0], reference_id=reference_id,
                          rng_state_equal=same_state,
                          passed=bool(same_id and same_state))
    cases.append(torch_case)

    return {"status": "complete" if all(case["passed"] for case in cases) else "failed",
            "test_count": len(cases),
            "passed": sum(bool(case["passed"]) for case in cases),
            "skipped": sum(bool(case.get("skipped")) for case in cases),
            "cases": cases}


def main() -> None:
    print(json.dumps(run_sampling_law_tests(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
