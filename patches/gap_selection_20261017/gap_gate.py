"""Pure route selection for the GaussianGPT position/feature decoder."""
from dataclasses import dataclass


P_ONLY = "p_only"
PF_K2 = "pf_k2"


@dataclass(frozen=True)
class RouteDecision:
    route: str
    gap: int | None
    reason: str


def choose_gap_route(candidate_position, previous_position, remaining_tokens,
                     p_vocab=8000):
    """Choose PF-K2 for serialized ID gap 1; otherwise use the P-only route."""
    if candidate_position is None:
        return RouteDecision(P_ONLY, None, "no_candidate")
    candidate = int(candidate_position)
    previous = int(previous_position)
    if remaining_tokens < 4:
        gap = candidate - previous
        return RouteDecision(P_ONLY, gap, "short_budget")
    if not (0 <= candidate < p_vocab and 0 <= previous < p_vocab):
        return RouteDecision(P_ONLY, candidate - previous, "invalid_position")
    gap = candidate - previous
    if gap <= 0:
        return RouteDecision(P_ONLY, gap, "nonpositive_gap")
    if gap == 1:
        return RouteDecision(PF_K2, gap, "unit_gap")
    return RouteDecision(P_ONLY, gap, "larger_gap")
