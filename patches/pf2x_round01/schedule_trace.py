#!/usr/bin/env python3
"""CPU-only event trace for the round01 PF schedules.

This models control flow and cache lengths from the source snapshot. It does
not load model code, draw probabilities, or measure device performance.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/project/artifacts/pf2x_loop_20261019/round01/results/schedule_traces.jsonl"
)
SOURCE_SNAPSHOT = (
    "docs/project/artifacts/pf2x_loop_20261019/round01/source_current/"
)
SCHEDULES = (
    "W1-K2",
    "W1-short-after-F-reject",
    "W2-lazy-short",
    "W2-sample-only-then-W1",
)


@dataclass
class RankerState:
    last_position: str = "P0"
    pending: str | None = None


class Trace:
    def __init__(self, schedule: str, scenario: str, budget: int, q2_outcome: str):
        assert budget >= 2 and budget % 2 == 0
        assert q2_outcome in {"accept", "reject"}
        self.schedule = schedule
        self.scenario = scenario
        self.budget = budget
        self.q2_outcome = q2_outcome
        self.prefix = ["BOS", "P0", "F0"]
        self.base_length = len(self.prefix)
        self.cache_tokens = list(self.prefix[:-1])
        self.initial_get_pos = len(self.cache_tokens)
        self.next_pair_index = 1
        self.round_index = 0
        self.ranker = RankerState()
        self.events: list[dict] = []
        self.target_calls: list[dict] = []
        self.dispatches: list[dict] = []
        self.target_input_tokens = 0
        self.max_get_pos = len(self.cache_tokens)
        self.ranker_choose_calls = 0
        self.ranker_api_operations = 0
        self.ranker_forks = 0
        self.feature_mlp_calls = 0
        self.q2_candidate_draws = 0
        self.q2_acceptance_decisions = 0
        self.q2_used = False
        self.terminal = False

        self.assert_aligned("initial complete prefix")
        self.event("start", budget_remaining=budget)

    @property
    def get_pos(self) -> int:
        return len(self.cache_tokens)

    @property
    def generated(self) -> list[str]:
        return self.prefix[self.base_length :]

    @property
    def remaining(self) -> int:
        return self.budget - len(self.generated)

    def pending_snapshot(self) -> str | None:
        return self.ranker.pending

    def begin_round(self) -> int:
        self.round_index += 1
        return self.round_index

    def event(self, name: str, **details) -> None:
        self.events.append(
            {
                "event": name,
                "get_pos": self.get_pos,
                "visible_prefix_length": len(self.prefix),
                "visible_output_tokens": len(self.generated),
                "round_index": self.round_index,
                "ranker_pending": self.pending_snapshot(),
                "ranker_last_position": self.ranker.last_position,
                **details,
            }
        )

    def assert_aligned(self, where: str) -> None:
        assert self.get_pos == len(self.prefix) - 1, (
            f"KV/prefix mismatch at {where}: get_pos={self.get_pos}, "
            f"visible_prefix_length={len(self.prefix)}"
        )
        assert self.cache_tokens == self.prefix[:-1], (
            f"KV token contents differ from visible prefix at {where}"
        )
        assert len(self.generated) <= self.budget
        if self.generated and self.generated[-1] == "EOS":
            assert self.terminal, "EOS must be a terminal state"

    def propose(self, state: RankerState, token: str, label: str) -> None:
        assert state.pending is None, f"Ranker already has pending {state.pending}"
        state.pending = token
        self.ranker_choose_calls += 1
        self.ranker_api_operations += 1
        self.event("ranker_propose", label=label, proposed=token)

    def fork(self, source: RankerState) -> RankerState:
        assert source.pending is not None
        shadow = copy.copy(source)
        self.ranker_forks += 1
        self.ranker_api_operations += 1
        self.event("ranker_fork", source_pending=source.pending)
        return shadow

    def finalize(self, state: RankerState, actual: str, label: str) -> None:
        assert state.pending is not None, f"Cannot finalize without a pending {label}"
        proposed = state.pending
        state.pending = None
        state.last_position = actual
        self.ranker_api_operations += 1
        self.event("ranker_finalize", label=label, proposed=proposed, actual=actual)

    def drop_pending(self, state: RankerState, reason: str) -> None:
        previous = state.pending
        state.pending = None
        self.ranker_api_operations += 1
        self.event("ranker_drop_pending", reason=reason, discarded=previous)

    def forward(self, tokens: list[str], phase: str) -> None:
        assert tokens, "Target forward needs at least one input token"
        self.assert_aligned(f"before {phase}")
        before = self.get_pos
        self.cache_tokens.extend(tokens)
        after = self.get_pos
        self.target_input_tokens += len(tokens)
        self.max_get_pos = max(self.max_get_pos, after)
        call = {
            "call": len(self.target_calls) + 1,
            "phase": phase,
            "input": list(tokens),
            "input_tokens": len(tokens),
            "get_pos_before": before,
            "get_pos_after": after,
            "ranker_pending": self.pending_snapshot(),
        }
        self.target_calls.append(call)
        self.event("target_forward", **call)

    def rewind(self, keep: int, reason: str) -> None:
        old = self.get_pos
        assert 0 <= keep <= old
        del self.cache_tokens[keep:]
        self.event("target_rewind", reason=reason, get_pos_before=old, get_pos_after=keep)

    def append(self, tokens: list[str], reason: str) -> None:
        assert not self.terminal
        self.prefix.extend(tokens)
        self.next_pair_index += sum(token.startswith("P") for token in tokens)
        if tokens and tokens[-1] == "EOS":
            self.terminal = True
        self.event("visible_output", reason=reason, appended=list(tokens))
        self.assert_aligned(reason)

    def dispatch(self, branch: str, remaining: int) -> None:
        row = {"branch": branch, "remaining_at_start": remaining}
        self.dispatches.append(row)
        self.event("budget_dispatch", **row)

    def round_k2(
        self,
        *,
        p1_result: str = "accept",
        f1_result: str = "accept",
        p2_result: str = "accept",
        q2_outcome: str | None = None,
    ) -> str:
        assert self.remaining >= 4
        assert p1_result in {"accept", "reject", "eos"}
        assert f1_result in {"accept", "reject"}
        assert p2_result in {"accept", "reject", "eos"}
        start = self.get_pos
        pending_f = self.prefix[-1]
        round_index = self.begin_round()
        pair1 = self.next_pair_index
        pair2 = pair1 + 1
        p1_token = f"P{pair1}@R{round_index}"
        f1a_token = f"F{pair1}a@R{round_index}"
        p2_token = f"P{pair2}@R{round_index}"
        self.dispatch("K2", self.remaining)

        live = self.ranker
        self.propose(live, p1_token, "first Position")
        shadow = self.fork(live)
        self.finalize(shadow, p1_token, "preview first Position")
        self.propose(shadow, p2_token, "second Position preview")
        self.feature_mlp_calls += 1
        self.event(
            "feature_propose", proposal=f1a_token, calls=self.feature_mlp_calls,
            input_tokens=1,
        )
        self.forward([pending_f, p1_token, f1a_token, p2_token], "K2 verification")

        if p1_result in {"reject", "eos"}:
            actual = "EOS" if p1_result == "eos" else f"P{pair1}_RESIDUAL@R{round_index}"
            self.rewind(start + 1, f"{p1_token} residual keeps pending Feature")
            if actual == "EOS":
                self.ranker = live
                self.drop_pending(live, f"terminal EOS after {p1_token} Target distribution")
                self.append(["EOS"], f"{p1_token} residual selected terminal EOS")
                return "eos"
            self.append([actual], f"{p1_token} residual")
            self.forward([actual], f"{p1_token} residual Feature draw")
            self.append([f"F{pair1}_TARGET@R{round_index}"], "Target Feature after P1 residual")
            self.ranker = live
            self.finalize(live, actual, f"{p1_token} residual")
            return "p1_reject"

        self.ranker = shadow
        if f1_result == "reject":
            self.event("feature_first_reject", proposal=f1a_token)
            if self.schedule in {"W2-lazy-short", "W2-sample-only-then-W1"}:
                outcome = q2_outcome or self.q2_outcome
                self.q2_used = True
                self.q2_candidate_draws += 1
                self.q2_acceptance_decisions += 1
                self.event(
                    "feature_second_candidate",
                    candidate=f"F{pair1}b@R{round_index}",
                    q2_mode=f"renormalized q{pair1} excluding {f1a_token}",
                    decision=outcome,
                    verified_suffix_tokens=0,
                )
                actual_f = (
                    f"F{pair1}b@R{round_index}"
                    if outcome == "accept"
                    else f"F{pair1}_RESIDUAL_2@R{round_index}"
                )
                if outcome == "reject":
                    self.event("feature_second_reject_residual", final_feature=actual_f)
                else:
                    self.event("feature_second_accept", final_feature=actual_f)
            else:
                actual_f = f"F{pair1}_RESIDUAL@R{round_index}"
            self.rewind(
                start + 2,
                f"{f1a_token} residual keeps pending Feature and accepted {p1_token}",
            )
            self.append([p1_token, actual_f], f"{p1_token} accepted; one final Feature committed")
            self.drop_pending(self.ranker, f"source W1 clears previewed {p2_token} after F rejection")
            return "f1_reject"

        self.event("feature_first_accept", proposal=f1a_token)
        if p2_result in {"reject", "eos"}:
            actual_p2 = "EOS" if p2_result == "eos" else f"P{pair2}_RESIDUAL@R{round_index}"
            self.rewind(start + 3, f"{p2_token} residual keeps the accepted {f1a_token} branch")
            if actual_p2 == "EOS":
                self.drop_pending(self.ranker, f"terminal EOS after {p2_token} Target distribution")
                self.append(
                    [p1_token, f1a_token, "EOS"],
                    f"{p2_token} residual selected terminal EOS",
                )
                return "eos"
            self.append([p1_token, f1a_token, actual_p2], f"{p2_token} residual")
            self.forward([actual_p2], f"{p2_token} residual Feature draw")
            self.append(
                [f"F{pair2}_TARGET@R{round_index}"],
                f"Target Feature after {p2_token} residual",
            )
            self.finalize(self.ranker, actual_p2, f"{p2_token} residual")
            return "p2_reject"

        self.append(
            [p1_token, f1a_token, p2_token, f"F{pair2}_TARGET_BONUS@R{round_index}"],
            f"{p1_token}, {f1a_token}, {p2_token} accepted; Target bonus Feature sampled",
        )
        self.finalize(self.ranker, p2_token, f"{p2_token} accepted")
        return "all_accept"

    def round_k1(self, p1_result: str = "accept") -> str:
        assert self.remaining == 2
        assert p1_result in {"accept", "reject", "eos"}
        start = self.get_pos
        pending_f = self.prefix[-1]
        round_index = self.begin_round()
        pair = self.next_pair_index
        p_token = f"P{pair}@R{round_index}"
        self.dispatch("K1 budget tail", self.remaining)
        self.propose(self.ranker, p_token, "tail Position")
        self.forward([pending_f, p_token], "K1 tail verification")
        if p1_result == "eos":
            self.rewind(start + 1, f"{p_token} residual keeps pending Feature")
            self.drop_pending(self.ranker, f"terminal EOS after {p_token} Target distribution")
            self.append(["EOS"], f"{p_token} residual selected terminal EOS")
            return "eos"
        if p1_result == "reject":
            self.rewind(start + 1, f"{p_token} residual keeps pending Feature")
            residual_p = f"P{pair}_RESIDUAL@R{round_index}"
            self.append([residual_p], f"{p_token} residual")
            self.forward([residual_p], f"{p_token} residual Feature draw")
            self.append([f"F{pair}_TARGET@R{round_index}"], f"Target Feature after {p_token} residual")
            self.finalize(self.ranker, residual_p, f"{p_token} residual")
            return "p1_reject"
        self.append(
            [p_token, f"F{pair}_TARGET@R{round_index}"],
            f"{p_token} accepted; Target Feature sampled",
        )
        self.finalize(self.ranker, p_token, f"{p_token} accepted")
        return "all_accept"

    def round_short(self, p2_result: str = "accept") -> str:
        assert self.remaining >= 2
        assert p2_result in {"accept", "reject", "eos"}
        start = self.get_pos
        pending_f = self.prefix[-1]
        round_index = self.begin_round()
        pair = self.next_pair_index
        p2_token = f"P{pair}@R{round_index}"
        self.dispatch("short K1 window after F rejection", self.remaining)
        self.propose(self.ranker, p2_token, "fresh short-window Position")
        self.forward([pending_f, p2_token], "short two-token Target window")
        if p2_result in {"reject", "eos"}:
            actual = "EOS" if p2_result == "eos" else f"P{pair}_RESIDUAL@R{round_index}"
            self.rewind(start + 1, f"{p2_token} residual keeps pending Feature")
            if actual == "EOS":
                self.drop_pending(self.ranker, f"terminal EOS after {p2_token} Target distribution")
                self.append(["EOS"], f"{p2_token} residual selected terminal EOS")
                return "eos"
            self.append([actual], f"{p2_token} residual")
            self.forward([actual], f"{p2_token} residual Feature draw")
            self.append(
                [f"F{pair}_TARGET@R{round_index}"],
                f"Target Feature after {p2_token} residual",
            )
            self.finalize(self.ranker, actual, f"{p2_token} residual")
            return "p2_reject"
        self.append(
            [p2_token, f"F{pair}_TARGET@R{round_index}"],
            f"{p2_token} accepted; Target Feature sampled",
        )
        self.finalize(self.ranker, p2_token, f"{p2_token} accepted")
        return "all_accept"

    def finish_budget(self, tail_p1_result: str = "accept") -> None:
        while self.remaining > 0 and not self.terminal:
            if self.remaining >= 4:
                self.round_k2()
            else:
                self.round_k1(tail_p1_result)
        if not self.terminal:
            assert len(self.generated) == self.budget

    def result(self) -> dict:
        self.assert_aligned("final committed state")
        output = self.generated
        eos = bool(output and output[-1] == "EOS")
        return {
            "record_type": "cpu_event_simulation",
            "schedule": self.schedule,
            "scenario": self.scenario,
            "budget_tokens": self.budget,
            "generated_tokens": len(output),
            "content_tokens_excluding_eos": len(output) - int(eos),
            "position_output_tokens": sum(token.startswith("P") for token in output),
            "feature_output_tokens": sum(token.startswith("F") for token in output),
            "eos_output_tokens": int(eos),
            "output_symbols": output,
            "eos": eos,
            "gpu_measured": False,
            "source_snapshot": SOURCE_SNAPSHOT,
            "target": {
                "calls": len(self.target_calls),
                "input_tokens_total": self.target_input_tokens,
                "input_lengths": [call["input_tokens"] for call in self.target_calls],
                "inputs": [call["input"] for call in self.target_calls],
                "initial_get_pos": self.initial_get_pos,
                "final_get_pos": self.get_pos,
                "max_trial_get_pos": self.max_get_pos,
                "committed_kv_slots": self.get_pos,
                "committed_kv_slots_added_during_trace": self.get_pos - self.initial_get_pos,
                "committed_kv_token_symbols": list(self.cache_tokens),
                "last_visible_token_is_unconsumed": bool(output and output[-1] != "EOS"),
                "terminal_eos_is_unconsumed": eos,
            },
            "ranker": {
                "decoder_ranker_calls": self.ranker_choose_calls,
                "choose_position_calls": self.ranker_choose_calls,
                "api_operations_including_state_updates": self.ranker_api_operations,
                "forks": self.ranker_forks,
                "pending_final": self.ranker.pending,
            },
            "feature": {
                "mlp_propose_calls": self.feature_mlp_calls,
                "q2_candidate_draws": self.q2_candidate_draws,
                "q2_acceptance_decisions": self.q2_acceptance_decisions,
                "q2_used": self.q2_used,
                "second_candidate_is_an_additional_visible_feature": False,
            },
            "budget": {
                "remaining_at_end": self.remaining,
                "dispatches": self.dispatches,
                "exact_non_eos_budget_match": not eos and len(output) == self.budget,
            },
            "trace": self.events,
            "target_calls_detail": self.target_calls,
            "validation_scope": [
                "Validates constructed control flow and cache accounting only.",
                "Does not validate model token IDs, actual Ranker numeric monotonicity, numeric 4D RoPE, or sampling probabilities.",
            ],
            "assumptions": [
                "Symbolic token names stand for event outcomes; no token IDs or probabilities are sampled.",
                "All listed paths are constructed CPU traces, not Target or GPU measurements.",
                "W2 q2 uses the returned q1 distribution with the first candidate removed and renormalized.",
                "After the source clears pending P2 on F rejection, short-window paths request a fresh Position proposal.",
                "At stage two, acceptance samples min(r1,q2(·|a))/beta(a), while rejection samples the normalized positive residual of r1-q2(·|a); together these branches produce exactly one final Feature.",
                "Different schedules need not produce identical token symbols or share a random seed.",
            ],
        }


def simulate(schedule: str, scenario: str, budget: int, q2_outcome: str) -> dict:
    sim = Trace(schedule, scenario, budget, q2_outcome)
    if scenario == "all_accept":
        sim.finish_budget()
    elif scenario == "p1_reject_then_continue":
        sim.round_k2(p1_result="reject")
        sim.finish_budget()
    elif scenario in {"f1_reject_q2_accept", "f1_reject_q2_reject"}:
        chosen_q2 = "accept" if scenario.endswith("accept") else "reject"
        sim.round_k2(f1_result="reject", q2_outcome=chosen_q2)
        if schedule in {"W1-short-after-F-reject", "W2-lazy-short"} and not sim.terminal:
            sim.round_short()
        sim.finish_budget()
    elif scenario == "p2_reject_then_continue":
        sim.round_k2(p2_result="reject")
        sim.finish_budget()
    elif scenario == "tail_k1_reject":
        sim.round_k2()
        sim.round_k1(p1_result="reject")
    elif scenario == "eos_after_p1_reject":
        sim.round_k2(p1_result="eos")
    elif scenario == "eos_after_p2_reject":
        sim.round_k2(p2_result="eos")
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    result = sim.result()
    result["q2_outcome_request"] = q2_outcome
    result["q2_outcome_used"] = (
        "accept" if scenario == "f1_reject_q2_accept" else
        "reject" if scenario == "f1_reject_q2_reject" else None
    ) if sim.q2_used else None
    assert result["target"]["final_get_pos"] == len(sim.prefix) - 1
    assert result["ranker"]["pending_final"] is None
    if scenario.startswith("eos_"):
        assert result["eos"]
    if scenario not in {"eos_after_p1_reject", "eos_after_p2_reject"}:
        assert result["budget"]["exact_non_eos_budget_match"]
    return result


def build_records() -> list[dict]:
    records: list[dict] = []
    cases = (
        ("all_accept", (6, 8), "accept"),
        ("p1_reject_then_continue", (6,), "accept"),
        ("f1_reject_q2_accept", (6, 8), "accept"),
        ("f1_reject_q2_reject", (6, 8), "reject"),
        ("p2_reject_then_continue", (6,), "accept"),
        ("tail_k1_reject", (6,), "accept"),
        ("eos_after_p1_reject", (6,), "accept"),
        ("eos_after_p2_reject", (6,), "accept"),
    )
    for scenario, budgets, q2_outcome in cases:
        for budget in budgets:
            for schedule in SCHEDULES:
                records.append(simulate(schedule, scenario, budget, q2_outcome))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = build_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "gpu_measured": False,
                "exact_budget_matches": sum(
                    row["budget"]["exact_non_eos_budget_match"] for row in records
                ),
                "terminal_eos_records": sum(row["eos"] for row in records),
                "boundary_k1_records": sum(
                    any(d["branch"] == "K1 budget tail" for d in row["budget"]["dispatches"])
                    for row in records
                ),
                "covered_scenarios": sorted({row["scenario"] for row in records}),
                "covered_schedules": list(SCHEDULES),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
