"""Exact sequential speculative sampling of typed GaussianGPT P/F blocks.

This is a separate experimental engine; the existing position-only engine is
unchanged. A block ends on a Feature slot (unless EOS/token limit intervenes).
Each candidate is accepted with min(1,p/q), using its own causal prefix and final
sampling policy. The first rejection commits a sample from normalized [p-q]+.
This is tokenwise rejection sampling, not a joint-pair acceptance approximation.

At a round boundary the target cache contains [BOS] + output[:-1]. The pending
last output (BOS initially) is the first verification input. If room remains,
the last candidate is also consumed to obtain one target bonus token. Draft may
trail by one additional token after this bonus; the next round synchronizes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .position_selective_speculative import (
    PositionSelectiveSpeculativeEngine, PositionSelectiveStats,
    PositionSelectiveResult, accept_or_correct_position, make_kv_cache,
    sample_from_probs, sampling_probs, truncate_kv_cache,
)


@dataclass
class StructuredStats(PositionSelectiveStats):
    rounds: int = 0
    drafted_tokens: int = 0
    feature_proposals: int = 0
    feature_accepts: int = 0
    feature_rejects: int = 0
    bonus_tokens: int = 0
    accepted_tokens: int = 0
    accepted_pairs: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    blocks: list = field(default_factory=list)

    def as_dict(self):
        result = super().as_dict()
        for name in ("rounds", "drafted_tokens", "feature_proposals", "feature_accepts",
                     "feature_rejects", "bonus_tokens", "accepted_tokens", "accepted_pairs",
                     "target_calls", "draft_calls", "blocks"):
            result[name] = getattr(self, name)
        result["empirical_alpha_f"] = (
            self.feature_accepts / self.feature_proposals if self.feature_proposals else None)
        result["committed_tokens_per_target_call"] = self.generated_tokens / self.target_calls
        return result


class StructuredSpeculativeEngine(PositionSelectiveSpeculativeEngine):
    def __init__(self, target_gpt, draft_gpt, *, k_pairs=2, measure_latency=False,
                 check_invariants=False, audit_callback: Optional[Callable] = None,
                 record_blocks=True):
        super().__init__(target_gpt, draft_gpt, measure_latency=measure_latency)
        if not isinstance(k_pairs, int) or isinstance(k_pairs, bool) or k_pairs < 1:
            raise ValueError("k_pairs must be a positive integer")
        # These determine token coordinates/slot semantics, beyond vocab equality.
        for name in ("rope_bos_coord", "dense_chunk_order"):
            if getattr(self.target.gpt_config, name) != getattr(self.draft.gpt_config, name):
                raise ValueError(f"target/draft mismatch for {name}")
        self.k_pairs = k_pairs
        self.check_invariants = check_invariants
        self.audit_callback = audit_callback
        self.record_blocks = record_blocks

    def _check_token(self, token, index):
        if index % 2 == 0:
            valid = 0 <= token < self.target.position_vocab_size or token == self.target.eos_token_id
        else:
            offset = 0 if self.target.gpt_config.sparse_shared_vocab else self.target.feature_token_offset
            valid = offset <= token < offset + self.target.feature_vocab_size
        if not valid:
            raise RuntimeError(f"Invalid token {token} at generated slot {index}")

    @torch.inference_mode()
    def generate(self, *, max_tokens, temperature=1.0, top_k=None, top_p=None,
                 seed=42, stop_on_eos=True, sampling_params_resolver=None):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not stop_on_eos:
            raise NotImplementedError("sparse structured decoding requires stop_on_eos=True")
        if max_tokens > min(self.target.gpt_config.sequence_len, self.draft.gpt_config.sequence_len):
            raise ValueError("max_tokens exceeds target/draft context length")
        stats = StructuredStats()
        rng = torch.Generator(device=self.device).manual_seed(int(seed))
        tc = make_kv_cache(self.target_model, max_tokens)
        dc = make_kv_cache(self.draft_model, max_tokens)
        bos = eos = int(self.target.eos_token_id)
        output = []

        def ids(values):
            return torch.tensor([values], dtype=torch.long, device=self.device)

        def probs(logits, index):
            params = self._params(index, temperature, top_k, top_p, sampling_params_resolver)
            return sampling_probs(logits, *params)

        def forward(model, values, cache, kind):
            if kind.startswith("target"):
                stats.target_calls += 1
            else:
                stats.draft_calls += 1
            return self._forward(model, ids(values), cache, kind=kind, stats=stats)

        while len(output) < max_tokens:
            start = len(output)
            history = [bos] + output
            if tc.get_pos() != start:
                raise RuntimeError("target must trail committed history by exactly one token")
            # A Position bonus/correction can leave the next round at a Feature.
            budget = min(2 * self.k_pairs - start % 2, max_tokens - start)
            proposal, q_policies = [], []
            q_logits = forward(self.draft_model, history[dc.get_pos():], dc, "draft_sync")[:, -1, :]
            for j in range(budget):
                q = probs(q_logits, start + j)
                token = int(sample_from_probs(q, rng).item())
                self._check_token(token, start + j)
                proposal.append(token)
                q_policies.append(q)
                if token == eos or j + 1 == budget:
                    break
                q_logits = forward(self.draft_model, [token], dc, "draft_step")[:, -1, :]

            stats.rounds += 1
            stats.drafted_tokens += len(proposal)
            bonus = proposal[-1] != eos and start + len(proposal) < max_tokens
            verify = [history[-1]] + (proposal if bonus else proposal[:-1])
            p_logits = forward(self.target_model, verify, tc, "target_block")
            if p_logits.shape[1] != len(verify):
                raise RuntimeError("target returned an unexpected number of causal logit rows")
            accepted = 0
            rejected_slot = None
            accepted_positions = set()
            for j, token in enumerate(proposal):
                p = probs(p_logits[:, j, :], start + j)
                committed, ok, _, overlap = accept_or_correct_position(ids([token]), p, q_policies[j], rng)
                value = int(committed.item())
                self._check_token(value, start + j)
                is_position = (start + j) % 2 == 0
                if is_position:
                    stats.position_proposals += 1
                    stats.position_accepts += int(ok)
                    stats.position_rejects += int(not ok)
                    stats.position_overlap_sum += overlap
                else:
                    stats.feature_proposals += 1
                    stats.feature_accepts += int(ok)
                    stats.feature_rejects += int(not ok)
                    stats.feature_observations += 1
                    stats.feature_overlap_sum += overlap
                output.append(value)
                if ok:
                    accepted += 1
                    stats.accepted_tokens += 1
                    if is_position:
                        accepted_positions.add(start + j)
                    elif start + j - 1 in accepted_positions:
                        stats.accepted_pairs += 1
                else:
                    rejected_slot = "position" if is_position else "feature"
                if value == eos:
                    stats.terminated_on_eos = True
                if not ok or stats.terminated_on_eos:
                    break

            if rejected_slot is None and not stats.terminated_on_eos and bonus:
                token = int(sample_from_probs(probs(p_logits[:, len(proposal), :], len(output)), rng).item())
                self._check_token(token, len(output))
                output.append(token)
                stats.bonus_tokens += 1
                stats.terminated_on_eos = token == eos

            # The correction/bonus stays pending, never retaining KV for a
            # rejected candidate or any descendant of its old P/F state.
            truncate_kv_cache(tc, len(output))
            truncate_kv_cache(dc, min(dc.get_pos(), len(output)))
            if self.check_invariants:
                committed_history = ids([bos] + output)
                for name, cache in (("target", tc), ("draft", dc)):
                    n = cache.get_pos()
                    if not torch.equal(cache.get_tokens(n), committed_history[:, :n]):
                        raise RuntimeError(f"{name} cache contains uncommitted tokens")
                    if bool((cache.token_cache[:, n:] != -1).any()):
                        raise RuntimeError(f"{name} stale token metadata after rollback")
            if self.audit_callback is not None:
                self.audit_callback(output, tc, dc)
            if self.record_blocks:
                stats.blocks.append(dict(start=start, proposed=len(proposal), accepted=accepted,
                                         committed=len(output)-start, rejected_slot=rejected_slot,
                                         verify_input_tokens=len(verify)))
            if stats.terminated_on_eos:
                break
        stats.generated_tokens = len(output)
        stats.generated_pairs = sum(output[i] != eos for i in range(0, len(output)-1, 2))
        return PositionSelectiveResult(tokens=ids(output), stats=stats)
