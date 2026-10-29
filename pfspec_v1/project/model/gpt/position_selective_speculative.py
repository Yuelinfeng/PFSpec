"""
Lossless Position-Selective Speculative Decoding for sparse GaussianGPT.

Design
------
GaussianGPT sparse streams alternate

    BOS, P_1, F_1, P_2, F_2, ..., EOS

This engine speculates only Position tokens with a draft model. Feature tokens are
always sampled from the target model. For stochastic decoding it implements the
standard speculative-sampling accept/reject rule on the *final sampling
distributions* (after slot/hard masks in the model and after temperature/top-k/
top-p filtering here):

    accept(x) = min(1, p(x) / q(x))

and, on rejection, samples from normalized [p - q]_+.

The target execution schedule is a 2-token block in steady state:

    target cache through P_{i-1}
        + [F_{i-1}, draft(P_i)]
        -> logits for P_i and F_i|draft(P_i)

If P_i is accepted, the second logit row is the target-only Feature distribution
(the "bonus" token). If P_i is rejected, the speculative P_i cache entry is
rolled back, the corrected P_i is fed once, and the resulting Feature logits are
used.

The draft cache is synchronized with the *committed* pair [P_i, F_i], which also
produces q_F for diagnostics and q_P for the next round.

Current scope
-------------
- sparse streams only
- exactly one Position token and one Feature token per latent (P,F pairs)
- batch size 1 (the repository KVCache assumes a uniform cache length; batched
  rows can diverge at rejection)
- stop_on_eos=True

No existing baseline source file is modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import mean, median
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .engine import KVCache, SamplingParamsResolver
from .draft_proposal import DraftProposal, DraftProposalState


@dataclass(frozen=True)
class TimingSample:
    """One synchronized forward-latency sample."""

    kind: str
    context_tokens: int
    input_tokens: int
    milliseconds: float


@dataclass
class PositionSelectiveStats:
    """Diagnostics collected during one selective-decoding run."""

    generated_tokens: int = 0
    generated_pairs: int = 0
    terminated_on_eos: bool = False

    position_proposals: int = 0
    position_accepts: int = 0
    position_rejects: int = 0
    position_overlap_sum: float = 0.0

    feature_observations: int = 0
    feature_overlap_sum: float = 0.0

    # Shadow joint [Position, Feature] acceptance diagnostics.
    feature_proposals: int = 0
    feature_accepts: int = 0
    feature_rejects: int = 0
    feature_accept_prob_sum: float = 0.0
    feature_shadow_overlap_sum: float = 0.0
    pair_accepts: int = 0

    timing: List[TimingSample] = field(default_factory=list)

    stage_timing: Dict[str, List[float]] = field(default_factory=dict)

    def record_stage(self, kind: str, milliseconds: float) -> None:
        self.stage_timing.setdefault(kind, []).append(float(milliseconds))

    def stage_timing_summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for kind, values in self.stage_timing.items():
            if not values:
                continue
            ordered = sorted(values)
            n = len(ordered)
            result[kind] = {
                "count": float(n),
                "mean_ms": float(sum(ordered) / n),
                "median_ms": float(ordered[n // 2]),
                "p95_ms": float(ordered[min(n - 1, int(round((n - 1) * 0.95)))]),
            }
        return result

    @property
    def empirical_alpha_p(self) -> float:
        if self.position_proposals == 0:
            return float("nan")
        return self.position_accepts / self.position_proposals

    @property
    def overlap_alpha_p(self) -> float:
        if self.position_proposals == 0:
            return float("nan")
        return self.position_overlap_sum / self.position_proposals

    @property
    def overlap_alpha_f(self) -> float:
        if self.feature_observations == 0:
            return float("nan")
        return self.feature_overlap_sum / self.feature_observations

    @property
    def empirical_alpha_f_shadow(self) -> float:
        if self.feature_proposals == 0:
            return float("nan")
        return self.feature_accepts / self.feature_proposals

    @property
    def overlap_alpha_f_shadow(self) -> float:
        if self.feature_proposals == 0:
            return float("nan")
        return self.feature_shadow_overlap_sum / self.feature_proposals

    @property
    def empirical_alpha_pair(self) -> float:
        if self.position_proposals == 0:
            return float("nan")
        return self.pair_accepts / self.position_proposals

    def timing_summary(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[float]] = {}
        for sample in self.timing:
            grouped.setdefault(sample.kind, []).append(sample.milliseconds)

        result: Dict[str, Dict[str, float]] = {}
        for kind, values in grouped.items():
            ordered = sorted(values)
            n = len(ordered)

            def percentile(p: float) -> float:
                if n == 1:
                    return ordered[0]
                idx = int(round((n - 1) * p))
                return ordered[max(0, min(idx, n - 1))]

            result[kind] = {
                "count": float(n),
                "mean_ms": float(mean(ordered)),
                "median_ms": float(median(ordered)),
                "p90_ms": float(percentile(0.90)),
                "p95_ms": float(percentile(0.95)),
            }
        return result

    def as_dict(self) -> Dict[str, object]:
        return {
            "generated_tokens": self.generated_tokens,
            "stage_timing": self.stage_timing_summary(),
            "generated_pairs": self.generated_pairs,
            "terminated_on_eos": self.terminated_on_eos,
            "position_proposals": self.position_proposals,
            "position_accepts": self.position_accepts,
            "position_rejects": self.position_rejects,
            "empirical_alpha_p": self.empirical_alpha_p,
            "overlap_alpha_p": self.overlap_alpha_p,
            "feature_observations": self.feature_observations,
            "overlap_alpha_f": self.overlap_alpha_f,
            "feature_shadow_proposals": self.feature_proposals,
            "feature_shadow_accepts": self.feature_accepts,
            "feature_shadow_rejects": self.feature_rejects,
            "feature_shadow_accept_prob_mean": (
                self.feature_accept_prob_sum / self.feature_proposals
                if self.feature_proposals else float("nan")
            ),
            "feature_shadow_overlap": self.overlap_alpha_f_shadow,
            "feature_shadow_alpha": self.empirical_alpha_f_shadow,
            "pair_shadow_accepts": self.pair_accepts,
            "pair_shadow_alpha": self.empirical_alpha_pair,
            "timing": self.timing_summary(),
        }


@dataclass
class PositionSelectiveResult:
    """Generated token stream (without BOS) plus diagnostics."""

    tokens: torch.Tensor  # shape (1, T)
    stats: PositionSelectiveStats


def unwrap_compiled(module):
    """Return the original nn.Module behind torch.compile when present."""

    return getattr(module, "_orig_mod", module)


def sampling_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert already-structurally-masked logits to the exact sampling distribution.

    This mirrors model/gpt/engine.py::sample_next_token. In particular,
    temperature==0 ignores top-k/top-p and becomes a point mass at argmax, which
    matches the repository baseline.

    Args:
        logits: (B, V) or (V,) logits. Model slot/monotonic masks must already
            have been applied by GPT.forward(..., kv_cache=...).
    Returns:
        probabilities with shape (B, V).
    """

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (B,V) or (V,), got {tuple(logits.shape)}")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    logits = logits.float()

    if temperature == 0.0:
        argmax = torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(1, argmax, 1.0)
        return probs

    filtered = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(int(top_k), filtered.size(-1))
        kth_values = torch.topk(filtered, k, dim=-1).values[:, [-1]]
        filtered = filtered.masked_fill(filtered < kth_values, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(
            sorted_indices_to_remove, float("-inf")
        )
        restored = torch.full_like(filtered, float("-inf"))
        restored.scatter_(1, sorted_indices, sorted_logits)
        filtered = restored

    probs = F.softmax(filtered, dim=-1)
    if not torch.isfinite(probs).all():
        raise FloatingPointError("sampling distribution contains NaN/Inf")
    return probs


def distribution_overlap(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return sum_x min(p(x), q(x)) for each batch row."""

    if p.shape != q.shape:
        raise ValueError(f"distribution shape mismatch: {tuple(p.shape)} != {tuple(q.shape)}")
    return torch.minimum(p, q).sum(dim=-1)


def sample_from_probs(probs: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Sample one token per batch row; returns shape (B,1)."""

    if probs.ndim != 2:
        raise ValueError(f"probs must be (B,V), got {tuple(probs.shape)}")
    # Preserve deterministic greedy behavior without consuming a random draw.
    is_point_mass = (probs.max(dim=-1).values == 1.0).all()
    if bool(is_point_mass.item()):
        return torch.argmax(probs, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=rng)


def accept_or_correct_position(
    proposal: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, bool, float, float]:
    """
    Standard speculative-sampling accept/reject + residual correction.

    Args:
        proposal: shape (1,1), sampled from q.
        p, q: shape (1,V), final target/draft sampling distributions.
    Returns:
        (committed_token, accepted, acceptance_probability, overlap)
    """

    if proposal.shape != (1, 1):
        raise ValueError("current implementation requires proposal shape (1,1)")
    if p.shape != q.shape or p.shape[0] != 1:
        raise ValueError("current implementation requires p,q shape (1,V)")

    token_id = int(proposal.item())
    qx = float(q[0, token_id].item())
    px = float(p[0, token_id].item())
    if qx <= 0.0:
        raise RuntimeError("proposal has zero probability under q; sampler invariant violated")

    accept_prob = min(1.0, px / qx)
    overlap = float(distribution_overlap(p, q)[0].item())

    # Avoid an unnecessary RNG draw for probability-0/1 decisions.
    if accept_prob >= 1.0:
        return proposal, True, accept_prob, overlap
    if accept_prob <= 0.0:
        accepted = False
    else:
        u = float(torch.rand((), device=p.device, generator=rng).item())
        accepted = u < accept_prob

    if accepted:
        return proposal, True, accept_prob, overlap

    residual = torch.clamp(p - q, min=0.0)
    mass = residual.sum(dim=-1, keepdim=True)
    if float(mass.item()) <= 1e-12:
        # The rejection event has zero probability when p == q. This fallback is
        # only for finite-precision corner cases and keeps the target distribution.
        residual = p
    else:
        residual = residual / mass
    corrected = sample_from_probs(residual, rng)
    return corrected, False, accept_prob, overlap


def accept_or_correct_position_compact(proposal_index: torch.Tensor, p: torch.Tensor, q: torch.Tensor, rng: torch.Generator) -> Tuple[torch.Tensor, bool, float, float]:
    if proposal_index.shape != (1, 1): raise ValueError('proposal shape must be (1,1)')
    if p.shape != q.shape or p.shape[0] != 1: raise ValueError('p,q shape mismatch')
    index = int(proposal_index.item()); qx = float(q[0,index].item()); px = float(p[0,index].item())
    if qx <= 0.0: raise RuntimeError('proposal has zero probability under q')
    accept_prob = min(1.0, px/qx); overlap = float(distribution_overlap(p,q)[0].item())
    if accept_prob >= 1.0: return proposal_index, True, accept_prob, overlap
    if accept_prob <= 0.0: accepted = False
    else: accepted = float(torch.rand((), device=p.device, generator=rng).item()) < accept_prob
    if accepted: return proposal_index, True, accept_prob, overlap
    residual = torch.clamp(p-q, min=0.0); mass = residual.sum(dim=-1, keepdim=True)
    residual = p if float(mass.item()) <= 1e-12 else residual/mass
    return sample_from_probs(residual, rng), False, accept_prob, overlap


def make_kv_cache(model, max_tokens: int) -> KVCache:
    """Allocate a batch-1 cache for a raw nanochat GPT model."""

    cfg = model.config
    device = model.get_device()
    dtype = next(model.parameters()).dtype
    return KVCache(
        batch_size=1,
        num_heads=cfg.n_kv_head,
        seq_len=max_tokens + 1,  # BOS + generated tokens
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer,
        device=device,
        dtype=dtype,
    )


def truncate_kv_cache(cache: KVCache, new_pos: int) -> None:
    """
    Roll back logical KV length without clearing K/V storage.

    Stale K/V entries at positions >= new_pos are ignored because cache_seqlens
    is reduced and will be overwritten by the next forward. token_cache *is*
    cleared because GaussianGPT's hard position constraints inspect it.
    """

    current = cache.get_pos()
    if not (0 <= new_pos <= current):
        raise ValueError(f"invalid cache truncate {current} -> {new_pos}")
    cache.cache_seqlens.fill_(int(new_pos))
    cache.token_cache[:, new_pos:].fill_(-1)


class PositionSelectiveSpeculativeEngine:
    """Lossless Position-selective speculative decoder for sparse GaussianGPT."""

    def __init__(self, target_gpt, draft_gpt, *, measure_latency: bool = False,
                 draft_proposer: Optional[DraftProposal] = None):
        # target_gpt/draft_gpt are NanoGPTGaussianModel instances (possibly the
        # _orig_mod behind torch.compile), not GaussianGPT Lightning wrappers.
        self.target = unwrap_compiled(target_gpt)
        self.draft = unwrap_compiled(draft_gpt)
        self.measure_latency = bool(measure_latency)
        self.draft_proposer = draft_proposer
        self._validate_models()

        self.target_model = self.target.transformer
        self.draft_model = self.draft.transformer
        self.device = self.target_model.get_device()
        self.position_token_ids = torch.cat((torch.arange(self.target_model.sparse_position_vocab_size, device=self.device, dtype=torch.long), torch.tensor([self.target_model.sparse_eos_token_id], device=self.device, dtype=torch.long)))
        feature_start = 0 if self.target_model.sparse_shared_vocab else self.target_model.sparse_feature_token_offset
        self.feature_token_ids = torch.arange(feature_start, feature_start + self.target_model.sparse_feature_vocab_size, device=self.device, dtype=torch.long)

    @staticmethod
    def _compact_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return logits.index_select(-1, token_ids)

    def _validate_models(self) -> None:
        target = self.target
        draft = self.draft

        if target.gpt_config.dense_chunks or draft.gpt_config.dense_chunks:
            raise NotImplementedError("Position-selective decoding is sparse-only")

        for name, model in (("target", target), ("draft", draft)):
            cfg = model.gpt_config
            if cfg.sparse_tokens_per_latent != 2:
                raise NotImplementedError(
                    f"{name}: requires sparse_tokens_per_latent=2 (one P + one F), "
                    f"got {cfg.sparse_tokens_per_latent}"
                )
            if cfg.sparse_num_position_tokens != 1:
                raise NotImplementedError(
                    f"{name}: requires one Position token per latent"
                )

        compatibility_attrs = (
            "position_vocab_size",
            "feature_vocab_size",
            "feature_token_offset",
            "eos_token_id",
            "pad_token_id",
            "vocab_size",
        )
        for attr in compatibility_attrs:
            tv = getattr(target, attr)
            dv = getattr(draft, attr)
            if tv != dv:
                raise ValueError(f"target/draft mismatch for {attr}: {tv} != {dv}")

        cfg_attrs = (
            "sparse_tokens_per_latent",
            "sparse_num_position_tokens",
            "sparse_position_vocab_size",
            "sparse_feature_vocab_size",
            "sparse_feature_token_offset",
            "sparse_shared_vocab",
            "sparse_position_hard_constraints",
            "rope_basis",
        )
        for attr in cfg_attrs:
            tv = getattr(target.gpt_config, attr)
            dv = getattr(draft.gpt_config, attr)
            if tv != dv:
                raise ValueError(f"target/draft GPTConfig mismatch for {attr}: {tv} != {dv}")

        target_device = target.transformer.get_device()
        draft_device = draft.transformer.get_device()
        if target_device != draft_device:
            raise ValueError(
                f"target and draft must be on the same device: {target_device} != {draft_device}"
            )

    def _params(
        self,
        generated_index: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        resolver: Optional[SamplingParamsResolver],
    ) -> Tuple[float, Optional[int], Optional[float]]:
        if resolver is None:
            return temperature, top_k, top_p
        return resolver(generated_index)

    def _forward(
        self,
        model,
        ids: torch.Tensor,
        cache: KVCache,
        *,
        kind: str,
        stats: PositionSelectiveStats,
    ) -> torch.Tensor:
        context_before = cache.get_pos()
        sparse_logit_types = {
            "draft_prefill_1": ("position",),
        "draft_feature_probe": ("feature",),
            "target_verify_2": ("position", "feature"),
            "target_correction_1": ("feature",),
            "draft_update_2": ("feature", "position"),
        }.get(kind)
        if not self.measure_latency:
            return model.forward(ids, kv_cache=cache, logit_types=sparse_logit_types)

        if ids.device.type == "cuda":
            torch.cuda.synchronize(ids.device)
        start = perf_counter()
        logits = model.forward(ids, kv_cache=cache, logit_types=sparse_logit_types)
        if ids.device.type == "cuda":
            torch.cuda.synchronize(ids.device)
        elapsed_ms = (perf_counter() - start) * 1000.0
        stats.timing.append(
            TimingSample(
                kind=kind,
                context_tokens=context_before,
                input_tokens=int(ids.shape[1]),
                milliseconds=float(elapsed_ms),
            )
        )
        return logits

    @torch.inference_mode()
    def generate(
        self,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: int = 42,
        feature_shadow: bool = False,
        stop_on_eos: bool = True,
        sampling_params_resolver: Optional[SamplingParamsResolver] = None,
        draft_proposer: Optional[DraftProposal] = None,
    ) -> PositionSelectiveResult:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not stop_on_eos:
            raise NotImplementedError(
                "sparse Position-selective decoding currently requires stop_on_eos=True"
            )
        if max_tokens > min(
            self.target.gpt_config.sequence_len,
            self.draft.gpt_config.sequence_len,
        ):
            raise ValueError("max_tokens exceeds target/draft context length")

        stats = PositionSelectiveStats()
        proposer = draft_proposer if draft_proposer is not None else self.draft_proposer
        if proposer is not None:
            reset = getattr(proposer, "reset", None)
            if reset is not None:
                reset()
        proposer_uses_draft_forward = (
            proposer is None or bool(getattr(proposer, "requires_draft_forward", True))
        )
        rng = torch.Generator(device=self.device)
        rng.manual_seed(int(seed))
        # Keep shadow acceptance sampling independent of the committed stream.
        shadow_rng = torch.Generator(device=self.device)
        shadow_rng.manual_seed(int(seed) + 1_000_003)

        target_cache = make_kv_cache(self.target_model, max_tokens=max_tokens)
        draft_cache = make_kv_cache(self.draft_model, max_tokens=max_tokens)

        eos = int(self.target.eos_token_id)
        bos = eos
        output: List[int] = []

        # Draft prefill: neural D8 computes q(P_1 | BOS); a table proposer
        # only advances the bookkeeping cache and supplies q(P) itself.
        bos_ids = torch.tensor([[bos]], device=self.device, dtype=torch.long)
        if proposer_uses_draft_forward:
            draft_logits = self._forward(
                self.draft_model,
                bos_ids,
                draft_cache,
                kind="draft_prefill_1",
                stats=stats,
            )
            q_position_logits = draft_logits[:, -1, :]
        else:
            draft_cache.store_tokens(bos_ids, draft_cache.get_pos())
            draft_cache.advance(int(bos_ids.shape[1]))
            q_position_logits = None

        while len(output) < max_tokens:
            position_index = len(output)
            pos_temp, pos_top_k, pos_top_p = self._params(
                position_index,
                temperature,
                top_k,
                top_p,
                sampling_params_resolver,
            )
            draft_position_post_start = perf_counter()
            proposal_state = None
            if proposer is None:
                q_position_probs = sampling_probs(
                    self._compact_logits(q_position_logits, self.position_token_ids),
                    temperature=pos_temp,
                    top_k=pos_top_k,
                    top_p=pos_top_p,
                )
                proposal_index = sample_from_probs(q_position_probs, rng)
                proposal = self.position_token_ids[proposal_index]
                proposal_id = int(proposal.item())
            else:
                position_history = tuple(
                    int(token)
                    for token in output[0::2]
                    if int(token) != eos
                    and 0 <= int(token) < int(self.target_model.sparse_position_vocab_size)
                )
                feature_token = int(output[-1]) if output and len(output) % 2 == 0 else None
                proposal_state = DraftProposalState(
                    position_history=position_history,
                    feature_token=feature_token,
                    q_position_logits=(
                        q_position_logits.detach() if q_position_logits is not None else None
                    ),
                    position_token_ids=self.position_token_ids,
                    eos_token_id=eos,
                    temperature=pos_temp,
                    top_k=pos_top_k,
                    top_p=pos_top_p,
                    rng=rng,
                    device=self.device,
                    step=position_index,
                    metadata={"target_cache_pos": target_cache.get_pos()},
                )
                proposal_result = proposer.propose_position(proposal_state)
                if proposal_result.metadata.get("cascade_needs_neural", False):
                    # The structured proposer intentionally keeps only a logical cache.
                    # Rebuild the neural cache once, then continue with normal D8 updates.
                    replay_cache = make_kv_cache(self.draft_model, max_tokens=max_tokens)
                    replay_ids = torch.tensor(
                        [[bos, *[int(token) for token in output]]],
                        device=self.device,
                        dtype=torch.long,
                    )
                    replay_logits = self._forward(
                        self.draft_model,
                        replay_ids,
                        replay_cache,
                        kind="draft_lazy_replay",
                        stats=stats,
                    )
                    draft_cache = replay_cache
                    q_position_logits = replay_logits[:, -1, :]
                    proposer_uses_draft_forward = True
                    proposal_state = replace(
                        proposal_state,
                        q_position_logits=q_position_logits.detach(),
                        metadata={**proposal_state.metadata, "cascade_retry": True},
                    )
                    proposal_result = proposer.propose_position(proposal_state)
                q_position_probs = proposal_result.probs
                proposal_index_value = proposal_result.index
                if isinstance(proposal_index_value, torch.Tensor):
                    proposal_index = proposal_index_value.to(
                        device=self.device, dtype=torch.long
                    ).reshape(1, 1)
                else:
                    proposal_index = torch.tensor(
                        [[int(proposal_index_value)]],
                        device=self.device,
                        dtype=torch.long,
                    )
                proposal_id = int(proposal_result.token_id)
            stats.record_stage("draft_position_sampling", (perf_counter() - draft_position_post_start) * 1000.0)

            # q_F is reused from the mandatory draft_update_2 forward below.
        # Steady state target cache trails committed history by one Feature:
            # cache ... P_{i-1}; feed [F_{i-1}, proposal P_i].
            # Round 1 uses [BOS, proposal P_1] from an empty target cache.
            target_start = target_cache.get_pos()
            previous_feature_or_bos = bos if not output else int(output[-1])
            if output and not (
                self.target.feature_token_offset
                <= previous_feature_or_bos
                < self.target.feature_token_offset + self.target.feature_vocab_size
            ):
                raise RuntimeError(
                    "selective-decoding invariant violated: history must end in Feature"
                )

            verify_ids = torch.tensor(
                [[previous_feature_or_bos, proposal_id]],
                device=self.device,
                dtype=torch.long,
            )
            verify_logits = self._forward(
                self.target_model,
                verify_ids,
                target_cache,
                kind="target_verify_2",
                stats=stats,
            )
            p_position_logits = verify_logits[:, 0, :]
            p_feature_if_accept_logits = verify_logits[:, 1, :]
            target_position_post_start = perf_counter()
            p_position_probs = sampling_probs(
                self._compact_logits(p_position_logits, self.position_token_ids),
                temperature=pos_temp,
                top_k=pos_top_k,
                top_p=pos_top_p,
            )

            committed_position_index, accepted, _, overlap_p = accept_or_correct_position_compact(
                proposal_index, p_position_probs, q_position_probs, rng
            )
            committed_position = self.position_token_ids[committed_position_index]
            committed_position_id = int(committed_position.item())
            stats.position_proposals += 1
            stats.position_overlap_sum += overlap_p
            stats.record_stage("target_position_postprocess", (perf_counter() - target_position_post_start) * 1000.0)
            if accepted:
                stats.position_accepts += 1
            else:
                stats.position_rejects += 1

            if proposer is not None and proposal_state is not None:
                if accepted:
                    proposer.commit(committed_position_id)
                else:
                    proposer.rewind(proposal_state)
                    proposer.commit(committed_position_id)

            output.append(committed_position_id)

            # EOS is a Position-side terminal event. No Feature follows it.
            if committed_position_id == eos:
                stats.terminated_on_eos = True
                break

            # Match baseline semantics if max_tokens ends after a Position.
            if len(output) >= max_tokens:
                break

            feature_index = len(output)
            feat_temp, feat_top_k, feat_top_p = self._params(
                feature_index,
                temperature,
                top_k,
                top_p,
                sampling_params_resolver,
            )

            if accepted:
                p_feature_logits = p_feature_if_accept_logits
            else:
                # Keep the correctly-consumed BOS/F_{i-1}, discard speculative P_i.
                cache_rewind_start = perf_counter()
                truncate_kv_cache(target_cache, target_start + 1)
                stats.record_stage("target_cache_rewind", (perf_counter() - cache_rewind_start) * 1000.0)

                # If residual correction selected EOS, terminate without Feature.
                # (Handled above, so this branch necessarily has a non-EOS Position.)
                correction_ids = torch.tensor(
                    [[committed_position_id]], device=self.device, dtype=torch.long
                )
                correction_logits = self._forward(
                    self.target_model,
                    correction_ids,
                    target_cache,
                    kind="target_correction_1",
                    stats=stats,
                )
                p_feature_logits = correction_logits[:, -1, :]

            target_feature_post_start = perf_counter()
            p_feature_probs = sampling_probs(
                self._compact_logits(p_feature_logits, self.feature_token_ids),
                temperature=feat_temp,
                top_k=feat_top_k,
                top_p=feat_top_p,
            )
            feature_index = sample_from_probs(p_feature_probs, rng)
            feature = self.feature_token_ids[feature_index]
            feature_id = int(feature.item())

            feature_start = int(self.target.feature_token_offset)
            feature_end = feature_start + int(self.target.feature_vocab_size)
            if not (feature_start <= feature_id < feature_end):
                raise RuntimeError(
                    f"target produced invalid Feature id {feature_id}; "
                    f"expected [{feature_start}, {feature_end})"
                )
            stats.record_stage("target_feature_sampling_postprocess", (perf_counter() - target_feature_post_start) * 1000.0)
            pair_commit_start = perf_counter()
            output.append(feature_id)
            stats.generated_pairs += 1

            # Synchronize the draft with the committed target-distributed pair.
            # The first output row gives q_F(F_i | H_i,P_i), useful for alpha_F
            # diagnostics; the second gives q_P(P_{i+1} | H_i,P_i,F_i).
            update_ids = torch.tensor(
                [[committed_position_id, feature_id]],
                device=self.device,
                dtype=torch.long,
            )
            stats.record_stage("pair_commit_prepare", (perf_counter() - pair_commit_start) * 1000.0)
            if proposer_uses_draft_forward:
                draft_update_logits = self._forward(
                    self.draft_model,
                    update_ids,
                    draft_cache,
                    kind="draft_update_2",
                    stats=stats,
                )
                q_feature_logits = draft_update_logits[:, 0, :]
                q_position_logits = draft_update_logits[:, 1, :]
            else:
                draft_cache.store_tokens(update_ids, draft_cache.get_pos())
                draft_cache.advance(int(update_ids.shape[1]))
                draft_update_logits = None
                q_feature_logits = torch.zeros(
                    (1, int(self.draft_model.config.vocab_size)),
                    device=self.device,
                    dtype=next(self.draft_model.parameters()).dtype,
                )
                q_position_logits = None
            # The first row of the mandatory Draft update is exactly q_F(F_i | H_i, P_i).
            # Reuse it for shadow joint acceptance without an extra Draft forward.
            if feature_shadow and accepted:
                feature_shadow_start = perf_counter()
                p_feature_shadow_probs = sampling_probs(
                    self._compact_logits(p_feature_if_accept_logits, self.feature_token_ids),
                    temperature=feat_temp,
                    top_k=feat_top_k,
                    top_p=feat_top_p,
                )
                q_feature_shadow_probs = sampling_probs(
                    self._compact_logits(q_feature_logits, self.feature_token_ids),
                    temperature=feat_temp,
                    top_k=feat_top_k,
                    top_p=feat_top_p,
                )
                feature_shadow_index = sample_from_probs(q_feature_shadow_probs, shadow_rng)
                (
                    _,
                    feature_shadow_accepted,
                    feature_shadow_accept_prob,
                    feature_shadow_overlap,
                ) = accept_or_correct_position_compact(
                    feature_shadow_index,
                    p_feature_shadow_probs,
                    q_feature_shadow_probs,
                    shadow_rng,
                )
                stats.feature_proposals += 1
                stats.feature_accept_prob_sum += float(feature_shadow_accept_prob)
                stats.feature_shadow_overlap_sum += float(feature_shadow_overlap)
                if feature_shadow_accepted:
                    stats.feature_accepts += 1
                    stats.pair_accepts += 1
                else:
                    stats.feature_rejects += 1
                stats.record_stage(
                    'feature_shadow_acceptance',
                    (perf_counter() - feature_shadow_start) * 1000.0,
                )

            draft_feature_diag_start = perf_counter()
            q_feature_probs = sampling_probs(
                self._compact_logits(q_feature_logits, self.feature_token_ids),
                temperature=feat_temp,
                top_k=feat_top_k,
                top_p=feat_top_p,
            )
            overlap_f = float(
                distribution_overlap(p_feature_probs, q_feature_probs)[0].item()
            )
            stats.feature_observations += 1
            stats.feature_overlap_sum += overlap_f
            stats.record_stage("draft_feature_diagnostic", (perf_counter() - draft_feature_diag_start) * 1000.0)

            cache_invariant_start = perf_counter()
            # Cache invariants after a committed (P_i,F_i) pair.
            expected_draft_pos = 1 + len(output)  # BOS + all committed tokens
            expected_target_pos = len(output)  # BOS + tokens through P_i, F_i trails
            if draft_cache.get_pos() != expected_draft_pos:
                raise RuntimeError(
                    f"draft cache invariant failed: {draft_cache.get_pos()} != {expected_draft_pos}"
                )
            if target_cache.get_pos() != expected_target_pos:
                raise RuntimeError(
                    f"target cache invariant failed: {target_cache.get_pos()} != {expected_target_pos}"
                )

            stats.record_stage("cache_invariant_checks", (perf_counter() - cache_invariant_start) * 1000.0)
        stats.generated_tokens = len(output)
        tokens = torch.tensor([output], device=self.device, dtype=torch.long)
        return PositionSelectiveResult(tokens=tokens, stats=stats)
