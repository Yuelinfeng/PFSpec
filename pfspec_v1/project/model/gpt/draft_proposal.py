"""Formal Position-side draft proposer interfaces."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
import torch

@dataclass(frozen=True)
class DraftProposalState:
    """Snapshot of the committed stream passed to a Position proposer."""
    position_history: tuple[int, ...]
    feature_token: Optional[int]
    q_position_logits: Optional[torch.Tensor]
    position_token_ids: torch.Tensor
    eos_token_id: int
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]
    rng: torch.Generator
    device: torch.device
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PositionProposal:
    token_id: int
    index: int
    probs: torch.Tensor
    source: str
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

class DraftProposal(Protocol):
    name: str
    def propose_position(self, state: DraftProposalState) -> PositionProposal: ...
    def commit(self, token: int) -> None: ...
    def rewind(self, state: DraftProposalState) -> None: ...
    def reset(self) -> None: ...

def _sampling_helpers():
    # Lazy import avoids a circular import while keeping the engine's exact
    # sampling implementation shared by every proposer.
    from model.gpt.position_selective_speculative import sampling_probs, sample_from_probs
    return sampling_probs, sample_from_probs

def _compact_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"q_position_logits must be rank 1/2, got {tuple(logits.shape)}")
    token_ids = token_ids.to(device=logits.device, dtype=torch.long)
    if logits.shape[-1] == token_ids.numel():
        return logits
    max_id = int(token_ids.max().item()) if token_ids.numel() else -1
    if logits.shape[-1] <= max_id:
        raise ValueError("q_position_logits does not contain Position support")
    return logits.index_select(-1, token_ids)

def _sample_position(logits: torch.Tensor, state: DraftProposalState) -> tuple[int, torch.Tensor]:
    sampling_probs, sample_from_probs = _sampling_helpers()
    probs = sampling_probs(
        _compact_logits(logits, state.position_token_ids),
        temperature=state.temperature, top_k=state.top_k, top_p=state.top_p,
    )
    if probs.ndim != 2 or probs.shape[0] != 1:
        raise ValueError(f"Position proposer expects one sample, got {tuple(probs.shape)}")
    if not torch.isfinite(probs).all() or (probs < 0).any() or float(probs.sum().item()) <= 0:
        raise ValueError("Position proposal distribution is not finite/non-negative")
    probs = probs / probs.sum(dim=-1, keepdim=True)
    index = int(sample_from_probs(probs, state.rng).reshape(-1)[0].item())
    return index, probs

class NeuralDraftProposal:
    """Adapter for canonical D8 neural draft Position logits."""
    name = "neural_d8"
    requires_draft_forward = True
    def __init__(self, *, name: str = "neural_d8") -> None:
        self.name = str(name)
    def propose_position(self, state: DraftProposalState) -> PositionProposal:
        if state.q_position_logits is None:
            raise ValueError("NeuralDraftProposal requires q_position_logits")
        index, probs = _sample_position(state.q_position_logits, state)
        token_id = int(state.position_token_ids[index].item())
        return PositionProposal(token_id, index, probs, self.name, float(probs[0, index].item()))
    def commit(self, token: int) -> None:
        del token
    def rewind(self, state: DraftProposalState) -> None:
        del state
    def reset(self) -> None:
        return None

class StructuredPositionProposal:
    """Formal wrapper around StructuredPositionGapDraft."""
    requires_draft_forward = False
    def __init__(self, adapter: Any, *, position_token_ids: torch.Tensor,
                 eos_token_id: int, name: str = "structured_position") -> None:
        self.adapter = adapter
        self.position_token_ids = position_token_ids.detach().clone()
        self.eos_token_id = int(eos_token_id)
        self.name = str(name)
        self._position_history: list[int] = []
    def _history_for(self, state: DraftProposalState) -> tuple[int, ...]:
        if tuple(self._position_history) != tuple(state.position_history):
            self._position_history = list(state.position_history)
        return tuple(self._position_history)
    def propose_position(self, state: DraftProposalState) -> PositionProposal:
        history = self._history_for(state)
        probs = self.adapter.query(history, device=state.device)
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        expected = (1, self.position_token_ids.numel())
        if probs.ndim != 2 or tuple(probs.shape) != expected:
            raise ValueError(f"Structured adapter returned {tuple(probs.shape)}, expected {expected}")
        index, sampled = _sample_position(torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny)), state)
        token_id = int(state.position_token_ids[index].item())
        return PositionProposal(token_id, index, sampled, self.name,
                                float(sampled[0, index].item()),
                                {"lookup_history_len": len(history)})
    def commit(self, token: int) -> None:
        token = int(token)
        if token != self.eos_token_id:
            self._position_history.append(token)
    def rewind(self, state: DraftProposalState) -> None:
        self._position_history = list(state.position_history)
    def reset(self) -> None:
        self._position_history.clear()
    def summary(self) -> dict[str, Any]:
        out = dict(self.adapter.summary()) if hasattr(self.adapter, "summary") else {}
        out.update({"proposal_interface": self.name,
                    "position_history_len": len(self._position_history),
                    "eos_token_id": self.eos_token_id})
        return out


class CascadePositionProposal:
    """Confidence-gated structured proposer with canonical D8 fallback."""

    # Start table-only.  The engine may rebuild the neural cache lazily when
    # a low-confidence proposal requests the D8 fallback.
    requires_draft_forward = False

    def __init__(
        self,
        structured: StructuredPositionProposal,
        neural: Optional[NeuralDraftProposal] = None,
        *,
        metric: str = "top1",
        threshold: float = 0.5,
        name: str = "cascade_position",
    ) -> None:
        metric = str(metric).lower()
        if metric not in {"top1", "entropy"}:
            raise ValueError(f"unsupported cascade metric: {metric}")
        if not torch.isfinite(torch.tensor(float(threshold))):
            raise ValueError("cascade threshold must be finite")
        self.structured = structured
        self.neural = neural if neural is not None else NeuralDraftProposal()
        self.metric = metric
        self.threshold = float(threshold)
        self.name = str(name)
        self._total = 0
        self._structured = 0
        self._fallback = 0
        self._metric_sum = 0.0
        self._metric_min = None
        self._metric_max = None

    @staticmethod
    def _confidence(proposal: PositionProposal) -> tuple[float, float]:
        probs = proposal.probs
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        probs = probs.detach().float().clamp_min(0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
        top1 = float(probs.max(dim=-1).values.item())
        entropy = float((-(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)).item())
        return top1, entropy

    def propose_position(self, state: DraftProposalState) -> PositionProposal:
        structured_proposal = self.structured.propose_position(state)
        top1, entropy = self._confidence(structured_proposal)
        score = top1 if self.metric == "top1" else entropy
        use_structured = top1 >= self.threshold if self.metric == "top1" else entropy <= self.threshold
        if use_structured:
            self._total += 1
            self._metric_sum += score
            self._metric_min = score if self._metric_min is None else min(self._metric_min, score)
            self._metric_max = score if self._metric_max is None else max(self._metric_max, score)
            self._structured += 1
            structured_proposal.metadata.update({
                "cascade_source": "structured",
                "cascade_metric": self.metric,
                "cascade_score": score,
                "cascade_top1": top1,
                "cascade_entropy": entropy,
            })
            return structured_proposal
        # A table-only engine has no D8 logits at this point.  Return an
        # explicit retry marker; the engine rebuilds the D8 cache from the
        # committed prefix, then calls this method again with q_position_logits.
        if state.q_position_logits is None:
            return PositionProposal(
                token_id=int(structured_proposal.token_id),
                index=int(structured_proposal.index),
                probs=structured_proposal.probs,
                source=self.name,
                confidence=float(score),
                metadata={
                    "cascade_needs_neural": True,
                    "cascade_deferred": True,
                    "cascade_metric": self.metric,
                    "cascade_score": score,
                    "cascade_top1": top1,
                    "cascade_entropy": entropy,
                },
            )
        self._total += 1
        self._metric_sum += score
        self._metric_min = score if self._metric_min is None else min(self._metric_min, score)
        self._metric_max = score if self._metric_max is None else max(self._metric_max, score)
        self._fallback += 1
        fallback = self.neural.propose_position(state)
        fallback.metadata.update({
            "cascade_source": "neural_d8",
            "cascade_metric": self.metric,
            "cascade_score": score,
            "cascade_top1": top1,
            "cascade_entropy": entropy,
        })
        return fallback

    def commit(self, token: int) -> None:
        self.structured.commit(token)
        self.neural.commit(token)

    def rewind(self, state: DraftProposalState) -> None:
        self.structured.rewind(state)
        self.neural.rewind(state)

    def reset(self) -> None:
        self.structured.reset()
        self.neural.reset()
        self._total = 0
        self._structured = 0
        self._fallback = 0
        self._metric_sum = 0.0
        self._metric_min = None
        self._metric_max = None

    def summary(self) -> dict[str, Any]:
        structured_summary = self.structured.summary() if hasattr(self.structured, "summary") else {}
        return {
            "proposal_interface": self.name,
            "metric": self.metric,
            "threshold": self.threshold,
            "total": self._total,
            "structured": self._structured,
            "neural_fallback": self._fallback,
            "structured_fraction": self._structured / self._total if self._total else 0.0,
            "score_mean": self._metric_sum / self._total if self._total else 0.0,
            "score_min": self._metric_min,
            "score_max": self._metric_max,
            "structured_summary": structured_summary,
        }
