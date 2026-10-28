from pathlib import Path

path = Path("model/gpt/position_selective_speculative.py")


def replace_once(old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, found {count}: {old[:100]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
'''    corrected = sample_from_probs(residual, rng)
    return corrected, False, accept_prob, overlap


def make_kv_cache''',
'''    corrected = sample_from_probs(residual, rng)
    return corrected, False, accept_prob, overlap


def accept_or_correct_position_compact(
    proposal_index: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, bool, float, float]:
    """Compact-vocabulary version of position speculative sampling."""

    if proposal_index.shape != (1, 1):
        raise ValueError("current implementation requires proposal shape (1,1)")
    if p.shape != q.shape or p.shape[0] != 1:
        raise ValueError("current implementation requires p,q shape (1,V)")

    index = int(proposal_index.item())
    qx = float(q[0, index].item())
    px = float(p[0, index].item())
    if qx <= 0.0:
        raise RuntimeError("proposal has zero probability under q; sampler invariant violated")

    accept_prob = min(1.0, px / qx)
    overlap = float(distribution_overlap(p, q)[0].item())
    if accept_prob >= 1.0:
        return proposal_index, True, accept_prob, overlap
    if accept_prob <= 0.0:
        accepted = False
    else:
        u = float(torch.rand((), device=p.device, generator=rng).item())
        accepted = u < accept_prob
    if accepted:
        return proposal_index, True, accept_prob, overlap

    residual = torch.clamp(p - q, min=0.0)
    mass = residual.sum(dim=-1, keepdim=True)
    if float(mass.item()) <= 1e-12:
        residual = p
    else:
        residual = residual / mass
    corrected = sample_from_probs(residual, rng)
    return corrected, False, accept_prob, overlap


def make_kv_cache''',
)

replace_once(
'''        self.target_model = self.target.transformer
        self.draft_model = self.draft.transformer
        self.device = self.target_model.get_device()

    def _validate_models''',
'''        self.target_model = self.target.transformer
        self.draft_model = self.draft.transformer
        self.device = self.target_model.get_device()
        self.position_token_ids = torch.cat(
            (
                torch.arange(
                    self.target_model.sparse_position_vocab_size,
                    device=self.device,
                    dtype=torch.long,
                ),
                torch.tensor(
                    [self.target_model.sparse_eos_token_id],
                    device=self.device,
                    dtype=torch.long,
                ),
            )
        )
        feature_start = (
            0
            if self.target_model.sparse_shared_vocab
            else self.target_model.sparse_feature_token_offset
        )
        self.feature_token_ids = torch.arange(
            feature_start,
            feature_start + self.target_model.sparse_feature_vocab_size,
            device=self.device,
            dtype=torch.long,
        )

    @staticmethod
    def _compact_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return logits.index_select(-1, token_ids)

    def _validate_models''',
)

replace_once(
'''            q_position_probs = sampling_probs(
                q_position_logits,
                temperature=pos_temp,
                top_k=pos_top_k,
                top_p=pos_top_p,
            )
            proposal = sample_from_probs(q_position_probs, rng)
            proposal_id = int(proposal.item())''',
'''            q_position_probs = sampling_probs(
                self._compact_logits(q_position_logits, self.position_token_ids),
                temperature=pos_temp,
                top_k=pos_top_k,
                top_p=pos_top_p,
            )
            proposal_index = sample_from_probs(q_position_probs, rng)
            proposal = self.position_token_ids[proposal_index]
            proposal_id = int(proposal.item())''',
)

replace_once(
'''            p_position_probs = sampling_probs(
                p_position_logits,
                temperature=pos_temp,
                top_k=pos_top_k,
                top_p=pos_top_p,
            )

            committed_position, accepted, _, overlap_p = accept_or_correct_position(
                proposal,
                p_position_probs,
                q_position_probs,
                rng,
            )
            committed_position_id = int(committed_position.item())''',
'''            p_position_probs = sampling_probs(
                self._compact_logits(p_position_logits, self.position_token_ids),
                temperature=pos_temp,
                top_k=pos_top_k,
                top_p=pos_top_p,
            )

            committed_position_index, accepted, _, overlap_p = accept_or_correct_position_compact(
                proposal_index,
                p_position_probs,
                q_position_probs,
                rng,
            )
            committed_position = self.position_token_ids[committed_position_index]
            committed_position_id = int(committed_position.item())''',
)

replace_once(
'''            p_feature_probs = sampling_probs(
                p_feature_logits,
                temperature=feat_temp,
                top_k=feat_top_k,
                top_p=feat_top_p,
            )
            feature = sample_from_probs(p_feature_probs, rng)
            feature_id = int(feature.item())''',
'''            p_feature_probs = sampling_probs(
                self._compact_logits(p_feature_logits, self.feature_token_ids),
                temperature=feat_temp,
                top_k=feat_top_k,
                top_p=feat_top_p,
            )
            feature_index = sample_from_probs(p_feature_probs, rng)
            feature = self.feature_token_ids[feature_index]
            feature_id = int(feature.item())''',
)

replace_once(
'''            q_feature_probs = sampling_probs(
                q_feature_logits,
                temperature=feat_temp,
                top_k=feat_top_k,
                top_p=feat_top_p,
            )''',
'''            q_feature_probs = sampling_probs(
                self._compact_logits(q_feature_logits, self.feature_token_ids),
                temperature=feat_temp,
                top_k=feat_top_k,
                top_p=feat_top_p,
            )''',
)

print("compact sampling patch applied")

