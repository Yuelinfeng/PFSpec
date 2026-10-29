from __future__ import annotations

from typing import Optional
import torch

from model.gpt.position_selective_speculative import (
    PositionSelectiveResult,
    PositionSelectiveStats,
    accept_or_correct_position_compact,
    make_kv_cache,
    sample_from_probs,
    sampling_probs,
    truncate_kv_cache,
)


@torch.inference_mode()
def generate_joint_feature(
    engine,
    *,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: int = 42,
    stop_on_eos: bool = True,
) -> PositionSelectiveResult:
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not stop_on_eos:
        raise NotImplementedError("joint sparse decoding requires stop_on_eos=True")
    if max_tokens > min(engine.target.gpt_config.sequence_len, engine.draft.gpt_config.sequence_len):
        raise ValueError("max_tokens exceeds model context length")

    stats = PositionSelectiveStats()
    rng = torch.Generator(device=engine.device)
    rng.manual_seed(int(seed))
    target_cache = make_kv_cache(engine.target_model, max_tokens=max_tokens)
    draft_cache = make_kv_cache(engine.draft_model, max_tokens=max_tokens)
    eos = int(engine.target.eos_token_id)
    bos = eos
    output = []

    bos_ids = torch.tensor([[bos]], device=engine.device, dtype=torch.long)
    draft_logits = engine._forward(engine.draft_model, bos_ids, draft_cache, kind="draft_prefill_1", stats=stats)
    q_position_logits = draft_logits[:, -1, :]

    while len(output) < max_tokens:
        position_index = len(output)
        pos_temp, pos_top_k, pos_top_p = engine._params(position_index, temperature, top_k, top_p, None)
        q_position_probs = sampling_probs(
            engine._compact_logits(q_position_logits, engine.position_token_ids),
            temperature=pos_temp, top_k=pos_top_k, top_p=pos_top_p,
        )
        proposal_index = sample_from_probs(q_position_probs, rng)
        proposal_id = int(engine.position_token_ids[proposal_index].item())
        draft_start = int(draft_cache.get_pos())

        q_feature_probs = None
        feature_proposal_index = None
        if proposal_id != eos:
            proposal_ids = torch.tensor([[proposal_id]], device=engine.device, dtype=torch.long)
            q_feature_all = engine._forward(
                engine.draft_model, proposal_ids, draft_cache,
                kind="draft_feature_probe", stats=stats,
            )
            q_feature_logits = q_feature_all[:, -1, :]
            feature_index = position_index + 1
            feat_temp, feat_top_k, feat_top_p = engine._params(feature_index, temperature, top_k, top_p, None)
            q_feature_probs = sampling_probs(
                engine._compact_logits(q_feature_logits, engine.feature_token_ids),
                temperature=feat_temp, top_k=feat_top_k, top_p=feat_top_p,
            )
            feature_proposal_index = sample_from_probs(q_feature_probs, rng)

        target_start = int(target_cache.get_pos())
        previous_feature = bos if not output else int(output[-1])
        verify_ids = torch.tensor([[previous_feature, proposal_id]], device=engine.device, dtype=torch.long)
        verify_logits = engine._forward(
            engine.target_model, verify_ids, target_cache,
            kind="target_verify_2", stats=stats,
        )
        p_position_logits = verify_logits[:, 0, :]
        p_feature_logits = verify_logits[:, 1, :]
        p_position_probs = sampling_probs(
            engine._compact_logits(p_position_logits, engine.position_token_ids),
            temperature=pos_temp, top_k=pos_top_k, top_p=pos_top_p,
        )
        committed_position_index, position_accepted, position_accept_prob, overlap_p = (
            accept_or_correct_position_compact(proposal_index, p_position_probs, q_position_probs, rng)
        )
        committed_position_id = int(engine.position_token_ids[committed_position_index].item())
        stats.position_proposals += 1
        stats.position_overlap_sum += float(overlap_p)
        if position_accepted:
            stats.position_accepts += 1
        else:
            stats.position_rejects += 1
            truncate_kv_cache(target_cache, target_start + 1)
            truncate_kv_cache(draft_cache, draft_start)
            correction_ids = torch.tensor([[committed_position_id]], device=engine.device, dtype=torch.long)
            correction_logits = engine._forward(
                engine.target_model, correction_ids, target_cache,
                kind="target_correction_1", stats=stats,
            )
            p_feature_logits = correction_logits[:, -1, :]

        output.append(committed_position_id)
        if committed_position_id == eos:
            stats.terminated_on_eos = True
            break
        if len(output) >= max_tokens:
            break

        feature_index = len(output)
        feat_temp, feat_top_k, feat_top_p = engine._params(feature_index, temperature, top_k, top_p, None)
        p_feature_probs = sampling_probs(
            engine._compact_logits(p_feature_logits, engine.feature_token_ids),
            temperature=feat_temp, top_k=feat_top_k, top_p=feat_top_p,
        )
        if position_accepted:
            if q_feature_probs is None or feature_proposal_index is None:
                raise RuntimeError("missing Draft Feature proposal on accepted Position")
            committed_feature_index, feature_accepted, feature_accept_prob, overlap_f = (
                accept_or_correct_position_compact(
                    feature_proposal_index, p_feature_probs, q_feature_probs, rng
                )
            )
            stats.feature_observations += 1
            stats.feature_proposals += 1
            stats.feature_accept_prob_sum += float(feature_accept_prob)
            stats.feature_shadow_overlap_sum += float(overlap_f)
            if feature_accepted:
                stats.feature_accepts += 1
            else:
                stats.feature_rejects += 1
            if feature_accepted:
                stats.pair_accepts += 1
            committed_feature_id = int(engine.feature_token_ids[committed_feature_index].item())
        else:
            committed_feature_index = sample_from_probs(p_feature_probs, rng)
            committed_feature_id = int(engine.feature_token_ids[committed_feature_index].item())

        output.append(committed_feature_id)
        stats.generated_pairs += 1

        if position_accepted:
            update_ids = torch.tensor([[committed_feature_id]], device=engine.device, dtype=torch.long)
            update_logits = engine._forward(
                engine.draft_model, update_ids, draft_cache,
                kind="draft_prefill_1", stats=stats,
            )
        else:
            update_ids = torch.tensor([[committed_position_id, committed_feature_id]], device=engine.device, dtype=torch.long)
            update_logits = engine._forward(
                engine.draft_model, update_ids, draft_cache,
                kind="draft_update_2", stats=stats,
            )
        q_position_logits = update_logits[:, -1, :]

        expected_draft_pos = 1 + len(output)
        expected_target_pos = len(output)
        if int(draft_cache.get_pos()) != expected_draft_pos:
            raise RuntimeError("joint draft cache invariant failed: %d != %d" % (draft_cache.get_pos(), expected_draft_pos))
        if int(target_cache.get_pos()) != expected_target_pos:
            raise RuntimeError("joint target cache invariant failed: %d != %d" % (target_cache.get_pos(), expected_target_pos))

    stats.generated_tokens = len(output)
    tokens = torch.tensor([output], device=engine.device, dtype=torch.long)
    return PositionSelectiveResult(tokens=tokens, stats=stats)

