#!/usr/bin/env python3
"""Prepare committed-history features and the Target Feature embedding on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POSITION_VOCAB = 8000
FEATURE_VOCAB = 4096
FEATURE_OFFSET = POSITION_VOCAB
EOS_ID = POSITION_VOCAB + FEATURE_VOCAB
SIDE = 20
EXPECTED_ROWS = 480
EMBEDDING_KEY = "gpt._orig_mod.transformer.transformer.wte.weight"
DIRECTIONS = (
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
)


def decode_position(position_id: int) -> tuple[int, int, int]:
    position_id = int(position_id)
    if not 0 <= position_id < POSITION_VOCAB:
        raise ValueError(f"Position ID outside 0..{POSITION_VOCAB - 1}: {position_id}")
    return position_id // 400, (position_id // 20) % 20, position_id % 20


def encode_position(x: int, y: int, z: int) -> int:
    if not (0 <= x < SIDE and 0 <= y < SIDE and 0 <= z < SIDE):
        raise ValueError(f"Position coordinates outside 0..{SIDE - 1}: {(x, y, z)}")
    return x * 400 + y * 20 + z


def committed_pairs(tokens: list[int]) -> list[tuple[int, int]]:
    ids = [int(token) for token in tokens]
    if not ids or ids[0] != EOS_ID:
        raise ValueError("context_prefix must begin with BOS id 12096")
    body = ids[1:]
    if EOS_ID in body:
        eos_at = body.index(EOS_ID)
        if eos_at != len(body) - 1:
            raise ValueError("EOS may appear only after the final complete PF pair")
        body = body[:-1]
    if len(body) % 2:
        raise ValueError("context_prefix must end with a complete PF pair")

    pairs = []
    seen_positions = set()
    for index in range(0, len(body), 2):
        position_id, feature_id = body[index:index + 2]
        decode_position(position_id)
        if not FEATURE_OFFSET <= feature_id < EOS_ID:
            raise ValueError(f"Feature ID outside {FEATURE_OFFSET}..{EOS_ID - 1}: {feature_id}")
        if position_id in seen_positions:
            raise ValueError(f"Repeated committed Position ID: {position_id}")
        seen_positions.add(position_id)
        pairs.append((position_id, feature_id))
    if not pairs:
        raise ValueError("context_prefix must contain at least one committed PF pair")
    return pairs


def _empty_slot() -> dict:
    return dict(position_id=-1, feature_local_id=-1, relative_xyz=[0, 0, 0],
                age_pf_pairs=-1, mask=False)


def derive_context_features(tokens: list[int], candidate_position: int) -> dict:
    """Use only committed PF history and the candidate P coordinate."""
    pairs = committed_pairs(tokens)
    candidate_position = int(candidate_position)
    candidate_xyz = decode_position(candidate_position)
    if any(position_id == candidate_position for position_id, _ in pairs):
        raise ValueError("candidate P is already present in committed history")
    latest = len(pairs) - 1
    by_position = {
        position_id: (feature_id - FEATURE_OFFSET, index)
        for index, (position_id, feature_id) in enumerate(pairs)
    }

    neighbors = []
    for dx, dy, dz in DIRECTIONS:
        nx, ny, nz = candidate_xyz[0] + dx, candidate_xyz[1] + dy, candidate_xyz[2] + dz
        if not (0 <= nx < SIDE and 0 <= ny < SIDE and 0 <= nz < SIDE):
            neighbors.append(_empty_slot())
            continue
        neighbor_position = encode_position(nx, ny, nz)
        entry = by_position.get(neighbor_position)
        if entry is None:
            neighbors.append(_empty_slot())
            continue
        feature_local_id, pair_index = entry
        neighbors.append(dict(
            position_id=neighbor_position,
            feature_local_id=feature_local_id,
            relative_xyz=[nx - candidate_xyz[0], ny - candidate_xyz[1], nz - candidate_xyz[2]],
            age_pf_pairs=latest - pair_index,
            mask=True,
        ))

    recent = []
    first_recent = max(0, len(pairs) - 6)
    for pair_index in range(first_recent, len(pairs)):
        position_id, feature_id = pairs[pair_index]
        xyz = decode_position(position_id)
        recent.append(dict(
            position_id=position_id,
            feature_local_id=feature_id - FEATURE_OFFSET,
            relative_xyz=[xyz[0] - candidate_xyz[0], xyz[1] - candidate_xyz[1],
                          xyz[2] - candidate_xyz[2]],
            age_pf_pairs=latest - pair_index,
            mask=True,
        ))
    recent = [_empty_slot() for _ in range(6 - len(recent))] + recent
    return dict(candidate_p_xyz=list(candidate_xyz), history_pf_pairs=len(pairs),
                history_position_ids=[position_id for position_id, _ in pairs],
                history_feature_global_ids=[feature_id for _, feature_id in pairs],
                neighbors=neighbors, recent_features=recent)


def _read_contexts(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def _rows_to_arrays(rows: list[dict], np) -> dict:
    neighbors = [row["_features"]["neighbors"] for row in rows]
    recent = [row["_features"]["recent_features"] for row in rows]
    history_lengths = [row["_features"]["history_pf_pairs"] for row in rows]
    history_offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    history_offsets[1:] = np.cumsum(history_lengths, dtype=np.int64)
    history_positions = [position for row in rows
                         for position in row["_features"]["history_position_ids"]]
    history_features = [feature for row in rows
                        for feature in row["_features"]["history_feature_global_ids"]]
    return dict(
        vector_index=np.asarray([row["vector_index"] for row in rows], dtype=np.int32),
        candidate_p_xyz=np.asarray([row["_features"]["candidate_p_xyz"] for row in rows],
                                   dtype=np.uint8),
        history_pf_pairs=np.asarray([row["_features"]["history_pf_pairs"] for row in rows],
                                    dtype=np.int32),
        history_pair_offsets=history_offsets,
        history_position_ids_flat=np.asarray(history_positions, dtype=np.int32),
        history_feature_global_ids_flat=np.asarray(history_features, dtype=np.int32),
        neighbor_position_ids=np.asarray(
            [[slot["position_id"] for slot in row] for row in neighbors], dtype=np.int32),
        neighbor_feature_local_ids=np.asarray(
            [[slot["feature_local_id"] for slot in row] for row in neighbors], dtype=np.int16),
        neighbor_relative_xyz=np.asarray(
            [[slot["relative_xyz"] for slot in row] for row in neighbors], dtype=np.int8),
        neighbor_age_pf_pairs=np.asarray(
            [[slot["age_pf_pairs"] for slot in row] for row in neighbors], dtype=np.int32),
        neighbor_mask=np.asarray([[slot["mask"] for slot in row] for row in neighbors], dtype=np.bool_),
        recent_position_ids=np.asarray(
            [[slot["position_id"] for slot in row] for row in recent], dtype=np.int32),
        recent_feature_local_ids=np.asarray(
            [[slot["feature_local_id"] for slot in row] for row in recent], dtype=np.int16),
        recent_relative_xyz=np.asarray(
            [[slot["relative_xyz"] for slot in row] for row in recent], dtype=np.int8),
        recent_age_pf_pairs=np.asarray(
            [[slot["age_pf_pairs"] for slot in row] for row in recent], dtype=np.int32),
        recent_mask=np.asarray([[slot["mask"] for slot in row] for row in recent], dtype=np.bool_),
        p1_id=np.asarray([row["p1_id"] for row in rows], dtype=np.int16),
        p1_accept_probability=np.asarray(
            [row["p1_accept_probability"] for row in rows], dtype=np.float32),
        p1_accepted_actual=np.asarray(
            [row["p1_accepted_actual"] for row in rows], dtype=np.bool_),
    )


def _self_test() -> dict:
    bos = EOS_ID
    feature = FEATURE_OFFSET
    history = [bos, 400, feature + 7, 1, feature + 9]
    result = derive_context_features(history, 0)
    slots = result["neighbors"]
    assert slots[0]["mask"] is False
    assert slots[1]["feature_local_id"] == 7
    assert slots[1]["relative_xyz"] == [1, 0, 0]
    assert slots[1]["age_pf_pairs"] == 1
    assert slots[5]["feature_local_id"] == 9
    assert slots[5]["relative_xyz"] == [0, 0, 1]
    assert slots[5]["age_pf_pairs"] == 0
    recent = result["recent_features"]
    assert [slot["feature_local_id"] for slot in recent] == [-1, -1, -1, -1, 7, 9]
    assert [slot["age_pf_pairs"] for slot in recent] == [-1, -1, -1, -1, 1, 0]
    assert result["candidate_p_xyz"] == [0, 0, 0]

    wrapped_history = [bos, 7600, feature + 11, 380, feature + 12, 19, feature + 13]
    wrapped = derive_context_features(wrapped_history, 0)
    assert not any(slot["mask"] for slot in wrapped["neighbors"])
    assert wrapped["history_position_ids"] == [7600, 380, 19]
    assert wrapped["history_feature_global_ids"] == [feature + 11, feature + 12, feature + 13]
    ended = derive_context_features([bos, 400, feature + 7, EOS_ID], 0)
    assert ended["history_pf_pairs"] == 1
    for invalid in ([bos, 0], [bos, EOS_ID, 400], [bos, 400, EOS_ID, 1]):
        try:
            committed_pairs(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid committed history was accepted: {invalid}")
    return dict(passed=True, direction_order="-x,+x,-y,+y,-z,+z",
                recent_order="oldest-to-newest with left padding", wraparound="absent",
                incomplete_pair="rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="Original Target Lightning checkpoint")
    parser.add_argument("--collection", type=Path,
                        help="R1 collection directory containing contexts.jsonl and vectors.npz")
    parser.add_argument("--output", type=Path,
                        help="New output directory; it must not exist")
    parser.add_argument("--self-test", action="store_true",
                        help="Run stdlib-only position and history examples")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), sort_keys=True))
        return 0
    if args.checkpoint is None or args.collection is None or args.output is None:
        parser.error("--checkpoint, --collection, and --output are required")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    collection = args.collection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("--output must name a new directory: " + str(output))
    for source in (checkpoint_path, collection / "contexts.jsonl", collection / "vectors.npz"):
        if not source.exists():
            parser.error("input does not exist: " + str(source))

    import numpy as np
    import torch

    contexts = _read_contexts(collection / "contexts.jsonl")
    if len(contexts) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} contexts, got {len(contexts)}")
    indices = [int(row["vector_index"]) for row in contexts]
    if indices != list(range(len(contexts))):
        raise ValueError("context vector_index must match original JSONL and vector array order")

    with np.load(collection / "vectors.npz", allow_pickle=False) as archive:
        required = ("p_f", "q1", "raw_logits")
        if set(required) - set(archive.files):
            raise ValueError("vectors.npz is missing one of p_f, q1, raw_logits")
        vectors = {name: np.array(archive[name], copy=True) for name in required}
    expected_shape = (len(contexts), FEATURE_VOCAB)
    for name, array in vectors.items():
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape {expected_shape}; got {array.shape}")

    # The existing trusted Lightning file contains OmegaConf configuration objects.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Target checkpoint must contain a state_dict")
    state = checkpoint["state_dict"]
    if EMBEDDING_KEY not in state:
        raise KeyError(f"Target token embedding key is absent: {EMBEDDING_KEY}")
    embedding_tensor = state[EMBEDDING_KEY]
    if tuple(embedding_tensor.shape) != (12160, 1024):
        raise ValueError(f"Unexpected Target token embedding shape: {tuple(embedding_tensor.shape)}")
    feature_embedding = embedding_tensor[FEATURE_OFFSET:EOS_ID].detach().float().cpu().numpy().copy()
    if feature_embedding.shape != (FEATURE_VOCAB, 1024):
        raise ValueError(f"Unexpected Feature embedding slice shape: {feature_embedding.shape}")
    del checkpoint, state, embedding_tensor

    prepared_rows = []
    for original in contexts:
        if "context_prefix" not in original or "p1_id" not in original:
            raise ValueError("Each context needs context_prefix and p1_id")
        derived = derive_context_features(original["context_prefix"], original["p1_id"])
        row = dict(
            vector_index=int(original["vector_index"]), source_id=str(original["scene"]),
            sample_id=str(original["sample_id"]), p1_id=int(original["p1_id"]),
            p1_accept_probability=float(original["p1_accept_probability"]),
            p1_accepted_actual=bool(original["p1_accepted_actual"]),
            f1_accepted_actual=original.get("f1_accepted_actual"),
            f1a_local_id=original.get("f1a_local_id"),
            f1_residual_local_id=original.get("f1_residual_local_id"),
            scene=str(original["scene"]), prefix=int(original["prefix"]),
            round_index=int(original["round_index"]),
            context_pf_pairs=derived["history_pf_pairs"],
            candidate_p_xyz=derived["candidate_p_xyz"],
            neighbor_features=derived["neighbors"],
            recent_features=derived["recent_features"],
        )
        prepared_rows.append((row, derived))

    rows_for_arrays = []
    for row, derived in prepared_rows:
        rows_for_arrays.append(dict(
            vector_index=row["vector_index"], p1_id=row["p1_id"],
            p1_accept_probability=row["p1_accept_probability"], _features=derived,
            p1_accepted_actual=row["p1_accepted_actual"],
        ))
    features = _rows_to_arrays(rows_for_arrays, np)
    if not np.array_equal(features["vector_index"], np.arange(len(contexts), dtype=np.int32)):
        raise AssertionError("derived row order differs from source vector order")

    output.mkdir(parents=True)
    np.savez(output / "prepared_data.npz", feature_embedding=feature_embedding,
             p_f=vectors["p_f"], q1=vectors["q1"], raw_logits=vectors["raw_logits"],
             **features)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for row, _derived in prepared_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metadata = dict(
        status="complete", checkpoint=str(checkpoint_path),
        collection=str(collection), output=str(output), row_count=len(contexts),
        embedding_key=EMBEDDING_KEY, target_embedding_shape=[12160, 1024],
        feature_embedding_slice=[FEATURE_OFFSET, EOS_ID],
        feature_embedding_shape=list(feature_embedding.shape),
        position_vocab=POSITION_VOCAB, feature_vocab=FEATURE_VOCAB,
        eos_id=EOS_ID, target_softcap=15.0,
        feature_logits="existing raw_logits from the same Feature MLP call; model output already uses Target softcap=15",
        policies=dict(temperature=0.9, top_p=0.9),
        vector_arrays={name: list(array.shape) for name, array in vectors.items()},
        full_history_arrays=dict(
            offsets="history_pair_offsets; row i uses [offsets[i], offsets[i+1])",
            positions="history_position_ids_flat",
            features_global="history_feature_global_ids_flat",
        ),
        vector_order="original NPZ order; each context vector_index was required to equal its JSONL row index",
        history_source="context_prefix only; includes the latest complete PF pair even when its F had not yet entered Target KV",
        recent_sequence_order="oldest to newest among the last six committed PF pairs, left-padded when fewer than six exist",
        relative_xyz="stored feature position minus candidate P position",
        age_pf_pairs="number of complete PF pairs since the latest committed pair; latest pair age is zero",
        absent_slots=dict(feature_id=-1, relative_xyz=[0, 0, 0], age_pf_pairs=-1, mask=False),
        target_probabilities_used_for_features=False,
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(dict(status="complete", rows=len(contexts),
                          embedding_shape=list(feature_embedding.shape), output=str(output)),
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
