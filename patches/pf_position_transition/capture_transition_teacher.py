#!/usr/bin/env python3
"""Collect PF-K2 transition teachers through the existing decoder path.

The module is inert at import time. Runtime loading and CUDA work happen only
when the caller invokes the explicit functions or the command line entry point.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


P_VOCAB = 8000
F_VOCAB = 4096
EOS = 12096
DEFAULT_FEATURE = Path("draft_a/runs/feature_target_continue_20261014/selected.pt")
DEFAULT_MANIFEST = Path("docs/project/artifacts/position_transition_20261020/source_manifest.json")
DEFAULT_OUT = Path("patches/pf_position_transition/results/teacher")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError("Cannot write value as JSON: " + type(value).__name__)


def _path(value: str | Path, project: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project / candidate).resolve()


def _check_deadline(deadline_monotonic: float | None, stage: str) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError("Shared collection limit reached during " + stage)


@contextlib.contextmanager
def _fp32_runtime(torch):
    old_precision = torch.get_float32_matmul_precision()
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def load_runtime(project: str | Path, *, feature_checkpoint: str | Path = DEFAULT_FEATURE,
                 device: str = "cuda") -> dict[str, Any]:
    """Load the Target, selected step-12000 MLP, and original Position Ranker once."""
    project_path = Path(project).expanduser().resolve()
    if not project_path.is_dir():
        raise FileNotFoundError("Project directory is missing: " + str(project_path))
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))
    os.chdir(project_path)

    import torch
    from joblib import load as joblib_load
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from draft_a import feature_target_v2
    from draft_a.corpus_v2 import EOS as corpus_eos, load_target, validate

    if device != "cuda":
        raise ValueError("G1 runtime is defined for CUDA FP32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the collector")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if corpus_eos != EOS:
        raise RuntimeError("Corpus EOS differs from PF token layout")

    feature_path = _path(feature_checkpoint, project_path)
    if not feature_path.is_file():
        raise FileNotFoundError("Selected Feature checkpoint is missing: " + str(feature_path))
    feature_meta = torch.load(feature_path, map_location="cpu", weights_only=True)
    feature_step = int(feature_meta.get("step", feature_meta.get("global_step", -1)))
    del feature_meta
    if feature_step != 12000:
        raise RuntimeError("Expected the existing step-12000 Feature MLP; found " + str(feature_step))
    target_wrapper = load_target().float().eval().requires_grad_(False)
    target = target_wrapper.transformer
    config = target.config
    if (str(config.rope_basis).lower() != "position" or bool(config.dense_chunks)
            or bool(config.learned_pos_embed) or int(config.n_embd) != 1024
            or int(config.n_layer) != 24 or int(config.n_head) != 16
            or int(target.rope_dim) != 4):
        raise RuntimeError("Loaded Target does not match the recorded sparse position 4D-RoPE config")
    if (int(target.sparse_position_vocab_size) != P_VOCAB
            or int(target.sparse_feature_vocab_size) != F_VOCAB
            or int(target.sparse_eos_token_id) != EOS):
        raise RuntimeError("Loaded Target token layout differs from PF P/F ranges")
    ranks = target.sparse_position_rank_map.detach().cpu().tolist()
    if len(ranks) != P_VOCAB:
        raise RuntimeError("Target rank map must contain 8000 Position entries")
    head_path = _path(feature_target_v2.HEAD, project_path)
    if not head_path.is_file():
        raise FileNotFoundError("Frozen Feature head is missing: " + str(head_path))
    feature, temperature, top_p = load_feature_model(
        feature_path, head_path, target, torch.device(device), P_VOCAB, EOS,
    )
    if (temperature, top_p) != (0.9, 0.9):
        raise RuntimeError("Existing Feature MLP must use temperature=.9 and top_p=.9")
    ranker_path = _path(feature_target_v2.RANKER, project_path)
    if not ranker_path.is_file():
        raise FileNotFoundError("Position Ranker is missing: " + str(ranker_path))
    ranker = FastHistRanker(joblib_load(ranker_path))
    return dict(
        torch=torch, target_wrapper=target_wrapper, target=target,
        feature=feature, ranker=ranker, ranks=ranks, head_path=head_path,
        ranker_path=ranker_path, feature_checkpoint=feature_path,
        feature_step=feature_step,
        corpus_validate=validate, project=project_path,
        policy_module=__import__("draft_a.pf_ranker_eval", fromlist=["policy"]),
    )


def _cpu_tensor(tensor, *, dtype=None):
    value = tensor.detach()
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.cpu().contiguous().clone()


class _CaptureController:
    def __init__(self, torch):
        self.torch = torch
        self.enabled = False
        self.current = None
        self.current_rows = None
        self.max_rows = 0
        self.row_index = 0
        self.decisions = []
        self.draw_seen = False
        self.position_policy = None

    def begin(self, row, rows, max_rows):
        if not self.enabled or len(rows) >= max_rows:
            self.current = None
            return False
        self.current = row
        self.current_rows = rows
        self.max_rows = max_rows
        self.decisions = []
        self.draw_seen = False
        self.position_policy = None
        rows.append(row)
        return True


class _CapturingMLPAdapter:
    """Runtime-created subclass preserving MLPAdapter outputs and sampling."""
    def __new__(cls, base, feature, controller, torch):
        class Adapter(base):
            def __init__(self):
                super().__init__(feature, bf16=False)
                self.capture_controller = controller
                self.rewind_trace = []
                self._hook = feature.register_forward_hook(self._feature_hook)

            def _feature_hook(self, _module, inputs, output):
                row = self.capture_controller.current
                if row is None:
                    return
                if len(inputs) < 3 or not isinstance(output, (tuple, list)) or len(output) < 2:
                    raise RuntimeError("Unexpected Feature MLP call signature")
                row["h0"] = _cpu_tensor(inputs[0].reshape(-1), dtype=torch.float32)
                row["feature_input_local_id"] = int(inputs[1].reshape(-1)[0].item())
                row["position_input_id"] = int(inputs[2].reshape(-1)[0].item())
                row["old_logits_feature"] = _cpu_tensor(output[1].reshape(-1), dtype=torch.float32)

            def propose(self, prefix, p1, h_prev, rng):
                cc = self.capture_controller
                if cc.current is not None:
                    if "target_logits_feature" not in cc.current:
                        raise RuntimeError("Prior teacher row missed its four-token Target pass")
                    cc.current = None
                if cc.enabled:
                    source = getattr(self, "_source_context", None)
                    if source is None:
                        raise RuntimeError("Source context was not assigned")
                    row = dict(
                        row_id=cc.row_index,
                        split=source["split"], source_id=source["scene_id"],
                        scene_id=source["scene_id"], source_group_id=source["group_id"],
                        prefix_id=f"{source['scene_id']}:prefix{source['prefix_length']}",
                        prefix_length=source["prefix_length"], rng_seed=source["seed"],
                        previous_position_id=int(prefix[-2]),
                        previous_feature_id=int(prefix[-1]),
                        candidate_position_id=int(p1),
                        prefix_tokens=[int(token) for token in prefix],
                        cache_start_position=None, cache_end_position=None,
                        window_type="full_four_token", decisions=[],
                    )
                    cc.row_index += 1
                    self._row_active = cc.begin(row, source["rows"], source["max_rows"])
                    if not self._row_active:
                        self._row_active = False
                else:
                    self._row_active = False
                q = super().propose(prefix, p1, h_prev, rng)
                row = cc.current
                if row is not None:
                    row["q_old"] = _cpu_tensor(q, dtype=torch.float32)
                    if row.get("h0") is None or row.get("old_logits_feature") is None:
                        raise RuntimeError("Feature hook did not capture h0 and old logits")
                    if row["feature_input_local_id"] != row["previous_feature_id"] - P_VOCAB:
                        raise RuntimeError("Feature input ID differs from the committed Feature token")
                    if row["position_input_id"] != row["candidate_position_id"]:
                        raise RuntimeError("Feature MLP Position input differs from Ranker P1")
                return q

            def rewind(self, keep):
                self.rewind_trace.append(int(keep))
                return super().rewind(keep)

            def close(self):
                return super().close()

            def shutdown(self):
                try:
                    self._hook.remove()
                finally:
                    return super().close()

        return Adapter()


def _install_capture(runtime, controller):
    import bench_gaussiangpt_pf_block2_cow as bench
    import draft_a.pf_ranker_eval as shared
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock as OriginalCapture
    from bench_gaussiangpt_pf_block2_cow import decide as original_decide
    from draft_a.pf_feature_unified import MLPAdapter

    class CaptureLastBlock:
        def __init__(self, target):
            self.inner = OriginalCapture(target)

        def run(self, target, cache, tokens, kinds, device):
            start = int(cache.get_pos())
            output = self.inner.run(target, cache, tokens, kinds, device)
            row = controller.current
            if controller.enabled and row is not None and len(tokens) == 4:
                logits, states = output
                if len(row["prefix_tokens"]) != start + 1:
                    raise RuntimeError("Prefix length and Target cache index differ")
                if int(tokens[0]) != row["previous_feature_id"] or int(tokens[1]) != row["candidate_position_id"]:
                    raise RuntimeError("Four-token Target window does not match the draft context")
                row["cache_start_position"] = start
                row["cache_end_position"] = int(cache.get_pos())
                row["verification_input_tokens"] = [int(token) for token in tokens]
                row["feature_candidate_id"] = int(tokens[2])
                row["next_position_candidate_id"] = int(tokens[3])
                row["hF"] = _cpu_tensor(states[0].reshape(-1), dtype=runtime["torch"].float32)
                row["hP"] = _cpu_tensor(states[1].reshape(-1), dtype=runtime["torch"].float32)
                row["target_logits_position"] = _cpu_tensor(
                    runtime["torch"].cat((logits[0, :P_VOCAB], logits[0, EOS:EOS + 1])),
                    dtype=runtime["torch"].float32,
                )
                row["target_logits_feature"] = _cpu_tensor(
                    logits[1, P_VOCAB:EOS], dtype=runtime["torch"].float32,
                )
                if row["cache_end_position"] != start + 4:
                    raise RuntimeError("Target KV length did not advance by four tokens")
            return output

        def close(self):
            return self.inner.close()

    def tracked_decide(alpha, rng, device):
        result = original_decide(alpha, rng, device)
        row = controller.current
        if controller.enabled and row is not None:
            entry = dict(alpha=float(alpha), accepted=bool(result))
            controller.decisions.append(entry)
            row["decisions"] = list(controller.decisions)
            if len(controller.decisions) == 1:
                row["p_p1"] = float(alpha)
                row["p1_accept_probability"] = float(alpha)
                row["p1_accepted"] = bool(result)
            elif len(controller.decisions) == 2:
                row["p_f1_accept_probability"] = float(alpha)
                row["f1_accepted"] = bool(result)
            elif len(controller.decisions) == 3:
                row["p_p2_accept_probability"] = float(alpha)
                row["p2_accepted"] = bool(result)
        return result

    original_draw = shared.draw

    def tracked_draw(probabilities, rng):
        token = original_draw(probabilities, rng)
        row = controller.current
        if controller.enabled and row is not None and "feature_candidate_id" not in row and not controller.draw_seen:
            row["feature_candidate_id"] = int(token) + P_VOCAB
            controller.draw_seen = True
        return token

    bench.CaptureLastBlock = CaptureLastBlock
    bench.decide = tracked_decide
    shared.draw = tracked_draw
    return dict(bench=bench, shared=shared, old_capture=OriginalCapture,
                old_decide=original_decide, old_draw=original_draw,
                base_adapter=MLPAdapter)


def _restore_capture(patch):
    patch["bench"].CaptureLastBlock = patch["old_capture"]
    patch["bench"].decide = patch["old_decide"]
    patch["shared"].draw = patch["old_draw"]


def _policy(runtime, logits):
    return runtime["policy_module"].policy(logits)


def _validate_replay(runtime, prompt, seed, budget=16):
    torch = runtime["torch"]
    from draft_a.pf_feature_unified import MLPAdapter, decode
    controller = _CaptureController(torch)
    patch = _install_capture(runtime, controller)
    normal = MLPAdapter(runtime["feature"], bf16=False)
    traced = _CapturingMLPAdapter(patch["base_adapter"], runtime["feature"], controller, torch)
    normal.rewind_trace = []
    normal_rewind = normal.rewind
    normal.rewind = lambda keep: (normal.rewind_trace.append(int(keep)), normal_rewind(keep))[1]
    replay_rows = []
    traced._source_context = dict(
        split="dev", scene_id="replay", group_id="replay",
        prefix_length=len(prompt) - 1, seed=seed, rows=replay_rows, max_rows=100,
    )
    try:
        with torch.inference_mode(), _fp32_runtime(torch):
            controller.enabled = False
            stats_a, output_a = decode(runtime["target"], normal, runtime["ranker"],
                                      runtime["ranks"], prompt, budget, seed)
            controller.enabled = True
            stats_b, output_b = decode(runtime["target"], traced, runtime["ranker"],
                                       runtime["ranks"], prompt, budget, seed)
            controller.enabled = False
        ignored = {"seconds", "tokens_per_second", "acceptance", "acceptance_proposed",
                   "acceptance_verified", "position_acceptance_proposed",
                   "position_acceptance_verified", "feature_acceptance_proposed",
                   "feature_acceptance_verified", "eos_acceptance_proposed",
                   "eos_acceptance_verified"}
        stable_a = {k: v for k, v in stats_a.items() if k not in ignored}
        stable_b = {k: v for k, v in stats_b.items() if k not in ignored}
        result = dict(
            budget=budget, seed=seed, output_equal=output_a == output_b,
            output_tokens=len(output_a), counters_equal=stable_a == stable_b,
            rewind_trace_equal=normal.rewind_trace == traced.rewind_trace,
            captured_rows=len(replay_rows),
            capture_rows_complete=all(
                all(key in row for key in ("h0", "hF", "hP", "old_logits_feature",
                                           "target_logits_feature", "p_p1", "p1_accepted"))
                for row in replay_rows
            ),
            ordinary_rewind_trace=normal.rewind_trace,
            capture_rewind_trace=traced.rewind_trace,
            ordinary_output=[int(x) for x in output_a],
            capture_output=[int(x) for x in output_b],
        )
        if not result["output_equal"] or not result["counters_equal"] or not result["rewind_trace_equal"]:
            raise RuntimeError("Capture replay differs from the ordinary MLP decoder")
        if not replay_rows or not result["capture_rows_complete"]:
            raise RuntimeError("Sixteen-token replay did not exercise complete teacher capture")
        return result
    finally:
        normal.close()
        traced.shutdown()
        _restore_capture(patch)


def _load_manifest_records(runtime, manifest_path: Path, project: Path):
    import torch

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Source manifest must contain a sources list")
    splits = {key: [row for row in sources if row.get("split") == key]
              for key in ("train", "val", "dev", "confirmation")}
    expected = {"train": 16, "val": 4, "dev": 10, "confirmation": 20}
    counts = {key: len(rows) for key, rows in splits.items()}
    if counts != expected:
        raise ValueError("Manifest split counts differ: " + repr(counts))
    if any(int(row.get("max_rows", 0)) != 0 for row in splits["confirmation"]):
        raise ValueError("Confirmation sources must remain unused in G1")
    seen_groups = {}
    seen_scenes = set()
    for split, entries in splits.items():
        for entry in entries:
            scene = str(entry["scene_id"])
            group = str(entry["group_id"])
            if scene in seen_scenes:
                raise ValueError("Scene id appears more than once: " + scene)
            seen_scenes.add(scene)
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError("Group id crosses source splits: " + group)
            seen_groups[group] = split
    selected = []
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for source in sources:
        if source["split"] not in {"train", "val", "dev"}:
            continue
        crops_path = _path(source["crops_path"], project)
        grouped.setdefault(crops_path, []).append(source)
    for crops_path, source_rows in grouped.items():
        if not crops_path.is_file():
            raise FileNotFoundError("Crop file is missing: " + str(crops_path))
        raw = torch.load(crops_path, map_location="cpu", weights_only=True)
        for source in source_rows:
            index = int(source["crop_index"])
            if index < 0 or index >= len(raw):
                raise IndexError("Crop index is outside crop file: " + str(index))
            record = raw[index]
            runtime["corpus_validate"](record)
            scene = record.get("scene")
            if not scene and record.get("path"):
                scene = Path(record["path"]).parents[1].name
            if str(scene) != str(source["scene_id"]):
                raise ValueError("Manifest crop does not match its scene id: "
                                 + str(source["scene_id"]) + " != " + str(scene))
            tokens = record["tokens"].detach().cpu().long().contiguous().clone()
            if len(tokens) < 64:
                raise ValueError("Crop has fewer than 64 tokens: " + str(scene))
            selected.append(dict(
                split=str(source["split"]), scene_id=str(source["scene_id"]),
                group_id=str(source["group_id"]), crops_path=str(crops_path),
                crop_index=index, crop_seed=int(source.get("crop_seed", 0)),
                max_rows=int(source["max_rows"]), tokens=tokens,
                record_path=str(record.get("path", "")),
                has_eos=bool(record.get("has_eos", False)),
            ))
        del raw
    order = {"train": 0, "val": 1, "dev": 2}
    selected.sort(key=lambda row: (order[row["split"]], row["scene_id"]))
    expected_rows = {"train": 1600, "val": 400, "dev": 400}
    planned_rows = {key: sum(row["max_rows"] for row in selected if row["split"] == key)
                    for key in expected_rows}
    if planned_rows != expected_rows:
        raise ValueError("Manifest row limits differ: " + repr(planned_rows))
    return manifest, selected


def _validate_teacher(runtime, rows, deadline_monotonic=None):
    import torch
    from model.gpt.nanochat_gpt import norm as target_norm

    if not rows:
        raise RuntimeError("No full-window teacher rows were collected")
    target = runtime["target"]
    device = next(target.parameters()).device
    max_logits = 0.0
    max_prob = 0.0
    max_tv = 0.0
    max_position_prob = 0.0
    checked = 0
    batch_size = 32
    for begin in range(0, len(rows), batch_size):
        _check_deadline(deadline_monotonic, "teacher probability validation")
        subset = rows[begin:begin + batch_size]
        hp = torch.stack([row["hP"] for row in subset]).to(device=device, dtype=torch.float32)
        normalized = target_norm(hp)
        full = target.lm_head(normalized).float()[..., :int(target.config.vocab_size)]
        full = 15.0 * torch.tanh(full / 15.0)
        full_feature = full[:, P_VOCAB:EOS]
        for j, row in enumerate(subset):
            captured_logits = row["target_logits_feature"].to(device=device, dtype=torch.float32)
            delta_logits = float((full_feature[j] - captured_logits).abs().max().item())
            max_logits = max(max_logits, delta_logits)
            p_captured = _policy(runtime, captured_logits).float()
            p_full = _policy(runtime, full_feature[j]).float()
            if not bool(torch.isfinite(p_captured).all()) or not bool(torch.isfinite(p_full).all()):
                raise RuntimeError("Target Feature policy contains nonfinite values")
            if bool((p_captured < 0).any()) or bool((p_full < 0).any()):
                raise RuntimeError("Target Feature policy contains negative values")
            if abs(float(p_captured.sum().item()) - 1.0) > 2e-6:
                raise RuntimeError("Captured Target Feature policy does not sum to one")
            if abs(float(p_full.sum().item()) - 1.0) > 2e-6:
                raise RuntimeError("Recomputed Target Feature policy does not sum to one")
            probability_error = float((p_captured - p_full).abs().max().item())
            tv = float((p_captured - p_full).abs().sum().item() * 0.5)
            max_prob = max(max_prob, probability_error)
            max_tv = max(max_tv, tv)
            row["p_f_policy"] = _cpu_tensor(p_captured, dtype=torch.float32)
            row["target_logits_full_projection_max_abs"] = delta_logits
            position_logits = row["target_logits_position"].to(device=device, dtype=torch.float32)
            p_position = _policy(runtime, position_logits).float()
            recomputed_p1 = float(p_position[int(row["candidate_position_id"])].item())
            position_error = abs(recomputed_p1 - float(row["p_p1"]))
            max_position_prob = max(max_position_prob, position_error)
            row["p_p1_recomputed"] = recomputed_p1
            if abs(float(row["q_old"].sum().item()) - 1.0) > 2e-6:
                raise RuntimeError("Old MLP q does not sum to one")
            if not bool(torch.isfinite(row["q_old"]).all()) or bool((row["q_old"] < 0).any()):
                raise RuntimeError("Old MLP q has invalid probability values")
            if row["p1_accept_probability"] < 0.0 or row["p1_accept_probability"] > 1.0:
                raise RuntimeError("P1 acceptance probability is outside [0,1]")
            if not (0 <= row["previous_position_id"] < P_VOCAB
                    and P_VOCAB <= row["previous_feature_id"] < EOS
                    and 0 <= row["candidate_position_id"] < P_VOCAB
                    and P_VOCAB <= row["feature_candidate_id"] < EOS
                    and 0 <= row["next_position_candidate_id"] < P_VOCAB):
                raise RuntimeError("Captured P/F token ids are outside their vocabulary ranges")
            if (row["old_logits_feature"].shape != (F_VOCAB,)
                    or not bool(torch.isfinite(row["old_logits_feature"]).all())
                    or float(row["old_logits_feature"].abs().max()) > 15.0001):
                raise RuntimeError("Old Feature logits are not the expected softcapped 4096-vector")
            for state_key in ("h0", "hF", "hP"):
                state = row[state_key]
                if state.shape != (int(target.config.n_embd),) or not bool(torch.isfinite(state).all()):
                    raise RuntimeError("Captured state has an invalid shape or value: " + state_key)
            expected_input = row["prefix_tokens"][-1]
            if row["verification_input_tokens"][0] != expected_input:
                raise RuntimeError("Target verification does not begin with pending Feature")
            if row["cache_start_position"] != len(row["prefix_tokens"]) - 1:
                raise RuntimeError("Target cache start differs from captured prefix index")
            if row["cache_end_position"] != row["cache_start_position"] + 4:
                raise RuntimeError("Target cache end does not reflect one four-token verify")
            checked += 1
    limits = dict(max_abs_probability=1e-5, total_variation=1e-5,
                  position_p1_probability=1e-5)
    if max_prob > limits["max_abs_probability"] or max_tv > limits["total_variation"]:
        raise RuntimeError("Full and typed Feature probability policies exceed the planned tolerance")
    if max_position_prob > limits["position_p1_probability"]:
        raise RuntimeError("Recorded and recomputed P1 probabilities differ")
    return dict(
        rows_checked=checked, full_vs_typed_logits_max_abs=max_logits,
        full_vs_typed_policy_max_abs=max_prob, full_vs_typed_policy_max_tv=max_tv,
        p_p1_recorded_vs_recomputed_max_abs=max_position_prob,
        limits=limits, batch_size=batch_size,
        comparison="full lm_head projection from captured pre-finalnorm hP versus typed Feature row",
    )


def _tensor_rows(torch, rows, field, dtype=None):
    if dtype is None:
        dtype = torch.float32
    return torch.stack([row[field].to(dtype=dtype).cpu() for row in rows])


def _build_teacher(runtime, rows, manifest_path: Path, replay: dict, validation: dict):
    torch = runtime["torch"]
    integer = torch.int64
    float_tensor = torch.float32
    bool_tensor = torch.bool
    optional_float = lambda key: [float(row.get(key, -1.0)) for row in rows]
    teacher = dict(
        schema_version=1,
        manifest_path=str(manifest_path),
        row_metadata=[{k: v for k, v in row.items()
                       if k not in {"h0", "hF", "hP", "old_logits_feature", "target_logits_feature",
                                    "target_logits_position", "q_old", "p_f_policy"}}
                      for row in rows],
        h0=_tensor_rows(torch, rows, "h0"),
        hF=_tensor_rows(torch, rows, "hF"),
        hP=_tensor_rows(torch, rows, "hP"),
        previous_feature_id=torch.tensor([r["previous_feature_id"] for r in rows], dtype=integer),
        previous_position_id=torch.tensor([r["previous_position_id"] for r in rows], dtype=integer),
        candidate_position_id=torch.tensor([r["candidate_position_id"] for r in rows], dtype=integer),
        feature_candidate_id=torch.tensor([r["feature_candidate_id"] for r in rows], dtype=integer),
        next_position_candidate_id=torch.tensor([r["next_position_candidate_id"] for r in rows], dtype=integer),
        old_logits_feature=_tensor_rows(torch, rows, "old_logits_feature"),
        target_logits_feature=_tensor_rows(torch, rows, "target_logits_feature"),
        target_logits_position=_tensor_rows(torch, rows, "target_logits_position"),
        target_logits_position_support_ids=torch.cat((
            torch.arange(P_VOCAB, dtype=integer), torch.tensor([EOS], dtype=integer),
        )),
        p_f_policy=_tensor_rows(torch, rows, "p_f_policy"),
        q_old=_tensor_rows(torch, rows, "q_old"),
        p_p1=torch.tensor([float(r["p_p1"]) for r in rows], dtype=float_tensor),
        p1_accept_probability=torch.tensor([float(r["p1_accept_probability"]) for r in rows], dtype=float_tensor),
        p1_accepted=torch.tensor([bool(r["p1_accepted"]) for r in rows], dtype=bool_tensor),
        p_p1_recomputed=torch.tensor([float(r["p_p1_recomputed"]) for r in rows], dtype=float_tensor),
        p_f1_accept_probability=torch.tensor(optional_float("p_f1_accept_probability"), dtype=float_tensor),
        f1_accepted=torch.tensor([int(r.get("f1_accepted", -1)) for r in rows], dtype=torch.int8),
        p_p2_accept_probability=torch.tensor(optional_float("p_p2_accept_probability"), dtype=float_tensor),
        p2_accepted=torch.tensor([int(r.get("p2_accepted", -1)) for r in rows], dtype=torch.int8),
        cache_start_position=torch.tensor([r["cache_start_position"] for r in rows], dtype=integer),
        cache_end_position=torch.tensor([r["cache_end_position"] for r in rows], dtype=integer),
        prefix_tokens=[torch.tensor(r["prefix_tokens"], dtype=integer) for r in rows],
        verification_input_tokens=[torch.tensor(r["verification_input_tokens"], dtype=integer) for r in rows],
        p1_acceptance_weighted=True,
        token_layout=dict(position_vocab=P_VOCAB, feature_start=P_VOCAB,
                          feature_end=EOS, eos_id=EOS, feature_ids="global"),
        policy=dict(temperature=0.9, top_p=0.9, target_softcap=15.0,
                    old_feature_logits_already_softcapped=True),
        replay=replay,
        validation=validation,
    )
    return teacher


def save_token_embeddings(runtime, out: str | Path) -> Path:
    """Save only the original Target token table to CPU for fixed downstream use."""
    torch = runtime["torch"]
    output = Path(out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "token_embeddings.pt"
    embedding = runtime["target"].transformer.wte.weight.detach().cpu().contiguous().clone()
    torch.save(embedding, path)
    return path


def collect_teacher(runtime, manifest_path: str | Path, out: str | Path,
                    *, deadline_monotonic: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect rows from train/val/dev using one original PF-K2 call per source."""
    torch = runtime["torch"]
    from draft_a.pf_feature_unified import decode

    project = runtime["project"]
    manifest_file = _path(manifest_path, project)
    output = _path(out, project)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be empty: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    _check_deadline(deadline_monotonic, "manifest and crop loading")
    manifest, selected_sources = _load_manifest_records(runtime, manifest_file, project)
    collection_seed = int(manifest.get("collection_seed", 39111))
    prefix_length = int(manifest.get("prefix_length", 64))
    if prefix_length != 64 or int(manifest.get("collection_budget_per_scene", 512)) != 512:
        raise ValueError("This collector is fixed to a 64-token prefix and 512-token source run")

    controller = _CaptureController(torch)
    patch = _install_capture(runtime, controller)
    from draft_a.pf_feature_unified import MLPAdapter
    adapter = _CapturingMLPAdapter(patch["base_adapter"], runtime["feature"], controller, torch)
    source_metrics = []
    all_rows = []
    replay = None
    started = time.monotonic()
    status = "running"
    _write_json(output / "collection.json", dict(
        status=status, manifest=str(manifest_file), project=str(project),
        feature_checkpoint=str(runtime["feature_checkpoint"]),
        feature_step=runtime["feature_step"],
        target_loaded_once=True, collection_seed=collection_seed,
        prefix_length=prefix_length, budget=512, temperature=0.9, top_p=0.9,
        split_source_counts={key: sum(x["split"] == key for x in selected_sources)
                             for key in ("train", "val", "dev")},
    ))
    try:
        with torch.inference_mode(), _fp32_runtime(torch):
            first_dev = next(row for row in selected_sources if row["split"] == "dev")
            prompt = [EOS] + [int(x) for x in first_dev["tokens"][:prefix_length].tolist()]
            replay = _validate_replay(runtime, prompt, collection_seed, budget=16)
            _write_json(output / "replay.json", replay)

            controller.enabled = True
            for source in selected_sources:
                _check_deadline(deadline_monotonic, "source decode " + source["scene_id"])
                source_rows = []
                source_context = dict(
                    split=source["split"], scene_id=source["scene_id"],
                    group_id=source["group_id"], prefix_length=prefix_length,
                    seed=collection_seed, rows=source_rows, max_rows=source["max_rows"],
                )
                adapter._source_context = source_context
                prompt = [EOS] + [int(x) for x in source["tokens"][:prefix_length].tolist()]
                if len(prompt) != prefix_length + 1:
                    raise RuntimeError("Prompt construction differs from 64 body tokens")
                source_begin = time.monotonic()
                stats, output_tokens = decode(
                    runtime["target"], adapter, runtime["ranker"], runtime["ranks"],
                    prompt, 512, collection_seed,
                )
                if controller.current is not None:
                    if "target_logits_feature" not in controller.current:
                        raise RuntimeError("Last sampled row missed its Target verify")
                    controller.current = None
                row_offset = len(all_rows)
                for local_index, row in enumerate(source_rows):
                    if "target_logits_feature" not in row:
                        raise RuntimeError("Collected row has no Target teacher logits")
                    if "p_p1" not in row:
                        raise RuntimeError("Collected row has no actual P1 acceptance decision")
                    if "feature_candidate_id" not in row or "next_position_candidate_id" not in row:
                        raise RuntimeError("Collected row has no sampled F1/P2 candidates")
                    row["row_id"] = row_offset + local_index
                all_rows.extend(source_rows)
                metric = dict(
                    split=source["split"], scene_id=source["scene_id"],
                    requested_rows=source["max_rows"], captured_rows=len(source_rows),
                    seed=collection_seed, prompt_length=len(prompt),
                    generated_tokens=[int(x) for x in output_tokens],
                    decode_stats={key: value for key, value in stats.items()
                                  if key not in {"seconds", "tokens_per_second"}},
                    elapsed_seconds=time.monotonic() - source_begin,
                )
                source_metrics.append(metric)
                with (output / "source_runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_jsonable(metric), ensure_ascii=False,
                                            allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                _write_json(output / "collection.json", dict(
                    status="collecting", manifest=str(manifest_file),
                    collected_sources=len(source_metrics), expected_sources=len(selected_sources),
                    collected_rows=len(all_rows), source_metrics=source_metrics,
                    elapsed_seconds=time.monotonic() - started,
                ))
            controller.enabled = False
            _check_deadline(deadline_monotonic, "full-head teacher probability validation")
            validation = _validate_teacher(runtime, all_rows, deadline_monotonic)
            teacher = _build_teacher(runtime, all_rows, manifest_file, replay, validation)
            teacher_path = output / "teacher.pt"
            temporary = teacher_path.with_suffix(".pt.tmp")
            torch.save(teacher, temporary)
            os.replace(temporary, teacher_path)
            token_embedding = runtime["target"].transformer.wte.weight.detach().cpu().contiguous().clone()
            embedding_path = output / "token_embeddings.pt"
            torch.save(token_embedding, embedding_path)
            split_metrics = {}
            for split in ("train", "val", "dev"):
                rows_for_split = [row for row in all_rows if row["split"] == split]
                split_metrics[split] = dict(
                    sources=sum(item["split"] == split for item in source_metrics),
                    captured_rows=len(rows_for_split),
                    max_rows=sum(item["requested_rows"] for item in source_metrics
                                 if item["split"] == split),
                )
            summary = dict(
                status="complete", manifest=str(manifest_file),
                source_count=len(source_metrics), row_count=len(all_rows),
                splits=split_metrics, replay=replay, teacher_validation=validation,
                token_embedding_path=str(embedding_path),
                token_embedding_shape=list(token_embedding.shape),
                teacher_path=str(teacher_path), elapsed_seconds=time.monotonic() - started,
                target_loaded_once=True, target_forward_repeats=0,
                capture_scope="ordinary PF-K2 full four-token verification windows only",
            )
            _write_json(output / "collection.json", summary)
            return teacher, summary
    finally:
        controller.enabled = False
        controller.current = None
        adapter.shutdown()
        _restore_capture(patch)


def run_collection(project: str | Path, *, manifest_path: str | Path = DEFAULT_MANIFEST,
                   out: str | Path = DEFAULT_OUT,
                   feature_checkpoint: str | Path = DEFAULT_FEATURE,
                   deadline_seconds: float = 900.0) -> tuple[dict[str, Any], dict[str, Any]]:
    if deadline_seconds <= 0 or deadline_seconds > 900:
        raise ValueError("deadline_seconds must be in (0, 900]")
    started = time.monotonic()
    deadline = started + deadline_seconds
    runtime = load_runtime(project, feature_checkpoint=feature_checkpoint)
    _check_deadline(deadline, "model loading")
    return collect_teacher(runtime, manifest_path, out, deadline_monotonic=deadline)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GaussianGPT project root")
    parser.add_argument("--source-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--feature-checkpoint", default=str(DEFAULT_FEATURE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--deadline-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.deadline_seconds <= 0 or args.deadline_seconds > 900:
        parser.error("--deadline-seconds must be in (0, 900]")
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    runtime = load_runtime(args.project, feature_checkpoint=args.feature_checkpoint)
    _check_deadline(deadline, "model loading")
    _, summary = collect_teacher(runtime, args.source_manifest, args.out,
                                 deadline_monotonic=deadline)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
