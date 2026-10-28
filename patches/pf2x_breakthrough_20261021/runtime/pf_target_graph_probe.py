#!/usr/bin/env python3
"""Measure fixed-position Target forwards with the real FP32 PF model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import statistics
import time
from pathlib import Path
from types import MethodType


EOS = 12096
P_VOCAB = 8000
F_OFFSET = 8000
DEFAULT_CONTEXTS = (64, 256, 768, 2048)
DEFAULT_SHAPES = (1, 2, 4)
MODES = (
    "sdpa_gpu_pos",
    "sdpa_host_pos",
    "explicit_host_pos",
    "graph_explicit_host_pos",
)
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-4
HIDDEN_ATOL = 1e-4
HIDDEN_RTOL = 1e-4
POLICY_TV_MAX = 5e-5


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("provide a nonempty list of unique integers")
    return values


def typed_kinds(start: int, count: int) -> list[str]:
    return ["position" if (start + offset) % 2 == 0 else "feature"
            for offset in range(count)]


def pair_order(pair_index: int, context_index: int, shape_index: int) -> list[str]:
    offset = (pair_index + context_index + shape_index) % len(MODES)
    return list(MODES[offset:]) + list(MODES[:offset])


def self_test() -> dict:
    import torch
    import torch.nn.functional as F

    for ctx in DEFAULT_CONTEXTS:
        assert ctx % 2 == 0
        assert typed_kinds(ctx, 4) == ["position", "feature", "position", "feature"]
    for context_index in range(4):
        for shape_index in range(3):
            orders = [pair_order(index, context_index, shape_index) for index in range(8)]
            for order in orders:
                assert sorted(order) == sorted(MODES)
            assert all(orders[index][0] != orders[index + 1][0]
                       for index in range(len(orders) - 1))

    timing_rows = []
    for pair_index in range(4):
        for mode, elapsed in zip(MODES, (8.0, 4.0, 2.0, 1.0)):
            timing_rows.append(dict(context_length=64, query_tokens=1,
                                    pair_index=pair_index, mode=mode,
                                    host_ms_per_call=elapsed,
                                    cuda_ms_per_call=elapsed / 2,
                                    host_minus_cuda_ms_per_call=elapsed / 2))
    summary = summarize_pairs(timing_rows)[0]
    assert summary["explicit_attention"]["reference_mode"] == "sdpa_gpu_pos"
    assert summary["explicit_attention"]["candidate_mode"] == "explicit_host_pos"
    assert summary["explicit_attention"]["speed_ratio_median"] == 4.0
    assert summary["graph_over_explicit"]["speed_ratio_median"] == 2.0

    torch.manual_seed(20261021)
    batch, q_heads, kv_heads, width = 1, 4, 2, 8
    pos, count, capacity = 5, 4, 12
    q = torch.randn(batch, count, q_heads, width, dtype=torch.float32)
    k = torch.randn(batch, count, kv_heads, width, dtype=torch.float32)
    v = torch.randn(batch, count, kv_heads, width, dtype=torch.float32)
    k_cache = torch.randn(batch, capacity, kv_heads, width, dtype=torch.float32)
    v_cache = torch.randn(batch, capacity, kv_heads, width, dtype=torch.float32)
    seq = torch.tensor([pos], dtype=torch.int32)
    impl = make_explicit_kvcache(torch, pos, count, torch.device("cpu"))
    actual = impl(q, k_cache, v_cache, k=k, v=v, cache_seqlens=seq,
                  causal=True, window_size=(-1, 0))
    k_full = k_cache[:, :pos + count].transpose(1, 2)
    v_full = v_cache[:, :pos + count].transpose(1, 2)
    groups = q_heads // kv_heads
    k_repeat = k_full.repeat_interleave(groups, dim=1)
    v_repeat = v_full.repeat_interleave(groups, dim=1)
    allow = impl.mask.view(1, 1, count, pos + count)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k_repeat, v_repeat, attn_mask=allow, is_causal=False
    ).transpose(1, 2)
    error = float((actual - expected).abs().max().item())
    if not torch.allclose(actual, expected, atol=2e-6, rtol=2e-6):
        raise AssertionError(f"explicit attention CPU error is too large: {error}")
    return dict(passed=True, contexts=len(DEFAULT_CONTEXTS), shapes=list(DEFAULT_SHAPES),
                paired_orders=4 * 3 * 8, explicit_attention_max_abs_error=error,
                pair_summary_direction="reference_over_candidate",
                cpu_only=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--teacher", type=Path, default=Path("checkpoints/gpt_vfront.ckpt"))
    parser.add_argument("--vqvae", type=Path, default=Path("checkpoints/vqvae_vfront.ckpt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contexts", type=parse_int_list,
                        default=DEFAULT_CONTEXTS)
    parser.add_argument("--shapes", type=parse_int_list, default=DEFAULT_SHAPES)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--pair-repeats", type=int, default=10)
    parser.add_argument("--calls-per-pair", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.device != "cuda":
        parser.error("this probe requires the production CUDA device")
    if not args.out_dir:
        parser.error("--out-dir is required")
    if any(ctx <= 0 or ctx % 2 for ctx in args.contexts):
        parser.error("each context length must be a positive even number")
    if any(shape not in {1, 2, 4} for shape in args.shapes):
        parser.error("--shapes accepts only 1, 2, and 4")
    if args.warmup_calls < 2 or args.pair_repeats < 4 or args.calls_per_pair < 1:
        parser.error("use at least two warmup calls, four pairs, and one call per pair")
    return args


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def make_explicit_kvcache(torch, pos: int, count: int, device):
    query_positions = torch.arange(count, device=device) + pos
    key_positions = torch.arange(pos + count, device=device)
    allow = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    mask = allow.view(1, 1, 1, count, pos + count)

    def explicit_kvcache(q, k_cache, v_cache, k=None, v=None,
                         cache_seqlens=None, causal=False,
                         window_size=(-1, -1)):
        if cache_seqlens is None:
            raise RuntimeError("cache_seqlens is required")
        if not causal:
            raise RuntimeError("only causal attention is supported")
        batch, actual_count, q_heads, width = q.shape
        if actual_count != count or k is None or v is None:
            raise RuntimeError("input shape differs from the captured block")
        if q_heads % k_cache.size(2):
            raise RuntimeError("query heads are not divisible by key/value heads")
        k_cache[:, pos:pos + count, :, :] = k
        v_cache[:, pos:pos + count, :, :] = v
        k_full = k_cache[:, :pos + count, :, :].transpose(1, 2)
        v_full = v_cache[:, :pos + count, :, :].transpose(1, 2)
        kv_heads = k_full.size(1)
        groups = q_heads // kv_heads
        q_grouped = q.transpose(1, 2).reshape(
            batch, kv_heads, groups, count, width
        )
        # PyTorch's SDPA math backend scales Q and K separately before QKᵀ.
        # Matching that order avoids accumulating a small FP32 discrepancy
        # across the 24-layer Target trunk.
        q_scale = width ** -0.25
        q_scaled = q_grouped * q_scale
        k_scaled = k_full * q_scale
        scores = torch.matmul(
            q_scaled, k_scaled.transpose(-1, -2).unsqueeze(2)
        )
        # Mirror the local SDPA KV-cache fallback, which ignores window_size.
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        values = torch.matmul(probs, v_full.unsqueeze(2))
        return values.reshape(batch, q_heads, count, width).transpose(1, 2)

    explicit_kvcache.mask = mask
    return explicit_kvcache


def make_prompt(cache_length: int) -> list[int]:
    tokens = [EOS]
    pair_count = cache_length // 2 + 1
    for pair in range(pair_count):
        tokens.append(pair % P_VOCAB)
        tokens.append(F_OFFSET + pair % 4096)
    tokens = tokens[:cache_length + 1]
    return tokens


def make_block(tokens: list[int], shape: int, p1: int, f1: int) -> list[int]:
    block = [tokens[-1], p1, f1, min(P_VOCAB - 1, p1 + 1)]
    return block[:shape]


def patch_fixed_position(cache, pos: int, gpu_scalar: bool) -> None:
    def no_advance(self, num_tokens):
        return None

    cache.advance = MethodType(no_advance, cache)
    if gpu_scalar:
        return

    def host_get_pos(self):
        return pos

    cache.get_pos = MethodType(host_get_pos, cache)


def make_host_position_attention(torch, original_attention, pos: int):
    host_position = torch.tensor([pos], dtype=torch.int32, device="cpu")

    def host_attention(*args, **kwargs):
        kwargs["cache_seqlens"] = host_position
        return original_attention(*args, **kwargs)

    return host_attention


def compare_tensors(torch, left, right, atol: float, rtol: float) -> dict:
    if tuple(left.shape) != tuple(right.shape):
        return dict(passed=False, shape_left=list(left.shape),
                    shape_right=list(right.shape))
    left_finite = torch.isfinite(left)
    right_finite = torch.isfinite(right)
    support_mismatches = int((left_finite != right_finite).sum().item())
    if support_mismatches:
        return dict(passed=False, support_mismatches=support_mismatches)
    finite = left_finite & right_finite
    if bool(torch.isnan(left).any().item()) or bool(torch.isnan(right).any().item()):
        return dict(passed=False, nan_values=True)
    if bool(finite.any().item()):
        delta = (left[finite] - right[finite]).abs()
        limit = atol + rtol * right[finite].abs()
        max_abs = float(delta.max().item())
        max_scaled = float((delta / limit.clamp_min(1e-30)).max().item())
        passed = bool(torch.all(delta <= limit).item())
    else:
        max_abs = 0.0
        max_scaled = 0.0
        passed = True
    return dict(passed=passed, support_mismatches=0,
                max_abs_error=max_abs, max_tolerance_fraction=max_scaled)


def save_output_tensors(torch, path: Path, context_length: int, query_tokens: int,
                        tokens: list[int], kinds: list[str], outputs: dict) -> None:
    def to_cpu(value):
        return value.detach().to(device="cpu").clone()

    payload = dict(context_length=context_length, query_tokens=query_tokens,
                   input_tokens=torch.tensor([tokens], dtype=torch.long),
                   cache_position=context_length, logit_types=list(kinds),
                   outputs={name: dict(logits=to_cpu(value[0]), hidden=to_cpu(value[1]))
                            for name, value in outputs.items()})
    torch.save(payload, path)


def policy_tv(torch, left, right, policy) -> float:
    values = []
    for row in range(left.size(0)):
        left_p = policy(left[row])
        right_p = policy(right[row])
        values.append(float((left_p - right_p).abs().sum().item()) * 0.5)
    return max(values, default=0.0)


class LastBlockTap:
    def __init__(self, target):
        self.output = None
        self.handle = target.transformer.h[-1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(value, "shape") or value.ndim != 3 or value.size(0) != 1:
            raise RuntimeError("unexpected last-block output")
        self.output = value.detach()

    def close(self):
        self.handle.remove()


def load_target(args, torch, device):
    from model.gaussian_vqvae import GaussianVQVAE
    from profiling.s2.s2_decode_faithful_agreement import load_model

    vqvae = GaussianVQVAE.load_from_checkpoint(str(args.vqvae), map_location="cpu")
    vqvae.eval().requires_grad_(False)
    teacher = load_model(args.teacher, vqvae)
    teacher.eval()
    target = teacher.gpt.transformer
    sparse_spec = (int(target.sparse_position_vocab_size),
                   int(target.sparse_feature_vocab_size),
                   int(target.sparse_feature_token_offset),
                   int(target.sparse_eos_token_id), bool(target.sparse_shared_vocab))
    if sparse_spec != (P_VOCAB, 4096, F_OFFSET, EOS, False):
        raise RuntimeError(f"unexpected sparse vocabulary layout: {sparse_spec}")
    target = target.to(device=device, dtype=torch.float32)
    target.eval().requires_grad_(False)
    return teacher, target


def create_cache(target, capacity, cache_for, torch, pos, gpu_scalar):
    allocation_started = time.perf_counter()
    cache = cache_for(target, capacity)
    allocation_seconds = time.perf_counter() - allocation_started
    prompt = make_prompt(pos)
    prefix = prompt[:-1]
    idx = torch.tensor([prefix], device="cuda", dtype=torch.long)
    from draft_a.pf_ranker_eval import typed_kinds
    kinds = typed_kinds(0, len(prefix))
    prefill_started = time.perf_counter()
    target(idx, kv_cache=cache, logit_types=kinds)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_started
    if cache.get_pos() != pos:
        raise AssertionError(f"prefill ended at {cache.get_pos()}, expected {pos}")
    patch_fixed_position(cache, pos, gpu_scalar)
    return cache, prompt, dict(allocation_seconds=allocation_seconds,
                               prefill_seconds=prefill_seconds)


def make_forward(torch, target, cache, tap, tokens, pos, device):
    from draft_a.pf_ranker_eval import typed_kinds

    kinds = typed_kinds(pos, len(tokens))

    def run_eager():
        tap.output = None
        idx = torch.tensor([tokens], device=device, dtype=torch.long)
        logits = target(idx, kv_cache=cache, logit_types=kinds)
        if tap.output is None:
            raise RuntimeError("last-block output was not captured")
        return logits[0].detach(), tap.output[0]

    return run_eager


def make_typed_ids(torch, target, device):
    p_end = int(target.sparse_position_vocab_size)
    feature_start = (0 if target.sparse_shared_vocab
                     else int(target.sparse_feature_token_offset))
    feature_end = feature_start + int(target.sparse_feature_vocab_size)
    position_ids = torch.cat((
        torch.arange(0, p_end, device=device, dtype=torch.long),
        torch.tensor([int(target.sparse_eos_token_id)], device=device,
                     dtype=torch.long),
    ))
    feature_ids = torch.arange(feature_start, feature_end, device=device,
                                dtype=torch.long)
    return dict(position=position_ids, feature=feature_ids)


def typed_logits(torch, target, final_x, kinds, ids_by_kind):
    softcap = 15
    vocab_size = int(target.config.vocab_size)
    logits = torch.full((final_x.size(0), final_x.size(1), vocab_size),
                        -torch.inf, dtype=torch.float32, device=final_x.device)
    for row, kind in enumerate(kinds):
        token_ids = ids_by_kind[kind]
        selected_weight = target.lm_head.weight.index_select(0, token_ids)
        selected = torch.nn.functional.linear(final_x[:, row, :], selected_weight)
        selected = softcap * torch.tanh(selected.float() / softcap)
        logits[:, row, :].index_copy_(1, token_ids, selected)
    return logits


def target_trunk(torch, target, idx, cache, pos, rope, kinds, ids_by_kind, norm):
    x = target.transformer.wte(idx)
    if target.learned_pos_embed:
        positions = torch.arange(pos, pos + idx.size(1), device=idx.device)
        x = x + target.transformer.wpe(positions)
    x = norm(x)
    x0 = x
    for layer_index, block in enumerate(target.transformer.h):
        x = target.resid_lambdas[layer_index] * x + target.x0_lambdas[layer_index] * x0
        value_embed = (target.value_embeds[str(layer_index)](idx)
                       if str(layer_index) in target.value_embeds else None)
        x = block(x, value_embed, rope, target.window_sizes[layer_index], cache)
    pre_final_norm = x
    final_x = norm(x)
    logits = typed_logits(torch, target, final_x, kinds, ids_by_kind)
    return logits, pre_final_norm


def copy_tensor_tree(torch, destination, source):
    if torch.is_tensor(destination):
        if not torch.is_tensor(source) or tuple(destination.shape) != tuple(source.shape):
            raise RuntimeError("RoPE tensor shape changed for the captured block")
        destination.copy_(source)
        return
    if isinstance(destination, tuple):
        if not isinstance(source, tuple) or len(destination) != len(source):
            raise RuntimeError("RoPE tuple layout changed for the captured block")
        for left, right in zip(destination, source):
            copy_tensor_tree(torch, left, right)
        return
    if isinstance(destination, list):
        if not isinstance(source, list) or len(destination) != len(source):
            raise RuntimeError("RoPE list layout changed for the captured block")
        for left, right in zip(destination, source):
            copy_tensor_tree(torch, left, right)
        return
    raise TypeError(f"unsupported RoPE value: {type(destination).__name__}")


def clone_tensor_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tensor_tree(torch, item) for item in value)
    if isinstance(value, list):
        return [clone_tensor_tree(torch, item) for item in value]
    raise TypeError(f"unsupported RoPE value: {type(value).__name__}")


def capture_graph(torch, target, cache, tokens, pos, device, warmup_calls: int,
                  original_get_rope, original_slot_mask, original_mono_mask):
    from model.gpt.nanochat_gpt import norm
    from draft_a.pf_ranker_eval import typed_kinds

    preparation_started = time.perf_counter()
    kinds = typed_kinds(pos, len(tokens))
    ids_by_kind = make_typed_ids(torch, target, device)
    graph_input = torch.tensor([tokens], device=device, dtype=torch.long)
    graph_outputs = {}
    initial_rope = original_get_rope(graph_input, kv_cache=cache)
    static_rope = clone_tensor_tree(torch, initial_rope)
    cache.store_tokens(graph_input, pos)
    torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - preparation_started

    def update_static_inputs():
        source = torch.tensor([tokens], device=device, dtype=torch.long)
        graph_input.copy_(source)
        rope_value = original_get_rope(graph_input, kv_cache=cache)
        copy_tensor_tree(torch, static_rope, rope_value)
        cache.store_tokens(graph_input, pos)

    def capture_forward():
        return target_trunk(torch, target, graph_input, cache, pos,
                            static_rope, kinds, ids_by_kind, norm)

    current = torch.cuda.current_stream(device)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(current)
    warmup_started = time.perf_counter()
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(warmup_calls):
            update_static_inputs()
            capture_forward()
    current.wait_stream(warmup_stream)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(device)
    capture_started = time.perf_counter()
    with torch.inference_mode(), torch.cuda.graph(graph):
        graph_outputs["logits"], graph_outputs["hidden"] = capture_forward()
    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - capture_started

    def run_graph():
        update_static_inputs()
        graph.replay()
        logits = graph_outputs["logits"]
        original_slot_mask(logits, start_pos=pos)
        original_mono_mask(logits, idx=graph_input, start_pos=pos, kv_cache=cache)
        return logits[0], graph_outputs["hidden"][0]

    graph_state = dict(capture_forward=capture_forward, graph_input=graph_input,
                       static_rope=static_rope, ids_by_kind=ids_by_kind,
                       kinds=kinds, cache=cache, target=target)
    return (graph, run_graph, graph_outputs, graph_state, preparation_seconds,
            warmup_seconds, capture_seconds)


def timed_calls(torch, fn, calls: int, device) -> dict:
    torch.cuda.synchronize(device)
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_started = time.perf_counter()
    begin_event.record(torch.cuda.current_stream(device))
    for _ in range(calls):
        fn()
    end_event.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    host_seconds = time.perf_counter() - host_started
    cuda_seconds = begin_event.elapsed_time(end_event) / 1000.0
    return dict(host_ms_per_call=host_seconds * 1000.0 / calls,
                cuda_ms_per_call=cuda_seconds * 1000.0 / calls,
                host_minus_cuda_ms_per_call=(host_seconds - cuda_seconds) * 1000.0 / calls)


def summarize_pairs(pair_rows: list[dict]) -> list[dict]:
    by_shape = defaultdict(lambda: defaultdict(list))
    for row in pair_rows:
        key = (row["context_length"], row["query_tokens"])
        by_shape[key][row["mode"]].append(row)
    comparisons = (
        ("sdpa_gpu_pos", "sdpa_host_pos", "host_position"),
        ("sdpa_gpu_pos", "explicit_host_pos", "explicit_attention"),
        ("sdpa_gpu_pos", "graph_explicit_host_pos", "graph_explicit"),
        ("explicit_host_pos", "graph_explicit_host_pos", "graph_over_explicit"),
    )
    summaries = []
    for (ctx, shape), modes in sorted(by_shape.items()):
        rounds = sorted(set.intersection(*[
            {row["pair_index"] for row in modes.get(mode, [])}
            for mode in ("sdpa_gpu_pos", "sdpa_host_pos", "explicit_host_pos",
                         "graph_explicit_host_pos")
        ]))
        item = dict(context_length=ctx, query_tokens=shape, paired_rounds=len(rounds))
        for reference_name, candidate_name, label in comparisons:
            reference_values, candidate_values, speed_ratios = [], [], []
            for pair_index in rounds:
                reference = next(row for row in modes[reference_name]
                                 if row["pair_index"] == pair_index)
                candidate = next(row for row in modes[candidate_name]
                                 if row["pair_index"] == pair_index)
                reference_ms = reference["host_ms_per_call"]
                candidate_ms = candidate["host_ms_per_call"]
                reference_values.append(reference_ms)
                candidate_values.append(candidate_ms)
                speed_ratios.append(reference_ms / candidate_ms)
            item[label] = dict(
                reference_mode=reference_name,
                candidate_mode=candidate_name,
                reference_ms_per_call_median=(statistics.median(reference_values)
                                              if reference_values else None),
                candidate_ms_per_call_median=(statistics.median(candidate_values)
                                              if candidate_values else None),
                speed_ratio_median=(statistics.median(speed_ratios)
                                    if speed_ratios else None),
                speed_ratio_mean=(statistics.mean(speed_ratios)
                                  if speed_ratios else None),
            )
        for mode in MODES:
            rows = [row for row in modes[mode] if row["pair_index"] in rounds]
            for metric in ("host_ms_per_call", "cuda_ms_per_call",
                           "host_minus_cuda_ms_per_call"):
                vals = [row[metric] for row in rows]
                item[mode + "_" + metric + "_median"] = statistics.median(vals) if vals else None
        summaries.append(item)
    return summaries


def run_experiment(args) -> dict:
    import torch
    from draft_a.pf_ranker_eval import cache_for, policy
    import model.gpt.flash_attention as flash_mod
    import model.gpt.nanochat_gpt as gpt_mod

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_precision = (torch.get_float32_matmul_precision()
                     if hasattr(torch, "get_float32_matmul_precision") else None)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    pair_path = args.out_dir / "pairs.jsonl"
    correctness_path = args.out_dir / "correctness.jsonl"

    result = dict(status="starting", experiment="pf_target_fixed_position_graph",
                  target=str(args.teacher), vqvae=str(args.vqvae), device=str(device),
                  torch_version=torch.__version__,
                  contexts=list(args.contexts), shapes=list(args.shapes),
                  warmup_calls=args.warmup_calls, pair_repeats=args.pair_repeats,
                  calls_per_pair=args.calls_per_pair,
                  modes=list(MODES), load_seconds=None,
                  tolerances=dict(logit_atol=LOGIT_ATOL, logit_rtol=LOGIT_RTOL,
                                  hidden_atol=HIDDEN_ATOL, hidden_rtol=HIDDEN_RTOL,
                                  policy_tv_max=POLICY_TV_MAX),
                  caveat="fixed input and cache position; graph result is a forward microbench")
    write_json(args.out_dir / "metadata.json", result)

    started = time.perf_counter()
    try:
        teacher, target = load_target(args, torch, device)
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - started
        if next(target.parameters()).dtype != torch.float32:
            raise RuntimeError("Target must run in FP32")
    except Exception as error:
        result.update(status="failed", load_seconds=time.perf_counter() - started,
                      error=f"{type(error).__name__}: {error}")
        write_json(args.out_dir / "metadata.json", result)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        if old_precision is not None:
            torch.set_float32_matmul_precision(old_precision)
        raise

    original_impl = flash_mod._override_impl
    original_attention = gpt_mod.flash_attn.flash_attn_with_kvcache
    original_get_rope = target._get_rope
    original_slot_mask = target._apply_slot_logits_mask
    original_mono_mask = target._apply_monotonic_position_mask
    tap = LastBlockTap(target)
    result.update(status="running", load_seconds=load_seconds,
                  device_name=torch.cuda.get_device_name(device),
                  target_dtype=str(next(target.parameters()).dtype),
                  target_window_sizes=[list(window) for window in target.window_sizes])
    write_json(args.out_dir / "metadata.json", result)

    try:
        flash_mod._override_impl = "sdpa"
        with torch.inference_mode():
            for context_index, ctx in enumerate(args.contexts):
                capacity = ctx + max(args.shapes) + 2
                gpu_cache, prompt, gpu_prep = create_cache(target, capacity, cache_for,
                                                            torch, ctx, True)
                host_cache, _, host_prep = create_cache(target, capacity, cache_for,
                                                        torch, ctx, False)
                host_position_attention = make_host_position_attention(
                    torch, original_attention, ctx)
                result.setdefault("cache_prep", []).append(
                    dict(context_length=ctx, gpu_position=gpu_prep,
                         host_position=host_prep,
                         gpu_cache_seqlens_device=str(gpu_cache.cache_seqlens.device),
                         host_cache_seqlens_device=str(host_cache.cache_seqlens.device),
                         host_attention_position_device="cpu"))
                for shape_index, shape in enumerate(args.shapes):
                    p1 = min(P_VOCAB - 1, ctx // 2)
                    f1 = F_OFFSET + (ctx // 2 % 4096)
                    tokens = make_block(prompt, shape, p1, f1)
                    if len(tokens) != shape:
                        raise AssertionError("generated block has the wrong length")
                    kinds = typed_kinds(ctx, shape)
                    if len(kinds) != shape:
                        raise AssertionError("typed rows do not match block length")

                    gpu_forward = make_forward(torch, target, gpu_cache, tap,
                                               tokens, ctx, device)
                    host_forward = make_forward(torch, target, host_cache, tap,
                                                tokens, ctx, device)
                    gpu_output = gpu_forward()
                    gpt_mod.flash_attn.flash_attn_with_kvcache = host_position_attention
                    host_output = host_forward()
                    gpt_mod.flash_attn.flash_attn_with_kvcache = original_attention
                    gpu_compare = dict(
                        logits=compare_tensors(torch, gpu_output[0], host_output[0],
                                               LOGIT_ATOL, LOGIT_RTOL),
                        hidden=compare_tensors(torch, gpu_output[1], host_output[1],
                                               HIDDEN_ATOL, HIDDEN_RTOL),
                        policy_tv=policy_tv(torch, gpu_output[0], host_output[0], policy),
                    )
                    output_path = args.out_dir / f"outputs_ctx{ctx}_T{shape}.pt"
                    save_output_tensors(
                        torch, output_path, ctx, shape, tokens, kinds,
                        dict(sdpa_gpu_pos=gpu_output, sdpa_host_pos=host_output),
                    )
                    if not gpu_compare["logits"]["passed"] or not gpu_compare["hidden"]["passed"] or gpu_compare["policy_tv"] > POLICY_TV_MAX:
                        row = dict(context_length=ctx, query_tokens=shape,
                                   cache_position=ctx, input_tokens=tokens,
                                   logit_types=kinds,
                                   comparison="gpu_position_vs_host_position", **gpu_compare)
                        append_jsonl(correctness_path, row)
                        raise RuntimeError(f"position mirror differs at ctx={ctx}, T={shape}")

                    mask_started = time.perf_counter()
                    explicit_impl = make_explicit_kvcache(torch, ctx, shape, device)
                    torch.cuda.synchronize(device)
                    explicit_mask_prepare_seconds = time.perf_counter() - mask_started
                    gpt_mod.flash_attn.flash_attn_with_kvcache = explicit_impl
                    explicit_forward = make_forward(torch, target, host_cache, tap,
                                                    tokens, ctx, device)
                    explicit_output = explicit_forward()
                    explicit_compare = dict(
                        logits=compare_tensors(torch, host_output[0], explicit_output[0],
                                               LOGIT_ATOL, LOGIT_RTOL),
                        hidden=compare_tensors(torch, host_output[1], explicit_output[1],
                                               HIDDEN_ATOL, HIDDEN_RTOL),
                        policy_tv=policy_tv(torch, host_output[0], explicit_output[0], policy),
                    )

                    (graph, graph_forward, graph_outputs, graph_state, graph_prepare_seconds,
                     graph_warmup_seconds, capture_seconds) = capture_graph(
                        torch, target, host_cache, tokens, ctx, device,
                        args.warmup_calls, original_get_rope,
                        original_slot_mask, original_mono_mask)
                    graph_state["attention_impl"] = explicit_impl
                    graph_state["attention_mask"] = explicit_impl.mask
                    graph_output = graph_forward()
                    torch.cuda.synchronize(device)
                    graph_compare = dict(
                        logits=compare_tensors(torch, explicit_output[0], graph_output[0],
                                               LOGIT_ATOL, LOGIT_RTOL),
                        hidden=compare_tensors(torch, explicit_output[1], graph_output[1],
                                               HIDDEN_ATOL, HIDDEN_RTOL),
                        policy_tv=policy_tv(torch, explicit_output[0], graph_output[0], policy),
                    )
                    save_output_tensors(
                        torch, output_path, ctx, shape, tokens, kinds,
                        dict(sdpa_gpu_pos=gpu_output, sdpa_host_pos=host_output,
                             explicit_host_pos=explicit_output,
                             graph_explicit_host_pos=graph_output),
                    )
                    correct = (
                        explicit_compare["logits"]["passed"]
                        and explicit_compare["hidden"]["passed"]
                        and explicit_compare["policy_tv"] <= POLICY_TV_MAX
                        and graph_compare["logits"]["passed"]
                        and graph_compare["hidden"]["passed"]
                        and graph_compare["policy_tv"] <= POLICY_TV_MAX
                    )
                    correctness = dict(context_length=ctx, query_tokens=shape,
                                       cache_position=ctx, input_tokens=tokens,
                                       logit_types=kinds,
                                       gpu_position_vs_host_position=gpu_compare,
                                       sdpa_vs_explicit=explicit_compare,
                                       explicit_eager_vs_graph=graph_compare,
                                       tensor_artifact=str(output_path),
                                       passed=bool(correct))
                    append_jsonl(correctness_path, correctness)
                    result.setdefault("correctness", []).append(correctness)
                    write_json(args.out_dir / "metadata.json", result)
                    if not correct:
                        raise RuntimeError(f"numeric tolerance exceeded at ctx={ctx}, T={shape}")

                    def gpu_mode():
                        gpt_mod.flash_attn.flash_attn_with_kvcache = original_attention
                        return gpu_forward()

                    def host_mode():
                        gpt_mod.flash_attn.flash_attn_with_kvcache = host_position_attention
                        return host_forward()

                    def explicit_mode():
                        gpt_mod.flash_attn.flash_attn_with_kvcache = explicit_impl
                        return explicit_forward()

                    def graph_mode():
                        gpt_mod.flash_attn.flash_attn_with_kvcache = explicit_impl
                        return graph_forward()

                    functions = dict(sdpa_gpu_pos=gpu_mode, sdpa_host_pos=host_mode,
                                     explicit_host_pos=explicit_mode,
                                     graph_explicit_host_pos=graph_mode)
                    mode_warmup_seconds = {}
                    for mode, fn in functions.items():
                        warmup_started = time.perf_counter()
                        for _ in range(args.warmup_calls):
                            fn()
                        torch.cuda.synchronize(device)
                        mode_warmup_seconds[mode] = time.perf_counter() - warmup_started
                    for pair_index in range(args.pair_repeats):
                        order = pair_order(pair_index, context_index, shape_index)
                        for order_index, mode in enumerate(order):
                            timing = timed_calls(torch, functions[mode],
                                                 args.calls_per_pair, device)
                            row = dict(context_length=ctx, query_tokens=shape,
                                       pair_index=pair_index, mode=mode,
                                       cache_position=ctx, input_tokens=tokens,
                                       logit_types=kinds,
                                       order_index=order_index, order=order,
                                       calls=args.calls_per_pair,
                                       warmup_seconds=mode_warmup_seconds[mode],
                                       graph_capture_warmup_seconds=graph_warmup_seconds,
                                       graph_prepare_seconds=graph_prepare_seconds,
                                       explicit_mask_prepare_seconds=explicit_mask_prepare_seconds,
                                       capture_seconds=capture_seconds if mode == "graph_explicit_host_pos" else 0.0,
                                       graph_input_copy_included=(mode == "graph_explicit_host_pos"),
                                       **timing)
                            append_jsonl(pair_path, row)
                    result.setdefault("shapes_completed", []).append(
                        dict(context_length=ctx, query_tokens=shape,
                             mode_warmup_seconds=mode_warmup_seconds,
                             explicit_mask_prepare_seconds=explicit_mask_prepare_seconds,
                             graph_prepare_seconds=graph_prepare_seconds,
                             graph_capture_warmup_seconds=graph_warmup_seconds,
                             capture_seconds=capture_seconds,
                             shared_mask_shape=list(explicit_impl.mask.shape)))
                    write_json(args.out_dir / "metadata.json", result)
                    del graph, graph_forward, graph_outputs, graph_state, gpu_output, host_output
                    del explicit_output, graph_output, explicit_impl
                    gpt_mod.flash_attn.flash_attn_with_kvcache = original_attention
                del gpu_cache, host_cache
                gc.collect()
                torch.cuda.empty_cache()
        pair_rows = [json.loads(line) for line in pair_path.read_text().splitlines() if line.strip()]
        result["paired_summary"] = summarize_pairs(pair_rows)
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        tap.close()
        gpt_mod.flash_attn.flash_attn_with_kvcache = original_attention
        target._get_rope = original_get_rope
        target._apply_slot_logits_mask = original_slot_mask
        target._apply_monotonic_position_mask = original_mono_mask
        flash_mod._override_impl = original_impl
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        if old_precision is not None:
            torch.set_float32_matmul_precision(old_precision)
        write_json(args.out_dir / "metadata.json", result)
    return result


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    result = run_experiment(args)
    print(json.dumps(dict(status=result["status"],
                          output=str(args.out_dir / "metadata.json"),
                          shapes=len(result.get("shapes_completed", []))),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
