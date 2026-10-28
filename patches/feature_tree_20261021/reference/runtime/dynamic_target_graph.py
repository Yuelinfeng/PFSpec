"""Experimental dynamic-position CUDA Graph wrapper for GaussianGPT Target."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import math
import time
from types import MethodType
from typing import Iterable


MAX_SUPPORTED_QUERY_TOKENS = 8
SUPPORTED_QUERY_TOKENS = tuple(range(1, MAX_SUPPORTED_QUERY_TOKENS + 1))
SUPPORTED_LOGIT_TYPES = ("position", "feature", "full")
_MISSING = object()


def _with_host_position(cache, pos: int, operation):
    previous = cache.__dict__.get("get_pos", _MISSING)
    cache.get_pos = MethodType(lambda self: int(pos), cache)
    try:
        return operation()
    finally:
        if previous is _MISSING:
            delattr(cache, "get_pos")
        else:
            cache.get_pos = previous


def power_of_two_buckets(maximum: int, minimum: int = 8) -> tuple[int, ...]:
    """Return read or cache capacities rounded to powers of two."""
    maximum = int(maximum)
    minimum = int(minimum)
    if maximum < 1 or minimum < 1:
        raise ValueError("bucket limits must be positive")
    values = []
    value = min(minimum, maximum)
    while value < maximum:
        values.append(value)
        value *= 2
    values.append(maximum)
    return tuple(sorted(set(values)))


def _ceil_bucket(value: int, buckets: Iterable[int]) -> int | None:
    for bucket in buckets:
        if int(bucket) >= int(value):
            return int(bucket)
    return None


def _copy_tree(torch, destination, source) -> None:
    if destination is None and source is None:
        return
    if torch.is_tensor(destination):
        if not torch.is_tensor(source) or tuple(destination.shape) != tuple(source.shape):
            raise RuntimeError("RoPE shape differs from the stored graph inputs")
        destination.copy_(source)
        return
    if isinstance(destination, tuple):
        if not isinstance(source, tuple) or len(destination) != len(source):
            raise RuntimeError("RoPE tuple layout differs from the stored graph inputs")
        for left, right in zip(destination, source):
            _copy_tree(torch, left, right)
        return
    if isinstance(destination, list):
        if not isinstance(source, list) or len(destination) != len(source):
            raise RuntimeError("RoPE list layout differs from the stored graph inputs")
        for left, right in zip(destination, source):
            _copy_tree(torch, left, right)
        return
    raise TypeError(f"unsupported RoPE value {type(destination).__name__}")


def _clone_tree(torch, value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(torch, item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(torch, item) for item in value]
    raise TypeError(f"unsupported RoPE value {type(value).__name__}")


class DynamicTargetGraph:
    """Wrap one GPT core with lazy graphs keyed by stable cache, shape and read bucket.

    The wrapper is deliberately limited to single-batch inference with up to
    ``max_query_tokens`` rows and the Target's full or typed sparse readouts.
    Unsupported calls use the original forward method.
    Call ``cache_for`` from the same location as the evaluator's imported
    ``cache_for`` function so graph replays keep stable KV tensor addresses.
    """

    def __init__(
        self,
        target,
        *,
        read_buckets=None,
        cache_capacity_buckets=None,
        max_query_tokens: int = MAX_SUPPORTED_QUERY_TOKENS,
        warmup_calls: int = 2,
        max_graphs: int = 128,
        record_outputs: bool = False,
    ):
        started = time.perf_counter()
        self.target = target
        self.max_query_tokens = int(max_query_tokens)
        if not 1 <= self.max_query_tokens <= 16:
            raise ValueError("max_query_tokens must be between 1 and 16")
        self.supported_query_tokens = tuple(range(1, self.max_query_tokens + 1))
        config = getattr(target, "config", None)
        max_sequence = int(getattr(config, "sequence_len", 16384))
        self.read_buckets = tuple(sorted(set(int(x) for x in (
            read_buckets or power_of_two_buckets(max_sequence, minimum=8)
        ))))
        self.cache_capacity_buckets = tuple(sorted(set(int(x) for x in (
            cache_capacity_buckets or power_of_two_buckets(max_sequence, minimum=256)
        ))))
        self.warmup_calls = max(1, int(warmup_calls))
        self.max_graphs = max(1, int(max_graphs))
        self.record_outputs = bool(record_outputs)
        self.enabled = False
        self._context_active = False
        self._original_forward = None
        self._instance_forward_existed = False
        self._instance_forward = None
        self._explicit_hidden_capture = None
        self._cache_pool = {}
        self._graphs = {}
        self._disabled_graphs = {}
        self._weight_views = None
        self._numerical_records = []
        self._calls = Counter()
        self._fallbacks = Counter()
        self._setup = dict(
            runtime_init_ms=(time.perf_counter() - started) * 1000.0,
            cache_events=[],
            graph_events=[],
        )

    def set_enabled(self, enabled: bool) -> bool:
        previous = self.enabled
        self.enabled = bool(enabled)
        return previous

    @contextmanager
    def attention_context(self, enabled: bool = True, hidden_capture=None):
        """Temporarily route eligible Target calls through this runtime."""
        if self._context_active:
            raise RuntimeError("this runtime already has an active forward context")
        self._context_active = True
        self._explicit_hidden_capture = hidden_capture
        previous_enabled = self.set_enabled(enabled)
        instance_dict = self.target.__dict__
        self._instance_forward_existed = "forward" in instance_dict
        if self._instance_forward_existed:
            self._instance_forward = instance_dict["forward"]
        self._original_forward = self.target.forward

        def wrapped_forward(idx, *args, **kwargs):
            return self._dispatch(idx, args, kwargs)

        self.target.forward = wrapped_forward
        try:
            yield self
        finally:
            if self._instance_forward_existed:
                self.target.forward = self._instance_forward
            else:
                delattr(self.target, "forward")
            self._original_forward = None
            self._instance_forward = None
            self._explicit_hidden_capture = None
            self.enabled = previous_enabled
            self._context_active = False

    def cache_for(self, core, requested_capacity: int, *, factory):
        """Return a pooled Target KVCache; use the supplied factory for other models."""
        requested_capacity = int(requested_capacity)
        if core is not self.target:
            return factory(core, requested_capacity)
        capacity = _ceil_bucket(requested_capacity, self.cache_capacity_buckets)
        if capacity is None:
            capacity = requested_capacity
        cache = self._cache_pool.get(capacity)
        if cache is None:
            started = time.perf_counter()
            cache = factory(core, capacity)
            elapsed = (time.perf_counter() - started) * 1000.0
            self._cache_pool[capacity] = cache
            self._setup["cache_events"].append(dict(
                event="allocate",
                requested_capacity=requested_capacity,
                cache_capacity=capacity,
                host_enqueue_ms=elapsed,
                kv_bytes=self._kv_bytes(cache),
                cache=cache,
            ))
            return cache

        started = time.perf_counter()
        cache.reset()
        elapsed = (time.perf_counter() - started) * 1000.0
        self._setup["cache_events"].append(dict(
            event="reset",
            requested_capacity=requested_capacity,
            cache_capacity=capacity,
            host_enqueue_ms=elapsed,
            kv_bytes=self._kv_bytes(cache),
            cache=cache,
        ))
        return cache

    def _kv_bytes(self, cache) -> int | None:
        try:
            return int(cache.k_cache.numel() * cache.k_cache.element_size()
                       + cache.v_cache.numel() * cache.v_cache.element_size())
        except Exception:
            return None

    def _model_cache_bytes(self, capacity: int, batch: int, dtype_size: int) -> int | None:
        config = getattr(self.target, "config", None)
        if config is None:
            return None
        try:
            layers = int(config.n_layer)
            kv_heads = int(config.n_kv_head)
            head_dim = int(config.n_embd) // int(config.n_head)
            return 2 * layers * batch * capacity * kv_heads * head_dim * dtype_size
        except Exception:
            return None

    def stats(self) -> dict:
        """Return setup detail, graph shapes, fallbacks and optional raw tensors."""
        def serial_event(event):
            return {key: value for key, value in event.items() if key != "cache"}

        return dict(
            runtime_init_ms=self._setup["runtime_init_ms"],
            max_query_tokens=self.max_query_tokens,
            supported_query_tokens=list(self.supported_query_tokens),
            cache_events=[serial_event(item) for item in self._setup["cache_events"]],
            graph_events=[copy.deepcopy(item) for item in self._setup["graph_events"]],
            graph_count=len(self._graphs),
            disabled_graphs={repr(key): value
                             for key, value in self._disabled_graphs.items()},
            calls=dict(self._calls),
            fallbacks=dict(self._fallbacks),
            numerical_records=len(self._numerical_records),
            numerical_recording=self.record_outputs,
            cache_bytes_by_capacity={str(cap): self._kv_bytes(cache)
                                     for cap, cache in self._cache_pool.items()},
        )

    def stats_snapshot(self) -> dict:
        return self.stats()

    def reset_call_counters(self) -> None:
        self._calls.clear()
        self._fallbacks.clear()

    def save_numerical_records(self, path) -> None:
        import torch

        torch.save(dict(records=self._numerical_records, stats=self.stats()), path)

    def _record(self, record: dict) -> None:
        if self.record_outputs:
            self._numerical_records.append(record)

    def _fallback(self, reason: str, idx, args, kwargs):
        self._fallbacks[reason] += 1
        self._calls["original_forward"] += 1
        return self._original_forward(idx, *args, **kwargs)

    def _dispatch(self, idx, args, kwargs):
        if not self.enabled:
            self._calls["disabled_forward"] += 1
            return self._original_forward(idx, *args, **kwargs)
        import torch

        if self.target.training:
            return self._fallback("training", idx, args, kwargs)
        if torch.is_grad_enabled():
            return self._fallback("grad_enabled", idx, args, kwargs)
        if args:
            return self._fallback("positional_optional_arguments", idx, args, kwargs)
        if kwargs.get("targets") is not None:
            return self._fallback("training_targets", idx, args, kwargs)
        cache = kwargs.get("kv_cache")
        if cache is None:
            return self._fallback("no_kv_cache_prefill", idx, args, kwargs)
        if not torch.is_tensor(idx) or idx.ndim != 2 or idx.size(0) != 1:
            return self._fallback("unsupported_input_shape", idx, args, kwargs)
        query_tokens = int(idx.size(1))
        if query_tokens not in self.supported_query_tokens:
            return self._fallback("unsupported_query_tokens", idx, args, kwargs)
        if idx.dtype != torch.long:
            return self._fallback("input_dtype", idx, args, kwargs)
        kinds = kwargs.get("logit_types")
        if kinds is not None:
            if len(kinds) != query_tokens:
                return self._fallback("logit_types_length", idx, args, kwargs)
            kinds = tuple(str(kind) for kind in kinds)
        if kinds is not None and any(kind not in SUPPORTED_LOGIT_TYPES for kind in kinds):
            return self._fallback("unsupported_logit_types", idx, args, kwargs)
        if cache.batch_size != 1 or cache.k_cache.device != idx.device:
            return self._fallback("cache_device_or_batch", idx, args, kwargs)
        if idx.device.type != "cuda":
            return self._fallback("non_cuda_device", idx, args, kwargs)

        try:
            pos = int(cache.get_pos())
        except Exception as error:
            self._record(dict(mode="fallback", reason="position_read_error",
                              error=f"{type(error).__name__}: {error}"))
            return self._fallback("position_read_error", idx, args, kwargs)
        end_pos = pos + query_tokens
        if end_pos > int(cache.max_seq_len):
            return self._fallback("cache_capacity_overflow", idx, args, kwargs)
        read_bucket = _ceil_bucket(end_pos, self.read_buckets)
        if read_bucket is None or read_bucket > int(cache.max_seq_len):
            return self._fallback("read_bucket_unavailable", idx, args, kwargs)
        if read_bucket == query_tokens:
            for window in getattr(self.target, "window_sizes", ()):
                if window[0] >= 0 and window[0] < query_tokens:
                    return self._fallback("unsupported_window_for_full_block",
                                          idx, args, kwargs)

        graph_key = (id(cache), query_tokens, read_bucket, kinds)
        if graph_key in self._disabled_graphs:
            return self._fallback("disabled_graph_shape", idx, args, kwargs)

        try:
            state = self._graphs.get(graph_key)
            if state is None:
                if len(self._graphs) >= self.max_graphs:
                    return self._fallback("graph_limit", idx, args, kwargs)
                state = self._build_graph(cache, idx, pos, read_bucket, kinds)
                self._graphs[graph_key] = state
            result = self._run_graph(state, cache, idx, pos, kinds, kwargs)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self._disabled_graphs[graph_key] = message
            self._record(dict(mode="fallback", reason="graph_error", pos=pos,
                              query_tokens=query_tokens, read_bucket=read_bucket,
                              logit_types=None if kinds is None else list(kinds), error=message))
            text = str(error).lower()
            if "device-side assert" in text or "illegal memory access" in text:
                raise
            try:
                torch.cuda.synchronize(idx.device)
            except Exception:
                raise
            return self._fallback("graph_error", idx, args, kwargs)
        state["advance"](state["query_tokens"])
        self._calls["graph_forward"] += 1
        self._calls[f"graph_forward_T{query_tokens}"] += 1
        if kinds is None or any(kind == "full" for kind in kinds):
            self._calls["graph_full_forward"] += 1
        if kwargs.get("return_hidden", False):
            self._calls["graph_return_hidden"] += 1
        return result

    def _make_weight_views(self, torch, device):
        if self._weight_views is not None:
            return self._weight_views
        target = self.target
        position_end = int(target.sparse_position_vocab_size)
        feature_start = (0 if bool(target.sparse_shared_vocab)
                         else int(target.sparse_feature_token_offset))
        feature_end = feature_start + int(target.sparse_feature_vocab_size)
        position_ids = torch.cat((
            torch.arange(position_end, device=device, dtype=torch.long),
            torch.tensor([int(target.sparse_eos_token_id)], device=device,
                         dtype=torch.long),
        ))
        feature_ids = torch.arange(feature_start, feature_end, device=device,
                                    dtype=torch.long)
        position_weight = target.lm_head.weight.index_select(0, position_ids)
        feature_weight = target.lm_head.weight.index_select(0, feature_ids)
        self._weight_views = dict(
            ids=dict(position=position_ids, feature=feature_ids),
            weights=dict(position=position_weight, feature=feature_weight),
            target=target,
        )
        return self._weight_views

    def _typed_logits(self, torch, final_x, kinds, views):
        import torch.nn.functional as F

        target = self.target
        vocab_size = int(target.config.vocab_size)
        if kinds is None or any(kind == "full" for kind in kinds):
            logits = target.lm_head(final_x)[..., :vocab_size].float()
            return 15.0 * torch.tanh(logits / 15.0)
        logits = torch.full((final_x.size(0), final_x.size(1), vocab_size),
                            -torch.inf, dtype=torch.float32, device=final_x.device)
        for row, kind in enumerate(kinds):
            selected = F.linear(final_x[:, row, :], views["weights"][kind])
            selected = 15.0 * torch.tanh(selected.float() / 15.0)
            logits[:, row, :].index_copy_(1, views["ids"][kind], selected)
        return logits

    def _make_attention(self, torch, pos_tensor, offsets, key_positions,
                        read_bucket: int, query_tokens: int):
        def explicit_attention(q, k_cache, v_cache, k=None, v=None,
                               cache_seqlens=None, causal=False,
                               window_size=(-1, -1)):
            if k is None or v is None or not causal:
                raise RuntimeError("dynamic graph attention requires causal K/V inputs")
            batch, actual_tokens, q_heads, width = q.shape
            kv_heads = int(k_cache.size(2))
            if (batch != 1 or actual_tokens != query_tokens or
                    q_heads % kv_heads != 0 or read_bucket > k_cache.size(1)):
                raise RuntimeError("attention dimensions differ from the captured graph")

            query_positions = pos_tensor.to(dtype=torch.long) + offsets
            k_cache.index_copy_(1, query_positions, k)
            v_cache.index_copy_(1, query_positions, v)
            k_full = k_cache[:, :read_bucket, :, :].transpose(1, 2)
            v_full = v_cache[:, :read_bucket, :, :].transpose(1, 2)
            groups = q_heads // kv_heads
            q_grouped = q.transpose(1, 2).reshape(
                batch, kv_heads, groups, query_tokens, width
            )

            # The math SDPA path scales Q and K separately before their product.
            scale = width ** -0.25
            q_scaled = q_grouped * scale
            k_scaled = k_full * scale
            scores = torch.matmul(
                q_scaled, k_scaled.transpose(-1, -2).unsqueeze(2)
            )
            allow = (key_positions.view(1, 1, 1, 1, read_bucket)
                     <= query_positions.view(1, 1, 1, query_tokens, 1))
            scores = scores.masked_fill(~allow, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            result = torch.matmul(probs, v_full.unsqueeze(2))
            return result.reshape(batch, q_heads, query_tokens, width).transpose(1, 2)

        return explicit_attention

    def _trunk(self, torch, idx, cache, pos_tensor, offsets, rope, kinds,
               weight_views, attention_impl):
        from model.gpt.nanochat_gpt import norm
        import model.gpt.nanochat_gpt as gpt_module

        x = self.target.transformer.wte(idx)
        if self.target.learned_pos_embed:
            x = x + self.target.transformer.wpe(
                pos_tensor.to(dtype=torch.long) + offsets
            )
        x = norm(x)
        x0 = x
        old_attention = gpt_module.flash_attn.flash_attn_with_kvcache
        gpt_module.flash_attn.flash_attn_with_kvcache = attention_impl
        try:
            for layer_index, block in enumerate(self.target.transformer.h):
                x = (self.target.resid_lambdas[layer_index] * x
                     + self.target.x0_lambdas[layer_index] * x0)
                value_embed = (
                    self.target.value_embeds[str(layer_index)](idx)
                    if str(layer_index) in self.target.value_embeds else None
                )
                x = block(x, value_embed, rope,
                          self.target.window_sizes[layer_index], cache)
        finally:
            gpt_module.flash_attn.flash_attn_with_kvcache = old_attention
        pre_final_norm = x
        post_final_norm = norm(x)
        logits = self._typed_logits(torch, post_final_norm, kinds, weight_views)
        return logits, pre_final_norm, post_final_norm

    def _build_graph(self, cache, idx, pos: int, read_bucket: int, kinds):
        import torch
        import model.gpt.nanochat_gpt as gpt_module

        device = idx.device
        query_tokens = int(idx.size(1))
        prep_started = time.perf_counter()
        rope_value = _with_host_position(
            cache, pos, lambda: self.target._get_rope(idx, kv_cache=cache)
        )
        static_idx = torch.empty_like(idx)
        static_idx.copy_(idx)
        static_rope = _clone_tree(torch, rope_value)
        pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
        offsets = torch.arange(query_tokens, dtype=torch.long, device=device)
        key_positions = torch.arange(read_bucket, dtype=torch.long, device=device)
        weight_views = self._make_weight_views(torch, device)
        attention_impl = self._make_attention(
            torch, pos_tensor, offsets, key_positions, read_bucket, query_tokens
        )
        cache.store_tokens(static_idx, pos)
        torch.cuda.synchronize(device)
        prepare_ms = (time.perf_counter() - prep_started) * 1000.0

        def update_inputs(new_idx, new_pos, new_rope):
            static_idx.copy_(new_idx)
            pos_tensor.fill_(int(new_pos))
            _copy_tree(torch, static_rope, new_rope)
            cache.store_tokens(static_idx, int(new_pos))

        def graph_forward():
            return self._trunk(torch, static_idx, cache, pos_tensor, offsets,
                               static_rope, kinds, weight_views, attention_impl)

        advance_bound = cache.advance
        old_advance = cache.__dict__.get("advance", None)
        had_instance_advance = "advance" in cache.__dict__
        cache.advance = MethodType(lambda self, count: None, cache)
        current = torch.cuda.current_stream(device)
        warm_stream = torch.cuda.Stream(device=device)
        warm_stream.wait_stream(current)
        warm_started = time.perf_counter()
        try:
            with torch.cuda.stream(warm_stream), torch.inference_mode():
                for _ in range(self.warmup_calls):
                    update_inputs(idx, pos, rope_value)
                    graph_forward()
            current.wait_stream(warm_stream)
            torch.cuda.synchronize(device)
            warmup_ms = (time.perf_counter() - warm_started) * 1000.0

            memory_before = int(torch.cuda.memory_allocated(device))
            reserved_before = int(torch.cuda.memory_reserved(device))
            graph = torch.cuda.CUDAGraph()
            capture_started = time.perf_counter()
            with torch.inference_mode(), torch.cuda.graph(graph):
                graph_outputs = graph_forward()
            torch.cuda.synchronize(device)
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            memory_after = int(torch.cuda.memory_allocated(device))
            reserved_after = int(torch.cuda.memory_reserved(device))
        finally:
            if had_instance_advance:
                cache.advance = old_advance
            else:
                delattr(cache, "advance")

        state = dict(
            graph=graph,
            graph_outputs=graph_outputs,
            static_idx=static_idx,
            pos_tensor=pos_tensor,
            offsets=offsets,
            key_positions=key_positions,
            static_rope=static_rope,
            rope_source=rope_value,
            attention_impl=attention_impl,
            weight_views=weight_views,
            kinds=None if kinds is None else tuple(kinds),
            cache=cache,
            target=self.target,
            graph_forward=graph_forward,
            read_bucket=read_bucket,
            query_tokens=query_tokens,
            advance=advance_bound,
        )
        self._setup["graph_events"].append(dict(
            cache_capacity=int(cache.max_seq_len),
            query_tokens=query_tokens,
            read_bucket=read_bucket,
            logit_types=None if kinds is None else list(kinds),
            first_pos=int(pos),
            prepare_ms=prepare_ms,
            warmup_ms=warmup_ms,
            capture_ms=capture_ms,
            memory_allocated_before=memory_before,
            memory_allocated_after=memory_after,
            memory_allocated_delta=memory_after - memory_before,
            memory_reserved_before=reserved_before,
            memory_reserved_after=reserved_after,
            memory_reserved_delta=reserved_after - reserved_before,
            kv_bytes=self._kv_bytes(cache),
            estimated_kv_bytes=self._model_cache_bytes(
                int(cache.max_seq_len), int(cache.batch_size),
                int(cache.k_cache.element_size()),
            ),
        ))
        state["setup_event"] = self._setup["graph_events"][-1]
        return state

    def _capture_owners(self):
        result = []
        try:
            last_block = self.target.transformer.h[-1]
            hooks = getattr(last_block, "_forward_hooks", {})
            for hook in hooks.values():
                owner = getattr(hook, "__self__", None)
                if owner is not None and hasattr(owner, "output"):
                    result.append(owner)
        except Exception:
            pass
        return result

    def _publish_hidden(self, hidden):
        value = hidden.detach()
        sinks = []
        if self._explicit_hidden_capture is not None:
            sinks.append(self._explicit_hidden_capture)
        sinks.extend(self._capture_owners())
        seen = set()
        for sink in sinks:
            if id(sink) in seen:
                continue
            seen.add(id(sink))
            if callable(sink) and not hasattr(sink, "output"):
                sink(value)
            else:
                sink.output = value

    def _run_graph(self, state, cache, idx, pos: int, kinds, kwargs):
        import torch

        rope_value = _with_host_position(
            cache, pos, lambda: self.target._get_rope(idx, kv_cache=cache)
        )
        state["static_idx"].copy_(idx)
        state["pos_tensor"].fill_(int(pos))
        _copy_tree(torch, state["static_rope"], rope_value)
        cache.store_tokens(state["static_idx"], pos)
        state["graph"].replay()
        logits, pre_final_norm_hidden, post_final_norm_hidden = state["graph_outputs"]
        self.target._apply_slot_logits_mask(logits, start_pos=pos)
        self.target._apply_monotonic_position_mask(
            logits, idx=state["static_idx"], start_pos=pos, kv_cache=cache
        )
        self._publish_hidden(pre_final_norm_hidden)
        if self.record_outputs:
            self._record(dict(
                mode="graph",
                pos=int(pos),
                input_tokens=idx.detach().to(device="cpu").clone(),
                logit_types=None if kinds is None else list(kinds),
                logits=logits.detach().to(device="cpu").clone(),
                pre_final_norm_hidden=pre_final_norm_hidden.detach().to(device="cpu").clone(),
                post_final_norm_hidden=post_final_norm_hidden.detach().to(device="cpu").clone(),
            ))
        return ((logits, post_final_norm_hidden)
                if kwargs.get("return_hidden", False) else logits)

