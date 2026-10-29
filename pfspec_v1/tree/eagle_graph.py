"""Graph only the recurrent T=1 EAGLE Draft forward.

The existing EAGLE decoder keeps ownership of its Draft caches, cache resets,
truncation and sampling. This runtime stages the active KV prefix into one
stable cache before replay and copies the newly written slot back afterwards.
Long reset/prefill calls remain eager. No decoder RNG or Target path is
captured.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
import time


_MISSING = object()


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _read_buckets(capacity: int, minimum: int = 8) -> tuple[int, ...]:
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError("max_cache_tokens must be positive")
    values = []
    size = min(int(minimum), capacity)
    while size < capacity:
        values.append(size)
        size *= 2
    values.append(capacity)
    return tuple(sorted(set(values)))


def _ceil_bucket(position_after_write: int, buckets: tuple[int, ...]) -> int | None:
    for bucket in buckets:
        if bucket >= position_after_write:
            return bucket
    return None


def _shape_signature(torch, values):
    result = []
    for value in values:
        if torch.is_tensor(value):
            result.append((tuple(value.shape), str(value.dtype), str(value.device)))
        else:
            result.append((type(value).__name__, value))
    return tuple(result)


class EagleDraftGraph:
    """Temporarily route eligible EAGLE Draft T=1 forwards through CUDA Graphs.

    Graph keys use only the fixed query shape and a read-length bucket. Cache
    position and RoPE coordinates are copied into graph inputs each call. A
    single fixed staging KV buffer is reused across cache objects, including
    the per-round caches used by EAGLE v3.
    """

    def __init__(
        self,
        draft,
        *,
        max_cache_tokens: int = 1024,
        read_buckets=None,
        warmup_calls: int = 2,
        max_graphs: int = 16,
        verify: bool = False,
        profile_copies: bool = False,
        verify_atol: float = 1e-5,
        verify_rtol: float = 1e-5,
    ):
        started = time.perf_counter()
        if draft is None or not callable(getattr(draft, "forward", None)):
            raise TypeError("an EAGLE Draft module is required")
        self.draft = draft
        self.max_cache_tokens = int(max_cache_tokens)
        if self.max_cache_tokens < 8:
            raise ValueError("max_cache_tokens must be at least eight")
        self.read_buckets = tuple(sorted(set(int(value) for value in (
            read_buckets or _read_buckets(self.max_cache_tokens)
        ))))
        if not self.read_buckets or self.read_buckets[-1] > self.max_cache_tokens:
            raise ValueError("read buckets must fit in max_cache_tokens")
        if any(value < 1 for value in self.read_buckets):
            raise ValueError("read buckets must be positive")
        self.warmup_calls = max(1, int(warmup_calls))
        self.max_graphs = max(1, int(max_graphs))
        self.verify_enabled = bool(verify)
        self.profile_copies = bool(profile_copies)
        self.verify_atol = float(verify_atol)
        self.verify_rtol = float(verify_rtol)
        self.enabled = True
        self._active = False
        self._original_forward = None
        self._instance_forward_existed = False
        self._instance_forward = None
        self._original_logits = None
        self._instance_logits_existed = False
        self._instance_logits = None
        self._stage_cache = None
        self._stage_source_cache = None
        self._stage_pos = None
        self._verify_cache = None
        self._graphs = {}
        self._logit_graphs = {}
        self._disabled = {}
        self._disabled_logits = {}
        self._calls = Counter()
        self._fallbacks = Counter()
        self._events = []
        self._copy_event_pairs = []
        self._copy_time_ms = Counter()
        self._copy_bytes = Counter()
        self._verify = Counter()
        self._verify_max = dict(
            hidden_row_l2_relative=0.0,
            policy_tv=0.0,
            kv_abs=0.0,
            logits_abs=0.0,
            logits_policy_tv=0.0,
        )
        self._verify_support_equal = True
        self._parameters_fp32 = None
        self._setup = dict(
            runtime_init_ms=(time.perf_counter() - started) * 1000.0,
            stage_cache_ms=0.0,
            stage_cache_bytes=0,
            verify_cache_ms=0.0,
        )
        self._flash_module = self._find_flash_module()
        self._fa3_check = getattr(
            getattr(self._flash_module, "flash_attn_with_kvcache"),
            "__globals__", {},
        ).get("_use_fa3")

    def _find_flash_module(self):
        blocks = getattr(self.draft, "blocks", None)
        if not blocks:
            raise TypeError("EAGLE Draft has no Transformer blocks")
        forward = blocks[0].attn.forward
        namespace = getattr(getattr(forward, "__func__", forward), "__globals__", {})
        module = namespace.get("flash_attn")
        if module is None or not callable(getattr(module, "flash_attn_with_kvcache", None)):
            raise RuntimeError("could not locate the Draft KV attention module")
        return module

    @contextmanager
    def context(self):
        """Install a temporary Draft.forward wrapper and restore it on exit."""
        if self._active:
            raise RuntimeError("this Draft graph context is already active")
        self._active = True
        self._original_forward = self.draft.forward
        self._instance_forward_existed = "forward" in self.draft.__dict__
        if self._instance_forward_existed:
            self._instance_forward = self.draft.__dict__["forward"]

        def wrapped_forward(*args, **kwargs):
            return self._dispatch(args, kwargs)

        self.draft.forward = wrapped_forward
        self._original_logits = self.draft.logits
        self._instance_logits_existed = "logits" in self.draft.__dict__
        if self._instance_logits_existed:
            self._instance_logits = self.draft.__dict__["logits"]

        def wrapped_logits(*args, **kwargs):
            return self._dispatch_logits(args, kwargs)

        self.draft.logits = wrapped_logits
        try:
            yield self
        finally:
            if self._instance_forward_existed:
                self.draft.forward = self._instance_forward
            else:
                delattr(self.draft, "forward")
            if self._instance_logits_existed:
                self.draft.logits = self._instance_logits
            else:
                delattr(self.draft, "logits")
            self._original_forward = None
            self._instance_forward = None
            self._original_logits = None
            self._instance_logits = None
            self._active = False

    def eager_forward(self, *args, **kwargs):
        """Call the unwrapped Draft method, useful for paired verification."""
        if self._original_forward is not None:
            return self._original_forward(*args, **kwargs)
        return self.draft.forward(*args, **kwargs)

    def eager_logits(self, *args, **kwargs):
        """Call the unwrapped Draft readout, useful for paired verification."""
        if self._original_logits is not None:
            return self._original_logits(*args, **kwargs)
        return self.draft.logits(*args, **kwargs)

    def invalidate_cache(self, cache=None):
        """Drop staged history after an external same-position K/V rewrite.

        Truncate and append operations are detected from cache identity and
        position. Call this when active K/V are overwritten while the cache
        position remains unchanged.
        """
        if cache is None or cache is self._stage_source_cache:
            self._invalidate_cache_state()

    def _invalidate_cache_state(self):
        self._stage_source_cache = None
        self._stage_pos = None
        self._calls["stage_invalidations"] += 1

    def _call_original(self, args, kwargs):
        self._calls["eager_forward"] += 1
        if self._original_forward is None:
            return self.draft.forward(*args, **kwargs)
        return self._original_forward(*args, **kwargs)

    def _dispatch(self, args, kwargs):
        import torch

        self._calls["draft_forward"] += 1
        reason, cache, values = self._eligibility(torch, args, kwargs)
        if reason is not None:
            self._fallbacks[reason] += 1
            if cache is not None:
                self._invalidate_cache_state()
            return self._call_original(args, kwargs)
        previous_hidden, actual_tokens, coordinates = values
        pos = int(cache.get_pos())
        self._flush_copy_events()
        bucket = _ceil_bucket(pos + 1, self.read_buckets)
        if pos < 0 or pos + 1 > self.max_cache_tokens or pos + 1 > int(cache.max_seq_len):
            self._fallbacks["cache_capacity"] += 1
            self._invalidate_cache_state()
            return self._call_original(args, kwargs)
        if bucket is None:
            self._fallbacks["read_bucket_capacity"] += 1
            self._invalidate_cache_state()
            return self._call_original(args, kwargs)
        key = (bucket, _shape_signature(torch, values))
        if key in self._disabled:
            self._fallbacks["disabled_graph"] += 1
            self._invalidate_cache_state()
            return self._call_original(args, kwargs)
        state = self._graphs.get(key)
        if state is None:
            if len(self._graphs) >= self.max_graphs:
                self._fallbacks["graph_limit"] += 1
                self._invalidate_cache_state()
                return self._call_original(args, kwargs)
            try:
                state = self._build_graph(torch, key, bucket, cache, pos, values)
            except Exception as error:
                self._disabled[key] = f"{type(error).__name__}: {error}"
                self._events.append(dict(
                    query_tokens=1,
                    read_bucket=int(bucket),
                    error=self._disabled[key],
                ))
                try:
                    torch.cuda.synchronize(previous_hidden.device)
                except Exception:
                    raise
                self._fallbacks["graph_capture_error"] += 1
                self._invalidate_cache_state()
                return self._call_original(args, kwargs)
            self._graphs[key] = state
            self._calls["graph_capture"] += 1
            self._copy_stage_input(torch, state, cache, pos, values)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                state["graph"].replay()
            self._calls["graph_forward"] += 1
            self._calls["graph_replay"] += 1
            self._copy_stage_output(cache, state["stage_cache"], pos)
            output = self._clone_output(torch, state["outputs"])
            if self.verify_enabled:
                self._verify_call(torch, state, cache, pos, values, output)
            return output

        self._copy_stage_input(torch, state, cache, pos, values)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            state["graph"].replay()
        self._calls["graph_forward"] += 1
        self._calls["graph_replay"] += 1
        output = self._clone_output(torch, state["outputs"])
        self._copy_stage_output(cache, state["stage_cache"], pos)
        if self.verify_enabled:
            self._verify_call(torch, state, cache, pos, values, output)
        return output

    def _dispatch_logits(self, args, kwargs):
        import torch

        self._calls["logits_call"] += 1
        if not self.enabled:
            return self._logits_fallback("logits_disabled", args, kwargs)
        if getattr(self.draft, "training", False):
            return self._logits_fallback("logits_training_mode", args, kwargs)
        if torch.is_grad_enabled():
            return self._logits_fallback("logits_grad_enabled", args, kwargs)
        hidden = args[0] if args else kwargs.get("hidden")
        kind = args[1] if len(args) > 1 else kwargs.get("kind")
        if kind != "F":
            return self._logits_fallback("logits_kind", args, kwargs)
        if not torch.is_tensor(hidden):
            return self._logits_fallback("logits_input", args, kwargs)
        if hidden.ndim != 1 or int(hidden.shape[0]) != int(self.draft.embedding.shape[1]):
            return self._logits_fallback("logits_shape", args, kwargs)
        if not hidden.is_cuda:
            return self._logits_fallback("logits_non_cuda", args, kwargs)
        if hidden.dtype != torch.float32 or self.draft.output_weight.dtype != torch.float32:
            return self._logits_fallback("logits_non_fp32", args, kwargs)
        if not self.draft.output_weight.is_cuda:
            return self._logits_fallback("logits_weight_non_cuda", args, kwargs)
        if self._parameters_fp32 is None:
            self._parameters_fp32 = all(
                not tensor.is_floating_point() or tensor.dtype == torch.float32
                for tensor in self.draft.parameters()
            )
        if not self._parameters_fp32:
            return self._logits_fallback("logits_non_fp32_parameter", args, kwargs)
        key = ("F", _shape_signature(torch, (hidden,)))
        if key in self._disabled_logits:
            return self._logits_fallback("logits_disabled_graph", args, kwargs)
        state = self._logit_graphs.get(key)
        if state is None:
            if len(self._graphs) + len(self._logit_graphs) >= self.max_graphs:
                return self._logits_fallback("logits_graph_limit", args, kwargs)
            try:
                state = self._build_logit_graph(torch, key, hidden, kind)
            except Exception as error:
                self._disabled_logits[key] = f"{type(error).__name__}: {error}"
                self._events.append(dict(event_kind="draft_logits", error=self._disabled_logits[key]))
                try:
                    torch.cuda.synchronize(hidden.device)
                except Exception:
                    raise
                return self._logits_fallback("logits_capture_error", args, kwargs)
            self._logit_graphs[key] = state
            self._calls["graph_capture"] += 1
            self._calls["logits_capture"] += 1
            state["static_hidden"].copy_(hidden)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                state["graph"].replay()
            self._calls["graph_logits"] += 1
            self._calls["logits_replay"] += 1
            output = self._clone_output(torch, state["outputs"])
            if self.verify_enabled:
                self._verify_logits(torch, hidden, output, kind)
            return output

        state["static_hidden"].copy_(hidden)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            state["graph"].replay()
        self._calls["graph_logits"] += 1
        self._calls["logits_replay"] += 1
        output = self._clone_output(torch, state["outputs"])
        if self.verify_enabled:
            self._verify_logits(torch, hidden, output, kind)
        return output

    def _logits_fallback(self, reason, args, kwargs):
        self._fallbacks[reason] += 1
        self._calls["eager_logits"] += 1
        if self._original_logits is None:
            return self.draft.logits(*args, **kwargs)
        return self._original_logits(*args, **kwargs)

    def _build_logit_graph(self, torch, key, hidden, kind):
        started = time.perf_counter()
        static_hidden = hidden.detach().clone()
        graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(hidden.device)
        warm_stream = torch.cuda.Stream(device=hidden.device)
        warm_stream.wait_stream(current)
        warm_started = time.perf_counter()
        with torch.cuda.stream(warm_stream), torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            for _ in range(self.warmup_calls):
                static_hidden.copy_(hidden)
                self._original_logits(static_hidden, kind)
        current.wait_stream(warm_stream)
        torch.cuda.synchronize(hidden.device)
        warmup_ms = (time.perf_counter() - warm_started) * 1000.0
        static_hidden.copy_(hidden)
        capture_stream = torch.cuda.Stream(device=hidden.device)
        capture_stream.wait_stream(current)
        capture_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            with torch.cuda.graph(graph, stream=capture_stream):
                outputs = self._original_logits(static_hidden, kind)
        current.wait_stream(capture_stream)
        torch.cuda.synchronize(hidden.device)
        capture_ms = (time.perf_counter() - capture_started) * 1000.0
        if not torch.is_tensor(outputs) or not outputs.is_cuda:
            raise TypeError("EAGLE Draft F readout must return a CUDA tensor")
        state = dict(
            graph=graph,
            outputs=outputs,
            static_hidden=static_hidden,
            capture_stream=capture_stream,
            key=key,
            kind=str(kind),
        )
        self._events.append(dict(
            event_kind="draft_logits",
            kind=str(kind),
            input_shape=list(hidden.shape),
            output_shape=list(outputs.shape),
            warmup_calls=self.warmup_calls,
            warmup_ms=warmup_ms,
            capture_ms=capture_ms,
            prepare_ms=(time.perf_counter() - started) * 1000.0,
        ))
        return state

    def _verify_logits(self, torch, hidden, candidate, kind):
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            expected = self._original_logits(hidden, kind).detach().clone()
        if tuple(candidate.shape) != tuple(expected.shape):
            self._verify["logits_calls"] += 1
            self._verify["logits_shape_mismatches"] += 1
            return
        left = candidate.float()
        right = expected.float()
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        if finite:
            max_abs = float((left - right).abs().max().item())
        else:
            max_abs = math.inf
        left_policy = self._policy(torch, candidate)
        right_policy = self._policy(torch, expected)
        tv = float((left_policy - right_policy).abs().sum().mul(0.5).item())
        support_equal = bool(torch.equal(left_policy > 0, right_policy > 0))
        all_close = finite and bool(torch.allclose(
            candidate, expected, atol=self.verify_atol, rtol=self.verify_rtol
        ))
        passed = all_close and tv <= 1e-5 and support_equal
        self._verify["logits_calls"] += 1
        self._verify["logits_passes"] += int(passed)
        self._verify["logits_support_mismatches"] += int(not support_equal)
        self._verify_max["logits_abs"] = max(self._verify_max["logits_abs"], max_abs)
        self._verify_max["logits_policy_tv"] = max(self._verify_max["logits_policy_tv"], tv)

    def _eligibility(self, torch, args, kwargs):
        if not self.enabled:
            return "disabled", None, None
        if getattr(self.draft, "training", False):
            return "training_mode", None, None
        if torch.is_grad_enabled():
            return "grad_enabled", None, None
        if self._fa3_check is not None and self._fa3_check():
            return "fa3_backend", None, None
        if len(args) >= 3:
            previous_hidden, actual_tokens, coordinates = args[:3]
        else:
            previous_hidden = kwargs.get("previous_hidden")
            actual_tokens = kwargs.get("actual_tokens")
            coordinates = kwargs.get("coordinates")
        cache = kwargs.get("cache")
        if cache is None and len(args) >= 4:
            cache = args[3]
        if cache is None:
            return "cache_missing", None, None
        if not all(torch.is_tensor(item) for item in
                   (previous_hidden, actual_tokens, coordinates)):
            return "invalid_inputs", cache, None
        values = (previous_hidden, actual_tokens, coordinates)
        if previous_hidden.ndim != 3 or previous_hidden.shape[:2] != actual_tokens.shape:
            return "input_shape", cache, values
        if int(actual_tokens.shape[0]) != 1:
            return "batch_size", cache, values
        if int(actual_tokens.shape[1]) != 1:
            return "unsupported_query_tokens", cache, values
        if not all(item.is_cuda for item in values):
            return "non_cuda_input", cache, values
        if len({item.device for item in values}) != 1:
            return "mixed_input_devices", cache, values
        if any(item.layout != torch.strided for item in values):
            return "non_strided_input", cache, values
        if previous_hidden.dtype != torch.float32:
            return "non_fp32_hidden", cache, values
        if coordinates.dtype not in (torch.int32, torch.int64):
            return "coordinate_dtype", cache, values
        if actual_tokens.dtype not in (torch.int32, torch.int64):
            return "token_dtype", cache, values
        if self._parameters_fp32 is None:
            self._parameters_fp32 = all(
                not tensor.is_floating_point() or tensor.dtype == torch.float32
                for tensor in self.draft.parameters()
            )
        if not self._parameters_fp32:
            return "non_fp32_parameter", cache, values
        if int(cache.batch_size) != 1:
            return "cache_batch_size", cache, values
        if cache.k_cache.dtype != torch.float32 or cache.v_cache.dtype != torch.float32:
            return "non_fp32_cache", cache, values
        if not cache.k_cache.is_cuda or not cache.v_cache.is_cuda:
            return "non_cuda_cache", cache, values
        if int(cache.n_layers) != int(self.draft.config.layers):
            return "cache_layer_count", cache, values
        if int(cache.n_heads) != int(self.draft.config.heads):
            return "cache_head_count", cache, values
        if int(cache.head_dim) != int(self.draft.config.width // self.draft.config.heads):
            return "cache_head_dim", cache, values
        return None, cache, values

    def _ensure_stage_cache(self, pos: int, like_cache):
        if self._stage_cache is not None:
            return
        started = time.perf_counter()
        self._stage_cache = self.draft.new_cache(self.max_cache_tokens)
        if self._stage_cache.k_cache.device != like_cache.k_cache.device:
            raise ValueError("Draft stage cache device differs from active cache")
        self._setup["stage_cache_ms"] = (time.perf_counter() - started) * 1000.0
        self._setup["stage_cache_bytes"] = self._kv_bytes(self._stage_cache)

    def _build_graph(self, torch, key, bucket, cache, pos, values):
        started = time.perf_counter()
        self._stage_source_cache = None
        self._stage_pos = None
        self._ensure_stage_cache(pos, cache)
        stage = self._stage_cache
        static_values = tuple(item.detach().clone() for item in values)
        graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(values[0].device)
        warm_stream = torch.cuda.Stream(device=values[0].device)
        warm_stream.wait_stream(current)
        warm_started = time.perf_counter()
        with torch.cuda.stream(warm_stream), torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            for _ in range(self.warmup_calls):
                self._copy_prefix_to_stage(stage, cache, pos)
                for destination, source in zip(static_values, values):
                    destination.copy_(source)
                with self._patched_attention(stage, bucket):
                    self._original_forward(*static_values, cache=stage)
        current.wait_stream(warm_stream)
        torch.cuda.synchronize(values[0].device)
        warmup_ms = (time.perf_counter() - warm_started) * 1000.0
        self._copy_prefix_to_stage(stage, cache, pos)
        for destination, source in zip(static_values, values):
            destination.copy_(source)

        capture_stream = torch.cuda.Stream(device=values[0].device)
        capture_stream.wait_stream(current)
        capture_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            with self._patched_attention(stage, bucket):
                with torch.cuda.graph(graph, stream=capture_stream):
                    outputs = self._original_forward(*static_values, cache=stage)
        current.wait_stream(capture_stream)
        torch.cuda.synchronize(values[0].device)
        capture_ms = (time.perf_counter() - capture_started) * 1000.0
        if not torch.is_tensor(outputs) or not outputs.is_cuda:
            raise TypeError("EAGLE Draft forward must return one CUDA tensor")
        if tuple(outputs.shape) != tuple(values[0].shape):
            raise ValueError("EAGLE Draft graph output shape changed")
        state = dict(
            graph=graph,
            outputs=outputs,
            static_values=static_values,
            stage_cache=stage,
            capture_stream=capture_stream,
            key=key,
            bucket=int(bucket),
        )
        self._events.append(dict(
            query_tokens=1,
            read_bucket=int(bucket),
            cache_capacity=self.max_cache_tokens,
            warmup_calls=self.warmup_calls,
            warmup_ms=warmup_ms,
            capture_ms=capture_ms,
            prepare_ms=(time.perf_counter() - started) * 1000.0,
            setup_prefix_copy_bytes=self._prefix_copy_bytes(cache, pos),
            verify=self.verify_enabled,
        ))
        self._stage_source_cache = None
        self._stage_pos = None
        return state

    @contextmanager
    def _patched_attention(self, stage, bucket):
        import torch
        import torch.nn.functional as F

        module = self._flash_module
        original = module.flash_attn_with_kvcache

        def graph_attention(q, k_cache, v_cache, k=None, v=None,
                            cache_seqlens=None, causal=False,
                            window_size=(-1, 0), **kwargs):
            if cache_seqlens is not stage.cache_seqlens:
                return original(q, k_cache, v_cache, k=k, v=v,
                                cache_seqlens=cache_seqlens, causal=causal,
                                window_size=window_size, **kwargs)
            if k is None or v is None or not causal:
                raise RuntimeError("unsupported EAGLE graph attention call")
            if q.shape[0] != 1 or q.shape[1] != 1 or k.shape[1] != 1:
                raise RuntimeError("EAGLE graph attention expects batch=1 and T=1")
            if k_cache.shape[1] < bucket or v_cache.shape[1] < bucket:
                raise RuntimeError("graph KV tensor is smaller than its read bucket")
            query_pos = cache_seqlens.to(dtype=torch.long)
            key_offsets = torch.arange(bucket, device=q.device, dtype=torch.long)
            write_index = query_pos.reshape(-1, 1).reshape(-1)
            k_cache.index_copy_(1, write_index, k)
            v_cache.index_copy_(1, write_index, v)
            valid = key_offsets.reshape(1, -1) <= query_pos.reshape(-1, 1)
            qh = q.transpose(1, 2)
            kh = k_cache[:, :bucket].transpose(1, 2)
            vh = v_cache[:, :bucket].transpose(1, 2)
            enable_gqa = qh.size(1) != kh.size(1)
            y = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=valid[:, None, None, :],
                is_causal=False,
                enable_gqa=enable_gqa,
            )
            return y.transpose(1, 2)

        module.flash_attn_with_kvcache = graph_attention
        try:
            yield
        finally:
            module.flash_attn_with_kvcache = original

    def _copy_prefix_to_stage(self, stage, source, pos):
        if pos:
            stage.k_cache[:, :, :pos].copy_(source.k_cache[:, :, :pos])
            stage.v_cache[:, :, :pos].copy_(source.v_cache[:, :, :pos])
        stage.cache_seqlens.copy_(source.cache_seqlens)

    def _copy_stage_input(self, torch, state, source_cache, pos, values):
        stage = state["stage_cache"]
        stream = torch.cuda.current_stream(values[0].device)
        started = torch.cuda.Event(enable_timing=True) if self.profile_copies else None
        ended = torch.cuda.Event(enable_timing=True) if self.profile_copies else None
        if started is not None:
            started.record(stream)
        is_synced = self._stage_source_cache is source_cache and self._stage_pos == int(pos)
        if not is_synced:
            self._copy_prefix_to_stage(stage, source_cache, pos)
            self._copy_bytes["input_kv"] += self._prefix_copy_bytes(source_cache, pos)
        for destination, source in zip(state["static_values"], values):
            destination.copy_(source)
        if ended is not None:
            ended.record(stream)
            self._copy_event_pairs.append(("input", started, ended))
        self._copy_bytes["input_values"] += sum(int(item.numel() * item.element_size())
                                                  for item in values)

    def _copy_stage_output(self, destination, stage, pos):
        import torch

        stream = torch.cuda.current_stream(stage.k_cache.device)
        started = torch.cuda.Event(enable_timing=True) if self.profile_copies else None
        ended = torch.cuda.Event(enable_timing=True) if self.profile_copies else None
        if started is not None:
            started.record(stream)
        destination.k_cache[:, :, pos:pos + 1].copy_(stage.k_cache[:, :, pos:pos + 1])
        destination.v_cache[:, :, pos:pos + 1].copy_(stage.v_cache[:, :, pos:pos + 1])
        destination.cache_seqlens.copy_(stage.cache_seqlens)
        if ended is not None:
            ended.record(stream)
            self._copy_event_pairs.append(("output", started, ended))
        self._copy_bytes["output_kv"] += self._slot_copy_bytes(destination)
        self._stage_source_cache = destination
        self._stage_pos = int(pos) + 1

    def _flush_copy_events(self):
        remaining = []
        for kind, started, ended in self._copy_event_pairs:
            if ended.query():
                self._copy_time_ms[kind] += float(started.elapsed_time(ended))
            else:
                remaining.append((kind, started, ended))
        self._copy_event_pairs = remaining

    @staticmethod
    def _prefix_copy_bytes(cache, pos):
        element = int(cache.k_cache.element_size())
        return 2 * int(cache.k_cache.shape[0] * cache.k_cache.shape[1] * pos
                       * cache.k_cache.shape[3] * cache.k_cache.shape[4] * element)

    @staticmethod
    def _slot_copy_bytes(cache):
        element = int(cache.k_cache.element_size())
        return 2 * int(cache.k_cache.shape[0] * cache.k_cache.shape[1]
                       * cache.k_cache.shape[3] * cache.k_cache.shape[4] * element)

    @staticmethod
    def _kv_bytes(cache):
        return int(cache.k_cache.numel() * cache.k_cache.element_size()
                   + cache.v_cache.numel() * cache.v_cache.element_size())

    @staticmethod
    def _clone_output(torch, output):
        return output.detach().clone()

    def _verify_call(self, torch, state, candidate_cache, pos, values, candidate_output):
        if self._verify_cache is None:
            started = time.perf_counter()
            self._verify_cache = self.draft.new_cache(self.max_cache_tokens)
            self._setup["verify_cache_ms"] = (time.perf_counter() - started) * 1000.0
        reference_cache = self._verify_cache
        if pos:
            reference_cache.k_cache[:, :, :pos].copy_(candidate_cache.k_cache[:, :, :pos])
            reference_cache.v_cache[:, :, :pos].copy_(candidate_cache.v_cache[:, :, :pos])
        reference_cache.cache_seqlens.copy_(candidate_cache.cache_seqlens)
        reference_cache.cache_seqlens.fill_(pos)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            expected = self._original_forward(*values, cache=reference_cache).detach().clone()
        hidden_error = self._max_row_l2_relative(torch, candidate_output, expected)
        self._verify["calls"] += 1
        self._verify_max["hidden_row_l2_relative"] = max(
            self._verify_max["hidden_row_l2_relative"], hidden_error
        )
        self._verify["hidden_passes"] += int(torch.allclose(
            candidate_output, expected, atol=self.verify_atol, rtol=self.verify_rtol
        ))
        if self.draft.config.mode == "feature":
            kinds = ("F",)
        else:
            kinds = ("P", "F")
        for kind in kinds:
            left = self.draft.logits(candidate_output[:, -1, :], kind)
            right = self.draft.logits(expected[:, -1, :], kind)
            left_policy = self._policy(torch, left)
            right_policy = self._policy(torch, right)
            tv = float((left_policy - right_policy).abs().sum().mul(0.5).item())
            equal_support = torch.equal(left_policy > 0, right_policy > 0)
            self._verify_max["policy_tv"] = max(self._verify_max["policy_tv"], tv)
            self._verify["policy_calls"] += 1
            self._verify["policy_support_mismatches"] += int(not equal_support)
            self._verify_support_equal = self._verify_support_equal and equal_support
        actual_k = candidate_cache.k_cache[:, :, pos:pos + 1]
        actual_v = candidate_cache.v_cache[:, :, pos:pos + 1]
        expected_k = reference_cache.k_cache[:, :, pos:pos + 1]
        expected_v = reference_cache.v_cache[:, :, pos:pos + 1]
        kv_error = max(
            float((actual_k.float() - expected_k.float()).abs().max().item()),
            float((actual_v.float() - expected_v.float()).abs().max().item()),
        )
        self._verify_max["kv_abs"] = max(self._verify_max["kv_abs"], kv_error)
        self._verify["kv_passes"] += int(torch.allclose(
            actual_k, expected_k, atol=self.verify_atol, rtol=self.verify_rtol
        ) and torch.allclose(actual_v, expected_v, atol=self.verify_atol, rtol=self.verify_rtol))
        self._verify["position_mismatches"] += int(
            int(candidate_cache.get_pos()) != int(reference_cache.get_pos())
        )

    @staticmethod
    def _max_row_l2_relative(torch, actual, expected):
        left = actual.detach().float().reshape(-1, actual.shape[-1])
        right = expected.detach().float().reshape(-1, expected.shape[-1])
        if left.shape != right.shape or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            return math.inf
        numer = torch.linalg.vector_norm(left - right, dim=-1)
        denom = torch.linalg.vector_norm(right, dim=-1).clamp_min(1e-12)
        return float((numer / denom).max().item()) if numer.numel() else 0.0

    @staticmethod
    def _policy(torch, logits):
        from speculative.pf_engine import policy
        return policy(logits, temperature=0.9, top_p=0.9)

    def stats(self):
        self._flush_copy_events()
        verify = dict(self._verify)
        verify.update(self._verify_max)
        verify["policy_support_equal"] = self._verify_support_equal
        verify["passed"] = (
            verify.get("calls", 0) > 0
            and verify.get("hidden_passes", 0) == verify.get("calls", 0)
            and verify.get("kv_passes", 0) == verify.get("calls", 0)
            and verify.get("logits_calls", 0) > 0
            and verify.get("logits_passes", 0) == verify.get("logits_calls", 0)
            and verify.get("policy_support_mismatches", 0) == 0
            and verify.get("logits_support_mismatches", 0) == 0
            and verify.get("position_mismatches", 0) == 0
            and verify.get("hidden_row_l2_relative", 0.0) <= 1e-5
            and verify.get("policy_tv", 0.0) <= 1e-5
            and verify.get("logits_policy_tv", 0.0) <= 1e-5
        ) if self.verify_enabled else None
        return dict(
            enabled=self.enabled,
            active=self._active,
            max_cache_tokens=self.max_cache_tokens,
            read_buckets=list(self.read_buckets),
            graph_count=len(self._graphs) + len(self._logit_graphs),
            forward_graph_count=len(self._graphs),
            logits_graph_count=len(self._logit_graphs),
            disabled_graphs={
                **{repr(key): value for key, value in self._disabled.items()},
                **{"logits:" + repr(key): value
                   for key, value in self._disabled_logits.items()},
            },
            calls=dict(self._calls),
            fallbacks=dict(self._fallbacks),
            graph_events=[dict(event) for event in self._events],
            runtime_init_ms=self._setup["runtime_init_ms"],
            stage_cache_ms=self._setup["stage_cache_ms"],
            stage_cache_bytes=self._setup["stage_cache_bytes"],
            verify_cache_ms=self._setup["verify_cache_ms"],
            staging_copy_ms=({key: float(value) for key, value in self._copy_time_ms.items()}
                             if self.profile_copies else None),
            staging_copy_pending=len(self._copy_event_pairs),
            staging_copy_bytes=dict(self._copy_bytes),
            verification=verify,
            output_lifetime="returned values are independent clones of graph output",
        )

    def close(self):
        if self._active:
            raise RuntimeError("leave EagleDraftGraph.context() before closing it")
        self._graphs.clear()
        self._logit_graphs.clear()
        self._disabled.clear()
        self._disabled_logits.clear()
        self._stage_cache = None
        self._stage_source_cache = None
        self._stage_pos = None
        self._verify_cache = None


def build_proposal(adapter, *, verify=False, max_cache_tokens=1024,
                   warmup_calls=2, max_graphs=16, profile_copies=False):
    """Build the proposal runtime expected by evaluate_proposals.py."""
    if getattr(adapter, "variant", None) != "eagle":
        raise TypeError("EAGLE proposal adapter required")
    draft = getattr(adapter, "model", None)
    if draft is None:
        raise TypeError("EAGLE adapter has no loaded Draft model")
    return EagleDraftGraph(
        draft,
        max_cache_tokens=max_cache_tokens,
        warmup_calls=warmup_calls,
        max_graphs=max_graphs,
        verify=verify,
        profile_copies=profile_copies,
    )
