"""Experimental fixed-width P/F tree verification over one GaussianGPT Target."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import time
from types import MethodType


def tree_layout(width: int):
    """Return node parents, logical depths, output types, and ancestor paths.

    Nodes are ordered as ``pendingF, P1, F0, P20, F1, P21, ...``.  This
    pure-Python helper is also useful to inspect the tree without importing
    PyTorch or loading a model.
    """
    width = int(width)
    if width < 1:
        raise ValueError("tree width must be at least one")
    parents = [-1, 0]
    depths = [0, 1]
    logit_types = ["position", "feature"]
    for branch in range(width):
        feature_node = 2 + 2 * branch
        position_node = feature_node + 1
        parents.extend((1, feature_node))
        depths.extend((2, 3))
        logit_types.extend(("position", "feature"))
    paths = []
    for node in range(len(parents)):
        path = []
        current = node
        while current >= 0:
            path.append(current)
            current = parents[current]
        paths.append(tuple(reversed(path)))
    return tuple(parents), tuple(depths), tuple(logit_types), tuple(paths)


def tree_key_mask(width: int, before: int, read_bucket: int,
                  window_left: int = -1):
    """Build a CPU boolean mask from shared history, ancestors, and logical depth."""
    parents, depths, _, paths = tree_layout(width)
    before = int(before)
    read_bucket = int(read_bucket)
    if before < 0 or read_bucket < before + len(parents):
        raise ValueError("read bucket must contain the prefix and every tree node")
    window_left = int(window_left)
    rows = []
    for node, path in enumerate(paths):
        query_position = before + depths[node]
        allowed = [
            index < before and
            (window_left < 0 or query_position - index <= window_left)
            for index in range(read_bucket)
        ]
        for ancestor in path:
            allowed[before + ancestor] = (
                window_left < 0 or depths[node] - depths[ancestor] <= window_left
            )
        rows.append(tuple(allowed))
    return tuple(rows)


def _sparse_position_rope_sources(before: int, tokens, depths, paths,
                                  tokens_per_latent: int):
    """Map each tree query to its Position-token source without scanning history."""
    before = int(before)
    tokens_per_latent = int(tokens_per_latent)
    if tokens_per_latent < 1:
        raise ValueError("tokens_per_latent must be positive")
    prefix_slots = []
    branch_token_ids = []
    from_prefix = []
    slot_indices = []
    bos_rows = []
    for depth, path in zip(depths, paths):
        absolute = before + int(depth)
        token_position = max(absolute - 1, 0)
        latent = token_position // tokens_per_latent
        slot = token_position % tokens_per_latent
        slot0 = 0 if absolute == 0 else 1 + latent * tokens_per_latent
        if slot0 < before:
            prefix_slots.append(slot0)
            branch_token_ids.append(0)
            from_prefix.append(True)
        else:
            path_offset = slot0 - before
            if not 0 <= path_offset < len(path):
                raise ValueError("Position RoPE source is outside the current node path")
            prefix_slots.append(0)
            branch_token_ids.append(int(tokens[path[path_offset]]))
            from_prefix.append(False)
        slot_indices.append(0 if absolute == 0 else slot)
        bos_rows.append(absolute == 0)
    return (tuple(prefix_slots), tuple(branch_token_ids), tuple(from_prefix),
            tuple(slot_indices), tuple(bos_rows))


def _ceil_power_of_two(value: int, capacity: int) -> int:
    value = int(value)
    capacity = int(capacity)
    if value < 1 or value > capacity:
        raise ValueError(f"required cache end {value} exceeds capacity {capacity}")
    bucket = max(8, 1 << (value - 1).bit_length())
    if bucket > capacity:
        bucket = capacity
    if bucket < value:
        raise ValueError(f"no read bucket can hold {value} cache entries")
    return bucket


def _clone_tree(torch, value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(torch, item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(torch, item) for item in value]
    raise TypeError(f"unsupported positional encoding value {type(value).__name__}")


def _copy_tree(torch, destination, source):
    if destination is None and source is None:
        return 0
    if torch.is_tensor(destination):
        if not torch.is_tensor(source) or tuple(destination.shape) != tuple(source.shape):
            raise RuntimeError("positional encoding shape changed for a captured tree")
        destination.copy_(source)
        return int(destination.numel() * destination.element_size())
    if isinstance(destination, tuple):
        if not isinstance(source, tuple) or len(destination) != len(source):
            raise RuntimeError("positional encoding tuple layout changed")
        return sum(_copy_tree(torch, left, right)
                   for left, right in zip(destination, source))
    if isinstance(destination, list):
        if not isinstance(source, list) or len(destination) != len(source):
            raise RuntimeError("positional encoding list layout changed")
        return sum(_copy_tree(torch, left, right)
                   for left, right in zip(destination, source))
    raise TypeError(f"unsupported positional encoding value {type(destination).__name__}")


@contextmanager
def _without_cache_advance(cache):
    """Prevent Target blocks from moving the logical cache during tree work."""
    instance = cache.__dict__
    had_instance_method = "advance" in instance
    previous = instance.get("advance")
    cache.advance = MethodType(lambda self, count: None, cache)
    try:
        yield
    finally:
        if had_instance_method:
            cache.advance = previous
        else:
            delattr(cache, "advance")


class TargetTreeRuntime:
    """Run a fixed P/F proposal tree through one Target and compact a chosen path.

    ``verify`` returns ``(logits[N, V], pre_final_hidden[N, H])``.  With graph
    mode enabled these are borrowed views of graph-owned output buffers.  The
    next verify using the same cache, width, and read bucket overwrites them;
    copy either tensor before retaining it beyond that point.

    A successful verify must be followed by ``commit`` before another verify.
    Graph capture errors propagate to the caller; this runtime never silently
    switches to the eager path.
    """

    def __init__(self, target, linear_runtime, *, graph: bool = True,
                 max_width: int = 4):
        started = time.perf_counter()
        if not hasattr(linear_runtime, "_trunk") or not hasattr(linear_runtime, "_make_weight_views"):
            raise TypeError("linear_runtime must expose DynamicTargetGraph trunk helpers")
        if linear_runtime.target is not target:
            raise ValueError("linear_runtime and target must reference the same core")
        self.target = target
        self.linear_runtime = linear_runtime
        self.graph_enabled = bool(graph)
        self.max_width = int(max_width)
        if not 1 <= self.max_width <= 4:
            raise ValueError("max_width must be between one and four")
        self._graphs = {}
        self._pending = None
        self._calls = Counter()
        self._fallbacks = Counter()
        self._copy_bytes = Counter()
        self._graph_events = []
        self._runtime_init_ms = (time.perf_counter() - started) * 1000.0
        self._active = False
        self._weight_views = None

    def stats(self):
        """Return cumulative graph, call, commit, and explicit-copy counters."""
        return dict(
            graph_enabled=self.graph_enabled,
            max_width=self.max_width,
            runtime_init_ms=self._runtime_init_ms,
            graph_count=len(self._graphs),
            graph_events=copy.deepcopy(self._graph_events),
            calls=dict(self._calls),
            fallbacks=dict(self._fallbacks),
            pending_verify=self._pending is not None,
            copy_bytes=dict(self._copy_bytes),
            output_lifetime=(
                "borrowed_until_next_matching_graph_verify"
                if self.graph_enabled else "eager_outputs_owned_by_call"
            ),
        )

    def _validate_model(self):
        target = self.target
        if bool(getattr(target, "training", False)):
            raise ValueError("tree verification requires target.eval()")
        if int(getattr(target, "sparse_tokens_per_latent", 2)) != 2:
            raise ValueError("P/F tree runtime requires two tokens per latent")
        if int(getattr(target, "sparse_num_position_tokens", 1)) != 1:
            raise ValueError("P/F tree runtime requires one Position token per latent")
        if not hasattr(target, "transformer") or not hasattr(target.transformer, "h"):
            raise TypeError("target must be the GaussianGPT transformer core")
        if not hasattr(target, "sparse_position_vocab_size"):
            raise TypeError("target is missing sparse vocabulary metadata")

    def _node_tokens(self, pending_f, p1, features, p2s):
        width = len(features)
        if width != len(p2s):
            raise ValueError("features and p2s must have the same width")
        if not 1 <= width <= self.max_width:
            raise ValueError(f"tree width must be in [1, {self.max_width}]")
        target = self.target
        p_vocab = int(target.sparse_position_vocab_size)
        f_start = (0 if bool(target.sparse_shared_vocab)
                   else int(target.sparse_feature_token_offset))
        f_end = f_start + int(target.sparse_feature_vocab_size)
        if not f_start <= int(pending_f) < f_end:
            raise ValueError("pending_f must be a global Feature token id")
        if not 0 <= int(p1) < p_vocab:
            raise ValueError("p1 must be a Position token id")
        if any(not f_start <= int(value) < f_end for value in features):
            raise ValueError("every feature must be a global Feature token id")
        if any(not 0 <= int(value) < p_vocab for value in p2s):
            raise ValueError("every p2 must be a Position token id")
        tokens = [int(pending_f), int(p1)]
        for feature, position in zip(features, p2s):
            tokens.extend((int(feature), int(position)))
        parents, depths, kinds, paths = tree_layout(width)
        return tokens, parents, depths, kinds, paths

    def _validate_cache(self, cache, tokens, width, graph_call):
        import torch

        if int(cache.batch_size) != 1:
            raise ValueError("tree verification currently requires batch_size=1")
        if cache.k_cache.device != self.target.transformer.wte.weight.device:
            raise ValueError("Target and cache must use the same device")
        if graph_call and cache.k_cache.device.type != "cuda":
            raise ValueError("graph=True requires a CUDA KV cache")
        if not torch.is_tensor(cache.cache_seqlens) or cache.cache_seqlens.numel() != 1:
            raise TypeError("cache must expose one cache_seqlens entry")
        if not torch.is_tensor(cache.token_cache) or cache.token_cache.size(0) != 1:
            raise TypeError("cache must expose a batch-one token_cache")
        if cache.k_cache.ndim != 5 or cache.v_cache.shape != cache.k_cache.shape:
            raise TypeError("cache K/V tensors must share (layers,batch,seq,heads,width) shape")
        if int(cache.k_cache.size(0)) != len(self.target.transformer.h):
            raise ValueError("cache layer count differs from the Target")
        if cache.token_cache.device != cache.k_cache.device:
            raise ValueError("token_cache and K/V cache must use the same device")
        if not hasattr(cache, "get_layer_cache") or not hasattr(cache, "get_pos"):
            raise TypeError("cache must expose get_layer_cache() and get_pos()")
        before = int(cache.get_pos())
        end = before + len(tokens)
        if end > int(cache.max_seq_len):
            raise ValueError(f"tree end {end} exceeds cache capacity {cache.max_seq_len}")
        if int(cache.token_cache.size(1)) < end:
            raise ValueError("token_cache is smaller than the requested tree span")
        if before < 2 or before % 2:
            raise ValueError("pending_f must follow a cached Position at an even sequence index")
        return before

    def _make_rope(self, cache, before, tokens, depths, paths, device):
        import torch

        target = self.target
        if not bool(getattr(target, "use_rope", True)):
            return None
        last_position = before + max(int(depth) for depth in depths)
        if int(target.rope_dim) == 1:
            if last_position >= int(target.cos.size(1)):
                raise ValueError("tree positions exceed the 1D RoPE table")
            if target.cos.device != device:
                raise ValueError("1D RoPE table and Target use different devices")
            positions = torch.tensor(
                [before + int(depth) for depth in depths],
                dtype=torch.long,
                device=device,
            )
            return (target.cos.index_select(1, positions),
                    target.sin.index_select(1, positions))
        layout = target.rope_layout
        if bool(layout.is_dense) and bool(layout.is_position):
            phases = target._get_dense_rope_phases(device)
            if last_position >= int(phases.size(1)):
                raise ValueError("tree positions exceed the dense RoPE table")
            positions = torch.tensor(
                [before + int(depth) for depth in depths],
                dtype=torch.long,
                device=device,
            )
            return phases.index_select(1, positions)
        if not bool(layout.is_sparse):
            raise ValueError("unsupported multi-axis RoPE layout for tree verification")
        if bool(layout.is_sequence):
            pieces = []
            for depth in depths:
                indices = target._build_sparse_sequence_indices(
                    batch_size=1,
                    start_pos=before + int(depth),
                    length=1,
                    device=device,
                )
                pieces.append(target.rope_embedder(indices))
            return torch.cat(pieces, dim=1)
        if not bool(layout.is_position):
            raise ValueError("sparse tree RoPE must use sequence or Position coordinates")

        prefix_slots, branch_ids, from_prefix, slot_indices, bos_rows = (
            _sparse_position_rope_sources(
                before,
                tokens,
                depths,
                paths,
                int(target.sparse_tokens_per_latent),
            )
        )
        prefix_tokens = cache.get_tokens(before)[0]
        prefix_slot_tensor = torch.tensor(prefix_slots, dtype=torch.long, device=device)
        cached_ids = prefix_tokens.index_select(0, prefix_slot_tensor)
        branch_id_tensor = torch.tensor(branch_ids, dtype=torch.long, device=device)
        prefix_source = torch.tensor(from_prefix, dtype=torch.bool, device=device)
        position_ids = torch.where(prefix_source, cached_ids, branch_id_tensor)
        valid_position_token = (position_ids >= 0) & (
            position_ids < int(target.sparse_position_vocab_size)
        )
        safe_position_ids = position_ids.masked_fill(~valid_position_token, 0).unsqueeze(0)
        from utils.pos_tokens import oned_to_threed_indices
        coords = oned_to_threed_indices(
            safe_position_ids,
            target.sparse_rope_side_length,
        ).to(dtype=torch.long)
        coords = coords.masked_fill(~valid_position_token.unsqueeze(0).unsqueeze(-1), 0)
        slot_tensor = torch.tensor(slot_indices, dtype=torch.long, device=device)
        bos_mask = torch.tensor(bos_rows, dtype=torch.bool, device=device)
        bos_coords = torch.tensor(target.rope_bos_coord, dtype=torch.long, device=device)
        coords = torch.where(
            bos_mask.view(1, -1, 1),
            bos_coords.view(1, 1, 3),
            coords,
        )
        slot_tensor = torch.where(bos_mask, torch.zeros_like(slot_tensor), slot_tensor)
        indices = torch.cat((coords, slot_tensor.view(1, -1, 1)), dim=-1)
        return target.rope_embedder(indices)

    def _make_attention(self, cache, before, width, read_bucket, masks_by_window):
        import torch

        _, _, _, paths = tree_layout(width)
        node_count = len(paths)
        physical_offsets = torch.arange(node_count, dtype=torch.long,
                                        device=cache.k_cache.device)

        def explicit_tree_attention(q, k_cache, v_cache, k=None, v=None,
                                    cache_seqlens=None, causal=False,
                                    window_size=(-1, -1)):
            if k is None or v is None or not causal:
                raise RuntimeError("tree attention requires causal K/V inputs")
            batch, actual_nodes, q_heads, head_dim = q.shape
            kv_heads = int(k_cache.size(2))
            if (batch != 1 or actual_nodes != node_count or
                    q_heads % kv_heads != 0 or read_bucket > k_cache.size(1)):
                raise RuntimeError("Target dimensions differ from the captured tree")
            window_left = int(window_size[0])
            if window_left not in masks_by_window:
                raise RuntimeError(f"no tree key mask for window_left={window_left}")
            write_positions = before + physical_offsets
            k_cache.index_copy_(1, write_positions, k)
            v_cache.index_copy_(1, write_positions, v)
            k_full = k_cache[:, :read_bucket, :, :].transpose(1, 2)
            v_full = v_cache[:, :read_bucket, :, :].transpose(1, 2)
            groups = q_heads // kv_heads
            q_grouped = q.transpose(1, 2).reshape(
                batch, kv_heads, groups, node_count, head_dim
            )
            scale = head_dim ** -0.25
            scores = torch.matmul(
                q_grouped * scale,
                (k_full * scale).transpose(-1, -2).unsqueeze(2),
            )
            scores = scores.masked_fill(
                ~masks_by_window[window_left], float("-inf")
            )
            probs = torch.softmax(scores, dim=-1)
            result = torch.matmul(probs, v_full.unsqueeze(2))
            return result.reshape(batch, q_heads, node_count, head_dim).transpose(1, 2)

        return explicit_tree_attention

    def _rank_mask(self, logits, cache, before, p1, width):
        import torch

        target = self.target
        if not bool(getattr(target, "sparse_position_hard_constraints", False)):
            return
        if not bool(target.rope_layout.is_sparse) or int(target.sparse_num_position_tokens) != 1:
            return
        p_vocab = int(target.sparse_position_vocab_size)
        rank_map = target.sparse_position_rank_map.to(logits.device)
        if rank_map.numel() < p_vocab:
            raise RuntimeError("Target rank map does not cover Position ids")
        previous_ids = torch.full((width + 1,), -1, dtype=torch.long,
                                  device=logits.device)
        previous_ids[0] = cache.token_cache[0, before - 1]
        if width:
            previous_ids[1:].fill_(int(p1))
        valid = (previous_ids >= 0) & (previous_ids < p_vocab)
        safe_ids = previous_ids.clamp(min=0, max=p_vocab - 1)
        thresholds = rank_map[safe_ids]
        invalid = (
            rank_map[:p_vocab].unsqueeze(0) <= thresholds.unsqueeze(1)
        ) & valid.unsqueeze(1)
        logits[0::2, :p_vocab].masked_fill_(invalid, float("-inf"))

    def _position_masks(self, width, before, read_bucket, device):
        import torch

        windows = tuple(getattr(self.target, "window_sizes", ()))
        if not windows:
            windows = ((-1, 0),) * len(self.target.transformer.h)
        if len(windows) != len(self.target.transformer.h):
            raise ValueError("Target window_sizes count differs from layer count")
        lefts = sorted(set(int(window[0]) for window in windows))
        masks = {}
        for left in lefts:
            rows = tree_key_mask(width, before, read_bucket, window_left=left)
            masks[left] = torch.tensor(rows, dtype=torch.bool, device=device).view(
                1, 1, len(rows), read_bucket
            )
        return masks

    def _ensure_weight_views(self, torch, device):
        if self._weight_views is not None:
            return
        with torch.inference_mode():
            views = self.linear_runtime._make_weight_views(torch, device)
        weights = views["weights"]
        if any(weight.requires_grad for weight in weights.values()):
            views = dict(views)
            views["weights"] = {
                kind: weight.detach() for kind, weight in weights.items()
            }
        self._weight_views = views

    def _prepare_inputs(self, cache, before, tokens, depths, paths, width,
                        read_bucket, *, include_idx=True):
        import torch

        device = cache.k_cache.device
        idx = (torch.tensor([tokens], dtype=torch.long, device=device)
               if include_idx else None)
        rope = self._make_rope(cache, before, tokens, depths, paths, device)
        masks = self._position_masks(width, before, read_bucket, device)
        return idx, rope, masks

    def _trunk(self, idx, cache, positions, offsets, rope, kinds, attention):
        import torch
        return self.linear_runtime._trunk(
            torch,
            idx,
            cache,
            positions,
            offsets,
            rope,
            kinds,
            self._weight_views,
            attention,
        )

    def _build_graph(self, cache, before, width, read_bucket, tokens,
                     depths, paths, kinds):
        import torch

        device = cache.k_cache.device
        setup_start = time.perf_counter()
        self._ensure_weight_views(torch, device)
        _, rope, masks = self._prepare_inputs(
            cache, before, tokens, depths, paths, width, read_bucket,
            include_idx=False,
        )
        node_count = len(tokens)
        static_idx = torch.empty((1, node_count), dtype=torch.long, device=device)
        static_rope = _clone_tree(torch, rope)
        static_positions = torch.empty(node_count, dtype=torch.long, device=device)
        static_offsets = torch.zeros(node_count, dtype=torch.long, device=device)
        static_masks = {left: torch.empty_like(mask) for left, mask in masks.items()}
        static_before = torch.empty((), dtype=torch.long, device=device)
        attention = self._make_attention(
            cache, static_before, width, read_bucket, static_masks
        )
        setup_ms = (time.perf_counter() - setup_start) * 1000.0

        def update(new_tokens, new_before, new_rope, new_masks):
            token_tensor = torch.tensor([new_tokens], dtype=torch.long, device=device)
            static_idx.copy_(token_tensor)
            static_positions.copy_(torch.tensor(
                [new_before + int(depth) for depth in depths],
                dtype=torch.long,
                device=device,
            ))
            static_before.fill_(int(new_before))
            _copy_tree(torch, static_rope, new_rope)
            for left, new_mask in new_masks.items():
                static_masks[left].copy_(new_mask)
            cache.store_tokens(static_idx, int(new_before))
            self._copy_bytes["static_token_inputs"] += int(
                static_idx.numel() * static_idx.element_size()
            )
            self._copy_bytes["rope_inputs"] += self._rope_numel_bytes(torch, new_rope)
            self._copy_bytes["attention_masks"] += sum(
                int(mask.numel() * mask.element_size())
                for mask in static_masks.values()
            )
            self._copy_bytes["token_cache_writes"] += int(
                static_idx.numel() * static_idx.element_size()
            )

        def graph_forward():
            return self._trunk(
                static_idx, cache, static_positions, static_offsets,
                static_rope, kinds, attention,
            )

        torch.cuda.synchronize(device)
        warm_stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        warm_stream.wait_stream(current)
        warm_start = time.perf_counter()
        with _without_cache_advance(cache), torch.cuda.stream(warm_stream), torch.inference_mode():
            for _ in range(2):
                update(tokens, before, rope, masks)
                graph_forward()
        current.wait_stream(warm_stream)
        torch.cuda.synchronize(device)
        warmup_ms = (time.perf_counter() - warm_start) * 1000.0

        memory_before = int(torch.cuda.memory_allocated(device))
        graph = torch.cuda.CUDAGraph()
        capture_start = time.perf_counter()
        with _without_cache_advance(cache), torch.inference_mode(), torch.cuda.graph(graph):
            graph_outputs = graph_forward()
        torch.cuda.synchronize(device)
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
        memory_after = int(torch.cuda.memory_allocated(device))

        state = dict(
            graph=graph,
            outputs=graph_outputs,
            cache=cache,
            width=width,
            read_bucket=read_bucket,
            kinds=tuple(kinds),
            static_idx=static_idx,
            static_rope=static_rope,
            static_positions=static_positions,
            static_offsets=static_offsets,
            static_masks=static_masks,
            static_before=static_before,
            attention=attention,
            update=update,
            graph_forward=graph_forward,
        )
        self._graph_events.append(dict(
            cache_id=id(cache),
            width=width,
            node_count=node_count,
            read_bucket=read_bucket,
            first_before=int(before),
            prepare_ms=setup_ms,
            warmup_ms=warmup_ms,
            capture_ms=capture_ms,
            memory_allocated_before=memory_before,
            memory_allocated_after=memory_after,
            memory_allocated_delta=memory_after - memory_before,
            kv_bytes=self._kv_bytes(cache),
        ))
        return state

    @staticmethod
    def _rope_numel_bytes(torch, value):
        if value is None:
            return 0
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, (tuple, list)):
            return sum(TargetTreeRuntime._rope_numel_bytes(torch, item) for item in value)
        return 0

    @staticmethod
    def _kv_bytes(cache):
        return int(
            cache.k_cache.numel() * cache.k_cache.element_size()
            + cache.v_cache.numel() * cache.v_cache.element_size()
        )

    def _run_graph(self, state, cache, before, tokens, depths, paths,
                   kinds, width, p1):
        import torch

        _, rope, masks = self._prepare_inputs(
            cache, before, tokens, depths, paths, width, state["read_bucket"],
            include_idx=False,
        )
        with torch.inference_mode():
            state["update"](tokens, before, rope, masks)
            state["graph"].replay()
            logits, pre_final_hidden, _post_final_hidden = state["outputs"]
            logits = logits[0]
            hidden = pre_final_hidden[0]
            self._rank_mask(logits, cache, before, p1, width)
        self._copy_bytes["graph_output_views"] += 0
        return logits, hidden

    def _run_eager(self, cache, before, tokens, depths, paths, kinds,
                   width, read_bucket, p1):
        import torch

        self._ensure_weight_views(torch, cache.k_cache.device)
        idx, rope, masks = self._prepare_inputs(
            cache, before, tokens, depths, paths, width, read_bucket
        )
        positions = torch.tensor(
            [before + int(depth) for depth in depths],
            dtype=torch.long,
            device=cache.k_cache.device,
        )
        offsets = torch.zeros_like(positions)
        attention = self._make_attention(cache, before, width, read_bucket, masks)
        cache.store_tokens(idx, before)
        self._copy_bytes["token_cache_writes"] += int(idx.numel() * idx.element_size())
        self._copy_bytes["rope_inputs"] += self._rope_numel_bytes(torch, rope)
        self._copy_bytes["attention_masks"] += sum(
            int(mask.numel() * mask.element_size()) for mask in masks.values()
        )
        with _without_cache_advance(cache), torch.inference_mode():
            logits, hidden, _post = self._trunk(
                idx, cache, positions, offsets, rope, kinds, attention
            )
            logits = logits[0]
            hidden = hidden[0]
            self._rank_mask(logits, cache, before, p1, width)
        return logits, hidden

    def verify(self, cache, pending_f: int, p1: int,
               features: list[int], p2s: list[int]):
        """Evaluate all sibling Feature branches without changing cache position."""
        if self._active:
            raise RuntimeError("nested or concurrent verify calls are unsupported")
        if self._pending is not None:
            raise RuntimeError("commit or clear the previous tree before verify")
        self._validate_model()
        tokens, parents, depths, kinds, paths = self._node_tokens(
            pending_f, p1, list(features), list(p2s)
        )
        width = len(features)
        before = self._validate_cache(cache, tokens, width, self.graph_enabled)
        read_bucket = _ceil_power_of_two(before + len(tokens), int(cache.max_seq_len))
        self._active = True
        self._calls["verify_attempts"] += 1
        try:
            if self.graph_enabled:
                graph_key = (id(cache), width, read_bucket)
                state = self._graphs.get(graph_key)
                if state is None:
                    state = self._build_graph(
                        cache, before, width, read_bucket, tokens,
                        depths, paths, kinds,
                    )
                    self._graphs[graph_key] = state
                if state["cache"] is not cache:
                    raise RuntimeError("graph cache identity was reused unexpectedly")
                logits, hidden = self._run_graph(
                    state, cache, before, tokens, depths, paths, kinds, width, p1
                )
                self._calls["graph_forward"] += 1
                self._calls[f"graph_forward_W{width}"] += 1
            else:
                logits, hidden = self._run_eager(
                    cache, before, tokens, depths, paths, kinds,
                    width, read_bucket, p1,
                )
                self._calls["eager_forward"] += 1
                self._calls[f"eager_forward_W{width}"] += 1
            self._pending = dict(
                cache=cache,
                before=before,
                width=width,
                node_tokens=tuple(tokens),
                parents=tuple(parents),
                depths=tuple(depths),
                read_bucket=read_bucket,
                output_logits=logits,
                output_hidden=hidden,
            )
            self._calls["verify"] += 1
            return logits, hidden
        except Exception:
            self._calls["verify_errors"] += 1
            raise
        finally:
            self._active = False

    def commit(self, cache, node_indices):
        """Compact one root-to-node path and advance the cache by its node count.

        ``node_indices`` must be a path prefix such as ``[0]``, ``[0, 1]``,
        ``[0, 1, 2]`` or ``[0, 1, 4, 5]``.  An empty path clears the staged tree
        and leaves the cache position unchanged.
        """
        import torch

        pending = self._pending
        if pending is None:
            raise RuntimeError("commit requires a successful verify")
        if pending["cache"] is not cache:
            raise ValueError("commit cache differs from the cache used by verify")
        if int(cache.get_pos()) != int(pending["before"]):
            raise RuntimeError("cache position changed between verify and commit")
        path = tuple(int(node) for node in node_indices)
        parents = pending["parents"]
        if len(set(path)) != len(path):
            raise ValueError("commit path cannot repeat a tree node")
        for index, node in enumerate(path):
            if not 0 <= node < len(parents):
                raise ValueError(f"commit node {node} is outside the verified tree")
            expected_parent = -1 if index == 0 else path[index - 1]
            if parents[node] != expected_parent:
                raise ValueError("commit nodes must form one root-to-node path")

        before = int(pending["before"])
        node_count = len(pending["node_tokens"])
        path_len = len(path)
        device = cache.k_cache.device
        with torch.inference_mode():
            contiguous = path == tuple(range(path_len))
            if contiguous:
                clear_start = before + path_len
                clear_count = node_count - path_len
                if clear_count:
                    cache.token_cache[:, clear_start:before + node_count].fill_(-1)
                    self._copy_bytes["commit_token_clear"] += int(
                        clear_count * cache.token_cache.element_size()
                    )
            else:
                cache.token_cache[:, before:before + node_count].fill_(-1)
                self._copy_bytes["commit_token_clear"] += int(
                    node_count * cache.token_cache.element_size()
                )
            if path_len and not contiguous:
                source = torch.tensor(
                    [before + node for node in path],
                    dtype=torch.long,
                    device=device,
                )
                staged_k = cache.k_cache.index_select(2, source)
                staged_v = cache.v_cache.index_select(2, source)
                payload = int(
                    staged_k.numel() * staged_k.element_size()
                    + staged_v.numel() * staged_v.element_size()
                )
                cache.k_cache[:, :, before:before + path_len].copy_(staged_k)
                cache.v_cache[:, :, before:before + path_len].copy_(staged_v)
                tokens = torch.tensor(
                    [[pending["node_tokens"][node] for node in path]],
                    dtype=torch.long,
                    device=device,
                )
                cache.token_cache[:, before:before + path_len].copy_(tokens)
                self._copy_bytes["commit_kv_payload"] += payload
                self._copy_bytes["commit_kv_device_moved"] += 2 * payload
                self._copy_bytes["commit_token_write"] += int(
                    tokens.numel() * tokens.element_size()
                )
            if path_len and contiguous:
                self._calls["commit_contiguous"] += 1
            elif path_len:
                self._calls["commit_branched"] += 1
            cache.cache_seqlens.fill_(before + path_len)
        after = before + path_len
        self._calls["commit"] += 1
        self._calls[f"commit_nodes_{path_len}"] += 1
        self._pending = None
        return after

