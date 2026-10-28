"""CPU-only state-safety tests for action_probe.py."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from action_probe import measure_action_operations, probe_action_costs


def _indices(shape):
    if not shape:
        yield ()
        return
    index = [0] * len(shape)
    while True:
        yield tuple(index)
        axis = len(shape) - 1
        while axis >= 0:
            index[axis] += 1
            if index[axis] < shape[axis]:
                break
            index[axis] = 0
            axis -= 1
        if axis < 0:
            return


class FakeTensor:
    def __init__(self, shape, *, fill=0, data=None, offset=0, strides=None):
        self.shape = tuple(int(value) for value in shape)
        self._offset = int(offset)
        self._strides = tuple(strides or self._make_strides(self.shape))
        size = 1
        for value in self.shape:
            size *= value
        self._data = data if data is not None else [fill] * size
        self.device = SimpleNamespace(type="cpu")

    @staticmethod
    def _make_strides(shape):
        strides = []
        stride = 1
        for value in reversed(shape):
            strides.append(stride)
            stride *= value
        return tuple(reversed(strides))

    def _flat(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape):
            raise IndexError("wrong number of indices")
        return self._offset + sum(int(item) * stride
                                  for item, stride in zip(index, self._strides))

    def get(self, index):
        return self._data[self._flat(index)]

    def set(self, index, value):
        self._data[self._flat(index)] = value

    def narrow(self, dim, start, length):
        dim = int(dim)
        start = int(start)
        length = int(length)
        if not 0 <= dim < len(self.shape):
            raise IndexError("narrow dimension out of range")
        if start < 0 or length < 0 or start + length > self.shape[dim]:
            raise IndexError("narrow range out of bounds")
        shape = list(self.shape)
        shape[dim] = length
        return FakeTensor(
            shape, data=self._data,
            offset=self._offset + start * self._strides[dim],
            strides=self._strides,
        )

    def detach(self):
        return self

    def clone(self):
        result = FakeTensor(self.shape)
        for index in _indices(self.shape):
            result.set(index, self.get(index))
        return result

    def copy_(self, other):
        if tuple(self.shape) != tuple(other.shape):
            raise ValueError("copy shape mismatch")
        for index in _indices(self.shape):
            self.set(index, other.get(index))
        return self

    def fill_(self, value):
        for index in _indices(self.shape):
            self.set(index, value)
        return self

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[int(dim)]

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def element_size(self):
        return 4

    def visible_values(self):
        return tuple(self.get(index) for index in _indices(self.shape))


class FakeCache:
    def __init__(self, capacity=32, position=8):
        self.max_seq_len = int(capacity)
        self.batch_size = 1
        self.k_cache = FakeTensor((2, 1, capacity, 1, 1))
        self.v_cache = FakeTensor((2, 1, capacity, 1, 1))
        self.token_cache = FakeTensor((1, capacity), fill=-1)
        self.cache_seqlens = FakeTensor((1,), fill=position)
        for slot in range(position):
            self.token_cache.set((0, slot), 100 + slot)
            for layer in range(2):
                self.k_cache.set((layer, 0, slot, 0, 0), layer * 1000 + slot)
                self.v_cache.set((layer, 0, slot, 0, 0), layer * 2000 + slot)

    def get_pos(self):
        return int(self.cache_seqlens.get((0,)))

    def dump(self):
        return (
            self.k_cache.visible_values(),
            self.v_cache.visible_values(),
            self.token_cache.visible_values(),
            self.cache_seqlens.visible_values(),
        )


class FakeRuntime:
    def __init__(self, *, rescue=False):
        self.rescue = bool(rescue)
        self.graph_enabled = True
        self._graphs = {}
        self._graphs_by_count = {1: self._graphs} if rescue else {}
        self._disabled_graphs = set()
        self._weight_views = None
        self._calls = Counter()
        self._fallbacks = Counter()
        self._copy_bytes = Counter()
        self._graph_events = []
        self._pending = None
        self._active = False
        self._backup_positions = None
        self.NODE_COUNT = 5 if rescue else 4
        self.linear_runtime = None

    def stats(self):
        count = (sum(len(group) for group in self._graphs_by_count.values())
                 if self.rescue else len(self._graphs))
        return {
            "graph_enabled": True,
            "graph_count": count,
            "graph_events": [dict(event) for event in self._graph_events],
            "calls": dict(self._calls),
            "fallbacks": dict(self._fallbacks),
            "copy_bytes": dict(self._copy_bytes),
        }

    def _graph_group(self):
        return self._graphs

    def prime(self, cache, node_count):
        key = (id(cache), int(node_count))
        self._graphs[key] = {
            "cache": cache,
            "static_idx": FakeTensor((1, node_count), fill=81),
            "static_positions": FakeTensor((node_count,), fill=82),
            "static_masks": {"left": FakeTensor((node_count, 2), fill=83)},
            "outputs": (FakeTensor((node_count, 2), fill=84),),
        }
        self._graph_events.append({"node_count": node_count, "source": "prime"})

    def verify(self, cache, pending_f, p1, features, p2s, *, backup_position=None):
        if self._active or self._pending is not None:
            raise RuntimeError("fake runtime is busy")
        node_count = 5 if backup_position is not None else 4
        before = cache.get_pos()
        key = (id(cache), node_count)
        graph = self._graphs.get(key)
        if graph is None:
            graph = {
                "cache": cache,
                "static_idx": FakeTensor((1, node_count)),
                "static_positions": FakeTensor((node_count,)),
                "static_masks": {"left": FakeTensor((node_count, 2))},
                "outputs": (FakeTensor((node_count, 2)),),
            }
            self._graphs[key] = graph
            self._graph_events.append({"node_count": node_count, "source": "capture"})
        tokens = [int(pending_f), int(p1), int(features[0]), int(p2s[0])]
        if backup_position is not None:
            tokens.append(int(backup_position))
        self._active = True
        try:
            for node, token in enumerate(tokens):
                cache.token_cache.set((0, before + node), token)
                for layer in range(2):
                    cache.k_cache.set((layer, 0, before + node, 0, 0), token + 10 * layer)
                    cache.v_cache.set((layer, 0, before + node, 0, 0), token + 20 * layer)
                graph["static_idx"].set((0, node), token)
                graph["static_positions"].set((node,), before + node)
                graph["static_masks"]["left"].set((node, 0), token)
                graph["outputs"][0].set((node, 0), token * 2)
            self._pending = {"cache": cache, "before": before, "tokens": tuple(tokens)}
            self._calls["graph_forward"] += 1
            return FakeTensor((node_count, 3)), FakeTensor((node_count, 2))
        finally:
            self._active = False

    def commit(self, cache, path):
        pending = self._pending
        if pending is None or pending["cache"] is not cache:
            raise RuntimeError("missing fake pending verify")
        before = pending["before"]
        node_count = len(pending["tokens"])
        path = tuple(int(node) for node in path)
        contiguous = path == tuple(range(len(path)))
        if not contiguous and path:
            staged = []
            for node in path:
                slot = before + node
                staged.append((
                    pending["tokens"][node],
                    [cache.k_cache.get((layer, 0, slot, 0, 0)) for layer in range(2)],
                    [cache.v_cache.get((layer, 0, slot, 0, 0)) for layer in range(2)],
                ))
            cache.token_cache.narrow(1, before, node_count).fill_(-1)
            for offset, (token, keys, values) in enumerate(staged):
                cache.token_cache.set((0, before + offset), token)
                for layer in range(2):
                    cache.k_cache.set((layer, 0, before + offset, 0, 0), keys[layer])
                    cache.v_cache.set((layer, 0, before + offset, 0, 0), values[layer])
        else:
            clear_start = before + len(path)
            if clear_start < before + node_count:
                cache.token_cache.narrow(
                    1, clear_start, before + node_count - clear_start
                ).fill_(-1)
        cache.cache_seqlens.set((0,), before + len(path))
        self._pending = None
        self._calls["commit"] += 1
        self._calls[f"commit_nodes_{len(path)}"] += 1
        return before + len(path)


class FakeLinearRuntime:
    def __init__(self):
        self._graphs = {}
        self._disabled_graphs = {}
        self._cache_pool = {}
        self._weight_views = {"source": "original"}
        self._calls = Counter()
        self._fallbacks = Counter()
        self._setup = {
            "runtime_init_ms": 1.25,
            "cache_events": [{"event": "original_cache"}],
            "graph_events": [],
        }
        self._numerical_records = [{"mode": "original", "value": 1}]

    def _new_state(self):
        return {
            "static_idx": FakeTensor((1, 1), fill=31),
            "graph_outputs": (
                FakeTensor((1, 2), fill=32),
                FakeTensor((1, 2), fill=33),
                FakeTensor((1, 2), fill=34),
            ),
            "pos_tensor": FakeTensor((1,), fill=35),
            "offsets": FakeTensor((1,), fill=36),
            "key_positions": FakeTensor((8,), fill=37),
            "static_rope": (FakeTensor((1, 1, 2)), FakeTensor((1, 1, 2))),
            "rope_source": FakeTensor((1, 1, 2), fill=38),
        }

    def stats(self):
        return {
            "runtime_init_ms": self._setup["runtime_init_ms"],
            "cache_events": [dict(event) for event in self._setup["cache_events"]],
            "graph_events": [dict(event) for event in self._setup["graph_events"]],
            "graph_count": len(self._graphs),
            "disabled_graphs": dict(self._disabled_graphs),
            "calls": dict(self._calls),
            "fallbacks": dict(self._fallbacks),
            "numerical_records": len(self._numerical_records),
            "numerical_recording": True,
            "cache_bytes_by_capacity": {},
        }

    def forward(self, cache, token):
        key = (id(cache), 1, 16, ("position",))
        state = self._graphs.get(key)
        if state is None:
            state = self._new_state()
            self._graphs[key] = state
            self._setup["graph_events"].append({
                "query_tokens": 1, "read_bucket": 16, "capture_ms": 2.0,
            })
        state["static_idx"].set((0, 0), int(token))
        state["pos_tensor"].set((0,), cache.get_pos())
        state["graph_outputs"][0].set((0, 0), int(token) * 3)
        state["static_rope"][0].set((0, 0, 0), int(token) * 4)
        self._calls["graph_forward"] += 1
        self._numerical_records.append({"mode": "graph", "token": int(token)})
        self._cache_pool[64] = cache
        self._weight_views["source"] = "probe"


def _linear_state_values(state):
    return (
        state["static_idx"].visible_values(),
        state["graph_outputs"][0].visible_values(),
        state["pos_tensor"].visible_values(),
        state["static_rope"][0].visible_values(),
        state["rope_source"].visible_values(),
    )


def _recovery(cache, token, aux):
    position = cache.get_pos()
    cache.token_cache.set((0, position), int(token))
    for layer in range(2):
        cache.k_cache.set((layer, 0, position, 0, 0), int(token) + 100 + layer)
        cache.v_cache.set((layer, 0, position, 0, 0), int(token) + 200 + layer)
    cache.cache_seqlens.set((0,), position + 1)
    aux.append((position, int(token)))


def _ready_objects():
    cache = FakeCache()
    tree = FakeRuntime()
    rescue = FakeRuntime(rescue=True)
    aux = []
    return cache, tree, rescue, aux


def test_probe_measures_requested_paths_and_restores_state():
    cache, tree, rescue, aux = _ready_objects()
    cache_before = cache.dump()
    tree_before = tree.stats()
    rescue_before = rescue.stats()
    aux_before = list(aux)

    result = measure_action_operations(
        cache=cache,
        tree_runtime=tree,
        rescue_runtime=rescue,
        pending_f=20,
        p1=3,
        features=[21],
        p2s=[4],
        candidate_backups=[5, 6],
        miss_position=7,
        forced_miss_token=True,
        recovery_forward=lambda active_cache, token: _recovery(
            active_cache, token, aux
        ),
        repeats=2,
        accepted_path_lengths=(1, 2, 3, 4),
        snapshot_aux_state=lambda: list(aux),
        restore_aux_state=lambda values: aux.__setitem__(slice(None), values),
    )

    actions = {row["action_id"]: row for row in result["actions"]}
    assert "k0_common_path_2" in actions
    assert actions["k0_eos_after_p1_rejection"]["commit_path"] == [0]
    assert actions["k0_eos_after_p1_rejection"]["repeat_details"][0]["operation"]["target_recovery_calls"] == 0
    assert actions["k0_eos_after_p1_rejection"]["repeat_details"][0]["operation"]["path_kind"] == "eos_after_p1_rejection"
    assert "k0_common_path_3" in actions
    assert "k0_common_path_4" in actions
    assert "k1_common_path_2_backup_5" in actions
    assert actions["k1_eos_after_p1_rejection_backup_5"]["commit_path"] == [0]
    assert actions["k1_eos_after_p1_rejection_backup_5"]["repeat_details"][0]["operation"]["target_recovery_calls"] == 0
    assert "k1_common_path_3_backup_5" in actions
    assert "k1_common_path_4_backup_5" in actions
    assert actions["k1_backup_hit_5"]["commit_path"] == [0, 4]
    miss = actions["k1_miss_recover_5"]["repeat_details"][0]
    assert miss["commit_path"] == [0]
    assert miss["operation"]["target_recovery_calls"] == 1
    assert miss["operation"]["cache_pos_after_action"] == 10
    assert actions["k0_miss_recover"]["recovery_token"] == 7
    assert actions["k1_miss_recover_5"]["recovery_token"] == 7
    assert len(result["prewarm"]) == 3
    assert all(row["phase"] == "prewarm" for row in result["prewarm"])
    assert all(len(row["repeat_wall_ms"]) == 2 for row in result["actions"])
    assert result["rng_used"] is False
    assert result["input"]["forced_miss_token"] is True
    assert result["cache_restored"] is True
    assert result["runtime_bookkeeping_restored"] is True
    assert cache.dump() == cache_before
    assert tree.stats() == tree_before
    assert rescue.stats() == rescue_before
    assert aux == aux_before


def test_preexisting_graph_buffers_are_restored():
    cache, tree, rescue, aux = _ready_objects()
    tree.prime(cache, 4)
    rescue.prime(cache, 5)
    graph_objects = [tree._graphs[(id(cache), 4)], rescue._graphs[(id(cache), 5)]]
    saved_buffers = [
        (
            state["static_idx"].visible_values(),
            state["static_positions"].visible_values(),
            state["static_masks"]["left"].visible_values(),
            state["outputs"][0].visible_values(),
        )
        for state in graph_objects
    ]
    cache_before = cache.dump()
    tree_before = tree.stats()
    rescue_before = rescue.stats()
    linear = FakeLinearRuntime()
    tree.linear_runtime = linear
    rescue.linear_runtime = linear
    linear_state = linear._new_state()
    linear_key = (id(cache), 1, 16, ("position",))
    linear._graphs[linear_key] = linear_state
    linear_before = linear.stats()
    linear_buffers_before = _linear_state_values(linear_state)

    probe_action_costs(
        cache=cache, tree_runtime=tree, rescue_runtime=rescue,
        pending_f=20, p1=3, feature=21, p2=4,
        backup_candidates=[5], miss_position=7,
        recovery_forward=lambda active_cache, token: (
            linear.forward(active_cache, token),
            _recovery(active_cache, token, aux),
        ),
        repeats=1, accepted_path_lengths=(2,),
    )

    restored_buffers = [
        (
            state["static_idx"].visible_values(),
            state["static_positions"].visible_values(),
            state["static_masks"]["left"].visible_values(),
            state["outputs"][0].visible_values(),
        )
        for state in graph_objects
    ]
    assert restored_buffers == saved_buffers
    assert cache.dump() == cache_before
    assert tree.stats() == tree_before
    assert rescue.stats() == rescue_before
    assert linear.stats() == linear_before
    assert _linear_state_values(linear_state) == linear_buffers_before


def test_linear_runtime_statistics_and_new_graph_setup_are_restored():
    cache, tree, rescue, aux = _ready_objects()
    linear = FakeLinearRuntime()
    tree.linear_runtime = linear
    rescue.linear_runtime = linear
    linear_before = linear.stats()

    result = probe_action_costs(
        cache=cache, tree_runtime=tree, rescue_runtime=rescue,
        pending_f=20, p1=3, feature=21, p2=4,
        backup_candidates=[5], miss_position=7,
        recovery_forward=lambda active_cache, token: (
            linear.forward(active_cache, token),
            _recovery(active_cache, token, aux),
        ),
        repeats=1, accepted_path_lengths=(2,),
    )

    assert linear.stats() == linear_before
    assert linear._graphs == {}
    assert linear._setup["graph_events"] == []
    assert linear._numerical_records == [{"mode": "original", "value": 1}]
    linear_summary = result["runtime_state_before"]["linear"]
    assert linear_summary["graph_count"] == linear_before["graph_count"]
    assert linear_summary["graph_events"] == linear_before["graph_events"]
    assert linear_summary["cache_events"] == linear_before["cache_events"]
    assert linear_summary["calls"] == linear_before["calls"]
    assert linear_summary["numerical_records"] == linear_before["numerical_records"]
    assert len(result["prewarm"]) == 3
    recovery_prewarm = result["prewarm"][-1]
    assert len(recovery_prewarm["linear_new_graph_events"]) == 1
    assert recovery_prewarm["linear_new_graph_events"][0]["query_tokens"] == 1


def test_exception_restores_cache_aux_and_runtime_and_surfaces_error():
    cache, tree, rescue, aux = _ready_objects()
    cache_before = cache.dump()
    tree_before = tree.stats()
    rescue_before = rescue.stats()
    aux_before = list(aux)

    def failing_recovery(active_cache, token):
        _recovery(active_cache, token, aux)
        raise LookupError("synthetic recovery failure")

    try:
        probe_action_costs(
            cache=cache, tree_runtime=tree, rescue_runtime=rescue,
            pending_f=20, p1=3, feature=21, p2=4,
            backup_candidates=[5], miss_position=7,
            recovery_forward=failing_recovery,
            repeats=1, accepted_path_lengths=(2,),
            snapshot_aux_state=lambda: list(aux),
            restore_aux_state=lambda values: aux.__setitem__(slice(None), values),
        )
    except LookupError as error:
        assert str(error) == "synthetic recovery failure"
    else:
        raise AssertionError("recovery exception was swallowed")

    assert cache.dump() == cache_before
    assert tree.stats() == tree_before
    assert rescue.stats() == rescue_before
    assert aux == aux_before
    assert tree._pending is None and rescue._pending is None
    assert tree._active is False and rescue._active is False
    assert rescue._backup_positions is None


if __name__ == "__main__":
    test_probe_measures_requested_paths_and_restores_state()
    test_preexisting_graph_buffers_are_restored()
    test_linear_runtime_statistics_and_new_graph_setup_are_restored()
    test_exception_restores_cache_aux_and_runtime_and_surfaces_error()
    print("action_probe CPU tests passed")
