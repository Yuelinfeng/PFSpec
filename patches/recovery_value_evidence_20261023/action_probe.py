"""Same-cache action-cost measurements for P/F tree verification.

The probe forces the listed commit paths. It never samples a token and it does
not measure Ranker, MLP, policy construction, or full decode time. Cache data and
runtime bookkeeping are restored after the probe; graph setup is reported as a
separate prewarm action and newly built graphs are removed afterward.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import statistics
import time
from typing import Any


class ProbeRestoreError(RuntimeError):
    """Raised when the probe cannot put the supplied objects back as found."""


_RUNTIME_FIELDS = (
    "_graphs", "_graphs_by_count", "_disabled_graphs", "_cache_pool",
    "_weight_views", "_calls", "_fallbacks", "_copy_bytes",
    "_graph_events", "_setup", "_numerical_records", "_pending",
    "_active", "_backup_positions", "NODE_COUNT",
)
_GRAPH_BUFFER_FIELDS = (
    "static_idx", "static_rope", "static_positions", "static_offsets",
    "static_masks", "static_before", "outputs", "graph_outputs",
    "pos_tensor", "offsets", "key_positions", "rope_source",
)


def _runtime_summaries(tree_runtime, rescue_runtime):
    result = {
        "tree": _runtime_summary(tree_runtime),
        "rescue": _runtime_summary(rescue_runtime),
    }
    tree_linear = getattr(tree_runtime, "linear_runtime", None)
    rescue_linear = getattr(rescue_runtime, "linear_runtime", None)
    if tree_linear is not None and tree_linear is rescue_linear:
        result["linear"] = _runtime_summary(tree_linear)
    else:
        if tree_linear is not None:
            result["tree_linear"] = _runtime_summary(tree_linear)
        if rescue_linear is not None:
            result["rescue_linear"] = _runtime_summary(rescue_linear)
    return result


def _slice_clone(tensor, dim: int, end: int):
    if not hasattr(tensor, "narrow"):
        raise TypeError("cache tensors must expose narrow()")
    view = tensor.narrow(int(dim), 0, int(end))
    return view.detach().clone()


def _slice_copy(destination, source, dim: int, end: int):
    if not hasattr(destination, "narrow"):
        raise TypeError("cache tensors must expose narrow()")
    destination.narrow(int(dim), 0, int(end)).copy_(source)


def _cache_snapshot(cache, end: int):
    before = int(cache.get_pos())
    end = int(end)
    if end < before:
        raise ValueError("snapshot extent ends before the logical cache position")
    if int(cache.k_cache.size(2)) < end or int(cache.v_cache.size(2)) < end:
        raise ValueError("K/V cache is shorter than the requested snapshot extent")
    if int(cache.token_cache.size(1)) < end:
        raise ValueError("token cache is shorter than the requested snapshot extent")
    return {
        "position": before,
        "extent": end,
        "k": _slice_clone(cache.k_cache, 2, end),
        "v": _slice_clone(cache.v_cache, 2, end),
        "tokens": _slice_clone(cache.token_cache, 1, end),
        "seqlens": cache.cache_seqlens.detach().clone(),
    }


def _restore_cache(cache, snapshot):
    end = int(snapshot["extent"])
    _slice_copy(cache.k_cache, snapshot["k"], 2, end)
    _slice_copy(cache.v_cache, snapshot["v"], 2, end)
    _slice_copy(cache.token_cache, snapshot["tokens"], 1, end)
    cache.cache_seqlens.copy_(snapshot["seqlens"])
    actual = int(cache.get_pos())
    if actual != int(snapshot["position"]):
        raise ProbeRestoreError(
            f"cache position restore failed: expected {snapshot['position']}, got {actual}"
        )


def _clone_buffer(value):
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, dict):
        return {key: _clone_buffer(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_buffer(item) for item in value)
    if isinstance(value, list):
        return [_clone_buffer(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    return value


def _restore_buffer(destination, snapshot):
    if snapshot is None:
        return
    if isinstance(destination, dict) and isinstance(snapshot, dict):
        for key in destination.keys() & snapshot.keys():
            _restore_buffer(destination[key], snapshot[key])
        return
    if isinstance(destination, tuple) and isinstance(snapshot, tuple):
        for left, right in zip(destination, snapshot):
            _restore_buffer(left, right)
        return
    if isinstance(destination, list) and isinstance(snapshot, list):
        for left, right in zip(destination, snapshot):
            _restore_buffer(left, right)
        return
    if hasattr(destination, "copy_") and hasattr(snapshot, "shape"):
        destination.copy_(snapshot)


def _mapping_state(value):
    if not isinstance(value, dict):
        return None
    nested = []
    for item in value.values():
        if isinstance(item, dict):
            nested.append((item, dict(item)))
    return value, dict(value), nested


def _restore_mapping(state):
    if state is None:
        return
    mapping, entries, nested = state
    for child, child_entries in nested:
        child.clear()
        child.update(child_entries)
    mapping.clear()
    mapping.update(entries)


def _snapshot_mutable_containers(value, seen=None):
    if seen is None:
        seen = set()
    if not isinstance(value, (dict, list)) or id(value) in seen:
        return None
    seen.add(id(value))
    if isinstance(value, dict):
        entries = list(value.items())
        children = [
            _snapshot_mutable_containers(item, seen)
            for _key, item in entries
        ]
        return ("dict", value, entries, children)
    entries = list(value)
    children = [_snapshot_mutable_containers(item, seen) for item in entries]
    return ("list", value, entries, children)


def _restore_mutable_containers(snapshot):
    if snapshot is None:
        return
    kind, container, entries, children = snapshot
    for child in children:
        _restore_mutable_containers(child)
    if kind == "dict":
        container.clear()
        container.update(entries)
    else:
        container.clear()
        container.extend(entries)


def _runtime_state(runtime):
    fields = {}
    mapping_states = {}
    list_states = {}
    nested_states = {}
    graph_buffers = []
    seen_graph_states = set()
    for name in _RUNTIME_FIELDS:
        if not hasattr(runtime, name):
            continue
        value = getattr(runtime, name)
        fields[name] = value
        if name in ("_setup", "_numerical_records"):
            nested_states[name] = _snapshot_mutable_containers(value)
        state = _mapping_state(value)
        if state is not None:
            mapping_states[name] = state
            if name in ("_graphs", "_graphs_by_count"):
                if name == "_graphs":
                    graph_states = [
                        graph_state for graph_state in value.values()
                        if isinstance(graph_state, dict)
                    ]
                else:
                    graph_states = [
                        graph_state
                        for group in value.values()
                        if isinstance(group, dict)
                        for graph_state in group.values()
                        if isinstance(graph_state, dict)
                    ]
                for graph_state in graph_states:
                    if id(graph_state) in seen_graph_states:
                        continue
                    seen_graph_states.add(id(graph_state))
                    buffers = {
                        field: _clone_buffer(graph_state.get(field))
                        for field in _GRAPH_BUFFER_FIELDS
                        if field in graph_state
                    }
                    graph_buffers.append((graph_state, buffers))
        elif isinstance(value, list):
            list_states[name] = (value, list(value))
    return {
        "runtime": runtime,
        "fields": fields,
        "mapping_states": mapping_states,
        "list_states": list_states,
        "nested_states": nested_states,
        "graph_buffers": graph_buffers,
    }


def _restore_runtime(snapshot):
    runtime = snapshot["runtime"]
    for state, buffers in snapshot["graph_buffers"]:
        for key, value in buffers.items():
            _restore_buffer(state.get(key), value)
    for name, state in snapshot["mapping_states"].items():
        _restore_mapping(state)
        setattr(runtime, name, state[0])
    for name, (items, values) in snapshot["list_states"].items():
        items.clear()
        items.extend(values)
        setattr(runtime, name, items)
    for name, state in snapshot["nested_states"].items():
        _restore_mutable_containers(state)
        setattr(runtime, name, state[1] if state is not None else snapshot["fields"][name])
    for name, value in snapshot["fields"].items():
        if name not in snapshot["mapping_states"] and name not in snapshot["list_states"]:
            setattr(runtime, name, value)


def _runtime_summary(runtime):
    stats_fn = getattr(runtime, "stats", None)
    stats = stats_fn() if callable(stats_fn) else {}
    return {
        "graph_enabled": stats.get("graph_enabled"),
        "graph_count": int(stats.get("graph_count", 0)),
        "graph_events": [dict(event) for event in stats.get("graph_events", ())],
        "cache_events": [dict(event) for event in stats.get("cache_events", ())],
        "calls": dict(stats.get("calls", {})),
        "fallbacks": dict(stats.get("fallbacks", {})),
        "copy_bytes": dict(stats.get("copy_bytes", {})),
        "disabled_graphs": dict(stats.get("disabled_graphs", {})),
        "numerical_records": stats.get("numerical_records"),
        "numerical_recording": stats.get("numerical_recording"),
        "cache_bytes_by_capacity": dict(stats.get("cache_bytes_by_capacity", {})),
        "runtime_init_ms": stats.get("runtime_init_ms"),
        "pending_verify": bool(getattr(runtime, "_pending", None) is not None),
        "active": bool(getattr(runtime, "_active", False)),
        "backup_positions": getattr(runtime, "_backup_positions", None),
    }


def _unique_objects(items):
    result = []
    seen = set()
    for item in items:
        if item is not None and id(item) not in seen:
            result.append(item)
            seen.add(id(item))
    return result


def _synchronize(cache, synchronize):
    if synchronize is not None:
        synchronize(cache)
        return "custom_synchronized_wall"
    device = getattr(cache.k_cache, "device", None)
    device_type = getattr(device, "type", "cpu")
    if device_type == "cuda":
        import torch
        torch.cuda.synchronize(device)
        return "cuda_synchronized_wall"
    return "host_wall"


def _shape0(value):
    shape = getattr(value, "shape", None)
    if shape:
        return int(shape[0])
    try:
        return len(value)
    except TypeError:
        return None


def _runtime_graphs_changed(before, after):
    old_events = before.get("graph_events", ())
    new_events = after.get("graph_events", ())
    return [dict(event) for event in new_events[len(old_events):]]


def _execute_action(action, *, cache, tree_runtime, rescue_runtime,
                    pending_f, p1, feature, p2, recovery_forward):
    before = int(cache.get_pos())
    if action["runtime"] == "tree":
        active_runtime = tree_runtime
        logits, hidden = active_runtime.verify(
            cache, int(pending_f), int(p1), [int(feature)], [int(p2)]
        )
    else:
        active_runtime = rescue_runtime
        logits, hidden = active_runtime.verify(
            cache, int(pending_f), int(p1), [int(feature)], [int(p2)],
            backup_position=int(action["backup_position"]),
        )
    if int(cache.get_pos()) != before:
        raise RuntimeError("tree verification changed logical cache position")
    expected_nodes = 4 if action["runtime"] == "tree" else 5
    if _shape0(logits) != expected_nodes or _shape0(hidden) != expected_nodes:
        raise RuntimeError("Target runtime returned an unexpected node count")
    pending_state = getattr(active_runtime, "_pending", None)
    if not isinstance(pending_state, dict):
        raise RuntimeError("Target runtime did not retain its pending tree state")
    read_bucket = pending_state.get("read_bucket")
    logical_width = pending_state.get("width")
    del logits, hidden
    after_commit = int(active_runtime.commit(cache, action["commit_path"]))
    expected_after_commit = before + len(action["commit_path"])
    if after_commit != expected_after_commit or int(cache.get_pos()) != expected_after_commit:
        raise RuntimeError("tree commit ended at an unexpected logical position")
    if action.get("recovery_token") is not None:
        recovery_forward(cache, int(action["recovery_token"]))
        if int(cache.get_pos()) != before + 2:
            raise RuntimeError("single-token Target recovery did not advance exactly once")
    if getattr(active_runtime, "_pending", None) is not None:
        raise RuntimeError("runtime retained a pending verify after commit")
    if bool(getattr(active_runtime, "_active", False)):
        raise RuntimeError("runtime remained active after commit")
    return {
        "cache_pos_before": before,
        "cache_pos_after_commit": expected_after_commit,
        "cache_pos_after_action": int(cache.get_pos()),
        "target_recovery_calls": 1 if action.get("recovery_token") is not None else 0,
        "node_count": expected_nodes,
        "read_bucket": None if read_bucket is None else int(read_bucket),
        "tree_width": None if logical_width is None else int(logical_width),
        "path_kind": action["path_kind"],
        "commit_path": list(action["commit_path"]),
    }


def probe_action_costs(*, cache, tree_runtime, rescue_runtime, pending_f: int,
                       p1: int, feature: int, p2: int,
                       backup_candidates, miss_position: int,
                       recovery_forward: Callable,
                       forced_miss_token: bool = False,
                       repeats: int = 3, max_candidates: int = 4,
                       accepted_path_lengths=(1, 2, 3, 4),
                       synchronize: Callable | None = None,
                       snapshot_aux_state: Callable | None = None,
                       restore_aux_state: Callable | None = None):
    """Measure same-state verify+commit actions and restore inputs afterward.

    ``recovery_forward(cache, token)`` must perform exactly one ordinary Target
    forward for ``token``. It must not sample or otherwise use the decode RNG.
    If it updates a non-cache object such as a capture hook, provide matching
    ``snapshot_aux_state`` and ``restore_aux_state`` callbacks.

    For each eligible backup ID, the result includes no-backup recovery to
    that ID, the expanded-tree miss recovery to one shared Position, the
    expanded-tree full-accept paths, and the backup-hit action. The shared
    miss Position makes the no-hit recovery pair use the same Target input.
    One untimed-from-the-main-decode perspective setup call per runtime shape
    records graph preparation cost separately; repeated samples use the warmed
    shape. Every timer surrounds only verify+commit and optional Target recovery.
    Snapshot, synchronization before the timer, and restoration are excluded.
    """
    repeats = int(repeats)
    max_candidates = int(max_candidates)
    if repeats < 1 or not 1 <= max_candidates <= 4:
        raise ValueError("repeats must be positive and max_candidates in [1, 4]")
    accepted_path_lengths = tuple(int(length) for length in accepted_path_lengths)
    if not accepted_path_lengths or any(length not in (1, 2, 3, 4)
                                        for length in accepted_path_lengths):
        raise ValueError("accepted_path_lengths must select from 1, 2, 3, and 4")
    if len(set(accepted_path_lengths)) != len(accepted_path_lengths):
        raise ValueError("accepted_path_lengths cannot contain duplicates")
    if not callable(recovery_forward):
        raise TypeError("recovery_forward must be callable")
    if (snapshot_aux_state is None) != (restore_aux_state is None):
        raise ValueError("auxiliary state needs both snapshot and restore callbacks")
    backups = [int(value) for value in backup_candidates]
    if not backups:
        raise ValueError("at least one real backup Position is required")
    if len(set(backups)) != len(backups):
        raise ValueError("backup candidates must be unique")
    backups = backups[:max_candidates]
    if int(p1) in backups:
        raise ValueError("backup candidates must differ from the main Position")
    miss_position = int(miss_position)
    if miss_position == int(p1) or miss_position in backups:
        raise ValueError("miss_position must differ from the main and backup Positions")
    if not all(hasattr(runtime, "verify") and hasattr(runtime, "commit")
               for runtime in (tree_runtime, rescue_runtime)):
        raise TypeError("both runtimes must expose verify() and commit()")

    runtimes = _unique_objects((tree_runtime, rescue_runtime,
                                getattr(tree_runtime, "linear_runtime", None),
                                getattr(rescue_runtime, "linear_runtime", None)))
    for runtime in runtimes:
        if getattr(runtime, "_pending", None) is not None:
            raise RuntimeError("all runtimes must start without a pending verify")
        if bool(getattr(runtime, "_active", False)):
            raise RuntimeError("all runtimes must start inactive")
        if getattr(runtime, "_backup_positions", None) is not None:
            raise RuntimeError("rescue runtime has a backup selection in progress")

    before = int(cache.get_pos())
    max_node_count = 4 + 1
    end = before + max_node_count
    if end > int(cache.max_seq_len):
        raise ValueError("cache capacity cannot hold the largest measured tree")
    initial_cache = _cache_snapshot(cache, end)
    runtime_snapshots = [_runtime_state(runtime) for runtime in runtimes]
    initial_aux = snapshot_aux_state() if snapshot_aux_state is not None else None
    runtime_before = _runtime_summaries(tree_runtime, rescue_runtime)
    measurement_mode = "host_wall"
    setup = []
    samples = []

    def run_one(action, *, phase, repeat_index=None):
        nonlocal measurement_mode
        _restore_cache(cache, initial_cache)
        if snapshot_aux_state is not None:
            restore_aux_state(_clone_buffer(initial_aux))
        runt_before = _runtime_summary(action["runtime_object"])
        linear_runtime = getattr(action["runtime_object"], "linear_runtime", None)
        linear_before = (
            None if linear_runtime is None else _runtime_summary(linear_runtime)
        )
        mode = _synchronize(cache, synchronize)
        start = time.perf_counter()
        operation = None
        failure = None
        try:
            operation = _execute_action(
                action, cache=cache, tree_runtime=tree_runtime,
                rescue_runtime=rescue_runtime, pending_f=pending_f,
                p1=p1, feature=feature, p2=p2,
                recovery_forward=recovery_forward,
            )
            _synchronize(cache, synchronize)
        except BaseException as error:
            failure = error
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        runt_after = _runtime_summary(action["runtime_object"])
        new_events = _runtime_graphs_changed(runt_before, runt_after)
        linear_after = (
            None if linear_runtime is None else _runtime_summary(linear_runtime)
        )
        linear_new_events = (
            [] if linear_before is None else
            _runtime_graphs_changed(linear_before, linear_after)
        )
        try:
            _clear_runtime_attempts(runtimes, cache)
            _synchronize(cache, synchronize)
            _restore_cache(cache, initial_cache)
            if snapshot_aux_state is not None:
                restore_aux_state(_clone_buffer(initial_aux))
        except BaseException as restore_error:
            raise ProbeRestoreError("could not restore state after a measured action") from restore_error
        measurement_mode = mode
        row = {
            "action_id": action["action_id"],
            "phase": phase,
            "repeat_index": repeat_index,
            "k": action["k"],
            "backup_position": action.get("backup_position"),
            "recovery_token": action.get("recovery_token"),
            "runtime": action["runtime"],
            "commit_path": list(action["commit_path"]),
            "node_count": 4 if action["runtime"] == "tree" else 5,
            "wall_ms": elapsed_ms,
            "timing_mode": mode,
            "operation": operation,
            "runtime_state_before": runt_before,
            "runtime_state_after_operation": runt_after,
            "new_graph_events": new_events,
            "linear_runtime_state_before": linear_before,
            "linear_runtime_state_after_operation": linear_after,
            "linear_new_graph_events": linear_new_events,
            "error": None if failure is None else {
                "type": type(failure).__name__, "message": str(failure),
            },
        }
        if failure is not None:
            raise failure
        return row

    def action(action_id, runtime_name, path, *, backup=None, recovery=None,
               path_kind="accepted_path"):
        runtime_object = tree_runtime if runtime_name == "tree" else rescue_runtime
        return {
            "action_id": action_id,
            "runtime": runtime_name,
            "runtime_object": runtime_object,
            "k": 0 if runtime_name == "tree" else 1,
            "backup_position": backup,
            "recovery_token": recovery,
            "path_kind": path_kind,
            "commit_path": tuple(path),
        }

    def common_action_id(runtime_name, length, backup=None):
        kind = "eos_after_p1_rejection" if length == 1 else f"common_path_{length}"
        suffix = "" if backup is None else f"_backup_{backup}"
        return f"{runtime_name}_{kind}{suffix}"

    actions = []
    for length in accepted_path_lengths:
        kind = "eos_after_p1_rejection" if length == 1 else "accepted_path"
        actions.append(action(
            common_action_id("k0", length), "tree", tuple(range(length)),
            path_kind=kind,
        ))
    actions.append(action(
        "k0_miss_recover", "tree", (0,), recovery=miss_position
    ))
    for backup in backups:
        actions.append(action(
            f"k0_recover_to_backup_{backup}", "tree", (0,), recovery=backup
        ))
        for length in accepted_path_lengths:
            kind = "eos_after_p1_rejection" if length == 1 else "accepted_path"
            actions.append(action(
                common_action_id("k1", length, backup), "rescue",
                tuple(range(length)), backup=backup, path_kind=kind,
            ))
        actions.extend((
            action(f"k1_backup_hit_{backup}", "rescue", (0, 4), backup=backup,
                   path_kind="p1_rejected_backup_hit"),
            action(f"k1_miss_recover_{backup}", "rescue", (0,),
                   backup=backup, recovery=miss_position,
                   path_kind="p1_rejected_backup_miss"),
        ))
    setup_path_len = max(accepted_path_lengths)
    tree_full = next(
        item for item in actions
        if item["runtime"] == "tree" and len(item["commit_path"]) == setup_path_len
    )
    first_rescue_full = next(
        item for item in actions
        if item["runtime"] == "rescue" and len(item["commit_path"]) == setup_path_len
    )
    k0_recovery = next(
        item for item in actions if item["action_id"] == "k0_miss_recover"
    )
    setup_actions = [tree_full, first_rescue_full, k0_recovery]
    try:
        for setup_action in setup_actions:
            if setup_action is None:
                continue
            setup.append(run_one(setup_action, phase="prewarm"))
        for measured_action in actions:
            measured = [run_one(measured_action, phase="measured", repeat_index=index)
                        for index in range(repeats)]
            samples.append({
                "action_id": measured_action["action_id"],
                "k": measured_action["k"],
                "backup_position": measured_action.get("backup_position"),
                "recovery_token": measured_action.get("recovery_token"),
                "runtime": measured_action["runtime"],
                "commit_path": list(measured_action["commit_path"]),
                "node_count": 4 if measured_action["runtime"] == "tree" else 5,
                "repeat_wall_ms": [row["wall_ms"] for row in measured],
                "median_wall_ms": float(statistics.median(
                    row["wall_ms"] for row in measured
                )),
                "repeat_details": measured,
            })
    finally:
        restore_errors = []
        try:
            _synchronize(cache, synchronize)
            _restore_cache(cache, initial_cache)
        except BaseException as error:
            restore_errors.append(error)
        for snapshot in reversed(runtime_snapshots):
            try:
                _restore_runtime(snapshot)
            except BaseException as error:
                restore_errors.append(error)
        if snapshot_aux_state is not None:
            try:
                restore_aux_state(_clone_buffer(initial_aux))
            except BaseException as error:
                restore_errors.append(error)
        try:
            _synchronize(cache, synchronize)
        except BaseException as error:
            restore_errors.append(error)
        if restore_errors:
            raise ProbeRestoreError(
                f"failed to restore {len(restore_errors)} probe object(s)"
            ) from restore_errors[0]

    cache_restored = int(cache.get_pos()) == before
    runtime_after = _runtime_summaries(tree_runtime, rescue_runtime)
    if not cache_restored:
        raise ProbeRestoreError("cache position differs after probe completion")
    if runtime_after != runtime_before:
        raise ProbeRestoreError("runtime bookkeeping differs after probe completion")
    return {
        "schema": "same_state_action_probe_v1",
        "scope": (
            "Synchronized verify+commit and optional one-token Target recovery only; "
            "excludes Ranker, MLP, policy, sampling, prefill, and full decode."
        ),
        "rng_used": False,
        "cache_restored": cache_restored,
        "runtime_bookkeeping_restored": True,
        "cache_position_before": before,
        "cache_snapshot_extent": end,
        "input": {
            "pending_f": int(pending_f), "p1": int(p1),
            "feature": int(feature), "p2": int(p2),
            "backup_candidates": backups, "miss_position": miss_position,
            "forced_miss_token": bool(forced_miss_token),
            "accepted_path_lengths": list(accepted_path_lengths),
        },
        "measurement_mode": measurement_mode,
        "repeats": repeats,
        "runtime_state_before": runtime_before,
        "runtime_state_after_restore": runtime_after,
        "prewarm": setup,
        "actions": samples,
        "restoration_scope": (
            "K/V and token-cache prefix through max tree span, cache_seqlens, "
            "runtime maps/counters/pending state, graph input/output buffers, "
            "and optional caller-supplied auxiliary state."
        ),
        "graph_policy": "temporary probe graphs are removed after all samples",
        "accepted_commit_paths": {
            "k0_and_k1": {
                str(length): list(range(length))
                for length in accepted_path_lengths
            },
            "path_1_semantics": "EOS after P1 rejection; no Target recovery forward",
            "path_2_to_4_semantics": "ordinary accepted P/F prefixes",
            "p1_rejected_backup_hit": [0, 4],
            "p1_rejected_backup_miss": [0],
        },
        "cost_comparison_note": (
            "This helper measures the named forced paths at one cache state. "
            "Use recorded policy probabilities outside this helper to combine "
            "common-accept, EOS, backup-hit, and miss/recovery costs; it does "
            "not estimate an unobserved Feature-stage branch probability."
        ),
    }


def measure_action_operations(*, cache, tree_runtime, rescue_runtime,
                              pending_f: int, p1: int, features, p2s,
                              candidate_backups, miss_position: int,
                              recovery_forward: Callable,
                              forced_miss_token: bool = False,
                              repeats: int = 3,
                              max_candidates: int = 4,
                              accepted_path_lengths=(1, 2, 3, 4),
                              synchronize: Callable | None = None,
                              snapshot_aux_state: Callable | None = None,
                              restore_aux_state: Callable | None = None):
    """Adapter-shaped entry point for one live W1/rescue decode state.

    This is for a state with one candidate Feature and one P2. The caller
    supplies ``miss_position`` from that same round so k0 and k1 miss paths
    recover the identical Position. ``recovery_forward(cache, token)`` should
    match the ordinary Target entry used by the decoder, for example:

    .. code-block:: python

        def recovery_forward(cache, token):
            start = int(cache.get_pos())
            kinds = shared.typed_kinds(start, 1)
            return capture.run(target, cache, [int(token)], kinds, device)

    Snapshot and restore the capture hook's mutable output through the optional
    auxiliary callbacks if the real recovery closure updates that object.
    """
    features = tuple(int(value) for value in features)
    p2s = tuple(int(value) for value in p2s)
    if len(features) != 1 or len(p2s) != 1:
        raise ValueError("action probe requires exactly one Feature and one P2")
    return probe_action_costs(
        cache=cache,
        tree_runtime=tree_runtime,
        rescue_runtime=rescue_runtime,
        pending_f=pending_f,
        p1=p1,
        feature=features[0],
        p2=p2s[0],
        backup_candidates=candidate_backups,
        miss_position=miss_position,
        recovery_forward=recovery_forward,
        forced_miss_token=forced_miss_token,
        repeats=repeats,
        max_candidates=max_candidates,
        accepted_path_lengths=accepted_path_lengths,
        synchronize=synchronize,
        snapshot_aux_state=snapshot_aux_state,
        restore_aux_state=restore_aux_state,
    )


def _clear_runtime_attempts(runtimes, cache):
    for runtime in runtimes:
        pending = getattr(runtime, "_pending", None)
        if isinstance(pending, dict) and pending.get("cache") is cache:
            runtime._pending = None
        if bool(getattr(runtime, "_active", False)):
            runtime._active = False
        if getattr(runtime, "_backup_positions", None) is not None:
            runtime._backup_positions = None

