"""Ranker+MLP decoding with a small exact Feature candidate tree.

The Target tree runtime supplies logits for all rows at once and compacts only
one accepted path. This decoder owns proposal, rejection, recovery, and EOS
logic; those Python decisions remain outside the captured Target forward.
"""
from __future__ import annotations

from collections import Counter
import time
from contextlib import nullcontext


class ProbeObserver:
    """Collect optional component timings and per-round records.

    ``mode='off'`` adds no per-component timer calls. ``mode='components'``
    records host enqueue time and CUDA event spans; CUDA is synchronized only
    at the requested prefill/round boundaries, never between components.
    """

    def __init__(self, torch, device, *, mode="off", sync_prefill=False,
                 sync_round=False):
        if mode not in {"off", "components"}:
            raise ValueError("observer mode must be 'off' or 'components'")
        self.torch = torch
        self.device = device
        self.mode = mode
        self.sync_prefill = bool(sync_prefill)
        self.sync_round = bool(sync_round)
        self._phases = {}
        self._phase_rows = []
        self._host_ms = Counter()
        self._events = []
        self._round_state = None
        self._synced = False

    def measure(self, name, fn, *args, **kwargs):
        if self.mode == "off":
            return fn(*args, **kwargs)
        started = time.perf_counter()
        pair = None
        if self.device.type == "cuda":
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            stream = self.torch.cuda.current_stream(self.device)
            start_event.record(stream)
            pair = (str(name), start_event, end_event)
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - started) * 1000.0
        self._host_ms[str(name)] += elapsed
        if pair is not None:
            pair[2].record(stream)
            self._events.append(pair)
        self._synced = False
        return result

    def begin_phase(self, name):
        self._phases[str(name)] = time.perf_counter()

    def end_phase(self, name, *, synchronize=False):
        name = str(name)
        if synchronize and self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self._synced = True
        started = self._phases.pop(name, None)
        if started is not None:
            self._phase_rows.append(dict(
                name=name,
                host_wall_ms=(time.perf_counter() - started) * 1000.0,
                synchronized=bool(synchronize),
            ))

    def begin_round(self, record):
        if self.mode == "components":
            self._round_state = dict(
                started=time.perf_counter(),
                host_before=dict(self._host_ms),
                event_start=len(self._events),
            )
        else:
            self._round_state = dict(started=time.perf_counter())
        record["round_wall_ms"] = None
        record["round_enqueue_ms"] = None

    def end_round(self, record, *, synchronize=False):
        state = self._round_state or dict(started=time.perf_counter())
        if synchronize and self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self._synced = True
        elapsed = (time.perf_counter() - state["started"]) * 1000.0
        record["round_enqueue_ms"] = elapsed
        record["round_wall_ms"] = elapsed if synchronize else None
        record["round_synchronized"] = bool(synchronize)
        if self.mode == "components":
            before = state.get("host_before", {})
            local_host = {
                name: value - before.get(name, 0.0)
                for name, value in self._host_ms.items()
                if value - before.get(name, 0.0) > 0.0
            }
            record["component_cpu_ms"] = dict(local_host)
            record["component_host_ms"] = dict(local_host)
            record["_event_range"] = [state.get("event_start", 0), len(self._events)]
        self._round_state = None

    def finalize(self, rounds):
        event_ms = Counter()
        for name, started, ended in self._events:
            event_ms[name] += float(started.elapsed_time(ended))
        if self.mode == "components":
            for row in rounds:
                bounds = row.pop("_event_range", [0, 0])
                first, last = bounds
                local = Counter()
                for name, started, ended in self._events[first:last]:
                    local[name] += float(started.elapsed_time(ended))
                row["component_device_ms"] = dict(local)
                row["component_cuda_event_ms"] = dict(local)
        return dict(
            mode=self.mode,
            synchronization=dict(prefill=self.sync_prefill, round=self.sync_round),
            phases=list(self._phase_rows),
            component_cpu_ms=dict(self._host_ms) if self.mode == "components" else {},
            component_host_ms=dict(self._host_ms) if self.mode == "components" else {},
            component_device_ms=dict(event_ms) if self.mode == "components" else {},
            component_cuda_event_ms=dict(event_ms) if self.mode == "components" else {},
        )

try:
    from .tree_sampling import (
        torch_acceptance_probability,
        torch_positive_residual,
        torch_proposals_without_replacement,
    )
except ImportError:
    from tree_sampling import (
        torch_acceptance_probability,
        torch_positive_residual,
        torch_proposals_without_replacement,
    )


def _decode_impl(
    target,
    adapter,
    ranker,
    ranks,
    prompt,
    budget,
    seed,
    *,
    tree_runtime,
    width=2,
    cache_factory=None,
    observer=None,
    max_rounds=None,
    force_k1_each_round=False,
):
    """Decode one sequence and return ``(stats, generated_suffix)``.

    ``tree_runtime.verify`` evaluates ``[pendingF, P1, F_a, P2_a, ...]`` and
    leaves the logical cache position unchanged. ``commit`` keeps the chosen
    root-to-node path. A single accepted Feature branch uses its own Target P2
    row and pre-finalnorm hidden state.
    """
    import torch

    from draft_a.pf_feature_unified import EOS, F_VOCAB, P_VOCAB, _shared
    from bench_gaussiangpt_pf_block2_cow import (
        CaptureLastBlock,
        decide,
        fork_position_state,
        residual_point,
        truncate,
    )
    from gpt_position_spec_decode_smoke import choose_position

    shared = _shared()
    require = shared._require
    ratio = shared._ratio
    width = int(width)
    budget = int(budget)
    max_rounds = None if max_rounds is None else int(max_rounds)
    if max_rounds is not None and max_rounds < 1:
        raise ValueError("max_rounds must be positive when supplied")
    if budget < 2 or budget % 2:
        raise ValueError("budget must be a positive even number of at least two")
    if width < 1 or width > int(getattr(tree_runtime, "max_width", width)):
        raise ValueError("width exceeds the tree runtime capacity")
    if width > F_VOCAB:
        raise ValueError("width exceeds the Feature vocabulary")

    prompt = [int(token) for token in prompt]
    require(bool(prompt), "Prompt must contain BOS")
    shared.validate_sequence(prompt, [], ranks)
    require((len(prompt) - 1) % 2 == 0, "Prompt must end at a complete pair")
    require(EOS not in prompt[1:], "Prompt must not contain terminal EOS")

    device = next(target.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    stats = {
        "target_calls": 0,
        "target_input_tokens": 0,
        "draft_calls": 0,
        "draft_input_tokens": 0,
        "ranker_calls": 0,
        "rounds": 0,
        "proposed": 0,
        "verified": 0,
        "accepted": 0,
        "accepted_lengths": [],
        "round_generated_lengths": [],
        "position_fallbacks": 0,
        "k1_steps": 0,
        "tree_verify_calls": 0,
        "candidate_tree_nodes": 0,
        "target_tree_nodes": 0,
        "feature_attempts": 0,
        "feature_branch_hits": 0,
        "feature_all_rejected": 0,
        "feature_branch_index_counts": {},
        "recovery_calls": 0,
        "recovery_target_input_tokens": 0,
    }
    for kind in shared.KINDS:
        for field in ("proposed", "verified", "accepted"):
            stats[kind + "_" + field] = 0

    def count(kind, field, amount=1):
        amount = int(amount)
        stats[field] += amount
        stats[kind + "_" + field] += amount

    def measured(name, fn, *args, **kwargs):
        if observer is None:
            return fn(*args, **kwargs)
        return observer.measure(name, fn, *args, **kwargs)

    def policy(logits):
        return measured("sampling_policy", shared.policy, logits)

    def draw(logits):
        return int(shared.draw(policy(logits), rng))

    current_round = None

    def finish_round(begin, accepted):
        stats["accepted_lengths"].append(int(accepted))
        stats["round_generated_lengths"].append(len(prefix) - begin)
        if current_round is not None:
            current_round.update(
                accepted_length=int(accepted),
                cache_pos_after=int(cache.get_pos()),
                generated_tokens=len(prefix) - begin,
                actual_tokens=[int(token) for token in prefix[begin:]],
                eos=bool(prefix[begin:] and int(prefix[-1]) == EOS),
                eos_stage=current_round.get("eos_stage"),
                target_calls=stats["target_calls"] - current_round["_target_calls_before"],
                target_input_tokens=stats["target_input_tokens"] - current_round["_target_input_before"],
                recovery_calls=stats["recovery_calls"] - current_round["_recovery_calls_before"],
                recovery_target_calls=stats["recovery_calls"] - current_round["_recovery_calls_before"],
                recovery_target_input_tokens=(stats["recovery_target_input_tokens"]
                                              - current_round["_recovery_input_before"]),
                ranker_calls=stats["ranker_calls"] - current_round["_ranker_calls_before"],
                tree_verify_calls=stats["tree_verify_calls"] - current_round["_tree_verify_before"],
                candidate_tree_nodes=stats["candidate_tree_nodes"] - current_round["_candidate_nodes_before"],
                target_tree_nodes=stats["target_tree_nodes"] - current_round["_target_tree_nodes_before"],
                tree_copy_bytes={
                    key: int(value - current_round["_tree_copy_bytes_before"].get(key, 0))
                    for key, value in dict(getattr(tree_runtime, "_copy_bytes", {})).items()
                    if value - current_round["_tree_copy_bytes_before"].get(key, 0)
                },
                draft_calls=int(getattr(adapter, "forward_calls", 0)) - current_round["_draft_calls_before"],
                draft_input_tokens=int(getattr(adapter, "input_tokens", 0)) - current_round["_draft_tokens_before"],
            )
            for key in tuple(current_round):
                if key.startswith("_"):
                    del current_round[key]
            if observer is not None:
                observer.end_round(current_round, synchronize=(max_rounds == 1))
            stats["probe"]["rounds"].append(dict(current_round))

    def validate_position(previous, token):
        require(0 <= int(token) < P_VOCAB, "Target emitted an invalid Position id")
        if previous is not None:
            require(int(ranks[token]) > int(ranks[previous]),
                    "Target emitted a non-increasing Position")

    stats["probe"] = dict(rounds=[], action_budget=int(budget), max_rounds=max_rounds,
                          force_k1_each_round=bool(force_k1_each_round))
    cache = None
    capture = None
    prefix = list(prompt)
    adapter._profile = None
    capacity = len(prompt) + budget + 6
    make_cache = cache_factory or shared.cache_for
    try:
        if observer is not None:
            observer.begin_phase("setup")
        cache = measured("cache_create_reset", make_cache, target, capacity)
        capture = measured("capture_hook_create", CaptureLastBlock, target)
        measured("adapter_prepare", adapter.prepare, prompt, capacity)
        if observer is not None:
            observer.end_phase("setup")

        def target_forward(tokens, *, recovery=False):
            tokens = [int(token) for token in tokens]
            if not tokens:
                raise ValueError("Target forward requires at least one token")
            start = int(cache.get_pos())
            kinds = shared.typed_kinds(start, len(tokens))
            stage = "target_recovery" if recovery else "target_forward"
            logits, states = measured(stage, capture.run, target, cache, tokens, kinds, device)
            if int(cache.get_pos()) != start + len(tokens):
                raise RuntimeError("ordinary Target call advanced the wrong number of tokens")
            stats["target_calls"] += 1
            stats["target_input_tokens"] += len(tokens)
            if recovery:
                stats["recovery_calls"] += 1
                stats["recovery_target_input_tokens"] += len(tokens)
            return logits, states

        def verify_tree(pending_f, p1, features, p2s):
            before = int(cache.get_pos())
            logits, states = measured("tree_verify", tree_runtime.verify,
                                      cache, pending_f, p1, features, p2s)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verify advanced the logical cache position")
            expected_nodes = 2 + 2 * len(features)
            if logits.ndim == 3:
                logits = logits[0]
            if states.ndim == 3:
                states = states[0]
            if logits.ndim != 2 or states.ndim != 2:
                raise RuntimeError("tree runtime must return rank-2 logits and hidden states")
            if logits.size(0) != expected_nodes or states.size(0) != expected_nodes:
                raise RuntimeError("tree runtime returned a different node count")
            stats["target_calls"] += 1
            stats["target_input_tokens"] += expected_nodes
            stats["tree_verify_calls"] += 1
            stats["candidate_tree_nodes"] += len(features)
            stats["target_tree_nodes"] += expected_nodes
            return logits, states

        def commit_path(before, path):
            after = measured("tree_commit", tree_runtime.commit, cache, path)
            expected = int(before) + len(path)
            if int(after) != expected or int(cache.get_pos()) != expected:
                raise RuntimeError("tree commit left the Target cache at an unexpected position")
            return after

        # A complete prompt leaves its final Feature pending. The Target cache
        # and h_prev stop at the preceding Position, matching the MLP adapter.
        online = None
        h_prev = None
        if observer is not None:
            observer.begin_phase("prefill")
        if len(prompt) > 1:
            _, states = target_forward(prompt[:-1])
            h_prev = states[-1].detach().clone()
            online = shared.position_state(prefix, ranks)
        if observer is not None:
            observer.end_phase("prefill", synchronize=(max_rounds == 1))

        # BOS-only prompts use the same Target P/F bootstrap as the reference.
        while (len(prefix) - len(prompt) < budget
               and (max_rounds is None or stats["rounds"] < max_rounds)):
            if online is None:
                logits, _ = target_forward(prefix[-1:])
                p0 = draw(logits[-1])
                prefix.append(p0)
                if p0 == EOS:
                    break
                validate_position(None, p0)
                logits, states = target_forward([p0])
                f0 = draw(logits[-1])
                require(P_VOCAB <= f0 < EOS, "Target emitted an invalid Feature id")
                prefix.append(f0)
                h_prev = states[-1].detach().clone()
                online = shared.position_state(prefix, ranks)
                continue

            if int(cache.get_pos()) != len(prefix) - 1:
                raise RuntimeError("Target cache does not end immediately before pending Feature")
            stats["rounds"] += 1
            begin = len(prefix)
            before = int(cache.get_pos())
            previous, pending_f = prefix[-2], prefix[-1]
            remaining = budget - (len(prefix) - len(prompt))
            current_round = dict(
                round_index=stats["rounds"] - 1,
                cache_pos_before=before,
                remaining_budget_before=remaining,
                pending_feature=int(pending_f),
                previous_position=int(previous),
                mode=None,
                proposed_p1=None,
                proposed_p2=None,
                candidate_features=[],
                accepted_p1=None,
                accepted_feature=None,
                accepted_p2=None,
                selected_feature_branch=None,
                commit_path=None,
                cache_truncate_to=None,
                eos_stage=None,
                _target_calls_before=stats["target_calls"],
                _target_input_before=stats["target_input_tokens"],
                _recovery_calls_before=stats["recovery_calls"],
                _recovery_input_before=stats["recovery_target_input_tokens"],
                _ranker_calls_before=stats["ranker_calls"],
                _draft_calls_before=int(getattr(adapter, "forward_calls", 0)),
                _draft_tokens_before=int(getattr(adapter, "input_tokens", 0)),
                _tree_verify_before=stats["tree_verify_calls"],
                _candidate_nodes_before=stats["candidate_tree_nodes"],
                _target_tree_nodes_before=stats["target_tree_nodes"],
                _tree_copy_bytes_before=dict(getattr(tree_runtime, "_copy_bytes", {})),
            )
            if observer is not None:
                observer.begin_round(current_round)

            stats["ranker_calls"] += 1
            first = measured("ranker_choose_position", choose_position,
                             ranker, online, previous)
            second = shadow = None
            if first is not None:
                current_round["proposed_p1"] = int(first[0])
                current_round["ranker_p1_candidate_count"] = int(first[1]) if len(first) > 1 else None
            can_tree = (first is not None and remaining >= 4
                        and not force_k1_each_round)
            if can_tree:
                shadow = fork_position_state(online)
                shadow.finalize(first[0])
                stats["ranker_calls"] += 1
                second = measured("ranker_choose_position", choose_position,
                                  ranker, shadow, first[0])
                can_tree = second is not None
                if second is not None:
                    current_round["proposed_p2"] = int(second[0])
                    current_round["ranker_p2_candidate_count"] = int(second[1]) if len(second) > 1 else None

            if not can_tree:
                current_round["mode"] = "k1" if first is not None else "ranker_fallback"
                current_round["node_count"] = 0
                current_round["tree_read_bucket"] = None
                stats["k1_steps"] += 1
                if first is None:
                    stats["position_fallbacks"] += 1
                    logits, _ = target_forward([pending_f])
                    actual_p1 = draw(logits[-1])
                    current_round["actual_p1"] = int(actual_p1)
                    prefix.append(actual_p1)
                    if actual_p1 == EOS:
                        current_round["eos_stage"] = "position_after_pending_feature"
                        online.drop_eos()
                        finish_round(begin, 0)
                        break
                    validate_position(previous, actual_p1)
                    logits, states = target_forward([actual_p1])
                    actual_f1 = draw(logits[-1])
                    require(P_VOCAB <= actual_f1 < EOS,
                            "Target emitted an invalid Feature id")
                    prefix.append(actual_f1)
                    current_round["actual_feature"] = int(actual_f1)
                    h_prev = states[-1].detach().clone()
                    online.finalize(actual_p1)
                    finish_round(begin, 0)
                    continue

                p1 = int(first[0])
                count("position", "proposed")
                logits, states = target_forward([pending_f, p1])
                p1_distribution = policy(logits[0])
                count("position", "verified")
                p1_accepted = bool(measured("position_acceptance_draw", decide,
                                            float(p1_distribution[p1].item()), rng, device))
                current_round["accepted_p1"] = p1_accepted
                if p1_accepted:
                    count("position", "accepted")
                    actual_p1 = p1
                    h_prev = states[1].detach().clone()
                    actual_f1 = draw(logits[1])
                    accepted = 1
                else:
                    actual_p1 = int(residual_point(p1_distribution, p1, rng))
                    current_round["actual_p1"] = actual_p1
                    current_round["cache_truncate_to"] = before + 1
                    measured("cache_truncate", truncate, cache, before + 1, before + 2)
                    if int(cache.get_pos()) != before + 1:
                        raise RuntimeError("K1 Position recovery left a bad cache length")
                    if actual_p1 == EOS:
                        current_round["eos_stage"] = "position_recovery"
                        prefix.append(actual_p1)
                        online.drop_eos()
                        finish_round(begin, 0)
                        break
                    validate_position(previous, actual_p1)
                    logits, states = target_forward([actual_p1], recovery=True)
                    actual_f1 = draw(logits[-1])
                    h_prev = states[-1].detach().clone()
                    accepted = 0
                require(P_VOCAB <= actual_f1 < EOS,
                        "Target emitted an invalid Feature id")
                current_round["actual_p1"] = int(actual_p1)
                current_round["actual_feature"] = int(actual_f1)
                prefix.extend([actual_p1, actual_f1])
                online.finalize(actual_p1)
                finish_round(begin, accepted)
                continue

            p1 = int(first[0])
            p2 = int(second[0])
            current_round["mode"] = f"tree_w{width}"
            current_round["tree_width"] = int(width)
            current_round["node_count"] = 2 + 2 * int(width)
            required_read = before + current_round["node_count"]
            current_round["tree_read_bucket"] = max(8, 1 << (required_read - 1).bit_length())
            count("position", "proposed", 2)
            count("feature", "proposed", width)
            q = measured("draft_proposal", adapter.propose, prefix, p1, h_prev, rng)
            require(torch.is_tensor(q) and q.dtype == torch.float32 and q.ndim == 1
                    and q.numel() == F_VOCAB,
                    "Feature proposal must be a full FP32 distribution of size 4096")
            q_mass = float(q.sum().item())
            require(abs(q_mass - 1.0) <= 1e-4,
                    "Feature proposal must be normalized after temperature/top-p")
            candidate_ids, candidate_q = measured(
                "feature_candidate_sampling", torch_proposals_without_replacement,
                q, width, rng,
            )
            features = [int(token) + P_VOCAB for token in candidate_ids]
            current_round["candidate_features"] = list(features)
            p2s = [p2] * width

            logits, states = verify_tree(pending_f, p1, features, p2s)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verification moved the cache before acceptance")

            count("position", "verified")
            p1_distribution = policy(logits[0])
            p1_accepted = bool(measured("position_acceptance_draw", decide,
                                        float(p1_distribution[p1].item()), rng, device))
            current_round["accepted_p1"] = p1_accepted
            if not p1_accepted:
                actual_p1 = int(residual_point(p1_distribution, p1, rng))
                current_round["actual_p1"] = actual_p1
                current_round["commit_path"] = [0]
                commit_path(before, [0])
                prefix.append(actual_p1)
                if actual_p1 == EOS:
                    current_round["eos_stage"] = "position_residual_p1"
                    online.drop_eos()
                    finish_round(begin, 0)
                    break
                validate_position(previous, actual_p1)
                online.finalize(actual_p1)
                logits_recovery, states_recovery = target_forward([actual_p1], recovery=True)
                actual_f1 = draw(logits_recovery[-1])
                require(P_VOCAB <= actual_f1 < EOS,
                        "Target emitted an invalid Feature id")
                prefix.append(actual_f1)
                h_prev = states_recovery[-1].detach().clone()
                finish_round(begin, 0)
                continue

            count("position", "accepted")
            online = shadow
            h_p1 = states[1].detach().clone()
            target_feature = policy(logits[1])[P_VOCAB:EOS].float().clone()
            residual = target_feature
            selected_branch = None
            for branch, (candidate_local, proposal_q) in enumerate(
                zip(candidate_ids, candidate_q)
            ):
                candidate_local = int(candidate_local)
                stats["feature_attempts"] += 1
                count("feature", "verified")
                alpha = torch_acceptance_probability(residual, proposal_q,
                                                     candidate_local)
                if bool(decide(alpha, rng, device)):
                    selected_branch = branch
                    count("feature", "accepted")
                    stats["feature_branch_hits"] += 1
                    branch_counts = stats["feature_branch_index_counts"]
                    key = str(branch)
                    branch_counts[key] = int(branch_counts.get(key, 0)) + 1
                    break
                residual = torch_positive_residual(residual, proposal_q)

            if selected_branch is None:
                current_round["accepted_feature"] = False
                current_round["commit_path"] = [0, 1]
                actual_f1 = int(shared.draw(residual, rng)) + P_VOCAB
                current_round["actual_feature"] = int(actual_f1)
                require(P_VOCAB <= actual_f1 < EOS,
                        "Feature residual emitted an invalid Feature id")
                commit_path(before, [0, 1])
                prefix.extend([p1, actual_f1])
                online.drop_eos()
                h_prev = h_p1
                stats["feature_all_rejected"] += 1
                finish_round(begin, 1)
                continue

            branch = int(selected_branch)
            current_round["accepted_feature"] = True
            current_round["selected_feature_branch"] = branch
            current_round["actual_feature"] = int(features[branch])
            feature_node = 2 + 2 * branch
            p2_node = feature_node + 1
            actual_f1 = features[branch]
            prefix.extend([p1, actual_f1])
            count("position", "verified")
            p2_distribution = policy(logits[feature_node])
            p2_accepted = bool(measured("position_acceptance_draw", decide,
                                        float(p2_distribution[p2].item()), rng, device))
            current_round["accepted_p2"] = p2_accepted
            if p2_accepted:
                count("position", "accepted")
                h_prev = states[p2_node].detach().clone()
                bonus_f2 = draw(logits[p2_node])
                require(P_VOCAB <= bonus_f2 < EOS,
                        "Target bonus emitted an invalid Feature id")
                current_round["commit_path"] = [0, 1, feature_node, p2_node]
                commit_path(before, [0, 1, feature_node, p2_node])
                prefix.extend([p2, bonus_f2])
                online.finalize(p2)
                finish_round(begin, 3)
                continue

            actual_p2 = int(residual_point(p2_distribution, p2, rng))
            current_round["actual_p2"] = actual_p2
            current_round["commit_path"] = [0, 1, feature_node]
            commit_path(before, [0, 1, feature_node])
            prefix.append(actual_p2)
            if actual_p2 == EOS:
                current_round["eos_stage"] = "position_residual_p2"
                online.drop_eos()
                finish_round(begin, 2)
                break
            validate_position(p1, actual_p2)
            online.finalize(actual_p2)
            logits_recovery, states_recovery = target_forward([actual_p2], recovery=True)
            actual_f2 = draw(logits_recovery[-1])
            require(P_VOCAB <= actual_f2 < EOS,
                    "Target recovery emitted an invalid Feature id")
            prefix.append(actual_f2)
            h_prev = states_recovery[-1].detach().clone()
            finish_round(begin, 2)

    finally:
        stats["draft_calls"] = int(getattr(adapter, "forward_calls", 0))
        stats["draft_input_tokens"] = int(getattr(adapter, "input_tokens", 0))
        try:
            adapter.close()
        finally:
            if capture is not None:
                capture.close()
            cache = None

    if device.type == "cuda":
        if observer is None or not observer._synced:
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    suffix = prefix[len(prompt):]
    shared.validate_sequence(prompt, suffix, ranks)
    require(len(suffix) <= budget, "Output exceeds the requested token budget")
    require(len(stats["accepted_lengths"]) == stats["rounds"]
            == len(stats["round_generated_lengths"]), "Round totals differ")
    for field in ("proposed", "verified", "accepted"):
        require(stats[field] == sum(stats[kind + "_" + field]
                                    for kind in shared.KINDS),
                "Typed counters differ")

    stats.update(
        generated_tokens=len(suffix),
        seconds=elapsed,
        tokens_per_second=ratio(len(suffix), elapsed),
        eos=bool(suffix and suffix[-1] == EOS),
        k=2,
        proposal_width=2 + width,
        tree_width=width,
        seed=int(seed),
        prompt_tokens=len(prompt) - 1,
        requested_tokens=budget,
        temperature=0.9,
        target_top_p=0.9,
        draft_top_p=0.9,
        variant=f"ranker_mlp_feature_tree_w{width}",
        mean_accepted_length=ratio(sum(stats["accepted_lengths"]), stats["rounds"]),
        mean_generated_per_round=ratio(sum(stats["round_generated_lengths"]),
                                       stats["rounds"]),
    )
    if observer is not None:
        timing = observer.finalize(stats["probe"]["rounds"])
        stats["probe"].update(timing)
        phases = {row["name"]: row for row in timing["phases"]}
        stats["probe"]["setup_ms"] = phases.get("setup", {}).get("host_wall_ms")
        stats["probe"]["prefill_ms"] = phases.get("prefill", {}).get("host_wall_ms")
        stats["probe"]["round_wall_definition"] = (
            "Excludes runtime constructor, cache/adapter setup and prefill; "
            "includes proposal, Target verification/recovery, acceptance and commit."
        )
    if hasattr(tree_runtime, "stats"):
        stats["tree_runtime_stats"] = tree_runtime.stats()
    return shared._add_rates(stats), suffix


def decode_probe(
    target,
    adapter,
    ranker,
    ranks,
    prompt,
    seed,
    *,
    action_id,
    action_budget,
    tree_runtime,
    cache_factory=None,
    runtime=None,
    max_rounds=None,
    diagnostics="off",
    force_k1_each_round=False,
):
    """Run one isolated K1/W1/W2 action with optional component diagnostics.

    K1 single-round timing uses ``action_budget=2`` so the existing decoder
    selects its K1 path. W1/W2 use budget 4 for one round. For a complete K1
    request, pass ``force_k1_each_round=True`` and a larger even budget.
    ``runtime`` is accepted for explicit pairing with ``tree_runtime``; the
    caller owns its attention context and Target cache factory.
    """
    import torch

    action_id = str(action_id).upper()
    if action_id not in {"K1", "W1", "W2"}:
        raise ValueError("action_id must be K1, W1, or W2")
    width = 1 if action_id in {"K1", "W1"} else 2
    budget = int(action_budget)
    if action_id == "K1" and int(max_rounds or 0) == 1 and budget != 2:
        raise ValueError("single-round K1 must use action_budget=2")
    if action_id in {"W1", "W2"} and int(max_rounds or 0) == 1 and budget != 4:
        raise ValueError("single-round W1/W2 must use action_budget=4")
    if runtime is not None and tree_runtime is not None:
        if getattr(tree_runtime, "linear_runtime", runtime) is not runtime:
            raise ValueError("tree_runtime and runtime must share one Target runtime")
    device = next(target.parameters()).device
    autocast = (torch.autocast(device_type=device.type, enabled=False)
                if device.type == "cuda" else nullcontext())
    observer = ProbeObserver(
        torch,
        device,
        mode=diagnostics,
        sync_prefill=(max_rounds == 1),
        sync_round=(max_rounds == 1),
    )
    with torch.inference_mode(), autocast:
        stats, suffix = _decode_impl(
            target, adapter, ranker, ranks, prompt, budget, seed,
            tree_runtime=tree_runtime,
            width=width,
            cache_factory=cache_factory,
            observer=observer,
            max_rounds=max_rounds,
            force_k1_each_round=force_k1_each_round,
        )
    stats["probe"].update(
        action_id=action_id,
        action_budget_tokens=budget,
        scope="single_round" if max_rounds == 1 else "full_request",
        force_k1_each_round=bool(force_k1_each_round),
    )
    return stats, suffix
