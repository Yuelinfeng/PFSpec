"""Five-node Target tree with one alternate Position branch for rejection recovery."""

from __future__ import annotations

from pathlib import Path
import sys


def rescue_layout():
    """Describe [pendingF, P1, F1, P2, P1b] and each node's ancestors."""
    parents = (-1, 0, 1, 2, 0)
    depths = (0, 1, 2, 3, 1)
    kinds = ("position", "feature", "position", "feature", "feature")
    paths = []
    for node in range(len(parents)):
        path = []
        current = node
        while current >= 0:
            path.append(current)
            current = parents[current]
        paths.append(tuple(reversed(path)))
    return parents, depths, kinds, tuple(paths)


def rescue_key_mask(before: int, read_bucket: int, window_left: int = -1):
    """Build a CPU attention mask from the shared prefix and each node path."""
    parents, depths, _kinds, paths = rescue_layout()
    before = int(before)
    read_bucket = int(read_bucket)
    window_left = int(window_left)
    if before < 0 or read_bucket < before + len(parents):
        raise ValueError("read bucket must contain the prefix and all five nodes")
    rows = []
    for node, path in enumerate(paths):
        query_depth = depths[node]
        query_position = before + query_depth
        allowed = [
            index < before and
            (window_left < 0 or query_position - index <= window_left)
            for index in range(read_bucket)
        ]
        for ancestor in path:
            allowed[before + ancestor] = (
                window_left < 0 or query_depth - depths[ancestor] <= window_left
            )
        rows.append(tuple(allowed))
    return tuple(rows)


try:
    from tree_runtime import TargetTreeRuntime
except ModuleNotFoundError as error:
    if error.name != "tree_runtime":
        raise
    local_tree_runtime = (
        Path(__file__).resolve().parents[1]
        / "feature_tree_20261021"
    )
    if not (local_tree_runtime / "tree_runtime.py").is_file():
        raise
    sys.path.insert(0, str(local_tree_runtime))
    from tree_runtime import TargetTreeRuntime


class PositionRescueRuntime(TargetTreeRuntime):
    """Run a fixed five-node tree with a sibling backup Position.

    The caller supplies a distinct legal ``backup_position`` from the same
    ranker candidate set as P1. The node is evaluated on every eligible W1
    round; it is used only if exact residual sampling returns that Position.
    All graph and cache state is owned by this runtime instance.
    """

    NODE_COUNT = 5

    def __init__(self, target, linear_runtime, *, graph: bool = True):
        super().__init__(target, linear_runtime, graph=graph, max_width=1)
        self._backup_position = None

    def stats(self):
        values = super().stats()
        values["layout"] = "pendingF_P1_F1_P2_P1b"
        values["fixed_node_count"] = self.NODE_COUNT
        return values

    def _node_tokens(self, pending_f, p1, features, p2s):
        tokens, _parents, _depths, _kinds, _paths = super()._node_tokens(
            pending_f, p1, features, p2s
        )
        if len(features) != 1 or len(p2s) != 1:
            raise ValueError("Position rescue supports one Feature branch")
        backup = self._backup_position
        if backup is None:
            raise RuntimeError("backup_position is required for this runtime")
        p_vocab = int(self.target.sparse_position_vocab_size)
        backup = int(backup)
        if not 0 <= backup < p_vocab:
            raise ValueError("backup_position must be a Position token id")
        if backup == int(p1):
            raise ValueError("backup_position must differ from P1")
        parents, depths, kinds, paths = rescue_layout()
        return tokens + [backup], parents, depths, kinds, paths

    def _position_masks(self, width, before, read_bucket, device):
        import torch

        if int(width) != 1:
            raise ValueError("Position rescue has a fixed single Feature branch")
        windows = tuple(getattr(self.target, "window_sizes", ()))
        if not windows:
            windows = ((-1, 0),) * len(self.target.transformer.h)
        if len(windows) != len(self.target.transformer.h):
            raise ValueError("Target window_sizes count differs from layer count")
        masks = {}
        for left in sorted(set(int(window[0]) for window in windows)):
            rows = rescue_key_mask(before, read_bucket, window_left=left)
            masks[left] = torch.tensor(rows, dtype=torch.bool, device=device).view(
                1, 1, self.NODE_COUNT, read_bucket
            )
        return masks

    def _make_attention(self, cache, before, width, read_bucket, masks_by_window):
        import torch

        if int(width) != 1:
            raise ValueError("Position rescue has a fixed single Feature branch")
        node_count = self.NODE_COUNT
        physical_offsets = torch.arange(
            node_count, dtype=torch.long, device=cache.k_cache.device
        )

        def explicit_tree_attention(q, k_cache, v_cache, k=None, v=None,
                                    cache_seqlens=None, causal=False,
                                    window_size=(-1, -1)):
            if k is None or v is None or not causal:
                raise RuntimeError("tree attention requires causal K/V inputs")
            batch, actual_nodes, q_heads, head_dim = q.shape
            kv_heads = int(k_cache.size(2))
            if (batch != 1 or actual_nodes != node_count or
                    q_heads % kv_heads != 0 or read_bucket > k_cache.size(1)):
                raise RuntimeError("Target dimensions differ from the rescue tree")
            window_left = int(window_size[0])
            if window_left not in masks_by_window:
                raise RuntimeError(f"no rescue key mask for window_left={window_left}")
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

        if int(width) != 1:
            raise ValueError("Position rescue has a fixed single Feature branch")
        target = self.target
        if not bool(getattr(target, "sparse_position_hard_constraints", False)):
            return
        if (not bool(target.rope_layout.is_sparse)
                or int(target.sparse_num_position_tokens) != 1):
            return
        p_vocab = int(target.sparse_position_vocab_size)
        rank_map = target.sparse_position_rank_map.to(logits.device)
        if rank_map.numel() < p_vocab:
            raise RuntimeError("Target rank map does not cover Position ids")
        previous_ids = torch.empty((2,), dtype=torch.long, device=logits.device)
        previous_ids[0] = cache.token_cache[0, int(before) - 1]
        previous_ids[1] = int(p1)
        valid = (previous_ids >= 0) & (previous_ids < p_vocab)
        safe_ids = previous_ids.clamp(min=0, max=p_vocab - 1)
        thresholds = rank_map[safe_ids]
        invalid = (
            rank_map[:p_vocab].unsqueeze(0) <= thresholds.unsqueeze(1)
        ) & valid.unsqueeze(1)
        logits[0:4:2, :p_vocab].masked_fill_(invalid, float("-inf"))

    def _build_graph(self, cache, before, width, read_bucket, tokens,
                     depths, paths, kinds):
        if len(tokens) != self.NODE_COUNT:
            raise RuntimeError("Position rescue graph must contain five nodes")
        state = super()._build_graph(
            cache, before, width, read_bucket, tokens, depths, paths, kinds
        )
        state["layout"] = "pendingF_P1_F1_P2_P1b"
        if self._graph_events:
            self._graph_events[-1]["layout"] = state["layout"]
            self._graph_events[-1]["node_count"] = self.NODE_COUNT
        return state

    def verify(self, cache, pending_f: int, p1: int,
               features: list[int], p2s: list[int], *, backup_position: int):
        """Evaluate one W1 branch plus P1's backup sibling without advancing pos."""
        if self._backup_position is not None:
            raise RuntimeError("nested or concurrent rescue verification is unsupported")
        self._backup_position = int(backup_position)
        try:
            result = super().verify(cache, pending_f, p1, features, p2s)
            self._calls["rescue_verify"] += 1
            return result
        finally:
            self._backup_position = None


def run_cpu_self_test():
    parents, depths, kinds, paths = rescue_layout()
    assert parents == (-1, 0, 1, 2, 0)
    assert depths == (0, 1, 2, 3, 1)
    assert kinds == ("position", "feature", "position", "feature", "feature")
    assert paths == ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 4))
    mask = rescue_key_mask(before=8, read_bucket=16, window_left=-1)
    assert len(mask) == 5 and all(len(row) == 16 for row in mask)
    assert mask[1][8] and mask[1][9] and not mask[1][12]
    assert mask[4][8] and mask[4][12] and not mask[4][9]
    finite = rescue_key_mask(before=8, read_bucket=16, window_left=2)
    assert not finite[4][5] and not finite[4][6]
    assert finite[4][7] and finite[4][8] and finite[4][12]
    try:
        rescue_key_mask(before=8, read_bucket=12)
    except ValueError:
        pass
    else:
        raise AssertionError("short read bucket was accepted")
    return dict(passed=True, parents=list(parents), depths=list(depths),
                kinds=list(kinds), paths=[list(path) for path in paths],
                all_mask_rows=len(mask), finite_window_left=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        import json
        print(json.dumps(run_cpu_self_test(), sort_keys=True))
    else:
        parser.error("Use --self-test for the CPU layout check")
