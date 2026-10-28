"""Target tree with up to four alternate Position recovery branches."""

from __future__ import annotations

from pathlib import Path
import sys


def rescue_layout(backup_count=1):
    """Describe the W1 path and sibling Position nodes sharing pendingF."""
    original_count = backup_count
    backup_count = int(backup_count)
    if isinstance(original_count, bool) or backup_count != original_count or not 1 <= backup_count <= 4:
        raise ValueError("backup_count must be between one and four")
    parents = (-1, 0, 1, 2) + (0,) * backup_count
    depths = (0, 1, 2, 3) + (1,) * backup_count
    kinds = ("position", "feature", "position", "feature") + ("feature",) * backup_count
    paths = []
    for node in range(len(parents)):
        path = []
        current = node
        while current >= 0:
            path.append(current)
            current = parents[current]
        paths.append(tuple(reversed(path)))
    return parents, depths, kinds, tuple(paths)


def rescue_key_mask(before: int, read_bucket: int, window_left: int = -1,
                    backup_count: int = 1):
    """Build a CPU attention mask from the shared prefix and each node path."""
    parents, depths, _kinds, paths = rescue_layout(backup_count)
    before = int(before)
    read_bucket = int(read_bucket)
    window_left = int(window_left)
    if before < 0 or read_bucket < before + len(parents):
        raise ValueError("read bucket must contain the prefix and every tree node")
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
    """Run a W1 tree with one to four sibling backup Positions.

    The caller supplies a distinct legal ``backup_position`` from the same
    ranker candidate set as P1. The node is evaluated on every eligible W1
    round; it is used only if exact residual sampling returns that Position.
    All graph and cache state is owned by this runtime instance.
    """

    NODE_COUNT = 5

    def __init__(self, target, linear_runtime, *, graph: bool = True,
                 backup_count: int = 1):
        super().__init__(target, linear_runtime, graph=graph, max_width=1)
        rescue_layout(backup_count)
        self.backup_count = int(backup_count)
        self.NODE_COUNT = 4 + self.backup_count
        self._backup_positions = None
        self._graphs_by_count = {self.backup_count: self._graphs}

    def stats(self):
        values = super().stats()
        values["layout"] = "pendingF_P1_F1_P2_P1b" if self.NODE_COUNT == 5 else f"pendingF_P1_F1_P2_backups{self.NODE_COUNT - 4}"
        values["fixed_node_count"] = self.NODE_COUNT
        values["backup_count"] = self.backup_count
        values["graph_count"] = sum(len(group) for group in self._graphs_by_count.values())
        values["graph_counts_by_backup_count"] = {
            str(count): len(group) for count, group in self._graphs_by_count.items()
        }
        return values

    def _node_tokens(self, pending_f, p1, features, p2s):
        tokens, _parents, _depths, _kinds, _paths = super()._node_tokens(
            pending_f, p1, features, p2s
        )
        if len(features) != 1 or len(p2s) != 1:
            raise ValueError("Position rescue supports one Feature branch")
        backups = self._backup_positions
        if backups is None:
            raise RuntimeError("backup_positions are required for this runtime")
        p_vocab = int(self.target.sparse_position_vocab_size)
        if any(not 0 <= backup < p_vocab for backup in backups):
            raise ValueError("backup_positions must be Position token ids")
        if int(p1) in backups or len(set(backups)) != len(backups):
            raise ValueError("backup_positions must be distinct and differ from P1")
        parents, depths, kinds, paths = rescue_layout(len(backups))
        return tokens + list(backups), parents, depths, kinds, paths

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
            rows = rescue_key_mask(before, read_bucket, window_left=left,
                                   backup_count=self.NODE_COUNT - 4)
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
            raise RuntimeError("Position rescue graph has an unexpected node count")
        state = super()._build_graph(
            cache, before, width, read_bucket, tokens, depths, paths, kinds
        )
        state["layout"] = "pendingF_P1_F1_P2_P1b" if self.NODE_COUNT == 5 else f"pendingF_P1_F1_P2_backups{self.NODE_COUNT - 4}"
        if self._graph_events:
            self._graph_events[-1]["layout"] = state["layout"]
            self._graph_events[-1]["node_count"] = self.NODE_COUNT
        return state

    def verify(self, cache, pending_f: int, p1: int,
               features: list[int], p2s: list[int], *, backup_position=None,
               backup_positions=None):
        """Evaluate the W1 path and supplied sibling Positions without advancing pos."""
        if self._backup_positions is not None or self._active:
            raise RuntimeError("nested or concurrent rescue verification is unsupported")
        if self._pending is not None:
            raise RuntimeError("commit or clear the previous tree before verify")
        if backup_positions is not None and backup_position is not None:
            raise ValueError("supply only one backup argument")
        backups = (tuple(int(value) for value in backup_positions)
                   if backup_positions is not None else
                   (() if backup_position is None else (int(backup_position),)))
        if not 1 <= len(backups) <= self.backup_count:
            raise ValueError("supplied backups must fit the configured count")
        self.NODE_COUNT = 4 + len(backups)
        self._graphs = self._graphs_by_count.setdefault(len(backups), {})
        self._backup_positions = backups
        try:
            result = super().verify(cache, pending_f, p1, features, p2s)
            self._calls["rescue_verify"] += 1
            return result
        finally:
            self._backup_positions = None


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
    for count in range(1, 5):
        many_parents, many_depths, many_kinds, many_paths = rescue_layout(count)
        many_mask = rescue_key_mask(8, 16, backup_count=count)
        assert len(many_parents) == 4 + count
        for index in range(4, 4 + count):
            assert many_parents[index] == 0 and many_depths[index] == 1
            assert many_kinds[index] == "feature" and many_paths[index] == (0, index)
            assert [key for key, allowed in enumerate(many_mask[index]) if allowed] == list(range(9)) + [8 + index]
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
