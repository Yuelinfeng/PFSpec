"""Small action selectors using only live pre-verification context."""
import time
from probability_model import predict


def choose(metadata, context, method):
    main = int(context["main_position"])
    rows = context.get("candidates", ())
    candidates = [r for r in rows if int(r["token_id"]) != main]
    if not candidates:
        return []
    graph = context.get("graph_state", {}).get("1", {})
    if not graph.get("candidate_available", False):
        return []
    if method == "score_rule":
        first = candidates[0]
        return [int(first["token_id"])] if float(first.get("ranker_score") or 0) >= metadata["score_threshold"] else []
    model = metadata["probability"]
    predictions = [(predict(model, context, r), int(r["token_id"])) for r in candidates]
    main_row = next((r for r in rows if int(r["token_id"]) == main), None)
    pmain = predict(model, context, main_row) if main_row is not None else 0.0
    global_cost = metadata["global_cost"]
    bucket = str(graph.get("read_bucket", context.get("read_bucket", 0)))
    cost = metadata["costs_by_bucket"].get(bucket, global_cost)
    if method == "probability_rule":
        cost = global_cost
    if method == "acceptance_rule":
        residual = 1.0-pmain
        return [int(candidates[0]["token_id"])] if residual >= metadata["acceptance_threshold"] else []
    best = None
    for mass, token in predictions:
        mass = min(mass, max(0.0, 1.0-pmain))
        if method == "probability_rule":
            net = mass*cost["hit_saved_ms"] - (1.0-mass)*cost["common_extra_ms"]
        else:
            net = mass*cost["hit_saved_ms"] - pmain*cost["common_extra_ms"] - max(0.0, 1.0-pmain-mass)*cost["miss_extra_ms"]
        if best is None or net > best[0]:
            best = (net, token)
    if best is None or best[0] <= 0:
        return []
    if method == "value_ranker":
        return [int(candidates[0]["token_id"])]
    return [best[1]]


class Selector:
    def __init__(self, metadata, method):
        if method not in ("operation_value", "probability_rule", "score_rule", "value_ranker", "acceptance_rule"):
            raise ValueError(method)
        self.metadata = metadata
        self.method = method
        self.calls = 0
        self.nodes = 0
        self.seconds = 0.0

    def __call__(self, context):
        start = time.perf_counter()
        result = choose(self.metadata, context, self.method)
        self.seconds += time.perf_counter()-start
        self.calls += 1
        self.nodes += len(result)
        return result

    def stats(self):
        return dict(calls=self.calls, selected_backup_nodes=self.nodes, wall_seconds=self.seconds)
