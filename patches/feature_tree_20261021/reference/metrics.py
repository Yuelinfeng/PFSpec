"""Source-weighted summaries for the paired existing-method experiment."""

import math
import random


def quantile(values, p):
    values = sorted(values)
    point = (len(values) - 1) * p
    low = int(point)
    high = min(low + 1, len(values) - 1)
    return values[low] * (high - point) + values[high] * (point - low) if high != low else values[low]


def ratio_summary(rows, numerator, denominator, budget, full_only=False):
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], {})[row["arm"]] = row
    logs = {}
    count = 0
    for arms in groups.values():
        if numerator not in arms or denominator not in arms:
            continue
        up, down = arms[numerator], arms[denominator]
        if full_only and (up["generated_tokens"] != budget or down["generated_tokens"] != budget):
            continue
        if min(up["generated_tokens"], down["generated_tokens"]) <= 0:
            continue
        value = math.log(up["generated_tokens"] / up["outer_wall_seconds"]) - math.log(
            down["generated_tokens"] / down["outer_wall_seconds"])
        logs.setdefault(up["source"], []).append(value)
        count += 1
    means = {source: math.fsum(values) / len(values) for source, values in logs.items()}
    names = sorted(means)
    result = dict(pair_count=count, source_count=len(names), source_equal_geometric_mean=None,
                  source_bootstrap_95_ci=None, per_source={})
    if not names:
        return result
    result["source_equal_geometric_mean"] = math.exp(math.fsum(means.values()) / len(names))
    result["per_source"] = {key: math.exp(means[key]) for key in names}
    result["source_bootstrap_seed"] = 402101
    result["source_bootstrap_replicates"] = 10000
    if len(names) > 1:
        rng = random.Random(402101)
        samples = [math.exp(math.fsum(means[rng.choice(names)] for _ in names) / len(names))
                   for _ in range(10000)]
        result["source_bootstrap_95_ci"] = [quantile(samples, .025), quantile(samples, .975)]
    return result


def summarize(rows, methods, executions, prompts, budget):
    arms = {}
    for execution in executions:
        for method in methods:
            arm = f"{execution}:{method}"
            chosen = [row for row in rows if row["arm"] == arm]
            tokens = sum(row["generated_tokens"] for row in chosen)
            seconds = math.fsum(row["outer_wall_seconds"] for row in chosen)
            lengths = {}
            for row in chosen:
                key = str(row["generated_tokens"])
                lengths[key] = lengths.get(key, 0) + 1
            arms[arm] = dict(runs=len(chosen), generated_tokens=tokens, outer_wall_seconds=seconds,
                             pooled_tokens_per_second=tokens / seconds if seconds else None,
                             eos_count=sum(row["eos"] for row in chosen), output_lengths=lengths)
    contrasts = {}

    def add(name, up, down):
        if up not in arms or down not in arms:
            return
        contrasts[name] = dict(numerator=up, denominator=down,
                              all_output_lengths=ratio_summary(rows, up, down, budget),
                              common_full_budget=ratio_summary(rows, up, down, budget, True))

    for method in methods:
        add(f"execution_gain:{method}", f"optimized:{method}", f"original:{method}")
        if method != "target":
            for execution in executions:
                add(f"algorithm_gain:{execution}:{method}", f"{execution}:{method}",
                    f"{execution}:target")
                if method != "pf_mlp":
                    add(f"versus_pf:{execution}:{method}", f"{execution}:{method}",
                        f"{execution}:pf_mlp")
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], {})[row["arm"]] = row
    parity = {}
    if "original" in executions and "optimized" in executions:
        for method in methods:
            pairs = []
            for pair_id, group in groups.items():
                old, new = group.get(f"original:{method}"), group.get(f"optimized:{method}")
                if old is None or new is None:
                    continue
                pairs.append(dict(pair_id=pair_id, output_equal=old["output_tokens"] == new["output_tokens"],
                                  counters_equal=old["counters"] == new["counters"],
                                  output_lengths=[old["generated_tokens"], new["generated_tokens"]]))
            parity[method] = dict(pair_count=len(pairs),
                                  output_equal_count=sum(item["output_equal"] for item in pairs),
                                  counters_equal_count=sum(item["counters_equal"] for item in pairs), pairs=pairs)
    expected = len(methods) * len(executions) * len(prompts)
    return dict(status="complete" if len(rows) == expected else "partial", expected_runs=expected,
                completed_runs=len(rows), methods=methods, executions=executions, prompt_count=len(prompts),
                arms=arms, contrasts=contrasts, same_seed_parity=parity)
