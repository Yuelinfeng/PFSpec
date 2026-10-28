#!/usr/bin/env python3
"""Analyze same-round Position rescue widths using matched source records."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ("target", "w1", "rescue1", "rescue2", "rescue3", "rescue4")
RESCUES = ARMS[2:]
BOOTSTRAP_COUNT = 10000
BOOTSTRAP_SEED = 104212
HIT_MAP_FIELDS = (
    "position_rescue_hits_by_rank",
    "position_rescue_hit_counts",
    "position_rescue_hits_by_node",
    "position_backup_hit_counts",
    "position_backup_hits_by_node",
    "rescue_hit_counts",
    "rescue_hits_by_node",
    "backup_hit_counts",
    "backup_hits_by_node",
    "backup_node_hits",
    "position_backup_node_hits",
    "position_rescue_hits",
)
ACCEPT_FIELDS = ("accepted", "position_accepted", "feature_accepted")


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _read_data(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(value)
        return path, rows
    value = json.loads(text)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and isinstance(value.get("records"), list):
        rows = value["records"]
    elif isinstance(value, dict):
        rows = [value]
    else:
        raise ValueError(f"Unsupported record structure in {path}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Every record in {path} must be an object")
    return path, rows


def _round_value(row, source_path, round_id):
    if round_id:
        return str(round_id)
    for key in ("round_id", "run_id", "experiment_id", "run_tag"):
        if row.get(key) is not None:
            return str(row[key])
    return source_path.parent.name


def _source(row):
    for key in ("source", "source_id", "scene"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    return "unknown"


def _parse_arm_label(value):
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in ("target", "target_only", "original_target", "original_target:target"):
        return "target"
    if label in ("w1", "width1", "width_1"):
        return "w1"
    if label == "rescue":
        return "rescue1"
    match = re.search(r"(?:^|[^a-z])rescue(?:[_-]?width)?[_-]?([1-4])(?:$|[^0-9])", label)
    if match:
        return "rescue" + match.group(1)
    return None


def _arm(row):
    labels = (row.get("arm"), row.get("verification_role"),
              row.get("action_id"), row.get("method"), row.get("candidate"))
    parsed_labels = [_parse_arm_label(label) for label in labels]
    for parsed in parsed_labels:
        if parsed and parsed.startswith("rescue") and parsed != "rescue1":
            return parsed
    has_rescue = any(str(label).strip().lower() == "rescue"
                     for label in labels if label is not None)
    if has_rescue:
        for key in ("rescue_width", "position_backup_width", "backup_width",
                    "num_backups", "position_backup_nodes"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                width = int(value)
                if 1 <= width <= 4:
                    return "rescue" + str(width)
    return next((parsed for parsed in parsed_labels if parsed), None)


def _budget(row):
    for value in (row.get("budget"), row.get("requested_tokens"),
                  (row.get("stats") or {}).get("requested_tokens")
                  if isinstance(row.get("stats"), dict) else None):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return 512


def _pair_id(row):
    value = row.get("pair_id")
    if value is None:
        value = row.get("context_id")
    if value is None:
        values = (_source(row), row.get("prefix"), row.get("seed"))
        value = "|".join(str(item) for item in values)
    return str(value)


def load_records(paths, *, round_id=None):
    loaded = []
    seen_paths = set()
    for input_path in paths:
        source_path, rows = _read_data(input_path)
        if source_path in seen_paths:
            continue
        seen_paths.add(source_path)
        for row in rows:
            arm = _arm(row)
            if arm is None:
                continue
            marked = dict(row)
            marked["_analysis_arm"] = arm
            marked["_analysis_round"] = _round_value(row, source_path, round_id)
            marked["_analysis_path"] = str(source_path)
            marked["_analysis_budget"] = _budget(row)
            marked["_analysis_source"] = _source(row)
            loaded.append(marked)
    rounds = sorted({row["_analysis_round"] for row in loaded})
    if len(rounds) > 1:
        raise ValueError(
            "Inputs contain multiple rounds; pass only one round or select one with --round-id"
        )
    return loaded, rounds[0] if rounds else (str(round_id) if round_id else None)


def _groups(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["_analysis_round"], _pair_id(row), row["_analysis_budget"])
        arm = row["_analysis_arm"]
        if arm in grouped[key]:
            raise ValueError(f"Duplicate {arm} record for pair {key[1]} budget {key[2]}")
        grouped[key][arm] = row
    complete = {}
    incomplete = []
    for key, arms in grouped.items():
        if all(arm in arms for arm in ARMS):
            sources = {arms[name]["_analysis_source"] for name in ARMS}
            prefixes = {str(arms[name].get("prefix")) for name in ARMS}
            seeds = {str(arms[name].get("seed")) for name in ARMS}
            if len(sources) != 1 or len(prefixes) != 1 or len(seeds) != 1:
                raise ValueError(f"Arm metadata differs within pair {key[1]}")
            complete[key] = arms
        else:
            incomplete.append(dict(
                round_id=key[0], pair_id=key[1], budget=key[2],
                source=next(iter(arms.values()))["_analysis_source"],
                present_arms=sorted(arms),
                missing_arms=[arm for arm in ARMS if arm not in arms],
            ))
    return grouped, complete, incomplete


def _generated(row):
    value = row.get("generated_tokens")
    if value is not None:
        return int(value)
    for key in ("output_tokens", "output", "tokens"):
        value = row.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _sequence(row):
    for key in ("output_tokens", "output", "tokens", "output_token_ids"):
        value = row.get(key)
        if isinstance(value, list):
            return [int(item) for item in value]
    return None


def _throughput(row):
    direct = _finite(row.get("tokens_per_second"))
    if direct is not None and direct >= 0.0:
        return direct
    count = _generated(row)
    seconds = _finite(row.get("outer_wall_seconds"))
    if seconds is None:
        seconds = _finite(row.get("elapsed_seconds"))
    if seconds is None and isinstance(row.get("stats"), dict):
        seconds = _finite(row["stats"].get("seconds"))
    if count is None or seconds is None or seconds <= 0.0:
        raise ValueError(
            f"Missing usable throughput fields for {_pair_id(row)} {row['_analysis_arm']}"
        )
    return count / seconds


def _has_new_graph(row):
    if bool(row.get("runtime_created")):
        return True
    if (_finite(row.get("graph_setup_ms")) or 0.0) > 0.0:
        return True
    if row.get("new_graph_events"):
        return True
    for key, value in row.items():
        if key.endswith("_delta") and isinstance(value, dict):
            if value.get("new_graph_events"):
                return True
    return False


def _subset_groups(complete):
    all_groups = list(complete.items())
    full = [(key, arms) for key, arms in all_groups
            if all(arms[name]["_analysis_budget"] == 512
                   and _generated(arms[name]) == 512 for name in ARMS)]
    warm = [(key, arms) for key, arms in all_groups
            if all(not _has_new_graph(arms[name]) for name in ARMS)]
    cold = [(key, arms) for key, arms in all_groups
            if any(_has_new_graph(arms[name]) for name in ARMS)]
    full_warm = [(key, arms) for key, arms in full
                 if all(not _has_new_graph(arms[name]) for name in ARMS)]
    full_cold = [(key, arms) for key, arms in full
                 if any(_has_new_graph(arms[name]) for name in ARMS)]
    return {
        "complete_six_arm": all_groups,
        "all_arms_full_512": full,
        "all_arms_no_new_graph": warm,
        "all_arms_with_new_graph": cold,
        "all_arms_full_512_no_new_graph": full_warm,
        "all_arms_full_512_with_new_graph": full_cold,
    }


def _ratio(left, right):
    numerator = _throughput(left)
    denominator = _throughput(right)
    if denominator == 0.0:
        return None
    return numerator / denominator


def _geometric_source_summary(group_rows, bootstrap_count, seed):
    by_source = defaultdict(list)
    undefined_pairs = []
    for key, source, value in group_rows:
        if value is None:
            undefined_pairs.append(key[1])
        else:
            by_source[source].append(math.log(value) if value > 0.0 else -math.inf)
    source_logs = {source: sum(values) / len(values)
                   for source, values in by_source.items() if values}
    all_sources = sorted(set(by_source) | {source for _key, source, _value in group_rows})
    if undefined_pairs or len(source_logs) != len(all_sources):
        source_ratios = {source: (math.exp(value) if math.isfinite(value)
                                  else 0.0 if value == -math.inf else None)
                         for source, value in source_logs.items()}
        source_directions = {}
        for source in all_sources:
            value = source_ratios.get(source)
            if value is None:
                source_directions[source] = "unavailable"
            elif value > 1.0:
                source_directions[source] = "faster"
            elif value < 1.0:
                source_directions[source] = "slower"
            else:
                source_directions[source] = "equal"
        direction_counts = Counter(source_directions.values())
        return dict(
            pair_count=len(group_rows), source_count=len(all_sources),
            ratio=None, ci95=None, bootstrap_samples=0,
            source_ratios=source_ratios,
            source_directions=source_directions,
            direction_counts={name: int(direction_counts.get(name, 0))
                              for name in ("faster", "equal", "slower", "unavailable")},
            undefined_pair_count=len(undefined_pairs),
            undefined_pair_ids=undefined_pairs,
        )
    source_ratios = {
        source: (math.exp(value) if math.isfinite(value)
                 else 0.0 if value == -math.inf else None)
        for source, value in source_logs.items()
    }
    point_log = sum(source_logs.values()) / len(source_logs) if source_logs else None
    point = math.exp(point_log) if point_log is not None and point_log != -math.inf else (
        0.0 if point_log == -math.inf else None
    )
    directions = Counter()
    source_directions = {}
    for source, value in source_ratios.items():
        if value is None:
            directions["unavailable"] += 1
            source_directions[source] = "unavailable"
        elif value > 1.0:
            directions["faster"] += 1
            source_directions[source] = "faster"
        elif value < 1.0:
            directions["slower"] += 1
            source_directions[source] = "slower"
        else:
            directions["equal"] += 1
            source_directions[source] = "equal"
    interval = None
    if source_logs and bootstrap_count > 0:
        rng = random.Random(int(seed))
        names = sorted(source_logs)
        samples = []
        for _ in range(int(bootstrap_count)):
            sampled_log = sum(source_logs[names[rng.randrange(len(names))]]
                              for _ in names) / len(names)
            samples.append(math.exp(sampled_log) if sampled_log != -math.inf else 0.0)
        samples.sort()
        lower = max(0, math.floor(0.025 * (len(samples) - 1)))
        upper = min(len(samples) - 1, math.ceil(0.975 * (len(samples) - 1)))
        interval = [samples[lower], samples[upper]]
    return dict(
        pair_count=len(group_rows), source_count=len(all_sources), ratio=point,
        ci95=interval, bootstrap_samples=int(bootstrap_count) if source_logs else 0,
        source_ratios=source_ratios,
        source_directions=source_directions,
        direction_counts={name: int(directions.get(name, 0))
                          for name in ("faster", "equal", "slower", "unavailable")},
        undefined_pair_count=0, undefined_pair_ids=[],
    )


def _comparison_specs():
    specs = [(arm, "target") for arm in ARMS[1:]]
    specs.extend((arm, "w1") for arm in RESCUES)
    specs.extend((arm, "rescue1") for arm in RESCUES[1:])
    specs.extend((f"rescue{width}", f"rescue{width - 1}")
                 for width in range(2, 5))
    return list(dict.fromkeys(specs))


def _summarize_comparisons(subsets, bootstrap_count, seed):
    result = {}
    for subset_name, items in subsets.items():
        comparisons = {}
        for left, right in _comparison_specs():
            observations = []
            for key, arms in items:
                observations.append((key, arms[left]["_analysis_source"],
                                     _ratio(arms[left], arms[right])))
            name = f"{left}_vs_{right}"
            comparisons[name] = _geometric_source_summary(
                observations, bootstrap_count, seed
            )
        result[subset_name] = comparisons
    return result


def _counter_value(row, name):
    for container_name in ("counters", "stats"):
        container = row.get(container_name)
        if isinstance(container, dict) and container.get(name) is not None:
            value = container[name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    value = row.get(name)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _rank_map(row, field_names):
    containers = []
    stats = row.get("stats")
    if isinstance(stats, dict):
        containers.append(("stats", stats))
    counters = row.get("counters")
    if isinstance(counters, dict):
        containers.append(("counters", counters))
    containers.append(("record", row))
    for container_name, container in containers:
        for field_name in field_names:
            value = container.get(field_name)
            if not isinstance(value, dict):
                continue
            parsed = {}
            valid = True
            for key, count in value.items():
                try:
                    rank = str(int(key))
                except (TypeError, ValueError):
                    valid = False
                    break
                parsed[rank] = _nested_count(count)
            if valid and all(count is not None and count >= 0
                             for count in parsed.values()):
                return parsed, f"{container_name}.{field_name}"
    return None, None


def _rounds(row):
    return _counter_value(row, "rounds")


def _requested_backup_count(row):
    for name in (
        "position_backup_requested_count",
        "requested_backup_count",
        "configured_backup_count",
        "backup_count",
        "num_backups",
    ):
        value = _counter_value(row, name)
        if value is not None:
            return value
    return None


def _nested_count(value):
    if isinstance(value, dict):
        for key in ("hits", "hit_count", "count", "value"):
            if key in value:
                return _nested_count(value[key])
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _hit_counts(row):
    containers = []
    for name in ("stats", "counters"):
        value = row.get(name)
        if isinstance(value, dict):
            containers.append(value)
    containers.append(row)
    for container in containers:
        for field in HIT_MAP_FIELDS:
            value = container.get(field)
            if isinstance(value, dict):
                parsed = {str(key): _nested_count(count) for key, count in value.items()}
                if parsed and all(count is not None and count >= 0
                                  for count in parsed.values()):
                    return parsed
        for field, value in container.items():
            lowered = str(field).lower()
            if (isinstance(value, dict) and "hit" in lowered
                    and ("rescue" in lowered or "backup" in lowered)
                    and "feature" not in lowered):
                parsed = {str(key): _nested_count(count) for key, count in value.items()}
                if parsed and all(count is not None and count >= 0
                                  for count in parsed.values()):
                    return parsed
    for container in containers:
        value = container.get("position_rescue_hits")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"backup_1": int(value)}
    return None


def _recoveries(row):
    return _counter_value(row, "recovery_calls")


def _sum_hits(hit_map):
    return None if hit_map is None else sum(hit_map.values())


def _ranked_counts(count_map):
    ranked = Counter()
    if count_map is None:
        return ranked
    for key, value in count_map.items():
        match = re.fullmatch(r"(?:backup|node|rank)?[_-]?(\d+)", str(key).lower())
        if match:
            ranked[str(int(match.group(1)))] += int(value)
        else:
            ranked[str(key)] += int(value)
    return ranked


def _pair_recovery_audit(items, left_arm, right_arm, *, incremental=False):
    paired = 0
    matching = 0
    missing = []
    mismatches = []
    saved_total = 0
    expected_total = 0
    for key, arms in items:
        left = arms[left_arm]
        right = arms[right_arm]
        left_recovery, right_recovery = _recoveries(left), _recoveries(right)
        left_hits, right_hits = _hit_counts(left), _hit_counts(right)
        if left_recovery is None or right_recovery is None or right_hits is None:
            missing.append(key[1])
            continue
        if incremental:
            if left_hits is None:
                missing.append(key[1])
                continue
            right_by_rank = _ranked_counts(right_hits)
            width = int(right_arm[-1])
            expected = right_by_rank.get(str(width), 0)
            expected_mode = f"new_rank_{width}_hits"
        else:
            expected = _sum_hits(right_hits)
            expected_mode = "all_backup_nodes"
        actual = left_recovery - right_recovery
        paired += 1
        saved_total += actual
        expected_total += expected
        if actual == expected:
            matching += 1
        else:
            mismatches.append(dict(
                pair_id=key[1], recovery_calls_saved=actual,
                hit_count_expected=expected, expected_from=expected_mode,
                left_arm=left_arm, right_arm=right_arm,
            ))
    return dict(
        left_arm=left_arm, right_arm=right_arm, paired_count=paired,
        equal_pair_count=matching, mismatch_count=len(mismatches),
        recovery_calls_saved=saved_total, hit_count_expected=expected_total,
        missing_counter_count=len(missing), missing_pair_ids=missing,
        mismatches=mismatches,
    )


def _recovery_summary(complete):
    items = list(complete.items())
    by_width = {}
    for arm in RESCUES:
        hit_totals = Counter()
        known_rows = 0
        missing_rows = 0
        recovery_total = 0
        recovery_rows = 0
        for _key, arms in items:
            row = arms[arm]
            hit_map = _hit_counts(row)
            if hit_map is None:
                missing_rows += 1
            else:
                known_rows += 1
                hit_totals.update(hit_map)
            value = _recoveries(row)
            if value is None:
                recovery_rows += 0
            else:
                recovery_total += value
                recovery_rows += 1
        by_width[arm] = dict(
            rows=len(items), hit_count_rows=known_rows,
            missing_hit_count_rows=missing_rows,
            hit_counts_by_rank=dict(sorted(_ranked_counts(hit_totals).items())),
            hit_counts_by_node=dict(sorted(hit_totals.items())),
            total_backup_hits=sum(hit_totals.values()),
            recovery_count_rows=recovery_rows,
            recovery_calls=recovery_total,
        )
    against_w1 = {
        arm: _pair_recovery_audit(items, "w1", arm) for arm in RESCUES
    }
    adjacent = {
        f"rescue{width}_vs_rescue{width - 1}": _pair_recovery_audit(
            items, f"rescue{width - 1}", f"rescue{width}", incremental=True
        )
        for width in range(2, 5)
    }
    return dict(
        by_width=by_width, versus_w1=against_w1, adjacent_widths=adjacent,
    )


def _backup_width_statistics(complete):
    items = list(complete.items())
    result = {}
    for arm in RESCUES:
        requested_count = int(arm[-1])
        requested_values = Counter()
        requested_missing = []
        requested_mismatch = []
        cumulative = Counter()
        exact = Counter({str(count): 0 for count in range(requested_count + 1)})
        extra_nodes = Counter()
        extra_sources = Counter()
        applicable_sources = Counter()
        extra_missing = []
        extra_mismatches = []
        invalid_applicable = []
        applicable_missing = []
        used_records = 0
        observed_rounds = 0
        hit_totals = Counter()
        hit_count_rows = 0
        hit_missing = []

        for key, arms in items:
            pair_id = key[1]
            row = arms[arm]
            configured = _requested_backup_count(row)
            if configured is None:
                requested_missing.append(pair_id)
            else:
                requested_values[str(configured)] += 1
                if configured != requested_count:
                    requested_mismatch.append(dict(
                        pair_id=pair_id, observed=int(configured),
                        expected=requested_count,
                    ))

            rounds = _rounds(row)
            applicable, applicable_source = _rank_map(
                row, ("position_backup_applicable_rounds_by_rank",)
            )
            if rounds is None or applicable is None:
                applicable_missing.append(pair_id)
            else:
                counts = [int(applicable.get(str(rank), 0))
                          for rank in range(1, requested_count + 1)]
                valid = (
                    rounds >= 0
                    and all(0 <= count <= rounds for count in counts)
                    and all(left >= right for left, right in zip(counts, counts[1:]))
                )
                if not valid:
                    invalid_applicable.append(dict(
                        pair_id=pair_id, rounds=int(rounds),
                        cumulative_by_rank={str(rank): count for rank, count in
                                            zip(range(1, requested_count + 1), counts)},
                    ))
                else:
                    used_records += 1
                    observed_rounds += int(rounds)
                    applicable_sources[applicable_source] += 1
                    for rank, count in enumerate(counts, 1):
                        cumulative[str(rank)] += count
                    exact["0"] += int(rounds) - counts[0] if counts else int(rounds)
                    for rank in range(1, requested_count):
                        exact[str(rank)] += counts[rank - 1] - counts[rank]
                    if counts:
                        exact[str(requested_count)] += counts[-1]

            extra, extra_source = _rank_map(
                row, (
                    "position_backup_extra_nodes_by_rank",
                    "position_backup_nodes_by_rank",
                    "nodes_by_rank",
                )
            )
            if extra is None:
                extra_missing.append(pair_id)
            else:
                extra_sources[extra_source] += 1
                for rank, value in extra.items():
                    extra_nodes[rank] += value
                if applicable is not None:
                    expected_extra = {
                        str(rank): int(applicable.get(str(rank), 0))
                        for rank in range(1, requested_count + 1)
                    }
                    observed_extra = {
                        str(rank): int(extra.get(str(rank), 0))
                        for rank in range(1, requested_count + 1)
                    }
                    if observed_extra != expected_extra:
                        extra_mismatches.append(dict(
                            pair_id=pair_id,
                            applicable_by_rank=expected_extra,
                            extra_nodes_by_rank=observed_extra,
                        ))

            hit_map = _hit_counts(row)
            if hit_map is None:
                hit_missing.append(pair_id)
            else:
                hit_count_rows += 1
                hit_totals.update(_ranked_counts(hit_map))

        actual_rounds = dict(sorted(exact.items(), key=lambda item: int(item[0])))
        reduced_rounds = sum(value for key, value in actual_rounds.items()
                             if int(key) < requested_count)
        result[arm] = dict(
            requested_count=requested_count,
            requested_count_values=dict(sorted(requested_values.items())),
            requested_count_fixed=(
                bool(items) and not requested_missing and not requested_mismatch
                and requested_values == Counter({str(requested_count): len(items)})
            ),
            requested_count_missing_rows=len(requested_missing),
            requested_count_missing_pair_ids=requested_missing,
            requested_count_mismatch_count=len(requested_mismatch),
            requested_count_mismatches=requested_mismatch,
            record_count=len(items),
            records_with_valid_applicable_stats=used_records,
            records_missing_applicable_stats=len(applicable_missing),
            records_with_invalid_applicable_stats=len(invalid_applicable),
            missing_applicable_pair_ids=applicable_missing,
            invalid_applicable_records=invalid_applicable,
            observed_rounds=observed_rounds,
            cumulative_applicable_rounds_by_rank={
                str(rank): cumulative.get(str(rank), 0)
                for rank in range(1, requested_count + 1)
            },
            actual_backup_rounds_by_count=actual_rounds,
            actual_backup_round_share_by_count={
                key: (value / observed_rounds if observed_rounds else None)
                for key, value in actual_rounds.items()
            },
            actual_less_than_requested_rounds=reduced_rounds,
            actual_full_requested_rounds=actual_rounds[str(requested_count)],
            rounds_with_any_backup=observed_rounds - actual_rounds["0"],
            maximum_observed_actual_count=max(
                (int(key) for key, value in actual_rounds.items() if value),
                default=0,
            ),
            applicable_field_usage=dict(sorted(applicable_sources.items())),
            extra_nodes_by_rank={
                str(rank): extra_nodes.get(str(rank), 0)
                for rank in range(1, requested_count + 1)
            },
            extra_nodes_total=sum(extra_nodes.values()),
            extra_node_field_usage=dict(sorted(extra_sources.items())),
            records_missing_extra_nodes=len(extra_missing),
            missing_extra_node_pair_ids=extra_missing,
            extra_nodes_matching_applicable=(
                bool(items) and not extra_missing and not extra_mismatches
            ),
            extra_nodes_mismatch_count=len(extra_mismatches),
            extra_nodes_mismatches=extra_mismatches,
            hit_counts_by_rank=dict(sorted(hit_totals.items())),
            total_rescue_hits=sum(hit_totals.values()),
            hit_count_rows=hit_count_rows,
            hit_count_missing_rows=len(hit_missing),
            hit_count_missing_pair_ids=hit_missing,
        )
    return result


def _acceptance_audit(items, left_arm, right_arm):
    by_field = {field: dict(available=0, equal=0, different=0, different_pair_ids=[])
                for field in ACCEPT_FIELDS}
    for key, arms in items:
        for field in ACCEPT_FIELDS:
            left = _counter_value(arms[left_arm], field)
            right = _counter_value(arms[right_arm], field)
            if left is None or right is None:
                continue
            target = by_field[field]
            target["available"] += 1
            if left == right:
                target["equal"] += 1
            else:
                target["different"] += 1
                target["different_pair_ids"].append(key[1])
    return by_field


def _output_audit(items, left_arm, right_arm):
    exact = 0
    exact_known = 0
    unknown = 0
    same_length = 0
    length_known = 0
    different_length = 0
    same_eos = 0
    eos_known = 0
    different_eos = 0
    differences = []
    for key, arms in items:
        left, right = arms[left_arm], arms[right_arm]
        left_sequence, right_sequence = _sequence(left), _sequence(right)
        if left_sequence is not None and right_sequence is not None:
            exact_known += 1
            equal = left_sequence == right_sequence
            exact += int(equal)
            if not equal:
                first = next((index for index, (a, b) in enumerate(
                    zip(left_sequence, right_sequence)) if a != b),
                    min(len(left_sequence), len(right_sequence)))
                differences.append(dict(
                    pair_id=key[1], left_tokens=len(left_sequence),
                    right_tokens=len(right_sequence), first_difference_index=first,
                ))
        else:
            unknown += 1
        left_count, right_count = _generated(left), _generated(right)
        if left_count is not None and right_count is not None:
            length_known += 1
            same_length += int(left_count == right_count)
            different_length += int(left_count != right_count)
        if left.get("eos") is not None and right.get("eos") is not None:
            eos_known += 1
            same_eos += int(bool(left["eos"]) == bool(right["eos"]))
            different_eos += int(bool(left["eos"]) != bool(right["eos"]))
    return dict(
        left_arm=left_arm, right_arm=right_arm, paired_count=len(items),
        exact_sequence_known_count=exact_known, exact_sequence_equal_count=exact,
        exact_sequence_different_count=exact_known - exact,
        exact_sequence_unknown_count=unknown,
        all_exact_sequences_equal=(exact == len(items) and unknown == 0),
        length_known_count=length_known, length_equal_count=same_length,
        length_different_count=different_length,
        eos_known_count=eos_known, eos_equal_count=same_eos,
        eos_different_count=different_eos,
        differing_pairs=differences,
        acceptance_counts=_acceptance_audit(items, left_arm, right_arm),
    )


def _parity_summary(complete):
    items = list(complete.items())
    specs = [(arm, "w1") for arm in RESCUES]
    specs.extend((f"rescue{width}", f"rescue{width - 1}")
                 for width in range(2, 5))
    return {
        f"{left}_vs_{right}": _output_audit(items, left, right)
        for left, right in specs
    }


def _read_cost_rows(paths):
    rows = []
    for input_path in paths or []:
        _path, values = _read_data(input_path)
        for row in values:
            nested = None
            for key in ("per_node", "node_costs", "nodes"):
                if isinstance(row.get(key), dict):
                    nested = row[key]
                    break
            if nested is not None:
                for node, detail in nested.items():
                    value = detail if isinstance(detail, (int, float)) else None
                    unit = "marginal_verify_ms"
                    if isinstance(detail, dict):
                        value = next((detail[name] for name in (
                            "marginal_verify_ms", "verify_ms", "cost_ms",
                            "marginal_verify_seconds", "verify_seconds",
                        ) if detail.get(name) is not None), None)
                        unit = next((name for name in (
                            "marginal_verify_ms", "verify_ms", "cost_ms",
                            "marginal_verify_seconds", "verify_seconds",
                        ) if detail.get(name) is not None), unit)
                    if value is not None:
                        rows.append(dict(node=str(node), value=value, unit=unit))
                continue
            node = next((row[name] for name in (
                "node", "node_id", "backup_node", "backup_index", "position"
            ) if row.get(name) is not None), None)
            field = next((name for name in (
                "marginal_verify_ms", "verify_ms", "cost_ms", "elapsed_ms",
                "marginal_verify_seconds", "verify_seconds"
            ) if row.get(name) is not None), None)
            if node is not None and field is not None:
                rows.append(dict(node=str(node), value=row[field], unit=field))
    return rows


def _cost_summary(paths):
    rows = _read_cost_rows(paths)
    grouped = defaultdict(list)
    for row in rows:
        value = _finite(row["value"])
        if value is None or value < 0.0:
            raise ValueError(f"Invalid node cost measurement: {row}")
        if row["unit"].endswith("seconds"):
            value *= 1000.0
        grouped[row["node"]].append(value)
    return dict(
        input_paths=[str(Path(path).expanduser().resolve()) for path in paths or []],
        unit="milliseconds per node verification",
        node_costs={node: dict(
            sample_count=len(values), mean_ms=statistics.fmean(values),
            median_ms=statistics.median(values), values_ms=values,
        ) for node, values in sorted(grouped.items())},
        separate_from_decode_throughput=True,
    )


def analyze_rows(rows, *, bootstrap_count=BOOTSTRAP_COUNT, seed=BOOTSTRAP_SEED,
                 cost_paths=None):
    grouped, complete, incomplete = _groups(rows)
    subsets = _subset_groups(complete)
    comparisons = _summarize_comparisons(subsets, bootstrap_count, seed)
    return _json_value(dict(
        schema="position-rescue-width-analysis-v1",
        round_id=rows[0]["_analysis_round"] if rows else None,
        required_arms=list(ARMS),
        row_count=len(rows),
        arm_row_counts=dict(sorted(Counter(row["_analysis_arm"] for row in rows).items())),
        pair_group_count=len(grouped),
        complete_six_arm_pair_count=len(complete),
        incomplete_pair_count=len(incomplete),
        incomplete_pairs=incomplete,
        included_pair_ids=[key[1] for key in sorted(complete)],
        source_count=len({arms["target"]["_analysis_source"] for arms in complete.values()}),
        bootstrap=dict(samples=int(bootstrap_count), seed=int(seed),
                       resampling_unit="source", source_weight="equal"),
        subsets={name: len(values) for name, values in subsets.items()},
        comparisons=comparisons,
        output_and_acceptance=(_parity_summary(complete)),
        recovery_and_hits=_recovery_summary(complete),
        actual_backup_count_statistics=_backup_width_statistics(complete),
        node_verify_costs=_cost_summary(cost_paths),
        analysis_policy=dict(
            partial_groups_included=False,
            complete_groups_require_all_six_arms=True,
            pair_selection_uses_completion_only=True,
            negative_throughput_results_retained=True,
            full_512_requires_all_six_arms_to_generate_512=True,
            no_new_graph_and_new_graph_subsets_require_all_six_arms=True,
            actual_backup_counts_derived_from_cumulative_rank_differences=True,
            requested_backup_count_is_fixed_by_arm=True,
            actual_backup_count_may_be_less_than_requested=True,
            node_costs_combined_into_end_to_end_rate=False,
        ),
    ))


def _ratio_text(summary):
    value = summary.get("ratio")
    interval = summary.get("ci95")
    if value is None:
        return "n/a"
    if interval is None:
        return f"{value:.4f}"
    return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def _count_map_text(values):
    return ", ".join(f"{key}:{values[key]}" for key in sorted(
        values, key=lambda item: int(item) if str(item).isdigit() else str(item)
    )) or "none"


def render_markdown(report):
    bootstrap_samples = report["bootstrap"]["samples"]
    lines = [
        "# Position Rescue Width Analysis",
        "",
        f"Round: `{report.get('round_id') or 'unknown'}`; "
        f"complete six-arm pairs: {report['complete_six_arm_pair_count']}; "
        f"incomplete groups: {report['incomplete_pair_count']}; "
        f"sources: {report['source_count']}.",
        "",
        "Throughput ratios use source-equal geometric means. Brackets give the "
        f"95% interval from {bootstrap_samples:,} source resamples. Every complete six-arm pair "
        "is retained, including slower cases.",
        "",
        "| Comparison | All complete | Full 512 | No new graph | New graph | "
        "Source direction (all complete) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    all_comparisons = report["comparisons"]["complete_six_arm"]
    for left, right in _comparison_specs():
        name = f"{left}_vs_{right}"
        full = report["comparisons"]["all_arms_full_512"][name]
        warm = report["comparisons"]["all_arms_no_new_graph"][name]
        cold = report["comparisons"]["all_arms_with_new_graph"][name]
        direction = all_comparisons[name]["direction_counts"]
        direction_text = (f"{direction['faster']} faster / {direction['equal']} equal / "
                          f"{direction['slower']} slower")
        lines.append(
            f"| `{left}/{right}` | {_ratio_text(all_comparisons[name])} | "
            f"{_ratio_text(full)} | {_ratio_text(warm)} | {_ratio_text(cold)} | "
            f"{direction_text} |"
        )
    lines.extend(["", "## Output and Acceptance"])
    lines.extend(["", "| Comparison | Exact sequence equal | Sequence different | "
                  "Length different | EOS different | Acceptance fields with differences |",
                  "|---|---:|---:|---:|---:|---|"])
    for name, audit in report["output_and_acceptance"].items():
        changed = []
        for field, values in audit["acceptance_counts"].items():
            if values["different"]:
                changed.append(f"{field}: {values['different']}")
        equal_text = (f"{audit['exact_sequence_equal_count']}/"
                      f"{audit['exact_sequence_known_count']} known")
        lines.append(
            f"| `{name}` | {equal_text} | {audit['exact_sequence_different_count']} | "
            f"{audit['length_different_count']} | {audit['eos_different_count']} | "
            f"{', '.join(changed) if changed else 'none observed'} |"
        )
    lines.extend(["", "## Actual Backup Count Frequencies", ""])
    lines.append(
        "Requested count is the fixed capacity of each arm: `rescue1`=1, "
        "`rescue2`=2, `rescue3`=3, `rescue4`=4. Actual count is derived from "
        "cumulative rank presence; every `0..k` bin is retained. A lower bin "
        "records a round where fewer than the requested nodes were available, "
        "so a 6/7/8-node layout does not mean that many nodes appeared every round. "
        "Extra-node values are cumulative rank appearances, not a per-round promise."
    )
    lines.extend(["", "| Arm | Requested | Requested fixed | Valid records / total; valid rounds | "
                  "Actual count frequency | Below requested | Extra nodes by rank | "
                  "Rescue hits by rank |",
                  "|---|---:|:---:|---:|---|---:|---|---|"])
    backup_stats = report["actual_backup_count_statistics"]
    for arm in RESCUES:
        values = backup_stats[arm]
        actual_text = _count_map_text(values["actual_backup_rounds_by_count"])
        extra_text = _count_map_text(values["extra_nodes_by_rank"])
        hit_text = _count_map_text(values["hit_counts_by_rank"])
        lines.append(
            f"| `{arm}` | {values['requested_count']} | "
            f"{'yes' if values['requested_count_fixed'] else 'no'} | "
            f"{values['records_with_valid_applicable_stats']}/{values['record_count']}; "
            f"{values['observed_rounds']} | {actual_text} | "
            f"{values['actual_less_than_requested_rounds']} | {extra_text} | "
            f"{hit_text} |"
        )
    lines.extend(["", "## Recovery Calls and Node Hits", ""])
    lines.extend(["| Width | Recovery calls | Backup hits | Saved calls vs W1 | "
                  "Pairs with exact difference/hit match |", "|---|---:|---:|---:|---:|"])
    for arm in RESCUES:
        width = report["recovery_and_hits"]["by_width"][arm]
        audit = report["recovery_and_hits"]["versus_w1"][arm]
        lines.append(
            f"| `{arm}` | {width['recovery_calls']} | {width['total_backup_hits']} | "
            f"{audit['recovery_calls_saved']} | "
            f"{audit['equal_pair_count']}/{audit['paired_count']} |"
        )
    cost_nodes = report["node_verify_costs"]["node_costs"]
    lines.extend(["", "Node verification cost records are reported separately from "
                  "decode throughput.", ""])
    if cost_nodes:
        lines.extend(["| Node | Samples | Mean ms | Median ms |", "|---|---:|---:|---:|"])
        for node, values in cost_nodes.items():
            lines.append(
                f"| `{node}` | {values['sample_count']} | {values['mean_ms']:.6f} | "
                f"{values['median_ms']:.6f} |"
            )
    else:
        lines.append("No per-node verification cost records were supplied.")
    lines.extend(["", "Incomplete groups do not enter any throughput, parity, or "
                  "recovery comparison. Output differences are counted as observed; "
                  "no equality claim is made for divergent pairs.", ""])
    return "\n".join(lines)


def _save_outputs(report, json_path, markdown_path):
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown outputs must use different paths")
    for path in (json_path, markdown_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def _fixture_row(source, pair, arm, rate, *, rescue_hits=None,
                 recovery_calls=20, full=True, new_graph=False,
                 output_change=False, acceptance_change=False, stats=None):
    tokens = list(range(512 if full else 511))
    if output_change:
        tokens[17] = 9999
    counters = dict(position_accepted=10, feature_accepted=9,
                     accepted=19, recovery_calls=recovery_calls)
    if rescue_hits is not None:
        counters["position_rescue_hit_counts"] = rescue_hits
    if acceptance_change:
        counters["feature_accepted"] -= 1
    row = dict(
        pair_id=pair, source=source, prefix=64, seed=7001, budget=512,
        method=arm, generated_tokens=len(tokens), output_tokens=tokens,
        outer_wall_seconds=len(tokens) / rate,
        tokens_per_second=rate, eos=False, counters=counters,
        runtime_created=new_graph,
    )
    if stats is not None:
        row["stats"] = stats
    return row


def _fixture_rows():
    rows = []
    rates = {
        "scene-a": dict(target=100, w1=200, rescue1=300, rescue2=600,
                        rescue3=400, rescue4=300),
        "scene-b": dict(target=100, w1=800, rescue1=400, rescue2=200,
                        rescue3=100, rescue4=50),
    }
    hit_totals = {
        "scene-a": (1, 2, 3, 4),
        "scene-b": (2, 4, 5, 6),
    }
    for source, arms in rates.items():
        totals = hit_totals[source]
        previous_recovery = 20 if source == "scene-a" else 30
        for arm in ARMS:
            width = int(arm[-1]) if arm.startswith("rescue") else 0
            if arm == "target":
                recovery = previous_recovery
                hits = None
            elif arm == "w1":
                recovery = previous_recovery
                hits = None
            else:
                recovery = previous_recovery - totals[width - 1]
                hits = {f"backup_{index + 1}": 1 for index in range(width)}
                if source == "scene-b":
                    hits["backup_1"] = 2
                    if width > 1:
                        hits["backup_2"] = 2
                if width < len(totals):
                    hits[f"backup_{width}"] = totals[width - 1] - sum(
                        hits.values()
                    ) + hits.get(f"backup_{width}", 0)
            stats = None
            if arm.startswith("rescue"):
                rounds = 10 if source == "scene-a" else 8
                applicable_values = (6, 2, 0, 0) if source == "scene-a" else (4, 1, 0, 0)
                applicable = {
                    str(rank): applicable_values[rank - 1]
                    for rank in range(1, width + 1)
                }
                rank_hits = {
                    key.rsplit("_", 1)[-1]: value
                    for key, value in (hits or {}).items()
                }
                stats = dict(
                    rounds=rounds,
                    backup_count=width,
                    position_backup_applicable_rounds_by_rank=applicable,
                    position_rescue_hits_by_rank=rank_hits,
                )
                if source == "scene-a":
                    stats["position_backup_extra_nodes_by_rank"] = dict(applicable)
                else:
                    stats["position_backup_nodes_by_rank"] = dict(applicable)
            rows.append(_fixture_row(
                source, source + "-pair", arm, arms[arm], rescue_hits=hits,
                recovery_calls=recovery,
                new_graph=(source == "scene-b" and arm == "rescue4"),
                output_change=(source == "scene-b" and arm == "rescue3"),
                acceptance_change=(source == "scene-b" and arm == "rescue3"),
                stats=stats,
            ))
    for arm in ARMS[:-1]:
        rows.append(_fixture_row(
            "scene-c", "partial-pair", arm, 100.0,
            rescue_hits={"backup_1": 1} if arm.startswith("rescue") else None,
        ))
    for row in rows:
        row["_analysis_arm"] = _arm(row)
        row["_analysis_round"] = "fixture"
        row["_analysis_path"] = "fixture/records.jsonl"
        row["_analysis_budget"] = _budget(row)
        row["_analysis_source"] = _source(row)
    return rows


def run_self_test():
    rows = _fixture_rows()
    grouped, complete, incomplete = _groups(rows)
    assert len(grouped) == 3 and len(complete) == 2 and len(incomplete) == 1
    assert incomplete[0]["missing_arms"] == ["rescue4"]
    report = analyze_rows(rows, bootstrap_count=10000, seed=104212)
    ratio = report["comparisons"]["complete_six_arm"]["w1_vs_target"]["ratio"]
    assert math.isclose(ratio, 4.0, rel_tol=1e-12)
    assert report["comparisons"]["complete_six_arm"]["w1_vs_target"]["bootstrap_samples"] == 10000
    assert report["comparisons"]["all_arms_full_512"]["w1_vs_target"]["pair_count"] == 2
    assert report["comparisons"]["all_arms_no_new_graph"]["w1_vs_target"]["pair_count"] == 1
    assert report["comparisons"]["all_arms_with_new_graph"]["w1_vs_target"]["pair_count"] == 1
    rescue3 = report["output_and_acceptance"]["rescue3_vs_w1"]
    assert rescue3["exact_sequence_different_count"] == 1
    assert rescue3["acceptance_counts"]["feature_accepted"]["different"] == 1
    rescue4_directions = report["comparisons"]["complete_six_arm"]["rescue4_vs_target"]["direction_counts"]
    assert rescue4_directions["faster"] == 1 and rescue4_directions["slower"] == 1
    for audit in report["recovery_and_hits"]["versus_w1"].values():
        assert audit["paired_count"] == 2 and audit["mismatch_count"] == 0
    for audit in report["recovery_and_hits"]["adjacent_widths"].values():
        assert audit["paired_count"] == 2 and audit["mismatch_count"] == 0
    backup_stats = report["actual_backup_count_statistics"]
    assert all(values["requested_count_fixed"] for values in backup_stats.values())
    assert backup_stats["rescue4"]["actual_backup_rounds_by_count"] == {
        "0": 8, "1": 7, "2": 3, "3": 0, "4": 0,
    }
    assert backup_stats["rescue4"]["actual_less_than_requested_rounds"] == 18
    assert backup_stats["rescue4"]["extra_nodes_by_rank"] == {
        "1": 10, "2": 3, "3": 0, "4": 0,
    }
    assert backup_stats["rescue4"]["extra_nodes_matching_applicable"] is True
    assert backup_stats["rescue4"]["extra_node_field_usage"] == {
        "stats.position_backup_extra_nodes_by_rank": 1,
        "stats.position_backup_nodes_by_rank": 1,
    }
    assert backup_stats["rescue4"]["hit_counts_by_rank"]
    with tempfile.TemporaryDirectory(prefix="width-analysis-") as temporary:
        cost_path = Path(temporary) / "node_costs.json"
        cost_path.write_text(json.dumps({"per_node": {
            "backup_1": {"marginal_verify_ms": 0.25},
            "backup_2": {"marginal_verify_seconds": 0.00035},
        }}), encoding="utf-8")
        report = analyze_rows(rows, bootstrap_count=10000, seed=104212,
                              cost_paths=[cost_path])
        assert math.isclose(
            report["node_verify_costs"]["node_costs"]["backup_1"]["mean_ms"],
            0.25, rel_tol=1e-12,
        )
        assert math.isclose(
            report["node_verify_costs"]["node_costs"]["backup_2"]["mean_ms"],
            0.35, rel_tol=1e-12,
        )
        json_path = Path(temporary) / "analysis.json"
        markdown_path = Path(temporary) / "analysis.md"
        _save_outputs(report, json_path, markdown_path)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        assert loaded["complete_six_arm_pair_count"] == 2
        assert "rescue3_vs_w1" in markdown
        assert "No new graph" in markdown
    assert report["node_verify_costs"]["separate_from_decode_throughput"] is True
    return dict(status="passed", complete_pairs=2, incomplete_pairs=1,
                source_equal_geomean=ratio, bootstrap_samples=10000,
                output_difference_detected=True,
                recovery_audits="all exact")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path)
    parser.add_argument("--round-id", default=None,
                        help="Select or join records from one explicitly named run")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_COUNT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--cost-records", nargs="*", type=Path, default=[])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.self_test:
        if args.records or args.json_out or args.markdown_out or args.cost_records:
            parser.error("--self-test cannot be combined with input or output paths")
        print(json.dumps(run_self_test(), ensure_ascii=False, sort_keys=True))
        return
    if not args.records or not args.json_out or not args.markdown_out:
        parser.error("--records, --json-out, and --markdown-out are required")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    rows, round_name = load_records(args.records, round_id=args.round_id)
    if not rows:
        parser.error("No recognized target, w1, or rescue1..rescue4 rows were found")
    report = analyze_rows(rows, bootstrap_count=args.bootstrap, seed=args.seed,
                          cost_paths=args.cost_records)
    report["round_id"] = round_name
    _save_outputs(report, args.json_out, args.markdown_out)
    print(json.dumps(dict(
        status="saved", round_id=round_name,
        complete_six_arm_pairs=report["complete_six_arm_pair_count"],
        incomplete_pairs=report["incomplete_pair_count"],
        json=str(args.json_out.resolve()), markdown=str(args.markdown_out.resolve()),
    ), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
