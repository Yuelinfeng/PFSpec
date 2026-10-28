#!/usr/bin/env python3
"""Catalog 3D-FRONT scene metadata and assign disjoint source-group splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path


STAGE_A_SCENES = {
    "030345aa-f00e-4922-b50f-b56055702045",
    "003ac11d-2abc-44f8-9836-4354e7dfa543",
}
SEED = 20261015


def stratum(row):
    if row["room_count"] <= 3:
        return "small"
    if row["room_count"] <= 6:
        return "medium"
    return "large"


def active_scene_data(data):
    furniture = {item.get("uid"): item for item in data.get("furniture", [])}
    model_ids = set()
    instances = 0
    for room in data.get("scene", {}).get("room", []):
        for child in room.get("children", []):
            ref = child.get("ref") or child.get("instanceid") or child.get("uid")
            item = furniture.get(ref)
            if item and item.get("valid") is True and item.get("jid"):
                instances += 1
                model_ids.add(item["jid"])
    return instances, model_ids


def proportional_quota(total, counts):
    denom = sum(counts.values())
    if not denom:
        return {key: 0 for key in counts}
    exact = {key: total * value / denom for key, value in counts.items()}
    quota = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quota.values())
    order = sorted(counts, key=lambda key: (exact[key] - quota[key], key), reverse=True)
    for key in order[:remainder]:
        quota[key] += 1
    return quota


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--models-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import zipfile

    with zipfile.ZipFile(args.models_zip) as archive:
        available_models = {
            parts[1]
            for name in archive.namelist()
            if len(parts := name.split("/")) >= 3
            and parts[0] == "3D-FUTURE-model"
            and parts[1]
        }

    files = sorted(args.scene_dir.glob("*.json"), key=lambda path: path.stem)
    rows = []
    progress = args.out_dir / "scene_catalog_progress.json"
    for index, path in enumerate(files, 1):
        row = {
            "scene_id": path.stem,
            "source_path": str(path),
            "json_bytes": path.stat().st_size,
            "readable": False,
            "parse_error": None,
            "uid": None,
            "jobid": None,
            "group_id": path.stem,
            "room_count": 0,
            "room_types": [],
            "active_furniture_instances": 0,
            "active_model_ids": [],
            "missing_model_ids": [],
            "model_coverage": 0.0,
            "has_living_room": False,
            "split": None,
        }
        try:
            data = json.loads(path.read_text())
            row["readable"] = True
            row["uid"] = data.get("uid")
            row["jobid"] = data.get("jobid")
            row["group_id"] = str(data.get("jobid") or data.get("uid") or path.stem)
            rooms = data.get("scene", {}).get("room", [])
            row["room_count"] = len(rooms)
            row["room_types"] = sorted(room.get("type", "") for room in rooms)
            row["has_living_room"] = any(
                room.get("type") in {"LivingRoom", "LivingDiningRoom"}
                for room in rooms
            )
            instances, model_ids = active_scene_data(data)
            missing = sorted(model_ids - available_models)
            row["active_furniture_instances"] = instances
            row["active_model_ids"] = sorted(model_ids)
            row["missing_model_ids"] = missing
            row["model_coverage"] = (
                (len(model_ids) - len(missing)) / len(model_ids) if model_ids else 1.0
            )
        except Exception as exc:
            row["parse_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 250 == 0 or index == len(files):
            progress.write_text(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(files),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                        "last_scene_id": path.stem,
                    },
                    indent=2,
                )
            )

    eligible = [
        row
        for row in rows
        if row["readable"]
        and row["room_count"] >= 2
        and row["active_furniture_instances"] >= 10
        and row["has_living_room"]
    ]
    all_groups = defaultdict(list)
    eligible_groups = defaultdict(list)
    eligible_ids = set()
    for row in rows:
        if row["readable"]:
            all_groups[row["group_id"]].append(row)
    for row in eligible:
        eligible_groups[row["group_id"]].append(row)
        eligible_ids.add(row["scene_id"])
    groups = {
        group_id: all_groups[group_id]
        for group_id in eligible_groups
    }
    by_stratum = defaultdict(list)
    for group_id, members in groups.items():
        representative = max(
            eligible_groups[group_id],
            key=lambda row: (row["active_furniture_instances"], row["room_count"], row["scene_id"]),
        )
        by_stratum[stratum(representative)].append(group_id)

    rng = random.Random(SEED)
    for group_ids in by_stratum.values():
        group_ids.sort()
        rng.shuffle(group_ids)

    split_groups = {"development": [], "final_test": [], "adaptation_pool": []}
    used = set()

    def assign(split_name, group_ids):
        selected = []
        for group_id in group_ids:
            if group_id in used:
                continue
            used.add(group_id)
            split_groups[split_name].append(group_id)
            for row in groups[group_id]:
                row["split"] = split_name
                selected.append(row)
        return selected

    stage_a_groups = sorted(
        {row["group_id"] for row in eligible if row["scene_id"] in STAGE_A_SCENES}
    )
    development_rows = assign("development", stage_a_groups)
    dev_remaining = max(0, 10 - len(development_rows))
    available_dev = {
        bucket: [gid for gid in values if gid not in used]
        for bucket, values in by_stratum.items()
    }
    dev_counts = {
        bucket: sum(len(eligible_groups[gid]) for gid in gids)
        for bucket, gids in available_dev.items()
    }
    dev_quota = proportional_quota(dev_remaining, dev_counts)
    dev_pick = []
    for bucket in ("small", "medium", "large"):
        target = dev_quota.get(bucket, 0)
        count = 0
        for group_id in available_dev.get(bucket, []):
            if count >= target:
                break
            dev_pick.append(group_id)
            count += len(eligible_groups[group_id])
    development_rows.extend(assign("development", dev_pick))

    def choose_next(split_name, target_count):
        remaining = {
            bucket: [gid for gid in values if gid not in used]
            for bucket, values in by_stratum.items()
        }
        counts = {
            bucket: sum(len(eligible_groups[gid]) for gid in gids)
            for bucket, gids in remaining.items()
        }
        quota = proportional_quota(target_count, counts)
        selected = []
        for bucket in ("small", "medium", "large"):
            wanted = quota.get(bucket, 0)
            count = 0
            for group_id in remaining.get(bucket, []):
                if count >= wanted:
                    break
                selected.append(group_id)
                count += len(eligible_groups[group_id])
        current = assign(split_name, selected)
        while len(current) < target_count:
            next_group = next(
                (gid for bucket in ("small", "medium", "large") for gid in by_stratum.get(bucket, []) if gid not in used),
                None,
            )
            if next_group is None:
                break
            current.extend(assign(split_name, [next_group]))
        return current

    final_test_rows = choose_next("final_test", 30)
    adaptation_rows = choose_next("adaptation_pool", 20)

    for row in rows:
        row["stratum"] = stratum(row) if row["room_count"] else "unreadable"
        row["related_scene_ids"] = sorted(
            member["scene_id"] for member in all_groups.get(row["group_id"], [])
        )

    catalog_json = args.out_dir / "scene_catalog.json"
    catalog_json.write_text(json.dumps(rows, indent=2))
    fields = [
        "scene_id", "uid", "jobid", "group_id", "split", "stratum", "readable",
        "parse_error", "json_bytes", "room_count", "room_types", "has_living_room",
        "active_furniture_instances", "active_model_count", "active_model_ids", "missing_model_count",
        "model_coverage", "missing_model_ids", "related_scene_ids", "source_path",
    ]
    with (args.out_dir / "scene_catalog.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["room_types"] = ";".join(row["room_types"])
            output["active_model_count"] = len(row["active_model_ids"])
            output["active_model_ids"] = ";".join(row["active_model_ids"])
            output["missing_model_count"] = len(row["missing_model_ids"])
            output["missing_model_ids"] = ";".join(row["missing_model_ids"])
            output["related_scene_ids"] = ";".join(row["related_scene_ids"])
            writer.writerow({key: output.get(key) for key in fields})

    split_records = {}
    for split_name, split_rows in [
        ("development", development_rows),
        ("final_test", final_test_rows),
        ("adaptation_pool", adaptation_rows),
    ]:
        split_records[split_name] = {
            "scene_ids": sorted(row["scene_id"] for row in split_rows),
            "eligible_scene_ids": sorted(row["scene_id"] for row in split_rows if row["scene_id"] in eligible_ids),
            "group_ids": split_groups[split_name],
            "scene_count": len(split_rows),
            "eligible_scene_count": sum(row["scene_id"] in eligible_ids for row in split_rows),
            "group_count": len(split_groups[split_name]),
            "stratum_counts": dict(Counter(stratum(row) for row in split_rows)),
            "missing_model_scenes": [
                row["scene_id"] for row in split_rows if row["missing_model_ids"]
            ],
        }
    split_data = {
        "selection_seed": SEED,
        "group_key_precedence": ["jobid", "uid", "scene_id"],
        "eligibility": "readable JSON; at least 2 rooms; at least 10 active valid furniture instances; a living or living-dining room",
        "strata": {"small": "2-3 rooms", "medium": "4-6 rooms", "large": "7+ rooms"},
        "stage_a_scenes_reserved_for_development": sorted(STAGE_A_SCENES),
        "split_group_overlap_count": sum(
            1
            for group_id in split_groups["development"]
            if group_id in set(split_groups["final_test"] + split_groups["adaptation_pool"])
        ),
        "catalog_scene_count": len(rows),
        "readable_scene_count": sum(row["readable"] for row in rows),
        "eligible_scene_count": len(eligible),
        "eligible_group_count": len(groups),
        "scene_counts_by_room_bin": dict(Counter(stratum(row) for row in eligible)),
        "splits": split_records,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (args.out_dir / "scene_splits.json").write_text(json.dumps(split_data, indent=2))
    print(json.dumps(split_data, indent=2))


if __name__ == "__main__":
    main()
