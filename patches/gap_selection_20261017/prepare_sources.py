"""Prepare selected 3D-FRONT sources in an isolated experiment directory."""

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_MANIFEST = Path(
    "/root/autodl-tmp/gaussiangpt_gap_selection_20261017/G5_source_manifest.json"
)
DEFAULT_ASSETS = Path(
    "/root/autodl-tmp/pf_multiscene_pilot_20261015/shared_assets"
)


def ensure_link(link: Path, target: Path) -> None:
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected link target: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"Expected a link, found an existing path: {link}")
    link.symlink_to(target, target_is_directory=True)


def prepare(manifest_path: Path, split: str, run_root: Path, assets_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    rows = manifest[split]
    scene_ids = []
    prepared = []
    for row in rows:
        scene_id = row["scene_id"]
        source = Path(row["source_path"])
        if not source.is_file():
            raise FileNotFoundError(source)

        scene_root = run_root / "scenes" / scene_id
        front_dir = scene_root / "3D-FRONT"
        front_dir.mkdir(parents=True, exist_ok=True)
        destination = front_dir / f"{scene_id}.json"
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Scene source differs from prior copy: {destination}")
        else:
            shutil.copy2(source, destination)

        ensure_link(scene_root / "3D-FUTURE-model", assets_root / "3D-FUTURE-model")
        ensure_link(scene_root / "3D-FRONT-texture", assets_root / "3D-FRONT-texture")
        scene_ids.append(scene_id)
        prepared.append({
            "scene_id": scene_id,
            "group_id": row["group_id"],
            "stratum": row["stratum"],
            "source_path": str(source),
            "prepared_scene_json": str(destination),
            "room_count": row["room_count"],
            "active_furniture_instances": row["active_furniture_instances"],
            "model_coverage": row["model_coverage"],
        })

    split_path = run_root / f"{split}_splits.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"splits": {"development": {"scene_ids": scene_ids}}}, indent=2) + "\n")
    inventory = {
        "split": split,
        "run_root": str(run_root),
        "assets_root": str(assets_root.resolve()),
        "scene_count": len(prepared),
        "scene_ids": scene_ids,
        "scenes": prepared,
        "scene_id_manifest": str(split_path),
    }
    output = run_root / f"{split}_source_inventory.json"
    output.write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("d1", "confirmation_t"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    result = prepare(args.manifest, args.split, args.run_root, args.assets_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
