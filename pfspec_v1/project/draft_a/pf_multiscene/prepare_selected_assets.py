#!/usr/bin/env python3
"""Stage selected 3D-FRONT scenes and only their referenced assets."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path


def scene_assets(data: dict) -> tuple[set[str], set[str]]:
    furniture = {item.get("uid"): item for item in data.get("furniture", [])}
    model_ids = set()
    for room in data.get("scene", {}).get("room", []):
        for child in room.get("children", []):
            ref = child.get("ref") or child.get("instanceid") or child.get("uid")
            item = furniture.get(ref)
            if item and item.get("valid") is True and item.get("jid"):
                model_ids.add(item["jid"])
    texture_ids = {
        item["jid"]
        for item in data.get("material", [])
        if item.get("jid") and not item.get("useColor")
    }
    return model_ids, texture_ids


def replace_link(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace a non-link path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--models-zip", type=Path, required=True)
    parser.add_argument("--textures-zip", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--split-names", nargs="+", default=["development", "final_test"]
    )
    args = parser.parse_args()
    started = time.perf_counter()
    split_data = json.loads(args.splits.read_text())
    selected_ids = sorted(
        {
            scene_id
            for name in args.split_names
            for scene_id in split_data["splits"][name]["scene_ids"]
        }
    )

    scene_records = {}
    model_ids = set()
    texture_ids = set()
    for scene_id in selected_ids:
        source = args.scene_dir / f"{scene_id}.json"
        data = json.loads(source.read_text())
        models, textures = scene_assets(data)
        model_ids.update(models)
        texture_ids.update(textures)
        scene_records[scene_id] = {
            "source_json": str(source),
            "model_ids": sorted(models),
            "texture_ids": sorted(textures),
        }

    assets_root = args.run_root / "shared_assets"
    model_root = assets_root / "3D-FUTURE-model"
    texture_root = assets_root / "3D-FRONT-texture"
    model_root.mkdir(parents=True, exist_ok=True)
    texture_root.mkdir(parents=True, exist_ok=True)
    extracted_models = set()
    extracted_textures = set()
    usable_textures = set()
    model_bytes = 0
    texture_bytes = 0

    with zipfile.ZipFile(args.models_zip) as archive:
        members = archive.infolist()
        for info in members:
            parts = info.filename.split("/")
            if info.filename == "3D-FUTURE-model/model_info.json":
                archive.extract(info, assets_root)
            elif (
                len(parts) >= 3
                and parts[0] == "3D-FUTURE-model"
                and parts[1] in model_ids
            ):
                archive.extract(info, assets_root)
                extracted_models.add(parts[1])
                model_bytes += info.file_size

    with zipfile.ZipFile(args.textures_zip) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if (
                len(parts) >= 3
                and parts[0] == "3D-FRONT-texture"
                and parts[1] in texture_ids
            ):
                archive.extract(info, assets_root)
                extracted_textures.add(parts[1])
                texture_bytes += info.file_size
                if parts[-1] == "texture.png":
                    usable_textures.add(parts[1])

    missing_models = sorted(model_ids - extracted_models)
    if missing_models:
        raise RuntimeError(f"Missing model assets: {missing_models[:20]}")

    for scene_id, record in scene_records.items():
        scene_root = args.run_root / "scenes" / scene_id
        json_dir = scene_root / "3D-FRONT"
        per_scene_models = scene_root / "3D-FUTURE-model"
        per_scene_textures = scene_root / "3D-FRONT-texture"
        json_dir.mkdir(parents=True, exist_ok=True)
        per_scene_models.mkdir(parents=True, exist_ok=True)
        per_scene_textures.mkdir(parents=True, exist_ok=True)
        source = Path(record["source_json"])
        (json_dir / f"{scene_id}.json").write_bytes(source.read_bytes())
        replace_link(per_scene_models / "model_info.json", model_root / "model_info.json")
        for model_id in record["model_ids"]:
            replace_link(per_scene_models / model_id, model_root / model_id)
        for texture_id in record["texture_ids"]:
            target = texture_root / texture_id
            if target.is_dir():
                replace_link(per_scene_textures / texture_id, target)
        record["missing_texture_ids"] = sorted(
            set(record["texture_ids"]) - usable_textures
        )

    manifest = {
        "run_root": str(args.run_root),
        "splits": args.split_names,
        "selected_scene_count": len(selected_ids),
        "selected_scene_ids": selected_ids,
        "model_ids_requested": len(model_ids),
        "model_ids_extracted": len(extracted_models),
        "model_bytes_extracted": model_bytes,
        "texture_ids_requested": len(texture_ids),
        "texture_ids_extracted": len(extracted_textures),
        "texture_png_ids_available": len(usable_textures),
        "texture_bytes_extracted": texture_bytes,
        "missing_model_ids": missing_models,
        "missing_texture_ids": sorted(texture_ids - usable_textures),
        "scenes": scene_records,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    path = args.run_root / "stage_b" / "selected_asset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in manifest if k != "scenes"}, indent=2))
    print(f"Manifest: {path}")


if __name__ == "__main__":
    main()
