"""Build a minimal asset tree for selected scenes without changing source data."""
import argparse
import json
import zipfile
from pathlib import Path


def scene_references(path: Path) -> tuple[set[str], set[str]]:
    data = json.loads(path.read_text())
    furniture = {
        item['uid']: str(item['jid'])
        for item in data.get('furniture', [])
        if item.get('valid') and item.get('uid') and item.get('jid')
    }
    meshes = {item['uid']: item for item in data.get('mesh', []) if item.get('uid')}
    materials = {item['uid']: item for item in data.get('material', []) if item.get('uid')}
    model_ids: set[str] = set()
    texture_ids: set[str] = set()
    for room in data.get('scene', {}).get('room', []):
        for child in room.get('children', []):
            ref = child.get('ref')
            if ref in furniture:
                model_ids.add(furniture[ref])
            elif ref in meshes:
                material_id = meshes[ref].get('material')
                material = materials.get(material_id, {})
                jid = material.get('jid')
                if jid and not material.get('useColor'):
                    texture_ids.add(str(jid))
    return model_ids, texture_ids


def link_available(source: Path, destination: Path) -> bool:
    if not source.is_dir():
        return False
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f'Unexpected link target: {destination}')
    elif destination.exists():
        raise FileExistsError(destination)
    else:
        destination.symlink_to(source, target_is_directory=True)
    return True


def extract_folders(archive_path: Path, prefix: str, ids: set[str], output_root: Path) -> set[str]:
    wanted = set(ids)
    if not wanted:
        return wanted
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if len(parts) < 3 or parts[0] != prefix or parts[1] not in wanted:
                continue
            if '..' in parts or Path(info.filename).is_absolute():
                raise RuntimeError(f'Unsafe archive path: {info.filename}')
            archive.extract(info, output_root)
    if prefix == '3D-FUTURE-model':
        return {model_id for model_id in ids
                if not (output_root / prefix / model_id / 'raw_model.obj').is_file()}
    return {texture_id for texture_id in ids
            if not any((output_root / prefix / texture_id / name).is_file()
                       for name in ('texture.png', 'texture.jpg'))}


def materialize(manifest_path: Path, splits: tuple[str, ...], source_assets: Path,
                model_zip: Path, texture_zip: Path, output_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    model_ids: set[str] = set()
    texture_ids: set[str] = set()
    scenes = []
    for split in splits:
        for row in manifest[split]:
            models, textures = scene_references(Path(row['source_path']))
            model_ids.update(models)
            texture_ids.update(textures)
            scenes.append({
                'split': split,
                'scene_id': row['scene_id'],
                'active_model_count': len(models),
                'layout_texture_count': len(textures),
            })

    model_root = output_root / '3D-FUTURE-model'
    texture_root = output_root / '3D-FRONT-texture'
    model_root.mkdir(parents=True, exist_ok=True)
    texture_root.mkdir(parents=True, exist_ok=True)
    metadata_link = model_root / 'model_info.json'
    metadata_source = source_assets / '3D-FUTURE-model' / 'model_info.json'
    if not metadata_link.exists():
        metadata_link.symlink_to(metadata_source)
    elif metadata_link.resolve() != metadata_source.resolve():
        raise RuntimeError(f'Unexpected metadata path: {metadata_link}')

    missing_models = set()
    for model_id in sorted(model_ids):
        source = source_assets / '3D-FUTURE-model' / model_id
        destination = model_root / model_id
        if not (source / 'raw_model.obj').is_file():
            missing_models.add(model_id)
        else:
            link_available(source, destination)
    unresolved_models = extract_folders(model_zip, '3D-FUTURE-model', missing_models, output_root)
    invalid_models = [model_id for model_id in model_ids
                      if not (model_root / model_id / 'raw_model.obj').is_file()]
    if invalid_models:
        raise FileNotFoundError(f'Missing model files after extraction: {invalid_models[:20]}')

    missing_textures = set()
    for texture_id in sorted(texture_ids):
        source = source_assets / '3D-FRONT-texture' / texture_id
        destination = texture_root / texture_id
        if not (source / 'texture.png').is_file():
            missing_textures.add(texture_id)
        else:
            link_available(source, destination)
    unresolved_textures = extract_folders(texture_zip, '3D-FRONT-texture', missing_textures, output_root)

    result = {
        'manifest': str(manifest_path),
        'splits': list(splits),
        'scene_count': len(scenes),
        'scenes': scenes,
        'source_assets': str(source_assets.resolve()),
        'model_archive': str(model_zip),
        'texture_archive': str(texture_zip),
        'output_root': str(output_root),
        'active_model_count': len(model_ids),
        'model_count_reused_by_link': len(model_ids) - len(missing_models),
        'model_count_extracted': len(missing_models) - len(unresolved_models),
        'layout_texture_count': len(texture_ids),
        'texture_count_extracted_or_linked': len(texture_ids) - len(unresolved_textures),
        'layout_texture_ids_without_png_or_jpg': sorted(unresolved_textures),
        'all_active_models_available': True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'asset_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', required=True)
    parser.add_argument('--source-assets', type=Path, required=True)
    parser.add_argument('--model-zip', type=Path, required=True)
    parser.add_argument('--texture-zip', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        args.manifest, tuple(args.splits), args.source_assets,
        args.model_zip, args.texture_zip, args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
