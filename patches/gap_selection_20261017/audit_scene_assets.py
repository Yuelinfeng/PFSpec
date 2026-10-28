"""Compare selected scene model references with the prepared asset library."""
import argparse
import json
from pathlib import Path


def audit(manifest_path: Path, assets_root: Path, splits: tuple[str, ...]) -> dict:
    manifest = json.loads(manifest_path.read_text())
    results = []
    for split in splits:
        for row in manifest[split]:
            source = Path(row['source_path'])
            data = json.loads(source.read_text())
            furniture = {
                item['uid']: str(item['jid'])
                for item in data.get('furniture', [])
                if item.get('valid') and item.get('uid') and item.get('jid')
            }
            active_refs = {
                child['ref']
                for room in data.get('scene', {}).get('room', [])
                for child in room.get('children', [])
                if child.get('ref') in furniture
            }
            model_ids = sorted({furniture[ref] for ref in active_refs})
            missing = [
                model_id for model_id in model_ids
                if not (assets_root / '3D-FUTURE-model' / model_id / 'raw_model.obj').is_file()
            ]
            results.append({
                'split': split,
                'scene_id': row['scene_id'],
                'active_instance_count': len(active_refs),
                'model_id_count': len(model_ids),
                'missing_model_ids': missing,
                'model_coverage_in_runtime_assets': (
                    (len(model_ids) - len(missing)) / len(model_ids) if model_ids else None
                ),
            })
    return {
        'manifest': str(manifest_path),
        'assets_root': str(assets_root.resolve()),
        'split_count': len(splits),
        'scene_count': len(results),
        'scenes': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['d1', 'confirmation_t'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.assets_root, tuple(args.splits))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
