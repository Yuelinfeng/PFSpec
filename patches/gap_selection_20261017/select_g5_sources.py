"""Select D1 and T scene groups from the prior unused catalog entries."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[2] / 'docs/project/artifacts/gap_selection_20261017'
STAGE_F_SCENES = ARTIFACTS.parent / 'stage_f_full3d/analysis/test_scenes.csv'
INVENTORY = ARTIFACTS / 'source_inventory'
SEED = 20261018


def room_stratum(room_count):
    if room_count <= 3:
        return 'small'
    if room_count <= 6:
        return 'medium'
    return 'large'


def main():
    splits = json.loads((INVENTORY / 'scene_splits.json').read_text())['splits']
    prior_3d_scenes = {
        row['scene'] for row in csv.DictReader(STAGE_F_SCENES.open()) if row.get('scene')
    }
    assert len(prior_3d_scenes) == 30
    catalog_rows = list(csv.DictReader((INVENTORY / 'scene_catalog.csv').open()))
    groups = defaultdict(list)
    for row in catalog_rows:
        if (row['split'] == '' and row['readable'] == 'True'
                and row['scene_id'] not in prior_3d_scenes
                and row['has_living_room'] == 'True'
                and int(row['room_count']) >= 2
                and int(row['active_furniture_instances']) >= 10
                and float(row['model_coverage']) == 1.0):
            groups[row['group_id']].append(row)

    candidates = []
    for group_id, members in groups.items():
        representative = max(
            members,
            key=lambda row: (
                int(row['active_furniture_instances']),
                int(row['room_count']),
                row['scene_id'],
            ),
        )
        candidates.append({
            'scene_id': representative['scene_id'],
            'group_id': group_id,
            'room_count': int(representative['room_count']),
            'stratum': room_stratum(int(representative['room_count'])),
            'active_furniture_instances': int(representative['active_furniture_instances']),
            'model_coverage': float(representative['model_coverage']),
            'source_path': representative['source_path'],
        })

    rng = random.Random(SEED)
    candidates.sort(key=lambda row: row['group_id'])
    rng.shuffle(candidates)
    assert len(candidates) >= 40
    d1, confirmation = candidates[:10], candidates[10:40]
    d1_groups = {row['group_id'] for row in d1}
    confirmation_groups = {row['group_id'] for row in confirmation}
    assert len(d1_groups) == 10 and len(confirmation_groups) == 30
    assert not d1_groups & confirmation_groups

    prior_ids = {name: set(split['scene_ids']) for name, split in splits.items()}
    for row in d1 + confirmation:
        assert all(row['scene_id'] not in ids for ids in prior_ids.values())
        assert row['scene_id'] not in prior_3d_scenes

    output = {
        'selection_seed': SEED,
        'candidate_count': len(candidates),
        'selection_rule': 'random order over eligible catalog groups with no prior assigned split; one representative scene per group',
        'eligibility': 'readable scene, at least two rooms, at least ten active furniture instances, living room, complete model coverage',
        'prior_assigned_source_counts': {name: len(ids) for name, ids in prior_ids.items()},
        'prior_stage_e_f_source_count': len(prior_3d_scenes),
        'd1': d1,
        'confirmation_t': confirmation,
        'pretraining_source_overlap': 'not auditable because the available Target training manifest has no complete scene ID list',
    }
    path = ARTIFACTS / 'G5_source_manifest.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({
        'candidate_count': len(candidates),
        'd1': [row['scene_id'] for row in d1],
        'confirmation_t': [row['scene_id'] for row in confirmation],
        'd1_strata': {name: sum(row['stratum'] == name for row in d1)
                      for name in ('small', 'medium', 'large')},
        't_strata': {name: sum(row['stratum'] == name for row in confirmation)
                     for name in ('small', 'medium', 'large')},
    }, indent=2))


if __name__ == '__main__':
    main()
