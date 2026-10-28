"""Align CPU coordinate names with utils.pos_tokens; sampling records stay equal."""
import gzip
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_value_20261017')
mapping = dict(x='id // 400',y='(id // 20) % 20',z='id % 20')
for phase in ['smoke','full']:
    dest = ROOT/phase
    receipt = dest/'coordinate_names.json'
    if receipt.exists():
        continue
    counts = {}
    for name in ['paths','pairs']:
        path = dest/(name+'.jsonl')
        data = path.read_bytes()
        with gzip.open(dest/(name+'_original_axis_names.jsonl.gz'),'wb') as stream:
            stream.write(data)
        rows = [json.loads(line) for line in data.splitlines()]
        count = 0
        for row in rows:
            for rr in row['rounds']:
                features = rr['pre_features']
                previous = features['previous_position']
                features.update(previous_x=previous // 400,previous_z=previous % 20)
                if 'p1' in rr:
                    p1 = rr['p1']
                    features.update(p1_x=p1 // 400,p1_z=p1 % 20,
                        p1_dx=p1 // 400-previous // 400,p1_dz=p1 % 20-previous % 20)
                count += 1
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        originals = [json.loads(line) for line in data.splitlines()]
        for old,new in zip(originals,rows):
            for old_r,new_r in zip(old['rounds'],new['rounds']):
                old_r.pop('pre_features');new_r.pop('pre_features')
            assert old == new
        counts[name] = count
    receipt.write_text(json.dumps(dict(mapping=mapping,rounds=counts,
        only_cpu_coordinate_names_changed=True,probabilities_and_outputs_unchanged=True),indent=2))
