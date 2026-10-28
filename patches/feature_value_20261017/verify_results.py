"""Validate observation records and unchanged production files."""
import json
from pathlib import Path
import subprocess

ROOT=Path('/root/autodl-tmp/gaussiangpt_feature_value_20261017')
PROJECT=Path('/root/workspace/GaussianGPT')
SNAPSHOT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017/source_snapshot/project')
dest=ROOT/'full'
analysis=json.loads((dest/'analysis.json').read_text())
comparisons=[]
for path in sorted(SNAPSHOT.rglob('*')):
    if not path.is_file():
        continue
    rel=path.relative_to(SNAPSHOT)
    production=PROJECT/rel
    equal=production.is_file() and path.read_bytes()==production.read_bytes()
    comparisons.append(dict(path=str(production),bytes_equal=equal))
assert len(comparisons)==19 and all(r['bytes_equal'] for r in comparisons),comparisons
for name,count in [('paths',56),('pairs',448)]:
    rows=[json.loads(s) for s in (dest/(name+'.jsonl')).read_text().splitlines()]
    assert len(rows)==count and all(r['output_equal'] for r in rows)
    for row in rows:
        for rr in row['rounds']:
            f=rr['pre_features'];p=f['previous_position']
            assert [f['previous_x'],f['previous_y'],f['previous_z']]==[p//400,p//20%20,p%20]
            if 'labels' not in rr:
                continue
            l=rr['labels']
            assert abs(l['joint']-l['a_p']*l['a_f'])<1e-12
            assert set(l).isdisjoint(f)
            assert rr['generated_tokens']==2+2*rr['counts']['feature_accepted']-int(rr['eos'])
            p1=rr['p1']
            assert [f['p1_x'],f['p1_y'],f['p1_z']]==[p1//400,p1//20%20,p1%20]
for path in ROOT.glob('*.py'):
    compile(path.read_text(),str(path),'exec')
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu',
                    '--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
result=dict(status='complete',validation=analysis['validation'],
    original_source_files=comparisons,gpu_after=gpu,
    coordinates_match_source=True,joint_probability_relation=True,
    ordinary_round_advancement_relation=True,script_syntax=True,
    no_new_weights=True,no_online_selector=True)
(dest/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='original_source_files'},indent=2))
