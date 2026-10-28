"""Prepare source groups unused by the recorded experiment scene folders."""
from pathlib import Path
import argparse
import concurrent.futures
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['select', 'assets_geometry', 'finish'], required=True)
    a = p.parse_args()
    root = a.root.resolve()
    project = Path('/root/workspace/GaussianGPT')
    prior = Path('/root/autodl-tmp/pf_multiscene_pilot_20261015')
    root.mkdir(parents=True, exist_ok=True)
    if a.phase == 'select':
        dest = root / 'source_manifest.json'
        if dest.exists(): raise FileExistsError(dest)
        catalog = json.loads((prior/'stage_b/scene_catalog.json').read_text())
        used = set()
        for path, dirs, files in os.walk('/root/autodl-tmp'):
            if len(Path(path).parts) > 8:
                dirs[:] = []
                continue
            if Path(path).name == 'scenes' and not str(path).startswith(str(root)):
                used.update(dirs)
                dirs[:] = []
        used.add('04396802-8d89-4377-8b28-4cadc9eb07c6')
        g5 = json.loads(Path('/root/autodl-tmp/gaussiangpt_gap_selection_20261017/G5_source_manifest.json').read_text())
        used.update(r['scene_id'] for k in ['d1','confirmation_t'] for r in g5[k])
        used_groups = {r['group_id'] for r in catalog if r['scene_id'] in used or r.get('split')}
        candidates = [r for r in catalog if r.get('readable') and r.get('has_living_room')
                      and r['room_count'] >= 2 and r['active_furniture_instances'] >= 10
                      and r['model_coverage'] == 1 and r['group_id'] not in used_groups]
        rng = random.Random(20261023)
        chosen = []
        seen = set()
        for label, count in [('small',1),('medium',5),('large',14)]:
            rows = sorted([r for r in candidates if r['stratum'] == label], key=lambda r:r['scene_id'])
            rng.shuffle(rows)
            group = []
            for row in rows:
                if row['group_id'] in seen: continue
                seen.add(row['group_id']); group.append(row)
                if len(group) == count: break
            if len(group) != count: raise ValueError('Insufficient eligible groups')
            chosen.extend(group)
        result = dict(selection_seed=20261023, test=chosen, prior_scene_ids=sorted(used),
                      prior_group_count=len(used_groups), candidate_count=len(candidates),
                      source_scope='New scene groups outside recorded prior scene folders and reserved splits. Target pretraining overlap unknown.')
        dest.write_text(json.dumps(result,indent=2))
        (root/'scene_splits.json').write_text(json.dumps({'splits':{'final_test':{'scene_ids':[r['scene_id'] for r in chosen]}}},indent=2))
        print(json.dumps({'selected':len(chosen),'excluded_groups':len(used_groups),'candidates':len(candidates)}),flush=True)
        return
    manifest = json.loads((root/'source_manifest.json').read_text())
    rows = manifest['test']
    if a.phase == 'assets_geometry':
        m = load('/root/autodl-tmp/gaussiangpt_gap_selection_20261017/materialize_source_assets.py','materialize')
        m.materialize(root/'source_manifest.json',('test',),prior/'shared_assets',
                      Path('/root/autodl-tmp/3D-FRONT-raw/3D-FUTURE-model.zip'),
                      Path('/root/autodl-tmp/3D-FRONT-raw/3D-FRONT-texture.zip'),root/'shared_assets')
        for row in rows:
            scene = root/'scenes'/row['scene_id']
            (scene/'3D-FRONT').mkdir(parents=True,exist_ok=True)
            shutil.copy2(row['source_path'],scene/'3D-FRONT'/f"{row['scene_id']}.json")
            for name in ['3D-FUTURE-model','3D-FRONT-texture']:
                dst=scene/name
                if not dst.exists(): dst.symlink_to(root/'shared_assets'/name,target_is_directory=True)

    def run_stage(row, stage, filename):
        scene_id=row['scene_id']; scene=root/'scenes'/scene_id
        work=root/'work'/scene_id; logs=root/'logs'/scene_id; scripts=root/'scripts'/scene_id
        for d in [work/'draft_a',logs,scripts]:d.mkdir(parents=True,exist_ok=True)
        (work/'draft_a/vfront_pilot_scene.txt').write_text(scene_id+'\n')
        record=logs/(stage+'.json')
        if record.exists() and json.loads(record.read_text()).get('exit_code') == 0:return
        source=project/'draft_a'/('pf_multiscene' if stage in ['depth','decode_render'] else '')/filename
        text=source.read_text().replace('/root/autodl-tmp/3D-FRONT-pilot',str(root/'scenes'))
        text=text.replace('"checkpoints/vqvae_vfront.ckpt"',f'"{project}/checkpoints/vqvae_vfront.ckpt"')
        if stage=='gaussian_fit':
            import numpy as np
            with np.load(scene/'gaussian_init/voxel_init_025m.npz') as data:n=len(data['xyz'])
            if n<=0:raise ValueError('Empty scene')
            text=text.replace('N = 30000',f'N = {min(n,30000)}')
        script=scripts/filename;script.write_text(text)
        command=[sys.executable,str(script)]
        if stage in ['depth','decode_render']:command+=['--scene-root',str(scene),'--scene-id',scene_id]
        if stage=='decode_render':command+=['--vqvae-checkpoint',str(project/'checkpoints/vqvae_vfront.ckpt')]
        env=os.environ.copy();env.update(PYTHONPATH=str(project),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        start=time.perf_counter()
        with (logs/(stage+'.log')).open('w') as f:
            ret=subprocess.run(command,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
        result=dict(scene_id=scene_id,stage=stage,seconds=time.perf_counter()-start,exit_code=ret.returncode)
        record.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        if ret.returncode:raise RuntimeError(f'{scene_id} {stage} failed')

    if a.phase=='assets_geometry':
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda r:run_stage(r,'geometry','export_vfront_pilot.py'),rows))
    else:
        stages=[('depth','render_depth_multiscene.py'),('rgb','render_vfront_rgb_cpu.py'),
                ('voxel','build_vfront_init_cloud.py'),('gaussian_fit','gsplat_vfront_multiview_export.py'),
                ('tokenize','tokenize_vfront_pilot.py'),('decode_render','decode_and_render.py')]
        for row in rows:
            for stage,filename in stages:run_stage(row,stage,filename)
        script=project/'draft_a/pf_multiscene/prepare_crop_records.py'
        subprocess.run([sys.executable,str(script),'--run-root',str(root),'--splits',str(root/'scene_splits.json'),
                        '--split','final_test','--target-crops','1','--max-seed','4',
                        '--output',str(root/'fresh_crops.pt')],cwd=project,check=True,
                       env={**os.environ, 'PYTHONPATH': str(project), 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    print('phase complete',a.phase,flush=True)


if __name__=='__main__':main()
