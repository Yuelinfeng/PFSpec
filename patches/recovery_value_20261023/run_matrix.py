"""Execute the declared independent-source jobs sequentially on an idle GPU."""
from pathlib import Path
import argparse
import datetime
import json
import subprocess
import sys
import tarfile
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    a=p.parse_args()
    spec=json.loads(a.spec.read_text())
    root=Path(spec['remote_root'])
    source=root/'source'
    state=root/'matrix_progress.json'
    if state.exists():raise FileExistsError(state)
    started=time.perf_counter()
    progress={'status':'starting','completed_jobs':[], 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def write():state.write_text(json.dumps(progress,indent=2)+'\n')
    write()
    try:
        for job in spec['jobs']:
            apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
            if apps:raise RuntimeError('Another GPU compute process is active: '+apps)
            cmd=[sys.executable,str(source/'run_experiment.py'),'--phase',job['phase'],
                 '--crops',str(root/'fresh_sources/fresh_crops.pt'),'--out',str(root/job['name']),
                 '--model',str(root/'selection_model.json'),'--methods',','.join(spec['methods']),
                 '--prefixes',','.join(map(str,spec['prefixes'])),'--sources',str(spec['source_count']),
                 '--seeds',str(job['seeds']),'--seed-start',str(spec['seed_start']),
                 '--budget',str(spec['budget']),'--repeat-index',str(job['repeat_index']),
                 '--max-wall-seconds',str(job['wall_limit'])]
            if job['phase']=='e2e':cmd+=['--warm-tokens',str(root/'warm_tokens.pt')]
            progress.update(status='running',current_job=job['name'],command=cmd);write()
            with (root/(job['name']+'.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=job['wall_limit']+60)
            subprocess.run([sys.executable,str(source/'analyze_results.py'),str(root/job['name']),
                            '--out',str(root/(job['name']+'_analysis.json')),'--best-fixed',spec['best_fixed']],check=True)
            progress['completed_jobs'].append(job['name']);write()
        for phase in ['timing','e2e']:
            paths=[str(root/j['name']) for j in spec['jobs'] if j['phase']==phase]
            subprocess.run([sys.executable,str(source/'analyze_results.py'),*paths,
                            '--out',str(root/(phase+'_combined_analysis.json')),'--best-fixed',spec['best_fixed']],check=True)
        paths=[str(root/j['name']) for j in spec['jobs'] if j['phase']=='e2e']
        progress.update(status='reading_products');write()
        subprocess.run([sys.executable,str(source/'audit_products.py'),*paths,'--out',str(root/'products_audit.json')],check=True,timeout=600)
        progress.update(status='complete',wall_seconds=time.perf_counter()-started);write()
    except Exception as error:
        progress.update(status='failed',error=repr(error),wall_seconds=time.perf_counter()-started);write()
        raise
    finally:
        with tarfile.open(root/'results_bundle.tgz','w:gz') as tar:
            for job in spec['jobs']:
                folder=root/job['name']
                if folder.exists():
                    for name in ['plan.json','prompts.json','progress.json','records.jsonl']:
                        f=folder/name
                        if f.exists():tar.add(f,arcname=str(f.relative_to(root)))
            for pattern in ['*_analysis.json','*audit.json','matrix_progress.json','evaluation_spec.json']:
                for f in root.glob(pattern):tar.add(f,arcname=f.name)


if __name__=='__main__':main()
