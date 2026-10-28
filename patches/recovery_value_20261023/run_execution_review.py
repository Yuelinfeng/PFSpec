"""Run a finite, paired implementation review using preserved development data."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import time


ROOT = Path('/root/autodl-tmp/gaussiangpt_recovery_value_execution_20261023')
OLD = Path('/root/autodl-tmp/gaussiangpt_recovery_value_20261023')
METHODS = 'w1,fixed1,same0,same1,old_value,value_reference,value'


def main():
    state_file = ROOT / 'progress.json'
    if state_file.exists():
        raise FileExistsError(state_file)
    started = time.perf_counter()
    state = dict(status='starting', completed=[],
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

    def save():
        state['wall_seconds'] = time.perf_counter() - started
        state_file.write_text(json.dumps(state, indent=2) + '\n')

    def call(name, arguments, limit):
        state.update(status='running', current=name, command=arguments)
        save()
        with (ROOT / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, *arguments], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=limit)
        state['completed'].append(name)
        save()

    def run(phase, repeat, limit):
        name = f'{phase}_run0{repeat + 1}'
        args = [str(ROOT / 'source/run_experiment.py'), '--phase', phase,
                '--crops', '/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt',
                '--out', str(ROOT / name), '--model', str(OLD / 'selection_model.json'),
                '--methods', METHODS, '--prefixes', '64,256', '--sources', '3',
                '--seeds', '2' if phase == 'timing' else '1',
                '--seed-start', '58101', '--budget', '512', '--repeat-index', str(repeat),
                '--max-wall-seconds', str(limit),
                '--legacy-adaptive', str(OLD / 'source/adaptive_decoder.py')]
        if phase == 'e2e':
            args += ['--warm-tokens', str(OLD / 'warm_tokens.pt')]
        call(name, args, limit + 60)
        call(name + '_validation', [str(ROOT / 'source/validate_execution.py'),
             str(ROOT / name), '--out', str(ROOT / (name + '_validation.json'))], 60)

    try:
        save()
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                               '--format=csv,noheader'], capture_output=True,
                              text=True, check=True).stdout.strip()
        if apps:
            raise RuntimeError('An existing GPU compute process is active: ' + apps)
        for repeat in range(2):
            run('timing', repeat, 900)
        call('timing_combined_validation', [str(ROOT / 'source/validate_execution.py'),
             str(ROOT / 'timing_run01'), str(ROOT / 'timing_run02'),
             '--out', str(ROOT / 'timing_combined_validation.json')], 60)
        for repeat in range(2):
            run('e2e', repeat, 600)
        call('e2e_combined_validation', [str(ROOT / 'source/validate_execution.py'),
             str(ROOT / 'e2e_run01'), str(ROOT / 'e2e_run02'),
             '--out', str(ROOT / 'e2e_combined_validation.json')], 60)
        call('products_audit', [str(ROOT / 'source/audit_products.py'),
             str(ROOT / 'e2e_run01'), str(ROOT / 'e2e_run02'),
             '--out', str(ROOT / 'products_audit.json')], 300)
        call('cost_observation', [str(ROOT / 'source/run_experiment.py'), '--phase', 'collect',
             '--crops', '/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt',
             '--out', str(ROOT / 'cost_observation'), '--prefixes', '64,256',
             '--sources', '2', '--seeds', '1', '--seed-start', '58101',
             '--budget', '512', '--max-wall-seconds', '180'], 240)
        state.update(status='complete')
        save()
    except Exception as error:
        state.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
