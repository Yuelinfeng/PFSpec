"""Finish this finite experiment after its training process has completed."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_extend_20261019')


def main():
    while True:
        state = json.loads((ROOT / 'pipeline.json').read_text())
        if state['status'] == 'trained':
            break
        if state['status'] in ['failed', 'complete']:
            return
        if not Path(f"/proc/{state['pid']}").exists():
            raise RuntimeError('Training process ended before recording completion')
        time.sleep(20)
    try:
        subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate_extension.py'), '--smoke'], check=True)
        subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate_extension.py')], check=True)
    except Exception as error:
        (ROOT / 'pipeline.json').write_text(json.dumps(dict(status='failed', phase='evaluation',
                                                           error=str(error)), indent=2))
        raise


if __name__ == '__main__':
    main()
