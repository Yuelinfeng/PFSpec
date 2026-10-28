#!/usr/bin/env python3
"""Compare the actual scene sampler with ordinary and graph execution."""
import gc
import json
import sys
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scene as run


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    run.write_json(out / 'progress.json', dict(status='loading'))
    try:
        with torch.inference_mode(), threadpool_limits(limits=1):
            gpt, model, context = run.load_models()
            adapter = run.api.load_method('pf_mlp', context)
            outputs = {}
            for name in ('pf_w1', 'pf_rescue'):
                for optimized in (False, True):
                    label = name + ('_graph' if optimized else '_ordinary')
                    run.write_json(out / 'progress.json', dict(status='running', method=label))
                    active = run.arm(name, context, adapter, optimized)
                    sampler = run.Sampler(model, context, active)
                    sampler.install()
                    try:
                        tokens, lengths, _ = run.scene._generate_sparse_scene(
                            gpt, **run.scene_args(gpt, 4, 0))
                        outputs[label] = dict(tokens=tokens.tolist(), lengths=lengths.tolist(),
                                              calls=sampler.records,
                                              runtime=run.compare.arm_stats(active))
                        print(json.dumps(dict(event='scene_validated', method=label,
                                              calls=len(sampler.records), tokens=int(tokens.numel()))), flush=True)
                    finally:
                        sampler.remove()
                        run.compare.close_arm(active)
                        gc.collect()
                        torch.cuda.empty_cache()
            for name in ('pf_w1', 'pf_rescue'):
                a, b = outputs[name + '_ordinary'], outputs[name + '_graph']
                if any(a[k] != b[k] for k in ('tokens', 'lengths', 'calls')):
                    raise AssertionError(name + ' scene outputs or actions differ')
            a, b = outputs['pf_w1_graph'], outputs['pf_rescue_graph']
            if a['tokens'] != b['tokens'] or a['lengths'] != b['lengths']:
                raise AssertionError('W1 and Rescue scene outputs differ')
            write = dict(passed=True, outputs=outputs)
            run.write_json(out / 'validation.json', write)
            run.write_json(out / 'progress.json', dict(status='complete'))
    except BaseException as exc:
        run.write_json(out / 'progress.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
