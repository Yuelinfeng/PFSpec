"""Collect teacher labels and measure proposal cost within one 900-second run."""
import time
STARTED = time.monotonic()
DEADLINE = STARTED + 900.0
import argparse
import json
from pathlib import Path
import signal
import statistics
import sys
import traceback


def write(path, value):
    def convert(x):
        if hasattr(x, 'detach'):
            return x.detach().cpu().tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert))


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--existing-teacher', type=Path)
    args = parser.parse_args()
    if args.deadline_unix is not None:
        DEADLINE = min(DEADLINE, time.monotonic()+args.deadline_unix-time.time())
    if time.monotonic() >= DEADLINE:
        raise TimeoutError('G1 original deadline has passed')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(stage='G1', status='running', limit_seconds=900,
                 started_unix=time.time() - (time.monotonic() - STARTED))
    write(out / 'pipeline.json', state)
    def alarm(*_):
        raise TimeoutError('G1 shared 900-second limit reached')
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.01, DEADLINE-time.monotonic()))
    try:
        import torch
        from capture_transition_teacher import load_runtime, collect_teacher
        from train_transition import load_teacher
        from transition_model import TransitionModel, fit_hidden_map, CONFIGS
        from transition_adapter import TransitionAdapter, cost_probe
        sys.path.insert(0, str(Path(args.project).resolve()))
        from draft_a.pf_feature_unified import MLPAdapter, decode
        runtime = load_runtime(args.project)
        teacher_dir = (args.existing_teacher.resolve() if args.existing_teacher else out/'teacher')
        if args.existing_teacher:
            collection = json.loads((teacher_dir/'collection.json').read_text())
            if collection.get('status') != 'complete':
                raise RuntimeError('Saved teacher collection is incomplete')
        else:
            _, collection = collect_teacher(runtime, args.manifest.resolve(),
                teacher_dir, deadline_monotonic=DEADLINE)
        state['collection'] = collection
        write(out / 'pipeline.json', state)
        data, data_summary = load_teacher(torch, teacher_dir / 'teacher.pt')
        state['data'] = data_summary
        train_ids = [i for i, s in enumerate(data['split']) if s == 'train']
        hidden = torch.cat([data[k][train_ids] for k in ('h0', 'hF', 'hP')])
        wte = torch.load(teacher_dir / 'token_embeddings.pt', map_location='cpu', weights_only=True)
        chosen = None
        costs = {}
        chosen_models = None
        with torch.inference_mode():
            for size in ('main', 'narrow'):
                torch.set_num_threads(4)
                mapping = fit_hidden_map(hidden, width=CONFIGS[size]['d'])
                torch.set_num_threads(1)
                models = {m: TransitionModel(m, wte, mapping, size=size).cuda().eval()
                          for m in ('direct', 'two_step')}
                case_ids = [min(i*len(data['split'])//16, len(data['split'])-1) for i in range(16)]
                cases = [dict(sample_id=str(i), prefix=[int(data['previous_position_id'][i]),
                    int(data['previous_feature_id'][i])], p1=int(data['candidate_position_id'][i]),
                    h0=data['h0'][i].cuda()) for i in case_ids]
                costs[size] = {}
                for method, model in models.items():
                    old = MLPAdapter(runtime['feature'])
                    new = TransitionAdapter(runtime['feature'], model)
                    old.prepare([12096], 1024)
                    new.prepare([12096], 1024)
                    def call(adapter, case):
                        return adapter.propose(case['prefix'], case['p1'], case['h0'], None)
                    result = cost_probe(lambda c: call(old, c), lambda c: call(new, c),
                        cases, pairs=64, warmup_pairs=4,
                        deadline_unix=time.time()+DEADLINE-time.monotonic(),
                        synchronize=torch.cuda.synchronize)
                    if result['status'] != 'complete':
                        raise TimeoutError('Incomplete proposal cost pairs')
                    deltas = [r['delta_seconds']*1000 for r in result['rows']]
                    result['mean_delta_ms'] = statistics.mean(deltas)
                    result['median_delta_ms'] = statistics.median(deltas)
                    costs[size][method] = result
                    old.close()
                    new.close()
                write(out / 'costs.json', costs)
                if costs[size]['two_step']['mean_delta_ms'] <= .5:
                    chosen, chosen_models = size, models
                    break
                del models
            state['selected_size'] = chosen
            if chosen is None:
                state['status'] = 'stopped_cost'
            else:
                manifest = json.loads(args.manifest.read_text())
                dev = sorted([s for s in manifest['sources'] if s['split']=='dev'],
                             key=lambda s: s['scene_id'])
                replay_cases = [(dev[0], p, b, 39101) for p in (0,64,256) for b in (16,64)]
                replay_cases += [(dev[0], 64, 2, 39101), (dev[0], 64, 4, 39101),
                                 (dev[5], 256, 64, 42454)]
                comparisons = []
                fields = ('target_calls','target_input_tokens','draft_calls','draft_input_tokens',
                    'ranker_calls','rounds','proposed','verified','accepted','accepted_lengths',
                    'round_generated_lengths','position_proposed','position_verified',
                    'position_accepted','feature_proposed','feature_verified','feature_accepted',
                    'eos_proposed','eos_verified','eos_accepted','position_fallbacks')
                for case_id, (source, prefix, budget, seed) in enumerate(replay_cases):
                    crops = torch.load(source['crops_path'], map_location='cpu', weights_only=True)
                    body = crops[source['crop_index']]['tokens'][:prefix].tolist()
                    prompt = [12096] + body
                    reference = None
                    for name in ('w1','direct','two_step'):
                        adapter = (MLPAdapter(runtime['feature']) if name=='w1' else
                                   TransitionAdapter(runtime['feature'], chosen_models[name], residual_scale=0))
                        rewinds, inputs = [], []
                        original_rewind = adapter.rewind
                        def traced_rewind(keep):
                            rewinds.append(int(keep))
                            return original_rewind(keep)
                        adapter.rewind = traced_rewind
                        hook = runtime['target'].register_forward_pre_hook(
                            lambda module, values: inputs.append(values[0].detach().cpu().tolist()))
                        try:
                            stats, output = decode(runtime['target'], adapter, runtime['ranker'],
                                runtime['ranks'], prompt, budget, seed)
                        finally:
                            hook.remove()
                        record = dict(case_id=case_id, method=name, source=source['scene_id'],
                            prefix=prefix, budget=budget, seed=seed, prompt=prompt, output=output,
                            counters={k:stats.get(k) for k in fields}, rewind_keep=rewinds,
                            target_inputs=inputs)
                        with (out/'zero_replay.jsonl').open('a') as f:
                            f.write(json.dumps(record)+'\n')
                        if reference is None:
                            reference = record
                        else:
                            equal = all(record[k]==reference[k] for k in
                                ('output','counters','rewind_keep','target_inputs'))
                            comparisons.append(dict(case_id=case_id,method=name,equal=equal))
                            if not equal:
                                raise RuntimeError('Zero residual replay differs from W1')
                state['zero_replay'] = dict(comparisons=comparisons, all_equal=True)
                state['status'] = 'complete'
        write(out / 'selected_size.json', dict(size=chosen, cost_rule='C mean full-proposal increment <=0.5ms; target0.25ms'))
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state['elapsed_seconds'] = time.monotonic()-STARTED
        state['finished_unix'] = time.time()
        write(out / 'pipeline.json', state)


if __name__ == '__main__':
    main()
