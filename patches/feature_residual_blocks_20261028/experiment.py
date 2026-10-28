"""Train matched continuations of zero, three and five added Feature blocks."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch

from block_model import expand_checkpoint, from_checkpoint
from draft_a import feature_target_v2 as training


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def configure():
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_initial(value, head, sample, out):
    from mlp_medusa_graph import ProposalForwardGraph
    configure()
    original = from_checkpoint(value, head).eval()
    inputs = tuple(sample[k][:16].cuda() for k in
                   ('previous_state', 'previous_feature', 'position'))
    reports = []
    with torch.inference_mode():
        expected = original(*inputs)
        for added in (0, 3, 5):
            expanded = expand_checkpoint(value, head, added)
            model = from_checkpoint(expanded, head).eval()
            actual = model(*inputs)
            exact = all(torch.equal(a, b) for a, b in zip(expected, actual))
            if not exact:
                raise ArithmeticError(f'Initial output differs for {added} extra blocks')
            graph = ProposalForwardGraph(model, 'forward', method_kind='mlp')
            with graph.context():
                graphed = model(*(x[:1] for x in inputs))
            eager = model(*(x[:1] for x in inputs))
            graph_equal = all(torch.equal(a, b) for a, b in zip(eager, graphed))
            graph_report = graph.stats()
            graph.close()
            if not graph_equal:
                raise ArithmeticError(f'Graph output differs for {added} extra blocks')
            reports.append(dict(added_blocks=added, total_blocks=1 + added,
                                parameters=sum(p.numel() for p in model.parameters()),
                                original_output_equal=exact, graph_output_equal=graph_equal,
                                graph=graph_report))
            del model
    del original
    # Verify that every added block can learn and serialization retains its output.
    model = from_checkpoint(expand_checkpoint(value, head, 5), head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    target = sample['target'][:16].cuda()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(*inputs)
        losses, _, _ = training.objective(logits, target, 2.)
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    learned = [bool(block.delta[-1].weight.detach().abs().max() > 0)
               for block in model.extra_blocks]
    if not all(learned):
        raise ArithmeticError('An added block did not receive a training update')
    saved = dict(model_config=dict(model.config), model_state=model.state_dict())
    out.mkdir(parents=True, exist_ok=True)
    torch.save(saved, out / 'roundtrip.pt')
    restored = from_checkpoint(torch.load(out / 'roundtrip.pt', weights_only=True), head)
    with torch.inference_mode():
        serialization_equal = all(torch.equal(a, b) for a, b in zip(model(*inputs), restored(*inputs)))
    if not serialization_equal:
        raise ArithmeticError('Reloaded model output differs')
    result = dict(passed=True, models=reports, all_new_blocks_updated=learned,
                  serialization_output_equal=serialization_equal)
    write(out / 'validation.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-plan', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--eval-every', type=int, default=400)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.root.exists():
        raise FileExistsError('Use a new output directory')
    args.root.mkdir(parents=True)
    started = time.monotonic()
    configure()
    try:
        value = torch.load(args.initial, map_location='cpu', weights_only=True)
        head = torch.load(training.HEAD, map_location='cpu', weights_only=True)['weight']
        plan = json.loads(args.data_plan.read_text())
        for split in ('train', 'val'):
            for row in plan['splits'][split]['records']:
                row['path'] = str(args.data_root / split / row['name'])
                if not Path(row['path']).is_file():
                    raise FileNotFoundError(row['path'])
            if args.smoke:
                rows = plan['splits'][split]['records'][:2]
                plan['splits'][split] = dict(scenes=len(rows), rows=sum(r['rows'] for r in rows), records=rows)
                plan['files'][split] = [r['name'] for r in rows]
        if set(plan['files']['train']) & set(plan['files']['val']):
            raise ValueError('Training and validation trajectories overlap')
        sample = torch.load(plan['splits']['val']['records'][0]['path'], weights_only=True)
        validate_initial(value, head, sample, args.root / 'implementation')
        del sample
        gc.collect()
        torch.cuda.empty_cache()
        write(args.root / 'experiment.json', dict(
            initial=str(args.initial), start_step=value.get('step'), added_blocks=[0, 3, 5],
            total_blocks=[1, 4, 6], shared_embeddings=True, train_all_blocks=True,
            per_block='LayerNorm + 1024-to-256 input + 512-to-512-to-512-to-1024 residual',
            initialization='Existing weights retained; each new output Linear starts at zero',
            steps=2 if args.smoke else args.steps, training_seed=24690,
            selection='full validation distribution overlap', smoke=args.smoke))
        training.model_from = lambda v, device='cuda': from_checkpoint(v, head, device)
        for added in (0, 3, 5):
            root = args.root / f'add{added}'
            root.mkdir()
            initial = expand_checkpoint(value, head, added)
            torch.save(initial, root / 'initial.pt')
            write(root / 'data_plan.json', plan)
            write(args.root / 'progress.json', dict(status='training', added_blocks=added,
                                                    pid=os.getpid(), seconds=time.monotonic()-started))
            training.train(root, initial_path=root/'initial.pt', chunk=256,
                           phase_plan=[dict(name='matched_continuation', lr=1e-5, weight=2.,
                                            seed=24690, steps=2 if args.smoke else args.steps)],
                           eval_every=1 if args.smoke else args.eval_every, train_probe=4)
            gc.collect()
            torch.cuda.empty_cache()
        write(args.root / 'progress.json', dict(status='complete', phase='training',
                                                seconds=time.monotonic()-started))
    except Exception:
        write(args.root / 'progress.json', dict(status='failed', error=traceback.format_exc(),
                                                seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
