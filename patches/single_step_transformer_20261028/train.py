"""Matched three-seed continuations on the current full Feature data."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch

from feature_model import load_model, SingleStepFeature
from legacy_models import SingleStepFeature as LegacySingleStep
from draft_a import feature_target_v2 as training


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def initial_value(root, seed, kind):
    legacy = 'b0' if kind == 'mlp' else 'b1_e'
    path = root/'initial_direct/b1_training'/str(seed)/legacy/'selected.pt'
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value['method'] != legacy or value['seed'] != seed:
        raise ValueError('Legacy model identity differs')
    return dict(model_state=value['state'], model_config=dict(architecture=kind,
                training_seed=seed, legacy_checkpoint=str(path), legacy_selected_epoch=value['epoch']),
                teacher='checkpoints/gpt_vfront.ckpt', student_temperature=.9,
                student_top_p=.9, step=0, trained_feature_rows=0, training_seed=seed,
                legacy_checkpoint=str(path), legacy_selected_epoch=value['epoch'])


def validate(root, assets, value, sample):
    from mlp_medusa_graph import ProposalForwardGraph
    model = load_model(value, assets).eval().requires_grad_(False)
    original = LegacySingleStep(assets['feature_weight'], assets['token_embeddings'])
    original.load_state_dict(value['model_state'], strict=True)
    original = original.cuda().float().eval().requires_grad_(False)
    inputs = tuple(sample[k][:16].cuda() for k in
                   ('previous_state','previous_feature','position'))
    with torch.inference_mode():
        expected, actual = original(*inputs), model(*inputs)
        static_equal = all(torch.equal(a,b) for a,b in zip(expected,actual))
        if not static_equal:
            raise ArithmeticError('Preallocated mask changed legacy output')
        one = tuple(x[:1] for x in inputs)
        eager = tuple(x.clone() for x in model(*one))
        graph = ProposalForwardGraph(model, 'forward', method_kind='mlp')
        with graph.context():
            graphed = model(*one)
        graph_equal = all(torch.equal(a,b) for a,b in zip(eager,graphed))
        stats = graph.stats()
        graph.close()
        if not graph_equal:
            raise ArithmeticError('Graph output differs from eager output')
        # The first causal token may not depend on the later candidate Position.
        ids = torch.stack((inputs[1]+8000,inputs[2]),-1)
        embed = torch.nn.functional.embedding(ids,model.token_embeddings)
        x = model.state_project(model.state_norm(inputs[0])).unsqueeze(1)+model.token_project(embed)
        x = x+model.role.weight.unsqueeze(0)
        y = model.block(x,src_mask=model.causal_mask)
        changed = x.clone(); changed[:,1,::2] += 3.
        z = model.block(changed,src_mask=model.causal_mask)
        causal_equal = torch.equal(y[:,0],z[:,0])
        if not causal_equal:
            raise ArithmeticError('First token depends on a future token')
        unmasked = model.block(x,src_mask=None)
        unmasked_changed = model.block(changed,src_mask=None)
        probe_sensitive = not torch.equal(unmasked[:,0],unmasked_changed[:,0])
        if not probe_sensitive:
            raise ArithmeticError('Causality probe is insensitive without its mask')
    del original, model
    gc.collect()
    torch.cuda.empty_cache()
    report = dict(seed=value['training_seed'], legacy_output_equal=static_equal,
                  graph_output_equal=graph_equal, first_token_causal=causal_equal,
                  unmasked_probe_sensitive=probe_sensitive, graph=stats)
    write(root/'implementation'/f'seed_{value["training_seed"]}.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-plan',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new training output directory')
    args.out.mkdir(parents=True)
    start=time.monotonic()
    try:
        torch.set_num_threads(6)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        assets=torch.load(args.root/'fixed_assets.pt',map_location='cpu',weights_only=True)
        plan=json.loads(args.data_plan.read_text())
        for split in ('train','val'):
            if args.smoke:
                rows=plan['splits'][split]['records'][:2]
                plan['splits'][split]=dict(scenes=2,rows=sum(x['rows'] for x in rows),records=rows)
                plan['files'][split]=[x['name'] for x in rows]
            for row in plan['splits'][split]['records']:
                if not Path(row['path']).is_file():
                    raise FileNotFoundError(row['path'])
        if set(plan['files']['train']) & set(plan['files']['val']):
            raise ValueError('Training and validation trajectories overlap')
        sample=torch.load(plan['splits']['train']['records'][0]['path'],weights_only=True)
        for seed in (20261018,20261019,20261020):
            validate(args.out,assets,initial_value(args.root,seed,'transformer'),sample)
        del sample
        training.model_from=lambda value,device='cuda':load_model(value,assets,device)
        write(args.out/'experiment.json',dict(seeds=[20261018,20261019,20261020],
              methods=['mlp','transformer'],steps=2 if args.smoke else args.steps,
              initialization='legacy same-budget selected models, per architecture and seed',
              objective='raw Target full-distribution CE + 2 * policy TV',lr=1e-4,final_lr=1e-5,
              train_rows=plan['splits']['train']['rows'],validation_rows=plan['splits']['val']['rows'],
              teacher='checkpoints/gpt_vfront.ckpt',test_used=False,
              trained_feature_rows_meaning='this continuation only; legacy search used 10 epochs',
              scope='two-token single-layer Feature proposal; shared fixed Target embedding and head; no Draft KV'))
        for seed in (20261018,20261019,20261020):
            for kind in ('mlp','transformer'):
                dest=args.out/str(seed)/kind
                dest.mkdir(parents=True)
                value=initial_value(args.root,seed,kind)
                torch.save(value,dest/'initial.pt')
                write(dest/'data_plan.json',plan)
                write(args.out/'progress.json',dict(status='training',seed=seed,method=kind,
                      pid=os.getpid(),seconds=time.monotonic()-start))
                training.train(dest,initial_path=dest/'initial.pt',chunk=256,
                    phase_plan=[dict(name='full_data_continuation',lr=1e-4,weight=2.,seed=seed,
                                     steps=2 if args.smoke else args.steps)],
                    eval_every=1 if args.smoke else 400,train_probe=4)
                gc.collect();torch.cuda.empty_cache()
        write(args.out/'progress.json',dict(status='complete',seconds=time.monotonic()-start))
    except Exception:
        write(args.out/'progress.json',dict(status='failed',error=traceback.format_exc(),
                                          seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    main()
