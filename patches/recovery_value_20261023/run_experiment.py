"""Collect diagnostic trajectories or time fixed, source-separated policies."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import os
import sys
import time


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['collect','timing','e2e'],required=True)
    p.add_argument('--crops',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--methods',default='w1,fixed1,value,mass,ranker_same_k')
    p.add_argument('--prefixes',default='64,256')
    p.add_argument('--sources',type=int,default=10)
    p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--seed-start',type=int,default=56101)
    p.add_argument('--budget',type=int,default=512)
    p.add_argument('--repeat-index',type=int,default=0)
    p.add_argument('--max-wall-seconds',type=float,default=1800)
    p.add_argument('--warm-tokens',type=Path)
    p.add_argument('--legacy-adaptive',type=Path,
                   help='Saved decoder used only for implementation comparisons')
    a=p.parse_args()
    project=Path('/root/workspace/GaussianGPT')
    source=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    old=Path('/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_widths')
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),str(old),str(source),str(source/'reference'),str(source/'reference/runtime'),str(project)]
    os.chdir(project)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context,load_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from tree_decoder import decode as decode_w1
    from position_rescue_runtime import PositionRescueRuntime
    from adaptive_decoder import decode as decode_adaptive
    from position_rescue_decoder import decode as decode_fixed
    from proposal_adapters import build_proposal
    from draft_a.pf_ranker_eval import cache_for,validate_sequence
    from draft_a.learning_eval import _fp32_runtime
    from evaluate_candidates import _prompt_rows,jsonable,runtime_delta
    original=Path('/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02')
    def old_module(filename,name):
        spec=importlib.util.spec_from_file_location(name,original/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    old_decoder=old_module('position_rescue_decoder.py','measured_single_rescue_decoder')
    old_runtime=old_module('position_rescue_runtime.py','measured_single_rescue_runtime')
    legacy_decoder=None
    if a.legacy_adaptive is not None:
        spec=importlib.util.spec_from_file_location('saved_adaptive_decoder',a.legacy_adaptive)
        legacy_decoder=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_decoder)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    raw=torch.load(a.crops,map_location='cpu',weights_only=True)
    prefixes=[int(x) for x in a.prefixes.split(',')]
    prompts,names=_prompt_rows(raw,prefixes,a.sources,a.seeds,a.seed_start,12096)
    if a.repeat_index:
        offset=(len(names)//2*a.repeat_index)%len(names)
        order=names[offset:]+names[:offset]
        source_order={name:i for i,name in enumerate(order)}
        prompts.sort(key=lambda row:(source_order[row['source']],row['prefix'],row['seed']))
    methods=['collect'] if a.phase=='collect' else a.methods.split(',')
    write(a.out/'prompts.json',{'prompts':prompts,'sources':names})
    started=time.perf_counter()
    progress={'status':'loading','completed':0,'expected':len(prompts)*len(methods)}
    write(a.out/'progress.json',progress)
    metadata=None if not a.model else json.loads(a.model.read_text())
    with threadpool_limits(limits=1),torch.inference_mode(),_fp32_runtime(torch),torch.autocast('cuda',enabled=False):
        context=build_context(project,device='cuda');target=context['target']
        adapters={m:load_method('pf_mlp',context) for m in methods}
        selectors={}
        selection_methods={'value','mass','ranker_same_k','value_reference',
                           'mass_reference','ranker_same_k_reference','old_value'}
        if any(m in selection_methods for m in methods):
            from selection import make_selector
            for m in methods:
                if m in selection_methods:
                    method_name='value' if m=='old_value' else m.removesuffix('_reference')
                    implementation='reference' if m=='old_value' or m.endswith('_reference') else 'compiled'
                    selectors[m]=make_selector(metadata,method_name,implementation=implementation)
        if 'old_value' in methods and legacy_decoder is None:
            raise ValueError('old_value requires --legacy-adaptive')
        vq=None
        if a.phase=='e2e':
            from model.gaussian_vqvae import GaussianVQVAE
            from draft_a.pf_multiscene.full_3d_eval import camera_views,gaussian_tensors,render_views,save_views
            import numpy as np
            vq=GaussianVQVAE.load_from_checkpoint(str(project/'checkpoints/vqvae_vfront.ckpt'),map_location='cpu').cuda().float().eval().requires_grad_(False)
            cameras=camera_views(20*vq.autoencoder.stride*vq.autoencoder.default_grid_size)
            body=torch.load(a.warm_tokens,map_location='cpu',weights_only=True)['tokens'].tolist()
            t,q=gaussian_tensors(vq,body);rgb,alpha=render_views(t,cameras)
            warm=a.out/'warmup_render';warm.mkdir();save_views(rgb,warm)
            del t,rgb,alpha
        def make_arm(method):
            linear=DynamicTargetGraph(target,cache_capacity_buckets=(2048,))
            return dict(runtime=linear,tree=TargetTreeRuntime(target,linear,graph=True,max_width=4),
                        rescue=(old_runtime.PositionRescueRuntime(target,linear,graph=True) if method=='fixed1'
                                else PositionRescueRuntime(target,linear,graph=True,backup_count=4)),
                        proposal=build_proposal(adapters[method],verify=False))
        def generate(method,row,arm,budget,events):
            factory=lambda core,length:arm['runtime'].cache_for(core,length,factory=cache_for)
            with arm['proposal'].context(),arm['runtime'].attention_context():
                kwargs=dict(tree_runtime=arm['tree'],width=1,cache_factory=factory)
                if method=='w1':return decode_w1(target,adapters[method].model,context['ranker'],context['ranks'],row['prompt'],budget,row['seed'],**kwargs)
                selector=selectors.get(method)
                if method=='collect':selector=lambda ctx:[]
                if method in ('same0','same1'):
                    return decode_adaptive(target,adapters[method].model,context['ranker'],context['ranks'],row['prompt'],budget,row['seed'],
                                           rescue_runtime=arm['rescue'],backup_count=int(method[-1]),**kwargs)
                if method.startswith('fixed'):
                    count=int(method[5:]);selector=None
                    if count==1:
                        return old_decoder.decode(target,adapters[method].model,context['ranker'],context['ranks'],row['prompt'],budget,row['seed'],
                                                  rescue_runtime=arm['rescue'],**kwargs)
                    return decode_fixed(target,adapters[method].model,context['ranker'],context['ranks'],row['prompt'],budget,row['seed'],
                                        rescue_runtime=arm['rescue'],backup_count=count,**kwargs)
                else:count=4
                diagnostic=dict(round_observer=events.append,observer_sync=True,cost_probe_stride=16,cost_probe_repeats=3,max_cost_probes=6) if method=='collect' else {}
                decoder=legacy_decoder.decode if method=='old_value' else decode_adaptive
                return decoder(target,adapters[method].model,context['ranker'],context['ranks'],row['prompt'],budget,row['seed'],
                                       rescue_runtime=arm['rescue'],backup_count=count,backup_selector=selector,**kwargs,**diagnostic)
        for method in methods:
            arm=make_arm(method)
            generate(method,prompts[0],arm,16,[])
            arm['proposal'].close();del arm;gc.collect()
        arms={};rows=[]
        write(a.out/'plan.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              methods=methods,sources=names,torch=torch.__version__,device=torch.cuda.get_device_name(),
              model=metadata,diagnostic_only=a.phase=='collect',max_backups=4,
              implementation_files=dict(w1=sys.modules[decode_w1.__module__].__file__,
                  fixed1_decoder=str(original/'position_rescue_decoder.py'),fixed1_runtime=str(original/'position_rescue_runtime.py'),
                  fixed_wide=sys.modules[decode_fixed.__module__].__file__,adaptive=sys.modules[decode_adaptive.__module__].__file__),
              scope='Independent persistent runtimes; fresh first measured graphs included; selector cost included. Model loading and warmup excluded.'))
        for index,prompt in enumerate(prompts):
            if time.perf_counter()-started>a.max_wall_seconds:raise TimeoutError('Predeclared wall limit exceeded')
            shift=(index+a.repeat_index)%len(methods)
            for method in methods[shift:]+methods[:shift]:
                progress.update(status='running',current={'source':prompt['source'],'prefix':prompt['prefix'],'method':method})
                write(a.out/'progress.json',progress)
                torch.cuda.synchronize();begin=time.perf_counter()
                created=method not in arms
                if created:arms[method]=make_arm(method)
                arm=arms[method]
                before={k:arm[k].stats() for k in ['runtime','tree','rescue','proposal']}
                events=[]
                selector_before=selectors[method].stats() if method in selectors else None
                stats,output=generate(method,prompt,arm,a.budget,events)
                torch.cuda.synchronize();generation_end=time.perf_counter()
                validate_sequence(prompt['prompt'],output,context['ranks'])
                stages={'generation_seconds':generation_end-begin}
                if vq is not None:
                    dest=a.out/'samples'/prompt['pair_id']/method;dest.mkdir(parents=True)
                    eos=bool(output and output[-1]==12096)
                    body=prompt['prompt'][1:]+(output[:-1] if eos else output)
                    torch.save(dict(tokens=torch.tensor(body),suffix=torch.tensor(output),has_eos=eos,prefix_tokens=prompt['prefix']),dest/'tokens.pt')
                    t,q=gaussian_tensors(vq,body);torch.cuda.synchronize();vtime=time.perf_counter()
                    np.savez_compressed(dest/'gaussians.npz',**{k:v.cpu().numpy() for k,v in t.items()})
                    rgb,alpha=render_views(t,cameras);save_views(rgb,dest);torch.cuda.synchronize()
                    stages.update(vq_and_pack_seconds=vtime-generation_end,render_and_save_seconds=time.perf_counter()-vtime,quality=jsonable(q))
                    del t,rgb,alpha
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                result=dict(**{k:v for k,v in prompt.items() if k!='prompt'},method=method,budget=a.budget,
                    generated_tokens=len(output),output_tokens=[int(x) for x in output],eos=bool(output and output[-1]==12096),
                    outer_wall_seconds=elapsed,tokens_per_second=len(output)/elapsed,runtime_created=created,
                    stats=jsonable(stats),stages=stages,repeat_index=a.repeat_index,
                    selector_stats_before=selector_before,
                    selector_stats_after=selectors[method].stats() if method in selectors else None,
                    runtime_deltas={k:runtime_delta(before[k],arm[k].stats()) for k in before})
                rows.append(result)
                with (a.out/'records.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
                if events:
                    with (a.out/'rounds.jsonl').open('a') as f:
                        for e in events:
                            e.update(source_id=prompt['source'],scene_id=prompt['source'],context_id=prompt['context_id'],seed=prompt['seed'],prefix_len=prompt['prefix'])
                            f.write(json.dumps(jsonable(e),allow_nan=False)+'\n')
                progress['completed']+=1;write(a.out/'progress.json',progress)
        for arm in arms.values():arm['proposal'].close()
        progress.update(status='complete',wall_seconds=time.perf_counter()-started)
        write(a.out/'progress.json',progress)
        print(json.dumps(progress),flush=True)


if __name__=='__main__':main()
