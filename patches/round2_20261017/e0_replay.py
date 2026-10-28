"""Replay existing EAGLE and retain per-row policy statistics."""
from pathlib import Path
import collections
import json
import sys
import torch

PROJECT = Path('/root/workspace/GaussianGPT')
OUT = Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
sys.path.insert(0, str(PROJECT))
from gaussiangpt_medusa_conditional_v3.medusa_conditional_heads import sampling_probs

rows = []
metadata = []


def record_rows(c):
    with torch.inference_mode():
        p = sampling_probs(c['p_logits'], c['temperature'], .9)
        q = sampling_probs(c['q_logits'], c['temperature'], .9)
        ol = torch.minimum(p, q).sum(-1).cpu()
        original = c['overlap_cpu']
        valid = c['valid_cpu']
        positions = c['positions_cpu']
        trans = c['transitions_cpu']
        for j in torch.where(valid)[0].tolist():
            rows.append(dict(file=c['path'].name, row=j, source_index=int(positions[j]),
                transition=int(trans[j]), overlap_softmax=float(original[j]), overlap_top_p_09=float(ol[j])))
        metadata.append(dict(file=c['path'].name, predicted_dtype=str(c['pred'].dtype),
            head_dtype=str(c['q_head'].dtype), source_dtype=str(c['source_h'].dtype),
            loaded_parameter_dtypes=dict(collections.Counter(str(v.dtype) for v in c['draft'].parameters())),
            model_config=c['checkpoint']['model_config'], training_config=c['training_cfg']))


torch.set_num_threads(1)
torch.set_float32_matmul_precision('highest')
source = PROJECT / 'profiling/eagle/e48_acceptance_audit.py'
code = source.read_text()
needle = '        file_payloads.append(\n'
assert code.count(needle) == 1
code = code.replace(needle, '        record_rows(locals())\n\n' + needle)
namespace = dict(__file__=str(source), __name__='round2_eagle_replay', record_rows=record_rows)
exec(compile(code, str(source), 'exec'), namespace)
checkpoint = PROJECT / 'profiling/eagle/results/e46_hprev_dual_20ep/last.pt'
blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
storage = dict(collections.Counter(str(v.dtype) for v in blob['model_state'].values() if torch.is_tensor(v)))
sys.argv = [str(source), '--candidate', 'e46='+str(checkpoint), '--amp', 'fp32',
    '--output-json', str(OUT/'E0_fp32_replay.json')]
namespace['main']()
groups = collections.defaultdict(list)
for r in rows:
    pos = r['source_index']
    bucket = '0_63' if pos < 64 else '64_255' if pos < 256 else '256_1023' if pos < 1024 else '1024_plus'
    groups[(r['transition'], bucket)].append(r)
summary = {f'{t}:{b}':dict(count=len(v), overlap_softmax=sum(r['overlap_softmax'] for r in v)/len(v),
    overlap_top_p_09=sum(r['overlap_top_p_09'] for r in v)/len(v)) for (t,b),v in groups.items()}
(OUT/'E0_rows.json').write_text(json.dumps(dict(storage_dtypes=storage, metadata=metadata, groups=summary,
    rows=rows, tf32_matmul=torch.backends.cuda.matmul.allow_tf32, matmul_precision=torch.get_float32_matmul_precision()), indent=2, default=str))
print(json.dumps(dict(storage_dtypes=storage, groups=summary), indent=2), flush=True)
