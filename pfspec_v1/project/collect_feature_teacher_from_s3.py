#!/usr/bin/env python3
"""Create Feature-only distillation rows from an existing S3 teacher trajectory corpus.

No new scene generation. Uses frozen FP32 GaussianGPT and on-policy-prefix
teacher-forcing to label selected Feature rows (T=.9, top_p=.9 by default).

For sequence P0,F0,P1,F1,...:
  teacher input = EOS,P0,F0,P1,...,P_last
  teacher output at index 2*i+1 is p(F_i | P_<=i,F_<i)
  student input = last context_tokens of [P0,F0,...,P_i] padded on left.

Outputs .pt tensors: input_ids[int16; N,L], teacher_probs[float16; N,4096].
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch

FEATURE_VOCAB = 4096


def policy_batch(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Same 1D nucleus rule as decoder.target_policy, applied to batches."""
    z = logits.float() / temperature
    if top_p < 1.0:
        vals, ids = torch.sort(z, dim=-1, descending=True)
        pp = torch.softmax(vals, dim=-1)
        remove = pp.cumsum(dim=-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        vals = vals.masked_fill(remove, float('-inf'))
        z = torch.full_like(z, float('-inf'))
        z.scatter_(1, ids, vals)
    return torch.softmax(z, dim=-1)


def make_inputs(seq: torch.Tensor, context_tokens: int, pad_id: int, p_vocab: int, eos: int):
    seq = seq.flatten().long().cpu()
    if len(seq) < 2 or len(seq) % 2:
        raise ValueError('Expected even-length interleaved P/F sequence')
    if bool(((seq[::2] < 0) | (seq[::2] >= p_vocab)).any()):
        raise ValueError('Invalid position token in corpus')
    if bool(((seq[1::2] < p_vocab) | (seq[1::2] >= eos)).any()):
        raise ValueError('Invalid feature token in corpus')
    n = len(seq) // 2
    result = torch.full((n, context_tokens + 1), pad_id, dtype=torch.int16)
    for i in range(n):
        # At position 2*i, F_i is not yet present. The last ID is P_i.
        prefix = seq[:2*i+1][-context_tokens-1:]
        result[i, -len(prefix):] = prefix.to(torch.int16)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', type=Path, default=Path(
        'profiling/s3/s3_onpolicy_feature_distill_32x2048_256step/teacher_trajectory_corpus.pt'))
    ap.add_argument('--gpt', type=Path, default=Path('checkpoints/gpt_vfront.ckpt'))
    ap.add_argument('--vqvae', type=Path, default=Path('checkpoints/vqvae_vfront.ckpt'))
    ap.add_argument('--output-dir', type=Path, default=Path('profiling/feature_draft/pilot_data'))
    ap.add_argument('--context-tokens', type=int, default=64)
    ap.add_argument('--train-rows', type=int, default=4096)
    ap.add_argument('--val-rows', type=int, default=1024)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.context_tokens < 2 or args.context_tokens % 2:
        ap.error('--context-tokens must be positive even')
    if min(args.train_rows,args.val_rows) < 1:
        ap.error('train-rows and val-rows must be positive')
    if not (args.temperature > 0 and 0 < args.top_p <= 1):
        ap.error('Invalid decoding policy')
    for file in [args.corpus, args.gpt, args.vqvae]:
        if not file.is_file():
            raise SystemExit(f'Missing required input file: {file}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')

    obj = torch.load(args.corpus, map_location='cpu', weights_only=True)
    if obj.get('format') != 'teacher_generated_sparse_token_sequences_without_bos_or_eos':
        raise SystemExit(f'Unknown S3 corpus format: {obj.get("format")}')
    train = obj['train_sequences']
    val = obj['val_sequences']
    if not train or not val:
        raise SystemExit('Both train_sequences and val_sequences required')
    print(f'Loaded existing S3 corpus: train_seq={len(train)} val_seq={len(val)}', flush=True)
    torch.set_float32_matmul_precision('highest')
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE
    vq = GaussianVQVAE.load_from_checkpoint(str(args.vqvae), map_location='cpu').eval()
    teacher = GaussianGPT.load_from_checkpoint(str(args.gpt), map_location='cpu', vqvae=vq).eval()
    teacher.gpt.to(args.device).eval().requires_grad_(False)
    base = getattr(teacher.gpt, '_orig_mod', teacher.gpt)
    model = base.transformer
    p_vocab = int(base.position_vocab_size)
    eos = int(base.eos_token_id)
    if eos-p_vocab != FEATURE_VOCAB:
        raise AssertionError(f'Expected Feature 4096, got {eos-p_vocab}')
    if eos > 32767:
        raise AssertionError('int16 storage would truncate token ID')
    pad = eos + 1
    if pad > 32767:
        raise AssertionError('int16 storage would truncate PAD ID')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(source_corpus=str(args.corpus), gpt=str(args.gpt),
                   teacher_temperature=args.temperature, teacher_top_p=args.top_p,
                   context_tokens=args.context_tokens, pad_id=pad,
                   feature_offset=p_vocab, feature_size=FEATURE_VOCAB, splits={})
    with torch.inference_mode():
        for name, seqs, limit in [('train', train, args.train_rows),
                                  ('val', val, args.val_rows)]:
            inputs, labels, taken = [], [], 0
            tic = time.perf_counter()
            for s, seq in enumerate(seqs):
                if taken >= limit:
                    break
                seq = seq.long().flatten().cpu()
                x_all = make_inputs(seq, args.context_tokens, pad, p_vocab, eos)
                # Single teacher-forced forward obtains all Feature distributions.
                inp = torch.cat([torch.tensor([eos]), seq[:-1]]).unsqueeze(0).to(args.device)
                out = model(inp)
                if not torch.is_tensor(out):
                    out = out[0]
                if out.ndim != 3 or out.shape[1] != seq.numel():
                    raise AssertionError(f'Unexpected Target logits shape={out.shape}')
                feature_logits = out[0, 1::2, p_vocab:eos]
                if feature_logits.shape != (seq.numel()//2, FEATURE_VOCAB):
                    raise AssertionError(f'Incorrect Feature logits shape={feature_logits.shape}')
                remaining = min(limit - taken, feature_logits.shape[0])
                probs = policy_batch(feature_logits[:remaining], args.temperature, args.top_p)
                if not bool(torch.isfinite(probs).all()):
                    raise AssertionError('Non-finite Teacher feature probabilities')
                torch.testing.assert_close(probs.sum(-1), torch.ones(remaining, device=probs.device), atol=1e-4, rtol=0)
                inputs.append(x_all[:remaining].contiguous())
                labels.append(probs.to(torch.float16).cpu().contiguous())
                taken += remaining
                print(f'{name}: seq={s}, collected={taken}/{limit}, '
                      f'seconds={time.perf_counter()-tic:.1f}', flush=True)
                del inp, out, feature_logits, probs
            if taken != limit:
                print(f'WARNING: {name} corpus only contained {taken}/{limit} rows', flush=True)
            if not inputs:
                raise RuntimeError(f'No rows collected for {name}')
            payload = dict(input_ids=torch.cat(inputs),teacher_probs=torch.cat(labels),
                           pad_id=pad, feature_offset=p_vocab, context_tokens=args.context_tokens,
                           temperature=args.temperature,top_p=args.top_p, source_corpus=str(args.corpus))
            path = args.output_dir / f'{name}.pt'
            torch.save(payload,path)
            summary['splits'][name] = dict(rows=taken, sequences_used=s+1, path=str(path))
            print(f'Saved {path}: rows={taken} shape={tuple(payload["input_ids"].shape)}',flush=True)
    (args.output_dir / 'metadata.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
    print('PASS: Distillation input excludes current Feature and teacher distributions are normalized.', flush=True)


if __name__ == '__main__':
    main()
