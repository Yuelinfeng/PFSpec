#!/usr/bin/env python3
"""Measure and compare node-wise and batched rotary phases on CUDA."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output already exists')
    sys.path.insert(0, str(args.project))
    import torch
    from model.gpt.rope import RotaryPositionEmbedder
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    with torch.inference_mode():
        for head_dim in [64, 128]:
            embedder = RotaryPositionEmbedder(head_dim, dim=4).cuda()
            for nodes in [4, 6, 10]:
                indices = torch.arange(nodes*4, device='cuda').reshape(1, nodes, 4) % 20
                indices[:, :, 3] %= 2
                def separate():
                    return torch.cat([embedder(indices[:, n:n+1]) for n in range(nodes)], dim=1)
                def batch():
                    return embedder(indices)
                a, b = separate(), batch()
                same = bool(torch.equal(a, b))
                timings = {}
                for name, function in [('separate', separate), ('batch', batch)]:
                    for _ in range(20):
                        function()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(300):
                        function()
                    torch.cuda.synchronize()
                    timings[name+'_wall_ms'] = (time.perf_counter()-started)*1000/300
                rows.append(dict(head_dim=head_dim, nodes=nodes, bit_equal=same,
                                 max_abs=float((a-b).abs().max()), **timings))
    result = dict(status='complete' if all(r['bit_equal'] for r in rows) else 'failed',
                  timing_scope='CUDA synchronized outer wall for ordinary rotary preparation', rows=rows)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if result['status'] != 'complete':
        raise ArithmeticError('CUDA rotary phases differ')


if __name__ == '__main__':
    main()
