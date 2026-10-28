#!/usr/bin/env python3
"""Read the completed multi-chunk results and summarize paired timings."""
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from conf.dataclasses import PCKeys

root = Path(sys.argv[1])
paths = {'target': root / 'full_target_01/target',
         'pf_w1': root / 'full_pf_01/pf_w1',
         'pf_rescue': root / 'full_pf_01/pf_rescue'}
reports = {}
tokens = {}
for name, path in paths.items():
    result = json.loads((path / 'result.json').read_text())
    saved = torch.load(path / 'tokens.pt', map_location='cpu', weights_only=True)
    length = saved['column_token_lengths']
    token_seq = saved['tokens_sequence']
    assert list(length.shape) == result['scene_cols']
    assert int(length.sum()) == int(token_seq.numel()) == result['tokens']
    assert int((length > 0).sum()) == result['nonempty_columns']
    assert result['total_columns'] == length.numel()
    tokens[name] = saved
    with torch.serialization.safe_globals([PCKeys]):
        scene = torch.load(path / 'gaussians.pt', map_location='cpu', weights_only=True)
    coords = scene['coords']
    assert coords.ndim == 2 and coords.shape[1] == 3 and coords.shape[0] == result['gaussians'] > 0
    for field, value in scene.items():
        if torch.is_tensor(value) and value.is_floating_point():
            assert bool(torch.isfinite(value).all()), (name, field)
    with Image.open(path / 'topdown.png') as image:
        image.load()
        assert image.size == (1024, 1024)
        assert any(low < high for low, high in image.convert('RGB').getextrema())
    reports[name] = dict(**result, empty_columns=result['total_columns'] - result['nonempty_columns'],
                         token_and_scene_readable=True, finite=True, image_nonuniform=True)
    del scene, coords

w1, rescue = tokens['pf_w1'], tokens['pf_rescue']
same_tokens = torch.equal(w1['tokens_sequence'], rescue['tokens_sequence'])
same_lengths = torch.equal(w1['column_token_lengths'], rescue['column_token_lengths'])
summary = dict(status='complete', methods=reports,
               w1_rescue_same_tokens=same_tokens, w1_rescue_same_column_lengths=same_lengths,
               time_ratios=dict(target_over_w1=reports['target']['total_s'] / reports['pf_w1']['total_s'],
                                target_over_rescue=reports['target']['total_s'] / reports['pf_rescue']['total_s'],
                                w1_over_rescue=reports['pf_w1']['total_s'] / reports['pf_rescue']['total_s']),
               interpretation='One seed; request durations can reflect different output lengths and occupancy.')
assert all(math.isfinite(x) for x in summary['time_ratios'].values())
(root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
