"""Recompute the two-checkpoint comparison from its saved individual runs."""
import json
import math
from pathlib import Path
import sys


def validate(directory):
    root = Path(directory)
    rows = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
    summary = json.loads((root / 'summary.json').read_text())
    assert len(rows) == 120
    keys = [(r['step'], r['source'], r['prefix'], r['seed_base']) for r in rows]
    assert len(set(keys)) == 120
    pairs = {}
    for row in rows:
        pairs.setdefault(row['pair_id'], {})[row['step']] = row
        tokens = row['output_tokens']
        assert len(tokens) == row['generated_tokens'] <= 512
        assert row['eos'] == bool(tokens and tokens[-1] == 12096)
        assert 12096 not in tokens[:-1]
        assert row['eos'] or len(tokens) == 512
        assert row['outer_wall_seconds'] >= row['seconds'] > 0
        assert sum(row['round_generated_lengths']) == len(tokens)
        assert len(row['round_generated_lengths']) == row['rounds']
        for field in ['proposed', 'verified', 'accepted']:
            assert row[field] == row['position_' + field] + row['feature_' + field]
        for i, token in enumerate(tokens):
            if token == 12096:
                assert i % 2 == 0
            elif i % 2 == 0:
                assert 0 <= token < 8000
            else:
                assert 8000 <= token < 12096
    assert len(pairs) == 60
    for pair in pairs.values():
        assert set(pair) == {12000, 48000}
        a, b = pair[12000], pair[48000]
        assert all(a[k] == b[k] for k in ['source', 'crop_seed', 'prefix', 'seed', 'seed_base', 'budget'])
        assert {a['order_index'], b['order_index']} == {0, 1}
    assert sum(pair[12000]['order_index'] == 0 for pair in pairs.values()) == 30

    def pooled(group):
        return sum(x['generated_tokens'] for x in group) / sum(x['outer_wall_seconds'] for x in group)

    sources = sorted({r['source'] for r in rows})
    assert len(sources) == 10
    ratios = {}
    for source in sources:
        a = [r for r in rows if r['source'] == source and r['step'] == 12000]
        b = [r for r in rows if r['source'] == source and r['step'] == 48000]
        assert len(a) == len(b) == 6
        ratios[source] = pooled(b) / pooled(a)
    geo = math.exp(sum(math.log(x) for x in ratios.values()) / 10)
    assert abs(geo - summary['speed_48000_over_12000']['source_equal_geometric_mean']) < 1e-12
    for source, value in ratios.items():
        assert abs(value - summary['speed_48000_over_12000']['per_source'][source]) < 1e-12
    by_step = {}
    for step in [12000, 48000]:
        group = [r for r in rows if r['step'] == step]
        by_step[str(step)] = dict(
            runs=len(group), tokens=sum(x['generated_tokens'] for x in group),
            outer_seconds=sum(x['outer_wall_seconds'] for x in group),
            pooled_tps=pooled(group), target_calls=sum(x['target_calls'] for x in group),
            target_input_tokens=sum(x['target_input_tokens'] for x in group),
            feature_accepted=sum(x['feature_accepted'] for x in group),
            feature_verified=sum(x['feature_verified'] for x in group),
            eos=sum(x['eos'] for x in group),
        )
        assert abs(by_step[str(step)]['pooled_tps'] - summary['by_checkpoint'][str(step)]['pooled_tokens_per_second']) < 1e-12
    full = [p for p in pairs.values() if p[12000]['generated_tokens'] == p[48000]['generated_tokens'] == 512]
    full_ratio = pooled([p[48000] for p in full]) / pooled([p[12000] for p in full])
    result = dict(status='passed', runs=120, pairs=60, sources=10,
                  first_order_pairs_each=30, by_step=by_step, per_source=ratios,
                  source_equal_speed_48000_over_12000=geo,
                  common_full_pairs=len(full), common_full_pooled_speed_ratio=full_ratio,
                  faster_sources_48000=sum(x > 1 for x in ratios.values()),
                  scope='Saved records; no new model runs or changes to the chosen matrix.')
    (root / 'primary_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    validate(sys.argv[1])
