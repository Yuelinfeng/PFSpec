"""Assemble the G0/G1 audit outputs into the requested JSON summary."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'docs/project/artifacts/gap_selection_20261017'


def read(name):
    return json.loads((ARTIFACTS / name).read_text())


def main():
    static = read('g0_static_audit.json')
    decoder = read('g1_decoder_checks.json')
    assets = read('g0_asset_inventory.json')

    assert static['gap_labels']['candidate_gaps_checked'] == 5122
    assert static['gap_labels']['first_100_matching'] == 100
    assert static['gap_labels']['negative_or_zero_gaps'] == 0
    assert static['route_boundaries']['status'] == 'passed'
    assert static['exact_distribution']['status'] == 'passed'
    assert decoder['status'] == 'complete'
    assert decoder['decoder_checks'] == 'passed'

    endpoints = decoder['endpoints']
    endpoint_rows = [row for row in endpoints if row['mode'] in ('pf', 'p_only')]
    audited = [row for row in endpoints if 'audited_rounds' in row]
    assert len(endpoint_rows) == 16
    assert all(row['output_equal'] and row['counts_equal'] for row in endpoint_rows)
    assert len(audited) == 4
    assert all(row['audited_rounds'] > 0 for row in audited)
    max_next_token_tv = max(row['max_next_token_tv'] for row in audited)
    assert max_next_token_tv < 0.01
    assert all(row['passed'] for row in decoder['forced_branches'])
    assert all(row['passed'] for row in decoder['eos'])
    transitions = decoder['transitions']
    assert transitions.get('pf_to_p_only', 0) > 0
    assert transitions.get('p_only_to_pf', 0) > 0

    output = {
        'status': 'passed',
        'g0': {
            'status': 'passed',
            'git_head': assets['git_head'],
            'asset_inventory': 'g0_asset_inventory.json',
            'ranker_proposal_is_deterministic': True,
            'gap_labels': {
                'total_recomputed': static['gap_labels']['candidate_gaps_checked'],
                'first_100_match': static['gap_labels']['first_100_matching'],
                'gap_one': static['gap_labels']['gap_counts']['1'],
                'gap_above_one': sum(
                    value for key, value in static['gap_labels']['gap_counts'].items()
                    if int(key) > 1
                ),
                'nonpositive': static['gap_labels']['negative_or_zero_gaps'],
                'details': 'g0_static_audit.json',
            },
            'route_boundaries_tested': len(static['route_boundaries']['cases']),
        },
        'exact_toy_sequence_enumeration': static['exact_distribution'],
        'decoder_regression': {
            'status': 'passed',
            'tested_prefixes': decoder['contexts_tested'],
            'forced_pf_and_p_only_endpoint_runs': len(endpoint_rows),
            'endpoint_outputs_and_counts_match': True,
            'audited_runs': len(audited),
            'forced_pf_branches': decoder['forced_branches'],
            'eos_cases': len(decoder['eos']),
            'switch_counts': transitions,
            'branch_counts': decoder['branches'],
            'maximum_next_token_total_variation': max_next_token_tv,
            'next_token_total_variation_limit': 0.01,
            'cache_diagnostics': {
                'maximum_k_cache_absolute_difference': max(
                    row['max_cache_abs_difference']['k_cache'] for row in audited
                ),
                'maximum_v_cache_absolute_difference': max(
                    row['max_cache_abs_difference']['v_cache'] for row in audited
                ),
                'position_replay_state_checks': 'passed for every audited round',
            },
            'full_prefill_hidden_diagnostics': {
                'maximum_absolute_difference': max(
                    row['max_full_prefill_hidden_abs_difference'] for row in audited
                ),
                'maximum_relative_l2': max(
                    row['max_full_prefill_hidden_relative_l2'] for row in audited
                ),
                'minimum_cosine_similarity': min(
                    row['min_full_prefill_hidden_cosine'] for row in audited
                ),
                'interpretation': 'Reported as a comparison with full prefill; this value is not used as a sampling-law pass criterion.',
            },
            'raw_results': 'g1_decoder_checks.json',
        },
        'sampling_scope': {
            'method': 'The route depends only on the committed prefix and deterministic P1 proposal. Existing point-proposal acceptance, Feature residual sampling, Target bonus sampling, and cache updates are reused.',
            'evidence': 'Exact two-pair finite-support enumeration plus real-decoder branch, EOS, endpoint, ranker-state, and Target next-token cache audits.',
            'limit': 'The complete real GaussianGPT sequence distribution was not exhaustively enumerated or Monte Carlo-tested across the full vocabulary.',
        },
    }
    (ARTIFACTS / 'G1_correctness.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + '\n'
    )
    print(json.dumps({
        'status': output['status'],
        'endpoints': len(endpoint_rows),
        'audited_runs': len(audited),
        'max_next_token_tv': max_next_token_tv,
        'output': str(ARTIFACTS / 'G1_correctness.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
