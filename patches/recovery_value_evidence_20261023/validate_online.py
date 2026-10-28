"""Compare live selected IDs, applied tree nodes and saved CPU decisions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from evidence_policy import choose


def read(path):
    return [json.loads(line) for line in path.open()]


def validate(trace, formal, metadata):
    online=read(trace/'records.jsonl')
    timed={(r['source'],r['prefix'],r['seed'],r['method']):r for r in read(formal/'records.jsonl')}
    events=defaultdict(list)
    for event in read(trace/'rounds.jsonl'):
        events[(event['source_id'],event['prefix_len'],event['seed'],event['method'])].append(event)
    contexts=0;applied=0;swaps=0;selected_mass=0.0;ranker_mass=0.0
    semantic=['accepted','proposed','verified','position_accepted','feature_accepted','eos_accepted',
              'accepted_lengths','round_generated_lengths','target_calls','target_input_tokens',
              'recovery_calls','position_rescue_hits','position_backup_nodes','target_tree_nodes']
    for row in online:
        key=(row['source'],row['prefix'],row['seed'],row['method'])
        reference=timed[key]
        assert row['output_tokens']==reference['output_tokens'] and row['eos']==reference['eos']
        assert all(row['stats'][k]==reference['stats'][k] for k in semantic),key
        by_round={r['round']:r for r in row['stats'].get('position_backup_diagnostics',[])}
        selected_count=0;expected_calls=0
        for event in events[key]:
            context=event['round_context'];method=row['method']
            expected=[] if method=='same0' else choose(metadata,context,method)
            actual=event['selected_backup_ids']
            assert expected==actual,(key,context['round_index'],expected,actual)
            candidates=[c for c in context['candidates'] if c['token_id']!=context.get('main_position')]
            if method!='same0' and candidates:
                expected_calls+=1
            contexts+=1;selected_count+=len(actual)
            if actual:
                diagnostic=by_round[context['round_index']]
                assert diagnostic['candidate_ids']==[context['main_position'],*actual]
                assert actual==[r for r in diagnostic['candidate_ids'][1:]]
                applied+=1
            if actual and method=='operation_value':
                first=candidates[0]['token_id'];token=actual[0]
                swaps+=token!=first
                mass=event['labels']['target_mass_by_token']
                selected_mass+=mass[str(token)];ranker_mass+=mass[str(first)]
        assert selected_count==row['stats']['position_backup_nodes']
        if row['method']!='same0':
            assert selected_count==len(by_round)
            assert expected_calls==row['stats']['backup_selector_calls']
    return dict(status='passed',requests=len(online),cpu_live_decision_comparisons=contexts,
                applied_backup_id_comparisons=applied,formal_output_and_counter_pairs=len(online),
                value_candidate_swaps=swaps,value_selected_target_mass_sum=selected_mass,
                same_count_ranker_target_mass_sum=ranker_mass,
                scope='Instrumented correctness only; diagnostic timing excluded from performance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--formal',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=validate(a.trace,a.formal,json.loads(a.model.read_text()))
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
