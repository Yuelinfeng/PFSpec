"""Independently recount paired single-round observations using only JSON."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = [json.loads(line) for line in (args.run/'records.jsonl').read_text().splitlines()]
    plan = json.loads((args.run/'plan.json').read_text())
    failures = []
    groups = defaultdict(dict)
    repetitions = defaultdict(list)
    counts = Counter()
    paths = Counter()
    for i, row in enumerate(rows):
        if row['scope'] != 'single_round':
            continue
        key = (row['context_id'], row['seed'], row['repeat_index'])
        if row['action_id'] in groups[key]:
            failures.append([i, 'duplicate action'])
        groups[key][row['action_id']] = row
        repetitions[(row['context_id'],row['seed'],row['action_id'])].append(row)
        checks = dict(
            token_count=row['R']==len(row['actual_tokens']),
            sequence=row['sequence_validated'] is True,
            round_time=row['round_wall_ms'] is not None and row['round_wall_ms']>0,
            prefill_time=row['prefill_ms'] is not None and row['prefill_ms']>0,
            eos=row['eos']==bool(row['actual_tokens'] and row['actual_tokens'][-1]==12096),
            recovery=row['recovery_calls']==row['recovery_target_calls'],
        )
        if row['node_count']>0:
            expected = 2+2*int(row['accepted_p1'] is True and row['accepted_feature'] is True)-int(row['eos'])
            checks['pair_progress'] = row['R']==expected
            checks['tree_calls'] = row['target_calls']==1+row['recovery_calls']
            checks['input_tokens'] = row['target_input_tokens']==row['node_count']+row['recovery_calls']
            counts['tree_rounds']+=1
        else:
            checks['one_pair_progress'] = row['R']==2-int(row['eos'])
            counts['one_pair_rounds']+=1
        for name, ok in checks.items():
            if not ok:
                failures.append([i,name])
        paths[(row['action_id'],str(row['accepted_p1']),str(row['accepted_feature']),
               str(row['accepted_p2']),str(row['eos_stage']))]+=1
    for key, actions in groups.items():
        if set(actions)!=set(plan['actions']):
            failures.append([list(key),'incomplete paired actions'])
    for key, local in repetitions.items():
        if len(local)!=plan['single_round_repeats']:
            failures.append([list(key),'incomplete timing repetitions'])
        if any((r['actual_tokens'],r['accepted_p1'],r['accepted_feature'],r['accepted_p2']) !=
               (local[0]['actual_tokens'],local[0]['accepted_p1'],local[0]['accepted_feature'],local[0]['accepted_p2'])
               for r in local):
            failures.append([list(key),'repetition output differs'])
    expected=plan['expected_single_round_rows']+plan['expected_full_request_rows']
    if len(rows)!=expected:
        failures.append(['total','incomplete row count'])
    report=dict(passed=not failures,rows=len(rows),expected_rows=expected,
                sources=len({r['source_id'] for r in rows}),
                contexts=len({r['context_id'] for r in rows}),paired_groups=len(groups),
                repeated_seed_actions=len(repetitions),counts=dict(counts),
                paths=[dict(action=k[0],p1=k[1],feature=k[2],p2=k[3],eos_stage=k[4],rows=v)
                       for k,v in sorted(paths.items())],failures=failures)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('paths','failures')}))
    if failures:
        raise AssertionError(f'{len(failures)} observation inconsistencies')


if __name__=='__main__':
    main()
