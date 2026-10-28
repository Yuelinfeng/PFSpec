"""Summarize measured operation costs without claiming a full-decode optimum."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(path):
    contexts = []
    bins = defaultdict(lambda: defaultdict(list))
    paths = [path] if isinstance(path, Path) else path
    for line in (line for item in paths for line in item.open()):
        event = json.loads(line)
        labels = event["labels"]
        probe = labels.get("complete_action_operations")
        if not probe:
            continue
        if not probe["cache_restored"] or not probe["runtime_bookkeeping_restored"]:
            raise AssertionError("Probe restoration failed")
        ctx = event["round_context"]
        bucket = ctx["graph_state"]["1"]["read_bucket"]
        actions = {a["action_id"]: a["median_wall_ms"] for a in probe["actions"]}
        pmain = float(labels["p_main"])
        peos = float(labels["p_eos"])
        candidates = []
        for token in probe["input"]["backup_candidates"]:
            mass = float(labels["target_mass_by_token"][str(token)])
            missed = max(0.0, 1.0-pmain-peos-mass)
            gains = []
            zero_eos = actions["k0_eos_after_p1_rejection"]
            one_eos = actions[f"k1_eos_after_p1_rejection_backup_{token}"]
            eos_gain = peos*(zero_eos-one_eos)
            for length in (2, 3, 4):
                common = actions[f"k0_common_path_{length}"] - actions[f"k1_common_path_{length}_backup_{token}"]
                hit = actions[f"k0_recover_to_backup_{token}"] - actions[f"k1_backup_hit_{token}"]
                miss = actions["k0_miss_recover"] - actions[f"k1_miss_recover_{token}"]
                gains.append(pmain*common + mass*hit + missed*miss + eos_gain)
                bins[bucket]["common_extra_ms"].append(-common)
                bins[bucket]["hit_saved_ms"].append(hit)
                bins[bucket]["miss_extra_ms"].append(-miss)
            candidates.append(dict(token_id=token, probability=mass,
                                   operation_gain_lower_ms=min(gains) if gains else None,
                                   operation_gain_upper_ms=max(gains) if gains else None))
        contexts.append(dict(source=event["source_id"], cache_pos=ctx["cache_pos"],
                             read_bucket=bucket, p_main=pmain, p_eos=peos,
                             candidates=candidates, actions=actions,
                             prewarm=probe["prewarm"]))
    def stats(values):
        return dict(n=len(values), median=statistics.median(values), mean=statistics.mean(values),
                    minimum=min(values), maximum=max(values))
    output = dict(context_count=len(contexts), contexts=contexts,
                  costs_by_bucket={str(k): {n:stats(v) for n,v in data.items()} for k,data in bins.items()},
                  scope="Forced verify/commit/recovery operations; excludes policy, sampling and CPU decisions.",
                  limitations=["Main-accepted Feature outcomes use a range over commit lengths 2..4.",
                    "Noncandidate recovery uses one explicitly forced legal token, assuming token-independent operation cost.",
                    "Repeated synchronized times are noisy and do not prove complete decoding benefit."])
    return output


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--rounds",type=Path,nargs="+",required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); result=summarize(a.rounds)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(context_count=result["context_count"], costs_by_bucket=result["costs_by_bucket"])))
