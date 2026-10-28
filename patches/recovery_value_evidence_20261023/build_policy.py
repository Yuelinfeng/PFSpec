"""Combine development probability fit and measured warm operation costs."""
import argparse
import json
from pathlib import Path
import statistics
from collections import defaultdict
from probability_model import predict


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--operations",type=Path,required=True)
    p.add_argument("--rounds",type=Path,required=True)
    p.add_argument("--split-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    operations=json.loads(a.operations.read_text())
    costs={bucket:{name:float(stats["median"]) for name,stats in values.items()}
           for bucket,values in operations["costs_by_bucket"].items()}
    global_cost={name:statistics.median(values[name] for values in costs.values())
                 for name in ("hit_saved_ms","common_extra_ms","miss_extra_ms")}
    probability=json.loads(a.model.read_text())
    train_sources=json.loads(a.split_report.read_text())["split"]["train_sources"]
    observations=[]
    for line in a.rounds.open():
        row=json.loads(line)
        if row["source_id"] not in train_sources or row.get("diagnostic_only"):
            continue
        context=row["round_context"];main=context.get("main_position")
        candidates=[c for c in context["candidates"] if c["token_id"]!=main]
        if not candidates:
            continue
        first=candidates[0]
        mass=row["labels"].get("target_mass_by_token",{}).get(str(first["token_id"]))
        main_row=next((c for c in context["candidates"] if c["token_id"]==main),None)
        if mass is None or main_row is None:
            continue
        observations.append(dict(source=row["source_id"],score=float(first.get("ranker_score") or 0),
            rejection=1-predict(probability,context,main_row),
            gain=float(mass)*global_cost["hit_saved_ms"]-(1-float(mass))*global_cost["common_extra_ms"]))
    choices={}
    for kind,field in [("score","score"),("acceptance","rejection")]:
        best=None
        for threshold in [i*.05 for i in range(21)]+[1.01]:
            values=defaultdict(list)
            for row in observations:
                values[row["source"]].append(row["gain"] if row[field]>=threshold else 0.0)
            score=statistics.mean(statistics.mean(v) for v in values.values())
            if best is None or score>best[0]:
                best=(score,threshold)
        choices[kind]=dict(training_source_equal_operation_proxy_ms=best[0],threshold=best[1])
    result=dict(probability=probability,costs_by_bucket=costs,
                global_cost=global_cost,score_threshold=choices["score"]["threshold"],
                acceptance_threshold=choices["acceptance"]["threshold"],control_threshold_fit=choices,
                scope="Development heuristic using warmed operation costs; graph creation remains in measured wall time.",
                limitations=["First graph creation is not optimized by this selector.",
                             "EOS probability is not estimated; remaining EOS mass is charged at miss-path cost.",
                             "Main Feature outcomes use pooled commit-length medians."])
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(costs_by_bucket=costs,global_cost=global_cost,control_threshold_fit=choices)))


if __name__=="__main__":
    main()
