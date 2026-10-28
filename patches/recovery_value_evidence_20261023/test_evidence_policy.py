"""Verify live-context equivalence and action legality on saved development inputs."""
import argparse
import copy
import json
from pathlib import Path
from evidence_policy import Selector, choose


def main():
    p=argparse.ArgumentParser();p.add_argument("--rounds",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();model=json.loads(a.model.read_text())
    methods=("operation_value","probability_rule","score_rule","value_ranker","acceptance_rule")
    names=("prefix_len","remaining_budget","cache_pos","previous_position_rank","main_position",
           "main_position_rank","main_score","candidates","position_vocab_size","eos_id","graph_state","read_bucket")
    compared=0; nonempty=0
    for line in a.rounds.open():
        event=json.loads(line);full=event["round_context"]
        light={key:full[key] for key in names if key in full}
        for method in methods:
            expected=choose(model,full,method);actual=choose(model,light,method)
            assert actual==expected
            assert len(actual)<=1 and len(actual)==len(set(actual))
            legal={int(row["token_id"]) for row in light["candidates"]
                   if int(row["token_id"])!=light.get("main_position")}
            assert set(actual)<=legal
            contaminated=dict(light,labels={"p_main":999,"target_mass_by_token":{}},prefix_tokens=[-1])
            assert choose(model,contaminated,method)==actual
            absent=copy.deepcopy(light)
            if "1" in absent.get("graph_state",{}):
                absent["graph_state"]["1"]["candidate_available"]=False
            assert choose(model,absent,method)==[]
            compared+=1;nonempty+=bool(actual)
    try:
        Selector(model,"unrecognized")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown method accepted")
    result=dict(context_method_comparisons=compared,nonempty_actions=nonempty,
        full_light_actions_equal=True,labels_unused=True,legal_subsets=True,unavailable_shapes_empty=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))


if __name__=="__main__":
    main()
