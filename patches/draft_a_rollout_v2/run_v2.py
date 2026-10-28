"""Run the authorized stages in order, retaining a readable process status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from draft_a.corpus_v2 import ROOT, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--generation-pid",type=int,required=True)
    args=ap.parse_args()
    root=args.root
    status=root/"pipeline_status.json"
    write_json(status,dict(stage="generation",state="running",pid=os.getpid(),generation_pid=args.generation_pid))
    try:
        while True:
            process=Path(f"/proc/{args.generation_pid}/cmdline")
            try:
                command=process.read_bytes()
            except FileNotFoundError:
                break
            if b"draft_a.corpus_v2" not in command:
                break
            time.sleep(30)
        completed=len(list((root/"corpus").glob("*.pt")))
        if completed!=500:
            raise RuntimeError(f"Generation stopped with {completed}/500 files")
        stages=[
            ("analysis",["draft_a.analyze_v2"]),
            ("decoder_validation",["draft_a.benchmark_v2","--self-test","--checkpoint","draft_a/checkpoints/ce_43_v2/best.pt"]),
            ("training",["draft_a.train_v2","--steps","5000","--eval-every","250","--lr","0.0003"]),
            ("validation_overlap",["draft_a.evaluate_v2","--split","val"]),
            ("test_overlap",["draft_a.evaluate_v2","--split","test"]),
            ("benchmark_previous",["draft_a.benchmark_v2","--checkpoint","draft_a/checkpoints/ce_43_v2/best.pt","--output-tag","previous"]),
            ("benchmark",["draft_a.benchmark_v2"]),
        ]
        for stage,arguments in stages:
            write_json(status,dict(stage=stage,state="running",pid=os.getpid(),started_epoch=time.time()))
            command=[sys.executable,"-u","-m",*arguments,"--root",str(root)]
            print(json.dumps(dict(stage=stage,command=command)),flush=True)
            with (root/f"{stage}.log").open("x") as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode:
                raise RuntimeError(f"Stage {stage} exited with code {result.returncode}; see {stage}.log")
        overlap=json.loads((root/"overlap_test.json").read_text())["summary"]["test"]
        previous=json.loads((root/"benchmark_previous.json").read_text())["summary"]
        current=json.loads((root/"benchmark_v2.json").read_text())["summary"]
        comparisons=[]
        for context in ("0","1024"):
            for k in ("2","4","8","16"):
                a,b=previous[context][k],current[context][k]
                comparisons.append(dict(context=int(context),k=int(k),
                    previous_acceptance=a["acceptance"],v2_acceptance=b["acceptance"],
                    previous_speedup=a["speedup"],v2_speedup=b["speedup"],
                    accepted_length=b["mean_accepted_length"]))
        write_json(root/"result_summary.json",dict(test_overlap=overlap,decoding=comparisons,
            scope="Three seeds per 256-token window at two context lengths; one scene for real-data comparison",
            real_kl_started=False))
        write_json(status,dict(stage="complete",state="complete",pid=os.getpid(),finished_epoch=time.time(),
            next="Review actual acceptance and timing before deciding on Real VFront KL adaptation"))
    except Exception as error:
        previous=json.loads(status.read_text())
        write_json(status,dict(previous,state="failed",error=str(error),failed_epoch=time.time()))
        traceback.print_exc()
        raise


if __name__=="__main__":
    main()
