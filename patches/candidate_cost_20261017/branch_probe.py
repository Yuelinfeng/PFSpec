"""Observe the existing decoder in an isolated module without changing its sampler."""
import inspect
import time
from pathlib import Path

import torch
from draft_a import pf_feature_unified as original


class RoundTimer:
    def __init__(self, enabled, torch_module, device, sync_p2=False, detail=True):
        self.enabled = enabled
        self.device = device
        self.sync_p2 = sync_p2
        self.detail = detail
        self.rounds = []
        self.setup = []
        self.current = None
        self.regions = {}

    def stamp(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return time.perf_counter(), event

    def run(self, stage, fn, *args, gpu=False, input_tokens=0,
            operation="", context_tokens=0, **kwargs):
        if not self.enabled or not self.detail:
            return fn(*args, **kwargs)
        sync = self.sync_p2 and operation == "p2_recompute"
        if sync:
            torch.cuda.synchronize(self.device)
        start_event = end_event = None
        if gpu:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        begin = time.perf_counter()
        result = fn(*args, **kwargs)
        if sync:
            torch.cuda.synchronize(self.device)
        host = time.perf_counter() - begin
        if gpu:
            end_event.record()
        row = dict(stage=stage, operation=operation, host_seconds=host,
                   input_tokens=input_tokens, context_tokens=context_tokens,
                   synchronized=sync, events=(start_event, end_event))
        (self.current["operations"] if self.current else self.setup).append(row)
        return result

    def begin_round(self, prefix, length, stats):
        if not self.enabled:
            return
        self.current = dict(prefix_body_tokens=len(prefix)-1,
                            output_start=len(prefix)-length, operations=[], notes={},
                            counts_before={k: v for k, v in stats.items() if type(v) is int},
                            stamp=self.stamp(), regions=[])

    def note(self, name, value):
        if self.enabled and self.current:
            self.current["notes"].setdefault(name, []).append(value)

    def open_region(self, name):
        if self.enabled:
            self.regions[name] = self.stamp()

    def close_region(self, name, non_eos):
        if self.enabled:
            begin, start = self.regions.pop(name)
            end_host, end_event = self.stamp()
            self.current["regions"].append(dict(name=name, non_eos=non_eos,
                host_seconds=end_host-begin, events=(start, end_event)))

    def finish_round(self, prefix, begin, accepted, stats):
        if not self.enabled:
            return
        end_host, end_event = self.stamp()
        row = self.current
        start_host, start_event = row.pop("stamp")
        counts = row.pop("counts_before")
        row["counts"] = {k: stats[k]-v for k, v in counts.items()}
        row.update(generated_tokens=len(prefix)-begin, accepted_prefix=accepted,
                   eos=prefix[-1] == original.EOS,
                   host_seconds=end_host-start_host, events=(start_event, end_event))
        if row["counts"]["feature_proposed"]:
            row["branch"] = ("p1_reject", "f1_reject", "p2_reject", "all_accept")[accepted]
        else:
            row["branch"] = "k1_accept" if accepted else (
                "fallback" if row["counts"]["position_proposed"] == 0 else "k1_reject")
        self.rounds.append(row)
        self.current = None

    def result(self):
        def convert(row):
            start, end = row.pop("events", (None, None))
            if start is not None:
                row["gpu_seconds"] = start.elapsed_time(end)/1000
            for key in ("operations", "regions"):
                for child in row.get(key, []):
                    convert(child)
        for row in self.rounds + self.setup:
            convert(row)
        return dict(rounds=self.rounds, outside_round=self.setup,
                    timing="host function duration and CUDA stream event interval; separate measures")


def build_decoder(destination):
    source = inspect.getsource(original._decode)
    def change(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, (old, source.count(old), count)
        source = source.replace(old, new)
    change("def _decode(target, adapter, ranker, ranks, prompt, budget, seed, profile):",
           "def observed_decode(target, adapter, ranker, ranks, prompt, budget, seed, profile, *, mode='pf', max_rounds=None, sync_p2=False, detail=True):")
    change("timer = _StageProfile(profile, torch, device)",
           "timer = RoundTimer(profile, torch, device, sync_p2, detail)")
    change('stats["accepted_lengths"].append(accepted)',
           'timer.finish_round(prefix, begin, accepted, stats)\n        stats["accepted_lengths"].append(accepted)')
    change("while len(prefix) - length < budget:",
           'while len(prefix) - length < budget and (max_rounds is None or stats["rounds"] < max_rounds):')
    change('stats["rounds"] += 1\n            begin = len(prefix)',
           'stats["rounds"] += 1\n            if max_rounds is not None:\n                torch.cuda.synchronize(device)\n            timer.begin_round(prefix, length, stats)\n            begin = len(prefix)')
    change("if first is not None and budget - (len(prefix) - length) >= 4:",
           "if mode == 'pf' and first is not None and budget - (len(prefix) - length) >= 4:")
    change('            else:\n                actual = point_residual(p, p2)\n                rewind(before + 3)',
           '            else:\n                timer.open_region("p2_repair")\n                actual = point_residual(p, p2)\n                rewind(before + 3)')
    change('                    logits, states = tf([actual])\n                    h_prev = hidden(states)\n                    final_f = sample(logits[-1])\n            prefix.append(actual)',
           '                    logits, states = tf([actual], operation="p2_recompute")\n                    h_prev = hidden(states)\n                    final_f = sample(logits[-1])\n                timer.close_region("p2_repair", actual != EOS)\n            prefix.append(actual)')
    change('stats["stage_profile"] = timer.result()', 'stats["round_profile"] = timer.result()')
    source = '"""Generated from the current shared K2 decoder; observation only."""\n' + source
    Path(destination).write_text(source)
    namespace = dict(vars(original), RoundTimer=RoundTimer)
    exec(compile(source, str(destination), "exec"), namespace)
    return namespace["observed_decode"]


def run_observed(fn, target, feature, ranker, ranks, prompt, budget, seed,
                 mode="pf", max_rounds=None, sync_p2=False, profile=True, detail=True):
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        return fn(target, original.MLPAdapter(feature), ranker, ranks, prompt,
                  budget, seed, profile, mode=mode, max_rounds=max_rounds,
                  sync_p2=sync_p2, detail=detail)
