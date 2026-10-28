"""Select the existing P-only or PF-K2 path before Feature generation."""
import inspect
import random
from collections import Counter
from pathlib import Path

from draft_a import pf_feature_unified as original
from gap_gate import PF_K2, choose_gap_route


class Selector:
    def __init__(self, mode, probability, seed, trace=False):
        if mode not in ('pf','p_only','gap','reverse_gap','overhead','random','alternate'):
            raise ValueError(mode)
        if not 0 <= probability <= 1:
            raise ValueError(probability)
        self.mode, self.probability = mode, probability
        self.rng = random.Random(seed + 91815541)
        self.trace = [] if trace else None
        self.counts = Counter()
        self.branches = Counter()
        self.previous_path = None
        self.route_reason = None
        self.candidate_position = None

    def begin(self, prefix, stats):
        self.start = len(prefix)
        self.feature_before = stats['feature_proposed']
        self.gap = None
        self.requested = False
        self.eligible = False
        self.route_reason = None
        self.candidate_position = None

    def choose(self, first, previous, remaining):
        self.candidate_position = None if first is None else int(first[0])
        if self.mode in ('gap', 'reverse_gap', 'overhead'):
            decision = choose_gap_route(
                None if first is None else first[0], previous, remaining
            )
            self.gap = decision.gap
            self.route_reason = decision.reason
            if decision.reason in ('unit_gap', 'larger_gap'):
                self.eligible = True
                self.counts['eligible'] += 1
                self.counts['unit_gap' if self.gap == 1 else 'other_gap'] += 1
            else:
                self.counts['fallback_' + decision.reason] += 1
            if self.mode == 'overhead':
                use_pf = first is not None and remaining >= 4
            elif self.mode == 'reverse_gap':
                use_pf = decision.reason == 'larger_gap'
            else:
                use_pf = decision.route == PF_K2
            self.requested = use_pf
            self.counts['requested_pf' if use_pf else 'requested_p_only'] += 1
            return use_pf
        if first is None or remaining < 4:
            self.route_reason = 'no_candidate' if first is None else 'short_budget'
            self.counts['boundary_or_no_candidate'] += 1
            return False
        self.gap = int(first[0]-previous)
        self.route_reason = 'forced'
        self.eligible = True
        self.counts['eligible'] += 1
        self.counts['unit_gap' if self.gap == 1 else 'other_gap'] += 1
        if self.gap <= 0:
            self.counts['nonpositive_gap'] += 1
        if self.mode == 'pf':
            use_pf = True
        elif self.mode == 'p_only':
            use_pf = False
        elif self.mode == 'gap':
            use_pf = self.gap == 1
        elif self.mode == 'alternate':
            use_pf = self.counts['eligible'] % 2 == 1
        else:
            use_pf = self.rng.random() < self.probability
        self.requested = use_pf
        self.counts['requested_pf' if use_pf else 'requested_p_only'] += 1
        return use_pf

    def finish(self, prefix, accepted, stats):
        path = 'pf' if stats['feature_proposed'] > self.feature_before else 'p_only'
        if path == 'pf':
            branch = ('p1_reject','f1_reject','p2_reject','all_accept')[accepted]
        elif stats['position_fallbacks'] > self.counts.get('last_fallbacks',0):
            branch = 'fallback'
        else:
            branch = 'p_accept' if accepted else 'p_reject'
        self.counts['last_fallbacks'] = stats['position_fallbacks']
        self.counts['executed_'+path] += 1
        if self.previous_path is not None and path != self.previous_path:
            self.counts[self.previous_path+'_to_'+path] += 1
        self.previous_path = path
        self.branches[path+'_'+branch] += 1
        if self.trace is not None:
            self.trace.append(dict(gap=self.gap, eligible=self.eligible,
                requested_pf=self.requested,path=path,branch=branch,
                reason=self.route_reason,candidate_position=self.candidate_position,
                generated_tokens=len(prefix)-self.start,
                eos=prefix[-1] == original.EOS))

    def result(self):
        return dict(mode=self.mode,probability=self.probability,
                    counts=dict(self.counts),branches=dict(self.branches),trace=self.trace)


def build_decoder(destination):
    source=inspect.getsource(original._decode)
    def change(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new)
    change('def _decode(target, adapter, ranker, ranks, prompt, budget, seed, profile):',
           "def selected_decode(target, adapter, ranker, ranks, prompt, budget, seed, profile=False, *, mode='gap', probability=1., trace=False, audit=None, max_rounds=None, decision_override=None):")
    change('    shared = _shared()',
           '    if decision_override is not None:\n        decide = decision_override\n    shared = _shared()')
    change('timer = _StageProfile(profile, torch, device)',
           'timer = _StageProfile(profile, torch, device)\n    selector = Selector(mode, probability, seed, trace)')
    change('    def finish_round(begin, accepted):\n        stats["accepted_lengths"].append(accepted)',
           '    def finish_round(begin, accepted):\n        selector.finish(prefix, accepted, stats)\n        if audit is not None:\n            audit(target, cache, capture, prefix, h_prev, online, stats)\n        stats["accepted_lengths"].append(accepted)')
    change('while len(prefix) - length < budget:',
           'while len(prefix) - length < budget and (max_rounds is None or stats["rounds"] < max_rounds):')
    change('stats["rounds"] += 1\n            begin = len(prefix)',
           'stats["rounds"] += 1\n            selector.begin(prefix, stats)\n            begin = len(prefix)')
    change('if first is not None and budget - (len(prefix) - length) >= 4:',
           'if selector.choose(first, previous, budget - (len(prefix) - length)):')
    change('    return shared._add_rates(stats), output',
           '    stats["selection"] = selector.result()\n    return shared._add_rates(stats), output')
    Path(destination).write_text(source)
    namespace=dict(vars(original),Selector=Selector)
    exec(compile(source,str(destination),'exec'),namespace)
    return namespace['selected_decode']


def run(fn,target,feature,ranker,ranks,prompt,budget,seed,**kwargs):
    import torch
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        return fn(target,original.MLPAdapter(feature),ranker,ranks,prompt,budget,seed,**kwargs)
