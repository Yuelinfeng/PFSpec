"""统一 K=2 PF 解码，沿用 pf_matched_eval / pf_ranker_eval 的采样策略。

服务器调用（模型和 ranker 由调用方加载）：
    stats, output = decode(target, MLPAdapter(feature, bf16=True), ranker,
                           ranks, prompt, budget=512, seed=24685)
    stats, output = decode(target, GPTAdapter(core), ranker, ranks,
                           prompt, budget=512, seed=24685, profile=True)

prompt 是真实 [BOS, P, F, ...]；prefix0/64/256 分别传入 1/65/257
个 token。h_prev 始终取最后一个已提交 P 的 Target hidden，末尾 F 待处理。
GPT 使用独立 FP32 KV，只计算 F head，输入仍包含完整 P/F 历史。
prepare 每次重新创建缓存；close 释放本次缓存，不改变模型参数。

stage_profile 各阶段分别给出 host_seconds 和 cuda_events.seconds，二者
分别解读，不能相加。CUDA event 只覆盖 GPU 相关操作，ranker 仅记录 host。
cache_allocate 和真实前缀 prefill 均计入总耗时。GPT 提案的历史与 P 同次
计算时，feature_forward.operations.context_and_forward 记录输入数量。
profile 不使用随机数；调用方用相同 seed 比较逐 token 输出及非计时统计。
round_generated_lengths 不含 prefix0 的首次 Target P/F，只对应 rounds；
Target bonus F 和拒绝后生成的 F 均计入各轮实际产出。

本模块不加载权重、不训练、不访问网络；导入时不触发 GPU 操作。
"""

from importlib import import_module
import time


P_VOCAB = 8000
F_VOCAB = 4096
EOS = P_VOCAB + F_VOCAB
_STAGES = (
    "cache_allocate", "feature_forward", "feature_context",
    "cache_maintenance", "ranker", "target_verify", "target_context", "sampling",
)


def _shared():
    """复用服务器工程的 ranker、policy、cache 与 matched row 工具。"""
    return import_module("draft_a.pf_ranker_eval")


class _StageProfile:
    def __init__(self, enabled, torch, device):
        self.enabled = enabled
        self.torch = torch
        self.device = device
        self.rows = {
            name: dict(calls=0, input_tokens=0, host_seconds=0.0,
                       cuda_events=dict(count=0, seconds=0.0), operations={})
            for name in _STAGES
        } if enabled else {}
        self.events = []

    def run(self, stage, fn, *args, gpu=False, input_tokens=0,
            operation="", context_tokens=0, **kwargs):
        if not self.enabled:
            return fn(*args, **kwargs)
        row = self.rows[stage]
        row["calls"] += 1
        row["input_tokens"] += input_tokens
        if operation:
            entry = row["operations"].setdefault(
                operation, dict(calls=0, input_tokens=0, context_tokens=0))
            entry["calls"] += 1
            entry["input_tokens"] += input_tokens
            entry["context_tokens"] += context_tokens
        begin = end = None
        if gpu:
            begin = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            begin.record(self.torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            row["host_seconds"] += time.perf_counter() - started
            if begin is not None:
                end.record(self.torch.cuda.current_stream(self.device))
                self.events.append((stage, begin, end))

    def result(self):
        # decode 已等待 GPU 完成；这里不插入逐阶段同步。
        for stage, begin, end in self.events:
            row = self.rows[stage]["cuda_events"]
            row["count"] += 1
            row["seconds"] += begin.elapsed_time(end) / 1000.0
        self.events.clear()
        return self.rows


class _Adapter:
    def __init__(self):
        self._profile = None
        self.forward_calls = 0
        self.input_tokens = 0
        self._prepared = False

    def _run(self, stage, fn, *args, **kwargs):
        if self._profile is None:
            for key in ("gpu", "input_tokens", "operation", "context_tokens"):
                kwargs.pop(key, None)
            return fn(*args, **kwargs)
        return self._profile.run(stage, fn, *args, **kwargs)

    def _reset(self):
        self.forward_calls = 0
        self.input_tokens = 0
        self._prepared = False


class MLPAdapter(_Adapter):
    """state MLP：Target prev hidden、末尾 F 和提案 P -> 全量 4096 q。"""

    def __init__(self, feature, bf16=False):
        super().__init__()
        self.feature = feature
        self.bf16 = bf16
        self.variant = "old_mlp_bf16" if bf16 else "old_mlp_fp32"
        self.feature_precision = "BF16 autocast" if bf16 else "FP32"

    def prepare(self, prompt, capacity):
        self.close()
        self._reset()
        self._prepared = True

    def propose(self, prefix, p1, h_prev, rng):
        import torch
        from collect_feature_teacher_from_s3 import policy_batch

        shared = _shared()
        shared._require(self._prepared and h_prev is not None,
                        "MLP needs prepared state and Target prev hidden")
        self.forward_calls += 1
        self.input_tokens += 1

        def forward():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.bf16):
                _, logits = self.feature(
                    h_prev.unsqueeze(0),
                    torch.tensor([prefix[-1] - P_VOCAB], device=h_prev.device),
                    torch.tensor([p1], device=h_prev.device),
                )
            return logits

        logits = self._run("feature_forward", forward, gpu=True,
                           input_tokens=1, operation="state_mlp")
        q = self._run("sampling", lambda: policy_batch(logits.float(), .9, .9)[0],
                      gpu=True, operation="feature_policy").float()
        shared._require(q.ndim == 1 and q.numel() == F_VOCAB,
                        "MLP Feature q must contain 4096 probabilities")
        return q

    def rewind(self, keep):
        # 此 adapter 没有 KV；h_prev 由统一循环从已提交 Target P 获取。
        pass

    def close(self):
        self._prepared = False


class GPTAdapter(_Adapter):
    """GPT：独立 FP32 KV，保留 P/F 输入，仅使用 Feature head。"""

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.cache = None
        self.variant = "current_gpt_fp32"
        self.feature_precision = "FP32"

    def _forward(self, tokens):
        import torch

        self.forward_calls += 1
        self.input_tokens += len(tokens)
        with torch.autocast(device_type="cuda", enabled=False):
            return self.core(
                torch.tensor([tokens], device="cuda"), kv_cache=self.cache,
                logit_types=["feature"] * len(tokens),
            )[0]

    def prepare(self, prompt, capacity):
        self.close()
        self._reset()
        shared = _shared()
        self.cache = self._run("cache_allocate", shared.cache_for, self.core,
                               capacity, gpu=True, operation="feature_kv")
        self._prepared = True
        if len(prompt) > 1:
            self._run("feature_context", self._forward, list(prompt[:-1]),
                      gpu=True, input_tokens=len(prompt) - 1, operation="prefill",
                      context_tokens=len(prompt) - 1)

    def propose(self, prefix, p1, h_prev, rng):
        shared = _shared()
        shared._require(self._prepared and self.cache is not None,
                        "GPT needs a cache created by prepare")
        trial = list(prefix) + [p1]
        start = int(self.cache.get_pos())
        shared._require(0 <= start < len(trial), "GPT KV position differs")
        tokens = trial[start:]
        context_tokens = len(tokens) - 1
        logits = self._run(
            "feature_forward", self._forward, tokens, gpu=True,
            input_tokens=len(tokens), context_tokens=context_tokens,
            operation="context_and_forward" if context_tokens else "forward",
        )
        q = self._run("sampling", shared.policy, logits[-1].float(),
                      gpu=True, operation="feature_policy")[P_VOCAB:EOS].float()
        shared._require(q.ndim == 1 and q.numel() == F_VOCAB,
                        "GPT Feature q must contain 4096 probabilities")
        return q

    def rewind(self, keep):
        if self.cache is None:
            return
        from bench_gaussiangpt_pf_block2_cow import truncate

        old = int(self.cache.get_pos())
        if old > keep:
            self._run("cache_maintenance", truncate, self.cache, keep, old,
                      gpu=True, operation="feature_rewind")

    def close(self):
        def release():
            self.cache = None
            self._prepared = False

        if self.cache is not None:
            self._run("cache_maintenance", release, operation="feature_close")
        else:
            self._prepared = False


def decode(target, adapter, ranker, ranks, prompt, budget, seed, profile=False):
    """返回 (matched-compatible stats, output)，profile 默认关闭。"""
    import torch

    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        return _decode(target, adapter, ranker, ranks, prompt, budget, seed, profile)


def _decode(target, adapter, ranker, ranks, prompt, budget, seed, profile):
    import torch
    from gpt_position_spec_decode_smoke import choose_position
    from bench_gaussiangpt_pf_block2_cow import (
        CaptureLastBlock, fork_position_state, truncate, residual_point,
        residual_sample, decide,
    )

    shared = _shared()
    require, ratio = shared._require, shared._ratio
    require(budget >= 2 and budget % 2 == 0, "Need an even token budget")
    prompt = list(prompt)
    require(bool(prompt), "Prompt must contain BOS")
    shared.validate_sequence(prompt, [], ranks)
    require((len(prompt) - 1) % 2 == 0, "Prompt must end at a complete pair")
    require(EOS not in prompt[1:], "Prompt must not contain a terminal EOS")
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    timer = _StageProfile(profile, torch, device)
    length = len(prompt)
    capacity = length + budget + 6
    stats = dict(target_calls=0, draft_calls=0, ranker_calls=0,
                 target_input_tokens=0, draft_input_tokens=0, rounds=0,
                 proposed=0, verified=0, accepted=0, accepted_lengths=[],
                 round_generated_lengths=[], position_fallbacks=0)
    for kind in shared.KINDS:
        for field in ("proposed", "verified", "accepted"):
            stats[kind + "_" + field] = 0
    cache = capture = None
    adapter._profile = timer

    def tf(tokens, stage="target_context", operation="forward"):
        stats["target_calls"] += 1
        stats["target_input_tokens"] += len(tokens)
        return timer.run(
            stage, capture.run, target, cache, tokens,
            shared.typed_kinds(int(cache.get_pos()), len(tokens)), device,
            gpu=True, input_tokens=len(tokens), operation=operation,
        )

    def hidden(states, index=-1):
        return timer.run("target_context", lambda: states[index].detach().clone(),
                         gpu=True, operation="hidden_snapshot")

    def sample(logits):
        return timer.run("sampling", lambda: shared.draw(shared.policy(logits), rng),
                         gpu=True, operation="target_draw")

    def point_accept(p, token):
        return timer.run("sampling", lambda: decide(float(p[token]), rng, device),
                         gpu=True, operation="position_accept")

    def feature_accept(p, q, token):
        return timer.run(
            "sampling",
            lambda: decide(min(1.0, float(p[token] / q[token].clamp_min(1e-30))),
                           rng, device), gpu=True, operation="feature_accept",
        )

    def probabilities(logits):
        return timer.run("sampling", shared.policy, logits, gpu=True,
                         operation="target_policy")

    def point_residual(p, token):
        return timer.run("sampling", residual_point, p, token, rng,
                         gpu=True, operation="position_residual")

    def rewind(keep):
        old = int(cache.get_pos())
        if old > keep:
            timer.run("cache_maintenance", truncate, cache, keep, old,
                      gpu=True, operation="target_rewind")
        # adapter 自己记录缓存操作，避免与 forward 的时间重叠。
        adapter.rewind(keep)

    def count(kind, field, n=1):
        stats[field] += n
        stats[kind + "_" + field] += n

    def rank_call(fn, *args, operation):
        return timer.run("ranker", fn, *args, operation=operation)

    def finish_round(begin, accepted):
        stats["accepted_lengths"].append(accepted)
        stats["round_generated_lengths"].append(len(prefix) - begin)

    prefix = list(prompt)
    h_prev = online = None
    try:
        cache = timer.run("cache_allocate", shared.cache_for, target, capacity,
                          gpu=True, operation="target_kv")
        capture = timer.run("cache_maintenance", CaptureLastBlock, target,
                            operation="capture_open")
        adapter.prepare(prompt, capacity)
        if length > 1:
            _, states = tf(prompt[:-1], operation="prefill")
            h_prev = hidden(states)
            online = rank_call(shared.position_state, prefix, ranks,
                               operation="prefix_replay")

        while len(prefix) - length < budget:
            require(int(cache.get_pos()) == len(prefix) - 1,
                    "Target KV position differs")
            if online is None:
                logits, _ = tf(prefix[-1:])
                p0 = sample(logits[-1])
                prefix.append(p0)
                if p0 == EOS:
                    break
                logits, states = tf([p0])
                prefix.append(sample(logits[-1]))
                h_prev = hidden(states)
                online = rank_call(shared.position_state, prefix, ranks,
                                   operation="prefix_replay")
                continue

            stats["rounds"] += 1
            begin = len(prefix)
            before = int(cache.get_pos())
            previous, pending_f = prefix[-2], prefix[-1]
            stats["ranker_calls"] += 1
            first = rank_call(choose_position, ranker, online, previous,
                              operation="position_proposal")
            second = shadow = None
            if first is not None and budget - (len(prefix) - length) >= 4:
                shadow = rank_call(fork_position_state, online, operation="cow_fork")
                rank_call(shadow.finalize, first[0], operation="cow_finalize")
                stats["ranker_calls"] += 1
                second = rank_call(choose_position, ranker, shadow, first[0],
                                   operation="position_proposal")

            if second is None:
                accepted = 0
                if first is None:
                    stats["position_fallbacks"] += 1
                    logits, _ = tf([pending_f])
                    actual = sample(logits[-1])
                else:
                    p1 = first[0]
                    count("position", "proposed")
                    logits, states = tf([pending_f, p1], "target_verify")
                    p = probabilities(logits[0])
                    count("position", "verified")
                    if point_accept(p, p1):
                        count("position", "accepted")
                        accepted = 1
                        actual = p1
                        h_prev = hidden(states, 1)
                        final_f = sample(logits[1])
                    else:
                        actual = point_residual(p, p1)
                        rewind(before + 1)
                prefix.append(actual)
                if actual == EOS:
                    rank_call(online.drop_eos, operation="drop_pending")
                    finish_round(begin, accepted)
                    break
                if not accepted:
                    logits, states = tf([actual])
                    h_prev = hidden(states)
                    final_f = sample(logits[-1])
                prefix.append(final_f)
                rank_call(online.finalize, actual, operation="finalize")
                finish_round(begin, accepted)
                continue

            p1, p2 = first[0], second[0]
            count("position", "proposed", 2)
            count("feature", "proposed")
            q = adapter.propose(prefix, p1, h_prev, rng)
            require(q.dtype == torch.float32 and q.ndim == 1 and q.numel() == F_VOCAB,
                    "Feature q must be a full FP32 distribution of size 4096")
            f_local = timer.run("sampling", shared.draw, q, rng, gpu=True,
                                operation="feature_draw")
            f1 = f_local + P_VOCAB
            logits, states = tf([pending_f, p1, f1, p2], "target_verify")
            count("position", "verified")
            p = probabilities(logits[0])
            if not point_accept(p, p1):
                actual = point_residual(p, p1)
                rewind(before + 1)
                prefix.append(actual)
                if actual == EOS:
                    rank_call(online.drop_eos, operation="drop_pending")
                    finish_round(begin, 0)
                    break
                logits, states = tf([actual])
                h_prev = hidden(states)
                prefix.append(sample(logits[-1]))
                rank_call(online.finalize, actual, operation="finalize")
                finish_round(begin, 0)
                continue

            count("position", "accepted")
            online = shadow
            h_p1 = hidden(states, 1)
            p_f = probabilities(logits[1])[P_VOCAB:EOS]
            count("feature", "verified")
            if not feature_accept(p_f, q, f_local):
                actual_f = timer.run("sampling", residual_sample, p_f, q, rng,
                                     gpu=True, operation="feature_residual") + P_VOCAB
                rewind(before + 2)
                prefix.extend([p1, actual_f])
                rank_call(online.drop_eos, operation="drop_pending")
                h_prev = h_p1
                finish_round(begin, 1)
                continue

            count("feature", "accepted")
            prefix.extend([p1, f1])
            count("position", "verified")
            p = probabilities(logits[2])
            if point_accept(p, p2):
                count("position", "accepted")
                actual = p2
                h_prev = hidden(states, 3)
                final_f = sample(logits[3])
                accepted = 3
            else:
                actual = point_residual(p, p2)
                rewind(before + 3)
                accepted = 2
                if actual != EOS:
                    logits, states = tf([actual])
                    h_prev = hidden(states)
                    final_f = sample(logits[-1])
            prefix.append(actual)
            if actual == EOS:
                rank_call(online.drop_eos, operation="drop_pending")
                finish_round(begin, accepted)
                break
            prefix.append(final_f)
            rank_call(online.finalize, actual, operation="finalize")
            finish_round(begin, accepted)
    finally:
        stats["draft_calls"] = adapter.forward_calls
        stats["draft_input_tokens"] = adapter.input_tokens
        try:
            adapter.close()
        finally:
            adapter._profile = None
            if capture is not None:
                timer.run("cache_maintenance", capture.close, operation="capture_close")
            # 局部 Target KV 也在本次耗时范围内释放。
            def release_target():
                nonlocal cache
                cache = None

            timer.run("cache_maintenance", release_target, operation="target_close")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = prefix[length:]
    shared.validate_sequence(prompt, output, ranks)
    require(len(output) <= budget, "Output exceeds token budget")
    require(len(stats["accepted_lengths"]) == stats["rounds"] ==
            len(stats["round_generated_lengths"]), "Round totals differ")
    for field in ("proposed", "verified", "accepted"):
        require(stats[field] == sum(stats[kind + "_" + field] for kind in shared.KINDS),
                "Typed totals differ")
    stats.update(
        generated_tokens=len(output), seconds=elapsed,
        tokens_per_second=ratio(len(output), elapsed), eos=bool(output and output[-1] == EOS),
        k=2, proposal_width=3, seed=seed, prompt_tokens=length - 1,
        mean_accepted_length=ratio(sum(stats["accepted_lengths"]), stats["rounds"]),
        mean_generated_per_round=ratio(sum(stats["round_generated_lengths"]), stats["rounds"]),
        variant=adapter.variant, mode="draft", requested_tokens=budget,
        feature_precision=adapter.feature_precision, target_precision="FP32",
        temperature=.9, draft_top_p=.9, target_top_p=.9,
    )
    if profile:
        stats["stage_profile"] = timer.result()
    return shared._add_rates(stats), output
