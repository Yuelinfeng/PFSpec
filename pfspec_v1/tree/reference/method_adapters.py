"""Adapters for existing GaussianGPT decoders used by evaluate_methods.py."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import sys


METHODS = (
    "target",
    "ranker_only",
    "pf_mlp",
    "eagle_v3_g3",
    "eagle_v3_g5",
    "eagle_v4_reuse_g3",
    "eagle_v2_joint_g3",
    "medusa_lowrank",
    "hydra_conditional",
    "d8_pf",
    "d8_joint",
    "lfq_scratch_pf_bos",
)

RANKER_PATH = Path(
    "/root/autodl-tmp/pfspec_repro_4090d_20261028/project/profiling/target_eval/results/"
    "ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib"
)
EAGLE_V2_JOINT = Path(
    "/root/autodl-tmp/gaussiangpt_eagle_pf_v2_20261019/"
    "training_extended/joint/selected.pt"
)
EAGLE_V3_FEATURE = Path(
    "/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt"
)
METHOD_COMPARE_ROOT = Path(
    "/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare"
)
METHOD_COMPARE_SCRIPT_ROOT = METHOD_COMPARE_ROOT.parent
MEDUSA_LOW_RANK = METHOD_COMPARE_ROOT / "medusa_lowrank/medusa_linear_heads_best.pt"
HYDRA_WIDTH177 = METHOD_COMPARE_ROOT / "hydra_width177/medusa_conditional_heads_best.pt"
D8_CHECKPOINT = Path(
    "/root/autodl-tmp/pfspec_repro_4090d_20261028/project/draft_a/runs/feature_acceptance_20261014/"
    "adapt_d8_acceptance/step_3600.pt"
)
MLP_CHECKPOINT = Path(
    "/root/autodl-tmp/pfspec_repro_4090d_20261028/project/draft_a/runs/feature_target_continue_20261014/selected.pt"
)
LFQ_CHECKPOINT = Path(
    "/root/autodl-tmp/pfspec_repro_4090d_20261028/project/profiling/feature_draft/lfq_scratch_30epoch/"
    "lfq_scratch_conditional_best.pt"
)
LFQ_HEAD = Path(
    "/root/autodl-tmp/pfspec_repro_4090d_20261028/project/profiling/feature_draft/state_generation_512/"
    "states/frozen_feature_head.pt"
)


@dataclass
class MethodAdapter:
    name: str
    metadata: dict
    model: object | None = None
    decode_fn: object | None = None
    variant: str | None = None
    context: dict | None = None


def available_methods():
    return METHODS


def _decode_with_output(position_decode):
    """Clone a k=0-compatible decoder and expose its existing output suffix."""
    source = inspect.getsource(position_decode)
    if source.count("def decode(") != 1 or source.count("return _add_rates(stats)") != 1:
        raise RuntimeError("The saved decoder no longer has the expected return statement")
    source = source.replace("def decode(", "def decode_with_output(", 1)
    source = source.replace("return _add_rates(stats)",
                             "return _add_rates(stats), output", 1)
    namespace = dict(position_decode.__globals__)
    exec(compile(source, position_decode.__code__.co_filename, "exec"), namespace)
    return namespace["decode_with_output"]


def build_context(project: Path, device="cuda"):
    """Load one FP32 Target core and shared Ranker for all methods."""
    project = Path(project).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))

    import torch
    from joblib import load
    from bench_fast_hgb import FastHistRanker
    from draft_a.corpus_v2 import EOS, load_target

    wrapper = load_target().to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    core = getattr(wrapper, "transformer", None)
    if core is None:
        raise RuntimeError("Target wrapper has no transformer core")
    core.eval().requires_grad_(False)
    ranker_model = FastHistRanker(load(RANKER_PATH))
    ranks_tensor = core.sparse_position_rank_map.detach().to(device="cpu", dtype=torch.long)
    ranks = ranks_tensor.tolist()
    if int(core.sparse_eos_token_id) != int(EOS):
        raise ValueError("Target and crop token EOS ids differ")
    return dict(
        project=project,
        device=torch.device(device),
        target_model=wrapper,
        target=core,
        ranker=ranker_model,
        ranks=ranks,
        ranks_tensor=ranks_tensor,
        eos=int(EOS),
        p_vocab=int(core.sparse_position_vocab_size),
        feature_vocab=int(core.sparse_feature_vocab_size),
        cache_for=importlib.import_module("draft_a.feature_acceptance_eval").cache_for,
        target_decode=_decode_with_output(importlib.import_module("draft_a.pf_ranker_eval").decode),
        joint_decode=_decode_with_output(importlib.import_module("draft_a.feature_acceptance_eval").decode),
    )


def _metadata(name, checkpoint=None, **values):
    item = {"method": name, **values}
    if checkpoint is not None:
        item["checkpoint"] = str(checkpoint)
    return item


def _load_d8(context):
    import torch
    from draft_a.feature_reuse import make_model

    checkpoint = D8_CHECKPOINT
    if not checkpoint.is_file():
        checkpoint = context["project"] / "draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    draft_model = make_model(saved["config"]).to(device=context["device"], dtype=torch.float32)
    draft_model.load_state_dict(saved["model"], strict=True)
    draft_model.eval().requires_grad_(False)
    draft_core = draft_model.transformer
    same_ranks = torch.equal(
        draft_core.sparse_position_rank_map.detach().cpu(), context["ranks_tensor"]
    )
    if not same_ranks:
        raise ValueError("D8 and Target Position ranks differ")
    return draft_model, draft_core, saved, checkpoint


def load_method(name: str, context: dict):
    """Load one existing draft. Target and Ranker objects stay shared."""
    import torch

    if name not in METHODS:
        raise KeyError(f"Unknown method {name!r}; available methods: {', '.join(METHODS)}")
    if name == "target":
        return MethodAdapter(name, _metadata(name, decoder="pf_ranker_eval.decode", k=0,
                                              hidden_state="unused"),
                             decode_fn=context["target_decode"], context=context)
    if name == "ranker_only":
        return MethodAdapter(name, _metadata(name, decoder="pf_ranker_eval.decode", k=1,
                                              draft="FastHistRanker", hidden_state="unused"),
                             decode_fn=context["target_decode"], context=context)
    if name == "pf_mlp":
        from bench_gaussiangpt_pf_block2_cow import load_feature_model
        from draft_a.feature_target_v2 import HEAD
        from draft_a.pf_feature_unified import MLPAdapter

        loaded = load_feature_model(
            MLP_CHECKPOINT, HEAD, context["target"], context["device"],
            context["p_vocab"], context["eos"],
        )
        feature = loaded[0]
        feature.eval().requires_grad_(False)
        adapter = MLPAdapter(feature, bf16=False)
        return MethodAdapter(name, _metadata(
            name, MLP_CHECKPOINT, decoder="pf_feature_unified.decode",
            hidden_state="pre-finalnorm", target_policy="temperature=0.9,top_p=0.9",
            feature_policy="FP32,temperature=0.9,top_p=0.9",
        ), model=adapter, context=context)

    if name.startswith("eagle_"):
        from eval_pf_draft import load_draft
        from speculative import pf_engine

        if name == "eagle_v2_joint_g3":
            checkpoint, mode, gamma, reuse = EAGLE_V2_JOINT, "joint", 3, False
        elif name == "eagle_v3_g3":
            checkpoint, mode, gamma, reuse = EAGLE_V3_FEATURE, "feature", 3, False
        elif name == "eagle_v3_g5":
            checkpoint, mode, gamma, reuse = EAGLE_V3_FEATURE, "feature", 5, False
        else:
            checkpoint, mode, gamma, reuse = EAGLE_V3_FEATURE, "feature", 3, True
        draft = load_draft(context["target"], checkpoint)
        if draft.config.mode != mode:
            raise ValueError(f"EAGLE checkpoint mode mismatch for {name}: {draft.config.mode}")
        return MethodAdapter(name, _metadata(
            name, checkpoint, decoder="speculative.pf_engine.decode", mode=mode,
            gamma=gamma, reuse_draft_cache=reuse,
            hidden_state="post-finalnorm", target_query_max=gamma + 1,
            target_cache_factory="speculative.pf_engine.target_cache",
        ), model=draft, variant="eagle", context=context)

    if name in {"medusa_lowrank", "hydra_conditional"}:
        for path in (METHOD_COMPARE_SCRIPT_ROOT, METHOD_COMPARE_ROOT):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        medusa = importlib.import_module("bench_gaussiangpt_medusa_linear")
        heads_path = MEDUSA_LOW_RANK if name == "medusa_lowrank" else HYDRA_WIDTH177
        if name == "medusa_lowrank":
            from stage5_medusa_lowrank import LowRankMedusaHeads

            previous_new_heads = medusa.new_heads

            def make_low_rank(target, p_vocab, eos):
                return LowRankMedusaHeads(
                    target.config.n_embd, target.lm_head.weight, p_vocab, eos,
                    base_class=medusa.MedusaLinearHeads,
                ).to(target.lm_head.weight.device)

            try:
                medusa.new_heads = make_low_rank
                heads, head_meta = medusa.load_heads(
                    heads_path, context["target"], context["p_vocab"], context["eos"]
                )
            finally:
                medusa.new_heads = previous_new_heads
            method_meta = dict(head_kind="independent", head_metadata=head_meta)
        else:
            hydra_module = importlib.import_module(
                "gaussiangpt_medusa_conditional_v3.medusa_conditional_heads"
            )
            spec = hydra_module.TokenSpec(
                position_vocab_size=context["p_vocab"],
                feature_vocab_size=context["feature_vocab"],
                feature_offset=int(context["target"].sparse_feature_token_offset),
                eos_id=context["eos"],
                total_vocab_size=int(context["target"].config.vocab_size),
            )
            heads = hydra_module.ConditionalHeads(context["target"], spec, width=177)
            saved = torch.load(heads_path, map_location="cpu", weights_only=False)
            heads.load_state_dict(saved["heads"], strict=True)
            heads.to(device=context["device"]).eval().requires_grad_(False)
            method_meta = dict(head_kind="conditional", width=177,
                               checkpoint_step=saved.get("step"),
                               training_epoch=saved.get("epoch"))
        heads.eval().requires_grad_(False)
        return MethodAdapter(name, _metadata(
            name, heads_path, decoder="step5_method_compare.decode_heads",
            hidden_state="RMSNorm(last-block output) via existing Medusa hook",
            target_query_max=4, target_cache_factory="per-call make_cache closure",
            output_normalization="append EOS when raw stats report natural termination",
            **method_meta,
        ), model=heads, variant="medusa_hydra", context=context)

    if name in {"d8_pf", "d8_joint"}:
        from draft_a.pf_feature_unified import GPTAdapter

        model, draft_core, saved, checkpoint = _load_d8(context)
        if name == "d8_pf":
            adapter = GPTAdapter(draft_core)
            decoder = "pf_feature_unified.decode"
            hidden_state = "D8 uses its own cached state and proposes Feature only"
            mode = "feature_plus_ranker"
            target_query_max = 4
        else:
            adapter = None
            decoder = "draft_a.feature_acceptance_eval.decode"
            hidden_state = "unused"
            mode = "joint_PF_draft"
            target_query_max = 3
        return MethodAdapter(name, _metadata(
            name, checkpoint, decoder=decoder, mode=mode, k=2 if name == "d8_joint" else None,
            hidden_state=hidden_state, target_query_max=target_query_max,
            checkpoint_step=saved.get("step"), model_config=saved["config"],
        ), model=(model, draft_core, adapter), variant="d8", context=context)

    if name == "lfq_scratch_pf_bos":
        from bench_lfq_scratch_pf_cow import load_feature_model

        feature, draft_temp, draft_top_p = load_feature_model(
            LFQ_CHECKPOINT, LFQ_HEAD, context["target"], context["device"],
            context["p_vocab"], context["eos"],
        )
        feature.eval().requires_grad_(False)
        return MethodAdapter(name, _metadata(
            name, LFQ_CHECKPOINT, decoder="bench_lfq_scratch_pf_cow.block2_decode",
            scope="BOS-only; prompt must equal [EOS]",
            hidden_state="pre-finalnorm Target block state",
            target_query_max=4, target_cache_factory="make_kv_cache closure",
            target_policy="temperature=0.9,top_p=0.9",
            draft_temperature=float(draft_temp), draft_top_p=float(draft_top_p),
            output_normalization="append EOS when raw stats report natural termination",
        ), model=feature, variant="lfq_bos", context=context)

    raise AssertionError(f"Unhandled method {name}")


@contextmanager
def _patch_namespace_cache(namespace: dict, runtime):
    if runtime is None:
        yield
        return
    if "cache_for" not in namespace:
        raise RuntimeError("Decoder namespace does not expose cache_for")
    original = namespace["cache_for"]
    namespace["cache_for"] = lambda core, capacity: runtime.cache_for(
        core, capacity, factory=original
    )
    try:
        yield
    finally:
        namespace["cache_for"] = original


@contextmanager
def _patch_module_cache(module, name: str, runtime):
    if runtime is None:
        yield
        return
    original = getattr(module, name)
    setattr(module, name, lambda core, capacity: runtime.cache_for(
        core, capacity, factory=original
    ))
    try:
        yield
    finally:
        setattr(module, name, original)


class _MedusaPostNormSink:
    """Convert the runtime's pre-finalnorm state as the existing hook does."""
    def __init__(self, target):
        self.target = target

    def __call__(self, hidden):
        import torch.nn.functional as F

        block = self.target.transformer.h[-1]
        owners = []
        for hook in getattr(block, "_forward_hooks", {}).values():
            owner = getattr(hook, "__self__", None)
            if owner is not None and hasattr(owner, "value"):
                owners.append(owner)
        if not owners:
            raise RuntimeError("Medusa/Hydra Target hook is not attached during graph replay")
        postnorm = F.rms_norm(hidden.detach(), (hidden.size(-1),))
        for owner in owners:
            owner.value = postnorm


def _runtime_context(runtime, hidden_capture=None):
    if runtime is None:
        return ExitStack()
    stack = ExitStack()
    stack.enter_context(runtime.attention_context(enabled=True, hidden_capture=hidden_capture))
    return stack


def _finish(stats, suffix):
    values = dict(stats)
    suffix = [int(token) for token in suffix]
    values.setdefault("generated_tokens", len(suffix))
    values.setdefault("eos", bool(suffix and suffix[-1] == 12096))
    return values, suffix


def _normalize_omitted_eos(stats, suffix, eos):
    """Expose an EOS sentinel omitted by a legacy decoder without changing its loop."""
    suffix = [int(token) for token in suffix]
    eos_count = int(stats.get("eos", 0))
    stats["raw_body_tokens"] = len(suffix)
    stats["raw_eos_count"] = eos_count
    if eos_count and (not suffix or suffix[-1] != int(eos)):
        suffix.append(int(eos))
    stats["generated_tokens"] = len(suffix)
    return suffix


def decode_method(adapter: MethodAdapter, target, prompt, budget, seed, runtime=None):
    """Run one existing decoder, returning its count dictionary and full suffix."""
    import torch

    prompt = [int(token) for token in prompt]
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        if adapter.name in {"target", "ranker_only"}:
            decoder = adapter.decode_fn
            if decoder is None:
                raise RuntimeError("k=0/k=1 output decoder was not installed")
            k = 0 if adapter.name == "target" else 1
            ranker = None if k == 0 else adapter.context["ranker"]
            ranks = adapter.context["ranks"]
            with _patch_namespace_cache(decoder.__globals__, runtime):
                with _runtime_context(runtime):
                    stats, suffix = decoder(target, None, ranker, ranks, prompt, budget, k, seed)
            return _finish(stats, suffix)

        if adapter.name == "pf_mlp":
            from draft_a import pf_ranker_eval
            from draft_a.pf_feature_unified import decode

            context = adapter.context
            with _patch_module_cache(pf_ranker_eval, "cache_for", runtime):
                with _runtime_context(runtime):
                    stats, suffix = decode(
                        target, adapter.model, context["ranker"], context["ranks"],
                        prompt, budget, seed,
                    )
            return _finish(stats, suffix)

        if adapter.variant == "eagle":
            from speculative import pf_engine

            context = adapter.context
            original = pf_engine.target_cache
            if runtime is not None:
                pf_engine.target_cache = lambda core, capacity: runtime.cache_for(
                    core, capacity, factory=original
                )
            try:
                with _runtime_context(runtime):
                    stats, suffix = pf_engine.decode(
                        target, adapter.model, prompt, budget,
                        mode=adapter.metadata["mode"],
                        gamma=adapter.metadata["gamma"],
                        ranker=(pf_engine.PositionRanker(context["ranker"], context["ranks"])
                                if adapter.metadata["mode"] == "feature" else None),
                        seed=seed,
                        reuse_draft_cache=adapter.metadata["reuse_draft_cache"],
                    )
            finally:
                pf_engine.target_cache = original
            return _finish(stats, suffix)

        if adapter.variant == "medusa_hydra":
            for path in (METHOD_COMPARE_SCRIPT_ROOT, METHOD_COMPARE_ROOT):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            runner = importlib.import_module("step5_method_compare")
            medusa = importlib.import_module("bench_gaussiangpt_medusa_linear")
            from draft_a import feature_acceptance_eval

            context = adapter.context
            def make_cache(limit):
                capacity = 1 + int(limit) + 16
                factory = feature_acceptance_eval.cache_for
                return (runtime.cache_for(target, capacity, factory=factory)
                        if runtime is not None else factory(target, capacity))

            capture = _MedusaPostNormSink(target) if runtime is not None else None
            with _runtime_context(runtime, capture):
                full, raw_stats = runner.decode_heads(
                    target, adapter.model,
                    "independent" if adapter.name == "medusa_lowrank" else "conditional",
                    make_cache, eos=context["eos"], p_vocab=context["p_vocab"],
                    ranks=context["ranks_tensor"], prompt=prompt, budget=budget,
                    seed=seed, device=context["device"], temp=0.9, top_p=0.9,
                )
            suffix = full[len(prompt) - 1:]
            stats = dict(raw_stats)
            if "eos" not in stats:
                stats["eos"] = 0
            suffix = _normalize_omitted_eos(stats, suffix, context["eos"])
            return _finish(stats, suffix)

        if adapter.variant == "d8":
            from draft_a import pf_ranker_eval, feature_acceptance_eval
            from draft_a.pf_feature_unified import decode as pf_decode

            context = adapter.context
            _, draft_core, feature_adapter = adapter.model
            if adapter.name == "d8_pf":
                with _patch_module_cache(pf_ranker_eval, "cache_for", runtime):
                    with _runtime_context(runtime):
                        stats, suffix = pf_decode(
                            target, feature_adapter, context["ranker"], context["ranks"],
                            prompt, budget, seed,
                        )
            else:
                decoder = context["joint_decode"]
                with _patch_namespace_cache(decoder.__globals__, runtime):
                    with _runtime_context(runtime):
                        stats, suffix = decoder(target, draft_core, prompt, budget, 2, seed)
            return _finish(stats, suffix)

        if adapter.variant == "lfq_bos":
            from bench_lfq_scratch_pf_cow import block2_decode
            from profiling.s3.s3_onpolicy_feature_distill_train import make_kv_cache

            context = adapter.context
            if prompt != [context["eos"]]:
                raise ValueError("LFQ BOS adapter only accepts prompt=[EOS]")
            if budget < 2 or budget % 2:
                raise ValueError("LFQ BOS budget must be positive and even")
            draft_temp = adapter.metadata["draft_temperature"]
            draft_top_p = adapter.metadata["draft_top_p"]
            capacity = 1 + budget + 8

            def make_cache():
                factory = lambda core, length: make_kv_cache(
                    core, length, context["device"]
                )
                return (runtime.cache_for(target, capacity, factory=factory)
                        if runtime is not None else factory(target, capacity))

            with _runtime_context(runtime):
                body, raw_stats = block2_decode(
                    target, make_cache, context["ranker"], adapter.model,
                    eos=context["eos"], temp=0.9, top_p=0.9, seed=seed,
                    max_tokens=budget, device=context["device"],
                    p_vocab=context["p_vocab"], ranks=context["ranks_tensor"],
                    draft_temp=draft_temp, draft_top_p=draft_top_p,
                )
            stats = dict(raw_stats)
            if "eos" not in stats:
                stats["eos"] = 0
            body = _normalize_omitted_eos(stats, body, context["eos"])
            return _finish(stats, body)

    raise KeyError(f"No decode path for {adapter.name}")


