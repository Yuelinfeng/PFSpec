"""Lazy CUDA Graphs for the PF-MLP and Medusa LowRank proposal forwards.

Usage::

    proposal = build_proposal(adapter, verify=False)
    try:
        with proposal.context():
            stats, suffix = decode_method(...)
        proposal_stats = proposal.stats()
    finally:
        proposal.close()

Only the neural proposal forward is captured. Inputs are copied into persistent
graph-owned tensors before each replay. Returned tensors are borrowed graph
outputs and remain valid until the next call with the same input signature.
Policy construction, sampling, Target calls, acceptance, residual draws, and
cache changes remain in the existing Python decoder.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import time


_MISSING = object()


def _signature(torch, value):
    if torch.is_tensor(value):
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
            str(value.layout),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_signature(torch, item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_signature(torch, item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _signature(torch, value[key])) for key in sorted(value)),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f"unsupported proposal argument: {type(value).__name__}")


def _clone_inputs(torch, value):
    if torch.is_tensor(value):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, tuple):
        return tuple(_clone_inputs(torch, item) for item in value)
    if isinstance(value, list):
        return [_clone_inputs(torch, item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_inputs(torch, item) for key, item in value.items()}
    return value


def _copy_inputs(torch, destination, source):
    if torch.is_tensor(destination):
        if (not torch.is_tensor(source)
                or destination.shape != source.shape
                or destination.dtype != source.dtype
                or destination.device != source.device):
            raise ValueError("proposal input layout changed after graph capture")
        destination.copy_(source)
        return
    if isinstance(destination, tuple):
        if not isinstance(source, tuple) or len(destination) != len(source):
            raise ValueError("proposal tuple inputs changed after graph capture")
        for left, right in zip(destination, source):
            _copy_inputs(torch, left, right)
        return
    if isinstance(destination, list):
        if not isinstance(source, list) or len(destination) != len(source):
            raise ValueError("proposal list inputs changed after graph capture")
        for left, right in zip(destination, source):
            _copy_inputs(torch, left, right)
        return
    if isinstance(destination, dict):
        if not isinstance(source, dict) or destination.keys() != source.keys():
            raise ValueError("proposal keyword inputs changed after graph capture")
        for key in destination:
            _copy_inputs(torch, destination[key], source[key])
        return
    if destination != source:
        raise ValueError("proposal scalar input changed after graph capture")


def _walk_tensors(torch, value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(torch, item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(torch, item)


def _output_pairs(torch, left, right):
    if torch.is_tensor(left):
        if not torch.is_tensor(right):
            raise TypeError("proposal output structure changed")
        yield left, right
        return
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            raise TypeError("proposal tuple output structure changed")
        for a, b in zip(left, right):
            yield from _output_pairs(torch, a, b)
        return
    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise TypeError("proposal list output structure changed")
        for a, b in zip(left, right):
            yield from _output_pairs(torch, a, b)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise TypeError("proposal dictionary output structure changed")
        for key in left:
            yield from _output_pairs(torch, left[key], right[key])
        return
    if left != right:
        raise TypeError("proposal output scalar changed")


@contextmanager
def _strict_fp32(torch):
    old_precision = torch.get_float32_matmul_precision()
    old_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_cuda_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)


class ProposalForwardGraph:
    """Temporarily wrap one deterministic FP32 module method with CUDA Graphs."""

    def __init__(
        self,
        owner,
        method_name: str,
        *,
        method_kind: str,
        adapter=None,
        warmup_calls: int = 2,
        max_graphs: int = 8,
        verify: bool = False,
        verify_atol: float = 1e-5,
        verify_rtol: float = 1e-5,
    ):
        started = time.perf_counter()
        if method_kind not in {"mlp", "medusa_lowrank"}:
            raise ValueError(f"unsupported proposal kind {method_kind!r}")
        self.owner = owner
        self.method_name = str(method_name)
        self.method_kind = method_kind
        self.adapter = adapter
        self.warmup_calls = max(1, int(warmup_calls))
        self.max_graphs = max(1, int(max_graphs))
        self.verify_enabled = bool(verify)
        self.verify_atol = float(verify_atol)
        self.verify_rtol = float(verify_rtol)
        self._graphs = {}
        self._graph_count_total = 0
        self._runtime_init_ms = (time.perf_counter() - started) * 1000.0
        self._model_precision_checked = False
        self._model_precision_fallback = None
        self._calls = Counter()
        self._fallbacks = Counter()
        self._graph_events = []
        self._attempt_events = []
        self._active = False
        self._closed = False
        self._patch_present = False
        self._had_instance_method = False
        self._instance_method = None
        self._original_method = None
        self._had_instance_distribution = False
        self._instance_distribution = None
        self._original_distribution = None
        self._references_by_pointer = {}
        self._verification = dict(
            calls=0,
            allclose_failures=0,
            max_abs=0.0,
            max_relative=0.0,
            policy_calls=0,
            max_policy_tv=0.0,
            policy_support_mismatches=0,
        )

    @contextmanager
    def context(self):
        """Install the temporary forward wrapper and restore it on exit."""
        if self._closed:
            raise RuntimeError("proposal graph has been closed")
        if self._active:
            raise RuntimeError("proposal graph context is already active")
        import torch

        self._check_model_precision(torch)
        self._active = True
        self._original_method = getattr(self.owner, self.method_name)
        self._had_instance_method = self.method_name in self.owner.__dict__
        if self._had_instance_method:
            self._instance_method = self.owner.__dict__[self.method_name]

        def wrapped(*args, **kwargs):
            return self._run(*args, **kwargs)

        setattr(self.owner, self.method_name, wrapped)
        self._patch_present = True
        try:
            if self.verify_enabled and self.method_kind == "medusa_lowrank":
                self._install_distribution_monitor()
            with _strict_fp32(torch):
                yield self
        finally:
            self._restore_patch()
            self._active = False

    def _install_distribution_monitor(self):
        if self.owner is None or not hasattr(self.owner, "distribution"):
            raise TypeError("Medusa proposal owner has no distribution method")
        self._original_distribution = self.owner.distribution
        self._had_instance_distribution = "distribution" in self.owner.__dict__
        if self._had_instance_distribution:
            self._instance_distribution = self.owner.__dict__["distribution"]

        def wrapped_distribution(*args, **kwargs):
            return self._measure_distribution(*args, **kwargs)

        self.owner.distribution = wrapped_distribution

    def _restore_patch(self):
        if self._patch_present:
            if self._had_instance_method:
                setattr(self.owner, self.method_name, self._instance_method)
            else:
                delattr(self.owner, self.method_name)
            self._patch_present = False
        if self._original_distribution is not None:
            if self._had_instance_distribution:
                self.owner.distribution = self._instance_distribution
            else:
                delattr(self.owner, "distribution")
            self._original_distribution = None
        self._references_by_pointer.clear()

    def _eligible(self, torch, args, kwargs):
        if getattr(self.owner, "training", False):
            return False, "training_mode", None
        if self._model_precision_fallback is not None:
            return False, self._model_precision_fallback, None
        tensors = list(_walk_tensors(torch, (args, kwargs)))
        if not tensors:
            return False, "no_tensor_inputs", None
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            return False, "mixed_input_devices", None
        device = next(iter(devices))
        if device.type != "cuda":
            return False, "non_cuda_input", device
        if torch.is_grad_enabled():
            return False, "grad_enabled", device
        if any(tensor.layout != torch.strided for tensor in tensors):
            return False, "non_strided_input", device
        if any(tensor.is_floating_point() and tensor.dtype != torch.float32
               for tensor in tensors):
            return False, "non_fp32_input", device
        return True, None, device

    def _check_model_precision(self, torch):
        if self._model_precision_checked:
            return
        self._model_precision_checked = True
        for tensor in self.owner.parameters(recurse=True):
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                self._model_precision_fallback = "non_fp32_parameter"
                return
        for tensor in self.owner.buffers(recurse=True):
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                self._model_precision_fallback = "non_fp32_buffer"
                return

    def _state_snapshot(self, torch):
        values = []
        seen = set()
        for _, tensor in list(self.owner.named_parameters(recurse=True)) + list(
                self.owner.named_buffers(recurse=True)):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            values.append((tensor, tensor.detach().clone()))
        return values

    def _restore_changed_state(self, torch, snapshot):
        changed = []
        with torch.no_grad():
            for index, (tensor, saved) in enumerate(snapshot):
                if not torch.equal(tensor, saved):
                    changed.append(index)
                    tensor.copy_(saved)
        return changed

    @staticmethod
    def _rng_snapshot(torch, device):
        return torch.random.get_rng_state(), torch.cuda.get_rng_state(device)

    @staticmethod
    def _restore_rng(torch, device, states):
        cpu_state, cuda_state = states
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)

    def _build_graph(self, torch, key, args, kwargs, device):
        started = time.perf_counter()
        static_args = _clone_inputs(torch, args)
        static_kwargs = _clone_inputs(torch, kwargs)
        graph = torch.cuda.CUDAGraph()
        rng = self._rng_snapshot(torch, device)
        snapshot = self._state_snapshot(torch)
        warmup_ms = 0.0
        capture_ms = 0.0
        state_change = []
        rng_changed = False
        graph_outputs = None

        def invoke_static():
            with torch.inference_mode(), torch.autocast(
                    device_type="cuda", enabled=False):
                return self._original_method(*static_args, **static_kwargs)

        try:
            current = torch.cuda.current_stream(device)
            warm_stream = torch.cuda.Stream(device=device)
            warm_stream.wait_stream(current)
            warm_started = time.perf_counter()
            with torch.cuda.stream(warm_stream):
                for _ in range(self.warmup_calls):
                    _copy_inputs(torch, static_args, args)
                    _copy_inputs(torch, static_kwargs, kwargs)
                    invoke_static()
            current.wait_stream(warm_stream)
            torch.cuda.synchronize(device)
            warmup_ms = (time.perf_counter() - warm_started) * 1000.0

            capture_started = time.perf_counter()
            with torch.inference_mode(), torch.cuda.graph(graph):
                graph_outputs = invoke_static()
            torch.cuda.synchronize(device)
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            if not list(_walk_tensors(torch, graph_outputs)):
                raise TypeError("proposal forward returned no tensors")

            state_change = self._restore_changed_state(torch, snapshot)
            if state_change:
                raise RuntimeError("proposal forward changed module state during capture")
            rng_after = self._rng_snapshot(torch, device)
            rng_changed = (not torch.equal(rng[0], rng_after[0])
                           or not torch.equal(rng[1], rng_after[1]))
            if rng_changed:
                raise RuntimeError("proposal forward consumed RNG during capture")
        finally:
            if state_change:
                self._restore_changed_state(torch, snapshot)
            self._restore_rng(torch, device, rng)

        state = dict(
            graph=graph,
            args=static_args,
            kwargs=static_kwargs,
            outputs=graph_outputs,
            device=device,
            signature=key,
        )
        event = dict(
            signature=repr(key),
            device=str(device),
            warmup_calls=self.warmup_calls,
            warmup_ms=warmup_ms,
            capture_ms=capture_ms,
            prepare_ms=max(0.0, (time.perf_counter() - started) * 1000.0
                           - warmup_ms - capture_ms),
            setup_ms=(time.perf_counter() - started) * 1000.0,
            verified=self.verify_enabled,
        )
        self._graph_events.append(event)
        self._graph_count_total += 1
        return state

    def _run(self, *args, **kwargs):
        import torch

        self._calls["proposal_forward"] += 1
        eligible, reason, device = self._eligible(torch, args, kwargs)
        if not eligible:
            self._fallbacks[reason] += 1
            self._calls["original_forward"] += 1
            return self._call_original(torch, args, kwargs)

        key = _signature(torch, (args, kwargs))
        state = self._graphs.get(key)
        if state is None:
            if len(self._graphs) >= self.max_graphs:
                self._fallbacks["graph_limit"] += 1
                self._calls["original_forward"] += 1
                return self._call_original(torch, args, kwargs)
            self._calls["graph_capture_attempts"] += 1
            try:
                state = self._build_graph(torch, key, args, kwargs, device)
            except Exception as error:
                self._calls["graph_capture_failures"] += 1
                self._attempt_events.append(dict(
                    signature=repr(key),
                    device=str(device),
                    error=f"{type(error).__name__}: {error}",
                ))
                message = str(error).lower()
                if "illegal memory access" in message or "device-side assert" in message:
                    raise
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    raise
                self._fallbacks["capture_error"] += 1
                self._calls["original_forward"] += 1
                return self._call_original(torch, args, kwargs)
            self._graphs[key] = state

        _copy_inputs(torch, state["args"], args)
        _copy_inputs(torch, state["kwargs"], kwargs)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            state["graph"].replay()
        self._calls["graph_forward"] += 1
        self._calls[f"graph_forward_{self.method_kind}"] += 1

        if self.verify_enabled:
            self._calls["verification_forward"] += 1
            reference = self._call_original(torch, args, kwargs)
            self._verify_outputs(torch, state["outputs"], reference)
            if self.method_kind == "medusa_lowrank":
                self._references_by_pointer = self._reference_pointer_map(
                    torch, state["outputs"], reference
                )
            else:
                self._verify_mlp_policy(torch, state["outputs"], reference)
        return state["outputs"]

    def _call_original(self, torch, args, kwargs):
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            return self._original_method(*args, **kwargs)

    def _verify_outputs(self, torch, graph_outputs, reference):
        self._verification["calls"] += 1
        passed = True
        for actual, expected in _output_pairs(torch, graph_outputs, reference):
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                passed = False
                self._verification["allclose_failures"] += 1
                continue
            delta = (actual.detach().float() - expected.detach().float()).abs()
            max_abs = float(delta.max().item()) if delta.numel() else 0.0
            denom = expected.detach().float().abs().clamp_min(1e-12)
            max_relative = float((delta / denom).max().item()) if delta.numel() else 0.0
            self._verification["max_abs"] = max(
                self._verification["max_abs"], max_abs
            )
            self._verification["max_relative"] = max(
                self._verification["max_relative"], max_relative
            )
            if not torch.allclose(actual, expected, atol=self.verify_atol,
                                  rtol=self.verify_rtol):
                passed = False
                self._verification["allclose_failures"] += 1
        self._calls["verification_passes" if passed else "verification_failures"] += 1

    def _verify_mlp_policy(self, torch, graph_outputs, reference):
        from collect_feature_teacher_from_s3 import policy_batch

        graph_logits = graph_outputs[1].float()
        reference_logits = reference[1].float()
        graph_policy = policy_batch(graph_logits, 0.9, 0.9)[0]
        reference_policy = policy_batch(reference_logits, 0.9, 0.9)[0]
        self._record_policy_comparison(torch, graph_policy, reference_policy)

    def _reference_pointer_map(self, torch, graph_outputs, reference):
        result = {}
        for graph_tensor, reference_tensor in _output_pairs(
                torch, graph_outputs, reference):
            result[int(graph_tensor.data_ptr())] = reference_tensor
        return result

    def _measure_distribution(self, *args, **kwargs):
        original = self._original_distribution
        q = original(*args, **kwargs)
        if args and hasattr(args[0], "data_ptr"):
            reference_logits = self._references_by_pointer.get(int(args[0].data_ptr()))
            if reference_logits is not None:
                if reference_logits.shape != args[0].shape:
                    if reference_logits.numel() != args[0].numel():
                        self._verification["policy_support_mismatches"] += 1
                        self._verification["policy_calls"] += 1
                        return q
                    reference_logits = reference_logits.reshape_as(args[0])
                ref_args = (reference_logits, *args[1:])
                reference_q = original(*ref_args, **kwargs)
                import torch

                self._record_policy_comparison(torch, q, reference_q)
        return q

    def _record_policy_comparison(self, torch, actual, expected):
        actual = actual.detach().float()
        expected = expected.detach().float()
        if actual.shape != expected.shape:
            self._verification["policy_support_mismatches"] += 1
            self._verification["policy_calls"] += 1
            return
        tv = float((actual - expected).abs().sum().mul(0.5).item())
        actual_support = actual > 0
        expected_support = expected > 0
        if not torch.equal(actual_support, expected_support):
            self._verification["policy_support_mismatches"] += 1
        self._verification["max_policy_tv"] = max(
            self._verification["max_policy_tv"], tv
        )
        self._verification["policy_calls"] += 1

    def _restore_instance_member(self, name, existed, saved):
        if existed:
            setattr(self.owner, name, saved)
        elif name in self.owner.__dict__:
            delattr(self.owner, name)

    def stats(self):
        verification = dict(self._verification)
        verification["enabled"] = self.verify_enabled
        verification["policy_tv_tolerance"] = 1e-5
        verification["passed"] = (None if not self.verify_enabled else
            verification["calls"] > 0
            and verification["policy_calls"] > 0
            and verification["allclose_failures"] == 0
            and verification["policy_support_mismatches"] == 0
            and verification["max_policy_tv"] <= verification["policy_tv_tolerance"])
        return dict(
            runtime_init_ms=self._runtime_init_ms,
            method_kind=self.method_kind,
            patched_method=self.method_name,
            active=self._active,
            closed=self._closed,
            cache_events=[],
            graph_count=self._graph_count_total,
            graph_events=[dict(event) for event in self._graph_events],
            calls=dict(self._calls),
            fallbacks=dict(self._fallbacks),
            attempt_events=[dict(event) for event in self._attempt_events],
            verification=verification,
            output_lifetime=(
                "borrowed graph-owned tensors; consume or clone before the next call"
            ),
        )

    def close(self):
        """Restore temporary patches and release graph-owned GPU memory."""
        self._restore_patch()
        self._active = False
        self._graphs.clear()
        self._references_by_pointer.clear()
        self._closed = True


def build_proposal(adapter, *, verify=False, warmup_calls=2, max_graphs=8,
                   verify_atol=1e-5, verify_rtol=1e-5):
    """Build the matching proposer wrapper for a loaded method adapter."""
    name = getattr(adapter, "name", None)
    if name == "pf_mlp":
        model = getattr(adapter, "model", None)
        owner = getattr(model, "feature", None)
        if owner is None:
            raise TypeError("PF-MLP adapter does not expose its FeatureStateDraft")
        return ProposalForwardGraph(
            owner,
            "forward",
            method_kind="mlp",
            adapter=adapter,
            warmup_calls=warmup_calls,
            max_graphs=max_graphs,
            verify=verify,
            verify_atol=verify_atol,
            verify_rtol=verify_rtol,
        )
    if name == "medusa_lowrank":
        owner = getattr(adapter, "model", None)
        if owner is None or not callable(getattr(owner, "raw", None)):
            raise TypeError("Medusa LowRank adapter does not expose heads.raw")
        return ProposalForwardGraph(
            owner,
            "raw",
            method_kind="medusa_lowrank",
            adapter=adapter,
            warmup_calls=warmup_calls,
            max_graphs=max_graphs,
            verify=verify,
            verify_atol=verify_atol,
            verify_rtol=verify_rtol,
        )
    raise KeyError(f"Proposal Graph is not defined for method {name!r}")
