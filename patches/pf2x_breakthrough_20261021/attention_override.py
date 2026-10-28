"""Temporary FP32 attention implementations for isolated decode experiments."""

from contextlib import contextmanager


@contextmanager
def attention_context(implementation):
    import torch
    import model.gpt.flash_attention as attention_module
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if implementation not in {"default", "math", "explicit_bmm"}:
        raise ValueError(implementation)
    original = attention_module._sdpa_attention

    def attention(q, k, v, window_size, enable_gqa):
        # Long prefill keeps the original algorithm and its causal convention.
        if q.size(2) > 4 or enable_gqa:
            return original(q, k, v, window_size, enable_gqa)
        if implementation == "math":
            with sdpa_kernel(SDPBackend.MATH):
                return original(q, k, v, window_size, enable_gqa)
        tq, tk = q.size(2), k.size(2)
        window = window_size[0]
        qk_scale = q.size(-1) ** -0.25
        scores = torch.matmul(q * qk_scale, (k * qk_scale).transpose(-2, -1))
        if tq == tk and (window < 0 or window >= tq):
            allowed = torch.ones((tq, tk), device=q.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, -torch.inf)
        elif tq != 1:
            if tq == tk:
                allowed = torch.ones((tq, tk), device=q.device, dtype=torch.bool).tril()
                if 0 < window < tq:
                    rows = torch.arange(tq, device=q.device).unsqueeze(1)
                    columns = torch.arange(tk, device=q.device).unsqueeze(0)
                    allowed &= (rows - columns) <= window
            else:
                prefix = tk - tq
                allowed = torch.zeros((tq, tk), device=q.device, dtype=torch.bool)
                allowed[:, :prefix] = True
                allowed[:, prefix:] = torch.ones((tq, tq), device=q.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, -torch.inf)
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    try:
        if implementation != "default":
            attention_module._sdpa_attention = attention
        yield
    finally:
        attention_module._sdpa_attention = original
