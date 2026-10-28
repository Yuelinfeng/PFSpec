"""Small Direct and TwoStep Feature-logit residual models."""
from __future__ import annotations

import math

POSITION_VOCAB = 8000
FEATURE_VOCAB = 4096
FEATURE_OFFSET = POSITION_VOCAB
EOS_ID = POSITION_VOCAB + FEATURE_VOCAB
TEMPERATURE = 0.9
TOP_P = 0.9
EMBED_WIDTH = 64

CONFIGS = {
    "main": {"d": 128, "r": 64, "rz": 32},
    "narrow": {"d": 64, "r": 32, "rz": 16},
}


def position_xyz(position_id: int) -> tuple[float, float, float]:
    """Map xyz-ordered sparse Position IDs to normalized 20-cube coordinates."""
    value = int(position_id)
    if not 0 <= value < POSITION_VOCAB:
        raise ValueError(f"Position ID out of range: {value}")
    x, rem = divmod(value, 400)
    y, z = divmod(rem, 20)
    return x / 19.0, y / 19.0, z / 19.0


def config_parameter_table(name: str = "main") -> dict[str, int | float | str]:
    if name not in CONFIGS:
        raise ValueError(f"Unknown model size: {name}")
    cfg = CONFIGS[name]
    d, r, rz = cfg["d"], cfg["r"], cfg["rz"]
    f_in = d + EMBED_WIDTH + 3
    p_in = d + EMBED_WIDTH + 9
    c_f = (f_in + 1) * r + (r + 1) * d
    c_p = (p_in + 1) * r + (r + 1) * d
    c_state = (2 * d + 1) * rz
    c_class = FEATURE_VOCAB * rz
    c_two = c_f + c_p + c_state + c_class
    direct_in = d + 2 * EMBED_WIDTH + 9
    direct_width = round((c_two - c_class - rz) / (direct_in + 1 + rz))
    c_direct = (direct_in + 1) * direct_width + (direct_width + 1) * rz + c_class
    return {
        "size": name,
        "d": d,
        "r": r,
        "rz": rz,
        "direct_width": direct_width,
        "two_step_trainable": c_two,
        "direct_trainable": c_direct,
        "difference_percent": 100.0 * abs(c_direct - c_two) / c_two,
        "fixed_feature_table": FEATURE_VOCAB * EMBED_WIDTH,
        "fixed_position_table": POSITION_VOCAB * EMBED_WIDTH,
        "hidden_input": "Target final-block state before final norm",
        "embedding_input": "first 64 channels of original Target token table",
    }


def parameter_table() -> list[dict[str, int | float | str]]:
    return [config_parameter_table(name) for name in ("main", "narrow")]


def _need_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor operations")


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None
    F = None


if torch is not None:
    def fixed_token_tables(target_wte: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read deterministic 64-channel slices from the original Target table."""
        if target_wte.ndim != 2 or target_wte.size(0) < EOS_ID or target_wte.size(1) < EMBED_WIDTH:
            raise ValueError(f"Expected Target token table with at least [{EOS_ID},{EMBED_WIDTH}]")
        table = target_wte.detach().float().cpu()
        positions = table[:POSITION_VOCAB, :EMBED_WIDTH].contiguous()
        features = table[FEATURE_OFFSET:EOS_ID, :EMBED_WIDTH].contiguous()
        if not torch.isfinite(positions).all() or not torch.isfinite(features).all():
            raise ValueError("Target token table contains non-finite values")
        return positions, features


    def fit_hidden_map(train_hidden: torch.Tensor, width: int = 128) -> dict[str, torch.Tensor]:
        """Fit a deterministic PCA map using only the supplied training states."""
        values = train_hidden.detach().float().cpu()
        if values.ndim != 2 or values.size(0) < 2 or not 1 <= width <= values.size(1):
            raise ValueError("Training hidden states must be [N,H] with N>=2 and width<=H")
        mean = values.mean(dim=0)
        centered = values - mean
        covariance = centered.double().T @ centered.double()
        covariance.div_(max(values.size(0) - 1, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[:width]
        basis = eigenvectors[:, order].float()
        # Fix each eigenvector's sign so saved maps are reproducible.
        pivots = basis.abs().argmax(dim=0)
        signs = torch.sign(basis[pivots, torch.arange(width)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis = basis * signs.unsqueeze(0)
        scales = eigenvalues[order].clamp_min(1e-8).sqrt().float()
        return {"mean": mean, "basis": basis, "scale": scales}


    def _position_features(p0: torch.Tensor, p1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if p0.ndim == 0:
            p0 = p0.unsqueeze(0)
        if p1.ndim == 0:
            p1 = p1.unsqueeze(0)
        p0 = p0.long()
        p1 = p1.long()
        c0 = torch.stack((p0 // 400, (p0 // 20) % 20, p0 % 20), dim=-1).float() / 19.0
        c1 = torch.stack((p1 // 400, (p1 // 20) % 20, p1 % 20), dim=-1).float() / 19.0
        return torch.cat((c0, c1, c1 - c0), dim=-1), c0


    class TransitionModel(nn.Module):
        def __init__(
            self,
            method: str,
            target_wte: torch.Tensor,
            hidden_map: dict[str, torch.Tensor],
            size: str = "main",
        ):
            super().__init__()
            if method not in {"direct", "two_step"}:
                raise ValueError("method must be direct or two_step")
            if size not in CONFIGS:
                raise ValueError(f"Unknown model size: {size}")
            cfg = CONFIGS[size]
            self.method = method
            self.size = size
            self.width = int(cfg["d"])
            self.rank = int(cfg["r"])
            self.logit_rank = int(cfg["rz"])
            position_table, feature_table = fixed_token_tables(target_wte)
            self.register_buffer("position_table", position_table)
            self.register_buffer("feature_table", feature_table)
            self.register_buffer("hidden_mean", hidden_map["mean"].detach().float().cpu())
            self.register_buffer("hidden_basis", hidden_map["basis"].detach().float().cpu())
            self.register_buffer("hidden_scale", hidden_map["scale"].detach().float().cpu())
            if self.hidden_basis.size(1) != self.width:
                raise ValueError("Hidden map width does not match model size")

            self.class_factors = nn.Parameter(torch.empty(FEATURE_VOCAB, self.logit_rank))
            nn.init.normal_(self.class_factors, mean=0.0, std=0.02)
            if method == "two_step":
                self.f_update = nn.Sequential(
                    nn.Linear(self.width + EMBED_WIDTH + 3, self.rank), nn.GELU(),
                    nn.Linear(self.rank, self.width),
                )
                self.p_update = nn.Sequential(
                    nn.Linear(self.width + EMBED_WIDTH + 9, self.rank), nn.GELU(),
                    nn.Linear(self.rank, self.width),
                )
                self.state_head = nn.Linear(2 * self.width, self.logit_rank)
                nn.init.zeros_(self.f_update[-1].weight)
                nn.init.zeros_(self.f_update[-1].bias)
                nn.init.zeros_(self.p_update[-1].weight)
                nn.init.zeros_(self.p_update[-1].bias)
                nn.init.zeros_(self.state_head.weight)
                nn.init.zeros_(self.state_head.bias)
                self.direct_head = None
            else:
                din = self.width + 2 * EMBED_WIDTH + 9
                hidden = round((config_parameter_table(size)["two_step_trainable"]
                                - FEATURE_VOCAB * self.logit_rank - self.logit_rank)
                               / (din + 1 + self.logit_rank))
                self.direct_head = nn.Sequential(
                    nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, self.logit_rank),
                )
                nn.init.zeros_(self.direct_head[-1].weight)
                nn.init.zeros_(self.direct_head[-1].bias)
                self.f_update = None
                self.p_update = None
                self.state_head = None

        def hidden_state(self, hidden: torch.Tensor) -> torch.Tensor:
            return ((hidden.float() - self.hidden_mean) @ self.hidden_basis) / self.hidden_scale

        def _forward_parts(
            self,
            old_logits: torch.Tensor,
            h0: torch.Tensor,
            f0_global: torch.Tensor,
            p0: torch.Tensor,
            p1: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            h0 = h0.float()
            if h0.ndim == 1:
                h0 = h0.unsqueeze(0)
            f0_global = torch.as_tensor(f0_global, device=h0.device, dtype=torch.long).reshape(-1)
            p0 = torch.as_tensor(p0, device=h0.device, dtype=torch.long).reshape(-1)
            p1 = torch.as_tensor(p1, device=h0.device, dtype=torch.long).reshape(-1)
            if old_logits.ndim == 1:
                old_logits = old_logits.unsqueeze(0)
            if old_logits.shape != (h0.size(0), FEATURE_VOCAB):
                raise ValueError("old_logits must have shape [batch,4096]")
            if not (f0_global.shape == p0.shape == p1.shape == (h0.size(0),)):
                raise ValueError("Feature and Position IDs must have one value per row")
            f_local = f0_global - FEATURE_OFFSET
            geom, c0 = _position_features(p0, p1)
            u0 = self.hidden_state(h0)
            ef = self.feature_table[f_local]
            ep = self.position_table[p1]
            if self.method == "two_step":
                u_f = u0 + self.f_update(torch.cat((u0, ef, c0), dim=-1))
                u_p = u_f + self.p_update(torch.cat((u_f, ep, geom), dim=-1))
                latent = self.state_head(torch.cat((u_p, u0), dim=-1))
            else:
                latent = self.direct_head(torch.cat((u0, ef, ep, geom), dim=-1))
                u_f = u0
                u_p = u0
            delta = latent @ self.class_factors.T / math.sqrt(self.logit_rank)
            return {"delta": delta, "u0": u0, "uF": u_f, "uP": u_p}

        def forward(
            self,
            old_logits: torch.Tensor,
            h0: torch.Tensor,
            previous_feature_id: torch.Tensor,
            previous_position_id: torch.Tensor,
            candidate_position_id: torch.Tensor,
            residual_scale: float = 1.0,
            return_aux: bool = False,
        ):
            aux = self._forward_parts(
                old_logits, h0, previous_feature_id, previous_position_id, candidate_position_id
            )
            scale = float(residual_scale)
            updated = old_logits.float().reshape_as(aux["delta"]) + scale * aux["delta"]
            return (updated, aux) if return_aux else updated

        def proposal_logits(
            self,
            old_logits,
            h0,
            previous_feature_id,
            previous_position_id,
            candidate_position_id,
            residual_scale: float = 1.0,
        ):
            return self.forward(
                old_logits, h0, previous_feature_id, previous_position_id,
                candidate_position_id, residual_scale=residual_scale,
            )

        def trainable_parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

else:
    def fixed_token_tables(target_wte):
        _need_torch()

    def fit_hidden_map(train_hidden, width: int = 128):
        _need_torch()

    class TransitionModel:
        def __init__(self, *args, **kwargs):
            _need_torch()

        def proposal_logits(self, *args, **kwargs):
            _need_torch()


def policy_batch(logits, temperature: float = TEMPERATURE, top_p: float = TOP_P):
    """Apply the existing temperature and shifted nucleus rule once."""
    _need_torch()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2 or logits.size(-1) != FEATURE_VOCAB:
        raise ValueError("Feature logits must have shape [batch,4096]")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling policy")
    z = logits.float() / float(temperature)
    if top_p < 1.0:
        values, ids = torch.sort(z, dim=-1, descending=True)
        probs = torch.softmax(values, dim=-1)
        remove = probs.cumsum(dim=-1) > float(top_p)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        values = values.masked_fill(remove, float("-inf"))
        z = torch.full_like(z, float("-inf"))
        z.scatter_(1, ids, values)
    return torch.softmax(z, dim=-1)


def standard_library_self_test() -> list[dict[str, int | float | str]]:
    assert position_xyz(0) == (0.0, 0.0, 0.0)
    assert position_xyz(7999) == (1.0, 1.0, 1.0)
    assert position_xyz(21) == (0.0, 1.0 / 19.0, 1.0 / 19.0)
    rows = parameter_table()
    for row in rows:
        if float(row["difference_percent"]) > 10.0:
            raise AssertionError(f"Parameter counts differ by more than 10%: {row}")
    return rows


def torch_cpu_self_test() -> dict[str, object]:
    """Exercise real tensor modules on CPU without loading project weights."""
    _need_torch()
    torch.manual_seed(17029)
    hidden_width = 128
    target_wte = torch.randn(EOS_ID + 1, hidden_width, dtype=torch.float32)
    train_hidden = torch.randn(24, hidden_width, dtype=torch.float32)
    rows = []
    for size in ("main", "narrow"):
        hidden_map = fit_hidden_map(train_hidden, width=int(CONFIGS[size]["d"]))
        for method in ("direct", "two_step"):
            model = TransitionModel(method, target_wte, hidden_map, size=size).cpu().eval()
            expected = config_parameter_table(size)[f"{method}_trainable"]
            actual = model.trainable_parameter_count()
            if actual != expected:
                raise AssertionError(f"{method}/{size} parameter count {actual} != {expected}")
            old = torch.randn(2, FEATURE_VOCAB)
            h0 = torch.randn(2, hidden_width)
            f0 = torch.tensor([8000, 12095])
            p0 = torch.tensor([0, 400])
            p1 = torch.tensor([1, 421])
            same = model.proposal_logits(old, h0, f0, p0, p1, residual_scale=0.0)
            if not torch.equal(same, old):
                raise AssertionError("Zero residual changed original logits")
            proposed = model.proposal_logits(old, h0, f0, p0, p1)
            probs = policy_batch(proposed, TEMPERATURE, TOP_P)
            if not torch.isfinite(proposed).all() or not torch.isfinite(probs).all():
                raise AssertionError("Model produced non-finite values")
            torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0)
            rows.append({"method": method, "size": size, "parameters": actual, "rows": 2})
    return {"device": "cpu", "torch": torch.__version__, "runs": rows}
