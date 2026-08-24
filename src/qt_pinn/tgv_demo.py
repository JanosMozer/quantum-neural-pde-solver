"""Shared TGV demo: eval, losses, presets. Circuit runs once per step, not per point."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from qt_pinn.fourier import FourierFeatureMapTGV
from pdes.ns2d.physics_loss import exact_solution, relative_l2, relative_l2_gauge, _grad

X_LO, X_HI = 0.0, 2.0 * math.pi
DEMO_NU = 0.1
DEMO_T_MAX = 5.0
EVAL_TIMES = (0.0, 0.5, 1.0)  # fractions of t_max

# ─── Presets ────────────────────────────────────────────────────────────────
# classical_hidden: bigger classical net (better accuracy, but slower inference)
# qt_hidden:        smaller deployed QT net (comparable accuracy, faster inference)
# The demo story: QT produces a qt_hidden-size net ≈ as accurate as the
# classical_hidden net, but with fewer deployed params → faster per-frame.

PRESETS: dict[str, dict[str, Any]] = {
    "scout": {
        "adam_steps": 2000,
        "n_colloc": 2048,
        "n_bc": 512,
        "classical_hidden": [64, 64],   # 5,059 params — classical baseline
        "qt_hidden": [16, 16],           #   499 params — deployed QT net (≤ classical)
        "n_qubits": 8,
        "n_layers": 4,
        "bottleneck_width": 64,
        "lambda_data": 5.0,
        "lambda_pde": 1.0,
        "lambda_bc": 0.0,               # hard IC → bc by construction
        "lr": 0.005,
        "log_every": 200,
        "eval_every": 400,
        "resample_every": 100,
        "budget_s": 900,
        "qc_arch": "expect",
        "t_weight": True,               # amplitude-weighted loss (fixes late-time error)
    },
    "demo": {
        "adam_steps": 10000,
        "n_colloc": 4096,
        "n_bc": 1024,
        # Classical: [64,64] = 5,059 params — strong baseline
        # QT deployed: [24,24] = 939 params — 5× smaller, comparable accuracy
        "classical_hidden": [64, 64],
        "qt_hidden": [24, 24],
        "n_qubits": 8,
        "n_layers": 6,           # larger circuit than scout (4L→6L)
        "bottleneck_width": 64,
        "lambda_data": 5.0,
        "lambda_pde": 1.0,
        "lambda_bc": 0.0,
        "lr": 0.003,
        "log_every": 500,
        "eval_every": 1000,
        "resample_every": 200,
        "budget_s": 4800,
        "qc_arch": "expect",
        "t_weight": True,
        "t_sample": "uniform",
    },
    # Longer train + tail-heavy t + slightly larger QT net (still ≪ classical).
    "polish": {
        "adam_steps": 20000,
        "n_colloc": 8192,
        "n_bc": 1024,
        "classical_hidden": [64, 64],   # 5,059 params
        "qt_hidden": [32, 32],           # 1,507 params (3.4× smaller)
        "n_qubits": 8,
        "n_layers": 8,
        "bottleneck_width": 64,
        "lambda_data": 10.0,
        "lambda_pde": 1.0,
        "lambda_bc": 0.0,
        "lr": 0.002,
        "log_every": 500,
        "eval_every": 1000,
        "resample_every": 200,
        "budget_s": 7200,
        "qc_arch": "expect",
        "t_weight": True,
        "t_sample": "tail",             # more points near t=T
    },
}


class DirectNSMLP(nn.Module):
    """Classical PINN: (x,y,t) -> (u,v,p). Hard IC."""

    def __init__(
        self,
        hidden: tuple[int, int] = (64, 64),
        hard_ic: bool = True,
        t_max: float = DEMO_T_MAX,
    ) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        self.fourier = FourierFeatureMapTGV(t_max=t_max)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(self.fourier.out_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 3),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.net(self.fourier(torch.stack([x, y, t], dim=-1)))
        if not self.hard_ic:
            return raw
        u0 = torch.sin(x) * torch.cos(y)
        v0 = -torch.cos(x) * torch.sin(y)
        p0 = (torch.cos(2.0 * x) + torch.cos(2.0 * y)) / 4.0
        return torch.stack([u0, v0, p0], dim=-1) + t.unsqueeze(-1) * raw


def make_colloc(
    n: int,
    device: torch.device,
    t_max: float,
    t_sample: str = "uniform",
):
    """Collocation points. t_sample='tail' uses Beta(2,1) (more mass near t=T)."""
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    u = torch.empty(n, device=device).uniform_(0.0, 1.0)
    if t_sample == "tail":
        t = (t_max * u.sqrt()).requires_grad_(True)  # CDF u=t² → denser at large t
    else:
        t = (t_max * u).requires_grad_(True)
    return x, y, t


def make_bc(n: int, device: torch.device):
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    return x, y, torch.zeros(n, device=device)


def data_loss(
    uvp: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float,
    t_weight: bool = False,
) -> torch.Tensor:
    """MSE to exact TGV.

    t_weight=True scales each point by exp(+2νt) so that the decayed late-time
    tail contributes the same to the loss as the t≈0 region. Without this,
    uniform-t sampling under-penalises errors at large t because the field
    amplitude e^{-2νt} is small there.
    """
    u_ex, v_ex, p_ex = exact_solution(x, y, t, nu)
    residuals = (
        (uvp[:, 0] - u_ex).pow(2)
        + (uvp[:, 1] - v_ex).pow(2)
        + (uvp[:, 2] - p_ex).pow(2)
    )
    if t_weight:
        w = torch.exp(2.0 * nu * t.detach()).clamp(max=20.0)
        return (w * residuals).mean()
    return residuals.mean()


def pde_loss(uvp: torch.Tensor, x, y, t, nu: float) -> torch.Tensor:
    u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]
    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    p_x = _grad(p, x); p_y = _grad(p, y)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y)
    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return f_c.pow(2).mean() + f_u.pow(2).mean() + f_v.pow(2).mean()


def predict(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if weights is None:
        return model(x, y, t)
    return model(x, y, t, weights)


@torch.no_grad()
def eval_rel_l2(
    model: nn.Module,
    device: torch.device,
    nu: float,
    t_max: float,
    weights: dict[str, torch.Tensor] | None = None,
    n_grid: int = 32,
) -> dict[str, dict[str, float]]:
    """Velocity + gauge-fixed pressure rel-L2 at t=0, T/2, T.

    Also measures inference ms/frame on a 256² grid.
    """
    was_training = model.training
    model.eval()
    xs = torch.linspace(X_LO, X_HI, n_grid, device=device)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x, y = xg.flatten(), yg.flatten()
    out: dict[str, dict[str, float]] = {}
    for frac in EVAL_TIMES:
        tv = frac * t_max
        t = torch.full_like(x, tv)
        pred = predict(model, x, y, t, weights)
        u_ex, v_ex, p_ex = exact_solution(x, y, t, nu)
        u_e = relative_l2(pred[:, 0], u_ex)
        v_e = relative_l2(pred[:, 1], v_ex)
        out[f"{frac:g}T"] = {
            "t": tv,
            "u": u_e,
            "v": v_e,
            "p": relative_l2_gauge(pred[:, 2], p_ex),
            "vel": 0.5 * (u_e + v_e),
        }

    # Inference timing: 256² forward pass
    x256 = torch.empty(256 * 256, device=device).uniform_(X_LO, X_HI)
    y256 = torch.empty(256 * 256, device=device).uniform_(X_LO, X_HI)
    t256 = torch.full_like(x256, t_max / 2)
    w256 = {k: v.detach() for k, v in weights.items()} if weights else None
    for _ in range(5):  # warmup
        predict(model, x256, y256, t256, w256)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        predict(model, x256, y256, t256, w256)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    out["_ms_per_frame_256sq"] = ms

    if was_training:
        model.train()
    return out


def vel_max(errs: dict[str, dict[str, float]]) -> float:
    return max(v["vel"] for k, v in errs.items() if not k.startswith("_"))


def gate_pass(errs: dict[str, dict[str, float]], limit: float = 0.02) -> bool:
    return vel_max(errs) <= limit


def prepare_run_dir(run_id: str, overwrite: bool) -> Path:
    run_dir = Path("checkpoints") / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"exists: {run_dir}  (pass --overwrite)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def write_run(run_dir: Path, config: dict, results: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
