"""Physics loss for 2D viscous Burgers' equation via automatic differentiation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from qt_pinn.config_loader import load as _load

NU = float(_load()["physics"]["nu"])   # ν = 0.01/π ≈ 0.003183


def ic_values(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IC at t=0 (divergence-free vortex on [-1,1]^2)."""
    u = torch.sin(math.pi * x) * torch.cos(math.pi * y)
    v = -torch.cos(math.pi * x) * torch.sin(math.pi * y)
    return u, v


def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def _model_uv(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    weights: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if weights is None:
        return model(x, y, t)
    return model(x, y, t, weights)


def compute_burgers_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    t_bc: torch.Tensor,
    u_bc: torch.Tensor,
    v_bc: torch.Tensor,
    weights: dict[str, torch.Tensor] | None = None,
    nu: float | None = None,
    return_residuals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PDE residual + IC loss.

    weights=None for direct models (VQC-PINN / classical PINN).
    """
    if nu is None:
        nu = NU

    uvp = _model_uv(model, x, y, t, weights)
    u, v = uvp[:, 0], uvp[:, 1]

    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y)

    f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
    pde_loss = f_u.pow(2).mean() + f_v.pow(2).mean()

    uv_bc = _model_uv(model, x_bc, y_bc, t_bc, weights)
    bc_loss = ((uv_bc[:, 0] - u_bc).pow(2).mean()
               + (uv_bc[:, 1] - v_bc).pow(2).mean())

    if return_residuals:
        return pde_loss, bc_loss, (f_u.pow(2) + f_v.pow(2)).detach()
    return pde_loss, bc_loss


def pde_rms(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    t_bc: torch.Tensor,
    u_bc: torch.Tensor,
    v_bc: torch.Tensor,
    weights: dict[str, torch.Tensor] | None = None,
    nu: float | None = None,
) -> float:
    pde, _ = compute_burgers_loss(
        model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc,
        weights=weights, nu=nu,
    )
    return math.sqrt(pde.item())


if __name__ == "__main__":
    from qt_pinn.pinn_target import TargetPINN
    from qt_pinn.qnn_generator import QuantumWeightGenerator

    model = TargetPINN()
    gen = QuantumWeightGenerator()
    weights = gen()
    for w in weights.values():
        w.retain_grad()

    N, M = 32, 16
    x = torch.rand(N, requires_grad=True)
    y = torch.rand(N, requires_grad=True)
    t = torch.rand(N, requires_grad=True)
    x_bc = torch.rand(M)
    y_bc = torch.rand(M)
    t_bc = torch.zeros(M)
    u_bc, v_bc = ic_values(x_bc, y_bc)

    pde_loss, bc_loss = compute_burgers_loss(
        model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, weights=weights)
    total = pde_loss + bc_loss
    total.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print(f"PASS hypernet: pde={pde_loss.item():.4f} bc={bc_loss.item():.4f}")

    from qt_pinn.burgers_vqc_pinn import BurgersVQCPINN
    vqc = BurgersVQCPINN(4, 2)
    x2 = torch.rand(N, requires_grad=True)
    y2 = torch.rand(N, requires_grad=True)
    t2 = torch.rand(N, requires_grad=True)
    pde2, bc2 = compute_burgers_loss(
        vqc, x2, y2, t2, x_bc, y_bc, t_bc, u_bc, v_bc, weights=None)
    (pde2 + bc2).backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0
    print(f"PASS VQC-PINN: pde={pde2.item():.4f} bc={bc2.item():.4f}")
