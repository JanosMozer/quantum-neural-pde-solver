"""Physics loss for 2D Kolmogorov flow (forced incompressible NS).

PDE on [0, 2pi]^2 x [0, T] with periodic BC (implicit in Fourier features):

  div u = 0
  u_t + (u.grad)u + grad p = nu lap u + f
  f = (A sin(n y), 0)

IC: u = v = p = 0 at t = 0 (standard PINN setup for forced flow).

There is no closed-form space-time exact solution. Evaluation uses PDE
residual RMS on holdout collocation points, not rel-L2 vs an analytic field.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NU = 0.01
N_FORCE = 4
F_AMP = 1.0


def forcing(
    y: torch.Tensor,
    n: int = N_FORCE,
    amplitude: float = F_AMP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Body force f = (A sin(n y), 0)."""
    fx = amplitude * torch.sin(n * y)
    fy = torch.zeros_like(y)
    return fx, fy


def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def compute_kol_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    t_bc: torch.Tensor,
    weights: dict[str, torch.Tensor],
    nu: float | torch.Tensor = NU,
    n_force: int = N_FORCE,
    f_amp: float = F_AMP,
    return_residuals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PDE residual + IC loss for Kolmogorov flow."""
    uvp = model(x, y, t, weights)
    u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]

    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    p_x = _grad(p, x); p_y = _grad(p, y)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y)

    fx, fy = forcing(y, n_force, f_amp)
    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) - fx
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) - fy

    pde_loss = f_c.pow(2).mean() + f_u.pow(2).mean() + f_v.pow(2).mean()

    uvp_bc = model(x_bc, y_bc, t_bc, weights)
    if getattr(model, "hard_ic", False):
        bc_loss = torch.zeros((), device=x.device)
    else:
        bc_loss = uvp_bc.pow(2).mean()

    if return_residuals:
        per_point = (f_c.pow(2) + f_u.pow(2) + f_v.pow(2)).detach()
        return pde_loss, bc_loss, per_point
    return pde_loss, bc_loss


def pde_rms(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    weights: dict[str, torch.Tensor],
    nu: float,
    n_force: int = N_FORCE,
    f_amp: float = F_AMP,
) -> float:
    """Root-mean-square PDE residual (continuity + momentum)."""
    pde, _ = compute_kol_loss(
        model, x, y, t,
        x[:1], y[:1], t[:1], weights,
        nu=nu, n_force=n_force, f_amp=f_amp,
    )
    return math.sqrt(pde.item())


if __name__ == "__main__":
    # Forcing must appear in x-momentum only
    y = torch.linspace(0, 2 * math.pi, 64)
    fx, fy = forcing(y)
    assert fy.abs().max() < 1e-12
    assert (fx - torch.sin(N_FORCE * y)).abs().max() < 1e-6
    print("PASS Kolmogorov forcing f = (sin(n y), 0)")
