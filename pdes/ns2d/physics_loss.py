"""Physics loss for 2D unsteady incompressible Navier-Stokes via automatic differentiation.

PDE system (primitive variables u, v, p):
  Continuity:   u_x + v_y = 0
  Momentum x:   u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
  Momentum y:   v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0

IC: Taylor-Green vortex decay on [0, 2pi]^2, t in [0, 1]
  u(x,y,0) =  sin(x)*cos(y)
  v(x,y,0) = -cos(x)*sin(y)
  p(x,y,0) = +(cos(2x)+cos(2y)) / 4

Exact solution (used for IC enforcement and validation):
  u(x,y,t) =  sin(x)*cos(y)*exp(-2*nu*t)
  v(x,y,t) = -cos(x)*sin(y)*exp(-2*nu*t)
  p(x,y,t) = +(cos(2x)+cos(2y))*exp(-4*nu*t) / 4

Pressure sign: for this velocity convention, advection gives
  (u·∇)u|_x = +½ sin(2x) D², so momentum balance requires
  p_x = -½ sin(2x) D² ⇒ p = +¼ (cos(2x)+cos(2y)) D².
  The textbook minus-sign formula belongs to the opposite velocity
  convention (u=-cos x sin y). Using the wrong sign makes the exact
  solution inconsistent with the coded residual (mom RMS ~0.7).

The pressure equation is not time-integrated separately; it enters through
the incompressibility constraint and the momentum residuals. The PINN learns
p as a direct output field.

Why harder than Burgers:
  - 3 coupled output fields vs 2
  - pressure-velocity coupling: continuity acts as a hard algebraic constraint
    that must be satisfied simultaneously with both momentum equations
  - pressure has no explicit IC/BC in the PDE — the network must infer it from
    the momentum residuals and the IC value alone (this is the PINN analogue
    of the pressure Poisson problem in classical NS solvers)
  - the exponential decay factor exp(-2*nu*t) means the network must learn
    a non-trivial multiplicative time dependence, not just spatial structure
"""

import math
import torch
import torch.nn as nn


NU = 0.01   # default kinematic viscosity, kept for back-compat with older scripts.
            # New code should pass nu explicitly: the parametric study varies it.


def exact_solution(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float | torch.Tensor = NU,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Taylor-Green exact solution. Returns (u, v, p).

    nu may be a scalar or a tensor broadcastable against t, so a batch of
    tasks with different viscosities can be evaluated in one call.
    """
    decay   = torch.exp(-2.0 * nu * t)
    decay2  = torch.exp(-4.0 * nu * t)
    u =  torch.sin(x) * torch.cos(y) * decay
    v = -torch.cos(x) * torch.sin(y) * decay
    p = +(torch.cos(2.0 * x) + torch.cos(2.0 * y)) * decay2 / 4.0
    return u, v, p


def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def compute_ns_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    t_bc: torch.Tensor,
    weights: dict[str, torch.Tensor],
    return_residuals: bool = False,
    nu: float | torch.Tensor = NU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute NS PDE residual + IC loss.

    Args:
        model: TargetPINNNS instance (outputs u, v, p as 3-vector)
        x, y, t: collocation points (N,), requires_grad=True
        x_bc, y_bc, t_bc: IC points (M,) at t=0
        weights: weight dict from generator
        nu: kinematic viscosity for this task

    Returns:
        (pde_loss, bc_loss) — pde_loss = continuity + momentum_x + momentum_y residuals
    """
    uvp = model(x, y, t, weights)   # (N, 3)
    u = uvp[:, 0]
    v = uvp[:, 1]
    p = uvp[:, 2]

    # --- first-order derivatives ---
    u_t = _grad(u, t);  u_x = _grad(u, x);  u_y = _grad(u, y)
    v_t = _grad(v, t);  v_x = _grad(v, x);  v_y = _grad(v, y)
    p_x = _grad(p, x);  p_y = _grad(p, y)

    # --- second-order derivatives ---
    u_xx = _grad(u_x, x);  u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x);  v_yy = _grad(v_y, y)

    # --- PDE residuals ---
    # continuity (incompressibility)
    f_c = u_x + v_y

    # momentum x: u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)

    # momentum y: v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    pde_loss = (torch.mean(f_c ** 2)
                + torch.mean(f_u ** 2)
                + torch.mean(f_v ** 2))

    # --- IC loss at t=0 ---
    uvp_bc = model(x_bc, y_bc, t_bc, weights)   # (M, 3)
    u_bc_exact, v_bc_exact, p_bc_exact = exact_solution(x_bc, y_bc, t_bc, nu)
    bc_loss = (torch.mean((uvp_bc[:, 0] - u_bc_exact) ** 2)
               + torch.mean((uvp_bc[:, 1] - v_bc_exact) ** 2)
               + torch.mean((uvp_bc[:, 2] - p_bc_exact) ** 2))

    if return_residuals:
        per_point = (f_c ** 2 + f_u ** 2 + f_v ** 2).detach()
        return pde_loss, bc_loss, per_point
    return pde_loss, bc_loss


def relative_l2(pred: torch.Tensor, exact: torch.Tensor) -> float:
    return (pred - exact).norm().item() / (exact.norm().item() + 1e-10)


def relative_l2_gauge(pred: torch.Tensor, exact: torch.Tensor) -> float:
    """Relative L2 after removing the spatial mean from both fields.

    Incompressible NS constrains only the pressure gradient, so p is defined
    up to an additive function of t. A network that learns p + C(t) has zero
    PDE residual but arbitrarily large raw L2 error. For pressure this is the
    physically meaningful metric; for velocity the two agree (mean is ~0).
    """
    p = pred - pred.mean()
    e = exact - exact.mean()
    return (p - e).norm().item() / (e.norm().item() + 1e-10)


if __name__ == "__main__":
    # A1 residual check: exact solution must satisfy the coded PDE
    N = 2000
    x = torch.FloatTensor(N).uniform_(0, 2 * math.pi).requires_grad_(True)
    y = torch.FloatTensor(N).uniform_(0, 2 * math.pi).requires_grad_(True)
    t = torch.FloatTensor(N).uniform_(0, 1).requires_grad_(True)
    u, v, p = exact_solution(x, y, t)
    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    p_x = _grad(p, x); p_y = _grad(p, y)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y)
    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    rms_c = f_c.pow(2).mean().sqrt().item()
    rms_u = f_u.pow(2).mean().sqrt().item()
    rms_v = f_v.pow(2).mean().sqrt().item()
    print(f"exact residual RMS: cont={rms_c:.2e}  mom_x={rms_u:.2e}  mom_y={rms_v:.2e}")
    assert rms_c < 1e-5 and rms_u < 1e-5 and rms_v < 1e-5, "A1 FAIL: exact solution inconsistent with PDE"
    print("PASS: A1 exact solution consistent with coded residual")
