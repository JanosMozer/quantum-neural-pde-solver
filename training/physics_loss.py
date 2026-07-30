"""Physics loss for 2D viscous Burgers' equation via automatic differentiation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from training.config_loader import load as _load

NU = _load()["physics"]["nu"]   # kinematic viscosity; lower = sharper gradients


def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """First-order partial derivative of scalar field output w.r.t. inp."""
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


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
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PDE residual + BC loss.

    Args:
        model: TargetPINN instance
        x, y, t: collocation points (N,), requires_grad=True
        x_bc, y_bc, t_bc: boundary/IC points (M,)
        u_bc, v_bc: boundary values (M,)
        weights: weight dict from QuantumWeightGenerator

    Returns:
        (pde_loss, bc_loss): scalar tensors
    """
    # PDE residuals at collocation points
    uvp = model(x, y, t, weights)  # (N, 2)
    u = uvp[:, 0]
    v = uvp[:, 1]

    # First-order derivatives
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    v_t = _grad(v, t)
    v_x = _grad(v, x)
    v_y = _grad(v, y)

    # Second-order derivatives
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x)
    v_yy = _grad(v_y, y)

    # Burgers residuals
    f_u = u_t + u * u_x + v * u_y - NU * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y - NU * (v_xx + v_yy)

    pde_loss = torch.mean(f_u ** 2) + torch.mean(f_v ** 2)

    # Boundary / IC loss
    uv_bc = model(x_bc, y_bc, t_bc, weights)  # (M, 2)
    bc_loss = torch.mean((uv_bc[:, 0] - u_bc) ** 2) + torch.mean((uv_bc[:, 1] - v_bc) ** 2)

    return pde_loss, bc_loss


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.pinn_target import TargetPINN
    from training.qnn_generator import QuantumWeightGenerator

    model = TargetPINN()
    gen = QuantumWeightGenerator()
    weights = gen()

    # Make weights require grad so we can test backprop
    for w in weights.values():
        w.retain_grad()

    N, M = 32, 16
    x = torch.rand(N, requires_grad=True)
    y = torch.rand(N, requires_grad=True)
    t = torch.rand(N, requires_grad=True)
    x_bc = torch.rand(M)
    y_bc = torch.rand(M)
    t_bc = torch.zeros(M)
    u_bc = torch.zeros(M)
    v_bc = torch.zeros(M)

    pde_loss, bc_loss = compute_burgers_loss(
        model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, weights
    )
    total = pde_loss + bc_loss
    total.backward()

    assert x.grad is not None and x.grad.abs().sum() > 0, "No gradient on x"
    print(f"PASS: pde_loss={pde_loss.item():.4f}, bc_loss={bc_loss.item():.4f}")
    print(f"      x.grad norm = {x.grad.norm().item():.4f}")
