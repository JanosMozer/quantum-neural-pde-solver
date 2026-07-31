"""2-stage training: Adam -> L-BFGS. Saves run to checkpoints/run_NNNN/. Run from the repo root."""

from pathlib import Path

import json
import torch
import numpy as np
from qt_pinn.config_loader import load as _load_cfg
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.qnn_generator import QuantumWeightGenerator
from qt_pinn.physics_loss import compute_burgers_loss

# ── Load all parameters from config.yaml ────────────────────────────────────
_cfg            = _load_cfg()["training"]
SEED            = _cfg["seed"]
N_COLLOC        = _cfg["n_colloc"]
N_BC            = _cfg["n_bc"]
LAMBDA_BC       = _cfg["lambda_bc"]
ADAPTIVE_LAMBDA = _cfg.get("adaptive_lambda", True)   # disable to fix lambda
LOG_EVERY       = _cfg["log_every"]
ADAM_LR         = _cfg["adam"]["lr"]
ADAM_STEPS      = _cfg["adam"]["steps"]
ADAM2_LR        = _cfg["adam2"]["lr"]
ADAM2_STEPS     = _cfg["adam2"]["steps"]
LBFGS_LR        = _cfg["lbfgs"]["lr"]
LBFGS_STEPS     = _cfg["lbfgs"]["steps"]
LBFGS_MAX_ITER  = _cfg["lbfgs"]["max_iter"]


def _next_run_dir(base: Path = Path("checkpoints")) -> tuple[str, Path]:
    """Auto-increment run ID: checkpoints/run_0001, run_0002, ..."""
    base.mkdir(exist_ok=True)
    existing = sorted(base.glob("run_*"))
    n = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_id = f"run_{n:04d}"
    return run_id, base / run_id


def make_colloc(n: int) -> tuple[torch.Tensor, ...]:
    x = torch.FloatTensor(n).uniform_(-1, 1).requires_grad_(True)
    y = torch.FloatTensor(n).uniform_(-1, 1).requires_grad_(True)
    t = torch.FloatTensor(n).uniform_(0, 1).requires_grad_(True)
    return x, y, t


def make_bc(n: int) -> tuple[torch.Tensor, ...]:
    """IC at t=0: u=sin(πx)cos(πy), v=-cos(πx)sin(πy)."""
    x = torch.FloatTensor(n).uniform_(-1, 1)
    y = torch.FloatTensor(n).uniform_(-1, 1)
    t = torch.zeros(n)
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    return x, y, t, u, v


def _grad_norm(loss: torch.Tensor, params: list) -> float:
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.stack([g.norm() for g in gs if g is not None]).mean().item()


def adaptive_lambda(gen, model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc,
                    current: float, alpha: float = 0.9) -> float:
    """Gradient-norm ratio λ = ‖∇pde‖ / ‖∇bc‖, smoothed with EMA α."""
    params = [p for p in gen.parameters() if p.requires_grad]
    w = gen()
    pde, bc = compute_burgers_loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, w)
    new_lam = _grad_norm(pde, params) / (_grad_norm(bc, params) + 1e-8)
    return alpha * current + (1 - alpha) * new_lam   # EMA — prevents jumping


def loss_fn(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc,
            lam: float) -> tuple[torch.Tensor, float, float]:
    weights = gen()
    pde, bc = compute_burgers_loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, weights)
    return pde + lam * bc, pde.item(), bc.item()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_id, run_dir = _next_run_dir()
    run_dir.mkdir(parents=True)
    print(f"Run: {run_id}  →  {run_dir}/")

    model  = TargetPINN()
    gen    = QuantumWeightGenerator(seed=_load_cfg()["quantum"].get("partition_seed", 0))
    params = list(gen.parameters())

    x, y, t                  = make_colloc(N_COLLOC)
    x_bc, y_bc, t_bc, u_bc, v_bc = make_bc(N_BC)

    lam = float(LAMBDA_BC)   # mutable lambda, starts from config value

    # ── Stage 1: Adam ──────────────────────────────────────────────────────
    opt = torch.optim.Adam(params, lr=ADAM_LR)
    print(f"\nStage 1  Adam  lr={ADAM_LR}  steps={ADAM_STEPS}")
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'λ_bc':>8}")
    for step in range(ADAM_STEPS):
        # adaptive lambda every 50 steps
        if ADAPTIVE_LAMBDA and step % 50 == 0 and step > 0:
            lam = adaptive_lambda(gen, model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, lam)
        opt.zero_grad()
        loss, pde, bc = loss_fn(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, lam)
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0:
            print(f"{step:6d}  {loss.item():12.7f}  {pde:12.7f}  {bc:12.7f}  {lam:8.2f}")

    # ── Stage 2: Adam (fine refinement) ────────────────────────────────────
    opt2 = torch.optim.Adam(params, lr=ADAM2_LR)
    print(f"\nStage 2  Adam  lr={ADAM2_LR}  steps={ADAM2_STEPS}")
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'λ_bc':>8}")
    for step in range(ADAM2_STEPS):
        if ADAPTIVE_LAMBDA and step % 50 == 0 and step > 0:
            lam = adaptive_lambda(gen, model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, lam)
        opt2.zero_grad()
        loss, pde, bc = loss_fn(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, lam)
        loss.backward()
        opt2.step()
        if step % LOG_EVERY == 0:
            print(f"{step:6d}  {loss.item():12.7f}  {pde:12.7f}  {bc:12.7f}  {lam:8.2f}")

    # ── Stage 3: L-BFGS (short cleanup) ────────────────────────────────────
    opt2    = torch.optim.LBFGS(params, lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                                 history_size=10, line_search_fn="strong_wolfe")
    counter = [0]
    print(f"\nStage 2  L-BFGS  lr={LBFGS_LR}  steps={LBFGS_STEPS}")
    print(f"{'closure':>7}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'λ_bc':>8}")

    def closure() -> torch.Tensor:
        nonlocal lam
        opt2.zero_grad()
        if ADAPTIVE_LAMBDA and counter[0] % 100 == 0 and counter[0] > 0:
            lam = adaptive_lambda(gen, model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, lam)
        loss, pde, bc = loss_fn(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, lam)
        loss.backward()
        if counter[0] % (LOG_EVERY * 2) == 0:
            print(f"{counter[0]:7d}  {loss.item():12.7f}  {pde:12.7f}  {bc:12.7f}  {lam:8.2f}")
        counter[0] += 1
        return loss

    for _ in range(LBFGS_STEPS):
        opt2.step(closure)

    # ── Save ────────────────────────────────────────────────────────────────
    torch.save(gen.state_dict(), run_dir / "q_weights.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "seed": SEED, "n_colloc": N_COLLOC, "n_bc": N_BC,
        "adam_lr": ADAM_LR, "adam_steps": ADAM_STEPS,
        "lbfgs_lr": LBFGS_LR, "lbfgs_steps": LBFGS_STEPS,
        "lambda_bc_init": LAMBDA_BC, "lambda_bc_final": round(lam, 4),
    }, indent=2))
    print(f"\nSaved  {run_dir}/q_weights.pt")
    print(f"       {run_dir}/config.json")


if __name__ == "__main__":
    main()
