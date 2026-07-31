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
ADAPTIVE_LAMBDA = _cfg.get("adaptive_lambda", True)
ALPHA           = _cfg.get("adaptive_lambda_alpha", 0.9)   # EMA momentum (paper: 0.9)
ADAPT_EVERY     = _cfg.get("adaptive_lambda_every", 1)     # update every N steps (paper: every step)
ADAPT_WARMUP    = _cfg.get("adaptive_lambda_warmup", 200)  # steps before annealing starts
LAMBDA_MAX      = _cfg.get("lambda_max", 100.0)            # cap on λ to prevent runaway at init
LOG_EVERY       = _cfg["log_every"]
ADAM_LR         = _cfg["adam"]["lr"]
ADAM_STEPS      = _cfg["adam"]["steps"]
LBFGS_LR        = _cfg["lbfgs"]["lr"]
LBFGS_STEPS     = _cfg["lbfgs"]["steps"]
LBFGS_MAX_ITER  = _cfg["lbfgs"]["max_iter"]
COSINE_ANNEAL   = _cfg.get("cosine_annealing", False)  # CosineAnnealingLR on stage 1 & 2
COSINE_ETA_MIN  = _cfg.get("cosine_eta_min", 1e-5)     # minimum lr at end of cosine cycle


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


def adaptive_lambda(pde: torch.Tensor, bc: torch.Tensor,
                    params: list, current: float, alpha: float = 0.9) -> float:
    """Wang et al. 2020 (arXiv:2001.04536), Algorithm 1.

    λ̂ = max_θ|∇_θ L_pde| / mean_θ|∇_θ L_bc|
    λ  = (1-α)·λ + α·λ̂          (EMA, paper recommends α=0.9)
    """
    g_pde = torch.autograd.grad(pde, params, retain_graph=True, allow_unused=True)
    g_bc  = torch.autograd.grad(bc,  params, retain_graph=True, allow_unused=True)

    max_pde  = max(g.abs().max() for g in g_pde if g is not None)
    mean_bc  = torch.cat([g.flatten() for g in g_bc  if g is not None]).abs().mean()

    lam_hat = min((max_pde / (mean_bc + 1e-8)).item(), LAMBDA_MAX)  # cap prevents init explosion
    return (1 - alpha) * current + alpha * lam_hat


def forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc):
    weights = gen()
    pde, bc = compute_burgers_loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, weights)
    return pde, bc


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

    lam = float(LAMBDA_BC)

    def _step(opt, step):
        """One gradient step with optional paper lambda update."""
        nonlocal lam
        opt.zero_grad()
        pde, bc = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc)
        if ADAPTIVE_LAMBDA and step >= ADAPT_WARMUP and step % ADAPT_EVERY == 0:
            lam = adaptive_lambda(pde, bc, params, lam, ALPHA)
        loss = pde + lam * bc
        loss.backward()
        opt.step()
        return loss.item(), pde.item(), bc.item()

    # ── Adam with optional cosine annealing ────────────────────────────────
    opt = torch.optim.Adam(params, lr=ADAM_LR)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAM_STEPS, eta_min=COSINE_ETA_MIN)
             if COSINE_ANNEAL else None)
    lr_end = COSINE_ETA_MIN if COSINE_ANNEAL else ADAM_LR
    print(f"\nAdam  lr={ADAM_LR}→{lr_end}  steps={ADAM_STEPS}  cosine={COSINE_ANNEAL}")
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'λ_bc':>8}  {'lr':>10}")
    for step in range(ADAM_STEPS):
        total, pde, bc = _step(opt, step)
        if sched: sched.step()
        if step % LOG_EVERY == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"{step:6d}  {total:12.7f}  {pde:12.7f}  {bc:12.7f}  {lam:8.4f}  {lr_now:.2e}")

    # ── Stage 3: L-BFGS ────────────────────────────────────────────────────
    opt3    = torch.optim.LBFGS(params, lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                                 history_size=10, line_search_fn="strong_wolfe")
    counter = [0]
    print(f"\nL-BFGS  lr={LBFGS_LR}  steps={LBFGS_STEPS}")
    print(f"{'closure':>7}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'λ_bc':>8}")

    def closure() -> torch.Tensor:
        nonlocal lam
        opt3.zero_grad()
        pde, bc = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc)
        if ADAPTIVE_LAMBDA and counter[0] % ADAPT_EVERY == 0:
            lam = adaptive_lambda(pde, bc, params, lam, ALPHA)
        loss = pde + lam * bc
        loss.backward()
        if counter[0] % (LOG_EVERY * 2) == 0:
            print(f"{counter[0]:7d}  {loss.item():12.7f}  {pde.item():12.7f}  {bc.item():12.7f}  {lam:8.4f}")
        counter[0] += 1
        return loss

    for _ in range(LBFGS_STEPS):
        opt3.step(closure)

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
