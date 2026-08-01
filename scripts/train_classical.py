"""Classical ablation: identical to train_lp.py but uses ClassicalWeightGenerator.

Run this after train_lp.py to compare quantum vs classical under identical conditions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qt_pinn.learned_proj.classical_generator import ClassicalWeightGenerator as QuantumWeightGenerator

# everything else is identical to train_lp.py
import json, torch, numpy as np
from qt_pinn.config_loader import load as _load_cfg
from qt_pinn.pinn_target import TargetPINN
from pdes.burgers2d.physics_loss import compute_burgers_loss
from scripts.train import (
    SEED, N_COLLOC, N_BC, LAMBDA_BC, ADAPTIVE_LAMBDA, ALPHA,
    ADAPT_EVERY, ADAPT_WARMUP, LAMBDA_MAX, LOG_EVERY,
    ADAM_LR, ADAM_STEPS, LBFGS_LR, LBFGS_STEPS, LBFGS_MAX_ITER,
    COSINE_ANNEAL, COSINE_ETA_MIN, WARMUP_STEPS,
    _next_run_dir, make_colloc, make_bc, adaptive_lambda, make_optimizer,
)
from qt_pinn.config_loader import load as _cfg_cl
_t = _cfg_cl()["training"]
WEIGHT_DECAY    = _t["adam"].get("weight_decay", 0.0)
RESAMPLE_EVERY  = _t.get("resample_every", 0)
STRUCTURED_BC   = _t.get("structured_bc", False)
WEIGHT_REG      = _t.get("weight_reg", 0.0)


def main_classical() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_id, run_dir = _next_run_dir()
    run_dir.mkdir(parents=True)
    print(f"Run (Classical): {run_id}  ->  {run_dir}/")

    model  = TargetPINN()
    gen    = QuantumWeightGenerator()
    params = list(gen.parameters())

    n_total = sum(p.numel() for p in gen.parameters())
    print(f"Params: {n_total:,}  (latent={gen.latent.numel()}, proj={n_total - gen.latent.numel()})")

    x, y, t               = make_colloc(N_COLLOC)
    x_bc, y_bc, t_bc, u_bc, v_bc = make_bc(N_BC, structured=STRUCTURED_BC)
    lam = float(LAMBDA_BC)

    def _step(opt, step):
        nonlocal lam
        opt.zero_grad()
        weights = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, weights)
        if ADAPTIVE_LAMBDA and step >= ADAPT_WARMUP and step % ADAPT_EVERY == 0:
            lam = adaptive_lambda(pde, bc, params, lam, ALPHA)
        reg  = sum(w.pow(2).mean() for w in weights.values()) if WEIGHT_REG > 0 else 0.0
        loss = pde + lam * bc + WEIGHT_REG * reg
        loss.backward()
        opt.step()
        return loss.item(), pde.item(), bc.item()

    opt, sched = make_optimizer(params, ADAM_LR, WEIGHT_DECAY)

    print(f"\nAdam  lr=0->{ADAM_LR}->{COSINE_ETA_MIN if COSINE_ANNEAL else ADAM_LR}  steps={ADAM_STEPS}")
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'lam':>8}  {'lr':>10}")
    for step in range(ADAM_STEPS):
        if RESAMPLE_EVERY > 0 and step % RESAMPLE_EVERY == 0 and step > 0:
            x, y, t = make_colloc(N_COLLOC)
        total, pde, bc = _step(opt, step)
        if sched: sched.step()
        if step % LOG_EVERY == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"{step:6d}  {total:12.7f}  {pde:12.7f}  {bc:12.7f}  {lam:8.4f}  {lr_now:.2e}")

    pde_f, bc_f = compute_burgers_loss(
        model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, gen())[0].item(), \
        compute_burgers_loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, gen())[1].item()
    print(f"\nFinal  pde={pde_f:.7f}  bc={bc_f:.7f}  sum={pde_f+bc_f:.7f}")

    torch.save(gen.state_dict(), run_dir / "q_weights.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "generator": "classical",
        "seed": SEED, "n_colloc": N_COLLOC, "n_bc": N_BC,
        "adam_lr": ADAM_LR, "adam_steps": ADAM_STEPS,
        "lambda_bc_init": LAMBDA_BC, "lambda_bc_final": round(lam, 4),
        "weight_reg": WEIGHT_REG,
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id, "generator": "classical",
        "pde_loss": round(pde_f, 8), "bc_loss": round(bc_f, 8),
        "total": round(pde_f + bc_f, 8),
        "n_params": n_total,
    }, indent=2))
    print(f"Saved  {run_dir}/results.json")


if __name__ == "__main__":
    main_classical()
