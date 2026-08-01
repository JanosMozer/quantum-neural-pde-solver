"""Train with learned linear projection generator (Option 2).

Identical to scripts/train.py but uses QuantumWeightGeneratorLP
instead of QuantumWeightGenerator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── swap only this import ────────────────────────────────────────────────────
from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP as QuantumWeightGenerator

import json, torch, numpy as np
from qt_pinn.config_loader import load as _load_cfg
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.physics_loss import compute_burgers_loss

# reuse helpers and constants from train.py
from scripts.train import (
    SEED, N_COLLOC, N_BC, LAMBDA_BC, ADAPTIVE_LAMBDA, ALPHA,
    ADAPT_EVERY, ADAPT_WARMUP, LAMBDA_MAX, LOG_EVERY,
    ADAM_LR, ADAM_STEPS, LBFGS_LR, LBFGS_STEPS, LBFGS_MAX_ITER,
    COSINE_ANNEAL, COSINE_ETA_MIN, WARMUP_STEPS,
    _next_run_dir, make_colloc, make_bc, forward_losses, adaptive_lambda,
)
from qt_pinn.config_loader import load as _cfg_lp
_t = _cfg_lp()["training"]
WEIGHT_DECAY    = _t["adam"].get("weight_decay", 0.0)
RESAMPLE_EVERY  = _t.get("resample_every", 0)
STRUCTURED_BC   = _t.get("structured_bc", False)
WEIGHT_REG      = _t.get("weight_reg", 0.0)  # L2 reg on generated MLP weights


def main_lp() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_id, run_dir = _next_run_dir()
    run_dir.mkdir(parents=True)
    print(f"Run (LP): {run_id}  →  {run_dir}/")

    model  = TargetPINN()
    gen    = QuantumWeightGenerator()
    params = list(gen.parameters())

    n_total = sum(p.numel() for p in gen.parameters())
    n_q     = gen.q_weights.numel()
    print(f"Params: {n_total:,}  (circuit={n_q}, proj={n_total - n_q})")

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
        reg = sum(w.pow(2).mean() for w in weights.values()) if WEIGHT_REG > 0 else 0.0
        loss = pde + lam * bc + WEIGHT_REG * reg
        loss.backward()
        opt.step()
        return loss.item(), pde.item(), bc.item()

    # ── Adam with optional warmup + cosine annealing ───────────────────────
    opt = torch.optim.Adam(params, lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    if COSINE_ANNEAL:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=max(WARMUP_STEPS, 1))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(ADAM_STEPS - WARMUP_STEPS, 1), eta_min=COSINE_ETA_MIN)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[WARMUP_STEPS])
    else:
        sched = None
    lr_end = COSINE_ETA_MIN if COSINE_ANNEAL else ADAM_LR
    print(f"\nAdam  lr=0→{ADAM_LR}→{lr_end}  warmup={WARMUP_STEPS}  steps={ADAM_STEPS}  cosine={COSINE_ANNEAL}")
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'λ':>8}  {'lr':>10}")
    for step in range(ADAM_STEPS):
        if RESAMPLE_EVERY > 0 and step % RESAMPLE_EVERY == 0 and step > 0:
            x, y, t = make_colloc(N_COLLOC)
            x_bc, y_bc, t_bc, u_bc, v_bc = make_bc(N_BC, structured=STRUCTURED_BC)
        total, pde, bc = _step(opt, step)
        if sched: sched.step()
        if step % LOG_EVERY == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"{step:6d}  {total:12.7f}  {pde:12.7f}  {bc:12.7f}  {lam:8.4f}  {lr_now:.2e}")

    # ── L-BFGS ───────────────────────────────────────────────────────────────
    if LBFGS_STEPS > 0:
        opt3    = torch.optim.LBFGS(params, lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                                     history_size=10, line_search_fn="strong_wolfe")
        counter = [0]
        print(f"\nL-BFGS  lr={LBFGS_LR}  steps={LBFGS_STEPS}")
        print(f"{'closure':>7}  {'total':>12}  {'pde':>12}  {'bc':>12}")

        def closure():
            nonlocal lam
            opt3.zero_grad()
            pde, bc = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc)
            loss = pde + lam * bc
            loss.backward()
            if counter[0] % (LOG_EVERY * 2) == 0:
                print(f"{counter[0]:7d}  {loss.item():12.7f}  {pde.item():12.7f}  {bc.item():12.7f}")
            counter[0] += 1
            return loss

        for _ in range(LBFGS_STEPS):
            opt3.step(closure)

    # ── Final eval ───────────────────────────────────────────────────────────
    pde_f, bc_f = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc)
    pde_final = pde_f.item(); bc_final = bc_f.item()
    print(f"\nFinal  pde={pde_final:.7f}  bc={bc_final:.7f}  sum={pde_final+bc_final:.7f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    torch.save(gen.state_dict(), run_dir / "q_weights.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "generator": "learned_proj",
        "seed": SEED, "n_colloc": N_COLLOC, "n_bc": N_BC,
        "adam_lr": ADAM_LR, "adam_steps": ADAM_STEPS,
        "lambda_bc_init": LAMBDA_BC, "lambda_bc_final": round(lam, 4),
        "cosine": COSINE_ANNEAL, "weight_decay": WEIGHT_DECAY,
        "bottleneck_width": gen.proj[0].out_features,  # was unrecoverable except by reverse-engineering param counts
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id,
        "pde_loss": round(pde_final, 8),
        "bc_loss":  round(bc_final, 8),
        "total":    round(pde_final + bc_final, 8),
        "lambda_final": round(lam, 6),
        "n_params": sum(p.numel() for p in gen.parameters()),
    }, indent=2))
    print(f"Saved  {run_dir}/q_weights.pt")
    print(f"       {run_dir}/results.json")


if __name__ == "__main__":
    main_lp()
