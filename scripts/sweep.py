"""Hyperparameter sweep: lambda_bc × adam_lr grid.

Runs all combinations with reduced steps for speed, saves results to
checkpoints/sweep_YYYYMMDD_HHMMSS.csv and prints a ranked summary table.

Usage: python scripts/sweep.py
"""

import sys, itertools, csv, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.qnn_generator import QuantumWeightGenerator
from pdes.burgers2d.physics_loss import compute_burgers_loss
from qt_pinn.fourier import FourierFeatureMap
from qt_pinn.config_loader import load as _load
from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP

# ── Sweep grid ───────────────────────────────────────────────────────────────
# Phase 1 (λ × lr): already run — winner was λ=1, lr=0.01
# Phase 2 (partition_seed): fixed best λ/lr, sweep random basis assignments
SWEEP_MODE = "lp"   # "full" | "partition" | "n_bc" | "adaptive_lambda" | "sigma" | "lp"

LAMBDA_BC_VALUES      = [1.0, 2.0, 3.0, 5.0]
ADAM_LR_VALUES        = [1e-3, 3e-3, 5e-3, 1e-2]
PARTITION_SEED_VALUES = [0, 1, 2, 42]
N_BC_VALUES           = [32, 64, 128, 256, 512]
SIGMA_VALUES          = [1.0, 1.5, 2.0, 3.0, 5.0]  # Fourier B matrix bandwidth

# Adaptive lambda sweep grid
LAMBDA_MAX_VALUES = [1.2, 1.5, 2.0, 2.5, 3.0]
WARMUP_VALUES     = [500, 1000, 2000]
ALPHA_VALUES      = [0.5, 0.9, 0.99]

# Best config from previous sweeps
BEST_LAMBDA_BC      = 1.0
BEST_ADAM_LR        = 1e-2
BEST_PARTITION_SEED = 0

# ── Per-trial settings (reduced from full run for speed) ─────────────────────
SWEEP_STEPS        = 3000   # adaptive lambda needs more steps: warmup + EMA convergence
ADAPTIVE_LAM_STEPS = 3000   # separate override for adaptive_lambda mode
N_COLLOC    = 256
N_BC        = 64
ETA_MIN     = 1e-5
SEED        = 0


def _make_colloc(n: int):
    x = torch.FloatTensor(n).uniform_(-1, 1).requires_grad_(True)
    y = torch.FloatTensor(n).uniform_(-1, 1).requires_grad_(True)
    t = torch.FloatTensor(n).uniform_(0, 1).requires_grad_(True)
    return x, y, t


def _make_bc(n: int):
    x = torch.FloatTensor(n).uniform_(-1, 1)
    y = torch.FloatTensor(n).uniform_(-1, 1)
    t = torch.zeros(n)
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    return x, y, t, u, v


def run_trial(lam: float, lr: float, partition_seed: int = 0, n_bc: int = N_BC,
              adaptive: bool = False, lam_max: float = 5.0,
              warmup: int = 500, alpha: float = 0.9, steps: int = SWEEP_STEPS,
              sigma: float | None = None, use_lp: bool = False) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model  = TargetPINN()
    if sigma is not None:
        fourier_seed = _load()["fourier"]["seed"]
        model.fourier = FourierFeatureMap(sigma=sigma, seed=fourier_seed)
    gen = QuantumWeightGeneratorLP() if use_lp else QuantumWeightGenerator(seed=partition_seed)
    params = list(gen.parameters())

    x, y, t               = _make_colloc(N_COLLOC)
    xb, yb, tb, ub, vb   = _make_bc(n_bc)

    opt   = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=ETA_MIN)

    cur_lam = lam
    pde_val = bc_val = 0.0
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)

        if adaptive and step >= warmup:
            # Wang et al. 2020: λ̂ = max|∇pde| / mean|∇bc|, capped at lam_max
            g_pde = torch.autograd.grad(pde, params, retain_graph=True, allow_unused=True)
            g_bc  = torch.autograd.grad(bc,  params, retain_graph=True, allow_unused=True)
            max_p  = max(g.abs().max() for g in g_pde if g is not None)
            mean_b = torch.cat([g.flatten() for g in g_bc if g is not None]).abs().mean()
            lam_hat = min((max_p / (mean_b + 1e-8)).item(), lam_max)
            cur_lam = (1 - alpha) * cur_lam + alpha * lam_hat

        (pde + cur_lam * bc).backward()
        opt.step()
        sched.step()
        pde_val, bc_val = pde.item(), bc.item()
    elapsed = time.perf_counter() - t0

    return {
        "lambda_bc":      lam,
        "lambda_final":   round(cur_lam, 4),
        "adam_lr":        lr,
        "partition_seed": partition_seed,
        "n_bc":           n_bc,
        "lambda_max":     lam_max,
        "warmup":         warmup,
        "alpha":          alpha,
        "sigma":          sigma,
        "pde":            pde_val,
        "bc":             bc_val,
        "pde+bc":         pde_val + bc_val,
        "weighted":       pde_val + lam * bc_val,
        "elapsed_s":      round(elapsed, 1),
    }


def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}"


def print_grid(results: list[dict], metric: str = "pde+bc") -> None:
    lrs  = sorted(set(r["adam_lr"]   for r in results))
    lams = sorted(set(r["lambda_bc"] for r in results))

    col_w = 12
    print(f"\n── {metric} grid (lower = better) {'─'*30}")
    header = f"{'λ \\ lr':>8}" + "".join(f"{_fmt_lr(lr):>{col_w}}" for lr in lrs)
    print(header)
    for lam in lams:
        row = f"{lam:>8.1f}"
        for lr in lrs:
            val = next(r[metric] for r in results if r["lambda_bc"]==lam and r["adam_lr"]==lr)
            row += f"{val:>{col_w}.5f}"
        print(row)


def main() -> None:
    if SWEEP_MODE == "lp":
        # λ × lr grid using the learned-projection generator
        grid = list(itertools.product([1.0, 2.0, 3.0, 5.0], [1e-3, 3e-3, 5e-3, 1e-2]))
        desc = "learned_proj  λ × lr"
    elif SWEEP_MODE == "partition":
        grid = [(BEST_LAMBDA_BC, BEST_ADAM_LR, s, N_BC) for s in PARTITION_SEED_VALUES]
        desc = f"partition_seed  (fixed λ={BEST_LAMBDA_BC}, lr={_fmt_lr(BEST_ADAM_LR)}, n_bc={N_BC})"
    elif SWEEP_MODE == "n_bc":
        grid = [(BEST_LAMBDA_BC, BEST_ADAM_LR, BEST_PARTITION_SEED, n) for n in N_BC_VALUES]
        desc = f"n_bc  (fixed λ={BEST_LAMBDA_BC}, lr={_fmt_lr(BEST_ADAM_LR)}, seed={BEST_PARTITION_SEED})"
    elif SWEEP_MODE == "sigma":
        grid = [(s,) for s in SIGMA_VALUES]
        desc = f"fourier.sigma  (fixed λ=1.0, lr=1e-02, seed={BEST_PARTITION_SEED})"
    elif SWEEP_MODE == "adaptive_lambda":
        # lambda_max × warmup grid; also do one alpha sweep at best lambda_max
        grid_lm_wu = list(itertools.product(LAMBDA_MAX_VALUES, WARMUP_VALUES))
        al_grid = [(lmx, wu, 0.9) for lmx, wu in grid_lm_wu]
        # alpha sweep at lambda_max=2.0, warmup=1000 (middle of grid)
        al_grid += [(2.0, 1000, a) for a in ALPHA_VALUES if a != 0.9]
        desc = f"adaptive_lambda  (fixed λ_init=1.0, lr=1e-02, steps={ADAPTIVE_LAM_STEPS})"
    else:
        grid = [(lam, lr, 0, N_BC) for lam, lr in itertools.product(LAMBDA_BC_VALUES, ADAM_LR_VALUES)]
        desc = f"λ_bc × adam_lr  (partition_seed=0)"

    if SWEEP_MODE == "lp":
        print(f"Sweep: {desc}")
        print(f"{len(grid)} trials × {SWEEP_STEPS} steps  (n_colloc={N_COLLOC}, n_bc={N_BC})\n")
        print(f"{'#':>3}  {'λ_bc':>6}  {'lr':>8}  {'pde':>12}  {'bc':>12}  {'pde+bc':>12}  {'t(s)':>6}")
        print("─" * 72)
        results = []
        for i, (lam, lr) in enumerate(grid, 1):
            print(f"{i:3d}  {lam:6.1f}  {_fmt_lr(lr):>8}  ", end="", flush=True)
            r = run_trial(lam, lr, use_lp=True)
            results.append({**r, "lambda_bc": lam, "adam_lr": lr,
                             "partition_seed": 0, "n_bc": N_BC})
            print(f"{r['pde']:12.6f}  {r['bc']:12.6f}  {r['pde+bc']:12.6f}  {r['elapsed_s']:6.1f}s")

    elif SWEEP_MODE == "adaptive_lambda":
        print(f"Sweep: {desc}")
        print(f"{len(al_grid)} trials × {ADAPTIVE_LAM_STEPS} steps  (n_colloc={N_COLLOC})\n")
        print(f"{'#':>3}  {'lam_max':>8}  {'warmup':>8}  {'alpha':>6}  {'λ_final':>8}  "
              f"{'pde':>12}  {'bc':>12}  {'pde+bc':>12}  {'t(s)':>6}")
        print("─" * 98)

        results = []
        for i, (lmx, wu, alp) in enumerate(al_grid, 1):
            print(f"{i:3d}  {lmx:8.2f}  {wu:8d}  {alp:6.2f}  ", end="", flush=True)
            r = run_trial(BEST_LAMBDA_BC, BEST_ADAM_LR, BEST_PARTITION_SEED, N_BC,
                          adaptive=True, lam_max=lmx, warmup=wu, alpha=alp,
                          steps=ADAPTIVE_LAM_STEPS)
            results.append(r)
            print(f"{r['lambda_final']:8.3f}  {r['pde']:12.6f}  {r['bc']:12.6f}  "
                  f"{r['pde+bc']:12.6f}  {r['elapsed_s']:6.1f}s")
    elif SWEEP_MODE == "sigma":
        print(f"Sweep: {desc}")
        print(f"{len(SIGMA_VALUES)} trials × {SWEEP_STEPS} steps  (n_colloc={N_COLLOC}, n_bc={N_BC})\n")
        print(f"{'#':>3}  {'sigma':>7}  {'pde':>12}  {'bc':>12}  {'pde+bc':>12}  {'t(s)':>6}")
        print("─" * 60)

        results = []
        for i, (sig,) in enumerate(grid, 1):
            print(f"{i:3d}  {sig:7.2f}  ", end="", flush=True)
            r = run_trial(BEST_LAMBDA_BC, BEST_ADAM_LR, BEST_PARTITION_SEED,
                          N_BC, sigma=sig)
            results.append(r)
            print(f"{r['pde']:12.6f}  {r['bc']:12.6f}  {r['pde+bc']:12.6f}  {r['elapsed_s']:6.1f}s")
    else:
        print(f"Sweep: {desc}")
        print(f"{len(grid)} trials × {SWEEP_STEPS} steps  (n_colloc={N_COLLOC}, cosine → {ETA_MIN:.0e})\n")
        print(f"{'#':>3}  {'λ_bc':>6}  {'lr':>8}  {'seed':>6}  {'n_bc':>6}  {'pde':>12}  {'bc':>12}  {'pde+bc':>12}  {'t(s)':>6}")
        print("─" * 88)

        results = []
        for i, (lam, lr, pseed, n_bc) in enumerate(grid, 1):
            print(f"{i:3d}  {lam:6.1f}  {_fmt_lr(lr):>8}  {pseed:>6}  {n_bc:>6}  ", end="", flush=True)
            r = run_trial(lam, lr, pseed, n_bc)
            results.append(r)
            print(f"{r['pde']:12.6f}  {r['bc']:12.6f}  {r['pde+bc']:12.6f}  {r['elapsed_s']:6.1f}s")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    Path("checkpoints").mkdir(exist_ok=True)
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    outcsv = Path(f"checkpoints/sweep_{stamp}.csv")
    with open(outcsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {outcsv}")

    # ── Rankings ─────────────────────────────────────────────────────────────
    by_pde = sorted(results, key=lambda r: r["pde"])
    by_sum = sorted(results, key=lambda r: r["pde+bc"])

    if SWEEP_MODE in ("adaptive_lambda",):
        hdr = f"{'lam_max':>8}  {'warmup':>8}  {'alpha':>6}  {'λ_final':>8}  {'pde':>10}  {'bc':>10}  {'pde+bc':>10}"
        def _row(r):
            return (f"{r['lambda_max']:8.2f}  {r['warmup']:8d}  {r['alpha']:6.2f}  "
                    f"{r['lambda_final']:8.3f}  {r['pde']:10.6f}  {r['bc']:10.6f}  {r['pde+bc']:10.6f}")
    elif SWEEP_MODE == "sigma":
        hdr = f"{'sigma':>7}  {'pde':>12}  {'bc':>12}  {'pde+bc':>12}"
        def _row(r):
            return f"{r['sigma']:7.2f}  {r['pde']:12.6f}  {r['bc']:12.6f}  {r['pde+bc']:12.6f}"
    else:
        print_grid(results, "pde")
        print_grid(results, "bc")
        print_grid(results, "pde+bc")
        hdr = f"{'λ_bc':>6}  {'lr':>8}  {'seed':>6}  {'pde':>10}  {'bc':>10}  {'pde+bc':>10}"
        def _row(r):
            return (f"{r['lambda_bc']:6.1f}  {_fmt_lr(r['adam_lr']):>8}  {r['partition_seed']:>6}  "
                    f"{r['pde']:10.6f}  {r['bc']:10.6f}  {r['pde+bc']:10.6f}")

    print(f"\n── Top 5 by pde {'─'*55}")
    print(hdr)
    for r in by_pde[:5]: print(_row(r))

    print(f"\n── Top 5 by pde+bc {'─'*53}")
    print(hdr)
    for r in by_sum[:5]: print(_row(r))

    best = by_sum[0]
    if SWEEP_MODE == "adaptive_lambda":
        print(f"\n★  Recommended:  lambda_max={best['lambda_max']}  "
              f"warmup={best['warmup']}  alpha={best['alpha']}")
    elif SWEEP_MODE == "sigma":
        print(f"\n★  Recommended:  fourier.sigma={best['sigma']}")
    else:
        print(f"\n★  Recommended:  lambda_bc={best['lambda_bc']}  adam.lr={_fmt_lr(best['adam_lr'])}"
              f"  partition_seed={best['partition_seed']}  n_bc={best['n_bc']}")


if __name__ == "__main__":
    main()
