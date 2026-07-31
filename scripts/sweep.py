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
from qt_pinn.physics_loss import compute_burgers_loss
from qt_pinn.config_loader import load as _load

# ── Sweep grid ───────────────────────────────────────────────────────────────
# Phase 1 (λ × lr): already run — winner was λ=1, lr=0.01
# Phase 2 (partition_seed): fixed best λ/lr, sweep random basis assignments
SWEEP_MODE = "n_bc"   # "full" | "partition" | "n_bc"

LAMBDA_BC_VALUES      = [1.0, 2.0, 3.0, 5.0]
ADAM_LR_VALUES        = [1e-3, 3e-3, 5e-3, 1e-2]
PARTITION_SEED_VALUES = [0, 1, 2, 42]
N_BC_VALUES           = [32, 64, 128, 256, 512]   # IC supervision density

# Best config from previous sweeps
BEST_LAMBDA_BC     = 1.0
BEST_ADAM_LR       = 1e-2
BEST_PARTITION_SEED = 0

# ── Per-trial settings (reduced from full run for speed) ─────────────────────
SWEEP_STEPS = 1500
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


def run_trial(lam: float, lr: float, partition_seed: int = 0, n_bc: int = N_BC) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model  = TargetPINN()
    gen    = QuantumWeightGenerator(seed=partition_seed)
    params = list(gen.parameters())

    x, y, t               = _make_colloc(N_COLLOC)
    xb, yb, tb, ub, vb   = _make_bc(n_bc)

    opt   = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SWEEP_STEPS, eta_min=ETA_MIN)

    pde_val = bc_val = 0.0
    t0 = time.perf_counter()
    for step in range(SWEEP_STEPS):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        (pde + lam * bc).backward()
        opt.step()
        sched.step()
        pde_val, bc_val = pde.item(), bc.item()
    elapsed = time.perf_counter() - t0

    return {
        "lambda_bc":      lam,
        "adam_lr":        lr,
        "partition_seed": partition_seed,
        "n_bc":           n_bc,
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
    if SWEEP_MODE == "partition":
        grid = [(BEST_LAMBDA_BC, BEST_ADAM_LR, s, N_BC) for s in PARTITION_SEED_VALUES]
        desc = f"partition_seed  (fixed λ={BEST_LAMBDA_BC}, lr={_fmt_lr(BEST_ADAM_LR)}, n_bc={N_BC})"
    elif SWEEP_MODE == "n_bc":
        grid = [(BEST_LAMBDA_BC, BEST_ADAM_LR, BEST_PARTITION_SEED, n) for n in N_BC_VALUES]
        desc = f"n_bc  (fixed λ={BEST_LAMBDA_BC}, lr={_fmt_lr(BEST_ADAM_LR)}, seed={BEST_PARTITION_SEED})"
    else:
        grid = [(lam, lr, 0, N_BC) for lam, lr in itertools.product(LAMBDA_BC_VALUES, ADAM_LR_VALUES)]
        desc = f"λ_bc × adam_lr  (partition_seed=0)"

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

    # ── Summary tables ────────────────────────────────────────────────────────
    print_grid(results, "pde")
    print_grid(results, "bc")
    print_grid(results, "pde+bc")

    # ── Rankings ─────────────────────────────────────────────────────────────
    by_pde = sorted(results, key=lambda r: r["pde"])
    by_sum = sorted(results, key=lambda r: r["pde+bc"])

    hdr = f"{'λ_bc':>6}  {'lr':>8}  {'seed':>6}  {'pde':>10}  {'bc':>10}  {'pde+bc':>10}"

    print(f"\n── Top 5 by pde {'─'*48}")
    print(hdr)
    for r in by_pde[:5]:
        print(f"{r['lambda_bc']:6.1f}  {_fmt_lr(r['adam_lr']):>8}  {r['partition_seed']:>6}  "
              f"{r['pde']:10.6f}  {r['bc']:10.6f}  {r['pde+bc']:10.6f}")

    print(f"\n── Top 5 by pde+bc {'─'*46}")
    print(hdr)
    for r in by_sum[:5]:
        print(f"{r['lambda_bc']:6.1f}  {_fmt_lr(r['adam_lr']):>8}  {r['partition_seed']:>6}  "
              f"{r['pde']:10.6f}  {r['bc']:10.6f}  {r['pde+bc']:10.6f}")

    best = by_sum[0]
    print(f"\n★  Recommended:  lambda_bc={best['lambda_bc']}  adam.lr={_fmt_lr(best['adam_lr'])}"
          f"  partition_seed={best['partition_seed']}  n_bc={best['n_bc']}")


if __name__ == "__main__":
    main()
