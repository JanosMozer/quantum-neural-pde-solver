"""Hyperparameter sweep: lambda_bc x adam_lr grid for the learned-projection generator.

Runs all combinations with reduced steps for speed, saves results to
checkpoints/sweep_YYYYMMDD_HHMMSS.csv and prints a ranked summary table.

Usage: python scripts/sweep.py
"""

import sys, itertools, csv, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from qt_pinn.pinn_target import TargetPINN
from pdes.burgers2d.physics_loss import compute_burgers_loss
from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP

LAMBDA_BC_VALUES = [1.0, 2.0, 3.0, 5.0]
ADAM_LR_VALUES   = [1e-3, 3e-3, 5e-3, 1e-2]

SWEEP_STEPS = 3000
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


def run_trial(lam: float, lr: float, steps: int = SWEEP_STEPS) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = TargetPINN()
    gen = QuantumWeightGeneratorLP()
    params = list(gen.parameters())

    x, y, t = _make_colloc(N_COLLOC)
    xb, yb, tb, ub, vb = _make_bc(N_BC)

    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=ETA_MIN)

    pde_val = bc_val = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        (pde + lam * bc).backward()
        opt.step()
        sched.step()
        pde_val, bc_val = pde.item(), bc.item()
    elapsed = time.perf_counter() - t0

    return {
        "lambda_bc": lam, "adam_lr": lr,
        "pde": pde_val, "bc": bc_val, "pde+bc": pde_val + bc_val,
        "elapsed_s": round(elapsed, 1),
    }


def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}"


def main() -> None:
    grid = list(itertools.product(LAMBDA_BC_VALUES, ADAM_LR_VALUES))
    print("Sweep: learned_proj  lambda_bc x lr")
    print(f"{len(grid)} trials x {SWEEP_STEPS} steps  (n_colloc={N_COLLOC}, n_bc={N_BC})\n")
    print(f"{'#':>3}  {'lam':>6}  {'lr':>8}  {'pde':>12}  {'bc':>12}  {'pde+bc':>12}  {'t(s)':>6}")
    print("-" * 72)

    results = []
    for i, (lam, lr) in enumerate(grid, 1):
        print(f"{i:3d}  {lam:6.1f}  {_fmt_lr(lr):>8}  ", end="", flush=True)
        r = run_trial(lam, lr)
        results.append(r)
        print(f"{r['pde']:12.6f}  {r['bc']:12.6f}  {r['pde+bc']:12.6f}  {r['elapsed_s']:6.1f}s")

    Path("checkpoints").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outcsv = Path(f"checkpoints/sweep_{stamp}.csv")
    with open(outcsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {outcsv}")

    by_sum = sorted(results, key=lambda r: r["pde+bc"])
    by_pde = sorted(results, key=lambda r: r["pde"])
    hdr = f"{'lam':>6}  {'lr':>8}  {'pde':>10}  {'bc':>10}  {'pde+bc':>10}"

    def _row(r):
        return f"{r['lambda_bc']:6.1f}  {_fmt_lr(r['adam_lr']):>8}  {r['pde']:10.6f}  {r['bc']:10.6f}  {r['pde+bc']:10.6f}"

    print("\nTop 5 by pde")
    print(hdr)
    for r in by_pde[:5]:
        print(_row(r))

    print("\nTop 5 by pde+bc")
    print(hdr)
    for r in by_sum[:5]:
        print(_row(r))

    best = by_sum[0]
    print(f"\nRecommended: lambda_bc={best['lambda_bc']}  adam.lr={_fmt_lr(best['adam_lr'])}")


if __name__ == "__main__":
    main()
