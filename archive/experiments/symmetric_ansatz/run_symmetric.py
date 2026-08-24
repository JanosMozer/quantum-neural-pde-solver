"""Step A4: train the symmetry-restricted ansatz (SymmetricQuantumWeightGenerator),
same PINN/loss/optimizer recipe as the B3 ablation and the depth sweep, with the
held-out generalization check run from the start this time (not bolted on after
the fact, per the lesson from the low_rank overfitting finding).

Usage: python run_symmetric.py <n_layers> <lambda_bc> [--steps N] [--n_colloc N]
       [--n_bc N] [--seed N] [--holdout_n N]
"""

import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.qnn_generator_symmetric import SymmetricQuantumWeightGenerator
from pdes.burgers2d.physics_loss import compute_burgers_loss
from qt_pinn.diagnostics import qgn


def make_colloc(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    y = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    t = torch.empty(n).uniform_(0, 1, generator=g).requires_grad_(True)
    return x, y, t


def make_bc(n: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.empty(n).uniform_(-1, 1, generator=g)
    y = torch.empty(n).uniform_(-1, 1, generator=g)
    t = torch.zeros(n)
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    return x, y, t, u, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("n_layers", type=int)
    parser.add_argument("lambda_bc", type=float)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--n_colloc", type=int, default=1024)
    parser.add_argument("--n_bc", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout_n", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = TargetPINN()
    gen = SymmetricQuantumWeightGenerator(n_layers=args.n_layers, seed=0)
    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)

    x, y, t = make_colloc(args.n_colloc, args.seed)
    xb, yb, tb, ub, vb = make_bc(args.n_bc, args.seed)

    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    print(f"n_layers={args.n_layers}  n_params={n_params}  lambda_bc={args.lambda_bc}  "
          f"n_colloc={args.n_colloc}  n_bc={args.n_bc}  steps={args.steps}")

    qgn_history = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        (pde + args.lambda_bc * bc).backward()
        qgn_history.append(qgn([gen.q_weights]))
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"{step:6d}  pde={pde.item():.6f}  bc={bc.item():.6f}  "
                  f"total={pde.item()+bc.item():.6f}  qgn={qgn_history[-1]:.6f}")
    elapsed = time.perf_counter() - t0

    # held-out generalization, from the start, not bolted on
    xh, yh, th = make_colloc(args.holdout_n, args.seed + 90000)
    xhb, yhb, thb, uhb, vhb = make_bc(args.holdout_n, args.seed + 90000)
    pde_h, bc_h = compute_burgers_loss(model, xh, yh, th, xhb, yhb, thb, uhb, vhb, gen())

    import numpy as np
    qgn_arr = np.array(qgn_history)
    result = {
        "n_layers": args.n_layers, "n_params": n_params, "lambda_bc": args.lambda_bc,
        "n_colloc": args.n_colloc, "n_bc": args.n_bc, "steps": args.steps, "seed": args.seed,
        "pde_loss": round(pde.item(), 8), "bc_loss": round(bc.item(), 8),
        "total": round(pde.item() + bc.item(), 8), "elapsed_s": round(elapsed, 1),
        "qgn_last_1000_mean": round(float(qgn_arr[-1000:].mean()), 6),
        "qgn_last_1000_std": round(float(qgn_arr[-1000:].std()), 6),
        "holdout_n": args.holdout_n,
        "holdout_pde_loss": round(pde_h.item(), 8),
        "holdout_bc_loss": round(bc_h.item(), 8),
        "holdout_total": round(pde_h.item() + bc_h.item(), 8),
    }
    print(f"\nFinal  pde={result['pde_loss']:.7f}  bc={result['bc_loss']:.7f}  "
          f"sum={result['total']:.7f}  ({elapsed:.1f}s)")
    print(f"Holdout  pde={result['holdout_pde_loss']:.7f}  bc={result['holdout_bc_loss']:.7f}  "
          f"sum={result['holdout_total']:.7f}")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"L{args.n_layers}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
