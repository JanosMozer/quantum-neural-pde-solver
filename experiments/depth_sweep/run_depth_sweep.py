"""Step 2 of the scaling-laws plan (revised after Step 1): Option 1 is underparametrized
by ~3236x relative to its DLA dimension (4^9-1=262143 vs 81 circuit params), and closing
that gap by depth alone is impractical (~9700 layers). This script tests the honest,
practical question instead: does a MODEST depth increase (a few x, not thousands x) show
any continuous improvement in bc_loss, and does it stay stable while doing so (Vyskubov et
al., arXiv:2604.06007, found depth-scaling is often unreliable/destabilizing for QNNs)?

N_LAYERS is baked into qt_pinn.qnn_generator at import time from pdes/burgers2d/config.yaml,
so the launcher (run_sweep.sh) edits that config's quantum.n_layers before each subprocess
call, exactly like scripts/train.py already requires. This script does not touch the config
itself, only reads the resulting N_LAYERS via the import, and records it in the output
for verification.

Usage: python run_depth_sweep.py <lambda_bc> [--steps N] [--n_colloc N] [--n_bc N] [--seed N]
"""

import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import numpy as np
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.qnn_generator import QuantumWeightGenerator, N_LAYERS, N_QUBITS
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
    parser.add_argument("lambda_bc", type=float)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--n_colloc", type=int, default=1024)
    parser.add_argument("--n_bc", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = TargetPINN()
    gen = QuantumWeightGenerator(seed=0)
    params = list(gen.parameters())
    n_circuit_params = gen.q_weights.numel()
    n_params = sum(p.numel() for p in params)

    x, y, t = make_colloc(args.n_colloc, args.seed)
    xb, yb, tb, ub, vb = make_bc(args.n_bc, args.seed)

    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    print(f"N_LAYERS={N_LAYERS}  N_QUBITS={N_QUBITS}  n_circuit_params={n_circuit_params}  "
          f"n_params={n_params}  lambda_bc={args.lambda_bc}  steps={args.steps}")

    qgn_history = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        (pde + args.lambda_bc * bc).backward()
        g = qgn([gen.q_weights])  # gradient-norm stability diagnostic (Vyskubov et al.)
        qgn_history.append(g)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"{step:6d}  pde={pde.item():.6f}  bc={bc.item():.6f}  "
                  f"total={pde.item()+bc.item():.6f}  qgn={g:.6f}")
    elapsed = time.perf_counter() - t0

    qgn_arr = np.array(qgn_history)
    result = {
        "n_layers": N_LAYERS, "n_qubits": N_QUBITS, "n_circuit_params": n_circuit_params,
        "n_params": n_params, "lambda_bc": args.lambda_bc,
        "n_colloc": args.n_colloc, "n_bc": args.n_bc, "steps": args.steps, "seed": args.seed,
        "pde_loss": round(pde.item(), 8), "bc_loss": round(bc.item(), 8),
        "total": round(pde.item() + bc.item(), 8), "elapsed_s": round(elapsed, 1),
        "qgn_mean": round(float(qgn_arr.mean()), 6),
        "qgn_std": round(float(qgn_arr.std()), 6),
        "qgn_max": round(float(qgn_arr.max()), 6),
        "qgn_last_1000_mean": round(float(qgn_arr[-1000:].mean()), 6),
        "qgn_last_1000_std": round(float(qgn_arr[-1000:].std()), 6),
    }
    print(f"\nFinal  pde={result['pde_loss']:.7f}  bc={result['bc_loss']:.7f}  "
          f"sum={result['total']:.7f}  qgn_mean={result['qgn_mean']:.4f}  "
          f"qgn_std={result['qgn_std']:.4f}  ({elapsed:.1f}s)")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"L{N_LAYERS}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
