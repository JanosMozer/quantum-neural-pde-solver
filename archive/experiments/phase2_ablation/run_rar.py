"""Step 3: Residual-based Adaptive Refinement (RAR), Lu, Meng, Mao, Karniadakis,
DeepXDE (arXiv:1907.04502), exact procedure (research/papers/EXTRACTED_NOTES.md):

  1. Train on the initial point set for a limited number of iterations.
  2. Estimate mean residual via Monte Carlo over a larger candidate pool S.
  3. If mean residual exceeds a threshold, add the m highest-residual points from
     S to the active set; repeat.

Starts from a small initial collocation set (256, matching the paper's own scale
and the original config) and grows to a size matched with the uniform-sampling
baseline (M=4096) by the end of training, so the comparison is fair: same final
point budget, different selection strategy. BC points stay fixed size (the paper's
own RAR only refines interior/residual points, not boundary points).

Uses the current best config from steps 1-2 (sigma=0.1, SIREN) by default -- set
config.yaml's fourier.sigma=0.1 before running, same convention as every other
experiment this session.

Usage: python run_rar.py [--siren] [--steps N] [--init_n N] [--target_n N] [--m N] [--add_every N]
"""

import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from qt_pinn.pinn_target import TargetPINN
from pdes.burgers2d.physics_loss import compute_burgers_loss
from run_ablation import DirectGenerator, make_bc


def sample_candidates(n: int, device) -> tuple:
    """Fresh, ungoverned-by-any-fixed-seed candidate pool for residual evaluation --
    RAR needs new random points each round, not the same fixed set reused."""
    x = torch.empty(n, device=device).uniform_(-1, 1).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(-1, 1).requires_grad_(True)
    t = torch.empty(n, device=device).uniform_(0, 1).requires_grad_(True)
    return x, y, t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_bc", type=float, nargs="?", default=1.0)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--init_n", type=int, default=256, help="initial colloc set size, matches DeepXDE's own scale")
    parser.add_argument("--target_n", type=int, default=4096, help="final size, matched to the uniform-sampling baseline")
    parser.add_argument("--pool_n", type=int, default=4096, help="candidate pool size per round")
    parser.add_argument("--m", type=int, default=64, help="points added per round")
    parser.add_argument("--add_every", type=int, default=250, help="steps between RAR rounds")
    parser.add_argument("--n_bc", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--siren", action="store_true")
    parser.add_argument("--holdout_n", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = TargetPINN(activation="siren" if args.siren else "tanh").to(device)
    gen = DirectGenerator(siren_init=args.siren).to(device)
    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)

    x, y, t = sample_candidates(args.init_n, device)
    xb, yb, tb, ub, vb = make_bc(args.n_bc, args.seed)
    xb, yb, tb, ub, vb = (v.to(device) for v in (xb, yb, tb, ub, vb))

    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    print(f"generator=direct(RAR)  device={device}  n_params={n_params}  siren={args.siren}  "
          f"init_n={args.init_n}  target_n={args.target_n}  m={args.m}  add_every={args.add_every}  "
          f"steps={args.steps}")

    n_additions = 0
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        (pde + args.lambda_bc * bc).backward()
        opt.step()
        sched.step()

        if step > 0 and step % args.add_every == 0 and x.shape[0] < args.target_n:
            xc, yc, tc = sample_candidates(args.pool_n, device)
            # no torch.no_grad() here: computing f_u/f_v needs real autograd.grad
            # through x/y/t; the residual tensor is already .detach()'d inside
            # compute_burgers_loss, that's the correct place to stop tracking, not here
            _, _, res = compute_burgers_loss(model, xc, yc, tc, xb, yb, tb, ub, vb, w,
                                              return_residuals=True)
            m = min(args.m, args.target_n - x.shape[0])
            top = torch.topk(res, m).indices
            x = torch.cat([x.detach(), xc[top].detach()]).requires_grad_(True)
            y = torch.cat([y.detach(), yc[top].detach()]).requires_grad_(True)
            t = torch.cat([t.detach(), tc[top].detach()]).requires_grad_(True)
            n_additions += 1

        if step % 2000 == 0:
            print(f"{step:6d}  n_colloc={x.shape[0]:5d}  pde={pde.item():.6f}  bc={bc.item():.6f}  "
                  f"total={pde.item()+bc.item():.6f}")
    elapsed = time.perf_counter() - t0

    print(f"\nFinal  n_colloc={x.shape[0]}  pde={pde.item():.7f}  bc={bc.item():.7f}  "
          f"sum={pde.item()+bc.item():.7f}  ({elapsed:.1f}s, {n_additions} RAR rounds)")

    xh, yh, th = sample_candidates(args.holdout_n, device)
    xhb, yhb, thb, uhb, vhb = make_bc(args.holdout_n, args.seed + 90000)
    xhb, yhb, thb, uhb, vhb = (v.to(device) for v in (xhb, yhb, thb, uhb, vhb))
    pde_h, bc_h = compute_burgers_loss(model, xh, yh, th, xhb, yhb, thb, uhb, vhb, gen())
    print(f"Held-out (n={args.holdout_n})  pde={pde_h.item():.7f}  bc={bc_h.item():.7f}  "
          f"sum={pde_h.item()+bc_h.item():.7f}")

    result = {
        "generator": "direct_rar", "n_params": n_params, "lambda_bc": args.lambda_bc,
        "init_n": args.init_n, "final_n_colloc": x.shape[0], "target_n": args.target_n,
        "m": args.m, "add_every": args.add_every, "n_bc": args.n_bc, "steps": args.steps,
        "seed": args.seed, "siren": args.siren,
        "pde_loss": round(pde.item(), 8), "bc_loss": round(bc.item(), 8),
        "total": round(pde.item() + bc.item(), 8), "elapsed_s": round(elapsed, 1),
        "holdout_n": args.holdout_n,
        "holdout_pde_loss": round(pde_h.item(), 8), "holdout_bc_loss": round(bc_h.item(), 8),
        "holdout_total": round(pde_h.item() + bc_h.item(), 8),
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = "_siren" if args.siren else ""
    out_path = out_dir / f"direct_rar_seed{args.seed}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
