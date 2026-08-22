"""Direct classical PINN for 2D Kolmogorov flow (no generator).

Gate: PDE RMS < 0.05 and IC satisfied before parametric QC comparison.

Run:
  .venv/bin/python scripts/train_kol_direct.py --adam-steps 8000 --run-id kol_direct_s0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from qt_pinn.fourier import FourierFeatureMapKolmogorov
from qt_pinn.pinn_target_ns import TargetPINNNS
from pdes.kolmogorov2d.physics_loss import compute_kol_loss, pde_rms, N_FORCE, F_AMP

X_LO, X_HI = 0.0, 2.0 * math.pi


class DirectKolMLP(nn.Module):
    def __init__(
        self,
        hidden: tuple[int, int] = (32, 32),
        n_force: int = N_FORCE,
        t_max: float = 5.0,
        hard_ic: bool = True,
    ) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        self.fourier = FourierFeatureMapKolmogorov(n_force=n_force, t_max=t_max)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(self.fourier.out_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 3),
        )

    def forward(self, x, y, t) -> torch.Tensor:
        raw = self.net(self.fourier(torch.stack([x, y, t], dim=-1)))
        if not self.hard_ic:
            return raw
        return t.unsqueeze(-1) * raw


def kol_loss_direct(model, x, y, t, x_bc, y_bc, t_bc, nu, n_force, f_amp):
    uvp = model(x, y, t)
    u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]
    from pdes.kolmogorov2d.physics_loss import _grad, forcing
    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    p_x = _grad(p, x); p_y = _grad(p, y)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y)
    fx, fy = forcing(y, n_force, f_amp)
    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) - fx
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) - fy
    pde = f_c.pow(2).mean() + f_u.pow(2).mean() + f_v.pow(2).mean()
    if model.hard_ic:
        bc = torch.zeros((), device=x.device)
    else:
        bc = model(x_bc, y_bc, t_bc).pow(2).mean()
    return pde, bc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--adam-steps", type=int, default=8000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--n-colloc", type=int, default=4096)
    p.add_argument("--n-bc", type=int, default=2048)
    p.add_argument("--lambda-bc", type=float, default=20.0)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--n-force", type=int, default=N_FORCE)
    p.add_argument("--f-amp", type=float, default=F_AMP)
    p.add_argument("--hard-ic", action="store_true", default=True)
    p.add_argument("--no-hard-ic", dest="hard_ic", action="store_false")
    p.add_argument("--run-id", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or "kol_direct_0001"
    run_dir = Path("checkpoints") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = DirectKolMLP(n_force=args.n_force, t_max=args.t_max,
                         hard_ic=args.hard_ic).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Run {run_id}  nu={args.nu}  T={args.t_max}  n_force={args.n_force}  "
          f"params={n_params:,}")

    x = torch.empty(args.n_colloc, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    y = torch.empty(args.n_colloc, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    t = torch.empty(args.n_colloc, device=device).uniform_(0, args.t_max).requires_grad_(True)
    x_bc = torch.empty(args.n_bc, device=device).uniform_(X_LO, X_HI)
    y_bc = torch.empty(args.n_bc, device=device).uniform_(X_LO, X_HI)
    t_bc = torch.zeros(args.n_bc, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.adam_steps, eta_min=1e-5)
    t0 = time.time()
    for step in range(args.adam_steps):
        opt.zero_grad()
        pde, bc = kol_loss_direct(model, x, y, t, x_bc, y_bc, t_bc,
                                  args.nu, args.n_force, args.f_amp)
        loss = pde + args.lambda_bc * bc
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            print(f"{step:6d}  total={loss.item():.6f}  pde={pde.item():.6f}  "
                  f"bc={bc.item():.6f}  [{time.time()-t0:.0f}s]")

    pde_f, bc_f = kol_loss_direct(model, x, y, t, x_bc, y_bc, t_bc,
                                  args.nu, args.n_force, args.f_amp)
    rms = math.sqrt(pde_f.item())
    gate = "PASS" if rms < 0.05 else "FAIL"
    print(f"\nFinal PDE RMS={rms:.5f}  bc={bc_f.item():.2e}  gate(<0.05): {gate}")

    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id, "pde_loss": pde_f.item(), "pde_rms": rms,
        "bc_loss": bc_f.item(), "gate_pass": rms < 0.05,
        "n_params": n_params, "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    print(f"Saved -> {run_dir}/")


if __name__ == "__main__":
    main()
