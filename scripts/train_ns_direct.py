"""Tier A: direct-MLP NS baseline (no quantum generator).

A3: train TargetPINN weights directly with corrected physics + TGV Fourier.
A4: optional hard IC ansatz  u = u_IC(x,y) + t * N_u  (bc_loss ≡ 0).

Gate: if A3 does not reach ≲5% mean rel-L2 at t=1, the generator question
is unanswerable on this problem — fix the target network first.

Run from repo root:
  # A3 soft IC
  .venv/bin/python scripts/train_ns_direct.py --adam-steps 6000

  # A4 hard IC
  .venv/bin/python scripts/train_ns_direct.py --hard-ic --adam-steps 6000
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

from qt_pinn.fourier import FourierFeatureMap, FourierFeatureMapTGV
from pdes.ns2d.physics_loss import exact_solution, _grad, relative_l2_gauge

X_LO, X_HI = 0.0, 2.0 * math.pi


class DirectNSMLP(nn.Module):
    """Classical PINN: (x,y,t) -> (u,v,p). Trainable weights, no hypernetwork."""

    def __init__(
        self,
        hidden: tuple[int, int] = (32, 32),
        fourier: str = "tgv",
        fourier_sigma: float = 0.25,
        hard_ic: bool = False,
        t_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        if fourier == "tgv":
            self.fourier: nn.Module = FourierFeatureMapTGV(t_max=t_max)
            in_dim = self.fourier.out_dim
        elif fourier == "random":
            self.fourier = FourierFeatureMap(sigma=fourier_sigma)
            in_dim = 6
        else:
            raise ValueError(f"unknown fourier={fourier!r}")

        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 3),
        )

    def _raw(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feats = self.fourier(torch.stack([x, y, t], dim=-1))
        return self.net(feats)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self._raw(x, y, t)
        if not self.hard_ic:
            return raw
        # Hard IC: field = IC(x,y) + t * network  ⇒ exact IC at t=0
        u0 = torch.sin(x) * torch.cos(y)
        v0 = -torch.cos(x) * torch.sin(y)
        p0 = (torch.cos(2.0 * x) + torch.cos(2.0 * y)) / 4.0
        ic = torch.stack([u0, v0, p0], dim=-1)
        return ic + t.unsqueeze(-1) * raw


def compute_ns_loss_direct(
    model: DirectNSMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    t_bc: torch.Tensor,
    nu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    uvp = model(x, y, t)
    u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]

    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    p_x = _grad(p, x); p_y = _grad(p, y)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y)

    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    pde = f_c.pow(2).mean() + f_u.pow(2).mean() + f_v.pow(2).mean()

    if model.hard_ic:
        # IC satisfied by construction
        bc = torch.zeros((), device=x.device)
    else:
        uvp_bc = model(x_bc, y_bc, t_bc)
        u_ex, v_ex, p_ex = exact_solution(x_bc, y_bc, t_bc, nu)
        bc = ((uvp_bc[:, 0] - u_ex).pow(2).mean()
              + (uvp_bc[:, 1] - v_ex).pow(2).mean()
              + (uvp_bc[:, 2] - p_ex).pow(2).mean())
    return pde, bc


def make_colloc(n: int, device: torch.device, t_max: float) -> tuple[torch.Tensor, ...]:
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    t = torch.empty(n, device=device).uniform_(0.0, t_max).requires_grad_(True)
    return x, y, t


def make_bc(n: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    t = torch.zeros(n, device=device)
    return x, y, t


def evaluate_exact(
    model: DirectNSMLP, device: torch.device, nu: float, t_max: float
) -> dict:
    model.eval()
    n = 32
    xs = torch.linspace(X_LO, X_HI, n, device=device)
    ys = torch.linspace(X_LO, X_HI, n, device=device)
    results = {}
    for frac in [0.0, 0.25, 0.5, 1.0]:
        tv = frac * t_max
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        x, y = xg.flatten(), yg.flatten()
        t = torch.full_like(x, tv)
        with torch.no_grad():
            pred = model(x, y, t)
        u_ex, v_ex, p_ex = exact_solution(x, y, t, nu)

        def rel(p, e):
            return ((p - e).norm() / (e.norm() + 1e-10)).item()

        results[round(tv, 4)] = {
            "u": rel(pred[:, 0], u_ex),
            "v": rel(pred[:, 1], v_ex),
            # pressure is only defined up to a gauge; compare mean-removed fields
            "p": relative_l2_gauge(pred[:, 2], p_ex),
            "p_raw": rel(pred[:, 2], p_ex),
        }
    model.train()
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A3/A4 direct-MLP NS baseline")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--adam-steps", type=int, default=6000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--n-colloc", type=int, default=4096)
    p.add_argument("--n-bc", type=int, default=4096)
    p.add_argument("--lambda-bc", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fourier", choices=["tgv", "random"], default="tgv")
    p.add_argument("--fourier-sigma", type=float, default=0.25)
    p.add_argument("--hard-ic", action="store_true",
                   help="A4: hard IC ansatz u=u_IC + t*N")
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--hidden", type=int, nargs=2, default=[32, 32])
    p.add_argument("--nu", type=float, default=0.1,
                   help="kinematic viscosity; 0.1 with T=5 gives a 63%% decay, "
                        "so the time channel is actually exercised")
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--resample-every", type=int, default=100,
                   help="0 = fixed collocation set (overfits the point cloud)")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def next_run_dir(tag: str) -> tuple[str, Path]:
    base = Path("checkpoints")
    base.mkdir(exist_ok=True)
    if tag:
        d = base / tag
        if d.exists():
            raise FileExistsError(f"exists: {d}")
        return tag, d
    existing = sorted(base.glob("ns_direct_*"))
    n = int(existing[-1].name.split("_")[-1]) + 1 if existing else 1
    run_id = f"ns_direct_{n:04d}"
    return run_id, base / run_id


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id, run_dir = next_run_dir(args.run_id)
    run_dir.mkdir(parents=True)

    model = DirectNSMLP(
        hidden=tuple(args.hidden),
        fourier=args.fourier,
        fourier_sigma=args.fourier_sigma,
        hard_ic=args.hard_ic,
        t_max=args.t_max,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    decay = math.exp(-2.0 * args.nu * args.t_max)
    print(f"Run: {run_id}  device={device}")
    print(f"A3/A4 direct MLP  hard_ic={args.hard_ic}  fourier={args.fourier}  "
          f"params={n_params:,}  lambda_bc={args.lambda_bc}")
    print(f"nu={args.nu}  T={args.t_max}  amplitude decay over horizon = {decay:.4f}")

    x, y, t = make_colloc(args.n_colloc, device, args.t_max)
    x_bc, y_bc, t_bc = make_bc(args.n_bc, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.adam_steps, eta_min=1e-5)

    t0 = time.time()
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'lr':>10}")
    for step in range(args.adam_steps):
        if args.resample_every and step and step % args.resample_every == 0:
            x, y, t = make_colloc(args.n_colloc, device, args.t_max)
            x_bc, y_bc, t_bc = make_bc(args.n_bc, device)
        opt.zero_grad()
        pde, bc = compute_ns_loss_direct(model, x, y, t, x_bc, y_bc, t_bc, args.nu)
        loss = pde if args.hard_ic else (pde + args.lambda_bc * bc)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            print(f"{step:6d}  {loss.item():12.6f}  {pde.item():12.6f}  "
                  f"{bc.item():12.6f}  {opt.param_groups[0]['lr']:.2e}  "
                  f"[{time.time()-t0:.0f}s]")

    pde_f, bc_f = compute_ns_loss_direct(model, x, y, t, x_bc, y_bc, t_bc, args.nu)
    elapsed = time.time() - t0
    print(f"\nFinal  pde={pde_f.item():.7f}  bc={bc_f.item():.7f}  [{elapsed:.0f}s]")

    exact_errs = evaluate_exact(model, device, args.nu, args.t_max)
    print("\nRelative L2 vs exact Taylor-Green (p is gauge-fixed):")
    print(f"  {'t':>6}  {'u_err%':>8}  {'v_err%':>8}  {'p_err%':>8}  {'mean%':>8}")
    for tv, errs in exact_errs.items():
        m = (errs["u"] + errs["v"] + errs["p"]) / 3
        print(f"  {tv:6.2f}  {errs['u']*100:8.3f}  {errs['v']*100:8.3f}  "
              f"{errs['p']*100:8.3f}  {m*100:8.3f}")
    e_end = exact_errs[round(args.t_max, 4)]
    mean_t1 = (e_end["u"] + e_end["v"] + e_end["p"]) / 3
    gate = "PASS" if mean_t1 < 0.05 else "FAIL"
    print(f"\nA3 gate (mean rel-L2 @ t=T < 5%): {gate}  ({mean_t1*100:.2f}%)")

    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "tier": "A3" if not args.hard_ic else "A4",
        "hard_ic": args.hard_ic, "fourier": args.fourier,
        "fourier_sigma": args.fourier_sigma, "device": str(device),
        "seed": args.seed, "n_colloc": args.n_colloc, "n_bc": args.n_bc,
        "lambda_bc": args.lambda_bc, "adam_steps": args.adam_steps,
        "lr": args.lr, "hidden": list(args.hidden), "n_params": n_params,
        "pressure_sign": "positive", "nu": args.nu, "t_max": args.t_max,
        "resample_every": args.resample_every,
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id,
        "pde_loss": round(pde_f.item(), 8),
        "bc_loss": round(bc_f.item(), 8),
        "elapsed_s": round(elapsed, 1),
        "n_params": n_params,
        "mean_rel_l2_t1": round(mean_t1, 6),
        "gate_pass": mean_t1 < 0.05,
        "exact_l2": {str(tv): {k: round(v, 6) for k, v in errs.items()}
                     for tv, errs in exact_errs.items()},
    }, indent=2))
    print(f"Saved -> {run_dir}/")


if __name__ == "__main__":
    main()
