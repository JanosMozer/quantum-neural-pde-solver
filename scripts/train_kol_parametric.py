"""Parametric Kolmogorov flow: nu -> PINN weights (quantum vs classical).

No exact solution — evaluation is holdout PDE residual RMS.

Run:
  .venv/bin/python scripts/train_kol_parametric.py --generator classical --seed 0 --run-id kol_par_c_s0
  .venv/bin/python scripts/train_kol_parametric.py --generator quantum   --seed 0 --run-id kol_par_q_s0
  .venv/bin/python scripts/compare_kol_parametric.py checkpoints/kol_par_c_s0 checkpoints/kol_par_q_s0
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

from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_cond import (
    ConditionedQuantumGenerator,
    ConditionedClassicalGenerator,
    ConditionedQuantumGeneratorV2,
    ConditionedClassicalGeneratorV2,
)
from pdes.kolmogorov2d.physics_loss import compute_kol_loss, N_FORCE, F_AMP

X_LO, X_HI = 0.0, 2.0 * math.pi


def sample_nu(n, lo, hi, device):
    u = torch.rand(n, device=device)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def make_colloc(n, device, t_max):
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    t = torch.empty(n, device=device).uniform_(0.0, t_max).requires_grad_(True)
    return x, y, t


def make_bc(n, device):
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    return x, y, torch.zeros(n, device=device)


def build_generator(args, in_dim, device):
    arch = args.qc_arch
    if arch == "expect":
        common = dict(
            in_dim=in_dim, h1=args.hidden[0], h2=args.hidden[1],
            n_qubits=args.n_qubits, bottleneck_width=args.bottleneck_width,
            nu_range=(args.nu_lo, args.nu_hi), freq_mode=args.freq_mode,
            nu_encode=args.nu_encode,
        )
        if args.generator == "quantum":
            gen = ConditionedQuantumGeneratorV2(n_layers=args.n_layers, **common)
        else:
            gen = ConditionedClassicalGeneratorV2(**common)
    else:
        common = dict(
            in_dim=in_dim, h1=args.hidden[0], h2=args.hidden[1],
            n_qubits=args.n_qubits, bottleneck_width=args.bottleneck_width,
            nu_range=(args.nu_lo, args.nu_hi), freq_mode=args.freq_mode,
        )
        if args.generator == "quantum":
            gen = ConditionedQuantumGenerator(n_layers=args.n_layers, **common)
        else:
            gen = ConditionedClassicalGenerator(**common)
    return gen.to(device)


def task_loss(model, gen, nus, x, y, t, x_bc, y_bc, t_bc, lambda_bc, n_force, f_amp):
    weights = gen(nus)
    pde_tot = torch.zeros((), device=x.device)
    bc_tot = torch.zeros((), device=x.device)
    for i in range(nus.shape[0]):
        w_i = {k: v[i] for k, v in weights.items()}
        pde, bc = compute_kol_loss(
            model, x, y, t, x_bc, y_bc, t_bc, w_i,
            nu=nus[i].item(), n_force=n_force, f_amp=f_amp)
        pde_tot = pde_tot + pde
        bc_tot = bc_tot + bc
    n = nus.shape[0]
    return pde_tot / n, bc_tot / n


def evaluate_pde_rms(model, gen, nus, device, t_max, n_force, f_amp, n_pts=1024):
    """Mean PDE RMS over space-time collocation for each nu."""
    x, y, t = make_colloc(n_pts, device, t_max)
    x_bc, y_bc, t_bc = make_bc(min(256, n_pts), device)
    weights = gen(nus)
    out = {}
    for i, nu in enumerate(nus.tolist()):
        w_i = {k: v[i] for k, v in weights.items()}
        pde, bc = compute_kol_loss(
            model, x, y, t, x_bc, y_bc, t_bc, w_i,
            nu=nu, n_force=n_force, f_amp=f_amp)
        out[round(nu, 5)] = {
            "pde_rms": math.sqrt(pde.item()),
            "bc_loss": bc.item(),
        }
    return out


def summarise(errs):
    return float(np.mean([v["pde_rms"] for v in errs.values()]))


def eval_grids(args, device):
    in_range = torch.tensor([0.012, 0.02, 0.035, 0.05, 0.07, 0.095], device=device)
    extrap_lo = torch.tensor([0.005, 0.007, 0.009], device=device)
    extrap_hi = torch.tensor([0.12, 0.15, 0.20], device=device)
    return {"in_range": in_range, "extrap_lo": extrap_lo, "extrap_hi": extrap_hi}


def parse_args():
    p = argparse.ArgumentParser(description="Kolmogorov parametric: nu -> weights")
    p.add_argument("--generator", choices=["quantum", "classical"], required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-steps", type=int, default=12000)
    p.add_argument("--patience", type=int, default=1500)
    p.add_argument("--plateau-window", type=int, default=200)
    p.add_argument("--min-delta", type=float, default=0.005)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-colloc", type=int, default=2048)
    p.add_argument("--n-bc", type=int, default=2048)
    p.add_argument("--n-tasks", type=int, default=8)
    p.add_argument("--lambda-bc", type=float, default=20.0)
    p.add_argument("--weight-reg", type=float, default=0.0)
    p.add_argument("--resample-every", type=int, default=50)
    p.add_argument("--nu-lo", type=float, default=0.01)
    p.add_argument("--nu-hi", type=float, default=0.1)
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--n-force", type=int, default=N_FORCE)
    p.add_argument("--f-amp", type=float, default=F_AMP)
    p.add_argument("--hidden", type=int, nargs=2, default=[32, 32])
    p.add_argument("--n-qubits", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--bottleneck-width", type=int, default=64)
    p.add_argument("--freq-mode", default="linear", choices=["linear", "geometric"])
    p.add_argument("--qc-arch", default="expect", choices=["reupload", "expect"])
    p.add_argument("--nu-encode", default="log", choices=["linear", "log"])
    p.add_argument("--hard-ic", action="store_true", default=True)
    p.add_argument("--no-hard-ic", dest="hard_ic", action="store_false")
    p.add_argument("--run-id", default="")
    return p.parse_args()


def resolve_device(arg):
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_split(name, errs, mean):
    print(f"\n[{name}] mean PDE RMS = {mean:.5f}")
    print(f"  {'nu':>8}  {'pde_rms':>10}  {'bc':>10}")
    for nu, rec in errs.items():
        print(f"  {nu:8.4f}  {rec['pde_rms']:10.5f}  {rec['bc_loss']:10.2e}")


def main():
    args = parse_args()
    if args.n_qubits is None:
        args.n_qubits = 6 if args.qc_arch == "expect" else 8
    if args.n_layers is None:
        args.n_layers = 6 if args.qc_arch == "expect" else 4
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or f"kol_par_{args.generator}_s{args.seed}"
    run_dir = Path("checkpoints") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = TargetPINNNS(
        fourier="kol", hard_ic=args.hard_ic,
        hidden=tuple(args.hidden), t_max=args.t_max, n_force=args.n_force,
    ).to(device)
    gen = build_generator(args, model.in_dim, device)
    n_params = sum(p.numel() for p in gen.parameters())
    grids = eval_grids(args, device)

    print(f"Run: {run_id}  Kolmogorov  generator={args.generator}  qc_arch={args.qc_arch}")
    print(f"nu in [{args.nu_lo}, {args.nu_hi}]  T={args.t_max}  n_force={args.n_force}  "
          f"weights={gen.total_weights}  gen_params={n_params:,}")

    x, y, t = make_colloc(args.n_colloc, device, args.t_max)
    x_bc, y_bc, t_bc = make_bc(args.n_bc, device)
    opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=1e-5)

    history, roll = [], []
    best_roll, stale = float("inf"), 0
    stopped_at = args.max_steps
    t0 = time.time()
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'roll':>10}")
    for step in range(args.max_steps):
        if args.resample_every and step and step % args.resample_every == 0:
            x, y, t = make_colloc(args.n_colloc, device, args.t_max)
            x_bc, y_bc, t_bc = make_bc(args.n_bc, device)
        nus = sample_nu(args.n_tasks, args.nu_lo, args.nu_hi, device)
        opt.zero_grad()
        pde, bc = task_loss(model, gen, nus, x, y, t, x_bc, y_bc, t_bc,
                            args.lambda_bc, args.n_force, args.f_amp)
        loss = pde + args.lambda_bc * bc
        if args.weight_reg:
            flat = torch.cat([v.flatten() for v in gen(nus).values()])
            loss = loss + args.weight_reg * flat.pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        sched.step()

        loss_v = loss.item()
        roll.append(loss_v)
        if len(roll) > args.plateau_window:
            roll.pop(0)
        roll_mean = float(np.mean(roll))
        if len(roll) == args.plateau_window:
            if roll_mean < best_roll * (1.0 - args.min_delta):
                best_roll, stale = roll_mean, 0
            else:
                stale += 1
        if step % args.log_every == 0:
            history.append({"step": step, "total": loss_v, "pde": pde.item(),
                            "bc": bc.item(), "roll": roll_mean})
            print(f"{step:6d}  {loss_v:12.6f}  {pde.item():12.6f}  "
                  f"{bc.item():12.6f}  {roll_mean:10.6f}  [{time.time()-t0:.0f}s]")
        if stale >= args.patience:
            stopped_at = step + 1
            print(f"\nplateau stop at step {stopped_at}")
            break

    elapsed = time.time() - t0
    splits, means = {}, {}
    for name, nus in grids.items():
        splits[name] = evaluate_pde_rms(
            model, gen, nus, device, args.t_max, args.n_force, args.f_amp)
        means[name] = summarise(splits[name])
        print_split(name, splits[name], means[name])

    torch.save(gen.state_dict(), run_dir / "generator.pt")
    (run_dir / "config.json").write_text(json.dumps(vars(args) | {
        "run_id": run_id, "pde": "kolmogorov2d", "device": str(device),
        "n_generator_params": n_params, "stopped_at": stopped_at,
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id, "generator": args.generator, "qc_arch": args.qc_arch,
        "pde": "kolmogorov2d",
        "elapsed_s": round(elapsed, 1), "stopped_at": stopped_at,
        "mean_pde_rms_in_range": round(means["in_range"], 6),
        "mean_pde_rms_extrap_lo": round(means["extrap_lo"], 6),
        "mean_pde_rms_extrap_hi": round(means["extrap_hi"], 6),
        "errors": splits, "history": history,
    }, indent=2))
    print(f"\nelapsed={elapsed:.0f}s  Saved -> {run_dir}/")


if __name__ == "__main__":
    main()
