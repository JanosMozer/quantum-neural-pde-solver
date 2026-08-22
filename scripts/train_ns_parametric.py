"""Tier B: parametric NS family — well-posed quantum vs classical test.

Task: learn nu -> PINN weights for Taylor-Green NS over a viscosity family.
Quantum: data re-uploading circuit. Classical: matched sin/cos encoder + MLP.
Both share the same frequency ladder, projection head, and evaluation protocol.

Eval splits (fixed grids, not random draws):
  in-range   — nu inside the training interval
  extrap-lo  — nu below nu_lo (true generalisation)
  extrap-hi  — nu above nu_hi

Training stops at plateau (rolling loss) or --max-steps, so quantum and
classical get equal *converged* footing rather than equal step counts.

Run from repo root:
  .venv/bin/python scripts/train_ns_parametric.py --generator classical --seed 0 --run-id ns_par_c_s0
  .venv/bin/python scripts/train_ns_parametric.py --generator quantum   --seed 0 --run-id ns_par_q_s0
  .venv/bin/python scripts/compare_ns_parametric.py checkpoints/ns_par_c_s0 checkpoints/ns_par_q_s0
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
from pdes.ns2d.physics_loss import compute_ns_loss, exact_solution

X_LO, X_HI = 0.0, 2.0 * math.pi


def sample_nu(n: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    """Log-uniform: spaces solutions roughly evenly via exp(-2 nu t)."""
    u = torch.rand(n, device=device)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def make_colloc(n: int, device: torch.device, t_max: float):
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    t = torch.empty(n, device=device).uniform_(0.0, t_max).requires_grad_(True)
    return x, y, t


def make_bc(n: int, device: torch.device):
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    return x, y, torch.zeros(n, device=device)


def build_generator(args, in_dim: int, device: torch.device) -> nn.Module:
    arch = args.qc_arch
    if arch == "expect":
        common = dict(
            in_dim=in_dim,
            h1=args.hidden[0],
            h2=args.hidden[1],
            n_qubits=args.n_qubits,
            bottleneck_width=args.bottleneck_width,
            nu_range=(args.nu_lo, args.nu_hi),
            freq_mode=args.freq_mode,
            nu_encode=args.nu_encode,
        )
        if args.generator == "quantum":
            gen = ConditionedQuantumGeneratorV2(n_layers=args.n_layers, **common)
        else:
            gen = ConditionedClassicalGeneratorV2(**common)
    else:
        common = dict(
            in_dim=in_dim,
            h1=args.hidden[0],
            h2=args.hidden[1],
            n_qubits=args.n_qubits,
            bottleneck_width=args.bottleneck_width,
            nu_range=(args.nu_lo, args.nu_hi),
            freq_mode=args.freq_mode,
        )
        if args.generator == "quantum":
            gen = ConditionedQuantumGenerator(n_layers=args.n_layers, **common)
        else:
            gen = ConditionedClassicalGenerator(**common)
    return gen.to(device)


def task_loss(model, gen, nus, x, y, t, x_bc, y_bc, t_bc, lambda_bc):
    weights = gen(nus)
    pde_tot = torch.zeros((), device=x.device)
    bc_tot = torch.zeros((), device=x.device)
    for i in range(nus.shape[0]):
        w_i = {k: v[i] for k, v in weights.items()}
        pde, bc = compute_ns_loss(
            model, x, y, t, x_bc, y_bc, t_bc, w_i, nu=nus[i].item())
        pde_tot = pde_tot + pde
        bc_tot = bc_tot + bc
    n = nus.shape[0]
    return pde_tot / n, bc_tot / n


@torch.no_grad()
def evaluate_family(model, gen, nus, device, t_max, n_grid=32, n_t=11) -> dict:
    """Space-time relative L2 vs exact TGV, per nu (pressure gauge-fixed)."""
    model.eval()
    xs = torch.linspace(X_LO, X_HI, n_grid, device=device)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x, y = xg.flatten(), yg.flatten()
    ts = torch.linspace(0.0, t_max, n_t, device=device)
    weights = gen(nus)
    out = {}
    for i, nu in enumerate(nus.tolist()):
        w_i = {k: v[i] for k, v in weights.items()}
        num = {"u": 0.0, "v": 0.0, "p": 0.0}
        den = {"u": 0.0, "v": 0.0, "p": 0.0}
        for tv in ts.tolist():
            t = torch.full_like(x, tv)
            pred = model(x, y, t, w_i)
            u_ex, v_ex, p_ex = exact_solution(x, y, t, nu)
            fields = {
                "u": (pred[:, 0], u_ex),
                "v": (pred[:, 1], v_ex),
                "p": (pred[:, 2] - pred[:, 2].mean(), p_ex - p_ex.mean()),
            }
            for k, (pv, ev) in fields.items():
                num[k] += (pv - ev).pow(2).sum().item()
                den[k] += ev.pow(2).sum().item()
        out[round(nu, 5)] = {
            "spacetime": {k: math.sqrt(num[k] / (den[k] + 1e-30)) for k in num},
        }
    model.train()
    return out


def summarise(errs: dict) -> float:
    per_nu = [sum(v["spacetime"][f] for f in ("u", "v", "p")) / 3
              for v in errs.values()]
    return float(np.mean(per_nu))


def eval_grids(args, device: torch.device) -> dict[str, torch.Tensor]:
    """Fixed grids: in-range interpolation + true extrapolation outside training."""
    in_range = torch.tensor(
        [0.06, 0.10, 0.18, 0.28, 0.38, 0.48], device=device)
    # keep amplitudes resolvable: at T=2, nu=0.8 → exp(-3.2)≈0.04
    extrap_lo = torch.tensor([0.02, 0.03, 0.04], device=device)
    extrap_hi = torch.tensor([0.60, 0.70, 0.80], device=device)
    assert float(in_range.min()) >= args.nu_lo and float(in_range.max()) <= args.nu_hi
    assert float(extrap_lo.max()) < args.nu_lo
    assert float(extrap_hi.min()) > args.nu_hi
    return {"in_range": in_range, "extrap_lo": extrap_lo, "extrap_hi": extrap_hi}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier B parametric NS: nu -> weights")
    p.add_argument("--generator", choices=["quantum", "classical"], required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-steps", type=int, default=12000,
                   help="hard cap; training usually stops earlier via --patience")
    p.add_argument("--patience", type=int, default=1500,
                   help="stop if rolling loss has not improved for this many steps")
    p.add_argument("--plateau-window", type=int, default=200)
    p.add_argument("--min-delta", type=float, default=0.005,
                   help="relative improvement required to reset patience "
                        "(0.005 = 0.5%% drop in rolling loss)")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-colloc", type=int, default=2048)
    p.add_argument("--n-bc", type=int, default=2048)
    p.add_argument("--n-tasks", type=int, default=8)
    p.add_argument("--lambda-bc", type=float, default=20.0)
    p.add_argument("--weight-reg", type=float, default=0.0)
    p.add_argument("--resample-every", type=int, default=50)
    p.add_argument("--nu-lo", type=float, default=0.05)
    p.add_argument("--nu-hi", type=float, default=0.5)
    p.add_argument("--t-max", type=float, default=2.0)
    p.add_argument("--hidden", type=int, nargs=2, default=[32, 32])
    p.add_argument("--n-qubits", type=int, default=None,
                   help="default 8 for reupload, 6 for expect")
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--bottleneck-width", type=int, default=64)
    p.add_argument("--freq-mode", choices=["linear", "geometric"], default="linear")
    p.add_argument("--qc-arch", choices=["reupload", "expect"], default="reupload",
                   help="reupload=v1 probs readout; expect=v2 log-nu + Z expectations")
    p.add_argument("--nu-encode", choices=["linear", "log"], default="log",
                   help="nu input normalisation (v2 default log; v1 ignores unless set)")
    p.add_argument("--hard-ic", action="store_true")
    p.add_argument("--run-id", type=str, default="")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_split(name: str, errs: dict, mean: float) -> None:
    print(f"\n[{name}] mean space-time rel-L2 = {mean*100:.3f}%")
    print(f"  {'nu':>8}  {'u%':>8}  {'v%':>8}  {'p%':>8}")
    for nu, rec in errs.items():
        e = rec["spacetime"]
        print(f"  {nu:8.4f}  {e['u']*100:8.3f}  {e['v']*100:8.3f}  {e['p']*100:8.3f}")


def main() -> None:
    args = parse_args()
    # architecture defaults
    if args.n_qubits is None:
        args.n_qubits = 6 if args.qc_arch == "expect" else 8
    if args.n_layers is None:
        args.n_layers = 6 if args.qc_arch == "expect" else 4
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or f"ns_par_{args.generator}_s{args.seed}"
    run_dir = Path("checkpoints") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = TargetPINNNS(
        fourier="tgv", hard_ic=args.hard_ic,
        hidden=tuple(args.hidden), t_max=args.t_max,
    ).to(device)
    gen = build_generator(args, model.in_dim, device)
    n_params = sum(p.numel() for p in gen.parameters())
    grids = eval_grids(args, device)

    print(f"Run: {run_id}  device={device}  generator={args.generator}  "
          f"qc_arch={args.qc_arch}")
    print(f"target weights={gen.total_weights}  generator params={n_params:,}  "
          f"freq_mode={args.freq_mode}")
    print(f"nu train [{args.nu_lo}, {args.nu_hi}]  T={args.t_max}  "
          f"decay {math.exp(-2*args.nu_hi*args.t_max):.4f} .. "
          f"{math.exp(-2*args.nu_lo*args.t_max):.4f}")
    print(f"stop: patience={args.patience} (rel delta {args.min_delta}) "
          f"or max_steps={args.max_steps}")

    x, y, t = make_colloc(args.n_colloc, device, args.t_max)
    x_bc, y_bc, t_bc = make_bc(args.n_bc, device)

    opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
    # Cosine over max_steps; if we stop early the schedule just hasn't annealed fully,
    # which is fine — plateau stop is the equal-footing criterion.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_steps, eta_min=1e-5)

    history = []
    roll = []
    best_roll = float("inf")
    stale = 0
    stopped_at = args.max_steps
    t0 = time.time()
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'roll':>10}  {'lr':>9}")
    for step in range(args.max_steps):
        if args.resample_every and step and step % args.resample_every == 0:
            x, y, t = make_colloc(args.n_colloc, device, args.t_max)
            x_bc, y_bc, t_bc = make_bc(args.n_bc, device)
        nus = sample_nu(args.n_tasks, args.nu_lo, args.nu_hi, device)

        opt.zero_grad()
        pde, bc = task_loss(model, gen, nus, x, y, t,
                            x_bc, y_bc, t_bc, args.lambda_bc)
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
                best_roll = roll_mean
                stale = 0
            else:
                stale += 1

        if step % args.log_every == 0:
            history.append({"step": step, "total": loss_v,
                            "pde": pde.item(), "bc": bc.item(),
                            "roll": roll_mean, "stale": stale})
            print(f"{step:6d}  {loss_v:12.6f}  {pde.item():12.6f}  "
                  f"{bc.item():12.6f}  {roll_mean:10.6f}  "
                  f"{opt.param_groups[0]['lr']:.2e}  [{time.time()-t0:.0f}s]")

        if stale >= args.patience:
            stopped_at = step + 1
            print(f"\nplateau stop at step {stopped_at}  "
                  f"(best rolling loss {best_roll:.6f})")
            break

    elapsed = time.time() - t0
    splits = {}
    means = {}
    for name, nus in grids.items():
        errs = evaluate_family(model, gen, nus, device, args.t_max)
        splits[name] = errs
        means[name] = summarise(errs)
        print_split(name, errs, means[name])

    print(f"\nelapsed={elapsed:.0f}s  stopped_at={stopped_at}/{args.max_steps}")

    torch.save(gen.state_dict(), run_dir / "generator.pt")
    (run_dir / "config.json").write_text(json.dumps(vars(args) | {
        "run_id": run_id, "device": str(device),
        "n_generator_params": n_params, "n_target_weights": gen.total_weights,
        "stopped_at": stopped_at,
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id,
        "generator": args.generator,
        "qc_arch": args.qc_arch,
        "nu_encode": args.nu_encode,
        "freq_mode": args.freq_mode,
        "n_generator_params": n_params,
        "elapsed_s": round(elapsed, 1),
        "stopped_at": stopped_at,
        "best_rolling_loss": None if best_roll == float("inf") else round(best_roll, 8),
        "mean_spacetime_rel_l2_in_range": round(means["in_range"], 6),
        "mean_spacetime_rel_l2_extrap_lo": round(means["extrap_lo"], 6),
        "mean_spacetime_rel_l2_extrap_hi": round(means["extrap_hi"], 6),
        "errors": splits,
        "history": history,
    }, indent=2, default=str))
    print(f"Saved -> {run_dir}/")


if __name__ == "__main__":
    main()
