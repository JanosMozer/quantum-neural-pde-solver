"""Train input-conditioned Burgers PINN: VQC vs matched classical.

Fair comparison: same collocation, IC, steps, lambda_bc; classical MLP has
>= the same number of trainable parameters as the VQC.

Presets
-------
  scout (default)  — ≤1 h quantum wall time; enough to accept/reject the idea
  full             — matched-budget long run (~many hours); use after scout looks promising

Scout (from repo root):
  .venv/bin/python scripts/train_burgers_vqc.py --model classical --seed 0 --run-id burg_vqc_c_scout
  .venv/bin/python scripts/train_burgers_vqc.py --model quantum  --seed 0 --run-id burg_vqc_q_scout
  .venv/bin/python scripts/compare_burgers_vqc.py checkpoints/burg_vqc_c_scout checkpoints/burg_vqc_q_scout

Full (only after scout suggests a quantum win is plausible):
  .venv/bin/python scripts/train_burgers_vqc.py --preset full --model classical --seed 0 --run-id burg_vqc_c_s0
  .venv/bin/python scripts/train_burgers_vqc.py --preset full --model quantum  --seed 0 --run-id burg_vqc_q_s0
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

from qt_pinn.burgers_vqc_pinn import (
    BurgersVQCPINN,
    BurgersClassicalPINN,
    count_params,
)
from pdes.burgers2d.physics_loss import compute_burgers_loss, ic_values, pde_rms, NU

X_LO, X_HI = -1.0, 1.0
T_HI = 1.0

# scout: ~7 s/step at 512 colloc on default.qubit → 400 steps ≈ 45–55 min
# full:  2048 colloc × 8000 steps ≈ tens of hours (same architecture)
PRESETS = {
    "scout": {
        "adam_steps": 400,
        "n_colloc": 512,
        "n_bc": 512,
        "n_holdout": 256,
        "n_holdout_bc": 256,
        "log_every": 25,
        "resample_every": 50,
    },
    "full": {
        "adam_steps": 8000,
        "n_colloc": 2048,
        "n_bc": 2048,
        "n_holdout": 1024,
        "n_holdout_bc": 512,
        "log_every": 200,
        "resample_every": 100,
    },
}


def make_colloc(n: int, device: torch.device):
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    t = torch.empty(n, device=device).uniform_(0.0, T_HI).requires_grad_(True)
    return x, y, t


def make_bc(n: int, device: torch.device):
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    t = torch.zeros(n, device=device)
    u, v = ic_values(x, y)
    return x, y, t, u, v


def build_model(args, device: torch.device) -> nn.Module:
    q_ref = BurgersVQCPINN(
        args.n_qubits, args.n_layers,
        hard_ic=args.hard_ic, ic_fn=ic_values if args.hard_ic else None,
    )
    if args.model == "quantum":
        model = q_ref
    else:
        model = BurgersClassicalPINN.matched_to(q_ref)
    return model.to(device)


@torch.no_grad()
def field_diagnostics(model: nn.Module, device: torch.device) -> dict[str, float]:
    """Detect the two degenerate solutions.

    collapse_ratio  — field rms at t=1 over IC rms; ~0 means u=v=0 collapse.
    correction_rms  — rms of the learned t-dependent part; ~0 under hard IC
                      means the model froze at the initial condition.
    """
    n = 4096
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    t1 = torch.ones(n, device=device)
    u0, v0 = ic_values(x, y)
    ic_rms = torch.sqrt((u0.pow(2) + v0.pow(2)).mean()).item()
    out = model(x, y, t1)
    field_rms = torch.sqrt(out.pow(2).sum(-1).mean()).item()
    ic_stack = torch.stack([u0, v0], dim=-1)
    corr_rms = torch.sqrt((out - ic_stack).pow(2).sum(-1).mean()).item()
    return {
        "ic_rms": ic_rms,
        "field_rms_t1": field_rms,
        "collapse_ratio": field_rms / max(ic_rms, 1e-12),
        "correction_rms": corr_rms,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Burgers VQC-PINN vs matched classical")
    p.add_argument("--model", choices=["quantum", "classical"], required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="scout",
                   help="scout ≤1h quantum; full = long matched run")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--adam-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="default 0.003 quantum, 0.005 classical")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-colloc", type=int, default=None)
    p.add_argument("--n-bc", type=int, default=None)
    p.add_argument("--n-holdout", type=int, default=None)
    p.add_argument("--n-holdout-bc", type=int, default=None)
    p.add_argument("--lambda-bc", type=float, default=10.0)
    p.add_argument("--hard-ic", dest="hard_ic", action="store_true", default=True,
                   help="u = u_IC + t*N ansatz (default); makes u=v=0 unreachable")
    p.add_argument("--soft-ic", dest="hard_ic", action="store_false",
                   help="penalise the IC instead; collapses to the trivial solution")
    p.add_argument("--nu", type=float, default=NU)
    p.add_argument("--resample-every", type=int, default=None)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--run-id", default="")
    return p.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    cfg = PRESETS[args.preset]
    for key, val in cfg.items():
        attr = key
        if getattr(args, attr) is None:
            setattr(args, attr, val)
    return args


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    args = apply_preset(args)
    if args.lr is None:
        args.lr = 0.003 if args.model == "quantum" else 0.005
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or f"burg_vqc_{args.model[0]}_{args.preset}_s{args.seed}"
    run_dir = Path("checkpoints") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    n_params = count_params(model)
    q_ref_n = count_params(BurgersVQCPINN(args.n_qubits, args.n_layers))

    print(f"Run: {run_id}  model={args.model}  preset={args.preset}  device={device}")
    print(f"params={n_params:,}  (VQC reference={q_ref_n:,})  "
          f"nu={args.nu}  lambda_bc={args.lambda_bc}")
    print(f"steps={args.adam_steps}  colloc={args.n_colloc}  bc={args.n_bc}  "
          f"holdout={args.n_holdout}")
    print(f"domain x,y in [{X_LO},{X_HI}]  t in [0,{T_HI}]  "
          f"ic={'hard' if args.hard_ic else 'soft'}")
    if args.preset == "scout" and args.model == "quantum":
        print("Note: scout target ≤1 h wall time; use --preset full only if scout looks promising.")

    x, y, t = make_colloc(args.n_colloc, device)
    x_bc, y_bc, t_bc, u_bc, v_bc = make_bc(args.n_bc, device)

    # fixed holdout sets for evaluation (smaller in scout so logging does not dominate)
    torch.manual_seed(args.seed + 777)
    x_h, y_h, t_h = make_colloc(args.n_holdout, device)
    x_h = x_h.detach().requires_grad_(True)
    y_h = y_h.detach().requires_grad_(True)
    t_h = t_h.detach().requires_grad_(True)
    x_bhc, y_bhc, t_bhc, u_bhc, v_bhc = make_bc(args.n_holdout_bc, device)
    torch.manual_seed(args.seed)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.adam_steps, eta_min=1e-5)

    history = []
    t0 = time.time()
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'hold_rms':>10}  "
          f"{'collapse':>8}  {'corr_rms':>8}")
    for step in range(args.adam_steps):
        if args.resample_every and step and step % args.resample_every == 0:
            x, y, t = make_colloc(args.n_colloc, device)
            x_bc, y_bc, t_bc, u_bc, v_bc = make_bc(args.n_bc, device)

        opt.zero_grad()
        pde, bc = compute_burgers_loss(
            model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc,
            weights=None, nu=args.nu)
        loss = pde + args.lambda_bc * bc
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            with torch.enable_grad():
                hrms = pde_rms(model, x_h, y_h, t_h,
                               x_bhc, y_bhc, t_bhc, u_bhc, v_bhc,
                               nu=args.nu)
            diag = field_diagnostics(model, device)
            history.append({"step": step, "total": loss.item(),
                            "pde": pde.item(), "bc": bc.item(),
                            "holdout_pde_rms": hrms, **diag})
            print(f"{step:6d}  {loss.item():12.6f}  {pde.item():12.6f}  "
                  f"{bc.item():12.6f}  {hrms:10.5f}  "
                  f"{diag['collapse_ratio']:8.3f}  {diag['correction_rms']:8.4f}  "
                  f"[{time.time()-t0:.0f}s]")

    elapsed = time.time() - t0
    pde_f, bc_f = compute_burgers_loss(
        model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, nu=args.nu)
    hold_rms = pde_rms(model, x_h, y_h, t_h,
                       x_bhc, y_bhc, t_bhc, u_bhc, v_bhc, nu=args.nu)
    train_rms = math.sqrt(pde_f.item())
    diag = field_diagnostics(model, device)

    print(f"\nFinal  train_pde_rms={train_rms:.5f}  bc={bc_f.item():.2e}  "
          f"holdout_pde_rms={hold_rms:.5f}  [{elapsed:.0f}s]")
    print(f"       collapse_ratio={diag['collapse_ratio']:.3f}  "
          f"correction_rms={diag['correction_rms']:.4f}  "
          f"ic_rms={diag['ic_rms']:.4f}")

    # A near-zero field satisfies Burgers exactly, so a low PDE RMS there is
    # meaningless; a near-zero correction means the model froze at the IC.
    valid = True
    if diag["collapse_ratio"] < 0.1:
        print("INVALID: collapsed to the trivial solution u=v=0 — PDE RMS is meaningless.")
        valid = False
    if args.hard_ic and diag["correction_rms"] < 0.01 * diag["ic_rms"]:
        print("INVALID: frozen at the initial condition — no time evolution learned.")
        valid = False
    if valid:
        print("OK: non-degenerate solution; PDE RMS is comparable across models.")

    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps(vars(args) | {
        "run_id": run_id, "device": str(device), "n_params": n_params,
        "vqc_reference_params": q_ref_n,
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id,
        "model": args.model,
        "preset": args.preset,
        "n_params": n_params,
        "elapsed_s": round(elapsed, 1),
        "train_pde_rms": round(train_rms, 6),
        "holdout_pde_rms": round(hold_rms, 6),
        "bc_loss": round(bc_f.item(), 8),
        "hard_ic": args.hard_ic,
        "valid": valid,
        **{k: round(v, 6) for k, v in diag.items()},
        "history": history,
    }, indent=2))
    print(f"Saved -> {run_dir}/")


if __name__ == "__main__":
    main()
