"""Train QT-PINN on 2D unsteady incompressible Navier-Stokes (Taylor-Green vortex).

PDE: continuity + momentum x + momentum y
IC:  Taylor-Green vortex at t=0, exact solution known for all t.

Settings mirror run_0072 (train_gpu.py):
  adam_lr=0.005, adam_steps=18000, cosine annealing, lbfgs off, seed=0,
  n_colloc=4096, n_bc=4096, lambda_bc=1.0, weight_reg=0.1, grad_clip=1.0,
  bottleneck_width=64, learned projection (quantum probs -> linear -> weights).

Run from repo root:
  .venv/bin/python scripts/train_ns.py

Saves to checkpoints/ns_run_NNNN/:
  q_weights.pt, config.json, results.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json, math, time
import torch
import torch.nn as nn
import numpy as np

from qt_pinn.pinn_target_ns import TargetPINNNS, W1_SIZE, W2_SIZE, W3_SIZE
from qt_pinn.qnn_generator_ns import QuantumWeightGeneratorNS, N_STATES, TOTAL_WEIGHTS
from pdes.ns2d.physics_loss import compute_ns_loss, exact_solution

# ---- hyperparameters (matching run_0072) ----
SEED            = 0
N_COLLOC        = 4096
N_BC            = 4096
LAMBDA_BC       = 1.0
ADAM_LR         = 0.005
ADAM_STEPS      = 18000
COSINE_ANNEAL   = True
COSINE_ETA_MIN  = 1e-5
WARMUP_STEPS    = 0
LOG_EVERY       = 500
GRAD_CLIP_NORM  = 1.0
WEIGHT_REG      = 0.1
BOTTLENECK_W    = 64

LBFGS_STEPS     = 0   # off, same as run_0072
LBFGS_LR        = 0.1
LBFGS_MAX_ITER  = 10

# domain: [0, 2pi]^2 x [0, 1]
X_LO, X_HI = 0.0, 2.0 * math.pi
T_HI = 1.0

def _next_run_dir(base: Path = Path("checkpoints")) -> tuple[str, Path]:
    base.mkdir(exist_ok=True)
    existing = sorted(base.glob("ns_run_*"))
    n = int(existing[-1].name.split("_")[-1]) + 1 if existing else 1
    run_id = f"ns_run_{n:04d}"
    return run_id, base / run_id


def _resolve_run_dir(base: Path, run_id_arg: str) -> tuple[str, Path]:
    """Resolve a unique run directory.

    If run_id_arg is empty, falls back to auto-increment ns_run_XXXX.
    If run_id_arg is non-empty, uses it as an exact directory name; errors if it exists.
    """
    if not run_id_arg:
        return _next_run_dir(base)
    base.mkdir(exist_ok=True)
    run_dir = base / run_id_arg
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run dir: {run_dir}")
    return run_id_arg, run_dir


def make_colloc(n: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI).requires_grad_(True)
    t = torch.empty(n, device=device).uniform_(0.0, T_HI).requires_grad_(True)
    return x, y, t


def make_bc(n: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """IC at t=0, exact Taylor-Green values."""
    x = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    y = torch.empty(n, device=device).uniform_(X_LO, X_HI)
    t = torch.zeros(n, device=device)
    return x, y, t


def make_optimizer(params, adam_lr: float, adam_steps: int):
    opt = torch.optim.Adam(params, lr=adam_lr)
    if COSINE_ANNEAL:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=max(WARMUP_STEPS, 1))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(adam_steps - WARMUP_STEPS, 1), eta_min=COSINE_ETA_MIN)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[WARMUP_STEPS])
    else:
        sched = None
    return opt, sched


def forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc):
    weights = gen()
    pde, bc = compute_ns_loss(model, x, y, t, x_bc, y_bc, t_bc, weights)
    return pde, bc, weights


def weight_reg_loss(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """L2 regularisation on generated MLP weights (not generator params)."""
    return sum(w.pow(2).mean() for w in weights.values())


def evaluate_exact(model, gen, device: torch.device) -> dict:
    """Evaluate relative L2 error against exact Taylor-Green solution on a grid."""
    model.eval()
    n = 32
    xs = torch.linspace(X_LO, X_HI, n, device=device)
    ys = torch.linspace(X_LO, X_HI, n, device=device)
    ts = [0.0, 0.25, 0.5, 1.0]
    results = {}
    with torch.no_grad():
        weights = gen()
    for tv in ts:
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        x = xg.flatten(); y = yg.flatten()
        t = torch.full_like(x, tv)
        with torch.no_grad():
            pred = model(x, y, t, weights)
        u_ex, v_ex, p_ex = exact_solution(x, y, t)
        def rel_l2(p, e):
            return ((p - e).norm() / (e.norm() + 1e-10)).item()
        results[tv] = {
            "u": rel_l2(pred[:, 0], u_ex),
            "v": rel_l2(pred[:, 1], v_ex),
            "p": rel_l2(pred[:, 2], p_ex),
        }
    model.train()
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NS QT-PINN with resource controls.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu-threads", type=int, default=0,
                   help="Torch intra-op CPU threads (0 keeps default).")
    p.add_argument("--batch-multiplier", type=int, default=1,
                   help="Scales collocation/BC batch sizes to increase utilization.")
    p.add_argument("--run-id", type=str, default="",
                   help="If non-empty, writes to checkpoints/<run-id>/ instead of auto ns_run_XXXX.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed used for collocation/IC and weight generator init.")
    p.add_argument("--adam-steps", type=int, default=ADAM_STEPS)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--lambda-bc", type=float, default=LAMBDA_BC)
    p.add_argument("--weight-reg", type=float, default=WEIGHT_REG)
    p.add_argument("--bottleneck-width", type=int, default=BOTTLENECK_W)
    p.add_argument("--fourier-sigma", type=float, default=1.0)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        # Keep inter-op modest to reduce thread oversubscription.
        torch.set_num_interop_threads(max(1, min(4, args.cpu_threads // 2)))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id, run_dir = _resolve_run_dir(Path("checkpoints"), args.run_id)
    run_dir.mkdir(parents=True)
    n_colloc_eff = N_COLLOC * args.batch_multiplier
    n_bc_eff = N_BC * args.batch_multiplier

    print(f"Run: {run_id}  ->  {run_dir}/  device={device}")
    print(f"PDE: 2D unsteady incompressible NS, Taylor-Green vortex IC")
    print(f"     {TOTAL_WEIGHTS} MLP weights, {N_STATES} quantum basis states ({N_STATES}-dim circuit)")
    print(f"Batching: colloc={n_colloc_eff} (base {N_COLLOC} x{args.batch_multiplier}), "
          f"bc={n_bc_eff} (base {N_BC} x{args.batch_multiplier})")
    print(f"Hyperparams: lambda_bc={args.lambda_bc} weight_reg={args.weight_reg} "
          f"bottleneck={args.bottleneck_width} fourier_sigma={args.fourier_sigma}")
    if args.cpu_threads > 0:
        print(f"CPU threads: intra_op={torch.get_num_threads()} inter_op={torch.get_num_interop_threads()}")

    model = TargetPINNNS(fourier_sigma=args.fourier_sigma).to(device)
    gen   = QuantumWeightGeneratorNS(bottleneck_width=args.bottleneck_width)
    gen.proj.to(device)   # projection on GPU; q_weights stay on CPU (PennyLane constraint)

    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"Generator params: {n_params:,}")

    x,    y,    t    = make_colloc(n_colloc_eff, device)
    x_bc, y_bc, t_bc = make_bc(n_bc_eff, device)

    lam = float(args.lambda_bc)
    t0 = time.time()

    def _step(opt, step):
        opt.zero_grad()
        pde, bc, weights = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc)
        wreg = weight_reg_loss(weights)
        loss = pde + lam * bc + args.weight_reg * wreg
        loss.backward()
        nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        opt.step()
        return loss.item(), pde.item(), bc.item()

    opt, sched = make_optimizer(params, ADAM_LR, args.adam_steps)
    lr_end = COSINE_ETA_MIN if COSINE_ANNEAL else ADAM_LR
    print(f"\nAdam  lr={ADAM_LR}->{lr_end}  steps={args.adam_steps}  cosine={COSINE_ANNEAL}")
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'lr':>10}")

    for step in range(args.adam_steps):
        total, pde_v, bc_v = _step(opt, step)
        if sched:
            sched.step()
        if step % args.log_every == 0:
            lr_now = opt.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"{step:6d}  {total:12.6f}  {pde_v:12.6f}  {bc_v:12.6f}  {lr_now:.2e}  [{elapsed:.0f}s]")

    # Final eval on training points — compute_ns_loss needs requires_grad on colloc pts
    pde_f_t, bc_f_t, _ = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc)
    pde_final = pde_f_t.item(); bc_final = bc_f_t.item()
    elapsed_s = time.time() - t0
    print(f"\nFinal  pde={pde_final:.7f}  bc={bc_final:.7f}  sum={pde_final+bc_final:.7f}  [{elapsed_s:.0f}s]")

    # Holdout eval — fresh colloc pts with requires_grad for AD
    torch.manual_seed(args.seed + 90000)
    xh, yh, th    = make_colloc(n_colloc_eff, device)
    xh_bc, yh_bc, th_bc = make_bc(n_bc_eff, device)
    pde_h_t, bc_h_t, _ = forward_losses(model, gen, xh, yh, th, xh_bc, yh_bc, th_bc)
    pde_hold = pde_h_t.item(); bc_hold = bc_h_t.item()
    print(f"Holdout pde={pde_hold:.7f}  bc={bc_hold:.7f}  pde_ratio={pde_hold/max(pde_final,1e-12):.4f}")

    # Exact relative L2 error
    exact_errs = evaluate_exact(model, gen, device)
    print("\nRelative L2 vs exact Taylor-Green:")
    print(f"  {'t':>5}  {'u_err%':>8}  {'v_err%':>8}  {'p_err%':>8}")
    for tv, errs in exact_errs.items():
        print(f"  {tv:5.2f}  {errs['u']*100:8.3f}  {errs['v']*100:8.3f}  {errs['p']*100:8.3f}")

    # Save
    torch.save(gen.state_dict(), run_dir / "q_weights.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "pde": "ns2d_taylor_green",
        "generator": "quantum_learned_proj", "device": str(device),
        "seed": args.seed, "n_colloc": n_colloc_eff, "n_bc": n_bc_eff,
        "batch_multiplier": args.batch_multiplier,
        "adam_lr": ADAM_LR, "adam_steps": args.adam_steps,
        "lambda_bc": args.lambda_bc, "weight_reg": args.weight_reg,
        "cosine": COSINE_ANNEAL, "bottleneck_width": args.bottleneck_width,
        "fourier_sigma": args.fourier_sigma,
        "n_qubits": 11, "n_states": N_STATES, "total_weights": TOTAL_WEIGHTS,
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id, "pde": "ns2d_taylor_green",
        "pde_loss": round(pde_final, 8), "bc_loss": round(bc_final, 8),
        "total": round(pde_final + bc_final, 8),
        "holdout_pde_loss": round(pde_hold, 8), "holdout_bc_loss": round(bc_hold, 8),
        "holdout_total": round(pde_hold + bc_hold, 8),
        "pde_ratio": round(pde_hold / max(pde_final, 1e-12), 4),
        "elapsed_s": round(elapsed_s, 1),
        "n_params": n_params,
        "exact_l2": {str(tv): {k: round(v, 6) for k, v in errs.items()}
                     for tv, errs in exact_errs.items()},
    }, indent=2))
    print(f"\nSaved -> {run_dir}/")


if __name__ == "__main__":
    main()
