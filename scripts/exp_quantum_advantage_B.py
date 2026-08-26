"""Experiment B: ν-family QNN vs matched classical generator on vortex merger.

Requires multi-ν DNS from scripts/gen_merger_dns_family.py.

Train on all ν except an optional held-out viscosity; evaluate train + holdout.

  .venv/bin/python scripts/exp_quantum_advantage_B.py --arm quantum --holdout-nu 0.008 --steps 25000
  .venv/bin/python scripts/exp_quantum_advantage_B.py --arm classical --holdout-nu 0.008 --steps 25000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdes.ns2d.physics_loss import _grad
from qt_pinn.ns2d_spectral import TWO_PI, ic_values_from_dns, sample_dns
from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_cond import ConditionedClassicalGeneratorV2, ConditionedQuantumGeneratorV2
from qt_pinn.tgv_demo import resolve_device
from scripts.exp_merger_omega import GATE_TIMES, HarmMLP, eval_fields, V3
from scripts.train_merger_qt_fast import grid_fd_loss

ROOT = Path(__file__).resolve().parents[1]
DNS_FAMILY = ROOT / "blog" / "checkpoint" / "v4" / "dns_family"
DEFAULT_OUT = ROOT / "blog" / "checkpoint" / "v4" / "advantage_B"

HIDDEN = (48, 48)
K_MAX = 3
N_QUBITS = 8
N_LAYERS = 8
BOTTLENECK = 64
NU_RANGE = (0.002, 0.02)
N_COLLOC = 8192
N_IC = 2048


def load_family(device: torch.device, holdout_nu: float | None = None) -> tuple[list[dict], list[dict]]:
    dirs = sorted(DNS_FAMILY.glob("nu_*"))
    if not dirs:
        raise FileNotFoundError(f"No DNS under {DNS_FAMILY}; run gen_merger_dns_family.py --all")
    train, hold = [], []
    for d in dirs:
        dns = torch.load(d / "reference.pt", map_location="cpu", weights_only=False)
        dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
        nu = float(dns_g["nu"])
        if holdout_nu is not None and abs(nu - holdout_nu) < 1e-9:
            hold.append(dns_g)
            print(f"holdout ν={nu} from {d.name}", flush=True)
        else:
            train.append(dns_g)
            print(f"train   ν={nu} from {d.name}", flush=True)
    if holdout_nu is not None and not hold:
        raise ValueError(f"holdout ν={holdout_nu} not found in {DNS_FAMILY}")
    if not train:
        raise ValueError("empty train family")
    return train, hold


def field_loss(pred, tgt, w_dns, x, y):
    """DNS fit + relative curl (absolute curl MSE blows up early)."""
    curl = _grad(pred[:, 1], x) - _grad(pred[:, 0], y)
    uv_mse = F.mse_loss(pred, tgt)
    curl_rel = ((curl - w_dns).pow(2).sum() / (w_dns.pow(2).sum() + 1e-8)).sqrt()
    return 50.0 * uv_mse + 25.0 * curl_rel


def load_teacher(device: torch.device) -> HarmMLP | None:
    cfg_path = V3 / "classical" / "config.json"
    model_path = V3 / "classical" / "model.pt"
    if not cfg_path.exists() or not model_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    teacher = HarmMLP(
        hidden=tuple(cfg["hidden"]),
        t_max=float(cfg.get("t_max", 15.0)),
        k_max=int(cfg.get("k_max", 6)),
        axis_extra=int(cfg.get("axis_extra", 0)),
    ).to(device)
    teacher.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def eval_on_family(predict_fn, family, device) -> dict:
    if not family:
        return {"per_nu": {}, "omega_mean": float("nan"), "omega_max": float("nan")}
    per = {}
    omegas = []
    for dns_g in family:
        nu = float(dns_g["nu"])
        errs = eval_fields(
            lambda a, b, c, nu=nu: predict_fn(a, b, c, nu).detach(),
            dns_g, device, times=GATE_TIMES, step=1,
        )
        per[f"{nu:.4f}"] = {"omega": errs["omega_max"], "vel": errs["vel_max"]}
        omegas.append(errs["omega_max"])
    return {"per_nu": per, "omega_mean": float(sum(omegas) / len(omegas)), "omega_max": float(max(omegas))}


def train_arm(
    gen,
    target,
    train_fam,
    eval_fam,
    device,
    steps: int,
    name: str,
    lr: float = 3e-4,
    teacher: HarmMLP | None = None,
    mid_only: bool = False,
):
    opt = torch.optim.Adam(list(gen.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    best = 1e9
    best_sd = None
    history = []
    t0 = time.time()
    cycle = list(GATE_TIMES)
    # Focus on mid-range ν first — extremes were collapsing prior B runs.
    pool = train_fam
    if mid_only:
        mid = [d for d in train_fam if 0.0035 <= float(d["nu"]) <= 0.015]
        if mid:
            pool = mid

    def pred_fn(a, b, c, nu_v):
        pk = gen(torch.tensor([nu_v], device=device, dtype=torch.float32))
        ww = {k: v[0] for k, v in pk.items()}
        return target(a, b, c, ww)

    for step in range(steps):
        dns_g = pool[step % len(pool)]
        if (not mid_only) and step % 4 == 0:
            # periodically revisit extremes if present (full-family mode only)
            hard = [d for d in train_fam if float(d["nu"]) >= 0.015 or float(d["nu"]) <= 0.0025]
            if hard:
                dns_g = hard[(step // 4) % len(hard)]
        nu = float(dns_g["nu"])
        x, y, t, tgt, w_dns = sample_dns(dns_g, N_COLLOC, device, t_sample="merger")
        x, y = x.requires_grad_(True), y.requires_grad_(True)
        packed = gen(torch.tensor([nu], device=device, dtype=torch.float32))
        w = {k: v[0] for k, v in packed.items()}
        pred = target(x, y, t, w)
        loss = field_loss(pred, tgt, w_dns, x, y)
        if teacher is not None and abs(nu - 0.005) < 1e-6:
            with torch.no_grad():
                teach = teacher(x, y, t)
            loss = loss + 20.0 * F.mse_loss(pred, teach)
        tv = cycle[step % len(cycle)]
        loss = loss + grid_fd_loss(
            lambda a, b, c: target(a, b, c, w), dns_g, tv, 48, 6.0, 20.0,
        )
        x_ic = torch.empty(N_IC, device=device).uniform_(0, TWO_PI)
        y_ic = torch.empty(N_IC, device=device).uniform_(0, TWO_PI)
        u0, v0, p0 = ic_values_from_dns(dns_g, x_ic, y_ic)
        loss = loss + 20.0 * F.mse_loss(
            target(x_ic, y_ic, torch.zeros(N_IC, device=device), w),
            torch.stack([u0, v0, p0], -1),
        )
        if not torch.isfinite(loss):
            print(f"{name} non-finite loss at step {step}; skip", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 0.5)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps - 1:
            metrics = eval_on_family(pred_fn, eval_fam, device)
            per = " ".join(f"{k}:{100*v['omega']:.1f}%" for k, v in metrics["per_nu"].items())
            print(
                f"{name} {step:6d}  ω_mean={100*metrics['omega_mean']:.2f}%  "
                f"ω_max={100*metrics['omega_max']:.2f}%  [{time.time()-t0:.0f}s]  {per}",
                flush=True,
            )
            history.append({
                "step": step,
                "omega_mean": metrics["omega_mean"],
                "omega_max": metrics["omega_max"],
                "per_nu": metrics["per_nu"],
            })
            if metrics["omega_mean"] < best:
                best = metrics["omega_mean"]
                best_sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
    if best_sd is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
    gen.load_state_dict(best_sd)
    final = eval_on_family(pred_fn, eval_fam, device)
    return gen, final, best_sd, history, time.time() - t0


def ablate_circuit(gen, target, family, device) -> dict:
    def pred_fn(a, b, c, nu_v):
        pk = gen(torch.tensor([nu_v], device=device, dtype=torch.float32))
        ww = {k: v[0] for k, v in pk.items()}
        return target(a, b, c, ww)

    full = eval_on_family(pred_fn, family, device)
    if not hasattr(gen, "q_weights"):
        return {"omega_full": full["omega_mean"], "circuit_used": False, "note": "no q_weights"}

    bak_q = gen.q_weights.detach().clone()
    with torch.no_grad():
        gen.q_weights.copy_(torch.randn_like(gen.q_weights) * 0.5)
    rand = eval_on_family(pred_fn, family, device)
    with torch.no_grad():
        gen.q_weights.copy_(bak_q)

    def pred_zero(a, b, c, nu_v):
        zeros = torch.zeros(1, gen.feat_dim, device=device)
        flat = gen.proj(zeros)
        ww = {k: v[0] for k, v in gen._split(flat).items()}
        return target(a, b, c, ww)

    zero = eval_on_family(pred_zero, family, device)
    used = (
        (rand["omega_mean"] - full["omega_mean"]) > 0.01
        or (zero["omega_mean"] - full["omega_mean"]) > 0.01
    )
    return {
        "omega_full": full["omega_mean"],
        "omega_random_q": rand["omega_mean"],
        "omega_zero_feats": zero["omega_mean"],
        "delta_random": rand["omega_mean"] - full["omega_mean"],
        "delta_zero": zero["omega_mean"] - full["omega_mean"],
        "circuit_used": used,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=("quantum", "classical"), required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-qubits", type=int, default=N_QUBITS)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--bottleneck", type=int, default=BOTTLENECK)
    ap.add_argument("--holdout-nu", type=float, default=0.008,
                    help="Viscosity held out of training for interpolation test")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mid-only", action="store_true",
                    help="Train primarily on mid-range ν (0.0035–0.015)")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--tag-suffix", type=str, default="",
                    help="Optional suffix for output directory name")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device("cuda")
    train_fam, hold_fam = load_family(device, holdout_nu=args.holdout_nu)
    all_fam = train_fam + hold_fam
    t_max = float(train_fam[0]["t_max"])
    target = TargetPINNNS(
        fourier="harm", hard_ic=False, hidden=HIDDEN, t_max=t_max,
        n_freqs=K_MAX, fourier_seed=0,
    ).to(device)
    teacher = None if args.no_teacher else load_teacher(device)
    if teacher is not None:
        # Align teacher t_max with DNS
        if hasattr(teacher, "t_max"):
            pass
        print("teacher loaded (ν≈0.005 distill assist)", flush=True)

    tag = f"{args.arm}_q{args.n_qubits}_L{args.n_layers}_hold{args.holdout_nu}"
    if args.mid_only:
        tag += "_mid"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    if args.arm == "quantum":
        gen = ConditionedQuantumGeneratorV2(
            in_dim=target.in_dim, h1=HIDDEN[0], h2=HIDDEN[1], out_dim=3,
            n_qubits=args.n_qubits, n_layers=args.n_layers, bottleneck_width=args.bottleneck,
            nu_range=NU_RANGE, nu_encode="log",
        ).to(device)
    else:
        gen = ConditionedClassicalGeneratorV2(
            in_dim=target.in_dim, h1=HIDDEN[0], h2=HIDDEN[1], out_dim=3,
            n_qubits=args.n_qubits, bottleneck_width=args.bottleneck,
            nu_range=NU_RANGE, nu_encode="log",
        ).to(device)

    n_gen = sum(p.numel() for p in gen.parameters())
    print(f"{args.arm} generator params={n_gen}  train_ν={[float(d['nu']) for d in train_fam]}", flush=True)

    gen, final_train, sd, hist, elapsed = train_arm(
        gen, target, train_fam, train_fam, device, args.steps, args.arm, lr=args.lr,
        teacher=teacher, mid_only=args.mid_only,
    )

    def pred_fn(a, b, c, nu_v):
        pk = gen(torch.tensor([nu_v], device=device, dtype=torch.float32))
        ww = {k: v[0] for k, v in pk.items()}
        return target(a, b, c, ww)

    final_hold = eval_on_family(pred_fn, hold_fam, device)
    final_all = eval_on_family(pred_fn, all_fam, device)
    ablation = ablate_circuit(gen, target, train_fam, device) if args.arm == "quantum" else None

    torch.save(sd, out / "generator.pt")
    payload = {
        "arm": args.arm,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "bottleneck": args.bottleneck,
        "n_gen_params": n_gen,
        "steps": args.steps,
        "elapsed_s": elapsed,
        "holdout_nu": args.holdout_nu,
        "train_nus": [float(d["nu"]) for d in train_fam],
        "final_train": final_train,
        "final_holdout": final_hold,
        "final_all": final_all,
        "ablation": ablation,
        "history": hist,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "arm": args.arm,
        "train_ω_mean": final_train["omega_mean"],
        "holdout_ω_mean": final_hold["omega_mean"],
        "ablation": ablation,
    }, indent=2), flush=True)
    if ablation is not None and not ablation["circuit_used"]:
        print("FAIL: circuit unused", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
