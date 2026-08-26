"""Experiment A: end-to-end QNN training on vortex merger.

Three arms — quantum generator, matched classical generator, direct HarmMLP —
all with deployed HarmMLP(48,48,k_max=3) / TargetPINNNS(harm) architecture.

  .venv/bin/python scripts/exp_quantum_advantage_A.py --steps 15000
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
from qt_pinn.qnn_generator_cond import (
    ConditionedClassicalGenerator,
    ConditionedClassicalGeneratorV2,
    ConditionedQuantumGenerator,
    ConditionedQuantumGeneratorV2,
)
from qt_pinn.tgv_demo import resolve_device
from scripts.exp_merger_omega import GATE_TIMES, HarmMLP, OMEGA_GATE, V3, eval_fields
from scripts.train_merger_qt_fast import BENCH_N, grid_fd_loss, latency_thr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "blog" / "checkpoint" / "v4" / "advantage_A"

HIDDEN = (48, 48)
K_MAX = 3
N_QUBITS = 8
N_LAYERS = 8
BOTTLENECK = 64
NU_RANGE = (0.001, 0.05)
N_COLLOC = 16384
N_IC = 3072
LR = 1e-3
CIRCUIT_DEGRAD_EPS = 0.002

W_DNS = 45.0
W_TEACH = 30.0
W_CURL = 40.0
W_GRID_UV = 8.0
W_GRID_CURL = 30.0
W_IC = 40.0


def build_target(t_max: float, device: torch.device) -> TargetPINNNS:
    return TargetPINNNS(
        fourier="harm",
        hard_ic=False,
        hidden=HIDDEN,
        t_max=t_max,
        n_freqs=K_MAX,
        fourier_seed=0,
    ).to(device)


def verify_harm_basis(target: TargetPINNNS, ref: HarmMLP, device: torch.device) -> None:
    d = (target.fourier.B - ref.fourier.B.to(device)).abs().max().item()
    if d > 1e-6:
        raise RuntimeError(f"Fourier B mismatch Δ={d}")


def load_teacher(dns_g: dict, device: torch.device) -> HarmMLP | None:
    cfg_path = V3 / "classical" / "config.json"
    model_path = V3 / "classical" / "model.pt"
    if not cfg_path.exists() or not model_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    teacher = HarmMLP(
        hidden=tuple(cfg["hidden"]),
        t_max=float(dns_g["t_max"]),
        k_max=int(cfg.get("k_max", 6)),
        axis_extra=int(cfg.get("axis_extra", 0)),
    ).to(device)
    teacher.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def packed_item(packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v[0] for k, v in packed.items()}


@torch.no_grad()
def quantum_forward_ablated(
    gen: nn.Module,
    nu: torch.Tensor,
    mode: str,
) -> dict[str, torch.Tensor]:
    out_device = next(gen.proj.parameters()).device
    if mode == "full":
        return packed_item(gen(nu))

    angles = gen._encode(nu)
    feat_dim = int(getattr(gen, "feat_dim", getattr(gen, "n_states")))
    if mode == "random_q":
        q_w = torch.randn_like(gen.q_weights) * 0.1
        raw = gen._circuit(angles.cpu(), q_w.cpu())
        if isinstance(raw, (list, tuple)):
            feats = torch.stack(raw, dim=-1).float().to(out_device)
        else:
            probs = raw.float()
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            feats = probs.to(out_device) * feat_dim - 1.0
    elif mode == "zero_feats":
        feats = torch.zeros(nu.shape[0], feat_dim, device=out_device)
    else:
        raise ValueError(f"unknown ablation mode={mode!r}")

    if feats.dim() == 1:
        feats = feats.unsqueeze(0)
    return packed_item(gen._split(gen.proj(feats)))


def train_direct(
    model: HarmMLP,
    dns_g: dict,
    device: torch.device,
    teacher: HarmMLP | None,
    steps: int,
    name: str,
) -> tuple[dict, dict[str, torch.Tensor], list]:
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-6)
    cycle = list(GATE_TIMES)
    best_w, best_sd = float("inf"), None
    history = []
    t0 = time.time()

    for step in range(steps):
        x, y, t, tgt, w_dns = sample_dns(dns_g, N_COLLOC, device, t_sample="merger")
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        pred = model(x, y, t)
        loss = W_DNS * F.mse_loss(pred, tgt)
        if teacher is not None:
            with torch.no_grad():
                teach = teacher(x, y, t)
            loss = loss + W_TEACH * F.mse_loss(pred, teach)
        curl = _grad(pred[:, 1], x) - _grad(pred[:, 0], y)
        loss = loss + W_CURL * F.mse_loss(curl, w_dns)
        tv = cycle[step % len(cycle)]
        loss = loss + grid_fd_loss(model, dns_g, tv, 64, W_GRID_UV, W_GRID_CURL)
        x_ic = torch.empty(N_IC, device=device).uniform_(0, TWO_PI)
        y_ic = torch.empty(N_IC, device=device).uniform_(0, TWO_PI)
        u0, v0, p0 = ic_values_from_dns(dns_g, x_ic, y_ic)
        loss = loss + W_IC * F.mse_loss(
            model(x_ic, y_ic, torch.zeros(N_IC, device=device)),
            torch.stack([u0, v0, p0], dim=-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps - 1:
            errs = eval_fields(lambda a, b, c: model(a, b, c).detach(), dns_g, device,
                               times=GATE_TIMES, step=1)
            print(
                f"{name} {step:6d}  loss={loss.item():.4f}  "
                f"vel={100*errs['vel_max']:.2f}%  ω={100*errs['omega_max']:.2f}%  "
                f"[{time.time()-t0:.0f}s]",
                flush=True,
            )
            history.append({
                "step": step, "loss": loss.item(),
                "vel_max": errs["vel_max"], "omega_max": errs["omega_max"],
            })
            if errs["omega_max"] < best_w:
                best_w = errs["omega_max"]
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if errs["omega_max"] <= OMEGA_GATE:
                print(f"{name} EARLY STOP ω≤{100*OMEGA_GATE:.0f}%", flush=True)
                break

    model.load_state_dict(best_sd)
    final = eval_fields(lambda a, b, c: model(a, b, c).detach(), dns_g, device,
                        times=GATE_TIMES, step=1)
    ms, thr = latency_thr(lambda a, b, c: model(a, b, c), device)
    result = {
        "arm": name,
        "elapsed_s": round(time.time() - t0, 1),
        "best_omega_during": best_w,
        "final": final,
        "latency_ms": ms,
        "throughput_pts_s": thr,
        "n_deployed_params": sum(p.numel() for p in model.parameters()),
        "history": history,
        "passed_omega_gate": final["omega_max"] <= OMEGA_GATE,
    }
    return result, best_sd, history


def train_generator(
    gen: nn.Module,
    target: TargetPINNNS,
    dns_g: dict,
    device: torch.device,
    teacher: HarmMLP | None,
    steps: int,
    name: str,
    lr: float = LR,
    circuit_lr_scale: float = 1.0,
    curl_scale: float = 1.0,
) -> tuple[dict, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    nu = float(dns_g["nu"])
    nu_t = torch.tensor([nu], device=device, dtype=torch.float32)
    w_curl = W_CURL * curl_scale
    w_grid_curl = W_GRID_CURL * curl_scale
    # Dual LR: keep circuit / angle params smaller than classical proj head.
    circuit_keys = ("q_weights", "freq_scale")
    circuit_params, head_params = [], []
    for n, p in gen.named_parameters():
        if any(k in n for k in circuit_keys):
            circuit_params.append(p)
        else:
            head_params.append(p)
    if circuit_params and circuit_lr_scale != 1.0:
        opt = torch.optim.Adam(
            [
                {"params": head_params, "lr": lr},
                {"params": circuit_params, "lr": lr * circuit_lr_scale},
            ]
        )
    else:
        opt = torch.optim.Adam(list(gen.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-6)
    cycle = list(GATE_TIMES)
    best_w, best_gen_sd = float("inf"), None
    best_deployed = None
    history = []
    t0 = time.time()

    def predict_from_w(w: dict[str, torch.Tensor]):
        def fn(x, y, t):
            return target(x, y, t, w)
        return fn

    for step in range(steps):
        packed = gen(nu_t)
        w = packed_item(packed)
        x, y, t, tgt, w_dns = sample_dns(dns_g, N_COLLOC, device, t_sample="merger")
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        pred = target(x, y, t, w)
        loss = W_DNS * F.mse_loss(pred, tgt)
        if teacher is not None:
            with torch.no_grad():
                teach = teacher(x, y, t)
            loss = loss + W_TEACH * F.mse_loss(pred, teach)
        curl = _grad(pred[:, 1], x) - _grad(pred[:, 0], y)
        loss = loss + w_curl * F.mse_loss(curl, w_dns)
        tv = cycle[step % len(cycle)]
        loss = loss + grid_fd_loss(predict_from_w(w), dns_g, tv, 64, W_GRID_UV, w_grid_curl)
        x_ic = torch.empty(N_IC, device=device).uniform_(0, TWO_PI)
        y_ic = torch.empty(N_IC, device=device).uniform_(0, TWO_PI)
        u0, v0, p0 = ic_values_from_dns(dns_g, x_ic, y_ic)
        loss = loss + W_IC * F.mse_loss(
            target(x_ic, y_ic, torch.zeros(N_IC, device=device), w),
            torch.stack([u0, v0, p0], dim=-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(gen.parameters()), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps - 1:
            pred_eval = predict_from_w(packed_item(gen(nu_t)))
            errs = eval_fields(lambda a, b, c: pred_eval(a, b, c).detach(), dns_g, device,
                               times=GATE_TIMES, step=1)
            print(
                f"{name} {step:6d}  loss={loss.item():.4f}  "
                f"vel={100*errs['vel_max']:.2f}%  ω={100*errs['omega_max']:.2f}%  "
                f"[{time.time()-t0:.0f}s]",
                flush=True,
            )
            history.append({
                "step": step, "loss": loss.item(),
                "vel_max": errs["vel_max"], "omega_max": errs["omega_max"],
            })
            if errs["omega_max"] < best_w:
                best_w = errs["omega_max"]
                best_gen_sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
                with torch.no_grad():
                    best_deployed = {
                        k: v.detach().cpu().clone()
                        for k, v in packed_item(gen(nu_t)).items()
                    }
            if errs["omega_max"] <= OMEGA_GATE:
                print(f"{name} EARLY STOP ω≤{100*OMEGA_GATE:.0f}%", flush=True)
                break

    gen.load_state_dict(best_gen_sd)
    with torch.no_grad():
        deployed = {k: v.detach().cpu().clone() for k, v in packed_item(gen(nu_t)).items()}

    pred_final = predict_from_w({k: v.to(device) for k, v in deployed.items()})
    final = eval_fields(lambda a, b, c: pred_final(a, b, c).detach(), dns_g, device,
                        times=GATE_TIMES, step=1)
    ms, thr = latency_thr(pred_final, device)
    n_dep = sum(v.numel() for v in deployed.values())
    result = {
        "arm": name,
        "elapsed_s": round(time.time() - t0, 1),
        "best_omega_during": best_w,
        "final": final,
        "latency_ms": ms,
        "throughput_pts_s": thr,
        "n_deployed_params": n_dep,
        "n_generator_params": sum(p.numel() for p in gen.parameters()),
        "history": history,
        "passed_omega_gate": final["omega_max"] <= OMEGA_GATE,
    }
    return result, best_gen_sd, deployed


def verify_circuit_usage(
    gen: nn.Module,
    target: TargetPINNNS,
    dns_g: dict,
    device: torch.device,
    nu: float,
) -> dict:
    nu_t = torch.tensor([nu], device=device, dtype=torch.float32)
    gen.eval()

    def make_pred(mode: str):
        w = quantum_forward_ablated(gen, nu_t, mode)
        w_dev = {k: v.to(device) for k, v in w.items()}

        def fn(x, y, t):
            return target(x, y, t, w_dev)

        return fn

    full = eval_fields(lambda a, b, c: make_pred("full")(a, b, c).detach(), dns_g, device,
                       times=GATE_TIMES, step=1)
    rand_q = eval_fields(lambda a, b, c: make_pred("random_q")(a, b, c).detach(), dns_g, device,
                         times=GATE_TIMES, step=1)
    zero = eval_fields(lambda a, b, c: make_pred("zero_feats")(a, b, c).detach(), dns_g, device,
                       times=GATE_TIMES, step=1)

    d_rand = rand_q["omega_max"] - full["omega_max"]
    d_zero = zero["omega_max"] - full["omega_max"]
    used = d_rand > CIRCUIT_DEGRAD_EPS or d_zero > CIRCUIT_DEGRAD_EPS
    return {
        "omega_full": full["omega_max"],
        "omega_random_q": rand_q["omega_max"],
        "omega_zero_feats": zero["omega_max"],
        "delta_random_q": d_rand,
        "delta_zero_feats": d_zero,
        "circuit_used": used,
    }


def save_gen_arm(
    out_dir: Path,
    gen_sd: dict,
    deployed: dict,
    result: dict,
    arm: str,
    gen_kind: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gen_sd, out_dir / "generator.pt")
    torch.save(deployed, out_dir / "deployed_weights.pt")
    cfg = {
        "arm": arm,
        "gen_kind": gen_kind,
        "qt_hidden": list(HIDDEN),
        "qt_fourier": "harm",
        "n_freqs": K_MAX,
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "bottleneck_width": BOTTLENECK,
        "nu_range": list(NU_RANGE),
        "nu_encode": "log",
        "n_deployed_params": result["n_deployed_params"],
        "bench_n_pts": BENCH_N,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out_dir / "results.json").write_text(json.dumps({
        "omega_rel_l2_max": result["final"]["omega_max"],
        "vel_rel_l2_max": result["final"]["vel_max"],
        "gate_pass_omega_2pct": result["passed_omega_gate"],
        "latency_ms": result["latency_ms"],
        "throughput_pts_s": result["throughput_pts_s"],
        "n_deployed_params": result["n_deployed_params"],
        "elapsed_s": result["elapsed_s"],
        "per_time": result["final"].get("times", {}),
        "history": result["history"],
    }, indent=2) + "\n")


def save_direct_arm(out_dir: Path, model_sd: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_sd, out_dir / "model.pt")
    cfg = {
        "arm": "classical_direct",
        "arch": "harm_mlp",
        "hidden": list(HIDDEN),
        "k_max": K_MAX,
        "n_deployed_params": result["n_deployed_params"],
        "bench_n_pts": BENCH_N,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out_dir / "results.json").write_text(json.dumps({
        "omega_rel_l2_max": result["final"]["omega_max"],
        "vel_rel_l2_max": result["final"]["vel_max"],
        "gate_pass_omega_2pct": result["passed_omega_gate"],
        "latency_ms": result["latency_ms"],
        "throughput_pts_s": result["throughput_pts_s"],
        "n_deployed_params": result["n_deployed_params"],
        "elapsed_s": result["elapsed_s"],
        "per_time": result["final"].get("times", {}),
        "history": result["history"],
    }, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment A: end-to-end QNN on vortex merger")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument(
        "--arm",
        choices=("all", "quantum", "classical_gen", "classical_direct"),
        default="all",
        help="Run one arm (for parallel jobs) or all sequentially",
    )
    ap.add_argument("--n-qubits", type=int, default=N_QUBITS)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--bottleneck", type=int, default=BOTTLENECK)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--circuit-lr-scale", type=float, default=1.0,
                    help="Multiply LR for q_weights/freq_scale (quantum only)")
    ap.add_argument("--curl-scale", type=float, default=1.0,
                    help="Multiply W_CURL and W_GRID_CURL")
    ap.add_argument(
        "--gen-version",
        choices=("v2", "v1"),
        default="v2",
        help="v2=Z-expectations; v1=basis probs (2^n features)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    dns = torch.load(V3 / "dns" / "reference.pt", map_location="cpu", weights_only=False)
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    t_max = float(dns["t_max"])
    nu = float(dns["nu"])

    teacher = None if args.no_teacher else load_teacher(dns_g, device)
    if teacher is not None:
        te = eval_fields(lambda a, b, c: teacher(a, b, c).detach(), dns_g, device,
                         times=GATE_TIMES, step=1)
        print(f"teacher ω={100*te['omega_max']:.3f}%", flush=True)
    else:
        print("teacher disabled", flush=True)

    target = build_target(t_max, device)
    harm_ref = HarmMLP(hidden=HIDDEN, t_max=t_max, k_max=K_MAX).to(device)
    verify_harm_basis(target, harm_ref, device)

    comparison: dict = {
        "steps": args.steps,
        "seed": args.seed,
        "nu": nu,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "bottleneck": args.bottleneck,
        "lr": args.lr,
        "circuit_lr_scale": args.circuit_lr_scale,
        "curl_scale": args.curl_scale,
        "gen_version": args.gen_version,
        "teacher": teacher is not None,
        "omega_gate": OMEGA_GATE,
        "arm": args.arm,
    }
    ablation = None

    run_q = args.arm in ("all", "quantum")
    run_cg = args.arm in ("all", "classical_gen")
    run_d = args.arm in ("all", "classical_direct")

    if run_q:
        print(
            f"\n=== quantum arm  steps={args.steps} q={args.n_qubits} L={args.n_layers} "
            f"ver={args.gen_version} ===",
            flush=True,
        )
        if args.gen_version == "v1":
            q_gen = ConditionedQuantumGenerator(
                in_dim=target.in_dim, h1=HIDDEN[0], h2=HIDDEN[1], out_dim=3,
                n_qubits=args.n_qubits, n_layers=args.n_layers, bottleneck_width=args.bottleneck,
                nu_range=NU_RANGE,
            ).to(device)
            q_kind = "quantum_v1_probs"
        else:
            q_gen = ConditionedQuantumGeneratorV2(
                in_dim=target.in_dim, h1=HIDDEN[0], h2=HIDDEN[1], out_dim=3,
                n_qubits=args.n_qubits, n_layers=args.n_layers, bottleneck_width=args.bottleneck,
                nu_range=NU_RANGE, nu_encode="log",
            ).to(device)
            q_kind = "quantum_v2"
        q_result, q_gen_sd, q_deployed = train_generator(
            q_gen, target, dns_g, device, teacher, args.steps, "quantum",
            lr=args.lr, circuit_lr_scale=args.circuit_lr_scale, curl_scale=args.curl_scale,
        )
        print("\n=== circuit ablation (quantum) ===", flush=True)
        ablation = verify_circuit_usage(q_gen, target, dns_g, device, nu)
        print(
            f"ω full={100*ablation['omega_full']:.3f}%  "
            f"random_q={100*ablation['omega_random_q']:.3f}%  "
            f"zero={100*ablation['omega_zero_feats']:.3f}%  "
            f"used={ablation['circuit_used']}",
            flush=True,
        )
        save_gen_arm(args.out / "quantum", q_gen_sd, q_deployed, q_result, "quantum", q_kind)
        comparison["quantum"] = {
            "omega_max": q_result["final"]["omega_max"],
            "vel_max": q_result["final"]["vel_max"],
            "latency_ms": q_result["latency_ms"],
            "throughput_pts_s": q_result["throughput_pts_s"],
            "n_deployed_params": q_result["n_deployed_params"],
        }
        comparison["circuit_ablation"] = ablation

    if run_cg:
        print(
            f"\n=== classical generator arm  steps={args.steps} ver={args.gen_version} ===",
            flush=True,
        )
        if args.gen_version == "v1":
            c_gen = ConditionedClassicalGenerator(
                in_dim=target.in_dim, h1=HIDDEN[0], h2=HIDDEN[1], out_dim=3,
                n_qubits=args.n_qubits, bottleneck_width=args.bottleneck,
                nu_range=NU_RANGE,
            ).to(device)
            c_kind = "classical_v1"
        else:
            c_gen = ConditionedClassicalGeneratorV2(
                in_dim=target.in_dim, h1=HIDDEN[0], h2=HIDDEN[1], out_dim=3,
                n_qubits=args.n_qubits, bottleneck_width=args.bottleneck,
                nu_range=NU_RANGE, nu_encode="log",
            ).to(device)
            c_kind = "classical_v2"
        cg_result, cg_gen_sd, cg_deployed = train_generator(
            c_gen, target, dns_g, device, teacher, args.steps, "classical_gen",
            lr=args.lr, circuit_lr_scale=1.0, curl_scale=args.curl_scale,
        )
        save_gen_arm(args.out / "classical_gen", cg_gen_sd, cg_deployed, cg_result,
                     "classical_gen", c_kind)
        comparison["classical_gen"] = {
            "omega_max": cg_result["final"]["omega_max"],
            "vel_max": cg_result["final"]["vel_max"],
            "latency_ms": cg_result["latency_ms"],
            "throughput_pts_s": cg_result["throughput_pts_s"],
            "n_deployed_params": cg_result["n_deployed_params"],
        }

    if run_d:
        print(f"\n=== classical direct arm  steps={args.steps} ===", flush=True)
        direct = HarmMLP(hidden=HIDDEN, t_max=t_max, k_max=K_MAX).to(device)
        d_result, d_sd, _ = train_direct(
            direct, dns_g, device, teacher, args.steps, "classical_direct",
        )
        save_direct_arm(args.out / "classical_direct", d_sd, d_result)
        comparison["classical_direct"] = {
            "omega_max": d_result["final"]["omega_max"],
            "vel_max": d_result["final"]["vel_max"],
            "latency_ms": d_result["latency_ms"],
            "throughput_pts_s": d_result["throughput_pts_s"],
            "n_deployed_params": d_result["n_deployed_params"],
        }

    tag = args.arm if args.arm != "all" else "all"
    (args.out / f"comparison_{tag}.json").write_text(json.dumps(comparison, indent=2) + "\n")
    if args.arm == "all":
        (args.out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    print("\n=== SUMMARY ===", flush=True)
    for arm in ("quantum", "classical_gen", "classical_direct"):
        if arm not in comparison:
            continue
        row = comparison[arm]
        print(
            f"  {arm}: ω={100*row['omega_max']:.3f}%  vel={100*row['vel_max']:.3f}%  "
            f"thr={row['throughput_pts_s']/1e6:.1f} Mpts/s",
            flush=True,
        )

    if ablation is not None and not ablation["circuit_used"]:
        print("FAIL: circuit unused — ablated ω did not degrade", flush=True)
        sys.exit(3)

    sys.exit(0)


if __name__ == "__main__":
    main()
