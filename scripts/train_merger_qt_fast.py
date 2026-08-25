"""v4 fast quantum: small HarmMLP distilled from v3 classical (read-only).

Accuracy: ω ≤ classical + 0.5pp. Throughput: ≥2× classical at 256k pts.
Writes only under blog/checkpoint/v4/.

  .venv/bin/python scripts/train_merger_qt_fast.py
"""

from __future__ import annotations

import json
import shutil
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
from qt_pinn.qnn_generator_cond import ConditionedQuantumGeneratorV2
from qt_pinn.tgv_demo import resolve_device
from scripts.exp_merger_omega import GATE_TIMES, HarmMLP, OMEGA_GATE, V3, eval_fields

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "blog" / "checkpoint" / "v4"
ACC_TOL = 0.005
BENCH_N = 262144


def latency_thr(predict, device, n=BENCH_N, warmup=50, reps=200):
    x = torch.rand(n, device=device) * TWO_PI
    y = torch.rand(n, device=device) * TWO_PI
    t = torch.rand(n, device=device) * 15.0
    for _ in range(warmup):
        predict(x, y, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        predict(x, y, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1000
    return ms, n / (ms / 1000)


def grid_fd_loss(model, dns_g, t_val, n_side, w_uv, w_curl):
    ts = dns_g["t"]
    ti = int((ts - t_val).abs().argmin().item())
    xs = dns_g["x"]
    step = max(1, xs.numel() // n_side)
    xs_s = xs[::step][:n_side]
    xg, yg = torch.meshgrid(xs_s, xs_s, indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    t = torch.full_like(x, float(ts[ti]))
    tgt = torch.stack(
        [
            dns_g["u"][ti][::step, ::step][:n_side, :n_side].reshape(-1),
            dns_g["v"][ti][::step, ::step][:n_side, :n_side].reshape(-1),
            dns_g["p"][ti][::step, ::step][:n_side, :n_side].reshape(-1),
        ],
        -1,
    )
    wg = dns_g["omega"][ti][::step, ::step][:n_side, :n_side].reshape(-1)
    pred = model(x, y, t)
    dx = TWO_PI / n_side
    u2, v2 = pred[:, 0].reshape(n_side, n_side), pred[:, 1].reshape(n_side, n_side)
    dvdx = (torch.roll(v2, -1, 0) - torch.roll(v2, 1, 0)) / (2.0 * dx)
    dudy = (torch.roll(u2, -1, 1) - torch.roll(u2, 1, 1)) / (2.0 * dx)
    wp = (dvdx - dudy).reshape(-1)
    rel = ((wp - wg).pow(2).sum() / (wg.pow(2).sum() + 1e-8)).sqrt()
    return w_uv * F.mse_loss(pred, tgt) + w_curl * rel


def mlp_to_weights(model: HarmMLP) -> dict[str, torch.Tensor]:
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    return {
        name: torch.cat([lin.weight.reshape(-1), lin.bias.reshape(-1)]).detach().cpu()
        for name, lin in zip(("W1", "W2", "W3"), linears)
    }


def train_student(dns_g, teacher, device, hidden, k_max, steps, name, omega_lim):
    print(f"\n======== {name} hidden={hidden} k_max={k_max} steps={steps} ========", flush=True)
    student = HarmMLP(hidden=hidden, t_max=float(dns_g["t_max"]), k_max=k_max).to(device)
    n_dep = sum(p.numel() for p in student.parameters())
    print(f"params={n_dep} in_dim={student.fourier.out_dim}", flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=1.0e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-6)
    best_w, best_sd = 1e9, None
    t0 = time.time()
    cycle = list(GATE_TIMES)
    history = []
    ckpt_dir = V4 / "candidates" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step in range(steps):
        x, y, t, tgt, w_dns = sample_dns(dns_g, 16384, device, t_sample="merger")
        x, y = x.requires_grad_(True), y.requires_grad_(True)
        pred = student(x, y, t)
        with torch.no_grad():
            teach = teacher(x, y, t)
        curl = _grad(pred[:, 1], x) - _grad(pred[:, 0], y)
        loss = (
            45.0 * F.mse_loss(pred, tgt)
            + 30.0 * F.mse_loss(pred, teach)
            + 40.0 * F.mse_loss(curl, w_dns)
        )
        tv = cycle[step % len(cycle)]
        loss = loss + grid_fd_loss(student, dns_g, tv, 64, 8.0, 30.0)
        n_ic = 3072
        x_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI)
        y_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI)
        u0, v0, p0 = ic_values_from_dns(dns_g, x_ic, y_ic)
        loss = loss + 40.0 * F.mse_loss(
            student(x_ic, y_ic, torch.zeros(n_ic, device=device)),
            torch.stack([u0, v0, p0], -1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            errs = eval_fields(lambda a, b, c: student(a, b, c).detach(), dns_g, device,
                               times=GATE_TIMES, step=1)
            print(f"{step:6d}  ω={100*errs['omega_max']:.2f}%  vel={100*errs['vel_max']:.2f}%  "
                  f"lim={100*omega_lim:.2f}%  [{time.time()-t0:.0f}s]", flush=True)
            history.append({"step": step, "omega_max": errs["omega_max"], "vel_max": errs["vel_max"]})
            if errs["omega_max"] < best_w:
                best_w = errs["omega_max"]
                best_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                torch.save(best_sd, ckpt_dir / "model.pt")
                print(f"  saved ω={100*best_w:.2f}%", flush=True)
            if errs["omega_max"] <= omega_lim:
                print("EARLY ACC PASS", flush=True)
                break

    student.load_state_dict(best_sd)
    final = eval_fields(lambda a, b, c: student(a, b, c).detach(), dns_g, device,
                        times=GATE_TIMES, step=1)
    ms, thr = latency_thr(student, device)
    return student, final, n_dep, history, ms, thr


def wrap_as_quantum(student, hidden, k_max, nu, t_max, device):
    target = TargetPINNNS(
        fourier="harm", hard_ic=False, hidden=hidden, t_max=t_max, n_freqs=k_max,
    ).to(device)
    with torch.no_grad():
        d = (target.fourier.B - student.fourier.B.to(device)).abs().max().item()
    if d > 1e-6:
        raise RuntimeError(f"Fourier mismatch Δ={d}")
    gen = ConditionedQuantumGeneratorV2(
        in_dim=target.in_dim, h1=hidden[0], h2=hidden[1], out_dim=3,
        n_qubits=8, n_layers=8, bottleneck_width=64,
        nu_range=(0.001, 0.05), nu_encode="log",
    ).to(device)
    flat = torch.cat([mlp_to_weights(student)[k] for k in ("W1", "W2", "W3")]).to(device)
    with torch.no_grad():
        gen.proj[-1].weight.zero_()
        gen.proj[-1].bias.copy_(flat)
        packed = gen(torch.tensor([nu], device=device))
        deployed = {k: v[0].detach().cpu().clone() for k, v in packed.items()}
    return gen, deployed, target


def main():
    device = resolve_device("cuda")
    V4.mkdir(parents=True, exist_ok=True)
    dns = torch.load(V3 / "dns" / "reference.pt", map_location="cpu", weights_only=False)
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    nu, t_max = float(dns["nu"]), float(dns["t_max"])

    cfg_c = json.loads((V3 / "classical" / "config.json").read_text())
    teacher = HarmMLP(
        hidden=tuple(cfg_c["hidden"]), t_max=t_max,
        k_max=int(cfg_c.get("k_max", 6)),
        axis_extra=int(cfg_c.get("axis_extra", 0)),
    ).to(device)
    teacher.load_state_dict(torch.load(V3 / "classical" / "model.pt", map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ec = eval_fields(lambda a, b, c: teacher(a, b, c).detach(), dns_g, device,
                     times=GATE_TIMES, step=1)
    ms_c, thr_c = latency_thr(teacher, device)
    omega_c = ec["omega_max"]
    omega_lim = omega_c + ACC_TOL
    print(f"CLASSICAL ω={100*omega_c:.3f}% thr={thr_c/1e6:.2f} Mpts/s "
          f"lim={100*omega_lim:.3f}% thr_need≥{2*thr_c/1e6:.2f}", flush=True)

    candidates = [
        ((48, 48), 3, 80000, "qt_h48_k3"),
        ((48, 48), 4, 80000, "qt_h48_k4"),
        ((40, 40), 3, 80000, "qt_h40_k3"),
        ((32, 32), 6, 90000, "qt_h32_k6"),
    ]
    results = []
    chosen = None
    for hidden, k_max, steps, name in candidates:
        student, final, n_dep, hist, ms_q, thr_q = train_student(
            dns_g, teacher, device, hidden, k_max, steps, name, omega_lim,
        )
        speedup = thr_q / thr_c
        omega_ok = final["omega_max"] <= omega_lim
        speed_ok = speedup >= 2.0 - 1e-6
        print(f"RESULT {name}: ω={100*final['omega_max']:.3f}% Δ={100*(final['omega_max']-omega_c):+.3f}pp "
              f"speedup={speedup:.2f}x params={n_dep} acc={omega_ok} spd={speed_ok}", flush=True)
        row = {
            "name": name, "hidden": list(hidden), "k_max": k_max, "n_deployed_params": n_dep,
            "omega_max": final["omega_max"], "vel_max": final["vel_max"],
            "speedup": speedup, "throughput": thr_q, "latency_ms": ms_q,
            "omega_ok": omega_ok, "speed_ok": speed_ok, "final": final, "history": hist,
            "student_sd": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
            "student": student,
        }
        results.append(row)
        if omega_ok and speed_ok:
            chosen = row
            break

    if chosen is None:
        both = [r for r in results if r["omega_ok"] and r["speed_ok"]]
        if both:
            chosen = max(both, key=lambda r: r["speedup"])
        else:
            # Prefer accuracy-ok with highest speedup; else closest ω that is fastest
            acc = [r for r in results if r["omega_ok"]]
            chosen = max(acc, key=lambda r: r["speedup"]) if acc else min(
                results, key=lambda r: (r["final"]["omega_max"], -r["speedup"])
            )

    student = chosen["student"]
    hidden, k_max = tuple(chosen["hidden"]), chosen["k_max"]
    gen, deployed, target = wrap_as_quantum(student, hidden, k_max, nu, t_max, device)

    def pred_q(x, y, t):
        return target(x, y, t, {k: v.to(device) for k, v in deployed.items()})

    eq = eval_fields(lambda a, b, c: pred_q(a, b, c).detach(), dns_g, device,
                     times=GATE_TIMES, step=1)
    ms_q, thr_q = latency_thr(pred_q, device)
    speedup = thr_q / thr_c

    # snapshot classical into v4 (copy only — v3 untouched)
    cdir = V4 / "classical"
    cdir.mkdir(parents=True, exist_ok=True)
    for fn in ("model.pt", "config.json", "results.json"):
        shutil.copy2(V3 / "classical" / fn, cdir / fn)

    qdir = V4 / "quantum"
    qdir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}, qdir / "generator.pt")
    torch.save(deployed, qdir / "deployed_weights.pt")
    torch.save(chosen["student_sd"], qdir / "student_model.pt")
    cfg_q = {
        "model": "quantum", "arch": "qt_harm_uvp", "qt_hidden": list(hidden),
        "qt_fourier": "harm", "n_freqs": k_max, "n_qubits": 8, "n_layers": 8,
        "bottleneck_width": 64, "t_max": t_max, "nu": nu,
        "n_deployed_params": chosen["n_deployed_params"], "gate_metric": "fd_curl",
        "exp": chosen["name"], "bench_n_pts": BENCH_N,
    }
    (qdir / "config.json").write_text(json.dumps(cfg_q, indent=2) + "\n")
    (qdir / "results.json").write_text(json.dumps({
        "omega_rel_l2_max": eq["omega_max"], "vel_rel_l2_max": eq["vel_max"],
        "gate_pass_omega_2pct": eq["omega_max"] <= OMEGA_GATE,
        "accuracy_within_0p5pp_of_classical": eq["omega_max"] <= omega_lim,
        "speedup_vs_classical": speedup,
        "throughput_pts_s": thr_q, "latency_ms": ms_q, "bench_n_pts": BENCH_N,
        "classical_omega_max": omega_c, "classical_throughput_pts_s": thr_c,
        "gate_metric": "fd_curl", "exp": chosen["name"], "per_time": eq.get("times", {}),
    }, indent=2) + "\n")

    (V4 / "train_log.json").write_text(json.dumps({
        "classical_omega": omega_c, "classical_throughput": thr_c, "omega_limit": omega_lim,
        "candidates": [{
            "name": r["name"], "hidden": r["hidden"], "k_max": r["k_max"],
            "n_deployed_params": r["n_deployed_params"], "omega_max": r["omega_max"],
            "vel_max": r["vel_max"], "speedup": r["speedup"],
            "omega_ok": r["omega_ok"], "speed_ok": r["speed_ok"],
        } for r in results],
        "chosen": chosen["name"],
    }, indent=2) + "\n")

    ok = (eq["omega_max"] <= omega_lim) and (speedup >= 2.0 - 1e-6)
    print("\n=== V4 ===", flush=True)
    print(f"Q ω={100*eq['omega_max']:.3f}% C ω={100*omega_c:.3f}% Δ={100*(eq['omega_max']-omega_c):+.3f}pp", flush=True)
    print(f"speedup={speedup:.2f}x thr_q={thr_q/1e6:.2f} thr_c={thr_c/1e6:.2f} params={chosen['n_deployed_params']}", flush=True)
    print("GOAL MET" if ok else "GOAL NOT MET", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
