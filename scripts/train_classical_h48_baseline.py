"""Classical h48_k3 distillation baseline — identical training to v4 student.

This is the missing control experiment: same architecture (HarmMLP 48-48, k<=3),
same loss, same steps, same teacher — but purely classical (no QNN wrapper).

If this matches QT h48_k3 accuracy, the QNN path adds nothing.
If this is worse, the QNN training path provides a genuine advantage.

  .venv/bin/python scripts/train_classical_h48_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdes.ns2d.physics_loss import _grad
from qt_pinn.ns2d_spectral import TWO_PI, ic_values_from_dns, sample_dns
from qt_pinn.tgv_demo import resolve_device
from scripts.exp_merger_omega import GATE_TIMES, HarmMLP, OMEGA_GATE, V3 as _V3_IMPORT, eval_fields
from scripts.train_merger_qt_fast import grid_fd_loss, latency_thr

V3 = _V3_IMPORT

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "blog" / "checkpoint" / "v4"
OUT = V4 / "classical_h48_baseline"
STEPS = 80000
HIDDEN = (48, 48)
K_MAX = 3
BENCH_N = 262144


def main() -> None:
    device = resolve_device("cuda")
    dns = torch.load(V3 / "dns" / "reference.pt", map_location="cpu", weights_only=False)
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    t_max = float(dns["t_max"])

    # Load teacher (v3 classical 96-96)
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

    ec = eval_fields(lambda a, b, c: teacher(a, b, c).detach(), dns_g, device, times=GATE_TIMES, step=1)
    ms_c, thr_c = latency_thr(lambda a, b, c: teacher(a, b, c), device)
    omega_c = ec["omega_max"]
    omega_lim = omega_c + 0.005  # same gate as v4
    print(f"TEACHER  ω={100*omega_c:.3f}%  thr={thr_c/1e6:.1f} Mpts/s", flush=True)

    # Student: identical architecture to v4 QT student
    student = HarmMLP(hidden=HIDDEN, t_max=t_max, k_max=K_MAX).to(device)
    n_dep = sum(p.numel() for p in student.parameters())
    print(f"STUDENT  arch=HarmMLP{list(HIDDEN)} k_max={K_MAX} params={n_dep}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=3e-6)

    best_w, best_sd = 1e9, None
    cycle = list(GATE_TIMES)
    t0 = time.time()
    history = []

    for step in range(STEPS):
        # Identical loss to train_student in train_merger_qt_fast.py
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

        if step % 2000 == 0 or step == STEPS - 1:
            errs = eval_fields(lambda a, b, c: student(a, b, c).detach(), dns_g, device,
                               times=GATE_TIMES, step=1)
            print(f"{step:6d}  ω={100*errs['omega_max']:.2f}%  vel={100*errs['vel_max']:.2f}%  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            history.append({"step": step, "omega": errs["omega_max"], "vel": errs["vel_max"]})
            if errs["omega_max"] < best_w:
                best_w = errs["omega_max"]
                best_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                torch.save(best_sd, OUT / "model.pt")
            if errs["omega_max"] <= omega_lim:
                print("EARLY PASS", flush=True)
                break

    student.load_state_dict(best_sd)
    final = eval_fields(lambda a, b, c: student(a, b, c).detach(), dns_g, device, times=GATE_TIMES, step=1)
    ms_s, thr_s = latency_thr(lambda a, b, c: student(a, b, c), device)
    speedup = thr_s / thr_c

    # Load v4 QT result for comparison
    qt_res = json.loads((V4 / "quantum" / "results.json").read_text())

    print("\n=== COMPARISON (same arch: HarmMLP 48-48 k<=3) ===", flush=True)
    print(f"Classical-distill h48k3:  ω={100*final['omega_max']:.3f}%  thr={thr_s/1e6:.1f} Mpts/s  params={n_dep}", flush=True)
    print(f"QT-distill      h48k3:    ω={100*qt_res['omega_rel_l2_max']:.3f}%  thr={qt_res['throughput_pts_s']/1e6:.1f} Mpts/s  params={n_dep}", flush=True)
    print(f"Teacher (96-96 k<=6):     ω={100*omega_c:.3f}%  thr={thr_c/1e6:.1f} Mpts/s  params=13347", flush=True)
    delta = final["omega_max"] - qt_res["omega_rel_l2_max"]
    print(f"\nΔω (classical - QT) = {100*delta:+.3f} pp  ({'QT better' if delta>0 else 'classical better or tied'})", flush=True)

    cfg = {"arch": "harm_mlp_h48_k3", "hidden": list(HIDDEN), "k_max": K_MAX,
           "n_deployed_params": n_dep, "training": "classical_distill",
           "teacher": "v3_classical_96_96_k6"}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (OUT / "results.json").write_text(json.dumps({
        "omega_rel_l2_max": final["omega_max"],
        "vel_rel_l2_max": final["vel_max"],
        "gate_pass_omega_2pct": final["omega_max"] <= OMEGA_GATE,
        "throughput_pts_s": thr_s, "latency_ms": ms_s,
        "speedup_vs_teacher": speedup,
        "qt_omega": qt_res["omega_rel_l2_max"],
        "delta_vs_qt_pp": delta * 100,
        "history": history,
    }, indent=2) + "\n")
    print(f"\nResults saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
