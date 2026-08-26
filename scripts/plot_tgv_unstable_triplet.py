#!/usr/bin/env python3
"""Perturbed TGV: boost bottom-left vortex → unstable evolution; DNS|Classical|Quantum GIF.

  .venv/bin/python -u scripts/plot_tgv_unstable_triplet.py --device cuda
  .venv/bin/python -u scripts/plot_tgv_unstable_triplet.py --device cpu --train-steps 4000
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from pdes.ns2d.physics_loss import _grad
from qt_pinn.ns2d_spectral import TWO_PI, SpectralNS2D, sample_dns
from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.tgv_demo import DirectNSMLP, pde_loss, resolve_device

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "blog" / "checkpoint" / "v2"
V5 = ROOT / "blog" / "checkpoint" / "v5"
OUT = V5 / "unstable"

CMAP = LinearSegmentedColormap.from_list(
    "deepblue_yellow",
    ["#061428", "#0B3D91", "#1A8FBF", "#7BC96F", "#F4D35E", "#FFF3B0"],
    N=256,
)

# Bottom-left lobe center in math coords (origin bottom-left)
BL_CENTER = (0.5 * math.pi, 0.5 * math.pi)


def tgv_omega(xs: torch.Tensor, ys: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    return 2.0 * k * torch.sin(k * xg) * torch.sin(k * yg)


def boost_bottom_left(
    omega: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    factor: float = 5.5,
    delta: float = 1.05,
) -> torch.Tensor:
    """Amplify the bottom-left TGV lobe with a smooth Gaussian mask (breaks balance)."""
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    cx, cy = BL_CENTER
    dx = (xg - cx + math.pi) % TWO_PI - math.pi
    dy = (yg - cy + math.pi) % TWO_PI - math.pi
    mask = torch.exp(-(dx * dx + dy * dy) / (delta * delta))
    # Only boost where ω already has the lobe's sign (positive near BL for k=1)
    sign = torch.sign(omega + 1e-12)
    boost = 1.0 + (factor - 1.0) * mask * (sign > 0).to(omega.dtype)
    out = omega * boost
    return out - out.mean()



def _step_rk4_if(sol: SpectralNS2D, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrating-factor RK4 (viscosity exact). Explicit viscous RK4 blows up at ν=0.1."""
    e1 = torch.exp(-sol.nu * sol.k2 * (0.5 * dt))
    e2 = torch.exp(-sol.nu * sol.k2 * dt)

    def N(om: torch.Tensor) -> torch.Tensor:
        u, v = sol.omega_to_uv(om)
        wh = torch.fft.fft2(om)
        wx = torch.fft.ifft2(1j * sol.kx * wh).real
        wy = torch.fft.ifft2(1j * sol.ky * wh).real
        adv = u * wx + v * wy
        return torch.fft.ifft2(-torch.fft.fft2(adv) * sol.mask).real

    k1 = N(omega)
    w2 = torch.fft.ifft2(e1 * torch.fft.fft2(omega + 0.5 * dt * k1)).real
    k2 = N(w2)
    w3 = torch.fft.ifft2(e1 * torch.fft.fft2(omega) + (0.5 * dt) * torch.fft.fft2(k2)).real
    k3 = N(w3)
    w4 = torch.fft.ifft2(e2 * torch.fft.fft2(omega) + dt * e1 * torch.fft.fft2(k3)).real
    k4 = N(w4)
    out_h = e2 * torch.fft.fft2(omega) + (dt / 6.0) * (
        e2 * torch.fft.fft2(k1)
        + 2.0 * e1 * torch.fft.fft2(k2)
        + 2.0 * e1 * torch.fft.fft2(k3)
        + torch.fft.fft2(k4)
    )
    return torch.fft.ifft2(out_h).real


@torch.no_grad()
def run_perturbed_dns(
    *,
    n: int,
    nu: float,
    t_max: float,
    n_save: int,
    factor: float,
    device: torch.device,
    cfl: float = 0.45,
    delta: float = 1.05,
) -> dict:
    sol = SpectralNS2D(n=n, nu=nu, device=device)
    omega = tgv_omega(sol.x, sol.y, k=1.0)
    omega = boost_bottom_left(omega, sol.x, sol.y, factor=factor, delta=delta)
    t_save = torch.linspace(0.0, t_max, n_save)
    omegas, us, vs, ps, times = [], [], [], [], []

    def snap(om, t):
        u, v = sol.omega_to_uv(om)
        p = sol.pressure(u, v)
        omegas.append(om.detach().cpu().clone())
        us.append(u.detach().cpu().clone())
        vs.append(v.detach().cpu().clone())
        ps.append(p.detach().cpu().clone())
        times.append(float(t))

    t = 0.0
    save_idx = 0
    snap(omega, 0.0)
    save_idx = 1
    step = 0
    print(f"DNS perturbed TGV n={n} ν={nu} T={t_max} boost={factor}", flush=True)
    while t < t_max - 1e-12 and save_idx < n_save:
        target = float(t_save[save_idx].item()) - t
        # never floor dt — that violates CFL and seeds NaNs when the flow stiffens
        dt = min(sol.cfl_dt(omega, cfl=cfl), target)
        if not (dt > 0.0) or dt != dt:
            raise RuntimeError(
                f"DNS timestep collapsed at t={t} cfl_dt={sol.cfl_dt(omega, cfl=cfl)}"
            )
        omega = _step_rk4_if(sol, omega, dt)
        t += dt
        step += 1
        if torch.isnan(omega).any():
            raise RuntimeError(f"DNS NaN at t={t} step={step}")
        if t >= float(t_save[save_idx].item()) - 1e-9:
            snap(omega, t)
            save_idx += 1
            if save_idx % 10 == 0 or save_idx >= n_save:
                print(f"  DNS t={t:5.2f}/{t_max}  |ω|_mean={float(omega.abs().mean()):.4f}", flush=True)

    return {
        "x": sol.x.cpu(),
        "y": sol.y.cpu(),
        "t": torch.tensor(times, dtype=torch.float32),
        "u": torch.stack(us),
        "v": torch.stack(vs),
        "p": torch.stack(ps),
        "omega": torch.stack(omegas),
        "nu": nu,
        "n": n,
        "t_max": t_max,
        "boost_factor": factor,
        "bl_center": list(BL_CENTER),
        "kind": "tgv_bottom_left_boost",
    }


def _init_classical(device, t_max: float, warm_start: bool = False) -> DirectNSMLP:
    m = DirectNSMLP(hidden=(128, 128), hard_ic=False, t_max=t_max).to(device)
    if not warm_start:
        return m
    pt = V2 / "classical" / "model.pt"
    if pt.exists():
        try:
            cfg = json.loads((V2 / "classical" / "config.json").read_text())
            warm = DirectNSMLP(
                hidden=tuple(cfg.get("classical_hidden") or cfg["hidden"]),
                hard_ic=False,
                t_max=float(cfg["t_max"]),
            ).to(device)
            warm.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
            m = warm
            print(f"warm-start classical from {pt}", flush=True)
        except Exception as e:
            print(f"classical warm-start skipped: {e}", flush=True)
    return m


def _quantum_weights_from_v2(device, warm_start: bool = False) -> dict[str, torch.Tensor]:
    wpath = V2 / "quantum" / "deployed_weights.pt"
    if not wpath.exists():
        raise SystemExit(f"need {wpath} for weight shapes")
    w = torch.load(wpath, map_location=device, weights_only=True)
    if warm_start:
        print(f"warm-start quantum weights from {wpath}", flush=True)
        return {k: v.to(device).detach().clone().requires_grad_(True) for k, v in w.items()}
    out = {}
    for k, v in w.items():
        t = torch.empty_like(v, device=device)
        if v.ndim >= 2:
            torch.nn.init.xavier_uniform_(t)
        else:
            torch.nn.init.normal_(t, std=0.05)
        out[k] = t.requires_grad_(True)
    print("quantum weights: random init (no TGV warm-start)", flush=True)
    return out


def train_classical(
    dns: dict, device: torch.device, steps: int, lr: float, *, warm_start: bool = False
) -> DirectNSMLP:
    t_max = float(dns["t_max"])
    nu = float(dns["nu"])
    model = _init_classical(device, t_max, warm_start=warm_start)
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    t0 = time.time()
    for step in range(steps):
        x, y, t, tgt, w_dns = sample_dns(dns_g, 8192, device, t_sample="uniform")
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        pred = model(x, y, t)
        curl = _grad(pred[:, 1], x) - _grad(pred[:, 0], y)
        # Data-heavy: PDE at ν=0.03 + strong boost is stiff; fit the field first
        loss = (
            80.0 * F.mse_loss(pred, tgt)
            + 100.0 * F.mse_loss(curl, w_dns)
            + 1.0 * pde_loss(pred, x, y, t, nu)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0 or step == 0:
            print(
                f"  [classical] {step+1:5d}/{steps}  loss={float(loss.detach()):.4e}  [{time.time()-t0:.0f}s]",
                flush=True,
            )
    return model


def train_quantum(
    dns: dict, device: torch.device, steps: int, lr: float, *, warm_start: bool = False
) -> tuple[TargetPINNNS, dict]:
    t_max = float(dns["t_max"])
    nu = float(dns["nu"])
    target = TargetPINNNS(
        fourier="tgv", hard_ic=False, hidden=(32, 32), t_max=t_max,
    ).to(device)
    weights = _quantum_weights_from_v2(device, warm_start=warm_start)
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    opt = torch.optim.Adam(list(weights.values()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    t0 = time.time()
    for step in range(steps):
        x, y, t, tgt, w_dns = sample_dns(dns_g, 6144, device, t_sample="uniform")
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        pred = target(x, y, t, weights)
        curl = _grad(pred[:, 1], x) - _grad(pred[:, 0], y)
        loss = (
            80.0 * F.mse_loss(pred, tgt)
            + 100.0 * F.mse_loss(curl, w_dns)
            + 1.0 * pde_loss(pred, x, y, t, nu)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0 or step == 0:
            print(
                f"  [quantum]   {step+1:5d}/{steps}  loss={float(loss.detach()):.4e}  [{time.time()-t0:.0f}s]",
                flush=True,
            )
    return target, weights


def _lerp_dns_omega(dns: dict, t: float) -> np.ndarray:
    ts = dns["t"].numpy()
    W = dns["omega"].numpy()
    if t <= ts[0]:
        return W[0]
    if t >= ts[-1]:
        return W[-1]
    i = int(np.searchsorted(ts, t) - 1)
    a = (t - ts[i]) / (ts[i + 1] - ts[i])
    return (1 - a) * W[i] + a * W[i + 1]


@torch.no_grad()
def pred_abs_omega(model, weights, xs_np: np.ndarray, t: float, device) -> np.ndarray:
    n = len(xs_np)
    xs = torch.tensor(xs_np, device=device, dtype=torch.float32)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    tt = torch.full_like(x, float(t))
    if weights is None:
        pred = model(x, y, tt)
    else:
        pred = model(x, y, tt, weights)
    u = pred[:, 0].cpu().numpy().reshape(n, n)
    v = pred[:, 1].cpu().numpy().reshape(n, n)
    dx = float(TWO_PI / n)
    dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
    dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    return np.abs(dvdx - dudy)


def write_triplet_gif(
    dns: dict,
    classical,
    quantum,
    qw,
    *,
    out: Path,
    dpi: int = 110,
    fps: int = 10,
    plot_n: int = 192,
) -> int:
    xs_full = dns["x"].numpy()
    step = max(1, len(xs_full) // plot_n)
    xs = xs_full[::step][:plot_n]
    times = dns["t"].numpy()
    # fixed vmax from early DNS |ω|
    w0 = np.abs(_lerp_dns_omega(dns, 0.0))[::step, ::step][:plot_n, :plot_n]
    vmax = float(np.percentile(w0, 99.5)) * 1.05
    device = next(classical.parameters()).device

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg required")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tgv_unst_") as tmp:
        tmp_path = Path(tmp)
        for i, t in enumerate(times):
            w_d = np.abs(_lerp_dns_omega(dns, float(t)))[::step, ::step][:plot_n, :plot_n]
            w_c = pred_abs_omega(classical, None, xs, float(t), device)
            w_q = pred_abs_omega(quantum, qw, xs, float(t), device)
            fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), constrained_layout=True)
            panels = [(w_d, "DNS"), (w_c, "Classical"), (w_q, "Quantum")]
            im = None
            for ax, (w, title) in zip(axes, panels):
                im = ax.pcolormesh(
                    xs, xs, w.T, cmap=CMAP, shading="auto", vmin=0.0, vmax=vmax,
                )
                # mark boosted center
                ax.plot([BL_CENTER[0]], [BL_CENTER[1]], "r+", ms=8, mew=1.2)
                ax.set_aspect("equal")
                ax.set_xlim(0, TWO_PI)
                ax.set_ylim(0, TWO_PI)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=12)
            fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label=r"$|\omega|$")
            fig.suptitle(
                rf"Unstable TGV (bottom-left boost)  $\nu={dns['nu']}$  $t={t:.2f}$",
                fontsize=13,
            )
            fig.savefig(tmp_path / f"frame_{i:04d}.png", dpi=dpi, facecolor="white")
            plt.close(fig)
            if i % 10 == 0 or i == len(times) - 1:
                print(f"  gif frame {i+1}/{len(times)} t={t:.2f}", flush=True)

        palette = tmp_path / "palette.png"
        pattern = str(tmp_path / "frame_%04d.png")
        subprocess.run(
            [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
             "-vf", "palettegen=stats_mode=diff", str(palette)],
            check=True, capture_output=True,
        )
        subprocess.run(
            [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern, "-i", str(palette),
             "-lavfi", "paletteuse=dither=bayer:bayer_scale=5", str(out)],
            check=True, capture_output=True,
        )
    print(f"Wrote {out} ({len(times)} frames @ {fps} fps ≈ {len(times)/fps:.1f}s)", flush=True)
    return len(times)


def plot_snapshots(dns, classical, quantum, qw, out: Path, times=None):
    t_max = float(dns["t_max"])
    if times is None:
        times = tuple(round(t_max * f, 2) for f in (0.0, 0.2, 0.45, 0.7, 1.0))
    xs_full = dns["x"].numpy()
    plot_n = 160
    step = max(1, len(xs_full) // plot_n)
    xs = xs_full[::step][:plot_n]
    device = next(classical.parameters()).device
    w0 = np.abs(_lerp_dns_omega(dns, 0.0))[::step, ::step][:plot_n, :plot_n]
    vmax = float(np.percentile(w0, 99.5)) * 1.05
    nrow, ncol = 3, len(times)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 7.2), constrained_layout=True)
    row_labels = ["DNS", "Classical", "Quantum"]
    for j, t in enumerate(times):
        panels = [
            np.abs(_lerp_dns_omega(dns, float(t)))[::step, ::step][:plot_n, :plot_n],
            pred_abs_omega(classical, None, xs, float(t), device),
            pred_abs_omega(quantum, qw, xs, float(t), device),
        ]
        for i, w in enumerate(panels):
            ax = axes[i, j]
            ax.pcolormesh(xs, xs, w.T, cmap=CMAP, shading="auto", vmin=0.0, vmax=vmax)
            ax.plot([BL_CENTER[0]], [BL_CENTER[1]], "r+", ms=6, mew=1.0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(rf"$t={t:g}$", fontsize=11)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=11)
    fig.suptitle("Unstable TGV — bottom-left vortex boosted", fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=192, help="DNS grid")
    ap.add_argument("--nu", type=float, default=0.03,
                    help="viscosity (lower → imbalance lives longer)")
    ap.add_argument("--t-max", type=float, default=12.0)
    ap.add_argument("--dt-save", type=float, default=0.1)
    ap.add_argument("--boost", type=float, default=5.5,
                    help="bottom-left lobe amplification (≫1 breaks TGV balance)")
    ap.add_argument("--boost-delta", type=float, default=1.05,
                    help="Gaussian mask width for the boosted lobe")
    ap.add_argument("--train-steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--warm-start", action="store_true",
                    help="warm-start from v2 TGV (usually hurts the boosted case)")
    ap.add_argument("--reuse-dns", action="store_true",
                    help="reuse existing dns/reference.pt")
    ap.add_argument("--skip-train", action="store_true", help="reuse saved models if present")
    args = ap.parse_args()

    device = resolve_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    n_save = int(round(args.t_max / args.dt_save)) + 1

    dns_path = OUT / "dns" / "reference.pt"
    if dns_path.exists() and (args.skip_train or args.reuse_dns):
        dns = torch.load(dns_path, map_location="cpu", weights_only=False)
        print(f"loaded DNS {dns_path}", flush=True)
    else:
        dns = run_perturbed_dns(
            n=args.n, nu=args.nu, t_max=args.t_max, n_save=n_save,
            factor=args.boost, device=device, delta=args.boost_delta,
        )
        dns_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dns, dns_path)
        (OUT / "dns" / "config.json").write_text(
            json.dumps(
                {
                    "nu": args.nu,
                    "t_max": args.t_max,
                    "n": args.n,
                    "boost": args.boost,
                    "boost_delta": args.boost_delta,
                    "bl_center": list(BL_CENTER),
                    "n_save": n_save,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"saved {dns_path}", flush=True)

    c_path = OUT / "classical" / "model.pt"
    q_path = OUT / "quantum" / "deployed_weights.pt"
    if args.skip_train and c_path.exists() and q_path.exists():
        classical = _init_classical(device, float(dns["t_max"]), warm_start=False)
        classical.load_state_dict(torch.load(c_path, map_location=device, weights_only=True))
        quantum = TargetPINNNS(
            fourier="tgv", hard_ic=False, hidden=(32, 32), t_max=float(dns["t_max"]),
        ).to(device)
        qw = {
            k: v.to(device)
            for k, v in torch.load(q_path, map_location=device, weights_only=True).items()
        }
        print("loaded trained classical/quantum", flush=True)
    else:
        print("Training classical on perturbed DNS…", flush=True)
        classical = train_classical(
            dns, device, args.train_steps, args.lr, warm_start=args.warm_start
        )
        print("Training quantum (deployed weights) on perturbed DNS…", flush=True)
        quantum, qw = train_quantum(
            dns, device, args.train_steps, args.lr, warm_start=args.warm_start
        )
        (OUT / "classical").mkdir(parents=True, exist_ok=True)
        (OUT / "quantum").mkdir(parents=True, exist_ok=True)
        torch.save(classical.state_dict(), c_path)
        torch.save({k: v.detach().cpu() for k, v in qw.items()}, q_path)
        (OUT / "classical" / "config.json").write_text(
            json.dumps({"arch": "DirectNSMLP", "hidden": [128, 128], "hard_ic": False,
                        "t_max": dns["t_max"], "nu": dns["nu"], "warm_start": args.warm_start},
                       indent=2) + "\n"
        )
        (OUT / "quantum" / "config.json").write_text(
            json.dumps({"arch": "TargetPINNNS", "fourier": "tgv", "hidden": [32, 32],
                        "hard_ic": False, "t_max": dns["t_max"], "nu": dns["nu"]}, indent=2) + "\n"
        )

    classical.eval()
    media = OUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    write_triplet_gif(
        dns, classical, quantum, qw,
        out=media / "tgv_unstable_triplet.gif", fps=args.fps,
    )
    plot_snapshots(
        dns, classical, quantum, qw, media / "tgv_unstable_snapshots.png",
    )
    (media / "CAPTION.md").write_text(
        "# Unstable Taylor–Green (bottom-left boost)\n\n"
        "Start from standard 2D TGV, then **amplify the bottom-left vortex** "
        f"(center $(\\pi/2,\\pi/2)$, boost×{args.boost}, mask δ={args.boost_delta}). "
        "That breaks the exact-solution balance so nonlinear advection turns on. "
        f"Lower viscosity (ν={args.nu}) keeps the imbalance visible through T={args.t_max}.\n\n"
        f"- DNS: spectral NS, ν={args.nu}, T={args.t_max}, Δt_save={args.dt_save}\n"
        "- Classical / Quantum: fine-tuned on this DNS (soft IC)\n"
        "- `tgv_unstable_triplet.gif` — DNS | Classical | Quantum "
        f"({int(round(args.t_max / args.dt_save)) + 1} frames @ {args.fps} fps "
        f"≈ {(int(round(args.t_max / args.dt_save)) + 1) / args.fps:.1f}s)\n"
        "- Red **+** marks the boosted center\n"
    )
    (OUT / "notes.md").write_text(
        "# v5 unstable TGV\n\n"
        f"Bottom-left lobe boost×{args.boost} (δ={args.boost_delta}), ν={args.nu}, "
        f"T={args.t_max} — destroys TGV exact-balance so vortices interact longer.\n"
        "See `media/tgv_unstable_triplet.gif`.\n"
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
