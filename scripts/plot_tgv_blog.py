"""Charts + dense eval for blog TGV checkpoints (exact | classical | quantum).

Outputs under blog/media/:
  tgv_triplet_snapshots.png   vorticity at t=0, T/2, T
  tgv_error_maps.png          |ω − ω_exact| at T/2, T
  tgv_metrics.png             rel-L2 bars
  tgv_triplet.gif             short animation (optional, needs ffmpeg)
  eval_report.json            quantitative gate check

  .venv/bin/python scripts/plot_tgv_blog.py
  .venv/bin/python scripts/plot_tgv_blog.py --no-gif
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pdes.ns2d.physics_loss import exact_solution, relative_l2, relative_l2_gauge, _grad
from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.tgv_demo import DirectNSMLP, pde_loss, resolve_device

ROOT = Path(__file__).resolve().parents[1]
X_LO, X_HI = 0.0, 2.0 * math.pi


def load_classical(run_dir: Path, device: torch.device):
    cfg = json.loads((run_dir / "config.json").read_text())
    h = tuple(cfg.get("classical_hidden") or cfg["hidden"])
    m = DirectNSMLP(hidden=h, hard_ic=cfg["hard_ic"], t_max=cfg["t_max"]).to(device)
    m.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    m.eval()
    return m, cfg


def load_quantum(run_dir: Path, device: torch.device):
    cfg = json.loads((run_dir / "config.json").read_text())
    h = tuple(cfg.get("qt_hidden") or cfg["hidden"])
    target = TargetPINNNS(
        fourier="tgv", hard_ic=cfg["hard_ic"], hidden=h, t_max=cfg["t_max"],
    ).to(device)
    w = torch.load(run_dir / "deployed_weights.pt", map_location=device, weights_only=True)
    w = {k: v.to(device) for k, v in w.items()}
    return target, w, cfg


def predict_field(kind: str, model, weights, x, y, t):
    if kind == "classical":
        return model(x, y, t)
    return model(x, y, t, weights)


def exact_vorticity(x, y, t, nu: float) -> torch.Tensor:
    return 2.0 * torch.sin(x) * torch.sin(y) * torch.exp(-2.0 * nu * t)


def fd_vorticity(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """ω ≈ ∂v/∂x − ∂u/∂y on a uniform periodic grid (shape N×N)."""
    dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    return dv_dx - du_dy


def grid_xy(n: int, device: torch.device):
    xs = torch.linspace(X_LO, X_HI, n, device=device)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    return xs, xg.flatten(), yg.flatten()


def eval_dense(
    classical, quantum, qw, nu: float, t_max: float, device: torch.device, n: int = 128,
) -> dict:
    """Rel-L2 + PDE residual RMS on a dense grid at several times."""
    xs, x, y = grid_xy(n, device)
    times = [0.0, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max]
    report: dict = {"n_grid": n, "times": {}, "gate_2pct": True, "gate_1pct": True}

    for tv in times:
        t = torch.full_like(x, tv)
        u_ex, v_ex, p_ex = exact_solution(x, y, t, nu)
        out_t: dict = {}
        for name, model, w in (
            ("classical", classical, None),
            ("quantum", quantum, qw),
        ):
            pred = predict_field("classical" if w is None else "quantum", model, w, x, y, t)
            u_e = relative_l2(pred[:, 0], u_ex)
            v_e = relative_l2(pred[:, 1], v_ex)
            p_e = relative_l2_gauge(pred[:, 2], p_ex)
            vel = 0.5 * (u_e + v_e)
            # PDE residual (needs grad)
            xg = x.detach().requires_grad_(True)
            yg = y.detach().requires_grad_(True)
            tg = t.detach().requires_grad_(True)
            with torch.enable_grad():
                pred_g = predict_field(
                    "classical" if w is None else "quantum", model, w, xg, yg, tg,
                )
                pde = pde_loss(pred_g, xg, yg, tg, nu).item()
            out_t[name] = {
                "u_rel_l2": u_e, "v_rel_l2": v_e, "p_rel_l2_gauge": p_e,
                "vel_rel_l2": vel, "pde_rms_sq": pde,
            }
            if vel > 0.02:
                report["gate_2pct"] = False
            if vel > 0.01:
                report["gate_1pct"] = False
        report["times"][f"{tv:g}"] = out_t

    # Max over t>0 for summary
    for name in ("classical", "quantum"):
        vmax = max(report["times"][k][name]["vel_rel_l2"] for k in report["times"])
        report[f"{name}_vel_rel_l2_max"] = vmax
    return report


def field_at(model, weights, kind: str, n: int, t_val: float, nu: float, device):
    xs, x, y = grid_xy(n, device)
    t = torch.full_like(x, t_val)
    with torch.no_grad():
        pred = predict_field(kind, model, weights, x, y, t)
    u = pred[:, 0].cpu().numpy().reshape(n, n)
    v = pred[:, 1].cpu().numpy().reshape(n, n)
    p = pred[:, 2].cpu().numpy().reshape(n, n)
    dx = (X_HI - X_LO) / n
    omega = fd_vorticity(u, v, dx)
    u_ex, v_ex, p_ex = exact_solution(x, y, t, nu)
    omega_ex = exact_vorticity(x, y, t, nu).cpu().numpy().reshape(n, n)
    return {
        "xs": xs.cpu().numpy(),
        "u": u, "v": v, "p": p, "omega": omega,
        "u_ex": u_ex.cpu().numpy().reshape(n, n),
        "v_ex": v_ex.cpu().numpy().reshape(n, n),
        "p_ex": p_ex.cpu().numpy().reshape(n, n),
        "omega_ex": omega_ex,
    }


def plot_snapshots(classical, quantum, qw, nu, t_max, device, out: Path, n: int = 128):
    times = [0.0, 0.5 * t_max, t_max]
    labels = ["Exact", "Classical", "Quantum"]
    fig, axes = plt.subplots(len(times), 3, figsize=(11, 9.5), constrained_layout=True)
    vmax = 2.0 * 1.05

    for row, tv in enumerate(times):
        f_c = field_at(classical, None, "classical", n, tv, nu, device)
        f_q = field_at(quantum, qw, "quantum", n, tv, nu, device)
        panels = [
            (f_c["omega_ex"], f_c["u_ex"], f_c["v_ex"]),
            (f_c["omega"], f_c["u"], f_c["v"]),
            (f_q["omega"], f_q["u"], f_q["v"]),
        ]
        xs = f_c["xs"]
        for col, ((omega, u, v), title) in enumerate(zip(panels, labels)):
            ax = axes[row, col]
            im = ax.pcolormesh(
                xs, xs, omega.T, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax,
            )
            # streamlines: show the swirl (TGV vortices are stationary, only amplitude decays)
            ax.streamplot(
                xs, xs, u.T, v.T,
                color="k", density=1.1, linewidth=0.7, arrowsize=0.7,
                broken_streamlines=False,
            )
            ax.set_xlim(X_LO, X_HI)
            ax.set_ylim(X_LO, X_HI)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"t = {tv:.2f}", fontsize=11)
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.046, pad=0.02)

    fig.suptitle(
        f"TGV vorticity + streamlines  ν={nu}  T={t_max}", fontsize=13, y=1.01,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_errors(classical, quantum, qw, nu, t_max, device, out: Path, n: int = 128):
    times = [0.5 * t_max, t_max]
    fig, axes = plt.subplots(len(times), 2, figsize=(8.5, 7.5), constrained_layout=True)
    for row, tv in enumerate(times):
        f_c = field_at(classical, None, "classical", n, tv, nu, device)
        f_q = field_at(quantum, qw, "quantum", n, tv, nu, device)
        err_c = np.abs(f_c["omega"] - f_c["omega_ex"])
        err_q = np.abs(f_q["omega"] - f_q["omega_ex"])
        vmax = max(err_c.max(), err_q.max(), 1e-6)
        xs = f_c["xs"]
        for col, (err, name) in enumerate(((err_c, "Classical"), (err_q, "Quantum"))):
            ax = axes[row, col]
            im = ax.pcolormesh(xs, xs, err.T, cmap="magma", shading="auto", vmin=0, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"|Δω| {name}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"t = {tv:.2f}", fontsize=11)
            ax.text(
                0.02, 0.98,
                f"max={err.max():.3e}\nmean={err.mean():.3e}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0),
            )
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.046, pad=0.02)
    fig.suptitle("Absolute vorticity error vs exact", fontsize=13, y=1.01)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_metrics(report: dict, out: Path):
    times = sorted(report["times"], key=float)
    names = ["classical", "quantum"]
    colors = {"classical": "#2c7bb6", "quantum": "#d7191c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax = axes[0]
    for name in names:
        ys = [100 * report["times"][t][name]["vel_rel_l2"] for t in times]
        ax.plot([float(t) for t in times], ys, "o-", label=name, color=colors[name])
    ax.axhline(2.0, color="gray", ls="--", lw=1, label="2% gate")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="1% target")
    ax.set_xlabel("t")
    ax.set_ylabel("velocity rel-L2 (%)")
    ax.set_title("Velocity error vs time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    vals = [100 * report[f"{n}_vel_rel_l2_max"] for n in names]
    bars = ax.bar(names, vals, color=[colors[n] for n in names])
    ax.axhline(2.0, color="gray", ls="--", lw=1)
    ax.set_ylabel("max velocity rel-L2 (%)")
    ax.set_title("Worst-time velocity error")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(2.5, max(vals) * 1.3))

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _seed_tracers(n_rings: int = 4, n_per: int = 12) -> np.ndarray:
    """Seed particles on circles around the four TGV vortex centers."""
    centers = [(math.pi / 2, math.pi / 2), (3 * math.pi / 2, math.pi / 2),
               (math.pi / 2, 3 * math.pi / 2), (3 * math.pi / 2, 3 * math.pi / 2)]
    pts = []
    for cx, cy in centers:
        for r in np.linspace(0.35, 1.1, n_rings):
            th = np.linspace(0, 2 * math.pi, n_per, endpoint=False)
            pts.append(np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1))
    return np.concatenate(pts, axis=0)  # (P, 2)


def _bilinear(field: np.ndarray, xs: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Sample field[i,j] at (px,py) with periodic wrap. field indexed (x,y)."""
    n = len(xs)
    # map to continuous index
    fx = (px - X_LO) / (X_HI - X_LO) * n
    fy = (py - X_LO) / (X_HI - X_LO) * n
    i0 = np.floor(fx).astype(int) % n
    j0 = np.floor(fy).astype(int) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def _advect(pos: np.ndarray, u: np.ndarray, v: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
    """One RK2 step, periodic domain."""
    px, py = pos[:, 0], pos[:, 1]
    u1 = _bilinear(u, xs, px, py)
    v1 = _bilinear(v, xs, px, py)
    px2 = px + 0.5 * dt * u1
    py2 = py + 0.5 * dt * v1
    u2 = _bilinear(u, xs, px2, py2)
    v2 = _bilinear(v, xs, px2, py2)
    out = np.empty_like(pos)
    out[:, 0] = (px + dt * u2 - X_LO) % (X_HI - X_LO) + X_LO
    out[:, 1] = (py + dt * v2 - X_LO) % (X_HI - X_LO) + X_LO
    return out


def make_gif(classical, quantum, qw, nu, t_max, device, out: Path, n: int = 96, n_frames: int = 60):
    """Vorticity + streamlines + tracer particles (so the swirl is visible)."""
    ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if ffmpeg.returncode != 0:
        print("ffmpeg not found — skip gif")
        return
    vmax = 2.0 * 1.05
    dt = t_max / (n_frames - 1)
    # independent tracers per panel
    tracers = {
        "exact": _seed_tracers(),
        "classical": _seed_tracers(),
        "quantum": _seed_tracers(),
    }
    trail_len = 12
    trails = {k: [tracers[k].copy()] for k in tracers}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in range(n_frames):
            tv = t_max * i / (n_frames - 1)
            f_c = field_at(classical, None, "classical", n, tv, nu, device)
            f_q = field_at(quantum, qw, "quantum", n, tv, nu, device)
            fields = {
                "exact": (f_c["omega_ex"], f_c["u_ex"], f_c["v_ex"]),
                "classical": (f_c["omega"], f_c["u"], f_c["v"]),
                "quantum": (f_q["omega"], f_q["u"], f_q["v"]),
            }
            xs = f_c["xs"]
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
            for ax, key, title in zip(
                axes, ("exact", "classical", "quantum"), ("Exact", "Classical", "Quantum"),
            ):
                omega, u, v = fields[key]
                im = ax.pcolormesh(
                    xs, xs, omega.T, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax,
                )
                ax.streamplot(
                    xs, xs, u.T, v.T,
                    color="0.15", density=0.95, linewidth=0.55, arrowsize=0.55,
                    broken_streamlines=False,
                )
                # draw particle trails
                hist = trails[key]
                for ti, pts in enumerate(hist[-trail_len:]):
                    alpha = 0.15 + 0.85 * (ti + 1) / min(len(hist), trail_len)
                    ax.scatter(pts[:, 0], pts[:, 1], s=4, c="yellow",
                               alpha=alpha, linewidths=0, zorder=5)
                ax.scatter(tracers[key][:, 0], tracers[key][:, 1],
                           s=8, c="white", edgecolors="k", linewidths=0.3, zorder=6)
                ax.set_xlim(X_LO, X_HI)
                ax.set_ylim(X_LO, X_HI)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title)

                # advance tracers for next frame
                if i < n_frames - 1:
                    tracers[key] = _advect(tracers[key], u, v, xs, dt)
                    trails[key].append(tracers[key].copy())

            fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.02)
            fig.suptitle(
                f"TGV swirl (vorticity + streamlines + tracers)   t = {tv:.2f} / {t_max:.1f}",
                fontsize=12,
            )
            fp = tmp / f"frame_{i:04d}.png"
            fig.savefig(fp, dpi=110)
            plt.close(fig)

        palette = tmp / "palette.png"
        pattern = str(tmp / "frame_%04d.png")
        subprocess.check_call([
            "ffmpeg", "-y", "-framerate", "15", "-i", pattern,
            "-vf", "palettegen", str(palette),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([
            "ffmpeg", "-y", "-framerate", "15", "-i", pattern,
            "-i", str(palette),
            "-lavfi", "paletteuse", str(out),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classical", default="blog/checkpoint/classical")
    p.add_argument("--quantum", default="blog/checkpoint/quantum")
    p.add_argument("--out", default="blog/media")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-grid", type=int, default=128)
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    classical, cfg_c = load_classical(ROOT / args.classical, device)
    quantum, qw, cfg_q = load_quantum(ROOT / args.quantum, device)
    nu = cfg_c["nu"]
    t_max = cfg_c["t_max"]
    assert abs(nu - cfg_q["nu"]) < 1e-12 and abs(t_max - cfg_q["t_max"]) < 1e-12

    print("=== Dense evaluation ===")
    report = eval_dense(classical, quantum, qw, nu, t_max, device, n=args.n_grid)
    report["classical_params"] = cfg_c.get("n_deployed_params")
    report["quantum_params"] = cfg_q.get("n_deployed_params")
    report["classical_path"] = args.classical
    report["quantum_path"] = args.quantum
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"classical vel_max = {100*report['classical_vel_rel_l2_max']:.3f}%")
    print(f"quantum   vel_max = {100*report['quantum_vel_rel_l2_max']:.3f}%")
    print(f"gate ≤2%: {report['gate_2pct']}   gate ≤1%: {report['gate_1pct']}")
    for t, block in report["times"].items():
        c, q = block["classical"], block["quantum"]
        print(f"  t={float(t):4.2f}  C vel={100*c['vel_rel_l2']:.3f}%  "
              f"Q vel={100*q['vel_rel_l2']:.3f}%  "
              f"C pde={c['pde_rms_sq']:.2e}  Q pde={q['pde_rms_sq']:.2e}")

    plot_snapshots(classical, quantum, qw, nu, t_max, device,
                   out / "tgv_triplet_snapshots.png", n=args.n_grid)
    plot_errors(classical, quantum, qw, nu, t_max, device,
                out / "tgv_error_maps.png", n=args.n_grid)
    plot_metrics(report, out / "tgv_metrics.png")
    if not args.no_gif:
        make_gif(classical, quantum, qw, nu, t_max, device,
                 out / "tgv_triplet.gif", n=96, n_frames=40)

    if not report["gate_2pct"]:
        raise SystemExit("FAIL: models do not meet 2% velocity gate on dense grid")
    print("PASS: both models ≤2% velocity rel-L2 on dense eval grid")


if __name__ == "__main__":
    main()
