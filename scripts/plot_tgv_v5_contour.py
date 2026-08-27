#!/usr/bin/env python3
"""v5 TGV media: multi-vortex exact + Exact|Classical|Quantum animations.

  # denser field matching the 4×4-lobe screenshot (k=2)
  .venv/bin/python -u scripts/plot_tgv_v5_contour.py --mode dense

  # Exact | Classical | Quantum triplet animation (k=1 — what v2 PINNs solve)
  .venv/bin/python -u scripts/plot_tgv_v5_contour.py --mode triplet

  # both
  .venv/bin/python -u scripts/plot_tgv_v5_contour.py --mode all
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
from matplotlib.colors import LinearSegmentedColormap

from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.tgv_demo import DirectNSMLP, resolve_device

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "blog" / "checkpoint" / "v2"
V5 = ROOT / "blog" / "checkpoint" / "v5"
X_LO, X_HI = 0.0, 2.0 * math.pi

CMAP = LinearSegmentedColormap.from_list(
    "deepblue_yellow",
    ["#061428", "#0B3D91", "#1A8FBF", "#7BC96F", "#F4D35E", "#FFF3B0"],
    N=256,
)


def exact_abs_omega(
    x: np.ndarray, y: np.ndarray, t: float, nu: float, u0: float = 1.0, k: float = 1.0
) -> np.ndarray:
    """|ω| = 2 k U0 |sin(kx) sin(ky)| exp(-2 ν k² t). k=2 → 4×4 lobes in [0,2π]²."""
    return (
        2.0
        * u0
        * k
        * np.abs(np.sin(k * x) * np.sin(k * y))
        * np.exp(-2.0 * nu * k * k * t)
    )


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
    return target, {kk: vv.to(device) for kk, vv in w.items()}, cfg


def fd_omega(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    return dv_dx - du_dy


@torch.no_grad()
def pred_abs_omega(model, weights, n: int, t_val: float, device) -> np.ndarray:
    xs = torch.linspace(X_LO, X_HI, n, device=device)
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    t = torch.full_like(x, float(t_val))
    pred = model(x, y, t) if weights is None else model(x, y, t, weights)
    u = pred[:, 0].cpu().numpy().reshape(n, n)
    v = pred[:, 1].cpu().numpy().reshape(n, n)
    dx = (X_HI - X_LO) / n
    return np.abs(fd_omega(u, v, dx))


def _encode_gif(frame_dir: Path, out: Path, fps: int, n_frames: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    palette = frame_dir / "palette.png"
    pattern = str(frame_dir / "frame_%04d.png")
    subprocess.run(
        [
            ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
            "-vf", "palettegen=stats_mode=diff", str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
            "-i", str(palette),
            "-lavfi", "paletteuse=dither=bayer:bayer_scale=5",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    print(f"Wrote {out}  ({n_frames} frames @ {fps} fps)", flush=True)


def write_exact_gif(
    *,
    n: int,
    nu: float,
    k: float,
    dpi: int,
    out: Path,
    t_end: float = 15.0,
    dt: float = 0.1,
    fps: int = 10,
) -> int:
    xs = np.linspace(X_LO, X_HI, n)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")
    times = np.arange(0.0, t_end + 0.5 * dt, dt)
    vmax = float(np.max(exact_abs_omega(xg, yg, 0.0, nu, k=k))) * 1.02
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tgv_exact_") as tmp:
        tmp_path = Path(tmp)
        for i, t in enumerate(times):
            w = exact_abs_omega(xg, yg, float(t), nu, k=k)
            fig, ax = plt.subplots(figsize=(8.0, 7.5), constrained_layout=True)
            im = ax.pcolormesh(
                xs, xs, w.T, cmap=CMAP, shading="auto", vmin=0.0, vmax=vmax,
            )
            ax.set_aspect("equal")
            ax.set_xlim(X_LO, X_HI)
            ax.set_ylim(X_LO, X_HI)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                rf"Taylor–Green Vortex $|\omega|$  $k={k:g}$  $\nu={nu:g}$  $t={t:.1f}$",
                fontsize=13,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$|\omega|$")
            fig.savefig(tmp_path / f"frame_{i:04d}.png", dpi=dpi, facecolor="white")
            plt.close(fig)
            if i % 25 == 0 or i == len(times) - 1:
                print(f"  [k={k:g}] frame {i+1}/{len(times)} t={t:.1f}", flush=True)
        _encode_gif(tmp_path, out, fps, len(times))
    return len(times)


def write_triplet_gif(
    *,
    n: int,
    nu: float,
    dpi: int,
    out: Path,
    device: torch.device,
    t_end: float = 5.0,
    dt: float = 0.1,
    fps: int = 10,
) -> int:
    """Exact | Classical | Quantum (k=1). Default t_end=5 matches v2 t_max."""
    classical, _ = load_classical(V2 / "classical", device)
    quantum, qw, _ = load_quantum(V2 / "quantum", device)
    xs = np.linspace(X_LO, X_HI, n)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")
    times = np.arange(0.0, t_end + 0.5 * dt, dt)
    vmax = float(np.max(exact_abs_omega(xg, yg, 0.0, nu, k=1.0))) * 1.02
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tgv_trip_") as tmp:
        tmp_path = Path(tmp)
        for i, t in enumerate(times):
            w_ex = exact_abs_omega(xg, yg, float(t), nu, k=1.0)
            w_c = pred_abs_omega(classical, None, n, float(t), device)
            w_q = pred_abs_omega(quantum, qw, n, float(t), device)
            fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
            panels = [
                (w_ex, "Exact (DNS)"),
                (w_c, "Classical"),
                (w_q, "Quantum"),
            ]
            im = None
            for ax, (w, title) in zip(axes, panels):
                im = ax.pcolormesh(
                    xs, xs, w.T, cmap=CMAP, shading="auto", vmin=0.0, vmax=vmax,
                )
                ax.set_aspect("equal")
                ax.set_xlim(X_LO, X_HI)
                ax.set_ylim(X_LO, X_HI)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=12)
            fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label=r"$|\omega|$")
            fig.suptitle(
                rf"Taylor–Green $|\omega|$  $\nu={nu:g}$  $t={t:.1f}$",
                fontsize=13,
            )
            fig.savefig(tmp_path / f"frame_{i:04d}.png", dpi=dpi, facecolor="white")
            plt.close(fig)
            if i % 25 == 0 or i == len(times) - 1:
                print(f"  [triplet] frame {i+1}/{len(times)} t={t:.1f}", flush=True)
        _encode_gif(tmp_path, out, fps, len(times))
    return len(times)


def plot_snapshots(
    n: int, nu: float, k: float, dpi: int, out: Path, times=(0, 3, 6, 9, 12, 15)
) -> None:
    xs = np.linspace(X_LO, X_HI, n)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")
    vmax = float(np.max(exact_abs_omega(xg, yg, 0.0, nu, k=k))) * 1.02
    fig, axes = plt.subplots(
        1, len(times), figsize=(3.0 * len(times), 3.2), constrained_layout=True
    )
    im = None
    for ax, t in zip(axes, times):
        w = exact_abs_omega(xg, yg, float(t), nu, k=k)
        im = ax.pcolormesh(
            xs, xs, w.T, cmap=CMAP, shading="auto", vmin=0.0, vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(X_LO, X_HI)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$t={t:g}$", fontsize=11)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label=r"$|\omega|$")
    fig.suptitle(
        rf"Taylor–Green Vortex $|\omega|$  $k={k:g}$, $\nu={nu:g}$  ({int(k)*2}×{int(k)*2} lobes)",
        fontsize=13,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def plot_triplet_still(
    n: int, nu: float, t: float, dpi: int, out: Path, device: torch.device
) -> None:
    classical, _ = load_classical(V2 / "classical", device)
    quantum, qw, _ = load_quantum(V2 / "quantum", device)
    xs = np.linspace(X_LO, X_HI, n)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")
    panels = [
        (exact_abs_omega(xg, yg, t, nu, k=1.0), "Exact (DNS)"),
        (pred_abs_omega(classical, None, n, t, device), "Classical"),
        (pred_abs_omega(quantum, qw, n, t, device), "Quantum"),
    ]
    vmax = max(float(np.max(panels[0][0])), 1e-8) * 1.02
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    im = None
    for ax, (w, title) in zip(axes, panels):
        im = ax.pcolormesh(
            xs, xs, w.T, cmap=CMAP, shading="auto", vmin=0.0, vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(X_LO, X_HI)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label=r"$|\omega|$")
    fig.suptitle(rf"Taylor–Green Vortex $|\omega|$  $\nu={nu:g}$, $t={t:g}$", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def plot_dense_still(n: int, nu: float, k: float, t: float, dpi: int, out: Path) -> None:
    xs = np.linspace(X_LO, X_HI, n)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")
    w = exact_abs_omega(xg, yg, t, nu, k=k)
    vmax = float(np.max(exact_abs_omega(xg, yg, 0.0, nu, k=k))) * 1.02
    fig, ax = plt.subplots(figsize=(10.0, 9.5), constrained_layout=True)
    im = ax.pcolormesh(xs, xs, w.T, cmap=CMAP, shading="auto", vmin=0.0, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(X_LO, X_HI)
    ticks = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
    labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(
        rf"Taylor–Green $|\omega|$  $k={k:g}$  ($\nu={nu:g}$, $t={t:g}$) — denser vortices",
        fontsize=14,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$|\omega|$")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("dense", "triplet", "all"),
        default="all",
        help="dense=k=2 multi-vortex exact; triplet=Exact|Classical|Quantum; all=both",
    )
    ap.add_argument("--k-dense", type=float, default=2.0, help="wavenumber for dense field")
    ap.add_argument("--n", type=int, default=640)
    ap.add_argument("--gif-n", type=int, default=384)
    ap.add_argument("--dpi", type=int, default=280)
    ap.add_argument("--gif-dpi", type=int, default=110)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--t-end", type=float, default=5.0,
                    help="triplet GIF end time (v2 models use t_max=5)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    media = V5 / "media"
    media.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    do_dense = args.mode in ("dense", "all")
    do_trip = args.mode in ("triplet", "all")

    n_dense = n_trip = 0
    if do_dense:
        print(f"=== dense multi-vortex k={args.k_dense:g} ===", flush=True)
        plot_dense_still(
            args.n, args.nu, args.k_dense, 0.0, args.dpi,
            media / "tgv_dense_t0.png",
        )
        plot_snapshots(
            min(args.n, 480), args.nu, args.k_dense, args.dpi,
            media / "tgv_dense_snapshots.png",
        )
        n_dense = write_exact_gif(
            n=args.gif_n, nu=args.nu, k=args.k_dense, dpi=args.gif_dpi,
            out=media / "tgv_dense.gif", fps=args.fps,
        )

    if do_trip:
        print("=== Exact | Classical | Quantum (k=1) ===", flush=True)
        plot_triplet_still(
            min(args.n, 384), args.nu, 0.0, args.dpi,
            media / "tgv_triplet_t0.png", device,
        )
        n_trip = write_triplet_gif(
            n=min(args.gif_n, 320), nu=args.nu, dpi=args.gif_dpi,
            out=media / "tgv_triplet.gif", device=device, fps=args.fps,
            t_end=args.t_end,
        )

    (media / "CAPTION.md").write_text(
        "# Taylor–Green vortex (v5)\n\n"
        "Deep blue → yellow colormap. Exact field "
        r"$|\omega|=2k|\sin(kx)\sin(ky)|e^{-2\nu k^{2}t}$"
        ".\n\n"
        "## Multi-vortex (screenshot-like density)\n\n"
        f"- `tgv_dense.gif` — **k={args.k_dense:g}** → denser lobes (4×4 for k=2), "
        f"t=0…15 Δt=0.1, {n_dense or 151} frames @ {args.fps} fps.\n"
        "- `tgv_dense_t0.png`, `tgv_dense_snapshots.png`\n\n"
        "## Exact | Classical | Quantum\n\n"
        "- `tgv_triplet.gif` — **k=1** (v2 PINN training mode), same time grid. "
        "Panels: Exact (analytic DNS) | Classical | Quantum.\n"
        "- `tgv_triplet_t0.png`\n\n"
        "Note: v2 classical/quantum are trained on **k=1** only; the denser k=2 "
        "field is exact-only.\n"
    )
    (V5 / "notes.md").write_text(
        "# v5 — Taylor–Green animations\n\n"
        "- **Dense / multi-vortex:** `media/tgv_dense.gif` with wavenumber "
        f"**k={args.k_dense:g}** (more vortices than the classic 2×2 k=1 pattern).\n"
        "- **Solver comparison:** `media/tgv_triplet.gif` — Exact | Classical | Quantum "
        "(k=1, v2 polish weights).\n"
        "- Colormap: merger-style deep blue → yellow.\n"
        "- Generator: `scripts/plot_tgv_v5_contour.py`\n"
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
