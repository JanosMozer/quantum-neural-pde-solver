"""Regenerate v4 merger media: vorticity gif + core markers (no streamlines).

Red + markers are the ω maxima of each panel (re-detected every frame + local
COM). They expose orbital dynamics: DNS co-rotates; current Classical/Quantum
checkpoints largely freeze after a brief early turn.
"""

from __future__ import annotations

import json
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

from qt_pinn.ns2d_spectral import TWO_PI
from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.tgv_demo import resolve_device
from scripts.exp_merger_omega import CMAP, HarmMLP, V3

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "blog" / "checkpoint" / "v4"

T_END = 15.0
DT = 0.05
FPS = 20


def _lerp_dns(dns: dict, name: str, t: float) -> np.ndarray:
    ts = dns["t"].numpy()
    if t <= ts[0]:
        return dns[name][0].numpy()
    if t >= ts[-1]:
        return dns[name][-1].numpy()
    j = int(np.searchsorted(ts, t) - 1)
    j = max(0, min(j, len(ts) - 2))
    a = (t - ts[j]) / (ts[j + 1] - ts[j])
    return ((1 - a) * dns[name][j] + a * dns[name][j + 1]).numpy()


def _fd_omega(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
    dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    return dvdx - dudy


def _phase(t: float) -> str:
    if t < 4:
        return "1. Four co-rotating vortices"
    if t < 10:
        return "2. Orbiting & stretching toward center"
    return "3. Merging into one core"


def _sample_uv(u: np.ndarray, v: np.ndarray, xs: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Bilinear sample of (u,v) at (x,y); arrays are indexing='ij'."""
    n = len(xs)
    # map to fractional index
    fx = (x / TWO_PI) * n
    fy = (y / TWO_PI) * n
    i0 = int(np.floor(fx)) % n
    j0 = int(np.floor(fy)) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    ax = fx - np.floor(fx)
    ay = fy - np.floor(fy)
    uu = (
        (1 - ax) * (1 - ay) * u[i0, j0]
        + ax * (1 - ay) * u[i1, j0]
        + (1 - ax) * ay * u[i0, j1]
        + ax * ay * u[i1, j1]
    )
    vv = (
        (1 - ax) * (1 - ay) * v[i0, j0]
        + ax * (1 - ay) * v[i1, j0]
        + (1 - ax) * ay * v[i0, j1]
        + ax * ay * v[i1, j1]
    )
    return float(uu), float(vv)


def _period_dist(x0: float, y0: float, x1: float, y1: float) -> float:
    dx = (x1 - x0 + np.pi) % TWO_PI - np.pi
    dy = (y1 - y0 + np.pi) % TWO_PI - np.pi
    return float(np.hypot(dx, dy))


def _local_com(
    omega: np.ndarray,
    xs: np.ndarray,
    x0: float,
    y0: float,
    radius: float = 0.55,
    power: float = 2.0,
) -> tuple[float, float]:
    """Sub-pixel center of mass of ω^power in a disk around (x0,y0)."""
    n = len(xs)
    dx = TWO_PI / n
    i_c = int(np.round((x0 / TWO_PI) * n)) % n
    j_c = int(np.round((y0 / TWO_PI) * n)) % n
    r_idx = max(3, int(radius / dx) + 1)
    w_sum = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for di in range(-r_idx, r_idx + 1):
        for dj in range(-r_idx, r_idx + 1):
            if di * di + dj * dj > r_idx * r_idx:
                continue
            i = (i_c + di) % n
            j = (j_c + dj) % n
            xi = xs[i]
            yj = xs[j]
            ddx = (xi - x0 + np.pi) % TWO_PI - np.pi
            ddy = (yj - y0 + np.pi) % TWO_PI - np.pi
            if ddx * ddx + ddy * ddy > radius * radius:
                continue
            w = float(max(omega[i, j], 0.0)) ** power
            if w <= 0:
                continue
            w_sum += w
            x_sum += w * (x0 + ddx)
            y_sum += w * (y0 + ddy)
    if w_sum < 1e-12:
        return x0, y0
    return float(x_sum / w_sum) % TWO_PI, float(y_sum / w_sum) % TWO_PI


def _global_peaks(omega: np.ndarray, xs: np.ndarray, k: int = 4) -> np.ndarray:
    """Top-k vorticity peaks on the *plotted* ω field (indexing='ij')."""
    a = omega.copy()
    n = a.shape[0]
    excl = max(6, n // 16)
    thr = 0.18 * float(max(omega.max(), 1e-8))
    peaks = []
    for _ in range(k):
        i, j = np.unravel_index(int(np.argmax(a)), a.shape)
        if a[i, j] < thr:
            break
        peaks.append((float(xs[i]), float(xs[j])))
        a[max(0, i - excl) : i + excl + 1, max(0, j - excl) : j + excl + 1] = 0.0
    if not peaks:
        return np.zeros((0, 2))
    return np.asarray(peaks, dtype=np.float64)


def _init_cores(omega: np.ndarray, xs: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Peaks of plotted ω, sorted by angle about domain center."""
    del seeds
    return _cores_from_omega(omega, xs)


def _cores_from_omega(omega: np.ndarray, xs: np.ndarray, k: int = 4) -> np.ndarray:
    """Detect ω peaks, merge neighbors, angle-sort, COM-refine.

    Markers are always maxima of the *plotted* panel ω (never velocity).
    Nearby peaks (merged core) collapse to one marker so late-time PINN
    lobes don't leave orphan + in the blue.
    """
    peaks = _global_peaks(omega, xs, k=k)
    if len(peaks) == 0:
        return np.zeros((0, 2))

    # Strength at each peak (for merge decisions)
    def peak_w(p):
        i = int(np.argmin(np.abs(xs - p[0])))
        j = int(np.argmin(np.abs(xs - p[1])))
        return float(omega[i, j])

    # Greedy merge: if two peaks closer than merge_dist, keep the stronger
    merge_dist = 0.55
    kept: list[np.ndarray] = []
    order = sorted(range(len(peaks)), key=lambda i: -peak_w(peaks[i]))
    for i in order:
        p = peaks[i]
        if any(_period_dist(float(p[0]), float(p[1]), float(q[0]), float(q[1])) < merge_dist for q in kept):
            continue
        kept.append(p)
    peaks = np.asarray(kept, dtype=np.float64)

    cx = cy = np.pi
    order = np.argsort(np.arctan2(peaks[:, 1] - cy, peaks[:, 0] - cx))
    peaks = peaks[order]
    out = []
    for px, py in peaks:
        x, y = _local_com(omega, xs, float(px), float(py), radius=0.35, power=4.0)
        out.append((x, y))
    return np.asarray(out, dtype=np.float64)


def _step_cores(
    cores: np.ndarray,
    omega: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    xs: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Re-detect ω peaks every frame (no velocity / no sticky COM walk)."""
    del cores, u, v, dt
    return _cores_from_omega(omega, xs)


def _draw_panel(ax, xs_s, omega, cores, vmax, title: str | None) -> None:
    ax.pcolormesh(xs_s, xs_s, omega.T, cmap=CMAP, shading="auto", vmin=0, vmax=vmax)
    if cores is not None and len(cores):
        ax.plot(cores[:, 0], cores[:, 1], "r+", ms=9, mew=1.6, alpha=0.95)
    ax.set_xlim(0, TWO_PI)
    ax.set_ylim(0, TWO_PI)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def main() -> None:
    device = resolve_device("cuda")
    dns = torch.load(V3 / "dns" / "reference.pt", map_location="cpu", weights_only=False)
    media = V4 / "media"
    media.mkdir(exist_ok=True)

    cfg_c = json.loads((V4 / "classical" / "config.json").read_text())
    model_c = HarmMLP(
        hidden=tuple(cfg_c.get("hidden", [96, 96])),
        t_max=float(dns["t_max"]),
        k_max=int(cfg_c.get("k_max", 6)),
        axis_extra=int(cfg_c.get("axis_extra", 0)),
        orbit_omega=float(cfg_c.get("orbit_omega", 0.0) or 0.0),
        orbit_frame=bool(cfg_c.get("orbit_frame", False)),
    ).to(device)
    model_c.load_state_dict(torch.load(V4 / "classical" / "model.pt", map_location=device, weights_only=True))
    model_c.eval()

    cfg_q = json.loads((V4 / "quantum" / "config.json").read_text())
    target = TargetPINNNS(
        fourier=cfg_q.get("qt_fourier", "harm"),
        hard_ic=False,
        hidden=tuple(cfg_q.get("qt_hidden", [48, 48])),
        t_max=dns["t_max"],
        n_freqs=cfg_q.get("n_freqs", 3),
        fourier_seed=0,
        orbit_omega=float(cfg_q.get("orbit_omega", 0.0) or 0.0),
    ).to(device)
    wq = torch.load(V4 / "quantum" / "deployed_weights.pt", map_location=device, weights_only=True)

    xs = dns["x"].numpy()
    step = max(1, len(xs) // 128)
    xs_s = xs[::step]
    n = len(xs_s)
    dx = TWO_PI / n
    xg, yg = np.meshgrid(xs_s, xs_s, indexing="ij")
    xt0 = torch.tensor(xg.ravel(), device=device, dtype=torch.float32)
    yt0 = torch.tensor(yg.ravel(), device=device, dtype=torch.float32)
    vmax = float(dns["omega"][0].max()) * 1.05
    seeds = np.asarray([(float(a), float(b)) for a, b in dns["centers"]], dtype=np.float64)

    def fields_at(tv: float) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        u_d = _lerp_dns(dns, "u", tv)[::step, ::step]
        v_d = _lerp_dns(dns, "v", tv)[::step, ::step]
        o_d = _lerp_dns(dns, "omega", tv)[::step, ::step]
        tt = torch.full_like(xt0, float(tv))
        with torch.no_grad():
            pc = model_c(xt0, yt0, tt).cpu().numpy().reshape(n, n, 3)
            pq = target(xt0, yt0, tt, wq).cpu().numpy().reshape(n, n, 3)
        return {
            "DNS": (o_d, u_d, v_d),
            "Classical": (_fd_omega(pc[..., 0], pc[..., 1], dx), pc[..., 0], pc[..., 1]),
            "Quantum": (_fd_omega(pq[..., 0], pq[..., 1], dx), pq[..., 0], pq[..., 1]),
        }

    # Snapshots first (fast check that markers sit on yellow peaks).
    t_want = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0]
    fig, axes = plt.subplots(len(t_want), 3, figsize=(11, 13), constrained_layout=True)
    for row, tv in enumerate(t_want):
        panels = fields_at(tv)
        for col, title in enumerate(["DNS", "Classical", "Quantum"]):
            ax = axes[row, col]
            omega, _, _ = panels[title]
            cores = _cores_from_omega(omega, xs_s)
            _draw_panel(ax, xs_s, omega, cores, vmax, title if row == 0 else None)
            if col == 0:
                ax.set_ylabel(f"t={tv:.1f}\n{_phase(tv)}", fontsize=9)
        fig.colorbar(
            axes[row, -1].collections[0], ax=axes[row, :].tolist(),
            fraction=0.046, pad=0.02, label="ω",
        )
    fig.suptitle(r"$\omega$ snapshots (red + = vortex relative maxima)")
    fig.savefig(media / "merger_triplet_snapshots.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {media / 'merger_triplet_snapshots.png'}", flush=True)

    t_frames = np.arange(0.0, T_END + 1e-9, DT)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fi, tv in enumerate(t_frames):
            panels = fields_at(tv)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)
            for ax, title in zip(axes, ["DNS", "Classical", "Quantum"]):
                omega, _, _ = panels[title]
                cores = _cores_from_omega(omega, xs_s)
                _draw_panel(ax, xs_s, omega, cores, vmax, title)
            fig.colorbar(
                axes[-1].collections[0], ax=axes.tolist(),
                fraction=0.046, pad=0.02, label="ω",
            )
            fig.suptitle(f"Co-rotating vortices    t = {tv:.2f} / {T_END:.0f}")
            fig.savefig(tmp / f"frame_{fi:04d}.png", dpi=110)
            plt.close(fig)
            if fi % 50 == 0:
                print(f"  frame {fi}/{len(t_frames)-1}  t={tv:.2f}", flush=True)

        out = media / "merger_triplet.gif"
        palette = tmp / "palette.png"
        pattern = str(tmp / "frame_%04d.png")
        subprocess.check_call(
            ["ffmpeg", "-y", "-framerate", str(FPS), "-i", pattern, "-vf", "palettegen", str(palette)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            [
                "ffmpeg", "-y", "-framerate", str(FPS), "-i", pattern, "-i", str(palette),
                "-lavfi", "paletteuse", str(out),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    (media / "CAPTION.md").write_text(
        "# Vortex merger\n\n"
        "Deep blue→yellow vorticity. Red **+** markers are the maxima of each "
        "panel’s own plotted ω (re-detected every frame + local COM refine). "
        "They are grounded in the colormap, not in passive tracers.\n\n"
        "Known fidelity gap (v4 Classical / Quantum used here): peak **radii** "
        "spiral inward similarly to DNS, but peak **angles** nearly freeze after "
        "~t=1 (DNS keeps co-rotating at roughly −50 to −60° per unit time). "
        "Pointwise ω Rel-L2 can still look good while orbital phase is wrong.\n"
        f"Gif: t=0..{T_END:.0f}, Δt={DT}, {FPS} fps. No streamlines.\n"
    )
    print(f"Wrote {media / 'merger_triplet.gif'}", flush=True)


if __name__ == "__main__":
    main()
