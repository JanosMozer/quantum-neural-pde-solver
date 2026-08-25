"""Regenerate v4 merger media: smooth ω animation, no tracers.

Snapshots include light streamlines for swirl; the gif is vorticity-only
at Δt=0.05 / 20 fps.
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

    t_want = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0]
    fig, axes = plt.subplots(len(t_want), 3, figsize=(11, 13), constrained_layout=True)
    for row, tv in enumerate(t_want):
        panels = fields_at(tv)
        for col, title in enumerate(["DNS", "Classical", "Quantum"]):
            ax = axes[row, col]
            omega, u, v = panels[title]
            im = ax.pcolormesh(xs_s, xs_s, omega.T, cmap=CMAP, shading="auto", vmin=0, vmax=vmax)
            ax.streamplot(
                xs_s, xs_s, u.T, v.T,
                color="0.92", density=0.7, linewidth=0.45, arrowsize=0.45,
                broken_streamlines=False,
            )
            ax.set_xlim(0, TWO_PI)
            ax.set_ylim(0, TWO_PI)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"t={tv:.1f}\n{_phase(tv)}", fontsize=9)
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.046, pad=0.02, label="ω")
    fig.suptitle("Vortex merger ω + streamlines (no tracers)")
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
                im = ax.pcolormesh(xs_s, xs_s, omega.T, cmap=CMAP, shading="auto", vmin=0, vmax=vmax)
                ax.set_xlim(0, TWO_PI)
                ax.set_ylim(0, TWO_PI)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title)
            fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.02, label="ω")
            fig.suptitle(f"{_phase(tv)}    t = {tv:.2f} / {T_END:.0f}")
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
        "Deep blue→yellow vorticity. No tracer particles.\n"
        f"Gif: t=0..{T_END:.0f}, Δt={DT}, {FPS} fps (~{T_END:.0f}s). "
        "Snapshots include streamlines for swirl.\n"
    )
    print(f"Wrote {media / 'merger_triplet.gif'}", flush=True)


if __name__ == "__main__":
    main()
