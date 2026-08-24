"""2D periodic incompressible NS — vorticity–streamfunction spectral solver (GPU).

Used for vortex-merger DNS reference (no closed-form solution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


TWO_PI = 2.0 * math.pi


@dataclass
class SpectralNS2D:
    n: int = 256
    nu: float = 0.01
    device: torch.device | str = "cuda"
    dealias: bool = True

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        n = self.n
        # wave numbers
        k = torch.fft.fftfreq(n, d=1.0 / n, device=self.device)  # 0..n/2-1, -n/2..
        kx = k.view(n, 1).expand(n, n)
        ky = k.view(1, n).expand(n, n)
        self.kx = kx
        self.ky = ky
        self.k2 = kx * kx + ky * ky
        self.k2_safe = self.k2.clone()
        self.k2_safe[0, 0] = 1.0  # avoid /0 for streamfunction
        # 2/3 dealias mask
        if self.dealias:
            kk = torch.fft.fftfreq(n, d=1.0 / n, device=self.device).abs()
            cut = n / 3.0
            self.mask = ((kk.view(n, 1) < cut) & (kk.view(1, n) < cut)).to(torch.float32)
        else:
            self.mask = torch.ones(n, n, device=self.device)
        xs = torch.linspace(0.0, TWO_PI, n + 1, device=self.device)[:-1]
        self.x = xs
        self.y = xs
        self.dx = float(TWO_PI / n)

    def omega_to_uv(self, omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert ∇²ψ = −ω, then u=∂ψ/∂y, v=−∂ψ/∂x (spectral)."""
        wh = torch.fft.fft2(omega)
        wh = wh * self.mask
        psi_h = wh / self.k2_safe
        psi_h[0, 0] = 0.0
        # u = ψ_y → ik_y ψ;  v = −ψ_x → −ik_x ψ
        u = torch.fft.ifft2(1j * self.ky * psi_h).real
        v = torch.fft.ifft2(-1j * self.kx * psi_h).real
        return u, v

    def pressure(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Solve ∇²p = −∇·((u·∇)u) spectrally (divergence-free projection residual)."""
        uh, vh = torch.fft.fft2(u), torch.fft.fft2(v)
        ux = torch.fft.ifft2(1j * self.kx * uh).real
        uy = torch.fft.ifft2(1j * self.ky * uh).real
        vx = torch.fft.ifft2(1j * self.kx * vh).real
        vy = torch.fft.ifft2(1j * self.ky * vh).real
        rhs = -(ux * ux + 2.0 * uy * vx + vy * vy)  # −(u_x² + 2 u_y v_x + v_y²)
        rh = torch.fft.fft2(rhs) * self.mask
        ph = rh / self.k2_safe
        ph[0, 0] = 0.0
        return torch.fft.ifft2(ph).real

    def rhs_omega(self, omega: torch.Tensor) -> torch.Tensor:
        u, v = self.omega_to_uv(omega)
        wh = torch.fft.fft2(omega)
        wx = torch.fft.ifft2(1j * self.kx * wh).real
        wy = torch.fft.ifft2(1j * self.ky * wh).real
        adv = u * wx + v * wy
        adv_h = torch.fft.fft2(adv) * self.mask
        # ω_t = −u·∇ω + ν ∇²ω
        return torch.fft.ifft2(-adv_h - self.nu * self.k2 * wh).real

    def step_rk4(self, omega: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.rhs_omega(omega)
        k2 = self.rhs_omega(omega + 0.5 * dt * k1)
        k3 = self.rhs_omega(omega + 0.5 * dt * k2)
        k4 = self.rhs_omega(omega + dt * k3)
        return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def cfl_dt(self, omega: torch.Tensor, cfl: float = 0.5) -> float:
        u, v = self.omega_to_uv(omega)
        umax = float(torch.maximum(u.abs().max(), v.abs().max()).item()) + 1e-8
        return cfl * self.dx / umax


def gaussian_vortices(
    xs: torch.Tensor,
    ys: torch.Tensor,
    centers: list[tuple[float, float]],
    gamma: float = 5.0,
    delta: float = 0.45,
) -> torch.Tensor:
    """Sum of same-sign Gaussian vortices (periodic images via minimum image)."""
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    omega = torch.zeros_like(xg)
    amp = gamma / (math.pi * delta * delta)
    for cx, cy in centers:
        # minimum-image distance on [0, 2π)
        dx = (xg - cx + math.pi) % TWO_PI - math.pi
        dy = (yg - cy + math.pi) % TWO_PI - math.pi
        omega = omega + amp * torch.exp(-(dx * dx + dy * dy) / (delta * delta))
    # remove mean (compatible with periodic streamfunction)
    omega = omega - omega.mean()
    return omega


def four_quadrant_centers(offset: float = 0.0) -> list[tuple[float, float]]:
    """Centers of the four quadrants, optionally pulled inward by `offset`."""
    # offset>0 moves vortices toward domain center → stronger merger
    c = math.pi / 2
    d = offset
    return [
        (c + d, c + d),
        (3 * c - d, c + d),
        (c + d, 3 * c - d),
        (3 * c - d, 3 * c - d),
    ]


@torch.no_grad()
def simulate(
    n: int = 256,
    nu: float = 0.01,
    t_max: float = 20.0,
    n_save: int = 81,
    gamma: float = 6.0,
    delta: float = 0.5,
    pull_in: float = 0.55,
    device: str = "cuda",
    cfl: float = 0.45,
) -> dict:
    """Run vortex-merger DNS. Returns tensors on CPU for checkpointing."""
    sol = SpectralNS2D(n=n, nu=nu, device=device)
    centers = four_quadrant_centers(offset=pull_in)
    omega = gaussian_vortices(sol.x, sol.y, centers, gamma=gamma, delta=delta)

    t_save = torch.linspace(0.0, t_max, n_save)
    omegas = []
    us, vs, ps = [], [], []
    times = []

    t = 0.0
    save_idx = 0
    # save t=0
    u, v = sol.omega_to_uv(omega)
    p = sol.pressure(u, v)
    omegas.append(omega.cpu().clone())
    us.append(u.cpu().clone())
    vs.append(v.cpu().clone())
    ps.append(p.cpu().clone())
    times.append(0.0)
    save_idx = 1

    step = 0
    while t < t_max - 1e-12 and save_idx < n_save:
        dt = min(sol.cfl_dt(omega, cfl=cfl), t_save[save_idx].item() - t)
        dt = max(dt, 1e-6)
        omega = sol.step_rk4(omega, dt)
        t += dt
        step += 1
        if t >= t_save[save_idx].item() - 1e-9 or abs(t - t_max) < 1e-9:
            u, v = sol.omega_to_uv(omega)
            p = sol.pressure(u, v)
            omegas.append(omega.cpu().clone())
            us.append(u.cpu().clone())
            vs.append(v.cpu().clone())
            ps.append(p.cpu().clone())
            times.append(t)
            save_idx += 1
            if step % 50 == 0 or save_idx >= n_save:
                circ = float(omega.abs().mean())
                print(f"  DNS t={t:6.3f}/{t_max}  |ω|_mean={circ:.4f}  steps={step}")

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
        "gamma": gamma,
        "delta": delta,
        "pull_in": pull_in,
        "centers": centers,
    }


def sample_dns(
    dns: dict,
    n_pts: int,
    device: torch.device,
    t_sample: str = "uniform",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random (x,y,t) samples with trilinear-ish lookup of (u,v,p) from DNS grids.

    Returns x,y,t, target(N,3).
    """
    u_all = dns["u"].to(device)  # (T,N,N)
    v_all = dns["v"].to(device)
    p_all = dns["p"].to(device)
    ts = dns["t"].to(device)
    n = u_all.shape[-1]
    n_t = ts.numel()
    t_max = float(dns["t_max"])

    # sample indices
    ix = torch.randint(0, n, (n_pts,), device=device)
    iy = torch.randint(0, n, (n_pts,), device=device)
    if t_sample == "tail":
        # bias toward later times (merger)
        it = (torch.rand(n_pts, device=device).sqrt() * (n_t - 1)).long()
    else:
        it = torch.randint(0, n_t, (n_pts,), device=device)

    x = dns["x"].to(device)[ix]
    y = dns["y"].to(device)[iy]
    t = ts[it]
    tgt = torch.stack([
        u_all[it, ix, iy],
        v_all[it, ix, iy],
        p_all[it, ix, iy],
    ], dim=-1)
    return x, y, t, tgt


def ic_values_from_dns(dns: dict, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bilinear sample of DNS fields at t=0 for hard-IC ansatz."""
    device = x.device
    u0 = dns["u"][0].to(device)
    v0 = dns["v"][0].to(device)
    p0 = dns["p"][0].to(device)
    n = u0.shape[0]
    # map to grid coords
    fx = (x / TWO_PI) * n
    fy = (y / TWO_PI) * n
    i0 = fx.floor().long() % n
    j0 = fy.floor().long() % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    tx = (fx - fx.floor()).clamp(0, 1)
    ty = (fy - fy.floor()).clamp(0, 1)

    def bil(field):
        return (
            (1 - tx) * (1 - ty) * field[i0, j0]
            + tx * (1 - ty) * field[i1, j0]
            + (1 - tx) * ty * field[i0, j1]
            + tx * ty * field[i1, j1]
        )

    return bil(u0), bil(v0), bil(p0)
