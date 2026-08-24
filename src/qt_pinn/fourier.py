"""Fourier feature maps: random RFF and deterministic TGV basis."""

import math
import torch
import torch.nn as nn
from qt_pinn.config_loader import load as _load

_f = _load()["fourier"]


class FourierFeatureMap(nn.Module):
    """Maps 3D (x, y, t) inputs to 6D Fourier feature space.

    Features: [sin(2π B x), cos(2π B x)] where B is (3,3) Gaussian matrix,
    giving 6 output features total.
    """

    def __init__(self, sigma: float = _f["sigma"], seed: int = _f["seed"]) -> None:
        super().__init__()
        torch.manual_seed(seed)
        B = torch.randn(3, 3) * sigma  # (in_dim=3, n_freqs=3)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3) -> features: (batch, 6)."""
        proj = x @ self.B  # (batch, 3)
        return torch.cat([torch.sin(2 * torch.pi * proj),
                          torch.cos(2 * torch.pi * proj)], dim=-1)  # (batch, 6)


class FourierFeatureMapWide(nn.Module):
    """Wide random Fourier features for non-TGV fields (e.g. vortex merger)."""

    def __init__(
        self,
        n_freqs: int = 32,
        sigma: float = 1.0,
        t_max: float = 40.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.t_max = float(t_max)
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(3, n_freqs, generator=g) * sigma
        B[2, :] *= 1.0 / max(self.t_max, 1e-6)
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        return torch.cat([torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)], dim=-1)


class FourierFeatureMapTGV(nn.Module):
    """Deterministic spatial Fourier basis + explicit time channel.

    Spatial: on domain [0, 2π], wavenumber k needs B_component = k / (2π).
      col0: k_x=1  → sin(x), cos(x)
      col1: k_y=1  → sin(y), cos(y)
      col2: k_x=2  → sin(2x), cos(2x)
      col3: k_y=2  → sin(2y), cos(2y)

    Temporal: TGV decays as exp(-2νt), which is NOT periodic, so sinusoidal
    features are the wrong basis for it. We append t normalised to [0,1] and
    let the tanh MLP build the envelope. Without this channel the network is
    a function of (x,y) only and can only ever reproduce the t=0 slice.
    """

    N_SPATIAL = 8

    def __init__(self, t_max: float = 1.0, n_time_feats: int = 2) -> None:
        super().__init__()
        if n_time_feats not in (1, 2):
            raise ValueError("n_time_feats must be 1 or 2")
        self.n_time_feats = n_time_feats
        self.t_max = float(t_max)
        inv_2pi = 1.0 / (2.0 * math.pi)
        # rows: [x, y, t]; cols: spatial frequency directions
        B = torch.tensor([
            [1.0 * inv_2pi, 0.0,          2.0 * inv_2pi, 0.0],
            [0.0,          1.0 * inv_2pi, 0.0,          2.0 * inv_2pi],
            [0.0,          0.0,          0.0,          0.0],
        ], dtype=torch.float32)
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return self.N_SPATIAL + self.n_time_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3) -> features: (batch, 8 + n_time_feats)."""
        proj = x @ self.B  # (batch, 4)
        spatial = torch.cat([torch.sin(2 * torch.pi * proj),
                             torch.cos(2 * torch.pi * proj)], dim=-1)
        t_norm = (x[:, 2:3] / self.t_max)
        if self.n_time_feats == 1:
            time_feats = t_norm
        else:
            time_feats = torch.cat([t_norm, t_norm ** 2], dim=-1)
        return torch.cat([spatial, time_feats], dim=-1)


class FourierFeatureMapKolmogorov(nn.Module):
    """Fourier basis tuned for Kolmogorov forcing f_x = A sin(n y).

    Includes x- and y-harmonics up to 2n to represent advected streaks and
    the primary forcing mode. Time enters as normalized t/T (non-periodic).
    """

    def __init__(
        self,
        n_force: int = 4,
        t_max: float = 5.0,
        n_time_feats: int = 2,
    ) -> None:
        super().__init__()
        if n_time_feats not in (1, 2):
            raise ValueError("n_time_feats must be 1 or 2")
        self.n_force = n_force
        self.t_max = float(t_max)
        self.n_time_feats = n_time_feats
        inv_2pi = 1.0 / (2.0 * math.pi)
        n = float(n_force)
        # columns: (kx, ky) wavenumber pairs
        pairs = [
            (1.0, 0.0), (0.0, 1.0), (0.0, n), (0.0, 2.0 * n),
            (2.0, 0.0), (n, 0.0), (1.0, n), (2.0, n),
        ]
        B = torch.zeros(3, len(pairs), dtype=torch.float32)
        for j, (kx, ky) in enumerate(pairs):
            B[0, j] = kx * inv_2pi
            B[1, j] = ky * inv_2pi
        self.register_buffer("B", B)
        self._n_spatial = 2 * len(pairs)

    @property
    def out_dim(self) -> int:
        return self._n_spatial + self.n_time_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        spatial = torch.cat([torch.sin(2 * math.pi * proj),
                             torch.cos(2 * math.pi * proj)], dim=-1)
        t_norm = x[:, 2:3] / self.t_max
        if self.n_time_feats == 1:
            time_feats = t_norm
        else:
            time_feats = torch.cat([t_norm, t_norm ** 2], dim=-1)
        return torch.cat([spatial, time_feats], dim=-1)


if __name__ == "__main__":
    model = FourierFeatureMap(sigma=0.25)
    x = torch.randn(32, 3)
    out = model(x)
    assert out.shape == (32, 6), f"Expected (32,6), got {out.shape}"
    print(f"PASS random: input {x.shape} -> output {out.shape}")

    tgv = FourierFeatureMapTGV(t_max=5.0)
    coords = torch.stack([
        torch.linspace(0, 2 * math.pi, 32),
        torch.zeros(32),
        torch.full((32,), 2.5),
    ], dim=-1)
    feats = tgv(coords)
    assert feats.shape == (32, tgv.out_dim) == (32, 10)
    x1 = coords[:, 0]
    assert (feats[:, 0] - torch.sin(x1)).abs().max() < 1e-5
    assert (feats[:, 4] - torch.cos(x1)).abs().max() < 1e-5
    assert (feats[:, 8] - 0.5).abs().max() < 1e-6, "time channel must be live"
    # time channel must actually vary with t
    c2 = coords.clone(); c2[:, 2] = 0.0
    assert (tgv(c2)[:, 8] - feats[:, 8]).abs().max() > 0.4
    print(f"PASS TGV: shape {feats.shape}, spatial basis + live time channel")

    kol = FourierFeatureMapKolmogorov(n_force=4, t_max=5.0)
    ck = torch.stack([
        torch.zeros(16),
        torch.linspace(0, 2 * math.pi, 16),
        torch.full((16,), 2.5),
    ], dim=-1)
    fk = kol(ck)
    assert fk.shape == (16, kol.out_dim)
    yk = ck[:, 1]
    # column 2 is (0, n) -> sin(4y) at index 2, cos(4y) at index 10
    assert (fk[:, 2] - torch.sin(4 * yk)).abs().max() < 1e-5
    assert (fk[:, 10] - torch.cos(4 * yk)).abs().max() < 1e-5
    print(f"PASS Kolmogorov: shape {fk.shape}, forcing mode cos(4y) present")
