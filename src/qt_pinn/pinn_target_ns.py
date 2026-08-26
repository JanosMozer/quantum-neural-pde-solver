"""Target PINN for 2D NS: (x,y,t) -> (u, v, p). Three output fields.

Supports:
  - random RFF (6 features) or deterministic TGV Fourier (8 features)
  - optional hard IC ansatz: field = IC(x,y) + t * N(x,y,t)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from qt_pinn.fourier import FourierFeatureMap, FourierFeatureMapTGV, FourierFeatureMapKolmogorov, FourierFeatureMapWide, FourierFeatureMapHarmonic

# Default sizes for random-RFF path (must match qnn_generator_ns.py when used)
H1_DEFAULT, H2_DEFAULT = 32, 32
OUT_DIM = 3


class TargetPINNNS(nn.Module):
    """MLP for 2D NS: (x,y,t) -> (u, v, p).

    Accepts weights dict from a weight generator with keys W1, W2, W3,
    OR trains with external flat weights via the same unpacking.
    """

    def __init__(
        self,
        fourier: str = "tgv",
        fourier_sigma: float = 0.25,
        fourier_seed: int = 42,
        hard_ic: bool = False,
        hidden: tuple[int, int] = (H1_DEFAULT, H2_DEFAULT),
        t_max: float = 1.0,
        n_force: int = 4,
        ic_fn=None,
        n_freqs: int = 32,
        fourier_sigma_wide: float = 1.2,
        orbit_omega: float = 0.0,
    ) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        self.ic_fn = ic_fn
        self.h1, self.h2 = hidden
        self.fourier_mode = fourier
        if fourier == "tgv":
            self.fourier: nn.Module = FourierFeatureMapTGV(t_max=t_max)
            self.in_dim = self.fourier.out_dim
        elif fourier == "kol":
            self.fourier = FourierFeatureMapKolmogorov(n_force=n_force, t_max=t_max)
            self.in_dim = self.fourier.out_dim
        elif fourier == "random":
            self.fourier = FourierFeatureMap(sigma=fourier_sigma, seed=fourier_seed)
            self.in_dim = 6
        elif fourier == "harm":
            k_max = n_freqs if n_freqs >= 2 else 6
            self.fourier = FourierFeatureMapHarmonic(
                k_max=k_max, t_max=t_max, orbit_omega=orbit_omega,
            )
            self.in_dim = self.fourier.out_dim
        elif fourier == "wide":
            self.fourier = FourierFeatureMapWide(
                n_freqs=n_freqs, sigma=fourier_sigma_wide, t_max=t_max, seed=fourier_seed,
            )
            self.in_dim = self.fourier.out_dim
        else:
            raise ValueError(f"unknown fourier={fourier!r}")

    @property
    def w1_size(self) -> int:
        return self.in_dim * self.h1 + self.h1

    @property
    def w2_size(self) -> int:
        return self.h1 * self.h2 + self.h2

    @property
    def w3_size(self) -> int:
        return self.h2 * OUT_DIM + OUT_DIM

    def _unpack(self, weights: dict[str, torch.Tensor]) -> tuple:
        W1_flat = weights["W1"]
        W2_flat = weights["W2"]
        W3_flat = weights["W3"]

        w1 = W1_flat[: self.in_dim * self.h1].reshape(self.h1, self.in_dim)
        b1 = W1_flat[self.in_dim * self.h1 :]

        w2 = W2_flat[: self.h1 * self.h2].reshape(self.h2, self.h1)
        b2 = W2_flat[self.h1 * self.h2 :]

        w3 = W3_flat[: self.h2 * OUT_DIM].reshape(OUT_DIM, self.h2)
        b3 = W3_flat[self.h2 * OUT_DIM :]
        return w1, b1, w2, b2, w3, b3

    def _raw(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        weights: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        coords = torch.stack([x, y, t], dim=-1)
        feats = self.fourier(coords)
        w1, b1, w2, b2, w3, b3 = self._unpack(weights)
        h = torch.tanh(F.linear(feats, w1, b1))
        h = torch.tanh(F.linear(h, w2, b2))
        return F.linear(h, w3, b3)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        weights: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns (batch, 3) tensor of [u, v, p] predictions."""
        raw = self._raw(x, y, t, weights)
        if not self.hard_ic:
            return raw
        if self.ic_fn is not None:
            u0, v0, p0 = self.ic_fn(x, y)
            ic = torch.stack([u0, v0, p0], dim=-1)
            return ic + t.unsqueeze(-1) * raw
        if self.fourier_mode == "kol":
            return t.unsqueeze(-1) * raw
        u0 = torch.sin(x) * torch.cos(y)
        v0 = -torch.cos(x) * torch.sin(y)
        p0 = (torch.cos(2.0 * x) + torch.cos(2.0 * y)) / 4.0
        ic = torch.stack([u0, v0, p0], dim=-1)
        return ic + t.unsqueeze(-1) * raw


# Back-compat aliases used by older quantum path (6-feature random RFF sizes)
IN_DIM = 6
H1, H2 = H1_DEFAULT, H2_DEFAULT
W1_SIZE = IN_DIM * H1 + H1
W2_SIZE = H1 * H2 + H2
W3_SIZE = H2 * OUT_DIM + OUT_DIM


if __name__ == "__main__":
    model = TargetPINNNS(fourier="tgv")
    B = 16
    dummy = {
        "W1": torch.randn(model.w1_size),
        "W2": torch.randn(model.w2_size),
        "W3": torch.randn(model.w3_size),
    }
    x = torch.rand(B); y = torch.rand(B); t = torch.rand(B)
    out = model(x, y, t, dummy)
    assert out.shape == (B, 3)
    print(f"PASS TGV: out={out.shape} W=({model.w1_size},{model.w2_size},{model.w3_size})")

    model_h = TargetPINNNS(fourier="tgv", hard_ic=True)
    out_h = model_h(x, y, torch.zeros(B), dummy)
    u0 = torch.sin(x) * torch.cos(y)
    assert (out_h[:, 0] - u0).abs().max() < 1e-5
    print("PASS hard IC at t=0")
