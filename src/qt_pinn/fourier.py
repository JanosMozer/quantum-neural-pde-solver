"""Fourier feature map: (B, 3) -> (B, 6)."""

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


if __name__ == "__main__":
    model = FourierFeatureMap(sigma=1.0)
    x = torch.randn(32, 3)
    out = model(x)
    assert out.shape == (32, 6), f"Expected (32,6), got {out.shape}"
    print(f"PASS: input {x.shape} -> output {out.shape}")
