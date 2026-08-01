"""Target PINN: classical MLP that accepts externally-generated weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from qt_pinn.fourier import FourierFeatureMap
from qt_pinn.config_loader import load as _load

_m = _load()["mlp"]

IN_DIM = 6
H1, H2 = _m["hidden"]
OUT_DIM = 2
OMEGA_0 = _m.get("omega_0", 30.0)  # SIREN periodic activation frequency, Sitzmann et al. 2020


class TargetPINN(nn.Module):
    """MLP for 2D Burgers: (x,y,t) -> (u, v).

    Accepts weights dict from QuantumWeightGenerator (or any static dict).
    Weights dict keys: W1 (IN_DIM*H1+H1,), W2 (H1*H2+H2,), W3 (H2*OUT_DIM+OUT_DIM,)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fourier = FourierFeatureMap()

    def _unpack(self, weights: dict[str, torch.Tensor]) -> tuple:
        W1_flat = weights["W1"]  # IN_DIM*H1 + H1
        W2_flat = weights["W2"]  # H1*H2 + H2
        W3_flat = weights["W3"]  # H2*OUT_DIM + OUT_DIM

        w1 = W1_flat[:IN_DIM * H1].reshape(H1, IN_DIM)
        b1 = W1_flat[IN_DIM * H1:]

        w2 = W2_flat[:H1 * H2].reshape(H2, H1)
        b2 = W2_flat[H1 * H2:]

        w3 = W3_flat[:H2 * OUT_DIM].reshape(OUT_DIM, H2)
        b3 = W3_flat[H2 * OUT_DIM:]
        return w1, b1, w2, b2, w3, b3

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        weights: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns (batch, 2) tensor of [u, v] predictions."""
        coords = torch.stack([x, y, t], dim=-1)  # (batch, 3)
        feats = self.fourier(coords)              # (batch, 6)

        w1, b1, w2, b2, w3, b3 = self._unpack(weights)

        # SIREN periodic activations (Sitzmann et al. 2020) instead of tanh — the
        # generated weights already play the role of SIREN's learned W, b; only the
        # nonlinearity changes. Output layer stays linear, as in the paper.
        h = torch.sin(OMEGA_0 * F.linear(feats, w1, b1))  # (batch, H1)
        h = torch.sin(OMEGA_0 * F.linear(h, w2, b2))       # (batch, H2)
        out = F.linear(h, w3, b3)                          # (batch, 2)
        return out


if __name__ == "__main__":
    from qt_pinn.qnn_generator import QuantumWeightGenerator

    model = TargetPINN()
    gen = QuantumWeightGenerator()
    weights = gen()

    B = 16
    x = torch.rand(B)
    y = torch.rand(B)
    t = torch.rand(B)

    out = model(x, y, t, weights)
    assert out.shape == (B, 2), f"Expected ({B}, 2), got {out.shape}"
    print(f"PASS: output shape {out.shape}, u range [{out[:,0].min():.3f}, {out[:,0].max():.3f}]")
