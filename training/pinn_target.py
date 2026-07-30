"""Target PINN: classical MLP that accepts externally-generated weights."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.fourier import FourierFeatureMap

IN_DIM = 6
H1 = 16
H2 = 16
OUT_DIM = 2


class TargetPINN(nn.Module):
    """MLP for 2D Burgers: (x,y,t) -> (u, v).

    Accepts weights dict from QuantumWeightGenerator (or any static dict).
    Weights dict keys: W1 (112,), W2 (272,), W3 (34,)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fourier = FourierFeatureMap()

    def _unpack(self, weights: dict[str, torch.Tensor]) -> tuple:
        W1_flat = weights["W1"]  # 112 = 6*16 + 16
        W2_flat = weights["W2"]  # 272 = 16*16 + 16
        W3_flat = weights["W3"]  # 34  = 16*2 + 2

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

        h = F.tanh(F.linear(feats, w1, b1))  # (batch, 16)
        h = F.tanh(F.linear(h, w2, b2))       # (batch, 16)
        out = F.linear(h, w3, b3)             # (batch, 2)
        return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.qnn_generator import QuantumWeightGenerator

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
