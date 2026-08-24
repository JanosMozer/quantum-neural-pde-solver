"""Pre-flight: 1D regression sin(4x)+sin(8x) with VQC vs matched classical.

If quantum cannot win here (Fourier-native target), it will not win on Burgers.

Run:
  .venv/bin/python scripts/sanity_vqc_regression.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from qt_pinn.burgers_vqc_pinn import (
    BurgersVQCPINN,
    BurgersClassicalPINN,
    count_params,
)


def target(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(4 * x) + torch.sin(8 * x)


class RegVQCPINN(BurgersVQCPINN):
    """Reuse Burgers VQC with y=t=0, scalar output."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        t = torch.zeros_like(x)
        return super().forward(x, y, t)[:, 0:1]


class RegClassicalPINN(BurgersClassicalPINN):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        t = torch.zeros_like(x)
        return super().forward(x, y, t)[:, 0:1]


def train(model, x, y, steps=1500, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        opt.step()
    return loss.item()


def main():
    torch.manual_seed(0)
    n = 256
    x = torch.linspace(-math.pi, math.pi, n).requires_grad_(False)
    y = target(x).reshape(-1, 1)

    q = RegVQCPINN(6, 4)
    c = RegClassicalPINN.matched_to(q)
    print(f"params  quantum={count_params(q)}  classical={count_params(c)}")

    lq = train(q, x, y, lr=0.005)
    lc = train(c, x, y, lr=0.01)
    with torch.no_grad():
        rq = (q(x) - y).pow(2).mean().sqrt().item()
        rc = (c(x) - y).pow(2).mean().sqrt().item()
    print(f"final RMSE  quantum={rq:.5f}  classical={rc:.5f}  Q/C={rq/max(rc,1e-12):.2f}")
    if rq < rc * 0.95:
        print("PASS: quantum better on Fourier-native regression")
    elif rq <= rc * 1.05:
        print("NOTE: parity on regression sanity check")
    else:
        print("FAIL: classical better — VQC expressivity may be insufficient")


if __name__ == "__main__":
    main()
