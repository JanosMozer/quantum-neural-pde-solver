"""Option 2: circuit probs, learned Linear(N_STATES, TOTAL_WEIGHTS), MLP weights.

Replaces the fixed random partition and tanh scaling with a trained projection.
Reuses circuit, constants, and device from qt_pinn.qnn_generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

import torch
import torch.nn as nn
from qt_pinn.qnn_generator import (    # reuse circuit + all constants
    _circuit, N_QUBITS, N_STATES, N_LAYERS,
    W1_SIZE, W2_SIZE, W3_SIZE, TOTAL_WEIGHTS,
)


class QuantumWeightGeneratorLP(nn.Module):
    """9-qubit basis probs, learned linear projection, MLP weight dict.

    Trainable parameters:
      q_weights: circuit rotation angles (N_LAYERS x N_QUBITS x 3)
      proj: Linear(N_STATES, TOTAL_WEIGHTS), no fixed partition
    """

    def __init__(self) -> None:
        super().__init__()
        q_shape = (N_LAYERS, N_QUBITS, 3)
        self.q_weights = nn.Parameter(torch.randn(q_shape) * 0.1)
        # bottleneck projection: N_STATES -> 64 -> TOTAL_WEIGHTS
        self.proj = nn.Sequential(
            nn.Linear(N_STATES, 64),
            nn.Tanh(),
            nn.Linear(64, TOTAL_WEIGHTS),
        )

    def forward(self) -> dict[str, torch.Tensor]:
        probs = _circuit(self.q_weights).float()
        flat  = self.proj(probs)
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


if __name__ == "__main__":
    gen    = QuantumWeightGeneratorLP()
    w      = gen()
    n_params = sum(p.numel() for p in gen.parameters())
    assert w["W1"].shape == (W1_SIZE,)
    assert w["W2"].shape == (W2_SIZE,)
    assert w["W3"].shape == (W3_SIZE,)
    print(f"PASS  W1={w['W1'].shape} W2={w['W2'].shape} W3={w['W3'].shape}")
    n_proj = sum(p.numel() for p in gen.proj.parameters())
    print(f"Params: {n_params:,}  (q_weights={gen.q_weights.numel()} + proj={n_proj})")
