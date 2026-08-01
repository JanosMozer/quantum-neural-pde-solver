"""Option 2: 9-qubit probs → learned Linear(512, 418) → MLP weights.

Replaces the fixed random partition + tanh scaling with a trained projection.
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
    """9-qubit basis probs → learned linear projection → MLP weight dict.

    Trainable parameters:
      q_weights : circuit rotation angles  (N_LAYERS × N_QUBITS × 3)
      proj      : Linear(N_STATES, TOTAL_WEIGHTS)  — no fixed partition
    """

    def __init__(self, bottleneck_width: int = 64) -> None:
        super().__init__()
        q_shape = (N_LAYERS, N_QUBITS, 3)
        self.q_weights = nn.Parameter(torch.randn(q_shape) * 0.1)
        self.proj = nn.Sequential(
            nn.Linear(N_STATES, bottleneck_width),
            nn.Tanh(),
            nn.Linear(bottleneck_width, TOTAL_WEIGHTS),
        )

    def forward(self, inputs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if inputs is None:
            inputs = torch.zeros(N_QUBITS)
        probs = _circuit(inputs, self.q_weights).float()   # (N_STATES,)
        flat  = self.proj(probs)                            # (TOTAL_WEIGHTS,)
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
    print(f"Params: {n_params:,}  "
          f"(q_weights={gen.q_weights.numel()} + proj={gen.proj.weight.numel()+gen.proj.bias.numel()})")
