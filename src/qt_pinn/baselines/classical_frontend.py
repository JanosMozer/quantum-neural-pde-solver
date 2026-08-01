"""Classical baseline 3: matched-budget classical frontend replacing the quantum circuit.

Same 99-param budget as the circuit (N_LAYERS*N_QUBITS*3), same softmax normalization to
mimic qml.probs() (non-negative, sums to 1), and the identical proj bottleneck used by
QuantumWeightGeneratorLP. Isolates whether the circuit itself contributes anything beyond
a classical 99-param generator feeding the same downstream architecture.
"""

import torch
import torch.nn as nn
from qt_pinn.qnn_generator import N_LAYERS, N_QUBITS, N_STATES, TOTAL_WEIGHTS, W1_SIZE, W2_SIZE, W3_SIZE
from qt_pinn.baselines.low_rank import LowRankGenerator

N_CIRCUIT_PARAMS = N_LAYERS * N_QUBITS * 3  # 99, matches the quantum circuit exactly


class ClassicalFrontendGeneratorLP(nn.Module):
    """frontend (99 params, low-rank) -> softmax -> proj (identical to QuantumWeightGeneratorLP)."""

    def __init__(self, bottleneck_width: int = 64) -> None:
        super().__init__()
        self.frontend = LowRankGenerator(out_dim=N_STATES, target_param_count=N_CIRCUIT_PARAMS)
        self.proj = nn.Sequential(
            nn.Linear(N_STATES, bottleneck_width),
            nn.Tanh(),
            nn.Linear(bottleneck_width, TOTAL_WEIGHTS),
        )

    def forward(self, inputs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        probs = torch.softmax(self.frontend(), dim=0)
        flat = self.proj(probs)
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


if __name__ == "__main__":
    gen = ClassicalFrontendGeneratorLP()
    n_frontend = sum(p.numel() for p in gen.frontend.parameters())
    assert n_frontend == N_CIRCUIT_PARAMS, f"expected {N_CIRCUIT_PARAMS}, got {n_frontend}"
    w = gen()
    assert w["W1"].shape == (W1_SIZE,) and w["W2"].shape == (W2_SIZE,) and w["W3"].shape == (W3_SIZE,)
    loss = sum(v.sum() for v in w.values())
    loss.backward()
    assert gen.frontend.coeffs.grad is not None and gen.frontend.coeffs.grad.norm() > 0
    print(f"PASS  frontend_params={n_frontend}  proj_params={sum(p.numel() for p in gen.proj.parameters())}")
