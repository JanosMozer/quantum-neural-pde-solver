"""Quantum weight generator for the NS-PINN (3 output fields: u, v, p).

Architecture: circuit probs (2^11=2048 values) -> bottleneck projection -> NS weight dict.

Weight dimensions (32/32/3 MLP with 6 Fourier inputs):
  W1: IN_DIM*H1 + H1 = 6*32 + 32 = 224
  W2: H1*H2 + H2     = 32*32 + 32 = 1056
  W3: H2*OUT_DIM + OUT_DIM = 32*3 + 3 = 99
  TOTAL: 1379  => ceil(log2(1379)) = 11 qubits, 2048 basis states
"""

import torch
import torch.nn as nn
import pennylane as qml

# Architecture — must match pinn_target_ns.py
IN_DIM  = 6
H1, H2  = 32, 32
OUT_DIM = 3

W1_SIZE = IN_DIM * H1 + H1     # 224
W2_SIZE = H1 * H2 + H2         # 1056
W3_SIZE = H2 * OUT_DIM + OUT_DIM  # 99
TOTAL_WEIGHTS = W1_SIZE + W2_SIZE + W3_SIZE  # 1379

N_QUBITS = 11     # ceil(log2(1379)) = 11
N_STATES = 2048   # 2^11
N_LAYERS = 3      # match Burgers config

_dev_ns = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev_ns, interface="torch", diff_method="backprop")
def _circuit_ns(weights: torch.Tensor) -> torch.Tensor:
    """Hadamard init then N_LAYERS StronglyEntanglingLayers -> basis probs."""
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    for l in range(N_LAYERS):
        qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))   # (2^11,)


class QuantumWeightGeneratorNS(nn.Module):
    """11-qubit circuit (2048 probs) -> learned bottleneck -> 1379 MLP weights.

    Trainable:
      q_weights: circuit rotation angles (N_LAYERS, N_QUBITS, 3) = (3, 11, 3) = 99 params
      proj:      Linear(2048, bw) + Tanh + Linear(bw, 1379)
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

    def forward(self) -> dict[str, torch.Tensor]:
        probs = _circuit_ns(self.q_weights).float()
        # q_weights lives on CPU (PennyLane statevector constraint);
        # proj may be on GPU — move probs across the device boundary here.
        probs = probs.to(next(self.proj.parameters()).device)
        flat  = self.proj(probs)
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


class ClassicalWeightGeneratorNS(nn.Module):
    """Classical baseline: same bottleneck architecture, no quantum circuit.

    Input: learnable embedding of size N_STATES (initialized ~ N(0,0.1)).
    Projection: same Linear(N_STATES, bw) -> Tanh -> Linear(bw, TOTAL_WEIGHTS).
    Parameter-matched to QuantumWeightGeneratorNS at same bottleneck_width.
    """

    def __init__(self, bottleneck_width: int = 64) -> None:
        super().__init__()
        # Replaces the quantum circuit: a single trainable vector of the same size
        self.embedding = nn.Parameter(torch.randn(N_STATES) * 0.1)
        self.proj = nn.Sequential(
            nn.Linear(N_STATES, bottleneck_width),
            nn.Tanh(),
            nn.Linear(bottleneck_width, TOTAL_WEIGHTS),
        )

    def forward(self) -> dict[str, torch.Tensor]:
        flat = self.proj(torch.sigmoid(self.embedding))   # sigmoid to keep in [0,1] like probs
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


if __name__ == "__main__":
    gen = QuantumWeightGeneratorNS(bottleneck_width=64)
    w   = gen()
    assert w["W1"].shape == (W1_SIZE,)
    assert w["W2"].shape == (W2_SIZE,)
    assert w["W3"].shape == (W3_SIZE,)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"PASS  W1={w['W1'].shape} W2={w['W2'].shape} W3={w['W3'].shape}")
    print(f"Params: {n_params:,}  (q_weights={gen.q_weights.numel()} + proj)")
