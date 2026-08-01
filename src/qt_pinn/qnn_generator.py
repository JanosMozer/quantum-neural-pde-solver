"""Quantum weight generator: basis-probability hypernetwork.

Architecture:
  N = ceil(log2(M)) qubits, 2^N basis-state probabilities (qml.probs)
  fixed random partition maps the 2^N probabilities to M weights directly
  scalar gamma is the only additional trainable parameter

N_QUBITS/N_STATES/TOTAL_WEIGHTS below are computed from config.yaml's mlp.hidden,
not fixed numbers, so they track whatever MLP size is configured.
"""

import math
import torch
import torch.nn as nn
import pennylane as qml
from qt_pinn.config_loader import load as _load

_cfg = _load()
_m   = _cfg["mlp"]
_q   = _cfg["quantum"]

# MLP dims, must match pinn_target.py
H1, H2  = _m["hidden"]
OUT_DIM = 2
IN_DIM  = 6   # Fourier features: 2 x n_freqs = 6, independent of N_QUBITS

W1_SIZE = IN_DIM * H1 + H1
W2_SIZE = H1 * H2   + H2
W3_SIZE = H2 * OUT_DIM + OUT_DIM
TOTAL_WEIGHTS = W1_SIZE + W2_SIZE + W3_SIZE

# qubit count: N = ceil(log2(M)), so 2^N >= M
N_QUBITS = math.ceil(math.log2(TOTAL_WEIGHTS))
N_STATES = 2 ** N_QUBITS
N_LAYERS = _q["n_layers"]  # circuit depth

# partition counts, fixed by M and N
N_PAIRED = N_STATES - TOTAL_WEIGHTS      # 2 basis probs map to 1 weight
N_SINGLE = 2 * TOTAL_WEIGHTS - N_STATES  # 1 basis prob maps to 1 weight
assert 2 * N_PAIRED + N_SINGLE == N_STATES, "partition check failed"

_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev, interface="torch", diff_method="backprop")
def _circuit(weights: torch.Tensor) -> torch.Tensor:
    """Hadamard init (uniform superposition) then N_LAYERS entangling layers."""
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    for l in range(N_LAYERS):
        qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))  # (2^N,) in [0,1], sums to 1


class QuantumWeightGenerator(nn.Module):
    """Maps quantum basis probabilities to an MLP weight dict via a fixed partition.

    Trainable: q_weights (circuit angles) + gamma (1 scalar). No dense layer.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)

        # learnable: circuit angles + scale scalar
        q_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
        self.q_weights = nn.Parameter(torch.randn(q_shape) * 0.1)  # gate angles
        self.gamma     = nn.Parameter(torch.tensor(1.0))          # weight magnitude

        # fixed: random basis assignment, never updated
        perm = torch.randperm(N_STATES)
        self.register_buffer("paired_bases", perm[:2 * N_PAIRED].reshape(N_PAIRED, 2))
        self.register_buffer("single_bases", perm[2 * N_PAIRED:])

        # fixed sign pattern: even index +1, odd index -1
        idx = torch.arange(TOTAL_WEIGHTS)
        self.register_buffer("signs", torch.where(idx % 2 == 0,
                                                   torch.ones(TOTAL_WEIGHTS),
                                                   -torch.ones(TOTAL_WEIGHTS)))

    def forward(self) -> dict[str, torch.Tensor]:
        probs = _circuit(self.q_weights).float()
        g     = self.gamma

        # paired weights: theta = gamma * tanh[sign * 2^(N-2) * gamma * (p_a + p_b)]
        p_sum    = probs[self.paired_bases].sum(dim=1)
        w_paired = g * torch.tanh(self.signs[:N_PAIRED] * (2**(N_QUBITS-2)) * g * p_sum)

        # single weights: theta = gamma * tanh[sign * 2^(N-1) * gamma * p_i]
        p_s      = probs[self.single_bases]
        w_single = g * torch.tanh(self.signs[N_PAIRED:] * (2**(N_QUBITS-1)) * g * p_s)

        flat = torch.cat([w_paired, w_single])
        return {"W1": flat[:W1_SIZE],
                "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
                "W3": flat[W1_SIZE + W2_SIZE:]}
