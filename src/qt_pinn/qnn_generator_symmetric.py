"""Step A4: symmetry-restricted (particle-number-conserving) alternative to
qnn_generator.py's StronglyEntanglingLayers circuit. Same downstream mapping
(fixed random partition + tanh formula, qml.probs readout), only the circuit
internals change: per-qubit RZ (phase only) + a ring of SingleExcitation gates
(exact generator 0.25(X_iY_j)-0.25(Y_iX_j), confirmed from PennyLane directly),
instead of full StronglyEntanglingLayers + CNOT-ring.

Verified before writing this (research/logs/2026-08-01-next-phase-plan.md, step A):
DLA of this generator set is 4n^2-2n (306 at n=9), vs 4^9-1=262143 for the
original, and its QCE (expressibility) does not collapse (0.64-0.83 vs 0.67).
"""

import math
import torch
import torch.nn as nn
import pennylane as qml
from qt_pinn.config_loader import load as _load
from qt_pinn.qnn_generator import (
    TOTAL_WEIGHTS, W1_SIZE, W2_SIZE, W3_SIZE, N_QUBITS, N_STATES, N_PAIRED, N_SINGLE,
)

_cfg = _load()


def weight_shape(n_layers: int) -> tuple:
    return (n_layers, 2, N_QUBITS)  # [:,0,:]=RZ angles, [:,1,:]=SingleExcitation angles


def _make_circuit(n_layers: int):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(weights: torch.Tensor) -> torch.Tensor:
        for w in range(N_QUBITS):
            qml.Hadamard(wires=w)
        for l in range(n_layers):
            for i in range(N_QUBITS):
                qml.RZ(weights[l, 0, i], wires=i)
            for i in range(N_QUBITS):
                qml.SingleExcitation(weights[l, 1, i], wires=[i, (i + 1) % N_QUBITS])
        return qml.probs(wires=range(N_QUBITS))

    return _circuit


class SymmetricQuantumWeightGenerator(nn.Module):
    """Same fixed-partition, fixed-tanh-formula mapping as QuantumWeightGenerator,
    swapping only the circuit that produces the basis probabilities.
    """

    def __init__(self, n_layers: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self._circuit = _make_circuit(n_layers)

        self.q_weights = nn.Parameter(torch.randn(weight_shape(n_layers)) * 0.1)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        perm = torch.randperm(N_STATES)
        self.register_buffer("paired_bases", perm[:2 * N_PAIRED].reshape(N_PAIRED, 2))
        self.register_buffer("single_bases", perm[2 * N_PAIRED:])

        idx = torch.arange(TOTAL_WEIGHTS)
        self.register_buffer("signs", torch.where(idx % 2 == 0,
                                                   torch.ones(TOTAL_WEIGHTS),
                                                   -torch.ones(TOTAL_WEIGHTS)))

    def forward(self, inputs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        probs = self._circuit(self.q_weights).float()
        g = self.gamma

        p_sum = probs[self.paired_bases].sum(dim=1)
        w_paired = g * torch.tanh(self.signs[:N_PAIRED] * (2 ** (N_QUBITS - 2)) * g * p_sum)

        p_s = probs[self.single_bases]
        w_single = g * torch.tanh(self.signs[N_PAIRED:] * (2 ** (N_QUBITS - 1)) * g * p_s)

        flat = torch.cat([w_paired, w_single])
        return {"W1": flat[:W1_SIZE],
                "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
                "W3": flat[W1_SIZE + W2_SIZE:]}


if __name__ == "__main__":
    for L in (5, 17):
        gen = SymmetricQuantumWeightGenerator(n_layers=L, seed=0)
        w = gen()
        n_params = sum(p.numel() for p in gen.parameters())
        assert w["W1"].shape[0] + w["W2"].shape[0] + w["W3"].shape[0] == TOTAL_WEIGHTS
        loss = sum(v.sum() for v in w.values())
        loss.backward()
        assert gen.q_weights.grad is not None and gen.q_weights.grad.norm().item() > 1e-6
        print(f"PASS L={L}: n_params={n_params}, output shapes ok, gradient flows")
