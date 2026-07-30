"""Quantum Weight Generator using PennyLane StronglyEntanglingLayers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import pennylane as qml
from training.config_loader import load as _load

_cfg = _load()
_q   = _cfg["quantum"]
_m   = _cfg["mlp"]

# ── Circuit ───────────────────────────────────────────────────────────────────
N_QUBITS = _q["n_qubits"]   # circuit width; one Pauli-Z readout per qubit
N_LAYERS = _q["n_layers"]   # StronglyEntanglingLayers depth

# ── MLP dims (must match pinn_target.py) ─────────────────────────────────────
IN_DIM  = N_QUBITS            # Fourier feature dim = 2 * (N_QUBITS/2) = N_QUBITS
H1, H2  = _m["hidden"]        # hidden layer widths
OUT_DIM = 2                   # u and v

# ── Flat weight sizes (W + b per layer) ──────────────────────────────────────
W1_SIZE = IN_DIM * H1 + H1        # e.g. 6×16 + 16 = 112
W2_SIZE = H1 * H2   + H2          # e.g. 16×16 + 16 = 272
W3_SIZE = H2 * OUT_DIM + OUT_DIM  # e.g. 16×2  + 2  = 34
TOTAL_WEIGHTS = W1_SIZE + W2_SIZE + W3_SIZE

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
    """StronglyEntanglingLayers circuit returning Pauli-Z expectations."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumWeightGenerator(nn.Module):
    """Generates MLP weights via a quantum circuit + linear projection.

    Input: scalar 'step' encoded as angle-embedded 6-vector.
    Output: dict with flattened W1, W2, W3 tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable quantum circuit weights
        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=N_LAYERS, n_wires=N_QUBITS
        )
        self.q_weights = nn.Parameter(
            torch.randn(weight_shape) * 0.1
        )
        # Linear projection: 6 Pauli-Z values -> total MLP weights
        self.proj = nn.Linear(N_QUBITS, TOTAL_WEIGHTS)

    def forward(self, inputs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """inputs: (6,) tensor of angles, or None to use zeros."""
        if inputs is None:
            inputs = torch.zeros(N_QUBITS)
        # quantum_circuit returns list of scalars -> stack to (6,)
        z_vals = torch.stack(quantum_circuit(inputs, self.q_weights)).float()  # (6,)
        flat = self.proj(z_vals)  # (TOTAL_WEIGHTS,)

        w1 = flat[:W1_SIZE]
        w2 = flat[W1_SIZE: W1_SIZE + W2_SIZE]
        w3 = flat[W1_SIZE + W2_SIZE:]
        return {"W1": w1, "W2": w2, "W3": w3}


if __name__ == "__main__":
    gen = QuantumWeightGenerator()
    out = gen()
    print(f"W1: {out['W1'].shape} (expected {W1_SIZE})")
    print(f"W2: {out['W2'].shape} (expected {W2_SIZE})")
    print(f"W3: {out['W3'].shape} (expected {W3_SIZE})")
    assert out["W1"].shape == (W1_SIZE,)
    assert out["W2"].shape == (W2_SIZE,)
    assert out["W3"].shape == (W3_SIZE,)
    print("PASS: all weight shapes correct")
