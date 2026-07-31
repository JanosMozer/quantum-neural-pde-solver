"""Classify the actual QT weight generator's circuit (imported directly, not reimplemented).

Reuses qt_pinn.qnn_generator's constants (N_QUBITS, N_LAYERS, device) as the single source
of truth for the ansatz. We only add a second QNode on the same device/ansatz that returns
qml.state() instead of qml.probs(), since entanglement entropy needs the full complex
amplitudes (relative phases), not just measurement probabilities.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import pennylane as qml

from qt_pinn.qnn_generator import N_QUBITS, N_LAYERS, _dev
from qt_pinn.classification import schmidt_ranks, entanglement_entropies, effective_ranks


@qml.qnode(_dev, interface="torch")
def state_circuit(weights: torch.Tensor):
    """Mirrors qnn_generator._circuit's gate sequence exactly, swapping probs for state."""
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    zeros = torch.zeros(N_QUBITS)
    for l in range(N_LAYERS):
        qml.AngleEmbedding(zeros, wires=range(N_QUBITS), rotation="Y")  # no-op: inputs are always 0
        qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(N_QUBITS))
    return qml.state()


def classify(weights: torch.Tensor, label: str) -> None:
    psi = state_circuit(weights).detach().numpy()
    ranks = schmidt_ranks(psi, N_QUBITS)
    entropies = entanglement_entropies(psi, N_QUBITS)
    eff = effective_ranks(psi, N_QUBITS, fidelity=0.99)
    max_possible = 2 ** (N_QUBITS // 2)
    max_entropy = np.log(max_possible)
    print(f"\n[{label}]  N_QUBITS={N_QUBITS}  N_LAYERS={N_LAYERS}")
    print(f"  Schmidt ranks per cut       : {ranks}  (max possible: {max_possible})")
    print(f"  entropies (nats)            : {[round(e, 4) for e in entropies]}  (max possible: {max_entropy:.4f})")
    print(f"  effective rank @99% fidelity: {eff}  (max possible: {max_possible})")
    print(f"  entropy as % of maximum     : {[round(100 * e / max_entropy, 1) for e in entropies]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="path to a q_weights.pt to classify, e.g. checkpoints/run_0035/q_weights.pt")
    args = parser.parse_args()

    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)

    # (a) fresh random init, matching QuantumWeightGenerator's own init scheme
    torch.manual_seed(0)
    random_weights = torch.randn(weight_shape) * 0.1
    classify(random_weights, "random init (seed 0)")

    torch.manual_seed(0)
    random_weights_repeat = torch.randn(weight_shape) * 0.1
    psi1 = state_circuit(random_weights).detach().numpy()
    psi2 = state_circuit(random_weights_repeat).detach().numpy()
    assert np.allclose(psi1, psi2), "classification is not deterministic given fixed seed"
    print("\nPASS: deterministic given fixed seed")

    # (b) an actual trained checkpoint, if one is given
    if args.ckpt:
        state_dict = torch.load(Path(args.ckpt), map_location="cpu", weights_only=True)
        classify(state_dict["q_weights"], f"trained ({args.ckpt})")
    else:
        print("\nNo --ckpt given, skipping trained-circuit classification. "
              "Run scripts/train.py first, then pass --ckpt checkpoints/run_NNNN/q_weights.pt")


if __name__ == "__main__":
    main()
