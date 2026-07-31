"""Classify the actual QT weight generator's circuit (imported directly, not reimplemented).

Reuses qt_pinn.qnn_generator as the single source of truth for the ansatz: N qubits (see
config.yaml), StronglyEntanglingLayers, range-based CNOT connectivity. We only add a second
QNode on the same device/ansatz that returns qml.state() instead of the per-qubit Z
expectations the training code uses.
"""

from pathlib import Path

import numpy as np
import torch
import pennylane as qml

from qt_pinn.qnn_generator import N_QUBITS, N_LAYERS, dev
from qt_pinn.classification import schmidt_ranks, entanglement_entropies, effective_ranks


@qml.qnode(dev, interface="torch")
def state_circuit(weights: torch.Tensor):
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.state()


def classify(weights: torch.Tensor, label: str) -> None:
    psi = state_circuit(weights).detach().numpy()
    ranks = schmidt_ranks(psi, N_QUBITS)
    entropies = entanglement_entropies(psi, N_QUBITS)
    eff = effective_ranks(psi, N_QUBITS, fidelity=0.99)
    max_possible = 2 ** (N_QUBITS // 2)
    max_entropy = np.log(max_possible)
    print(f"\n[{label}]")
    print(f"  Schmidt ranks per cut       : {ranks}  (max possible: {max_possible})")
    print(f"  entropies (nats)            : {[round(e, 4) for e in entropies]}  (max possible: {max_entropy:.4f})")
    print(f"  effective rank @99% fidelity: {eff}  (max possible: {max_possible})")
    print(f"  entropy as % of maximum     : {[round(100*e/max_entropy, 1) for e in entropies]}")


def main() -> None:
    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)

    # (a) fresh random init, Janos's own init scheme
    torch.manual_seed(0)
    random_weights = torch.randn(weight_shape) * 0.1
    classify(random_weights, "random init (seed 0)")

    # repeat with the same seed to confirm determinism (Gate 2 pass criterion)
    torch.manual_seed(0)
    random_weights_repeat = torch.randn(weight_shape) * 0.1
    psi1 = state_circuit(random_weights).detach().numpy()
    psi2 = state_circuit(random_weights_repeat).detach().numpy()
    assert np.allclose(psi1, psi2), "classification is not deterministic given fixed seed"
    print("\nPASS: deterministic given fixed seed")

    # (b) the actual trained checkpoint (run from the repo root)
    ckpt_path = Path("checkpoints") / "run_0004" / "q_weights.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    trained_weights = state_dict["q_weights"]
    classify(trained_weights, "trained (run_0004)")


if __name__ == "__main__":
    main()
