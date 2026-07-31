"""Step A3: does the number-conserving ansatz's expressibility (QCE) collapse
relative to the original, before spending any training compute on it?
"""

import torch
import pennylane as qml

from qt_pinn.diagnostics import qce

N_QUBITS = 9
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def original_state(weights):
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.state()


@qml.qnode(dev, interface="torch")
def symmetric_state(weights):
    """weights shape (L, 2, N_QUBITS): [:,0,:] = RZ angles, [:,1,:] = SingleExcitation
    angles on the ring (i, i+1 mod N_QUBITS)."""
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(N_QUBITS):
            qml.RZ(weights[l, 0, i], wires=i)
        for i in range(N_QUBITS):
            qml.SingleExcitation(weights[l, 1, i], wires=[i, (i + 1) % N_QUBITS])
    return qml.state()


if __name__ == "__main__":
    L_ORIGINAL = 3
    orig_shape = qml.StronglyEntanglingLayers.shape(n_layers=L_ORIGINAL, n_wires=N_QUBITS)
    q_orig = qce(original_state, orig_shape, n_samples=50)
    print(f"original ansatz (L={L_ORIGINAL}, {orig_shape[0]*orig_shape[1]*orig_shape[2]} params) "
          f"QCE = {q_orig:.4f}")

    for L_SYM in (5, 17):
        sym_shape = (L_SYM, 2, N_QUBITS)
        q_sym = qce(symmetric_state, sym_shape, n_samples=50)
        n_params = L_SYM * 2 * N_QUBITS
        print(f"symmetric ansatz (L={L_SYM}, {n_params} params) QCE = {q_sym:.4f}")
