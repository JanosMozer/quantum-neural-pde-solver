"""Reusable diagnostics for the scaling-laws plan (step 4c): DLA dimension, QCE, QGN.

QCE and QGN definitions are exact, quoted formulas from Vyskubov, Vyskubov, Innan,
Shafique, "Scaling Laws for Hybrid Quantum Neural Networks" (arXiv:2604.06007):
  QCE = 1 - (1/(N(N-1))) * sum_{i<j} |<psi_i|psi_j>|^2   (average pairwise fidelity)
  QGN = ||grad_{theta_q} L||_2

EEE (their third diagnostic, entanglement entropy of the reduced density matrix on
the first floor(Q/2) qubits) is NOT duplicated here: qt_pinn.classification already
computes this (entanglement_entropies), used by scripts/classify_circuit.py.

DLA dimension is only computed for small n (verified methodology in
archive/experiments/dla/compute_dla.py); at n=9 it is intractable to compute directly, so
`dla_dimension` raises rather than hang, and callers should use the verified
extrapolation (4^n - 1 for this circuit family) documented in
research/logs/2026-07-31-scaling-laws-plan.md instead.
"""

import numpy as np
import torch
import pennylane as qml
from pennylane import Y, Z


def cnot_ring_unitary(n: int) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def _circuit():
        for i in range(n):
            qml.CNOT(wires=[i, (i + 1) % n])
        return qml.state()

    return qml.matrix(_circuit)()


def dla_dimension(n_qubits: int, rounds: int = 2, max_n: int = 6) -> int:
    """Full-rank-verified methodology from archive/experiments/dla/compute_dla.py. Only
    tractable for small n; raises above max_n rather than hang.
    """
    if n_qubits > max_n:
        raise ValueError(
            f"dla_dimension is only tractable up to n={max_n} qubits (dense closure "
            f"over up to 4^n-1 Pauli terms). For n={n_qubits}, use the verified "
            f"extrapolation 4^n-1 from research/logs/2026-07-31-scaling-laws-plan.md "
            f"instead of computing directly."
        )
    U = cnot_ring_unitary(n_qubits)
    local = [qml.matrix(Y(i), wire_order=list(range(n_qubits))) for i in range(n_qubits)]
    local += [qml.matrix(Z(i), wire_order=list(range(n_qubits))) for i in range(n_qubits)]

    gens = [Y(i) for i in range(n_qubits)] + [Z(i) for i in range(n_qubits)]
    current = local
    for _ in range(rounds):
        conjugated = [U @ g @ U.conj().T for g in current]
        for mat in conjugated:
            gens.append(qml.pauli_decompose(mat, wire_order=list(range(n_qubits))))
        current = conjugated
    return len(qml.lie_closure(gens))


def qce(state_fn, weight_shape: tuple, n_samples: int = 50, seed: int = 0) -> float:
    """Expressibility: 1 - average pairwise fidelity of statevectors from n_samples
    independent random parameter draws through state_fn(weights) -> statevector.
    """
    gen = torch.Generator().manual_seed(seed)
    states = []
    for _ in range(n_samples):
        w = torch.randn(weight_shape, generator=gen) * 0.1
        states.append(np.asarray(state_fn(w).detach()))

    fidelities = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            overlap = np.vdot(states[i], states[j])
            fidelities.append(abs(overlap) ** 2)
    n = n_samples
    return 1.0 - (1.0 / (n * (n - 1))) * sum(fidelities)


def qgn(params) -> float:
    """L2 norm of gradients on an iterable of torch parameters (call after .backward())."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


if __name__ == "__main__":
    for n in (3, 4):
        dim = dla_dimension(n)
        assert dim == 4 ** n - 1, f"n={n}: expected {4**n-1}, got {dim}"
    print("PASS: dla_dimension matches 4^n-1 at n=3,4")

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev, interface="torch")
    def _toy_state(weights):
        qml.StronglyEntanglingLayers(weights, wires=range(3))
        return qml.state()

    shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
    q = qce(_toy_state, shape, n_samples=20)
    assert 0.0 <= q <= 1.0, f"QCE out of [0,1]: {q}"
    print(f"PASS: qce in range, value={q:.4f}")

    x = torch.nn.Parameter(torch.randn(5))
    loss = (x ** 2).sum()
    loss.backward()
    g = qgn([x])
    expected = x.grad.norm().item()
    assert abs(g - expected) < 1e-9, f"qgn mismatch: {g} vs {expected}"
    print(f"PASS: qgn matches direct norm, value={g:.4f}")
