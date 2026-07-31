"""Gate 1: analytic ground-truth self-tests for classification.py.

Two known-answer cases, checked before this code is trusted on anything real:
  - a product state (independent single-qubit rotations, no entangling gates):
    entropy ~ 0 at every cut, Schmidt rank 1.
  - a GHZ state (H then a chain of CNOTs): entropy = ln(2) at every cut, rank exactly 2.

Plain assert-based self-test, no pytest, matching the project's existing convention.
"""

import numpy as np
import pennylane as qml

from qt_pinn.classification import schmidt_ranks, entanglement_entropies, effective_ranks

N = 6
dev = qml.device("default.qubit", wires=N)


@qml.qnode(dev)
def product_state():
    for w in range(N):
        qml.RY(0.37 + 0.1 * w, wires=w)  # arbitrary distinct angles, no entangling gates
    return qml.state()


@qml.qnode(dev)
def ghz_state():
    qml.Hadamard(wires=0)
    for w in range(N - 1):
        qml.CNOT(wires=[w, w + 1])
    return qml.state()


def test_product_state() -> None:
    psi = product_state()
    ranks = schmidt_ranks(psi, N)
    entropies = entanglement_entropies(psi, N)
    eff = effective_ranks(psi, N)
    assert all(r == 1 for r in ranks), f"expected all ranks 1, got {ranks}"
    assert all(e < 1e-8 for e in entropies), f"expected ~0 entropy, got {entropies}"
    assert all(r == 1 for r in eff), f"expected all effective ranks 1, got {eff}"
    print(f"PASS product state: ranks={ranks} entropies={[round(e, 10) for e in entropies]} eff={eff}")


def test_ghz_state() -> None:
    psi = ghz_state()
    ranks = schmidt_ranks(psi, N)
    entropies = entanglement_entropies(psi, N)
    eff = effective_ranks(psi, N)
    assert all(r == 2 for r in ranks), f"expected all ranks 2, got {ranks}"
    expected = np.log(2)
    assert all(abs(e - expected) < 1e-8 for e in entropies), (
        f"expected all entropies {expected}, got {entropies}"
    )
    assert all(r == 2 for r in eff), f"expected all effective ranks 2 (equal split), got {eff}"
    print(f"PASS GHZ state: ranks={ranks} entropies={[round(e, 10) for e in entropies]} eff={eff}")


if __name__ == "__main__":
    test_product_state()
    test_ghz_state()
    print("PASS: all Gate 1 analytic self-tests passed")
