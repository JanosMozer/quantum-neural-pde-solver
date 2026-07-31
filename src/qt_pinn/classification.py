"""Circuit classification: exact Schmidt rank / entanglement entropy at every contiguous
bipartition, computed directly from the exact statevector via SVD.

Why exact SVD instead of PennyLane's classical_shadow/shadow_expval: those exist to estimate
a state's properties when you can't fully access it (real hardware, limited measurements).
We only ever run on classical simulators, we always have the full statevector, so the exact
computation is both simpler and strictly more rigorous than an estimate would be.
"""

import numpy as np


def schmidt_ranks(psi: np.ndarray, n_qubits: int, tol: float = 1e-8) -> list[int]:
    """Schmidt rank at each of the n_qubits - 1 contiguous bipartitions.

    Rank at cut k = number of non-negligible singular values when psi is reshaped
    into a (2**k, 2**(n_qubits-k)) matrix. This is exactly the MPS bond dimension
    needed for an exact (lossless) representation at that cut.
    """
    psi = np.asarray(psi).reshape(-1)
    assert psi.shape == (2 ** n_qubits,), f"expected {2**n_qubits}-dim state, got {psi.shape}"
    ranks = []
    for cut in range(1, n_qubits):
        mat = psi.reshape(2 ** cut, 2 ** (n_qubits - cut))
        s = np.linalg.svd(mat, compute_uv=False)
        ranks.append(int((s > tol).sum()))
    return ranks


def effective_ranks(psi: np.ndarray, n_qubits: int, fidelity: float = 0.99) -> list[int]:
    """How many leading Schmidt coefficients are needed to capture `fidelity` of the
    state's norm at each cut. This is what actually matters for classical simulability:
    exact Schmidt rank counts every technically-nonzero singular value, which is nearly
    always "full rank" for a generic (even weakly entangled) state, so it hides whether
    a small classical bond dimension already captures almost all of the state.
    """
    psi = np.asarray(psi).reshape(-1)
    assert psi.shape == (2 ** n_qubits,), f"expected {2**n_qubits}-dim state, got {psi.shape}"
    out = []
    for cut in range(1, n_qubits):
        mat = psi.reshape(2 ** cut, 2 ** (n_qubits - cut))
        s = np.linalg.svd(mat, compute_uv=False)
        p = s ** 2
        p = p / p.sum()
        cum = np.cumsum(np.sort(p)[::-1])
        out.append(int(np.searchsorted(cum, fidelity) + 1))
    return out


def entanglement_entropies(psi: np.ndarray, n_qubits: int, tol: float = 1e-12) -> list[float]:
    """Von Neumann entanglement entropy (natural log) at each contiguous bipartition."""
    psi = np.asarray(psi).reshape(-1)
    assert psi.shape == (2 ** n_qubits,), f"expected {2**n_qubits}-dim state, got {psi.shape}"
    entropies = []
    for cut in range(1, n_qubits):
        mat = psi.reshape(2 ** cut, 2 ** (n_qubits - cut))
        s = np.linalg.svd(mat, compute_uv=False)
        p = s ** 2
        p = p[p > tol]
        entropies.append(float(-(p * np.log(p)).sum()))
    return entropies


if __name__ == "__main__":
    # quick manual sanity print, the real assertions live in test_classification.py
    n = 6
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0  # |000000>, a product state
    print("product state ranks:", schmidt_ranks(psi, n))
    print("product state entropies:", entanglement_entropies(psi, n))
