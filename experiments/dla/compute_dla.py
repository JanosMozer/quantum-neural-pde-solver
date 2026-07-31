"""Step 1 of the scaling-laws plan: is Option 1's 81 circuit parameters above or
below its dynamical Lie algebra (DLA) dimension (Larocca et al.'s overparametrization
threshold M_c, upper-bounded by dim(DLA))?

Two-stage verification, not a single trusted number:
  1. Reproduce the known, PROVEN result from Larocca, Sauvage, Sbahi, Verdon, Coles,
     Cerezo, "Diagnosing barren plateaus with tools from quantum optimal control"
     (arXiv:2105.14377), Proposition (App. proof_of_hea_sg): the generator set
     G_HEA = {X_i, Y_i}_i union {sum_i Z_i Z_{i+1}} gives a full-rank DLA of
     dimension 4^n - 1. This validates the lie_closure methodology itself against a
     published, proven ground truth before trusting it on our own circuit.
  2. Apply the SAME methodology to our actual circuit's literal gate structure:
     per-qubit Rot = RZ.RY.RZ (generators Z, Y) from StronglyEntanglingLayers, plus
     the real fixed CNOT-ring entangler (range=1, periodic boundary, confirmed from
     PennyLane's own compute_decomposition source), conjugated 1-2 rounds. Confirm
     the SAME 4^n-1 scaling holds for our literal gates, not just the paper's
     abstracted representative generator.

n=9 (our real qubit count) is not attempted directly: 4^9-1 = 262143 dimensional
closure is intractable to compute densely on this hardware. The small-n match to a
proven, exponentially-scaling theorem is the evidence for the n=9 extrapolation, not
a numerical run at n=9 itself.
"""

import numpy as np
import pennylane as qml
from pennylane import X, Y, Z


def paper_hea_generators(n: int) -> list:
    """G_HEA from Larocca et al. arXiv:2105.14377, Proposition (proof_of_hea_sg)."""
    gens = []
    for i in range(n):
        gens.append(X(i))
        gens.append(Y(i))
    zz = sum(Z(i) @ Z(i + 1) for i in range(n - 1))
    gens.append(zz)
    return gens


def cnot_ring_unitary(n: int) -> np.ndarray:
    """Exact matrix for one StronglyEntanglingLayers entangling block, range=1,
    periodic boundary: CNOT(i, i+1 mod n) for i in range(n), matching
    StronglyEntanglingLayers.compute_decomposition's actual op sequence.
    """
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def _circuit():
        for i in range(n):
            qml.CNOT(wires=[i, (i + 1) % n])
        return qml.state()

    return qml.matrix(_circuit)()


def conjugate_to_pauli_sentence(mat: np.ndarray, n: int):
    """U G U^dagger, decomposed back into a PennyLane Pauli-sum operator."""
    return qml.pauli_decompose(mat, wire_order=list(range(n)), pauli=False)


def our_circuit_generators(n: int, rounds: int = 2) -> list:
    """Real gate-level generators: per-qubit {Y_i, Z_i} (from Rot=RZ.RY.RZ) plus
    `rounds` conjugated copies through the actual fixed CNOT-ring unitary, matching
    what a periodic StronglyEntanglingLayers ansatz actually reaches across layers.
    """
    U = cnot_ring_unitary(n)
    local = []
    for i in range(n):
        local.append(qml.matrix(Y(i), wire_order=list(range(n))))
        local.append(qml.matrix(Z(i), wire_order=list(range(n))))

    gens = [Y(i) for i in range(n)] + [Z(i) for i in range(n)]
    current = local
    for _ in range(rounds):
        conjugated = [U @ g @ U.conj().T for g in current]
        for mat in conjugated:
            gens.append(conjugate_to_pauli_sentence(mat, n))
        current = conjugated
    return gens


def run(n: int) -> dict:
    full_rank = 4 ** n - 1

    paper_dla = qml.lie_closure(paper_hea_generators(n))
    ours_dla = qml.lie_closure(our_circuit_generators(n))

    return {
        "n": n,
        "full_rank_target": full_rank,
        "paper_generators_dim": len(paper_dla),
        "our_generators_dim": len(ours_dla),
        "paper_matches_full_rank": len(paper_dla) == full_rank,
        "ours_matches_full_rank": len(ours_dla) == full_rank,
    }


if __name__ == "__main__":
    for n in (3, 4, 5):
        result = run(n)
        print(result)
        assert result["paper_matches_full_rank"], (
            f"n={n}: paper's own proven generator set did NOT reach 4^n-1, "
            f"lie_closure methodology itself is suspect, stop before trusting n=9"
        )
        assert result["ours_matches_full_rank"], (
            f"n={n}: our circuit's literal gate generators did NOT reach 4^n-1, "
            f"our ansatz may be less than full rank, do not extrapolate to n=9"
        )
    print("PASS: both generator sets reach full rank 4^n-1 at all tested n.")
    print("Extrapolation: at n=9 (our real circuit), dim(DLA) = 4^9 - 1 = "
          f"{4**9 - 1}, vs 81 trainable circuit parameters. "
          f"Underparametrized by a factor of {(4**9 - 1) / 81:.0f}x.")
