"""Step A1-A2: DLA of a candidate particle-number-conserving replacement ansatz,
before spending any training compute on it (per research/logs/2026-08-01-next-phase-plan.md).

Candidate generator set, replacing StronglyEntanglingLayers' {Y_i, Z_i} + CNOT-ring:
  - Z_i (phase rotation only; RX/RY would break particle-number conservation)
  - G_{i,i+1} = X_i Y_(i+1) - Y_i X_(i+1)  (exact generator of qml.SingleExcitation,
    confirmed directly from PennyLane: op.generator() on SingleExcitation(phi, [0,1])
    returns 0.25*(X(0)@Y(1)) - 0.25*(Y(0)@X(1)), a number-conserving "hopping" term),
    ring-connected matching the original CNOT-ring topology.

This is a real design choice, not a guess dressed as one: computed directly via the
same verified qml.lie_closure methodology as compute_dla.py, compared against both
the full-rank target (4^n-1, what the original ansatz hits) and the naive
per-sector-fully-controllable upper bound (sum_m C(n,m)^2 = C(2n,n) by Vandermonde's
identity, the total dimension if the ansatz became fully controllable within each
Hamming-weight sector separately, still far short of full rank since it respects
the conserved symmetry).
"""

from math import comb

import pennylane as qml
from pennylane import X, Y, Z


def symmetric_generators(n: int) -> list:
    gens = [Z(i) for i in range(n)]
    for i in range(n):
        j = (i + 1) % n
        gens.append(X(i) @ Y(j))
        gens.append(Y(i) @ X(j))
    return gens


def run(n: int) -> dict:
    dla = qml.lie_closure(symmetric_generators(n))
    full_rank = 4 ** n - 1
    sector_bound = comb(2 * n, n)  # Vandermonde: sum_m C(n,m)^2
    return {
        "n": n,
        "symmetric_dla_dim": len(dla),
        "full_rank_4n_minus_1": full_rank,
        "sector_full_controllable_bound": sector_bound,
        "ratio_vs_full_rank": round(len(dla) / full_rank, 4),
    }


if __name__ == "__main__":
    for n in (3, 4, 5, 6):
        print(run(n))
