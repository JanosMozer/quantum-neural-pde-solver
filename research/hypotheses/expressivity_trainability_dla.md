# Fracture candidates: DLA dimension vs trainability (2026-07-30)

## Candidate 1: Empirically test the LASA variance-scaling law (Ragone et al. Thm 1)
outside its proven regime -- specifically for observables/states NOT inside i𝔤, on ansatz
classes not covered by Diaz et al.'s matchgate relaxation (e.g. Hamiltonian Variational
Ansatz for a non-integrable model, or a hardware-efficient ansatz with generic local Pauli-Z
observable).

- Gap type: A (proved theorem exists: Ragone et al. Thm 1, restricted to O/ρ ∈ i𝔤).
  Restriction is the open sub-case.
- Primary source: https://arxiv.org/abs/2309.09342 -- Theorem 1 (quoted in
  sources/ragone2024_lie_algebraic_bp.md).
- 2-week feasibility: HIGH. PennyLane's qml.lie_closure computes DLA dimension directly
  (https://docs.pennylane.ai/en/stable/code/api/pennylane.lie_closure.html); gradient-variance
  sampling over random parameters + qubit-count scan is standard, GPU-cheap up to ~20-24
  qubits statevector sim. Comparing measured Var[grad] vs 1/dim(g_j) prediction, and checking
  whether/how it breaks when O ∉ i𝔤, is a concrete, boundable 2-week numerical study.

## Candidate 2: Extend Diaz et al.'s "beyond-DLA" generalized expressiveness quantity
(dimension of Lie group modules) from matchgate/free-fermion circuits to one non-integrable
ansatz family, checking if the same module-dimension scaling holds.

- Gap type: A, but confidence LOWER -- I have not yet quoted Diaz et al.'s literal theorem
  text verbatim (only two independent paraphrased search summaries agree). Must fetch full
  text before treating this as verified.
- Primary source: https://arxiv.org/abs/2310.11505 (verbatim theorem quote still pending --
  see sources/diaz2023_beyond_dla.md, flagged NEEDS FOLLOW-UP).
- 2-week feasibility: MEDIUM. Requires first fully understanding Lie group module theory used
  in that paper (heavier group-representation machinery than Candidate 1) before any numerics --
  more front-loaded reading, less certain to land a clean result in 2 weeks.

Neither candidate yet run through the full fracture cascade (precision gate passed; importance/
attack/named-victim steps not yet done). This file records the literature gap only.
