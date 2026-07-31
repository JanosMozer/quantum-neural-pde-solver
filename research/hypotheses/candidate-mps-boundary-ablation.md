## Candidate: does Quantum-Train weight generation beat a classical tensor-network generator outside the MPS-efficient regime?

**Status: DRAFT — not yet run through full fracture, waiting on the broader landscape sweep (2nd research agent) before ranking against alternatives.**

### Precise statement (draft)
Construct a Quantum-Train-style weight generator (PQC → readout → linear map → classical PINN weights) for the 2D Burgers' equation using a strongly-entangling ansatz with non-local (circular) connectivity — i.e. deliberately outside the "1D or geometrically local Hamiltonian" regime the QT authors concede is MPS-simulable. Build a matched-parameter-count classical baseline generator (tensor-network/MPS-based, or a plain random low-rank projection) producing the same-size classical weight matrices. Train both to solve the same PDE instance, same data, same optimizer budget. Question: does the quantum generator achieve measurably better PDE residual / generalization at matched parameter count, or is it statistically indistinguishable from the classical baseline?

### Why this might be fracture-shaped
- **Precision gate:** passes as drafted above — object (weight generator), input class (2D Burgers' PINN, strongly-entangling non-local ansatz), behavioral requirement (matched parameter count, matched training budget), output (quantitative comparison).
- **Gap type (tentative): C** — performance-gap-with-structural-reason, where the structural reason is literally handed to us by the primary source: the authors' own caveat draws an explicit boundary (1D/local vs not) where their necessity proof doesn't apply. That boundary is the "crack."
- **Primary source:** Liu et al. 2024 (arXiv:2402.16465), quote in `ours/sources/liu2024_quantum_train.md`.
- **Attack:** concrete and buildable — pennylane-lightning[gpu]/cuQuantum for the PQC side (compute confirmed feasible at 10-12 qubits per the friend's design doc), an MPS/tensor-network classical baseline (libraries: `tensornetwork`, `quimb`, or a hand-rolled low-rank random projection as a cheaper first cut), same PINN backbone, same Burgers' data.
- **Open risk:** does this cascade to more than one downstream thing, or is it a leaf result? Tentatively: (1) it would directly inform whether the entire QT-for-PDE research direction is worth pursuing further (blocks/unblocks a research programme), (2) it's a template ablation methodology reusable for any QT-style claim, not just this one instance. Needs sharper articulation once the landscape sweep is in.

### What's still needed before this gets a real fracture verdict
- The 2nd background agent's broader QNN/QML sweep — need to know what else is out there before committing 2 weeks to this specific ablation instead of a different candidate.
- Infrastructure check: does a good classical MPS/tensor-network baseline implementation exist off the shelf (quimb? tensornetwork?), or would building a fair one eat into the 2-week budget?
- Cascade check needs sharpening — "informs whether QT-for-PDE is worth pursuing" is directionally right but not yet phrased as two concrete downstream unlocks per the fracture cascade step.
