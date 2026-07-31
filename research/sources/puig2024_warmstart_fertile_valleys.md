## Puig, Drudis, Thanasilp, Holmes — "Variational quantum simulation: a case study for understanding warm starts"

**arXiv:** 2404.10044v4, PRX Quantum 6, 010317 — https://arxiv.org/html/2404.10044v4
**Code:** public, https://github.com/MarcDrudis/WarmStartCaseStudy

### Exact setup
10-qubit Ising Hamiltonian `H = Σ XᵢXᵢ₊₁ − 0.95 Σ Yᵢ`, 1D lattice, 2-layer Hamiltonian Variational Ansatz, iterative-Trotter-compression warm-start scheme.

### The open question (authors' own words)
*"In both cases this is evidence for toy problems and at a small problem sizes (10 qubits). To what extent these phenomena occur at larger problem sizes, for more interesting problems and for relevant time-step sizes, remains entirely open."* (Sec. III.5)

Detection of a "fertile valley" (a connecting path of non-vanishing gradient between a local warm-start minimum and a better, discontinuously-jumped-to minimum) was demonstrated manually, once, post-hoc: *"We managed to successfully train from this initial minimum to the new minimum using the BFGS algorithm"* — no systematic search/detection method was proposed or tested at scale.

### Sub-scoped 2-week problem
Object: their exact HVA/Ising/iterative-Trotter setup, ported to PennyLane-lightning-gpu/cuQuantum. Input class: qubit count n ∈ {10,14,18,22,26}, fixed timestep range, sampled minimum-jump instances. Behavioral requirement: for each jump instance, run a pre-registered multistart-BFGS + gradient-norm-threshold path search between old and new minimum (success criterion fixed in advance, no post-hoc curve-fitting). Output: empirical fertile-valley prevalence vs. n, and gradient-magnitude scaling along successful paths vs. n. Directly answers the authors' own stated open question.

### Feasibility
Week 1: port + reproduce n=10 baseline exactly (public code exists). Week 2: scale n, produce prevalence/scaling curves. Caveat: result is evidence for this one ansatz/Hamiltonian family only — stays gap type C, does not become a general theorem.

### Citation check
30 citing works (OpenAlex W4406760972) — none found addressing fertile-valley prevalence/detection specifically. Not redundant.

### Status
VERIFIED, full text, precision-gate-clean. No connection to the peer's PDE/PINN work — standalone VQE/quantum-simulation candidate.
