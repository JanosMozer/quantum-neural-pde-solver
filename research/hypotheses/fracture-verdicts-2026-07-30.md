## Fracture verdicts — 2026-07-30, post-sweep

Ten candidates came back from the landscape sweep (full list + sources in the task log / `ours/sources/`). Full fracture applied to the three strongest; one is a synthesis of two sweep candidates that turned out to compose.

---

### TOP PICK (synthesis of sweep candidates C + E): Is the Quantum-Train PDE weight-generator classically simulable, and does that predict the ablation result?

**Precision gate:** For the specific ansatz the peer's QT-PINN design would use (strongly-entangling, circular-CNOT, shallow, small-angle-init — chosen because it's known to *avoid* barren plateaus), (1) classify it against Cerezo et al.'s named classically-simulable-via-light-cone-reduction + classical-shadow families; (2) if it falls inside that regime, this *theoretically predicts* no measurable advantage over a classical generator; (3) empirically test that prediction with a matched-parameter-count ablation (QT generator vs. classical MPS/low-rank generator) on 2D Burgers', measuring both PDE residual and generalization. Passes — object, input class, behavioral requirement, and output are all named.

**Conjunction test:**
- *Importance:* if the ansatz falls in the classically-simulable regime AND the ablation confirms no advantage, this directly resolves (for a real PDE-solving use case, not a toy) whether the peer's whole architecture family is doing anything a classical hypernetwork couldn't — a foundational validity check for an active project, not a hypothetical. If it falls *outside* that regime (i.e. Cerezo's classical-simulability criteria don't apply here — plausible, since PDE weight generation isn't a BP-avoidance-motivated ansatz choice in the way their surveyed families are) and the ablation still shows no advantage, that's a *new* empirical data point outside their studied case list. Either branch produces a real, usable answer.
- *Attack:* concrete three-step pipeline above, all off-the-shelf tooling (PennyLane-lightning-gpu/cuQuantum for the PQC, quimb/tensornetwork or a plain random low-rank projection for the classical control, `qml.lie_closure`-adjacent classification tools for step 1). No missing infrastructure.
- Verdict: PASSES.

**Gap type:** C, with the structural reason handed to us by two independent primary sources: Liu et al.'s own 1D/local-Hamiltonian caveat on their necessity proof, and Cerezo et al.'s classical-simulability criteria for BP-avoiding ansatz families. The specific joint question (does a QT-style PDE weight-generator fall into Cerezo's simulable set, and does that predict the ablation) has not been asked by anyone — verified via OpenAlex citation check (18 citing papers, zero ablations) plus full-text keyword sweep of the QT paper itself (zero mentions of "ablation"/"classical shadow"/"light-cone").

**Primary sources (already quoted in full in `ours/sources/`):**
- Liu et al. 2024, arXiv:2402.16465 — necessity proof + 1D/local caveat.
- Cerezo et al. 2023, arXiv:2312.09121 — "many commonly used models whose loss landscapes avoid barren plateaus can also admit classical simulation, provided that one can collect some classical data from quantum devices during an initial data acquisition phase."

**Infrastructure readiness:** READY. GPU statevector sim confirmed feasible at the relevant qubit counts (8-14) per the friend's own compute estimates; classical MPS/shadow tooling is standard and installable; no dependency on infrastructure another team is building.

**First-principles check:** My own re-derivation (see `ours/logs/2026-07-30-setup.md`) already predicted that if unitarity buys anything, it's more likely *implicit regularization quality* (smoothness/generalization) than raw expressivity — so the experiment should measure both PDE residual AND out-of-distribution generalization, not residual alone. This is a genuine first-principles addition, not just confirming the literature's framing.

**Cascade:** (1) directly informs the peer's own project's foundational framing — immediate, concrete collaborator value, not a side result; (2) produces a reusable methodology (classify-then-ablate) for evaluating any future "quantum generates classical X" claim, which is the majority framing of the entire Quantum-Train literature family. Passes — two distinct, real downstream unlocks.

**Verdict: PROCEED.** Strongest candidate: synergizes directly with the peer's actual repo/problem, clean 2-week attack, two independent primary sources supply the structural reasoning, genuinely unasked question.

---

### Backup 1: DLA variance-scaling law (Ragone et al. Thm 1) tested outside its proven LASA regime

**Precision gate:** passes (see `ours/hypotheses/expressivity_trainability_dla.md` Candidate 1).
**Gap type:** A (proved theorem; the restriction O/ρ∈i𝔤 is the open sub-case).
**Attack:** `qml.lie_closure` for DLA dimension, gradient-variance sampling vs. qubit count, GPU-cheap to ~20-24 qubits. Very clean, very executable.
**Cascade:** weaker than the top pick — this is closer to a leaf (confirms/extends one theorem's boundary) unless framed against a second application. Still a legitimate, well-scoped 2-week numerical study with a real theorem behind it.
**Verdict: PROCEED as a fallback** if the top pick's classical-simulability classification step (stage 1) turns out to be a dead end (e.g. the QT ansatz doesn't cleanly map onto any of Cerezo's named families) — this is the cleanest pure-theory-testing option with zero PDE/PINN engineering overhead.

### Backup 2: "Black hole" barren-plateau entanglement ablation in physics-consistency QPINNs (authors' own stated future work)

**Precision gate:** passes.
**Gap type:** C.
**Attack:** reproduce the 2D Maxwell QPINN with energy-conservation loss (arXiv:2506.23246), then run exactly the entanglement-ablation-at-fixed-parameter-count sweep the authors explicitly named as future work. GPU removes their stated compute bottleneck.
**Cascade:** single-paper-scoped (a named follow-up experiment, not a self-discovered gap) — genuine but narrower value than the top pick.
**Verdict: HOLD** — good candidate if the top pick stalls, but it's answering someone else's named next step rather than a gap we found ourselves; lower strategic value given the peer-synergy angle is absent here.

---

### Upgraded after verification pass (B, D, J) — all now precision-gate-clean

**Candidate B — warm-start "fertile valley" prevalence (Puig et al., arXiv:2404.10044).** Sub-scoped: reproduce their exact 10-qubit HVA/Ising/iterative-Trotter warm-start setup (public code exists), then scale qubit count n∈{10,14,18,22,26} and run a pre-registered multistart-BFGS path search between warm-start minima and jumped-to minima, measuring fertile-valley prevalence and gradient scaling vs. n. Directly answers the authors' own named open question ("to what extent these phenomena occur at larger problem sizes... remains entirely open"). Gap type C (stays C — evidence for one ansatz family, not a theorem). Week 1 reproduce, week 2 scale. No connection to the peer's work. **PROCEED-eligible, standalone.**

**Candidate D — RFF dequantization failure (Sweke et al., arXiv:2309.11647).** Corrected theorem numbering (Theorem 1 + Lemma 3, not "Thm 4.2"). The failure case (uniform/product-induced frequency re-weighting) is the *default* case for generic data-reuploading PQC regressors, not adversarial-narrow. No numerical companion exists anywhere — build the specific failing PQC regressor, run RFF against it, confirm the M-scaling matches Lemma 3's prediction. Gap type A/B, HIGH confidence, clean GPU-light build. **PROCEED-eligible, standalone.**

**Candidate J — DLA generalization bound fit (Ohno, arXiv:2504.09771).** Sub-scoped to the most tractable explanation: the authors' own language points at training-convergence noise (fixed-epoch budget, not fixed-quality) rather than a wrong complexity measure. Rerun their exact TFIM setup with a fixed-low-RMSE stopping rule instead of fixed epochs; if R² rises and correlates with per-run RMSE, training noise is confirmed (and the theorem is tighter than the original figure suggests); if not, pivot to testing effective-dimension/Fisher-information as the complexity measure instead. Either outcome is a real, informative result. Gap type C, HIGH confidence now, clean self-contained numerics. **PROCEED-eligible, standalone.**

All three are now legitimate 2-week candidates. None connect to the peer's project the way the top pick (C+E synthesis) does — that remains the strongest pick on the peer-synergy axis, but B/D/J are all real, clean, standalone alternatives if a pure-QML (non-PDE) angle is preferred.

---

## FINAL PLAN — D as warm-up, C+E as main project (decided 2026-07-30)

### Fracture, stated sharp

- **Precision gate:** PASS. Object: a weight-generator ablation. Input class: 2D Burgers' PINN, strongly-entangling non-local (circular-CNOT) PQC ansatz. Behavioral requirement: matched parameter count, matched optimizer budget, against two classical baselines (random low-rank, MPS/tensor-network). Output: PDE residual + OOD generalization, compared against a prior classification-based prediction.
- **Conjunction test — importance:** peer is training this exact architecture with no critical framing attached. Whichever way the ablation lands, it changes what the peer's paper can honestly claim: "quantum advantage" (if QT wins) or "quantum-inspired compression trick, here's the honest classical baseline" (if it doesn't) — both are publishable, only one is currently being assumed. **attack:** concrete, three-stage, all off-the-shelf (PennyLane-lightning-gpu/cuQuantum, quimb for MPS, sklearn for the D warm-up). PASS.
- **Gap type:** C. Structural reason supplied by two independent primary sources (Liu et al.'s 1D/local caveat; Cerezo et al.'s classical-simulability criteria for BP-avoiding ansätze) — not a vibes-based "let's just try it."
- **Primary sources:** arXiv:2402.16465, arXiv:2312.09121 (top pick); arXiv:2309.11647 (D warm-up). All full-text verified, quotes pinned in `ours/sources/`.
- **Infrastructure readiness:** READY. GPU statevector sim confirmed feasible at 8-14 qubits (friend's own compute math, independently sane per barren-plateau depth thresholds found in the sweep). No missing library. Building our own QT-PINN implementation, independent of the peer's repo state, so we're not blocked on their pace.
- **Cascade — concrete downstream applications, not abstract:**
  1. **Directly changes the peer's paper's claim.** Negative result → they honestly reframe as compression/regularization, avoiding a reviewer takedown at any QML venue that knows the dequantization literature. Positive result → they get a defensible, two-theorem-grounded core claim instead of an unexamined one.
  2. **Real engineering lever for scientific ML.** PINNs are used for CFD surrogates, structural/thermal simulation, and real-time PDE-based control. If QT-style generation genuinely compresses/regularizes better than classical hypernetworks at matched parameter count, that's a usable technique for edge-deployed PDE surrogates. If it doesn't, that redirects compute/grant effort in the wider scientific-ML-meets-QML community away from a dead end — a real service, not a null result.
  3. **Reusable audit methodology.** "Classify ansatz by classical-simulability criteria, then predict-and-ablate" becomes a checklist applicable to the whole growing Quantum-Train family (QTRL, QT-LoRA/ICLR2025, future follow-ups) — turns "is this quantum trick real" from a vibes call into a repeatable procedure, usable by other researchers or reviewers.
  4. **D warm-up has its own standalone payoff.** A clean empirical confirmation of Sweke et al.'s dequantization-failure boundary is a citable, self-contained result even if the main project stalls — not wasted time.
- **Verdict: PROCEED.**

### What we will actually do (≈18 working days, 3.5 weeks)

**Phase 0 (days 1-4) — D warm-up, builds Feynman-level grounding before the harder case.**
Build a data-reuploading PQC regressor with uniform/product-induced frequency re-weighting (Sweke Sec 4.5 construction) + a matched classical RFF baseline (sklearn). Sweep required feature count M vs. problem dimension d. Confirm the empirical M-scaling matches Lemma 3's super-polynomial prediction. Deliverable: one clean figure, standalone.

**Phase 1 (days 5-9) — build our own QT-PINN, independent of peer's repo state.**
Classical Fourier front-end + strongly-entangling circular-CNOT PQC + weight-mapping layer + PDE loss + Adam→L-BFGS (per friend's design, testing both the ⟨Z_i⟩ cheap readout and the literal QT full-2^N-distribution readout as two configs). Classify the ansatz against Cerezo et al.'s light-cone-reduction/classical-shadow criteria. Build two classical baselines: random low-rank projection, MPS/tensor-network (quimb). Deliverable: in/out-of-classically-simulable-regime verdict + a stated prediction for Phase 2's outcome.

**Phase 2 (days 10-14) — run the ablation.**
Train all three generators (QT, low-rank, MPS) with identical PINN backbone/data/optimizer budget on 2D Burgers'. Metrics: PDE residual, IC/BC loss, held-out-region generalization, cross-seed stability. Compare against Phase 1's prediction. Deliverable: quantitative ablation table.

**Phase 3 (days 15-18) — writeup + reconcile with peer.**
Short technical note / arXiv-draft framing: "Is Quantum-Train Necessary for PDE-Constrained Weight Generation? A Classifiability-Predicted Ablation on 2D Burgers'." Cross-check against whatever the peer has actually trained by then. Finalize `ours/qm-foundations.md`.
