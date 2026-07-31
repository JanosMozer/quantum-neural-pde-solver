## Sweke, Recio-Armengol, Jerbi, Gil-Fuster, Fuller, Eisert, Meyer — "Potential and limitations of random Fourier features for dequantizing quantum machine learning"

**arXiv:** 2309.11647v4, published Quantum (2024-07-10) — https://arxiv.org/pdf/2309.11647v4
**Verification:** full PDF text extraction (HTML render loses equation numbering — use the PDF for this paper).

### Theorem/lemma pair (corrects an earlier pass's wrong theorem number — it's not "Theorem 4.2", numbering is global)
- **Theorem 1** (Section 4.2, p.13) — sufficient side: RFF dequantizes with sample complexity `n ≥ max{c₁²log⁴(1/δ)/ϵ², n₀}`, `M ≥ c₀√n log(108√n/δ)`, polynomial in problem size `d` iff `‖T_{K_D}‖ = Ω(1/poly(d))` and `C = O(poly(d))` (kernel operator norm / RKHS norm conditions).
- **Lemma 3** (Section 4.7, p.26) — necessity side: for integer-frequency encodings, `ϵ̂ ≥ ‖f*‖₂²(1 − 2M·p_max)`, forcing `M = Ω(1/p_max)`. If `p_max` decays super-polynomially in `d`, RFF needs super-polynomial `M` — provably fails to dequantize efficiently.

### Failure class — concrete, not adversarial-narrow
Uniform re-weighting distributions, or product-induced distributions with all non-trivial component factors over the PQC frequency set (Section 4.5, Observation 3 + Eq. 72-74), give `p_max` decaying inverse-exponentially in `d`. Standard Pauli-encoding data-reuploading circuits with generic (non hand-adapted) weight vectors produce exactly this — **this is the default case for generic angle embeddings, not a cherry-picked pathology.**

### The gap
No numerical companion exists in this paper (Section 4 is pure theory) or anywhere found — nobody has built the concrete failing PQC regressor and empirically confirmed the M-scaling matches Lemma 3's prediction vs. Theorem 1's regime. A cited prior paper (Schreiber-Eisert-Meyer, "surrogate models") shows exponential frequency-set blowup empirically but for a different (classical surrogate) method, not RFF.

### 2-week feasibility: HIGH, and now precisely stated
Object: an RFF regressor. Input class: PQC data-reuploading regressors with uniform/product-induced frequency re-weighting (the generic/default case, easy to construct). Behavioral requirement: measure whether required RFF feature count `M` scales polynomially or super-polynomially in problem size `d`, matching Lemma 3's bound. Output: empirical scaling plot vs. the theorem's predicted regime boundary. Infrastructure: PennyLane-lightning-gpu (PQC side) + classical RFF regression (sklearn or hand-rolled) — nothing missing.

### Status
VERIFIED, full text. Candidate D upgraded from "needs verification" to precision-gate-clean.
