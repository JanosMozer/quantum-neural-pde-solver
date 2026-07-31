## Ohno — "Generalization analysis of quantum neural networks using dynamical Lie algebras"

**arXiv:** 2504.09771v1 — https://arxiv.org/html/2504.09771v1 (full text obtained directly)

### Theorem 1 result
Generalization gap bound `O(sqrt(dim(DLA)))` for DLA-generated ansätze — improvement on parameter-count bounds. Empirical fit on TFIM QNNs is loose: R²≈0.6.

### Exact setup (for a faithful reproduction)
TFIM Hamiltonian `H = Σ ZᵢZᵢ₊₁ + Σ Xᵢ` (open: n_f=n-1 bonds; closed: n_f=n). L=2 layers, K=10 → 20 parameters. Train size M=10, test size 100, 20 dataset resamples, qubit sweep 2-8. Two optimizers tried: SPSA (init in [-0.01,0.01)) and random search (init at zero), both 200 epochs fixed budget.

### Exact author speculation on the R²≈0.6 mismatch (their own words, not inferred)
- *"the RMSE results exhibited relatively large values, which may affect the generalization gap... it is inadvisable to perform statistical tests between the open and closed boundary conditions. Further investigation is needed."*
- *"for both training algorithms, the RMSEs for eight qubits were somewhat large. This may be because the approximation ability of the circuit U (Eq. 38) is low."*
They lean toward **training-convergence noise** (fixed-epoch budget, not fixed-quality), not toward "DLA is the wrong measure" or "the bound's constant is loose" — though they don't rule those out explicitly either.

### Sub-scoped 2-week hypothesis (recommended reading of the evidence)
The R²≈0.6 scatter is driven by training-convergence noise, not a wrong complexity measure — because a loose constant only rescales the fit line, it can't by itself produce scatter, and the authors' own diagnostic language points at RMSE/training quality repeatedly, not optimizer choice (they tried two and got "almost equivalent" results per their own comparison).

**Precise reproduction + test:** rerun Ohno's exact TFIM setup, but replace the fixed-200-epoch stopping rule with a fixed-low-RMSE convergence criterion (using Adam or L-BFGS on parameter-shift gradients instead of SPSA/random-search). Regress generalization gap on √dim(DLA) again.
- If R² rises and per-run residuals correlate with per-run training RMSE → training noise confirmed, and the theorem's fit is actually much tighter than the original figure suggested.
- If R² stays ~0.6 regardless of convergence quality → rules out training noise, points toward dim(DLA) being the wrong complexity measure for this regime (next test: refit against effective dimension / Fisher-information trace instead).
Either outcome is a real, informative, publishable result about where a real theorem's practical predictive power actually holds.

### Citation check
Only one citing paper found (OpenAlex, SN Computer Science 2026, background citation only) — this specific discrepancy has not been investigated by anyone else. Not redundant.

### Status
VERIFIED, full text, precision-gate-clean. Standalone candidate — clean, self-contained numerics, no PDE/PINN overhead, doesn't connect to the peer's project.
