# NS2D Experiment Analysis (Taylor–Green + Kolmogorov)

Summary of Navier–Stokes experiments in this repo: what was tried, what worked, what failed, and why the quantum hypernetwork path has not shown advantage. Written as measurement notes for future runs.

**Tier A–B (archived):** Taylor–Green vortex (TGV) on \([0,2\pi]^2 \times [0,T]\) — exact solution available.

**Tier C (complete):** 2D Kolmogorov flow — forced NS, no exact solution; metric is PDE residual RMS.

**Architecture pattern:** a *generator* (quantum or classical) produces weights for a small target MLP (`TargetPINNNS`, 3 outputs: \(u,v,p\)). Training minimizes PDE residual + IC loss (unless hard IC).

### TGV exact solution (velocity convention in code)

\[
u = \sin x \cos y \, e^{-2\nu t},\quad
v = -\cos x \sin y \, e^{-2\nu t},\quad
p = +\tfrac{1}{4}(\cos 2x + \cos 2y)\, e^{-4\nu t}
\]

### Kolmogorov PDE (Tier C)

\[
\partial_t u + (u\cdot\nabla)u + \nabla p = \nu \nabla^2 u + f,\quad \nabla\cdot u = 0,\quad
f = (A\sin(n y),\, 0),\; n=4
\]

IC: \(u=v=p=0\) at \(t=0\) (hard IC: field \(= t \cdot \mathrm{network}\)).

---

## Experiment timeline (tiers)

| Tier | Goal | Scripts / runs |
|------|------|----------------|
| **0** | First NS quantum run (Burgers run_0072 settings) | `train_ns.py` → `ns_run_0001` |
| **0b** | Hyperparameter sweep (λ\_bc, bottleneck, Fourier σ, w\_reg) | `ns_sweep_8x.py` → `ns_sweep_*` |
| **A** | Fix physics & target network before any QC claim | `train_ns_direct.py` → `ns_direct_a3/a4`, `a3_nu0p1_T5` |
| **B v1** | Well-posed family test: \(\nu \mapsto \theta(\nu)\) | `train_ns_parametric.py` → `ns_par_c_s0`, `ns_par_q_s0` |
| **B v2** | Redesign QC: log-\(\nu\) + Z expectations | `train_ns_parametric.py --qc-arch expect` → `ns_par_c_v2_s0`, `ns_par_q_v2_s0` |
| **C** | Kolmogorov forced flow (no exact soln) | `train_kol_direct.py`, `train_kol_parametric.py` → `kol_*` |

---

## Critical bugs found (invalidated early results)

These were not hyperparameter problems; they made the problem unlearnable or the metric meaningless.

### 1. Pressure sign inconsistency (`physics_loss.py`)

- **Symptom:** `bc_loss ≈ 0.56`, `pde_loss` small, rel-L2 ≈ 100% — collapse to \(u=v=p=0\).
- **Cause:** Exact pressure used the wrong sign for the chosen velocity convention; coded residuals disagreed with the analytical solution (momentum RMS ~ 0.7).
- **Fix:** \(p = +(\cos 2x + \cos 2y)\, e^{-4\nu t}/4\). Residual check → RMS \(< 10^{-8}\).
- **Runs affected:** All pre-fix runs (`ns_run_0001`, entire `ns_sweep_*` cohort).

### 2. Time channel dead in TGV Fourier map (`fourier.py`)

- **Symptom:** `ns_direct_a3` and `ns_direct_a3_time` **byte-identical** results; no temporal learning.
- **Cause:** Row for \(t\) in the TGV Fourier matrix was all zeros → MLP was a function of \((x,y)\) only.
- **Fix:** Explicit normalized time features \(t/T\) (and optionally \(t^2/T^2\)); spatial part stays deterministic \(k=1,2\) basis.
- **Note:** TGV decay is \(e^{-2\nu t}\), not periodic — sinusoidal time Fourier was wrong anyway.

### 3. Regime too easy to ignore time

- Default \(\nu=0.01, T=1\) → amplitude decay \(e^{-0.02} \approx 0.98\). “Ignore time” is nearly optimal.
- **Fix for meaningful dynamics:** e.g. \(\nu=0.1, T=5\) (decay to ~37% at horizon).

### 4. Constant-input quantum generator (conceptual bug)

- Original `qnn_generator_ns.py` feeds a **fixed** input to the circuit → one fixed probability vector → one fixed weight vector.
- That is a reparameterization of a point, not a computation. No quantum advantage is possible in principle; classical lookup of converged weights reproduces it at zero cost.

### 5. Evaluation pitfalls (measurement notes)

| Pitfall | Effect | Mitigation |
|---------|--------|------------|
| Relative L2 at \(t=T\) only | At high \(\nu\), exact field \(\propto e^{-2\nu T}\) → tiny denominator → error explodes (e.g. 4000% artifacts) | **Space-time** rel-L2 over \([0,T]\); gauge-fixed pressure |
| “Held-out” = resampled train range | Two i.i.d. log-uniform draws look identical (`ns_par_c_s0`: 2.638% vs 2.638%) | Fixed grids: `in_range`, `extrap_lo`, `extrap_hi` |
| Equal step count, quantum still descending | Unfair if quantum loss still falling at step cap | Plateau stop (`patience`); report `stopped_at` |
| Linear \(\nu\) encoding + log-uniform training | Extrap below \(\nu_{lo}\) off-manifold for the circuit | v2: `log(ν)` encoding |
| Geometric freq ladder \(2^i\pi\) | Heavy aliasing on RY (period \(4\pi\)) → erratic per-\(\nu\) errors | Default **linear** ladder \((i+1)\pi |

---

## Results by experiment class

### Tier 0 — Initial quantum NS (`ns_run_0001`, sweeps)

**Setup:** Quantum weight generator, random RFF, soft IC, \(\nu=0.01\), \(T=1\), settings copied from Burgers best run.

| Metric | Typical value | Interpretation |
|--------|---------------|----------------|
| `bc_loss` | ~0.54 | IC not satisfied |
| rel-L2 vs exact | ~98–102% | Trivial \(u=v=p=0\) or nonsense |
| `pde_loss` | low | Zero field satisfies PDE if IC is weak |

**Sweeps (`ns_sweep_*`, 16 configs):** Higher `lambda_bc`, wider bottleneck, weaker `weight_reg` did not fix collapse — consistent with **wrong physics**, not tuning.

**Verdict:** FAIL — did not work until pressure sign fix. Sweeps are historical only.

---

### Tier A — Direct classical PINN (no generator)

**Purpose:** Can the *target* network solve NS at all? If not, generator comparisons are meaningless.

| Run | Config | Mean rel-L2 @ \(t=T\) | Gate (<5%) | Notes |
|-----|--------|------------------------|------------|-------|
| `ns_direct_a3` | TGV Fourier, \(\nu=0.01, T=1\), soft IC | ~2.7% | PASS | Works after physics fix |
| `ns_direct_a3_time` | Same | **Identical to a3** | PASS | Proved time bug |
| `a3_nu0p1_T5` | \(\nu=0.1, T=5\), time channel live | ~7.8% mean @ \(t=5\) | FAIL (pressure-dominated) | \(u,v\) ~3.8–4.0% at \(t=5\) |
| `ns_direct_a4` | Hard IC | Pressure gauge issues | — | \(p\) only defined up to \(C(t)\) |

**What worked:** Classical direct MLP with TGV Fourier + corrected physics reaches **~1–4% velocity error** in-range.

**What did not:** Hard-IC pressure metric without gauge fix; very long horizons with raw pressure rel-L2.

**Verdict:** PASS — target network is viable. Quantum must beat **~2–4%** class of solution, not “better than 100%.”

---

### Tier B v1 — Parametric family (\(\nu \in [0.05, 0.5]\), `reupload` arch)

**Purpose:** Learn hypernetwork \(\nu \mapsto \text{MLP weights}\). First test where the circuit must represent a **function**, not a constant.

**Setup:** `freq_mode=linear`, 8 qubits, 4 layers, \(T=2\), log-uniform \(\nu\) during training, 12k steps (both hit cap), seed 0.

| Split | Classical `ns_par_c_s0` | Quantum `ns_par_q_s0` | Q/C |
|-------|-------------------------|------------------------|-----|
| **in-range** | **1.33%** | 1.72% | 1.29 |
| **extrap-lo** (\(\nu < 0.05\)) | **2.09%** | 49.0% | **23.4** |
| **extrap-hi** (\(\nu > 0.5\)) | 85.7% | 212.6% | 2.48 |
| Wall time | 1182 s | 2146 s | 1.82× |
| Final rolling loss | \(1.1\times 10^{-4}\) | \(2.1\times 10^{-4}\) | — |

**Per-\(\nu\) patterns:**

- **Classical in-range:** Smooth, monotone in \(\nu\) (0.85% → 1.95% for \(u,v\) as \(\nu\) increases).
- **Quantum in-range:** Acceptable (~1.7% mean) but slightly worse throughout.
- **Classical extrap-lo:** Improves toward boundary (0.6% at \(\nu=0.04\)).
- **Quantum extrap-lo:** **Erratic** — 52%, 85%, 20% at \(\nu=0.02,0.03,0.04\) (non-physical ordering).
- **Both extrap-hi:** Poor; partly metric + vanishing amplitude at \(\nu=0.8, T=2\).

**Verdict:**

- PASS — Classical hypernetwork: strong interpolation + reasonable extrapolation below training range.
- NOTE — Quantum v1: **interpolation OK**, **extrapolation below range failed** (memorization / encoding mismatch).
- FAIL — No quantum advantage; classical wins on accuracy and speed.

---

### Tier B v2 — Redesigned QC (`expect` arch, log-\(\nu\))

**Changes:** `log(ν)` encoding, Pauli-\(Z\) expectations (6 qubits × 6 layers), matched classical v2.

**Setup:** Same training protocol as v1 (12k steps, seed 0, \(T=2\), `expect` + `log`).

| Split | Classical `ns_par_c_v2_s0` | Quantum `ns_par_q_v2_s0` | Q/C |
|-------|----------------------------|---------------------------|-----|
| **in-range** | 1.99% | 2.21% | 1.11 |
| **extrap-lo** | 31.0% | 31.3% | 1.01 |
| **extrap-hi** | 41.5% | 83.7% | 2.01 |
| Wall time | 1171 s | 2259 s | 1.93× |
| Final rolling loss | \(1.6\times 10^{-4}\) | \(2.6\times 10^{-4}\) | — |

**Comparison to v1:**

| Model | in-range | extrap-lo |
|-------|----------|-----------|
| Classical v1 | 1.33% | **2.09%** |
| Classical v2 | 1.99% | **31.0%** (regression) |
| Quantum v1 | 1.72% | 49.0% |
| Quantum v2 | 2.21% | 31.3% (fixed erratic pattern) |

Log-\(\nu\) + 6-dim readout brought quantum extrap-lo to **parity with classical v2**, but **destroyed classical v1’s extrap-lo advantage** (2% → 31%). The v2 A/B comparison is fair between architectures; **best overall classical remains v1 reupload** (`ns_par_c_s0`).

**Verdict:** FAIL — classical wins (extrap Q/C = 1.58). Credible **null result** for VQC-hypernetwork on TGV. Pivot to Kolmogorov flow (Tier C).

**Compare command used:**

```bash
.venv/bin/python scripts/compare_ns_parametric.py checkpoints/ns_par_c_v2_s0 checkpoints/ns_par_q_v2_s0
```

---

### Tier C — Kolmogorov flow (forced NS)

**Purpose:** Harder benchmark with sustained nonlinearity (body force \(f_x = \sin 4y\)). No exact solution — evaluate **PDE residual RMS** on fixed holdout grids, not rel-L2 vs analytic fields.

**Setup:** `FourierFeatureMapKolmogorov`, hard IC, \(\nu \in [0.01, 0.1]\) log-uniform during training, \(T=5\), `expect` + log-\(\nu\), 6 qubits × 6 layers, 12k steps, seed 0.

#### C0 — Direct baseline (`kol_direct_s0`)

| Metric | Value |
|--------|-------|
| PDE RMS | **0.00246** |
| Gate (< 0.05) | PASS |
| Params | 1,763 |
| Time | 95 s |

Confirms target MLP + Kolmogorov physics are learnable before any generator comparison.

#### C1 — Parametric family (`kol_par_c_s0` vs `kol_par_q_s0`)

| Split | Classical | Quantum | Q/C |
|-------|-----------|---------|-----|
| **in-range** | **0.00288** | 0.00534 | 1.85 |
| **extrap-lo** (\(\nu < 0.01\)) | 0.282 | 0.392 | 1.39 |
| **extrap-hi** (\(\nu > 0.1\)) | 1.459 | 2.214 | 1.52 |
| Wall time | 1145 s | 2119 s | 1.85× |

Combined extrap PDE RMS Q/C = **1.50** (primary decision metric).

**In-range per \(\nu\) (PDE RMS):**

| \(\nu\) | Classical | Quantum |
|---------|-----------|---------|
| 0.012 | 0.0027 | 0.0033 |
| 0.020 | 0.0020 | 0.0027 |
| 0.035 | 0.0020 | 0.0061 |
| 0.050 | 0.0026 | 0.0087 |
| 0.070 | 0.0034 | 0.0047 |
| 0.095 | 0.0046 | 0.0066 |

Classical mean 0.0029 — matches direct baseline. Quantum ~2× worse in-range but still small in absolute terms.

**extrap-lo per \(\nu\):**

| \(\nu\) | Classical | Quantum |
|---------|-----------|---------|
| 0.005 | 0.472 | 0.355 |
| 0.007 | 0.312 | 0.367 |
| 0.009 | **0.061** | 0.454 |

Classical improves toward training boundary (0.009 is near 0.01); quantum stays flat ~0.35–0.45 with no monotone trend.

**extrap-hi per \(\nu\):**

| \(\nu\) | Classical | Quantum |
|---------|-----------|---------|
| 0.12 | 0.162 | 0.743 |
| 0.15 | 1.101 | 1.299 |
| 0.20 | 3.113 | 4.599 |

Both fail at high extrap; eval grid reaches 2× training \(\nu_{\max}\).

**Compare to TGV (best classical runs):**

| Benchmark | in-range | extrap-lo | QC win? |
|-----------|----------|-----------|---------|
| TGV v1 (`ns_par_c_s0`) | 1.3% rel-L2 | **2.1%** rel-L2 | No |
| Kolmogorov (`kol_par_c_s0`) | 0.29% PDE RMS | **28%** PDE RMS | No |

Kolmogorov is a genuinely harder extrapolation problem for hypernetworks (nonlinear forcing, no analytic anchor). The quantum result is unchanged: **no advantage**.

**Verdict:**

- PASS — Direct + classical parametric in-range (PDE RMS \(\lesssim 0.005\)).
- FAIL — Both hypernetworks generalize poorly outside training \(\nu\); quantum worse on every split.
- FAIL — **Project-level null result** for VQC-hypernetwork on forced NS. Same answer as TGV, on a harder PDE.

**Compare command:**

```bash
.venv/bin/python scripts/compare_kol_parametric.py checkpoints/kol_par_c_s0 checkpoints/kol_par_q_s0
```

See also `pdes/kolmogorov2d/README.md`.

---

## What worked (overall)

1. **Corrected TGV physics** — enables learning; exact solution validates residuals.
2. **Deterministic TGV spatial Fourier** — matches wavenumbers \(k=1,2\); much better than mis-scaled random RFF.
3. **Explicit time channel** — required for unsteady TGV; MLP must see \(t\).
4. **Direct classical PINN** — proof the PDE + network capacity are sufficient (~1–4% velocity error).
5. **Parametric classical generator** — learns \(\nu \mapsto \weights\) with ~1.3% in-range, ~2% extrap-lo (v1).
6. **Fixed eval protocol** — space-time rel-L2, gauge-fixed \(p\), true extrap grids, plateau stopping.
7. **Xavier-init generator head** — prevents hypernetwork weight blow-up at step 0.
8. **Kolmogorov direct PINN** — PDE RMS 0.0025; forced NS with nonlinear dynamics is learnable.
9. **Kolmogorov classical hypernetwork** — in-range PDE RMS 0.0029 across \(\nu \in [0.01, 0.1]\).

---

## What did not work

1. **Quantum NS with constant circuit input** — equivalent to static weights; no functional QC.
2. **All pre-fix quantum/classical NS runs** — pressure sign bug → trivial collapse.
3. **Hyperparameter sweeps without physics fix** — wasted compute on unlearnable objective.
4. **Quantum v1 parametric (`ns_par_q_s0`)** — loses to classical on every decision metric; catastrophic extrap-lo.
5. **Quantum v2 parametric (`ns_par_q_v2_s0`)** — parity with classical v2 on extrap-lo at ~31%, but both worse than v1 classical; no speed or accuracy win.
6. **Using TGV alone to “prove quantum advantage”** — see structural limits below.
7. **Hard IC without pressure gauge handling** — false failure on \(p\) (TGV).
8. **Kolmogorov parametric quantum (`kol_par_q_s0`)** — in-range 1.85× worse PDE RMS; extrap 1.4–1.5× worse; 1.85× slower.
9. **Hypernetwork extrapolation on Kolmogorov** — both architectures fail outside training \(\nu\) (classical extrap-lo mean 28%, quantum 39%).
10. **VQC-hypernetwork line (Tiers 0–C)** — no setting showed quantum matching or beating classical on the primary metric.

---

## Why quantum has not worked overall (measurement + structural notes)

### A. Implementation / measurement (fixable)

1. **Encoding ≠ training distribution** — log-uniform \(\nu\) samples with linear \(\nu\) angles punished extrap-lo (quantum 49% vs classical 2%).
2. **Harsh readout** — full Hilbert-space probabilities → 256-dim simplex → optimization noise; expectations are smoother.
3. **Slower, noisier training** — quantum rolling loss ~2× classical at same step cap; gradient path through PennyLane CPU sim + entangling layers.
4. **Extrap-hi metric** — both models fail; do not over-interpret without amplitude-aware metrics.

### B. Problem / architecture (fundamental for this benchmark)

1. **TGV is nearly linear diffusion** — nonlinear advection and pressure gradient cancel; family indexed by \(\nu\) is mostly “decay rate of sines.” Low-dimensional manifold; not a stress test for entanglement.
2. **Hypernetwork maps \(\nu \to 1507\) weights** — high-dimensional output from low-dimensional input; classical MLP encoder may simply be a better inductive bias than a small VQC.
3. **No input-conditioned spatial QC** — generator only sees \(\nu\), not \((x,y,t)\). Quantum never participates in field evaluation; it only shapes weights offline. Advantage claims need a clearer story than “replace MLP encoder with circuit.”
4. **Simulation cost** — ~1.8–1.9× wall time with worse accuracy on every completed parametric run.

### C. Tier C confirmation (Kolmogorov)

1. **Harder PDE, same answer** — forcing maintains nonlinearity, yet quantum still loses in-range and on extrap (Q/C 1.50).
2. **No exact solution did not help QC** — metric is PDE RMS, removing rel-L2 artifacts; outcome unchanged.
3. **Hypernetwork extrapolation is the weak link** — in-range fits are good; failure is \(\nu \mapsto \weights\) generalization, not field network capacity.

### D. Decision criteria (historical)

| Observation | Interpretation |
|-------------|----------------|
| extrap-lo Q/C ≤ 0.90 (v2+) | Worth deeper investigation on this family |
| extrap-lo Q/C 0.90–1.15 | Parity only — not advantage |
| extrap-lo Q/C ≫ 1.15 (v1: **23×**) | Drop this QC design for TGV |
| Win on TGV only | Weak evidence; move to **Kolmogorov flow** or forced NS with sustained nonlinearity |
| Loss on TGV after v2 | Null result on TGV — confirmed by Tier C |
| Kolmogorov extrap PDE RMS Q/C > 1.15 | Null result on forced NS — **stop this QC design** |
| Kolmogorov in-range Q/C ~ 1.85 | Confirms gap is not only an extrapolation artifact |

**Current recommendation:** Do not invest further in ν-conditioned VQC hypernetworks for PINNs in this repo unless the architecture changes fundamentally (e.g. input-conditioned spatial QC, not \(\nu \to 1700\) static weights).

---

## Key checkpoints (reference)

| Run ID | Type | Headline result |
|--------|------|-----------------|
| `ns_run_0001` | QG v0, broken physics | ~100% rel-L2, collapse |
| `ns_sweep_*` | Sweep v0 | Same failure mode |
| `ns_direct_a3` | Direct MLP | ~2.7% @ \(t=1\), gate pass |
| `a3_nu0p1_T5` | Direct MLP, harder regime | \(u,v\) ~4% @ \(t=5\) |
| `ns_par_c_s0` | Classical hypernet v1 | 1.3% in-range, 2.1% extrap-lo |
| `ns_par_q_s0` | Quantum hypernet v1 | 1.7% in-range, **49% extrap-lo** |
| `ns_par_c_v2_s0` | Classical hypernet v2 | 2.0% in-range, 31% extrap-lo |
| `ns_par_q_v2_s0` | Quantum hypernet v2 | 2.2% in-range, 31% extrap-lo; extrap Q/C 1.58 |
| `kol_direct_s0` | Direct Kolmogorov MLP | PDE RMS 0.0025, gate pass |
| `kol_par_c_s0` | Classical hypernet | in-range RMS 0.0029; extrap-lo 0.28 |
| `kol_par_q_s0` | Quantum hypernet | in-range RMS 0.0053; extrap Q/C 1.50 |

---

## Files & commands

| File | Role |
|------|------|
| `pdes/ns2d/config.yaml` | Default PDE / training hyperparameters |
| `pdes/ns2d/physics_loss.py` | TGV residual, IC, exact solution, gauge-fixed metrics |
| `src/qt_pinn/fourier.py` | `FourierFeatureMapTGV` (spatial + time) |
| `src/qt_pinn/pinn_target_ns.py` | Target MLP \((x,y,t)\to(u,v,p)\) |
| `src/qt_pinn/qnn_generator_ns.py` | Legacy constant-input QG (Tier 0) |
| `src/qt_pinn/qnn_generator_cond.py` | v1 `reupload`, v2 `expect` conditioned generators |
| `scripts/train_ns_direct.py` | Tier A direct baseline |
| `scripts/train_ns_parametric.py` | Tier B TGV family training |
| `scripts/compare_ns_parametric.py` | TGV side-by-side + verdict |
| `pdes/kolmogorov2d/` | **Tier C** Kolmogorov PDE + config |
| `scripts/train_kol_direct.py` | Kolmogorov direct baseline |
| `scripts/train_kol_parametric.py` | Kolmogorov parametric QC vs classical |
| `scripts/compare_kol_parametric.py` | Kolmogorov compare (PDE RMS) |

---

## Recommended next steps

1. **Archive TGV + Kolmogorov hypernetwork QC** — null result is established across easy (TGV) and hard (Kolmogorov) PDEs.
2. **If continuing quantum PINNs:** change the research question — e.g. input-conditioned circuits at \((x,y,t)\), or QC as optimizer (QNG/VQE on weights), not \(\nu \to\) flat weight vector.
3. **If continuing classical PINNs only:** v1 TGV classical (`ns_par_c_s0`) remains best extrapolator; Kolmogorov direct (`kol_direct_s0`) is the field-network reference.
4. **Optional:** tighten Kolmogorov extrap-hi grid or add spectral-DNS reference for qualitative validation.
5. **Do not** compare constant-input quantum generators — invalid experiment (Tier 0).

---

*Last updated: 2026-08-22. Tiers 0–C complete. Null result: classical hypernetwork beats VQC-hypernetwork on TGV and Kolmogorov.*
