# Blog draft: Evaluating quantum circuits in Physics-Informed Neural Networks

> **Status:** methodology + working 2D NS merger solver + closed fair-advantage null + TGV media (v5). Numbers through **2026-08-26**.
>
> **Companion:** [MODEL_CARD.md](MODEL_CARD.md) (best classical / quantum metrics). Checkpoints: [checkpoint/README.md](checkpoint/README.md).
>
> **Working title options** (pick one):
> 1. *A PINN PDE solver for vortex merger: classical and quantum-trained nets, same accuracy, faster inference*
> 2. *From failed VQC-PINNs to a working NS solver — without claiming quantum advantage*
> 3. *Debugging quantum PINNs, then solving a real 2D flow*

---

## 0. One-sentence thesis

**Narrow claim (recommended):** Input-conditioned VQCs and ν-hypernetworks did **not** beat matched classical models on Burgers / TGV / Kolmogorov. The project then produced a working PINN solver for 2D vortex merger (**ω ≤ 2%** vs spectral DNS). In a **fair** end-to-end test, a quantum weight generator does **not** reliably beat a matched classical generator at equal deployed latency (Q wins 2/6 seeds; classical mean slightly better). The earlier v4 “2.5× faster quantum” product checkpoint was an inject-only no-op (**circuit unused**); the speedup is a smaller MLP.

**Do not claim:** “A variational quantum circuit evaluates Navier–Stokes at query time,” swirl/orbit fidelity on product weights, or that every QNN hyperparameter wins.

---

## 0.1 Where we are now (2026-08-26)

| Layer | State |
|-------|-------|
| **Product merger solvers** | Classical HarmMLP 96–96: **ω 1.29%**. Quantum inject h48: **ω 1.78%**, ~2.5× faster inference. Shipped under `checkpoint/v4/{classical,quantum,dns,media}`. |
| **Fair QT advantage** | **Null.** Closed; evidence in `v4/archive/advantage_scoreboard.md`. |
| **Orbit / swirl co-rotation** | **Not promoted.** Pointwise ω can pass while peaks freeze; orbit Fourier helps but Rel≤2% *and* swirl≤5% never landed in one product checkpoint. |
| **TGV media (v5)** | Dense \|ω\| contours + Exact\|Classical\|Quantum triplet + **unstable** TGV (bottom-left boost breaks exact balance). |
| **Blog** | This draft + model card. Hero still: merger triplet gif. |

---

## 0.2 What worked vs what did not

### Worked
| Method | Why it mattered |
|--------|-----------------|
| Hard IC \(u = u_{IC} + t N\) | Killed soft-IC collapse to the trivial zero field |
| Harmonic Fourier features (merger) | Hit FD-curl ω ≤ 2% where plain RFF / streamfunction failed |
| FP32 curl gate on fixed DNS times | Stops “looks good in MSE, wrong vorticity” |
| Distill → small HarmMLP 48–48 | Same ω band, ~2.5× inference throughput |
| Degeneracy guards (`collapse_ratio`, `correction_rms`, circuit ablation) | Invalidated fake quantum wins |
| Integrating-factor RK4 for stiff TGV DNS | Explicit viscous RK4 blew up at ν~0.1 |
| `orbit_omega` Fourier terms | Unfroze early peak motion (research path; not product) |

### Did not work / do not claim
| Attempt | Outcome |
|---------|---------|
| ν-hypernetwork VQC (TGV, Kolmogorov) | Classical wins; quantum extrap often catastrophic |
| Soft-IC Burgers VQC | Fake quantum win via harder collapse |
| Hard-IC Burgers input VQC | Classical learns; quantum freezes near IC |
| Constant-input quantum generator | No functional quantum computation |
| v4 inject “quantum speedup” | Circuit unused; classical distill matches ω |
| Fair e2e QNN vs classical gen (Exp A) | No robust advantage (mean +0.10 pp against Q) |
| Multi-ν family generators (Exp B) | Both arms ~33%+ ω — not solvers |
| Orbit-gated product merger | Rel / swirl tradeoff; not shipped |
| Dense FD polish at n_side=96 near gate | Blew up near-converged models |

---

## 1. Audience & tone

- Audience: ML + scientific computing practitioners (not QC hype)
- Tone: engineering lab notebook → polished narrative
- Honesty: lead with methodology and failure modes; merger solver is the positive product story; QT advantage is a **null**

---

## 2. Article outline (sections to write)

### Hook
- Motivation: PINNs + VQCs as a popular “quantum for science” story
- Question asked: *Can a variational quantum circuit beat a matched classical model on a PDE residual?*

### Setup (what we mean by fair)
- Same collocation / IC / steps / λ when soft IC
- Matched parameter counts (or classical ≥ quantum)
- Same encoding features where relevant (angles → sin/cos for classical)
- Primary metrics defined up front
- Degeneracy guards: collapse to \(u=0\), freeze at IC

### Experiment ladder
1. Hypernetwork path (ν → MLP weights) — TGV, then Kolmogorov — **null**
2. Architecturally valid path — input-conditioned VQC as \(f(x,y,t)\) on Burgers — **null / collapse**
3. Soft IC → hard IC after collapse artifact
4. **PDE solver (the actual goal):** 2D vortex merger vs spectral DNS — classical + quantum-trained PINNs
5. Fast deployed net (v4): same accuracy band, ≥2× inference — **size**, not circuit
6. Fair e2e generators — **null advantage**
7. (Coda) TGV return + unstable perturbation media (v5)

### Bugs that invalidate early results
### Results tables (+ model card)
### Media plan (§11a)
### Why quantum lost (measurement vs structural)
### Cost realism (simulator wall time)
### What would count as a future positive result
### Conclusion

---

## 3. Framing: research question evolution

| Stage | Question | Outcome |
|-------|----------|---------|
| Early | Can a quantum weight generator solve NS? | Invalid (bugs + constant-input circuit) |
| Tier A | Can a classical target MLP solve NS at all? | Yes (~2–4% velocity error) |
| Tier B | Does ν → weights with a VQC beat a matched MLP? | No (TGV) |
| Tier C | Same on harder forced NS (Kolmogorov)? | No |
| Burgers VQC | Does input-conditioned VQC beat matched MLP as \(f(x,y,t)\)? | Soft IC: fake win via collapse; hard IC: classical wins, quantum freezes |
| Merger v3 | Can a PINN hit FD-curl ω ≤ 2% vs DNS? | **Yes** — HarmMLP 96–96, k≤6 (**1.29%**); QT distill same net |
| Merger v4 product | Same band, ≥2× inference | **Yes** size win (1.78% ω, ~2.5×) — circuit unused |
| Merger v4 fair | Does e2e QNN beat matched classical gen? | **No** (Q 2/6; mean favors classical) |
| Orbit fidelity | Pointwise ω *and* peak co-rotation? | Partial; **not promoted** |
| v5 TGV | Sharper \|ω\| media + unstable boost | Media / demo (not an advantage claim) |

**Architectural insight:** Hypernetworks only see ν; the circuit never evaluates the field. Input-conditioned VQC-PINN is the first setup where the circuit *is* the solution map.

---

## 4. Methodology checklist (copy into article)

### Fairness
- [x] Matched (or classical ≥) trainable parameters
- [x] Same domain, ν, collocation protocol
- [x] Fixed holdout grids (not “resampled train range”)
- [x] Soft vs hard IC explicit
- [x] Report wall time, not only steps

### Metrics
| Problem | Primary metric | Secondary |
|---------|----------------|-----------|
| TGV (exact soln) | Space-time / gauge-aware rel-L2; in-range vs extrap ν | Wall time |
| Kolmogorov (no exact) | Holdout PDE residual RMS | Extrapolation in ν |
| Burgers VQC | Holdout PDE RMS under **hard IC** | `collapse_ratio`, `correction_rms` |

### Degeneracy guards (Burgers)
| Diagnostic | Meaning | Fail if |
|------------|---------|---------|
| `collapse_ratio` = field_rms(t=1) / IC_rms | Near-zero field | < 0.1 |
| `correction_rms` under hard IC \(u = u_{IC} + t N\) | Learned time evolution | ≪ 0.01 × IC_rms (freeze) |
| Soft IC `bc_loss ≈ 0.5` | Matches \(u=v=0\) predictor | Plateau at ~0.5 with falling PDE → collapse |

**Analytic check:** frozen IC (\(N=0\)) on Burgers has PDE RMS ≈ **1.56** (advection-dominated). Quantum hard-IC scout landed at **1.63** — functionally frozen.

### Success criteria used in this project
- Quantum win: primary metric ≤ 90% of classical, *and* non-degenerate
- Parity: within ~10%
- Else: classical wins / null result
- Do **not** promote soft-IC PDE RMS alone

---

## 5. Critical bugs (invalidate early “results”)

Use these as a “debugging quantum ML” section — high value for a professional blog.

### 5.1 Pressure sign inconsistency (TGV)
- **Symptom:** `bc_loss ≈ 0.56`, low PDE loss, ~100% rel-L2 → collapse to \(u=v=p=0\)
- **Cause:** Exact pressure sign disagreed with velocity convention → coded residual ≠ analytic
- **Fix:** \(p = +\frac14(\cos 2x + \cos 2y)\,e^{-4\nu t}\)
- **Lesson:** Low PDE residual can mean “learned the zero field,” not “solved the PDE”

### 5.2 Dead time channel (TGV Fourier map)
- **Symptom:** Adding “time features” changed nothing (byte-identical runs)
- **Cause:** \(t\) row of Fourier matrix was all zeros
- **Fix:** Explicit \(t/T\) (and optionally \(t^2/T^2\)); TGV decay is not periodic in \(t\)

### 5.3 Easy regime hides time dependence
- Default \(\nu=0.01, T=1\) → amplitude \(\approx e^{-0.02}\approx 0.98\); ignoring time is nearly optimal
- Harder regime example: \(\nu=0.1, T=5\)

### 5.4 Constant-input quantum generator
- Circuit saw a fixed vector → one weight vector forever
- Equivalent to classical lookup of static weights; **no functional quantum computation**
- Invalidates Tier-0 “quantum generator” advantage claims

### 5.5 Soft IC + exact trivial solution (Burgers)
- \(u=v=0\) solves Burgers with zero residual
- Soft IC with moderate \(\lambda_{bc}\) → basin of “be zero”
- Scout: both models sat at `bc ≈ 0.5` (= MSE of IC vs zero); quantum looked better only because it collapsed *harder*
- **Fix:** hard IC \(u = u_{IC} + t\,N\)

### 5.6 Encoding / frequency bugs (Burgers VQC)
- Double-\(\pi\) scaling on frequencies (awkward non-harmonics of IC)
- Even time frequencies alias \(t=0\) with \(t=1\) over \([0,1]\)
- Fix: dimensionless ladder; spatial even freqs for IC fundamental; **odd** freqs for time

### 5.7 Evaluation pitfalls (general)
| Pitfall | Effect |
|---------|--------|
| Rel-L2 only at final \(t\) with tiny amplitude | Explosive % error |
| “Holdout” = another i.i.d. draw of train range | Fake generalization |
| Equal step count while quantum still descending | Unfair |
| Linear ν encoding + log-uniform train ν | Extrapolation off-manifold |
| Geometric \(2^i\pi\) RY ladder | Aliasing / erratic errors |

---

## 6. Results: Tier A–C (hypernetworks)

Source of truth for NS: `pdes/ns2d/EXPERIMENT_ANALYSIS.md`.

### Tier A — Direct classical PINN (sanity gate)
| Run | Headline |
|-----|----------|
| `ns_direct_a3` | ~2.7% mean rel-L2 @ \(t=1\) after physics fix — **gate pass** |
| `ns_direct_a3_time` | Identical → proved dead time channel |
| `a3_nu0p1_T5` | Harder regime; velocity ~4% @ \(t=5\) |
| `kol_direct_s0` | Kolmogorov PDE RMS **0.00246** — learnable |

**Point for article:** Quantum must beat a working classical baseline (~2–4%), not “better than 100% collapse.”

### Tier B v1 — TGV parametric (\(\nu \mapsto\) weights), `reupload`
| Split | Classical `ns_par_c_s0` | Quantum `ns_par_q_s0` | Q/C |
|-------|-------------------------|------------------------|-----|
| in-range | **1.33%** | 1.72% | 1.29 |
| extrap-lo | **2.09%** | 49.0% | **23.4** |
| Wall time | 1182 s | 2146 s | 1.82× |

**Verdict:** Classical wins. Quantum interpolates OK, extrapolates catastrophically.

### Tier B v2 — `expect` + log-ν
| Split | Classical v2 | Quantum v2 | Q/C |
|-------|--------------|------------|-----|
| in-range | 1.99% | 2.21% | 1.11 |
| extrap-lo | 31.0% | 31.3% | 1.01 |

Fixed quantum extrap-lo erraticism but **hurt** classical vs v1. Best classical overall remains **v1**. Still no QC win (extrap Q/C ≈ 1.58 vs best classical framing).

### Tier C — Kolmogorov forced NS
| Split | Classical `kol_par_c_s0` | Quantum `kol_par_q_s0` | Q/C |
|-------|--------------------------|------------------------|-----|
| in-range PDE RMS | **0.00288** | 0.00534 | 1.85 |
| Combined extrap | — | — | **1.50** |
| Wall time | 1145 s | 2119 s | 1.85× |

**Verdict:** Same null on a harder PDE with sustained nonlinearity. Hypernetwork VQC line closed.

---

## 7. Results: Burgers input-conditioned VQC-PINN

### Architecture
```
(x,y,t) → angle encode → re-uploading VQC → ⟨Z_i⟩ → linear → (u,v)   [~92 params]
(x,y,t) → same angles → sin/cos → tanh MLP → (u,v)                 [~98 params]
```
Domain: \(x,y\in[-1,1]\), \(t\in[0,1]\), \(\nu=0.01/\pi\approx 0.003183\).

Hard IC (default after collapse discovery):
\[
u = u_{IC}(x,y) + t\,N_u,\quad
u_{IC}=\sin(\pi x)\cos(\pi y),\quad
v_{IC}=-\cos(\pi x)\sin(\pi y).
\]

Scripts: `scripts/train_burgers_vqc.py` (`--preset scout|full`), `compare_burgers_vqc.py`.
Model: `src/qt_pinn/burgers_vqc_pinn.py`.

### Soft IC scout (INVALID — do not report as a win)
| | Classical | Quantum |
|--|-----------|---------|
| Holdout PDE RMS | ~0.21 | ~0.018 |
| `bc` plateau | ~0.50 | ~0.50 |
| Field | near zero | nearer zero |

**Interpretation:** Quantum “won” by collapsing harder to the trivial solution.

### Hard IC scout (VALID null)
Checkpoints: `burg_vqc_c_scout`, `burg_vqc_q_scout` (400 steps, 512 colloc, ~12–13 min quantum).

| Metric | Classical | Quantum | Frozen IC (\(N=0\)) |
|--------|-----------|---------|---------------------|
| Holdout PDE RMS | **0.799** | 1.631 | **1.563** |
| `collapse_ratio` | 0.67 | 0.996 | 1.0 |
| `correction_rms` | 0.57 | **0.024** | 0 |
| Wall time | 4 s | 732 s | — |
| Q/C | — | **2.04** | — |

**Behavior:**
- Classical: `corr_rms` rises; field decays — learning viscous/advective evolution
- Quantum: `corr_rms` falls 0.48 → 0.024; plateaus at frozen-IC residual from ~step 125

**Decision:** Do **not** run `--preset full` (~60 h) on this architecture. Failure is basin / expressivity, not undertraining.

### Cost note (why quantum “trains forever”)
- PennyLane `default.qubit` is CPU; `.cpu()` on angles/weights each forward
- PINN needs **second** autograd through the circuit
- Scout ~7 s/step at 512 points; full 2048×8000 was multi-day territory
- Classical 98-param MLP: milliseconds/step on GPU

---

## 8. Structural reasons quantum lost (for discussion section)

### Hypernetwork path
1. Map \(\nu \to \sim 1500\) weights is a classical-friendly encoder problem
2. Circuit never sees \((x,y,t)\)
3. TGV family is nearly linear diffusion (advection/pressure cancel) — weak stress test for entanglement
4. Simulation cost 1.8–2× with worse accuracy

### Input-conditioned Burgers VQC
1. Linear head on \(\langle Z\rangle\) + small re-uploading circuit ≈ low-order trigonometric features
2. Hard IC residual needs a **second harmonic** of the IC (advection); MLP mixes sin/cos and can form it; VQC froze at \(N=0\)
3. \(N=0\) is a critical point of a small linear head — attractive basin
4. Matched param count does not imply matched *useful* function class after quantum constraints

---

## 9. The actual goal: a PDE solver for 2D vortex merger

Earlier sections are about **failed** quantum architectures. This section is the product: a PINN that reconstructs a **four same-sign vortex merger** against spectral DNS (no closed form) — and a **fair** test of whether an end-to-end quantum weight generator beats a matched classical generator.

**Problem.** Periodic box \([0,2\pi]^2\), \(\nu=0.005\), four co-rotating Gaussian vortices → one core by \(t\sim 15\). Reference: 256² spectral DNS in `blog/checkpoint/v3/dns/`. Gate times \(t\in\{0,2,5,8,12,15\}\).

**Metric.** Finite-difference curl \(\omega = \partial v/\partial x - \partial u/\partial y\) on a subsampled grid, **FP32 only**. Target for the product solver: max relative L2 of ω vs DNS **≤ 2%**.

**Hero media (no tracers):** `blog/checkpoint/v4/media/merger_triplet.gif`.

### 9.1 Product checkpoint (v4 deployed solver)

| | Classical teacher | v4 “quantum” inject |
|--|-------------------|---------------------|
| Deployed net | HarmMLP 96–96, \(k\le 6\) | HarmMLP **48–48**, \(k\le 3\) |
| Params | 13 347 | **3 795** |
| ω max | **1.29%** | **1.78%** |
| Throughput | ~326 Mpts/s | **~2.5×** |

**Correction:** the v4 inject path zeroed the generator projection weight and copied a classically distilled student into the bias. The circuit did **not** contribute. A classical distill of the same h48k3 hits **1.75%** (`classical_h48_baseline/`). That “2.5× faster quantum” claim was architecture size, not quantum computation.

### 9.2 Fair quantum advantage (Experiment A)

**Protocol.** Train `ConditionedQuantumGeneratorV2` end-to-end (circuit + proj get gradients) vs `ConditionedClassicalGeneratorV2` at matched `n_qubits` / bottleneck. Both emit weights for the **same** deployed HarmMLP 48–48, \(k\le 3\). Require circuit ablation (randomize `q_weights` / zero feats) to degrade ω.

**Multi-seed (q=8, L=4, bn=64; seeds 0–5, deduped):**

| | Quantum | Classical-gen |
|--|---------|---------------|
| Mean ω | 2.325% ± 0.38% | **2.220% ± 0.24%** |
| Seed wins | 2/6 | **4/6** |
| Best seed | 1.895% (s1) | 1.987% (s1) / long **1.823%** |

Δ mean **+0.10 pp against quantum**. Circuit ablation still passes on quantum runs. Other sweeps (bn16, V1 probs, long+curl×2) likewise favor classical or are within noise; full tables: `blog/checkpoint/v4/archive/advantage_scoreboard.md`.

**Claim (honest).** On this task we **did not** find a robust quantum advantage. End-to-end QNNs can run with a live circuit at matched deployed latency, but they do not reliably beat matched classical generators on ω. v4’s earlier “2.5× faster quantum” story was an inject/no-op artifact.

### 9.3 Experiment B (ν-family) — null

Multi-ν DNS family (ν∈{0.002…0.02}), hold out ν=0.008. Both quantum and classical generators plateau ~**33–50%** train ω_mean (≫2% gate) — useless as solvers. Mid-range + teacher variants do not rescue either arm. Not used for an advantage claim.

### 9.4 Orbit / swirl campaign — closed, not promoted

Pointwise Rel-L2 can look fine while vorticity **maxima freeze** (DNS peaks keep orbiting). Adding `orbit_omega≈−1.22` Fourier features unfroze early motion; full-horizon swirl ≤5% with Rel≤2% was **not** achieved in one promoted checkpoint (best compromise ~1.9% Rel / ~7% swirl). Product `v4/classical` and `v4/quantum` remain the pre-orbit inject pair. Details: `v4/archive/orbit_fidelity/NOTES.md`.

### 9.5 What we tried (ω campaign + advantage)

| Approach | Outcome |
|----------|---------|
| Pointwise \((u,v)\)+curl, streamfunction, uvpw, wide RFF | Did not hit 2% ω |
| HarmMLP 96–96, \(k\le 6\) | **1.29%** classical teacher |
| Distill h48k3 + inject into QNN (v4) | 1.78% / ~2.5× faster — **circuit unused** |
| Classical distill h48k3 (Control C) | **1.75%** — explains v4 |
| End-to-end QNN vs matched classical gen (A) | **No advantage** (Q wins 2/6; mean +0.10 pp vs Q) |
| ν-family generators (B) | Both fail ~33%+ ω |
| Orbit-gated Rel + swirl | Partial; **not product** |

Reproduce:
```bash
.venv/bin/python scripts/exp_quantum_advantage_A.py --arm quantum --n-qubits 8 --n-layers 4 --steps 20000
.venv/bin/python scripts/exp_quantum_advantage_A.py --arm classical_gen --n-qubits 8 --steps 20000
```

### 9.6 v5 — Taylor–Green return (media)

Stable TGV is a weak stress test (advection cancels into pressure). v5 uses it for **visual** fidelity:

- Dense \|ω\| (k=2): `checkpoint/v5/media/tgv_dense.gif`
- Exact \| Classical \| Quantum (k=1): `tgv_triplet.gif`
- **Unstable:** amplify bottom-left lobe so balance breaks → nonlinear interaction: `v5/unstable/media/tgv_unstable_triplet.gif`

Not an advantage claim — a demo that PINNs can follow DNS when the exact solution no longer holds.

## 10. Optional Fourier-toy figure (optional; merger gif is the hero)

Goal: one clean “quantum can do something” plot without claiming PDE superiority.

### Candidate A — Fourier regression (already sketched)
- Target: \(f(x)=\sin(4x)+\sin(8x)\)
- Script: `scripts/sanity_vqc_regression.py`
- Prior run: Q/C ≈ **1.01** (parity) — weak as a “win,” OK as “competitive on Fourier-native toy”
- [ ] Re-run with fixed seed, report RMSE + params + wall time
- [ ] Figure: prediction overlay + residual

### Candidate B — Sample-efficiency story (not run)
- Same toy or 1D linear PDE; vary # training points
- Claim shape: “at N points, quantum matches classical” — only if data supports it

### Candidate C — Skip positive figure
- Pure methodology / negative-result article is still publishable if debugging narrative is strong

**Recommendation:** Prefer skip for Fourier toy — the **merger gif** (§9) is the positive figure. Do not invent a circuit-in-the-loop PDE win.

---

## 11. What to put in figures / tables (production checklist)

**Hero (shipped):** 1×3 vortex-merger vorticity — DNS | classical | quantum-trained. `blog/checkpoint/v4/media/merger_triplet.gif` (+ snapshots). Gate: FD-curl ω ≤ 2% at listed times. **No tracer particles.**

- [x] Fig: merger triplet gif / snapshots (v4)
- [x] Table: best classical vs quantum — see [MODEL_CARD.md](MODEL_CARD.md)
- [x] Table: ω-campaign methods — worked vs failed (§0.2, §9)
- [x] Fig: TGV dense + triplet + unstable (v5)
- [ ] Fig: experiment ladder (hypernet → Burgers VQC → merger → fair null)
- [ ] Table: critical bugs + symptom → fix
- [ ] Fig: Burgers soft-IC collapse vs hard-IC trajectories
- [ ] Box: “When not to trust PDE residual”
- [ ] Box: “Quantum-trained ≠ circuit at inference”

---

## 11a. Media placement plan (for the published post)

Keep assets versioned under `blog/checkpoint/`; the article embeds relative paths (or copies into a static `blog/assets/` at publish time).

| Slot in article | Asset | Path |
|-----------------|-------|------|
| **Hero / after thesis** | Merger triplet GIF (DNS \| Classical \| Quantum) | `checkpoint/v4/media/merger_triplet.gif` |
| Hero still (OG / thumbnail) | Merger snapshots strip | `checkpoint/v4/media/merger_snapshots.png` (or first frame) |
| Debugging: collapse | Soft-IC Burgers field → near zero (optional still) | regenerate from scout checkpoints |
| Debugging: TGV polish | Exact vs models at late \(t\) | `checkpoint/v2/media/` |
| Product metrics callout | Link / inline table | [MODEL_CARD.md](MODEL_CARD.md) |
| Fair advantage null | Small bar/table (mean ω Q vs C) | numbers from `v4/archive/advantage_scoreboard.md` |
| Coda: stable TGV beauty | Dense \|ω\| decay | `checkpoint/v5/media/tgv_dense.gif` |
| Coda: solver comparison | Exact \| Classical \| Quantum | `checkpoint/v5/media/tgv_triplet.gif` |
| Coda: when balance breaks | Unstable BL-boost triplet | `checkpoint/v5/unstable/media/tgv_unstable_triplet.gif` |
| Caption footnotes | Per-folder `CAPTION.md` | next to each gif |

**Layout suggestion**
1. Open with merger gif + one-sentence claim (solver works; QT advantage null).
2. Mid-article: one bug figure (pressure sign or soft-IC collapse) + hypernetwork null table.
3. Solver section: model-card table + merger stills.
4. Close with v5 TGV strip (stable → unstable) as “what the physics looks like,” not as a QC win.

**Do not** put tracer particles on merger gifs; **do** mark the boosted center (`+`) on unstable TGV.

---

## 12. Recommended conclusion bullets (draft)

1. Fair evaluation of quantum PINNs requires matched capacity, fixed holdouts, and **degeneracy checks**.
2. Several attractive early results were invalidated by physics bugs or trivial solutions (zero field, frozen IC, constant-input generators).
3. On TGV and Kolmogorov, VQC hypernetworks underperformed matched classical generators (~1.5–2× error, ~2× slower **training**).
4. On input-conditioned Burgers, soft IC produced a fake quantum win; hard IC showed classical better, quantum frozen near IC.
5. The **goal that landed** is a PINN PDE solver for 2D vortex merger: HarmMLP vs spectral DNS, FD-curl ω **≤ 2%** (best classical **1.29%**).
6. Fair end-to-end QNN vs matched classical **generators** (same deployed 48–48 net): **no robust quantum advantage** (Q wins 2/6 seeds; classical mean slightly better).
7. v4’s “~2.5× faster quantum” result was an **inject/no-op** (proj weight zeroed); classical distill of the same small net matches it (**1.75%** vs **1.78%**).
8. Orbit/swirl co-rotation remains open: product weights still show frozen late peaks vs DNS.
9. What worked for the solver: harmonic features, FP32 curl gate, distillation. What did not: claiming quantum advantage from unequal architectures or unused circuits.

---

## 13. What not to write / common traps

- [ ] Don’t headline soft-IC Burgers quantum PDE RMS
- [ ] Don’t compare constant-input generators as “quantum computing”
- [ ] Don’t claim advantage from equal steps when quantum is 100–800× slower per step
- [ ] Don’t use TGV alone as “proof of quantum advantage” (too easy / linear)
- [ ] Don’t promote v2 classical regression as a QC win because quantum matched a *worse* classical
- [ ] Don’t run 60 h full Burgers VQC without a new architecture that already passes scout
- [ ] Don’t say the **circuit** is 2.5× faster at PDE queries — the **deployed MLP** is
- [ ] Don’t treat v3 QT as a speed win (same 13k-param net as classical)
- [ ] Don’t imply a physics-only PINN; merger training uses DNS
- [ ] Don’t put tracer particles back on the merger gif (misleading if DNS-only)

---

## 14. Future experiments (only if continuing research)

Priority order if the blog needs a follow-up series:

1. **Tighten freeze guard** — flag `correction_rms` near frozen-IC residual floor, not only 0.01×ic_rms
2. **Richer quantum head** — nonlinear MLP on \(\langle Z\rangle\) with ablation vs classical-only head
3. **Explicit 2nd-harmonic features** in encoding (align circuit with Burgers advection residual)
4. **1D Burgers or linear heat** with hard IC — smaller, faster conclusive scout
5. **Sample-efficiency** protocol (vary collocation N at fixed params)
6. Archive ν-hypernetwork VQC line unless architecture changes fundamentally
7. Merger follow-ups: physics-only (no DNS collocation) at the 2% ω gate; true circuit-in-the-forward-pass if that is the scientific claim; time-varying \(k_{\max}\) / adaptive width without losing the 0.5 pp budget

---

## 15. Repo map (for footnotes / “reproduce”)

| Path | Role |
|------|------|
| `pdes/ns2d/EXPERIMENT_ANALYSIS.md` | Full NS measurement notes |
| `pdes/ns2d/physics_loss.py` | TGV residuals / exact / gauge |
| `pdes/kolmogorov2d/` | Forced NS Tier C |
| `pdes/burgers2d/physics_loss.py` | Burgers residual + `ic_values` |
| `src/qt_pinn/qnn_generator_cond.py` | Conditioned generators v1/v2 |
| `src/qt_pinn/burgers_vqc_pinn.py` | Input-conditioned VQC + matched classical |
| `docs/PLAN_TGV_DEMO.md` | 3-panel TGV animation + quality-model plan |
| `scripts/train_ns_parametric.py` | TGV family |
| `scripts/train_kol_parametric.py` | Kolmogorov family |
| `scripts/train_burgers_vqc.py` | Burgers scout/full |
| `scripts/compare_*.py` | Side-by-side verdicts |
| `scripts/sanity_vqc_regression.py` | Fourier toy |
| `scripts/train_vortex_merger.py` | Merger PINN training entry |
| `scripts/exp_merger_omega.py` | HarmMLP, FD-curl gate, v3 media |
| `scripts/train_merger_qt_fast.py` | v4 distill + inject |
| `scripts/bench_merger.py` | Classical vs QT throughput / ω |
| `scripts/regenerate_v4_media.py` | v4 gif + snapshots (no tracers) |
| `blog/checkpoint/v3/` | DNS + matched-size classical/QT |
| `blog/checkpoint/v4/` | Product + bench + media; fair-advantage/orbit under `archive/` |
| `blog/checkpoint/v5/` | TGV dense / triplet / unstable media |
| `blog/MODEL_CARD.md` | Best classical & quantum metrics |

### Reproduce Burgers scout
```bash
.venv/bin/python scripts/train_burgers_vqc.py --model classical --seed 0 --run-id burg_vqc_c_scout
.venv/bin/python scripts/train_burgers_vqc.py --model quantum  --seed 0 --run-id burg_vqc_q_scout
.venv/bin/python scripts/compare_burgers_vqc.py checkpoints/burg_vqc_c_scout checkpoints/burg_vqc_q_scout
```

### Key checkpoints
| ID | Role |
|----|------|
| `ns_direct_a3` | Classical NS works |
| `ns_par_c_s0` / `ns_par_q_s0` | Best TGV classical; quantum extrap fail |
| `ns_par_*_v2_s0` | Expect/log redesign |
| `kol_direct_s0`, `kol_par_c_s0`, `kol_par_q_s0` | Forced NS null |
| `burg_vqc_c_scout`, `burg_vqc_q_scout` | Hard-IC Burgers null (valid) |
| Soft-IC Burgers scouts | **Discarded / invalid** — collapse |
| `blog/checkpoint/v3/classical` | Merger HarmMLP 96–96 (1.29% ω) |
| `blog/checkpoint/v4/classical` | Same teacher (product) |
| `blog/checkpoint/v4/quantum` | Fast HarmMLP 48–48 inject (~2.5×; circuit unused) |
| `blog/checkpoint/v4/archive/advantage_*` | Fair e2e null |
| `blog/checkpoint/v5/unstable/` | Strong BL-boost TGV triplet |

---

## 16. Open writing TODOs for the author

- [ ] Choose final title
- [ ] Write intro narrative (problem → failed VQC paths → working merger solver)
- [ ] Lead with merger solver + v4 speed; keep Burgers/TGV as “how we almost shipped a fake win”
- [ ] Decide how much of the hypernetwork era to keep vs compress
- [ ] Compress Tier B/C tables for readability (one “hypernetwork era” figure)
- [ ] Draft “how to evaluate quantum PINNs” box from §4
- [ ] Add author bio / disclosure (simulator-only; no hardware)
- [ ] Cite PennyLane, PINN literature, related QC-for-PDE papers (fair related work)
- [ ] Proofread that no soft-IC Burgers number appears as a success

---

## 17. Changelog of this document

| Date | Note |
|------|------|
| 2026-08-24 | Initial structure from experiment log + Burgers hard-IC scout + blog strategy notes |
| 2026-08-25 | Added §9 vortex-merger PDE solver (v3 2% ω gate, v4 ~2.5× QT inference); updated thesis/conclusion so they match shipped checkpoints |
| 2026-08-26 | Corrected fair-advantage thesis to **null**; added §0.1–0.2 status + worked/failed; orbit close-out; v5 TGV + unstable media plan (§11a); linked [MODEL_CARD.md](MODEL_CARD.md) |

*Last evidence cutoff: merger v4 product + fair-advantage archive, Burgers hard-IC scout, NS Tiers 0–C, v5 TGV media. Product status: working DNS-gated merger PINN; no robust quantum advantage; orbit fidelity not promoted.*
