# Blog draft: Evaluating quantum circuits in Physics-Informed Neural Networks

> **Status:** methodology draft **plus** a working 2D Navier–Stokes solver (vortex merger). Numbers through 2026-08-25.
>
> **Working title options** (pick one):
> 1. *A PINN PDE solver for vortex merger: classical and quantum-trained nets, same accuracy, faster inference*
> 2. *From failed VQC-PINNs to a working NS solver — and why the quantum path is faster*
> 3. *Debugging quantum PINNs, then solving a real 2D flow*

---

## 0. One-sentence thesis

**Narrow claim (recommended):** Input-conditioned VQCs and ν-hypernetworks did **not** beat matched classical models on Burgers / TGV / Kolmogorov. The project then **did** produce a working PINN solver for 2D vortex merger (ω ≤ 2% vs spectral DNS). At the same vorticity accuracy, a quantum-**trained** (distilled) deployed net is **~2.5× faster** at inference because it is a smaller HarmMLP — not because a circuit runs at query time.

**Do not claim:** “A variational quantum circuit evaluates Navier–Stokes faster than a GPU MLP.” Inference is a classical MLP whose weights were produced by a QNN generator + distillation.

---

## 1. Audience & tone

- [ ] Audience: ML + scientific computing practitioners (not QC hype)
- [ ] Tone: engineering lab notebook → polished narrative
- [ ] Honesty: lead with methodology and failure modes; one limited positive figure is optional balance, not the headline

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
5. Fast quantum-trained deployed net (v4): same accuracy band, ≥2× inference throughput

### Bugs that invalidate early results
### Results tables
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
| Merger v3 | Can a PINN hit FD-curl ω ≤ 2% vs DNS? | **Yes** — HarmMLP 96–96, k≤6 (1.29%); QT via weight distill (same net) |
| Merger v4 | Same accuracy band, ≥2× inference throughput | **Yes** — QT HarmMLP 48–48, k≤3: ω 1.78% vs 1.29% (+0.49 pp), **~2.5×** pts/s |

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

Earlier sections are about **failed** quantum architectures. This section is the product: a PINN that reconstructs a **four same-sign vortex merger** against spectral DNS (no closed form).

**Problem.** Periodic box \([0,2\pi]^2\), \(\nu=0.005\), four co-rotating Gaussian vortices → one core by \(t\sim 15\). Reference: 256² spectral DNS in `blog/checkpoint/v3/dns/`. Gate times \(t\in\{0,2,5,8,12,15\}\).

**Metric (not “PDE residual looks small”).** Finite-difference curl \(\omega = \partial v/\partial x - \partial u/\partial y\) on a subsampled grid, **FP32 only** (TF32 inflated the same checkpoint from 1.29% to 2.38%). Target: max relative L2 of ω vs DNS **≤ 2%**. Velocity error is secondary.

**Hero media (no tracer particles):** `blog/checkpoint/v4/media/merger_triplet.gif` — DNS | Classical | Quantum vorticity, \(t=0..15\), Δt=0.05, 20 fps. Snapshots: `merger_triplet_snapshots.png` (streamlines on stills only).

### Headline result (v4)

| | Classical | Quantum-trained |
|--|-----------|-----------------|
| Deployed net | HarmMLP 96–96, harmonic \(k\le 6\) | HarmMLP **48–48**, \(k\le 3\) |
| Params | 13 347 | **3 795** (~3.5× smaller) |
| ω rel-L2 max (FP32) | **1.29%** | **1.78%** (Δ **+0.49 pp**) |
| vel rel-L2 max | 3.75% | 2.95% |
| Throughput @ 262 144 pts | ~326 Mpts/s | **~829 Mpts/s (~2.54×)** |
| Gate ω ≤ 2% | pass | pass |
| Gate ω within 0.5 pp of classical | — | pass |
| Gate ≥2× throughput | — | pass |

Checkpoints: `blog/checkpoint/v4/classical/`, `v4/quantum/`. Numbers: `v4/bench.json`. v3 kept a **matched-size** QT copy of the 96–96 net (same 1.29% ω, not faster).

**How to read “quantum is faster.”** Query-time evaluation is a **classical** harmonic MLP. The QNN (`ConditionedQuantumGeneratorV2`) emits those weights (distill + inject into projection bias). Speed comes from **fewer deployed weights**, not from running a circuit on every \((x,y,t)\). That is still a valid product claim if the training path is quantum-trained; it is not a claim about quantum hardware latency.

### How the solver is trained

1. Spectral DNS is **read-only** teacher / gate.
2. Classical: train HarmMLP on DNS + physics (velocity, curl, IC). Winner: continue from a good IC (`E27_harm_cont`).
3. Quantum v3 (`E31_qnn_distill`): copy those weights into the generator so the deployed QT PINN **matches** classical (accuracy demo, not speed).
4. Quantum v4: distill a **smaller** student HarmMLP from the v3 teacher (DNS MSE + teacher MSE + curl + 64² FD-curl + IC), inject into the generator, bench at 262k points (tiny batches understate speedup).

Reproduce v4:
```bash
.venv/bin/python scripts/train_merger_qt_fast.py
.venv/bin/python scripts/bench_merger.py --out blog/checkpoint/v4/bench.json
.venv/bin/python scripts/regenerate_v4_media.py
```

### What we tried (ω campaign)

Architectures and losses, in the order they failed or succeeded:

| Approach | What we hoped | Outcome |
|----------|----------------|---------|
| Pointwise \((u,v)\) + curl on random collocation | Match DNS velocity | Plateaus **~7–9%** ω: filamentary structure under-sampled |
| Streamfunction \(\psi\) (\(u=\psi_y\), \(v=-\psi_x\), \(\omega=-\Delta\psi\)) | Div-free by construction + grid ω loss | Did not hit 2%; extra autograd cost |
| Direct \(\omega\) head / UVP+\(\omega\) (`uvpw`) | Supervise vorticity as an output | Did not beat harmonic MLP |
| Wide / deep UVP, large random Fourier features | Capacity would resolve filaments | Over-parameterized, worse or similar ω, slow |
| Dense-grid Fourier transform of a converged field | Post-hoc spectral match | Did not replace a better architecture |
| Harmonic Fourier MLP, \(k\le 6\), **16-wide** | Tiny + fast | Fast, stuck **~19%** ω |
| HarmMLP **96–96**, \(k\le 6\) | Align basis with 4-fold merger | **Worked: 1.29%** (`E25_harm_direct` ~2.5%, then `E27_harm_cont`) |
| Train QNN from scratch on the small accurate net | Circuit discovers weights | Much harder than distill-then-inject |
| Post-inject QNN fine-tune | Recover extra accuracy | Did not beat inject-only |
| Same \(k=6\) with 8–16 width (v4 speed hunt) | ≥2× and within 0.5 pp | Speed yes, **accuracy no** |
| HarmMLP **48–48**, \(k\le 3\), distill from 96–96 | Cut params, keep ω | **Worked: 1.78%**, **~2.5×** |

**What actually mattered**

- **Harmonic features** (\(k\le 6\)) beat generic RFF for this 4-fold co-rotating flow.
- **FD-curl ω in FP32** is the gate that matches what the gif shows; velocity L2 can look fine while the spiral looks “slow” (smoothed filaments, not under-spinning \(|v|\)).
- **Distill + inject** is how QT matches or nearly matches classical. The generator is not a better optimizer from scratch on this task.
- **Inference speed = deployed MLP size.** Shrinking \(k_{\max}\) and width is the speed lever; shrinking too far loses the 0.5 pp budget.

**Honest caveats**

- These PINNs are **data-assisted** (DNS collocation), not “solve NS from residual + IC only” at this accuracy.
- PDE residual MSE in `bench.json` is still large (momentum terms); the published gate is **field error vs DNS**, not residual-to-zero.
- v3 QT and classical are the **same** 13k-param net. v4 QT is a **different, smaller** net — similar accuracy, not identical.

---

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

Optional TGV stills remain in `blog/checkpoint/v1/`–`v2/` if the debugging narrative needs them.

- [x] Fig: merger triplet gif / snapshots (v4)
- [x] Table: v4 classical vs quantum (ω, vel, params, throughput)
- [x] Table: ω-campaign methods — worked vs failed (§9)
- [ ] Fig: experiment ladder (hypernet → Burgers VQC → merger solver → v4 speed)
- [ ] Table: critical bugs + symptom → fix
- [ ] Table: TGV v1 classical vs quantum (in-range / extrap-lo / time)
- [ ] Table: Kolmogorov in-range + extrap Q/C
- [ ] Fig: Burgers soft-IC collapse vs hard-IC trajectories
- [ ] Table: hard-IC scout + frozen-IC analytic floor
- [ ] Box: “When not to trust PDE residual” (and: residual vs DNS field error)
- [ ] Box: “Quantum-trained ≠ circuit at inference”

---

## 12. Recommended conclusion bullets (draft)

1. Fair evaluation of quantum PINNs requires matched capacity, fixed holdouts, and **degeneracy checks**.
2. Several attractive early results were invalidated by physics bugs or trivial solutions (zero field, frozen IC, constant-input generators).
3. On TGV and Kolmogorov, VQC hypernetworks underperformed matched classical generators (~1.5–2× error, ~2× slower **training**).
4. On input-conditioned Burgers, soft IC produced a fake quantum win; hard IC showed classical better, quantum frozen near IC.
5. The **goal that landed** is a PINN PDE solver for 2D vortex merger: HarmMLP vs spectral DNS, FD-curl ω **≤ 2%**.
6. Classical (96–96, \(k\le6\)) and quantum-trained (48–48, \(k\le3\)) sit in the **same accuracy band** (1.29% vs 1.78% ω); the quantum-trained deployed net is **~2.5× faster** because it has fewer weights.
7. What worked for the solver: harmonic features, FP32 curl gate, distill-then-inject. What did not: random collocation + curl, skinny \(k=6\) nets, training the QNN from scratch, treating residual MSE as the product metric.
8. Do not confuse “quantum-trained weights” with “quantum circuit in the forward pass.”

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
| `blog/checkpoint/v4/` | Fast QT + bench + media |

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
| `blog/checkpoint/v3/quantum` | Distilled same net (E31) |
| `blog/checkpoint/v4/quantum` | Fast HarmMLP 48–48, k≤3 (~2.5×) |

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

*Last evidence cutoff: merger v4 (`blog/checkpoint/v4/`), Burgers hard-IC scout, NS Tiers 0–C. Product status: working DNS-gated merger PINN; quantum-trained deployed net similar ω, faster because smaller.*
