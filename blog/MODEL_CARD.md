# Model card — qt-pinn PDE solvers

**Status:** evidence cutoff 2026-08-26. Simulator-only (PennyLane `default.qubit` / Torch); no hardware.

**Scope:** best *product* classical and quantum-trained solvers we are willing to cite, plus honest fair-advantage numbers. Checkpoints live under `blog/checkpoint/`.

---

## Headline metrics (vortex merger — primary product)

DNS reference: 256² spectral NS, four co-rotating Gaussians, ν=0.005, T=40. Gate: FD-curl ω rel-L2 max at t∈{0,2,5,8,12,15}, FP32. Target: ω ≤ 2%.

| | **Best classical** | **Best quantum (product)** | **Best fair e2e quantum** |
|--|--------------------|----------------------------|---------------------------|
| Checkpoint | `v4/classical/` (= v3 HarmMLP teacher) | `v4/quantum/` (inject / distill student) | `v4/archive/advantage_A/` seed-1 (`A_q8l4_s1`) |
| Deployed net | HarmMLP **96–96**, k≤6 | HarmMLP **48–48**, k≤3 | Same 48–48 k≤3 |
| Trainable / deployed params | 13 347 | 3 795 deployed | Circuit + proj → same 3 795 MLP |
| **ω rel-L2 max** | **1.29%** | **1.78%** | **1.90%** |
| Velocity rel-L2 max | 3.75% | 2.95% | ~2.7% |
| ω ≤ 2% gate | **pass** | **pass** | **pass** |
| Inference (256² batch) | ~326 Mpts/s | ~829 Mpts/s (**~2.5×**) | same arch ⇒ same latency as classical-gen |
| Circuit at inference? | n/a | **No** (proj weight zeroed; inject) | **No** (weights baked into MLP) |
| Circuit during training? | n/a | **No** (unused) | **Yes** (ablation degrades ω) |

**How to read this.** The classical teacher is the most accurate solver we ship. The v4 “quantum” product is a *smaller* deployed MLP that stays inside the 2% ω band and is ~2.5× faster; that speedup is **architecture size**, not quantum compute. The best *fair* end-to-end quantum generator (circuit actually trained) hits ~1.90% on one lucky seed; multi-seed mean is **worse** than a matched classical generator (see below).

Source of truth: `blog/checkpoint/v4/bench.json`, `v4/notes.md`, `v4/archive/advantage_scoreboard.md`.

### Per-time ω (classical teacher vs v4 inject)

| t | Classical ω | Quantum inject ω |
|---|-------------|------------------|
| 0 | 1.11% | 1.78% |
| 2 | **1.29%** | 1.74% |
| 5 | 0.97% | 1.19% |
| 8 | 0.77% | 0.74% |
| 12 | 0.74% | 0.58% |
| 15 | 0.68% | 0.62% |

---

## Fair quantum advantage (same deployed size)

Protocol: `ConditionedQuantumGeneratorV2` vs `ConditionedClassicalGeneratorV2`, both emit HarmMLP 48–48 k≤3, ν=0.005, q=8 L=4 bn=64, seeds 0–5.

| | Quantum gen | Classical gen |
|--|-------------|---------------|
| Mean ω ± pstdev | 2.325% ± 0.38% | **2.220% ± 0.24%** |
| Seed wins | 2/6 | **4/6** |
| Best seed ω | 1.895% (s1) | 1.823% (long classical) |
| Absolute best small net | — | Classical distill **1.75%** |

**Verdict: no robust quantum advantage.** Full table: `v4/archive/advantage_scoreboard.md`.

---

## Secondary: Taylor–Green polish (v2)

Exact TGV, ν=0.1, T=5. Soft metrics on velocity vs analytic solution.

| | Classical | Quantum-trained |
|--|-----------|-----------------|
| Checkpoint | `v2/classical/` | `v2/quantum/` |
| Deployed params | 5 059 | 1 507 (+ circuit at train) |
| **vel rel-L2 max** | **0.61%** | **0.62%** |
| Gate (~1%) | pass | pass |

Parity on a near-linear decay problem — useful demo media, not an advantage claim.

---

## What these models are / are not

**Are**
- PINNs that reconstruct velocity (and pressure) against spectral DNS or exact TGV.
- Deployed as classical MLPs at query time (even when a VQC trained the weights).
- Honest about soft vs hard IC, FP32 curl gates, and degeneracy checks.

**Are not**
- Circuit-in-the-forward-pass NS solvers at inference.
- Swirl/orbit-faithful merger solvers (peak co-rotation ≤5% with Rel≤2% was **not** promoted; see `v4/archive/orbit_fidelity/`).
- Multi-ν parametric solvers (Experiment B both arms ~33%+ ω).

---

## Reproduce

```bash
# Merger bench (classical teacher vs v4 inject)
.venv/bin/python scripts/bench_merger.py

# Fair advantage (example)
.venv/bin/python scripts/exp_quantum_advantage_A.py --arm quantum --n-qubits 8 --n-layers 4 --steps 20000
.venv/bin/python scripts/exp_quantum_advantage_A.py --arm classical_gen --n-qubits 8 --steps 20000
```

Media: `v4/media/merger_triplet.gif` · TGV: `v2/media/`, `v5/media/` · Unstable TGV: `v5/unstable/media/`.
