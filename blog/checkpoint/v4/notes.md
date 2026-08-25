# v4 — fast quantum merger PINN

Goal: **≥2× classical inference throughput** with ω accuracy **within 0.5 percentage
points** of classical, by shrinking the **deployed** MLP. v3 models/data were
**not modified**.

## Result

| | Classical (v3 snapshot in `v4/classical/`) | Quantum (`v4/quantum/`) |
|--|---------------------------------------------|-------------------------|
| Architecture | HarmMLP 96–96, harmonic k≤6 | HarmMLP **48–48**, harmonic **k≤3** |
| Deployed params | 13 347 | **3 795** (~3.5× fewer) |
| ω rel-L2 max (FP32) | **1.288%** | **1.779%** (Δ = **+0.492 pp**) |
| vel rel-L2 max | 3.747% | ~2.95% |
| Throughput (256k pts) | ~230 Mpts/s | ~**2.5–2.7×** classical |
| Gate ω ≤ classical+0.5pp | — | **pass** |
| Gate ≥2× throughput | — | **pass** |

Raw numbers: [`bench.json`](bench.json), [`quantum/results.json`](quantum/results.json).

## How

1. Keep DNS reference and classical teacher **read-only** from `v3/`.
2. Distill a smaller student HarmMLP (DNS + teacher MSE + curl + 64² FD-curl + IC).
3. Inject student weights into a `ConditionedQuantumGeneratorV2` projection bias
   (same single-ν pattern as E31) so the **deployed** QT PINN matches the student.
4. Bench both at **262 144** points (compute-bound); tiny batches understate speedup.

## What worked

- **Fewer deployed weights** (not a smaller generator circuit) drive inference speed.
- **Slightly reduced harmonic basis (k≤3)** + mid-width MLP (48–48) hits both
  accuracy and ≥2× speed. Full k=6 with a 16-wide net was fast but stuck ~19% ω.
- Knowledge distillation from the 96–96 classical teacher helps the small net.

## What did not

- Same k=6 basis with 8–16 width: ≥2× speed, **cannot** reach within 0.5pp of classical.
- Expecting the quantum *generator* to discover a small accurate net from scratch
  was much harder than distill-then-inject.

## Layout

```
v4/
  notes.md          this file
  bench.json        classical vs quantum metrics
  classical/        copy of v3 classical (for self-contained bench)
  quantum/          fast QT deployed weights + generator
  candidates/       intermediate student checkpoints
```

Reproduce:
```bash
.venv/bin/python scripts/train_merger_qt_fast.py
```
