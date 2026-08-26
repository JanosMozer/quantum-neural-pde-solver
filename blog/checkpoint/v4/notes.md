# v4 — fast quantum merger PINN + advantage experiments (closed)

## Disposition

**v4 closed as-is.** Product solvers stay in `classical/` and `quantum/`. Advantage,
orbit/swirl, dns_family, and related experiment trees live under **`archive/`**
(not deleted). Next: **v5** returns to Taylor–Green (non-merger) with larger/sharper
contour media.

## Product checkpoint (deployed solver)

Goal: **≥2× classical inference throughput** with ω within 0.5 pp of the
96–96 teacher, by shrinking the **deployed** MLP.

| | Classical (v3 snapshot) | Quantum v4 inject |
|--|-------------------------|-------------------|
| Deployed | HarmMLP 96–96, k≤6 | HarmMLP **48–48**, k≤3 |
| Params | 13 347 | **3 795** |
| ω max (FP32) | **1.288%** | **1.779%** |
| Throughput | ~326 Mpts/s | **~2.5×** |

**Important correction:** v4 inject set `proj[-1].weight = 0` and copied the
student into the bias — the **circuit did not contribute**. Classical distill
of the same h48k3 hits **1.746%** (`archive/classical_h48_baseline/`). Speed was from
architecture size, not quantum.

## Advantage experiments (A/B/C) — null

See [`archive/advantage_scoreboard.md`](archive/advantage_scoreboard.md).

### Headline (Experiment A)

Same deployed HarmMLP 48–48 k≤3; matched generator capacity; circuit ablation pass.

| Seeds 0–5 (q=8 L=4 bn=64) | Quantum | Classical-gen |
|--|--|--|
| Mean ω | 2.325% ± 0.38% | **2.220% ± 0.24%** |
| Wins | 2/6 | **4/6** |

**No robust quantum advantage.** bn16 / V1 / long+curl sweeps agree or favor classical.

### Experiment B (ν-family)

Both arms ~33–50% train ω_mean — **fail** as solvers (including mid+teacher).

### Control C

Classical h48 distill **1.746%** ≈ QT inject **1.779%** with unused circuit.

## Orbit / swirl campaign — not promoted

See [`archive/orbit_fidelity/NOTES.md`](archive/orbit_fidelity/NOTES.md).

Tried to make ω **maxima co-rotate** like DNS (not just pointwise Rel-L2).
`orbit_omega` Fourier features fixed early freeze; full-horizon ≤5% swirl with
Rel≤2% was **not** achieved in a single promoted checkpoint. Best evidence kept
under `archive/orbit_fidelity/` (s4 Rel, s7 swirl, s9 compromise). Product
`classical/` / `quantum/` remain the pre-orbit inject pair. Media gif still
shows the frozen-orbit gap on those weights.

## Layout

```
v4/
  notes.md, bench.json
  classical/, quantum/   # product inject checkpoint
  dns/                   # merger DNS reference (from v3)
  media/
  archive/               # advantage_*, orbit_fidelity, dns_family, baselines, …
```
