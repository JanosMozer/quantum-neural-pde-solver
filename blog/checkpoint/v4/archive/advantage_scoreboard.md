# Quantum advantage experiments (v4) — verified scoreboard

Date: 2026-08-25. Deployed net for all generator arms: **HarmMLP / TargetPINNNS 48–48, harmonic k≤3** (3 795 params). Gate metric: FD-curl ω rel-L2 max at t∈{0,2,5,8,12,15}, FP32.

**Final verdict: no robust quantum advantage.** Control C + Experiments A and B all fail the success gate under fair matched-architecture tests.

## Control C — inject-only was a no-op

| Model | ω max | Notes |
|-------|-------|-------|
| Classical distill h48k3 | **1.746%** | Same training as v4 student |
| v4 QT inject (proj weight=0) | 1.779% | Circuit unused |

**Verdict:** v4 “~2.5× speedup” was architecture shrink, not quantum.

## Experiment A — end-to-end generator (single ν=0.005)

Same deployed HarmMLP 48–48 k≤3; matched generator bottleneck; circuit ablation required for quantum.

### Multi-seed (q=8, L=4, bn=64) — primary matchup

`A_q8l4` was mis-tagged seed=1 (duplicate of `A_q8l4_s1`). True seeds 0–5:

| Seed | Quantum | Classical-gen | Winner |
|------|---------|---------------|--------|
| 0 | 2.302% | 2.241% | **C** |
| 1 | 1.895% | 1.987% | Q |
| 2 | 3.029% | 2.726% | **C** |
| 3 | 2.246% | 2.188% | **C** |
| 4 | 1.956% | 2.047% | Q |
| 5 | 2.519% | 2.133% | **C** |
| **Mean ± pstdev** | **2.325% ± 0.38%** | **2.220% ± 0.24%** | **Δ +0.10 pp (C)** |

Q wins **2/6**. Circuit used on all quantum runs.

### Long + curl×2 (40k steps, circuit LR×0.3)

| Seed | Quantum | Classical-gen |
|------|---------|---------------|
| 0 | 1.943% | 1.972% |
| 1 | *(mid ~4%)* | **1.823%** |
| 2 | *(mid ~4%)* | **1.928%** |
| 3 | *(mid ~4%)* | **1.944%** |

Classical long mean already **1.917%** — better than the only finished quantum long (1.943%). Seed-0’s −0.03 pp Q edge does not generalize.

### Other configs

| Config | Quantum | Classical-gen | Note |
|--------|---------|---------------|------|
| bn16 multi-seed (q8 L8) | mean **2.873%** | mean **2.287%** | Q wins 1/4 |
| V1 probs q6 L4 s0 | 3.316% | **2.125%** | C |
| q8 L6 bn32 long s0 | 1.974% | **1.960%** | C |
| Best absolute (any method) | — | distill h48 **1.746%** / teacher 96–96 **1.29%** | classical |

## Experiment B — ν-family (holdout ν=0.008)

| Arm | Train ω_mean | Holdout ω_mean | Circuit |
|-----|--------------|----------------|---------|
| quantum q8/q6 (full family) | ~33% | ~100%+ | yes |
| classical q8/q6 (full family) | ~33–34% | ~180–276% | — |
| classical mid+teacher | 35–49% | 19–25% | — |
| quantum mid+teacher | **33.7%** | **126%** | yes |

**Verdict:** Neither arm is a working multi-ν solver (≫2% gate). No advantage claim possible.

## Success-gate check

| Requirement | Result |
|-------------|--------|
| Same deployed arch | yes |
| QT ω better at matched latency (robust) | **no** |
| QT latency better at matched ω | **no** (same arch ⇒ same latency) |
| Circuit influences weights | **yes** on e2e QNN (ablations pass); **no** on v4 inject |

## Deliverables

- Logs / comparisons: `blog/checkpoint/v4/advantage_A/`, `advantage_B/`, `classical_h48_baseline/`, `parallel_logs/`
- This scoreboard + `notes.md` + `blog/blog.md` §9
