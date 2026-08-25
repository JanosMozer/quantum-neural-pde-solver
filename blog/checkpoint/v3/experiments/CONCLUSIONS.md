# ω campaign — vortex merger (kept runs)

**Gate:** max FD-curl ω rel-L2 at t ∈ {0, 2, 5, 8, 12, 15}, FP32. Target 2%. **Met.**

See `blog/checkpoint/v4/notes.md` for full lessons + benchmark.

## Promoted

| Model | curl ω_max | vel_max | exp |
|-------|------------|---------|-----|
| Classical HarmMLP 96-96, k≤6 | **1.29%** | 3.75% | E27_harm_cont |
| Quantum (distilled same weights) | **1.29%** | 3.75% | E31_qnn_distill |

## Kept experiment dirs

- `E25_harm_direct` — first harmonic MLP that beat RFF (~2.5%)
- `E27_harm_cont` — IC continue → promoted classical
- `E31_qnn_distill` — weight inject → promoted quantum
