# Live scripts

All commands from repo root: `.venv/bin/python scripts/<name>.py`

## Taylor–Green demo (live)

| Script | Role |
|--------|------|
| `train_tgv_demo.py` | Classical direct or single-ν Quantum-Train; `--preset scout\|demo` |
| `eval_tgv_grid.py` | rel-L2 at \(t=0,T/2,T\) |
| `sweep_tgv.py` | Cartesian grid from YAML / `--grid` |
| `configs/tgv_demo/` | scout, demo, sweep_scout, sweep_data |

```bash
.venv/bin/python scripts/train_tgv_demo.py --model classical --preset scout
.venv/bin/python scripts/train_tgv_demo.py --model quantum   --preset scout
.venv/bin/python scripts/sweep_tgv.py --config configs/tgv_demo/sweep_data.yaml
.venv/bin/python scripts/eval_tgv_grid.py checkpoints/tgv_demo_c_scout_s0
```

## Navier–Stokes / Taylor–Green (family / legacy)

| Script | Role |
|--------|------|
| `train_ns_direct.py` | Classical direct PINN `(x,y,t)→(u,v,p)` |
| `train_ns_parametric.py` | ν → MLP weights, quantum vs matched classical |
| `compare_ns_parametric.py` | Side-by-side + verdict |

## Kolmogorov (forced NS)

| Script | Role |
|--------|------|
| `train_kol_direct.py` | Direct baseline |
| `train_kol_parametric.py` | Parametric quantum vs classical |
| `compare_kol_parametric.py` | Compare PDE RMS |

## Burgers VQC-PINN (input-conditioned)

| Script | Role |
|--------|------|
| `train_burgers_vqc.py` | Scout/full, `--preset scout\|full`, hard IC default |
| `compare_burgers_vqc.py` | Refuses degenerate runs |
| `sanity_vqc_regression.py` | Fourier toy, not a PDE claim |

## Original Burgers Quantum-Train (weight generator → deploy MLP)

| Script | Role |
|--------|------|
| `train.py` / `train_gpu.py` / `train_lp.py` | Quantum generator |
| `train_classical.py` | Matched classical generator |
| `export_weights.py` | Drop circuit; save static MLP weights |
| `inference.py` | Plot from exported weights |
| `classify_circuit.py` | Entanglement / simulability diagnostics |

Dead NS sweeps and Burgers one-offs: `archive/`.
Demo quality + 3-panel animations: `docs/PLAN_TGV_DEMO.md`.
