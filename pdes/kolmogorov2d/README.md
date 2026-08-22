# Kolmogorov flow (Tier C)

2D forced Navier–Stokes benchmark with sustained nonlinearity. Replaces Taylor–Green as the active QC vs classical testbed.

## PDE

On \([0,2\pi]^2 \times [0,T]\) with periodic boundaries:

\[
\partial_t u + (u\cdot\nabla)u + \nabla p = \nu \nabla^2 u + f,\qquad \nabla\cdot u = 0
\]

\[
f = (A\sin(n y),\, 0), \quad n=4 \text{ by default}
\]

IC: \(u=v=p=0\) at \(t=0\) (hard IC: field \(= t \cdot \mathrm{network}\)).

**No exact solution** — quality is measured by **PDE residual RMS** on holdout points.

## Scripts

```bash
# 1. Direct baseline (run first)
.venv/bin/python scripts/train_kol_direct.py --adam-steps 8000 --run-id kol_direct_s0

# 2. Parametric nu family
.venv/bin/python scripts/train_kol_parametric.py --generator classical --seed 0 --run-id kol_par_c_s0
.venv/bin/python scripts/train_kol_parametric.py --generator quantum   --seed 0 --run-id kol_par_q_s0

# 3. Compare
.venv/bin/python scripts/compare_kol_parametric.py checkpoints/kol_par_c_s0 checkpoints/kol_par_q_s0
```

Defaults: `qc-arch expect`, log-ν encoding, \(\nu \in [0.01, 0.1]\), \(T=5\), `n_force=4`.

## Files

| File | Role |
|------|------|
| `config.yaml` | Default hyperparameters |
| `physics_loss.py` | Forced NS residual + zero IC |
| `../ns2d/EXPERIMENT_ANALYSIS.md` | Full TGV history + pivot notes |
