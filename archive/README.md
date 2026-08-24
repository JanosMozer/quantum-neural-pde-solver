# Archive

Historical / invalidated entry points. Not part of the live training path.
Kept for provenance (bugs, nulls, Burgers one-offs). Do not use these for new runs.

Run from repo root if you must reproduce an old figure. Import paths were updated
for the extra `archive/` directory.

## `archive/scripts/`

| File | Why archived |
|------|----------------|
| `train_ns.py` | Tier 0 NS: constant-input quantum generator (not a function) |
| `ns_sweep_8x.py`, `ns_sweep_pick_best.py` | Sweeps of broken TGV physics (wrong pressure sign) |
| `plot_ns.py` | Hardcoded to `checkpoints/ns_run_0001` (invalid collapse run) |
| `sweep.py`, `ablation.py` | Burgers hyperparam / baseline one-offs |

## `archive/experiments/`

Self-contained Burgers studies (RFF warmup, DLA, depth, symmetric ansatz, STMFF, run_0051 verify).
Results JSON under each folder is the record.

## Live replacements

| Need | Use instead |
|------|-------------|
| TGV classical PINN | `scripts/train_ns_direct.py` |
| TGV quantum vs classical family | `scripts/train_ns_parametric.py` |
| Kolmogorov | `scripts/train_kol_*.py` |
| Burgers input-conditioned VQC | `scripts/train_burgers_vqc.py` |
| Demo animations / quality models | `docs/PLAN_TGV_DEMO.md` |
