# qt-pinn

Physics-informed neural nets with variational quantum circuits as **weight generators** (Quantum-Train: circuit produces MLP weights, then the circuit is discarded) and as **input-conditioned** field models.

Primary demo target: 2D Taylor–Green Navier–Stokes, classical vs quantum, 3-panel animation.  
Plan: [`docs/PLAN_TGV_DEMO.md`](docs/PLAN_TGV_DEMO.md).  
NS measurement log: [`pdes/ns2d/EXPERIMENT_ANALYSIS.md`](pdes/ns2d/EXPERIMENT_ANALYSIS.md).  
Blog outline: [`blog/blog.md`](blog/blog.md).

## Setup

```bash
uv venv --python 3.12.12 .venv
uv pip install --python .venv/bin/python -e .
```

PennyLane `default.qubit` (CPU) is the simulator in use. Circuit training is much slower than a small GPU MLP; animations must use **deployed classical MLPs**, not per-pixel VQCs.

## Live commands

From repo root. Full list: [`scripts/README.md`](scripts/README.md).

```bash
# Classical TGV PINN (direct)
.venv/bin/python scripts/train_ns_direct.py --adam-steps 6000

# Quantum vs classical hypernetwork (ν → MLP weights)
.venv/bin/python scripts/train_ns_parametric.py --generator classical --seed 0 --run-id ns_par_c_s0
.venv/bin/python scripts/train_ns_parametric.py --generator quantum   --seed 0 --run-id ns_par_q_s0
.venv/bin/python scripts/compare_ns_parametric.py checkpoints/ns_par_c_s0 checkpoints/ns_par_q_s0

# Original Burgers Quantum-Train
.venv/bin/python scripts/train_gpu.py
.venv/bin/python scripts/export_weights.py --run latest
.venv/bin/python scripts/inference.py --run latest
```

## Tests

```bash
.venv/bin/python tests/verify_env.py
.venv/bin/python tests/test_classification.py
.venv/bin/python tests/test_quimb_autograd.py
.venv/bin/python tests/test_baselines.py
```

## Layout

```
pdes/burgers2d/     Burgers residual + config
pdes/ns2d/          Taylor–Green NS + experiment log
pdes/kolmogorov2d/  Forced NS
src/qt_pinn/        Circuits, PINN targets, Fourier maps, generators
scripts/            Live entry points
archive/            Dead sweeps, broken NS trainer, Burgers one-offs
checkpoints/        Runs (see checkpoints/README.md)
docs/               Demo plan
blog/               Article outline
tests/
```
