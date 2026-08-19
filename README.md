# Quantum Neural Solvers for PDEs

A Quantum-Train-style weight generator (a parameterized quantum circuit generates
the weights of a classical physics-informed neural network solving the 2D Burgers'
equation, then the circuit is discarded and only the classical weights are deployed),
classified against known classical-simulability criteria, and ablated against two
classical weight generators at matched parameter count.

Full research writeup, citations, and the implementation plan: `research/qt_pde_ablation_prop.pdf`
and `research/qt_pde_ablation_implementation_plan.pdf`. This README is just the practical
"how to run things" reference.

## Setup

```bash
uv venv --python 3.12.12 .venv
uv pip install --python .venv/bin/python -e .
```

No GPU required at the qubit counts this project currently uses; `pennylane`'s CPU
`default.qubit` device is enough. If qubit count ever grows past what's comfortable on CPU,
`pennylane-lightning[gpu]` is already a listed dependency.

## Running things

All commands run from the repo root.

```bash
# train the QT-PINN (2-stage Adam -> L-BFGS), saves to checkpoints/run_NNNN/
.venv/bin/python scripts/train.py

# export the trained quantum generator's static classical weights (drops the quantum dependency)
.venv/bin/python scripts/export_weights.py --run latest

# pure-classical inference + a solution plot, using the exported static weights
.venv/bin/python scripts/inference.py --run latest

# classify a circuit's entanglement structure (Schmidt rank / entropy at every cut)
.venv/bin/python scripts/classify_circuit.py
```

## Tests

Plain assert-based self-tests, no framework, run directly:

```bash
.venv/bin/python tests/verify_env.py          # environment/dependency sanity check
.venv/bin/python tests/test_classification.py # analytic ground truth (product state, GHZ state)
.venv/bin/python tests/test_quimb_autograd.py # quimb MPS -> torch gradient smoke test
.venv/bin/python tests/test_baselines.py      # both classical baseline generators
```

## Layout

```
pdes/burgers2d/        the PDE: physics residual (physics_loss.py) + config.yaml (circuit/training hyperparameters)
src/qt_pinn/           the library: circuit, PINN, classification, classical baselines
scripts/               entry points: train, export, inference, classify
tests/                 self-tests for the library code
experiments/           self-contained one-off studies (currently: the RFF dequantization warm-up)
checkpoints/           trained runs (be deliberate about committing new ones, see .gitignore)
research/              the two LaTeX writeups + PDFs, verified paper source notes, hypotheses, dated logs
```
