# qt-pinn

Is the quantum circuit necessary? A Quantum-Train-style weight generator (a parameterized
quantum circuit generates the weights of a classical physics-informed neural network
solving the 2D Burgers' equation, then the circuit is discarded and only the classical
weights are deployed), classified against known classical-simulability criteria, and
ablated against two classical weight generators at matched parameter count.

Full research writeup, citations, and the implementation plan: `research/qt_pde_ablation_prop.pdf`
and `research/qt_pde_ablation_implementation_plan.pdf`. This README is just the practical
"how to run things" reference.

## Who does what

- **Janos**: the weight generator itself, `src/qt_pinn/qnn_generator.py`, training (`scripts/train.py`),
  and getting it to converge with a correct (parameter-efficient) weight-mapping scheme.
- **Petya**: circuit classification (`src/qt_pinn/classification.py`, `scripts/classify_circuit.py`),
  the classical baseline generators (`src/qt_pinn/baselines/`), and the ablation comparing all three.

Current status and the exact open issues: `research/logs/2026-07-30-janos-first-push.md` and
`research/logs/2026-07-31-phase1b-classification.md`.

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
src/qt_pinn/          the library: circuit, PINN, physics loss, classification, classical baselines
scripts/               entry points: train, export, inference, classify
tests/                 self-tests for the library code
experiments/           self-contained one-off studies (currently: the RFF dequantization warm-up)
checkpoints/           trained runs (be deliberate about committing new ones, see .gitignore)
research/              the two LaTeX writeups + PDFs, verified paper source notes, hypotheses, dated logs
config.yaml            single source of truth for circuit/training hyperparameters
```
