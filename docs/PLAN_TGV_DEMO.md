# TGV demo — requirements and structure

## Deliverable

Three side-by-side vorticity animations:

1. Exact Taylor–Green
2. Classical PINN
3. Quantum-trained PINN (deployed MLP)

## Architecture

```
ν → [VQC | matched classical generator] → θ_MLP     # quantum: once per step
(x,y,t) → TargetPINNNS(θ_MLP) → (u,v,p)            # every collocation / frame
```

- Same hidden width and TGV Fourier for both.
- Hard IC: \(u = u_{IC} + t\,N\).
- No per-pixel VQC (that path is 60 h-class; not used).
- Single viscosity \(\nu=0.1\) (circuit is a weight generator, not a field net).

Exact field (panel 1):

\[
u=\sin x\cos y\,e^{-2\nu t},\quad
v=-\cos x\sin y\,e^{-2\nu t},\quad
p=\tfrac14(\cos 2x+\cos 2y)\,e^{-4\nu t}
\]

Vorticity \(\omega=2\sin x\sin y\,e^{-2\nu t}\).

## Specs

| | |
|--|--|
| Domain | \(x,y\in[0,2\pi]\) |
| Regime | \(\nu=0.1\), \(t\in[0,5]\) |
| Grid | \(256^2\) |
| Field | vorticity, shared color scale |
| Frames | 80–120, mp4 + gif |
| Layout | 1×3, \(t\) overlay |
| Accuracy | velocity rel-L2 \(\le 2\%\) at \(t=0,T/2,T\) (target \(\le 1\%\)) |
| Quantum vs classical | rel-L2 \(\le 1.2\times\) classical at \(\nu=0.1\), and \(\le 2\%\) absolute |
| Quantum wall-clock | scout \(\le 15\) min (`--budget-s 900`); demo \(\le 40\) min (`--budget-s 2400`) |

Do not render until both models meet the accuracy row on a held-out grid.

## How accuracy improves (without a 60 h QNN)

| Lever | Classical | Quantum |
|--------|-----------|---------|
| Hard IC | required | required |
| \(\lambda_{\mathrm{data}}\) on exact TGV | primary visual lever | same; first-order through MLP only |
| `--loss-mode data` | optional warm start | **use for scout**; skips \(u_{xx}\) |
| Hidden | `[32,32]` (small target = fewer generated weights) | same |
| Circuit | — | `expect`, 4q×3L scout / 6q×4L demo |
| Tasks / step | — | **1** (fixed \(\nu=0.1\)) |

Loss:

\[
\mathcal{L}=\lambda_{\mathrm{pde}}\mathcal{L}_{\mathrm{PDE}}+\lambda_{\mathrm{IC}}\mathcal{L}_{\mathrm{IC}}+\lambda_{\mathrm{data}}\lVert\hat u-u_{\mathrm{exact}}\rVert^2
\]

## Experiment interface

| | |
|--|--|
| Train | `scripts/train_tgv_demo.py --model {classical,quantum} --preset {scout,demo}` |
| Eval | `scripts/eval_tgv_grid.py checkpoints/<run>` |
| Sweep | `scripts/sweep_tgv.py --config configs/tgv_demo/sweep_scout.yaml` |
| Configs | `configs/tgv_demo/*.yaml` (CLI overrides) |
| `--overwrite` | rerun same `run-id` |
| `--budget-s` | hard wall-clock stop |

Presets live in `src/qt_pinn/tgv_demo.py` (`PRESETS`).

## Outputs

| Artifact | Path |
|----------|------|
| Classical | `checkpoints/tgv_demo_c_<preset>_s<seed>/` |
| Quantum | `checkpoints/tgv_demo_q_<preset>_s<seed>/` |
| Sweep table | `checkpoints/sweeps/<preset>_classical-quantum.json` |
| Animation | `scripts/animate_tgv.py` → `blog/media/tgv_triplet.{mp4,gif}` (after gate) |
