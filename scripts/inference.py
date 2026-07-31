"""Pure classical inference (no PennyLane). Loads static_weights.pt, measures latency."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── MLP dims (must match qnn_generator.py) ──────────────────────────────────
IN_DIM  = 6    # Fourier feature width
H1      = 16   # first hidden layer
H2      = 16   # second hidden layer
OUT_DIM = 2    # u, v outputs


class _FourierMap(nn.Module):
    """Inline Fourier map — mirrors models/fourier.py, avoids any package import."""

    def __init__(self, sigma: float = 1.0, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)                    # same seed as training → identical B
        self.register_buffer("B", torch.randn(3, 3) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        return torch.cat([torch.sin(2 * math.pi * proj),
                          torch.cos(2 * math.pi * proj)], dim=-1)


class PureClassicalPINN(nn.Module):
    """Standalone MLP for Burgers inference — zero quantum dependencies."""

    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        self.fourier = _FourierMap()
        w = torch.load(weights_path, map_location="cpu", weights_only=True)

        self.fc1 = nn.Linear(IN_DIM, H1)
        self.fc1.weight = nn.Parameter(w["W1"][:IN_DIM * H1].reshape(H1, IN_DIM))
        self.fc1.bias   = nn.Parameter(w["W1"][IN_DIM * H1:])

        self.fc2 = nn.Linear(H1, H2)
        self.fc2.weight = nn.Parameter(w["W2"][:H1 * H2].reshape(H2, H1))
        self.fc2.bias   = nn.Parameter(w["W2"][H1 * H2:])

        self.fc3 = nn.Linear(H2, OUT_DIM)
        self.fc3.weight = nn.Parameter(w["W3"][:H2 * OUT_DIM].reshape(OUT_DIM, H2))
        self.fc3.bias   = nn.Parameter(w["W3"][H2 * OUT_DIM:])

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feats = self.fourier(torch.stack([x, y, t], dim=-1))
        h = F.tanh(self.fc1(feats))
        h = F.tanh(self.fc2(h))
        return self.fc3(h)


def _latest(base: Path = Path("checkpoints")) -> str:
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError("No runs found in checkpoints/")
    return runs[-1].name


def benchmark(model: nn.Module, batch: int = 1000, reps: int = 200) -> float:
    x, y, t = torch.rand(batch), torch.rand(batch), torch.rand(batch)
    for _ in range(10):       # warmup
        model(x, y, t)
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x, y, t)
    return (time.perf_counter() - t0) / reps * 1000


def plot_solution(
    model: nn.Module,
    run_id: str,
    t_values: list[float] = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    y_slice: float = 0.0,   # cross-section to plot along x; try 0.0 or 0.5
    n_points: int = 300,    # x resolution per curve
) -> None:
    """Plot u and v along x at a fixed y for multiple time snapshots."""

    x = torch.linspace(-1, 1, n_points)
    y = torch.full((n_points,), y_slice)

    colors = cm.plasma(np.linspace(0.1, 0.9, len(t_values)))

    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    with torch.no_grad():
        for t_val, color in zip(t_values, colors):
            t = torch.full((n_points,), t_val)
            pred = model(x, y, t)
            u = pred[:, 0].numpy()
            v = pred[:, 1].numpy()
            ax_u.plot(x.numpy(), u, color=color, label=f"t = {t_val}")
            ax_v.plot(x.numpy(), v, color=color, label=f"t = {t_val}")

    # reference IC from formula (t=0 ground truth)
    x_np = torch.linspace(-1, 1, n_points).numpy()
    u_ic = np.sin(np.pi * x_np) * np.cos(np.pi * y_slice)
    v_ic = -np.cos(np.pi * x_np) * np.sin(np.pi * y_slice)
    ax_u.plot(x_np, u_ic, "k--", linewidth=1.2, label="IC exact (t=0)")
    ax_v.plot(x_np, v_ic, "k--", linewidth=1.2, label="IC exact (t=0)")

    for ax, field in [(ax_u, "u"), (ax_v, "v")]:
        ax.set_xlabel("x")
        ax.set_ylabel(field)
        ax.set_title(f"{field}(x, y={y_slice}, t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)

    fig.suptitle(f"[{run_id}]  Burgers solution — time evolution at y={y_slice}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"checkpoints/{run_id}/solution_plot.png", dpi=150)
    plt.show()
    print(f"Saved checkpoints/{run_id}/solution_plot.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="latest",
                        help="run ID (e.g. run_0001) or 'latest'")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip solution plot")
    args = parser.parse_args()

    run_id = _latest() if args.run == "latest" else args.run
    path   = Path("checkpoints") / run_id / "static_weights.pt"

    model = PureClassicalPINN(path)
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor([0.5]), torch.tensor([0.3]), torch.tensor([0.1]))
    print(f"[{run_id}]  x=0.5 y=0.3 t=0.1  →  u={pred[0,0]:.6f}  v={pred[0,1]:.6f}")
    print(f"Latency: {benchmark(model):.3f} ms / batch-1000")

    if not args.no_plot:
        plot_solution(model, run_id)


if __name__ == "__main__":
    main()
