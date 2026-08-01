"""Pure classical inference (no PennyLane). Loads static_weights.pt, measures accuracy."""

from pathlib import Path
import argparse, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from qt_pinn.config_loader import load as _load_cfg

_cfg = _load_cfg()
IN_DIM, OUT_DIM = 6, 2
H1, H2 = _cfg["mlp"]["hidden"]
OMEGA_0 = _cfg["mlp"].get("omega_0", 30.0)
FOURIER_SIGMA = _cfg["fourier"]["sigma"]
FOURIER_SEED  = _cfg["fourier"]["seed"]


class _FourierMap(nn.Module):
    def __init__(self, sigma: float = FOURIER_SIGMA, seed: int = FOURIER_SEED) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("B", torch.randn(3, 3) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        return torch.cat([torch.sin(2 * math.pi * proj),
                          torch.cos(2 * math.pi * proj)], dim=-1)


class PureClassicalPINN(nn.Module):
    """Standalone MLP for Burgers inference, zero quantum dependencies."""

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
        h = torch.sin(OMEGA_0 * self.fc1(feats))
        h = torch.sin(OMEGA_0 * self.fc2(h))
        return self.fc3(h)


def _latest(base: Path = Path("checkpoints")) -> str:
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError("No runs found in checkpoints/")
    return runs[-1].name


def benchmark(model: nn.Module, batch: int = 1000, reps: int = 200) -> float:
    x, y, t = torch.rand(batch), torch.rand(batch), torch.rand(batch)
    for _ in range(10):
        model(x, y, t)
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x, y, t)
    return (time.perf_counter() - t0) / reps * 1000


def evaluate_ic(model: nn.Module, n: int = 20) -> dict:
    """Measure accuracy at t=0 on nxn grid where exact solution is known.

    IC:  u(x,y,0) = sin(πx)·cos(πy)
         v(x,y,0) = −cos(πx)·sin(πy)
    """
    xs = torch.linspace(-1, 1, n)
    ys = torch.linspace(-1, 1, n)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    x = xg.flatten(); y = yg.flatten()
    t = torch.zeros(n * n)

    u_exact = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v_exact = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)

    with torch.no_grad():
        pred = model(x, y, t)
    u_pred = pred[:, 0]; v_pred = pred[:, 1]

    def _metrics(p, e):
        err = p - e
        l2_rel  = (err.norm() / (e.norm() + 1e-10)).item()
        mae     = err.abs().mean().item()
        max_err = err.abs().max().item()
        rmse    = err.pow(2).mean().sqrt().item()
        return {"l2_rel": l2_rel, "mae": mae, "max_err": max_err, "rmse": rmse}

    return {
        "n":       n * n,
        "u":       _metrics(u_pred, u_exact),
        "v":       _metrics(v_pred, v_exact),
        "x": x, "y": y,
        "u_pred": u_pred, "v_pred": v_pred,
        "u_exact": u_exact, "v_exact": v_exact,
    }


def print_ic_report(model: nn.Module) -> None:
    """Full IC accuracy report with sample point table."""
    res = evaluate_ic(model, n=20)

    print("\n---  IC Accuracy  (t=0, 20x20 grid, 400 points)  ---")
    print(f"  {'':6}  {'L2 rel%':>9}  {'RMSE':>9}  {'MAE':>9}  {'Max err':>9}")
    print(f"  {'u':6}  {res['u']['l2_rel']*100:9.2f}  {res['u']['rmse']:9.5f}  {res['u']['mae']:9.5f}  {res['u']['max_err']:9.5f}")
    print(f"  {'v':6}  {res['v']['l2_rel']*100:9.2f}  {res['v']['rmse']:9.5f}  {res['v']['mae']:9.5f}  {res['v']['max_err']:9.5f}")

    # sample point table
    test = [
        (-1.0, -1.0), (-0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0),  # diagonal
        (-0.5,  0.0), (0.5,  0.0),                                           # y=0 axis
        ( 0.0, -0.5), (0.0,  0.5),                                           # x=0 axis
    ]
    print(f"\n{'x':>6}  {'y':>6} | {'u_pred':>8} {'u_exact':>8} {'|err|':>8} | {'v_pred':>8} {'v_exact':>8} {'|err|':>8}")
    print("-" * 74)
    for xi, yi in test:
        xT = torch.tensor([xi]); yT = torch.tensor([yi]); tT = torch.zeros(1)
        u_ex = math.sin(math.pi * xi) * math.cos(math.pi * yi)
        v_ex = -math.cos(math.pi * xi) * math.sin(math.pi * yi)
        with torch.no_grad():
            p = model(xT, yT, tT)
        up, vp = p[0, 0].item(), p[0, 1].item()
        print(f"{xi:6.2f}  {yi:6.2f} | {up:8.4f} {u_ex:8.4f} {abs(up-u_ex):8.4f} | {vp:8.4f} {v_ex:8.4f} {abs(vp-v_ex):8.4f}")

    # time evolution at one probe point
    # probe at two physically interesting points
    probes = [
        (0.5,  0.0, "sin(π·0.5)·cos(0)=1.0",  "−cos(π·0.5)·sin(0)=0.0"),
        (0.0,  0.5, "sin(0)·cos(π·0.5)=0.0",   "−cos(0)·sin(π·0.5)=−1.0"),
    ]
    for px, py, u_desc, v_desc in probes:
        u_ic = math.sin(math.pi * px) * math.cos(math.pi * py)
        v_ic = -math.cos(math.pi * px) * math.sin(math.pi * py)
        print(f"\n  Probe  x={px}, y={py}  |  IC: u={u_ic:.4f} ({u_desc})  v={v_ic:.4f} ({v_desc})")
        print(f"  {'t':>5}  {'u_pred':>9}  {'u_err':>8}  {'v_pred':>9}  {'v_err':>8}")
        for tv in [0.0, 0.1, 0.25, 0.5, 1.0]:
            with torch.no_grad():
                p = model(torch.tensor([px]), torch.tensor([py]), torch.tensor([tv]))
            up, vp = p[0,0].item(), p[0,1].item()
            u_e = f"{abs(up-u_ic):.4f}" if tv == 0.0 else "      -"
            v_e = f"{abs(vp-v_ic):.4f}" if tv == 0.0 else "      -"
            print(f"  {tv:5.2f}  {up:9.4f}  {u_e:>8}  {vp:9.4f}  {v_e:>8}")


def plot_solution(
    model: nn.Module,
    run_id: str,
    t_values: list[float] = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    y_slice: float = 0.0,
    n_points: int = 300,
) -> None:
    x = torch.linspace(-1, 1, n_points)
    y = torch.full((n_points,), y_slice)
    colors = cm.plasma(np.linspace(0.1, 0.9, len(t_values)))
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(13, 4))

    with torch.no_grad():
        for t_val, color in zip(t_values, colors):
            t = torch.full((n_points,), t_val)
            pred = model(x, y, t)
            ax_u.plot(x.numpy(), pred[:, 0].numpy(), color=color, label=f"t = {t_val}")
            ax_v.plot(x.numpy(), pred[:, 1].numpy(), color=color, label=f"t = {t_val}")

    x_np = x.numpy()
    ax_u.plot(x_np, np.sin(np.pi * x_np) * np.cos(np.pi * y_slice), "k--", lw=1.2, label="IC exact (t=0)")
    ax_v.plot(x_np, -np.cos(np.pi * x_np) * np.sin(np.pi * y_slice), "k--", lw=1.2, label="IC exact (t=0)")

    for ax, field in [(ax_u, "u"), (ax_v, "v")]:
        ax.set_xlabel("x"); ax.set_ylabel(field)
        ax.set_title(f"{field}(x, y={y_slice}, t)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)

    fig.suptitle(f"[{run_id}]  Burgers, time evolution at y={y_slice}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"checkpoints/{run_id}/solution_plot.png", dpi=150)
    plt.show()
    print(f"Saved checkpoints/{run_id}/solution_plot.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="latest")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_id = _latest() if args.run == "latest" else args.run
    path   = Path("checkpoints") / run_id / "static_weights.pt"

    print(f"[{run_id}]")
    model = PureClassicalPINN(path)
    model.eval()

    print(f"Latency: {benchmark(model):.3f} ms / batch-1000")
    print_ic_report(model)

    if not args.no_plot:
        plot_solution(model, run_id)


if __name__ == "__main__":
    main()
