"""Phase 2, the actual research-proposition ablation: Option 1 (quantum, 82 params) vs.
two classical generators at Option 1's own matched parameter count (low-rank exact at 82,
MPS at its nearest achievable, 64). Same PINN backbone, same data, same optimizer schedule,
only the generator differs.

Usage: python run_ablation.py <generator> <lambda_bc> [--steps N] [--n_colloc N] [--n_bc N]
  generator: option1 | low_rank | mps
"""

import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.qnn_generator import QuantumWeightGenerator, W1_SIZE, W2_SIZE, W3_SIZE, TOTAL_WEIGHTS
from qt_pinn.physics_loss import compute_burgers_loss
from qt_pinn.baselines.low_rank import LowRankGenerator
from qt_pinn.baselines.mps import MPSGenerator

OPTION1_PARAMS = 82  # circuit (81) + gamma (1), the real matched target for the baselines


class FlatToWeightDict(torch.nn.Module):
    """Adapter: wraps a flat-418-output generator (low_rank/mps) in Option 1's
    {"W1":..., "W2":..., "W3":...} interface, so TargetPINN doesn't need to know which
    generator produced its weights.
    """

    def __init__(self, flat_gen: torch.nn.Module) -> None:
        super().__init__()
        self.flat_gen = flat_gen

    def forward(self, inputs=None):
        flat = self.flat_gen().float()  # mps generator is float64 internally; PINN expects float32
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE:W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


def build_generator(name: str) -> torch.nn.Module:
    if name == "option1":
        return QuantumWeightGenerator(seed=0)
    if name == "low_rank":
        return FlatToWeightDict(LowRankGenerator(out_dim=TOTAL_WEIGHTS, target_param_count=OPTION1_PARAMS))
    if name == "mps":
        return FlatToWeightDict(MPSGenerator(out_dim=TOTAL_WEIGHTS, target_param_count=OPTION1_PARAMS))
    raise ValueError(f"unknown generator {name!r}")


def make_colloc(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    y = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    t = torch.empty(n).uniform_(0, 1, generator=g).requires_grad_(True)
    return x, y, t


def make_bc(n: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1)  # different seed than colloc, still fixed
    x = torch.empty(n).uniform_(-1, 1, generator=g)
    y = torch.empty(n).uniform_(-1, 1, generator=g)
    t = torch.zeros(n)
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    return x, y, t, u, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generator", choices=["option1", "low_rank", "mps"])
    parser.add_argument("lambda_bc", type=float)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--n_colloc", type=int, default=1024)
    parser.add_argument("--n_bc", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = TargetPINN()
    gen = build_generator(args.generator)
    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)

    x, y, t = make_colloc(args.n_colloc, args.seed)
    xb, yb, tb, ub, vb = make_bc(args.n_bc, args.seed)

    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    print(f"generator={args.generator}  n_params={n_params}  lambda_bc={args.lambda_bc}  "
          f"n_colloc={args.n_colloc}  n_bc={args.n_bc}  steps={args.steps}")

    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        (pde + args.lambda_bc * bc).backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"{step:6d}  pde={pde.item():.6f}  bc={bc.item():.6f}  total={pde.item()+bc.item():.6f}")
    elapsed = time.perf_counter() - t0

    result = {
        "generator": args.generator, "n_params": n_params, "lambda_bc": args.lambda_bc,
        "n_colloc": args.n_colloc, "n_bc": args.n_bc, "steps": args.steps, "seed": args.seed,
        "pde_loss": round(pde.item(), 8), "bc_loss": round(bc.item(), 8),
        "total": round(pde.item() + bc.item(), 8), "elapsed_s": round(elapsed, 1),
    }
    print(f"\nFinal  pde={result['pde_loss']:.7f}  bc={result['bc_loss']:.7f}  sum={result['total']:.7f}  "
          f"({elapsed:.1f}s)")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.generator}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
