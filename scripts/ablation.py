"""Benchmark the quantum weight generator against architecturally-matched classical
(low-rank) and tensor-network (MPS) generators.

All three produce the same TOTAL_WEIGHTS-sized flat vector for the same TargetPINN, are
matched at the quantum generator's own total trainable parameter count (circuit + proj
head), and use the same training recipe (config.yaml) and the same data. Only the
generator's internal structure differs. Run from the repo root.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from qt_pinn.config_loader import load as _load_cfg
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.qnn_generator import TOTAL_WEIGHTS, W1_SIZE, W2_SIZE, W3_SIZE
from qt_pinn.baselines.low_rank import LowRankGenerator
from qt_pinn.baselines.mps import MPSGenerator
from qt_pinn.baselines.classical_frontend import ClassicalFrontendGeneratorLP
from pdes.burgers2d.physics_loss import compute_burgers_loss
from scripts.train_gpu import GPUWeightGeneratorLP, DEVICE, make_colloc, make_bc

_t = _load_cfg()["training"]
SEED       = _t["seed"]
N_COLLOC   = _t["n_colloc"]
N_BC       = _t["n_bc"]
LAMBDA_BC  = _t["lambda_bc"]
WEIGHT_REG = _t.get("weight_reg", 0.0)
GRAD_CLIP  = _t.get("grad_clip_norm", 1.0)
LR         = _t["adam"]["lr"]
STEPS      = _t["adam"]["steps"]
ETA_MIN    = _t.get("cosine_eta_min", 1e-5)


class FlatToWeightDict(torch.nn.Module):
    """Wraps a flat-TOTAL_WEIGHTS-output generator in TargetPINN's {"W1","W2","W3"}
    dict interface, so it doesn't need to know which generator produced its weights."""

    def __init__(self, flat_gen: torch.nn.Module) -> None:
        super().__init__()
        self.flat_gen = flat_gen

    def forward(self, inputs=None) -> dict[str, torch.Tensor]:
        flat = self.flat_gen().float()
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


def build_generators(device: torch.device, bottleneck_width: int = 64) -> tuple[dict[str, torch.nn.Module], int]:
    quantum = GPUWeightGeneratorLP(device, bottleneck_width)
    n_target = sum(p.numel() for p in quantum.parameters())
    low_rank = FlatToWeightDict(LowRankGenerator(TOTAL_WEIGHTS, n_target)).to(device)
    mps      = FlatToWeightDict(MPSGenerator(TOTAL_WEIGHTS, n_target)).to(device)
    classical_frontend = ClassicalFrontendGeneratorLP().to(device)
    return {"quantum": quantum, "low_rank": low_rank, "mps": mps,
            "classical_frontend": classical_frontend}, n_target


def run_one(name: str, gen: torch.nn.Module, model: torch.nn.Module, steps: int,
            x, y, t, xb, yb, tb, ub, vb, xh, yh, th, xhb, yhb, thb, uhb, vhb) -> dict:
    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)

    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=ETA_MIN)

    print(f"\n=== {name}  n_params={n_params:,} ===")
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        reg = sum(v.pow(2).mean() for v in w.values())
        loss = pde + LAMBDA_BC * bc + WEIGHT_REG * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"{step:6d}  pde={pde.item():.6f}  bc={bc.item():.6f}  total={pde.item() + bc.item():.6f}")
    elapsed = time.perf_counter() - t0

    result = {
        "generator": name, "n_params": n_params, "elapsed_s": round(elapsed, 1),
        "pde_loss": round(pde.item(), 8), "bc_loss": round(bc.item(), 8),
        "total": round(pde.item() + bc.item(), 8),
    }

    # holdout still needs autograd for the PDE residual, so no torch.no_grad() here
    pde_h, bc_h = compute_burgers_loss(model, xh, yh, th, xhb, yhb, thb, uhb, vhb, gen())
    result.update({
        "holdout_pde_loss": round(pde_h.item(), 8),
        "holdout_bc_loss": round(bc_h.item(), 8),
        "holdout_total": round(pde_h.item() + bc_h.item(), 8),
        "pde_ratio": round(pde_h.item() / max(pde.item(), 1e-12), 4),
    })
    print(f"Final    pde={result['pde_loss']:.6f}  bc={result['bc_loss']:.6f}  "
          f"total={result['total']:.6f}  ({elapsed:.1f}s)")
    print(f"Holdout  pde={result['holdout_pde_loss']:.6f}  bc={result['holdout_bc_loss']:.6f}  "
          f"total={result['holdout_total']:.6f}  ratio={result['pde_ratio']:.2f}x")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generators", nargs="+", default=["quantum", "low_rank", "mps"],
                         choices=["quantum", "low_rank", "mps", "classical_frontend"])
    parser.add_argument("--steps", type=int, default=STEPS, help="override adam.steps from config")
    parser.add_argument("--bottleneck_width", type=int, default=64,
                         help="quantum/proj bottleneck width (only affects the 'quantum' generator)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    x, y, t = make_colloc(N_COLLOC, DEVICE)
    xb, yb, tb, ub, vb = make_bc(N_BC, DEVICE)
    torch.manual_seed(SEED + 90000)
    xh, yh, th = make_colloc(N_COLLOC, DEVICE)
    xhb, yhb, thb, uhb, vhb = make_bc(N_BC, DEVICE)

    model = TargetPINN().to(DEVICE)
    gens, n_target = build_generators(DEVICE, args.bottleneck_width)
    print(f"Matched generator parameter budget: {n_target:,} (quantum's actual total, "
          f"bottleneck_width={args.bottleneck_width})")
    print(f"lambda_bc={LAMBDA_BC}  n_colloc={N_COLLOC}  n_bc={N_BC}  steps={args.steps}  "
          f"grad_clip={GRAD_CLIP}  device={DEVICE}")

    out_dir = Path("checkpoints/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_bw{args.bottleneck_width}"
    results = []
    for name in args.generators:
        result = run_one(name, gens[name], model, args.steps,
                          x, y, t, xb, yb, tb, ub, vb, xh, yh, th, xhb, yhb, thb, uhb, vhb)
        results.append(result)
        torch.save(gens[name].state_dict(), out_dir / f"{name}{suffix}_gen.pt")

    print("\n" + "=" * 78)
    print(f"{'generator':10s}  {'params':>10s}  {'train pde':>10s}  {'train bc':>10s}  "
          f"{'holdout total':>14s}  {'ratio':>7s}")
    for r in results:
        print(f"{r['generator']:10s}  {r['n_params']:10,d}  {r['pde_loss']:10.5f}  "
              f"{r['bc_loss']:10.5f}  {r['holdout_total']:14.5f}  {r['pde_ratio']:6.2f}x")

    out_path = out_dir / f"results{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
