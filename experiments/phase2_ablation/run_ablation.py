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
from qt_pinn.pinn_target import TargetPINN, IN_DIM, H1, H2, OUT_DIM
from qt_pinn.qnn_generator import QuantumWeightGenerator, W1_SIZE, W2_SIZE, W3_SIZE, TOTAL_WEIGHTS
from qt_pinn.physics_loss import compute_burgers_loss
from qt_pinn.baselines.low_rank import LowRankGenerator
from qt_pinn.baselines.mps import MPSGenerator

OPTION1_PARAMS = 82  # circuit (81) + gamma (1), the real matched target for the baselines


class DirectGenerator(torch.nn.Module):
    """Task 1, the missing ceiling: no compression at all. Plain nn.Linear layers,
    PyTorch's default init, trained directly. Not matched to any budget on purpose,
    418 free parameters is the point: what can this exact MLP/loss/data reach with
    no generator in the way.

    siren_init=True applies Sitzmann et al. 2020's exact initialization (verbatim
    from the official repo, vsitzmann/siren modules.py) instead of PyTorch's
    default. Only meaningful when TargetPINN(activation="siren") is used downstream
    -- this init is derived specifically to keep sin(30*Wx+b) statistics stable
    through depth, it has no special meaning for a tanh network.
    """

    def __init__(self, siren_init: bool = False) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(IN_DIM, H1)
        self.l2 = torch.nn.Linear(H1, H2)
        self.l3 = torch.nn.Linear(H2, OUT_DIM)
        if siren_init:
            with torch.no_grad():
                n_in = self.l1.weight.size(-1)
                self.l1.weight.uniform_(-1 / n_in, 1 / n_in)
                for layer in (self.l2, self.l3):
                    n_in = layer.weight.size(-1)
                    bound = (6 / n_in) ** 0.5 / 30
                    layer.weight.uniform_(-bound, bound)

    def forward(self, inputs=None):
        return {
            "W1": torch.cat([self.l1.weight.flatten(), self.l1.bias]),
            "W2": torch.cat([self.l2.weight.flatten(), self.l2.bias]),
            "W3": torch.cat([self.l3.weight.flatten(), self.l3.bias]),
        }


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


def build_generator(name: str, mps_target: int = OPTION1_PARAMS, low_rank_target: int = OPTION1_PARAMS,
                     seed: int = 0, siren: bool = False) -> torch.nn.Module:
    if name == "option1":
        return QuantumWeightGenerator(seed=0)
    if name == "direct":
        return DirectGenerator(siren_init=siren)
    if name == "low_rank":
        return FlatToWeightDict(LowRankGenerator(out_dim=TOTAL_WEIGHTS, target_param_count=low_rank_target))
    if name == "mps":
        # MPS_rand_state has its own RNG, not covered by torch.manual_seed (confirmed
        # 2026-08-01: two unseeded calls gave different tensors) -- must pass seed explicitly
        # or every "different seed" MPS run silently reuses whatever quimb's default state gives.
        return FlatToWeightDict(MPSGenerator(out_dim=TOTAL_WEIGHTS, target_param_count=mps_target, seed=seed))
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
    parser.add_argument("generator", choices=["option1", "low_rank", "mps", "direct"])
    parser.add_argument("lambda_bc", type=float)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--n_colloc", type=int, default=1024)
    parser.add_argument("--n_bc", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", action="store_true",
                         help="also evaluate on a large, never-trained-on set (step 4a)")
    parser.add_argument("--holdout_n", type=int, default=4096)
    parser.add_argument("--mps_target", type=int, default=OPTION1_PARAMS,
                         help="only affects generator=mps; step C bond-dim robustness sweep")
    parser.add_argument("--siren", action="store_true",
                         help="TargetPINN(activation='siren') + SIREN init on DirectGenerator "
                              "(only meaningful for generator=direct)")
    parser.add_argument("--low_rank_target", type=int, default=OPTION1_PARAMS,
                         help="only affects generator=low_rank; task 2 capacity-matched scaling")
    args = parser.parse_args()

    # GPU only for the pure-classical generators. Benchmarked directly (not assumed):
    # the quantum circuit (option1) is *slower* on CUDA (28.3ms/step vs 13.4ms/step CPU,
    # 9-qubit/512-amplitude statevectors are too small to amortize kernel-launch overhead).
    # The classical MLP + PDE-residual autograd is a wash at M=4096 (7.3 vs 8.2ms/step) but
    # ~2x faster on CUDA at M=16384 (16.8 vs 8.2ms/step) with GPU time nearly flat as M grows
    # further -- real, growing payoff, zero cost at today's M, so default to it when available.
    device = torch.device("cuda" if torch.cuda.is_available() and args.generator != "option1" else "cpu")

    torch.manual_seed(args.seed)
    model = TargetPINN(activation="siren" if args.siren else "tanh").to(device)
    gen = build_generator(args.generator, mps_target=args.mps_target,
                           low_rank_target=args.low_rank_target, seed=args.seed,
                           siren=args.siren).to(device)
    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)

    x, y, t = make_colloc(args.n_colloc, args.seed)
    xb, yb, tb, ub, vb = make_bc(args.n_bc, args.seed)
    x, y, t = (v.to(device).detach().requires_grad_(True) for v in (x, y, t))
    xb, yb, tb, ub, vb = (v.to(device) for v in (xb, yb, tb, ub, vb))

    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    print(f"generator={args.generator}  device={device}  n_params={n_params}  lambda_bc={args.lambda_bc}  "
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
        "siren": args.siren, "pde_loss": round(pde.item(), 8), "bc_loss": round(bc.item(), 8),
        "total": round(pde.item() + bc.item(), 8), "elapsed_s": round(elapsed, 1),
    }
    print(f"\nFinal  pde={result['pde_loss']:.7f}  bc={result['bc_loss']:.7f}  sum={result['total']:.7f}  "
          f"({elapsed:.1f}s)")

    # ── Step 4a: held-out generalization check ──────────────────────────────
    # Weights are frozen (no opt.step() below); a large, never-trained-on set with a
    # seed offset guaranteed disjoint from any training seed used in this sweep.
    if args.holdout:
        xh, yh, th = make_colloc(args.holdout_n, args.seed + 90000)
        xhb, yhb, thb, uhb, vhb = make_bc(args.holdout_n, args.seed + 90000)
        xh, yh, th = (v.to(device).detach().requires_grad_(True) for v in (xh, yh, th))
        xhb, yhb, thb, uhb, vhb = (v.to(device) for v in (xhb, yhb, thb, uhb, vhb))
        w_final = gen()
        pde_h, bc_h = compute_burgers_loss(model, xh, yh, th, xhb, yhb, thb, uhb, vhb, w_final)
        result["holdout_n"] = args.holdout_n
        result["holdout_pde_loss"] = round(pde_h.item(), 8)
        result["holdout_bc_loss"] = round(bc_h.item(), 8)
        result["holdout_total"] = round(pde_h.item() + bc_h.item(), 8)
        print(f"Held-out (n={args.holdout_n})  pde={result['holdout_pde_loss']:.7f}  "
              f"bc={result['holdout_bc_loss']:.7f}  sum={result['holdout_total']:.7f}")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = ""
    if args.generator == "mps" and args.mps_target != OPTION1_PARAMS:
        suffix = f"_mpstarget{args.mps_target}"
    if args.generator == "low_rank" and args.low_rank_target != OPTION1_PARAMS:
        suffix = f"_target{args.low_rank_target}"
    out_path = out_dir / f"{args.generator}_seed{args.seed}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")

    weights_path = out_dir / f"{args.generator}_seed{args.seed}{suffix}_weights.pt"
    torch.save({k: v.cpu() for k, v in gen.state_dict().items()}, weights_path)
    print(f"Saved {weights_path}")


if __name__ == "__main__":
    main()
