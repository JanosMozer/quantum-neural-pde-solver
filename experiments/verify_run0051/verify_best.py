"""Holdout-verify the two most recent, most relevant checkpoints:
run_0057 (learned_proj / quantum, stated "best model so far") and
run_0059 (classical ablation, ClassicalWeightGenerator). Neither
run saved weights, and neither was ever checked against held-out points,
despite THEORY.md itself naming held-out relative-L2 error the "gold
standard" metric. Both sit at Nt/M ~= 188-189 (60083-60514 params on
256-576 training points), deep in the regime already shown this session
to produce severe overfitting. Reproduces each exactly via the project's
own training code (scripts.train's make_colloc/make_bc, same generator
classes), not reimplemented, then adds the missing holdout check.

Usage: python verify_janos_best.py learned_proj   # reproduces run_0057
       python verify_janos_best.py classical       # reproduces run_0059
"""

import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from qt_pinn.pinn_target import TargetPINN
from pdes.burgers2d.physics_loss import compute_burgers_loss
from scripts.train import make_colloc, make_bc

RECIPES = {
    # exact values from checkpoints/run_0057/config.json
    "learned_proj": dict(n_colloc=256, n_bc=64, lambda_bc=7.0, weight_decay=0.001,
                          weight_reg=0.0, structured_bc=False, steps=18000, seed=0),
    # exact values from checkpoints/run_0059/config.json + config.yaml at time of that run
    "classical": dict(n_colloc=512, n_bc=64, lambda_bc=7.0, weight_decay=0.0,
                       weight_reg=0.1, structured_bc=False, steps=18000, seed=0),
}


def build_generator(name: str):
    if name == "learned_proj":
        from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP
        return QuantumWeightGeneratorLP()
    if name == "classical":
        from qt_pinn.learned_proj.classical_generator import ClassicalWeightGenerator
        return ClassicalWeightGenerator()
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generator", choices=list(RECIPES))
    args = parser.parse_args()
    cfg = RECIPES[args.generator]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model = TargetPINN()
    gen = build_generator(args.generator)
    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"generator={args.generator}  n_params={n_params}  cfg={cfg}")

    x, y, t = make_colloc(cfg["n_colloc"])
    xb, yb, tb, ub, vb = make_bc(cfg["n_bc"], structured=cfg["structured_bc"])

    opt = torch.optim.Adam(params, lr=0.01, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"], eta_min=1e-5)

    t0 = time.perf_counter()
    for step in range(cfg["steps"]):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        reg = sum(v.pow(2).mean() for v in w.values()) if cfg["weight_reg"] > 0 else 0.0
        loss = pde + cfg["lambda_bc"] * bc + cfg["weight_reg"] * reg
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"{step:6d}  pde={pde.item():.6f}  bc={bc.item():.6f}  total={pde.item()+bc.item():.6f}")
    elapsed = time.perf_counter() - t0

    print(f"\nTrain (matches the checkpoint's own set)  pde={pde.item():.7f}  bc={bc.item():.7f}  "
          f"sum={pde.item()+bc.item():.7f}  ({elapsed:.1f}s)")

    # fresh draw continuing the same global RNG stream -- disjoint from training data,
    # same mechanism scripts/train.py itself uses, not a separately-seeded generator
    xh, yh, th = make_colloc(4096)
    xhb, yhb, thb, uhb, vhb = make_bc(4096, structured=False)
    pde_h, bc_h = compute_burgers_loss(model, xh, yh, th, xhb, yhb, thb, uhb, vhb, gen())
    print(f"Held-out (n=4096, never trained on)  pde={pde_h.item():.7f}  bc={bc_h.item():.7f}  "
          f"sum={pde_h.item()+bc_h.item():.7f}")

    result = {
        "generator": args.generator, "n_params": n_params, **cfg,
        "train_pde": round(pde.item(), 8), "train_bc": round(bc.item(), 8),
        "train_total": round(pde.item() + bc.item(), 8),
        "holdout_pde": round(pde_h.item(), 8), "holdout_bc": round(bc_h.item(), 8),
        "holdout_total": round(pde_h.item() + bc_h.item(), 8),
        "pde_ratio": round(pde_h.item() / pde.item(), 2) if pde.item() > 0 else None,
        "elapsed_s": round(elapsed, 1),
    }
    out = Path(__file__).resolve().parent / f"result_{args.generator}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
