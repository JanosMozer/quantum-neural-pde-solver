"""Evaluate a TGV demo checkpoint on the held-out grid (t = 0, T/2, T).

  .venv/bin/python scripts/eval_tgv_grid.py checkpoints/tgv_demo_c
  .venv/bin/python scripts/eval_tgv_grid.py blog/checkpoint/classical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_cond import (
    ConditionedQuantumGenerator,
    ConditionedQuantumGeneratorV2,
)
from qt_pinn.tgv_demo import (
    DirectNSMLP,
    eval_rel_l2,
    gate_pass,
    resolve_device,
    vel_max,
)


def _hidden(cfg: dict, model: str) -> tuple[int, int]:
    if "hidden" in cfg:
        return tuple(cfg["hidden"])
    key = "classical_hidden" if model == "classical" else "qt_hidden"
    return tuple(cfg[key])


def load_run(run_dir: Path, device: torch.device):
    cfg = json.loads((run_dir / "config.json").read_text())
    hidden = _hidden(cfg, cfg["model"])
    if cfg["model"] == "classical":
        model = DirectNSMLP(hidden=hidden, hard_ic=cfg["hard_ic"], t_max=cfg["t_max"])
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
        model.to(device).eval()
        return model, None, cfg

    target = TargetPINNNS(
        fourier="tgv", hard_ic=cfg["hard_ic"], hidden=hidden, t_max=cfg["t_max"],
    ).to(device)
    # Prefer frozen deployed weights (no circuit needed for inference).
    dep = run_dir / "deployed_weights.pt"
    if dep.exists():
        w = torch.load(dep, map_location=device, weights_only=True)
        return target, ("weights", w), cfg

    common = dict(
        in_dim=target.in_dim, h1=hidden[0], h2=hidden[1],
        n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"],
        bottleneck_width=cfg["bottleneck_width"],
        nu_range=(0.05, 0.2), freq_mode="linear",
    )
    if cfg.get("qc_arch", "expect") == "expect":
        gen = ConditionedQuantumGeneratorV2(nu_encode="log", **common)
    else:
        gen = ConditionedQuantumGenerator(**common)
    gen.load_state_dict(torch.load(run_dir / "generator.pt", map_location=device, weights_only=True))
    gen.to(device).eval()
    return target, ("generator", gen), cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-grid", type=int, default=64)
    args = p.parse_args()
    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    model, packed, cfg = load_run(run_dir, device)
    w = None
    if packed is not None:
        kind, obj = packed
        if kind == "weights":
            w = {k: v.to(device) for k, v in obj.items()}
        else:
            with torch.no_grad():
                out = obj(torch.tensor([cfg["nu"]], device=device))
                w = {k: v[0] for k, v in out.items()}
    errs = eval_rel_l2(model, device, cfg["nu"], cfg["t_max"], w, n_grid=args.n_grid)
    vmax = vel_max(errs)
    print(f"{run_dir.name}  model={cfg['model']}  vel_max={100*vmax:.3f}%  "
          f"gate={gate_pass(errs)}")
    for k, e in errs.items():
        if k.startswith("_"):
            print(f"  {k}={e:.4f}" if isinstance(e, float) else f"  {k}={e}")
            continue
        print(f"  t={e['t']:4.2f}  u={100*e['u']:.3f}%  v={100*e['v']:.3f}%  "
              f"p={100*e['p']:.3f}%  vel={100*e['vel']:.3f}%")


if __name__ == "__main__":
    main()
