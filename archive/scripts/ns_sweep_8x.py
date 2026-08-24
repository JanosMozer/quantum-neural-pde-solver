"""Run 8 NS-PINN training runs in parallel for parameter scanning.

This is a *probe sweep*: use fewer Adam steps (default 6000) to quickly
identify promising hyperparameters, then rerun the best config with full steps.

Run with (example CPU sweep):
  .venv/bin/python scripts/ns_sweep_8x.py --device cpu --cpu-threads 4 --adam-steps 6000

or GPU sweep:
  .venv/bin/python scripts/ns_sweep_8x.py --device cuda --batch-multiplier 1 --adam-steps 6000
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--cpu-threads", type=int, default=4,
                   help="Torch intra-op CPU threads per training process.")
    p.add_argument("--batch-multiplier", type=int, default=1,
                   help="Scales collocation/BC sizes per process.")
    p.add_argument("--adam-steps", type=int, default=6000)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--weight-reg", type=float, default=0.01)
    p.add_argument("--grid", choices=["high_ic", "capacity_only"], default="high_ic",
                   help="8-point hyperparameter preset.")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run even if checkpoints/<run-id>/results.json already exists.")
    return p.parse_args()


def combos(grid: str, weight_reg: float) -> list[dict[str, Any]]:
    # Diagnosis from ns_run_0005:
    # - pde_loss ~ 5e-4 (almost perfect), while bc_loss ~ 0.564, and
    # - relative L2 errors are ~100%.
    #
    # Since the trivial solution u=v=p=0 has bc_mse ~ 0.5625 for this exact IC,
    # the optimizer is collapsing to the trivial PDE-satisfying solution.
    #
    # Therefore next probe should use stronger IC forcing and larger capacity.
    if grid == "high_ic":
        # Strong lambda_bc + bottleneck_width up to 256.
        pts = [
            {"lambda_bc": 10.0,  "bottleneck_width": 64,  "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 10.0,  "bottleneck_width": 128, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 10.0,  "bottleneck_width": 256, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 50.0,  "bottleneck_width": 64,  "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 50.0,  "bottleneck_width": 128, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 50.0,  "bottleneck_width": 256, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 100.0, "bottleneck_width": 128, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 100.0, "bottleneck_width": 256, "fourier_sigma": 2.0, "weight_reg": weight_reg},
        ]
        assert len(pts) == 8
        return pts

    if grid == "capacity_only":
        # Hold lambda_bc high and scan capacity up to 512.
        pts = [
            {"lambda_bc": 50.0,  "bottleneck_width": 128, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 50.0,  "bottleneck_width": 256, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 50.0,  "bottleneck_width": 512, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 100.0, "bottleneck_width": 128, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 100.0, "bottleneck_width": 256, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 100.0, "bottleneck_width": 512, "fourier_sigma": 2.0, "weight_reg": weight_reg},
            {"lambda_bc": 50.0,  "bottleneck_width": 256, "fourier_sigma": 1.0, "weight_reg": weight_reg},
            {"lambda_bc": 100.0, "bottleneck_width": 512, "fourier_sigma": 1.0, "weight_reg": weight_reg},
        ]
        assert len(pts) == 8
        return pts

    raise ValueError(f"unknown grid {grid!r}")


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    base = root / "checkpoints"
    log_dir = base / "ns_sweep_logs"
    safe_makedirs(log_dir)
    skip_existing = not args.no_skip_existing

    runs = combos(args.grid, args.weight_reg)
    procs: list[subprocess.Popen] = []
    active_run_ids: list[str] = []
    skipped: list[str] = []

    # Launch all 8 at once.
    for i, cfg in enumerate(runs):
        run_id = (
            f"ns_sweep_{i:02d}_lam{cfg['lambda_bc']}_bw{cfg['bottleneck_width']}"
            f"_f{cfg['fourier_sigma']}_wreg{cfg['weight_reg']}"
        ).replace(".", "p")

        out_log = log_dir / f"{run_id}.log"
        results_path = base / run_id / "results.json"
        if skip_existing and results_path.exists():
            print(f"Skipping existing run: {run_id}")
            skipped.append(run_id)
            continue
        active_run_ids.append(run_id)

        cmd = [
            str(root / ".venv" / "bin" / "python"),
            str(root / "scripts" / "train_ns.py"),
            "--device", args.device,
            "--cpu-threads", str(args.cpu_threads),
            "--batch-multiplier", str(args.batch_multiplier),
            "--adam-steps", str(args.adam_steps),
            "--log-every", str(args.log_every),
            "--lambda-bc", str(cfg["lambda_bc"]),
            "--weight-reg", str(args.weight_reg),
            "--bottleneck-width", str(cfg["bottleneck_width"]),
            "--fourier-sigma", str(cfg["fourier_sigma"]),
            "--run-id", run_id,
            "--seed", str(args.seed_base + i),
        ]

        print("Launching:", " ".join(cmd))
        f = open(out_log, "w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy()))

    # Wait for all.
    exit_codes: list[int] = []
    for proc in procs:
        exit_codes.append(proc.wait())

    # Summarize any runs that produced results.json.
    print("\nSweep summary:")
    for run_id, code in zip(active_run_ids, exit_codes):
        if code != 0:
            print(f"  {run_id}: process exit_code={code}")
            continue
        results_path = base / run_id / "results.json"
        if not results_path.exists():
            print(f"  {run_id}: no results.json (unexpected)")
            continue
        res = json.loads(results_path.read_text())
        exact_l2 = res.get("exact_l2", {})
        # scalarize by averaging u/v/p at t=1.0 if present, else use total
        score = None
        if "1.0" in exact_l2:
            score = (exact_l2["1.0"]["u"] + exact_l2["1.0"]["v"] + exact_l2["1.0"]["p"]) / 3
        else:
            score = res.get("total", None)
        print(f"  {run_id}: total={res.get('total')} pde={res.get('pde_loss')} bc={res.get('bc_loss')} score={score}")

    if skipped:
        print("\nSkipped (existing results):")
        for run_id in skipped:
            res_path = base / run_id / "results.json"
            if res_path.exists():
                res = json.loads(res_path.read_text())
                print(f"  {run_id}: total={res.get('total')} pde={res.get('pde_loss')} bc={res.get('bc_loss')}")


if __name__ == "__main__":
    main()

