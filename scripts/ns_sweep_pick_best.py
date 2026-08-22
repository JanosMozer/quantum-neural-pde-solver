"""Pick the best NS sweep run and print a full training command.

Usage:
  .venv/bin/python scripts/ns_sweep_pick_best.py --device cuda --adam-steps-full 18000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--batch-multiplier", type=int, default=2)
    p.add_argument("--adam-steps-full", type=int, default=18000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Adds to the sweep seed when rerunning (default: 0).")
    return p.parse_args()


def score_from_results(res: dict[str, Any]) -> float:
    exact_l2 = res.get("exact_l2", {})
    if "1.0" in exact_l2:
        u = exact_l2["1.0"]["u"]
        v = exact_l2["1.0"]["v"]
        p = exact_l2["1.0"]["p"]
        return float((u + v + p) / 3.0)
    # fallback: lower total loss
    return float(res.get("total", 1e9))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    ckpt = root / "checkpoints"

    best = None  # tuple(score, run_dir, cfg, res)
    for d in sorted(ckpt.glob("ns_sweep_*")):
        results_path = d / "results.json"
        cfg_path = d / "config.json"
        if not results_path.exists() or not cfg_path.exists():
            continue
        res = json.loads(results_path.read_text())
        cfg = json.loads(cfg_path.read_text())
        s = score_from_results(res)
        if best is None or s < best[0]:
            best = (s, d, cfg, res)

    if best is None:
        raise SystemExit("No ns_sweep_* runs with results/config found in checkpoints/.")

    score, run_dir, cfg, res = best
    run_name = run_dir.name

    rerun_id = f"{run_name}_FULL"

    # Build command
    cmd = [
        str(root / ".venv" / "bin" / "python"),
        str(root / "scripts" / "train_ns.py"),
        "--device", args.device,
        "--run-id", rerun_id,
        "--seed", str(int(cfg["seed"]) + args.seed_offset),
        "--batch-multiplier", str(args.batch_multiplier),
        "--adam-steps", str(args.adam_steps_full),
        "--log-every", str(args.log_every),
        "--lambda-bc", str(cfg["lambda_bc"]),
        "--weight-reg", str(cfg["weight_reg"]),
        "--bottleneck-width", str(cfg["bottleneck_width"]),
        "--fourier-sigma", str(cfg["fourier_sigma"]),
    ]
    if args.device == "cpu" and args.cpu_threads > 0:
        cmd += ["--cpu-threads", str(args.cpu_threads)]
    elif args.device == "cuda":
        # keep threads unspecified unless user set cpu-threads explicitly
        if args.cpu_threads > 0:
            cmd += ["--cpu-threads", str(args.cpu_threads)]

    print("Best sweep run:", run_name)
    print("  score(mean rel-L2 at t=1.0):", score)
    print("  total/pde/bc:", res.get("total"), res.get("pde_loss"), res.get("bc_loss"))
    print("\nRerun command (copy/paste):")
    print(" ".join(cmd))


if __name__ == "__main__":
    main()

