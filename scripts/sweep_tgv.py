"""Cartesian sweep over TGV demo hyperparameters. Sequential; unique run-ids.

  .venv/bin/python scripts/sweep_tgv.py --config configs/tgv_demo/sweep_scout.yaml
  .venv/bin/python scripts/sweep_tgv.py --model classical --preset scout --grid '{"lambda_data":[0.3,1,3],"lr":[0.003,0.005]}'
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep train_tgv_demo.py")
    p.add_argument("--config", default="")
    p.add_argument("--model", choices=["classical", "quantum"], default=None)
    p.add_argument("--preset", default=None)
    p.add_argument("--grid", default="", help='JSON dict of lists, e.g. {"lr":[0.003,0.005]}')
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    file_cfg = load_cfg(args.config)
    models = file_cfg.get("model", [args.model] if args.model else ["classical"])
    if isinstance(models, str):
        models = [models]
    preset = args.preset or file_cfg.get("preset", "scout")
    grid = dict(file_cfg.get("grid") or {})
    if args.grid:
        grid.update(json.loads(args.grid))
    extra = {k: v for k, v in (file_cfg.get("extra") or {}).items()}

    keys = list(grid)
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = list(itertools.product(*values)) if keys else [()]
    jobs = []
    for model in models:
        for combo in combos:
            kv = dict(zip(keys, combo))
            tag = "_".join(f"{k}{v}" for k, v in kv.items()).replace(".", "p")
            run_id = f"tgv_sw_{model[0]}_{preset}" + (f"_{tag}" if tag else "")
            cmd = [
                sys.executable, str(ROOT / "scripts" / "train_tgv_demo.py"),
                "--model", model, "--preset", preset,
                "--run-id", run_id, "--device", args.device,
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            for k, v in {**extra, **kv}.items():
                flag = "--" + k.replace("_", "-")
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                elif isinstance(v, (list, tuple)):
                    cmd.extend([flag, *[str(x) for x in v]])
                else:
                    cmd.extend([flag, str(v)])
            jobs.append((run_id, cmd))

    print(f"{len(jobs)} jobs  preset={preset}  models={models}")
    for run_id, cmd in jobs:
        print(" ", " ".join(cmd[2:]))
    if args.dry_run:
        return

    out_dir = ROOT / "checkpoints" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for run_id, cmd in jobs:
        print(f"\n=== {run_id} ===")
        rc = subprocess.call(cmd, cwd=ROOT)
        rec = {"run_id": run_id, "rc": rc}
        res_path = ROOT / "checkpoints" / run_id / "results.json"
        if res_path.exists():
            rec.update(json.loads(res_path.read_text()))
        summary.append(rec)

    summary_path = out_dir / f"{preset}_{'-'.join(models)}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'run_id':<40} {'vel%':>8} {'time':>7} {'gate':>6}")
    for rec in summary:
        vel = rec.get("vel_rel_l2_max")
        vel_s = f"{100*vel:.3f}" if isinstance(vel, (int, float)) else "  FAIL"
        print(f"{rec['run_id']:<40} {vel_s:>8} {rec.get('elapsed_s','?'):>7} "
              f"{str(rec.get('gate_pass', rec['rc']==0)):>6}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
