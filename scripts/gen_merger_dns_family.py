"""Generate multi-ν vortex-merger DNS family for Experiment B (parallel-friendly).

  .venv/bin/python scripts/gen_merger_dns_family.py --nu 0.002
  .venv/bin/python scripts/gen_merger_dns_family.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qt_pinn.ns2d_spectral import simulate
from qt_pinn.tgv_demo import resolve_device

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog" / "checkpoint" / "v4" / "dns_family"

# Match v3 reference IC / grid; vary only ν
NUS = [0.002, 0.003, 0.005, 0.008, 0.012, 0.02]
T_MAX = 40.0
N = 256
N_SAVE = 81
GAMMA = 8.0
DELTA = 0.65
PULL_IN = 0.95


def run_one(nu: float, device: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"nu_{nu:.4f}".replace(".", "p")
    dest = OUT / tag
    dest.mkdir(parents=True, exist_ok=True)
    pt = dest / "reference.pt"
    if pt.exists():
        print(f"skip existing {pt}", flush=True)
        return dest
    print(f"DNS ν={nu} n={N} T={T_MAX} on {device}", flush=True)
    t0 = time.time()
    dns = simulate(
        n=N, nu=nu, t_max=T_MAX, n_save=N_SAVE,
        gamma=GAMMA, delta=DELTA, pull_in=PULL_IN,
        device=device, cfl=0.45,
    )
    elapsed = time.time() - t0
    torch.save(dns, pt)
    cfg = {
        "nu": nu, "t_max": T_MAX, "n": N, "n_save": N_SAVE,
        "gamma": GAMMA, "delta": DELTA, "pull_in": PULL_IN,
        "elapsed_s": elapsed, "centers": dns["centers"],
    }
    (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {pt} in {elapsed:.1f}s", flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = str(resolve_device(args.device))
    if args.nu is not None:
        run_one(args.nu, device)
    elif args.all:
        for nu in NUS:
            run_one(nu, device)
    else:
        ap.error("pass --nu FLOAT or --all")


if __name__ == "__main__":
    main()
