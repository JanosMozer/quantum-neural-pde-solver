"""Compare Kolmogorov parametric runs by holdout PDE RMS (lower is better).

Usage:
  .venv/bin/python scripts/compare_kol_parametric.py checkpoints/kol_par_c_s0 checkpoints/kol_par_q_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


KEYS = (
    "mean_pde_rms_in_range",
    "mean_pde_rms_extrap_lo",
    "mean_pde_rms_extrap_hi",
)


def load(path: Path) -> dict:
    r = json.loads((path / "results.json").read_text())
    r["_path"] = str(path)
    return r


def mean_of(runs, key):
    return sum(r[key] for r in runs) / len(runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    runs = [load(Path(r)) for r in args.runs]
    classical = [r for r in runs if r["generator"] == "classical"]
    quantum = [r for r in runs if r["generator"] == "quantum"]
    if not classical or not quantum:
        print("need classical and quantum results.json", file=sys.stderr)
        sys.exit(1)

    print(f"classical: {len(classical)}   quantum: {len(quantum)}\n")
    print(f"{'split':<12}  {'classical':>10}  {'quantum':>10}  {'Q/C':>6}")
    for key, label in zip(KEYS, ("in-range", "extrap-lo", "extrap-hi")):
        c, q = mean_of(classical, key), mean_of(quantum, key)
        print(f"{label:<12}  {c:10.5f}  {q:10.5f}  {q/max(c,1e-12):6.2f}")

    c_t = mean_of(classical, "elapsed_s")
    q_t = mean_of(quantum, "elapsed_s")
    print(f"{'wall-time':<12}  {c_t:9.0f}s  {q_t:9.0f}s  {q_t/max(c_t,1e-12):6.2f}")

    c_ex = 0.5 * (mean_of(classical, "mean_pde_rms_extrap_lo")
                  + mean_of(classical, "mean_pde_rms_extrap_hi"))
    q_ex = 0.5 * (mean_of(quantum, "mean_pde_rms_extrap_lo")
                  + mean_of(quantum, "mean_pde_rms_extrap_hi"))
    ratio = q_ex / max(c_ex, 1e-12)
    print(f"\nextrap PDE RMS Q/C = {ratio:.2f}  (primary; lower quantum is better)")
    if ratio <= 0.90:
        print("VERDICT: quantum lower extrap PDE RMS — keep investigating.")
    elif ratio <= 1.15:
        print("VERDICT: parity on PDE residual — no advantage yet.")
    else:
        print("VERDICT: classical wins on Kolmogorov parametric task.")


if __name__ == "__main__":
    main()
