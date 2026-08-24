"""Compare Burgers VQC-PINN vs matched classical (lower holdout PDE RMS wins).

Usage (scout, ≤1 h quantum):
  .venv/bin/python scripts/compare_burgers_vqc.py checkpoints/burg_vqc_c_scout checkpoints/burg_vqc_q_scout

Usage (full, after scout looks promising):
  .venv/bin/python scripts/compare_burgers_vqc.py checkpoints/burg_vqc_c_s0 checkpoints/burg_vqc_q_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    r = json.loads((path / "results.json").read_text())
    r["_path"] = str(path)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    runs = [load(Path(r)) for r in args.runs]
    classical = [r for r in runs if r["model"] == "classical"]
    quantum = [r for r in runs if r["model"] == "quantum"]
    if not classical or not quantum:
        print("need one classical and one quantum run", file=sys.stderr)
        sys.exit(1)

    def mean(rs, key):
        return sum(r[key] for r in rs) / len(rs)

    legacy = [r for r in runs if "collapse_ratio" not in r]
    degenerate = [r for r in runs if not r.get("valid", True)]
    if legacy:
        print("REFUSING TO COMPARE — runs predate the degeneracy check:")
        for r in legacy:
            print(f"  {r['run_id']}")
        print("Re-run them so collapse_ratio/correction_rms are recorded.")
        sys.exit(2)
    if degenerate:
        print("REFUSING TO COMPARE — degenerate runs detected:")
        for r in degenerate:
            print(f"  {r['run_id']}: collapse_ratio={r.get('collapse_ratio')} "
                  f"correction_rms={r.get('correction_rms')}")
        print("A near-zero or frozen field trivially satisfies the PDE; "
              "re-run with --hard-ic before comparing.")
        sys.exit(2)

    c_h, q_h = mean(classical, "holdout_pde_rms"), mean(quantum, "holdout_pde_rms")
    c_t, q_t = mean(classical, "train_pde_rms"), mean(quantum, "train_pde_rms")
    c_time = mean(classical, "elapsed_s")
    q_time = mean(quantum, "elapsed_s")
    ratio = q_h / max(c_h, 1e-12)

    print(f"classical runs: {len(classical)}   quantum runs: {len(quantum)}\n")
    print(f"{'metric':<20}  {'classical':>10}  {'quantum':>10}  {'Q/C':>6}")
    print(f"{'holdout PDE RMS':<20}  {c_h:10.5f}  {q_h:10.5f}  {ratio:6.2f}")
    print(f"{'train PDE RMS':<20}  {c_t:10.5f}  {q_t:10.5f}  {q_t/max(c_t,1e-12):6.2f}")
    print(f"{'wall-time':<20}  {c_time:9.0f}s  {q_time:9.0f}s  {q_time/max(c_time,1e-12):6.2f}")

    print(f"\nPrimary: holdout PDE RMS Q/C = {ratio:.2f}  (lower quantum is better)")
    if ratio <= 0.90:
        print("VERDICT: quantum wins on holdout PDE residual.")
    elif ratio <= 1.10:
        print("VERDICT: parity — no clear quantum advantage.")
    else:
        print("VERDICT: classical wins.")


if __name__ == "__main__":
    main()
