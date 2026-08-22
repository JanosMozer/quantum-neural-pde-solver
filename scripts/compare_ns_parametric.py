"""Compare classical vs quantum parametric NS runs and print a go/no-go verdict.

Usage:
  .venv/bin/python scripts/compare_ns_parametric.py \\
      checkpoints/ns_par_c_s0 checkpoints/ns_par_q_s0
  # optional more seeds:
  .venv/bin/python scripts/compare_ns_parametric.py \\
      checkpoints/ns_par_c_s{0,1,2} checkpoints/ns_par_q_s{0,1,2}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


KEYS = (
    "mean_spacetime_rel_l2_in_range",
    "mean_spacetime_rel_l2_extrap_lo",
    "mean_spacetime_rel_l2_extrap_hi",
)


def load(path: Path) -> dict:
    r = json.loads((path / "results.json").read_text())
    r["_path"] = str(path)
    return r


def mean_of(runs: list[dict], key: str) -> float:
    return sum(r[key] for r in runs) / len(runs)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def verdict(c_in: float, q_in: float, c_lo: float, q_lo: float,
            c_hi: float, q_hi: float, c_time: float, q_time: float) -> str:
    """
    Worth using quantum only if it is at least as accurate on the hard test
    (extrapolation) and not much worse on cost. In-range alone is too easy
    to game with memorisation of the training interval.
    """
    lines = []
    # Primary: extrapolation mean (lo+hi)/2
    c_ex = 0.5 * (c_lo + c_hi)
    q_ex = 0.5 * (q_lo + q_hi)
    ratio_ex = q_ex / max(c_ex, 1e-12)
    ratio_in = q_in / max(c_in, 1e-12)
    ratio_t = q_time / max(c_time, 1e-12)

    lines.append(f"in-range   Q/C error ratio = {ratio_in:.2f}")
    lines.append(f"extrap     Q/C error ratio = {ratio_ex:.2f}  "
                 f"(primary decision metric)")
    lines.append(f"wall-time  Q/C ratio       = {ratio_t:.2f}")

    # Decision thresholds (deliberately strict: TGV is an easy family)
    match_ex = ratio_ex <= 1.15          # within 15% of classical on extrap
    beat_ex = ratio_ex <= 0.90           # ≥10% better on extrap
    not_awful_in = ratio_in <= 1.50
    cost_ok = ratio_t <= 3.0             # sim is slower; >3x is a hard sell

    if beat_ex and not_awful_in:
        if cost_ok:
            lines.append("VERDICT: KEEP investigating quantum — "
                         "beats classical on extrapolation.")
        else:
            lines.append("VERDICT: accuracy win, but wall-time >3x — "
                         "only interesting if a real device / better simulator "
                         "can close the cost gap.")
    elif match_ex and not_awful_in and cost_ok:
        lines.append("VERDICT: quantum matches classical (parity). "
                     "Worth keeping as a baseline; not yet evidence of advantage. "
                     "Move to a harder family (Kolmogorov) before claiming more.")
    else:
        lines.append("VERDICT: classical wins. Drop this QC design for TGV; "
                     "either redesign the circuit or switch PDE. "
                     "Do not interpret as 'quantum PINNs don't work' — "
                     "only that this hypernetwork+TGV setup lost.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="checkpoint dirs (classical and quantum)")
    args = p.parse_args()

    runs = [load(Path(r)) for r in args.runs]
    classical = [r for r in runs if r["generator"] == "classical"]
    quantum = [r for r in runs if r["generator"] == "quantum"]
    if not classical or not quantum:
        print("need at least one classical and one quantum results.json", file=sys.stderr)
        sys.exit(1)

    print(f"classical runs: {len(classical)}   quantum runs: {len(quantum)}")
    print()
    print(f"{'split':<12}  {'classical':>10}  {'quantum':>10}  {'Q/C':>6}")
    for key, label in (
        ("mean_spacetime_rel_l2_in_range", "in-range"),
        ("mean_spacetime_rel_l2_extrap_lo", "extrap-lo"),
        ("mean_spacetime_rel_l2_extrap_hi", "extrap-hi"),
    ):
        c = mean_of(classical, key)
        q = mean_of(quantum, key)
        print(f"{label:<12}  {fmt_pct(c):>10}  {fmt_pct(q):>10}  {q/max(c,1e-12):6.2f}")

    c_t = mean_of(classical, "elapsed_s")
    q_t = mean_of(quantum, "elapsed_s")
    print(f"{'wall-time':<12}  {c_t:9.0f}s  {q_t:9.0f}s  {q_t/max(c_t,1e-12):6.2f}")
    print()
    print(verdict(
        mean_of(classical, "mean_spacetime_rel_l2_in_range"),
        mean_of(quantum, "mean_spacetime_rel_l2_in_range"),
        mean_of(classical, "mean_spacetime_rel_l2_extrap_lo"),
        mean_of(quantum, "mean_spacetime_rel_l2_extrap_lo"),
        mean_of(classical, "mean_spacetime_rel_l2_extrap_hi"),
        mean_of(quantum, "mean_spacetime_rel_l2_extrap_hi"),
        c_t, q_t,
    ))


if __name__ == "__main__":
    main()
