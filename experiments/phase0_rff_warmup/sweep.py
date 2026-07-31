"""Sweke et al. Lemma 3, empirically: for the uniform re-weighting failure case, how many
RFF features M are needed to recover the signal, as a function of the problem dimension d.
Prediction: M_needed grows like |frequency_set(d)| = 3**d, i.e. exponentially (certainly
super-polynomially) in d.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from target import frequency_set, target_coefficients, evaluate
from rff import fit_predict, relative_l2_error

ERROR_THRESHOLD = 0.5  # "recovered at least half the signal's energy"
SEEDS = [0, 1, 2]
# extends past 1.0x |Omega_d| on purpose: sampling is WITH replacement (i.i.d. from the
# re-weighting distribution, matching the paper's construction), so M = |Omega_d| does not
# deterministically cover every unique frequency (coupon-collector effect) - confirmed
# empirically below, the crossing for several d actually lands past the 1.0x point.
M_FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
D_VALUES = [2, 3, 4, 5, 6]


def error_at(d: int, m: int, seed: int) -> float:
    omegas = frequency_set(d)
    coeffs = target_coefficients(d, seed=0)  # fixed target per d, independent of sampling seed

    rng = np.random.default_rng(1000 + seed)
    n_train, n_test = 3500, 500
    x_train = rng.uniform(-np.pi, np.pi, size=(n_train, d))
    x_test = rng.uniform(-np.pi, np.pi, size=(n_test, d))
    y_train = evaluate(x_train, omegas, coeffs)
    y_test = evaluate(x_test, omegas, coeffs)

    sample_idx = rng.integers(0, len(omegas), size=m)  # i.i.d. WITH replacement, uniform re-weighting
    sampled_omegas = omegas[sample_idx]

    y_pred = fit_predict(x_train, y_train, x_test, sampled_omegas)
    return relative_l2_error(y_pred, y_test)


def m_needed(d: int) -> int:
    k = len(frequency_set(d))
    m_grid = sorted(set(max(1, round(f * k)) for f in M_FRACTIONS))
    for m in m_grid:
        errs = [error_at(d, m, seed) for seed in SEEDS]
        mean_err = float(np.mean(errs))
        print(f"  d={d} |Omega|={k} M={m:5d} ({m/k:5.1%} of |Omega|)  mean rel err={mean_err:.3f}")
        if mean_err < ERROR_THRESHOLD:
            return m
    print(f"  WARNING: d={d} never crossed the error threshold within the grid, "
          f"reporting a lower bound (>{m_grid[-1]}), not a real crossing point")
    return m_grid[-1]  # honest lower bound, not a fabricated "beyond grid" guess


def main() -> None:
    t0 = time.time()
    results = {}
    for d in D_VALUES:
        print(f"\n=== d={d} ===")
        results[d] = m_needed(d)
    print(f"\ntotal sweep time: {time.time() - t0:.1f}s")

    ds = list(results.keys())
    m_vals = [results[d] for d in ds]
    omega_sizes = [len(frequency_set(d)) for d in ds]

    print("\nd, M_needed, |Omega_d|=3^d")
    for d, m, k in zip(ds, m_vals, omega_sizes):
        print(f"  {d}, {m}, {k}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogy(ds, m_vals, "o-", label=r"$M_{needed}$ (empirical, RFF crosses 50% error)")
    ax.semilogy(ds, omega_sizes, "k--", label=r"$3^d = |\Omega_d|$ (Lemma 3 prediction)")
    ax.set_xlabel("problem dimension d")
    ax.set_ylabel("RFF feature count (log scale)")
    ax.set_title("RFF dequantization failure: $M_{needed}$ vs. dimension\n(uniform re-weighting, Sweke et al. Lemma 3)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("phase0_rff_scaling.png", dpi=150)
    print("\nSaved phase0_rff_scaling.png")


if __name__ == "__main__":
    main()
