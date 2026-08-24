"""The Sweke et al. Section 4.5 failure construction: a data-reuploading PQC's Fourier
series, target function built from ALL frequencies in a d-dimensional integer frequency
set with a flat (uniform-magnitude) spectrum, matching one data-upload layer (frequencies
in {-1,0,1} per dimension, the simplest non-trivial data-reuploading spectrum).

This is the "uniform re-weighting" failure case from Lemma 3: sampling frequencies
uniformly at random from a set whose size grows exponentially in d gives p_max ~ 1/3**d,
decaying inverse-exponentially, forcing M = Omega(3**d) for RFF to recover the signal.
"""

import itertools
import numpy as np


def frequency_set(d: int) -> np.ndarray:
    """All (d,)-integer frequency vectors with entries in {-1, 0, 1}. Size 3**d."""
    return np.array(list(itertools.product([-1, 0, 1], repeat=d)))


def target_coefficients(d: int, seed: int) -> np.ndarray:
    """Fixed +-1 (Rademacher) coefficient per frequency, flat spectrum: every frequency
    contributes equally to the target's energy, which is exactly the worst case for RFF
    dequantization, no single dominant frequency for a small M to get lucky on.
    """
    omegas = frequency_set(d)
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=len(omegas))


def evaluate(x: np.ndarray, omegas: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """f*(x) = (1/sqrt(|Omega|)) * sum_omega c_omega * cos(omega . x).

    x: (N, d) array. omegas: (K, d) array of frequency vectors. coeffs: (K,) array.
    Returns: (N,) array of target values.
    """
    proj = x @ omegas.T  # (N, K)
    return (np.cos(proj) @ coeffs) / np.sqrt(len(coeffs))


if __name__ == "__main__":
    d = 3
    omegas = frequency_set(d)
    assert omegas.shape == (27, d), f"expected 27 frequencies for d=3, got {omegas.shape}"
    coeffs = target_coefficients(d, seed=0)
    assert coeffs.shape == (27,)
    assert set(np.unique(coeffs)) <= {-1.0, 1.0}

    rng = np.random.default_rng(1)
    x = rng.uniform(-np.pi, np.pi, size=(5, d))
    y = evaluate(x, omegas, coeffs)
    assert y.shape == (5,)
    print(f"PASS: frequency_set/target_coefficients/evaluate shapes and ranges ok, sample y={y}")
