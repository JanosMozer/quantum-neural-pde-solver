"""Hand-written random Fourier features (RFF) regressor matching Sweke et al.'s
construction: sample M frequencies i.i.d. from a re-weighting distribution D over the
PQC's frequency set (here: uniform over frequency_set(d)), fit a linear model over
[cos(omega.x), sin(omega.x)] features, by ordinary least squares.

Not sklearn's RBFSampler: that assumes a Gaussian/RBF kernel and samples frequencies from
a Gaussian, a different construction from the integer-frequency-set re-weighting this
paper's theorem is actually about. Plain numpy lstsq is all this needs.
"""

import numpy as np


def features(x: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    """x: (N, d). omegas: (M, d). Returns (N, 2M): [cos(omega_m . x), sin(omega_m . x)]."""
    proj = x @ omegas.T  # (N, M)
    return np.concatenate([np.cos(proj), np.sin(proj)], axis=1)


def fit_predict(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray,
    sampled_omegas: np.ndarray,
) -> np.ndarray:
    """Ordinary least squares on the sampled-frequency feature map, no new dependency
    (sklearn isn't installed and isn't needed for plain lstsq).
    """
    phi_train = features(x_train, sampled_omegas)
    weights, *_ = np.linalg.lstsq(phi_train, y_train, rcond=None)
    phi_test = features(x_test, sampled_omegas)
    return phi_test @ weights


def relative_l2_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true))


if __name__ == "__main__":
    from target import frequency_set, target_coefficients, evaluate

    d = 3
    omegas = frequency_set(d)
    coeffs = target_coefficients(d, seed=0)

    rng = np.random.default_rng(2)
    x_train = rng.uniform(-np.pi, np.pi, size=(500, d))
    x_test = rng.uniform(-np.pi, np.pi, size=(500, d))
    y_train = evaluate(x_train, omegas, coeffs)
    y_test = evaluate(x_test, omegas, coeffs)

    # positive control: fit using the TRUE full frequency set (not a random sample),
    # this must recover the target almost exactly, or the regression code is wrong.
    y_pred_exact = fit_predict(x_train, y_train, x_test, omegas)
    err_exact = relative_l2_error(y_pred_exact, y_test)
    assert err_exact < 1e-6, f"exact-recovery self-test failed: rel err {err_exact}"
    print(f"PASS: exact recovery with full frequency set, rel err {err_exact:.2e}")

    # a tiny random sample of frequencies should NOT recover the signal well
    small_sample = omegas[rng.choice(len(omegas), size=3, replace=False)]
    y_pred_small = fit_predict(x_train, y_train, x_test, small_sample)
    err_small = relative_l2_error(y_pred_small, y_test)
    assert err_small > 0.5, f"expected a small random sample to fail badly, got rel err {err_small}"
    print(f"PASS: undersampled frequencies fail as expected, rel err {err_small:.2f}")
