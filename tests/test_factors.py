# -*- coding: utf-8 -*-
"""Tests of factor extraction.

Pins the numerical equivalences and conventions that factor extraction relies on.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from volfactor.factors import (
    extract_full_sample_factors,
    extract_pca,
    extract_spca_residual,
    har_residuals,
    scaling_betas,
)
from volfactor.io import HAR_COLS, stock_columns


def test_pca_solver_full_matches_variance_ratio_spec(train_window):
    """``PCA(n_components=4, svd_solver="full")`` must equal the first 4 columns of
    ``PCA(n_components=0.99)`` (exact SVD, deterministic).
    """
    X, _ = train_window
    stocks = X[stock_columns(X)]

    original = PCA(n_components=0.99).fit(stocks)
    ours = PCA(n_components=4, svd_solver="full").fit(stocks)

    assert original.n_components_ >= 4
    np.testing.assert_allclose(
        np.abs(original.transform(stocks)[:, :4]),
        np.abs(ours.transform(stocks)),
        atol=1e-9,
    )


def test_pca_components_are_nested(train_window):
    """Principal components are nested: the first 4 columns are identical whether 4 or 8 are kept."""
    X, _ = train_window
    stocks = X[stock_columns(X)]

    f4 = PCA(n_components=4, svd_solver="full").fit_transform(stocks)
    f8 = PCA(n_components=8, svd_solver="full").fit_transform(stocks)

    np.testing.assert_allclose(f4, f8[:, :4], atol=1e-9)


def test_scaling_betas_match_sklearn_loop(train_window):
    """The closed form ``cov(x, r) / var(x)`` must equal column-by-column ``LinearRegression``."""
    X, Y = train_window
    stocks = stock_columns(X)[:30]  # 30 stocks suffice; all 332 would be slow

    resid = har_residuals(X, Y)
    fast = scaling_betas(X[stocks], resid)

    ols = LinearRegression()
    slow = np.array(
        [ols.fit(X[[c]], resid.reshape(-1, 1)).coef_[0][-1] for c in stocks]
    )

    np.testing.assert_allclose(fast, slow, rtol=1e-10)


def test_har_residuals_are_orthogonal_to_design(train_window):
    """Residuals must be orthogonal to the HAR design matrix — the OLS first-order condition."""
    X, Y = train_window
    resid = har_residuals(X, Y)

    design = np.column_stack([np.ones(len(X)), X[HAR_COLS].to_numpy()])
    np.testing.assert_allclose(design.T @ resid, 0.0, atol=1e-6)


def test_extract_pca_output_shape(train_window):
    X, _ = train_window
    stocks = X[stock_columns(X)]
    f_train, f_apply = extract_pca(stocks.iloc[:1000], stocks.iloc[1000:])

    assert list(f_train.columns) == ["VF1", "VF2", "VF3", "VF4"]
    assert len(f_train) == 1000
    assert len(f_apply) == len(stocks) - 1000


def test_extract_spca_uses_training_betas_only(train_window):
    """β must depend on the training window only: changing the application data must not affect training factors.

    This guarantees that residual sPCA has no look-ahead.
    """
    X, Y = train_window
    X_tr, Y_tr = X.iloc[:1000], Y.iloc[:1000]

    f_train_a, _ = extract_spca_residual(X_tr, Y_tr, X.iloc[1000:])

    perturbed = X.iloc[1000:].copy()
    perturbed.iloc[:, 5:] *= 1.5  # large perturbation of the application period
    f_train_b, _ = extract_spca_residual(X_tr, Y_tr, perturbed)

    np.testing.assert_allclose(f_train_a.to_numpy(), f_train_b.to_numpy(), atol=1e-12)


def test_full_sample_factors_sign_convention(panel):
    """Sign convention of the paper: sVF_k must correlate positively with VF_k.

    The four correlations in Table 1 are 0.997 / 0.983 / 0.974 / 0.916, all positive.
    """
    X, Y = panel
    vf, svf = extract_full_sample_factors(X, Y)

    assert list(vf.columns) == ["VF1", "VF2", "VF3", "VF4"]
    assert list(svf.columns) == ["sVF1", "sVF2", "sVF3", "sVF4"]

    for k in range(4):
        rho = np.corrcoef(vf.iloc[:, k], svf.iloc[:, k])[0, 1]
        assert rho > 0, f"sVF{k + 1} should correlate positively with VF{k + 1}, got {rho:.3f}"


def test_full_sample_factors_are_deterministic(panel):
    """The same input must give identical factors — the sign convention must not depend on randomness."""
    X, Y = panel
    a_vf, a_svf = extract_full_sample_factors(X, Y)
    b_vf, b_svf = extract_full_sample_factors(X, Y)

    np.testing.assert_array_equal(a_vf.to_numpy(), b_vf.to_numpy())
    np.testing.assert_array_equal(a_svf.to_numpy(), b_svf.to_numpy())


# --------------------------------------------------------------------------
# Correlation-matrix specification (§4.3.3 / Table 10)
# --------------------------------------------------------------------------

def test_correlation_pca_matches_manual_eigendecomposition(train_window):
    """Must match the correlation-matrix algorithm element by element.

    The algorithm has two steps on different scales, which are easy to mix up::

        R = corrcoef(X_train)        # loadings from the correlation matrix (scale-free)
        F = (X − mean(X_train)) · V  # but the projected data are on the original scale, only demeaned

    The reference is recomputed by hand here.
    """
    from scipy.linalg import eigh

    X, _ = train_window
    stocks = X[stock_columns(X)]

    eigvals, eigvecs = eigh(np.corrcoef(stocks.to_numpy().T))
    basis = eigvecs[:, np.argsort(eigvals)[::-1][:4]]
    expected = (stocks.to_numpy() - stocks.to_numpy().mean(axis=0)) @ basis

    got, _ = extract_pca(stocks, stocks, use_correlation=True, align_signs=False)

    np.testing.assert_allclose(np.abs(got.to_numpy()), np.abs(expected), rtol=1e-10)


def test_correlation_pca_projects_uncentered_scale(train_window):
    """The projected data are only demeaned, not standardized.

    Eigenvectors come from the correlation matrix, but the data projected onto them are on the
    original scale (only demeaned). Projecting z-scored data instead would cancel the β-weighting of sPCA
    — see :func:`test_correlation_spca_keeps_beta_weighting`.
    """
    X, _ = train_window
    stocks = X[stock_columns(X)]

    ours, _ = extract_pca(stocks, stocks, use_correlation=True, align_signs=False)

    standardized = (stocks - stocks.mean()) / stocks.std()
    textbook, _ = extract_pca(
        standardized, standardized, use_correlation=True, align_signs=False
    )

    # Same direction (same loadings) but a different scale — otherwise the implementation would be the standard one
    rho = abs(np.corrcoef(ours.iloc[:, 0], textbook.iloc[:, 0])[0, 1])
    assert rho > 0.99
    ratio = ours.iloc[:, 0].std() / textbook.iloc[:, 0].std()
    assert not np.isclose(ratio, 1.0, atol=1e-3), "projection should keep the original scale"


def test_correlation_spca_keeps_beta_weighting(train_window):
    """With correlation-matrix factors, sVF must still differ from VF.

    Standardizing after the β-weighting would rescale every column to unit variance and cancel
    the weighting, making sVF identical to VF. In Table 10 the two differ
    (1-day: VF1 = 1.367 vs sVF1 = 1.498); this test guards that difference.
    """
    X, Y = train_window
    stocks = X[stock_columns(X)]

    vf, _ = extract_pca(stocks, stocks, use_correlation=True)
    svf, _ = extract_spca_residual(X, Y, X, use_correlation=True)

    rho = abs(np.corrcoef(vf["VF1"], svf["sVF1"])[0, 1])
    assert rho < 0.9999, "sVF1 should not collapse to VF1 — the β-weighting was cancelled"


def test_correlation_flag_defaults_to_covariance(train_window):
    """The default must use the covariance matrix, as in the main results (Tables 3/4)."""
    X, _ = train_window
    stocks = X[stock_columns(X)]

    default, _ = extract_pca(stocks, stocks)
    explicit, _ = extract_pca(stocks, stocks, use_correlation=False)

    np.testing.assert_array_equal(default.to_numpy(), explicit.to_numpy())
