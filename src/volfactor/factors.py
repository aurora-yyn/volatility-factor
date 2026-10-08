# -*- coding: utf-8 -*-
"""Volatility factor extraction: PCA (VF) and residual scaled PCA (sVF).

This is the single implementation of factor extraction in the repository; all
chapter scripts call it.

The two methods
---------------
- **PCA (VF)**: principal component analysis on the LV panel of the 332 stocks.
  Unsupervised.
- **Residual sPCA (sVF)**: the three-step procedure of §2.2.2 — regress the target on
  the three HAR terms to obtain residuals, regress the residuals on each stock
  separately to obtain β_n, then apply PCA to the scaled panel (β_n · X_n).
  Supervised: stocks with more predictive power for the target receive larger weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA

from .io import HAR_COLS

#: Number of factors used in the paper (§2.2.1 / §2.2.2, supported by the
#: Pelger (2019) test in §3.3)
N_FACTORS = 4


def _fit_pca(train: pd.DataFrame, n_factors: int) -> PCA:
    """Fit PCA on the training panel.

    Uses the exact full SVD. With an integer ``n_components``, sklearn's default
    ``svd_solver="auto"`` switches to a randomized (approximate) SVD at this problem
    size, so the solver is set explicitly. Principal components are nested, so the
    first 4 components equal the first 4 obtained when more components are kept.
    """
    return PCA(n_components=n_factors, svd_solver="full").fit(train)


def _signs(pca: PCA) -> np.ndarray:
    """Sign convention: the loading of each component on the **first stock** is positive.

    The sign of a principal component is arbitrary — if v is an eigenvector, so is −v.
    sklearn's internal convention has changed across versions, so a version-independent
    convention is fixed here. Signs do not affect out-of-sample R² or other results —
    the regression coefficient absorbs the sign.
    """
    return np.sign(pca.components_[:, 0])


def _as_factor_frame(arr: np.ndarray, index: pd.Index, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        arr, index=index, columns=[f"{prefix}{i + 1}" for i in range(arr.shape[1])]
    )


def _project(
    train: pd.DataFrame,
    apply_: pd.DataFrame,
    n_factors: int,
    use_correlation: bool,
    align_signs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the training / application panels onto the first ``n_factors`` components.

    ``use_correlation=False`` (default, main results) uses the covariance matrix,
    i.e. standard PCA.

    ``use_correlation=True`` is the approach of §4.3.3 (Table 10): **eigenvectors are
    taken from the correlation matrix, and the projected data are only demeaned, not
    divided by their standard deviations**::

        R = corrcoef(X_train)                    # loadings from the correlation matrix
        F = (X − mean(X_train)) · V              # projection keeps the original scale

    The projected data are not standardized so that the β-weighting of sPCA is
    preserved (standardizing would rescale every column to unit variance and undo it).
    """
    if use_correlation:
        mu = train.to_numpy().mean(axis=0)
        eigvals, eigvecs = eigh(np.corrcoef(train.to_numpy().T))
        basis = eigvecs[:, np.argsort(eigvals)[::-1][:n_factors]]
        s = np.sign(basis[0]) if align_signs else 1.0
        return (
            ((train.to_numpy() - mu) @ basis) * s,
            ((apply_.to_numpy() - mu) @ basis) * s,
        )

    pca = _fit_pca(train, n_factors)
    s = _signs(pca) if align_signs else 1.0
    return pca.transform(train) * s, pca.transform(apply_) * s


def extract_pca(
    X_train: pd.DataFrame,
    X_apply: pd.DataFrame,
    n_factors: int = N_FACTORS,
    prefix: str = "VF",
    align_signs: bool = True,
    use_correlation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PCA volatility factors (VF, §2.2.1).

    Parameters
    ----------
    X_train
        LV panel of the training window, without the three HAR columns
        (select columns with :func:`volfactor.io.stock_columns`).
    X_apply
        Panel to which the same transformation is applied (a single test row or a
        whole sample).
    n_factors
        Number of factors, default 4.
    prefix
        Prefix of the output columns, default ``VF`` -> VF1…VF4.
    align_signs
        Whether to align component signs with the convention of :func:`_signs`.
    use_correlation
        If True, loadings are eigenvectors of the correlation matrix (see
        :func:`_project`), as in the robustness check of §4.3.3 (Table 10). The default
        False uses the covariance matrix, as in the main results.

    Returns
    -------
    (F_train, F_apply)
        Factors for the training and application periods, columns
        ``{prefix}1`` … ``{prefix}{n_factors}``.
    """
    f_train, f_apply = _project(
        X_train, X_apply, n_factors, use_correlation, align_signs
    )
    return (
        _as_factor_frame(f_train, X_train.index, prefix),
        _as_factor_frame(f_apply, X_apply.index, prefix),
    )


def pca_loadings(
    panel: pd.DataFrame, n_factors: int = N_FACTORS, align_signs: bool = True
) -> pd.DataFrame:
    """Loading matrix of the full-sample PCA (used in §5.1 to analyse factor composition).

    Parameters
    ----------
    panel
        Stock LV panel without the three HAR columns.

    Returns
    -------
    DataFrame
        Rows ``VF1`` … ``VF{n_factors}``, columns are tickers, entries are the loadings.
    """
    pca = _fit_pca(panel, n_factors)
    components = pca.components_
    if align_signs:
        components = components * _signs(pca)[:, None]

    return pd.DataFrame(
        components,
        index=[f"VF{i + 1}" for i in range(n_factors)],
        columns=panel.columns,
    )


def scaling_betas(X_train: pd.DataFrame, residuals: np.ndarray) -> np.ndarray:
    """Regress the residuals on each stock's LV and keep the slope β_n (step 2 of §2.2.2).

    Equivalent to a univariate OLS with intercept for every column::

        residual_t = α_n + β_n · X_{n,t} + ε_t

    The closed form β = cov(x, r) / var(x) is computed for all stocks at once and is
    numerically equivalent to column-by-column OLS (see tests/test_factors.py).
    """
    xc = X_train.to_numpy() - X_train.to_numpy().mean(axis=0)
    rc = residuals.ravel() - residuals.mean()
    return (xc * rc[:, None]).sum(axis=0) / (xc ** 2).sum(axis=0)


def har_residuals(
    X_train: pd.DataFrame, Y_train: pd.Series, har_cols: list[str] | None = None
) -> np.ndarray:
    """Residuals from regressing the target on the three HAR terms (step 1 of §2.2.2)."""
    har_cols = har_cols or HAR_COLS
    A = np.column_stack([np.ones(len(X_train)), X_train[har_cols].to_numpy()])
    y = Y_train.to_numpy()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def extract_spca_residual(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_apply: pd.DataFrame,
    har_cols: list[str] | None = None,
    n_factors: int = N_FACTORS,
    prefix: str = "sVF",
    align_signs: bool = True,
    use_correlation: bool = False,
    beta_on_fitted: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual scaled-PCA volatility factors (sVF, §2.2.2).

    Three steps: HAR residuals -> per-stock β -> PCA on the β-scaled panel.

    Parameters
    ----------
    X_train, X_apply
        Full LV panels **including** the three HAR columns. Stock columns are selected
        internally; the HAR columns are only used for the residuals in step 1.
    Y_train
        Target of the training window (LV scale).
    use_correlation
        If True, loadings are eigenvectors of the correlation matrix (see
        :func:`_project`), as in the robustness check of §4.3.3 (Table 10). β is still
        estimated on the original panel.

    Returns
    -------
    (F_train, F_apply)

    Notes
    -----
    β is estimated on the training window only and then applied to ``X_apply``. This
    involves **no look-ahead**: β uses information from the training period only, and
    X_{n,t} is observable at time t.

    ``beta_on_fitted=True`` is the β specification of §6.3 (Table 16): β is obtained by
    regressing the HAR fitted values on each column, with the sign reversed. Only
    Table 16 uses it; all other sections use the default branch.
    """
    har_cols = har_cols or HAR_COLS
    stocks = [c for c in X_train.columns if c not in har_cols]

    resid = har_residuals(X_train, Y_train, har_cols)
    if beta_on_fitted:
        # §6.3 specification: regress on the HAR fitted values, sign reversed (see Notes).
        beta = scaling_betas(X_train[stocks], -(Y_train.to_numpy() - resid))
    else:
        beta = scaling_betas(X_train[stocks], resid)

    scaled_train = X_train[stocks] * beta
    scaled_apply = X_apply[stocks] * beta

    f_train, f_apply = _project(
        scaled_train, scaled_apply, n_factors, use_correlation, align_signs
    )
    return (
        _as_factor_frame(f_train, X_train.index, prefix),
        _as_factor_frame(f_apply, X_apply.index, prefix),
    )


def pair_align(factors: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Flip each column of ``factors`` so that it correlates positively with ``reference``.

    sPCA is the supervised version of PCA: the k-th sVF and the k-th VF capture the same
    common factor and should point in the same direction. The correlations between VF_k
    and sVF_k in Table 1 are all positive (0.997 / 0.983 / 0.974 / 0.916).
    """
    signs = np.sign(
        [
            np.corrcoef(factors.iloc[:, k], reference.iloc[:, k])[0, 1]
            for k in range(factors.shape[1])
        ]
    )
    return factors * signs


def extract_full_sample_factors(
    X: pd.DataFrame, Y: pd.Series, n_factors: int = N_FACTORS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-off factor extraction for the in-sample analyses (§3.2.1, §4.1, §5).

    Extracts VF and sVF on the **full sample** and applies the paper's sign convention:
    each component has a positive loading on the first stock, then sVF_k is flipped to
    correlate positively with VF_k.

    The out-of-sample part (§4.2 onward) uses the rolling extraction of
    :func:`volfactor.forecast.rolling_factor_forecast`, where signs are irrelevant —
    the regression coefficient absorbs them.

    Returns
    -------
    (VF, sVF)
        Two DataFrames with columns VF1…VF4 and sVF1…sVF4.
    """
    stocks = [c for c in X.columns if c not in HAR_COLS]

    vf, _ = extract_pca(X[stocks], X[stocks], n_factors)
    svf, _ = extract_spca_residual(X, Y, X, n_factors=n_factors)

    return vf, pair_align(svf, vf)


def eigenvalues(data: pd.DataFrame | np.ndarray, k: int = 20) -> np.ndarray:
    """The k largest eigenvalues of the panel covariance matrix, in descending order.

    Equivalent to ``PCA(n_components=k).fit(data).explained_variance_``, but computed
    by a symmetric eigendecomposition of the N×N covariance matrix, which is much faster
    than an SVD of the T×N panel (``eigvalsh`` on 332×332 vs SVD on 1260×332).
    """
    arr = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    return np.linalg.eigvalsh(np.cov(arr, rowvar=False))[::-1][:k]


def estimate_n_factors(
    data: pd.DataFrame | np.ndarray, gamma: float = 0.08, max_components: int = 20
) -> int:
    """Number of factors by the perturbed eigenvalue ratio test of Pelger (2019) (§3.3).

    A perturbation ``g = sqrt(median(λ))`` is added to the first ``max_components``
    eigenvalues to stabilize the ratios, and adjacent ratios are checked in order::

        (λ_j + g) / (λ_{j+1} + g) ≤ 1 + γ

    The first time the condition holds, the remaining eigenvalues are treated as noise
    and the number of preceding components is returned.

    Parameters
    ----------
    gamma
        Threshold parameter. The main specification uses 0.08 (critical value 1.08);
        Appendix H uses the more conservative 0.30.

    Notes
    -----
    The median is taken over the first ``max_components`` eigenvalues, not all N.
    """
    lam = eigenvalues(data, max_components)
    g = np.sqrt(np.median(lam))
    ratio = (lam[:-1] + g) / (lam[1:] + g)

    for idx, r in enumerate(ratio, start=1):
        if r <= 1 + gamma:
            return idx - 1
    return max_components
