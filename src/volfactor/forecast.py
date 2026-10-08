# -*- coding: utf-8 -*-
"""Rolling-window out-of-sample forecasts.

All models share the same rolling scheme: estimate on the past `window_years` years
of data, forecast one step ahead, and move the window forward by one day. In §4.2 the
first 1260 days (5 years of 252 trading days) form the initial estimation window.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .factors import N_FACTORS, extract_pca, extract_spca_residual
from .io import HAR_COLS, stock_columns

#: Trading days per year
DAYS_PER_YEAR = 252

Method = Literal["pca", "spca"]


def _ols_fit_predict(
    design_train: np.ndarray, y_train: np.ndarray, design_test: np.ndarray
) -> np.ndarray:
    """OLS with intercept: fit and predict.

    Equivalent to ``sm.OLS(y, sm.add_constant(X)).fit().predict(X_test)``; ``lstsq``
    avoids the per-call overhead of statsmodels (about 2000 rolling fits).
    """
    A = np.column_stack([np.ones(len(design_train)), design_train])
    coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    A_test = np.column_stack([np.ones(len(design_test)), design_test])
    return A_test @ coef


def _window_slices(n: int, window: int):
    """Yield (training start, training end) such that a test row exists.

    There is no next day to forecast after the last window, so the number of forecasts
    is ``n - window``.
    """
    for i in range(n - window):
        yield i, i + window


def rolling_har_forecast(
    X: pd.DataFrame, Y: pd.Series, window_years: int = 5
) -> pd.Series:
    """Rolling out-of-sample forecasts of the HAR benchmark (§2.1)."""
    window = DAYS_PER_YEAR * window_years
    har = X[HAR_COLS].to_numpy()
    y = Y.to_numpy()

    dates, preds = [], []
    for i, j in _window_slices(len(X), window):
        pred = _ols_fit_predict(har[i:j], y[i:j], har[j : j + 1])
        dates.append(X.index[j])
        preds.append(pred[0])

    return pd.Series(preds, index=pd.DatetimeIndex(dates, name="date"), name="HAR")


def rolling_ols_forecast(
    features: pd.DataFrame,
    Y: pd.Series,
    window_years: int = 5,
    name: str = "pred",
) -> pd.Series:
    """Rolling one-step OLS forecasts for an arbitrary set of features.

    Uses the same windows and estimation as :func:`rolling_har_forecast`, but the
    caller supplies the full feature set, which makes it easy to build "HAR plus an
    extra variable" specifications: the industry factors of §5.1, the VIX / EPU / SKEW
    benchmarks of §5.2, etc.

    Parameters
    ----------
    features
        Design matrix (**without** intercept; added internally). Usually the three HAR
        columns plus extra variables.
    name
        Name of the returned series.

    Returns
    -------
    Series
        Indexed by forecast date.
    """
    window = DAYS_PER_YEAR * window_years
    design = features.to_numpy()
    y = Y.to_numpy()

    dates: list[pd.Timestamp] = []
    preds: list[float] = []

    for i, j in _window_slices(len(features), window):
        pred = _ols_fit_predict(design[i:j], y[i:j], design[j : j + 1])
        dates.append(features.index[j])
        preds.append(float(pred[0]))

    return pd.Series(preds, index=pd.DatetimeIndex(dates, name="date"), name=name)


def rolling_factor_forecast(
    X: pd.DataFrame,
    Y: pd.Series,
    method: Method,
    window_years: int = 5,
    n_factors: int = N_FACTORS,
    use_correlation: bool = False,
    extra: pd.DataFrame | None = None,
    base_cols: list[str] | None = None,
    factor_cols: list[str] | None = None,
    beta_on_fitted: bool = False,
) -> pd.DataFrame:
    """Rolling out-of-sample forecasts of HAR + one volatility factor.

    Factors are extracted once per rolling window, and **each factor is added to the
    HAR regression separately** (the paper adds them one at a time to assess the
    information content of each factor)::

        LV_{t,t+h} = HAR_t + β · F_{k,t-1} + ε

    Parameters
    ----------
    method
        ``"pca"`` -> VF1…VF4; ``"spca"`` -> sVF1…sVF4 (residual scaled PCA).
    use_correlation
        If True, factors are extracted from the correlation matrix, as in the
        robustness check of §4.3.3 (Table 10).
    extra
        Additional regressors (same index as ``X``) added for every factor. Used for
        benchmark 2 in §5.2 — VIX / EPU / SKEW are controlled for on top of HAR before
        asking whether the factors add information.
    base_cols
        Fixed regressors, also used as the baseline for the sPCA residuals. Default
        ``HAR_COLS``. §6.3 passes ``HAR_COLS + [current partial variance component]``:
        its baseline is "HAR + this component", and the factors explain the residual.
    factor_cols
        Columns fed into the PCA. Default: all columns of ``X`` other than
        ``base_cols`` (the 332 stocks in the main analysis). §6.3 uses 5 columns
        (the other 4 partial variance components + ``RV_final``).
    beta_on_fitted
        The sPCA β specification of §6.3, used only for Table 16.
        See :func:`~volfactor.factors.extract_spca_residual`.

    Returns
    -------
    DataFrame
        Indexed by forecast date; one column of forecasts per factor.

    Notes
    -----
    Factors are extracted once per window, followed by one regression per factor.
    """
    window = DAYS_PER_YEAR * window_years
    base_cols = base_cols or HAR_COLS
    stocks = factor_cols if factor_cols is not None else [
        c for c in X.columns if c not in base_cols
    ]
    prefix = "VF" if method == "pca" else "sVF"

    base = X[base_cols].to_numpy()
    if extra is not None:
        base = np.column_stack([base, extra.reindex(X.index).to_numpy()])
    y = Y.to_numpy()

    dates: list[pd.Timestamp] = []
    rows: list[np.ndarray] = []

    for i, j in _window_slices(len(X), window):
        X_train, X_test = X.iloc[i:j], X.iloc[j : j + 1]

        if method == "pca":
            F_train, F_test = extract_pca(
                X_train[stocks], X_test[stocks], n_factors, prefix,
                use_correlation=use_correlation,
            )
        else:
            F_train, F_test = extract_spca_residual(
                X_train[base_cols + stocks], Y.iloc[i:j],
                X_test[base_cols + stocks], base_cols, n_factors, prefix,
                use_correlation=use_correlation,
                beta_on_fitted=beta_on_fitted,
            )

        f_tr, f_te = F_train.to_numpy(), F_test.to_numpy()
        preds = [
            _ols_fit_predict(
                np.column_stack([base[i:j], f_tr[:, k]]),
                y[i:j],
                np.column_stack([base[j : j + 1], f_te[:, k]]),
            )[0]
            for k in range(n_factors)
        ]

        dates.append(X.index[j])
        rows.append(np.asarray(preds))

    return pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(dates, name="date"),
        columns=[f"{prefix}{k + 1}" for k in range(n_factors)],
    )
