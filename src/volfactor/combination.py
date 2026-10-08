# -*- coding: utf-8 -*-
"""Combination forecasts (§2.3).

All five combination methods share the same building block: an augmented HAR model
is fitted separately for **every stock** n::

    LV_{t,t+h} = c + β₁·RV_1Day + β₂·RV_5Day + β₃·RV_22Day + β_n·X_{n,t} + ε

which gives N = 332 forecasts that are then aggregated into one forecast:

======  ==========================================================
MC      mean combination
MDC     median combination
TMC     trimmed mean combination (drop one maximum and one minimum)
DMSPE   weights inversely proportional to the discounted MSPE, ξ = 1 or 0.9
======  ==========================================================

Implementation
--------------
Each rolling window requires 332 single-stock regressions. They are vectorized into
one least-squares problem with multiple right-hand sides via the Frisch–Waugh–Lovell
theorem: the target and every stock's LV are first residualized on the HAR design,

    β_n = (x̃_n' ỹ) / (x̃_n' x̃_n)

and the individual forecasts are recovered as
``pred_n = HAR forecast + β_n · (x_{n,test} − ĥ_{n,test})``. This is mathematically
identical to running the regressions one by one (see tests/test_combination.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import DAYS_PER_YEAR, _window_slices
from .io import HAR_COLS, stock_columns


def _design(X: pd.DataFrame) -> np.ndarray:
    """HAR design matrix [1, RV_1Day, RV_5Day, RV_22Day]."""
    return np.column_stack([np.ones(len(X)), X[HAR_COLS].to_numpy()])


def _individual_predictions(
    H_tr: np.ndarray, y_tr: np.ndarray, Xs_tr: np.ndarray,
    H_ap: np.ndarray, Xs_ap: np.ndarray,
) -> np.ndarray:
    """Forecasts of all N augmented HAR models over the application period, via FWL.

    Returns
    -------
    ndarray, shape (len(H_ap), N)
    """
    coef_y, *_ = np.linalg.lstsq(H_tr, y_tr, rcond=None)      # (4,)
    coef_X, *_ = np.linalg.lstsq(H_tr, Xs_tr, rcond=None)     # (4, N)

    res_y = y_tr - H_tr @ coef_y                              # (T,)
    res_X = Xs_tr - H_tr @ coef_X                             # (T, N)

    beta = (res_X * res_y[:, None]).sum(axis=0) / (res_X ** 2).sum(axis=0)

    har_pred = H_ap @ coef_y                                  # (m,)
    x_hat = H_ap @ coef_X                                     # (m, N)
    return har_pred[:, None] + beta * (Xs_ap - x_hat)


def _trimmed_mean(v: np.ndarray) -> float:
    """Mean after dropping one maximum and one minimum.

    Exactly one extreme value is removed on each side; this is not a percentage trim.
    """
    if len(v) > 2:
        v = np.delete(v, [v.argmax(), v.argmin()])
    return float(v.mean())


def rolling_combination_forecast(
    X: pd.DataFrame, Y: pd.Series, window_years: int = 5
) -> pd.DataFrame:
    """Rolling out-of-sample MC / MDC / TMC combination forecasts."""
    window = DAYS_PER_YEAR * window_years
    stocks = stock_columns(X)
    H_all = _design(X)
    Xs_all = X[stocks].to_numpy()
    y_all = Y.to_numpy()

    dates, rows = [], []
    for i, j in _window_slices(len(X), window):
        preds = _individual_predictions(
            H_all[i:j], y_all[i:j], Xs_all[i:j],
            H_all[j : j + 1], Xs_all[j : j + 1],
        )[0]
        dates.append(X.index[j])
        rows.append((preds.mean(), np.median(preds), _trimmed_mean(preds)))

    return pd.DataFrame(
        rows, index=pd.DatetimeIndex(dates, name="date"), columns=["MC", "MDC", "TMC"]
    )


def _dmspe_weights(
    val_preds: np.ndarray, val_actual: np.ndarray, xi: float
) -> np.ndarray:
    """Discounted mean squared prediction error weights.

    Earlier observations are discounted more (weights ξ^(T-1) … ξ⁰), and the final
    weights are inversely proportional to the discounted sum of squared errors::

        w_n ∝ 1 / Σ_t ξ^{T-t} (ŷ_{n,t} − y_t)²
    """
    discount = xi ** np.arange(len(val_actual))[::-1]
    errors = (val_preds - val_actual[:, None]) ** 2
    w = 1.0 / (errors * discount[:, None]).sum(axis=0)
    return w / w.sum()


def rolling_dmspe_forecast(
    X: pd.DataFrame,
    Y: pd.Series,
    xi: float = 0.9,
    window_years: int = 5,
    validate_years: int = 2,
) -> pd.Series:
    """DMSPE combination forecast (discounted weighting à la Rapach–Strauss–Zhou).

    Unlike MC/MDC/TMC, DMSPE needs a validation period to assess each forecaster:
    the estimation window is split into "training + validation", where **the
    validation period is fixed at 2 years and the training period is the remainder**,
    i.e. w=4 -> 2+2, w=5 -> 3+2, w=6 -> 4+2. Weights are estimated on the validation
    period; the stock models are then refitted on the full window and their one-step
    forecasts are combined with these weights.

    Parameters
    ----------
    xi
        Discount factor. The paper uses 1.0 (no discounting) and 0.9 (more weight on
        recent performance).
    window_years
        Total estimation window; must match the other models.

    Notes
    -----
    The total window matches MC/MDC/TMC and the other models so that all models share
    the same forecast period. The validation part is ``validate_years`` years and the
    training part is the remainder.
    """
    train_years = window_years - validate_years
    if train_years < 1:
        raise ValueError(
            f"window_years ({window_years}) must exceed validate_years ({validate_years}) by at least 1"
        )

    train = DAYS_PER_YEAR * train_years
    validate = DAYS_PER_YEAR * validate_years
    total = train + validate

    stocks = stock_columns(X)
    H_all = _design(X)
    Xs_all = X[stocks].to_numpy()
    y_all = Y.to_numpy()

    dates, preds = [], []
    for i, j in _window_slices(len(X), total):
        k = i + train  # training / validation split

        val_preds = _individual_predictions(
            H_all[i:k], y_all[i:k], Xs_all[i:k],
            H_all[k:j], Xs_all[k:j],
        )
        weights = _dmspe_weights(val_preds, y_all[k:j], xi)

        test_preds = _individual_predictions(
            H_all[i:j], y_all[i:j], Xs_all[i:j],
            H_all[j : j + 1], Xs_all[j : j + 1],
        )[0]

        dates.append(X.index[j])
        preds.append(float(test_preds @ weights))

    name = f"DMSPE_{xi:g}"
    return pd.Series(preds, index=pd.DatetimeIndex(dates, name="date"), name=name)


def full_period_dmspe_forecast(
    X: pd.DataFrame,
    Y: pd.Series,
    xis: tuple[float, ...] = (0.9, 1.0),
    window_years: int = 5,
) -> pd.DataFrame:
    """DMSPE combination forecast with weights set once over the whole out-of-sample period (Table 10).

    The rolling forecasts of the N augmented HAR models are first obtained for the
    whole out-of-sample period; a single set of weights is then computed from the
    discounted forecast errors over that period (:func:`_dmspe_weights`) and applied
    to the forecasts of every period.

    Returns
    -------
    DataFrame
        Indexed by forecast date, columns ``DMSPE_{xi}``.
    """
    window = DAYS_PER_YEAR * window_years
    stocks = stock_columns(X)
    H_all = _design(X)
    Xs_all = X[stocks].to_numpy()
    y_all = Y.to_numpy()

    dates, rows = [], []
    for i, j in _window_slices(len(X), window):
        rows.append(_individual_predictions(
            H_all[i:j], y_all[i:j], Xs_all[i:j],
            H_all[j : j + 1], Xs_all[j : j + 1],
        )[0])
        dates.append(X.index[j])

    preds = np.vstack(rows)                       # (M, N)
    actual = Y.loc[dates].to_numpy()
    out = {f"DMSPE_{xi:g}": preds @ _dmspe_weights(preds, actual, xi) for xi in xis}
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates, name="date"))
