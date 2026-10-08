# -*- coding: utf-8 -*-
"""Out-of-sample evaluation: OOS R², Clark–West test, five loss functions, and the
model confidence set (MCS).

Scale convention
----------------
All ``actual`` / ``pred`` arguments in this module are on the **LV (log variance)
scale**, i.e. ``LV = log(RV²)``, consistent with :mod:`volfactor.io`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

#: Random seed used throughout the repository. Fixed so that the MCS block bootstrap
#: is reproducible.
DEFAULT_SEED = 42


# --------------------------------------------------------------------------
# Loss functions
# --------------------------------------------------------------------------

def loss_mse(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Squared error, per period."""
    return (pred - actual) ** 2


def loss_mae(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Absolute error, per period."""
    return np.abs(pred - actual)


def loss_qlike(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Standard Patton (2011) QLIKE on the variance scale, written in log differences::

        QLIKE_t = exp(LV_t − LV̂_t) − (LV_t − LV̂_t) − 1

    Since LV = log(RV²), ``exp(LV − LV̂)`` equals the variance ratio ``RV²/RV̂²``, so
    this is identical to the textbook form ``σ²/σ̂² − log(σ²/σ̂²) − 1``, but numerically
    stable in log differences (defined for any real input, always ≥ 0, and 0 for a
    perfect forecast).
    """
    d = actual - pred
    return np.exp(d) - d - 1.0


def loss_hmse(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Heteroskedasticity-adjusted MSE ``(1 − forecast/actual)²``, ratio on the log-variance scale."""
    return (1.0 - pred / actual) ** 2


def loss_hmae(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Heteroskedasticity-adjusted MAE ``|1 − forecast/actual|``; see :func:`loss_hmse`."""
    return np.abs(1.0 - pred / actual)


#: The five loss functions used in Tables 7/8
LOSS_FUNCTIONS = {
    "MSE": loss_mse,
    "MAE": loss_mae,
    "QLIKE": loss_qlike,
    "HMSE": loss_hmse,
    "HMAE": loss_hmae,
}


def loss_matrix(
    actual: pd.Series, predictions: pd.DataFrame, loss: str
) -> pd.DataFrame:
    """Turn the forecasts of several models into a per-period loss matrix for the MCS.

    Parameters
    ----------
    actual
        Realized values (LV scale), indexed by date.
    predictions
        One column of forecasts per model, indexed by date.
    loss
        A key of ``LOSS_FUNCTIONS``.

    Returns
    -------
    DataFrame
        Same shape as ``predictions`` with per-period losses; only dates common to
        both inputs are kept.
    """
    if loss not in LOSS_FUNCTIONS:
        raise KeyError(f"unknown loss {loss!r}; choose from {sorted(LOSS_FUNCTIONS)}")

    fn = LOSS_FUNCTIONS[loss]
    joined = predictions.join(actual.rename("__actual__"), how="inner").dropna()
    a = joined["__actual__"].to_numpy()

    return pd.DataFrame(
        {col: fn(a, joined[col].to_numpy()) for col in predictions.columns},
        index=joined.index,
    )


# --------------------------------------------------------------------------
# Forecast accuracy tests
# --------------------------------------------------------------------------

def oos_r2(actual: np.ndarray, pred: np.ndarray, benchmark: np.ndarray) -> float:
    """Campbell–Thompson (2008) out-of-sample R²::

        R²_OS = 1 − MSPE_model / MSPE_benchmark

    Positive values mean the model beats the benchmark. Returned as a fraction; the
    paper's tables report it multiplied by 100.
    """
    actual, pred, benchmark = map(np.asarray, (actual, pred, benchmark))
    mask = ~(np.isnan(pred) | np.isnan(benchmark) | np.isnan(actual))
    mspe_m = np.mean((actual[mask] - pred[mask]) ** 2)
    mspe_b = np.mean((actual[mask] - benchmark[mask]) ** 2)
    return float(1.0 - mspe_m / mspe_b)


def cw_adjustment(
    actual: np.ndarray, pred: np.ndarray, benchmark: np.ndarray
) -> np.ndarray:
    """Per-period Clark–West adjusted loss differential::

        f_t = (y_t − ŷ^b_t)² − [(y_t − ŷ^m_t)² − (ŷ^m_t − ŷ^b_t)²]

    The last term corrects the small-sample bias with nested models: the larger model
    estimates extra parameters, so its MSPE is biased upward even under the null.
    Observations missing in any of the three inputs are dropped.
    """
    actual, pred, benchmark = map(np.asarray, (actual, pred, benchmark))
    mask = ~(np.isnan(pred) | np.isnan(benchmark) | np.isnan(actual))
    a, p, b = actual[mask], pred[mask], benchmark[mask]
    return (a - b) ** 2 - ((a - p) ** 2 - (p - b) ** 2)


def mspe_adjusted(
    actual: np.ndarray, pred: np.ndarray, benchmark: np.ndarray
) -> float:
    """The "MSPE-adj" column of the paper's tables — the mean of :func:`cw_adjustment`."""
    return float(cw_adjustment(actual, pred, benchmark).mean())


def clark_west(
    actual: np.ndarray,
    pred: np.ndarray,
    benchmark: np.ndarray,
    maxlags: int | None = None,
    two_sided: bool = False,
    dist: str = "t",
) -> tuple[float, float]:
    """Clark–West (2007) MSPE-adjusted test.

    Null: the benchmark and the model have equal forecast accuracy; alternative: the
    model is more accurate. The statistic is the mean of :func:`cw_adjustment`
    divided by its standard error.

    Parameters
    ----------
    maxlags
        ``None`` (default) uses ``std(f, ddof=1) / sqrt(n)`` as the standard error, as
        in the paper. An integer switches to a HAC standard error.
    two_sided
        True for a two-sided p-value, False (default) for one-sided. The convention
        used for each table is set at the call site.
    dist
        Reference distribution of the p-value: ``"t"`` (default, n−1 degrees of
        freedom, as in the paper) or ``"normal"``.

    Returns
    -------
    (statistic, p-value)
    """
    f = cw_adjustment(actual, pred, benchmark)

    if maxlags is None:
        cw = float(f.mean() / (np.std(f, ddof=1) / np.sqrt(len(f))))
    else:
        fit = sm.OLS(f, np.ones((len(f), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        cw = float(fit.tvalues[0])

    if dist == "t":
        sf = stats.t.sf(abs(cw), df=len(f) - 1), stats.t.sf(cw, df=len(f) - 1)
    elif dist == "normal":
        sf = stats.norm.sf(abs(cw)), stats.norm.sf(cw)
    else:
        raise ValueError(f"dist must be 't' or 'normal', got {dist!r}")

    if two_sided:
        return cw, float(2.0 * sf[0])
    return cw, float(sf[1])


def significance_stars(p: float) -> str:
    """p-value -> significance stars (1% / 5% / 10%, as in the paper's tables)."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# --------------------------------------------------------------------------
# Model confidence set
# --------------------------------------------------------------------------

def run_mcs(
    losses: pd.DataFrame,
    size: float = 0.05,
    reps: int = 1000,
    block_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> pd.Series:
    """Model confidence set (Hansen–Lunde–Nason, 2011); returns the MCS p-value of each model.

    ``arch.bootstrap.MCS`` computes p-values by block bootstrap; the random seed
    :data:`DEFAULT_SEED` is fixed so that results are reproducible.

    Parameters
    ----------
    losses
        Per-period loss matrix, one column per model (build with :func:`loss_matrix`).
    size
        Significance level of the MCS. It only affects the ``included`` / ``excluded``
        split, not the p-values returned.

    Returns
    -------
    Series
        Indexed by model name; values are MCS p-values.
    """
    from arch.bootstrap import MCS  # lazy import: only the MCS script needs arch

    mcs = MCS(losses, size=size, reps=reps, block_size=block_size, seed=seed)
    mcs.compute()
    return mcs.pvalues.iloc[:, 0] if mcs.pvalues.ndim > 1 else mcs.pvalues


def evaluate_models(
    actual: pd.Series,
    predictions: pd.DataFrame,
    benchmark_col: str = "HAR",
    two_sided: bool = True,
) -> pd.DataFrame:
    """OOS R² and Clark–West test for a set of models.

    ``two_sided`` defaults to True, the two-sided convention of Tables 4 / 9.

    Returns
    -------
    DataFrame
        Indexed by model name, with columns ``OOS_R2`` (%), ``MSPE_adj``, ``CW_stat``,
        ``CW_pvalue`` and ``stars``. The benchmark's own row has OOS R² 0 and NaN tests.
    """
    joined = predictions.join(actual.rename("__actual__"), how="inner").dropna()
    a = joined["__actual__"].to_numpy()
    bench = joined[benchmark_col].to_numpy()

    rows = {}
    for col in predictions.columns:
        p = joined[col].to_numpy()
        if col == benchmark_col:
            rows[col] = {"OOS_R2": 0.0, "MSPE_adj": np.nan, "CW_stat": np.nan,
                         "CW_pvalue": np.nan, "stars": ""}
            continue
        r2 = oos_r2(a, p, bench) * 100.0
        cw, pv = clark_west(a, p, bench, two_sided=two_sided)
        rows[col] = {"OOS_R2": r2, "MSPE_adj": mspe_adjusted(a, p, bench), "CW_stat": cw,
                     "CW_pvalue": pv, "stars": significance_stars(pv)}

    return pd.DataFrame(rows).T
