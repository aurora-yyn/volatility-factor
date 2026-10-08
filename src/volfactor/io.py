# -*- coding: utf-8 -*-
"""Data loading and log transformation.

**Unit convention (single source of truth for the whole repository)**

`data/raw/lasso_data.csv` and `targets.csv` store realized volatility RV on the
**standard-deviation scale**. LV in the paper denotes the log realized **variance**::

    LV = log(RV ** 2) = 2 * log(RV)

All modelling, forecasting and loss functions operate on the LV scale. This matters
for QLIKE in particular: `exp(LV - LV_hat)` equals the variance ratio `RV² / RV̂²`,
consistent with the variance-scale definition of Patton (2011).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .paths import PREDICTORS_DIR, RAW_DIR

#: The three fixed HAR regressors (daily / weekly / monthly realized volatility)
HAR_COLS = ["RV_1Day", "RV_5Day", "RV_22Day"]

#: horizon -> target column in targets.csv
TARGET_COL = {1: "RV_target", 5: "RV_target_5day", 22: "RV_target_22day"}

Horizon = Literal[1, 5, 22]


def to_log_variance(rv: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Standard-deviation-scale RV -> log variance LV = log(RV²)."""
    return np.log(rv ** 2)


def load_panel(horizon: Horizon = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Load the predictor matrix X and target Y, both on the LV scale.

    Parameters
    ----------
    horizon
        Forecast horizon: 1, 5 or 22 days.

    Returns
    -------
    X : DataFrame
        Indexed by Date; columns are ``HAR_COLS`` + the 332 stocks; values are LV.
    Y : Series
        Indexed by Date; the forecast target on the LV scale.

    Notes
    -----
    Multi-day targets are missing at the end of the sample (they look h days ahead),
    so X is trimmed to the valid range of Y. The returned X and Y therefore have the
    same number of rows: h=1 -> 3251, h=5 -> 3247, h=22 -> 3230.
    """
    panel = pd.read_csv(RAW_DIR / "lasso_data.csv", parse_dates=["Date"]).set_index("Date")
    targets = pd.read_csv(RAW_DIR / "targets.csv", parse_dates=["Date"]).set_index("Date")

    col = TARGET_COL[horizon]
    if col not in targets.columns:
        raise KeyError(f"targets.csv has no column {col!r}; available: {list(targets.columns)}")

    y = targets[col].dropna()
    x = panel.loc[y.index]

    return to_log_variance(x), to_log_variance(y)


def stock_columns(X: pd.DataFrame) -> list[str]:
    """Stock columns of X other than the three HAR columns (the N = 332 constituents)."""
    return [c for c in X.columns if c not in HAR_COLS]


#: The four volatility decompositions used in the §6 extensions.
#:
#: - ``shift``: constant added before taking logs. About 93% of the cells in the jump
#:   panel are zero (no jump on most trading days) and the continuous panel also has
#:   zeros, so the paper uses ``log(1 + x)`` for these two; the good / bad panels have
#:   no zeros (minimum about 1e-4) and are logged directly.
#: - ``data_end``: last year of the **panel**; affects the rolling windows and the
#:   number of forecasts.
#: - ``eval_end``: last year of the **evaluation window**; only affects which period
#:   enters the R² computation.
#:
#: The §6.1 panel ends in 2014; §6.2 generates forecasts on the full sample and
#: evaluates them up to 2014.
COMPONENTS = {
    "continuous": {
        "shift": 1.0, "data_end": 2014, "eval_end": None,
        "desc": "Continuous component (BPV): smooth everyday variation (§6.1; factors CVF / CsVF)",
    },
    "jump": {
        "shift": 1.0, "data_end": 2014, "eval_end": None,
        "desc": "Jump component (RV − BPV): sudden events (§6.1; factors JVF / JsVF)",
    },
    "good": {
        "shift": 0.0, "data_end": None, "eval_end": 2014,
        "desc": "Upside (good) volatility from positive returns (§6.2; factors GVF / GsVF)",
    },
    "bad": {
        "shift": 0.0, "data_end": None, "eval_end": 2014,
        "desc": "Downside (bad) volatility from negative returns (§6.2; factors BVF / BsVF)",
    },
}


def load_component_panel(
    component: str, horizon: Horizon = 1, end_year: int | None = -1
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the panel and target for one volatility component, on the LV scale (§6).

    Differs from :func:`load_panel` only in where the stock columns come from: the HAR
    columns and the target are still taken from the market RV, because the object being
    forecast is always **market** volatility — only the cross-sectional information used
    to extract the factors changes.

    Scaling
    -------
    - The component panels are already on the **variance** scale, so they are
      transformed with ``log(shift + x)``, not ``log(x²)``; see :data:`COMPONENTS` for
      ``shift``.
    - ``market_rv.csv`` is on the standard-deviation scale and is squared before taking
      logs, so that both sides are on the same scale.
    - The target is ``RV_{h}Day.shift(-h)``.

    Parameters
    ----------
    component
        A key of :data:`COMPONENTS`.
    end_year
        Last year of the panel. The default ``-1`` uses the component's ``data_end``
        (see :data:`COMPONENTS`); ``None`` uses the full data range; a year overrides it.

    Returns
    -------
    (X, Y)
        X has columns ``HAR_COLS`` + the component's stock columns; Y is the market LV target.
    """
    if component not in COMPONENTS:
        raise KeyError(
            f"unknown component {component!r}; choose from {sorted(COMPONENTS)}"
        )

    spec = COMPONENTS[component]
    if end_year == -1:
        end_year = spec["data_end"]

    path = RAW_DIR / "decomposition" / f"{component}_rv.csv.gz"
    panel = (
        pd.read_csv(path, parse_dates=["Datetime"])
        .rename(columns={"Datetime": "Date"})
        .set_index("Date")
    )
    panel = panel + spec["shift"]

    market = load_market_rv().drop(columns=["RV_final"], errors="ignore") ** 2

    # Truncate before shift(-h): targets for the last h days of the window are missing
    # and dropped, rather than borrowing data from outside the window.
    if end_year is not None:
        market = market[market.index.year <= end_year]

    target_col = {1: "RV_1Day", 5: "RV_5Day", 22: "RV_22Day"}[horizon]

    joined = panel.join(market[HAR_COLS], how="inner").dropna()
    target = market[target_col].reindex(joined.index).shift(-horizon).dropna()
    joined = joined.loc[target.index]

    stocks = [c for c in panel.columns if c in joined.columns]
    return np.log(joined[HAR_COLS + stocks]), np.log(target)


#: The five realized partial variance components of §6.3, split by quantiles of the
#: high-frequency returns (Bollerslev et al., 2022). Order matches Panels A–E of Table 16.
PARTIAL_COMPONENTS = [
    "RV_ExtremeNeg",    # < 10th percentile — extreme negative returns
    "RV_ModerateNeg",   # 10th – 40th
    "RV_Neutral",       # 40th – 60th
    "RV_ModeratePos",   # 60th – 90th
    "RV_ExtremePos",    # > 90th — extreme positive returns
]


def load_partial_variance_panel(
    component: str, horizon: Horizon = 1
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the panel and target for one partial variance component (§6.3, Table 16).

    Setup of this section:

    1. Market RV enters as ``log(RV)`` (standard-deviation scale). At evaluation the
       forecasts are multiplied by 2 to be compared with the ``log(RV²)`` benchmark.
    2. The PCA input is the other 4 partial variance components plus ``RV_final``
       (5 columns), from which 4 principal components are extracted.
    3. The full 2004–2016 sample is used.

    Returns
    -------
    (X, Y, pca_cols)
        X has columns ``HAR_COLS`` + the current component + ``pca_cols``, all logged;
        Y is ``log(RV_{h}Day.shift(-h))`` on the standard-deviation scale;
        ``pca_cols`` are the columns fed into the PCA for this component.
    """
    if component not in PARTIAL_COMPONENTS:
        raise KeyError(
            f"unknown partial variance component {component!r}; "
            f"choose from {PARTIAL_COMPONENTS}"
        )

    panel = (
        pd.read_csv(RAW_DIR / "decomposition" / "partial_rv.csv.gz",
                    parse_dates=["Datetime"])
        .rename(columns={"Datetime": "Date"})
        .set_index("Date")
    )

    market = load_market_rv()          # not squared, and RV_final is kept
    target_col = {1: "RV_1Day", 5: "RV_5Day", 22: "RV_22Day"}[horizon]

    joined = panel.join(market, how="inner").dropna()
    target = joined[target_col].shift(-horizon).dropna()
    joined = joined.loc[target.index]

    # PCA input: the other 4 partial variance components + RV_final; the current
    # component goes into the fixed regressors
    pca_cols = [c for c in panel.columns if c != component] + ["RV_final"]
    X = np.log(joined[HAR_COLS + [component] + pca_cols])
    return X, np.log(target), pca_cols


def load_market_rv() -> pd.DataFrame:
    """Raw S&P 500 realized volatility table (standard-deviation scale, not logged)."""
    return pd.read_csv(RAW_DIR / "market_rv.csv", parse_dates=["Date"]).set_index("Date")


def load_actual_lv(horizon: Horizon = 1) -> pd.Series:
    """Realized values for out-of-sample evaluation, on the LV scale.

    Takes column ``RV_{h}Day`` of ``market_rv.csv``, aligns it to the forecast period
    with ``shift(-h)``, and converts it to LV.

    - h=1  : ``RV_1Day.shift(-1)``   — next-day realized volatility
    - h=5  : ``RV_5Day.shift(-5)``   — next 5 days
    - h=22 : ``RV_22Day.shift(-22)`` — next 22 days

    This is the same quantity as the training target Y returned by :func:`load_panel`.
    """
    col = {1: "RV_1Day", 5: "RV_5Day", 22: "RV_22Day"}[horizon]
    actual = load_market_rv()[col].shift(-horizon).dropna()
    actual.index.name = "date"
    return to_log_variance(actual).rename("actual")


def load_industry_map() -> pd.DataFrame:
    """14-industry SIC classification with columns ``ticker`` / ``industry``."""
    return pd.read_csv(RAW_DIR / "industry_mapping.csv")


def load_predictors() -> pd.DataFrame:
    """VIX / EPU / SKEW merged into one date-indexed table.

    The three series use different calendars (EPU is daily on calendar days, VIX/SKEW
    on trading days). An outer join keeps all dates and leaves NaN where a series is
    missing; the caller decides how to align them.
    """
    frames = []
    for name in ("vix", "epu", "skew"):
        df = pd.read_csv(PREDICTORS_DIR / f"{name}.csv", parse_dates=["date"])
        frames.append(df.set_index("date"))
    return pd.concat(frames, axis=1).sort_index()
