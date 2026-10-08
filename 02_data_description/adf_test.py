#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tables A.17 / A.18 — ADF stationarity tests for the 332 constituents (Appendix A).

An ADF test is run on the log realized variance of each stock, with the lag order selected by AIC.
The paper splits the results into two tables (part 1 / part 2); this script writes one complete table
and additionally a blocked version laid out like the paper (3 companies per row).

A series is classified as stationary when **p ≤ 0.10**.

Usage
-----
    python 02_data_description/adf_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from volfactor.io import load_panel, stock_columns
from volfactor.paths import TABLES_DIR, ensure_output_dirs

#: Significance level for classifying a series as "Stationary"
ALPHA = 0.10

#: Number of companies per row in the paper's tables
COLUMNS_PER_ROW = 3


def run_adf(series: pd.Series) -> dict:
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {
        "ADF": stat,
        "p_value": pvalue,
        "stationary": pvalue <= ALPHA,
        "used_lag": used_lag,
        "nobs": nobs,
    }


def main() -> None:
    ensure_output_dirs()

    # ADF is run on the stocks only, excluding the HAR columns and the target
    X, _ = load_panel(horizon=1)
    stocks = stock_columns(X)

    rows = {ticker: run_adf(X[ticker]) for ticker in stocks}

    # from_dict(orient="index") keeps the original dtype of each column;
    # DataFrame(rows).T would turn the whole table into object and break the boolean aggregation.
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "company"

    path = TABLES_DIR / "tableA17_A18_adf_test.csv"
    result.to_csv(path)

    n_stationary = int(result["stationary"].sum())
    print(f"ADF test on {len(result)} SPX constituents (log realized variance)")
    print(f"  stationary at {ALPHA:.0%} level : {n_stationary}/{len(result)}")
    print(f"  ADF statistic range           : "
          f"[{result['ADF'].min():.3f}, {result['ADF'].max():.3f}]")
    print(f"  max p-value                   : {result['p_value'].max():.4f}")

    if n_stationary < len(result):
        failed = result[~result["stationary"]]
        print(f"  non-stationary ({len(failed)}): {list(failed.index)}")

    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
