#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Appendix Table G.27 — predictive power of VIX / EPU / SKEW on their own, relative to HAR.

This prepares Table 13 (benchmark 2): how much does each common predictor improve HAR **on its own**, before
asking whether the volatility factors add information on top of it. Only VIX significantly improves HAR
on its own; EPU and SKEW do not.

    LV_{t,t+h} = HAR_t + γ · X_t + ε        vs.        LV_{t,t+h} = HAR_t + ε

Setup
-----
1. Full sample (to 2016).
2. The "MSPE-adj" column reports the Clark–West statistic.
3. p-values are two-sided.

Usage
-----
    python 06_information_content/standalone_predictors.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from volfactor.evaluation import clark_west, oos_r2, significance_stars
from volfactor.forecast import rolling_ols_forecast
from volfactor.io import HAR_COLS, load_panel, load_predictors
from volfactor.paths import TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)

#: Row order and labels of the table
PREDICTORS = {"vix": "VIX", "epu": "EPU", "skew": "SKEW"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=list(HORIZONS),
                    choices=list(HORIZONS), help="forecast horizons in days, default 1 5 22")
    ap.add_argument("-w", "--window", type=int, default=5, help="estimation window in years, default 5")
    args = ap.parse_args()

    ensure_output_dirs()
    rows = []

    for h in args.horizons:
        X, Y = load_panel(horizon=h)
        predictors = load_predictors().reindex(X.index).ffill()

        benchmark = rolling_ols_forecast(X[HAR_COLS], Y, args.window, name="HAR")
        actual = Y.reindex(benchmark.index)

        print(f"\n=== horizon={h}d " + "=" * 42)
        for var, label in PREDICTORS.items():
            pred = rolling_ols_forecast(
                pd.concat([X[HAR_COLS], predictors[[var]]], axis=1),
                Y, args.window, name=var,
            )
            r2 = oos_r2(actual, pred, benchmark) * 100
            cw, pval = clark_west(actual, pred, benchmark, two_sided=True)

            rows.append({
                "horizon": h,
                "model": f"HAR-{label}",
                "oos_r2": r2,
                "mspe_adj": cw,       # the MSPE-adj column of this table is the CW statistic
                "cw_pvalue": pval,
                "stars": significance_stars(pval),
            })
            print(f"    HAR-{label:<6s} R2={r2:7.3f}{significance_stars(pval):<3s}"
                  f"  MSPE-adj={cw:7.3f}  p={pval:.3f}")

    path = TABLES_DIR / "appendix_standalone_predictors.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
