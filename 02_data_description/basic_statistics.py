#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 1 — descriptive statistics of LV and the volatility factors (§3.2.1).

Two blocks are produced for each forecast horizon:

1. **Descriptive statistics**: mean, standard deviation, Ljung–Box Q(5) and Q(22) with significance
2. **Pearson correlation matrix** (lower triangle): pairwise correlations of LV and the 8 factors

Difference from the out-of-sample part
--------------------------------------
Factors are extracted once on the **full sample** (in-sample analysis), not in rolling windows.
The LV row is the HAR column of X **matching the horizon** — ``RV_1Day`` for h=1,
``RV_5Day`` for h=5 and ``RV_22Day`` for h=22 — rather than the forecast target Y.

Usage
-----
    python 02_data_description/basic_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from volfactor.evaluation import significance_stars
from volfactor.factors import extract_full_sample_factors
from volfactor.io import load_panel
from volfactor.paths import TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)

#: Column shown as "LV" in Table 1 for each horizon
LV_COLUMN = {1: "RV_1Day", 5: "RV_5Day", 22: "RV_22Day"}

LJUNG_BOX_LAGS = (5, 22)


def build_panel(horizon: int) -> pd.DataFrame:
    """Full-sample factors + LV: the 9-column panel needed for Table 1."""
    X, Y = load_panel(horizon=horizon)
    vf, svf = extract_full_sample_factors(X, Y)
    lv = X[LV_COLUMN[horizon]].rename("LV")
    return pd.concat([lv, vf, svf], axis=1)


def summarize(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation (ddof=1, as in pandas ``describe``) and Ljung–Box tests."""
    rows = []
    for name in panel.columns:
        series = panel[name]
        row = {"variable": name, "mean": series.mean(), "std": series.std()}

        for lag in LJUNG_BOX_LAGS:
            lb = sm.stats.acorr_ljungbox(series, lags=[lag], return_df=True)
            stat = lb.loc[lag, "lb_stat"]
            pval = lb.loc[lag, "lb_pvalue"]
            row[f"Q{lag}"] = stat
            row[f"Q{lag}_pvalue"] = pval
            row[f"Q{lag}_stars"] = significance_stars(pval)

        rows.append(row)

    return pd.DataFrame(rows).set_index("variable")


def lower_triangle(panel: pd.DataFrame) -> pd.DataFrame:
    """Lower triangle (with diagonal) of the Pearson correlation matrix; upper triangle left empty."""
    corr = panel.corr()
    mask = np.tri(len(corr), k=0, dtype=bool)
    return corr.where(mask)


def main() -> None:
    ensure_output_dirs()
    all_stats = []

    for h in HORIZONS:
        panel = build_panel(h)
        stats = summarize(panel)
        corr = lower_triangle(panel)

        corr.to_csv(TABLES_DIR / f"table1_correlation_{h}day.csv")

        stats.insert(0, "horizon", h)
        all_stats.append(stats)

        print(f"\n=== horizon = {h} day " + "=" * 40)
        for name, r in stats.iterrows():
            print(f"  {name:<6s} mean={r['mean']:>11.3e}  std={r['std']:>8.3f}  "
                  f"Q(5)={r['Q5']:>9.0f}{r['Q5_stars']:<3s} "
                  f"Q(22)={r['Q22']:>9.0f}{r['Q22_stars']:<3s}")

    combined = pd.concat(all_stats)
    out = TABLES_DIR / "table1_summary_statistics.csv"
    combined.to_csv(out)
    print(f"\n-> {out}")
    print(f"-> table1_correlation_{{1,5,22}}day.csv")


if __name__ == "__main__":
    main()
