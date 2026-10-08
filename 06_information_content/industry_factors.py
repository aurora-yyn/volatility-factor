#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 12 — out-of-sample performance of industry factors (§5.1).

The 332 stocks are grouped into 14 industries; each industry factor is the **cross-sectional equal-weighted average of
the stock LVs within the industry**, added to HAR and evaluated out of sample. This gives an interpretable
counterpart to the statistical factors: the finance factor performs closest to VF1, and the oil factor, like VF3, at long horizons.

Industry factors are daily means of the stock LVs within each industry; they involve no estimated parameters and are computed once on the full sample.

The sample ends in 2014, as in the paper; adjust with ``--end-year``. The HAR benchmark is re-estimated
over the same period.

p-values are two-sided.
The "MSPE-adj" column is the mean of the Clark–West adjusted loss differential.

Usage
-----
    python 06_information_content/industry_factors.py
    python 06_information_content/industry_factors.py --end-year 2016
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from volfactor.evaluation import (
    clark_west,
    mspe_adjusted,
    oos_r2,
    significance_stars,
)
from volfactor.forecast import rolling_ols_forecast
from volfactor.io import HAR_COLS, load_industry_map, load_panel
from volfactor.paths import TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)

#: Row order of Table 12; keys are the names in industry_mapping.csv
INDUSTRY_LABELS = {
    "Oil and gas": "Oil",
    "Finance": "Finance",
    "Electricity": "Electricity",
    "Technology": "Technology",
    "Food": "Food",
    "Manufacturing": "Manufacturing",
    "Pharma & Chemicals": "Pharma & Chemicals",
    "Primary manufacturing": "Primary manufacturing",
    "Machinery": "Machinery",
    "Health": "Health",
    "Transportation": "Transportation",
    "Trade": "Trade",
    "Services": "Services",
    "Mining": "Mining",
}


def industry_groups(columns: list[str]) -> dict[str, list[str]]:
    """Industry -> the stock columns of that industry in the panel."""
    mapping = load_industry_map()
    present = set(columns)

    groups = {}
    for industry in INDUSTRY_LABELS:
        tickers = mapping.loc[mapping["industry"] == industry, "ticker"]
        cols = [t for t in tickers if t in present]
        if cols:
            groups[industry] = cols
    return groups


def run_horizon(horizon: int, window_years: int, end_year: int) -> pd.DataFrame:
    """Out-of-sample performance of the 14 industry factors at one horizon."""
    X, Y = load_panel(horizon=horizon)
    mask = X.index.year <= end_year
    X, Y = X[mask], Y[mask]

    benchmark = rolling_ols_forecast(X[HAR_COLS], Y, window_years, name="HAR")
    actual = Y.reindex(benchmark.index)

    rows = []
    for industry, cols in industry_groups(list(X.columns)).items():
        factor = X[cols].mean(axis=1).rename("industry_factor")
        features = pd.concat([X[HAR_COLS], factor], axis=1)

        pred = rolling_ols_forecast(features, Y, window_years, name="industry")

        r2 = oos_r2(actual, pred, benchmark) * 100
        adj = mspe_adjusted(actual, pred, benchmark)
        cw, pval = clark_west(actual, pred, benchmark, two_sided=True)
        label = INDUSTRY_LABELS[industry]

        rows.append({
            "horizon": horizon,
            "industry": label,
            "n_stocks": len(cols),
            "oos_r2": r2,
            "mspe_adj": adj,
            "cw_stat": cw,
            "cw_pvalue": pval,
            "stars": significance_stars(pval),
        })
        print(f"    {label:<24s} n={len(cols):<3d} R2={r2:7.3f}"
              f"{significance_stars(pval):<3s} MSPE-adj={adj:.3f} p={pval:.3f}")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=list(HORIZONS),
                    choices=list(HORIZONS), help="forecast horizons in days, default 1 5 22")
    ap.add_argument("-w", "--window", type=int, default=5, help="estimation window in years, default 5")
    ap.add_argument("--end-year", type=int, default=2014, help="last year of the sample, default 2014")
    args = ap.parse_args()

    ensure_output_dirs()
    frames = []

    for h in args.horizons:
        print(f"\n=== horizon={h}d  industry factors (≤{args.end_year}) " + "=" * 22)
        frames.append(run_horizon(h, args.window, args.end_year))

    path = TABLES_DIR / "table12_industry_factors.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
