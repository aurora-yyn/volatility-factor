#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 3 — in-sample estimation results (§4.1).

Each volatility factor is added **separately** to the HAR model and estimated on the full sample::

    LV_{t,t+h} = c + β₁·RV_1Day + β₂·RV_5Day + β₃·RV_22Day + β_F·F_t + ε

The factor coefficient β_F, its t-statistic and the model R² are reported, with plain HAR as the benchmark row.
Factors are extracted on the full sample (in-sample analysis), unlike the rolling windows of §4.2.

Usage
-----
    python 03_in_sample_results/in_sample_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import statsmodels.api as sm

from volfactor.evaluation import significance_stars
from volfactor.factors import extract_full_sample_factors
from volfactor.io import HAR_COLS, load_panel
from volfactor.paths import TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
FACTORS = ["VF1", "VF2", "VF3", "VF4", "sVF1", "sVF2", "sVF3", "sVF4"]


def fit_one(Y: pd.Series, design: pd.DataFrame, factor: str | None) -> dict:
    """Fit HAR (factor=None) or HAR + one factor and return the statistics of the factor."""
    cols = HAR_COLS + ([factor] if factor else [])
    res = sm.OLS(Y, sm.add_constant(design[cols])).fit()

    row = {
        "model": f"HAR-{factor}" if factor else "HAR",
        "r2": res.rsquared,
        "adj_r2": res.rsquared_adj,
    }
    if factor:
        row.update(
            beta=res.params[factor],
            t_stat=res.tvalues[factor],
            p_value=res.pvalues[factor],
            stars=significance_stars(res.pvalues[factor]),
        )
    return row


def main() -> None:
    ensure_output_dirs()
    out = []

    for h in HORIZONS:
        X, Y = load_panel(horizon=h)
        vf, svf = extract_full_sample_factors(X, Y)
        design = pd.concat([X[HAR_COLS], vf, svf], axis=1)

        rows = [fit_one(Y, design, None)] + [fit_one(Y, design, f) for f in FACTORS]
        frame = pd.DataFrame(rows)
        frame.insert(0, "horizon", h)
        out.append(frame)

        print(f"\n=== horizon = {h} day " + "=" * 46)
        print(f"  {'model':<10s} {'R2':>7s} {'adj.R2':>8s} {'beta':>10s} {'t-stat':>9s}")
        for r in rows:
            beta = f"{r['beta']:.3f}{r['stars']}" if "beta" in r else "—"
            tstat = f"{r['t_stat']:.3f}" if "t_stat" in r else "—"
            print(f"  {r['model']:<10s} {r['r2']:7.3f} {r['adj_r2']:8.3f} "
                  f"{beta:>10s} {tstat:>9s}")

    path = TABLES_DIR / "table3_in_sample_regression.csv"
    pd.concat(out).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
