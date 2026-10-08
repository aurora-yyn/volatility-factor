#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 13 — incremental information of the volatility factors over other predictors (§5.2, benchmark 2).

The benchmark of the main text (benchmark 1) is plain HAR. This table uses a stricter benchmark: one of EPU / VIX / SKEW
is first added to HAR (benchmark 2), and the question is whether the volatility factors still reduce forecast errors
**on top of it**. If so, the factors carry information beyond these known predictors.

    Benchmark 2:    LV_{t,t+h} = HAR_t + γ · X_t + ε
    Extended model: LV_{t,t+h} = HAR_t + γ · X_t + β · VF_{k,t} + ε

Data
----
EPU / VIX / SKEW enter the regressions in levels, without the log-variance transformation. VIX and SKEW each miss
one day of the trading calendar, which is filled with the previous value.

The sample ends in 2014, as in the paper; adjust with ``--end-year``.
(The appendix table that evaluates VIX / EPU / SKEW on their own uses the full sample; see
``standalone_predictors.py``.)

Usage
-----
    python 06_information_content/incremental_predictability.py
    python 06_information_content/incremental_predictability.py -H 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from volfactor.evaluation import (
    clark_west,
    mspe_adjusted,
    oos_r2,
    significance_stars,
)
from volfactor.forecast import rolling_factor_forecast, rolling_ols_forecast
from volfactor.io import HAR_COLS, load_panel, load_predictors
from volfactor.paths import TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
N_FACTORS = 4

#: Group order and row labels of Table 13
PREDICTORS = {"epu": "EPU", "vix": "VIX", "skew": "SKEW"}


def run_horizon(horizon: int, window_years: int, end_year: int) -> pd.DataFrame:
    """Incremental performance of the factors over each of the three benchmark-2 models at one horizon."""
    X, Y = load_panel(horizon=horizon)
    mask = X.index.year <= end_year
    X, Y = X[mask], Y[mask]
    predictors = load_predictors().reindex(X.index).ffill()

    rows = []
    for var, label in PREDICTORS.items():
        extra = predictors[[var]]

        benchmark = rolling_ols_forecast(
            pd.concat([X[HAR_COLS], extra], axis=1), Y, window_years, name="benchmark"
        )
        models = rolling_factor_forecast(
            X, Y, "pca", window_years, N_FACTORS, extra=extra
        )
        actual = Y.reindex(benchmark.index)

        for k in range(1, N_FACTORS + 1):
            pred = models[f"VF{k}"]
            r2 = oos_r2(actual, pred, benchmark) * 100
            adj = mspe_adjusted(actual, pred, benchmark)
            cw, pval = clark_west(actual, pred, benchmark, two_sided=True)

            rows.append({
                "horizon": horizon,
                "benchmark": f"HAR-{label}",
                "model": f"HAR-{label}-VF{k}",
                "oos_r2": r2,
                "mspe_adj": adj,
                "cw_stat": cw,
                "cw_pvalue": pval,
                "stars": significance_stars(pval),
            })
            print(f"    HAR-{label}-VF{k:<3d} R2={r2:7.3f}{significance_stars(pval):<3s}"
                  f" MSPE-adj={adj:6.3f}  p={pval:.3f}")

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
        t0 = time.time()
        print(f"\n=== horizon={h}d  benchmark 2 (≤{args.end_year}) " + "=" * 26)
        frames.append(run_horizon(h, args.window, args.end_year))
        print(f"  ({time.time() - t0:.0f}s)")

    path = TABLES_DIR / "table13_incremental.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
