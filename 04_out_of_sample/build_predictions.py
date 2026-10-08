#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rolling out-of-sample forecasts — common input of Tables 4, 5, 7–10, 16 and Figures 5, 6.

For each (estimation window, horizon) pair, a matrix of rolling out-of-sample forecasts is built for the
**14 candidate models** listed in Table 7:

    HAR benchmark  +  4 HAR-VF  +  4 HAR-sVF  +  5 combination forecasts

The main specification is window = 5 years, horizon = 1/5/22 days; windows of 4/6 years are used in §4.3.2.
All nine combinations take about one hour.

Usage
-----
    python 04_out_of_sample/build_predictions.py                    # all 9 combinations
    python 04_out_of_sample/build_predictions.py -w 5 -H 1          # main specification only

Output
------
    outputs/predictions/w{W}_h{H}.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from volfactor.combination import rolling_combination_forecast, rolling_dmspe_forecast
from volfactor.evaluation import evaluate_models
from volfactor.forecast import DAYS_PER_YEAR, rolling_factor_forecast, rolling_har_forecast
from volfactor.io import load_actual_lv, load_panel
from volfactor.paths import PREDICTIONS_DIR, ensure_output_dirs

#: The 14 models of Table 7, in table order
MODEL_ORDER = [
    "HAR",
    "VF1", "VF2", "VF3", "VF4",
    "sVF1", "sVF2", "sVF3", "sVF4",
    "MC", "MDC", "TMC", "DMSPE_0.9", "DMSPE_1",
]


def build_one(window_years: int, horizon: int) -> pd.DataFrame:
    """Build the 14-model forecast matrix for one (window, horizon) pair."""
    X, Y = load_panel(horizon=horizon)

    parts = [
        rolling_har_forecast(X, Y, window_years),
        rolling_factor_forecast(X, Y, "pca", window_years),
        rolling_factor_forecast(X, Y, "spca", window_years),
        rolling_combination_forecast(X, Y, window_years),
        rolling_dmspe_forecast(X, Y, xi=0.9, window_years=window_years),
        rolling_dmspe_forecast(X, Y, xi=1.0, window_years=window_years),
    ]

    preds = pd.concat(parts, axis=1)[MODEL_ORDER]
    actual = load_actual_lv(horizon=horizon)

    out = preds.join(actual, how="inner")

    # All models must share the estimation window so that they cover the same forecast period.
    expected = len(X) - DAYS_PER_YEAR * window_years
    if len(out) != expected:
        raise RuntimeError(
            f"w={window_years} h={horizon}: got {len(out)} forecasts, expected {expected}. "
            f"Check that all models use the same window."
        )

    return out[["actual"] + MODEL_ORDER]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-w", "--windows", type=int, nargs="+", default=[4, 5, 6],
                    help="estimation window length in years, default 4 5 6")
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=[1, 5, 22],
                    help="forecast horizons in days, default 1 5 22")
    args = ap.parse_args()

    ensure_output_dirs()

    for w in args.windows:
        for h in args.horizons:
            t0 = time.time()
            print(f"\n=== window={w}y  horizon={h}d " + "=" * 40)

            df = build_one(w, h)
            path = PREDICTIONS_DIR / f"w{w}_h{h}.csv"
            df.to_csv(path)

            metrics = evaluate_models(df["actual"], df[MODEL_ORDER], "HAR")
            print(f"  {len(df)} predictions -> {path.name}   ({time.time() - t0:.0f}s)")
            print("  OOS R2 (%) vs HAR benchmark:")
            for name in MODEL_ORDER[1:]:
                r = metrics.loc[name]
                print(f"    {name:<10s} {r['OOS_R2']:7.3f}{r['stars']:<3s}"
                      f"  CW={r['CW_stat']:6.2f}")


if __name__ == "__main__":
    main()
