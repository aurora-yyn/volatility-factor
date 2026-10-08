#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 10 — out-of-sample performance of factors extracted from the correlation matrix (§4.3.3).

The main factors come from the **covariance matrix** of the stock LV panel, where more volatile stocks receive
larger loadings. This script extracts the factors from the **correlation matrix** of each training window
instead, to check whether the conclusions depend on this choice.

Combination forecasts
---------------------
MC / MDC / TMC do not involve PCA and are taken from ``outputs/predictions/w5_h{h}.csv``.
The DMSPE weights are set once over the whole out-of-sample period; see
:func:`volfactor.combination.full_period_dmspe_forecast`.

Usage
-----
    python 05_robustness/correlation_matrix_factors.py
    python 05_robustness/correlation_matrix_factors.py -H 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from volfactor.combination import full_period_dmspe_forecast
from volfactor.evaluation import evaluate_models
from volfactor.forecast import rolling_factor_forecast
from volfactor.io import load_actual_lv, load_panel
from volfactor.paths import PREDICTIONS_DIR, TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
FACTOR_MODELS = [f"VF{k}" for k in range(1, 5)] + [f"sVF{k}" for k in range(1, 5)]
COMBINATIONS = ["MC", "MDC", "TMC", "DMSPE_0.9", "DMSPE_1"]
MODEL_ORDER = ["HAR"] + FACTOR_MODELS + COMBINATIONS


def build_one(horizon: int, window_years: int) -> pd.DataFrame:
    """14-model forecast matrix for one horizon, correlation-matrix factors."""
    X, Y = load_panel(horizon=horizon)

    factors = pd.concat(
        [
            rolling_factor_forecast(X, Y, "pca", window_years, use_correlation=True),
            rolling_factor_forecast(X, Y, "spca", window_years, use_correlation=True),
        ],
        axis=1,
    )

    # HAR and MC / MDC / TMC do not involve PCA; reuse the main-specification results
    base_path = PREDICTIONS_DIR / f"w{window_years}_h{horizon}.csv"
    if not base_path.exists():
        raise FileNotFoundError(
            f"{base_path.name} is missing; run "
            f"`python 04_out_of_sample/build_predictions.py -w {window_years} -H {horizon}`"
        )
    base = pd.read_csv(base_path, index_col=0, parse_dates=True)

    dmspe = full_period_dmspe_forecast(X, Y, window_years=window_years)
    out = factors.join(base[["HAR", "MC", "MDC", "TMC"]], how="inner").join(dmspe, how="inner")
    out = out.join(load_actual_lv(horizon=horizon), how="inner")

    if len(out) != len(base):
        raise RuntimeError(
            f"h={horizon}: {len(out)} forecasts with correlation-matrix factors, "
            f"{len(base)} in the main specification; they should be equal."
        )

    return out[["actual"] + MODEL_ORDER]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=list(HORIZONS),
                    choices=list(HORIZONS), help="forecast horizons in days, default 1 5 22")
    ap.add_argument("-w", "--window", type=int, default=5, help="estimation window in years, default 5")
    args = ap.parse_args()

    ensure_output_dirs()
    frames = []

    for h in args.horizons:
        t0 = time.time()
        print(f"\n=== horizon={h}d  (correlation matrix) " + "=" * 32)

        preds = build_one(h, args.window)
        preds.to_csv(PREDICTIONS_DIR / f"w{args.window}_h{h}_corr.csv")

        metrics = evaluate_models(preds["actual"], preds[MODEL_ORDER], "HAR")
        metrics = metrics.loc[MODEL_ORDER].reset_index(names="model")
        metrics.insert(0, "horizon", h)
        frames.append(metrics)

        print(f"  {len(preds)} predictions   ({time.time() - t0:.0f}s)")
        for name in MODEL_ORDER[1:]:
            r = metrics[metrics.model == name].iloc[0]
            print(f"    {name:<10s} R2={r['OOS_R2']:7.3f}{r['stars']:<3s}"
                  f"  CW={r['CW_stat']:6.2f}  p={r['CW_pvalue']:.3f}")

    path = TABLES_DIR / "table10_correlation_matrix.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
