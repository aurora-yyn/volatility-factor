#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 16 — out-of-sample performance of the five realized partial variance components (§6.3).

The daily realized variance of the S&P 500 is split into five parts by quantiles of the high-frequency returns
(Bollerslev et al., 2022); each part forms one panel:

    Panel A  RV_ExtremeNeg    < 10th percentile   extreme negative returns
    Panel B  RV_ModerateNeg   10th – 40th
    Panel C  RV_Neutral       40th – 60th
    Panel D  RV_ModeratePos   60th – 90th
    Panel E  RV_ExtremePos    > 90th              extreme positive returns

The model is "HAR + this component + one factor"::

    LV_{t,t+h} = HAR_t + γ · C_t + β · F_{k,t} + ε

where C is the current component and F_k is the k-th principal component extracted from **the other four components + RV_final**.

Setup
-----
1. PCA on 5 columns: the other 4 partial variance components and ``RV_final``.
2. X and Y are ``log(RV)``; forecasts are multiplied by 2 before evaluation against the ``log(RV²)`` benchmark.
3. The HAR benchmark is taken from ``outputs/predictions/w5_h{h}.csv``, as in Table 4.
4. The evaluation window ends in 2014.
5. The "MSPE-adj" column reports the Clark–West statistic; p-values are two-sided.

Usage
-----
    python 07_extensions/partial_variance_factors.py
    python 07_extensions/partial_variance_factors.py -c RV_Neutral -H 1
    python 07_extensions/partial_variance_factors.py --resume     # continue from existing results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from volfactor.evaluation import clark_west, oos_r2, significance_stars
from volfactor.forecast import rolling_factor_forecast
from volfactor.io import HAR_COLS, PARTIAL_COMPONENTS, load_partial_variance_panel
from volfactor.paths import PREDICTIONS_DIR, TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
N_FACTORS = 4

#: Factor to convert forecasts from log(RV) to log(RV²): log(RV²) = 2·log(RV).
LOG_VARIANCE_SCALE = 2.0

#: Last year of the evaluation window (forecasts are generated on the full 2004–2016 sample).
EVAL_END_YEAR = 2014


def _benchmark(horizon: int) -> tuple[pd.Series, pd.Series]:
    """HAR benchmark and realized values of the main analysis (both on the log(RV²) scale)."""
    path = PREDICTIONS_DIR / f"w5_h{horizon}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run 04_out_of_sample/build_predictions.py -w 5 -H {horizon} first"
        )
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return d["HAR"], d["actual"]


def run_one(component: str, horizon: int, window_years: int) -> pd.DataFrame:
    """Out-of-sample performance of the 8 factor models for one (component, horizon) pair."""
    X, Y, pca_cols = load_partial_variance_panel(component, horizon=horizon)
    base_cols = HAR_COLS + [component]

    preds = pd.concat(
        [
            rolling_factor_forecast(X, Y, m, window_years, N_FACTORS,
                                    base_cols=base_cols, factor_cols=pca_cols,
                                    beta_on_fitted=True)
            for m in ("pca", "spca")
        ],
        axis=1,
    )
    preds *= LOG_VARIANCE_SCALE

    benchmark, actual = _benchmark(horizon)
    idx = preds.index.intersection(benchmark.index)
    idx = idx[idx.year <= EVAL_END_YEAR]
    preds, benchmark, actual = preds.reindex(idx), benchmark.reindex(idx), actual.reindex(idx)

    rows = []
    for col in preds.columns:
        pred = preds[col]
        r2 = oos_r2(actual, pred, benchmark) * 100
        cw, pval = clark_west(actual, pred, benchmark, two_sided=True)
        # VF1 -> PVF1, sVF1 -> PsVF1 (row labels of Table 16)
        label = f"PsVF{col[3:]}" if col.startswith("sVF") else f"PVF{col[2:]}"
        rows.append({
            "component": component, "horizon": horizon, "model": f"HAR-{label}",
            "oos_r2": r2, "cw_stat": cw, "cw_pvalue": pval,
            "stars": significance_stars(pval),
        })
        print(f"    HAR-{label:<7s} R2={r2:8.3f}{significance_stars(pval):<3s}"
              f" CW={cw:7.2f}  p={pval:.3f}")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--components", nargs="+", default=list(PARTIAL_COMPONENTS),
                    choices=list(PARTIAL_COMPONENTS), help="partial variance components, default: all five")
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=list(HORIZONS),
                    choices=list(HORIZONS), help="forecast horizons in days, default 1 5 22")
    ap.add_argument("-w", "--window", type=int, default=5, help="estimation window in years, default 5")
    ap.add_argument("--resume", action="store_true",
                    help="continue from existing results, skipping completed combinations")
    args = ap.parse_args()

    ensure_output_dirs()
    path = TABLES_DIR / "table16_partial_variances.csv"

    frames, done = [], set()
    if args.resume and path.exists():
        prev = pd.read_csv(path)
        frames.append(prev)
        done = set(zip(prev["component"], prev["horizon"]))
        print(f"Continuing {path.name}: {len(done)} combinations already done")

    for component in args.components:
        for h in args.horizons:
            if (component, h) in done:
                print(f"\n=== {component} / horizon={h}d — already done, skipped")
                continue
            t0 = time.time()
            print(f"\n=== {component} / horizon={h}d " + "=" * 30)
            frames.append(run_one(component, h, args.window))
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            print(f"  ({time.time() - t0:.0f}s, saved to {path.name})")

    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
