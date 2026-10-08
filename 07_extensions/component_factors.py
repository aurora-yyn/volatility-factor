#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tables 14 / 15 — factors extracted from volatility components (§6).

The main factors come from each stock's **total** realized volatility. This section uses different cross-sectional
information: each stock's volatility is split into two components, and factors are extracted from each to see
where the predictive power comes from.

Two decompositions
------------------
========== ============================================= ==================
Component   Meaning                                                  Factors
========== ============================================= ==================
continuous  continuous component (BPV): smooth everyday variation   CVF / CsVF
jump        jump component (RV − BPV): sudden events                 JVF / JsVF
good        upside volatility from positive returns                  GVF / GsVF
bad         downside volatility from negative returns                BVF / BsVF
========== ============================================= ==================

The first two are §6.1 (Table 14), the last two §6.2 (Table 15).

The forecast target is unchanged
--------------------------------
The HAR terms and the target always come from **market** realized volatility; only the cross-sectional panel used to
extract the factors changes. Tables 14 / 15 therefore share the benchmark of Table 4.

Usage
-----
    python 07_extensions/component_factors.py
    python 07_extensions/component_factors.py -c continuous -H 1
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
from volfactor.io import COMPONENTS, HAR_COLS, load_component_panel
from volfactor.paths import PREDICTIONS_DIR, TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
N_FACTORS = 4

#: Component -> (PCA factor prefix, sPCA factor prefix), the row labels of the paper's tables
PREFIXES = {
    "continuous": ("CVF", "CsVF"),
    "jump": ("JVF", "JsVF"),
    "bad": ("BVF", "BsVF"),
    "good": ("GVF", "GsVF"),
}


def run_one(component: str, horizon: int, window_years: int, end_year: int | None):
    """Out-of-sample performance of the 8 factor models for one (component, horizon) pair."""
    X, Y = load_component_panel(component, horizon=horizon, end_year=end_year)

    benchmark = rolling_ols_forecast(X[HAR_COLS], Y, window_years, name="HAR")

    pca_prefix, spca_prefix = PREFIXES[component]
    preds = pd.concat(
        [
            rolling_factor_forecast(X, Y, "pca", window_years, N_FACTORS),
            rolling_factor_forecast(X, Y, "spca", window_years, N_FACTORS),
        ],
        axis=1,
    )

    # The evaluation window is set by eval_end in COMPONENTS (§6.2 ends in 2014).
    eval_end = COMPONENTS[component]["eval_end"]
    idx = benchmark.index
    if eval_end is not None:
        idx = idx[idx.year <= eval_end]
    benchmark = benchmark.reindex(idx)
    preds = preds.reindex(idx)
    actual = Y.reindex(idx)

    rows = []
    for src, prefix in (("VF", pca_prefix), ("sVF", spca_prefix)):
        for k in range(1, N_FACTORS + 1):
            pred = preds[f"{src}{k}"]
            r2 = oos_r2(actual, pred, benchmark) * 100
            adj = mspe_adjusted(actual, pred, benchmark)
            cw, pval = clark_west(actual, pred, benchmark, two_sided=True)

            rows.append({
                "component": component,
                "horizon": horizon,
                "model": f"HAR-{prefix}{k}",
                "oos_r2": r2,
                "mspe_adj": adj,
                "cw_stat": cw,
                "cw_pvalue": pval,
                "stars": significance_stars(pval),
            })
            print(f"    HAR-{prefix}{k:<5d} R2={r2:8.3f}{significance_stars(pval):<3s}"
                  f" MSPE-adj={adj:7.3f}  p={pval:.3f}")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--components", nargs="+", default=list(PREFIXES),
                    choices=list(PREFIXES), help="volatility components, default: all four")
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=list(HORIZONS),
                    choices=list(HORIZONS), help="forecast horizons in days, default 1 5 22")
    ap.add_argument("-w", "--window", type=int, default=5, help="estimation window in years, default 5")
    ap.add_argument("--end-year", type=int, default=-1,
                    help="last year of the sample; the default -1 uses the period of each component in the paper")
    ap.add_argument("--resume", action="store_true",
                    help="continue from existing results, skipping completed combinations")
    args = ap.parse_args()

    ensure_output_dirs()
    path = TABLES_DIR / "table14_table15_components.csv"

    # A full run takes about 45 minutes. Results are saved after each combination, so an interrupted run
    # can be continued with --resume.
    frames = []
    done = set()
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
            print(f"\n=== {component} / horizon={h}d " + "=" * 34)
            frames.append(run_one(component, h, args.window, args.end_year))
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            print(f"  ({time.time() - t0:.0f}s, saved to {path.name})")

    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
