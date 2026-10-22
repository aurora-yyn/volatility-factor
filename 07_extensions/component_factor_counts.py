#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures E.11 / E.12 — number of factors in the volatility components over rolling windows (Appendix E).

The counterpart of Figure H.17 for the four volatility components of §6: for every 5-year rolling window,
the number of factors is estimated with the perturbed eigenvalue ratio test of Pelger (2019), applied to the
PCA panel and to the sPCA scaled panel of each component.

========== ======================== =========================
Figure     Panels                   Component panels
========== ======================== =========================
E.11       (a) CVF  (b) JVF         continuous / jump, PCA
           (c) CsVF (d) JsVF        continuous / jump, sPCA
E.12       (a) BVF  (b) GVF         bad / good, PCA
           (c) BsVF (d) GsVF        bad / good, sPCA
========== ======================== =========================

Specification
-------------
- Sample 2004–2014, 5-year rolling windows (1,260 observations), γ = 0.30 (critical value 1.30).
- Panels as in §6: ``log(1 + x)`` for the continuous and jump components, ``log(x)`` for the good and bad
  components (these two panels contain no zeros); see ``COMPONENTS`` in ``volfactor.io``.
- sPCA scaling coefficients β_n are obtained by regressing the target LV (1-day horizon) directly on each
  stock's component, as in ``01_factor_construction/eigenvalue_ratio.py``. A stock whose component is
  constant within a window (no jumps in the whole window) gets β_n = 0.

Usage
-----
    python 07_extensions/component_factor_counts.py
    python 07_extensions/component_factor_counts.py -c jump continuous
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volfactor.factors import estimate_n_factors, scaling_betas
from volfactor.forecast import DAYS_PER_YEAR
from volfactor.io import load_component_panel, stock_columns
from volfactor.paths import FIGURES_DIR, TABLES_DIR, ensure_output_dirs

END_YEAR = 2014
WINDOW_YEARS = 5

#: (component, method) -> (figure file, panel label); order follows the panels of Figures E.11 / E.12
PANELS = {
    ("continuous", "pca"): ("figureE11a_factor_count_cvf.png", "CVF"),
    ("jump", "pca"): ("figureE11b_factor_count_jvf.png", "JVF"),
    ("continuous", "spca"): ("figureE11c_factor_count_csvf.png", "CsVF"),
    ("jump", "spca"): ("figureE11d_factor_count_jsvf.png", "JsVF"),
    ("bad", "pca"): ("figureE12a_factor_count_bvf.png", "BVF"),
    ("good", "pca"): ("figureE12b_factor_count_gvf.png", "GVF"),
    ("bad", "spca"): ("figureE12c_factor_count_bsvf.png", "BsVF"),
    ("good", "spca"): ("figureE12d_factor_count_gsvf.png", "GsVF"),
}
COMPONENT_ORDER = ["continuous", "jump", "bad", "good"]


def rolling_counts(component: str, gamma: float) -> pd.DataFrame:
    """Number of PCA and sPCA factors of one component for every rolling window."""
    X, Y = load_component_panel(component, horizon=1, end_year=END_YEAR)
    stocks = X[stock_columns(X)]
    if not np.isfinite(stocks.to_numpy()).all():
        raise ValueError(f"{component}: the logged panel contains non-finite values")

    window = DAYS_PER_YEAR * WINDOW_YEARS
    y = Y.to_numpy()

    dates, n_pca, n_spca = [], [], []
    for i in range(len(stocks) - window + 1):
        train = stocks.iloc[i : i + window]

        n_pca.append(estimate_n_factors(train, gamma=gamma))

        with np.errstate(invalid="ignore", divide="ignore"):
            beta = scaling_betas(train, y[i : i + window])
        beta = np.nan_to_num(beta, nan=0.0)
        n_spca.append(estimate_n_factors(train * beta, gamma=gamma))

        dates.append(stocks.index[i + window - 1])  # labelled by the window's last date

    return pd.DataFrame({
        "window_end": dates,
        f"{component}_pca": n_pca,
        f"{component}_spca": n_spca,
    })


def plot_panel(counts: pd.Series, path: Path) -> None:
    """One panel in the layout of the paper: counts by rolling window, x axis unlabelled."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(counts.to_numpy(), marker="o", linestyle="-", color="b")
    ax.set_ylabel("Number of Volatility Factors", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(True)
    ax.set_xticks([])
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--components", nargs="+", default=COMPONENT_ORDER,
                    choices=COMPONENT_ORDER, help="volatility components, default: all four")
    ap.add_argument("--gamma", type=float, default=0.30,
                    help="Pelger threshold, default 0.30")
    args = ap.parse_args()

    ensure_output_dirs()

    frames = []
    for component in [c for c in COMPONENT_ORDER if c in args.components]:
        t0 = time.time()
        counts = rolling_counts(component, args.gamma)
        print(f"\n=== {component}: {len(counts)} windows, "
              f"{counts['window_end'].min().date()} … {counts['window_end'].max().date()} "
              f"({time.time() - t0:.0f}s)")

        for method in ("pca", "spca"):
            col = f"{component}_{method}"
            fname, label = PANELS[(component, method)]
            tally = counts[col].value_counts().sort_index()
            dist = "  ".join(f"K={k}: {n / len(counts) * 100:.1f}%" for k, n in tally.items())
            print(f"  {label:<5s} median {counts[col].median():.1f}   {dist}")

            path = FIGURES_DIR / fname
            plot_panel(counts[col], path)
            print(f"        -> {path.name}")

        frames.append(counts.set_index("window_end"))

    tag = f"gamma{args.gamma:g}".replace(".", "")
    out = TABLES_DIR / f"figureE11_E12_factor_counts_{tag}.csv"
    pd.concat(frames, axis=1).reset_index().to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
