#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 4 / Figure H.17 — number of volatility factors over rolling windows (§3.3, Appendix H).

For every 5-year rolling window, the number of factors is estimated with the perturbed
eigenvalue ratio test of Pelger (2019), applied to the PCA panel and to the sPCA
scaled panel. The main specification uses γ = 0.08 (critical value 1.08); the
appendix uses the more conservative γ = 0.30.

The cumulative explained variance of the full-sample PCA is also reported, supporting
the choice K = 4.

Specification
-------------
In this script the sPCA scaling coefficients β_n are obtained by regressing the
target LV directly on each stock's LV.

Usage
-----
    python 01_factor_construction/eigenvalue_ratio.py                # γ=0.08 (Figure 4)
    python 01_factor_construction/eigenvalue_ratio.py --gamma 0.30   # Figure H.17
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from volfactor.factors import eigenvalues, estimate_n_factors, scaling_betas
from volfactor.forecast import DAYS_PER_YEAR
from volfactor.io import load_panel, stock_columns
from volfactor.paths import FIGURES_DIR, TABLES_DIR, ensure_output_dirs


def rolling_factor_counts(
    X: pd.DataFrame, Y: pd.Series, gamma: float, window_years: int = 5
) -> pd.DataFrame:
    """Number of PCA and sPCA factors for every rolling window."""
    window = DAYS_PER_YEAR * window_years
    stocks = stock_columns(X)
    Xs = X[stocks]
    y = Y.to_numpy()

    dates, n_pca, n_spca = [], [], []
    for i in range(len(X) - window + 1):
        train = Xs.iloc[i : i + window]

        n_pca.append(estimate_n_factors(train, gamma=gamma))

        beta = scaling_betas(train, y[i : i + window])
        n_spca.append(estimate_n_factors(train * beta, gamma=gamma))

        dates.append(X.index[i + window - 1])  # labelled by the window's last date

    return pd.DataFrame(
        {"window_end": dates, "n_factors_pca": n_pca, "n_factors_spca": n_spca}
    )


def describe(counts: list[int], label: str) -> None:
    total = len(counts)
    tally = Counter(counts)
    top2 = sum(tally[k] for k in (4, 5)) / total * 100

    print(f"\n--- {label} ---")
    print(f"  range {min(counts)}–{max(counts)}   median {np.median(counts):.1f}   "
          f"mean {np.mean(counts):.2f}")
    print(f"  K=4 or 5 : {top2:.1f}% of rolling windows")
    for k in sorted(tally):
        pct = tally[k] / total * 100
        print(f"    K={k}: {tally[k]:>5} ({pct:5.1f}%) {'#' * int(pct / 2)}")


def plot_counts(ax, dates, counts, title, color, median_color, oos_start) -> None:
    """One panel: factor counts over time + median line + K=4 reference + out-of-sample start.

    The reference lines keep the figure readable in black and white.
    """
    ax.plot(dates, counts, color=color, alpha=0.7, linewidth=0.8)

    med = np.median(counts)
    ax.axhline(med, color=median_color, linestyle="--", linewidth=1.5,
               label=f"Median = {med:.1f}")
    ax.axhline(4, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, label="K = 4")
    ax.axvline(oos_start, color="black", linestyle="-", linewidth=1.2, alpha=0.7)

    ax.set_ylabel("Number of Factors", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_yticks(range(0, 9))
    ax.set_ylim(-0.5, 8.5)
    ax.text(oos_start, 8.2, " Out-of-sample →", fontsize=8, va="top", ha="left", alpha=0.7)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def make_figures(counts: pd.DataFrame, oos_start, tag: str) -> list[Path]:
    """Save the two panels of Figure 4 as separate files."""
    specs = [
        ("n_factors_pca", "(a) VF (PCA)", "tab:red", "blue", f"figure4a_factor_count_pca_{tag}.png"),
        ("n_factors_spca", "(b) sVF (sPCA)", "tab:blue", "red", f"figure4b_factor_count_spca_{tag}.png"),
    ]
    paths = []
    for col, title, color, med_color, fname in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_counts(ax, counts["window_end"], counts[col].tolist(),
                    title, color, med_color, oos_start)
        fig.tight_layout()
        path = FIGURES_DIR / fname
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gamma", type=float, default=0.08,
                    help="Pelger threshold, default 0.08 (Appendix H uses 0.30)")
    args = ap.parse_args()

    ensure_output_dirs()
    X, Y = load_panel(horizon=1)

    # --- Full-sample explained variance: supports the choice K = 4 ---------
    # The denominator is the total variance of the panel (sum over all N components),
    # not the sum of the first k eigenvalues
    stocks = X[stock_columns(X)]
    lam = eigenvalues(stocks, k=10)
    ratio = lam / stocks.var(ddof=1).sum() * 100

    print("Full-sample PCA explained variance")
    print(f"  cumulative over first 4 factors : {ratio[:4].sum():.2f}%")
    print(f"  marginal contribution of 5th    : {ratio[4]:.2f}%")

    # --- Rolling-window factor counts ---------------------------------------
    print(f"\nRolling-window factor counts (Pelger 2019, gamma={args.gamma})")
    counts = rolling_factor_counts(X, Y, gamma=args.gamma)
    print(f"  {len(counts)} windows, "
          f"{counts['window_end'].min().date()} … {counts['window_end'].max().date()}")

    describe(counts["n_factors_pca"].tolist(), "PCA (VF)")
    describe(counts["n_factors_spca"].tolist(), "sPCA (sVF)")

    tag = f"gamma{args.gamma:g}".replace(".", "")
    path = TABLES_DIR / f"factor_counts_{tag}.csv"
    counts.to_csv(path, index=False)

    # Start of the out-of-sample period: the first 1260 observations form the
    # initial estimation window
    oos_start = X.index[DAYS_PER_YEAR * 5]
    figs = make_figures(counts, oos_start, tag)

    print(f"\n-> {path}")
    for f in figs:
        print(f"-> {f}")


if __name__ == "__main__":
    main()
