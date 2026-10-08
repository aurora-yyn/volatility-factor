#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 1 — ACF / PACF of LV and the volatility factors (§3.2.2).

For each of the 9 series (LV + VF1–4 + sVF1–4), the autocorrelation and partial autocorrelation functions up to lag 50
are plotted with 95% confidence bands. Layout as in the paper: 3 variables per row, each with an ACF and a PACF panel.

The sample ends in 2014, as in the paper; adjust with ``--end-year``.

Usage
-----
    python 02_data_description/acf_pacf_plot.py
    python 02_data_description/acf_pacf_plot.py --end-year 2016
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # render without a display
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from volfactor.factors import extract_full_sample_factors
from volfactor.io import load_panel
from volfactor.paths import FIGURES_DIR, ensure_output_dirs

MAX_LAGS = 50
ALPHA = 0.05
VARS_PER_ROW = 3

#: Font used in the paper's figures; matplotlib falls back to the default sans-serif if missing
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def build_panel(end_year: int) -> pd.DataFrame:
    """LV + 8 factors in the paper's order. Factors are extracted on the truncated sample."""
    X, Y = load_panel(horizon=1)
    mask = X.index.year <= end_year
    X, Y = X[mask], Y[mask]

    vf, svf = extract_full_sample_factors(X, Y)
    return pd.concat([X["RV_1Day"].rename("LV"), vf, svf], axis=1)


def plot_grid(data: pd.DataFrame, path: Path) -> None:
    n_rows = (len(data.columns) + VARS_PER_ROW - 1) // VARS_PER_ROW
    fig, axes = plt.subplots(n_rows, VARS_PER_ROW * 2, figsize=(20, 12))
    fig.suptitle(
        "Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) Analysis",
        fontsize=16,
        y=1.00,
    )
    axes = axes.flatten()

    for i, col in enumerate(data.columns):
        series = data[col].to_numpy()

        ax = axes[i * 2]
        plot_acf(series, lags=MAX_LAGS, ax=ax, alpha=ALPHA)
        ax.set_title(f"{col} - ACF", fontsize=10)
        ax.set_xlabel("Lag", fontsize=8)
        ax.set_ylabel("ACF", fontsize=8)

        ax = axes[i * 2 + 1]
        plot_pacf(series, lags=MAX_LAGS, ax=ax, alpha=ALPHA, method="ywm")
        ax.set_title(f"{col} - PACF", fontsize=10)
        ax.set_xlabel("Lag", fontsize=8)
        ax.set_ylabel("PACF", fontsize=8)

    for j in range(len(data.columns) * 2, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--end-year", type=int, default=2014,
                    help="last year of the sample, default 2014")
    args = ap.parse_args()

    ensure_output_dirs()
    panel = build_panel(args.end_year)

    suffix = "" if args.end_year == 2014 else f"_to{args.end_year}"
    path = FIGURES_DIR / f"figure1_acf_pacf{suffix}.png"
    plot_grid(panel, path)

    print(f"{len(panel.columns)} series, lags={MAX_LAGS}, "
          f"sample {panel.index.min().date()} – {panel.index.max().date()} "
          f"({len(panel)} obs)")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
