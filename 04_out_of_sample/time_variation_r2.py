#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 5 — out-of-sample R² by year (§4.2.2).

The out-of-sample period is split by calendar year, and the out-of-sample R² of the four factors relative to HAR
is computed within each year and drawn as grouped bars: 3 horizons × {PCA, sPCA} = 6 panels.

The vertical axis shows R² as a **fraction** (not a percentage), as in the paper.

Usage
-----
    python 04_out_of_sample/time_variation_r2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volfactor.evaluation import oos_r2
from volfactor.paths import FIGURES_DIR, TABLES_DIR, PREDICTIONS_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
SOURCES = {"pca": "VF", "spca": "sVF"}

BAR_WIDTH = 0.1
AXIS_FONT, TICK_FONT, LEGEND_FONT = 18, 16, 14


def yearly_r2(preds: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Out-of-sample R² (fraction) by year × factor."""
    years = sorted(preds.index.year.unique())
    rows = {}
    for k in range(1, 5):
        rows[f"{prefix}{k}"] = [
            oos_r2(
                preds.loc[preds.index.year == y, "actual"],
                preds.loc[preds.index.year == y, f"{prefix}{k}"],
                preds.loc[preds.index.year == y, "HAR"],
            )
            for y in years
        ]
    return pd.DataFrame(rows, index=years).T


def plot_yearly(results: pd.DataFrame, path: Path) -> None:
    """Grouped bars: years on the horizontal axis, four bars per year, one per factor.

    The legend is always shown and each factor has a fixed bar position, so identification does not rely on colour alone.
    """
    n_years, n_models = results.shape[1], results.shape[0]
    index = np.arange(n_years) * (n_models + 1) * BAR_WIDTH

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, model in enumerate(results.index):
        ax.bar(index + i * BAR_WIDTH, results.loc[model], BAR_WIDTH, label=model)

    ax.set_xlabel("Year", fontsize=AXIS_FONT)
    ax.set_ylabel(r"$R^2$ Value", fontsize=AXIS_FONT, labelpad=15)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT)
    ax.set_xticks(index + BAR_WIDTH * (n_models - 1) / 2)
    ax.set_xticklabels(results.columns)
    ax.legend(title="Volatility Factors", fontsize=LEGEND_FONT,
              title_fontsize=LEGEND_FONT)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    tables = []

    for h in HORIZONS:
        preds = pd.read_csv(PREDICTIONS_DIR / f"w5_h{h}.csv", index_col=0, parse_dates=True)

        for source, prefix in SOURCES.items():
            results = yearly_r2(preds, prefix)

            path = FIGURES_DIR / f"figure5_yearly_r2_{source}_{h}day.png"
            plot_yearly(results, path)

            tidy = results.reset_index(names="factor").melt(
                id_vars="factor", var_name="year", value_name="oos_r2"
            )
            tidy.insert(0, "horizon", h)
            tables.append(tidy)

            rng = f"[{results.to_numpy().min():+.3f}, {results.to_numpy().max():+.3f}]"
            print(f"  h={h:2d}d  {prefix:<4s} years {results.columns[0]}–{results.columns[-1]}"
                  f"  R2 range {rng}  -> {path.name}")

    out = TABLES_DIR / "figure5_yearly_r2.csv"
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
