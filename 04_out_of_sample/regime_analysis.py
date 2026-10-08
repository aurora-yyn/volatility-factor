#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 5 — out-of-sample R² across volatility regimes (§4.2, regime analysis).

The out-of-sample period is split into three regimes by the **same-day 1-day realized volatility**, and within
each regime the out-of-sample R² and Clark–West test of every factor relative to HAR are computed:

    Low     RV < 0.075
    Normal  0.075 ≤ RV ≤ 0.115
    High    RV > 0.115

The thresholds apply to RV on the standard-deviation scale, so the modelling LV is transformed back: ``RV = exp(LV / 2)``.

Factors
-------
Table 5 uses the factors extracted by residual sPCA, i.e. the default ``--factor-source spca``.

Usage
-----
    python 04_out_of_sample/regime_analysis.py
    python 04_out_of_sample/regime_analysis.py --factor-source pca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volfactor.evaluation import clark_west, oos_r2, significance_stars
from volfactor.io import load_panel
from volfactor.paths import FIGURES_DIR, PREDICTIONS_DIR, TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
REGIMES = ("Low", "Normal", "High")

#: Regime thresholds for RV (standard-deviation scale)
RV_LOW, RV_HIGH = 0.075, 0.115

#: Colour scale of the heatmaps, as in the paper
HEATMAP_VMIN, HEATMAP_VMAX = -8, 13

REGIME_TITLES = {
    "Low": "Low Volatility",
    "Normal": "Normal Volatility",
    "High": "High Volatility",
}
HORIZON_LABELS = ["1-day", "5-day", "22-day"]
FACTOR_LABELS = ["VF1", "VF2", "VF3", "VF4"]


def regime_labels(horizon: int) -> pd.Series:
    """Label every trading day with its regime according to the same-day RV_1Day."""
    X, _ = load_panel(horizon=horizon)
    rv = np.exp(X["RV_1Day"] / 2)  # LV = log(RV²) ⇒ RV = exp(LV/2)

    labels = pd.Series("Normal", index=X.index, name="regime")
    labels[rv < RV_LOW] = "Low"
    labels[rv > RV_HIGH] = "High"
    return labels


def regime_matrix(result: pd.DataFrame, regime: str) -> np.ndarray:
    """Extract the 4 factors × 3 horizons matrix for one regime."""
    return np.array([
        [result[(result.model == f"HAR-VF{k}") & (result.horizon == h)
                & (result.regime == regime)]["oos_r2"].iloc[0]
         for h in HORIZONS]
        for k in range(1, 5)
    ])


def plot_heatmap(matrix: np.ndarray, title: str, path) -> None:
    """Heatmap for one regime (one panel of Figure 6).

    Uses the RdYlGn diverging colour map, as in the paper. **Every cell is annotated with its value**,
    so reading the figure does not depend on colour.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(HORIZON_LABELS, fontsize=11)
    ax.set_yticklabels(FACTOR_LABELS, fontsize=11)

    for i in range(4):
        for j in range(3):
            v = matrix[i, j]
            color = "white" if abs(v) > 6 else "black"
            weight = "bold" if v > 0 else "normal"
            if v == matrix[:, j].max() and v > 0:  # best factor at this horizon
                weight = "bold"
                color = "darkgreen" if v > 8 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight=weight)

    ax.set_xlabel("Prediction Horizon", fontsize=12, fontweight="bold")
    ax.set_ylabel("Volatility Factor", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$R^2_{os}$ (%)", rotation=270, labelpad=15, fontsize=11)

    ax.set_xticks(np.arange(3) - 0.5, minor=True)
    ax.set_yticks(np.arange(4) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--factor-source", choices=["spca", "pca"], default="spca",
                    help="which factors to use, default spca (as in Table 5)")
    args = ap.parse_args()

    ensure_output_dirs()
    prefix = "sVF" if args.factor_source == "spca" else "VF"

    rows = []
    for h in HORIZONS:
        preds = pd.read_csv(PREDICTIONS_DIR / f"w5_h{h}.csv", index_col=0, parse_dates=True)
        labels = regime_labels(h).reindex(preds.index)

        for k in range(1, 5):
            for regime in REGIMES:
                mask = labels == regime
                actual = preds.loc[mask, "actual"]
                bench = preds.loc[mask, "HAR"]
                pred = preds.loc[mask, f"{prefix}{k}"]

                r2 = oos_r2(actual, pred, bench) * 100
                cw, pval = clark_west(actual, pred, bench)

                rows.append({
                    "model": f"HAR-VF{k}",
                    "horizon": h,
                    "regime": regime,
                    "n_obs": int(mask.sum()),
                    "oos_r2": r2,
                    "cw_stat": cw,
                    "cw_pvalue": pval,
                    "stars": significance_stars(pval),
                })

    result = pd.DataFrame(rows)
    path = TABLES_DIR / "table5_regime_analysis.csv"
    result.to_csv(path, index=False)

    # Print in the paper's layout: rows = factors, columns = (horizon, regime)
    print(f"Out-of-sample R2 (%) by volatility regime   [factors: {prefix}]")
    counts = result.groupby("regime")["n_obs"].first()
    print(f"  regime sizes (h=1): " +
          ", ".join(f"{r}={counts.get(r, 0)}" for r in REGIMES))
    header = "".join(f"{f'{h}d-{r}':>12s}" for h in HORIZONS for r in REGIMES)
    print(f"\n  {'Model':<10s}{header}")
    for k in range(1, 5):
        cells = []
        for h in HORIZONS:
            for r in REGIMES:
                row = result[(result.model == f"HAR-VF{k}") &
                             (result.horizon == h) & (result.regime == r)].iloc[0]
                cells.append(f"{row['oos_r2']:.2f}{row['stars']}")
        print(f"  {'HAR-VF' + str(k):<10s}" + "".join(f"{c:>12s}" for c in cells))

    print(f"\n-> {path}")

    # Figure 6: one heatmap per regime
    for regime in REGIMES:
        fig_path = FIGURES_DIR / f"figure6_regime_heatmap_{regime.lower()}.png"
        plot_heatmap(regime_matrix(result, regime), REGIME_TITLES[regime], fig_path)
        print(f"-> {fig_path}")


if __name__ == "__main__":
    main()
