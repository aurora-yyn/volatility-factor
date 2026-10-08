#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 2 — LV time series of the S&P 500 and the volatility factors (§3.2.2).

The 9 panels pair each sVF with its VF, with LV in the last cell:

    sVF1  VF1
    sVF2  VF2
    sVF3  VF3
    sVF4  VF4
    LV    (empty)

so that the sPCA and PCA factors of the same order sit side by side for comparison.

Usage
-----
    python 02_data_description/factor_timeseries_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from volfactor.factors import extract_full_sample_factors
from volfactor.io import load_panel
from volfactor.paths import FIGURES_DIR, ensure_output_dirs

#: Panel order: sVF and VF side by side, LV last
PLOT_ORDER = ["sVF1", "VF1", "sVF2", "VF2", "sVF3", "VF3", "sVF4", "VF4", "LV"]


def build_panel() -> pd.DataFrame:
    X, Y = load_panel(horizon=1)
    vf, svf = extract_full_sample_factors(X, Y)
    panel = pd.concat([X["RV_1Day"].rename("LV"), vf, svf], axis=1)
    return panel[PLOT_ORDER]


def main() -> None:
    ensure_output_dirs()
    panel = build_panel()

    fig, axes = plt.subplots(5, 2, figsize=(12, 9))
    fig.tight_layout(pad=3.0)

    for i, col in enumerate(panel.columns):
        ax = axes[i // 2, i % 2]
        ax.plot(panel.index, panel[col])
        ax.set_title(col)

    axes[4, 1].axis("off")  # 9 series do not fill the 5×2 grid

    path = FIGURES_DIR / "figure2_factor_timeseries.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"{len(panel.columns)} series, {panel.index.min().date()} – {panel.index.max().date()}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
