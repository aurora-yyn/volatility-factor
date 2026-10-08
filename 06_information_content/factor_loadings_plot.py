#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 7 — the 50 companies with the largest loadings on each volatility factor, by industry (§5.1).

VF1 and VF3 are extracted statistically and carry no economic label by themselves. This figure interprets them
through their loadings: the 50 stocks with the largest absolute loading on a factor are drawn as bars coloured
by industry. VF1 is dominated by financial firms, VF3 by oil and gas firms.

Plot settings
-------------
- Loadings come from the **full-sample** PCA (not rolling windows).
- Stocks are selected by ``argsort(-|loading|)[:50]`` and sorted by (industry, ticker), so bars of the same industry are adjacent.
- Bar heights are the **signed** loadings; the loadings of the first component share one sign, so all bars are positive.
- The 14 industries have fixed colours, and the legend lists all industries (even those not in the top 50).

Usage
-----
    python 06_information_content/factor_loadings_plot.py
    python 06_information_content/factor_loadings_plot.py --factors 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from volfactor.factors import pca_loadings
from volfactor.io import load_industry_map, load_panel, stock_columns
from volfactor.paths import FIGURES_DIR, TABLES_DIR, ensure_output_dirs

TOP_N = 50

#: Fixed colours of the 14 industries (0–255 converted to 0–1)
INDUSTRY_COLORS = {
    "Oil and gas": (218, 0, 0),
    "Finance": (227, 97, 20),
    "Electricity": (240, 189, 36),
    "Technology": (232, 253, 60),
    "Food": (174, 254, 62),
    "Manufacturing": (146, 253, 59),
    "Pharma & Chemicals": (145, 254, 86),
    "Primary manufacturing": (144, 255, 163),
    "Machinery": (145, 255, 255),
    "Health": (89, 158, 253),
    "Transportation": (37, 65, 250),
    "Trade": (32, 0, 251),
    "Services": (109, 0, 251),
    "Mining": (191, 0, 251),
}
COLOR_MAP = {k: tuple(c / 255 for c in v) for k, v in INDUSTRY_COLORS.items()}

#: Short industry names used in the legend
LEGEND_LABELS = {"Oil and gas": "Oil"}


def top_loadings(loadings: pd.Series, industries: dict[str, str]) -> pd.DataFrame:
    """The TOP_N stocks with the largest absolute loadings, sorted by (industry, ticker)."""
    top = loadings.reindex(loadings.abs().sort_values(ascending=False).index[:TOP_N])

    frame = pd.DataFrame({
        "ticker": top.index,
        "industry": [industries.get(t, "Unknown") for t in top.index],
        "loading": top.to_numpy(),
    })
    return frame.sort_values(["industry", "ticker"]).reset_index(drop=True)


def plot_loadings(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(
        frame["ticker"],
        frame["loading"],
        color=[COLOR_MAP.get(i, (0.5, 0.5, 0.5)) for i in frame["industry"]],
    )
    ax.set_xticks([])  # 50 tickers do not fit; hidden as in the paper
    ax.set_ylabel("Factor Loadings")

    # The legend lists all 14 industries so that panels can be compared
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[i]) for i in INDUSTRY_COLORS]
    labels = [LEGEND_LABELS.get(i, i) for i in INDUSTRY_COLORS]
    ax.legend(handles=handles, labels=labels)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 3],
                    help="factors to plot, default 1 3 (the two panels of Figure 7)")
    args = ap.parse_args()

    ensure_output_dirs()

    X, _ = load_panel(horizon=1)
    panel = X[stock_columns(X)]
    loadings = pca_loadings(panel, n_factors=max(args.factors))

    mapping = load_industry_map()
    industries = dict(zip(mapping["ticker"], mapping["industry"]))

    frames = []
    for k in args.factors:
        frame = top_loadings(loadings.loc[f"VF{k}"], industries)

        path = FIGURES_DIR / f"figure7_loadings_vf{k}.png"
        plot_loadings(frame, path)

        counts = frame["industry"].value_counts()
        top3 = ", ".join(f"{n}×{c}" for n, c in counts.head(3).items())
        print(f"  VF{k}: industry composition of the top {TOP_N} -> {top3}"
              f"   loading range [{frame['loading'].min():.4f}, {frame['loading'].max():.4f}]")
        print(f"        -> {path.name}")

        frame.insert(0, "factor", f"VF{k}")
        frames.append(frame)

    out = TABLES_DIR / "figure7_top_loadings.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
