#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 3 — percentage of stocks in each sector (§3.1).

The 332 S&P 500 constituents of the sample are classified into 14 industries by their SIC codes
(``data/raw/industry_mapping.csv``). The figure shows the share of each industry; financial firms form the
largest group. Slice labels are whole-number percentages that sum to 100%.

Usage
-----
    python 02_data_description/sector_composition_plot.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volfactor.io import load_industry_map, load_panel, stock_columns
from volfactor.paths import FIGURES_DIR, TABLES_DIR, ensure_output_dirs

#: Industries in the order of the paper's legend -> (label, colour)
SECTORS = {
    "Oil and gas": ("Oil", "#4F81FF"),
    "Finance": ("Finance", "#16CC8A"),
    "Electricity": ("Electricity", "#FFC619"),
    "Technology": ("Technology", "#FF8041"),
    "Food": ("Food", "#F95F5F"),
    "Manufacturing": ("Manufacturing", "#A15CFF"),
    "Pharma & Chemicals": ("Pharmaceuticals and chemicals", "#0039C8"),
    "Primary manufacturing": ("Primary manufacturing", "#0D7A53"),
    "Machinery": ("Machinery", "#A87E00"),
    "Health": ("Health", "#C04000"),
    "Transportation": ("Transportation", "#C70707"),
    "Trade": ("Trade", "#5800D0"),
    "Services": ("Services", "#729AFF"),
    "Mining": ("Mining", "#32E9A7"),
}

#: Slice pulled out of the pie
HIGHLIGHT = "Finance"


def rounded_percentages(shares: pd.Series) -> list[int]:
    """Whole-number percentages that sum to 100 (largest remainder method), as used for the slice labels."""
    raw = shares / shares.sum() * 100
    floor = raw.astype(int)
    extra = 100 - int(floor.sum())
    order = (raw - floor).sort_values(ascending=False).index[:extra]
    floor.loc[order] += 1
    return floor.tolist()


def sector_shares() -> pd.DataFrame:
    """Number and share of sample stocks in each industry, in the legend order."""
    X, _ = load_panel(horizon=1)
    stocks = stock_columns(X)

    mapping = load_industry_map().set_index("ticker")["industry"]
    counts = mapping.reindex(stocks).value_counts()

    frame = pd.DataFrame({"industry": list(SECTORS)})
    frame["label"] = [SECTORS[i][0] for i in frame["industry"]]
    frame["n_stocks"] = [int(counts.get(i, 0)) for i in frame["industry"]]
    frame["share"] = frame["n_stocks"] / len(stocks)
    return frame


def plot_pie(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    wedges, _ = ax.pie(
        frame["share"],
        colors=[SECTORS[i][1] for i in frame["industry"]],
        explode=[0.04 if i == HIGHLIGHT else 0 for i in frame["industry"]],
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    for wedge, pct in zip(wedges, rounded_percentages(frame["share"])):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = wedge.center
        ax.text(x + 0.78 * np.cos(angle), y + 0.78 * np.sin(angle), f"{pct}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    ax.legend(wedges, frame["label"], loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=12)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args()

    ensure_output_dirs()

    frame = sector_shares()
    print(f"{frame['n_stocks'].sum()} stocks in {len(frame)} industries")
    for _, row in frame.iterrows():
        print(f"  {row['label']:<30s} {row['n_stocks']:>3d}  {row['share'] * 100:5.1f}%")

    table = TABLES_DIR / "figure3_sector_shares.csv"
    frame.to_csv(table, index=False)

    path = FIGURES_DIR / "figure3_sector_composition.png"
    plot_pie(frame, path)

    print(f"\n-> {table}\n-> {path}")


if __name__ == "__main__":
    main()
