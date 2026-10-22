#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 7 / Figures B.9, B.10, F.13–F.16 — factor loadings by industry (§5.1, Appendices B and F).

VF1 and VF3 are extracted statistically and carry no economic label by themselves. Figure 7 interprets them
through their loadings: the 50 stocks with the largest absolute loading on a factor are drawn as bars coloured
by industry. VF1 is dominated by financial firms, VF3 by oil and gas firms. The appendix figures apply the same
plot to the remaining factors and to the factors of the volatility components.

========== ============================================= ======================= =========
Figure      Panel                                          Factors                 Stocks
========== ============================================= ======================= =========
7           total realized volatility, LV = log(RV²)       VF1, VF3                top 50
B.9         total realized volatility, LV = log(RV²)       VF2, VF4                top 50
B.10        total realized volatility, LV = log(RV²)       VF1–VF4                 all 332
F.13        jump component, log(1 + x), 2004–2014          JVF1–JVF4               all 332
F.14        continuous component, log(1 + x), 2004–2014    CVF1–CVF4               all 332
F.15        bad volatility, log(1 + x), 2004–2014          BVF1–BVF4               all 332
F.16        good volatility, log(1 + x), 2004–2014         GVF1–GVF4               all 332
========== ============================================= ======================= =========

Plot settings
-------------
- Loadings come from a **full-sample** PCA (not rolling windows).
- Stocks are sorted by (industry, ticker), so bars of the same industry are adjacent. Top-50 panels keep the
  stocks selected by ``argsort(-|loading|)[:50]``.
- Bar heights are the **signed** loadings. The sign of a principal component is arbitrary; the figures fix it
  so that the factor score with the largest absolute value is positive (the convention of scikit-learn's PCA,
  see :func:`display_loadings`).
- The 14 industries have fixed colours, and the legend lists all industries (even those not in a panel).
- The component panels are those of §6 (τ = 5 for the jump / continuous split). The loading figures of
  Appendix F apply ``log(1 + x)`` to all four component panels over 2004–2014.

Usage
-----
    python 06_information_content/factor_loadings_plot.py                 # all figures
    python 06_information_content/factor_loadings_plot.py --figures 7 B9  # selected figures
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

from volfactor.factors import pca_loadings
from volfactor.io import COMPONENTS, load_component_panel, load_industry_map, load_panel, stock_columns
from volfactor.paths import FIGURES_DIR, TABLES_DIR, ensure_output_dirs

TOP_N = 50

#: Last year of the component panels used in Appendix F
COMPONENT_END_YEAR = 2014

#: Paper figure -> (panel, factors, number of stocks shown; None = all)
FIGURES = {
    "7": ("total", (1, 3), TOP_N),
    "B9": ("total", (2, 4), TOP_N),
    "B10": ("total", (1, 2, 3, 4), None),
    "F13": ("jump", (1, 2, 3, 4), None),
    "F14": ("continuous", (1, 2, 3, 4), None),
    "F15": ("bad", (1, 2, 3, 4), None),
    "F16": ("good", (1, 2, 3, 4), None),
}

#: Panel -> factor prefix used in the paper
PREFIXES = {"total": "VF", "jump": "JVF", "continuous": "CVF", "bad": "BVF", "good": "GVF"}

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


def stock_panel(panel: str) -> pd.DataFrame:
    """Stock columns of the panel whose loadings are plotted."""
    if panel == "total":
        X, _ = load_panel(horizon=1)
        return X[stock_columns(X)]

    X, _ = load_component_panel(panel, horizon=1, end_year=COMPONENT_END_YEAR)
    stocks = X[stock_columns(X)]
    # load_component_panel returns log(shift + x); rewrite it as log(1 + x)
    return np.log1p(np.exp(stocks) - COMPONENTS[panel]["shift"])


def display_loadings(panel: pd.DataFrame, n_factors: int = 4) -> pd.DataFrame:
    """Full-sample PCA loadings, signed so that each factor's largest absolute score is positive.

    Flipping the sign of a component changes neither its information nor any forecast; it only
    decides whether the bars of a panel point up or down.
    """
    loadings = pca_loadings(panel, n_factors=n_factors)
    scores = (panel - panel.mean()).to_numpy() @ loadings.to_numpy().T
    signs = np.sign(scores[np.argmax(np.abs(scores), axis=0), np.arange(n_factors)])
    return loadings * signs[:, None]


def sorted_loadings(
    loadings: pd.Series, industries: dict[str, str], top_n: int | None
) -> pd.DataFrame:
    """Stocks sorted by (industry, ticker); with ``top_n``, only the largest absolute loadings."""
    if top_n is not None:
        loadings = loadings.reindex(loadings.abs().sort_values(ascending=False).index[:top_n])

    frame = pd.DataFrame({
        "ticker": loadings.index,
        "industry": [industries.get(t, "Unknown") for t in loadings.index],
        "loading": loadings.to_numpy(),
    })
    return frame.sort_values(["industry", "ticker"]).reset_index(drop=True)


def plot_loadings(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(
        frame["ticker"],
        frame["loading"],
        color=[COLOR_MAP.get(i, (0.5, 0.5, 0.5)) for i in frame["industry"]],
    )
    ax.set_xticks([])  # tickers do not fit; hidden as in the paper
    ax.set_ylabel("Factor Loadings")

    # The legend lists all 14 industries so that panels can be compared
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[i]) for i in INDUSTRY_COLORS]
    labels = [LEGEND_LABELS.get(i, i) for i in INDUSTRY_COLORS]
    ax.legend(handles=handles, labels=labels)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def file_name(figure: str, prefix: str, k: int) -> str:
    if figure == "7":
        return f"figure7_loadings_vf{k}.png"
    return f"figure{figure}_loadings_{prefix.lower()}{k}.png"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--figures", nargs="+", default=list(FIGURES), choices=list(FIGURES),
                    help="paper figures to draw, default: all")
    args = ap.parse_args()

    ensure_output_dirs()

    mapping = load_industry_map()
    industries = dict(zip(mapping["ticker"], mapping["industry"]))

    loadings_by_panel: dict[str, pd.DataFrame] = {}
    frames = []
    for figure in args.figures:
        panel, factors, top_n = FIGURES[figure]
        if panel not in loadings_by_panel:
            loadings_by_panel[panel] = display_loadings(stock_panel(panel))
        loadings = loadings_by_panel[panel]
        prefix = PREFIXES[panel]

        shown = f"top {top_n}" if top_n else "all stocks"
        print(f"Figure {figure} — {panel} panel, {shown}")
        for k in factors:
            frame = sorted_loadings(loadings.loc[f"VF{k}"], industries, top_n)

            path = FIGURES_DIR / file_name(figure, prefix, k)
            plot_loadings(frame, path)

            counts = frame.reindex(frame["loading"].abs().sort_values(ascending=False).index)
            top3 = ", ".join(f"{n}×{c}" for n, c in
                             counts["industry"].head(TOP_N).value_counts().head(3).items())
            print(f"  {prefix}{k}: industries among the {TOP_N} largest |loadings| -> {top3}"
                  f"   loading range [{frame['loading'].min():.4f}, {frame['loading'].max():.4f}]")
            print(f"        -> {path.name}")

            frame.insert(0, "factor", f"{prefix}{k}")
            frame.insert(0, "figure", figure)
            frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    if "7" in args.figures:
        out = TABLES_DIR / "figure7_top_loadings.csv"
        table[table["figure"] == "7"].drop(columns="figure").to_csv(out, index=False)
        print(f"\n-> {out}")
    appendix = table[table["figure"] != "7"]
    if not appendix.empty:
        out = TABLES_DIR / "appendix_figure_loadings.csv"
        appendix.to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
