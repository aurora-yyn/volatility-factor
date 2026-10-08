#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 11, Figure 8 — incremental generalized correlation analysis (§5.1).

The loading analysis asks which stocks carry the largest weights in a factor. This section takes another angle:
how strongly does each volatility factor co-move with the **overall volatility of each industry**? The two answers
can differ — VF1 is constructed mainly from financial stocks, yet co-moves strongly with all industries.

Method (Bai & Ng, 2006)
-----------------------
1. Within each SIC industry, PCA on the stock LVs; the **first principal component** is the industry factor (14 in total).
   Note the difference from Table 12, where the industry factors are equal-weighted averages.
2. For each volatility factor F_k, industries are ranked by |Pearson correlation| with F_k.
3. The top m industry factors are added one at a time; the generalized correlation is ``GC_{k,m} = sqrt(R²_{k,m})``.

The GC of a single industry is ``|corr(F_k, I_j)|`` (sqrt(R²) of a univariate regression is the absolute correlation).

Usage
-----
    python 06_information_content/incremental_correlation.py
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

from volfactor.factors import extract_pca
from volfactor.io import load_industry_map, load_panel, stock_columns
from volfactor.paths import FIGURES_DIR, TABLES_DIR, ensure_output_dirs

N_FACTORS = 4

#: Short industry names used in the table
SHORT = {
    "Oil and gas": "Oil and gas",
    "Primary manufacturing": "Primary mfg.",
    "Pharma & Chemicals": "Pharma & Chem.",
}


def industry_factors(panel: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """One column per industry: the first principal component of the stock LVs within the industry."""
    present = set(panel.columns)
    out = {}

    for industry, group in mapping.groupby("industry"):
        cols = [t for t in group["ticker"] if t in present]
        if not cols:
            continue
        sub = panel[cols]
        factor, _ = extract_pca(sub, sub, n_factors=1, prefix="I")
        out[industry] = factor["I1"]

    return pd.DataFrame(out, index=panel.index)


def incremental_gc(target: pd.Series, factors: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Add industries in order of |correlation|; return (ranked industries, GC at each step)."""
    order = (
        factors.corrwith(target).abs().sort_values(ascending=False).index.tolist()
    )

    y = target.to_numpy()
    y_centered = y - y.mean()
    tss = float((y_centered ** 2).sum())

    gcs = []
    for m in range(1, len(order) + 1):
        design = np.column_stack(
            [np.ones(len(factors)), factors[order[:m]].to_numpy()]
        )
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        rss = float(((y - design @ coef) ** 2).sum())
        gcs.append(np.sqrt(max(0.0, 1.0 - rss / tss)))

    return order, np.array(gcs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args()

    ensure_output_dirs()

    X, _ = load_panel(horizon=1)
    panel = X[stock_columns(X)]
    vf, _ = extract_pca(panel, panel, n_factors=N_FACTORS)

    inds = industry_factors(panel, load_industry_map())
    print(f"Built {inds.shape[1]} industry factors (first principal component within each industry)\n")

    rows, curves = [], {}
    for k in range(1, N_FACTORS + 1):
        name = f"VF{k}"
        order, gcs = incremental_gc(vf[name], inds)
        curves[name] = gcs

        top3 = [(SHORT.get(order[i], order[i]), gcs[i] if i == 0 else None)
                for i in range(3)]
        # GC of a single industry = |corr|, computed one by one
        singles = inds.corrwith(vf[name]).abs().sort_values(ascending=False)

        rows.append({
            "factor": name,
            "top1_industry": SHORT.get(order[0], order[0]),
            "top1_gc": singles.iloc[0],
            "top2_industry": SHORT.get(order[1], order[1]),
            "top2_gc": singles.iloc[1],
            "top3_industry": SHORT.get(order[2], order[2]),
            "top3_gc": singles.iloc[2],
            "all14_gc": gcs[-1],
            "improvement": gcs[-1] - singles.iloc[0],
        })

        print(f"  {name}: " + "  ".join(
            f"{SHORT.get(order[i], order[i])}={singles.iloc[i]:.3f}" for i in range(3)
        ) + f"   All-14={gcs[-1]:.3f}  (+{gcs[-1] - singles.iloc[0]:.3f})")

        # Rank of Finance (the VF1 case discussed in the text)
        if name == "VF1":
            rank = list(singles.index).index("Finance") + 1
            print(f"        Finance ranks {rank}, GC = {singles['Finance']:.3f}; "
                  f"lowest GC among the 14 industries = {singles.min():.3f}")

    table = pd.DataFrame(rows)
    path = TABLES_DIR / "incremental_gc_summary.csv"
    table.to_csv(path, index=False)

    # Incremental GC curves
    fig, ax = plt.subplots(figsize=(8, 5))
    m = np.arange(1, inds.shape[1] + 1)
    for name, gcs in curves.items():
        ax.plot(m, gcs, marker="o", label=name)

    ax.set_xlabel("Number of industry factors included ($m$)")
    ax.set_ylabel("Generalized correlation $GC_{k,m}$")
    ax.set_xticks(m)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(title="Volatility factor")

    fig_path = FIGURES_DIR / "incremental_generalized_correlation.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n-> {path}\n-> {fig_path}")


if __name__ == "__main__":
    main()
