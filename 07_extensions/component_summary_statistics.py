#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tables F.24–F.26 — summary statistics of the component volatility factors (Appendix F).

Full-sample PCA and sPCA factors are extracted from each component panel of §6 and described by their mean,
standard deviation, quantiles, skewness and (excess) kurtosis.

=========== ====================================== ========================= =====================
Table        Factors                                 Panel                      Sample
=========== ====================================== ========================= =====================
F.24         JVF / JsVF, τ = 4, 4.5, 5, 5.5, 6        jump, ``log(1 + x)``        2004–2014
F.25         CVF / CsVF, τ = 4, 4.5, 5, 5.5, 6        continuous, ``log(1 + x)``  2004–2014
F.26         GVF / GsVF, BVF / BsVF                   good / bad, ``log(x)``      2004–2014
=========== ====================================== ========================= =====================

Specification
-------------
- The panels use the same transformations as the forecasting exercise of §6 (Tables 14, 15).
- PCA factors are extracted from all dates of the panel. sPCA factors use the dates on which the
  one-day-ahead target is available, with the residual scaling of §2.2.2.
- The sign of a principal component is arbitrary, and it determines the sign of the minimum, maximum, quantiles
  and skewness. Each factor is oriented so that its largest absolute value is positive.
- Skewness and kurtosis are the moment estimators of ``scipy.stats`` (excess kurtosis).

Usage
-----
    python 07_extensions/component_summary_statistics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from volfactor.factors import N_FACTORS, extract_pca, extract_spca_residual
from volfactor.io import TAUS, load_component_panel, read_component_panel
from volfactor.paths import TABLES_DIR, ensure_output_dirs

END_YEAR = 2014

#: Component -> (table id, PCA factor prefix, sPCA factor prefix)
SPECS = {
    "jump": ("F.24", "JVF", "JsVF"),
    "continuous": ("F.25", "CVF", "CsVF"),
    "good": ("F.26", "GVF", "GsVF"),
    "bad": ("F.26", "BVF", "BsVF"),
}

STATISTICS = ["mean", "std", "min", "25%", "50%", "75%", "max", "skew", "kurtosis"]


def orient(factors: pd.DataFrame) -> pd.DataFrame:
    """Flip each factor so that its largest absolute value is positive."""
    arr = factors.to_numpy()
    signs = np.sign(arr[np.abs(arr).argmax(axis=0), np.arange(arr.shape[1])])
    return factors * signs


def describe(factors: pd.DataFrame) -> pd.DataFrame:
    """Rows of the paper's tables: describe() + skewness + excess kurtosis.

    PCA and sPCA factors cover different dates, so each column is described over its own non-missing values.
    """
    stats = factors.describe().drop(index="count")
    stats.loc["skew"] = factors.apply(lambda s: skew(s.dropna()))
    stats.loc["kurtosis"] = factors.apply(lambda s: kurtosis(s.dropna()))
    return stats.loc[STATISTICS]


def component_factors(component: str, tau: float) -> pd.DataFrame:
    """Full-sample PCA and sPCA factors of one component panel, oriented by :func:`orient`."""
    _, pca_prefix, spca_prefix = SPECS[component]

    panel = np.log(read_component_panel(component, tau=tau, end_year=END_YEAR))
    vf, _ = extract_pca(panel, panel, N_FACTORS, prefix=pca_prefix, align_signs=False)

    X, Y = load_component_panel(component, horizon=1, end_year=END_YEAR, tau=tau)
    svf, _ = extract_spca_residual(X, Y, X, n_factors=N_FACTORS, prefix=spca_prefix,
                                   align_signs=False)

    return pd.concat([orient(vf), orient(svf)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args()

    ensure_output_dirs()

    runs = [(c, tau) for c in ("jump", "continuous") for tau in TAUS]
    runs += [("good", 5), ("bad", 5)]

    rows = []
    for component, tau in runs:
        table = SPECS[component][0]
        stats = describe(component_factors(component, tau))

        label = f"tau={tau:g}" if component in ("jump", "continuous") else ""
        print(f"\n=== Table {table}: {component} {label}")
        print(stats.round(3).to_string())

        for stat, values in stats.iterrows():
            for factor, value in values.items():
                rows.append({"table": table, "component": component,
                             "tau": tau if component in ("jump", "continuous") else np.nan,
                             "statistic": stat, "factor": factor, "value": value})

    path = TABLES_DIR / "tableF24_F26_summary_statistics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
