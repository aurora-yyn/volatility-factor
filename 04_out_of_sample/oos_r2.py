#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 4 / Table 9 — out-of-sample R² and Clark–West tests (§4.2.1, §4.3.2).

Reads the rolling forecast matrices produced by ``build_predictions.py`` in ``outputs/predictions/``
and computes, for each of the 14 models, the out-of-sample R² relative to HAR, the MSPE-adjusted statistic and the p-value.

- ``--windows 5``       -> Table 4 (main specification, 5-year window)
- ``--windows 4 6``     -> Table 9 (window robustness, §4.3.2)

Usage
-----
    python 04_out_of_sample/oos_r2.py                 # all three windows
    python 04_out_of_sample/oos_r2.py --windows 5     # Table 4 only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from volfactor.evaluation import evaluate_models
from volfactor.paths import PREDICTIONS_DIR, TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)

#: Row order of the paper's tables
MODEL_ORDER = [
    "HAR",
    "VF1", "VF2", "VF3", "VF4",
    "sVF1", "sVF2", "sVF3", "sVF4",
    "MC", "MDC", "TMC", "DMSPE_0.9", "DMSPE_1",
]


def evaluate_window(window_years: int) -> pd.DataFrame:
    """Evaluation of all models at the three horizons for one estimation window."""
    frames = []
    for h in HORIZONS:
        path = PREDICTIONS_DIR / f"w{window_years}_h{h}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} is missing; run "
                f"`python 04_out_of_sample/build_predictions.py -w {window_years} -H {h}`"
            )
        preds = pd.read_csv(path, index_col=0, parse_dates=True)
        metrics = evaluate_models(preds["actual"], preds[MODEL_ORDER], "HAR")

        metrics = metrics.loc[MODEL_ORDER].reset_index(names="model")
        metrics.insert(0, "horizon", h)
        metrics.insert(0, "window_years", window_years)
        metrics["n_obs"] = len(preds)
        frames.append(metrics)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-w", "--windows", type=int, nargs="+", default=[4, 5, 6],
                    help="estimation window length in years, default 4 5 6")
    args = ap.parse_args()

    ensure_output_dirs()
    out = []

    for w in args.windows:
        table = evaluate_window(w)
        out.append(table)

        label = "Table 4 (main)" if w == 5 else "Table 9 (robustness)"
        print(f"\n=== window = {w} years — {label} " + "=" * 24)
        print(f"  {'Model':<12s}" + "".join(f"{f'{h}-day':>14s}" for h in HORIZONS))
        for name in MODEL_ORDER[1:]:
            cells = []
            for h in HORIZONS:
                r = table[(table.horizon == h) & (table.model == name)].iloc[0]
                cells.append(f"{r['OOS_R2']:.3f}{r['stars']}")
            print(f"  {name:<12s}" + "".join(f"{c:>14s}" for c in cells))

    path = TABLES_DIR / "table4_table9_oos_r2.csv"
    pd.concat(out, ignore_index=True).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
