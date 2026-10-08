#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 7 / Table 8 — model confidence set (MCS) tests (§4.3.1, Appendix).

Runs the MCS procedure of Hansen–Lunde–Nason (2011) for 5 loss functions × 3 horizons and
reports the MCS p-value of each model. Higher p-values mean the model is less likely to be excluded.

Three model sets
----------------
=========== ====================================================== ====
``--panel``  Models                                                  Count
=========== ====================================================== ====
``all``      HAR + VF1–4 + sVF1–4 + 5 combinations -> **Table 7**        14
``pca``      HAR + VF1–4 + 5 combinations -> **Table 8 Panel A**         10
``spca``     HAR + sVF1–4 + 5 combinations -> **Table 8 Panel B**        10
=========== ====================================================== ====

Table 8 compares PCA and sPCA separately so that the two sets of factors do not crowd each other out
(the "level-playing-field check" in the appendix).

MCS p-values come from a block bootstrap; the random seed
:data:`volfactor.evaluation.DEFAULT_SEED` is fixed so that results are reproducible.

Usage
-----
    python 05_robustness/mcs_test.py                      # all three panels
    python 05_robustness/mcs_test.py --panel all -H 1     # Table 7, 1-day horizon only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from volfactor.evaluation import DEFAULT_SEED, LOSS_FUNCTIONS, loss_matrix, run_mcs
from volfactor.paths import PREDICTIONS_DIR, TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
COMBINATIONS = ["MC", "MDC", "TMC", "DMSPE_0.9", "DMSPE_1"]

PANELS = {
    "all": ["HAR"] + [f"VF{k}" for k in range(1, 5)]
           + [f"sVF{k}" for k in range(1, 5)] + COMBINATIONS,
    "pca": ["HAR"] + [f"VF{k}" for k in range(1, 5)] + COMBINATIONS,
    "spca": ["HAR"] + [f"sVF{k}" for k in range(1, 5)] + COMBINATIONS,
}

PANEL_LABELS = {
    "all": "Table 7 (all 14 models)",
    "pca": "Table 8 Panel A (PCA only)",
    "spca": "Table 8 Panel B (sPCA only)",
}

#: MCS significance level. Only affects the included/excluded split, not the reported p-values.
MCS_SIZE = 0.10

#: Threshold above which values are set in bold in the paper
BOLD_THRESHOLD = 0.25

#: Last year of the evaluation sample for each loss: QLIKE uses the full sample (to 2016), the others end in 2014
EVAL_END_YEAR = {"MSE": 2014, "MAE": 2014, "QLIKE": None, "HMSE": 2014, "HMAE": 2014}


def run_panel(panel: str, horizons, window: int, seed: int, reps: int) -> pd.DataFrame:
    """MCS p-values for all (horizon, loss) pairs of one model set."""
    models = PANELS[panel]
    rows = []

    for h in horizons:
        path = PREDICTIONS_DIR / f"w{window}_h{h}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} is missing; run "
                f"`python 04_out_of_sample/build_predictions.py -w {window} -H {h}`"
            )
        preds = pd.read_csv(path, index_col=0, parse_dates=True)

        for loss_name in LOSS_FUNCTIONS:
            end = EVAL_END_YEAR[loss_name]
            sub = preds if end is None else preds[preds.index.year <= end]
            losses = loss_matrix(sub["actual"], sub[models], loss_name)
            pvalues = run_mcs(losses, size=MCS_SIZE, reps=reps, seed=seed)

            for model in models:
                rows.append({
                    "panel": panel,
                    "horizon": h,
                    "loss": loss_name,
                    "model": model,
                    "mcs_pvalue": float(pvalues[model]),
                })
            print(f"    h={h:2d}d  {loss_name:<6s} done")

    return pd.DataFrame(rows)


def print_panel(result: pd.DataFrame, panel: str) -> None:
    """Print in the paper's layout: rows = losses, columns = models, one block per horizon."""
    models = PANELS[panel]
    print(f"\n  {PANEL_LABELS[panel]}   (* = p > {BOLD_THRESHOLD}, bold in the paper)")

    for h in sorted(result.horizon.unique()):
        print(f"\n  --- {h} day horizon " + "-" * 52)
        print(f"  {'Loss':<7s}" + "".join(f"{m:>11s}" for m in models))
        for loss_name in LOSS_FUNCTIONS:
            sub = result[(result.horizon == h) & (result.loss == loss_name)]
            cells = []
            for m in models:
                p = sub[sub.model == m]["mcs_pvalue"].iloc[0]
                cells.append(f"{p:.3f}*" if p > BOLD_THRESHOLD else f"{p:.3f} ")
            print(f"  {loss_name:<7s}" + "".join(f"{c:>11s}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel", choices=list(PANELS) + ["every"], default="every",
                    help="model set, default: all three")
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=list(HORIZONS),
                    choices=list(HORIZONS), help="forecast horizons, default 1 5 22")
    ap.add_argument("-w", "--window", type=int, default=5, help="estimation window in years, default 5")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"block bootstrap random seed, default {DEFAULT_SEED}")
    ap.add_argument("--reps", type=int, default=1000, help="number of bootstrap replications, default 1000")
    args = ap.parse_args()

    ensure_output_dirs()
    panels = list(PANELS) if args.panel == "every" else [args.panel]
    out = []

    for panel in panels:
        print(f"\n{'=' * 70}\n{PANEL_LABELS[panel]}  "
              f"[seed={args.seed}, reps={args.reps}]\n{'=' * 70}")
        result = run_panel(panel, args.horizons, args.window, args.seed, args.reps)
        print_panel(result, panel)
        out.append(result)

    path = TABLES_DIR / "table7_table8_mcs.csv"
    pd.concat(out, ignore_index=True).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
