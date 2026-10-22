#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tables C.19–C.22 / D.23 — robustness of the continuous and jump volatility factors (Appendices C, D).

Both tables repeat the out-of-sample exercise of Table 14 (§6.1) with a different input panel; the HAR
benchmark, the rolling 5-year window and the Clark–West test are unchanged.

Appendix C — jump-test threshold τ
----------------------------------
The split of each stock's RV into a continuous and a jump component depends on the threshold τ of the jump
test. Table 14 uses τ = 5; Tables C.19–C.22 use τ = 4, 4.5, 5.5 and 6 (panels ``{component}_rv_tau{τ}.csv.gz``,
2004–2014, transformed with ``log(1 + x)`` as in Table 14).

Appendix D — ``log(CC)`` instead of ``log(1 + CC)``
---------------------------------------------------
Table D.23 extracts the continuous factors from ``log(CC)`` (τ = 5, zeros replaced by the smallest non-zero
value of the column) over the full 2004–2016 sample, so that the results are comparable with Table 4.

Output columns
--------------
``oos_r2`` (%), ``mspe_adj`` (mean of the Clark–West adjusted loss differential), ``cw_stat`` (its
t-statistic) and the two-sided ``cw_pvalue``. Tables C.19–C.22 report ``mspe_adj``; Table D.23 reports
``cw_stat``.

Usage
-----
    python 07_extensions/component_robustness.py                 # Appendices C and D
    python 07_extensions/component_robustness.py --part D        # Table D.23 only
    python 07_extensions/component_robustness.py --taus 4 --jobs 4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from component_factors import run_one
from volfactor.paths import TABLES_DIR, ensure_output_dirs

HORIZONS = (1, 5, 22)
TAUS = (4, 4.5, 5.5, 6)

#: Table id of each τ in Appendix C
TABLE_IDS = {4: "C.19", 4.5: "C.20", 5.5: "C.21", 6: "C.22"}

OUTPUTS = {
    "C": TABLES_DIR / "tableC19_C22_tau_robustness.csv",
    "D": TABLES_DIR / "tableD23_logcc.csv",
}


def tasks(parts: list[str], taus: list[float], horizons: list[int]) -> list[tuple]:
    """(part, tau, component, horizon) for every combination to run."""
    out = []
    if "C" in parts:
        out += [("C", tau, comp, h) for tau in taus for comp in ("continuous", "jump") for h in horizons]
    if "D" in parts:
        out += [("D", 5, "continuous", h) for h in horizons]
    return out


def run_task(task: tuple, window_years: int) -> tuple[tuple, pd.DataFrame, str]:
    """Run one combination; returns its rows and the printed output."""
    part, tau, comp, h = task
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if part == "C":
            # Same specification as Table 14: log(1 + x), panel and evaluation up to 2014
            frame = run_one(comp, h, window_years, end_year=-1, tau=tau)
        else:
            # log(CC): no shift, zeros replaced, full 2004–2016 sample
            frame = run_one(comp, h, window_years, end_year=None, eval_end=None,
                            tau=5, shift=0.0, replace_zeros=True)
    frame.insert(0, "tau", tau)
    frame.insert(0, "table", TABLE_IDS[tau] if part == "C" else "D.23")
    return task, frame, buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--part", nargs="+", default=["C", "D"], choices=["C", "D"],
                    help="appendix tables to reproduce, default: C D")
    ap.add_argument("--taus", type=float, nargs="+", default=list(TAUS), choices=list(TAUS),
                    help="jump-test thresholds for Appendix C, default 4 4.5 5.5 6")
    ap.add_argument("-H", "--horizons", type=int, nargs="+", default=list(HORIZONS),
                    choices=list(HORIZONS), help="forecast horizons in days, default 1 5 22")
    ap.add_argument("-w", "--window", type=int, default=5, help="estimation window in years, default 5")
    ap.add_argument("--jobs", type=int, default=1,
                    help="number of combinations run in parallel, default 1")
    ap.add_argument("--resume", action="store_true",
                    help="continue from existing results, skipping completed combinations")
    args = ap.parse_args()

    ensure_output_dirs()
    taus = [int(t) if float(t).is_integer() else t for t in args.taus]

    # Results are saved after each combination, so an interrupted run can be continued with --resume.
    frames = {part: [] for part in args.part}
    done = set()
    if args.resume:
        for part in args.part:
            if OUTPUTS[part].exists():
                prev = pd.read_csv(OUTPUTS[part])
                frames[part].append(prev)
                done |= {(part, t, c, h) for t, c, h in zip(prev["tau"], prev["component"], prev["horizon"])}
        print(f"Continuing: {len(done)} combinations already done")

    todo = [t for t in tasks(args.part, taus, args.horizons) if t not in done]
    print(f"{len(todo)} combinations to run, {args.jobs} at a time\n")

    def save(task, frame, text, seconds):
        part, tau, comp, h = task
        print(f"=== {part}: tau={tau:g} / {comp} / horizon={h}d ({seconds:.0f}s) " + "=" * 20)
        print(text, end="")
        frames[part].append(frame)
        pd.concat(frames[part], ignore_index=True).to_csv(OUTPUTS[part], index=False)
        sys.stdout.flush()

    t_all = time.time()
    if args.jobs <= 1:
        for task in todo:
            t0 = time.time()
            save(*run_task(task, args.window), time.time() - t0)
    else:
        # One BLAS thread per worker, so that the workers do not compete for cores
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_task, task, args.window): time.time() for task in todo}
            for fut in as_completed(futures):
                save(*fut.result(), time.time() - futures[fut])

    for part in args.part:
        if frames[part]:
            table = pd.concat(frames[part], ignore_index=True)
            table = table.sort_values(["tau", "component", "horizon"], kind="stable")
            table.to_csv(OUTPUTS[part], index=False)
            print(f"-> {OUTPUTS[part]}")
    print(f"\nTotal time {(time.time() - t_all) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
