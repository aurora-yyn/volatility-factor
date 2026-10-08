#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run all scripts in order to reproduce every table and figure of the paper.

Usage
-----
    python run_all.py              # run everything (about 2–3 hours)
    python run_all.py --list       # list all steps
    python run_all.py --from 12    # start from step 12

The output of each step is written to ``outputs/logs/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "outputs" / "logs"

#: (script, arguments, what it reproduces)
STEPS = [
    ("04_out_of_sample/build_predictions.py", [], "Rolling forecasts (input of Tables 4, 5, 7–10, 16 and Figures 5, 6)"),
    ("02_data_description/basic_statistics.py", [], "Table 1"),
    ("02_data_description/long_memory_test.py", [], "Table 2"),
    ("02_data_description/acf_pacf_plot.py", [], "Figure 1"),
    ("02_data_description/factor_timeseries_plot.py", [], "Figure 2"),
    ("02_data_description/adf_test.py", [], "Tables A.17, A.18"),
    ("01_factor_construction/eigenvalue_ratio.py", [], "Figure 4"),
    ("01_factor_construction/eigenvalue_ratio.py", ["--gamma", "0.30"], "Figure H.17"),
    ("03_in_sample_results/in_sample_regression.py", [], "Table 3"),
    ("04_out_of_sample/oos_r2.py", [], "Tables 4, 9"),
    ("04_out_of_sample/regime_analysis.py", [], "Table 5, Figure 6"),
    ("04_out_of_sample/multi_factor_combinations.py", [], "Table 6"),
    ("04_out_of_sample/time_variation_r2.py", [], "Figure 5"),
    ("05_robustness/mcs_test.py", [], "Tables 7, 8"),
    ("05_robustness/correlation_matrix_factors.py", [], "Table 10"),
    ("06_information_content/incremental_correlation.py", [], "Table 11, Figure 8"),
    ("06_information_content/industry_factors.py", [], "Table 12"),
    ("06_information_content/incremental_predictability.py", [], "Table 13"),
    ("06_information_content/factor_loadings_plot.py", [], "Figure 7"),
    ("06_information_content/standalone_predictors.py", [], "Table G.27"),
    ("07_extensions/component_factors.py", [], "Tables 14, 15"),
    ("07_extensions/partial_variance_factors.py", [], "Table 16"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1, help="step to start from, default 1")
    ap.add_argument("--list", action="store_true", help="list the steps without running them")
    args = ap.parse_args()

    if args.list:
        for i, (script, extra, what) in enumerate(STEPS, 1):
            print(f"{i:>2d}. {' '.join([script] + extra):<58s} {what}")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for i, (script, extra, what) in enumerate(STEPS, 1):
        if i < args.start:
            continue
        cmd = [sys.executable, script] + extra
        log = LOG_DIR / f"step{i:02d}_{Path(script).stem}{'_' + '_'.join(extra) if extra else ''}.log"
        print(f"[{i:>2d}/{len(STEPS)}] {what:<40s}", end=" ", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            ret = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"{time.time() - t0:>6.0f}s")
        if ret != 0:
            print(f"\nStep {i} failed; see {log}\nAfter fixing it, continue with `python run_all.py --from {i}`.")
            sys.exit(ret)
    print(f"\nAll done in {(time.time() - t_all) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
