# -*- coding: utf-8 -*-
"""Table 6 — multi-factor combinations (§4.2.3).

Setup (as in Table 4):
  - Input: LV = log(RV²) of the 332 constituents
  - Fixed HAR regressors: RV_1Day, RV_5Day, RV_22Day
  - Factors: PCA on the non-HAR columns of X_train in every rolling window, without standardization
    (covariance matrix); the first 4 principal components are VF1–VF4
  - Rolling window 252*5 = 1260 (5 years), step = 1, one-step-ahead forecasts
  - OOS R²: Campbell–Thompson (2008), relative to plain HAR

Usage:
    python 04_out_of_sample/multi_factor_combinations.py            # horizons 1/5/22
    python 04_out_of_sample/multi_factor_combinations.py --horizon 1
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volfactor.evaluation import clark_west, oos_r2, significance_stars
from volfactor.factors import extract_pca
from volfactor.io import HAR_COLS, load_panel, stock_columns
from volfactor.paths import TABLES_DIR, ensure_output_dirs

FIXED_COLUMNS = HAR_COLS


class MultiFactorHARPCA:
    """Multi-factor HAR combination models (standard PCA factors)."""

    def __init__(self, horizon, window_size=252 * 5, n_factors=4):
        self.horizon = horizon
        self.window_size = window_size
        self.n_factors = n_factors
        self.fixed_columns = FIXED_COLUMNS
        self._load_data()

    def _load_data(self):
        self.X, Y = load_panel(horizon=self.horizon)
        self.Y = Y.to_frame()
        self.other_columns = stock_columns(self.X)

    # ---- Factor extraction: standard PCA (as in Table 4) ----
    def extract_pca_factors(self, X_train, X_full):
        """PCA on the non-HAR columns of X_train; returns the first n_factors components on the full sample.

        Calls :func:`volfactor.factors.extract_pca`, the same implementation as Table 4
        (including ``svd_solver="full"``).
        """
        cols = self.other_columns
        _, factors_df = extract_pca(X_train[cols], X_full[cols], self.n_factors)
        return factors_df, None

    def fit_har_model(self, X_train, Y_train, factor_columns=None):
        cols = self.fixed_columns + (factor_columns or [])
        model = sm.OLS(Y_train, sm.add_constant(X_train[cols])).fit()
        return model, cols

    def fit_lasso_selection(self, X_train, Y_train, factor_columns):
        all_cols = self.fixed_columns + factor_columns
        Xc = sm.add_constant(X_train[all_cols])
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(Xc, Y_train.values.ravel())
        names = ["const"] + all_cols
        mask = np.abs(lasso.coef_) > 1e-5
        selected = [n for n, m in zip(names, mask) if m and n != "const"]
        return [f for f in selected if f in factor_columns]

    def generate_all_factor_combinations(self):
        fl = [f"VF{i}" for i in range(1, self.n_factors + 1)]
        combos = {f"HAR_{f}": [f] for f in fl}
        for r in (2, 3):
            for c in combinations(fl, r):
                combos["HAR_" + "_".join(c)] = list(c)
        combos["HAR_all_factors"] = fl
        return combos

    def rolling_forecast(self, step=1):
        all_combos = self.generate_all_factor_combinations()
        res = {"dates": [], "actual": [], "HAR_benchmark": []}
        for name in all_combos:
            res[name] = []
        res["HAR_LASSO_selected"] = []
        res["BIC_best_model"] = []

        n = len(self.X)
        for i in range(0, n - self.window_size + 1, step):
            X_train = self.X.iloc[i:i + self.window_size]
            Y_train = self.Y.iloc[i:i + self.window_size]
            X_test = self.X.iloc[i + self.window_size:i + self.window_size + 1]
            Y_test = self.Y.iloc[i + self.window_size:i + self.window_size + 1]
            if X_test.empty or Y_test.empty:
                continue

            factors_df, _ = self.extract_pca_factors(X_train, self.X)
            f_train = factors_df.loc[X_train.index]
            f_test = factors_df.loc[X_test.index]
            Xtr = X_train[self.fixed_columns].join(f_train)
            Xte = X_test[self.fixed_columns].join(f_test)

            res["dates"].append(X_test.index.strftime("%Y-%m-%d")[0])
            res["actual"].append(Y_test.values[0, 0])

            # HAR benchmark
            m_har, _ = self.fit_har_model(Xtr, Y_train, None)
            Xte_har = sm.add_constant(Xte[self.fixed_columns], has_constant="add")
            res["HAR_benchmark"].append(m_har.predict(Xte_har)[0])
            bic_scores = {"HAR": m_har.bic}
            bic_preds = {"HAR": res["HAR_benchmark"][-1]}

            # Fixed combinations
            for name, fcols in all_combos.items():
                m, cols = self.fit_har_model(Xtr, Y_train, fcols)
                Xte_c = sm.add_constant(Xte[cols], has_constant="add")
                pred = m.predict(Xte_c)[0]
                res[name].append(pred)
                bic_scores[name] = m.bic
                bic_preds[name] = pred

            # BIC-selected model
            best = min(bic_scores, key=bic_scores.get)
            res["BIC_best_model"].append(bic_preds[best])

            # LASSO selection
            sel = self.fit_lasso_selection(Xtr, Y_train, [f"VF{i}" for i in range(1, self.n_factors + 1)])
            if sel:
                m, cols = self.fit_har_model(Xtr, Y_train, sel)
                Xte_c = sm.add_constant(Xte[cols], has_constant="add")
                res["HAR_LASSO_selected"].append(m.predict(Xte_c)[0])
            else:
                res["HAR_LASSO_selected"].append(res["HAR_benchmark"][-1])

        return res

    def compute_metrics(self, res):
        actual = np.asarray(res["actual"])
        bench = np.asarray(res["HAR_benchmark"])
        rows = []
        for name, preds in res.items():
            if name in ("dates", "actual", "HAR_benchmark"):
                continue
            preds = np.asarray(preds)
            ok = ~(np.isnan(preds) | np.isnan(bench))
            a, p, b = actual[ok], preds[ok], bench[ok]
            r2 = oos_r2(a, p, b)
            # Table 6 stars use one-sided p-values
            cw, pval = clark_west(a, p, b, two_sided=False)
            rows.append({
                "Model": name,
                "OOS_R2": r2,
                "OOS_R2_pct": round(r2 * 100, 3),
                "CW_stat": round(cw, 3),
                "CW_pvalue": round(pval, 4),
                "signif": significance_stars(pval),
                "R2_with_stars": f"{r2 * 100:.2f}{significance_stars(pval)}",
            })
        return pd.DataFrame(rows).set_index("Model")


def run_horizon(horizon, window_days=252 * 5):
    if horizon not in (1, 5, 22):
        raise ValueError("horizon must be 1, 5, or 22")

    print(f"\n{'=' * 70}\n[PCA] horizon={horizon}day  window={window_days}({window_days // 252}y)\n{'=' * 70}")
    model = MultiFactorHARPCA(horizon, window_size=window_days)
    print(f"  X={model.X.shape}, factors from {len(model.other_columns)} cols")
    res = model.rolling_forecast(step=1)
    print(f"  {len(res['dates'])} predictions")
    metrics = model.compute_metrics(res)
    ensure_output_dirs()
    out = TABLES_DIR / f"multi_factor_combinations_PCA_{horizon}day.csv"
    metrics.to_csv(out)
    print(f"  saved -> {out}")
    # Self-check: print VF1
    if "HAR_VF1" in metrics.index:
        print(f"  [self-check] HAR_VF1 OOS R2 = {metrics.loc['HAR_VF1', 'OOS_R2_pct']}%  (Table4 VF1: 1d=1.500,5d=2.505,22d=4.762)")
    return metrics


#: Row order and labels of Table 6 (the per-horizon CSVs also contain single-factor rows HAR_VF2–VF4, not shown in the paper)
PAPER_ROWS = {
    "HAR_VF1": "HAR-VF1",
    "HAR_VF1_VF2": "HAR-VF1+VF2", "HAR_VF1_VF3": "HAR-VF1+VF3", "HAR_VF1_VF4": "HAR-VF1+VF4",
    "HAR_VF2_VF3": "HAR-VF2+VF3", "HAR_VF2_VF4": "HAR-VF2+VF4", "HAR_VF3_VF4": "HAR-VF3+VF4",
    "HAR_VF1_VF2_VF3": "HAR-VF1+VF2+VF3", "HAR_VF1_VF2_VF4": "HAR-VF1+VF2+VF4",
    "HAR_VF1_VF3_VF4": "HAR-VF1+VF3+VF4", "HAR_VF2_VF3_VF4": "HAR-VF2+VF3+VF4",
    "HAR_all_factors": "HAR-all-factors",
    "BIC_best_model": "BIC-selected", "HAR_LASSO_selected": "LASSO-selected",
}


def build_paper_table():
    """Assemble the three per-horizon results into the layout of Table 6 (R² in % + stars)."""
    cols = {}
    for h in (1, 5, 22):
        path = TABLES_DIR / f"multi_factor_combinations_PCA_{h}day.csv"
        if not path.exists():
            print(f"  {path.name} is missing; skipping the summary table")
            return
        cols[f"{h}-day"] = pd.read_csv(path, index_col="Model")["R2_with_stars"]
    table = pd.DataFrame(cols).reindex(list(PAPER_ROWS)).rename(index=PAPER_ROWS)
    table.index.name = "Model"
    out = TABLES_DIR / "table6_multi_factor_combinations.csv"
    table.to_csv(out)
    print(f"\n{table.to_string()}\n\n-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, choices=[1, 5, 22], default=None)
    args = ap.parse_args()
    horizons = [args.horizon] if args.horizon else [1, 5, 22]
    for h in horizons:
        run_horizon(h)
    build_paper_table()


if __name__ == "__main__":
    main()
