#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 2 — long-memory tests (§3.2.2).

Three estimators are computed for market LV and for the 4 PCA / 4 sPCA factors:

- GPH estimate of d (Geweke & Porter-Hudak, 1983)
- Local Whittle estimate of d (Robinson, 1995)
- Hurst exponent from R/S analysis

The sample ends in 2014; factors are extracted on the full sample.

Usage
-----
    python 02_data_description/long_memory_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from volfactor.io import HAR_COLS, load_panel
from volfactor.paths import TABLES_DIR


# ============================================================
# Parameters and outputs
# ============================================================

OUTPUT_TABLE = TABLES_DIR / "long_memory_test_results.csv"
# Same columns as Table 2 of the paper: 3 decimals + stars
OUTPUT_PAPER_CSV = TABLES_DIR / "long_memory_table2_paper_format.csv"
# LaTeX table rows
OUTPUT_LATEX_ROWS = TABLES_DIR / "long_memory_table2_rows.tex"

# Sample range (≤2014, the in-sample period of the paper)
SAMPLE_END_YEAR = 2014

# Number of factors
N_FACTORS = 4


# ============================================================
# 1. Data: read CSV -> extract PCA / sPCA factors
# ============================================================

def prepare_data():
    """Read RV, transform to log(RV²), and extract 4 PCA and 4 sPCA factors.

    Returns a DataFrame with 9 columns:
        LV        — market log realized variance
        VF1-VF4   — the 4 PCA factors
        sVF1-sVF4 — the 4 sPCA factors
    """
    # Load data (already on the LV = log(RV²) scale) and truncate at the last sample year
    X, Y = load_panel(horizon=1)
    keep = X.index.year <= SAMPLE_END_YEAR
    X, Y = X[keep], Y[keep].to_frame()

    # The three fixed HAR regressors
    fixed_cols = HAR_COLS
    other_cols = [c for c in X.columns if c not in fixed_cols]
    X_feature = X[other_cols]

    # ---------- PCA: directly on the LV of the 332 stocks ----------
    pca = PCA(n_components=N_FACTORS, random_state=100).fit(X_feature)
    pc_df = pd.DataFrame(
        pca.transform(X_feature),
        index=X_feature.index,
        columns=[f"VF{i+1}" for i in range(N_FACTORS)],
    )

    # ---------- sPCA: β for each stock from the HAR residuals, then PCA on the scaled panel ----------
    # Step 1: regress Y on RV_1Day / 5Day / 22Day and keep the residuals
    ols_fixed = LinearRegression().fit(X[fixed_cols], Y)
    residuals = Y.values.flatten() - ols_fixed.predict(X[fixed_cols]).flatten()

    # Step 2: regress the residuals on each stock separately to obtain β
    scaled = {}
    for col in other_cols:
        beta = LinearRegression().fit(X[[col]], residuals).coef_[0]
        scaled[col] = X[col] * beta
    scaled_df = pd.DataFrame(scaled, index=X.index)

    # Step 3: PCA on the scaled matrix
    spca = PCA(n_components=N_FACTORS, random_state=100).fit(scaled_df)
    spc_df = pd.DataFrame(
        spca.transform(scaled_df),
        index=scaled_df.index,
        columns=[f"sVF{i+1}" for i in range(N_FACTORS)],
    )

    # Combine: LV (RV_1Day) + 4 VF + 4 sVF
    merged = pd.concat(
        [X[["RV_1Day"]].rename(columns={"RV_1Day": "LV"}), pc_df, spc_df],
        axis=1,
    )
    return merged


# ============================================================
# 2. GPH estimator (Geweke & Porter-Hudak, 1983)
# ============================================================
#
# Idea: for a long-memory process the spectral density near frequency zero is approximately
#       f(λ) ≈ C * λ^(-2d),  λ → 0
# Taking logs gives
#       log I(λ_j) ≈ c - d * log(4 sin²(λ_j/2)) + u_j
# so an OLS regression of the log periodogram at low frequencies has slope ≈ -d
# (equivalently, with X = -log(4 sin²(λ/2)), slope = d).

def gph_estimator(series, m=None):
    """GPH estimate of d.

    Parameters
    ----------
    series : 1D array, time series
    m : number of low frequencies used (default sqrt(T), as in GPH)

    Returns
    -------
    d_hat : OLS estimate of d
    se    : standard error of d_hat
    p_val : two-sided p-value for H0: d=0
    """
    series = np.asarray(series, dtype=float)
    T = len(series)
    if m is None:
        m = int(T ** 0.5)            # bandwidth recommended by GPH

    # Demean (otherwise the DC component is the squared mean of the series)
    y = series - series.mean()

    # Fourier frequencies: λ_j = 2π j / T,  j = 1, ..., m
    freqs = 2 * np.pi * np.arange(1, m + 1) / T

    # Periodogram from FFT terms 1..m (skipping the DC term 0), aligned with freqs
    fft_vals = np.fft.fft(y)[1:m + 1]
    periodogram = np.abs(fft_vals) ** 2 / (2 * np.pi * T)

    # GPH regression: log I = c - d * log(4 sin²(λ/2)) + u
    # With X = -log(4 sin²(λ/2)), the OLS slope is d
    X_reg = -np.log(4 * np.sin(freqs / 2) ** 2).reshape(-1, 1)
    y_reg = np.log(periodogram)

    reg = LinearRegression().fit(X_reg, y_reg)
    d_hat = reg.coef_[0]

    # OLS standard error
    resid = y_reg - reg.predict(X_reg)
    sigma2 = (resid ** 2).sum() / (m - 2)
    se = np.sqrt(sigma2 / ((X_reg - X_reg.mean()) ** 2).sum())

    # Two-sided p-value
    t_stat = d_hat / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=m - 2))

    return d_hat, se, p_val


# ============================================================
# 3. Local Whittle estimator (Robinson, 1995)
# ============================================================
#
# Idea: at low frequencies the spectral density is f(λ) ≈ G * (4 sin²(λ/2))^(-d).
# Plugging this into the Whittle likelihood and profiling out G
# gives the objective function in d:
#       R(d) = log( (1/m) Σ G(λ_j)^d * I(λ_j) )  -  d * (1/m) Σ log G(λ_j)
# where G(λ) = 4 sin²(λ/2). Minimizing R(d) gives the estimate of d.

def whittle_estimator(series, m=None):
    """Local Whittle estimate of d.

    Parameters
    ----------
    series : 1D array
    m : number of frequencies (default T^0.65, Robinson 1995)

    Returns
    -------
    d_hat : Whittle estimate
    se    : asymptotic standard error
    """
    series = np.asarray(series, dtype=float)
    T = len(series)
    if m is None:
        m = int(T ** 0.65)

    y = series - series.mean()
    freqs = 2 * np.pi * np.arange(1, m + 1) / T

    # Periodogram from FFT terms 1..m (skipping the DC term)
    fft_vals = np.fft.fft(y)[1:m + 1]
    periodogram = np.abs(fft_vals) ** 2 / (2 * np.pi * T)

    G = 4 * np.sin(freqs / 2) ** 2
    log_G = np.log(G)

    # Profiled objective function of Robinson (1995)
    def objective(d):
        return np.log(np.mean(periodogram * G ** d)) - d * log_G.mean()

    # Search interval (-0.5, 0.99), allowing strong long memory with d > 0.5
    result = minimize_scalar(objective, bounds=(-0.5, 0.99), method="bounded")
    d_hat = result.x

    # Asymptotic standard error (Robinson 1995): 1 / sqrt(4m)
    se = 1.0 / np.sqrt(4 * m)
    return d_hat, se


# ============================================================
# 4. Hurst exponent (R/S analysis)
# ============================================================
#
# R/S analysis estimates Hurst directly, without d. In theory H = d + 0.5.

def hurst_rs(series):
    """Hurst exponent from R/S analysis."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    lags = range(2, min(100, n // 2))

    rs_means = []
    valid_lags = []

    for lag in lags:
        # Split the series into segments of length lag
        n_seg = n // lag
        rs_vals = []
        for i in range(n_seg):
            seg = series[i * lag:(i + 1) * lag]
            # Range of cumulative deviations R
            cum = np.cumsum(seg - seg.mean())
            R = cum.max() - cum.min()
            # Within-segment standard deviation S
            S = seg.std(ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs_means.append(np.mean(rs_vals))
            valid_lags.append(lag)

    # Linear regression in log-log coordinates; slope = Hurst
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_means)
    H = np.polyfit(log_lags, log_rs, 1)[0]
    return H


# ============================================================
# 5. Main routine: all 9 variables -> result table
# ============================================================

def format_p_value(p):
    """Format a p-value as a string with stars."""
    if p < 0.01:
        stars = "***"
    elif p < 0.05:
        stars = "**"
    elif p < 0.10:
        stars = "*"
    else:
        stars = ""
    return f"{p:.3f}{stars}"


# ============================================================
# 6. Paper-format output: CSV (as in Table 2) + LaTeX table rows
# ============================================================

def split_pvalue_and_stars(p_str):
    """Split '0.000***' into ('0.000', '***')."""
    for stars in ("***", "**", "*"):
        if p_str.endswith(stars):
            return p_str[:-len(stars)], stars
    return p_str, ""


def export_paper_csv(result, output_path):
    """Write the compact CSV: same columns as Table 2, 3 decimals, p-values with stars.

    Columns of Table 2: Variable, d_GPH, se_GPH, p_value, d_Whittle, Hurst
    (without se_Whittle and without the unrounded values)
    """
    paper = pd.DataFrame({
        "Variable":  result["Variable"],
        "d_GPH":     result["d_GPH"].map(lambda x: f"{x:.3f}"),
        "se_GPH":    result["se_GPH"].map(lambda x: f"{x:.3f}"),
        "p_value":   result["p_value"],
        "d_Whittle": result["d_Whittle"].map(lambda x: f"{x:.3f}"),
        "Hurst":     result["Hurst"].map(lambda x: f"{x:.3f}"),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(output_path, index=False, encoding="utf-8-sig")
    return paper


def export_latex_rows(result, output_path):
    """Write the LaTeX table rows (.tex), ready to \\input into the paper.

    Row format:
        LV    & $0.568$ & $0.086$ & $0.000^{***}$ & $0.605$ & $0.921$ \\
    """
    lines = []
    for _, row in result.iterrows():
        # Split the stars off '0.000***' and set them as a LaTeX superscript
        p_num, stars = split_pvalue_and_stars(row["p_value"])
        if stars:
            p_latex = f"{p_num}^{{{stars}}}"
        else:
            p_latex = p_num

        line = (
            f"  {row['Variable']:<6}"
            f"& ${row['d_GPH']:>6.3f}$ "
            f"& ${row['se_GPH']:>5.3f}$ "
            f"& ${p_latex}$ "
            f"& ${row['d_Whittle']:>5.3f}$ "
            f"& ${row['Hurst']:>5.3f}$ \\\\"
        )
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Long memory test rows — generated by long_memory_test.py\n")
        f.write("% Paste the 9 rows below between \\midrule and \\bottomrule of Table 2.\n")
        f.write("% Column order: Variable, d_GPH, se_GPH, p_value, d_Whittle, Hurst\n\n")
        f.write("\n".join(lines) + "\n")
    return lines


def run_all():
    print("Long-memory tests (Table 2)")

    data = prepare_data()
    rows = []
    for col in data.columns:
        series = data[col].dropna().values
        d_gph, se_gph, p_gph = gph_estimator(series)
        d_wh, se_wh = whittle_estimator(series)
        H = hurst_rs(series)
        rows.append({
            "Variable": col,
            "d_GPH": round(d_gph, 4),
            "se_GPH": round(se_gph, 4),
            "p_value": format_p_value(p_gph),
            "d_Whittle": round(d_wh, 4),
            "se_Whittle": round(se_wh, 4),
            "Hurst": round(H, 4),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")
    export_paper_csv(result, OUTPUT_PAPER_CSV)
    export_latex_rows(result, OUTPUT_LATEX_ROWS)
    print(f"\n-> {OUTPUT_TABLE}\n-> {OUTPUT_PAPER_CSV}\n-> {OUTPUT_LATEX_ROWS}")


if __name__ == "__main__":
    run_all()
