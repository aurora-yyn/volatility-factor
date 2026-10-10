# Replication Package: The Information Content of Realized Volatility Factors

Replication code and data for

> Ye, W., Yun, Y., & Wu, B. (2026). The Information Content of Realized Volatility Factors.
> *International Journal of Forecasting*.

The package reproduces all tables and figures in the main text of the paper, together with
supporting results reported in the Appendix.

## Package information

| | |
|---|---|
| Assembled | 9 October 2026 |
| Prepared by | Yining Yun |
| Contact | Bin Wu (corresponding author), bin.w@ustc.edu.cn |

## Special requirements

**None.** The code runs on a standard laptop: no GPU, cluster, or parallel computing setup is needed,
and 16 GB of RAM is sufficient. The full pipeline takes about 2 hours 40 minutes; see
[Expected runtime](#expected-runtime). There are no manual steps.

## Computing environment

| | |
|---|---|
| Operating system | Windows 11 (64-bit) |
| Processor | Intel Core i7-1195G7, 2.90 GHz, 4 cores / 8 threads |
| Memory | 16 GB |
| Language | Python 3.14.0 |

All required packages and their exact versions are listed in `requirements.txt`:

| Package | Version |
|---|---|
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scipy | 1.16.3 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.6 |
| arch | 8.0.0 |
| matplotlib | 3.10.7 |
| pytest (tests only) | 9.1.1 |

### Recreating the environment

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The versions are pinned because other versions of scikit-learn and arch may give slightly different
PCA and MCS results.

## Data

All data needed to reproduce the results are included in `data/raw/`; no data has to be obtained
elsewhere. The sample period is 2 January 2004 to 31 December 2016 (3,251 trading days for the stock panel).
All realized volatilities (RV) are annualized. CSV files are plain text; `.csv.gz` files are gzip-compressed CSV.

| File | Content |
|---|---|
| `lasso_data.csv` | Daily RV of 332 S&P 500 constituents, and `RV_1Day`, `RV_5Day`, `RV_22Day` of the S&P 500 index |
| `targets.csv` | Forecast targets: S&P 500 RV over the next 1 / 5 / 22 days |
| `market_rv.csv` | Daily S&P 500 RV (`RV_final`) and its 1 / 5 / 22-day averages |
| `industry_mapping.csv` | Industry of each of the 332 stocks (14 SIC industries) |
| `predictors/vix.csv`, `epu.csv`, `skew.csv` | VIX, Economic Policy Uncertainty index (EPU), SKEW index |
| `decomposition/continuous_rv.csv.gz`, `jump_rv.csv.gz` | Continuous and jump components of the stocks' RV (§6.1) |
| `decomposition/good_rv.csv.gz`, `bad_rv.csv.gz` | Upside and downside realized variances of the stocks (§6.2) |
| `decomposition/partial_rv.csv.gz` | Five realized partial variances of the S&P 500 index (§6.3) |

### Sources

- **Individual stocks**: high-frequency returns of the 332 constituents are from Pelger (2020),
  publicly available with the article: Pelger, M. (2020). Understanding systematic risk: A
  high-frequency approach. *The Journal of Finance*, 75(4), 2179–2220.
  https://doi.org/10.1111/jofi.12898
- **S&P 500 index**: high-frequency data are from Pi-Trading (https://www.pitrading.com).
- **VIX and SKEW**: Cboe Global Markets. **EPU**: Baker, Bloom & Davis (https://www.policyuncertainty.com).

### Pre-processing

The files in `data/raw/` are daily series constructed from 5-minute returns between 9:35 and 16:00
(77 returns per day). The realized volatility is defined in §2.1 and §3.1 of the paper, the
continuous / jump and good / bad decompositions in §6.1 and §6.2, and the realized partial variances in
§6.3. The code reproduces all results starting from these daily series; the construction of the daily
series from the high-frequency returns is not part of this package.

### Intermediary datasets

`04_out_of_sample/build_predictions.py` generates the rolling out-of-sample forecasts in
`outputs/predictions/w{4,5,6}_h{1,5,22}.csv` (about 50 minutes), which are inputs to Tables 4, 5, 7–10, 16
and Figures 5, 6. `05_robustness/correlation_matrix_factors.py` generates `outputs/predictions/w5_h{1,5,22}_corr.csv`.
These files are included in the package, so the scripts that read them can be run directly without
regenerating them. All other outputs (`outputs/tables/`, `outputs/figures/`) are also included for comparison
with the paper.

## Repository structure

```
volatility-factor/
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.py                    # runs the full pipeline in order
├── data/
│   ├── README.md
│   └── raw/                      # input data (see Data)
├── src/volfactor/                # shared functions: data loading, factor extraction,
│                                 #   rolling forecasts, evaluation
├── 01_factor_construction/       # § 3.3  number of factors
├── 02_data_description/          # § 3    data description
├── 03_in_sample_results/         # § 4.1  in-sample estimation
├── 04_out_of_sample/             # § 4.2  out-of-sample forecasting
├── 05_robustness/                # § 4.3  robustness checks
├── 06_information_content/       # § 5    information content of the factors
├── 07_extensions/                # § 6    extensions
├── outputs/
│   ├── predictions/              # intermediary rolling forecasts
│   ├── tables/                   # tables (CSV)
│   ├── figures/                  # figures (PNG)
│   └── logs/                     # run logs (created by run_all.py)
└── tests/                        # unit tests: pytest tests/
```

Each numbered folder contains a `README.md` listing its scripts.

## Running the code

All commands are run from the repository root. To reproduce everything:

```bash
python run_all.py
```

This runs the 22 steps below in order and writes a log for each step to `outputs/logs/`.
`python run_all.py --list` lists the steps; `python run_all.py --from N` resumes from step N.
Each script can also be run on its own, e.g. `python 02_data_description/basic_statistics.py`.

## Which code produces which outputs

Results are written to `outputs/tables/` and `outputs/figures/`. Scripts marked † read the
intermediary forecasts produced in step 1.

| Step | Output in the paper | Script | Runtime |
|---|---|---|---|
| 1 | Rolling forecasts (intermediary) | `04_out_of_sample/build_predictions.py` | ~52 min |
| 2 | Table 1 | `02_data_description/basic_statistics.py` | < 1 min |
| 3 | Table 2 | `02_data_description/long_memory_test.py` | < 1 min |
| 4 | Figure 1 | `02_data_description/acf_pacf_plot.py` | < 1 min |
| 5 | Figure 2 | `02_data_description/factor_timeseries_plot.py` | < 1 min |
| 6 | Tables A.17, A.18 | `02_data_description/adf_test.py` | < 1 min |
| 7 | Figure 4 | `01_factor_construction/eigenvalue_ratio.py` | ~1 min |
| 8 | Figure H.17 | `01_factor_construction/eigenvalue_ratio.py --gamma 0.30` | ~1 min |
| 9 | Table 3 | `03_in_sample_results/in_sample_regression.py` | < 1 min |
| 10 | Tables 4, 9 † | `04_out_of_sample/oos_r2.py` | < 1 min |
| 11 | Table 5, Figure 6 † | `04_out_of_sample/regime_analysis.py` | < 1 min |
| 12 | Table 6 | `04_out_of_sample/multi_factor_combinations.py` | ~22 min |
| 13 | Figure 5 † | `04_out_of_sample/time_variation_r2.py` | < 1 min |
| 14 | Tables 7, 8 † | `05_robustness/mcs_test.py` | ~1 min |
| 15 | Table 10 † | `05_robustness/correlation_matrix_factors.py` | ~7 min |
| 16 | Table 11, Figure 8 | `06_information_content/incremental_correlation.py` | < 1 min |
| 17 | Table 12 | `06_information_content/industry_factors.py` | < 1 min |
| 18 | Table 13 | `06_information_content/incremental_predictability.py` | ~17 min |
| 19 | Figure 7 | `06_information_content/factor_loadings_plot.py` | < 1 min |
| 20 | Table G.27 | `06_information_content/standalone_predictors.py` | < 1 min |
| 21 | Tables 14, 15 | `07_extensions/component_factors.py` | ~54 min |
| 22 | Table 16 † | `07_extensions/partial_variance_factors.py` | ~4 min |

Figure 3 of the paper (sector composition of the sample) was prepared separately and is not
produced by the code.

The MCS p-values in Tables 7 and 8 are computed by block bootstrap with a fixed random seed
(`DEFAULT_SEED = 42` in `src/volfactor/evaluation.py`).

## Expected runtime

On the machine described above, the full pipeline (`python run_all.py`) takes about 2 hours 40 minutes.
Most of the time is spent in steps 1, 12, 18 and 21; all other steps finish within a few minutes.
Because the intermediary forecasts are included, all steps except 1 can be run directly.

## Licence

- **Code**: [MIT License](LICENSE).
- **Data**: provided solely for replicating the results of this paper. Users of the individual stock data
  should cite Pelger (2020); the S&P 500 high-frequency data are from Pi-Trading.

## Citation

> Ye, W., Yun, Y., & Wu, B. (2026). The Information Content of Realized Volatility Factors.
> *International Journal of Forecasting*.
