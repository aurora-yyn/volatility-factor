# Tables and figures of the paper ↔ code

All commands are run from the repository root. `python run_all.py` runs everything in order.

## Step 1: rolling forecasts

```bash
python 04_out_of_sample/build_predictions.py
```

Writes `outputs/predictions/w{4,5,6}_h{1,5,22}.csv`, which are read by the scripts marked † below.

## Tables in the main text

| Paper | Content | Script |
|---|---|---|
| Table 1 | Descriptive statistics and correlations | `02_data_description/basic_statistics.py` |
| Table 2 | Long-memory tests | `02_data_description/long_memory_test.py` |
| Table 3 | In-sample estimation | `03_in_sample_results/in_sample_regression.py` |
| Table 4 | Out-of-sample R² | `04_out_of_sample/oos_r2.py` † |
| Table 5 | Out-of-sample R² across volatility regimes | `04_out_of_sample/regime_analysis.py` † |
| Table 6 | Multi-factor combinations | `04_out_of_sample/multi_factor_combinations.py` |
| Tables 7, 8 | MCS tests | `05_robustness/mcs_test.py` † |
| Table 9 | Alternative estimation windows | `04_out_of_sample/oos_r2.py` † |
| Table 10 | Factors from the correlation matrix | `05_robustness/correlation_matrix_factors.py` † |
| Table 11 | Incremental generalized correlation | `06_information_content/incremental_correlation.py` |
| Table 12 | Industry factors | `06_information_content/industry_factors.py` |
| Table 13 | Incremental information over VIX / EPU / SKEW | `06_information_content/incremental_predictability.py` |
| Tables 14, 15 | Continuous / jump and good / bad volatility factors | `07_extensions/component_factors.py` |
| Table 16 | Realized partial variances | `07_extensions/partial_variance_factors.py` † |

## Figures in the main text

| Paper | Content | Script |
|---|---|---|
| Figure 1 | ACF / PACF | `02_data_description/acf_pacf_plot.py` |
| Figure 2 | Factor time series | `02_data_description/factor_timeseries_plot.py` |
| Figure 4 | Number of factors | `01_factor_construction/eigenvalue_ratio.py` |
| Figure 5 | Out-of-sample R² by year | `04_out_of_sample/time_variation_r2.py` † |
| Figure 6 | Volatility regime heatmaps | `04_out_of_sample/regime_analysis.py` † |
| Figure 7 | The 50 companies with the largest factor loadings | `06_information_content/factor_loadings_plot.py` |
| Figure 8 | Incremental generalized correlation | `06_information_content/incremental_correlation.py` |

## Appendix

| Paper | Content | Script |
|---|---|---|
| Tables A.17, A.18 | ADF tests | `02_data_description/adf_test.py` |
| Table G.27 | VIX / EPU / SKEW on their own | `06_information_content/standalone_predictors.py` |
| Figure H.17 | Number of factors (γ = 0.30) | `01_factor_construction/eigenvalue_ratio.py --gamma 0.30` |
