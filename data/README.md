# Data

Sample period: 2 January 2004 to 31 December 2016. All realized volatilities (RV) are annualized.

| File | Content |
|---|---|
| `raw/lasso_data.csv` | Daily RV of 332 S&P 500 constituents, and `RV_1Day`, `RV_5Day`, `RV_22Day` of the S&P 500 index |
| `raw/targets.csv` | Forecast targets: S&P 500 RV over the next 1 / 5 / 22 days |
| `raw/market_rv.csv` | Daily S&P 500 RV (`RV_final`) and its 1 / 5 / 22-day averages |
| `raw/industry_mapping.csv` | Industry of each of the 332 stocks (14 SIC industries) |
| `raw/predictors/vix.csv`, `epu.csv`, `skew.csv` | VIX, Economic Policy Uncertainty index (EPU), SKEW index |
| `raw/decomposition/continuous_rv.csv.gz`, `jump_rv.csv.gz` | Continuous and jump components of the 332 stocks' RV (§6.1) |
| `raw/decomposition/good_rv.csv.gz`, `bad_rv.csv.gz` | Upside and downside realized variances of the 332 stocks (§6.2) |
| `raw/decomposition/partial_rv.csv.gz` | Five realized partial variances of the S&P 500 index (§6.3) |

## Sources

- High-frequency returns of the individual stocks are from Pelger (2020):
  Pelger, M. (2020). Understanding systematic risk: A high-frequency approach.
  *The Journal of Finance*, 75(4), 2179–2220. https://doi.org/10.1111/jofi.12898
- High-frequency data of the S&P 500 index are from Pi-Trading.
- VIX and SKEW are from CBOE; EPU is from Baker, Bloom & Davis (https://www.policyuncertainty.com).

Daily RV is computed from 5-minute returns between 9:35 and 16:00, giving 77 returns per day.

If you use these data, please cite this paper and Pelger (2020).
