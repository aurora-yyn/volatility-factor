# 04 — Out-of-sample forecasting

| Script | Reproduces |
|---|---|
| `build_predictions.py` | Rolling forecasts used by Tables 4, 5, 7–10, 16 and Figures 5, 6; run first |
| `oos_r2.py` | Tables 4, 9 |
| `regime_analysis.py` | Table 5, Figure 6 |
| `multi_factor_combinations.py` | Table 6 |
| `time_variation_r2.py` | Figure 5 |

## Usage

Run from the repository root:

```bash
python 04_out_of_sample/build_predictions.py      # run first
python 04_out_of_sample/oos_r2.py                 # Tables 4 and 9
python 04_out_of_sample/regime_analysis.py
python 04_out_of_sample/multi_factor_combinations.py
python 04_out_of_sample/time_variation_r2.py
```

Results are written to `outputs/tables/` and `outputs/figures/`.
