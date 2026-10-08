# 05 — Robustness checks

| Script | Reproduces |
|---|---|
| `mcs_test.py` | Tables 7, 8 |
| `correlation_matrix_factors.py` | Table 10 |

## Usage

Run from the repository root:

```bash
python 05_robustness/mcs_test.py
python 05_robustness/correlation_matrix_factors.py
```

Run `04_out_of_sample/build_predictions.py` first. Table 9 is produced by `04_out_of_sample/oos_r2.py`.

Results are written to `outputs/tables/` and `outputs/figures/`.
