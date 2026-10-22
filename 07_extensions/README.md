# 07 — Extensions

| Script | Reproduces |
|---|---|
| `component_factors.py` | Tables 14, 15 |
| `partial_variance_factors.py` | Table 16 |
| `component_robustness.py` | Appendix Tables C.19–C.22, D.23 |
| `component_factor_counts.py` | Appendix Figures E.11, E.12 |
| `component_summary_statistics.py` | Appendix Tables F.24–F.26 |

## Usage

Run from the repository root:

```bash
python 07_extensions/component_factors.py
python 07_extensions/partial_variance_factors.py
python 07_extensions/component_robustness.py --jobs 4
python 07_extensions/component_factor_counts.py
python 07_extensions/component_summary_statistics.py
```

`component_robustness.py` runs 27 rolling-forecast combinations; `--jobs` sets the number of parallel
processes (default 1).

`partial_variance_factors.py` requires `04_out_of_sample/build_predictions.py` to be run first.

Results are written to `outputs/tables/` and `outputs/figures/`.
