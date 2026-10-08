# 01 — Number of factors

| Script | Reproduces |
|---|---|
| `eigenvalue_ratio.py` | Figure 4; Appendix Figure H.17 |

## Usage

Run from the repository root:

```bash
python 01_factor_construction/eigenvalue_ratio.py                # Figure 4
python 01_factor_construction/eigenvalue_ratio.py --gamma 0.30   # Figure H.17
```

PCA / sPCA factor extraction is implemented in `src/volfactor/factors.py` and called by the chapter scripts.

Results are written to `outputs/tables/` and `outputs/figures/`.
