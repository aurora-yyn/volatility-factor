# The Information Content of Realized Volatility Factors — Replication Code

This repository contains the code and data to reproduce all results in the main text of
*The Information Content of Realized Volatility Factors* (*International Journal of Forecasting*).

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.14.

## Running the code

All commands are run from the repository root. To reproduce everything:

```bash
python run_all.py
```

This runs all 22 steps in order (about 2–3 hours). `python run_all.py --list` lists the steps, and
`python run_all.py --from N` resumes from step N.

Individual scripts can also be run on their own. The rolling forecasts must be generated first:

```bash
python 04_out_of_sample/build_predictions.py
```

[`docs/paper_to_code.md`](docs/paper_to_code.md) lists which script reproduces each table and figure of
the paper; the `README.md` in each folder lists that folder's scripts and commands. Results are written
to `outputs/tables/` and `outputs/figures/`.

## Repository structure

| Folder | Section of the paper |
|---|---|
| `01_factor_construction/` | § 3.3 Number of factors |
| `02_data_description/` | § 3 Data |
| `03_in_sample_results/` | § 4.1 In-sample estimation |
| `04_out_of_sample/` | § 4.2 Out-of-sample forecasting |
| `05_robustness/` | § 4.3 Robustness checks |
| `06_information_content/` | § 5 Information content of the factors |
| `07_extensions/` | § 6 Extensions |
| `src/volfactor/` | Shared functions (data loading, factor extraction, rolling forecasts, evaluation) |
| `data/` | Input data; see [`data/README.md`](data/README.md) |
| `outputs/` | Results |
| `tests/` | Unit tests, `pytest tests/` |

## License

The code is released under the [MIT License](LICENSE). Data sources and citation requirements are
described in [`data/README.md`](data/README.md).

## Citation

> Ye, W., Yun, Y., & Wu, B. (2026). The Information Content of Realized Volatility Factors.
> *International Journal of Forecasting*.
