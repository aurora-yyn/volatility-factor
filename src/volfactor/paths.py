# -*- coding: utf-8 -*-
"""Path constants. All paths are resolved relative to the repository root."""

from __future__ import annotations

from pathlib import Path

#: Repository root (src/volfactor/paths.py -> two levels up)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREDICTORS_DIR = RAW_DIR / "predictors"

OUTPUT_DIR = REPO_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

DOCS_DIR = REPO_ROOT / "docs"


def ensure_output_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for d in (PREDICTIONS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
