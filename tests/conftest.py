# -*- coding: utf-8 -*-
"""Shared pytest configuration: put src/ on the import path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def panel():
    """Full data panel (X, Y) for horizon=1, loaded once per test session."""
    from volfactor.io import load_panel

    return load_panel(horizon=1)


@pytest.fixture(scope="session")
def train_window(panel):
    """A typical 5-year training window for the factor-extraction equivalence tests."""
    X, Y = panel
    return X.iloc[:1260], Y.iloc[:1260]
