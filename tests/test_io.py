# -*- coding: utf-8 -*-
"""Tests of data loading and the scale convention."""

from __future__ import annotations

import numpy as np
import pytest

from volfactor.io import (
    HAR_COLS,
    load_actual_lv,
    load_industry_map,
    load_panel,
    stock_columns,
    to_log_variance,
)


def test_panel_shape(panel):
    """The panel has 332 stocks + 3 HAR columns."""
    X, Y = panel
    assert X.shape == (3251, 335)
    assert len(stock_columns(X)) == 332
    assert all(c in X.columns for c in HAR_COLS)
    assert len(Y) == len(X)


@pytest.mark.parametrize("horizon,n_rows", [(1, 3251), (5, 3247), (22, 3230)])
def test_horizon_row_counts(horizon, n_rows):
    """Multi-day targets look h days ahead, so the number of rows decreases accordingly."""
    X, Y = load_panel(horizon=horizon)
    assert len(X) == len(Y) == n_rows


def test_log_variance_transform():
    """LV = log(RV²) = 2·log(RV)."""
    import pandas as pd

    rv = pd.Series([0.08, 0.15, 0.2])
    np.testing.assert_allclose(to_log_variance(rv), 2 * np.log(rv))


@pytest.mark.parametrize("horizon", [1, 5, 22])
def test_target_matches_actual(horizon):
    """The training target Y and the evaluation actuals come from two separate files and must be identical.

    Y comes from RV_target_* in targets.csv; actual comes from
    RV_{h}Day.shift(-h) in market_rv.csv.
    """
    _, Y = load_panel(horizon=horizon)
    actual = load_actual_lv(horizon=horizon)
    joined = Y.rename("y").to_frame().join(actual, how="inner")

    assert len(joined) == len(Y)
    np.testing.assert_allclose(joined["y"], joined["actual"], atol=1e-12)


def test_industry_map_covers_all_stocks(panel):
    """The 14 industries must cover exactly the 332 stocks."""
    X, _ = panel
    ind = load_industry_map()

    assert len(ind) == 332
    assert ind["industry"].nunique() == 14
    assert set(ind["ticker"]) == set(stock_columns(X))
