# -*- coding: utf-8 -*-
"""Combination forecasts: the FWL vectorization must equal stock-by-stock OLS."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from volfactor.combination import _design, _individual_predictions, _trimmed_mean
from volfactor.io import stock_columns


def test_fwl_matches_individual_ols(train_window):
    """The batched FWL solution must equal running ``sm.OLS`` (``LV ~ const + HAR3 + X_n``) for each stock."""
    X, Y = train_window
    stocks = stock_columns(X)[:40]  # 40 stocks suffice; all 332 would be slow

    sub = X[["RV_1Day", "RV_5Day", "RV_22Day"] + stocks]
    X_tr, X_te = sub.iloc[:1000], sub.iloc[1000:1001]
    y_tr = Y.iloc[:1000].to_numpy()

    fast = _individual_predictions(
        _design(X_tr), y_tr, X_tr[stocks].to_numpy(),
        _design(X_te), X_te[stocks].to_numpy(),
    )[0]

    slow = []
    for col in stocks:
        cols = ["RV_1Day", "RV_5Day", "RV_22Day", col]
        fit = sm.OLS(y_tr, sm.add_constant(X_tr[cols])).fit()
        slow.append(fit.predict(sm.add_constant(X_te[cols], has_constant="add")).iloc[0])

    np.testing.assert_allclose(fast, np.array(slow), rtol=1e-9)


def test_trimmed_mean_drops_one_extreme_each_side():
    """TMC drops one maximum and one minimum; it is not a percentage trim."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert _trimmed_mean(v) == np.mean([2.0, 3.0, 4.0])


def test_trimmed_mean_degenerate_cases():
    """With fewer than 3 elements it falls back to the plain mean."""
    assert _trimmed_mean(np.array([5.0, 7.0])) == 6.0
    assert _trimmed_mean(np.array([5.0])) == 5.0
