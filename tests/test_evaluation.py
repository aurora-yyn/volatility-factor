# -*- coding: utf-8 -*-
"""Tests of the loss functions, forecast accuracy tests and MCS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volfactor.evaluation import (
    DEFAULT_SEED,
    LOSS_FUNCTIONS,
    clark_west,
    loss_matrix,
    loss_qlike,
    oos_r2,
    run_mcs,
    significance_stars,
)


# ---------------------------------------------------------------- QLIKE

def test_qlike_is_zero_for_perfect_forecast():
    x = np.array([-2.5, -3.0, -1.8])
    np.testing.assert_allclose(loss_qlike(x, x), 0.0, atol=1e-15)


def test_qlike_is_nonnegative():
    """Patton QLIKE is always ≥ 0, for any real input."""
    rng = np.random.default_rng(0)
    actual = rng.normal(-2.5, 1.0, 10_000)
    pred = rng.normal(-2.5, 1.0, 10_000)
    assert (loss_qlike(actual, pred) >= 0).all()


def test_qlike_matches_variance_scale_definition():
    """QLIKE written in log-variance differences must equal the textbook variance-scale form::

        σ²/σ̂² − log(σ²/σ̂²) − 1

    i.e. the standard form of Patton (2011).
    """
    rng = np.random.default_rng(1)
    var_actual = rng.uniform(0.01, 0.5, 500)
    var_pred = rng.uniform(0.01, 0.5, 500)

    textbook = var_actual / var_pred - np.log(var_actual / var_pred) - 1
    ours = loss_qlike(np.log(var_actual), np.log(var_pred))

    np.testing.assert_allclose(ours, textbook, rtol=1e-10)


# ---------------------------------------------------------------- OOS R²

def test_oos_r2_zero_against_itself():
    rng = np.random.default_rng(2)
    actual, pred = rng.normal(size=100), rng.normal(size=100)
    assert oos_r2(actual, pred, pred) == pytest.approx(0.0)


def test_oos_r2_positive_when_model_beats_benchmark():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    good = actual + 0.1
    bad = actual + 1.0
    assert oos_r2(actual, good, bad) > 0
    assert oos_r2(actual, bad, good) < 0


# ---------------------------------------------------------------- Clark–West

def test_clark_west_detects_genuine_improvement():
    rng = np.random.default_rng(3)
    actual = rng.normal(size=500)
    bench = np.zeros(500)
    pred = 0.5 * actual  # a genuinely informative forecast

    stat, pval = clark_west(actual, pred, bench)
    assert stat > 0 and pval < 0.01


def test_significance_stars():
    assert significance_stars(0.005) == "***"
    assert significance_stars(0.03) == "**"
    assert significance_stars(0.08) == "*"
    assert significance_stars(0.5) == ""


# ---------------------------------------------------------------- loss matrix

def test_loss_matrix_aligns_on_common_dates():
    idx = pd.date_range("2010-01-01", periods=5)
    actual = pd.Series([-2.0] * 5, index=idx)
    preds = pd.DataFrame({"A": [-2.0] * 5, "B": [-1.0] * 5}, index=idx)

    lm = loss_matrix(actual, preds, "MSE")
    assert list(lm.columns) == ["A", "B"]
    np.testing.assert_allclose(lm["A"], 0.0)
    np.testing.assert_allclose(lm["B"], 1.0)


def test_loss_matrix_rejects_unknown_loss():
    idx = pd.date_range("2010-01-01", periods=3)
    with pytest.raises(KeyError):
        loss_matrix(
            pd.Series([-2.0] * 3, index=idx),
            pd.DataFrame({"A": [-2.0] * 3}, index=idx),
            "NOT_A_LOSS",
        )


def test_all_five_losses_available():
    """All five loss functions used in Tables 7/8 must be available."""
    assert set(LOSS_FUNCTIONS) == {"MSE", "MAE", "QLIKE", "HMSE", "HMAE"}


# ---------------------------------------------------------------- MCS

def test_mcs_is_reproducible_with_fixed_seed():
    """The same seed must give identical MCS p-values."""
    rng = np.random.default_rng(4)
    idx = pd.date_range("2010-01-01", periods=400)
    losses = pd.DataFrame(
        {
            "good": rng.chisquare(2, 400) * 0.5,
            "mid": rng.chisquare(2, 400) * 0.8,
            "bad": rng.chisquare(2, 400) * 1.5,
        },
        index=idx,
    )

    a = run_mcs(losses, size=0.05, reps=100, seed=DEFAULT_SEED)
    b = run_mcs(losses, size=0.05, reps=100, seed=DEFAULT_SEED)

    pd.testing.assert_series_equal(a, b)


def test_mcs_different_seeds_still_rank_consistently():
    """The seed changes the exact p-values, but a clearly worse model must still be eliminated."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2010-01-01", periods=400)
    losses = pd.DataFrame(
        {"good": rng.chisquare(2, 400) * 0.5, "bad": rng.chisquare(2, 400) * 3.0},
        index=idx,
    )

    for seed in (1, 42, 999):
        p = run_mcs(losses, size=0.05, reps=100, seed=seed)
        assert p["good"] > p["bad"]
