"""Tests for the validation metrics library."""

from __future__ import annotations

import math

import numpy as np

from aqi_india.validation import metrics


def test_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert metrics.rmse(y, y) == 0.0
    assert metrics.mae(y, y) == 0.0
    assert metrics.mbe(y, y) == 0.0
    assert math.isclose(metrics.pearson_r(y, y), 1.0, abs_tol=1e-9)
    assert math.isclose(metrics.r2(y, y), 1.0, abs_tol=1e-9)
    assert math.isclose(metrics.ioa(y, y), 1.0, abs_tol=1e-9)
    slope, intercept = metrics.rma_slope_intercept(y, y)
    assert math.isclose(slope, 1.0, abs_tol=1e-6)
    assert math.isclose(intercept, 0.0, abs_tol=1e-6)


def test_known_rmse_mae_mbe() -> None:
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0, -2.0])
    # errors: 1,-1,2,-2 -> squared 1,1,4,4 mean 2.5 -> rmse sqrt(2.5)
    assert math.isclose(metrics.rmse(y_true, y_pred), math.sqrt(2.5), abs_tol=1e-9)
    assert math.isclose(metrics.mae(y_true, y_pred), 1.5, abs_tol=1e-9)
    assert math.isclose(metrics.mbe(y_true, y_pred), 0.0, abs_tol=1e-9)


def test_known_correlation() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0])  # perfectly linear, slope 2
    assert math.isclose(metrics.pearson_r(y_true, y_pred), 1.0, abs_tol=1e-9)
    slope, intercept = metrics.rma_slope_intercept(y_true, y_pred)
    assert math.isclose(slope, 2.0, abs_tol=1e-6)
    assert math.isclose(intercept, 0.0, abs_tol=1e-6)


def test_known_ioa() -> None:
    # Hand-computable IOA case.
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])  # one unit error on last
    # num = sum((p-o)^2) = 1
    # o_mean = 2.5; den terms: (|p-2.5|+|o-2.5|)^2
    #  -> (1.5+1.5)^2=9, (0.5+0.5)^2=1, (0.5+0.5)^2=1, (2.5+1.5)^2=16 -> 27
    expected = 1.0 - 1.0 / 27.0
    assert math.isclose(metrics.ioa(y_true, y_pred), expected, abs_tol=1e-9)


def test_nmb_nme() -> None:
    y_true = np.array([10.0, 20.0, 30.0])  # sum 60
    y_pred = np.array([12.0, 18.0, 33.0])  # diffs +2,-2,+3 -> sum 3
    assert math.isclose(metrics.nmb(y_true, y_pred), 3.0 / 60.0, abs_tol=1e-9)
    assert math.isclose(metrics.nme(y_true, y_pred), 7.0 / 60.0, abs_tol=1e-9)


def test_nan_safety() -> None:
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, np.nan, 4.0])
    # Only indices 0 and 3 survive -> perfect on those.
    assert metrics.rmse(y_true, y_pred) == 0.0
    table = metrics.metrics_table(y_true, y_pred)
    assert table["n"] == 2.0


def test_metrics_table_keys() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.3, 4.8])
    table = metrics.metrics_table(y_true, y_pred)
    for key in (
        "n", "rmse", "mae", "mbe", "r", "r2", "nmb", "nme", "ioa",
        "rma_slope", "rma_intercept",
    ):
        assert key in table


def test_stratified_metrics() -> None:
    y_true = np.array([10.0, 20.0, 150.0, 250.0])
    y_pred = np.array([12.0, 18.0, 160.0, 240.0])
    bands = [(0.0, 100.0), (100.0, 300.0)]
    strat = metrics.stratified_metrics(y_true, y_pred, bands)
    assert "[0,100)" in strat
    assert "[100,300)" in strat
    assert strat["[0,100)"]["n"] == 2.0
