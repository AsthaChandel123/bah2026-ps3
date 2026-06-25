"""Tests for the cross-validation ladder (`aqi_india.validation.cv`).

These assert the leakage-avoidance guarantees that make the CV ladder
trustworthy:
    * leave-station-out never puts a station in both train and test,
    * spatiotemporal-blocked folds share no held-out day between train and test,
    * spatial-block CV leaves a buffer dead-zone between train and test,
    * forward-chaining only ever trains on the past,
    * the ladder driver and confusion matrix produce well-formed outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqi_india.validation import cv
from aqi_india.validation.confusion import (
    aqi_category_confusion,
    assign_categories,
)


# --------------------------------------------------------------------------- #
# Fixtures (plain helpers — no heavy deps)
# --------------------------------------------------------------------------- #
def _toy_panel(n_stations: int = 8, n_days: int = 20, seed: int = 0):
    """Build a small station-day panel with lat/lon/time/station_id arrays."""
    rng = np.random.default_rng(seed)
    st_lat = rng.uniform(8.0, 35.0, n_stations)
    st_lon = rng.uniform(70.0, 95.0, n_stations)
    st_id = np.array([f"S{i:02d}" for i in range(n_stations)])
    days = pd.date_range("2023-10-01", periods=n_days, freq="D")

    rows = []
    for s in range(n_stations):
        for d in days:
            rows.append(
                {
                    "station_id": st_id[s],
                    "lat": st_lat[s],
                    "lon": st_lon[s],
                    "time": d,
                    "x": rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------- #
# leave_station_out — the core fairness guarantee
# --------------------------------------------------------------------------- #
def test_leave_station_out_disjoint() -> None:
    df = _toy_panel()
    groups = df["station_id"].to_numpy()
    n_folds = 0
    for train_idx, test_idx in cv.leave_station_out(df, groups, n_splits=4):
        train_stations = set(groups[train_idx])
        test_stations = set(groups[test_idx])
        # A station must never appear in both train and test.
        assert train_stations.isdisjoint(test_stations)
        # And the split must cover only valid, non-overlapping row indices.
        assert set(train_idx).isdisjoint(set(test_idx))
        n_folds += 1
    assert n_folds == 4


def test_leave_one_station_out_default() -> None:
    df = _toy_panel(n_stations=5)
    groups = df["station_id"].to_numpy()
    folds = list(cv.leave_station_out(df, groups, n_splits=None))
    # One fold per station; each test set is exactly one station.
    assert len(folds) == 5
    for _, test_idx in folds:
        assert len(set(groups[test_idx])) == 1


# --------------------------------------------------------------------------- #
# spatiotemporal_blocked_cv — no temporal overlap, no spatial overlap
# --------------------------------------------------------------------------- #
def test_spatiotemporal_no_temporal_overlap() -> None:
    df = _toy_panel(n_stations=10, n_days=40)
    day = df["time"].to_numpy().astype("datetime64[D]")
    sblock_i = np.floor((df["lat"] - df["lat"].min()) / 4.0).astype(int)
    sblock_j = np.floor((df["lon"] - df["lon"].min()) / 4.0).astype(int)
    sblock = np.array([f"{i}_{j}" for i, j in zip(sblock_i, sblock_j)])

    folds = list(
        cv.spatiotemporal_blocked_cv(
            df,
            df["lat"].to_numpy(),
            df["lon"].to_numpy(),
            df["time"].to_numpy(),
            block_deg=4.0,
            block_days=7.0,
        )
    )
    assert folds, "expected at least one spatiotemporal fold"
    for train_idx, test_idx in folds:
        # No held-out day appears in the training set.
        test_days = set(day[test_idx].tolist())
        train_days = set(day[train_idx].tolist())
        assert test_days.isdisjoint(train_days)
        # No held-out spatial block appears in the training set.
        test_blocks = set(sblock[test_idx].tolist())
        train_blocks = set(sblock[train_idx].tolist())
        assert test_blocks.isdisjoint(train_blocks)
        # Indices are disjoint.
        assert set(train_idx).isdisjoint(set(test_idx))


# --------------------------------------------------------------------------- #
# spatial_block_cv — buffer dead-zone is respected
# --------------------------------------------------------------------------- #
def test_spatial_block_buffer() -> None:
    df = _toy_panel(n_stations=12)
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    buffer = 1.0
    for train_idx, test_idx in cv.spatial_block_cv(
        df, lat, lon, block_deg=3.0, buffer_deg=buffer, checkerboard=3
    ):
        assert set(train_idx).isdisjoint(set(test_idx))
        # Every training station must be >= buffer from every test station
        # (Chebyshev distance in degrees).
        for tr in train_idx:
            d = np.maximum(
                np.abs(lat[tr] - lat[test_idx]),
                np.abs(lon[tr] - lon[test_idx]),
            ).min()
            assert d >= buffer


# --------------------------------------------------------------------------- #
# forward_chaining — train strictly on the past
# --------------------------------------------------------------------------- #
def test_forward_chaining_past_only() -> None:
    df = _toy_panel(n_stations=4, n_days=30)
    times = df["time"].to_numpy().astype("datetime64[D]")
    folds = list(cv.forward_chaining(df, df["time"].to_numpy(), n_splits=4))
    assert folds
    for train_idx, test_idx in folds:
        max_train_day = times[train_idx].max()
        min_test_day = times[test_idx].min()
        # All training days strictly precede every test day.
        assert max_train_day < min_test_day


def test_leave_time_out_blocks_disjoint() -> None:
    df = _toy_panel(n_stations=3, n_days=15)
    times = df["time"].to_numpy().astype("datetime64[D]")
    for train_idx, test_idx in cv.leave_time_out(
        df, df["time"].to_numpy(), n_splits=3
    ):
        # A held-out day never appears in training.
        assert set(times[test_idx]).isdisjoint(set(times[train_idx]))


# --------------------------------------------------------------------------- #
# random_kfold — partition is complete and disjoint
# --------------------------------------------------------------------------- #
def test_random_kfold_partition() -> None:
    X = np.zeros((50, 3))
    seen: set[int] = set()
    for train_idx, test_idx in cv.random_kfold(X, n_splits=5):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) + len(test_idx) == 50
        seen.update(test_idx.tolist())
    # Every row was tested exactly once across folds.
    assert seen == set(range(50))


# --------------------------------------------------------------------------- #
# cross_validate + run_cv_ladder — end-to-end with a trivial estimator
# --------------------------------------------------------------------------- #
class _MeanRegressor:
    """A dependency-free estimator: predicts the training-target mean."""

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MeanRegressor":
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mean)


def test_cross_validate_outputs() -> None:
    df = _toy_panel(n_stations=8, n_days=15)
    y = (
        2.0 * df["x"].to_numpy()
        + 0.5 * df["lat"].to_numpy()
        + np.random.default_rng(1).normal(0, 0.1, len(df))
    )
    result = cv.cross_validate(
        _MeanRegressor,
        df[["x", "lat", "lon"]],
        y,
        scheme="leave_station_out",
        groups=df["station_id"].to_numpy(),
    )
    assert result["scheme"] == "leave_station_out"
    assert result["n_folds"] > 0
    assert "rmse" in result["oof"]
    assert len(result["per_fold"]) == result["n_folds"]


def test_run_cv_ladder_table() -> None:
    df = _toy_panel(n_stations=10, n_days=20)
    y = 2.0 * df["x"].to_numpy() + 0.5 * df["lat"].to_numpy()
    table = cv.run_cv_ladder(
        _MeanRegressor,
        df[["x", "lat", "lon"]],
        y,
        groups=df["station_id"].to_numpy(),
        times=df["time"].to_numpy(),
        lats=df["lat"].to_numpy(),
        lons=df["lon"].to_numpy(),
    )
    # All six rungs should run given full metadata.
    assert set(table["scheme"]) <= set(cv.LADDER_SCHEMES)
    assert "leakage_vs_random" in table.columns
    assert "oof_r" in table.columns


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
def test_assign_categories_edges() -> None:
    # 25->Good(0), 75->Satisfactory(1), 150->Moderate(2), 250->Poor(3),
    # 350->Very Poor(4), 450->Severe(5); NaN -> -1.
    cats = assign_categories([25, 75, 150, 250, 350, 450, np.nan])
    assert cats.tolist() == [0, 1, 2, 3, 4, 5, -1]


def test_aqi_category_confusion_perfect() -> None:
    y = np.array([25, 75, 150, 250, 350, 450], dtype=float)
    out = aqi_category_confusion(y, y)
    assert out["accuracy"] == 1.0
    assert out["n"] == 6
    # Every diagonal entry is 1; off-diagonals zero.
    cm = out["matrix"]
    assert np.array_equal(cm, np.eye(6, dtype=cm.dtype))
    # Hazardous recall is perfect.
    assert out["hazardous_recall"] == 1.0


def test_aqi_category_confusion_row_normalized() -> None:
    y_true = np.array([250, 250, 450], dtype=float)
    y_pred = np.array([250, 150, 450], dtype=float)  # one Poor mis-called Moderate
    out = aqi_category_confusion(y_true, y_pred, normalize="true")
    # Poor (index 3) recall = 1 correct / 2 = 0.5.
    assert np.isclose(out["per_class"]["Poor"]["recall"], 0.5)
    # Row-normalised rows that have support sum to 1.
    cm = out["matrix"]
    assert np.isclose(cm[3].sum(), 1.0)
