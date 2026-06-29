"""Light, pure unit tests for the runnable model layer (LightGBM + helpers).

These exercise the dependency-light path only (numpy + pandas + lightgbm +
scikit-learn). The torch models (SA-ConvLSTM / CNN-LSTM) are intentionally *not*
imported or run here — they live behind the ``deep`` extra. Every test builds a
tiny synthetic ``(X, y)`` and asserts shapes / dict keys / coverage behaviour.

Run with ``pytest tests/test_model.py`` once the light deps are installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _synthetic_matrix(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic station-day feature matrix with an AOD->PM signal."""
    rng = np.random.default_rng(seed)
    aod = rng.uniform(0.1, 1.5, n)
    blh = rng.uniform(200, 2000, n)
    rh = rng.uniform(20, 95, n)
    no2_col = rng.uniform(1e-5, 5e-4, n)
    # Surface pollutants with a real (monotone-in-AOD) signal + noise.
    pm25 = 40 + 60 * aod - 0.01 * blh + rng.normal(0, 5, n)
    pm10 = 60 + 90 * aod + rng.normal(0, 8, n)
    no2 = 10 + 4e4 * no2_col + rng.normal(0, 3, n)
    co = 0.4 + 0.5 * aod + rng.normal(0, 0.05, n)  # mg/m3
    o3 = 30 + 0.02 * rh + rng.normal(0, 4, n)
    so2 = 8 + 5 * aod + rng.normal(0, 2, n)

    df = pd.DataFrame(
        {
            "station_id": np.repeat([f"S{i}" for i in range(n // 12)], 12)[:n],
            "time": pd.date_range("2023-10-01", periods=12).tolist() * (n // 12),
            "lat": rng.uniform(8, 36, n),
            "lon": rng.uniform(70, 96, n),
            "aod": aod,
            "blh": blh,
            "rh": rh,
            "no2_col": no2_col,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "co": np.clip(co, 0, None),
            "o3": o3,
        }
    )
    return df


# --------------------------------------------------------------------------- #
# MultiPollutantLGBM
# --------------------------------------------------------------------------- #
def test_lgbm_fit_predict_shapes() -> None:
    pytest.importorskip("lightgbm")
    from aqi_india.models.lightgbm_model import MultiPollutantLGBM, SURFACE_POLLUTANTS

    df = _synthetic_matrix()
    feature_cols = ["aod", "blh", "rh", "no2_col"]
    X = df[feature_cols]
    y_dict = {p: df[p].to_numpy() for p in SURFACE_POLLUTANTS}

    model = MultiPollutantLGBM(quantiles=(0.05, 0.5, 0.95)).fit(X, y_dict)

    preds = model.predict(X)
    assert set(preds) == set(SURFACE_POLLUTANTS)
    for pol, arr in preds.items():
        assert arr.shape == (len(df),)
        assert np.all(arr >= 0.0)  # concentrations are clipped non-negative

    quant = model.predict_quantiles(X)
    for pol, qd in quant.items():
        alphas = sorted(qd)
        lo, hi = qd[alphas[0]], qd[alphas[-1]]
        assert lo.shape == (len(df),)
        assert np.all(hi >= lo - 1e-6)  # well-ordered quantiles


def test_lgbm_monotone_constraint_vector() -> None:
    from aqi_india.models.lightgbm_model import monotone_constraints_for

    cons = monotone_constraints_for(["aod", "aod_pbl", "blh", "rh", "no2_col"])
    assert cons == [1, 1, 0, 0, 0]


def test_lgbm_feature_importances() -> None:
    pytest.importorskip("lightgbm")
    from aqi_india.models.lightgbm_model import MultiPollutantLGBM

    df = _synthetic_matrix()
    feature_cols = ["aod", "blh", "rh", "no2_col"]
    model = MultiPollutantLGBM(pollutants=("pm25", "pm10"), quantiles=None)
    model.fit(df[feature_cols], {"pm25": df["pm25"].to_numpy(), "pm10": df["pm10"].to_numpy()})
    imp = model.feature_importances()
    assert list(imp.index) == sorted(feature_cols)
    assert {"pm25", "pm10"}.issubset(imp.columns)


def test_lgbm_skips_all_nan_label() -> None:
    pytest.importorskip("lightgbm")
    from aqi_india.models.lightgbm_model import MultiPollutantLGBM

    df = _synthetic_matrix()
    feature_cols = ["aod", "blh"]
    y = {"pm25": df["pm25"].to_numpy(), "so2": np.full(len(df), np.nan)}
    model = MultiPollutantLGBM(pollutants=("pm25", "so2"), quantiles=None)
    model.fit(df[feature_cols], y)
    assert model.fitted_pollutants == ("pm25",)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def test_linear_and_rf_baseline_shapes() -> None:
    from aqi_india.models.baseline import LinearBaseline, RandomForestBaseline

    df = _synthetic_matrix()
    feature_cols = ["aod", "blh", "rh"]
    X = df[feature_cols]
    y = {"pm25": df["pm25"].to_numpy(), "pm10": df["pm10"].to_numpy()}

    lin = LinearBaseline(pollutants=("pm25", "pm10")).fit(X, y)
    assert set(lin.predict(X)) == {"pm25", "pm10"}

    rf = RandomForestBaseline(pollutants=("pm25", "pm10"), n_estimators=30).fit(X, y)
    preds = rf.predict(X)
    assert preds["pm25"].shape == (len(df),)
    q = rf.predict_quantiles(X)
    assert set(q["pm25"]) == {0.05, 0.5, 0.95}
    assert np.all(q["pm25"][0.95] >= q["pm25"][0.05] - 1e-6)


# --------------------------------------------------------------------------- #
# UQ: split-conformal
# --------------------------------------------------------------------------- #
def test_split_conformal_coverage() -> None:
    from aqi_india.models.uq import (
        SplitConformalRegressor,
        coverage_report,
        picp,
        mpiw,
    )
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(size=(n, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 1.0, n)
    Xtr, ytr = X[:400], y[:400]
    Xte, yte = X[400:], y[400:]

    scr = SplitConformalRegressor(LinearRegression(), alpha=0.1)
    scr.fit_calibrate(Xtr, ytr, cal_fraction=0.4)
    lo, point, hi = scr.predict_interval(Xte)

    assert point.shape == (len(yte),)
    cov = picp(yte, lo, hi)
    # Conformal guarantee is marginal >= 1 - alpha up to finite-sample noise.
    assert cov >= 0.80
    assert mpiw(lo, hi) > 0
    report = coverage_report(yte, lo, hi, alpha=0.1)
    assert set(report) >= {"picp", "mpiw", "target_coverage", "coverage_gap", "n"}


def test_applicability_flag() -> None:
    from aqi_india.models.uq import applicability_flag

    Xtr = np.random.default_rng(2).normal(size=(200, 4))
    Xq = np.vstack([np.zeros((3, 4)), np.full((2, 4), 100.0)])  # 2 far out-of-range
    flag = applicability_flag(Xtr, Xq, n_std=3.0)
    assert flag.shape == (5,)
    assert flag[:3].all()        # in-envelope rows are applicable
    assert not flag[3:].any()    # extreme rows abstain


# --------------------------------------------------------------------------- #
# Ensemble stack
# --------------------------------------------------------------------------- #
def test_ensemble_stack_mean_and_optimal() -> None:
    from aqi_india.models.baseline import LinearBaseline, RandomForestBaseline
    from aqi_india.models.stack import EnsembleStack, fit_oof_weights

    df = _synthetic_matrix()
    feature_cols = ["aod", "blh", "rh"]
    X = df[feature_cols]
    y = {"pm25": df["pm25"].to_numpy()}

    members = {
        "linear": LinearBaseline(pollutants=("pm25",)),
        "rf": RandomForestBaseline(pollutants=("pm25",), n_estimators=30),
    }
    stack = EnsembleStack(members=members, pollutants=("pm25",), strategy="mean").fit(X, y)
    out = stack.predict(X)
    assert out["pm25"].shape == (len(df),)

    # OOF weight fitting lives on the simplex.
    w = fit_oof_weights(
        [df["pm25"].to_numpy() + 1.0, df["pm25"].to_numpy() - 1.0],
        df["pm25"].to_numpy(),
    )
    assert w.shape == (2,)
    assert abs(w.sum() - 1.0) < 1e-6
    assert np.all(w >= -1e-9)


# --------------------------------------------------------------------------- #
# train_obj1 end-to-end (light, in-memory)
# --------------------------------------------------------------------------- #
def test_train_obj1_in_memory(tmp_path) -> None:
    pytest.importorskip("lightgbm")
    from aqi_india.models.train import train_obj1

    df = _synthetic_matrix()
    result = train_obj1(
        df,
        cv_scheme="leave_station_out",
        run_ladder=False,  # keep it fast; CV ladder exercised in cv tests
        models_dir=tmp_path / "models",
        model_name="lightgbm",
    )
    assert result["model"].fitted_pollutants  # at least one pollutant trained
    assert result["model_path"].exists()
