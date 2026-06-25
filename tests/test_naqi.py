"""Golden-vector tests for the CPCB NAQI engine.

These vectors are taken from the CPCB (2014) breakpoint tables and worked
examples. They pin the sub-index interpolation, the max-of-sub-index
aggregation, the responsible-pollutant argmax, the validity mask (>=3 pollutants
AND >=1 PM), and the CO-in-mg/m3 unit rule. Getting these exactly right is the
single most important correctness requirement of the whole project.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aqi_india.aqi import naqi
from aqi_india.aqi.breakpoints import category_name


# --------------------------------------------------------------------------- #
# Single-pollutant sub-index golden vectors                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("pollutant", "conc", "expected", "tol"),
    [
        # PM2.5 (ug/m3, 24h)
        ("pm25", 0.0, 0.0, 1e-9),
        ("pm25", 30.0, 50.0, 1e-9),        # exact breakpoint -> band top
        ("pm25", 45.0, 75.5, 0.6),         # mid band 30-60 -> ~75
        ("pm25", 60.0, 100.0, 1e-9),
        ("pm25", 90.0, 200.0, 1e-9),       # exact breakpoint
        ("pm25", 120.0, 300.0, 1e-9),
        ("pm25", 125.0, 304.8, 0.6),       # band 120-250 -> ~302-305
        ("pm25", 250.0, 400.0, 1e-9),
        # PM10
        ("pm10", 100.0, 100.0, 1e-9),
        ("pm10", 250.0, 200.0, 1e-9),
        # NO2
        ("no2", 40.0, 50.0, 1e-9),
        ("no2", 180.0, 200.0, 1e-9),
        # SO2
        ("so2", 80.0, 100.0, 1e-9),
        ("so2", 380.0, 200.0, 1e-9),
        # CO in mg/m3 (8h)
        ("co", 1.0, 50.0, 1e-9),
        ("co", 2.0, 100.0, 1e-9),          # CO 2.0 mg/m3 -> AQI 100
        ("co", 10.0, 200.0, 1e-9),
        # O3 (ug/m3, 8h)
        ("o3", 100.0, 100.0, 1e-9),
        ("o3", 168.0, 200.0, 1e-9),        # O3 168 -> AQI 200
        # NH3
        ("nh3", 400.0, 100.0, 1e-9),
        ("nh3", 800.0, 200.0, 1e-9),
    ],
)
def test_sub_index_golden(pollutant: str, conc: float, expected: float, tol: float) -> None:
    got = float(naqi.sub_index(conc, pollutant))
    assert math.isclose(got, expected, abs_tol=tol), (
        f"{pollutant} @ {conc}: got {got}, expected {expected} (tol {tol})"
    )


def test_sub_index_clamps_to_500() -> None:
    """Above the CPCB top cap the sub-index clamps to 500, never beyond."""
    assert float(naqi.sub_index(1000.0, "pm25")) == pytest.approx(500.0)
    assert float(naqi.sub_index(5000.0, "co")) == pytest.approx(500.0)


def test_sub_index_nan_and_negative_are_invalid() -> None:
    assert math.isnan(float(naqi.sub_index(np.nan, "pm25")))
    assert math.isnan(float(naqi.sub_index(-5.0, "pm25")))


def test_sub_index_vectorized() -> None:
    out = naqi.sub_index(np.array([30.0, 60.0, 90.0]), "pm25")
    np.testing.assert_allclose(out, [50.0, 100.0, 200.0])


# --------------------------------------------------------------------------- #
# Multi-pollutant aggregation, responsible pollutant, validity                #
# --------------------------------------------------------------------------- #
def test_aqi_is_max_with_correct_responsible() -> None:
    # PM2.5=90 -> 200, PM10=100 -> 100, NO2=40 -> 50, CO=1 -> 50.
    sub = {
        "pm25": naqi.sub_index(90.0, "pm25"),
        "pm10": naqi.sub_index(100.0, "pm10"),
        "no2": naqi.sub_index(40.0, "no2"),
        "co": naqi.sub_index(1.0, "co"),
    }
    aqi, resp = naqi.aqi_from_subindices(sub, return_responsible=True)
    assert float(aqi) == pytest.approx(200.0)
    assert str(resp) == "pm25"
    assert category_name(float(aqi)) == "Moderate"


def test_validity_two_pollutants_is_nan() -> None:
    """Fewer than 3 valid pollutants -> invalid (NaN)."""
    sub = {"pm25": naqi.sub_index(90.0, "pm25"), "no2": naqi.sub_index(40.0, "no2")}
    aqi = naqi.aqi_from_subindices(sub)
    assert math.isnan(float(aqi))


def test_validity_three_pollutants_no_pm_is_nan() -> None:
    """Three valid pollutants but no PM2.5/PM10 -> invalid (NaN)."""
    sub = {
        "no2": naqi.sub_index(40.0, "no2"),
        "so2": naqi.sub_index(80.0, "so2"),
        "co": naqi.sub_index(2.0, "co"),
    }
    aqi, resp = naqi.aqi_from_subindices(sub, return_responsible=True)
    assert math.isnan(float(aqi))
    assert resp is None or (np.ndim(resp) == 0 and resp.item() is None)


def test_validity_three_pollutants_with_pm_is_valid() -> None:
    sub = {
        "pm10": naqi.sub_index(100.0, "pm10"),
        "no2": naqi.sub_index(40.0, "no2"),
        "co": naqi.sub_index(2.0, "co"),
    }
    aqi = naqi.aqi_from_subindices(sub)
    assert float(aqi) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# DataFrame entry point                                                       #
# --------------------------------------------------------------------------- #
def test_compute_aqi_dataframe() -> None:
    df = pd.DataFrame(
        {
            "pm25": [90.0, 250.0, 30.0],
            "pm10": [100.0, 430.0, 110.0],
            "no2": [40.0, 180.0, 40.0],
            "co": [1.0, 10.0, 1.0],
        }
    )
    out = naqi.compute_aqi(df)
    # Row 1: pm25=90->200 (max). Row 2: pm25=250->400 (max).
    # Row 3: pm10=110->~107 (max), the others give 50.
    np.testing.assert_allclose(out["aqi"].to_numpy(), [200.0, 400.0, 107.0], atol=1.0)
    assert list(out["aqi_responsible"]) == ["pm25", "pm25", "pm10"]
    assert list(out["aqi_category"]) == ["Moderate", "Very Poor", "Moderate"]


def test_compute_aqi_dataframe_invalid_row_is_nan() -> None:
    # Only two pollutants -> invalid.
    df = pd.DataFrame({"no2": [40.0], "so2": [80.0]})
    out = naqi.compute_aqi(df)
    assert math.isnan(out["aqi"].iloc[0])
    assert out["aqi_category"].iloc[0] is None


def test_category_colors() -> None:
    name, color = naqi.category(450.0)
    assert name == "Severe"
    assert color.startswith("#")
