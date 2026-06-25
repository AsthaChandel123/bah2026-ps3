"""Tests for the gridded NAQI utilities (:mod:`aqi_india.aqi.grid`).

These pin the shape and a *known* cell value of the gridded AQI output, the
responsible-pollutant code, the derived category grid, and the colour-LUT
helpers — mirroring the golden vectors used for the tabular engine in
``test_naqi.py``. The module is pure (numpy + xarray only), so the suite runs
under the light dependency set.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from aqi_india.aqi import naqi
from aqi_india.aqi.breakpoints import CATEGORY_COLORS, CATEGORY_NAMES
from aqi_india.aqi.grid import (
    apply_naqi_grid,
    aqi_to_rgba,
    category_color_lut,
    category_grid,
)


def _toy_grid() -> xr.Dataset:
    """A 2x2 surface-pollutant grid with hand-computable NAQI per cell.

    Cells (row-major over lat then lon):
        (0,0): pm25=90 ->200, pm10=100 ->100, no2=40 ->50      -> AQI 200 (pm25)
        (0,1): pm25=30 ->50,  pm10=100 ->100, no2=40 ->50      -> AQI 100 (pm10)
        (1,0): no2=40, so2=80, co=2  (no PM)                   -> invalid (NaN)
        (1,1): pm25=250 ->400, pm10=430 ->400, no2=180 ->200   -> AQI 400 (pm25)
    """
    lat = np.array([10.0, 20.0], dtype=np.float64)
    lon = np.array([70.0, 80.0], dtype=np.float64)
    nan = np.nan
    pm25 = np.array([[90.0, 30.0], [nan, 250.0]], dtype=np.float32)
    pm10 = np.array([[100.0, 100.0], [nan, 430.0]], dtype=np.float32)
    no2 = np.array([[40.0, 40.0], [40.0, 180.0]], dtype=np.float32)
    so2 = np.array([[nan, nan], [80.0, nan]], dtype=np.float32)
    co = np.array([[nan, nan], [2.0, nan]], dtype=np.float32)
    return xr.Dataset(
        {
            "pm25": (("lat", "lon"), pm25),
            "pm10": (("lat", "lon"), pm10),
            "no2": (("lat", "lon"), no2),
            "so2": (("lat", "lon"), so2),
            "co": (("lat", "lon"), co),
        },
        coords={"lat": lat, "lon": lon},
    )


def test_apply_naqi_grid_shape_and_vars() -> None:
    out = apply_naqi_grid(_toy_grid())
    assert set(("aqi", "aqi_responsible", "aqi_category")).issubset(out.data_vars)
    assert out["aqi"].dims == ("lat", "lon")
    assert out["aqi"].shape == (2, 2)
    assert out["aqi"].dtype == np.float32
    assert out["aqi_responsible"].dtype == np.int16
    assert out["aqi_category"].dtype == np.int16


def test_apply_naqi_grid_known_cell_values() -> None:
    out = apply_naqi_grid(_toy_grid())
    aqi = out["aqi"].values

    # Known per-cell AQI (see _toy_grid docstring).
    assert float(aqi[0, 0]) == pytest.approx(200.0)
    assert float(aqi[0, 1]) == pytest.approx(100.0)
    assert math.isnan(float(aqi[1, 0]))  # no PM -> invalid
    assert float(aqi[1, 1]) == pytest.approx(400.0)


def test_apply_naqi_grid_responsible_codes() -> None:
    out = apply_naqi_grid(_toy_grid())
    codes = out["aqi_responsible"].values
    mapping = dict(
        pair.split(":")  # type: ignore[misc]
        for pair in out["aqi_responsible"].attrs["pollutant_codes"].split(",")
    )
    code_to_name = {int(k): v for k, v in mapping.items()}

    assert code_to_name[int(codes[0, 0])] == "pm25"  # pm25 drives 200
    assert code_to_name[int(codes[0, 1])] == "pm10"  # pm10 drives 100
    assert int(codes[1, 0]) == -1  # invalid cell sentinel
    assert code_to_name[int(codes[1, 1])] == "pm25"


def test_category_grid_matches_engine() -> None:
    out = apply_naqi_grid(_toy_grid())
    cats = out["aqi_category"].values
    # 200 -> Moderate (idx 2); 100 -> Satisfactory (idx 1); 400 -> Very Poor (idx 4).
    assert int(cats[0, 0]) == 2
    assert int(cats[0, 1]) == 1
    assert int(cats[1, 0]) == -1
    assert int(cats[1, 1]) == 4
    assert cats.dtype == np.int16


def test_category_grid_attrs_carry_legend() -> None:
    aqi = naqi.compute_aqi_grid(_toy_grid())["aqi"]
    cats = category_grid(aqi)
    assert "category_names" in cats.attrs
    assert cats.attrs["category_names"].split(",")[0] == "0:Good"
    assert len(cats.attrs["category_colors"].split(",")) == len(CATEGORY_NAMES)


def test_category_color_lut() -> None:
    lut_rgb = category_color_lut(with_alpha=False)
    lut_rgba = category_color_lut(with_alpha=True)
    assert lut_rgb.shape == (len(CATEGORY_COLORS), 3)
    assert lut_rgba.shape == (len(CATEGORY_COLORS), 4)
    assert np.all((lut_rgb >= 0.0) & (lut_rgb <= 1.0))
    # First category "#009865" -> (0, 0x98/255, 0x65/255).
    np.testing.assert_allclose(lut_rgb[0], [0.0, 0x98 / 255.0, 0x65 / 255.0], atol=1e-9)
    # Alpha channel is opaque.
    np.testing.assert_allclose(lut_rgba[:, 3], 1.0)


def test_aqi_to_rgba_colours_and_nodata() -> None:
    out = apply_naqi_grid(_toy_grid())
    rgba = aqi_to_rgba(out["aqi"].values)
    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype == np.float32

    lut = category_color_lut(with_alpha=True).astype(np.float32)
    # Cell (0,0) AQI 200 -> category 2 (Moderate) colour.
    np.testing.assert_allclose(rgba[0, 0], lut[2], atol=1e-6)
    # Cell (1,0) invalid -> transparent (alpha 0).
    assert float(rgba[1, 0, 3]) == pytest.approx(0.0)
