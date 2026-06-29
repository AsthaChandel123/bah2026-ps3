"""Tests for the aqi_india.ingest adapters.

These cover the pure logic (no network / GEE / credentials): the QA/asset-id
constants, the synthetic offline paths producing the DEV_CONTRACT schema, the
CPCB station-DB helpers (stable ids, OpenAQ dedup, tagging, hourly->daily), and
the AERONET expected-error envelope + Angstrom interpolation.

The real GEE/CDS/earthaccess code paths are intentionally not exercised here —
they require heavy credentialed deps and are run by the Demo/Verify agents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Light-import guarantee + constants                                          #
# --------------------------------------------------------------------------- #
def test_package_imports_light() -> None:
    """`import aqi_india.ingest` must not pull heavy/credentialed deps."""
    import sys

    import aqi_india.ingest  # noqa: F401

    for heavy in ("ee", "cdsapi", "earthaccess"):
        assert heavy not in sys.modules, f"{heavy} imported at ingest import time"


def test_s5p_asset_ids_and_qa() -> None:
    from aqi_india.ingest import s5p

    assert s5p.S5P_COLLECTIONS["no2"] == "COPERNICUS/S5P/OFFL/L3_NO2"
    assert s5p.S5P_COLLECTIONS["hcho"] == "COPERNICUS/S5P/OFFL/L3_HCHO"
    # Blueprint QA convention: NO2 >= 0.75; HCHO/SO2/CO >= 0.5.
    assert s5p.S5P_QA_THRESHOLDS["no2"] == 0.75
    for p in ("hcho", "so2", "co", "o3"):
        assert s5p.S5P_QA_THRESHOLDS[p] == 0.5
    assert s5p.PRODUCT_TO_GRID_VAR["hcho"] == "hcho_col"


def test_maiac_constants() -> None:
    from aqi_india.ingest import maiac

    assert maiac.MAIAC_COLLECTION == "MODIS/061/MCD19A2_GRANULES"
    assert maiac.MAIAC_BANDS["aod_055"] == "Optical_Depth_055"


def test_firms_sources_and_template() -> None:
    from aqi_india.ingest import firms

    assert "VIIRS_SNPP_NRT" in firms.FIRMS_NRT_SOURCES
    assert "{map_key}" in firms.FIRMS_AREA_TEMPLATE


# --------------------------------------------------------------------------- #
# Synthetic offline paths -> DEV_CONTRACT schema                              #
# --------------------------------------------------------------------------- #
START = "2023-10-01"
END = "2023-10-09"
SMALL_AOI = (74.0, 26.0, 82.0, 32.0)


def test_fetch_s5p_synthetic_hcho_schema() -> None:
    from aqi_india.ingest import s5p

    ds = s5p.fetch_s5p("hcho", START, END, SMALL_AOI, synthetic=True)
    assert "hcho_col" in ds.data_vars
    assert set(ds["hcho_col"].dims) == {"time", "lat", "lon"}
    assert ds.sizes["time"] == 8
    # ascending coords
    assert np.all(np.diff(ds["lat"].values) > 0)
    assert np.all(np.diff(ds["lon"].values) > 0)


def test_fetch_all_columns_synthetic() -> None:
    from aqi_india.ingest import s5p

    ds = s5p.fetch_all_columns(START, END, SMALL_AOI, synthetic=True)
    for v in ("no2_col", "so2_col", "co_col", "o3_col", "hcho_col"):
        assert v in ds.data_vars


def test_fetch_maiac_synthetic_aod() -> None:
    from aqi_india.ingest import maiac

    ds = maiac.fetch_maiac(START, END, SMALL_AOI, synthetic=True)
    assert "aod" in ds.data_vars
    assert ds.sizes["time"] == 8


def test_fetch_insat_synthetic_and_access_doc() -> None:
    from aqi_india.ingest import insat_aod

    ds = insat_aod.fetch_insat_aod(START, END, SMALL_AOI, synthetic=True)
    assert "aod" in ds.data_vars
    doc = insat_aod.describe_access()
    assert "mosdac" in doc and doc["access"].startswith("http_download")
    # Real path without files must error clearly.
    with pytest.raises(ValueError):
        insat_aod.fetch_insat_aod(START, END, SMALL_AOI)


def test_fetch_era5_synthetic_met() -> None:
    from aqi_india.ingest import era5_cds

    ds = era5_cds.fetch_era5(START, END, SMALL_AOI, synthetic=True)
    for v in ("blh", "rh", "wind_u", "wind_v", "t2m", "ssrd"):
        assert v in ds.data_vars


def test_fetch_imdaa_synthetic_and_real_requires_files() -> None:
    from aqi_india.ingest import imdaa

    ds = imdaa.fetch_imdaa(START, END, SMALL_AOI, synthetic=True)
    assert "blh" in ds.data_vars
    with pytest.raises(ValueError):
        imdaa.fetch_imdaa(START, END, SMALL_AOI)


def test_fetch_merra2_synthetic_and_pm25() -> None:
    from aqi_india.ingest import merra2

    ds = merra2.fetch_merra2(START, END, SMALL_AOI, collection="aer", synthetic=True)
    assert "TOTEXTTAU" in ds.data_vars
    pm = merra2.derive_pm25_from_speciated(ds)
    assert pm.name == "pm25"
    assert float(pm.max()) >= 0.0
    flx = merra2.fetch_merra2(START, END, SMALL_AOI, collection="flx", synthetic=True)
    assert "PBLH" in flx.data_vars


def test_fetch_cams_synthetic_gapfree() -> None:
    from aqi_india.ingest import cams

    ds = cams.fetch_cams(START, END, SMALL_AOI, synthetic=True)
    # gap-free prior: no NaNs in the produced column fields
    assert "hcho_col" in ds.data_vars
    assert not bool(np.isnan(ds["hcho_col"].values).any())


def test_fetch_firms_synthetic_schema() -> None:
    from aqi_india.ingest import firms

    gdf = firms.fetch_firms(START, "2023-11-30", SMALL_AOI, synthetic=True)
    for col in ("date", "lat", "lon", "frp", "sensor", "h3_res7"):
        assert col in gdf.columns
    assert gdf.crs is not None
    assert len(gdf) > 0


def test_fetch_cpcb_synthetic_schema_and_tags() -> None:
    from aqi_india.ingest import cpcb

    gdf = cpcb.fetch_cpcb(START, END, SMALL_AOI, synthetic=True, n_stations=40)
    for col in ("station_id", "h3_res7", "region", "time", "tag", "source"):
        assert col in gdf.columns
    for p in ("pm25", "pm10", "no2", "so2", "co", "o3"):
        assert p in gdf.columns
    # Tags present; a sensor never appears in both train and validation.
    tags = set(gdf["tag"].unique())
    assert tags <= {"label", "validation", "gapfill"}
    by_station = gdf.groupby("station_id")["tag"].nunique()
    assert (by_station == 1).all()


# --------------------------------------------------------------------------- #
# CPCB station-DB pure helpers                                                #
# --------------------------------------------------------------------------- #
def test_make_station_id_stable() -> None:
    from aqi_india.ingest import cpcb

    a = cpcb.make_station_id("Anand Vihar, Delhi", 28.6469, 77.3160)
    b = cpcb.make_station_id("anand vihar, delhi", 28.6469, 77.3160)
    assert a == b and a.startswith("CPCB-")
    c = cpcb.make_station_id("Other", 19.0, 73.0)
    assert c != a


def test_dedup_against_openaq() -> None:
    from aqi_india.ingest import cpcb

    cpcb_df = pd.DataFrame(
        {
            "station_id_raw": ["A", "B"],
            "name": ["A", "B"],
            "lat": [28.60, 19.00],
            "lon": [77.20, 73.00],
            "time": pd.to_datetime(["2023-10-01", "2023-10-01"]),
        }
    )
    openaq = pd.DataFrame(
        {
            "station_id_raw": ["mirror", "genuine"],
            "name": ["mirror", "genuine"],
            "lat": [28.6005, 13.0],  # ~55 m from A -> mirror; far -> kept
            "lon": [77.2003, 80.0],
            "time": pd.to_datetime(["2023-10-01", "2023-10-01"]),
        }
    )
    merged = cpcb.dedup_against_openaq(cpcb_df, openaq, radius_km=1.0)
    raws = set(merged["station_id_raw"])
    assert "mirror" not in raws
    assert "genuine" in raws and {"A", "B"} <= raws


def test_aggregate_hourly_to_daily_completeness_and_8h() -> None:
    from aqi_india.ingest import cpcb

    times = pd.date_range("2023-10-01", periods=24, freq="1h")
    hourly = pd.DataFrame(
        {
            "station_id_raw": ["S1"] * 24,
            "name": ["S1"] * 24,
            "lat": [28.6] * 24,
            "lon": [77.2] * 24,
            "time": times,
            "pm25": np.full(24, 100.0),
            "co": np.arange(24, dtype=float),  # ramp -> 8h-max near the top
        }
    )
    for p in ("pm10", "no2", "so2", "o3", "nh3"):
        hourly[p] = np.nan
    daily = cpcb.aggregate_hourly_to_daily(hourly)
    assert len(daily) == 1
    row = daily.iloc[0]
    assert np.isclose(row["pm25"], 100.0)  # 24h mean
    # max 8h rolling mean of 0..23 is the mean of 16..23 = 19.5
    assert np.isclose(row["co"], 19.5, atol=1e-6)

    # Below 75% completeness -> NaN.
    sparse = hourly.copy()
    sparse.loc[sparse.index[5:], "pm25"] = np.nan  # only 5/24 valid
    daily2 = cpcb.aggregate_hourly_to_daily(sparse)
    assert np.isnan(daily2.iloc[0]["pm25"])


# --------------------------------------------------------------------------- #
# AERONET validation helpers                                                  #
# --------------------------------------------------------------------------- #
def test_aeronet_expected_error_envelope() -> None:
    from aqi_india.ingest import aeronet

    ref = np.array([0.2, 0.5, 1.0])
    # Within envelope EE = 0.05 + 0.15*ref -> [0.08, 0.125, 0.20]
    sat_in = ref + np.array([0.07, -0.12, 0.19])
    sat_out = ref + np.array([0.20, 0.20, 0.30])
    assert aeronet.within_expected_error(sat_in, ref).all()
    assert not aeronet.within_expected_error(sat_out, ref).any()


def test_aeronet_angstrom_interpolation() -> None:
    from aqi_india.ingest.aeronet import _interp_to_550

    # Flat spectrum (alpha=0) -> AOD_550 == AOD_440.
    out = _interp_to_550(np.array([0.5]), np.array([0.5]))
    assert np.isclose(out[0], 0.5, atol=1e-9)
    # 550 should fall between 440 and 675 values for a normal decreasing spectrum.
    out2 = _interp_to_550(np.array([0.6]), np.array([0.3]))
    assert 0.3 < out2[0] < 0.6


def test_aeronet_synthetic_sites() -> None:
    from aqi_india.ingest import aeronet

    df = aeronet.fetch_aeronet(START, END, ("Kanpur", "Pune"), synthetic=True)
    assert set(df["site"]) == {"Kanpur", "Pune"}
    assert (df["aod_550"] > 0).all()
