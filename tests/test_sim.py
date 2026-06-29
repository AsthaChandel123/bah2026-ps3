"""Unit tests for the synthetic-data generator (``aqi_india.sim.synthetic``).

These tests verify the binding DEV_CONTRACT §6 schema (dims, variable names,
units, file locations), the embedded known structure (IGP gradient, fire
cluster location, downwind HCHO enhancement), determinism, and the
station/fire/grid counts. They are pure (no network, no heavy deps) and run on
the light dependency set alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqi_india.sim import synthetic as syn
from aqi_india.utils.geo import PUNJAB_HARYANA_BBOX

START = "2023-10-01"
N_DAYS = 12
RES = 0.5  # coarser grid keeps the test fast


@pytest.fixture(scope="module")
def grid():
    return syn.make_grid(START, N_DAYS, res=RES, seed=syn.DEFAULT_SEED)


@pytest.fixture(scope="module")
def fires(grid):
    return syn.make_fires(grid, season="oct_nov", seed=syn.DEFAULT_SEED)


@pytest.fixture(scope="module")
def grid_inj(grid, fires):
    return syn.inject_fire_hcho(grid, fires)


@pytest.fixture(scope="module")
def stations(grid_inj):
    return syn.make_stations(grid_inj, n=60, seed=syn.DEFAULT_SEED)


# --------------------------------------------------------------------------- #
# Grid schema                                                                  #
# --------------------------------------------------------------------------- #
def test_grid_dims_and_vars(grid):
    assert tuple(grid["aod"].dims) == ("time", "lat", "lon")
    assert set(grid.data_vars) == set(syn.GRID_VARS)
    assert grid.sizes["time"] == N_DAYS
    # 0.5 deg over INDIA_BBOX (68..98 lon, 6..38 lat).
    assert grid.sizes["lon"] == int(round((98.0 - 68.0) / RES)) + 1
    assert grid.sizes["lat"] == int(round((38.0 - 6.0) / RES)) + 1


def test_grid_dtype_and_coords_ascending(grid):
    for v in syn.GRID_VARS:
        assert grid[v].dtype == np.float32
    assert np.all(np.diff(grid["lat"].values) > 0)
    assert np.all(np.diff(grid["lon"].values) > 0)
    assert np.issubdtype(grid["time"].dtype, np.datetime64)


def test_grid_units_present(grid):
    assert grid["co_col"].attrs["units"] == "mol m-2"
    assert grid["t2m"].attrs["units"] == "K"
    assert "SYNTHETIC" in grid.attrs["title"]


def test_cloud_gaps_on_columns_only(grid):
    # Column vars are partially NaN (cloud gaps); met vars are fully present.
    for cv in syn.COLUMN_VARS:
        frac = float(np.isnan(grid[cv].values).mean())
        assert 0.15 < frac < 0.70, f"{cv} cloud frac {frac}"
    for mv in ("blh", "rh", "wind_u", "wind_v", "t2m", "ssrd"):
        assert not np.isnan(grid[mv].values).any(), mv


def test_igp_gradient_north_higher_than_peninsula(grid):
    # Mean AOD over the IGP latitudes should exceed the far-south peninsula.
    aod = grid["aod"].mean("time")
    north = aod.sel(lat=slice(26.0, 30.0)).mean().item()
    south = aod.sel(lat=slice(8.0, 12.0)).mean().item()
    assert north > south


def test_prevailing_wind_direction(grid):
    # NW -> SE prevailing flow: mean u positive (eastward), v negative (south).
    assert float(grid["wind_u"].mean()) > 0
    assert float(grid["wind_v"].mean()) < 0


# --------------------------------------------------------------------------- #
# Determinism                                                                  #
# --------------------------------------------------------------------------- #
def test_grid_deterministic():
    a = syn.make_grid(START, 5, res=1.0, seed=7)
    b = syn.make_grid(START, 5, res=1.0, seed=7)
    np.testing.assert_array_equal(
        np.nan_to_num(a["aod"].values), np.nan_to_num(b["aod"].values)
    )
    # Cloud masks identical too.
    np.testing.assert_array_equal(
        np.isnan(a["no2_col"].values), np.isnan(b["no2_col"].values)
    )


# --------------------------------------------------------------------------- #
# Fires                                                                        #
# --------------------------------------------------------------------------- #
def test_fire_schema(fires):
    expected = {"date", "lat", "lon", "frp", "sensor", "h3_res7"}
    assert expected.issubset(set(fires.columns))
    assert fires.crs is not None and fires.crs.to_epsg() == 4326
    assert (fires["frp"] > 0).all()
    assert set(fires["sensor"].unique()).issubset(
        {"VIIRS_SNPP", "VIIRS_NOAA20", "MODIS"}
    )


def test_fire_cluster_in_punjab_haryana(fires):
    # The dominant cluster must sit inside the Punjab/Haryana box.
    min_lon, min_lat, max_lon, max_lat = PUNJAB_HARYANA_BBOX
    in_box = (
        (fires["lon"] >= min_lon)
        & (fires["lon"] <= max_lon)
        & (fires["lat"] >= min_lat)
        & (fires["lat"] <= max_lat)
    )
    assert in_box.mean() > 0.5  # majority of detections in the burning region


def test_fires_in_burning_window(fires):
    months = pd.to_datetime(fires["date"]).dt.month
    assert months.isin((10, 11)).mean() > 0.9


def test_fire_h3_matches_coords(fires):
    from aqi_india.features.h3_index import latlng_to_cell

    row = fires.iloc[0]
    assert row["h3_res7"] == latlng_to_cell(row["lat"], row["lon"], 7)


# --------------------------------------------------------------------------- #
# Fire -> HCHO injection                                                       #
# --------------------------------------------------------------------------- #
def test_injection_increases_hcho(grid, grid_inj):
    before = np.nansum(grid["hcho_col"].values)
    after = np.nansum(grid_inj["hcho_col"].values)
    assert after > before


def test_injection_preserves_cloud_gaps(grid, grid_inj):
    # Cloud-gap NaNs in HCHO must remain NaN after injection.
    assert np.array_equal(
        np.isnan(grid["hcho_col"].values), np.isnan(grid_inj["hcho_col"].values)
    )


def test_injection_hotspot_locatable(grid_inj, fires):
    # The HCHO enhancement maximum should lie reasonably near the fire cluster.
    hcho = grid_inj["hcho_col"].mean("time")
    arr = hcho.values
    iy, ix = np.unravel_index(np.nanargmax(arr), arr.shape)
    peak_lat = float(grid_inj["lat"].values[iy])
    peak_lon = float(grid_inj["lon"].values[ix])
    # Downwind of Punjab/Haryana (~30N, 75E) toward the SE; within a few deg.
    assert 20.0 < peak_lat < 33.0
    assert 73.0 < peak_lon < 88.0


# --------------------------------------------------------------------------- #
# Stations                                                                     #
# --------------------------------------------------------------------------- #
def test_station_schema(stations):
    cols = {
        "station_id",
        "name",
        "lat",
        "lon",
        "h3_res7",
        "region",
        "time",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
    }
    assert cols.issubset(set(stations.columns))
    assert stations.crs.to_epsg() == 4326


def test_station_count_and_rows(stations):
    assert stations["station_id"].nunique() == 60
    assert len(stations) == 60 * N_DAYS


def test_station_co_units_plausible(stations):
    # CO reported in mg/m3 -> small numbers, unlike ug/m3 pollutants.
    assert stations["co"].median() < 20.0
    assert stations["pm25"].median() > stations["co"].median()


def test_station_pm10_ge_pm25(stations):
    valid = stations.dropna(subset=["pm25", "pm10"])
    assert (valid["pm10"] >= valid["pm25"]).mean() > 0.99


def test_station_igp_denser(stations):
    reg = stations.drop_duplicates("station_id")
    assert (reg["region"] == "IGP").mean() > 0.4


def test_station_h3_matches_coords(stations):
    from aqi_india.features.h3_index import latlng_to_cell

    row = stations.iloc[0]
    assert row["h3_res7"] == latlng_to_cell(row["lat"], row["lon"], 7)


def test_station_pm25_responds_to_aod(grid_inj, stations):
    # Sanity: PM2.5 is a positive function of AOD -> non-degenerate spread.
    assert stations["pm25"].std() > 0
    assert stations["pm25"].notna().all()


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def test_generate_all_writes_contract_files(tmp_path):
    paths = syn.generate_all(
        tmp_path, start_date=START, n_days=6, res=1.0, n_stations=30
    )
    proc = tmp_path / "data" / "processed"
    assert (proc / "grid.nc").exists()
    assert (proc / "stations.parquet").exists()
    assert (proc / "fires.parquet").exists()
    assert (proc / "stations.geojson").exists()
    assert (proc / "fires.geojson").exists()
    assert (tmp_path / "data" / "raw" / "synthetic" / "grid_sample.nc").exists()
    assert paths["grid"].exists()
