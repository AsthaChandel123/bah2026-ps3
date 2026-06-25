"""Unit tests for the :mod:`aqi_india.fusion` package.

These tests use only the light dependency set (numpy / scipy / xarray / pandas /
sklearn). They are pure assertions and are NOT executed here — the Verify/Demo
agents run the suite.

Coverage:
* DINEOF fills a masked hole in a low-rank space-time matrix and recovers the
  hidden values reasonably, reporting explained variance and a gap fraction.
* :func:`fill_gaps` removes all gaps from a synthetic cloud-gapped cube and
  reports a lower gap fraction afterwards.
* :func:`quantile_map` aligns a biased satellite distribution to the observation
  distribution (CDF matching).
* IDW recovers a smooth field from scattered samples.
* :func:`stack_sources` / :func:`fuse_aod` produce provenance + uncertainty.
* :func:`to_grid` resamples onto a coarser target grid with the right shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from aqi_india.fusion.biascorrect import ml_residual, quantile_map
from aqi_india.fusion.dineof import dineof
from aqi_india.fusion.gapfill import fill_gaps
from aqi_india.fusion.harmonize import fuse_aod, stack_sources
from aqi_india.fusion.kriging import idw, regression_kriging
from aqi_india.fusion.regrid import make_grid_from_bbox, to_grid


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _low_rank_cube(n_t=40, n_lat=12, n_lon=15, rank=3, seed=0):
    """Build a smooth low-rank (time, lat, lon) cube reconstructible by DINEOF."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(6.0, 38.0, n_lat)
    lon = np.linspace(68.0, 98.0, n_lon)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    spatial = [np.sin(0.1 * xx + k) * np.cos(0.1 * yy - k) for k in range(rank)]
    temporal = [np.sin(np.linspace(0, (k + 1) * np.pi, n_t)) for k in range(rank)]
    cube = sum(
        t[:, None, None] * s[None]
        for t, s in zip(temporal, spatial, strict=True)
    )
    cube = cube + 5.0 + 0.01 * rng.standard_normal(cube.shape)
    return cube.astype("float64"), lat, lon


# --------------------------------------------------------------------------- #
# DINEOF                                                                        #
# --------------------------------------------------------------------------- #
def test_dineof_fills_masked_hole():
    cube, _, _ = _low_rank_cube()
    truth = cube.copy()

    # Realistic cloud mask: scattered random gaps plus a small contiguous block.
    rng = np.random.default_rng(123)
    gapped = cube.copy()
    cloud = rng.random(cube.shape) < 0.25
    gapped[cloud] = np.nan
    gapped[12:16, 5:7, 6:9] = np.nan  # a coherent cloud patch
    hole = ~np.isfinite(gapped)
    assert hole.any()

    res = dineof(gapped, k_max=8, seed=1)

    # No NaNs remain and the originally-present data are untouched.
    assert np.isfinite(res.filled).all()
    np.testing.assert_allclose(res.filled[~hole], truth[~hole], rtol=0, atol=1e-4)

    # The filled hole tracks the hidden truth far better than a naive mean fill.
    err = np.abs(res.filled[hole] - truth[hole])
    naive = np.abs(np.nanmean(gapped) - truth[hole])
    assert err.mean() < 0.5 * naive.mean()

    # Diagnostics are sane.
    assert 0.0 <= res.explained_variance <= 1.0
    assert res.explained_variance > 0.9
    assert 1 <= res.n_eof <= 8
    assert 0.0 < res.gap_fraction < 1.0
    assert res.uncertainty.shape == res.filled.shape
    # Uncertainty is higher inside the hole than outside.
    assert res.uncertainty[hole].mean() > res.uncertainty[~hole].mean()


def test_dineof_no_gaps_is_passthrough():
    cube, _, _ = _low_rank_cube(seed=2)
    res = dineof(cube, k_max=5)
    assert res.gap_fraction == 0.0
    np.testing.assert_allclose(res.filled, cube.astype("float32"), atol=1e-5)


# --------------------------------------------------------------------------- #
# fill_gaps cascade                                                            #
# --------------------------------------------------------------------------- #
def test_fill_gaps_removes_all_gaps():
    cube, lat, lon = _low_rank_cube(seed=3)
    rng = np.random.default_rng(7)
    gapped = cube.copy()
    # Random cloud mask ~35% missing.
    mask = rng.random(cube.shape) < 0.35
    gapped[mask] = np.nan

    ds = xr.Dataset(
        {"aod": (("time", "lat", "lon"), gapped.astype("float32"))},
        coords={
            "time": pd.date_range("2024-11-01", periods=cube.shape[0]),
            "lat": lat,
            "lon": lon,
        },
    )
    before = float(np.isnan(ds["aod"].values).mean())
    assert before > 0.0

    out = fill_gaps(ds, method="auto", seed=5)

    after = float(np.isnan(out["aod"].values).mean())
    assert after == 0.0
    assert after < before
    # Uncertainty channel was added.
    assert "aod_uncertainty" in out.data_vars
    # Reports recorded and consistent.
    reports = out.attrs["gapfill_reports"]
    assert reports and reports[0]["variable"] == "aod"
    assert reports[0]["gap_after"] <= reports[0]["gap_before"]


def test_fill_gaps_dataarray_input():
    cube, lat, lon = _low_rank_cube(seed=11)
    cube[5, 2, 2] = np.nan
    da = xr.DataArray(
        cube.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(cube.shape[0]), "lat": lat, "lon": lon},
        name="no2_col",
    )
    out = fill_gaps(da, method="dineof")
    assert isinstance(out, xr.DataArray)
    assert np.isfinite(out.values).all()


# --------------------------------------------------------------------------- #
# quantile mapping / bias correction                                          #
# --------------------------------------------------------------------------- #
def test_quantile_map_aligns_distributions():
    rng = np.random.default_rng(0)
    n = 5000
    obs = rng.gamma(shape=2.0, scale=20.0, size=n)  # ground truth distribution
    # Satellite is biased high and stretched: a monotone distortion of obs.
    sat = 1.4 * obs + 15.0 + rng.normal(0, 1.0, n)

    mapper = quantile_map(sat, obs, n_quantiles=200)
    corrected = mapper.transform(sat)

    # After CDF matching the corrected distribution matches obs much better.
    for q in (10, 25, 50, 75, 90):
        cm = np.percentile(corrected, q)
        ob = np.percentile(obs, q)
        raw = np.percentile(sat, q)
        assert abs(cm - ob) < abs(raw - ob)

    # Mean and std are close to the observation distribution.
    assert abs(np.mean(corrected) - np.mean(obs)) < 0.05 * np.mean(obs)
    assert abs(np.std(corrected) - np.std(obs)) < 0.10 * np.std(obs)


def test_quantile_map_regionalized():
    rng = np.random.default_rng(1)
    n = 4000
    obs = rng.gamma(2.0, 20.0, n)
    region = np.where(rng.random(n) < 0.5, "IGP", "Peninsula")
    sat = np.where(region == "IGP", 1.5 * obs + 10, 0.8 * obs - 5)
    sat = sat + rng.normal(0, 1.0, n)

    mapper = quantile_map(sat, obs, region=region, n_quantiles=100)
    assert "IGP" in mapper.sat_q and "Peninsula" in mapper.sat_q
    corrected = mapper.transform(sat, region=region)
    assert abs(np.median(corrected) - np.median(obs)) < 0.1 * np.median(obs)


def test_ml_residual_reduces_bias():
    rng = np.random.default_rng(2)
    n = 2000
    blh = rng.uniform(200, 2000, n)
    rh = rng.uniform(20, 95, n)
    sat = rng.uniform(10, 120, n)
    # True obs has a nonlinear, met-dependent residual on top of sat.
    obs = sat + 30.0 * (1.0 - blh / 2000.0) + 0.1 * rh + rng.normal(0, 2.0, n)
    feats = pd.DataFrame({"blh": blh, "rh": rh})

    corrector = ml_residual(sat, obs, feats)
    corrected = corrector.transform(sat, feats)

    raw_bias = np.mean(np.abs(obs - sat))
    new_bias = np.mean(np.abs(obs - corrected))
    assert new_bias < raw_bias
    assert corrector.backend in ("lightgbm", "sklearn_hgb")
    assert corrector.feature_names == ["blh", "rh"]


# --------------------------------------------------------------------------- #
# kriging / IDW                                                                #
# --------------------------------------------------------------------------- #
def test_idw_recovers_smooth_field():
    rng = np.random.default_rng(0)
    grid_lon = np.linspace(70, 90, 25)
    grid_lat = np.linspace(10, 30, 25)

    def field(lon, lat):
        return 50 + 10 * np.sin(0.2 * lon) + 5 * np.cos(0.2 * lat)

    obs_lon = rng.uniform(70, 90, 120)
    obs_lat = rng.uniform(10, 30, 120)
    obs_val = field(obs_lon, obs_lat)

    res = idw(obs_lon, obs_lat, obs_val, grid_lon, grid_lat, k=8)
    assert res.method == "idw"
    assert res.estimate.shape == (grid_lat.size, grid_lon.size)
    assert np.isfinite(res.estimate).all()

    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
    truth = field(mesh_lon, mesh_lat)
    # Interior agreement should be good for a smooth field.
    interior = (
        (mesh_lon > 73) & (mesh_lon < 87) & (mesh_lat > 13) & (mesh_lat < 27)
    )
    rmse = np.sqrt(np.mean((res.estimate[interior] - truth[interior]) ** 2))
    assert rmse < 2.0


def test_regression_kriging_uses_covariates():
    rng = np.random.default_rng(3)
    grid_lon = np.linspace(70, 90, 20)
    grid_lat = np.linspace(10, 30, 20)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
    covar_grid = (0.3 * mesh_lon)[..., None]  # (lat, lon, 1)

    obs_lon = rng.uniform(70, 90, 80)
    obs_lat = rng.uniform(10, 30, 80)
    obs_cov = (0.3 * obs_lon)[:, None]
    obs_val = 5.0 + 2.0 * obs_cov[:, 0] + rng.normal(0, 0.5, 80)

    res = regression_kriging(
        obs_lon, obs_lat, obs_val, obs_cov, grid_lon, grid_lat, covar_grid
    )
    assert res.method == "regression_kriging"
    assert res.estimate.shape == (grid_lat.size, grid_lon.size)
    assert np.isfinite(res.estimate).all()


# --------------------------------------------------------------------------- #
# regrid                                                                        #
# --------------------------------------------------------------------------- #
def test_to_grid_resamples_shape_and_values():
    lat = np.linspace(6, 38, 33)
    lon = np.linspace(68, 98, 31)
    data = (lat[:, None] + lon[None, :]).astype("float32")
    da = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    target = make_grid_from_bbox((68, 6, 98, 38), 2.0, name="coarse")
    out = to_grid(da, target, method="bilinear")

    assert out.shape == target.shape
    # Linear field is reproduced by bilinear interpolation.
    mesh_lat, mesh_lon = np.meshgrid(target.lat, target.lon, indexing="ij")
    np.testing.assert_allclose(out.values, mesh_lat + mesh_lon, atol=1e-3)


def test_to_grid_nearest_preserves_dataset():
    lat = np.linspace(6, 38, 17)
    lon = np.linspace(68, 98, 16)
    ds = xr.Dataset(
        {"aod": (("lat", "lon"), np.ones((17, 16), dtype="float32"))},
        coords={"lat": lat, "lon": lon},
    )
    target = make_grid_from_bbox((70, 10, 90, 30), 1.0)
    out = to_grid(ds, target, method="nearest")
    assert isinstance(out, xr.Dataset)
    assert out["aod"].shape == target.shape
    assert np.allclose(out["aod"].values, 1.0)


# --------------------------------------------------------------------------- #
# harmonize: stacking + AOD fusion                                            #
# --------------------------------------------------------------------------- #
def _aod_source(name, lat, lon, base, gap_mask=None, seed=0):
    rng = np.random.default_rng(seed)
    vals = base + 0.02 * rng.standard_normal((1, lat.size, lon.size))
    if gap_mask is not None:
        vals = np.where(gap_mask[None], np.nan, vals)
    ds = xr.Dataset(
        {"aod": (("time", "lat", "lon"), vals.astype("float32"))},
        coords={"time": [0], "lat": lat, "lon": lon},
        attrs={"source": name},
    )
    return ds


def test_stack_sources_provenance_and_uncertainty():
    lat = np.linspace(6, 38, 10)
    lon = np.linspace(68, 98, 12)
    base = (0.1 * lat[:, None] + 0.0 * lon[None, :]).astype("float32")
    g1 = np.zeros((lat.size, lon.size), dtype=bool)
    g1[:3] = True  # maiac missing in a band
    s_maiac = _aod_source("maiac", lat, lon, base, gap_mask=g1, seed=1)
    s_merra = _aod_source("merra2", lat, lon, base + 0.05, seed=2)

    target = make_grid_from_bbox((68, 6, 98, 38), 3.0)
    res = stack_sources(
        [s_maiac, s_merra],
        target,
        variable="aod",
        reliability={"maiac": 1.0, "merra2": 0.6},
    )
    assert res.source_names == ["maiac", "merra2"]
    assert "aod_fused" in res.dataset.data_vars
    # Fused field has no all-source gaps (merra2 is gap-free).
    assert np.isfinite(res.dataset["aod_fused"].values).all()
    # Provenance codes are within range.
    prov = res.provenance.values
    assert prov.min() >= 0 and prov.max() <= 1
    assert res.uncertainty.shape == res.provenance.shape


def test_fuse_aod_rf_imputation():
    lat = np.linspace(6, 38, 14)
    lon = np.linspace(68, 98, 16)
    rng = np.random.default_rng(0)
    truth = 0.3 + 0.01 * (lat[:, None] + lon[None, :])
    truth = np.broadcast_to(truth, (2, lat.size, lon.size)).astype("float64")

    maiac = truth + 0.01 * rng.standard_normal(truth.shape)
    # MAIAC has cloud gaps; other sensors observe (correlated) values there.
    gaps = rng.random(truth.shape) < 0.4
    maiac_g = np.where(gaps, np.nan, maiac)
    viirs = truth + 0.02 * rng.standard_normal(truth.shape)
    merra = truth + 0.03 * rng.standard_normal(truth.shape)

    def _da(v):
        return xr.DataArray(
            v.astype("float32"),
            dims=("time", "lat", "lon"),
            coords={"time": [0, 1], "lat": lat, "lon": lon},
        )

    out = fuse_aod(
        {"maiac": _da(maiac_g), "viirs": _da(viirs), "merra2": _da(merra)},
        use_rf=True,
    )
    assert "aod" in out.data_vars
    assert "aod_uncertainty" in out.data_vars
    assert "aod_provenance" in out.data_vars
    # Fusion removed the MAIAC gaps using the other sensors.
    assert np.isfinite(out["aod"].values).all()
    # Imputed field stays near the truth (sensors are informative).
    err = np.abs(out["aod"].values - truth)
    assert err.mean() < 0.05
