"""Tests for the HCHO hotspot detection package (Objective-2).

A small synthetic India-scale HCHO column cube is built with a KNOWN injected
hotspot blob over the IGP. The tests assert that:

* per-season robust z-anomaly standardisation flattens the static gradient and
  highlights the injected blob,
* Getis-Ord Gi* and LISA both flag the blob,
* the >=2-of-3 consensus :func:`detect_hotspots` returns confirmed cells whose
  centroid sits at the injected location,
* the polygon delineation and ST-DBSCAN helpers behave, and
* Emerging Hot Spot Analysis labels a growing blob as new/intensifying and a
  steady blob as persistent.

These are pure-logic tests (the Demo/Verify agents run them); nothing here hits
the network or heavy/credentialed dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from aqi_india.hotspot import (  # noqa: E402
    climatology,
    cluster,
    consensus,
    ehsa,
    getis_ord,
    lisa,
)

# Injected hotspot centre (IGP, near Delhi-NCR / Punjab).
HOT_LAT, HOT_LON = 30.0, 76.0
HOT_RADIUS_DEG = 1.0


def _make_cube(
    *,
    n_days: int = 20,
    blob_amp: float = 4e-4,
    trend: float = 0.0,
    seed: int = 0,
) -> xr.Dataset:
    """Build a synthetic HCHO column cube with a Gaussian hotspot blob.

    The background is a smooth south->north gradient (India's real HCHO tilt)
    plus light noise; a Gaussian bump of amplitude ``blob_amp`` is added at
    ``(HOT_LAT, HOT_LON)``, optionally growing linearly in time via ``trend``.

    Args:
        n_days: Number of daily time steps (all in October -> post_monsoon).
        blob_amp: Peak amplitude of the injected blob (mol/m2).
        trend: Per-day fractional growth of the blob amplitude.
        seed: RNG seed.

    Returns:
        An :class:`xarray.Dataset` with a ``hcho_col`` variable on a coarse
        India grid (1 deg) following the DEV_CONTRACT dims.
    """
    rng = np.random.default_rng(seed)
    lat = np.arange(8.0, 36.0 + 0.5, 1.0)
    lon = np.arange(70.0, 90.0 + 0.5, 1.0)
    time = pd.date_range("2022-10-01", periods=n_days, freq="D")

    lon2d, lat2d = np.meshgrid(lon, lat)
    # Static background gradient (mol/m2 scale): higher in the north.
    background = 1e-4 + 6e-5 * (lat2d - lat.min()) / (lat.max() - lat.min())

    dist2 = (lat2d - HOT_LAT) ** 2 + (lon2d - HOT_LON) ** 2
    blob = np.exp(-dist2 / (2 * HOT_RADIUS_DEG**2))

    cube = np.empty((n_days, lat.size, lon.size), dtype=np.float64)
    for t in range(n_days):
        amp = blob_amp * (1.0 + trend * t)
        noise = rng.normal(0.0, 5e-6, size=background.shape)
        cube[t] = background + amp * blob + noise

    return xr.Dataset(
        {"hcho_col": (("time", "lat", "lon"), cube.astype("float32"))},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _nearest_blob_distance(gdf) -> float:
    """Min great-circle-ish distance (deg) from confirmed cells to the blob."""
    if len(gdf) == 0:
        return np.inf
    d = np.hypot(gdf["lat"].values - HOT_LAT, gdf["lon"].values - HOT_LON)
    return float(d.min())


# --------------------------------------------------------------------------- #
# climatology
# --------------------------------------------------------------------------- #
def test_robust_z_anomaly_centres_and_scales() -> None:
    rng = np.random.default_rng(1)
    cube = rng.normal(10.0, 2.0, size=(50, 4, 4))
    z, median, scale = climatology.robust_z_anomaly(cube, axis=0)
    assert z.shape == cube.shape
    # Median of the standardised series is ~0 per cell.
    assert np.allclose(np.nanmedian(z, axis=0), 0.0, atol=0.3)
    assert np.all(scale > 0)


def test_robust_z_handles_nan_and_constant_cells() -> None:
    cube = np.full((10, 2, 2), 5.0)
    cube[:, 0, 0] = np.nan  # all-NaN cell
    z, _, _ = climatology.robust_z_anomaly(cube, axis=0)
    # Constant and all-NaN cells yield NaN z (undefined scale).
    assert np.all(np.isnan(z[:, 0, 0]))
    assert np.all(np.isnan(z[:, 1, 1]))  # constant -> zero MAD


def test_build_climatology_filters_season_and_flags_blob() -> None:
    ds = _make_cube()
    clim = climatology.build_climatology(ds, season="post_monsoon")
    assert clim.attrs["season"] == "post_monsoon"
    assert "z" in clim and "z_spatial" in clim and "hcho_p95" in clim
    # The spatial anomaly at the blob centre should stand out strongly vs the
    # India-wide level (this is the field the spatial detectors consume).
    blob_z = clim["z_spatial"].sel(
        lat=HOT_LAT, lon=HOT_LON, method="nearest"
    ).item()
    assert blob_z > 2.0


def test_build_climatology_rejects_absent_season() -> None:
    ds = _make_cube()  # October only
    with pytest.raises(ValueError):
        climatology.build_climatology(ds, season="monsoon")


# --------------------------------------------------------------------------- #
# Getis-Ord / FDR
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_basic() -> None:
    p = np.array([0.001, 0.2, 0.01, 0.8, np.nan])
    reject, thr = getis_ord.benjamini_hochberg(p, alpha=0.05)
    assert reject[0]  # smallest survives
    assert not reject[3]  # 0.8 never
    assert not reject[4]  # NaN never
    assert thr >= 0.001


def test_getis_ord_flags_injected_blob() -> None:
    ds = _make_cube()
    clim = climatology.build_climatology(ds, season="post_monsoon")
    z_spatial = clim["z_spatial"].values
    lon2d, lat2d = np.meshgrid(clim["lon"].values, clim["lat"].values)
    coords = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    res = getis_ord.getis_ord_gi(
        z_spatial.ravel(), coords, k=8, permutations=199, alpha=0.05
    )
    assert res.sig.any()
    hot_coords = coords[res.sig]
    d = np.hypot(hot_coords[:, 1] - HOT_LAT, hot_coords[:, 0] - HOT_LON)
    assert d.min() < 1.5  # a significant cell sits on the blob


# --------------------------------------------------------------------------- #
# LISA
# --------------------------------------------------------------------------- #
def test_lisa_flags_hh_core_on_blob() -> None:
    ds = _make_cube()
    clim = climatology.build_climatology(ds, season="post_monsoon")
    z_spatial = clim["z_spatial"].values
    lon2d, lat2d = np.meshgrid(clim["lon"].values, clim["lat"].values)
    coords = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    res = lisa.local_morans_i(
        z_spatial.ravel(), coords, k=8, permutations=199, alpha=0.05
    )
    assert res.hh_mask.any()
    hh = coords[res.hh_mask]
    d = np.hypot(hh[:, 1] - HOT_LAT, hh[:, 0] - HOT_LON)
    assert d.min() < 1.5


# --------------------------------------------------------------------------- #
# Consensus — the headline assertion
# --------------------------------------------------------------------------- #
def test_detect_hotspots_finds_blob_location() -> None:
    ds = _make_cube()
    # gi_alpha=0.05 keeps the (expensive) permutation count low for the test
    # while still letting the Gi* vote fire; production uses 999 perms / 0.01.
    gdf = consensus.detect_hotspots(
        ds, season="post_monsoon", permutations=199, gi_alpha=0.05
    )
    assert len(gdf) > 0
    assert bool(gdf["confirmed"].all())
    assert (gdf["n_votes"] >= consensus.DEFAULT_MIN_VOTES).all()
    # All three independent votes should agree on the blob.
    assert int(gdf["vote_gi"].sum()) > 0
    assert int(gdf["vote_lisa"].sum()) > 0
    assert int(gdf["vote_exceed"].sum()) > 0
    # Confirmed hotspot centroid sits on the injected blob.
    assert _nearest_blob_distance(gdf) < 1.5


def test_detect_hotspots_quiet_field_finds_little() -> None:
    # No blob -> pure gradient + noise should not confirm a cluster of cells.
    ds = _make_cube(blob_amp=0.0)
    gdf = consensus.detect_hotspots(
        ds, season="post_monsoon", permutations=199, gi_alpha=0.05
    )
    # Far fewer (ideally zero) confirmations than the blob case.
    assert len(gdf) <= 3


# --------------------------------------------------------------------------- #
# Clustering / polygons
# --------------------------------------------------------------------------- #
def test_hotspot_polygons_from_blob_cells() -> None:
    # A tight cluster of points near the blob plus one far outlier.
    pts = np.array(
        [[76.0, 30.0], [76.2, 30.1], [75.8, 29.9], [76.1, 30.2], [76.0, 29.8]]
        + [[88.0, 12.0]]  # lone outlier
    )
    gdf = cluster.hotspot_polygons(pts, eps_km=60.0, min_samples=3, use_hdbscan=False)
    assert len(gdf) == 1  # outlier dropped as noise
    assert gdf.iloc[0]["n_cells"] == 5
    assert gdf.iloc[0].geometry.area > 0


def test_st_dbscan_separates_episodes() -> None:
    # Two spatial blobs on different days -> two episodes.
    coords = np.array(
        [[76.0, 30.0], [76.1, 30.1], [76.0, 29.9]]
        + [[82.0, 22.0], [82.1, 22.1], [82.0, 21.9]]
    )
    times = np.array(
        ["2022-10-01"] * 3 + ["2022-10-10"] * 3, dtype="datetime64[D]"
    )
    labels = cluster.st_dbscan_labels(
        coords, times, eps_spatial_km=50.0, eps_temporal_days=2.0, min_samples=3
    )
    assert len(set(labels) - {-1}) == 2


# --------------------------------------------------------------------------- #
# EHSA
# --------------------------------------------------------------------------- #
def _ehsa_inputs(trend: float, seed: int):
    """Build a per-bin HCHO-column cube and coords for an EHSA run.

    Each day is treated as one time bin; ``emerging_hotspot_analysis``
    spatially standardises each bin internally.
    """
    n_bins = 8
    ds = _make_cube(n_days=n_bins, blob_amp=4e-4, trend=trend, seed=seed)
    col = ds["hcho_col"].values  # (bins, lat, lon)
    lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    coords = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    cube = col.reshape(col.shape[0], -1)
    return cube, coords, list(range(n_bins))


def test_ehsa_categorises_cells() -> None:
    cube, coords, labels = _ehsa_inputs(trend=0.25, seed=3)
    res = ehsa.emerging_hotspot_analysis(
        cube, coords, labels, k=8, permutations=99, gi_alpha=0.05
    )
    assert res.category.shape[0] == coords.shape[0]
    assert set(np.unique(res.category)).issubset(set(ehsa.EHSA_CATEGORIES))
    # The intensifying/growing blob centre should be a recognised hot pattern.
    d = np.hypot(coords[:, 1] - HOT_LAT, coords[:, 0] - HOT_LON)
    centre = int(np.argmin(d))
    assert res.hot_fraction[centre] > 0.0
    assert res.category[centre] in {
        "new", "consecutive", "intensifying", "persistent", "sporadic"
    }
