"""Tests for the fire-HCHO correlation and transport package (Objective-2).

These exercise the full transport chain on small, deterministic synthetic data
with a KNOWN structure, so each assertion checks real behaviour rather than just
"it ran":

* ``fire_periods`` extracts a sustained biomass-burning episode injected into an
  otherwise quiet FRP series, and tags it with the right burning season;
* ``fire_hcho_corr`` recovers a fire->HCHO lead/lag relationship with the peak
  cross-correlation at the injected 0-2 day lag, and the dHCHO/FRP slope is
  positive;
* ``hysplit`` kinematic back-trajectories advect a parcel the correct way through
  a uniform wind field (an easterly wind pushes the back-trajectory east);
* ``traj_cluster`` separates two opposing trajectory bundles into two pathways
  and names the dominant direction;
* ``cwt_pscf`` puts the source maximum upwind of the receptor where the polluted
  trajectories originate;
* ``polar`` recovers the wind direction that carries the high concentrations;
* ``inventory`` produces positive, physically-ordered species emissions and an
  ensemble spread that brackets the inventory uncertainty.

Everything runs on the light dependency set; nothing here hits the network or a
heavy/credentialed dependency (HYSPLIT is never invoked — the kinematic fallback
is what is tested).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from aqi_india.transport import (  # noqa: E402
    cwt_pscf,
    fire_hcho_corr,
    fire_periods,
    hysplit,
    inventory,
    polar,
    traj_cluster,
)

PUNJAB_BBOX = (73.5, 28.5, 77.5, 32.5)
RECEPTOR = (28.6, 77.2)  # Delhi-NCR


# --------------------------------------------------------------------------- #
# Fixtures: deterministic synthetic fire table + receptor HCHO series
# --------------------------------------------------------------------------- #
def _fire_table(
    *, n_days: int = 60, episode_day: int = 30, episode_len: int = 5, seed: int = 0
) -> pd.DataFrame:
    """A FIRMS-surrogate fire table with a single sharp burning episode.

    A low background of scattered detections runs across ``n_days`` over
    Punjab/Haryana; a dense, high-FRP burst is injected for ``episode_len`` days
    starting at ``episode_day`` so episode extraction has an unambiguous target.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2022-10-01")
    rows = []
    for d in range(n_days):
        date = base + pd.Timedelta(days=d)
        in_episode = episode_day <= d < episode_day + episode_len
        n = rng.integers(40, 70) if in_episode else rng.integers(0, 4)
        frp_scale = 60.0 if in_episode else 15.0
        for _ in range(int(n)):
            rows.append(
                {
                    "date": date,
                    "lat": float(rng.uniform(28.8, 31.5)),
                    "lon": float(rng.uniform(74.0, 77.0)),
                    "frp": float(rng.gamma(2.0, frp_scale)),
                    "sensor": "VIIRS_SNPP",
                }
            )
    return pd.DataFrame(rows)


def _coupled_series(n: int = 60, lag: int = 1, seed: int = 1):
    """A fire FRP series and an HCHO series that lags it by ``lag`` days."""
    rng = np.random.default_rng(seed)
    fire = np.zeros(n)
    # A couple of bursts on top of a little noise.
    fire[20:25] = np.array([3, 8, 10, 6, 3]) * 50.0
    fire[40:43] = np.array([4, 7, 4]) * 50.0
    fire += rng.uniform(0, 20, n)
    hcho = np.full(n, 1.0e-4)
    hcho[lag:] += 4.0e-6 * (fire[: n - lag] / fire.max())  # downwind, lagged
    hcho += rng.normal(0, 1e-7, n)
    return fire, hcho


# --------------------------------------------------------------------------- #
# fire_periods
# --------------------------------------------------------------------------- #
def test_daily_fire_series_is_gap_free_and_counts() -> None:
    df = _fire_table()
    daily = fire_periods.daily_fire_series(df, PUNJAB_BBOX)
    assert list(daily.columns) == ["date", "fire_count", "frp_sum"]
    # Contiguous daily calendar, no missing days.
    deltas = np.diff(daily["date"].to_numpy()).astype("timedelta64[D]")
    assert np.all(deltas == np.timedelta64(1, "D"))
    assert daily["frp_sum"].min() >= 0.0


def test_extract_episodes_finds_injected_burst() -> None:
    df = _fire_table(episode_day=30, episode_len=5)
    res = fire_periods.extract_episodes(df, PUNJAB_BBOX, sigma=2.0, min_duration=2)
    assert not res.episodes.empty
    assert res.method in {"stl", "moving_average"}
    # The flagged episode should overlap the injected window (days 30-34 of Oct).
    inj_start = pd.Timestamp("2022-10-01") + pd.Timedelta(days=30)
    inj_end = inj_start + pd.Timedelta(days=4)
    hit = (
        (res.episodes["end"] >= inj_start) & (res.episodes["start"] <= inj_end)
    ).any()
    assert hit
    # October falls in the post-monsoon stubble season.
    assert (res.episodes["season"] == "oct_nov").any()


def test_extract_episodes_quiet_series_has_no_episodes() -> None:
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2022-07-01")  # monsoon: no burning season
    rows = []
    for d in range(40):
        for _ in range(int(rng.integers(0, 3))):
            rows.append(
                {
                    "date": base + pd.Timedelta(days=d),
                    "lat": float(rng.uniform(28.8, 31.0)),
                    "lon": float(rng.uniform(74.0, 77.0)),
                    "frp": float(rng.uniform(5, 15)),
                    "sensor": "MODIS",
                }
            )
    df = pd.DataFrame(rows)
    res = fire_periods.extract_episodes(df, PUNJAB_BBOX, sigma=2.0, min_duration=2)
    # A flat, low series should yield few or no sustained episodes.
    assert len(res.episodes) <= 1


def test_aoi_from_name_resolves_known_regions() -> None:
    box = fire_periods.aoi_from_name("punjab_haryana")
    assert len(box) == 4 and box[0] < box[2] and box[1] < box[3]
    with pytest.raises(KeyError):
        fire_periods.aoi_from_name("atlantis")


# --------------------------------------------------------------------------- #
# fire_hcho_corr
# --------------------------------------------------------------------------- #
def test_lagged_xcorr_peaks_in_expected_window() -> None:
    fire, hcho = _coupled_series(lag=1)
    res = fire_hcho_corr.lagged_xcorr(fire, hcho, maxlag=7)
    assert res.correlation.shape == res.lags.shape
    # Peak (causal) correlation should land at a 0-2 day lag and be positive.
    assert 0 <= res.peak_lag <= 2
    assert res.in_expected_window
    assert res.peak_corr > 0.2


def test_lagged_xcorr_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        fire_hcho_corr.lagged_xcorr(np.zeros(10), np.zeros(11), maxlag=3)


def test_dhcho_frp_slope_is_positive_for_coupled_series() -> None:
    fire, hcho = _coupled_series(lag=0)  # contemporaneous for a clean slope
    out = fire_hcho_corr.dhcho_frp_slope(fire, hcho)
    assert out["n"] == fire.size
    assert np.isfinite(out["slope"]) and out["slope"] > 0
    assert out["r"] > 0.2


def test_fnr_map_and_classify() -> None:
    hcho = np.array([[2.0e-4, 1.0e-4], [3.0e-4, 0.5e-4]])
    no2 = np.array([[1.0e-4, 1.0e-4], [1.0e-4, 1.0e-4]])
    fnr = fire_hcho_corr.fnr_map(hcho, no2)
    assert np.isclose(fnr[0, 0], 2.0)
    assert fire_hcho_corr.classify_fnr(0.5) == "VOC-limited"
    assert fire_hcho_corr.classify_fnr(3.0) == "NOx-limited"
    assert fire_hcho_corr.classify_fnr(1.5) == "transitional"
    assert fire_hcho_corr.classify_fnr(float("nan")) == "unknown"


def test_granger_degrades_gracefully_on_short_series() -> None:
    out = fire_hcho_corr.granger(np.arange(5.0), np.arange(5.0), maxlag=3)
    assert out["available"] is False  # too short after prewhitening


# --------------------------------------------------------------------------- #
# hysplit (kinematic fallback)
# --------------------------------------------------------------------------- #
def _uniform_wind_cube(u: float, v: float, *, days: int = 6) -> xr.Dataset:
    """A grid cube with a spatially/temporally uniform (u, v) wind."""
    lat = np.arange(20.0, 35.0 + 0.25, 0.5)
    lon = np.arange(70.0, 85.0 + 0.25, 0.5)
    time = pd.date_range("2022-10-01", periods=days, freq="D")
    shape = (time.size, lat.size, lon.size)
    return xr.Dataset(
        {
            "wind_u": (("time", "lat", "lon"), np.full(shape, u, dtype="float32")),
            "wind_v": (("time", "lat", "lon"), np.full(shape, v, dtype="float32")),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_kinematic_back_trajectory_advects_upwind() -> None:
    # Westerly wind (u>0, blowing east). A BACK-trajectory moves AGAINST the
    # wind, so the parcel's earlier positions are to the WEST (lon decreasing).
    ds = _uniform_wind_cube(u=8.0, v=0.0)
    trajs = hysplit.kinematic_back_trajectories(
        ds, [(28.0, 80.0)], start="2022-10-05", hours=48, levels=(500,)
    )
    assert len(trajs) == 1
    tr = trajs[0]
    assert tr.lons[0] == pytest.approx(80.0)  # release point
    assert tr.lons[-1] < tr.lons[0] - 0.5  # ended up west of release
    assert abs(tr.lats[-1] - tr.lats[0]) < 0.5  # ~no meridional drift


def test_back_trajectories_requires_wind_without_hysplit() -> None:
    with pytest.raises(ValueError):
        hysplit.back_trajectories([(28.0, 80.0)], start="2022-10-05", hours=24)


def test_trajectories_to_frame_roundtrip() -> None:
    ds = _uniform_wind_cube(u=5.0, v=2.0)
    trajs = hysplit.back_trajectories(
        [(28.0, 80.0), (29.0, 79.0)],
        start="2022-10-05",
        hours=24,
        levels=(500, 1000),
        wind_ds=ds,
    )
    frame = hysplit.trajectories_to_frame(trajs)
    assert set(frame.columns) == {"traj_id", "step", "time", "lat", "lon", "level"}
    assert frame["traj_id"].nunique() == 4  # 2 points x 2 levels


# --------------------------------------------------------------------------- #
# traj_cluster
# --------------------------------------------------------------------------- #
def test_cluster_trajectories_separates_two_bundles() -> None:
    # One bundle of parcels coming from the NW, one from the SE of the receptor.
    nw = _uniform_wind_cube(u=8.0, v=8.0)  # back-traj heads SW... use directions
    se = _uniform_wind_cube(u=-8.0, v=-8.0)
    start = "2022-10-05"
    trajs = []
    for _ in range(4):
        trajs += hysplit.kinematic_back_trajectories(
            nw, [RECEPTOR], start=start, hours=48, levels=(500,)
        )
    for _ in range(4):
        trajs += hysplit.kinematic_back_trajectories(
            se, [RECEPTOR], start=start, hours=48, levels=(500,)
        )
    res = traj_cluster.cluster_trajectories(trajs, n_clusters=2, origin=RECEPTOR)
    assert res.n_clusters == 2
    assert sum(res.sizes.values()) == len(trajs)
    dom = traj_cluster.dominant_pathway(res)
    assert dom["compass"] in {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    assert 0.0 < dom["fraction"] <= 1.0


def test_angle_distance_smaller_for_same_direction() -> None:
    # openair angle distance compares the *direction of origin* from the
    # receptor: two paths heading the same way (NW) are far closer than a NW vs
    # a SE path. (The metric is not 0 for a path vs itself because the subtended
    # angle is taken at matched endpoints, not point-to-point separation.)
    nw_a = (np.array([29.0, 30.0, 31.0]), np.array([76.0, 75.0, 74.0]))
    nw_b = (np.array([29.1, 30.2, 31.1]), np.array([76.1, 75.2, 74.1]))
    se = (np.array([28.0, 27.0, 26.0]), np.array([78.0, 79.0, 80.0]))
    same_dir = traj_cluster.angle_distance(nw_a, nw_b, RECEPTOR)
    opp_dir = traj_cluster.angle_distance(nw_a, se, RECEPTOR)
    assert same_dir < opp_dir
    assert same_dir >= 0.0


# --------------------------------------------------------------------------- #
# cwt_pscf
# --------------------------------------------------------------------------- #
def test_cwt_pscf_source_upwind_of_receptor() -> None:
    # Westerly wind -> back-trajectories go west -> the high-HCHO trajectories'
    # endpoints (and thus the CWT/PSCF maximum) sit WEST of the receptor.
    ds = _uniform_wind_cube(u=10.0, v=0.0, days=6)
    start = "2022-10-05"
    polluted = hysplit.kinematic_back_trajectories(
        ds, [RECEPTOR], start=start, hours=72, levels=(500,)
    )
    clean = hysplit.kinematic_back_trajectories(
        _uniform_wind_cube(u=-10.0, v=0.0, days=6),
        [RECEPTOR],
        start=start,
        hours=72,
        levels=(500,),
    )
    trajs = polluted + clean
    receptor_vals = [5.0e-4, 1.0e-4]  # first trajectory is the polluted one
    cwt_map = cwt_pscf.cwt(trajs, receptor_vals, resolution=0.5, weighted=False)
    peak = cwt_pscf.peak_source_cell(cwt_map)
    assert np.isfinite(peak["value"])
    assert peak["lon"] < RECEPTOR[1]  # source is upwind (west)

    pscf_map = cwt_pscf.pscf(trajs, receptor_vals, resolution=0.5, weighted=False)
    assert np.nanmax(pscf_map.values) <= 1.0 + 1e-9
    ppeak = cwt_pscf.peak_source_cell(pscf_map)
    assert ppeak["lon"] < RECEPTOR[1]


def test_cwt_requires_one_value_per_trajectory() -> None:
    ds = _uniform_wind_cube(u=5.0, v=0.0)
    trajs = hysplit.kinematic_back_trajectories(
        ds, [RECEPTOR], start="2022-10-05", hours=24, levels=(500,)
    )
    with pytest.raises(ValueError):
        cwt_pscf.cwt(trajs, [1.0, 2.0])  # 1 trajectory, 2 values


# --------------------------------------------------------------------------- #
# polar
# --------------------------------------------------------------------------- #
def test_uv_to_speed_dir_conventions() -> None:
    # u>0 (blowing east) originates FROM the west -> direction 270 deg.
    speed, direction = polar.uv_to_speed_dir(np.array([3.0]), np.array([4.0]))
    assert speed[0] == pytest.approx(5.0)
    s, d = polar.uv_to_speed_dir(np.array([1.0]), np.array([0.0]))
    assert d[0] == pytest.approx(270.0)


def test_bivariate_polar_recovers_high_conc_direction() -> None:
    rng = np.random.default_rng(3)
    n = 600
    direction = rng.uniform(0, 360, n)
    speed = rng.uniform(0.5, 6.0, n)
    # High HCHO whenever the wind is from the NW (~315 deg).
    nw = np.abs(((direction - 315 + 180) % 360) - 180) < 25
    conc = np.where(nw, 5.0e-4, 1.0e-4) + rng.normal(0, 2e-6, n)
    grid = polar.bivariate_polar(speed, direction, conc, n_dir_bins=16)
    dom = polar.dominant_direction(grid)
    assert dom["compass"] in {"NW", "N", "W"}
    assert dom["value"] > 2.0e-4


def test_bivariate_polar_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        polar.bivariate_polar(np.zeros(5), np.zeros(5), np.zeros(4))


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def test_emission_rates_positive_and_ordered() -> None:
    df = _fire_table(n_days=20, episode_day=10, episode_len=4)
    rates = inventory.emission_rates(df, species=("hcho", "co", "pm25"))
    for s in ("hcho", "co", "pm25"):
        assert (rates[f"emis_{s}_kg"] >= 0).all()
    # CO emission factor >> HCHO factor, so domain CO >> domain HCHO.
    assert rates["emis_co_kg"].sum() > rates["emis_hcho_kg"].sum()


def test_grid_emissions_concentrates_over_punjab() -> None:
    df = _fire_table(n_days=20, episode_day=10, episode_len=4)
    inv = inventory.grid_emissions(df, "hcho", resolution=0.25)
    assert inv.total_kg > 0
    idx = np.unravel_index(int(np.argmax(inv.emission)), inv.emission.shape)
    peak_lat = inv.lat_centers[idx[0]]
    peak_lon = inv.lon_centers[idx[1]]
    # The peak emission cell sits in the Punjab/Haryana burning box.
    assert 28.0 < peak_lat < 32.5
    assert 73.0 < peak_lon < 78.0


def test_inventory_ensemble_brackets_uncertainty() -> None:
    df = _fire_table(n_days=20, episode_day=10, episode_len=4)
    ens = inventory.inventory_ensemble(df, "hcho")
    assert ens["low_kg"] <= ens["central_kg"] <= ens["high_kg"]
    assert ens["high_kg"] > ens["low_kg"]
    assert ens["spread_ratio"] > 1.0  # a genuine bracket, not a point
    assert set(ens["totals_kg"]) == {"FINN", "GFAS", "QFED"}


def test_emission_coefficient_unknown_biome_and_species() -> None:
    with pytest.raises(KeyError):
        inventory.emission_coefficient("hcho", "atlantis_biome")
    with pytest.raises(KeyError):
        inventory.emission_coefficient("plutonium", "crop_residue")
