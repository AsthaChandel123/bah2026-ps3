"""Deterministic synthetic-data generator for the BAH 2026 PS3 demo.

This module fabricates a small, physically-plausible **synthetic** India dataset
that stands in for the real satellite / reanalysis / CPCB sources the
``aqi_india.ingest`` adapters target. Its sole purpose is to make the whole
pipeline runnable, deterministic and *verifiable offline* using only the light
dependency set (numpy / pandas / scipy / xarray / geopandas / shapely / h3) —
no Google Earth Engine credentials, no multi-gigabyte downloads, no GPU.

The generated fields embed **known structure** so that downstream detection and
validation can be checked against ground truth:

* a realistic Indo-Gangetic-Plain (IGP) pollution gradient — high over the
  northern plains (~26-30N / 75-88E), low over the peninsula and oceans;
* an annual seasonal cycle (post-monsoon / winter pollution build-up) and a
  crude diurnal-ish boundary-layer-height (BLH) modulation;
* prevailing north-west -> south-east winds (the IGP transport corridor);
* spatially-correlated noise (Gaussian-blurred white noise) rather than pixel
  white noise, so gap-fill / smoothing behave realistically;
* **cloud gaps** — 30-50% of the satellite *column* fields are randomly masked
  to ``NaN`` per day to exercise the gap-fill stage;
* **fire clusters** in Punjab/Haryana during the stubble-burning window plus a
  separate forest-fire cluster, with a **downwind HCHO enhancement** at a 0-2
  day lag so that the fire->HCHO correlation and the HCHO hotspots are real and
  locatable;
* **CPCB-like stations** whose surface concentrations are a *known* nonlinear
  function of the co-located columns + BLH + RH (so a model can actually learn
  the satellite->surface mapping).

Everything is seeded; calling any function twice with the same arguments yields
bit-identical output. The default 60-day / 0.25 deg run completes in a few
seconds.

The on-disk schema produced by :func:`generate_all` matches ``DEV_CONTRACT`` §6
exactly (dims, variable names, units, file locations) so every other module
agent's code composes against it unchanged.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..features.h3_index import cells_for_arrays
from ..utils.geo import (
    FOREST_BELT_BBOX,
    INDIA_BBOX,
    PUNJAB_HARYANA_BBOX,
)
from ..utils.io import (
    ensure_dir,
    save_geojson,
    save_geoparquet,
    save_netcdf,
    save_parquet,
)
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd
    import xarray as xr

logger = get_logger("sim.synthetic")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Master seed; every stochastic step derives a child generator from this.
DEFAULT_SEED: int = 42

#: Gridded satellite/met variables written to ``grid.nc`` (DEV_CONTRACT §6.1).
GRID_VARS: tuple[str, ...] = (
    "aod",
    "no2_col",
    "so2_col",
    "co_col",
    "o3_col",
    "hcho_col",
    "blh",
    "rh",
    "wind_u",
    "wind_v",
    "t2m",
    "ssrd",
)

#: Satellite *column* variables that suffer cloud gaps (set NaN per day).
COLUMN_VARS: tuple[str, ...] = (
    "aod",
    "no2_col",
    "so2_col",
    "co_col",
    "o3_col",
    "hcho_col",
)

#: CPCB surface pollutants generated at each station (CO in mg/m3, rest ug/m3).
SURFACE_POLLUTANTS: tuple[str, ...] = ("pm25", "pm10", "no2", "so2", "co", "o3")

#: Centre (lon, lat) of the Indo-Gangetic-Plain pollution maximum.
_IGP_CENTRE: tuple[float, float] = (80.0, 28.0)

#: Approximate physical scales used so columns sit in believable ``mol/m2`` /
#: AOD ranges. These are illustrative, not retrieval-accurate.
_COL_SCALE: dict[str, float] = {
    "aod": 1.0,  # unitless
    "no2_col": 8.0e-5,  # mol/m2 (tropospheric NO2)
    "so2_col": 4.0e-4,  # mol/m2
    "co_col": 3.0e-2,  # mol/m2 (total column CO ~ 2e-2..4e-2)
    "o3_col": 1.5e-1,  # mol/m2 (total O3 column ~ DU*4.46e-4)
    "hcho_col": 2.0e-4,  # mol/m2 (tropospheric HCHO)
}


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #
def _spawn(seed: int, stream: int) -> np.random.Generator:
    """Return an independent, reproducible RNG for a named sub-stream."""
    return np.random.default_rng([seed, stream])


def _coerce_date(start_date: str | date | datetime) -> pd.Timestamp:
    """Normalise the accepted date inputs to a midnight ``pd.Timestamp``."""
    ts = pd.Timestamp(start_date)
    return ts.normalize()


def _gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian blur to a 2-D field.

    Uses :func:`scipy.ndimage.gaussian_filter` when SciPy is available and falls
    back to an FFT-free identity when ``sigma <= 0``. SciPy is part of the light
    dependency set, so the import is module-local but reliable.

    Args:
        field: 2-D array (lat, lon).
        sigma: Blur standard deviation in grid cells.

    Returns:
        The blurred field, same shape and dtype family as the input.
    """
    if sigma <= 0:
        return field
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(field, sigma=sigma, mode="nearest")


def _correlated_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """Generate spatially-correlated, zero-mean, unit-ish noise.

    White noise is blurred with a Gaussian kernel and re-standardised so the
    output has ~unit standard deviation regardless of ``sigma``.

    Args:
        rng: Source generator.
        shape: (nlat, nlon).
        sigma: Spatial correlation length in grid cells.

    Returns:
        2-D float64 array of correlated noise.
    """
    white = rng.standard_normal(shape)
    blurred = _gaussian_blur(white, sigma)
    std = float(blurred.std())
    if std > 0:
        blurred = blurred / std
    return blurred


def _make_axes(
    res: float, bbox: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Build ascending 1-D lat/lon coordinate axes for ``bbox`` at ``res``."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = np.arange(min_lon, max_lon + 0.5 * res, res, dtype=np.float64)
    lat = np.arange(min_lat, max_lat + 0.5 * res, res, dtype=np.float64)
    return lat, lon


def _igp_gradient(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """Return a 0..1 pollution-burden weight peaking over the IGP.

    The weight is a 2-D anisotropic Gaussian centred on :data:`_IGP_CENTRE`,
    stretched east-west along the plains, multiplied by a land mask that decays
    over the oceans / far peninsula so coastal and marine cells stay clean.

    Args:
        lat2d: 2-D latitude grid (lat, lon).
        lon2d: 2-D longitude grid (lat, lon).

    Returns:
        2-D float array in [0, 1].
    """
    lon0, lat0 = _IGP_CENTRE
    # Anisotropic: broad in longitude (plains run W->E), tight in latitude.
    sig_lon, sig_lat = 9.0, 3.0
    g = np.exp(
        -(((lon2d - lon0) / sig_lon) ** 2 + ((lat2d - lat0) / sig_lat) ** 2)
    )
    # A second, weaker urban node over central/western India for variety.
    g2 = 0.35 * np.exp(
        -(((lon2d - 73.0) / 4.0) ** 2 + ((lat2d - 19.0) / 3.0) ** 2)
    )
    # Crude land/sea + peninsula taper: pollution fades south of ~12N and at
    # the extreme west (Arabian Sea) / east-coast edges.
    south_taper = 1.0 / (1.0 + np.exp(-(lat2d - 10.0) * 0.6))
    weight = np.clip(g + g2, 0.0, 1.5) * south_taper
    return np.clip(weight, 0.0, 1.0)


def _seasonal_factor(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Annual pollution seasonality: high in winter (DJF), low in monsoon (JAS).

    Args:
        timestamps: Daily time index.

    Returns:
        1-D array (n_days,) of multiplicative factors centred on ~1.0.
    """
    doy = timestamps.dayofyear.to_numpy().astype(np.float64)
    # Peak near day-of-year ~330 (late Nov), trough near ~200 (mid-Jul).
    phase = 2.0 * np.pi * (doy - 330.0) / 365.25
    return 1.0 + 0.6 * np.cos(phase)


# --------------------------------------------------------------------------- #
# 1. Gridded fields                                                            #
# --------------------------------------------------------------------------- #
def make_grid(
    start_date: str | date | datetime,
    n_days: int,
    res: float = 0.25,
    bbox: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    seed: int = DEFAULT_SEED,
    cloud_frac: tuple[float, float] = (0.30, 0.50),
) -> "xr.Dataset":
    """Generate the synthetic gridded satellite + meteorology cube.

    Produces an :class:`xarray.Dataset` with dims ``(time, lat, lon)`` and the
    twelve data variables of DEV_CONTRACT §6.1, all ``float32``. The fields
    encode an IGP pollution gradient, an annual seasonal cycle, a diurnal-ish
    BLH, prevailing NW->SE winds, and spatially-correlated noise. The six
    satellite *column* variables additionally receive per-day **cloud gaps**
    (random contiguous-ish NaN masks covering ``cloud_frac`` of the domain).

    Args:
        start_date: First day (inclusive); ``str`` / ``date`` / ``datetime``.
        n_days: Number of daily time steps (>= 1).
        res: Grid resolution in degrees (default 0.25).
        bbox: Domain as ``(min_lon, min_lat, max_lon, max_lat)``.
        seed: Master RNG seed for full determinism.
        cloud_frac: ``(low, high)`` fraction of pixels to mask as cloud per day.

    Returns:
        The synthetic ``xarray.Dataset`` (a ``ground_truth`` copy of each clean
        column is preserved in the variable attributes' provenance, not as
        separate vars, to keep the schema exact).

    Raises:
        ValueError: If ``n_days < 1`` or ``res <= 0``.
    """
    import xarray as xr

    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    if res <= 0:
        raise ValueError("res must be > 0")

    t0 = _coerce_date(start_date)
    times = pd.date_range(t0, periods=n_days, freq="D")
    lat, lon = _make_axes(res, bbox)
    nlat, nlon = lat.size, lon.size
    lon2d, lat2d = np.meshgrid(lon, lat)  # both (nlat, nlon)

    burden = _igp_gradient(lat2d, lon2d)  # (nlat, nlon) in [0,1]
    season = _seasonal_factor(times)  # (n_days,)
    blur_sigma = max(1.0, 0.5 / res)  # ~2 cells at 0.25 deg

    # Pre-allocate output arrays.
    out = {v: np.empty((n_days, nlat, nlon), dtype=np.float32) for v in GRID_VARS}

    rng_noise = _spawn(seed, 1)
    rng_cloud = _spawn(seed, 2)
    lo_frac, hi_frac = cloud_frac

    # Latitude-driven base temperature (warmer south) reused across days.
    base_t2m = 300.0 - 0.6 * (lat2d - 8.0)  # K, ~300K south -> ~282K north

    for ti, ts in enumerate(times):
        s = float(season[ti])
        # Independent correlated-noise realisation per day.
        n_aer = _correlated_noise(rng_noise, (nlat, nlon), blur_sigma)
        n_gas = _correlated_noise(rng_noise, (nlat, nlon), blur_sigma)
        n_met = _correlated_noise(rng_noise, (nlat, nlon), blur_sigma)

        # --- Aerosol optical depth -------------------------------------- #
        aod = 0.08 + 0.9 * burden * s + 0.08 * n_aer
        aod = np.clip(aod, 0.01, 3.0)
        out["aod"][ti] = aod * _COL_SCALE["aod"]

        # --- Trace-gas columns (track the burden + gas noise) ----------- #
        base_gas = burden * s
        out["no2_col"][ti] = (
            _COL_SCALE["no2_col"] * (0.15 + 1.0 * base_gas + 0.12 * n_gas)
        ).clip(min=0.0)
        out["so2_col"][ti] = (
            _COL_SCALE["so2_col"] * (0.10 + 0.7 * base_gas + 0.15 * n_gas)
        ).clip(min=0.0)
        out["co_col"][ti] = (
            _COL_SCALE["co_col"] * (0.6 + 0.8 * base_gas + 0.08 * n_gas)
        ).clip(min=0.0)
        # O3 total column is large + smooth, weak pollution dependence.
        out["o3_col"][ti] = (
            _COL_SCALE["o3_col"] * (0.9 + 0.15 * base_gas + 0.04 * n_gas)
        ).clip(min=0.0)
        # HCHO baseline biogenic field; fire enhancement added later.
        out["hcho_col"][ti] = (
            _COL_SCALE["hcho_col"] * (0.3 + 0.5 * base_gas + 0.10 * n_gas)
        ).clip(min=0.0)

        # --- Meteorology (gap-free, never clouded) ---------------------- #
        # BLH proxy: lower where pollution accumulates (winter-stable IGP),
        # higher where warmer, with a multi-day "diurnal-ish" oscillation that
        # mimics the day-to-day ventilation cycle.
        doy = ts.dayofyear
        diurnal = 200.0 * np.sin(2.0 * np.pi * (doy % 7) / 7.0)
        blh = (
            900.0
            - 350.0 * burden * s
            + diurnal
            + 120.0 * n_met
            + 0.5 * (300.0 - base_t2m)  # higher BLH where warmer
        )
        out["blh"][ti] = np.clip(blh, 120.0, 2500.0)

        # Relative humidity: higher in monsoon (inverse of pollution season).
        rh = 45.0 + 25.0 * (2.0 - s) + 8.0 * n_met
        out["rh"][ti] = np.clip(rh, 5.0, 100.0)

        # Prevailing NW -> SE flow: u>0 (eastward), v<0 (southward), with a
        # mild spatial perturbation so transport is non-uniform.
        out["wind_u"][ti] = (2.5 + 1.5 * n_met).astype(np.float32)
        out["wind_v"][ti] = (-2.0 + 1.5 * np.roll(n_met, 3, axis=1)).astype(
            np.float32
        )

        # 2 m temperature with seasonal swing (cooler in the pollution season).
        t2m = base_t2m - 6.0 * (s - 1.0) + 1.5 * n_met
        out["t2m"][ti] = t2m.astype(np.float32)

        # Surface solar radiation downwards (W/m2), reduced by aerosol load.
        ssrd = 260.0 - 40.0 * (aod / (aod + 0.5)) + 20.0 * n_met
        out["ssrd"][ti] = np.clip(ssrd, 40.0, 360.0)

        # --- Cloud gaps on the column fields ---------------------------- #
        frac = float(rng_cloud.uniform(lo_frac, hi_frac))
        mask = _cloud_mask(rng_cloud, (nlat, nlon), frac, blur_sigma)
        for cv in COLUMN_VARS:
            out[cv][ti][mask] = np.nan

    ds = xr.Dataset(
        {v: (("time", "lat", "lon"), out[v]) for v in GRID_VARS},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    _attach_attrs(ds, res, seed)
    logger.info(
        "make_grid: %d days x %d lat x %d lon (%.2f deg), cloud gaps %.0f-%.0f%%",
        n_days,
        nlat,
        nlon,
        res,
        lo_frac * 100,
        hi_frac * 100,
    )
    return ds


def _cloud_mask(
    rng: np.random.Generator,
    shape: tuple[int, int],
    frac: float,
    blur_sigma: float,
) -> np.ndarray:
    """Build a spatially-coherent boolean cloud mask covering ~``frac`` cells.

    A smooth random field is thresholded at the ``frac`` quantile so the masked
    region forms contiguous cloud-like blobs rather than salt-and-pepper holes.

    Args:
        rng: Source generator.
        shape: (nlat, nlon).
        frac: Target fraction of cells to mask, in (0, 1).
        blur_sigma: Spatial smoothness of the cloud field.

    Returns:
        Boolean array; ``True`` where data should be masked (clouded).
    """
    field = _gaussian_blur(rng.standard_normal(shape), blur_sigma * 1.5)
    thresh = np.quantile(field, frac)
    return field < thresh


def _attach_attrs(ds: "xr.Dataset", res: float, seed: int) -> None:
    """Attach CF-ish units/long-name metadata and global provenance attrs."""
    units = {
        "aod": "1",
        "no2_col": "mol m-2",
        "so2_col": "mol m-2",
        "co_col": "mol m-2",
        "o3_col": "mol m-2",
        "hcho_col": "mol m-2",
        "blh": "m",
        "rh": "%",
        "wind_u": "m s-1",
        "wind_v": "m s-1",
        "t2m": "K",
        "ssrd": "W m-2",
    }
    longname = {
        "aod": "MAIAC-like aerosol optical depth at 550 nm",
        "no2_col": "tropospheric NO2 column",
        "so2_col": "SO2 column",
        "co_col": "CO column",
        "o3_col": "O3 column",
        "hcho_col": "tropospheric HCHO column",
        "blh": "boundary layer height",
        "rh": "relative humidity",
        "wind_u": "10 m eastward wind",
        "wind_v": "10 m northward wind",
        "t2m": "2 m temperature",
        "ssrd": "surface solar radiation downwards",
    }
    for v in GRID_VARS:
        ds[v].attrs["units"] = units[v]
        ds[v].attrs["long_name"] = longname[v]
    ds.attrs.update(
        {
            "title": "SYNTHETIC India AQI/HCHO demo cube (aqi_india.sim)",
            "source": "aqi_india.sim.synthetic.make_grid",
            "note": (
                "Synthetic data standing in for MAIAC/TROPOMI/INSAT/reanalysis; "
                "for offline demo and verification only — NOT real observations."
            ),
            "resolution_deg": float(res),
            "seed": int(seed),
            "Conventions": "CF-1.8 (approximate)",
        }
    )


# --------------------------------------------------------------------------- #
# 2. Fires                                                                      #
# --------------------------------------------------------------------------- #
def make_fires(
    grid: "xr.Dataset",
    season: str = "oct_nov",
    *,
    seed: int = DEFAULT_SEED,
    n_punjab: int = 600,
    n_forest: int = 150,
) -> "gpd.GeoDataFrame":
    """Generate synthetic FIRMS-like fire detections.

    Creates a dense stubble-burning cluster over Punjab/Haryana
    (:data:`PUNJAB_HARYANA_BBOX`, ~29-31N / 74-77E) concentrated within the
    requested burning window, plus a smaller forest-fire cluster in the
    forest belt for variety. Each detection carries an FRP, a sensor label, an
    acquisition date and its H3 res-7 cell (DEV_CONTRACT §6.3).

    Args:
        grid: The grid Dataset from :func:`make_grid` (defines the time span).
        season: Burning window selector — ``"oct_nov"`` (post-monsoon stubble,
            default) or ``"apr_may"`` (pre-monsoon forest fires). Detections are
            weighted toward this window; out-of-window days get sparse activity.
        seed: RNG seed.
        n_punjab: Approximate number of Punjab/Haryana stubble detections.
        n_forest: Approximate number of forest-belt detections.

    Returns:
        A point :class:`geopandas.GeoDataFrame` (EPSG:4326) with columns
        ``date, lat, lon, frp, sensor, h3_res7`` and a ``geometry`` column.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    times = pd.to_datetime(grid["time"].values)
    rng = _spawn(seed, 10)

    window = _burning_window_mask(times, season)
    if not window.any():
        # No in-window days in this span: fall back to all days, lightly.
        window = np.ones(len(times), dtype=bool)

    sensors = np.array(
        ["VIIRS_SNPP", "VIIRS_NOAA20", "MODIS"], dtype=object
    )
    sensor_p = np.array([0.5, 0.35, 0.15])

    frames: list[dict[str, np.ndarray]] = []
    frames.append(
        _sample_fire_cluster(
            rng,
            times,
            window,
            bbox=PUNJAB_HARYANA_BBOX,
            n=n_punjab,
            frp_scale=45.0,
            sensors=sensors,
            sensor_p=sensor_p,
        )
    )
    # Forest cluster: pick a compact box inside the forest belt (central India).
    forest_box = (78.0, 21.0, 82.0, 24.0)
    _ = FOREST_BELT_BBOX  # documented provenance of the forest region
    frames.append(
        _sample_fire_cluster(
            rng,
            times,
            window,
            bbox=forest_box,
            n=n_forest,
            frp_scale=25.0,
            sensors=sensors,
            sensor_p=sensor_p,
        )
    )

    rec = {k: np.concatenate([f[k] for f in frames]) for k in frames[0]}
    h3 = cells_for_arrays(rec["lat"], rec["lon"], 7)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(rec["date"]),
            "lat": rec["lat"].astype(np.float64),
            "lon": rec["lon"].astype(np.float64),
            "frp": rec["frp"].astype(np.float64),
            "sensor": rec["sensor"].astype(object),
            "h3_res7": h3.astype(object),
        }
    )
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    geom = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    logger.info(
        "make_fires: %d detections (%s window), FRP %.0f-%.0f MW",
        len(gdf),
        season,
        gdf["frp"].min(),
        gdf["frp"].max(),
    )
    return gdf


def _burning_window_mask(times: pd.DatetimeIndex, season: str) -> np.ndarray:
    """Boolean per-day mask for the requested biomass-burning window."""
    month = times.month.to_numpy()
    if season == "oct_nov":
        return np.isin(month, (10, 11))
    if season == "apr_may":
        return np.isin(month, (4, 5))
    raise ValueError(f"unknown season {season!r}; use 'oct_nov' or 'apr_may'")


def _sample_fire_cluster(
    rng: np.random.Generator,
    times: pd.DatetimeIndex,
    window: np.ndarray,
    bbox: tuple[float, float, float, float],
    n: int,
    frp_scale: float,
    sensors: np.ndarray,
    sensor_p: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sample ``n`` fire points inside ``bbox`` on in-window days.

    Points are drawn from a 2-D Gaussian centred in the box (so the cluster is
    spatially compact, like a real burning hotspot) and clipped to the box. FRP
    follows an exponential distribution (many small + few large fires).

    Returns:
        Dict of equal-length arrays: ``date, lat, lon, frp, sensor``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    cx = 0.5 * (min_lon + max_lon)
    cy = 0.5 * (min_lat + max_lat)
    sx = 0.30 * (max_lon - min_lon)
    sy = 0.30 * (max_lat - min_lat)

    lon = np.clip(rng.normal(cx, sx, n), min_lon, max_lon)
    lat = np.clip(rng.normal(cy, sy, n), min_lat, max_lat)
    frp = rng.exponential(frp_scale, n) + 5.0

    in_days = np.flatnonzero(window)
    # Weight days within the window by a smooth peak so activity rises and falls.
    centre = in_days.mean() if in_days.size else 0.0
    w = np.exp(-0.5 * ((in_days - centre) / max(in_days.size / 4.0, 1.0)) ** 2)
    w = w / w.sum()
    day_idx = rng.choice(in_days, size=n, p=w)
    dates = times.to_numpy()[day_idx]

    sensor = rng.choice(sensors, size=n, p=sensor_p)
    return {
        "date": dates,
        "lat": lat,
        "lon": lon,
        "frp": frp,
        "sensor": sensor,
    }


# --------------------------------------------------------------------------- #
# 3. Fire -> HCHO injection                                                     #
# --------------------------------------------------------------------------- #
def inject_fire_hcho(
    grid: "xr.Dataset",
    fires: "gpd.GeoDataFrame",
    *,
    lag_days: int = 1,
    radius_deg: float = 1.5,
    strength: float = 6.0,
) -> "xr.Dataset":
    """Add a downwind HCHO enhancement from fire clusters into the grid.

    For each fire-day, the per-cell FRP total is smeared **downwind** (along the
    domain-mean wind vector for that day) and added to ``hcho_col`` on the same
    day and the next ``lag_days`` days. This makes the fire->HCHO correlation and
    the HCHO hotspots both *real* and *locatable*, anchored to the Punjab/Haryana
    and forest clusters from :func:`make_fires`.

    The enhancement respects existing cloud-gap NaNs (a clouded HCHO pixel stays
    NaN) so the gap-fill stage still has work to do.

    Args:
        grid: Dataset from :func:`make_grid` (modified copy is returned).
        fires: Fire detections from :func:`make_fires`.
        lag_days: Number of days the plume persists downwind (0-2 typical).
        radius_deg: Gaussian smear radius of the plume in degrees.
        strength: Peak enhancement as a multiple of the HCHO scale per unit
            normalised daily FRP.

    Returns:
        A copy of ``grid`` with ``hcho_col`` enhanced and provenance attrs added
        recording the injected cluster centroids for verification.
    """
    ds = grid.copy(deep=True)
    times = pd.to_datetime(ds["time"].values)
    lat = ds["lat"].values
    lon = ds["lon"].values
    res = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 0.25
    sigma_cells = max(radius_deg / res, 1.0)

    # Aggregate FRP onto the grid per day: (n_days, nlat, nlon).
    frp_grid = _rasterise_fire_frp(fires, times, lat, lon)
    if frp_grid.max() > 0:
        frp_norm = frp_grid / frp_grid.max()
    else:
        frp_norm = frp_grid

    hcho = ds["hcho_col"].values  # (time, lat, lon), may contain NaN
    nan_mask = np.isnan(hcho)
    enh_total = np.zeros_like(hcho)

    nlon = lon.size
    for ti in range(times.size):
        if frp_norm[ti].max() <= 0:
            continue
        # Downwind shift: convert mean wind (m/s) to a cell offset.
        u = float(np.nanmean(ds["wind_u"].values[ti]))
        v = float(np.nanmean(ds["wind_v"].values[ti]))
        # Smear the source, then translate downwind, then deposit with lag.
        smeared = _gaussian_blur(frp_norm[ti], sigma_cells)
        shift_x = int(np.clip(round(u * 0.5), -nlon // 4, nlon // 4))
        shift_y = int(np.clip(round(v * 0.5), -lat.size // 4, lat.size // 4))
        plume = np.roll(np.roll(smeared, shift_x, axis=1), shift_y, axis=0)
        # Re-normalise the plume to keep peak ~1 after blurring.
        pmax = plume.max()
        if pmax > 0:
            plume = plume / pmax

        for dl in range(lag_days + 1):
            tj = ti + dl
            if tj >= times.size:
                break
            decay = 0.6**dl
            enh_total[tj] += strength * _COL_SCALE["hcho_col"] * plume * decay

    hcho = hcho + enh_total
    hcho[nan_mask] = np.nan  # keep cloud gaps as gaps
    ds["hcho_col"].values[...] = hcho.astype(np.float32)

    # Record cluster centroids (where downwind HCHO maxima should appear).
    ds["hcho_col"].attrs["fire_injection"] = "downwind FRP-weighted enhancement"
    ds.attrs["fire_hcho_lag_days"] = int(lag_days)
    if len(fires):
        ds.attrs["fire_cluster_centroid_lat"] = float(fires["lat"].mean())
        ds.attrs["fire_cluster_centroid_lon"] = float(fires["lon"].mean())
    logger.info(
        "inject_fire_hcho: peak enhancement %.2e mol/m2 over %d days (lag=%d)",
        float(enh_total.max()),
        times.size,
        lag_days,
    )
    return ds


def _rasterise_fire_frp(
    fires: "gpd.GeoDataFrame",
    times: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Bin fire FRP onto the (time, lat, lon) grid by nearest cell."""
    out = np.zeros((times.size, lat.size, lon.size), dtype=np.float64)
    if len(fires) == 0:
        return out
    time_index = {pd.Timestamp(t).normalize(): i for i, t in enumerate(times)}
    res_lat = float(np.abs(np.diff(lat)).mean()) if lat.size > 1 else 0.25
    res_lon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 0.25
    lat0, lon0 = lat[0], lon[0]
    for _, row in fires.iterrows():
        ti = time_index.get(pd.Timestamp(row["date"]).normalize())
        if ti is None:
            continue
        iy = int(round((row["lat"] - lat0) / res_lat))
        ix = int(round((row["lon"] - lon0) / res_lon))
        if 0 <= iy < lat.size and 0 <= ix < lon.size:
            out[ti, iy, ix] += float(row["frp"])
    return out


# --------------------------------------------------------------------------- #
# 4. Stations                                                                   #
# --------------------------------------------------------------------------- #
def make_stations(
    grid: "xr.Dataset",
    n: int = 120,
    *,
    seed: int = DEFAULT_SEED,
) -> "gpd.GeoDataFrame":
    """Generate synthetic CPCB-like stations with daily surface concentrations.

    Stations are placed **denser in the IGP / cities and sparser elsewhere**
    (DEV_CONTRACT §6.2). Each station-day's surface pollutants are a *known
    nonlinear* function of the co-located satellite columns, BLH and RH plus
    noise, so a model can learn the satellite->surface mapping:

    * ``pm25`` ~ AOD scaled by BLH (PBL normalisation) and RH (hygroscopic
      growth), i.e. ``PM2.5 ∝ AOD / BLH * f(RH)`` — the textbook relationship;
    * ``pm10`` ~ a coarse-mode multiple of ``pm25``;
    * ``no2 / so2 / co`` ~ monotone (sub-linear) functions of their columns;
    * ``o3`` ~ photochemistry driven by SSRD, temperature and NO2.

    CO is reported in ``mg/m3``; all other pollutants in ``ug/m3``.

    Args:
        grid: Dataset from :func:`make_grid` (and ideally after
            :func:`inject_fire_hcho`); supplies the predictor fields.
        n: Number of stations.
        seed: RNG seed.

    Returns:
        A point :class:`geopandas.GeoDataFrame` (EPSG:4326), one row per
        station-day, with columns ``station_id, name, lat, lon, h3_res7,
        region, time, pm25, pm10, no2, so2, co, o3`` and a ``geometry`` column.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    rng = _spawn(seed, 20)
    lat = grid["lat"].values
    lon = grid["lon"].values
    lat_lo, lat_hi = float(lat.min()), float(lat.max())
    lon_lo, lon_hi = float(lon.min()), float(lon.max())

    # --- Station placement: mixture of IGP-dense + uniform-sparse ------- #
    n_igp = int(round(0.6 * n))
    n_rest = n - n_igp
    igp_lon = np.clip(rng.normal(80.0, 5.0, n_igp), lon_lo, lon_hi)
    igp_lat = np.clip(rng.normal(28.0, 1.8, n_igp), lat_lo, lat_hi)
    rest_lon = rng.uniform(lon_lo, lon_hi, n_rest)
    rest_lat = rng.uniform(lat_lo + 2.0, lat_hi, n_rest)  # avoid far-south ocean
    s_lon = np.concatenate([igp_lon, rest_lon])
    s_lat = np.concatenate([igp_lat, rest_lat])

    station_ids = [f"SYN{ i:04d}" for i in range(len(s_lat))]
    names = [f"Station-{i:04d}" for i in range(len(s_lat))]
    regions = np.array([_region_for(la, lo) for la, lo in zip(s_lat, s_lon)])
    h3_static = cells_for_arrays(s_lat, s_lon, 7)

    # --- Sample the grid at station locations (nearest cell) ------------ #
    times = pd.to_datetime(grid["time"].values)
    iy = _nearest_index(lat, s_lat)
    ix = _nearest_index(lon, s_lon)

    # Pull predictor cubes once; columns may have NaN (cloud) -> fill with the
    # gap-free MERRA-like behaviour by nearest-time interpolation per station.
    aod = _sample_series(grid["aod"].values, iy, ix)
    no2c = _sample_series(grid["no2_col"].values, iy, ix)
    so2c = _sample_series(grid["so2_col"].values, iy, ix)
    coc = _sample_series(grid["co_col"].values, iy, ix)
    o3c = _sample_series(grid["o3_col"].values, iy, ix)
    blh = _sample_series(grid["blh"].values, iy, ix)
    rh = _sample_series(grid["rh"].values, iy, ix)
    ssrd = _sample_series(grid["ssrd"].values, iy, ix)
    t2m = _sample_series(grid["t2m"].values, iy, ix)

    # Surface concentrations are always defined (ground sensors don't cloud out)
    # so fill NaN column inputs by per-station forward/back fill before mapping.
    aod = _fill_time_gaps(aod)
    no2c = _fill_time_gaps(no2c)
    so2c = _fill_time_gaps(so2c)
    coc = _fill_time_gaps(coc)
    o3c = _fill_time_gaps(o3c)

    nst, nt = len(s_lat), times.size
    noise = rng.normal(0.0, 1.0, (nst, nt))

    # Normalised predictors (broadcasting-friendly).
    aod_u = aod / _COL_SCALE["aod"]
    blh_km = blh / 1000.0
    frh = 1.0 / (1.0 - np.clip(rh, 0, 95) / 100.0) ** 0.5  # hygroscopic growth

    # PM2.5: PBL-normalised, RH-amplified AOD. Known nonlinear law.
    pm25 = 35.0 * (aod_u / np.clip(blh_km, 0.2, None)) * frh
    pm25 = pm25 * (1.0 + 0.10 * noise)
    pm25 = np.clip(pm25, 2.0, 900.0)
    pm10 = pm25 * (1.6 + 0.4 * rng.random((nst, nt))) + 8.0

    no2 = 25.0 * (no2c / _COL_SCALE["no2_col"]) ** 0.8 * (1 + 0.08 * noise)
    so2 = 12.0 * (so2c / _COL_SCALE["so2_col"]) ** 0.7 * (1 + 0.10 * noise)
    co = 0.8 * (coc / _COL_SCALE["co_col"]) ** 0.9 * (1 + 0.08 * noise)  # mg/m3
    # O3: photochemistry — sunlight + heat + NOx, damped at very high NO2 (titr.)
    o3 = (
        18.0
        + 0.20 * (ssrd - 200.0)
        + 0.8 * (t2m - 295.0)
        + 6.0 * (o3c / _COL_SCALE["o3_col"])
        - 0.05 * no2
        + 5.0 * noise
    )

    surface = {
        "pm25": np.clip(pm25, 1.0, 1000.0),
        "pm10": np.clip(pm10, 2.0, 1200.0),
        "no2": np.clip(no2, 0.5, 500.0),
        "so2": np.clip(so2, 0.2, 600.0),
        "co": np.clip(co, 0.05, 50.0),
        "o3": np.clip(o3, 1.0, 400.0),
    }

    # --- Sparse missingness: a few stations drop a pollutant some days --- #
    miss = rng.random((nst, nt)) < 0.05
    for p in ("so2", "o3"):
        surface[p] = np.where(miss, np.nan, surface[p])

    # --- Build the long station-day table ------------------------------- #
    st_idx = np.repeat(np.arange(nst), nt)
    t_idx = np.tile(np.arange(nt), nst)
    rows = {
        "station_id": np.array(station_ids, dtype=object)[st_idx],
        "name": np.array(names, dtype=object)[st_idx],
        "lat": s_lat[st_idx],
        "lon": s_lon[st_idx],
        "h3_res7": h3_static[st_idx],
        "region": regions[st_idx],
        "time": times.to_numpy()[t_idx],
    }
    for p in SURFACE_POLLUTANTS:
        rows[p] = surface[p].reshape(-1)
    df = pd.DataFrame(rows)
    geom = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    logger.info(
        "make_stations: %d stations x %d days = %d rows (IGP-dense)",
        nst,
        nt,
        len(gdf),
    )
    return gdf


def _region_for(lat: float, lon: float) -> str:
    """Coarse region label used to stratify validation."""
    if 24.0 <= lat <= 31.0 and 73.0 <= lon <= 88.0:
        return "IGP"
    if lat <= 15.0:
        return "Peninsula"
    if lon <= 72.5 or lon >= 90.0:
        return "Coastal"
    return "Central"


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return indices of the nearest ``axis`` cell for each value (ascending)."""
    idx = np.searchsorted(axis, values)
    idx = np.clip(idx, 1, axis.size - 1)
    left = axis[idx - 1]
    right = axis[idx]
    idx = np.where(np.abs(values - left) <= np.abs(values - right), idx - 1, idx)
    return idx.astype(int)


def _sample_series(cube: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Extract (n_stations, n_time) series from a (time, lat, lon) cube."""
    # cube[t, iy, ix] -> want shape (n_stations, n_time)
    return cube[:, iy, ix].T


def _fill_time_gaps(series: np.ndarray) -> np.ndarray:
    """Forward/back-fill NaNs along the time axis of a (n_station, n_time) array."""
    df = pd.DataFrame(series.T)  # (time, station)
    df = df.ffill().bfill()
    # Any all-NaN station column -> fill with the global mean of finite values.
    if df.isna().any().any():
        finite_mean = np.nanmean(series) if np.isfinite(series).any() else 0.0
        df = df.fillna(finite_mean)
    return df.to_numpy().T


# --------------------------------------------------------------------------- #
# 5. Orchestration                                                             #
# --------------------------------------------------------------------------- #
def generate_all(
    out_dir: str | Path,
    start_date: str | date | datetime = "2023-10-01",
    n_days: int = 60,
    *,
    res: float = 0.25,
    n_stations: int = 120,
    season: str = "oct_nov",
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate and write the full synthetic dataset, returning written paths.

    Writes, under ``<out_dir>``::

        data/processed/grid.nc                 # gridded cube (§6.1)
        data/processed/stations.parquet        # station daily labels (§6.2)
        data/processed/stations.geojson        # station registry (geometry)
        data/processed/fires.parquet           # fire detections (§6.3)
        data/processed/fires.geojson           # fire detections (geometry)
        data/raw/synthetic/grid_sample.nc      # tiny 3-day raw-style sample
        data/raw/synthetic/stations_sample.parquet
        data/raw/synthetic/fires_sample.parquet

    The pipeline is: ``make_grid -> make_fires -> inject_fire_hcho ->
    make_stations`` (stations sample the *post-injection* grid so their HCHO-
    correlated pollutants reflect the fire signal).

    Args:
        out_dir: Project root under which ``data/processed`` and ``data/raw``
            are created.
        start_date: First day of the simulation.
        n_days: Number of daily steps.
        res: Grid resolution in degrees.
        n_stations: Number of synthetic CPCB stations.
        season: Biomass-burning window for fires (``"oct_nov"`` / ``"apr_may"``).
        seed: Master RNG seed.

    Returns:
        Mapping of artifact name -> written :class:`pathlib.Path`.
    """
    out_root = Path(out_dir)
    processed = out_root / "data" / "processed"
    raw = out_root / "data" / "raw" / "synthetic"
    ensure_dir(processed / "_")
    ensure_dir(raw / "_")

    logger.info("generate_all: building synthetic India dataset (seed=%d)", seed)
    grid = make_grid(start_date, n_days, res=res, seed=seed)
    fires = make_fires(grid, season=season, seed=seed)
    grid = inject_fire_hcho(grid, fires)
    stations = make_stations(grid, n=n_stations, seed=seed)

    paths: dict[str, Path] = {}
    paths["grid"] = save_netcdf(grid, processed / "grid.nc")
    # Station daily long table (parquet) + geometry registry (geojson).
    stations_df = pd.DataFrame(stations.drop(columns="geometry"))
    paths["stations"] = save_parquet(stations_df, processed / "stations.parquet")
    paths["stations_geojson"] = save_geojson(
        _station_registry(stations), processed / "stations.geojson"
    )
    paths["stations_geoparquet"] = save_geoparquet(
        stations, processed / "stations.geoparquet"
    )
    paths["fires"] = save_parquet(
        pd.DataFrame(fires.drop(columns="geometry")), processed / "fires.parquet"
    )
    paths["fires_geojson"] = save_geojson(fires, processed / "fires.geojson")
    paths["fires_geoparquet"] = save_geoparquet(
        fires, processed / "fires.geoparquet"
    )

    # Tiny raw-style samples (first few days / rows) for inspection.
    sample_days = min(3, n_days)
    paths["grid_sample"] = save_netcdf(
        grid.isel(time=slice(0, sample_days)), raw / "grid_sample.nc"
    )
    paths["stations_sample"] = save_parquet(
        stations_df.head(200), raw / "stations_sample.parquet"
    )
    paths["fires_sample"] = save_parquet(
        pd.DataFrame(fires.drop(columns="geometry")).head(200),
        raw / "fires_sample.parquet",
    )

    _print_summary(grid, stations, fires, paths)
    return paths


def _station_registry(stations: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Collapse the station-day table to a static one-row-per-station registry."""
    cols = ["station_id", "name", "lat", "lon", "h3_res7", "region", "geometry"]
    reg = stations[cols].drop_duplicates(subset="station_id").reset_index(
        drop=True
    )
    return reg


def _print_summary(
    grid: "xr.Dataset",
    stations: "gpd.GeoDataFrame",
    fires: "gpd.GeoDataFrame",
    paths: dict[str, Path],
) -> None:
    """Emit a concise human-readable summary of what was generated."""
    n_st = stations["station_id"].nunique()
    cloud = float(np.isnan(grid["no2_col"].values).mean()) * 100.0
    lines = [
        "=" * 64,
        "SYNTHETIC AQI/HCHO demo dataset generated (aqi_india.sim)",
        "-" * 64,
        f"  grid      : {dict(grid.sizes)}  vars={list(grid.data_vars)}",
        f"  cloud gaps: ~{cloud:.0f}% of NO2 column pixels are NaN",
        f"  stations  : {n_st} stations, {len(stations)} station-days",
        f"  fires     : {len(fires)} detections "
        f"({fires['sensor'].nunique()} sensors)",
        f"  HCHO inj. : centroid "
        f"({grid.attrs.get('fire_cluster_centroid_lat', float('nan')):.2f}N, "
        f"{grid.attrs.get('fire_cluster_centroid_lon', float('nan')):.2f}E)",
        "-" * 64,
        "  files:",
    ]
    for name, p in paths.items():
        lines.append(f"    {name:18s} -> {p}")
    lines.append("=" * 64)
    msg = "\n".join(lines)
    logger.info("generate_all complete:\n%s", msg)
    print(msg)


__all__ = [
    "DEFAULT_SEED",
    "GRID_VARS",
    "COLUMN_VARS",
    "SURFACE_POLLUTANTS",
    "make_grid",
    "make_fires",
    "inject_fire_hcho",
    "make_stations",
    "generate_all",
]
