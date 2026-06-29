"""ERA5 reanalysis meteorology adapter (Copernicus CDS — cdsapi).

Pulls ERA5 **single-level** meteorology (and optionally **pressure-level winds**)
from the Copernicus Climate Data Store via ``cdsapi``, then aggregates the hourly
fields to the daily ``time/lat/lon`` schema (``blh, rh, wind_u, wind_v, t2m,
ssrd`` + ``precip, mslp``) used across the project.

.. important::
   **ERA5 boundary-layer height (BLH) and pressure-level winds are only available
   from CDS — NOT from GEE's ERA5 mirror.** This adapter is the canonical BLH /
   upper-wind source for the physics-guided features (AOD/BLH PBL normalisation);
   never substitute a GEE-ERA5 BLH (it does not exist there).

``cdsapi`` (credentialed) is lazy-imported. The ``synthetic=True`` path delegates
to :func:`aqi_india.sim.synthetic.make_grid` and returns its gap-free met fields,
so the offline pipeline always has BLH/RH/wind/T/SSRD without a CDS account.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("ingest.era5_cds")

#: CDS dataset id for ERA5 hourly single-level reanalysis.
CDS_SINGLE_LEVELS: str = "reanalysis-era5-single-levels"

#: CDS dataset id for ERA5 hourly pressure-level reanalysis (upper winds).
CDS_PRESSURE_LEVELS: str = "reanalysis-era5-pressure-levels"

#: CDS variable -> canonical project grid var. BLH is CDS-only (see module note).
CDS_TO_GRID_VAR: dict[str, str] = {
    "boundary_layer_height": "blh",
    "10m_u_component_of_wind": "wind_u",
    "10m_v_component_of_wind": "wind_v",
    "2m_temperature": "t2m",
    "surface_solar_radiation_downwards": "ssrd",
    "mean_sea_level_pressure": "mslp",
    "total_precipitation": "precip",
}

#: ERA5 short-name -> canonical grid var (as found in the downloaded NetCDF).
ERA5_SHORTNAME_TO_GRID_VAR: dict[str, str] = {
    "blh": "blh",
    "u10": "wind_u",
    "v10": "wind_v",
    "t2m": "t2m",
    "ssrd": "ssrd",
    "msl": "mslp",
    "tp": "precip",
    "d2m": "d2m",
}

#: Default single-level request variable list (BLH first — it is the headline).
DEFAULT_VARIABLES: tuple[str, ...] = (
    "boundary_layer_height",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_solar_radiation_downwards",
    "mean_sea_level_pressure",
    "total_precipitation",
)

DateLike = str | date | datetime


def fetch_era5(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    pressure_levels: tuple[int, ...] | None = None,
    cache_path: str | Path | None = None,
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Fetch daily-aggregated ERA5 meteorology over an AOI.

    Real path: submits a ``cdsapi`` request for ``variables`` over ``[start, end)``
    and the AOI, downloads the NetCDF, renames ERA5 short-names to the canonical
    grid vars, derives ``rh`` from 2 m temperature + dewpoint, and resamples the
    hourly fields to daily means (precip/SSRD summed). With ``pressure_levels``
    set, additionally pulls upper-level U/V winds (CDS-only) as ``wind_u_<lvl>`` /
    ``wind_v_<lvl>`` for transport features.

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_grid` and
    returns its gap-free met variables (``blh, rh, wind_u, wind_v, t2m, ssrd``).

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` clip footprint.
        variables: CDS single-level variable names (default :data:`DEFAULT_VARIABLES`).
        pressure_levels: Optional hPa levels for upper-wind retrieval (CDS-only).
        cache_path: Where to write the downloaded NetCDF (real path).
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A daily :class:`xarray.Dataset` of the requested met variables.
    """
    if synthetic:
        return _synthetic_era5(start, end, aoi, seed=seed)

    return _fetch_era5_cds(
        start, end, aoi, variables=variables,
        pressure_levels=pressure_levels, cache_path=cache_path,
    )


def _synthetic_era5(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    seed: int,
) -> xr.Dataset:
    """Return synthetic gap-free met fields (offline path)."""
    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)
    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
    met_vars = ["blh", "rh", "wind_u", "wind_v", "t2m", "ssrd"]
    out = grid[met_vars].copy()
    out.attrs["source"] = "SYNTHETIC ERA5 single-levels (BLH/RH/wind/T/SSRD)"
    out["blh"].attrs["note"] = "BLH is CDS-only in real ERA5 (absent from GEE ERA5)."
    logger.info("fetch_era5(synthetic): %s, %d days", met_vars, n_days)
    return out


def _fetch_era5_cds(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    variables: tuple[str, ...],
    pressure_levels: tuple[int, ...] | None,
    cache_path: str | Path | None,
) -> xr.Dataset:
    """Submit the cdsapi request, download, rename, derive RH and aggregate daily."""
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "cdsapi is required for live ERA5 ingestion (pip install cdsapi) and "
            "a configured ~/.cdsapirc. For offline runs call "
            "fetch_era5(..., synthetic=True)."
        ) from exc
    import xarray as xr

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize() - pd.Timedelta(days=1)
    min_lon, min_lat, max_lon, max_lat = aoi
    # CDS area order is North/West/South/East.
    area = [max_lat, min_lon, min_lat, max_lon]
    days = pd.date_range(t0, t1, freq="D")
    request = {
        "product_type": "reanalysis",
        "variable": list(variables),
        "year": sorted({d.strftime("%Y") for d in days}),
        "month": sorted({d.strftime("%m") for d in days}),
        "day": sorted({d.strftime("%d") for d in days}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "format": "netcdf",
    }

    cache = Path(cache_path) if cache_path else Path("data/raw/era5/era5_single.nc")
    cache.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "fetch_era5(CDS): %s vars=%d [%s..%s] area=%s",
        CDS_SINGLE_LEVELS,
        len(variables),
        t0.date(),
        t1.date(),
        area,
    )
    client = cdsapi.Client()
    client.retrieve(CDS_SINGLE_LEVELS, request, str(cache))

    ds = xr.open_dataset(cache)
    ds = _standardise_era5(ds)
    daily = _aggregate_daily(ds)

    if pressure_levels:
        upper = _fetch_era5_upper_winds(
            client, t0, t1, area, pressure_levels, cache.parent
        )
        daily = xr.merge([daily, upper], compat="override")
    return daily


def _fetch_era5_upper_winds(
    client,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    area: list[float],
    levels: tuple[int, ...],
    cache_dir: Path,
) -> xr.Dataset:
    """Pull ERA5 pressure-level U/V winds (CDS-only) and aggregate daily."""
    import xarray as xr

    days = pd.date_range(t0, t1, freq="D")
    request = {
        "product_type": "reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(lvl) for lvl in levels],
        "year": sorted({d.strftime("%Y") for d in days}),
        "month": sorted({d.strftime("%m") for d in days}),
        "day": sorted({d.strftime("%d") for d in days}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "format": "netcdf",
    }
    cache = cache_dir / "era5_pressure.nc"
    client.retrieve(CDS_PRESSURE_LEVELS, request, str(cache))
    ds = xr.open_dataset(cache)
    rename = {}
    for short, var in (("u", "wind_u"), ("v", "wind_v")):
        if short in ds:
            rename[short] = var
    ds = ds.rename(rename)
    daily = ds.resample(time="1D").mean() if "time" in ds.dims else ds
    return daily


def _standardise_era5(ds: xr.Dataset) -> xr.Dataset:
    """Rename ERA5 short-names to canonical vars and derive RH; ensure asc coords."""
    rename = {
        short: var
        for short, var in ERA5_SHORTNAME_TO_GRID_VAR.items()
        if short in ds
    }
    ds = ds.rename(rename)
    # Harmonise coordinate names (CDS uses latitude/longitude).
    coord_rename = {}
    if "latitude" in ds.coords:
        coord_rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        coord_rename["longitude"] = "lon"
    if coord_rename:
        ds = ds.rename(coord_rename)
    # Ensure ascending lat (ERA5 ships descending).
    if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))

    if "t2m" in ds and "d2m" in ds:
        ds["rh"] = _relative_humidity(ds["t2m"], ds["d2m"])
        ds = ds.drop_vars("d2m")
    return ds


def _relative_humidity(t2m: xr.DataArray, d2m: xr.DataArray) -> xr.DataArray:
    """Compute % relative humidity from 2 m temperature and dewpoint (K).

    Uses the Magnus/August-Roche-Magnus saturation-vapour-pressure approximation
    ``RH = 100 * e_s(Td) / e_s(T)`` with ``e_s(T) = 6.112 * exp(17.62*Tc /
    (243.12 + Tc))``.

    Args:
        t2m: 2 m temperature (K).
        d2m: 2 m dewpoint temperature (K).

    Returns:
        Relative humidity (%) clipped to ``[0, 100]``, named ``rh``.
    """
    tc = t2m - 273.15
    tdc = d2m - 273.15
    es = 6.112 * np.exp(17.62 * tc / (243.12 + tc))
    ed = 6.112 * np.exp(17.62 * tdc / (243.12 + tdc))
    rh = 100.0 * ed / es
    rh = rh.clip(0.0, 100.0)
    rh.name = "rh"
    rh.attrs["units"] = "%"
    return rh


def _aggregate_daily(ds: xr.Dataset) -> xr.Dataset:
    """Resample hourly ERA5 to daily (means; precip & SSRD summed as totals)."""
    if "time" not in ds.dims:
        return ds
    accum_vars = [v for v in ("precip", "ssrd") if v in ds]
    mean_vars = [v for v in ds.data_vars if v not in accum_vars]
    parts = []
    if mean_vars:
        parts.append(ds[mean_vars].resample(time="1D").mean())
    if accum_vars:
        parts.append(ds[accum_vars].resample(time="1D").sum())
    import xarray as xr

    return xr.merge(parts) if len(parts) > 1 else parts[0]


__all__ = [
    "CDS_SINGLE_LEVELS",
    "CDS_PRESSURE_LEVELS",
    "CDS_TO_GRID_VAR",
    "ERA5_SHORTNAME_TO_GRID_VAR",
    "DEFAULT_VARIABLES",
    "fetch_era5",
]
