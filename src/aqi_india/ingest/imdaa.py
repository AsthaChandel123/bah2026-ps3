"""IMDAA 12 km Indian regional reanalysis adapter (NCMRWF).

IMDAA is the **highest-resolution reanalysis over India** (0.12 deg ~12 km) and
the preferred regional met predictor (BLH/HPBL, 10 m winds, T, RH, precip, SSRD).
It is distributed as GRIB/NetCDF via NCMRWF's portal (http_download, registration
required) — not a cloud/GEE collection — so this adapter consumes already-
downloaded files and standardises them to the project's daily ``time/lat/lon``
schema.

Heavy deps (``xarray``/``cfgrib``) are lazy-imported. The ``synthetic=True`` path
delegates to :func:`aqi_india.sim.synthetic.make_grid`, returning its met fields
as an IMDAA surrogate so the offline pipeline runs without an NCMRWF account.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("ingest.imdaa")

#: NCMRWF reanalysis portal (registration required; http_download).
IMDAA_ENDPOINT: str = "https://nwp.ncmrwf.gov.in/reanalysis"

#: Native IMDAA resolution in degrees (~12 km).
IMDAA_RES_DEG: float = 0.12

#: IMDAA variable name -> canonical project grid var (probed across conventions).
IMDAA_TO_GRID_VAR: dict[str, str] = {
    "HPBL": "blh",
    "hpbl": "blh",
    "blh": "blh",
    "UGRD_10m": "wind_u",
    "u10": "wind_u",
    "10u": "wind_u",
    "VGRD_10m": "wind_v",
    "v10": "wind_v",
    "10v": "wind_v",
    "TMP_2m": "t2m",
    "t2m": "t2m",
    "2t": "t2m",
    "RH_2m": "rh",
    "r2": "rh",
    "rh": "rh",
    "DSWRF": "ssrd",
    "ssrd": "ssrd",
    "APCP": "precip",
    "tp": "precip",
    "precip": "precip",
}

DateLike = str | date | datetime


def fetch_imdaa(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    files: list[str | Path] | None = None,
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Read / standardise daily IMDAA 12 km meteorology over an AOI.

    Real path: opens the provided NCMRWF GRIB/NetCDF ``files``, renames IMDAA
    variables to the canonical grid vars, clips to ``aoi``, ensures ascending
    coords, and resamples hourly fields to daily (precip/SSRD summed). Acquisition
    of the files is an out-of-band registered NCMRWF download (see
    :data:`IMDAA_ENDPOINT`); this adapter consumes them.

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_grid` and
    returns its met fields as the IMDAA surrogate.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` clip footprint.
        files: Local IMDAA GRIB/NetCDF file paths (real path).
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A daily :class:`xarray.Dataset` of IMDAA met variables.

    Raises:
        ValueError: If neither ``synthetic`` nor ``files`` is provided.
    """
    if synthetic:
        return _synthetic_imdaa(start, end, aoi, seed=seed)

    if not files:
        raise ValueError(
            "IMDAA is an NCMRWF http_download product (not GEE-native): provide "
            "downloaded file paths via `files=`, or run with synthetic=True. "
            f"Register/download from {IMDAA_ENDPOINT}."
        )

    return _read_imdaa_files(files, start, end, aoi)


def _synthetic_imdaa(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    seed: int,
) -> xr.Dataset:
    """Return synthetic met fields as the IMDAA surrogate (offline path)."""
    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)
    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
    met_vars = ["blh", "rh", "wind_u", "wind_v", "t2m", "ssrd"]
    out = grid[met_vars].copy()
    out.attrs["source"] = "SYNTHETIC IMDAA 12km regional reanalysis (NCMRWF)"
    logger.info("fetch_imdaa(synthetic): %s, %d days", met_vars, n_days)
    return out


def _read_imdaa_files(
    files: list[str | Path],
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
) -> xr.Dataset:
    """Open + standardise + clip + daily-aggregate IMDAA files."""
    import xarray as xr

    datasets = []
    for fp in files:
        ds = _open_any(Path(fp))
        if ds is not None:
            datasets.append(ds)
    if not datasets:
        raise ValueError("no IMDAA files could be opened.")

    combined = xr.merge(datasets, compat="override", join="outer")
    combined = _standardise(combined, aoi)
    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    if "time" in combined.dims:
        combined = combined.sel(time=slice(t0, t1 - pd.Timedelta(seconds=1)))
        combined = _aggregate_daily(combined)
    logger.info("fetch_imdaa: standardised %d files", len(files))
    return combined


def _open_any(path: Path) -> xr.Dataset | None:
    """Open a NetCDF or GRIB IMDAA file (cfgrib for GRIB), or warn and skip."""
    import xarray as xr

    try:
        if path.suffix.lower() in (".grb", ".grib", ".grib2", ".grb2"):
            return xr.open_dataset(path, engine="cfgrib")
        return xr.open_dataset(path)
    except (ValueError, OSError, ImportError) as exc:
        logger.warning("could not open IMDAA file %s: %s", path, exc)
        return None


def _standardise(ds: xr.Dataset, aoi: tuple[float, float, float, float]) -> xr.Dataset:
    """Rename IMDAA vars/coords to canonical names, clip to AOI, ascending coords."""
    rename = {v: g for v, g in IMDAA_TO_GRID_VAR.items() if v in ds}
    ds = ds.rename(rename)
    coord_rename = {}
    if "latitude" in ds.coords:
        coord_rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        coord_rename["longitude"] = "lon"
    if coord_rename:
        ds = ds.rename(coord_rename)
    if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))

    min_lon, min_lat, max_lon, max_lat = aoi
    if "lat" in ds.coords and "lon" in ds.coords:
        ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    return ds


def _aggregate_daily(ds: xr.Dataset) -> xr.Dataset:
    """Resample hourly IMDAA to daily (means; precip & SSRD summed)."""
    import xarray as xr

    accum = [v for v in ("precip", "ssrd") if v in ds]
    means = [v for v in ds.data_vars if v not in accum]
    parts = []
    if means:
        parts.append(ds[means].resample(time="1D").mean())
    if accum:
        parts.append(ds[accum].resample(time="1D").sum())
    return xr.merge(parts) if len(parts) > 1 else parts[0]


__all__ = ["IMDAA_ENDPOINT", "IMDAA_RES_DEG", "IMDAA_TO_GRID_VAR", "fetch_imdaa"]
