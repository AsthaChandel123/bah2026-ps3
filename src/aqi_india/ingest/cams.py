"""CAMS EAC4 / NRT gap-free atmospheric-composition prior adapter (ADS — cdsapi).

CAMS is the **L0 gap-free physical prior** of the architecture: a 4D-Var
reanalysis (EAC4) / forecast (NRT) supplying cloud-free NO2/SO2/CO/O3/HCHO/PM/AOD
fields that fill cloud-masked satellite pixels *before* the ML and serve as
always-available feature channels. We **download** it — never reimplement 4D-Var.

CAMS is served by the **Atmosphere Data Store (ADS)** via ``cdsapi`` (a distinct
endpoint/key from the Climate Data Store). ``cdsapi`` is lazy-imported. The
``synthetic=True`` path delegates to :func:`aqi_india.sim.synthetic.make_grid`,
mapping its column/aerosol fields to the CAMS variable names so the offline
pipeline has a coherent gap-free prior.
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

logger = get_logger("ingest.cams")

#: ADS dataset ids for the reanalysis (EAC4) and NRT composition forecast.
CAMS_DATASETS: dict[str, str] = {
    "eac4": "cams-global-reanalysis-eac4",
    "nrt": "cams-global-atmospheric-composition-forecasts",
}

#: Default ADS endpoint (distinct from the CDS climate endpoint).
ADS_URL: str = "https://ads.atmosphere.copernicus.eu/api"

#: CAMS variable -> canonical project grid var (column priors map to ``*_col``).
CAMS_TO_GRID_VAR: dict[str, str] = {
    "nitrogen_dioxide": "no2_col",
    "sulphur_dioxide": "so2_col",
    "carbon_monoxide": "co_col",
    "ozone": "o3_col",
    "formaldehyde": "hcho_col",
    "particulate_matter_2.5um": "pm25",
    "particulate_matter_10um": "pm10",
    "total_aerosol_optical_depth_550nm": "aod",
}

#: CAMS NetCDF short-name -> canonical grid var.
CAMS_SHORTNAME_TO_GRID_VAR: dict[str, str] = {
    "no2": "no2_col",
    "so2": "so2_col",
    "co": "co_col",
    "go3": "o3_col",
    "o3": "o3_col",
    "hcho": "hcho_col",
    "pm2p5": "pm25",
    "pm10": "pm10",
    "aod550": "aod",
}

#: Default CAMS variables to request (the AQI pollutant set + HCHO + AOD).
DEFAULT_VARIABLES: tuple[str, ...] = (
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "ozone",
    "formaldehyde",
    "particulate_matter_2.5um",
    "particulate_matter_10um",
    "total_aerosol_optical_depth_550nm",
)

DateLike = str | date | datetime


def fetch_cams(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    product: str = "eac4",
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    cache_path: str | Path | None = None,
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Fetch the daily-aggregated CAMS gap-free composition prior over an AOI.

    Real path: submits a ``cdsapi`` request to the **ADS** for ``variables`` over
    ``[start, end)`` and the AOI, downloads the NetCDF, renames CAMS short-names
    to canonical grid vars, and resamples to daily means.

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_grid`, mapping
    its ``*_col`` / ``aod`` fields to the CAMS variable names (a gap-free prior
    with no cloud NaNs).

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` clip footprint.
        product: ``"eac4"`` (reanalysis) or ``"nrt"`` (forecast).
        variables: CAMS variable names (default :data:`DEFAULT_VARIABLES`).
        cache_path: Where to write the downloaded NetCDF (real path).
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A daily :class:`xarray.Dataset` of the CAMS prior fields.

    Raises:
        ValueError: If ``product`` is unknown.
    """
    if product not in CAMS_DATASETS:
        raise ValueError(
            f"unknown CAMS product {product!r}; valid: {sorted(CAMS_DATASETS)}"
        )

    if synthetic:
        return _synthetic_cams(start, end, aoi, variables, seed=seed)

    return _fetch_cams_ads(
        start, end, aoi, product, variables, cache_path=cache_path
    )


def _synthetic_cams(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    variables: tuple[str, ...],
    *,
    seed: int,
) -> xr.Dataset:
    """Return a synthetic gap-free CAMS-shaped prior (offline path).

    Builds the synthetic grid, injects the fire-HCHO signal, then **fills cloud
    gaps** in the column fields (CAMS has no orbit/cloud holes) and renames to the
    requested CAMS variable names.
    """
    import xarray as xr

    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)
    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
    fires = sim.make_fires(grid, seed=seed)
    grid = sim.inject_fire_hcho(grid, fires)

    out_vars: dict[str, xr.DataArray] = {}
    for cams_name in variables:
        gvar = CAMS_TO_GRID_VAR.get(cams_name)
        if gvar is None:
            continue
        if gvar in grid:
            da = grid[gvar]
        elif gvar == "pm25":
            da = _surrogate_pm(grid, coarse=False)
        elif gvar == "pm10":
            da = _surrogate_pm(grid, coarse=True)
        else:
            continue
        # Emit canonical grid-var names (matching the real-CAMS rename path) and,
        # since CAMS is gap-free, fill any synthetic cloud NaNs without bottleneck.
        out_vars[gvar] = _fill_gaps(da).astype(np.float32)

    ds = xr.Dataset(out_vars)
    ds.attrs["source"] = "SYNTHETIC CAMS EAC4 gap-free composition prior"
    logger.info("fetch_cams(synthetic): vars=%s, %d days", list(out_vars), n_days)
    return ds


def _fill_gaps(da: xr.DataArray) -> xr.DataArray:
    """Fill NaNs in a (time, lat, lon) array without bottleneck.

    Replaces each missing cell with the per-cell temporal mean (gap-free prior
    behaviour); any cell that is NaN for all days falls back to the global mean.
    Pure NumPy so the light dependency set suffices.

    Args:
        da: A ``(time, lat, lon)`` DataArray that may contain NaNs.

    Returns:
        A copy of ``da`` with NaNs filled.
    """
    values = np.asarray(da.values, dtype=np.float64)
    cell_mean = np.nanmean(values, axis=0, keepdims=True)
    cell_mean = np.where(np.isfinite(cell_mean), cell_mean, np.nan)
    filled = np.where(np.isnan(values), cell_mean, values)
    global_mean = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    filled = np.where(np.isnan(filled), global_mean, filled)
    out = da.copy(data=filled)
    return out


def _surrogate_pm(grid: xr.Dataset, *, coarse: bool) -> xr.DataArray:
    """Build a crude gap-free PM surrogate (ug/m3) from AOD for the offline prior."""
    aod = grid["aod"] / 1.0  # aod scale is 1.0 in the synthetic schema
    pm = aod * (90.0 if not coarse else 150.0)
    pm.name = "pm10" if coarse else "pm25"
    pm.attrs["units"] = "ug/m3"
    return pm


def _fetch_cams_ads(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    product: str,
    variables: tuple[str, ...],
    *,
    cache_path: str | Path | None,
) -> xr.Dataset:
    """Submit the ADS cdsapi request, download, rename and aggregate daily."""
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "cdsapi is required for live CAMS ingestion (pip install cdsapi) with "
            "an ADS (atmosphere.copernicus.eu) key in ~/.cdsapirc. For offline "
            "runs call fetch_cams(..., synthetic=True)."
        ) from exc
    import xarray as xr

    dataset = CAMS_DATASETS[product]
    min_lon, min_lat, max_lon, max_lat = aoi
    area = [max_lat, min_lon, min_lat, max_lon]  # N/W/S/E
    t0 = pd.Timestamp(start).strftime("%Y-%m-%d")
    t1 = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    request: dict[str, object] = {
        "variable": list(variables),
        "date": f"{t0}/{t1}",
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
        "area": area,
        "format": "netcdf",
    }
    if product == "eac4":
        request["model_level"] = "60"  # surface-most model level for column proxy

    cache = Path(cache_path) if cache_path else Path(f"data/raw/cams/{product}.nc")
    cache.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetch_cams(ADS): %s vars=%d [%s..%s]", dataset, len(variables), t0, t1)
    client = cdsapi.Client(url=ADS_URL)
    client.retrieve(dataset, request, str(cache))

    ds = xr.open_dataset(cache)
    rename = {s: g for s, g in CAMS_SHORTNAME_TO_GRID_VAR.items() if s in ds}
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
    daily = ds.resample(time="1D").mean() if "time" in ds.dims else ds
    daily.attrs["source"] = f"CAMS {dataset} (ADS)"
    return daily


__all__ = [
    "CAMS_DATASETS",
    "ADS_URL",
    "CAMS_TO_GRID_VAR",
    "CAMS_SHORTNAME_TO_GRID_VAR",
    "DEFAULT_VARIABLES",
    "fetch_cams",
]
