"""MERRA-2 aerosol + flux diagnostics adapter (NASA GMAO via earthaccess).

MERRA-2 supplies a **gap-free** speciated aerosol prior and an independent
boundary-layer height:

* ``M2T1NXAER`` (tavg1_2d_aer_Nx): total AOD ``TOTEXTTAU`` + speciated surface
  mass (``BCSMASS, OCSMASS, DUSMASS, DUSMASS25, SSSMASS, SO4SMASS``) from which
  ``PM2.5 ~ 1.375*SO4 + 1.6*OC + BC + DUSMASS25 + 0.5*SSSMASS`` can be derived;
* ``M2T1NXFLX`` (tavg1_2d_flx_Nx): planetary-boundary-layer height ``PBLH``.

These fill cloud/monsoon holes in MODIS/TROPOMI (the fields never have orbit/cloud
gaps) and serve as ML feature channels. Granules are pulled with NASA
``earthaccess`` (Earthdata login). The ``synthetic=True`` path delegates to
:func:`aqi_india.sim.synthetic.make_grid`, mapping its ``aod``/``blh`` to
``TOTEXTTAU``/``PBLH`` so the offline pipeline has a gap-free prior.
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

logger = get_logger("ingest.merra2")

#: Earthdata short-names for the two MERRA-2 collections used here.
MERRA2_COLLECTIONS: dict[str, str] = {
    "aer": "M2T1NXAER",
    "flx": "M2T1NXFLX",
}

#: AER variables (total AOD + speciated surface mass, kg/m3 for the *SMASS).
MERRA2_AER_VARS: tuple[str, ...] = (
    "TOTEXTTAU",
    "BCSMASS",
    "OCSMASS",
    "DUSMASS",
    "DUSMASS25",
    "SSSMASS",
    "SO4SMASS",
)

#: FLX variable: planetary boundary-layer height (m).
MERRA2_FLX_VARS: tuple[str, ...] = ("PBLH",)

#: GEE mirror of the MERRA-2 AER collection (documented; this adapter uses earthaccess).
MERRA2_GEE_COLLECTION: str = "NASA/GSFC/MERRA/aer/2"

DateLike = str | date | datetime


def fetch_merra2(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    collection: str = "aer",
    variables: tuple[str, ...] | None = None,
    cache_dir: str | Path | None = None,
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Fetch daily-aggregated MERRA-2 aerosol or flux diagnostics over an AOI.

    Real path: uses ``earthaccess`` to search + download the requested
    ``M2T1NXAER``/``M2T1NXFLX`` granules for ``[start, end)``, opens them, clips
    to ``aoi``, and resamples the hourly fields to daily means.

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_grid`, exposing
    ``TOTEXTTAU`` (from ``aod``) and ``PBLH`` (from ``blh``) plus crude speciated
    surrogates so the gap-free-prior consumers downstream have coherent fields.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` clip footprint.
        collection: ``"aer"`` (AOD + speciated mass) or ``"flx"`` (PBLH).
        variables: Override the variable subset (else collection defaults).
        cache_dir: Directory for downloaded granules (real path).
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A daily :class:`xarray.Dataset` of the requested MERRA-2 variables.

    Raises:
        ValueError: If ``collection`` is unknown.
    """
    if collection not in MERRA2_COLLECTIONS:
        raise ValueError(
            f"unknown MERRA-2 collection {collection!r}; valid: "
            f"{sorted(MERRA2_COLLECTIONS)}"
        )
    if variables is None:
        variables = MERRA2_AER_VARS if collection == "aer" else MERRA2_FLX_VARS

    if synthetic:
        return _synthetic_merra2(start, end, aoi, collection, variables, seed=seed)

    return _fetch_merra2_earthaccess(
        start, end, aoi, collection, variables, cache_dir=cache_dir
    )


def _synthetic_merra2(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    collection: str,
    variables: tuple[str, ...],
    *,
    seed: int,
) -> xr.Dataset:
    """Return synthetic gap-free MERRA-2 surrogates (offline path)."""
    import xarray as xr

    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)
    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)

    out_vars: dict[str, xr.DataArray] = {}
    aod = grid["aod"]
    if "TOTEXTTAU" in variables:
        out_vars["TOTEXTTAU"] = aod.copy()
    if "PBLH" in variables:
        out_vars["PBLH"] = grid["blh"].copy()
    # Crude speciated surface-mass surrogates (kg/m3) scaled off AOD so the
    # PM2.5 = 1.375*SO4 + 1.6*OC + BC + dust25 + 0.5*ss formula is exercisable.
    spec_frac = {
        "SO4SMASS": 0.25e-8,
        "OCSMASS": 0.20e-8,
        "BCSMASS": 0.05e-8,
        "DUSMASS": 0.30e-8,
        "DUSMASS25": 0.15e-8,
        "SSSMASS": 0.05e-8,
    }
    for name, frac in spec_frac.items():
        if name in variables:
            out_vars[name] = (aod * frac).astype(np.float32)

    ds = xr.Dataset(out_vars)
    ds.attrs["source"] = f"SYNTHETIC MERRA-2 {MERRA2_COLLECTIONS[collection]}"
    logger.info(
        "fetch_merra2(synthetic): %s vars=%s, %d days",
        collection,
        list(out_vars),
        n_days,
    )
    return ds


def _fetch_merra2_earthaccess(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    collection: str,
    variables: tuple[str, ...],
    *,
    cache_dir: str | Path | None,
) -> xr.Dataset:
    """Search + download MERRA-2 granules via earthaccess and aggregate daily."""
    try:
        import earthaccess
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "earthaccess is required for live MERRA-2 ingestion "
            "(pip install earthaccess) with Earthdata credentials. For offline "
            "runs call fetch_merra2(..., synthetic=True)."
        ) from exc
    import xarray as xr

    short_name = MERRA2_COLLECTIONS[collection]
    min_lon, min_lat, max_lon, max_lat = aoi
    t0 = pd.Timestamp(start).strftime("%Y-%m-%d")
    t1 = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    earthaccess.login()
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(t0, t1),
        bounding_box=(min_lon, min_lat, max_lon, max_lat),
    )
    cache = Path(cache_dir) if cache_dir else Path(f"data/raw/merra2/{collection}")
    cache.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download(results, str(cache))

    ds = xr.open_mfdataset(paths, combine="by_coords")[list(variables)]
    coord_rename = {}
    if "latitude" in ds.coords:
        coord_rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        coord_rename["longitude"] = "lon"
    if coord_rename:
        ds = ds.rename(coord_rename)
    ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    daily = ds.resample(time="1D").mean() if "time" in ds.dims else ds
    daily.attrs["source"] = f"MERRA-2 {short_name} (earthaccess)"
    logger.info("fetch_merra2(earthaccess): %s -> %d granules", short_name, len(paths))
    return daily.load()


def derive_pm25_from_speciated(ds: xr.Dataset) -> xr.DataArray:
    """Derive surface PM2.5 (ug/m3) from MERRA-2 speciated surface mass.

    Applies the standard GOCART relation
    ``PM2.5 = 1.375*SO4 + 1.6*OC + BC + DUSMASS25 + 0.5*SSSMASS`` and converts
    the MERRA-2 ``*SMASS`` units (kg/m3) to ug/m3 (x1e9).

    Args:
        ds: Dataset carrying ``SO4SMASS, OCSMASS, BCSMASS, DUSMASS25, SSSMASS``
            (any missing component is treated as zero).

    Returns:
        A ``DataArray`` named ``pm25`` in ug/m3.
    """
    def g(name: str):
        return ds[name] if name in ds else 0.0

    pm = (
        1.375 * g("SO4SMASS")
        + 1.6 * g("OCSMASS")
        + g("BCSMASS")
        + g("DUSMASS25")
        + 0.5 * g("SSSMASS")
    ) * 1.0e9
    if hasattr(pm, "rename"):
        pm = pm.rename("pm25")
        pm.attrs["units"] = "ug/m3"
        pm.attrs["source"] = "MERRA-2 speciated surface mass (GOCART PM2.5)"
    return pm


__all__ = [
    "MERRA2_COLLECTIONS",
    "MERRA2_AER_VARS",
    "MERRA2_FLX_VARS",
    "MERRA2_GEE_COLLECTION",
    "fetch_merra2",
    "derive_pm25_from_speciated",
]
