"""MODIS MAIAC 1 km aerosol optical depth (AOD) adapter — the AOD backbone.

MAIAC (``MODIS/061/MCD19A2_GRANULES``, band ``Optical_Depth_055``) is the
quantitative 1 km AOD anchor for the AOD->PM2.5 branch (Objective-1). This
adapter pulls a daily MAIAC AOD cube from Google Earth Engine, applies the
``AOD_QA`` best-quality bit filter, and median-composites the (up to two) daily
Terra+Aqua granules per pixel. Output is a single-variable
:class:`xarray.Dataset` named ``aod`` matching ``DEV_CONTRACT`` §6.1.

``import ee`` is lazy; the offline ``synthetic=True`` path delegates to
:func:`aqi_india.sim.synthetic.make_grid` and returns its ``aod`` field.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import ee
    import xarray as xr

logger = get_logger("ingest.maiac")

#: Exact GEE collection id for MAIAC combined Terra+Aqua AOD granules.
MAIAC_COLLECTION: str = "MODIS/061/MCD19A2_GRANULES"

#: Bands of interest; 055 (550 nm) is the primary PM2.5 predictor.
MAIAC_BANDS: dict[str, str] = {
    "aod_055": "Optical_Depth_055",
    "aod_047": "Optical_Depth_047",
}

#: QA band carrying the per-pixel AOD quality bitmask.
MAIAC_QA_BAND: str = "AOD_QA"

#: MAIAC scale factor for the Optical_Depth_* bands (DN -> AOD).
MAIAC_SCALE_FACTOR: float = 0.001

DateLike = str | date | datetime


def fetch_maiac(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    band: str = "aod_055",
    best_quality_only: bool = True,
    scale_m: float = 1000.0,
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Fetch a daily MAIAC 1 km AOD cube over an AOI.

    Real path: filters ``MODIS/061/MCD19A2_GRANULES`` to ``[start, end)`` and the
    AOD, optionally keeps only best-quality pixels (``AOD_QA`` mask QA bits 8-11
    == 0), scales DN by :data:`MAIAC_SCALE_FACTOR`, and per-pixel median-composites
    the daily granules. Output var is ``aod`` (unitless).

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_grid` and
    returns its ``aod`` field on the same schema.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` footprint.
        band: ``"aod_055"`` (default, 550 nm) or ``"aod_047"`` (470 nm).
        best_quality_only: Keep only best-quality retrievals via the QA mask.
        scale_m: Sampling scale in metres (native ~1 km).
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A single-variable daily :class:`xarray.Dataset` named ``aod``.

    Raises:
        ValueError: If ``band`` is unknown.
    """
    if band not in MAIAC_BANDS:
        raise ValueError(f"unknown MAIAC band {band!r}; valid: {sorted(MAIAC_BANDS)}")

    if synthetic:
        return _synthetic_maiac(start, end, aoi, seed=seed)

    return _fetch_maiac_gee(
        start, end, aoi, band=band, best_quality_only=best_quality_only,
        scale_m=scale_m,
    )


def _synthetic_maiac(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    seed: int,
) -> xr.Dataset:
    """Return the synthetic ``aod`` field (offline path)."""
    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)
    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
    out = grid[["aod"]].copy()
    out["aod"].attrs.setdefault("source", "SYNTHETIC MAIAC AOD 550nm")
    logger.info("fetch_maiac(synthetic): aod, %d days", n_days)
    return out


def _fetch_maiac_gee(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    band: str,
    best_quality_only: bool,
    scale_m: float,
) -> xr.Dataset:
    """Query, QA-mask, scale and median-composite daily MAIAC AOD from GEE."""
    import ee

    from .gee_auth import init_ee

    init_ee()
    ee_band = MAIAC_BANDS[band]
    region = ee.Geometry.Rectangle(list(aoi))
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")

    def _prep(img: ee.Image) -> ee.Image:
        aod = img.select(ee_band).multiply(MAIAC_SCALE_FACTOR)
        if best_quality_only:
            # AOD_QA bits 8-11 encode the QA confidence; 0 == best quality.
            qa = img.select(MAIAC_QA_BAND).rightShift(8).bitwiseAnd(0xF)
            aod = aod.updateMask(qa.eq(0))
        return aod.rename("aod")

    days = pd.date_range(start_s, end_s, freq="D", inclusive="left")
    images: list[ee.Image] = []
    for day in days:
        d0 = day.strftime("%Y-%m-%d")
        d1 = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        daily = (
            ee.ImageCollection(MAIAC_COLLECTION)
            .filterDate(d0, d1)
            .filterBounds(region)
            .map(_prep)
        )
        comp = daily.median().set("system:time_start", ee.Date(d0).millis())
        images.append(comp)

    composite = ee.ImageCollection(images)
    logger.info(
        "fetch_maiac(GEE): %s band=%s [%s, %s) best_quality=%s",
        MAIAC_COLLECTION,
        ee_band,
        start_s,
        end_s,
        best_quality_only,
    )
    return _maiac_to_xarray(composite, region, scale_m, days)


def _maiac_to_xarray(
    collection: ee.ImageCollection,
    region: ee.Geometry,
    scale_m: float,
    times: pd.DatetimeIndex,
) -> xr.Dataset:
    """Materialise daily MAIAC composites into an xarray Dataset via ``xee``."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xarray is required to assemble MAIAC cubes.") from exc

    try:
        ds = xr.open_dataset(
            collection,
            engine="ee",
            geometry=region,
            scale=scale_m / 111_320.0,
        )
    except (ValueError, ImportError) as exc:  # pragma: no cover - needs xee
        raise ImportError(
            "Reading MAIAC composites into xarray requires the 'xee' backend "
            "(pip install xee) and valid GEE credentials. For an offline run "
            "call fetch_maiac(..., synthetic=True)."
        ) from exc

    ds["aod"].attrs["source"] = "MODIS MAIAC MCD19A2 1km (GEE)"
    ds["aod"].attrs["units"] = "1"
    return ds


__all__ = [
    "MAIAC_COLLECTION",
    "MAIAC_BANDS",
    "MAIAC_QA_BAND",
    "MAIAC_SCALE_FACTOR",
    "fetch_maiac",
]
