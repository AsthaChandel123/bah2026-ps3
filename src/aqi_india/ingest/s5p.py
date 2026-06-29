"""Sentinel-5P TROPOMI L3 trace-gas / aerosol column adapter.

Fetches daily Sentinel-5P TROPOMI Level-3 columns from Google Earth Engine using
the **exact** ``COPERNICUS/S5P/OFFL/L3_*`` asset IDs and the blueprint QA-masking
conventions, then median-composites per day over the AOI. Supported products map
one-to-one to the synthetic ``*_col`` grid variables of ``DEV_CONTRACT`` §6.1 so
the real and offline paths are interchangeable downstream:

================  =========================================  =====================
product           GEE asset (OFFL L3)                        synthetic grid var
================  =========================================  =====================
``no2``           ``COPERNICUS/S5P/OFFL/L3_NO2``             ``no2_col``
``so2``           ``COPERNICUS/S5P/OFFL/L3_SO2``             ``so2_col``
``co``            ``COPERNICUS/S5P/OFFL/L3_CO``              ``co_col``
``o3``            ``COPERNICUS/S5P/OFFL/L3_O3``              ``o3_col``
``hcho``          ``COPERNICUS/S5P/OFFL/L3_HCHO``            ``hcho_col``
``ch4``           ``COPERNICUS/S5P/OFFL/L3_CH4``             (auxiliary)
``aer_ai``        ``COPERNICUS/S5P/OFFL/L3_AER_AI``          (auxiliary)
``aer_lh``        ``COPERNICUS/S5P/OFFL/L3_AER_LH``          (auxiliary)
``cloud``         ``COPERNICUS/S5P/OFFL/L3_CLOUD``           (QA layer)
================  =========================================  =====================

QA convention (single source of truth, mirrors ``conf/datasets/s5p.yaml``):
NO2 ``qa_value>=0.75``; HCHO / SO2 / CO / CH4 / AER_LH ``>=0.5``. Compositing is
per-pixel **median** to denoise (HCHO especially — single-pixel error 30-100%).

``import ee`` is lazy (inside functions); the module imports under the light set
and the ``synthetic=True`` path delegates to :mod:`aqi_india.sim.synthetic`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import ee
    import xarray as xr

logger = get_logger("ingest.s5p")

#: Exact Earth Engine OFFL L3 asset id per product key.
S5P_COLLECTIONS: dict[str, str] = {
    "no2": "COPERNICUS/S5P/OFFL/L3_NO2",
    "so2": "COPERNICUS/S5P/OFFL/L3_SO2",
    "co": "COPERNICUS/S5P/OFFL/L3_CO",
    "o3": "COPERNICUS/S5P/OFFL/L3_O3",
    "hcho": "COPERNICUS/S5P/OFFL/L3_HCHO",
    "ch4": "COPERNICUS/S5P/OFFL/L3_CH4",
    "aer_ai": "COPERNICUS/S5P/OFFL/L3_AER_AI",
    "aer_lh": "COPERNICUS/S5P/OFFL/L3_AER_LH",
    "cloud": "COPERNICUS/S5P/OFFL/L3_CLOUD",
}

#: Primary science band per product (mol/m2 unless noted).
S5P_BANDS: dict[str, str] = {
    "no2": "tropospheric_NO2_column_number_density",
    "so2": "SO2_column_number_density",
    "co": "CO_column_number_density",
    "o3": "O3_column_number_density",
    "hcho": "tropospheric_HCHO_column_number_density",
    "ch4": "CH4_column_volume_mixing_ratio_dry_air",
    "aer_ai": "absorbing_aerosol_index",
    "aer_lh": "aerosol_height",
}

#: QA floor per product (blueprint: NO2 0.75; others 0.5).
S5P_QA_THRESHOLDS: dict[str, float] = {
    "no2": 0.75,
    "so2": 0.50,
    "co": 0.50,
    "o3": 0.50,
    "hcho": 0.50,
    "ch4": 0.50,
    "aer_lh": 0.50,
}

#: Name of the per-pixel quality band in the S5P L3 products.
QA_BAND: str = "qa_value"

#: Map each product key to the synthetic ``grid.nc`` variable it stands in for.
PRODUCT_TO_GRID_VAR: dict[str, str] = {
    "no2": "no2_col",
    "so2": "so2_col",
    "co": "co_col",
    "o3": "o3_col",
    "hcho": "hcho_col",
}

DateLike = str | date | datetime


def fetch_s5p(
    product: str,
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    qa_min: float | None = None,
    scale_m: float = 1113.0,
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Fetch a daily Sentinel-5P L3 product column over an AOI.

    Real path (``synthetic=False``): filters the exact OFFL L3 ``ImageCollection``
    to ``[start, end)`` and the AOI, QA-masks each daily image (``qa_value`` floor
    per :data:`S5P_QA_THRESHOLDS`, overridable via ``qa_min``), then per-pixel
    **median**-composites the product band into a daily cube. The result is a
    single-variable :class:`xarray.Dataset` whose data-var is the canonical
    ``*_col`` name (e.g. ``hcho_col``) on ascending ``time/lat/lon`` coords.

    Offline path (``synthetic=True``): delegates to
    :func:`aqi_india.sim.synthetic.make_grid` and returns the matching synthetic
    column variable on the same schema — no credentials, no network.

    Args:
        product: One of :data:`S5P_COLLECTIONS` keys (``no2``, ``so2``, ``co``,
            ``o3``, ``hcho``, ``ch4``, ``aer_ai``, ``aer_lh``).
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` footprint.
        qa_min: Override the per-product QA floor (else use the blueprint value).
        scale_m: Reprojection/sampling scale in metres for the GEE export.
        synthetic: If ``True``, return synthetic data instead of querying GEE.
        seed: Seed forwarded to the synthetic generator.

    Returns:
        A single-variable daily :class:`xarray.Dataset` for ``product``.

    Raises:
        ValueError: If ``product`` is unknown.
    """
    if product not in S5P_COLLECTIONS:
        raise ValueError(
            f"unknown S5P product {product!r}; valid: {sorted(S5P_COLLECTIONS)}"
        )

    if synthetic:
        return _synthetic_s5p(product, start, end, aoi, seed=seed)

    return _fetch_s5p_gee(
        product, start, end, aoi, qa_min=qa_min, scale_m=scale_m
    )


# --------------------------------------------------------------------------- #
# Synthetic (offline) path                                                     #
# --------------------------------------------------------------------------- #
def _synthetic_s5p(
    product: str,
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    seed: int,
) -> xr.Dataset:
    """Return the synthetic column field matching ``product`` (offline path)."""
    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)

    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
    fires = sim.make_fires(grid, seed=seed)
    grid = sim.inject_fire_hcho(grid, fires)

    grid_var = PRODUCT_TO_GRID_VAR.get(product)
    if grid_var is None:
        # ch4 / aer_* have no synthetic counterpart: surrogate from hcho/aod
        # shape so the offline pipeline still has a coherent field to consume.
        surrogate = "hcho_col" if product in ("ch4", "aer_lh") else "aod"
        out = grid[[surrogate]].rename({surrogate: f"{product}_col"})
        out[f"{product}_col"].attrs["note"] = (
            f"SYNTHETIC surrogate for S5P {product} (no native synthetic field)"
        )
        logger.info("fetch_s5p(synthetic): %s -> surrogate of %s", product, surrogate)
        return out

    out = grid[[grid_var]].copy()
    out[grid_var].attrs.setdefault("source", f"SYNTHETIC S5P {product}")
    logger.info(
        "fetch_s5p(synthetic): %s -> %s, %d days", product, grid_var, n_days
    )
    return out


# --------------------------------------------------------------------------- #
# Real (GEE) path                                                              #
# --------------------------------------------------------------------------- #
def _fetch_s5p_gee(
    product: str,
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    qa_min: float | None,
    scale_m: float,
) -> xr.Dataset:
    """Query, QA-mask and median-composite a daily S5P L3 column from GEE."""
    import ee

    from .gee_auth import init_ee

    init_ee()

    asset = S5P_COLLECTIONS[product]
    band = S5P_BANDS.get(product)
    if band is None:
        raise ValueError(f"product {product!r} has no science band defined")
    qa = qa_min if qa_min is not None else S5P_QA_THRESHOLDS.get(product, 0.5)

    region = ee.Geometry.Rectangle(list(aoi))
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    has_qa = product not in ("aer_ai",)  # AER_AI L3 lacks a qa_value band

    def _mask(img: ee.Image) -> ee.Image:
        masked = img
        if has_qa:
            masked = img.updateMask(img.select(QA_BAND).gte(qa))
        return masked.select(band)

    days = pd.date_range(start_s, end_s, freq="D", inclusive="left")
    images: list[ee.Image] = []
    for day in days:
        d0 = day.strftime("%Y-%m-%d")
        d1 = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        daily = (
            ee.ImageCollection(asset)
            .filterDate(d0, d1)
            .filterBounds(region)
            .map(_mask)
        )
        # Per-pixel median composite for the day (denoise + overpass fill).
        comp = daily.median().set("system:time_start", ee.Date(d0).millis())
        images.append(comp)

    composite = ee.ImageCollection(images)
    grid_var = PRODUCT_TO_GRID_VAR.get(product, f"{product}_col")
    logger.info(
        "fetch_s5p(GEE): %s [%s, %s) qa>=%.2f -> %s (export via xee/COG)",
        asset,
        start_s,
        end_s,
        qa,
        grid_var,
    )
    return _ee_collection_to_xarray(
        composite, region, scale_m, band, grid_var, days
    )


def _ee_collection_to_xarray(
    collection: ee.ImageCollection,
    region: ee.Geometry,
    scale_m: float,
    band: str,
    grid_var: str,
    times: pd.DatetimeIndex,
) -> xr.Dataset:
    """Materialise a daily GEE composite collection into an xarray Dataset.

    Prefers the ``xee`` Earth Engine xarray backend (lazy, server-side) when
    available; otherwise raises a clear, actionable error. Kept separate so the
    transport (export vs in-memory) can evolve without touching the science.

    Args:
        collection: One image per day (median composites).
        region: AOI geometry to clip/export.
        scale_m: Sampling scale in metres.
        band: Source band name to rename to ``grid_var``.
        grid_var: Canonical output variable name (e.g. ``hcho_col``).
        times: Daily time index the composites correspond to.

    Returns:
        A single-variable daily :class:`xarray.Dataset`.

    Raises:
        ImportError: If the ``xee`` backend is unavailable.
    """
    import ee  # noqa: F401  (ensures ee initialised in caller's context)

    try:
        import xarray as xr  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xarray is required to assemble S5P cubes.") from exc

    try:
        ds = xr.open_dataset(
            collection,
            engine="ee",
            geometry=region,
            scale=scale_m / 111_320.0,  # metres -> approx degrees for xee
        )
    except (ValueError, ImportError) as exc:  # pragma: no cover - needs xee+creds
        raise ImportError(
            "Reading S5P composites into xarray requires the 'xee' backend "
            "(pip install xee) and valid GEE credentials. For an offline run "
            "call fetch_s5p(..., synthetic=True)."
        ) from exc

    if band in ds:
        ds = ds.rename({band: grid_var})
    ds[grid_var].attrs["source"] = "Sentinel-5P TROPOMI L3 (GEE OFFL)"
    return ds


def fetch_all_columns(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    products: tuple[str, ...] = ("no2", "so2", "co", "o3", "hcho"),
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Fetch several S5P columns and merge them into one daily cube.

    Convenience wrapper that calls :func:`fetch_s5p` per product and merges the
    single-variable Datasets on the shared ``time/lat/lon`` grid — yielding the
    five canonical ``*_col`` variables of the synthetic schema.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: AOI bbox.
        products: Product keys to fetch and merge.
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A merged daily :class:`xarray.Dataset` with one ``*_col`` var per product.
    """
    import xarray as xr

    parts = [
        fetch_s5p(p, start, end, aoi, synthetic=synthetic, seed=seed)
        for p in products
    ]
    merged = xr.merge(parts, compat="override", join="outer")
    logger.info("fetch_all_columns: merged %d products", len(products))
    return merged


def column_to_surface_units(values: np.ndarray, product: str) -> np.ndarray:
    """Pass-through documenting that column->surface conversion is the ML's job.

    Satellite columns are kept in ``mol/m2`` (the synthetic schema); converting a
    column to a CPCB surface concentration (ug/m3, CO mg/m3) is a *learned*
    mapping done by the models stage, **never** a fixed factor here. This helper
    exists so adapters never accidentally bake in a bogus constant; it returns
    the input unchanged and is documented for clarity.

    Args:
        values: Column values in ``mol/m2``.
        product: Product key (for logging only).

    Returns:
        ``values`` unchanged.
    """
    logger.debug(
        "column_to_surface_units: %s columns kept in mol/m2 (ML maps to surface)",
        product,
    )
    return np.asarray(values)


__all__ = [
    "S5P_COLLECTIONS",
    "S5P_BANDS",
    "S5P_QA_THRESHOLDS",
    "QA_BAND",
    "PRODUCT_TO_GRID_VAR",
    "fetch_s5p",
    "fetch_all_columns",
    "column_to_surface_units",
]
