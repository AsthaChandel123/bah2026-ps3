"""Artifact exporters bridging Earth Engine / in-memory data to local formats.

Thin, well-documented helpers that turn ingest outputs into the project's
standard, fast-platform artifacts:

* :func:`gee_image_to_cog` — export an ``ee.Image`` as a Cloud-Optimized GeoTIFF
  (Drive/GCS) for COG/TiTiler serving;
* :func:`to_geoparquet` — write a (Geo)DataFrame as Hilbert-/date-friendly
  GeoParquet, the keyed feature-matrix format consumed by the fusion/ML stages;
* :func:`ee_table_to_df` — materialise a small ``ee.FeatureCollection`` (e.g. the
  CPCB ``reduceRegions`` output) into a pandas DataFrame.

All Earth Engine work is delegated to :mod:`aqi_india.ingest.gee_auth` and
``import ee`` stays lazy; the GeoParquet path needs only the light dep set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..utils.io import save_geoparquet
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import ee
    import geopandas as gpd
    import pandas as pd

logger = get_logger("ingest.exporters")


def gee_image_to_cog(
    image: ee.Image,
    *,
    description: str,
    file_name_prefix: str,
    region: ee.Geometry,
    bucket: str | None = None,
    scale: float = 1113.0,
    crs: str = "EPSG:4326",
    start: bool = True,
) -> ee.batch.Task:
    """Export an Earth Engine image as a Cloud-Optimized GeoTIFF.

    Delegates to :func:`aqi_india.ingest.gee_auth.export_image_to_cog` (GCS when
    ``bucket`` is given, else Drive), ensuring the COG ``formatOptions`` are set so
    the artifact streams via HTTP range reads in the serving layer.

    Args:
        image: The ``ee.Image`` to export (e.g. a daily AQI/HCHO composite).
        description: Earth Engine task description.
        file_name_prefix: Output object/file prefix.
        region: Export footprint geometry.
        bucket: GCS bucket name; ``None`` exports to Drive.
        scale: Output pixel size in metres.
        crs: Output CRS.
        start: Whether to start the export task immediately.

    Returns:
        The created (optionally started) ``ee.batch.Task``.
    """
    from .gee_auth import export_image_to_cog as _export

    return _export(
        image,
        description=description,
        bucket=bucket,
        file_name_prefix=file_name_prefix,
        region=region,
        scale=scale,
        crs=crs,
        start=start,
    )


def to_geoparquet(
    data: gpd.GeoDataFrame | pd.DataFrame,
    path: str | Path,
    *,
    sort_by: str | None = None,
    geometry: str | None = None,
) -> Path:
    """Write a (Geo)DataFrame to GeoParquet, optionally sorted for locality.

    If ``data`` is a plain DataFrame with point columns, pass ``geometry`` (a
    ``(lon_col, lat_col)`` tuple is not accepted here — supply an existing
    geometry column name) to promote it to a GeoDataFrame first. Sorting by a
    date/H3 column (``sort_by``) improves predicate-pushdown locality for the
    DuckDB/feature-matrix consumers.

    Args:
        data: The frame to write. A ``GeoDataFrame`` is written as-is; a plain
            ``DataFrame`` is written via geopandas if it already carries a
            geometry column named ``geometry`` (or ``geometry=``).
        path: Output ``.geoparquet`` path.
        sort_by: Optional column to sort rows by before writing (e.g. ``"date"``
            or ``"h3_res7"``).
        geometry: Name of an existing geometry column to set as active geometry.

    Returns:
        The written :class:`pathlib.Path`.
    """
    import geopandas as gpd

    df = data.copy()
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, kind="stable").reset_index(drop=True)

    if not isinstance(df, gpd.GeoDataFrame):
        geom_col = geometry or ("geometry" if "geometry" in df.columns else None)
        if geom_col is None:
            raise ValueError(
                "to_geoparquet needs a GeoDataFrame or a DataFrame with a "
                "geometry column (pass geometry=<col>)."
            )
        df = gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:4326")

    out = save_geoparquet(df, path)
    logger.info("to_geoparquet: wrote %d rows -> %s", len(df), out)
    return out


def ee_table_to_df(collection: ee.FeatureCollection) -> pd.DataFrame:
    """Materialise a small ``ee.FeatureCollection`` into a pandas DataFrame.

    Wraps :func:`aqi_india.ingest.gee_auth.ee_to_dataframe`. Use only on compact
    reduced tables (e.g. station ``reduceRegions`` output) — never on a raw
    planetary collection.

    Args:
        collection: A small ``ee.FeatureCollection``.

    Returns:
        A pandas DataFrame of the collection's feature properties.
    """
    from .gee_auth import ee_to_dataframe

    return ee_to_dataframe(collection)


__all__ = ["gee_image_to_cog", "to_geoparquet", "ee_table_to_df"]
