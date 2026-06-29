"""Google Earth Engine authentication and server-side reduction helpers.

This module is the single entry point for everything that talks to **Google
Earth Engine (GEE)** in the ingest layer. GEE co-locates almost every PS3 input
as analysis-ready ``ImageCollection``s and runs *lazy, server-side* reductions,
so the analyst only ever pulls compact tables / COGs — O(1) analyst effort and
no bulk download (see ``docs/DATA_SOURCES.md`` and the synthesis blueprint).

Nothing here is imported at module top: ``earthengine-api`` is a heavy, optional,
credentialed dependency, so ``import ee`` is performed lazily *inside* every
function. The package therefore imports under the light dependency set alone and
the offline synthetic demo never touches GEE.

Typical use::

    from aqi_india.ingest.gee_auth import init_ee, mask_qa, composite_median
    init_ee(project="my-gcp-project")          # once per process
    ic = composite_median(filtered_collection)  # server-side median image

Authentication supports three modes, tried in order:

1. an explicit **service-account** key file (headless / CI — recommended for the
   reproducible pipeline);
2. the cached **user credentials** written by ``earthengine authenticate``;
3. **Application Default Credentials** (ADC) on GCP.

All reduction helpers (`reduce_regions`, `mask_qa`, `composite_median`) are thin,
well-documented wrappers that keep the QA-masking and compositing conventions of
the blueprint in exactly one place so every source adapter masks identically.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import ee
    import pandas as pd

logger = get_logger("ingest.gee_auth")

#: Process-level flag so ``init_ee`` is idempotent and cheap to re-call.
_EE_INITIALISED: bool = False

#: Default reduction scale (m) for national-scale ``reduceRegions`` calls. 1113 m
#: matches the S5P L3 ~0.01 deg grid; pair with a large ``tileScale`` to avoid
#: "computation timed out" on India-wide reductions.
DEFAULT_SCALE_M: float = 1113.0

#: Default ``tileScale`` for national reductions (trades memory for parallelism).
DEFAULT_TILE_SCALE: int = 8


def init_ee(
    *,
    service_account: str | None = None,
    key_file: str | None = None,
    project: str | None = None,
    force: bool = False,
) -> bool:
    """Initialise the Earth Engine client (idempotent).

    Authentication is attempted in priority order: explicit service-account key
    file, then cached user credentials, then Application Default Credentials. The
    GCP project is taken from the ``project`` argument or the
    ``EARTHENGINE_PROJECT`` / ``GOOGLE_CLOUD_PROJECT`` environment variables.

    Args:
        service_account: Service-account e-mail. If given together with
            ``key_file`` (or ``EE_SERVICE_ACCOUNT_KEY``), headless service-account
            auth is used — the recommended mode for CI / the reproducible run.
        key_file: Path to the service-account JSON key. Falls back to the
            ``EE_SERVICE_ACCOUNT_KEY`` environment variable.
        project: GCP/Earth-Engine cloud project id. Falls back to
            ``EARTHENGINE_PROJECT`` / ``GOOGLE_CLOUD_PROJECT``.
        force: Re-initialise even if a previous call already succeeded.

    Returns:
        ``True`` once Earth Engine is initialised in this process.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
        RuntimeError: If every authentication mode fails.
    """
    global _EE_INITIALISED
    if _EE_INITIALISED and not force:
        return True

    try:
        import ee
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "earthengine-api is required for GEE ingestion. Install the 'gee' "
            "extra (pip install 'aqi_india[gee]') or run the offline demo with "
            "synthetic=True."
        ) from exc

    project = project or os.getenv("EARTHENGINE_PROJECT") or os.getenv(
        "GOOGLE_CLOUD_PROJECT"
    )
    key_file = key_file or os.getenv("EE_SERVICE_ACCOUNT_KEY")

    # --- 1. Service-account (headless) --------------------------------- #
    if key_file:
        sa = service_account or os.getenv("EE_SERVICE_ACCOUNT")
        try:
            credentials = ee.ServiceAccountCredentials(sa, key_file)
            ee.Initialize(credentials, project=project, opt_url=None)
            _EE_INITIALISED = True
            logger.info("Earth Engine initialised via service account (%s).", sa)
            return True
        except Exception as exc:  # noqa: BLE001 - fall through to next mode
            logger.warning("Service-account EE auth failed: %s", exc)

    # --- 2. Cached user credentials / 3. ADC --------------------------- #
    try:
        ee.Initialize(project=project)
        _EE_INITIALISED = True
        logger.info("Earth Engine initialised via cached/default credentials.")
        return True
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not initialise Earth Engine. Provide a service-account key "
            "(key_file=/EE_SERVICE_ACCOUNT_KEY) or run 'earthengine "
            "authenticate' first. Original error: " + str(exc)
        ) from exc


def is_initialised() -> bool:
    """Return whether :func:`init_ee` has succeeded in this process."""
    return _EE_INITIALISED


def mask_qa(image: ee.Image, qa_band: str, threshold: float) -> ee.Image:
    """Mask an Earth Engine image where a QA band falls below ``threshold``.

    Implements the blueprint QA convention (NO2 ``qa_value>=0.75``;
    HCHO/SO2/CO/CH4/AER_LH ``>=0.5``). The mask is applied with
    ``updateMask`` so the masked pixels become transparent (excluded from any
    subsequent reduction / compositing), not zeroed.

    Args:
        image: Source ``ee.Image`` carrying a continuous 0-1 QA band.
        qa_band: Name of the QA band (e.g. ``"qa_value"``,
            ``"cloud_fraction"``).
        threshold: Minimum acceptable QA value; pixels strictly below are masked.

    Returns:
        The QA-masked ``ee.Image``.
    """
    qa = image.select(qa_band)
    return image.updateMask(qa.gte(threshold))


def composite_median(
    collection: ee.ImageCollection,
    band: str | None = None,
) -> ee.Image:
    """Reduce an image collection to a per-pixel **median** composite.

    Median compositing is the blueprint default for denoising daily L3 columns
    and filling single-overpass holes before the ML stage. Optionally selects a
    single band first to keep the server-side computation lean.

    Args:
        collection: The (already filtered / QA-masked) ``ee.ImageCollection``.
        band: Optional band to ``select`` before reducing.

    Returns:
        A single median-composited ``ee.Image``.
    """
    ic = collection.select(band) if band is not None else collection
    return ic.median()


def reduce_regions(
    image: ee.Image,
    regions: ee.FeatureCollection,
    *,
    reducer: ee.Reducer | None = None,
    scale: float = DEFAULT_SCALE_M,
    tile_scale: int = DEFAULT_TILE_SCALE,
) -> ee.FeatureCollection:
    """Server-side ``reduceRegions`` of an image over point/polygon features.

    This is the workhorse that samples satellite columns at CPCB station
    locations (turning a planetary image into a compact table the analyst can
    export). Defaults to a mean reducer at the national S5P scale with a large
    ``tileScale`` so India-wide reductions do not time out.

    Args:
        image: The ``ee.Image`` to sample (e.g. a median composite).
        regions: Features to reduce over (CPCB station points/buffers).
        reducer: Earth Engine reducer; defaults to ``ee.Reducer.mean()``.
        scale: Nominal reduction scale in metres.
        tile_scale: GEE ``tileScale`` (memory/parallelism trade-off).

    Returns:
        A ``ee.FeatureCollection`` with the reduced values attached to each
        input feature.
    """
    import ee

    reducer = reducer if reducer is not None else ee.Reducer.mean()
    return image.reduceRegions(
        collection=regions,
        reducer=reducer,
        scale=scale,
        tileScale=tile_scale,
    )


def export_image_to_cog(
    image: ee.Image,
    *,
    description: str,
    bucket: str | None = None,
    file_name_prefix: str,
    region: ee.Geometry,
    scale: float = DEFAULT_SCALE_M,
    crs: str = "EPSG:4326",
    start: bool = True,
) -> ee.batch.Task:
    """Export an image as a Cloud-Optimized GeoTIFF (Drive or GCS).

    When ``bucket`` is provided the export targets Google Cloud Storage
    (``Export.image.toCloudStorage``) with a COG-friendly format option;
    otherwise it falls back to Google Drive. The task is started unless
    ``start=False`` (useful for dry-runs/tests).

    Args:
        image: Image to export.
        description: Human-readable Earth Engine task description.
        bucket: GCS bucket name; ``None`` exports to Drive instead.
        file_name_prefix: Output object/file prefix.
        region: Export footprint geometry.
        scale: Output pixel size in metres.
        crs: Output CRS.
        start: Whether to immediately ``task.start()``.

    Returns:
        The created (optionally started) ``ee.batch.Task``.
    """
    import ee

    fmt_opts = {"cloudOptimized": True}
    if bucket:
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=description,
            bucket=bucket,
            fileNamePrefix=file_name_prefix,
            region=region,
            scale=scale,
            crs=crs,
            fileFormat="GeoTIFF",
            formatOptions=fmt_opts,
            maxPixels=1e13,
        )
    else:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            fileNamePrefix=file_name_prefix,
            region=region,
            scale=scale,
            crs=crs,
            fileFormat="GeoTIFF",
            formatOptions=fmt_opts,
            maxPixels=1e13,
        )
    if start:
        task.start()
        logger.info("Started COG export task %r.", description)
    return task


def export_table(
    collection: ee.FeatureCollection,
    *,
    description: str,
    bucket: str | None = None,
    file_name_prefix: str,
    file_format: str = "CSV",
    start: bool = True,
) -> ee.batch.Task:
    """Export a feature collection as a table (CSV / GeoJSON) to GCS or Drive.

    Args:
        collection: The ``ee.FeatureCollection`` to export (e.g. the output of
            :func:`reduce_regions`).
        description: Earth Engine task description.
        bucket: GCS bucket name; ``None`` exports to Drive.
        file_name_prefix: Output object/file prefix.
        file_format: ``"CSV"`` or ``"GeoJSON"``.
        start: Whether to immediately start the task.

    Returns:
        The created (optionally started) ``ee.batch.Task``.
    """
    import ee

    kwargs: dict[str, Any] = {
        "collection": collection,
        "description": description,
        "fileNamePrefix": file_name_prefix,
        "fileFormat": file_format,
    }
    if bucket:
        kwargs["bucket"] = bucket
        task = ee.batch.Export.table.toCloudStorage(**kwargs)
    else:
        task = ee.batch.Export.table.toDrive(**kwargs)
    if start:
        task.start()
        logger.info("Started table export task %r.", description)
    return task


def ee_to_dataframe(collection: ee.FeatureCollection) -> pd.DataFrame:
    """Materialise a (small) feature collection into a pandas DataFrame.

    Pulls the collection client-side via ``getInfo`` and flattens each feature's
    ``properties`` into a row. Intended for compact reduced tables only — never
    call on a raw planetary collection.

    Args:
        collection: A small ``ee.FeatureCollection``.

    Returns:
        A pandas DataFrame of the features' properties.
    """
    import pandas as pd

    info = collection.getInfo()
    rows = [feat.get("properties", {}) for feat in info.get("features", [])]
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_SCALE_M",
    "DEFAULT_TILE_SCALE",
    "init_ee",
    "is_initialised",
    "mask_qa",
    "composite_median",
    "reduce_regions",
    "export_image_to_cog",
    "export_table",
    "ee_to_dataframe",
]
