"""NASA FIRMS active-fire adapter (VIIRS 375 m primary + MODIS climatology).

Fetches active-fire detections from the **NASA FIRMS area API** (CSV) for one or
more sensors and assembles them into the project's fire ``GeoDataFrame`` schema
(``DEV_CONTRACT`` §6.3): ``date, lat, lon, frp, sensor, h3_res7`` + point
geometry. VIIRS 375 m (S-NPP + NOAA-20 + NOAA-21) is the primary source —
3-5x more small stubble fires than MODIS — with MODIS 1 km for the long
climatology (Objective-2).

The FIRMS API needs a free ``MAP_KEY`` (env ``FIRMS_MAP_KEY``); ``requests`` is
lazy-imported. The ``synthetic=True`` path delegates to
:func:`aqi_india.sim.synthetic.make_fires` so the offline pipeline has a realistic
Punjab/Haryana + forest fire cloud without any network call.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..features.h3_index import cells_for_arrays
from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

logger = get_logger("ingest.firms")

#: FIRMS area-CSV API template (filled per request).
FIRMS_AREA_TEMPLATE: str = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{map_key}/{source}/{w},{s},{e},{n}/{day_range}/{date}"
)

#: VIIRS 375 m NRT sources (primary) + MODIS NRT.
FIRMS_NRT_SOURCES: tuple[str, ...] = (
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
)

#: Archive (standard-processing) sources for historical climatology.
FIRMS_ARCHIVE_SOURCES: tuple[str, ...] = ("VIIRS_SNPP_SP", "MODIS_SP")

#: Map a raw FIRMS source token to the canonical ``sensor`` label in the schema.
_SENSOR_LABELS: dict[str, str] = {
    "VIIRS_SNPP_NRT": "VIIRS_SNPP",
    "VIIRS_SNPP_SP": "VIIRS_SNPP",
    "VIIRS_NOAA20_NRT": "VIIRS_NOAA20",
    "VIIRS_NOAA21_NRT": "VIIRS_NOAA21",
    "MODIS_NRT": "MODIS",
    "MODIS_SP": "MODIS",
}

#: Environment variable holding the FIRMS MAP_KEY.
MAP_KEY_ENV: str = "FIRMS_MAP_KEY"

#: FIRMS API caps each area request at 10 days.
MAX_DAY_RANGE: int = 10

DateLike = str | date | datetime


def fetch_firms(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    sources: tuple[str, ...] = FIRMS_NRT_SOURCES,
    map_key: str | None = None,
    synthetic: bool = False,
    season: str = "oct_nov",
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Fetch FIRMS active-fire detections as a fire ``GeoDataFrame``.

    Real path: for each sensor in ``sources`` and each <=10-day chunk in
    ``[start, end)``, calls the FIRMS area-CSV API over ``aoi``, parses the CSV,
    normalises columns to the schema, assigns ``h3_res7``, and concatenates.

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_fires` (built
    on a synthetic grid spanning the window) and returns the same schema.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` query box.
        sources: FIRMS source tokens (default VIIRS 375 m trio + MODIS).
        map_key: FIRMS MAP_KEY; falls back to the ``FIRMS_MAP_KEY`` env var.
        synthetic: Offline synthetic path when ``True``.
        season: Burning window forwarded to the synthetic generator.
        seed: Synthetic seed.

    Returns:
        A point :class:`geopandas.GeoDataFrame` (EPSG:4326) with columns
        ``date, lat, lon, frp, sensor, h3_res7`` + ``geometry``.

    Raises:
        RuntimeError: If no MAP_KEY is available on the real path.
    """
    if synthetic:
        return _synthetic_firms(start, end, aoi, season=season, seed=seed)

    map_key = map_key or os.getenv(MAP_KEY_ENV)
    if not map_key:
        raise RuntimeError(
            "FIRMS MAP_KEY required for live fire ingestion. Set the "
            f"{MAP_KEY_ENV} environment variable (free key from "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/) or call "
            "fetch_firms(..., synthetic=True)."
        )

    frames = [
        _fetch_firms_source(src, start, end, aoi, map_key)
        for src in sources
    ]
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        logger.warning("FIRMS returned no detections for the requested window.")
        return _empty_fire_gdf()

    df = pd.concat(frames, ignore_index=True)
    return _to_fire_gdf(df)


def _synthetic_firms(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    season: str,
    seed: int,
) -> gpd.GeoDataFrame:
    """Return synthetic FIRMS-like fires for the window (offline path)."""
    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)
    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
    fires = sim.make_fires(grid, season=season, seed=seed)
    logger.info("fetch_firms(synthetic): %d detections (%s)", len(fires), season)
    return fires


def _fetch_firms_source(
    source: str,
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    map_key: str,
) -> pd.DataFrame | None:
    """Fetch + parse all <=10-day chunks for a single FIRMS source."""
    import io

    import requests

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    min_lon, min_lat, max_lon, max_lat = aoi
    chunks: list[pd.DataFrame] = []

    cursor = t0
    while cursor < t1:
        span = min(MAX_DAY_RANGE, int((t1 - cursor).days))
        url = FIRMS_AREA_TEMPLATE.format(
            map_key=map_key,
            source=source,
            w=min_lon,
            s=min_lat,
            e=max_lon,
            n=max_lat,
            day_range=span,
            date=cursor.strftime("%Y-%m-%d"),
        )
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("FIRMS request failed (%s, %s): %s", source, cursor.date(), exc)
            cursor += pd.Timedelta(days=span)
            continue
        text = resp.text.strip()
        if text and not text.lower().startswith("no data"):
            try:
                part = pd.read_csv(io.StringIO(text))
                if len(part):
                    part["__source"] = source
                    chunks.append(part)
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                logger.warning("FIRMS CSV parse failed (%s): %s", source, exc)
        cursor += pd.Timedelta(days=span)

    if not chunks:
        return None
    out = pd.concat(chunks, ignore_index=True)
    logger.info("FIRMS %s: %d detections", source, len(out))
    return out


def _to_fire_gdf(raw: pd.DataFrame) -> gpd.GeoDataFrame:
    """Normalise raw FIRMS CSV columns into the fire schema GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import Point

    lat = pd.to_numeric(raw.get("latitude"), errors="coerce").to_numpy()
    lon = pd.to_numeric(raw.get("longitude"), errors="coerce").to_numpy()
    frp = pd.to_numeric(raw.get("frp"), errors="coerce").to_numpy()
    acq = pd.to_datetime(raw.get("acq_date"), errors="coerce")
    sensor = (
        raw.get("__source", pd.Series(["UNKNOWN"] * len(raw)))
        .map(lambda s: _SENSOR_LABELS.get(s, str(s)))
        .to_numpy()
    )

    df = pd.DataFrame(
        {
            "date": acq.dt.normalize().to_numpy(),
            "lat": lat.astype(np.float64),
            "lon": lon.astype(np.float64),
            "frp": frp.astype(np.float64),
            "sensor": sensor.astype(object),
        }
    )
    df = df.dropna(subset=["lat", "lon", "date"]).reset_index(drop=True)
    df["h3_res7"] = cells_for_arrays(
        df["lat"].to_numpy(), df["lon"].to_numpy(), 7
    ).astype(object)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    geom = [Point(xy) for xy in zip(df["lon"], df["lat"], strict=False)]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    logger.info("fetch_firms: %d total detections normalised", len(gdf))
    return gdf


def _empty_fire_gdf() -> gpd.GeoDataFrame:
    """Return an empty fire GeoDataFrame with the correct schema."""
    import geopandas as gpd

    cols = ["date", "lat", "lon", "frp", "sensor", "h3_res7"]
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")


__all__ = [
    "FIRMS_AREA_TEMPLATE",
    "FIRMS_NRT_SOURCES",
    "FIRMS_ARCHIVE_SOURCES",
    "MAP_KEY_ENV",
    "fetch_firms",
]
