"""CPCB CAAQMS ground-truth adapter + unified station-database builder.

CPCB Continuous Ambient Air Quality Monitoring Stations are the **primary
training labels** (the ``y``) and the RMSE/R/MAE reference for Objective-1. This
module:

* fetches the latest CPCB observations from the **data.gov.in OGD** resource
  (``3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69``) — JSON over HTTP, ``requests`` lazy;
* builds a **unified station database** keyed by a *stable* ``station_id``,
  de-duplicates CPCB against OpenAQ by a ``<1 km`` coordinate match (OpenAQ
  mirrors CPCB, so the two must not double-count), and **tags** every station as
  ``label`` / ``validation`` / ``gapfill`` so the model never trains and validates
  on the same physical sensor (the central CV-leakage mitigation);
* aggregates hourly observations to the daily NAQI schema (mean of valid hours,
  >= 75% completeness; CO/O3 as 8-hour-max, others as 24-hour mean).

The ``synthetic=True`` path delegates to
:func:`aqi_india.sim.synthetic.make_stations` so the offline pipeline has the full
station-day label table without any data.gov.in / OpenAQ key.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..aqi.breakpoints import AVERAGING_HOURS
from ..features.h3_index import cells_for_arrays
from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

logger = get_logger("ingest.cpcb")

#: data.gov.in OGD resource id for the latest CPCB AQI feed.
OGD_RESOURCE_ID: str = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

#: OGD JSON endpoint template.
OGD_TEMPLATE: str = (
    "https://api.data.gov.in/resource/{resource_id}"
    "?api-key={key}&format=json&limit={limit}&offset={offset}"
)

#: Environment variable holding the data.gov.in API key.
API_KEY_ENV: str = "DATA_GOV_IN_API_KEY"

#: CPCB pollutant labels (canonical project keys; CO is mg/m3, rest ug/m3).
CPCB_POLLUTANTS: tuple[str, ...] = ("pm25", "pm10", "no2", "so2", "co", "o3", "nh3")

#: OGD raw pollutant-name spellings -> canonical keys.
_POLLUTANT_ALIASES: dict[str, str] = {
    "pm2.5": "pm25",
    "pm25": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "so2": "so2",
    "co": "co",
    "ozone": "o3",
    "o3": "o3",
    "nh3": "nh3",
}

#: Minimum hourly completeness for a valid daily aggregate (CPCB QA rule).
MIN_HOURLY_COMPLETENESS: float = 0.75

#: Dedup radius (km): OpenAQ <-> CPCB matches within this distance are merged.
DEDUP_RADIUS_KM: float = 1.0

#: Station role tags used to prevent train/validate leakage on the same sensor.
STATION_TAGS: tuple[str, ...] = ("label", "validation", "gapfill")

DateLike = str | date | datetime


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def fetch_cpcb(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    api_key: str | None = None,
    openaq: pd.DataFrame | None = None,
    validation_frac: float = 0.18,
    seed: int = 42,
    synthetic: bool = False,
    n_stations: int = 120,
) -> gpd.GeoDataFrame:
    """Fetch CPCB station labels and build the tagged unified station DB.

    Real path: pulls the OGD resource, normalises to an hourly long table,
    aggregates to the daily NAQI schema, merges with an optional ``openaq`` frame
    (de-duplicating within :data:`DEDUP_RADIUS_KM`), assigns stable ``station_id``
    and ``h3_res7``, and tags stations ``label``/``validation``/``gapfill``.

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_stations` and
    then applies the same stable-id / tagging post-processing so downstream CV
    code sees an identical schema.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` clip footprint.
        api_key: data.gov.in key; falls back to ``DATA_GOV_IN_API_KEY``.
        openaq: Optional OpenAQ station-day frame to reconcile/dedup against.
        validation_frac: Fraction of stations to lock as the validation set.
        seed: RNG seed for the deterministic validation split.
        synthetic: Offline synthetic path when ``True``.
        n_stations: Station count for the synthetic path.

    Returns:
        A point :class:`geopandas.GeoDataFrame` (EPSG:4326), one row per
        station-day, with the §6.2 columns plus ``source`` and ``tag``.
    """
    if synthetic:
        from ..sim import synthetic as sim

        t0 = pd.Timestamp(start).normalize()
        t1 = pd.Timestamp(end).normalize()
        n_days = max(int((t1 - t0).days), 1)
        grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
        stations = sim.make_stations(grid, n=n_stations, seed=seed)
        stations = _attach_source(stations, "CPCB_SYN")
        return tag_stations(stations, validation_frac=validation_frac, seed=seed)

    api_key = api_key or os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            "data.gov.in API key required for live CPCB ingestion. Set "
            f"{API_KEY_ENV} or call fetch_cpcb(..., synthetic=True)."
        )

    hourly = _fetch_ogd_hourly(api_key, aoi)
    daily = aggregate_hourly_to_daily(hourly)
    daily = _attach_source(daily, "CPCB")

    if openaq is not None and len(openaq):
        oaq = _attach_source(openaq, "OpenAQ")
        daily = dedup_against_openaq(daily, oaq, radius_km=DEDUP_RADIUS_KM)

    daily = _finalise_station_db(daily, start, end)
    return tag_stations(daily, validation_frac=validation_frac, seed=seed)


# --------------------------------------------------------------------------- #
# OGD fetch + normalisation                                                    #
# --------------------------------------------------------------------------- #
def _fetch_ogd_hourly(
    api_key: str, aoi: tuple[float, float, float, float]
) -> pd.DataFrame:
    """Page through the OGD resource and return a normalised hourly long table."""
    import requests

    rows: list[dict] = []
    offset, limit = 0, 1000
    while True:
        url = OGD_TEMPLATE.format(
            resource_id=OGD_RESOURCE_ID, key=api_key, limit=limit, offset=offset
        )
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("OGD request failed at offset %d: %s", offset, exc)
            break
        records = payload.get("records", [])
        if not records:
            break
        rows.extend(records)
        if len(records) < limit:
            break
        offset += limit

    if not rows:
        logger.warning("OGD returned no CPCB records.")
        return _empty_hourly()

    return _normalise_ogd_records(rows, aoi)


def _normalise_ogd_records(
    records: list[dict], aoi: tuple[float, float, float, float]
) -> pd.DataFrame:
    """Reshape raw OGD records into a tidy hourly long table.

    The OGD feed is one row per (station, pollutant) with ``pollutant_id`` and a
    value column; this pivots it to one row per station-hour with canonical
    pollutant columns, clips to the AOI, and parses timestamps.

    Args:
        records: Raw OGD JSON records.
        aoi: Clip bbox.

    Returns:
        Hourly long table with ``station_id_raw, name, lat, lon, time`` + one
        column per canonical pollutant.
    """
    raw = pd.DataFrame.from_records(records)

    def _col(*names: str) -> pd.Series:
        for n in names:
            if n in raw:
                return raw[n]
        return pd.Series([np.nan] * len(raw))

    name = _col("station", "station_name").astype(str)
    city = _col("city").astype(str)
    state = _col("state").astype(str)
    lat = pd.to_numeric(_col("latitude"), errors="coerce")
    lon = pd.to_numeric(_col("longitude"), errors="coerce")
    pollutant = _col("pollutant_id", "pollutant").astype(str).str.lower().str.strip()
    value = pd.to_numeric(
        _col("pollutant_avg", "avg_value", "pollutant_value"), errors="coerce"
    )
    ts = pd.to_datetime(
        _col("last_update", "last_updated"), errors="coerce", dayfirst=True
    )

    tidy = pd.DataFrame(
        {
            "name": name,
            "city": city,
            "state": state,
            "lat": lat,
            "lon": lon,
            "time": ts,
            "pollutant": pollutant.map(_POLLUTANT_ALIASES),
            "value": value,
        }
    ).dropna(subset=["lat", "lon", "pollutant", "time"])

    min_lon, min_lat, max_lon, max_lat = aoi
    tidy = tidy[
        tidy["lat"].between(min_lat, max_lat) & tidy["lon"].between(min_lon, max_lon)
    ]
    # Pivot pollutants into columns, one row per station-hour.
    tidy["station_id_raw"] = tidy["name"].str.strip() + "|" + tidy["city"].str.strip()
    idx = ["station_id_raw", "name", "city", "state", "lat", "lon", "time"]
    wide = (
        tidy.pivot_table(
            index=idx, columns="pollutant", values="value", aggfunc="mean"
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for p in CPCB_POLLUTANTS:
        if p not in wide:
            wide[p] = np.nan
    logger.info("OGD: normalised %d records -> %d station-hours", len(raw), len(wide))
    return wide


def _empty_hourly() -> pd.DataFrame:
    cols = ["station_id_raw", "name", "city", "state", "lat", "lon", "time"]
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    for p in CPCB_POLLUTANTS:
        df[p] = pd.Series(dtype="float64")
    return df


# --------------------------------------------------------------------------- #
# Hourly -> daily aggregation (NAQI averaging rule)                            #
# --------------------------------------------------------------------------- #
def aggregate_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an hourly station table to daily NAQI-rule values.

    Per station-day: require >= :data:`MIN_HOURLY_COMPLETENESS` valid hours; then
    CO and O3 are reported as the maximum 8-hour rolling mean
    (:data:`AVERAGING_HOURS` == 8) and all other pollutants as the 24-hour mean.

    Args:
        hourly: Hourly long table with ``station_id_raw, name, lat, lon, time`` +
            one column per pollutant.

    Returns:
        Daily table, one row per station-day, with the canonical pollutant
        columns plus ``station_id_raw, name, lat, lon, time``.
    """
    if hourly.empty:
        return hourly.assign(time=pd.to_datetime([]))

    df = hourly.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].dt.normalize()
    pollutants = [p for p in CPCB_POLLUTANTS if p in df]

    out_rows: list[dict] = []
    meta_cols = ["station_id_raw", "name", "lat", "lon"]
    for (sid, day), grp in df.groupby(["station_id_raw", "day"], sort=False):
        row: dict[str, object] = {
            "station_id_raw": sid,
            "name": grp["name"].iloc[0],
            "lat": float(grp["lat"].iloc[0]),
            "lon": float(grp["lon"].iloc[0]),
            "time": day,
        }
        for p in pollutants:
            series = grp[[ "time", p]].dropna()
            completeness = len(series) / 24.0
            if completeness < MIN_HOURLY_COMPLETENESS:
                row[p] = np.nan
                continue
            if AVERAGING_HOURS.get(p, 24) == 8:
                row[p] = _max_rolling_mean(series.set_index("time")[p], window=8)
            else:
                row[p] = float(series[p].mean())
        out_rows.append(row)

    daily = pd.DataFrame(out_rows)
    for p in CPCB_POLLUTANTS:
        if p not in daily:
            daily[p] = np.nan
    _ = meta_cols
    logger.info("aggregate_hourly_to_daily: %d station-days", len(daily))
    return daily


def _max_rolling_mean(series: pd.Series, window: int) -> float:
    """Return the maximum ``window``-hour rolling mean of an hourly series."""
    if series.empty:
        return float("nan")
    hourly = series.sort_index().resample("1h").mean()
    roll = hourly.rolling(window=window, min_periods=max(1, int(0.75 * window))).mean()
    val = roll.max()
    return float(val) if pd.notna(val) else float("nan")


# --------------------------------------------------------------------------- #
# Unified station DB: stable id, dedup, tagging                                #
# --------------------------------------------------------------------------- #
def make_station_id(name: str, lat: float, lon: float) -> str:
    """Return a stable, collision-resistant ``station_id`` for a sensor.

    The id is deterministic in the station name and its rounded coordinates
    (~100 m precision), so the same physical sensor always maps to the same id
    across runs and networks — the prerequisite for the tag-based leakage guard.

    Args:
        name: Station/site name.
        lat: Latitude (deg).
        lon: Longitude (deg).

    Returns:
        A ``"CPCB-XXXXXXXX"`` style stable id.
    """
    key = f"{str(name).strip().lower()}|{round(float(lat), 3)}|{round(float(lon), 3)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8].upper()
    return f"CPCB-{digest}"


def dedup_against_openaq(
    cpcb: pd.DataFrame, openaq: pd.DataFrame, *, radius_km: float = DEDUP_RADIUS_KM
) -> pd.DataFrame:
    """Merge CPCB and OpenAQ station-day tables, dropping OpenAQ mirrors.

    OpenAQ ingests CPCB, so an OpenAQ station within ``radius_km`` of a CPCB
    station is the **same physical sensor** and is dropped to avoid double-
    counting (and train/validate leakage). OpenAQ stations with no nearby CPCB
    match are kept (genuine spatial densification). Uses a cKDTree on an
    equirectangular projection for an O(n log n) match.

    Args:
        cpcb: CPCB daily station table (``lat``/``lon`` present).
        openaq: OpenAQ daily station table (``lat``/``lon`` present).
        radius_km: Match radius in kilometres.

    Returns:
        The concatenated table with mirrored OpenAQ rows removed.
    """
    if openaq.empty:
        return cpcb
    from scipy.spatial import cKDTree

    cpcb_pts = cpcb.drop_duplicates("station_id_raw")[["lat", "lon"]].to_numpy()
    if len(cpcb_pts) == 0:
        return pd.concat([cpcb, openaq], ignore_index=True)

    def _xy(arr: np.ndarray) -> np.ndarray:
        lat0 = np.deg2rad(arr[:, 0])
        x = np.deg2rad(arr[:, 1]) * np.cos(lat0) * 6371.0
        y = np.deg2rad(arr[:, 0]) * 6371.0
        return np.column_stack([x, y])

    tree = cKDTree(_xy(cpcb_pts))
    oaq_sites = openaq.drop_duplicates("station_id_raw")
    oaq_pts = oaq_sites[["lat", "lon"]].to_numpy()
    dist, _ = tree.query(_xy(oaq_pts), k=1)
    mirror_ids = set(oaq_sites.loc[dist <= radius_km, "station_id_raw"])
    keep = openaq[~openaq["station_id_raw"].isin(mirror_ids)]
    logger.info(
        "dedup_against_openaq: dropped %d/%d OpenAQ mirror sites (<%.1f km)",
        len(mirror_ids),
        len(oaq_sites),
        radius_km,
    )
    return pd.concat([cpcb, keep], ignore_index=True)


def tag_stations(
    stations: gpd.GeoDataFrame | pd.DataFrame,
    *,
    validation_frac: float = 0.18,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Assign stable ids and ``label``/``validation``/``gapfill`` tags.

    A deterministic, region-stratified split locks ~``validation_frac`` of the
    *unique* stations as the held-out ``validation`` set (never tuned on); the
    rest are ``label`` (training). Rows sourced from a gap-free network (source
    not starting with ``CPCB``/``OpenAQ``) are tagged ``gapfill``. The split is by
    ``station_id`` so every row of a sensor shares its tag — the sensor never
    appears in both train and validation.

    Args:
        stations: Station-day table with ``name, lat, lon`` (and optionally
            ``station_id``, ``region``, ``source``).
        validation_frac: Fraction of stations to hold out for validation.
        seed: RNG seed for the deterministic split.

    Returns:
        A :class:`geopandas.GeoDataFrame` with ``station_id``, ``h3_res7``,
        ``source`` and ``tag`` columns ensured.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    df = pd.DataFrame(stations).copy()
    if "source" not in df:
        df["source"] = "CPCB"
    if "station_id" not in df or df["station_id"].isna().any():
        df["station_id"] = [
            make_station_id(n, la, lo)
            for n, la, lo in zip(
                df.get("name", df.get("station_id_raw", df.index)),
                df["lat"],
                df["lon"],
                strict=False,
            )
        ]
    if "h3_res7" not in df:
        df["h3_res7"] = cells_for_arrays(
            df["lat"].to_numpy(), df["lon"].to_numpy(), 7
        ).astype(object)

    # Deterministic region-stratified validation hold-out by station.
    rng = np.random.default_rng(seed)
    df["tag"] = "label"
    sites = df.drop_duplicates("station_id").copy()
    region_col = "region" if "region" in sites else None
    val_ids: set[str] = set()
    groups = sites.groupby(region_col) if region_col else [(None, sites)]
    for _, grp in groups:
        ids = grp["station_id"].to_numpy()
        k = int(round(validation_frac * len(ids)))
        if k > 0 and len(ids) > 0:
            val_ids.update(rng.choice(ids, size=min(k, len(ids)), replace=False))
    df.loc[df["station_id"].isin(val_ids), "tag"] = "validation"
    # Gap-free-prior sources serve only as gapfill (never as scored labels).
    gapfill_mask = ~df["source"].astype(str).str.startswith(("CPCB", "OpenAQ"))
    df.loc[gapfill_mask, "tag"] = "gapfill"

    geom = [Point(xy) for xy in zip(df["lon"], df["lat"], strict=False)]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    n_val = df.loc[df["tag"] == "validation", "station_id"].nunique()
    n_lab = df.loc[df["tag"] == "label", "station_id"].nunique()
    logger.info(
        "tag_stations: %d label + %d validation stations (frac=%.2f)",
        n_lab,
        n_val,
        validation_frac,
    )
    return gdf


def _attach_source(df: pd.DataFrame | gpd.GeoDataFrame, source: str):
    """Return a copy of ``df`` with a ``source`` column set (if absent)."""
    out = df.copy()
    if "source" not in out:
        out["source"] = source
    if "station_id_raw" not in out and "station_id" in out:
        out["station_id_raw"] = out["station_id"]
    return out


def _finalise_station_db(
    daily: pd.DataFrame, start: DateLike, end: DateLike
) -> pd.DataFrame:
    """Clip the daily table to the window and add stable id + region/h3."""
    df = daily.copy()
    if "time" in df:
        t0 = pd.Timestamp(start).normalize()
        t1 = pd.Timestamp(end).normalize()
        df = df[(df["time"] >= t0) & (df["time"] < t1)]
    df["station_id"] = [
        make_station_id(n, la, lo)
        for n, la, lo in zip(
            df.get("name", df["station_id_raw"]), df["lat"], df["lon"], strict=False
        )
    ]
    df["h3_res7"] = cells_for_arrays(
        df["lat"].to_numpy(), df["lon"].to_numpy(), 7
    ).astype(object)
    if "region" not in df:
        df["region"] = [
            _region_for(la, lo) for la, lo in zip(df["lat"], df["lon"], strict=False)
        ]
    return df


def _region_for(lat: float, lon: float) -> str:
    """Coarse region label (mirrors the synthetic generator's stratification)."""
    if 24.0 <= lat <= 31.0 and 73.0 <= lon <= 88.0:
        return "IGP"
    if lat <= 15.0:
        return "Peninsula"
    if lon <= 72.5 or lon >= 90.0:
        return "Coastal"
    return "Central"


__all__ = [
    "OGD_RESOURCE_ID",
    "OGD_TEMPLATE",
    "API_KEY_ENV",
    "CPCB_POLLUTANTS",
    "MIN_HOURLY_COMPLETENESS",
    "DEDUP_RADIUS_KM",
    "STATION_TAGS",
    "fetch_cpcb",
    "aggregate_hourly_to_daily",
    "make_station_id",
    "dedup_against_openaq",
    "tag_stations",
]
