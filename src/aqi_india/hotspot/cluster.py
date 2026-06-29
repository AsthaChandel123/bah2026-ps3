"""Delineate hotspot polygons from a significance mask via density clustering.

Once the consensus step has flagged individual significant cells, they must be
grouped into contiguous **hotspot polygons** for mapping and transport overlay.
This module:

1.  Clusters the significant cell centroids with **DBSCAN** using the great-circle
    (``haversine``) metric so ``eps`` is a real distance in kilometres. If the
    optional ``hdbscan`` package is installed it is used instead (variable
    density, no single ``eps``), falling back to DBSCAN otherwise (lazy import,
    blueprint method 38).
2.  Wraps each cluster's points in a polygon — a concave **alpha-shape** when the
    optional ``alphashape`` package is present, otherwise the convex hull — and
    returns a :class:`geopandas.GeoDataFrame` in EPSG:4326.
3.  Provides an **ST-DBSCAN** variant (separate spatial and temporal ``eps``)
    that groups significant cells across days into multi-day **plume episodes**
    for the transport analysis (blueprint method 39).

``sklearn``/``geopandas``/``shapely`` are in the light dependency set;
``hdbscan`` and ``alphashape`` are optional and imported lazily.
"""

from __future__ import annotations

import numpy as np

from ..utils.geo import CRS_WGS84
from ..utils.logging import get_logger

logger = get_logger("hotspot.cluster")

#: Mean Earth radius (km) for converting haversine radians to distance.
EARTH_RADIUS_KM: float = 6371.0088


def _haversine_eps(eps_km: float) -> float:
    """Convert a distance in km to the radian ``eps`` used by haversine DBSCAN."""
    return float(eps_km) / EARTH_RADIUS_KM


def cluster_labels(
    coords: np.ndarray,
    *,
    eps_km: float = 25.0,
    min_samples: int = 5,
    use_hdbscan: bool = True,
) -> np.ndarray:
    """Cluster lon/lat points by spatial density.

    Args:
        coords: ``(n, 2)`` array of ``(lon, lat)`` significant-cell centroids.
        eps_km: Neighbourhood radius in km for DBSCAN.
        min_samples: Minimum points to form a dense core.
        use_hdbscan: Try ``hdbscan`` first (variable density). Falls back to
            DBSCAN if it is not installed.

    Returns:
        Integer cluster labels per point; ``-1`` marks noise.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.empty(0, dtype=int)

    # haversine expects radians, ordered (lat, lon).
    rad = np.radians(coords[:, ::-1])

    if use_hdbscan:
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min_samples),
                metric="haversine",
                allow_single_cluster=True,
            )
            labels = clusterer.fit_predict(rad)
            logger.info("clustered %d points with HDBSCAN", coords.shape[0])
            return labels.astype(int)
        except ImportError:
            logger.info("hdbscan unavailable; falling back to DBSCAN")

    from sklearn.cluster import DBSCAN

    db = DBSCAN(
        eps=_haversine_eps(eps_km),
        min_samples=min_samples,
        metric="haversine",
    )
    labels = db.fit_predict(rad)
    logger.info(
        "clustered %d points with DBSCAN (eps=%.1f km)", coords.shape[0], eps_km
    )
    return labels.astype(int)


def st_dbscan_labels(
    coords: np.ndarray,
    times: np.ndarray,
    *,
    eps_spatial_km: float = 25.0,
    eps_temporal_days: float = 2.0,
    min_samples: int = 5,
) -> np.ndarray:
    """Cluster cells in space *and* time (ST-DBSCAN) into plume episodes.

    Two points are density-reachable only if they are within ``eps_spatial_km``
    *and* within ``eps_temporal_days``. This is implemented as DBSCAN on a
    combined metric where the temporal distance is rescaled so that the temporal
    threshold maps onto the same radian ``eps`` as the spatial one.

    Args:
        coords: ``(n, 2)`` array of ``(lon, lat)`` centroids.
        times: ``(n,)`` array of times — ``datetime64``, or numeric day indices.
        eps_spatial_km: Spatial neighbourhood radius in km.
        eps_temporal_days: Temporal neighbourhood radius in days.
        min_samples: Minimum points to form a dense core.

    Returns:
        Integer episode labels per point; ``-1`` marks noise.
    """
    from sklearn.cluster import DBSCAN

    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)

    times = np.asarray(times)
    if np.issubdtype(times.dtype, np.datetime64):
        day = times.astype("datetime64[D]").astype("int64").astype(np.float64)
    else:
        day = times.astype(np.float64)

    eps_rad = _haversine_eps(eps_spatial_km)
    lat = np.radians(coords[:, 1])
    lon = np.radians(coords[:, 0])

    # Equirectangular projection (radians) — locally accurate for hotspot scales.
    lat0 = np.mean(lat)
    x = lon * np.cos(lat0)
    y = lat
    # Scale the temporal axis so eps_temporal_days maps onto eps_rad.
    t_scaled = (day - day.min()) * (eps_rad / max(eps_temporal_days, 1e-9))
    feats = np.column_stack([x, y, t_scaled])

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(feats)
    logger.info(
        "ST-DBSCAN: %d points -> %d episodes (eps_s=%.1f km, eps_t=%.1f d)",
        n,
        len(set(labels) - {-1}),
        eps_spatial_km,
        eps_temporal_days,
    )
    return labels.astype(int)


def _cluster_polygon(points: np.ndarray, *, alpha: float | None):
    """Build a polygon for one cluster's (lon, lat) points.

    Uses a concave alpha-shape when ``alphashape`` is available, otherwise the
    convex hull. Degenerate clusters (1-2 points) are buffered to a small disc.

    Args:
        points: ``(m, 2)`` array of ``(lon, lat)``.
        alpha: Alpha parameter; ``None`` lets ``alphashape`` auto-optimise.

    Returns:
        A shapely geometry (Polygon/MultiPolygon).
    """
    from shapely.geometry import MultiPoint

    mp = MultiPoint([tuple(p) for p in points])
    if points.shape[0] < 3:
        # ~5 km buffer so single/paired cells still render as an area.
        return mp.buffer(0.05)

    try:
        import alphashape

        a = alphashape.optimizealpha(points) if alpha is None else alpha
        geom = alphashape.alphashape(points, a)
        if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
            return mp.convex_hull
        return geom
    except ImportError:
        return mp.convex_hull


def hotspot_polygons(
    coords: np.ndarray,
    *,
    values: np.ndarray | None = None,
    eps_km: float = 25.0,
    min_samples: int = 5,
    use_hdbscan: bool = True,
    alpha: float | None = None,
):
    """Delineate hotspot polygons from significant-cell centroids.

    Args:
        coords: ``(n, 2)`` array of ``(lon, lat)`` significant-cell centroids.
        values: Optional per-cell intensity (e.g. mean HCHO z-anomaly); when
            given the polygon's mean/max intensity are reported.
        eps_km: DBSCAN neighbourhood radius in km.
        min_samples: Minimum points per cluster.
        use_hdbscan: Prefer ``hdbscan`` if installed.
        alpha: Alpha-shape parameter (``None`` = auto / convex-hull fallback).

    Returns:
        A :class:`geopandas.GeoDataFrame` (EPSG:4326) with one row per cluster:
        ``cluster_id``, ``n_cells``, ``geometry`` and, if ``values`` given,
        ``mean_z`` / ``max_z``. Empty input yields an empty frame.
    """
    import geopandas as gpd

    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return gpd.GeoDataFrame(
            {"cluster_id": [], "n_cells": []}, geometry=[], crs=CRS_WGS84
        )

    labels = cluster_labels(
        coords, eps_km=eps_km, min_samples=min_samples, use_hdbscan=use_hdbscan
    )
    vals = None if values is None else np.asarray(values, dtype=np.float64)

    records = []
    geoms = []
    for cid in sorted(set(labels) - {-1}):
        sel = labels == cid
        pts = coords[sel]
        geoms.append(_cluster_polygon(pts, alpha=alpha))
        rec = {"cluster_id": int(cid), "n_cells": int(sel.sum())}
        if vals is not None:
            cvals = vals[sel]
            rec["mean_z"] = float(np.nanmean(cvals))
            rec["max_z"] = float(np.nanmax(cvals))
        records.append(rec)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=CRS_WGS84)
    logger.info("delineated %d hotspot polygons", len(gdf))
    return gdf


__all__ = [
    "EARTH_RADIUS_KM",
    "cluster_labels",
    "st_dbscan_labels",
    "hotspot_polygons",
]
