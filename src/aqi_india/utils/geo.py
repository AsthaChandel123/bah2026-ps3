"""Geospatial constants and helpers for the India domain.

Provides the canonical India and Indo-Gangetic-Plain (IGP) bounding boxes,
named source-region boxes used by Objective-2, and :func:`get_india_boundary`
which returns an India boundary as a GeoDataFrame — preferring the Natural Earth
admin-0 dataset when available and falling back to a bundled simplified GeoJSON
(``aqi_india/data/india_simplified.geojson``) so the package never requires a
network download to run.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

#: India analysis bounding box as (min_lon, min_lat, max_lon, max_lat).
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)

#: Indo-Gangetic Plain bounding box (Punjab through West Bengal).
IGP_BBOX: tuple[float, float, float, float] = (73.0, 24.0, 88.0, 31.0)

#: Punjab + Haryana stubble-burning region (Objective-2 primary source area).
PUNJAB_HARYANA_BBOX: tuple[float, float, float, float] = (73.5, 28.5, 77.5, 32.5)

#: Central-Indian / Himalayan forest-fire belt (pre-monsoon Apr-May fires).
FOREST_BELT_BBOX: tuple[float, float, float, float] = (73.0, 18.0, 96.0, 31.0)

#: WGS84 / EPSG:4326.
CRS_WGS84: str = "EPSG:4326"

#: UTM zones spanning India (43N west, 44N east) for metric operations.
UTM_43N: str = "EPSG:32643"
UTM_44N: str = "EPSG:32644"

#: Path (within the package) to the bundled simplified India boundary.
_SIMPLIFIED_GEOJSON = "india_simplified.geojson"


def bbox_to_polygon(bbox: tuple[float, float, float, float]):
    """Return a shapely box polygon for a (min_lon, min_lat, max_lon, max_lat)."""
    from shapely.geometry import box

    return box(*bbox)


def get_india_boundary(simplify: bool = False) -> gpd.GeoDataFrame:
    """Return the India national boundary as a GeoDataFrame (EPSG:4326).

    The function first tries the Natural Earth admin-0 dataset (via geopandas'
    bundled sample data, if present in the installed geopandas version). If that
    is unavailable, it falls back to the bundled simplified GeoJSON so the call
    always succeeds offline.

    Args:
        simplify: If True, apply a light topology-preserving simplification
            (tolerance ~0.05 deg) to the returned geometry.

    Returns:
        A single-row :class:`geopandas.GeoDataFrame` with the India polygon.
    """
    import geopandas as gpd

    gdf: gpd.GeoDataFrame | None = None
    try:  # geopandas <1.0 ships naturalearth_lowres; newer versions may not.
        path = gpd.datasets.get_path("naturalearth_lowres")  # type: ignore[attr-defined]
        world = gpd.read_file(path)
        name_col = "name" if "name" in world.columns else "NAME"
        india = world[world[name_col].str.lower() == "india"]
        if len(india):
            gdf = india[[india.geometry.name]].reset_index(drop=True)
    except Exception:
        gdf = None

    if gdf is None:
        gdf = _load_bundled_boundary()

    gdf = gdf.set_crs(CRS_WGS84, allow_override=True)
    if simplify:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(0.05, preserve_topology=True)
    return gdf


def _load_bundled_boundary() -> gpd.GeoDataFrame:
    """Load the bundled simplified India boundary GeoJSON."""
    import geopandas as gpd

    with resources.files("aqi_india.data").joinpath(_SIMPLIFIED_GEOJSON).open(
        "r", encoding="utf-8"
    ) as fh:
        geojson = json.load(fh)
    return gpd.GeoDataFrame.from_features(geojson["features"], crs=CRS_WGS84)


def clip_to_india(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip a GeoDataFrame to the India boundary (uses the spatial index)."""
    import geopandas as gpd

    india = get_india_boundary()
    return gpd.clip(gdf, india)


__all__ = [
    "INDIA_BBOX",
    "IGP_BBOX",
    "PUNJAB_HARYANA_BBOX",
    "FOREST_BELT_BBOX",
    "CRS_WGS84",
    "UTM_43N",
    "UTM_44N",
    "bbox_to_polygon",
    "get_india_boundary",
    "clip_to_india",
]
