"""Uber H3 indexing — the O(1) universal fusion join key.

Every satellite pixel, CPCB station and fire detection is assigned a single H3
cell (default resolution 7, ~5 km edge) so that all spatial joins between
heterogeneous sources collapse from O(n*m) geometric overlays into O(n) hash
group-bys. ``cell_to_parent`` enables Indo-Gangetic-Plain rollups; ``grid_disk``
gives hotspot neighbourhoods.

The module transparently supports both the h3-py v4 API (``latlng_to_cell``,
``grid_disk``, ...) and the legacy v3 API (``geo_to_h3``, ``k_ring``, ...). h3 is
imported at module scope because it is part of the light dependency set, but the
v3/v4 capability detection is done once and cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h3
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import xarray as xr

#: Default H3 resolution for the fusion grid (~5.16 km average edge length).
DEFAULT_RES: int = 7

#: True if the installed h3 exposes the v4 API, else the v3 API is used.
_H3_V4: bool = hasattr(h3, "latlng_to_cell")


def h3_api_version() -> int:
    """Return the major version of the installed h3 API (3 or 4)."""
    return 4 if _H3_V4 else 3


def latlng_to_cell(lat: float, lon: float, res: int = DEFAULT_RES) -> str:
    """Index a single (lat, lon) point to an H3 cell (O(1)).

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        res: H3 resolution (0..15).

    Returns:
        The H3 cell index as a hex string.
    """
    if _H3_V4:
        return h3.latlng_to_cell(lat, lon, res)
    return h3.geo_to_h3(lat, lon, res)  # type: ignore[attr-defined]


def cell_to_latlng(cell: str) -> tuple[float, float]:
    """Return the (lat, lon) centroid of an H3 cell."""
    if _H3_V4:
        return h3.cell_to_latlng(cell)
    return h3.h3_to_geo(cell)  # type: ignore[attr-defined]


def cell_to_parent(cell: str, res: int) -> str:
    """Return the parent H3 cell of ``cell`` at coarser resolution ``res``."""
    if _H3_V4:
        return h3.cell_to_parent(cell, res)
    return h3.h3_to_parent(cell, res)  # type: ignore[attr-defined]


def grid_disk(cell: str, k: int = 1) -> list[str]:
    """Return all H3 cells within grid distance ``k`` of ``cell`` (inclusive)."""
    if _H3_V4:
        return list(h3.grid_disk(cell, k))
    return list(h3.k_ring(cell, k))  # type: ignore[attr-defined]


def grid_disk_size(k: int) -> int:
    """Return the number of cells in a grid disk of radius ``k`` (= 3k(k+1)+1)."""
    return 3 * k * (k + 1) + 1


def get_resolution(cell: str) -> int:
    """Return the H3 resolution of a cell index."""
    if _H3_V4:
        return h3.get_resolution(cell)
    return h3.h3_get_resolution(cell)  # type: ignore[attr-defined]


def cells_for_points(
    df: pd.DataFrame,
    lat: str = "lat",
    lon: str = "lon",
    res: int = DEFAULT_RES,
    *,
    out_col: str = "h3",
) -> pd.DataFrame:
    """Vectorized point -> H3 cell assignment for a DataFrame.

    Args:
        df: DataFrame containing latitude/longitude columns.
        lat: Name of the latitude column.
        lon: Name of the longitude column.
        res: H3 resolution.
        out_col: Name of the output cell-index column (e.g. ``h3``,
            ``h3_res7``).

    Returns:
        A copy of ``df`` with an added ``out_col`` of H3 cell hex strings. Rows
        with non-finite coordinates receive ``None``.
    """
    import pandas as pd

    result = df.copy()
    lats = df[lat].to_numpy(dtype=np.float64)
    lons = df[lon].to_numpy(dtype=np.float64)
    cells: list[str | None] = []
    for la, lo in zip(lats, lons, strict=False):
        if not (np.isfinite(la) and np.isfinite(lo)):
            cells.append(None)
        else:
            cells.append(latlng_to_cell(float(la), float(lo), res))
    # Force object dtype so None is preserved (not coerced to NaN).
    result[out_col] = pd.Series(cells, index=df.index, dtype=object)
    return result


def cells_for_arrays(
    lats: np.ndarray, lons: np.ndarray, res: int = DEFAULT_RES
) -> np.ndarray:
    """Map flat latitude/longitude arrays to an object array of H3 cells.

    Args:
        lats: 1-D array of latitudes.
        lons: 1-D array of longitudes (same length as ``lats``).
        res: H3 resolution.

    Returns:
        Object ``np.ndarray`` of H3 cell strings (``None`` for non-finite
        coordinates), same shape as the inputs.
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")
    out = np.empty(lats.shape, dtype=object)
    for i in range(lats.size):
        la, lo = lats[i], lons[i]
        out[i] = (
            latlng_to_cell(float(la), float(lo), res)
            if np.isfinite(la) and np.isfinite(lo)
            else None
        )
    return out


def parents_for_cells(cells: Any, res: int) -> np.ndarray:
    """Roll up an iterable of H3 cells to coarser-resolution parents.

    Args:
        cells: Iterable of H3 cell strings (``None`` allowed).
        res: Target (coarser) resolution.

    Returns:
        Object ``np.ndarray`` of parent cell strings (``None`` preserved).
    """
    arr = np.asarray(list(cells), dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for i, c in enumerate(arr):
        out[i] = cell_to_parent(c, res) if c is not None else None
    return out


def neighborhood(cell: str, k: int = 1) -> list[str]:
    """Alias for :func:`grid_disk` — the k-ring neighbourhood of a cell."""
    return grid_disk(cell, k)


def assign_grid_cells(
    grid: xr.Dataset | xr.DataArray,
    lat: str = "lat",
    lon: str = "lon",
    res: int = DEFAULT_RES,
) -> xr.DataArray:
    """Assign an H3 cell to every (lat, lon) node of a gridded dataset.

    Args:
        grid: An xarray Dataset/DataArray with 1-D ``lat`` and ``lon`` coords.
        lat: Name of the latitude coordinate.
        lon: Name of the longitude coordinate.
        res: H3 resolution.

    Returns:
        A 2-D object ``xr.DataArray`` (dims ``(lat, lon)``) of H3 cell strings.
    """
    import xarray as xr  # local import keeps module import light

    lats = np.asarray(grid[lat].values, dtype=np.float64)
    lons = np.asarray(grid[lon].values, dtype=np.float64)
    cells = np.empty((lats.size, lons.size), dtype=object)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            cells[i, j] = latlng_to_cell(float(la), float(lo), res)
    return xr.DataArray(
        cells,
        dims=(lat, lon),
        coords={lat: grid[lat], lon: grid[lon]},
        name=f"h3_res{res}",
    )


__all__ = [
    "DEFAULT_RES",
    "h3_api_version",
    "latlng_to_cell",
    "cell_to_latlng",
    "cell_to_parent",
    "grid_disk",
    "grid_disk_size",
    "get_resolution",
    "cells_for_points",
    "cells_for_arrays",
    "parents_for_cells",
    "neighborhood",
    "assign_grid_cells",
]
