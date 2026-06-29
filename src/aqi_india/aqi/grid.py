"""Gridded NAQI utilities — apply the CPCB engine over an xarray cube.

This module is the thin, dask-friendly bridge between the pure NAQI engine in
:mod:`aqi_india.aqi.naqi` and the gridded products the visualisation and serving
layers consume. It does **not** reimplement any NAQI maths — the validity rule
(``>= 3`` sub-indices and ``>= 1`` of PM2.5/PM10), the piecewise-linear
sub-index and the max-of-sub-index aggregation all live in the engine and are
called here. What this module adds is:

* :func:`apply_naqi_grid` — run the engine across an :class:`xarray.Dataset` of
  *surface* pollutant fields and return the AQI grid, the responsible-pollutant
  code grid **and** a derived integer category grid (0..5), as a single
  Dataset, preserving coordinates and staying lazy under dask.
* :func:`category_grid` — derive the 0..5 CPCB category band from an AQI grid
  (vectorized, NaN-safe), with a ``category_names`` attribute for legends.
* Colour look-up-table helpers (:func:`category_color_lut`,
  :func:`colorize_aqi_grid`, :func:`aqi_to_rgba`) that turn an AQI / category
  grid into an RGBA image array using the six-band CPCB colour scale from
  :mod:`aqi_india.aqi.breakpoints`. These feed both the matplotlib figures and
  the COG/web exporters.

The canonical pollutant data-var names are the keys of
:data:`aqi_india.aqi.breakpoints.POLLUTANTS`; pass a ``var_map`` to override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aqi_india.aqi.breakpoints import (
    AQI_MAX,
    CATEGORY_COLORS,
    CATEGORY_NAMES,
)
from aqi_india.aqi.naqi import compute_aqi_grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

#: Internal AQI-band upper edges used to assign category indices (0..5).
#: A cell with AQI <= 50 -> band 0 (Good); (400, 500] -> band 5 (Severe).
_CATEGORY_UPPER_EDGES: np.ndarray = np.array(
    [50.0, 100.0, 200.0, 300.0, 400.0], dtype=np.float64
)


# --------------------------------------------------------------------------- #
# Vectorized category assignment
# --------------------------------------------------------------------------- #
def _category_index_array(aqi: np.ndarray) -> np.ndarray:
    """Map an AQI array to CPCB category indices 0..5 (NaN -> ``-1``).

    This is the array analogue of
    :func:`aqi_india.aqi.breakpoints.category_index`; it is branch-free and
    NaN-safe so it can run lazily under :func:`xarray.apply_ufunc`.

    Args:
        aqi: AQI values of any shape. NaN marks invalid cells.

    Returns:
        ``int16`` array of band indices in ``[0, 5]`` with ``-1`` where the
        input is NaN.
    """
    a = np.asarray(aqi, dtype=np.float64)
    finite = np.isfinite(a)
    clamped = np.clip(np.where(finite, a, 0.0), 0.0, AQI_MAX)
    idx = np.searchsorted(_CATEGORY_UPPER_EDGES, clamped, side="left")
    idx = np.clip(idx, 0, len(CATEGORY_NAMES) - 1).astype(np.int16)
    return np.where(finite, idx, np.int16(-1)).astype(np.int16)


def category_grid(aqi: "xr.DataArray", *, name: str = "aqi_category") -> "xr.DataArray":
    """Derive the CPCB category-band grid (0..5) from an AQI grid.

    Args:
        aqi: AQI :class:`~xarray.DataArray` (NaN where invalid). May be dask-
            backed; the operation is applied lazily.
        name: Name of the returned DataArray.

    Returns:
        An ``int16`` :class:`~xarray.DataArray` of category indices in
        ``[0, 5]`` with ``-1`` where the AQI is NaN. The
        ``attrs["category_names"]`` / ``attrs["category_colors"]`` carry the
        legend mapping and ``_FillValue`` is ``-1``.
    """
    import xarray as xr

    cats = xr.apply_ufunc(
        _category_index_array,
        aqi,
        dask="parallelized",
        output_dtypes=[np.int16],
    ).rename(name)
    cats.attrs.update(
        {
            "long_name": "CPCB AQI category index",
            "category_names": ",".join(f"{i}:{n}" for i, n in enumerate(CATEGORY_NAMES)),
            "category_colors": ",".join(CATEGORY_COLORS),
            "_FillValue": -1,
            "valid_range": "0 5",
        }
    )
    return cats


# --------------------------------------------------------------------------- #
# Top-level grid driver
# --------------------------------------------------------------------------- #
def apply_naqi_grid(
    ds: "xr.Dataset",
    *,
    var_map: dict[str, str] | None = None,
    aqi_name: str = "aqi",
    responsible_name: str = "aqi_responsible",
    category_name: str = "aqi_category",
) -> "xr.Dataset":
    """Apply the CPCB NAQI engine across a gridded pollutant cube.

    Thin orchestration wrapper around
    :func:`aqi_india.aqi.naqi.compute_aqi_grid` that additionally attaches the
    derived integer category-band grid. The result is a single Dataset carrying
    the AQI field, the responsible-pollutant code field and the category-index
    field, all sharing the input coordinates/dims and remaining lazy when the
    input is dask-backed.

    Args:
        ds: Dataset of **surface** pollutant concentration fields named with the
            canonical pollutant keys (any subset; CO interpreted in mg/m3). Dims
            are typically ``(time, lat, lon)`` but any shape is accepted.
        var_map: Optional ``{pollutant_key: dataset_variable}`` override for
            non-canonical variable names.
        aqi_name: Output AQI variable name.
        responsible_name: Output responsible-pollutant-code variable name.
        category_name: Output category-index variable name.

    Returns:
        A Dataset with ``float32`` ``{aqi_name}``, ``int16``
        ``{responsible_name}`` (``-1`` = invalid, code->name mapping in its
        ``pollutant_codes`` attribute) and ``int16`` ``{category_name}``
        (``-1`` = invalid, names/colours in its attributes).
    """
    out = compute_aqi_grid(
        ds,
        var_map=var_map,
        aqi_name=aqi_name,
        responsible_name=responsible_name,
    )
    out[category_name] = category_grid(out[aqi_name], name=category_name)
    return out


# --------------------------------------------------------------------------- #
# Colour look-up table helpers
# --------------------------------------------------------------------------- #
def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert a ``#rrggbb`` hex string to a 0-1 RGB float triple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def category_color_lut(*, with_alpha: bool = True) -> np.ndarray:
    """Return the six-band CPCB colour LUT as a float array.

    Args:
        with_alpha: If True, append an opaque alpha channel (shape ``(6, 4)``);
            otherwise return ``(6, 3)`` RGB.

    Returns:
        ``float64`` array in ``[0, 1]`` ordered Good..Severe.
    """
    rgb = np.array([_hex_to_rgb(c) for c in CATEGORY_COLORS], dtype=np.float64)
    if not with_alpha:
        return rgb
    alpha = np.ones((rgb.shape[0], 1), dtype=np.float64)
    return np.concatenate([rgb, alpha], axis=1)


def aqi_to_rgba(
    aqi: np.ndarray,
    *,
    nodata_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Colourise an AQI array into an RGBA image using the CPCB band LUT.

    Each cell is binned into its CPCB category and assigned that category's
    colour (discrete, not interpolated — the official AQI scale is categorical).
    Invalid (NaN) cells receive ``nodata_rgba`` (transparent by default).

    Args:
        aqi: AQI array of shape ``(*grid,)``. NaN marks invalid cells.
        nodata_rgba: RGBA tuple (0-1) for invalid cells.

    Returns:
        ``float32`` RGBA array of shape ``(*grid, 4)`` with values in ``[0, 1]``.
    """
    idx = _category_index_array(aqi)  # (*grid,) int16, -1 where NaN
    lut = category_color_lut(with_alpha=True).astype(np.float32)  # (6, 4)
    safe = np.where(idx >= 0, idx, 0)
    rgba = lut[safe]
    rgba[idx < 0] = np.asarray(nodata_rgba, dtype=np.float32)
    return rgba.astype(np.float32)


def colorize_aqi_grid(
    aqi: "xr.DataArray",
    *,
    nodata_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Colourise an AQI :class:`~xarray.DataArray` into an RGBA image array.

    Convenience wrapper around :func:`aqi_to_rgba` that first materialises the
    DataArray (loading any dask chunks). Intended for a single 2-D ``(lat, lon)``
    slice; a leading ``time`` dimension is allowed and produces a stacked
    ``(time, lat, lon, 4)`` array.

    Args:
        aqi: AQI DataArray (2-D, or 3-D with a leading time axis).
        nodata_rgba: RGBA tuple (0-1) for invalid cells.

    Returns:
        ``float32`` RGBA array with a trailing length-4 axis.
    """
    return aqi_to_rgba(np.asarray(aqi.values), nodata_rgba=nodata_rgba)


__all__ = [
    "apply_naqi_grid",
    "category_grid",
    "category_color_lut",
    "aqi_to_rgba",
    "colorize_aqi_grid",
]
