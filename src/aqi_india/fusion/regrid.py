"""Regridding of gridded fields onto a common analysis grid.

This module re-projects multi-source satellite / reanalysis fields onto the
project's common lat/lon analysis grid (``latlon_0p05`` at 0.05 deg, or the
0.25 deg synthetic-demo grid). The default path is dependency-light — it uses
:mod:`scipy` interpolation over the regular grids that every project artifact
carries — so it runs on the demo's light dependency set alone. A conservative
(area-weighted) regridder is provided as a lazily-imported :mod:`xesmf` adapter
for the heavy production path; if ``xesmf`` is unavailable it degrades to a
bilinear scipy interpolation with a clear warning.

Conventions (DEV_CONTRACT §3):

* Spatial dims/coords are ``lat`` / ``lon``, 1-D and ascending, CRS EPSG:4326.
* The time dim/coord is ``time``; regridding is applied independently per time
  slice and any extra leading dims are preserved.
* Missing pixels are ``NaN`` (never sentinels); NaNs propagate through the
  interpolation rather than being silently treated as zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("fusion.regrid")

Method = Literal["bilinear", "nearest", "linear", "conservative"]


@dataclass(frozen=True)
class TargetGrid:
    """A regular lat/lon target grid.

    Attributes:
        lat: Ascending 1-D latitude coordinate (degrees north).
        lon: Ascending 1-D longitude coordinate (degrees east).
        name: Optional human-readable grid name (e.g. ``"latlon_0p05"``).
    """

    lat: np.ndarray
    lon: np.ndarray
    name: str = "target"

    @property
    def shape(self) -> tuple[int, int]:
        """Return the ``(n_lat, n_lon)`` shape of the grid."""
        return (int(self.lat.size), int(self.lon.size))


def make_grid_from_bbox(
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
    *,
    name: str = "target",
) -> TargetGrid:
    """Build a :class:`TargetGrid` from a bounding box and resolution.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
        resolution_deg: Grid spacing in degrees (e.g. ``0.05``).
        name: Optional grid name carried for provenance.

    Returns:
        A :class:`TargetGrid` with ascending ``lat``/``lon`` cell-edge-aligned
        coordinates spanning ``bbox`` inclusive of the lower edge.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")
    lon = np.arange(min_lon, max_lon + 1e-9, resolution_deg, dtype="float64")
    lat = np.arange(min_lat, max_lat + 1e-9, resolution_deg, dtype="float64")
    return TargetGrid(lat=lat, lon=lon, name=name)


def target_grid_from_cfg(cfg) -> TargetGrid:  # noqa: ANN001 - Hydra config object
    """Construct a :class:`TargetGrid` from a composed Hydra ``cfg``.

    Reads ``cfg.grid`` (``bbox`` + ``resolution_deg``) when present, otherwise
    falls back to the India bbox at 0.25 deg (the synthetic-demo grid).

    Args:
        cfg: Composed Hydra config exposing ``cfg.grid``.

    Returns:
        The analysis :class:`TargetGrid`.
    """
    from ..utils.geo import INDIA_BBOX

    grid_cfg = getattr(cfg, "grid", None)
    bbox = tuple(getattr(grid_cfg, "bbox", INDIA_BBOX)) if grid_cfg else INDIA_BBOX
    res = float(getattr(grid_cfg, "resolution_deg", 0.25)) if grid_cfg else 0.25
    name = str(getattr(grid_cfg, "name", "target")) if grid_cfg else "target"
    return make_grid_from_bbox(bbox, res, name=name)  # type: ignore[arg-type]


def _coerce_target(target_grid) -> TargetGrid:  # noqa: ANN001
    """Coerce various target-grid spellings into a :class:`TargetGrid`.

    Accepts a :class:`TargetGrid`, an ``xarray`` object carrying ``lat``/``lon``
    coords, or a mapping with ``lat``/``lon`` keys.
    """
    if isinstance(target_grid, TargetGrid):
        return target_grid
    # xarray-like (Dataset / DataArray) with lat/lon coords.
    coords = getattr(target_grid, "coords", None)
    if coords is not None and "lat" in coords and "lon" in coords:
        return TargetGrid(
            lat=np.asarray(coords["lat"].values, dtype="float64"),
            lon=np.asarray(coords["lon"].values, dtype="float64"),
            name=str(getattr(target_grid, "name", "target") or "target"),
        )
    # Mapping with lat/lon arrays.
    try:
        return TargetGrid(
            lat=np.asarray(target_grid["lat"], dtype="float64"),
            lon=np.asarray(target_grid["lon"], dtype="float64"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(
            "target_grid must be a TargetGrid, an xarray object with lat/lon "
            "coords, or a mapping with 'lat'/'lon' arrays"
        ) from exc


def _interp_slice(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    values: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    method: str,
) -> np.ndarray:
    """Interpolate one 2-D ``(lat, lon)`` slice onto the target mesh.

    Uses :class:`scipy.interpolate.RegularGridInterpolator` which requires
    strictly ascending source coordinates (guaranteed by the contract). NaNs in
    the source propagate to any target cell whose interpolation stencil touches
    a NaN — this is intentional so cloud gaps are not silently filled here
    (gap-filling is the job of :mod:`aqi_india.fusion.gapfill`).
    """
    from scipy.interpolate import RegularGridInterpolator

    scipy_method = "nearest" if method == "nearest" else "linear"
    interp = RegularGridInterpolator(
        (src_lat, src_lon),
        values,
        method=scipy_method,
        bounds_error=False,
        fill_value=np.nan,
    )
    mesh_lat, mesh_lon = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
    pts = np.stack([mesh_lat.ravel(), mesh_lon.ravel()], axis=-1)
    out = interp(pts).reshape(mesh_lat.shape)
    return out.astype("float32")


def to_grid(
    ds,  # noqa: ANN001 - xarray Dataset | DataArray
    target_grid,  # noqa: ANN001
    *,
    method: Method = "bilinear",
    variables: Iterable[str] | None = None,
):
    """Regrid an xarray object onto ``target_grid``.

    The default ``"bilinear"`` / ``"nearest"`` / ``"linear"`` paths use
    :mod:`scipy` and run on the light dependency set. ``"conservative"`` lazily
    imports :mod:`xesmf`; if that is unavailable it falls back to bilinear with
    a warning (so the call still succeeds in the demo environment).

    Args:
        ds: Source :class:`xarray.Dataset` or :class:`xarray.DataArray` with
            ascending 1-D ``lat``/``lon`` coords (and optionally ``time``).
        target_grid: Destination grid (a :class:`TargetGrid`, an xarray object
            carrying ``lat``/``lon``, or a ``{"lat": ..., "lon": ...}`` mapping).
        method: ``"bilinear"`` (alias of linear), ``"nearest"``, ``"linear"`` or
            ``"conservative"``.
        variables: Optional subset of data-vars to regrid (Dataset input only).

    Returns:
        The regridded object of the same type as ``ds``, with new ``lat``/``lon``
        coords and all non-spatial dims/coords preserved.
    """
    import xarray as xr

    tgt = _coerce_target(target_grid)

    if method == "conservative":
        result = _to_grid_conservative(ds, tgt, variables=variables)
        if result is not None:
            return result
        logger.warning(
            "xesmf unavailable; falling back to bilinear regridding for "
            "conservative request"
        )
        method = "bilinear"

    is_dataarray = isinstance(ds, xr.DataArray)
    src = ds.to_dataset(name=ds.name or "var") if is_dataarray else ds

    src_lat = np.asarray(src["lat"].values, dtype="float64")
    src_lon = np.asarray(src["lon"].values, dtype="float64")

    names = list(variables) if variables is not None else list(src.data_vars)
    out_vars: dict[str, xr.DataArray] = {}
    for name in names:
        da = src[name]
        if "lat" not in da.dims or "lon" not in da.dims:
            out_vars[name] = da
            continue
        # Move (lat, lon) to the trailing axes so we can vectorize the loop.
        other_dims = [d for d in da.dims if d not in ("lat", "lon")]
        da_t = da.transpose(*other_dims, "lat", "lon")
        arr = np.asarray(da_t.values, dtype="float64")
        lead_shape = arr.shape[:-2]
        flat = arr.reshape((-1, len(src_lat), len(src_lon))) if lead_shape else arr[None]
        out = np.empty((flat.shape[0], tgt.lat.size, tgt.lon.size), dtype="float32")
        for i in range(flat.shape[0]):
            out[i] = _interp_slice(
                src_lat, src_lon, flat[i], tgt.lat, tgt.lon, method
            )
        out = out.reshape((*lead_shape, tgt.lat.size, tgt.lon.size))
        coords = {d: da_t.coords[d] for d in other_dims if d in da_t.coords}
        coords["lat"] = tgt.lat
        coords["lon"] = tgt.lon
        out_vars[name] = xr.DataArray(
            out, dims=(*other_dims, "lat", "lon"), coords=coords, attrs=dict(da.attrs)
        )

    result_ds = xr.Dataset(out_vars, attrs=dict(src.attrs))
    result_ds["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    result_ds["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})
    result_ds.attrs["regrid_method"] = method
    result_ds.attrs["regrid_target"] = tgt.name

    if is_dataarray:
        out_name = ds.name or "var"
        return result_ds[out_name].rename(ds.name) if ds.name else result_ds[out_name]
    return result_ds


def _to_grid_conservative(ds, tgt: TargetGrid, *, variables):  # noqa: ANN001
    """Conservative (area-weighted) regrid via lazily-imported :mod:`xesmf`.

    Returns ``None`` if :mod:`xesmf` cannot be imported so the caller can fall
    back to a scipy method.
    """
    try:
        import xarray as xr
        import xesmf as xe  # type: ignore
    except Exception:
        return None

    is_dataarray = isinstance(ds, xr.DataArray)
    src = ds.to_dataset(name=ds.name or "var") if is_dataarray else ds
    if variables is not None:
        src = src[list(variables)]

    target_ds = xr.Dataset({"lat": ("lat", tgt.lat), "lon": ("lon", tgt.lon)})
    regridder = xe.Regridder(src, target_ds, "conservative", periodic=False)
    out = regridder(src, keep_attrs=True)
    out.attrs["regrid_method"] = "conservative"
    out.attrs["regrid_target"] = tgt.name
    if is_dataarray:
        name = ds.name or "var"
        return out[name]
    return out


__all__ = [
    "TargetGrid",
    "Method",
    "make_grid_from_bbox",
    "target_grid_from_cfg",
    "to_grid",
]
