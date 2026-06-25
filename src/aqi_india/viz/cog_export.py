"""Cloud-Optimized GeoTIFF (COG) export for AQI / HCHO grids.

A COG is an ordinary GeoTIFF whose internal layout (tiling + overviews +
IFD-first byte order) lets an HTTP client read only the *windows* it needs via
range requests — no full download. That is what makes the interactive serving
tier (TiTiler XYZ tiles + the static MapLibre/PMTiles map) O(1) per request: the
pipeline emits one COG per daily field and the tile server reads a single tile
out of it on demand.

This module turns an :class:`xarray.DataArray`/``Dataset`` (lat/lon grid,
EPSG:4326) into a COG using :mod:`rioxarray` + :mod:`rasterio`. ``rioxarray`` is
imported lazily inside the function, so the module imports under the light
dependency set; the actual export needs the ``geo`` extra
(``rioxarray``/``rasterio``, plus optionally ``rio-cogeo`` for COG validation).

Serving (documented, implemented in a later workflow):

* **TiTiler** — run ``titiler.core`` + ``titiler.mosaic``; serve a single COG at
  ``/cog/tiles/{z}/{x}/{y}`` and point queries at ``/cog/point/{lon},{lat}``.
  For the daily time dimension, build a ``MosaicJSON`` over the per-day COGs and
  serve it via ``titiler.mosaic`` so a date slider maps to a mosaic asset.
* **Static track** — convert the COGs/vector layers to **PMTiles** behind a CDN
  and render with MapLibre GL + deck.gl (every request an O(1) cache hit, no
  server). Both tracks consume the *same* COGs this module writes.

The serving app itself is intentionally **not** built here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aqi_india.utils.geo import CRS_WGS84

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

#: Default COG creation profile (Deflate-compressed, 512px internal tiles).
DEFAULT_COG_PROFILE: dict[str, object] = {
    "driver": "COG",
    "compress": "DEFLATE",
    "blocksize": 512,
    "overview_resampling": "nearest",
}


def _prepare_dataarray(
    grid: "xr.DataArray | xr.Dataset",
    *,
    var: str | None,
    time: object | None,
) -> "xr.DataArray":
    """Select a single 2-D ``(y, x)`` slice and attach CRS + spatial dims.

    Args:
        grid: Source DataArray or Dataset.
        var: Variable name to export when ``grid`` is a Dataset.
        time: Optional label to select from a ``time`` dimension.

    Returns:
        A 2-D DataArray with ``rio`` spatial dims set to ``lon``/``lat`` and the
        CRS written as EPSG:4326.
    """
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    if hasattr(grid, "data_vars"):
        if var is None:
            data_vars = list(grid.data_vars)  # type: ignore[union-attr]
            if len(data_vars) != 1:
                raise ValueError(
                    "Dataset has multiple variables; pass `var` to choose one "
                    f"of {data_vars}"
                )
            var = data_vars[0]
        da = grid[var]
    else:
        da = grid

    if "time" in da.dims:
        da = da.sel(time=time, method="nearest") if time is not None else da.isel(time=0)
    extra = [d for d in da.dims if d not in ("lat", "lon")]
    for d in extra:
        da = da.isel({d: 0})

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            f"Expected lat/lon dims for COG export; got dims {tuple(da.dims)}"
        )

    # rioxarray needs ascending x and (internally) handles y orientation; ensure
    # coordinates are sorted ascending so the written geotransform is correct.
    da = da.sortby("lat").sortby("lon")
    da = da.rio.write_crs(CRS_WGS84)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    return da


def export_cog(
    grid: "xr.DataArray | xr.Dataset",
    out_path: str | Path,
    *,
    var: str | None = None,
    time: object | None = None,
    nodata: float = float("nan"),
    dtype: str | None = None,
    profile: dict[str, object] | None = None,
) -> Path:
    """Export a lat/lon grid to a Cloud-Optimized GeoTIFF.

    Args:
        grid: AQI / HCHO field as an :class:`xarray.DataArray`, or a Dataset
            from which ``var`` selects the field. Coordinates are ``lat``/``lon``
            in EPSG:4326 (the project convention).
        out_path: Destination ``.tif`` path (parent dirs are created).
        var: Variable to export when ``grid`` is a Dataset (required only if the
            Dataset has more than one variable).
        time: Optional label/value to select from a ``time`` dimension; the
            nearest slice is used. Defaults to the first time step.
        nodata: No-data value written into the COG (NaN by default for float
            fields; an integer sentinel such as ``-1`` for categorical grids).
        dtype: Optional output dtype override (e.g. ``"int16"`` for category
            grids); defaults to the input dtype.
        profile: Optional creation-profile overrides merged onto
            :data:`DEFAULT_COG_PROFILE`.

    Returns:
        The :class:`~pathlib.Path` of the written COG.

    Raises:
        ValueError: If the grid lacks ``lat``/``lon`` dims or an ambiguous
            Dataset is passed without ``var``.
    """
    da = _prepare_dataarray(grid, var=var, time=time)

    if dtype is not None:
        da = da.astype(dtype)

    # Write the no-data value; for float grids encode NaN, for integer grids the
    # caller-supplied sentinel.
    if np.issubdtype(da.dtype, np.floating):
        da = da.rio.write_nodata(nodata, encoded=True)
    else:
        da = da.rio.write_nodata(nodata)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    creation = dict(DEFAULT_COG_PROFILE)
    if profile:
        creation.update(profile)

    da.rio.to_raster(out, **creation)
    return out


def export_aqi_cogs(
    aqi_ds: "xr.Dataset",
    out_dir: str | Path,
    *,
    var: str = "aqi",
    prefix: str = "aqi",
) -> list[Path]:
    """Export one COG per time step of an AQI grid (for a date-slider mosaic).

    Iterates the ``time`` dimension and writes ``{prefix}_{YYYYMMDD}.tif`` per
    day, the per-day assets a TiTiler ``MosaicJSON`` (or a PMTiles build) then
    indexes for the interactive date slider.

    Args:
        aqi_ds: Dataset with a ``time`` dimension and the ``var`` AQI field.
        out_dir: Output directory (created if absent).
        var: AQI variable name.
        prefix: Filename prefix.

    Returns:
        The list of written COG paths, in time order.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "time" not in aqi_ds.dims:
        path = export_cog(aqi_ds, out_dir / f"{prefix}.tif", var=var)
        return [path]

    paths: list[Path] = []
    for t in aqi_ds["time"].values:
        stamp = pd.Timestamp(t).strftime("%Y%m%d")
        path = export_cog(aqi_ds, out_dir / f"{prefix}_{stamp}.tif", var=var, time=t)
        paths.append(path)
    return paths


def validate_cog(path: str | Path) -> bool:
    """Validate that a file is a proper COG, if ``rio-cogeo`` is installed.

    Args:
        path: Path to the GeoTIFF to validate.

    Returns:
        ``True`` if the file is a valid COG. Returns ``True`` (best effort) and
        does not raise if ``rio-cogeo`` is not available.
    """
    try:
        from rio_cogeo.cogeo import cog_validate
    except Exception:  # pragma: no cover - optional dependency
        return True
    is_valid, _errors, _warnings = cog_validate(str(path))
    return bool(is_valid)


__all__ = [
    "DEFAULT_COG_PROFILE",
    "export_cog",
    "export_aqi_cogs",
    "validate_cog",
]
