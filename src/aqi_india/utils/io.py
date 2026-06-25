"""Lightweight I/O helpers for the project's standard artifact formats.

Thin, dependency-lazy wrappers around xarray (NetCDF), pandas/pyarrow (Parquet)
and geopandas (GeoParquet / GeoJSON), plus a small Hydra/OmegaConf-free YAML
loader. Directory-creating ``save_*`` helpers keep call sites terse and ensure
parent folders exist.

These helpers are intentionally minimal: they centralise format choices (engine,
compression) so every pipeline stage reads and writes artifacts the same way and
the data contract stays stable across module agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd
    import pandas as pd
    import xarray as xr

PathLike = str | Path


def ensure_dir(path: PathLike) -> Path:
    """Create the parent directory of ``path`` if needed; return ``Path(path)``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# NetCDF (xarray)                                                             #
# --------------------------------------------------------------------------- #
def load_netcdf(path: PathLike, **kwargs: Any) -> xr.Dataset:
    """Open a NetCDF file as an xarray Dataset (netCDF4 engine by default)."""
    import xarray as xr

    kwargs.setdefault("engine", "netcdf4")
    return xr.open_dataset(path, **kwargs)


def save_netcdf(ds: xr.Dataset, path: PathLike, **kwargs: Any) -> Path:
    """Write an xarray Dataset to NetCDF with zlib compression on all vars."""
    p = ensure_dir(path)
    encoding = kwargs.pop("encoding", None)
    if encoding is None:
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(p, engine=kwargs.pop("engine", "netcdf4"), encoding=encoding, **kwargs)
    return p


# --------------------------------------------------------------------------- #
# Parquet (pandas) / GeoParquet (geopandas)                                   #
# --------------------------------------------------------------------------- #
def load_parquet(path: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame (pyarrow engine)."""
    import pandas as pd

    return pd.read_parquet(path, engine=kwargs.pop("engine", "pyarrow"), **kwargs)


def save_parquet(df: pd.DataFrame, path: PathLike, **kwargs: Any) -> Path:
    """Write a pandas DataFrame to Parquet (pyarrow, snappy)."""
    p = ensure_dir(path)
    df.to_parquet(
        p,
        engine=kwargs.pop("engine", "pyarrow"),
        compression=kwargs.pop("compression", "snappy"),
        index=kwargs.pop("index", False),
        **kwargs,
    )
    return p


def load_geoparquet(path: PathLike, **kwargs: Any) -> gpd.GeoDataFrame:
    """Read a GeoParquet file into a GeoDataFrame."""
    import geopandas as gpd

    return gpd.read_parquet(path, **kwargs)


def save_geoparquet(gdf: gpd.GeoDataFrame, path: PathLike, **kwargs: Any) -> Path:
    """Write a GeoDataFrame to GeoParquet."""
    p = ensure_dir(path)
    gdf.to_parquet(p, **kwargs)
    return p


# --------------------------------------------------------------------------- #
# GeoJSON (geopandas)                                                         #
# --------------------------------------------------------------------------- #
def load_geojson(path: PathLike, **kwargs: Any) -> gpd.GeoDataFrame:
    """Read a GeoJSON file into a GeoDataFrame."""
    import geopandas as gpd

    return gpd.read_file(path, **kwargs)


def save_geojson(gdf: gpd.GeoDataFrame, path: PathLike, **kwargs: Any) -> Path:
    """Write a GeoDataFrame to GeoJSON (EPSG:4326)."""
    p = ensure_dir(path)
    out = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf
    out.to_file(p, driver="GeoJSON", **kwargs)
    return p


# --------------------------------------------------------------------------- #
# YAML                                                                         #
# --------------------------------------------------------------------------- #
def load_yaml(path: PathLike) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(data: dict[str, Any], path: PathLike) -> Path:
    """Dump a dictionary to a YAML file (block style, keys preserved)."""
    import yaml

    p = ensure_dir(path)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    return p


__all__ = [
    "PathLike",
    "ensure_dir",
    "load_netcdf",
    "save_netcdf",
    "load_parquet",
    "save_parquet",
    "load_geoparquet",
    "save_geoparquet",
    "load_geojson",
    "save_geojson",
    "load_yaml",
    "save_yaml",
]
