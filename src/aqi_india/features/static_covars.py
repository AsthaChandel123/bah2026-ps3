"""Static (time-invariant) covariate layers over the India grid.

Coarse satellite columns and ~50 km reanalysis fields carry little sub-grid
spatial structure. Static, high-resolution covariates inject that structure into
the ML and downscaling layers, letting the model resolve urban/rural and
terrain-driven gradients the dynamic predictors cannot:

* **Terrain** — elevation (DEM), slope, Topographic Position Index (TPI):
  controls cold-air pooling, valley inversions, orographic effects.
* **Land cover** — fractional urban/cropland/forest: emission-source context.
* **NDVI** — vegetation greenness: biogenic VOC, dust suppression proxy.
* **Nightlights** (VIIRS VNP46A2) — anthropogenic activity / combustion proxy.
* **Population** (WorldPop) — emission and exposure proxy.
* **Road density** (OSM/GHSL) — traffic NOx/PM source proxy.

For the offline demo there are no bundled real layers, so this module
**synthesizes physically plausible fields** over the India grid using smooth,
deterministic spatial functions seeded from the grid geometry and the named
source regions (IGP, Punjab/Haryana, forest belt). Synthetic fields are clearly
flagged via ``Dataset.attrs["synthetic"] = True`` and each variable's
``source`` attribute. Real loaders are provided as clearly-marked adapters that
read a raster and resample it onto the grid (lazy ``rioxarray`` import).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.geo import (
    FOREST_BELT_BBOX,
    IGP_BBOX,
    PUNJAB_HARYANA_BBOX,
)
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    import xarray as xr

_log = get_logger("features.static_covars")

#: Canonical static covariate variable names.
STATIC_COVARIATES: tuple[str, ...] = (
    "dem",
    "slope",
    "tpi",
    "landcover_urban",
    "landcover_crop",
    "landcover_forest",
    "ndvi",
    "nightlights",
    "population",
    "road_density",
)

#: Deterministic seed so the synthetic static fields are reproducible.
_SYNTH_SEED: int = 42


def _grid_latlon(grid: "xr.Dataset | xr.DataArray") -> tuple[np.ndarray, np.ndarray]:
    """Return the 1-D ``lat`` and ``lon`` coordinate arrays of a grid."""
    lat = np.asarray(grid["lat"].values, dtype=np.float64)
    lon = np.asarray(grid["lon"].values, dtype=np.float64)
    return lat, lon


def _gaussian_bump(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    amplitude: float = 1.0,
    spread: float = 1.0,
) -> np.ndarray:
    """A smooth 2-D Gaussian centred on a bounding box (used for source regions).

    Args:
        lon2d: 2-D longitude mesh (lat, lon).
        lat2d: 2-D latitude mesh (lat, lon).
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` whose centre is the bump.
        amplitude: Peak value of the bump.
        spread: Multiplies the box half-width to set the Gaussian sigma.

    Returns:
        A 2-D field with a soft peak over the box, decaying outward.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    c_lon = 0.5 * (min_lon + max_lon)
    c_lat = 0.5 * (min_lat + max_lat)
    sx = max(spread * 0.5 * (max_lon - min_lon), 1e-3)
    sy = max(spread * 0.5 * (max_lat - min_lat), 1e-3)
    return amplitude * np.exp(
        -(((lon2d - c_lon) ** 2) / (2 * sx**2) + ((lat2d - c_lat) ** 2) / (2 * sy**2))
    )


def _synthesize_static(grid: "xr.Dataset | xr.DataArray") -> "xr.Dataset":
    """Build deterministic, plausible synthetic static covariates over the grid.

    The fields are smooth functions of latitude/longitude plus correlated noise,
    arranged so that emission-like layers (nightlights, population, roads, urban
    land cover) peak over the Indo-Gangetic Plain and Punjab/Haryana, vegetation
    peaks over the forest belt, and terrain rises toward the Himalayan north.

    Args:
        grid: Gridded ``Dataset``/``DataArray`` with 1-D ``lat``/``lon`` coords.

    Returns:
        An :class:`xarray.Dataset` of the :data:`STATIC_COVARIATES`
        (dims ``(lat, lon)``, ``float32``), with ``attrs["synthetic"] = True``.
    """
    import xarray as xr

    lat, lon = _grid_latlon(grid)
    lon2d, lat2d = np.meshgrid(lon, lat)  # shape (nlat, nlon)
    rng = np.random.default_rng(_SYNTH_SEED)

    # Smooth correlated noise field via separable low-frequency sinusoids.
    def smooth_noise(scale: float) -> np.ndarray:
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        kx = rng.uniform(0.1, 0.5)
        ky = rng.uniform(0.1, 0.5)
        field = np.sin(kx * lon2d + phase_x) * np.cos(ky * lat2d + phase_y)
        return scale * field

    # --- Terrain: rises toward the north (Himalaya) + smooth relief. ----------
    north = (lat2d - lat2d.min()) / max(lat2d.max() - lat2d.min(), 1e-6)
    himalaya = _gaussian_bump(
        lon2d, lat2d, (74.0, 30.0, 95.0, 37.0), amplitude=4000.0, spread=1.5
    )
    relief = 300.0 * (1.0 + smooth_noise(1.0))
    dem = np.clip(50.0 + 1500.0 * north**2 + himalaya + relief, 0.0, 8000.0)

    # Slope as the gradient magnitude of the DEM (degrees, rough scaling).
    gy, gx = np.gradient(dem)
    slope = np.clip(np.degrees(np.arctan(np.hypot(gx, gy) / 5000.0)), 0.0, 60.0)

    # TPI: cell elevation minus local mean (positive = ridge, negative = valley).
    tpi = dem - _box_mean(dem, radius=2)

    # --- Land cover fractions (sum-normalized urban/crop/forest). -------------
    urban = (
        _gaussian_bump(lon2d, lat2d, IGP_BBOX, amplitude=0.6, spread=0.8)
        + _gaussian_bump(lon2d, lat2d, (72.5, 18.5, 73.5, 19.5), amplitude=0.5)
        + _gaussian_bump(lon2d, lat2d, (77.0, 12.5, 78.0, 13.5), amplitude=0.4)
    )
    crop = _gaussian_bump(
        lon2d, lat2d, IGP_BBOX, amplitude=0.7, spread=1.4
    ) + _gaussian_bump(lon2d, lat2d, PUNJAB_HARYANA_BBOX, amplitude=0.6)
    forest = _gaussian_bump(
        lon2d, lat2d, FOREST_BELT_BBOX, amplitude=0.7, spread=1.2
    )
    urban = np.clip(urban, 0.0, 1.0)
    crop = np.clip(crop, 0.0, 1.0)
    forest = np.clip(forest, 0.0, 1.0)
    total = urban + crop + forest + 0.2  # +0.2 = "other" so fractions <1.
    urban, crop, forest = urban / total, crop / total, forest / total

    # --- NDVI: high over forest/crop, low over urban/desert. ------------------
    ndvi = np.clip(
        0.2 + 0.6 * forest + 0.4 * crop - 0.3 * urban + 0.05 * smooth_noise(1.0),
        -0.1,
        0.95,
    )

    # --- Nightlights / population / roads: anthropogenic, peak over IGP. -------
    anthropo = (
        _gaussian_bump(lon2d, lat2d, IGP_BBOX, amplitude=1.0, spread=0.9)
        + _gaussian_bump(lon2d, lat2d, (76.5, 28.0, 77.8, 29.2), amplitude=1.0)  # Delhi
        + _gaussian_bump(lon2d, lat2d, (72.5, 18.5, 73.3, 19.4), amplitude=0.9)  # Mumbai
        + _gaussian_bump(lon2d, lat2d, (88.0, 22.3, 88.6, 22.9), amplitude=0.7)  # Kolkata
    )
    anthropo = np.clip(anthropo + 0.05 * np.abs(smooth_noise(1.0)), 0.0, None)
    nightlights = 63.0 * anthropo / max(anthropo.max(), 1e-6)  # VIIRS-like [0,63]
    population = np.clip(
        20000.0 * (anthropo / max(anthropo.max(), 1e-6)) ** 1.5, 0.0, None
    )
    road_density = np.clip(
        10.0 * anthropo / max(anthropo.max(), 1e-6) + 0.2 * crop, 0.0, None
    )

    fields: dict[str, tuple[np.ndarray, str]] = {
        "dem": (dem, "Elevation above sea level (m)"),
        "slope": (slope, "Terrain slope (degrees)"),
        "tpi": (tpi, "Topographic Position Index (m)"),
        "landcover_urban": (urban, "Urban land-cover fraction"),
        "landcover_crop": (crop, "Cropland land-cover fraction"),
        "landcover_forest": (forest, "Forest land-cover fraction"),
        "ndvi": (ndvi, "Normalized Difference Vegetation Index"),
        "nightlights": (nightlights, "VIIRS nighttime lights radiance (nW/cm2/sr)"),
        "population": (population, "Population count per cell (WorldPop-like)"),
        "road_density": (road_density, "Road density (km/km2, OSM/GHSL-like)"),
    }

    coords = {"lat": grid["lat"], "lon": grid["lon"]}
    data_vars = {
        name: (
            ("lat", "lon"),
            arr.astype("float32"),
            {"long_name": desc, "source": "synthetic", "synthetic": "true"},
        )
        for name, (arr, desc) in fields.items()
    }
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs["synthetic"] = True
    ds.attrs["description"] = (
        "SYNTHETIC static covariates (aqi_india.features.static_covars). "
        "Plausible deterministic fields for the offline demo; replace with real "
        "DEM/WorldCover/NDVI/VIIRS-VNP46/WorldPop/OSM layers for production."
    )
    return ds


def _box_mean(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """Mean over a square ``(2*radius+1)`` window via cumulative sums (edge-safe).

    Args:
        arr: 2-D array.
        radius: Half-window size in cells.

    Returns:
        The windowed local mean, same shape as ``arr``.
    """
    if radius < 1:
        return arr.copy()
    padded = np.pad(arr, radius, mode="edge")
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    win = 2 * radius + 1
    n0, n1 = arr.shape
    out = (
        csum[win : win + n0, win : win + n1]
        - csum[0:n0, win : win + n1]
        - csum[win : win + n0, 0:n1]
        + csum[0:n0, 0:n1]
    )
    return out / (win * win)


def load_raster_covariate(
    path: "str | Path", grid: "xr.Dataset | xr.DataArray", *, name: str
) -> "xr.DataArray":
    """Load a real raster covariate and resample it onto the grid.

    External-data adapter (lazy ``rioxarray`` import). Reads a georeferenced
    raster (GeoTIFF/COG) and bilinearly resamples it to the target grid's
    ``lat``/``lon``.

    Args:
        path: Path to a georeferenced raster.
        grid: Target grid providing ``lat``/``lon`` coordinates.
        name: Name to assign the returned DataArray.

    Returns:
        A 2-D ``xr.DataArray`` (dims ``(lat, lon)``) aligned to ``grid``.
    """
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    da = xr.open_dataarray(path, engine="rasterio")
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    da = da.rename({da.rio.x_dim: "lon", da.rio.y_dim: "lat"})
    resampled = da.interp(lat=grid["lat"], lon=grid["lon"], method="linear")
    resampled.name = name
    resampled.attrs["source"] = str(path)
    return resampled


def get_static_covariates(
    grid: "xr.Dataset | xr.DataArray",
    *,
    layers: "dict[str, str | Path] | None" = None,
) -> "xr.Dataset":
    """Return the static covariate stack aligned to ``grid``.

    For any covariate whose real raster path is provided in ``layers`` the raster
    is loaded and resampled; all remaining covariates fall back to the
    deterministic synthetic fields (clearly flagged). With ``layers=None`` (the
    demo default) the entire stack is synthetic.

    Args:
        grid: Gridded ``Dataset``/``DataArray`` whose ``lat``/``lon`` define the
            target geometry (DEV_CONTRACT S6.1 cube).
        layers: Optional mapping ``{covariate_name: raster_path}`` of real layers
            to load instead of synthesizing.

    Returns:
        An :class:`xarray.Dataset` of static covariates with dims ``(lat, lon)``;
        ``attrs["synthetic"]`` is ``True`` when any field is synthetic.
    """
    ds = _synthesize_static(grid)
    if not layers:
        _log.info(
            "Using fully SYNTHETIC static covariates over the India grid "
            "(no real layers supplied)."
        )
        return ds

    any_synthetic = False
    for name in STATIC_COVARIATES:
        if name in layers:
            try:
                ds[name] = load_raster_covariate(layers[name], grid, name=name)
                _log.info("Loaded real static covariate %r from %s", name, layers[name])
            except Exception as exc:  # pragma: no cover - adapter/IO path
                _log.warning(
                    "Failed to load %r (%s); falling back to synthetic.", name, exc
                )
                any_synthetic = True
        else:
            any_synthetic = True
    ds.attrs["synthetic"] = any_synthetic
    return ds


__all__ = [
    "STATIC_COVARIATES",
    "get_static_covariates",
    "load_raster_covariate",
]
