"""INSAT-3D / 3DR Imager AOD (550 nm) adapter — the PS-mandated diurnal AOD.

INSAT-3D/3DR is the **problem-statement-mandated** AOD input. It is geostationary
(~15-30 min cadence) so it fills the twice-daily MODIS/VIIRS gap and captures
dust-storm / diurnal evolution, but it is **NOT** Earth-Engine-native: products
are HDF5 files distributed by ISRO MOSDAC / VEDAS that must be downloaded and
regridded. This adapter documents that non-GEE access path and parses the HDF5
grids into the project's common ``time/lat/lon`` schema.

Because INSAT AOD correlates poorly with AERONET relative to MODIS, downstream it
is used for **cadence / diurnal gap-fill only** and bias-corrected against the
MAIAC 1 km anchor (regionalized quantile mapping) — see the blueprint.

Heavy / credentialed deps (``h5py``/``netCDF4``/``requests``) are lazy-imported
inside functions. The ``synthetic=True`` path delegates to the synthetic ``aod``
field so the offline pipeline still has an INSAT-shaped diurnal AOD source.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("ingest.insat_aod")

#: Documented MOSDAC / VEDAS access endpoints (free registration required).
MOSDAC_ENDPOINT: str = "https://www.mosdac.gov.in/insat-3d-data-products"
VEDAS_ENDPOINT: str = "https://vedas.sac.gov.in"

#: Candidate HDF5 dataset paths for the AOD 550 nm field across product versions.
INSAT_AOD_HDF_PATHS: tuple[str, ...] = ("AOD", "AOD_550", "Aerosol_Optical_Depth")

#: Common-grid fallback resolution (deg) used when regridding INSAT to the cube.
DEFAULT_REGRID_RES: float = 0.25

DateLike = str | date | datetime


def fetch_insat_aod(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float] = INDIA_BBOX,
    *,
    files: list[str | Path] | None = None,
    regrid_res: float = DEFAULT_REGRID_RES,
    synthetic: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Read / regrid daily INSAT-3D/3DR Imager AOD (550 nm) to the common grid.

    Real path: parses the provided MOSDAC/VEDAS HDF5 ``files`` (one or more per
    day), extracts the AOD 550 nm grid, regrids to a regular ``lat/lon`` grid at
    ``regrid_res``, and stacks them daily into an ``aod`` cube. (Acquisition of
    the HDF5 files themselves is an out-of-band, credentialed MOSDAC download —
    :func:`describe_access` documents it; this adapter consumes already-downloaded
    files, mirroring the external-source ingestion stage of the blueprint.)

    Offline path: delegates to :func:`aqi_india.sim.synthetic.make_grid` and
    returns the synthetic ``aod`` field (the INSAT diurnal surrogate) so the
    pipeline runs without MOSDAC credentials.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` clip footprint.
        files: Local HDF5 file paths to parse (real path). If ``None`` and not
            ``synthetic``, a clear error is raised pointing at MOSDAC.
        regrid_res: Target regular-grid resolution in degrees.
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A single-variable daily :class:`xarray.Dataset` named ``aod``.

    Raises:
        ValueError: If neither ``synthetic`` nor ``files`` is provided.
    """
    if synthetic:
        return _synthetic_insat(start, end, aoi, seed=seed)

    if not files:
        raise ValueError(
            "INSAT-3D AOD is not GEE-native: provide downloaded MOSDAC/VEDAS "
            "HDF5 file paths via `files=`, or run with synthetic=True. "
            f"Register and download from {MOSDAC_ENDPOINT}."
        )

    return _read_insat_hdf5_files(
        files, start, end, aoi, regrid_res=regrid_res
    )


def _synthetic_insat(
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    seed: int,
) -> xr.Dataset:
    """Return a synthetic INSAT diurnal-AOD surrogate (the ``aod`` field)."""
    from ..sim import synthetic as sim

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    n_days = max(int((t1 - t0).days), 1)
    grid = sim.make_grid(t0, n_days, bbox=aoi, seed=seed)
    out = grid[["aod"]].copy()
    out["aod"].attrs["source"] = "SYNTHETIC INSAT-3D Imager AOD 550nm (diurnal)"
    out["aod"].attrs["note"] = (
        "Stands in for MOSDAC INSAT HDF5; used for diurnal/cloud gap-fill, "
        "bias-corrected against MAIAC downstream."
    )
    logger.info("fetch_insat_aod(synthetic): aod, %d days", n_days)
    return out


def _read_insat_hdf5_files(
    files: list[str | Path],
    start: DateLike,
    end: DateLike,
    aoi: tuple[float, float, float, float],
    *,
    regrid_res: float,
) -> xr.Dataset:
    """Parse + regrid a list of INSAT HDF5 AOD granules into a daily cube."""
    import xarray as xr

    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    target_lat, target_lon = _common_axes(aoi, regrid_res)

    per_day: dict[pd.Timestamp, list[np.ndarray]] = {}
    for fp in files:
        day, aod2d = _parse_insat_granule(Path(fp), target_lat, target_lon, aoi)
        if day is None or aod2d is None:
            continue
        if not (t0 <= day < t1):
            continue
        per_day.setdefault(day, []).append(aod2d)

    if not per_day:
        raise ValueError(
            "no INSAT AOD granules fell within the requested window; check the "
            "file list and date range."
        )

    days = sorted(per_day)
    stack = np.stack(
        [np.nanmean(np.stack(per_day[d], axis=0), axis=0) for d in days], axis=0
    ).astype(np.float32)
    ds = xr.Dataset(
        {"aod": (("time", "lat", "lon"), stack)},
        coords={"time": pd.DatetimeIndex(days), "lat": target_lat, "lon": target_lon},
    )
    ds["aod"].attrs.update(
        {"source": "INSAT-3D/3DR Imager AOD 550nm (MOSDAC HDF5)", "units": "1"}
    )
    logger.info("fetch_insat_aod: parsed %d granules -> %d days", len(files), len(days))
    return ds


def _parse_insat_granule(
    path: Path,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    aoi: tuple[float, float, float, float],
) -> tuple[pd.Timestamp | None, np.ndarray | None]:
    """Extract (date, regridded AOD 2-D) from a single INSAT HDF5 file.

    The MOSDAC INSAT AOD HDF5 layout varies by version; this reader probes the
    common dataset names in :data:`INSAT_AOD_HDF_PATHS` and the lat/lon grids,
    applies the file's fill value, and bilinearly bins onto the target grid.

    Args:
        path: HDF5 granule path.
        target_lat: Ascending target latitude axis.
        target_lon: Ascending target longitude axis.
        aoi: Clip bbox.

    Returns:
        ``(date, aod_2d)`` or ``(None, None)`` if the file cannot be parsed.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "h5py is required to read INSAT HDF5 granules (pip install h5py)."
        ) from exc

    try:
        with h5py.File(path, "r") as fh:
            var = next((p for p in INSAT_AOD_HDF_PATHS if p in fh), None)
            if var is None:
                logger.warning("no AOD dataset in %s; skipping", path.name)
                return None, None
            dset = fh[var]
            aod = np.asarray(dset[...], dtype=np.float64).squeeze()
            fill = dset.attrs.get("_FillValue")
            if fill is not None:
                aod = np.where(aod == float(np.asarray(fill).ravel()[0]), np.nan, aod)
            scale = float(np.asarray(dset.attrs.get("scale_factor", 1.0)).ravel()[0])
            aod = aod * scale

            lat = _read_hdf_axis(fh, ("latitude", "Latitude", "lat"))
            lon = _read_hdf_axis(fh, ("longitude", "Longitude", "lon"))
            day = _infer_granule_date(fh, path)
    except OSError as exc:
        logger.warning("could not open INSAT granule %s: %s", path, exc)
        return None, None

    if lat is None or lon is None or aod.ndim != 2:
        logger.warning("INSAT granule %s missing grid axes; skipping", path.name)
        return None, None

    regridded = _regrid_to_axes(aod, lat, lon, target_lat, target_lon)
    return day, regridded


def _read_hdf_axis(fh, names: tuple[str, ...]) -> np.ndarray | None:
    """Return the first present 1-D coordinate array among ``names``."""
    for n in names:
        if n in fh:
            arr = np.asarray(fh[n][...], dtype=np.float64).squeeze()
            if arr.ndim == 2:  # 2-D geolocation -> take a representative line
                arr = arr[arr.shape[0] // 2, :] if n.lower().startswith("lon") else arr[:, arr.shape[1] // 2]
            return arr
    return None


def _infer_granule_date(fh, path: Path) -> pd.Timestamp | None:
    """Infer the acquisition date from HDF5 attrs or the filename."""
    for key in ("Acquisition_Date", "date", "Date"):
        if key in fh.attrs:
            try:
                return pd.Timestamp(str(np.asarray(fh.attrs[key]).ravel()[0])).normalize()
            except (ValueError, TypeError):
                pass
    # Fall back to an 8-digit YYYYMMDD token in the filename.
    import re

    m = re.search(r"(20\d{6})", path.stem)
    if m:
        return pd.Timestamp(m.group(1)).normalize()
    return None


def _common_axes(
    aoi: tuple[float, float, float, float], res: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build ascending target lat/lon axes for ``aoi`` at ``res`` (deg)."""
    min_lon, min_lat, max_lon, max_lat = aoi
    lon = np.arange(min_lon, max_lon + 0.5 * res, res, dtype=np.float64)
    lat = np.arange(min_lat, max_lat + 0.5 * res, res, dtype=np.float64)
    return lat, lon


def _regrid_to_axes(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour regrid a 2-D field onto target lat/lon axes.

    Uses index lookups (O(n)) rather than a heavyweight xESMF regridder so the
    light dependency set suffices; this is adequate for the coarse INSAT->common
    grid step (a quantile-mapping bias correction follows downstream anyway).

    Args:
        field: Source 2-D AOD aligned to ``(src_lat, src_lon)``.
        src_lat: Source latitude axis (may be descending).
        src_lon: Source longitude axis.
        tgt_lat: Ascending target latitude axis.
        tgt_lon: Ascending target longitude axis.

    Returns:
        The regridded 2-D field of shape ``(len(tgt_lat), len(tgt_lon))``.
    """
    # Ensure ascending source axes (flip the field to match).
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        field = field[::-1, :]
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]
        field = field[:, ::-1]

    iy = np.clip(np.searchsorted(src_lat, tgt_lat), 0, src_lat.size - 1)
    ix = np.clip(np.searchsorted(src_lon, tgt_lon), 0, src_lon.size - 1)
    return field[np.ix_(iy, ix)]


def describe_access() -> dict[str, str]:
    """Return a human-readable description of the non-GEE INSAT access path.

    Returns:
        Mapping with the MOSDAC/VEDAS endpoints, product, format and the
        documented role/caveat — useful for logging and the README/serving layer.
    """
    return {
        "provider": "ISRO MOSDAC / VEDAS",
        "access": "http_download (free registration; NOT GEE-native)",
        "mosdac": MOSDAC_ENDPOINT,
        "vedas": VEDAS_ENDPOINT,
        "product": "INSAT-3D/3DR Imager L2 AOD 550nm (HDF5)",
        "role": "diurnal/cloud AOD gap-fill; bias-correct vs MAIAC 1km",
        "caveat": "poor AERONET correlation vs MODIS; use for cadence only",
    }


__all__ = [
    "MOSDAC_ENDPOINT",
    "VEDAS_ENDPOINT",
    "INSAT_AOD_HDF_PATHS",
    "fetch_insat_aod",
    "describe_access",
]
