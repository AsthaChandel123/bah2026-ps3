"""Per-season robust z-anomaly standardisation of the HCHO column field.

TROPOMI HCHO is the noisiest S5P product (single-pixel error 30-100 %) and over
India it carries a strong static spatial gradient (high IGP/industrial baseline,
low ocean/desert) on top of a seasonal cycle. Running Getis-Ord / LISA on the
raw column therefore produces *striping* artefacts and confounds "always high"
industrial cells with genuine *biomass-burning anomalies*.

This module removes both nuisance signals so that one nationwide threshold is
comparable everywhere (blueprint method 33):

1.  Restrict the time-cube to the requested season (the meteorological grouping
    of each time step, e.g. ``post_monsoon``).
2.  Compute a per-cell **robust centre and scale** over the seasonal climatology
    using the *median* and the *median absolute deviation* (MAD), which are
    insensitive to the heavy-tailed HCHO retrieval noise.
3.  Standardise every observation to a robust z-anomaly
    ``z = (x - median) / (1.4826 * MAD)`` (the 1.4826 factor makes the scaled
    MAD a consistent estimator of the standard deviation for Gaussian data).
4.  Additionally expose the per-cell seasonal **95th-percentile surface** which
    feeds the third consensus vote (exceedance) downstream.

The public entry point is :func:`build_climatology`. It is pure NumPy/xarray and
imports under the light dependency set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("hotspot.climatology")

#: Scale factor making ``k * MAD`` a consistent estimator of the Gaussian sigma.
MAD_TO_SIGMA: float = 1.4826

#: Mapping from calendar month (1-12) to the meteorological season label used to
#: group the HCHO climatology. Aligned with the Hydra ``dates`` season tags
#: (``post_monsoon`` = the Oct-Nov paddy-residue burning window).
MONTH_TO_SEASON: dict[int, str] = {
    1: "winter",
    2: "winter",
    3: "pre_monsoon",
    4: "pre_monsoon",
    5: "pre_monsoon",
    6: "monsoon",
    7: "monsoon",
    8: "monsoon",
    9: "monsoon",
    10: "post_monsoon",
    11: "post_monsoon",
    12: "winter",
}


def season_of(months: np.ndarray) -> np.ndarray:
    """Map an array of month numbers to season labels.

    Args:
        months: Integer month numbers (1-12), any shape.

    Returns:
        An object ``ndarray`` of the same shape holding season strings.
    """
    months = np.asarray(months, dtype=int)
    lut = np.array([MONTH_TO_SEASON[m] for m in range(1, 13)], dtype=object)
    return lut[(months - 1) % 12]


def robust_z_anomaly(
    cube: np.ndarray,
    *,
    axis: int = 0,
    min_periods: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise a stack of fields to per-cell robust z-anomalies.

    The centre and scale are estimated per cell over ``axis`` (time) with the
    median and the scaled median absolute deviation, which resist the heavy
    HCHO retrieval noise far better than mean/std.

    Args:
        cube: Array shaped ``(time, ...)`` (or with time on ``axis``). May hold
            ``NaN`` for cloud/orbit gaps; those are ignored in the statistics.
        axis: The axis to reduce over (the time axis). Defaults to ``0``.
        min_periods: Minimum number of finite samples a cell needs before a
            z-anomaly is computed; cells below this stay ``NaN``.

    Returns:
        A tuple ``(z, median, scale)`` where ``z`` has the same shape as
        ``cube`` and ``median``/``scale`` are the per-cell reductions (time axis
        collapsed). Cells with a zero or undefined scale yield ``NaN`` z-values.
    """
    arr = np.asarray(cube, dtype=np.float64)
    moved = np.moveaxis(arr, axis, 0)

    median = np.nanmedian(moved, axis=0)
    abs_dev = np.abs(moved - median[np.newaxis, ...])
    mad = np.nanmedian(abs_dev, axis=0)
    scale = MAD_TO_SIGMA * mad

    # Guard against degenerate (constant or near-constant) cells.
    valid_count = np.sum(np.isfinite(moved), axis=0)
    bad_scale = ~np.isfinite(scale) | (scale <= 0)
    too_few = valid_count < min_periods
    safe_scale = np.where(bad_scale, np.nan, scale)

    z_moved = (moved - median[np.newaxis, ...]) / safe_scale[np.newaxis, ...]
    invalid = (bad_scale | too_few)[np.newaxis, ...]
    z_moved = np.where(np.broadcast_to(invalid, z_moved.shape), np.nan, z_moved)

    z = np.moveaxis(z_moved, 0, axis)
    return z, median, scale


def _fit_poly_surface(field: np.ndarray, degree: int) -> np.ndarray:
    """Least-squares fit a low-order 2-D polynomial trend surface.

    Args:
        field: 2-D field (``lat``, ``lon``) with optional ``NaN``.
        degree: Polynomial degree (``1`` = plane, ``2`` = quadratic).

    Returns:
        The fitted smooth surface, same shape as ``field``. Returns a constant
        median surface if there are too few finite cells to fit.
    """
    ny, nx = field.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    # Normalise coordinates to [-1, 1] for conditioning.
    xn = (xx / max(nx - 1, 1)) * 2 - 1
    yn = (yy / max(ny - 1, 1)) * 2 - 1

    terms = [np.ones_like(xn)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            terms.append((xn ** (d - i)) * (yn ** i))
    design = np.stack([t.ravel() for t in terms], axis=1)

    flat = field.ravel()
    finite = np.isfinite(flat)
    if finite.sum() < design.shape[1] + 1:
        return np.full_like(field, np.nanmedian(field))
    coef, *_ = np.linalg.lstsq(design[finite], flat[finite], rcond=None)
    return (design @ coef).reshape(field.shape)


def spatial_robust_anomaly(field: np.ndarray, *, detrend_degree: int = 2) -> np.ndarray:
    """Standardise one spatial field to a detrended robust spatial anomaly.

    Removes the smooth India-wide spatial gradient (a low-order polynomial
    surface) and then standardises the *residual* by its global median and
    scaled MAD. Unlike :func:`robust_z_anomaly` (which reduces over *time* and
    so erases a persistently-high blob), this keeps localised spatial contrast,
    so a persistent hotspot stands out while the broad gradient does NOT — the
    blueprint's "removes India spatial gradient + seasonal cycle so one
    threshold is comparable nationwide" (method 33). This is the field the
    spatial detectors (Getis-Ord Gi*, LISA) consume.

    Args:
        field: A 2-D spatial field (``lat``, ``lon``), possibly with ``NaN``.
        detrend_degree: Degree of the polynomial trend removed first
            (``0`` = no detrend, just centre/scale; ``2`` = quadratic, default).

    Returns:
        The detrended robust spatial anomaly, same shape; all-NaN or constant
        fields return all-``NaN``.
    """
    arr = np.asarray(field, dtype=np.float64)
    # The polynomial trend can only be fit on a 2-D (lat, lon) grid; flat input
    # (e.g. an already-vectorised cell list) skips the detrend step.
    if detrend_degree and detrend_degree > 0 and arr.ndim == 2:
        residual = arr - _fit_poly_surface(arr, detrend_degree)
    else:
        residual = arr

    median = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - median))
    scale = MAD_TO_SIGMA * mad
    if not np.isfinite(scale) or scale <= 0:
        return np.full_like(arr, np.nan)
    return (residual - median) / scale


def percentile_surface(
    cube: np.ndarray,
    *,
    q: float = 95.0,
    axis: int = 0,
) -> np.ndarray:
    """Compute the per-cell ``q``-th percentile surface over the time axis.

    Args:
        cube: Array shaped ``(time, ...)`` with optional ``NaN`` gaps.
        q: Percentile in ``[0, 100]``. Defaults to ``95``.
        axis: Time axis to reduce over.

    Returns:
        The per-cell percentile field with the time axis removed.
    """
    arr = np.asarray(cube, dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.nanpercentile(arr, q, axis=axis)


def build_climatology(
    hcho_cube: xr.DataArray | xr.Dataset,
    season: str | None = None,
    *,
    var: str = "hcho_col",
    q: float = 95.0,
    min_periods: int = 3,
) -> xr.Dataset:
    """Build the per-season robust climatology used by all hotspot detectors.

    The cube is filtered to the requested ``season`` (derived from each time
    step's month), then standardised to robust z-anomalies and reduced to a
    seasonal 95th-percentile surface.

    Args:
        hcho_cube: A HCHO time-cube. Either an :class:`xarray.DataArray` of the
            column field or a :class:`xarray.Dataset` containing ``var``. Must
            have a ``time`` dimension and spatial dims ``lat``/``lon`` per the
            DEV_CONTRACT schema (column units ``mol/m2``).
        season: Season label to keep (one of the values of
            :data:`MONTH_TO_SEASON`, e.g. ``"post_monsoon"``). If ``None`` the
            whole cube is used (every time step contributes).
        var: Data-variable name when ``hcho_cube`` is a Dataset.
        q: Exceedance percentile for the threshold surface. Defaults to ``95``.
        min_periods: Minimum finite samples per cell for a valid z-anomaly.

    Returns:
        An :class:`xarray.Dataset` aligned to the (filtered) input grid with:

        * ``z`` — per-cell *temporal* robust z-anomaly, dims ``(time, lat, lon)``
          (for Emerging Hot Spot Analysis trend tests).
        * ``z_spatial`` — *spatial* robust anomaly of the seasonal-mean column,
          dims ``(lat, lon)`` (the field the spatial detectors Gi*/LISA use).
        * ``hcho_mean`` — seasonal-mean HCHO column, dims ``(lat, lon)``.
        * ``hcho_median`` / ``hcho_scale`` — per-cell robust centre/scale.
        * ``hcho_p{q}`` — per-cell seasonal percentile surface.
        * ``exceedance`` — boolean per-cell-time mask ``hcho_col >= hcho_p{q}``.

        ``ds.attrs["season"]`` records the season filter applied.
    """
    import xarray as xr

    if isinstance(hcho_cube, xr.Dataset):
        if var not in hcho_cube:
            raise KeyError(
                f"variable {var!r} not found in dataset; available: "
                f"{list(hcho_cube.data_vars)}"
            )
        da = hcho_cube[var]
    else:
        da = hcho_cube

    if "time" not in da.dims:
        raise ValueError("hcho_cube must have a 'time' dimension")

    if season is not None:
        months = da["time"].dt.month.values
        keep = season_of(months) == season
        if not keep.any():
            raise ValueError(
                f"no time steps fall in season {season!r}; "
                "check the cube's date range"
            )
        da = da.isel(time=np.flatnonzero(keep))
        logger.info(
            "season=%s -> %d/%d time steps retained",
            season,
            int(keep.sum()),
            keep.size,
        )

    time_axis = da.get_axis_num("time")
    values = da.values

    z, median, scale = robust_z_anomaly(
        values, axis=time_axis, min_periods=min_periods
    )
    p_surface = percentile_surface(values, q=q, axis=time_axis)

    # Spatial anomaly of the seasonal-mean column: removes the India-wide
    # gradient/level while preserving spatial contrast for Gi*/LISA.
    mean_field = np.nanmean(values, axis=time_axis)
    z_spatial = spatial_robust_anomaly(mean_field)
    exceed = values >= p_surface[np.newaxis, ...] if time_axis == 0 else (
        values >= np.expand_dims(p_surface, axis=time_axis)
    )

    spatial_dims = [d for d in da.dims if d != "time"]
    spatial_coords = {d: da[d] for d in spatial_dims if d in da.coords}

    p_name = f"hcho_p{int(round(q))}"
    ds = xr.Dataset(
        data_vars={
            "z": (da.dims, z.astype("float32")),
            "z_spatial": (spatial_dims, z_spatial.astype("float32")),
            "hcho_mean": (spatial_dims, mean_field.astype("float32")),
            "hcho_median": (spatial_dims, median.astype("float32")),
            "hcho_scale": (spatial_dims, scale.astype("float32")),
            p_name: (spatial_dims, p_surface.astype("float32")),
            "exceedance": (da.dims, np.nan_to_num(exceed, nan=False).astype(bool)),
        },
        coords={**{"time": da["time"]}, **spatial_coords},
    )
    ds.attrs["season"] = season if season is not None else "all"
    ds.attrs["percentile"] = float(q)
    ds["z"].attrs["long_name"] = "robust HCHO z-anomaly (median/MAD standardised)"
    ds[p_name].attrs["long_name"] = f"seasonal {q:g}th-percentile HCHO surface"
    return ds


__all__ = [
    "MAD_TO_SIGMA",
    "MONTH_TO_SEASON",
    "season_of",
    "robust_z_anomaly",
    "spatial_robust_anomaly",
    "percentile_surface",
    "build_climatology",
]
