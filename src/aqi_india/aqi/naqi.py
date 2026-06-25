"""Pure, vectorized CPCB National Air Quality Index (NAQI) engine.

The engine implements the deterministic O(1) CPCB (2014) NAQI computation:

1. ``sub_index(conc, pollutant)`` — piecewise-linear sub-index of a single
   pollutant concentration via ``np.searchsorted`` over the per-pollutant
   breakpoint array, using
   ``Ip = ((I_Hi - I_Lo)/(BP_Hi - BP_Lo)) * (C - BP_Lo) + I_Lo``.
2. ``aqi_from_subindices`` — the max-of-sub-index aggregation with the CPCB
   validity mask (>= 3 valid pollutants AND at least one of PM2.5 / PM10),
   plus the responsible pollutant (argmax).
3. ``category`` — map an AQI value to (name, colour).
4. ``compute_aqi`` / ``compute_aqi_grid`` — DataFrame and xarray entry points.

Everything is NaN-safe and vectorized over arbitrary array shapes. CO is carried
in mg/m3; all other pollutants in ug/m3 (see :mod:`aqi_india.aqi.breakpoints`).

All functions are pure (no I/O, no global state) and import only NumPy at module
scope so the engine is fast, dask-friendly and trivially unit-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from aqi_india.aqi.breakpoints import (
    AQI_BAND_HI,
    AQI_BAND_LO,
    AQI_MAX,
    BREAKPOINTS,
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    MIN_VALID_POLLUTANTS,
    PM_POLLUTANTS,
    POLLUTANTS,
    category_index,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import xarray as xr


def _segment_indices(conc: np.ndarray, bp: np.ndarray) -> np.ndarray:
    """Return the band index (0..5) each concentration falls into.

    A concentration exactly at an internal breakpoint maps to the *lower*
    segment that ends at that breakpoint (so e.g. PM2.5=90 -> AQI 200, not 201),
    matching CPCB worked examples. Concentrations at/above the top cap map to
    band 5 (handled by clamping in :func:`sub_index`).

    Args:
        conc: Concentration array (any shape), assumed finite.
        bp: Length-7 breakpoint array for the pollutant.

    Returns:
        Integer band-index array in [0, 5] with the same shape as ``conc``.
    """
    # bp has 7 edges -> 6 segments. searchsorted(side="left") on the 6 interior
    # edges bp[1:6] gives 0 for conc <= bp[1], counting how many interior edges
    # are < conc. Using side="left" makes a value exactly on an interior edge
    # belong to the lower segment.
    interior = bp[1:6]  # length 5: edges between the 6 bands
    idx = np.searchsorted(interior, conc, side="left")
    return np.clip(idx, 0, 5).astype(np.intp)


def sub_index(conc: Any, pollutant: str) -> np.ndarray:
    """Compute the CPCB sub-index for one pollutant (vectorized, NaN-safe).

    Args:
        conc: Concentration(s) in the pollutant's CPCB unit (CO in mg/m3, all
            others ug/m3). Scalar or array-like of any shape.
        pollutant: Pollutant key, one of
            ``{"pm25","pm10","no2","so2","co","o3","nh3","pb"}``.

    Returns:
        Float64 ``np.ndarray`` of sub-index values in [0, 500] with the same
        shape as ``conc``. NaN inputs and negative inputs propagate to NaN.

    Raises:
        KeyError: If ``pollutant`` is not a recognised CPCB pollutant.
    """
    if pollutant not in BREAKPOINTS:
        raise KeyError(
            f"Unknown pollutant {pollutant!r}; expected one of {tuple(BREAKPOINTS)}"
        )
    bp = BREAKPOINTS[pollutant]
    c = np.asarray(conc, dtype=np.float64)
    scalar_input = c.ndim == 0
    c = np.atleast_1d(c)

    out = np.full(c.shape, np.nan, dtype=np.float64)
    # Valid = finite and non-negative (negative concentrations are invalid).
    valid = np.isfinite(c) & (c >= 0.0)
    if not valid.any():
        return out[0] if scalar_input else out

    cv = c[valid]
    seg = _segment_indices(cv, bp)
    bp_lo = bp[seg]
    bp_hi = bp[seg + 1]
    i_lo = AQI_BAND_LO[seg]
    i_hi = AQI_BAND_HI[seg]

    # Piecewise-linear interpolation; guard against a degenerate zero-width band.
    span = bp_hi - bp_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(span > 0.0, (cv - bp_lo) / span, 0.0)
    sub = frac * (i_hi - i_lo) + i_lo

    # Above the top cap -> clamp to AQI_MAX; clamp the whole result to [0, 500].
    sub = np.clip(sub, 0.0, AQI_MAX)
    out[valid] = sub
    return out[0] if scalar_input else out


def _stack_subindices(
    subindices: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Stack a pollutant->sub-index dict into an array and a PM mask.

    Returns:
        Tuple ``(stacked, names, is_pm)`` where ``stacked`` has shape
        ``(n_pollutants, *grid)``, ``names`` is the ordered pollutant list, and
        ``is_pm`` is a boolean vector flagging PM2.5/PM10 rows.
    """
    names = [p for p in POLLUTANTS if p in subindices]
    if not names:
        raise ValueError("No recognised pollutant sub-indices supplied")
    arrays = [np.asarray(subindices[p], dtype=np.float64) for p in names]
    stacked = np.stack(np.broadcast_arrays(*arrays), axis=0)
    is_pm = np.array([p in PM_POLLUTANTS for p in names], dtype=bool)
    return stacked, names, is_pm


def aqi_from_subindices(
    subindices: dict[str, Any] | np.ndarray,
    pollutants: list[str] | None = None,
    *,
    return_responsible: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Aggregate per-pollutant sub-indices into the NAQI (max-of-sub-index).

    Applies the CPCB validity rule: a cell yields a valid AQI only if at least
    :data:`~aqi_india.aqi.breakpoints.MIN_VALID_POLLUTANTS` pollutant
    sub-indices are present AND at least one of PM2.5 / PM10 is among them.
    Otherwise the AQI is NaN.

    Args:
        subindices: Either a mapping ``{pollutant: sub_index_array}`` or a raw
            array of shape ``(n_pollutants, *grid)`` (then ``pollutants`` must
            give the row order).
        pollutants: Row order when ``subindices`` is a raw array.
        return_responsible: If True, also return the responsible pollutant per
            cell (the argmax), as an object array of pollutant names with
            ``None`` where the AQI is invalid.

    Returns:
        The AQI array, or ``(aqi, responsible)`` if ``return_responsible``.
    """
    if isinstance(subindices, dict):
        stacked, names, is_pm = _stack_subindices(subindices)
    else:
        stacked = np.asarray(subindices, dtype=np.float64)
        if stacked.ndim < 1:
            raise ValueError("Array input must have a leading pollutant axis")
        if pollutants is None or len(pollutants) != stacked.shape[0]:
            raise ValueError(
                "`pollutants` must list one name per row of the sub-index array"
            )
        names = list(pollutants)
        is_pm = np.array([p in PM_POLLUTANTS for p in names], dtype=bool)

    valid_mask = np.isfinite(stacked)  # (n, *grid)
    n_valid = valid_mask.sum(axis=0)
    pm_present = (valid_mask & is_pm[(...,) + (None,) * (stacked.ndim - 1)]).any(axis=0)
    cell_valid = (n_valid >= MIN_VALID_POLLUTANTS) & pm_present

    with np.errstate(invalid="ignore"):
        aqi = np.nanmax(np.where(valid_mask, stacked, -np.inf), axis=0)
    aqi = np.where(cell_valid, aqi, np.nan)

    if not return_responsible:
        return aqi

    # argmax over the (NaN -> -inf) stacked array; mask invalid cells to None.
    filled = np.where(valid_mask, stacked, -np.inf)
    arg = np.argmax(filled, axis=0)
    names_arr = np.array(names, dtype=object)
    responsible = names_arr[arg]
    responsible = np.where(cell_valid, responsible, None)
    if responsible.ndim == 0:
        responsible = responsible.reshape(())
    return aqi, responsible


def responsible_pollutant(
    subindices: dict[str, Any] | np.ndarray,
    pollutants: list[str] | None = None,
) -> np.ndarray:
    """Return the responsible (max sub-index) pollutant per cell.

    See :func:`aqi_from_subindices`; this is a convenience wrapper returning only
    the responsible-pollutant array.
    """
    _, resp = aqi_from_subindices(  # type: ignore[misc]
        subindices, pollutants, return_responsible=True
    )
    return resp


def category(aqi: float) -> tuple[str, str]:
    """Return ``(category_name, hex_color)`` for an AQI value."""
    idx = category_index(aqi)
    return CATEGORY_NAMES[idx], CATEGORY_COLORS[idx]


def compute_aqi(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    out_prefix: str = "",
) -> pd.DataFrame:
    """Compute NAQI for a DataFrame of pollutant concentration columns.

    Args:
        df: DataFrame whose columns hold pollutant concentrations. By default the
            columns are expected to be named with the canonical pollutant keys
            (``pm25``, ``pm10``, ``no2``, ``so2``, ``co``, ``o3``, ``nh3``,
            ``pb``); any subset present is used. CO is interpreted in mg/m3.
        column_map: Optional ``{pollutant_key: dataframe_column}`` override for
            non-canonical column names.
        out_prefix: Prefix for the output columns (``aqi``, ``aqi_category``,
            ``aqi_responsible``).

    Returns:
        A copy of ``df`` with added columns ``{prefix}aqi`` (float),
        ``{prefix}aqi_category`` (str or None) and ``{prefix}aqi_responsible``
        (str or None).
    """
    import pandas as pd  # local import keeps module import light

    result = df.copy()
    mapping = column_map or {p: p for p in POLLUTANTS}
    present = {p: col for p, col in mapping.items() if col in df.columns and p in BREAKPOINTS}
    if not present:
        raise ValueError(
            "No pollutant columns found in DataFrame; expected any of "
            f"{tuple(BREAKPOINTS)} (or supply column_map)"
        )

    subidx = {
        p: sub_index(df[col].to_numpy(dtype=np.float64), p) for p, col in present.items()
    }
    aqi, resp = aqi_from_subindices(subidx, return_responsible=True)  # type: ignore[misc]

    cats = pd.Series(
        [None if not np.isfinite(a) else category(a)[0] for a in np.atleast_1d(aqi)],
        index=df.index,
        dtype=object,
    )
    result[f"{out_prefix}aqi"] = np.atleast_1d(aqi)
    result[f"{out_prefix}aqi_category"] = cats
    result[f"{out_prefix}aqi_responsible"] = pd.Series(
        np.atleast_1d(resp), index=df.index, dtype=object
    )
    return result


def compute_aqi_grid(
    ds: xr.Dataset,
    *,
    var_map: dict[str, str] | None = None,
    aqi_name: str = "aqi",
    responsible_name: str = "aqi_responsible",
) -> xr.Dataset:
    """Compute gridded NAQI from an xarray Dataset of pollutant fields.

    Vectorized and dask-friendly: sub-indices are computed per data variable and
    aggregated with the same validity mask as the tabular path. The responsible
    pollutant is returned as an integer code variable plus a ``pollutant_codes``
    attribute mapping codes to names (xarray cannot hold object dtype lazily).

    Args:
        ds: Dataset whose data variables are pollutant concentration fields named
            with canonical pollutant keys (any subset). CO is interpreted in
            mg/m3.
        var_map: Optional ``{pollutant_key: dataset_variable}`` override.
        aqi_name: Name of the output AQI variable.
        responsible_name: Name of the output responsible-pollutant-code variable.

    Returns:
        A new Dataset with the AQI field and an integer responsible-pollutant
        code field (``-1`` where invalid); ``coords``/``dims`` follow the inputs.
    """
    import xarray as xr  # local import keeps module import light

    mapping = var_map or {p: p for p in POLLUTANTS}
    present = {p: v for p, v in mapping.items() if v in ds.data_vars and p in BREAKPOINTS}
    if not present:
        raise ValueError(
            "No pollutant variables found in Dataset; expected any of "
            f"{tuple(BREAKPOINTS)} (or supply var_map)"
        )

    names = [p for p in POLLUTANTS if p in present]
    subidx_arrays = []
    template = ds[present[names[0]]]
    for p in names:
        arr = xr.apply_ufunc(
            sub_index,
            ds[present[p]],
            kwargs={"pollutant": p},
            dask="parallelized",
            output_dtypes=[np.float64],
        )
        subidx_arrays.append(arr)

    stacked = xr.concat(subidx_arrays, dim="pollutant")
    stacked = stacked.assign_coords(pollutant=names)

    valid = np.isfinite(stacked)
    n_valid = valid.sum(dim="pollutant")
    is_pm = xr.DataArray(
        [p in PM_POLLUTANTS for p in names], dims="pollutant", coords={"pollutant": names}
    )
    pm_present = (valid & is_pm).any(dim="pollutant")
    cell_valid = (n_valid >= MIN_VALID_POLLUTANTS) & pm_present

    aqi = stacked.max(dim="pollutant", skipna=True)
    aqi = aqi.where(cell_valid)

    # Integer responsible-pollutant code (-1 = invalid).
    filled = stacked.where(valid, -np.inf)
    code = filled.argmax(dim="pollutant")
    code = code.where(cell_valid, -1).astype(np.int16)

    out = xr.Dataset(
        {aqi_name: aqi.astype(np.float32), responsible_name: code},
        coords=template.coords,
    )
    out[aqi_name].attrs.update(
        {"long_name": "CPCB National Air Quality Index", "units": "1", "valid_range": "0 500"}
    )
    out[responsible_name].attrs.update(
        {
            "long_name": "responsible pollutant code",
            "pollutant_codes": ",".join(f"{i}:{n}" for i, n in enumerate(names)),
            "_FillValue": -1,
        }
    )
    return out


__all__ = [
    "sub_index",
    "aqi_from_subindices",
    "responsible_pollutant",
    "category",
    "compute_aqi",
    "compute_aqi_grid",
]
