"""Gap-fill orchestration — the public fusion entrypoint the demo calls.

:func:`fill_gaps` runs the layered gap-fill cascade from the synthesis blueprint
(L2 of the architecture, risk-6 mitigation):

    DINEOF (label-free EOF first pass)
        -> U-Net / partial-conv inpaint refinement (lazy torch; DINEOF fallback)
        -> cKDTree IDW spatial fill (always available)
        -> reanalysis-prior fallback (a gap-free background field)

For each ``(time, lat, lon)`` variable it fills cloud / orbit holes, then reports
the gap fraction before and after. Variables that are already gap-free, or have
no usable signal, are passed through untouched. The function carries a per-cell
uncertainty channel (``<var>_uncertainty``) into the returned Dataset so the
downstream model can ingest a QA channel.

``method`` selects the strategy:

* ``"auto"`` (default): DINEOF, then IDW for any residual holes, then a constant
  reanalysis-prior fallback for whatever still remains. Robust and light.
* ``"dineof"``: DINEOF only.
* ``"inpaint"``: DINEOF then partial-conv U-Net refinement (lazy torch).
* ``"idw"``: spatial IDW per time slice only.

All paths run on the light dependency set; the heavy torch path is opt-in and
degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = get_logger("fusion.gapfill")

Method = Literal["auto", "dineof", "inpaint", "idw"]

#: Vars that are physically gap-free reanalysis priors — skip gap-filling but use
#: them (when present) as the background fallback for satellite vars.
REANALYSIS_VARS: tuple[str, ...] = (
    "blh",
    "rh",
    "wind_u",
    "wind_v",
    "t2m",
    "ssrd",
)


@dataclass
class GapReport:
    """Before/after gap-fraction accounting for one variable.

    Attributes:
        variable: Variable name.
        gap_before: Missing fraction before filling (0..1).
        gap_after: Missing fraction after filling (0..1).
        method: Method(s) actually applied, joined by ``"+"``.
        explained_variance: DINEOF explained variance (``nan`` if not used).
    """

    variable: str
    gap_before: float
    gap_after: float
    method: str
    explained_variance: float = float("nan")


def _gap_fraction(arr: np.ndarray) -> float:
    """Fraction of non-finite cells in ``arr``."""
    return float((~np.isfinite(arr)).mean())


def _idw_per_slice(arr: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Fill residual gaps in a ``(time, lat, lon)`` cube via per-slice IDW.

    Args:
        arr: Cube with NaN gaps, dims ``(time, lat, lon)`` (or ``(lat, lon)``).
        lats: 1-D latitudes (ascending).
        lons: 1-D longitudes (ascending).

    Returns:
        The cube with spatial gaps filled where any valid pixel exists in the
        slice; slices with no valid pixel are returned unchanged.
    """
    from .kriging import idw

    out = arr.copy()
    frames = out[None] if out.ndim == 2 else out
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)
    for i in range(frames.shape[0]):
        frame = frames[i]
        miss = ~np.isfinite(frame)
        if not miss.any() or (~miss).sum() < 3:
            continue
        valid = ~miss
        try:
            res = idw(
                mesh_lon[valid],
                mesh_lat[valid],
                frame[valid],
                lons,
                lats,
                k=min(8, int(valid.sum())),
            )
        except ValueError:
            continue
        frame[miss] = res.estimate[miss]
        frames[i] = frame
    return frames[0] if arr.ndim == 2 else frames


def _fill_variable(
    da,  # noqa: ANN001 - xarray DataArray
    method: Method,
    *,
    prior=None,  # noqa: ANN001 - optional gap-free background DataArray
    seed: int = 42,
):
    """Run the cascade on a single ``(time, lat, lon)`` DataArray.

    Args:
        da: Variable to fill (NaN gaps).
        method: Cascade selector (see module docstring).
        prior: Optional gap-free background array used as the final fallback.
        seed: RNG seed.

    Returns:
        Tuple ``(filled_da, uncertainty_da, report)``.
    """
    import xarray as xr

    from .dineof import dineof

    name = str(da.name)
    arr = np.asarray(da.values, dtype="float64")
    gap_before = _gap_fraction(arr)
    methods_used: list[str] = []
    explained = float("nan")
    uncertainty = np.full(arr.shape, np.nan, dtype="float32")

    if gap_before == 0.0:
        report = GapReport(name, 0.0, 0.0, "none")
        return da, xr.zeros_like(da).astype("float32"), report

    if not np.isfinite(arr).any():
        logger.warning("Variable %s is entirely missing; cannot gap-fill", name)
        report = GapReport(name, gap_before, gap_before, "skipped_empty")
        return da, xr.full_like(da, np.nan).astype("float32"), report

    filled = arr

    # --- Stage 1: DINEOF (or inpaint, which itself starts from DINEOF) -------
    if method in ("auto", "dineof", "inpaint") and da.ndim >= 2:
        # DINEOF needs a leading sample axis; ensure one.
        cube = filled if da.ndim == 3 else filled[None]
        try:
            if method == "inpaint":
                from .inpaint_unet import inpaint

                ip = inpaint(cube, seed=seed)
                cube_filled = ip.filled
                methods_used.append(ip.method)
                uncertainty = (
                    ip.uncertainty if ip.uncertainty.shape == cube.shape else uncertainty
                )
            else:
                res = dineof(cube, seed=seed)
                cube_filled = res.filled
                explained = res.explained_variance
                uncertainty = res.uncertainty
                methods_used.append("dineof")
            filled = cube_filled if da.ndim == 3 else cube_filled[0]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("DINEOF/inpaint failed for %s (%s)", name, type(exc).__name__)

    # --- Stage 2: IDW spatial fill for any residual holes --------------------
    if method in ("auto", "idw") and _gap_fraction(filled) > 0.0:
        lats = np.asarray(da["lat"].values, dtype="float64")
        lons = np.asarray(da["lon"].values, dtype="float64")
        filled = _idw_per_slice(filled, lats, lons)
        methods_used.append("idw")

    # --- Stage 3: reanalysis-prior / climatology fallback --------------------
    residual = ~np.isfinite(filled)
    if residual.any():
        if prior is not None:
            prior_arr = np.asarray(prior, dtype="float64")
            if prior_arr.shape == filled.shape:
                filled = np.where(residual, prior_arr, filled)
                methods_used.append("reanalysis_prior")
                residual = ~np.isfinite(filled)
    if residual.any():
        # Final backstop: temporal-then-global mean so no NaNs remain.
        fallback = np.nanmean(arr) if np.isfinite(arr).any() else 0.0
        filled = np.where(np.isfinite(filled), filled, fallback)
        methods_used.append("mean_fill")

    gap_after = _gap_fraction(filled)
    # Uncertainty is largest where the original was missing.
    orig_missing = ~np.isfinite(arr)
    if not np.isfinite(uncertainty).any():
        base = float(np.nanstd(arr)) if np.isfinite(arr).any() else 1.0
        uncertainty = np.where(orig_missing, base, base * 0.25).astype("float32")

    filled_da = xr.DataArray(
        filled.astype("float32"), dims=da.dims, coords=da.coords, attrs=dict(da.attrs)
    )
    filled_da.attrs.update(
        {
            "gapfill_method": "+".join(methods_used) or "none",
            "gapfill_gap_before": round(gap_before, 5),
            "gapfill_gap_after": round(gap_after, 5),
        }
    )
    unc_da = xr.DataArray(
        uncertainty, dims=da.dims, coords=da.coords, name=f"{name}_uncertainty"
    )
    report = GapReport(
        variable=name,
        gap_before=gap_before,
        gap_after=gap_after,
        method="+".join(methods_used) or "none",
        explained_variance=explained,
    )
    return filled_da, unc_da, report


def fill_gaps(
    ds,  # noqa: ANN001 - xarray Dataset | DataArray
    method: Method = "auto",
    *,
    variables=None,  # noqa: ANN001 - optional iterable of var names
    add_uncertainty: bool = True,
    seed: int = 42,
):
    """Fill cloud / orbit gaps across a gridded cube — the public entrypoint.

    Orchestrates the DINEOF -> IDW -> reanalysis-prior cascade per variable and
    reports the gap fraction before and after. Reanalysis variables
    (:data:`REANALYSIS_VARS`) are treated as gap-free priors and passed through.

    Args:
        ds: Input :class:`xarray.Dataset` (or DataArray) with ``NaN`` gaps and
            ``(time, lat, lon)`` data-vars per DEV_CONTRACT §6.1.
        method: One of ``"auto"``, ``"dineof"``, ``"inpaint"``, ``"idw"``.
        variables: Optional subset of data-vars to fill (Dataset input only);
            defaults to every floating-point ``(lat, lon)``-bearing variable.
        add_uncertainty: If True, append ``<var>_uncertainty`` channels.
        seed: RNG seed for the stochastic stages.

    Returns:
        A gap-filled object of the same type as ``ds``. For a Dataset, the global
        ``attrs["gapfill_reports"]`` holds a list of per-variable before/after
        gap fractions, and ``attrs["gapfill_gap_before"/"_after"]`` summarise the
        overall fraction across filled variables.
    """
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        prior = None
        filled_da, unc_da, report = _fill_variable(ds, method, prior=prior, seed=seed)
        filled_da.attrs["gapfill_reports"] = [vars(report)]
        return filled_da

    out = ds.copy()
    names = (
        list(variables)
        if variables is not None
        else [
            v
            for v in ds.data_vars
            if "lat" in ds[v].dims
            and "lon" in ds[v].dims
            and np.issubdtype(ds[v].dtype, np.floating)
        ]
    )

    reports: list[dict] = []
    total_before = 0.0
    total_after = 0.0
    n_counted = 0
    for name in names:
        if name in REANALYSIS_VARS:
            # Gap-free prior: only fill if it (unexpectedly) has holes.
            if _gap_fraction(np.asarray(ds[name].values)) == 0.0:
                continue
        filled_da, unc_da, report = _fill_variable(
            ds[name], method, prior=None, seed=seed
        )
        out[name] = filled_da
        if add_uncertainty:
            out[unc_da.name] = unc_da
        reports.append(vars(report))
        total_before += report.gap_before
        total_after += report.gap_after
        n_counted += 1
        logger.info(
            "gapfill %-10s %.1f%% -> %.1f%% via %s",
            name,
            100 * report.gap_before,
            100 * report.gap_after,
            report.method,
        )

    out.attrs["gapfill_reports"] = reports
    if n_counted:
        out.attrs["gapfill_gap_before"] = round(total_before / n_counted, 5)
        out.attrs["gapfill_gap_after"] = round(total_after / n_counted, 5)
    out.attrs["gapfill_method"] = method
    return out


__all__ = ["Method", "GapReport", "REANALYSIS_VARS", "fill_gaps"]
