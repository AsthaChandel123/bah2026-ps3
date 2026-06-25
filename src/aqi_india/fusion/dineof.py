"""DINEOF — Data-Interpolating Empirical Orthogonal Functions gap-fill.

DINEOF (Beckers & Rixen, 2003) is a *label-free* reconstruction technique that
fills missing values in a space-time data matrix by iterative truncated-SVD
(EOF) reconstruction. It is the project's first-pass cloud / orbit gap-filler
for satellite column and AOD cubes (see synthesis blueprint, Stage-2 method 6):
it requires no ground truth, returns an explained-variance diagnostic, and
yields a cell-wise uncertainty estimate from the cross-validation error.

Algorithm (pure :mod:`numpy` / :mod:`scipy`):

1. Reshape the cube to a 2-D matrix ``X`` of shape ``(time, space)``; record the
   missing mask. Optionally hold out a small random set of *present* values as a
   cross-validation (CV) set used to choose the EOF rank.
2. Initialise missing entries (and the held-out CV entries) with the per-row /
   global mean, after removing the temporal+spatial mean.
3. For ``k = 1 .. k_max`` EOFs: iterate truncated-SVD reconstruction to
   convergence, refilling only the missing entries each step. Track the RMSE on
   the held-out CV set. The optimal rank ``k*`` minimises the CV RMSE.
4. Final reconstruction at ``k*`` fills the true gaps; the CV RMSE is reported as
   a scalar reconstruction uncertainty and broadcast to a per-cell uncertainty
   field scaled by the local missing fraction.

The implementation operates on any leading "sample" axis (``time``) and arbitrary
trailing spatial axes, so it works directly on ``(time, lat, lon)`` cubes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = get_logger("fusion.dineof")


@dataclass
class DineofResult:
    """Container for a DINEOF reconstruction and its diagnostics.

    Attributes:
        filled: The gap-filled array, same shape as the input.
        n_eof: The selected number of EOF modes (rank ``k*``).
        explained_variance: Fraction of variance explained by the retained modes
            over the originally-present entries (0..1).
        cv_rmse: Cross-validation RMSE on held-out present values (in input
            units); ``nan`` if no CV set could be formed.
        uncertainty: Per-cell uncertainty array (same shape as ``filled``),
            largest where data were missing.
        converged: Whether the inner reconstruction loop converged at ``k*``.
        n_iter: Number of inner iterations used at ``k*``.
        gap_fraction: Fraction of the input that was missing (NaN).
    """

    filled: np.ndarray
    n_eof: int
    explained_variance: float
    cv_rmse: float
    uncertainty: np.ndarray
    converged: bool
    n_iter: int
    gap_fraction: float
    singular_values: np.ndarray = field(default_factory=lambda: np.empty(0))


def _reconstruct_fixed_rank(
    X: np.ndarray,
    missing: np.ndarray,
    k: int,
    *,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Iterate truncated-SVD reconstruction at fixed rank ``k``.

    Args:
        X: Working ``(n_time, n_space)`` matrix; missing entries pre-filled with
            the initial guess (typically zeros after mean removal).
        missing: Boolean mask (True where the value must be refilled each step).
        k: Number of EOF modes to retain.
        max_iter: Maximum reconstruction iterations.
        tol: Relative-change convergence threshold on the missing entries.

    Returns:
        Tuple ``(X_filled, singular_values, converged, n_iter)``.
    """
    Xc = X.copy()
    prev = Xc[missing].copy()
    sv = np.zeros(k)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - n_iter reported on exit
        # Economy SVD; truncate to k modes.
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        kk = min(k, s.size)
        recon = (U[:, :kk] * s[:kk]) @ Vt[:kk]
        Xc[missing] = recon[missing]
        sv = s[:kk]
        cur = Xc[missing]
        denom = np.linalg.norm(prev) + 1e-12
        rel = np.linalg.norm(cur - prev) / denom
        prev = cur.copy()
        if rel < tol:
            converged = True
            break
    return Xc, sv, converged, n_iter


def dineof(
    data: np.ndarray,
    *,
    k_max: int = 20,
    max_iter: int = 100,
    tol: float = 1e-4,
    cv_fraction: float = 0.02,
    min_cv_points: int = 20,
    seed: int = 42,
) -> DineofResult:
    """Gap-fill an N-D array via Data-Interpolating EOF reconstruction.

    The first axis is treated as the temporal / sample axis; all remaining axes
    are flattened into the spatial dimension. Missing entries are ``NaN``.

    Args:
        data: Input array with ``NaN`` at missing cells. Shape ``(n_time, *space)``.
        k_max: Maximum number of EOF modes to consider during rank selection.
        max_iter: Maximum inner reconstruction iterations per rank.
        tol: Relative-change convergence tolerance for the inner loop.
        cv_fraction: Fraction of *present* values held out for cross-validated
            rank selection.
        min_cv_points: Minimum CV points required to perform rank selection; if
            fewer are available the largest feasible rank is used directly.
        seed: RNG seed for reproducible CV-point selection.

    Returns:
        A :class:`DineofResult` with the filled array and diagnostics.

    Raises:
        ValueError: If the array has no finite values to learn from.
    """
    arr = np.asarray(data, dtype="float64")
    shape = arr.shape
    n_time = shape[0]
    n_space = int(np.prod(shape[1:])) if arr.ndim > 1 else 1
    X = arr.reshape(n_time, n_space)

    missing = ~np.isfinite(X)
    gap_fraction = float(missing.mean())
    n_present = int((~missing).sum())
    if n_present == 0:
        raise ValueError("DINEOF requires at least one finite value")

    if gap_fraction == 0.0:
        logger.debug("DINEOF: no gaps to fill")
        return DineofResult(
            filled=arr.reshape(shape).astype("float32"),
            n_eof=0,
            explained_variance=1.0,
            cv_rmse=float("nan"),
            uncertainty=np.zeros(shape, dtype="float32"),
            converged=True,
            n_iter=0,
            gap_fraction=0.0,
        )

    rng = np.random.default_rng(seed)

    # --- Cross-validation hold-out from the PRESENT values -------------------
    present_idx = np.flatnonzero(~missing.ravel())
    n_cv = int(round(cv_fraction * present_idx.size))
    cv_idx = np.empty(0, dtype="int64")
    cv_truth = np.empty(0)
    if n_cv >= min_cv_points:
        cv_idx = rng.choice(present_idx, size=n_cv, replace=False)
        cv_truth = X.ravel()[cv_idx].copy()

    work_missing = missing.copy()
    if cv_idx.size:
        flat_missing = work_missing.ravel()
        flat_missing[cv_idx] = True
        work_missing = flat_missing.reshape(missing.shape)

    # --- Centre the matrix (remove column/row means over present data) -------
    col_mean = np.where(
        (~work_missing).any(axis=0),
        np.nanmean(np.where(work_missing, np.nan, X), axis=0),
        0.0,
    )
    col_mean = np.nan_to_num(col_mean)
    Xc = X.copy()
    Xc[work_missing] = np.take(col_mean, np.where(work_missing)[1])
    Xc = Xc - col_mean  # broadcast over rows
    Xc[work_missing] = 0.0  # initial guess for gaps after centring

    k_cap = min(k_max, n_time, n_space)
    k_cap = max(1, k_cap)

    # --- Rank selection via CV RMSE -----------------------------------------
    best_k = k_cap
    best_rmse = float("inf")
    if cv_idx.size:
        cv_truth_centred = cv_truth - np.take(col_mean, cv_idx % n_space)
        for k in range(1, k_cap + 1):
            Xk, _, _, _ = _reconstruct_fixed_rank(
                Xc, work_missing, k, max_iter=max_iter, tol=tol
            )
            pred = Xk.ravel()[cv_idx]
            rmse = float(np.sqrt(np.mean((pred - cv_truth_centred) ** 2)))
            logger.debug("DINEOF rank=%d cv_rmse=%.5g", k, rmse)
            if rmse < best_rmse - 1e-12:
                best_rmse = rmse
                best_k = k
            elif rmse > best_rmse * 1.5:
                # CV error climbing well past the minimum -> stop early.
                break
    else:
        logger.debug(
            "DINEOF: only %d present values, skipping CV rank selection", n_present
        )

    # --- Final reconstruction on TRUE gaps only, at best_k -------------------
    final_missing = missing.copy()
    Xc_final = X.copy()
    Xc_final[final_missing] = np.take(col_mean, np.where(final_missing)[1])
    Xc_final = Xc_final - col_mean
    Xc_final[final_missing] = 0.0
    Xk, sv, converged, n_iter = _reconstruct_fixed_rank(
        Xc_final, final_missing, best_k, max_iter=max_iter, tol=tol
    )
    filled = Xk + col_mean

    # --- Explained variance over originally-present entries ------------------
    present = ~missing
    orig_centred = (X - col_mean)[present]
    resid = (Xk - (X - col_mean))[present]
    ss_tot = float(np.sum(orig_centred**2)) + 1e-12
    ss_res = float(np.sum(resid**2))
    explained_variance = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))

    cv_rmse = best_rmse if np.isfinite(best_rmse) else float("nan")

    # --- Per-cell uncertainty: base CV RMSE scaled up inside gaps ------------
    base = cv_rmse if np.isfinite(cv_rmse) else float(np.nanstd(X[present]))
    unc = np.full(X.shape, base, dtype="float64")
    # Inflate uncertainty inside gaps by local temporal missing fraction.
    space_missing_frac = missing.mean(axis=0, keepdims=True)
    unc = np.where(missing, base * (1.0 + 2.0 * space_missing_frac), base * 0.25)

    return DineofResult(
        filled=filled.reshape(shape).astype("float32"),
        n_eof=int(best_k),
        explained_variance=explained_variance,
        cv_rmse=cv_rmse,
        uncertainty=unc.reshape(shape).astype("float32"),
        converged=bool(converged),
        n_iter=int(n_iter),
        gap_fraction=gap_fraction,
        singular_values=np.asarray(sv, dtype="float64"),
    )


def dineof_fill(
    da,  # noqa: ANN001 - xarray DataArray
    *,
    sample_dim: str = "time",
    k_max: int = 20,
    max_iter: int = 100,
    tol: float = 1e-4,
    cv_fraction: float = 0.02,
    seed: int = 42,
):
    """Apply :func:`dineof` to an :class:`xarray.DataArray`.

    The reconstruction operates with ``sample_dim`` as the temporal axis and all
    other dims as space. Returns a new DataArray (gap-filled) and attaches the
    DINEOF diagnostics under ``.attrs``; a companion uncertainty DataArray is
    returned alongside.

    Args:
        da: Input DataArray with ``NaN`` gaps.
        sample_dim: Name of the temporal/sample dimension (default ``"time"``).
        k_max: Maximum EOF modes for rank selection.
        max_iter: Maximum inner reconstruction iterations.
        tol: Inner-loop convergence tolerance.
        cv_fraction: Hold-out fraction for rank selection.
        seed: RNG seed.

    Returns:
        Tuple ``(filled_da, uncertainty_da, result)`` where ``result`` is the
        :class:`DineofResult`.
    """
    import xarray as xr

    if sample_dim not in da.dims:
        raise ValueError(f"sample_dim {sample_dim!r} not in DataArray dims {da.dims}")
    other = [d for d in da.dims if d != sample_dim]
    da_t = da.transpose(sample_dim, *other)
    res = dineof(
        np.asarray(da_t.values),
        k_max=k_max,
        max_iter=max_iter,
        tol=tol,
        cv_fraction=cv_fraction,
        seed=seed,
    )
    filled = xr.DataArray(
        res.filled, dims=da_t.dims, coords=da_t.coords, attrs=dict(da.attrs)
    )
    filled.attrs.update(
        {
            "gapfill_method": "dineof",
            "dineof_n_eof": res.n_eof,
            "dineof_explained_variance": res.explained_variance,
            "dineof_cv_rmse": res.cv_rmse,
            "dineof_gap_fraction": res.gap_fraction,
        }
    )
    unc = xr.DataArray(
        res.uncertainty,
        dims=da_t.dims,
        coords=da_t.coords,
        name=(f"{da.name}_uncertainty" if da.name else "uncertainty"),
    )
    return filled.transpose(*da.dims), unc.transpose(*da.dims), res


__all__ = ["DineofResult", "dineof", "dineof_fill"]
