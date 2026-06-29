"""Multi-sensor harmonization onto the common analysis grid.

This module stacks heterogeneous AOD / column sources (MAIAC, VIIRS, INSAT-3D,
MERRA-2, TROPOMI, ...) onto the project's common lat/lon grid and fuses them into
a single seamless field with an accompanying **provenance** channel (which sensor
supplied each cell) and a **QA / uncertainty** channel.

Two public surfaces:

* :func:`stack_sources` — align a list of source datasets onto the common grid
  (via :func:`aqi_india.fusion.regrid.to_grid`), returning a stacked Dataset plus
  provenance and per-cell uncertainty channels.
* :func:`fuse_aod` — multi-sensor AOD fusion (MAIAC + VIIRS + INSAT + MERRA-2)
  via Random-Forest imputation (synthesis blueprint method 11), producing a
  gap-reduced daily AOD field with a QA/uncertainty channel; falls back to a
  reliability-weighted blend when scikit-learn is unavailable.
* :func:`run` — the Hydra ``cfg`` entrypoint declared in DEV_CONTRACT §6.5
  (``aqi_india.fusion.harmonize.run(cfg) -> xr.Dataset``): loads the synthetic
  grid cube, harmonizes / gap-fills it, and writes the fused cube.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..utils.logging import get_logger
from .regrid import target_grid_from_cfg, to_grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("fusion.harmonize")

#: Default per-sensor reliability weights for AOD fusion (higher = more trusted).
#: MAIAC 1km is the quantitative anchor; INSAT is cadence-only (down-weighted);
#: reanalysis is gap-free but coarse. See the blueprint's AOD fusion matrix.
DEFAULT_AOD_RELIABILITY: dict[str, float] = {
    "maiac": 1.0,
    "modis": 0.9,
    "viirs": 0.8,
    "insat": 0.5,
    "merra2": 0.6,
    "cams": 0.6,
}


@dataclass
class StackResult:
    """A harmonized multi-source stack.

    Attributes:
        dataset: Stacked dataset with a leading ``source`` dim; each variable is
            on the common grid.
        provenance: Integer-coded provenance per cell (which source) for the
            fused variable, dims ``(time, lat, lon)``; ``-1`` where all missing.
        uncertainty: Per-cell uncertainty for the fused variable, same dims.
        source_names: Ordered source names matching the provenance codes.
    """

    dataset: xr.Dataset
    provenance: xr.DataArray
    uncertainty: xr.DataArray
    source_names: list[str]


def _source_name(ds, idx: int) -> str:  # noqa: ANN001
    """Derive a stable source name from a dataset's attrs or a fallback index."""
    for key in ("source", "sensor", "name", "instrument"):
        val = ds.attrs.get(key)
        if val:
            return str(val).lower()
    return f"source{idx}"


def stack_sources(
    list_of_ds: Sequence[xr.Dataset],
    target_grid,  # noqa: ANN001
    *,
    variable: str,
    method: str = "bilinear",
    reliability: Mapping[str, float] | None = None,
) -> StackResult:
    """Align multiple sensor datasets onto a common grid with provenance + QA.

    Each input dataset is regridded onto ``target_grid``; the named ``variable``
    is stacked along a new ``source`` dimension. A reliability-weighted nan-aware
    composite is computed as the fused field, recording which source won each
    cell (provenance) and a spread-based uncertainty.

    Args:
        list_of_ds: Source datasets (each with ``lat``/``lon`` and ``variable``).
        target_grid: Common analysis grid (see :func:`aqi_india.fusion.regrid.to_grid`).
        variable: Name of the data-var to harmonize (must exist in each source).
        method: Regridding method passed to :func:`to_grid`.
        reliability: Optional ``source_name -> weight`` map; unseen sources get 0.5.

    Returns:
        A :class:`StackResult`.

    Raises:
        ValueError: If ``list_of_ds`` is empty.
    """
    import xarray as xr

    if not list_of_ds:
        raise ValueError("stack_sources requires at least one source dataset")

    reliability = dict(reliability or {})
    names: list[str] = []
    regridded: list[xr.DataArray] = []
    for i, ds in enumerate(list_of_ds):
        name = _source_name(ds, i)
        names.append(name)
        rg = to_grid(ds[[variable]], target_grid, method=method)
        regridded.append(rg[variable])

    stacked = xr.concat(regridded, dim="source")
    stacked = stacked.assign_coords(source=("source", names))

    weights = np.array([reliability.get(n, 0.5) for n in names], dtype="float64")
    fused, prov, unc = _weighted_composite(stacked.values, weights)

    spatial_dims = list(regridded[0].dims)
    coords = {d: regridded[0].coords[d] for d in spatial_dims if d in regridded[0].coords}
    fused_da = xr.DataArray(fused, dims=spatial_dims, coords=coords, name=variable)
    prov_da = xr.DataArray(
        prov, dims=spatial_dims, coords=coords, name=f"{variable}_provenance"
    )
    prov_da.attrs["source_codes"] = dict(enumerate(names))
    unc_da = xr.DataArray(
        unc, dims=spatial_dims, coords=coords, name=f"{variable}_uncertainty"
    )

    out = stacked.to_dataset(name=variable)
    out[f"{variable}_fused"] = fused_da
    out.attrs["harmonize_sources"] = ",".join(names)
    return StackResult(
        dataset=out, provenance=prov_da, uncertainty=unc_da, source_names=names
    )


def _weighted_composite(stack: np.ndarray, weights: np.ndarray):
    """Reliability-weighted nan-aware composite over the leading source axis.

    Args:
        stack: Array ``(n_source, *spatial)``.
        weights: Per-source reliability ``(n_source,)``.

    Returns:
        Tuple ``(fused, provenance, uncertainty)``:
        * ``fused`` weighted mean over available sources (NaN where all missing);
        * ``provenance`` index of the highest-reliability *available* source per
          cell (``-1`` where all missing);
        * ``uncertainty`` weighted std across available sources, inflated where
          fewer sources contribute.
    """
    valid = np.isfinite(stack)
    w = weights[(slice(None),) + (None,) * (stack.ndim - 1)]
    wmask = np.where(valid, w, 0.0)
    wsum = wmask.sum(axis=0)
    vals = np.where(valid, stack, 0.0)
    fused = np.where(wsum > 0, (wmask * vals).sum(axis=0) / np.where(wsum > 0, wsum, 1), np.nan)

    # Provenance: highest-reliability available source.
    rel = np.where(valid, w, -np.inf)
    prov = np.argmax(rel, axis=0).astype("int32")
    prov = np.where(wsum > 0, prov, -1)

    # Weighted spread across available sources, inflated by sparsity.
    diff2 = np.where(valid, (stack - fused) ** 2, 0.0)
    var = np.where(wsum > 0, (wmask * diff2).sum(axis=0) / np.where(wsum > 0, wsum, 1), np.nan)
    n_avail = valid.sum(axis=0)
    sparsity = np.where(n_avail > 0, 1.0 / n_avail, np.nan)
    unc = np.sqrt(var) * (1.0 + sparsity)

    return fused.astype("float32"), prov, unc.astype("float32")


def fuse_aod(
    sources: Mapping[str, xr.DataArray],
    *,
    reliability: Mapping[str, float] | None = None,
    use_rf: bool = True,
    random_state: int = 42,
) -> xr.Dataset:
    """Fuse multi-sensor AOD into a seamless field with a QA/uncertainty channel.

    Implements the blueprint's multi-sensor AOD fusion (method 11): MAIAC +
    VIIRS + INSAT + MERRA-2 are combined by Random-Forest imputation — for each
    cell missing in the anchor (MAIAC), an RF trained on co-present cells across
    the other sensors predicts the value. When scikit-learn is unavailable (or
    ``use_rf=False``) it degrades to a reliability-weighted composite.

    All sources must already share the same ``(time, lat, lon)`` grid (call
    :func:`stack_sources` or :func:`to_grid` first).

    Args:
        sources: Mapping ``sensor_name -> DataArray`` of AOD on the common grid.
        reliability: Optional per-sensor reliability weights (defaults to
            :data:`DEFAULT_AOD_RELIABILITY`).
        use_rf: If True, attempt the RF imputation path.
        random_state: RNG seed for the RF.

    Returns:
        A Dataset with ``aod`` (fused), ``aod_uncertainty`` and ``aod_provenance``.

    Raises:
        ValueError: If ``sources`` is empty or grids are inconsistent.
    """
    import xarray as xr

    if not sources:
        raise ValueError("fuse_aod requires at least one source")
    reliability = {**DEFAULT_AOD_RELIABILITY, **(reliability or {})}

    names = list(sources.keys())
    arrays = [np.asarray(sources[n].values, dtype="float64") for n in names]
    ref = sources[names[0]]
    base_shape = arrays[0].shape
    if any(a.shape != base_shape for a in arrays):
        raise ValueError("all AOD sources must share the same grid shape")

    stack = np.stack(arrays, axis=0)  # (n_source, *grid)
    weights = np.array([reliability.get(n, 0.5) for n in names], dtype="float64")
    fused, prov, unc = _weighted_composite(stack, weights)

    if use_rf and stack.shape[0] >= 2:
        fused = _rf_impute(stack, weights, fused, random_state)

    dims = ref.dims
    coords = {d: ref.coords[d] for d in dims if d in ref.coords}
    out = xr.Dataset(
        {
            "aod": (dims, fused.astype("float32")),
            "aod_uncertainty": (dims, unc.astype("float32")),
            "aod_provenance": (dims, prov),
        },
        coords=coords,
    )
    out["aod_provenance"].attrs["source_codes"] = dict(enumerate(names))
    out.attrs["aod_sources"] = ",".join(names)
    out.attrs["aod_fusion"] = "rf_impute" if use_rf else "weighted_composite"
    return out


def _rf_impute(
    stack: np.ndarray,
    weights: np.ndarray,
    fallback: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Random-Forest imputation of the anchor sensor from the others.

    The first source (index 0, highest-priority anchor) is the prediction
    target; the remaining sources are predictors. Cells where the anchor is
    missing but >=1 predictor is present are imputed by an RF trained on cells
    where the anchor is present. Cells with no predictor fall back to the
    weighted composite.

    Args:
        stack: ``(n_source, *grid)`` source stack (anchor at index 0).
        weights: Per-source reliability (unused here; kept for signature parity).
        fallback: Weighted-composite field used where RF cannot predict.
        random_state: RNG seed.

    Returns:
        The imputed anchor field reshaped to the spatial grid.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:  # pragma: no cover - sklearn is in the light set
        logger.info("sklearn unavailable (%s); weighted composite", type(exc).__name__)
        return fallback

    n_source = stack.shape[0]
    flat = stack.reshape(n_source, -1)
    target = flat[0]
    predictors = flat[1:]  # (n_source-1, n_cells)

    train = np.isfinite(target) & np.any(np.isfinite(predictors), axis=0)
    if int(train.sum()) < 50:
        logger.debug("RF impute: too few training cells (%d); composite", int(train.sum()))
        return fallback

    Xtr = predictors[:, train].T
    ytr = target[train]
    # Impute NaNs in predictors with per-source means for training/prediction.
    pred_means = np.nan_to_num(np.nanmean(predictors, axis=1))
    Xtr = np.where(np.isfinite(Xtr), Xtr, pred_means[None, :])

    rf = RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=-1, max_depth=12
    )
    rf.fit(Xtr, ytr)

    out = flat[0].copy()
    need = ~np.isfinite(target) & np.any(np.isfinite(predictors), axis=0)
    if need.any():
        Xpr = predictors[:, need].T
        Xpr = np.where(np.isfinite(Xpr), Xpr, pred_means[None, :])
        out[need] = rf.predict(Xpr)
    # Remaining gaps -> weighted composite.
    still = ~np.isfinite(out)
    out[still] = fallback.reshape(-1)[still]
    return out.reshape(stack.shape[1:])


def _netcdf_safe_attrs(ds):  # noqa: ANN001
    """Return a copy of ``ds`` with NetCDF-incompatible attrs JSON-encoded.

    NetCDF attributes must be scalars / numeric arrays / strings. Lists of dicts
    (e.g. the gap-fill reports) and mapping attrs (provenance codes) are encoded
    to JSON strings so the cube can round-trip through :func:`save_netcdf`.

    Args:
        ds: The dataset to sanitise.

    Returns:
        A shallow copy with sanitised ``attrs`` on the dataset and every var.
    """
    import json

    def _clean(attrs: dict) -> dict:
        out = {}
        for key, val in attrs.items():
            if isinstance(val, (list, tuple, dict)):
                out[key] = json.dumps(val, default=str)
            elif val is None:
                out[key] = "null"
            else:
                out[key] = val
        return out

    safe = ds.copy()
    safe.attrs = _clean(dict(ds.attrs))
    for name in safe.data_vars:
        safe[name].attrs = _clean(dict(safe[name].attrs))
    return safe


def run(cfg):  # noqa: ANN001 - Hydra config
    """Harmonize + gap-fill the synthetic grid cube (DEV_CONTRACT §6.5 entrypoint).

    Loads ``data/processed/grid.nc`` (the synthetic fused cube produced by the
    Demo agent), regrids it onto the configured analysis grid, runs the
    gap-fill cascade (:func:`aqi_india.fusion.gapfill.fill_gaps`), and writes the
    harmonized cube. Returns the in-memory gap-filled Dataset so the demo can
    chain directly into feature building.

    Args:
        cfg: Composed Hydra config (``cfg.grid``, ``cfg.paths.data_processed``).

    Returns:
        The harmonized, gap-filled :class:`xarray.Dataset`.
    """
    from pathlib import Path

    from ..utils.io import load_netcdf, save_netcdf
    from .gapfill import fill_gaps

    processed = Path(getattr(cfg.paths, "data_processed", "data/processed"))
    grid_path = processed / "grid.nc"
    logger.info("Harmonize: loading grid cube from %s", grid_path)
    ds = load_netcdf(grid_path)

    target = target_grid_from_cfg(cfg)
    if (ds["lat"].size, ds["lon"].size) != target.shape:
        logger.info("Regridding cube onto %s %s", target.name, target.shape)
        ds = to_grid(ds, target, method="bilinear")

    logger.info("Running gap-fill cascade on %d data-vars", len(ds.data_vars))
    filled = fill_gaps(ds, method="auto")

    out_path = processed / "grid_fused.nc"
    save_netcdf(_netcdf_safe_attrs(filled), out_path)
    logger.info("Harmonized cube written to %s", out_path)
    return filled


__all__ = [
    "StackResult",
    "DEFAULT_AOD_RELIABILITY",
    "stack_sources",
    "fuse_aod",
    "run",
]
