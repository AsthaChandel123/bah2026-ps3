"""Objective-1 inference — predict surface grids -> daily AQI grid + uncertainty.

Given a trained :class:`~aqi_india.models.lightgbm_model.MultiPollutantLGBM` and
the full-grid feature stack from
:func:`aqi_india.features.feature_matrix.build_grid_features`, this module:

1. flattens the gridded features into the tidy predictor matrix the tabular
   model expects (one row per ``(time, lat, lon)`` cell, columns in training
   order),
2. predicts every surface pollutant and reshapes back to gridded data-vars named
   with the canonical ``POLLUTANTS`` keys (``pm25``..``o3``) — exactly the schema
   :func:`aqi_india.aqi.naqi.compute_aqi_grid` consumes,
3. computes per-pollutant **uncertainty grids** from the quantile heads (the 90%
   interval half-width), and
4. runs the deterministic NAQI engine to produce the daily AQI grid + the
   responsible-pollutant code grid, saving everything under
   ``data/processed`` / ``data/artifacts``.

:func:`predict_grid` is the importable workhorse; :func:`run` is the Hydra
chaining hook (``aqi_india.models.predict.run(cfg) -> xr.Dataset``) returning the
predicted surface-concentration Dataset (pollutant-keyed) for direct feeding into
``compute_aqi_grid``. Heavy imports (xarray) are lazy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..features.feature_matrix import build_grid_features
from ..utils.logging import get_logger
from .lightgbm_model import SURFACE_POLLUTANTS, MultiPollutantLGBM

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

_log = get_logger("models.predict")


def _grid_feature_table(
    feat_grid: "xr.Dataset", feature_cols: "list[str]"
) -> tuple[np.ndarray, tuple[int, int, int], Any]:
    """Flatten gridded features to a ``(n_cells, n_features)`` matrix.

    Each predictor column is broadcast to the full ``(time, lat, lon)`` shape
    (static and time-only features are broadcast across the missing dims), then
    flattened row-major. Missing predictor columns are filled with NaN so the
    model's internal handling applies uniformly.

    Args:
        feat_grid: The full-grid feature Dataset from
            :func:`features.feature_matrix.build_grid_features`.
        feature_cols: Predictor names in training order.

    Returns:
        ``(X, shape, coords)`` where ``X`` is the flattened predictor matrix,
        ``shape`` is ``(n_time, n_lat, n_lon)`` and ``coords`` is the broadcast
        template DataArray carrying the output coordinates/dims.
    """
    import xarray as xr

    nt = feat_grid.sizes["time"]
    nlat = feat_grid.sizes["lat"]
    nlon = feat_grid.sizes["lon"]
    template = xr.DataArray(
        np.empty((nt, nlat, nlon), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": feat_grid["time"],
            "lat": feat_grid["lat"],
            "lon": feat_grid["lon"],
        },
    )

    cols: list[np.ndarray] = []
    for name in feature_cols:
        if name in feat_grid.data_vars:
            da = feat_grid[name].broadcast_like(template).transpose("time", "lat", "lon")
            cols.append(np.asarray(da.values, dtype=np.float64).reshape(-1))
        else:
            # Predictor absent on the grid (e.g. lag/fire columns) -> NaN.
            cols.append(np.full(nt * nlat * nlon, np.nan, dtype=np.float64))
    X = np.column_stack(cols) if cols else np.empty((nt * nlat * nlon, 0))
    return X, (nt, nlat, nlon), template


def _reshape_to_grid(
    values: np.ndarray, shape: tuple[int, int, int], template: Any, name: str
) -> "xr.DataArray":
    """Reshape a flat prediction vector back to a named gridded DataArray."""
    import xarray as xr

    arr = np.asarray(values, dtype=np.float32).reshape(shape)
    return xr.DataArray(
        arr,
        dims=("time", "lat", "lon"),
        coords=template.coords,
        name=name,
    )


def predict_grid(
    model: MultiPollutantLGBM,
    grid_features: "xr.Dataset",
    *,
    feature_cols: "list[str] | None" = None,
    with_uncertainty: bool = True,
) -> "xr.Dataset":
    """Predict per-pollutant surface concentration grids from a feature stack.

    Args:
        model: A fitted :class:`MultiPollutantLGBM`.
        grid_features: The full-grid feature Dataset (already augmented with the
            physics/temporal channels — pass the output of
            :func:`features.feature_matrix.build_grid_features`, or a raw cube
            which will be augmented here).
        feature_cols: Predictor order used at training time; defaults to the
            model's captured ``feature_names_``.
        with_uncertainty: If True and the model has quantile heads, also emit a
            ``<pollutant>_unc`` half-width grid per pollutant.

    Returns:
        An :class:`xarray.Dataset` with one ``float32`` data-var per fitted
        pollutant (canonical keys ``pm25``..``o3``, dims ``(time, lat, lon)``),
        plus optional ``<pollutant>_unc`` uncertainty grids — ready to feed
        :func:`aqi_india.aqi.naqi.compute_aqi_grid`.
    """
    import xarray as xr

    # Ensure the engineered channels exist on the grid.
    if "aod_pbl" not in grid_features.data_vars and "aod" in grid_features.data_vars:
        grid_features = build_grid_features(grid_features)

    cols = feature_cols or model.feature_names_
    if not cols:
        raise ValueError(
            "feature_cols unknown: pass feature_cols or use a model fitted on a "
            "named DataFrame."
        )

    X, shape, template = _grid_feature_table(grid_features, cols)

    out = xr.Dataset(coords=template.coords)
    point = model.predict(X)
    for pol in SURFACE_POLLUTANTS:
        if pol in point:
            out[pol] = _reshape_to_grid(point[pol], shape, template, pol)

    if with_uncertainty and model.quantile_models_:
        quant = model.predict_quantiles(X)
        for pol, qd in quant.items():
            alphas = sorted(qd)
            half_width = 0.5 * (qd[alphas[-1]] - qd[alphas[0]])
            out[f"{pol}_unc"] = _reshape_to_grid(
                half_width, shape, template, f"{pol}_unc"
            )

    out.attrs["model"] = "aqi_india.models.lightgbm_model.MultiPollutantLGBM"
    out.attrs["predicted_pollutants"] = ",".join(model.fitted_pollutants)
    _log.info(
        "Predicted surface grids for %s over shape %s.",
        model.fitted_pollutants,
        shape,
    )
    return out


def predict_aqi_grid(
    model: MultiPollutantLGBM,
    grid_features: "xr.Dataset",
    *,
    feature_cols: "list[str] | None" = None,
) -> "xr.Dataset":
    """Predict surface grids and fold them into the daily AQI grid.

    Convenience wrapper that runs :func:`predict_grid` then the deterministic
    NAQI engine, merging the AQI + responsible-pollutant code fields and any
    uncertainty grids into one Dataset.

    Args:
        model: A fitted :class:`MultiPollutantLGBM`.
        grid_features: The full-grid feature Dataset.
        feature_cols: Predictor order (defaults to the model's captured names).

    Returns:
        A Dataset with the predicted pollutant grids, the ``aqi`` field, the
        ``aqi_responsible`` code field and the uncertainty grids.
    """
    from ..aqi.naqi import compute_aqi_grid

    surface = predict_grid(
        model, grid_features, feature_cols=feature_cols, with_uncertainty=True
    )
    pollutant_vars = {p: p for p in SURFACE_POLLUTANTS if p in surface.data_vars}
    aqi = compute_aqi_grid(surface[list(pollutant_vars)])
    return surface.assign(aqi=aqi["aqi"], aqi_responsible=aqi["aqi_responsible"])


def run(cfg: Any) -> "xr.Dataset":
    """Hydra entry point: predict surface grids from the composed config.

    Implements the chaining contract ``aqi_india.models.predict.run(cfg) ->
    xr.Dataset`` whose pollutant data-vars are the canonical ``POLLUTANTS`` keys,
    so the result feeds straight into ``compute_aqi_grid``. It loads the fused
    grid + the trained model, builds the grid feature stack, predicts, computes
    the AQI grid, writes ``aqi_grid.nc`` (+ a surface-concentration artifact) and
    returns the predicted **surface-concentration** Dataset.

    Args:
        cfg: The composed Hydra config. Uses
            ``cfg.paths.{data_processed,models,artifacts}`` and ``cfg.model.name``;
            all accessed defensively.

    Returns:
        The predicted surface-concentration :class:`xarray.Dataset`
        (pollutant-keyed).
    """
    from ..aqi.naqi import compute_aqi_grid
    from ..utils.io import load_netcdf, save_netcdf
    from .train import load_model

    processed = Path(_cfg_get(cfg, "paths", "data_processed", default="data/processed"))
    models_dir = Path(_cfg_get(cfg, "paths", "models", default="data/models"))
    artifacts = Path(_cfg_get(cfg, "paths", "artifacts", default="data/artifacts"))
    model_name = str(_cfg_get(cfg, "model", "name", default="lightgbm"))

    grid = load_netcdf(processed / "grid.nc")
    bundle = load_model(models_dir / model_name / "model.joblib")
    model: MultiPollutantLGBM = bundle["model"]
    feature_cols = bundle.get("feature_cols")

    feat_grid = build_grid_features(grid)
    surface = predict_grid(model, feat_grid, feature_cols=feature_cols)

    # Daily AQI grid from the predicted surface concentrations.
    pollutant_vars = [p for p in SURFACE_POLLUTANTS if p in surface.data_vars]
    aqi = compute_aqi_grid(surface[pollutant_vars])
    aqi_out = surface.assign(aqi=aqi["aqi"], aqi_responsible=aqi["aqi_responsible"])

    save_netcdf(surface, processed / "surface_pred.nc")
    save_netcdf(aqi_out, processed / "aqi_grid.nc")
    artifacts.mkdir(parents=True, exist_ok=True)
    _log.info("Wrote surface_pred.nc and aqi_grid.nc under %s.", processed)
    return surface


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse a nested Hydra/dict config, returning ``default`` on miss."""
    node: Any = cfg
    for key in keys:
        if node is None:
            return default
        node = node.get(key, None) if isinstance(node, dict) else getattr(node, key, None)
    return node if node is not None else default


__all__ = [
    "predict_grid",
    "predict_aqi_grid",
    "run",
]
