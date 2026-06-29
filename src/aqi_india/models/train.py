"""Objective-1 training driver — fit the MultiPollutantLGBM + run the CV ladder.

This module wires the runnable model layer together:

1. load the engineered feature matrix (the tidy station-day table produced by
   :func:`aqi_india.features.feature_matrix.build`),
2. select predictors (via :func:`features.feature_matrix.feature_columns`) and
   the per-pollutant labels,
3. run the **CV ladder** (random -> leave-station-out -> spatiotemporal-blocked)
   per pollutant via :mod:`aqi_india.validation.cv`, surfacing the leakage gap,
4. fit a final :class:`~aqi_india.models.lightgbm_model.MultiPollutantLGBM` on
   all data (point + quantile heads), and
5. persist the trained model + a tidy metrics table under ``data/models/``.

:func:`train_obj1` is the importable workhorse; :func:`run` is the Hydra
chaining hook (``aqi_india.models.train.run(cfg)``) the Demo agent calls and the
``aqi train lightgbm`` CLI command targets. All heavy/learner imports stay lazy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..features.feature_matrix import feature_columns
from ..utils.io import save_parquet
from ..utils.logging import get_logger
from .lightgbm_model import (
    SURFACE_POLLUTANTS,
    LGBMParams,
    MultiPollutantLGBM,
    make_single_pollutant_factory,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_log = get_logger("models.train")

#: Default CV rungs to evaluate during training (a representative ladder slice).
DEFAULT_CV_LADDER: tuple[str, ...] = (
    "random_kfold",
    "leave_station_out",
    "spatiotemporal_blocked_cv",
)


def _label_arrays(matrix: "pd.DataFrame") -> dict[str, np.ndarray]:
    """Extract the per-pollutant label arrays present in the feature matrix."""
    return {
        pol: matrix[pol].to_numpy(dtype=np.float64)
        for pol in SURFACE_POLLUTANTS
        if pol in matrix.columns
    }


def _cv_metadata(matrix: "pd.DataFrame") -> dict[str, Any]:
    """Pull the grouping columns the CV ladder needs from the feature matrix."""
    return {
        "groups": matrix["station_id"].to_numpy() if "station_id" in matrix else None,
        "times": matrix["time"].to_numpy() if "time" in matrix else None,
        "lats": matrix["lat"].to_numpy() if "lat" in matrix else None,
        "lons": matrix["lon"].to_numpy() if "lon" in matrix else None,
    }


def run_cv_for_pollutant(
    matrix: "pd.DataFrame",
    pollutant: str,
    feature_cols: "list[str]",
    *,
    params: "LGBMParams | None" = None,
    schemes: "tuple[str, ...]" = DEFAULT_CV_LADDER,
) -> "pd.DataFrame":
    """Run the CV ladder for a single pollutant and return its leakage table.

    Rows with a missing label or any missing predictor are dropped first (the
    CV driver expects clean arrays). The LightGBM single-pollutant factory is
    used so the monotone-AOD constraint is honoured inside every fold.

    Args:
        matrix: The engineered feature matrix.
        pollutant: The pollutant to validate.
        feature_cols: Predictor column names.
        params: Optional LightGBM hyperparameters.
        schemes: CV ladder rungs to evaluate.

    Returns:
        The :func:`aqi_india.validation.cv.run_cv_ladder` comparison table with a
        ``pollutant`` column prepended (empty if too few clean rows).
    """
    import pandas as pd

    from ..validation.cv import run_cv_ladder

    sub = matrix.dropna(subset=[pollutant, *feature_cols])
    if len(sub) < 10:
        _log.warning("Too few clean rows for %s (%d); skipping CV.", pollutant, len(sub))
        return pd.DataFrame()

    X = sub[feature_cols]
    y = sub[pollutant].to_numpy(dtype=np.float64)
    meta = _cv_metadata(sub)
    factory = make_single_pollutant_factory(
        pollutant, params=params, feature_names=list(feature_cols)
    )
    table = run_cv_ladder(
        factory,
        X,
        y,
        groups=meta["groups"],
        times=meta["times"],
        lats=meta["lats"],
        lons=meta["lons"],
        schemes=schemes,
    )
    table.insert(0, "pollutant", pollutant)
    return table


def train_obj1(
    feature_matrix_path: "str | Path | pd.DataFrame",
    *,
    cv_scheme: str = "leave_station_out",
    params: "LGBMParams | None" = None,
    run_ladder: bool = True,
    models_dir: "str | Path" = "data/models",
    model_name: str = "lightgbm",
    seed: int = 42,
) -> dict[str, Any]:
    """Train the Objective-1 surface model and run the CV ladder.

    Fits one LightGBM regressor per surface pollutant (point + quantile heads)
    with the monotone-AOD constraint, evaluates the CV ladder per pollutant to
    expose the spatial/temporal leakage gap, and saves the model + metrics.

    Args:
        feature_matrix_path: Path to a feature-matrix parquet, or an in-memory
            :class:`pandas.DataFrame` already in that schema.
        cv_scheme: The headline CV scheme reported as the model's honest skill
            (default ``"leave_station_out"``). It is always included in the
            evaluated ladder.
        params: Optional LightGBM hyperparameters (defaults to
            :class:`LGBMParams`).
        run_ladder: Whether to run the full CV ladder (set ``False`` to skip CV
            and only fit the final model — faster for tiny demos).
        models_dir: Directory to persist the trained model + metrics under.
        model_name: Sub-directory / artifact name for this model.
        seed: Random seed threaded into the hyperparameters.

    Returns:
        Dict with:
            ``model`` — the fitted :class:`MultiPollutantLGBM`,
            ``metrics`` — the concatenated per-pollutant CV ladder table,
            ``model_path`` — where the model was written,
            ``metrics_path`` — where the metrics table was written,
            ``fitted_pollutants`` — the pollutants that trained successfully.
    """
    import pandas as pd

    if isinstance(feature_matrix_path, pd.DataFrame):
        matrix = feature_matrix_path
    else:
        from ..utils.io import load_parquet

        matrix = load_parquet(Path(feature_matrix_path))
    _log.info("Loaded feature matrix: %d rows x %d cols.", len(matrix), matrix.shape[1])

    params = params or LGBMParams(seed=seed)
    feature_cols = feature_columns(matrix)
    if not feature_cols:
        raise ValueError("Feature matrix has no usable predictor columns.")
    labels = _label_arrays(matrix)
    if not labels:
        raise ValueError(
            f"Feature matrix has none of the label columns {SURFACE_POLLUTANTS}."
        )

    # --- CV ladder per pollutant -------------------------------------------- #
    metrics_tables: list[pd.DataFrame] = []
    if run_ladder:
        schemes = tuple(dict.fromkeys((*DEFAULT_CV_LADDER, cv_scheme)))
        for pol in labels:
            tbl = run_cv_for_pollutant(
                matrix, pol, feature_cols, params=params, schemes=schemes
            )
            if not tbl.empty:
                metrics_tables.append(tbl)
    metrics = (
        pd.concat(metrics_tables, ignore_index=True)
        if metrics_tables
        else pd.DataFrame()
    )

    # --- Final fit on all data (point + quantile heads) --------------------- #
    model = MultiPollutantLGBM(
        pollutants=tuple(labels.keys()), params=params
    )
    X_all = matrix[feature_cols]
    model.fit(X_all, labels, fit_quantiles=True)
    _log.info("Final model fitted for pollutants: %s", model.fitted_pollutants)

    # --- Persist ------------------------------------------------------------ #
    out_dir = Path(models_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    _save_model(model, model_path, feature_cols)
    metrics_path = out_dir / "cv_metrics.parquet"
    if not metrics.empty:
        save_parquet(metrics, metrics_path)
    _log.info("Saved model -> %s ; metrics -> %s", model_path, metrics_path)

    return {
        "model": model,
        "metrics": metrics,
        "model_path": model_path,
        "metrics_path": metrics_path if not metrics.empty else None,
        "fitted_pollutants": model.fitted_pollutants,
        "feature_cols": feature_cols,
    }


def _save_model(
    model: MultiPollutantLGBM, path: "str | Path", feature_cols: "list[str]"
) -> Path:
    """Persist a fitted model + its feature schema via joblib.

    joblib is pulled in transitively by scikit-learn (light set). The feature
    column order is stored alongside so :func:`aqi_india.models.predict` can
    assemble inference matrices in the exact training order.

    Args:
        model: The fitted estimator.
        path: Destination ``.joblib`` path.
        feature_cols: Ordered predictor column names used at fit time.

    Returns:
        The path written.
    """
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": list(feature_cols)}, p)
    return p


def load_model(path: "str | Path") -> dict[str, Any]:
    """Load a model bundle saved by :func:`_save_model`.

    Args:
        path: Path to the ``.joblib`` bundle.

    Returns:
        Dict with ``model`` (the :class:`MultiPollutantLGBM`) and
        ``feature_cols`` (the training-time predictor order).
    """
    import joblib

    return joblib.load(Path(path))


def run(cfg: Any) -> str:
    """Hydra entry point: train Objective-1 from the composed config.

    Implements the chaining contract ``aqi_india.models.train.run(cfg)``. Reads
    the feature matrix from ``cfg.paths.data_processed``, maps the ``cfg.model``
    hyperparameters onto :class:`LGBMParams`, trains and persists the model under
    ``cfg.paths.models``, and returns the saved model path (as a string).

    Args:
        cfg: The composed Hydra config. Uses ``cfg.paths.{data_processed,models}``,
            ``cfg.project.seed`` and the ``cfg.model`` hyperparameters when
            present; all are accessed defensively with sensible defaults.

    Returns:
        The filesystem path of the trained model artifact, as a string.
    """
    processed = Path(_cfg_get(cfg, "paths", "data_processed", default="data/processed"))
    models_dir = Path(_cfg_get(cfg, "paths", "models", default="data/models"))
    seed = int(_cfg_get(cfg, "project", "seed", default=42))
    model_cfg = _cfg_get(cfg, "model", default=None)
    model_name = str(_cfg_get(cfg, "model", "name", default="lightgbm"))

    params = LGBMParams.from_cfg(model_cfg)
    params.seed = seed

    matrix_path = processed / "feature_matrix.parquet"
    if not matrix_path.exists():
        # Fall back to building the matrix from upstream artifacts on the fly.
        _log.info("feature_matrix.parquet absent; building it from processed data.")
        from ..features.feature_matrix import build as build_matrix

        build_matrix(cfg)

    result = train_obj1(
        matrix_path,
        cv_scheme=str(_cfg_get(cfg, "model", "cv_scheme", default="leave_station_out")),
        params=params,
        models_dir=models_dir,
        model_name=model_name,
        seed=seed,
    )
    return str(result["model_path"])


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse a nested Hydra/dict config, returning ``default`` on miss."""
    node: Any = cfg
    for key in keys:
        if node is None:
            return default
        node = node.get(key, None) if isinstance(node, dict) else getattr(node, key, None)
    return node if node is not None else default


__all__ = [
    "DEFAULT_CV_LADDER",
    "train_obj1",
    "run_cv_for_pollutant",
    "load_model",
    "run",
]
