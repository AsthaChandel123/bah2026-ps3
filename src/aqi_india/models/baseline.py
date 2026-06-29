"""Simple baseline learners for the CV-ladder comparison.

The headline validation deliverable is the **leakage gap** between random and
spatiotemporally-blocked CV (see :mod:`aqi_india.validation.cv`). To make that
story honest you also want a baseline-skill anchor: a model so simple it cannot
overfit the autocorrelation, against which the LightGBM/deep models must show a
real lift. This module provides two such anchors, each a multi-pollutant wrapper
mirroring :class:`aqi_india.models.lightgbm_model.MultiPollutantLGBM`'s
``fit(X, y_dict)`` / ``predict(X) -> dict`` contract so they drop into the same
training and CV plumbing:

* :class:`LinearBaseline` — a standardised ridge regression (the cheap,
  interpretable LME/LUR-style anchor).
* :class:`RandomForestBaseline` — a bagged tree ensemble (the robust, strong
  classical baseline; IGP 16-city RF R2 ~0.94 in the literature), which also
  exposes quantile intervals via the empirical spread of its trees.

scikit-learn is in the light dependency set, so these import eagerly; nothing
heavy or credentialed is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.logging import get_logger
from .lightgbm_model import SURFACE_POLLUTANTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_log = get_logger("models.baseline")


def _coerce_X(X: Any) -> np.ndarray:
    """Return a 2-D float array from a DataFrame or array-like ``X``."""
    try:
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64)
    except Exception:  # pragma: no cover - pandas is in the light set
        pass
    arr = np.asarray(X, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _clean_xy(Xv: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_clean, y_clean, mask)`` dropping rows with NaN in X or y."""
    mask = np.isfinite(y) & np.isfinite(Xv).all(axis=1)
    return Xv[mask], y[mask], mask


@dataclass
class _MultiTargetBase:
    """Shared multi-pollutant fit/predict scaffolding for the baselines."""

    pollutants: tuple[str, ...] = SURFACE_POLLUTANTS
    seed: int = 42
    models_: dict[str, Any] = field(default_factory=dict, init=False)
    feature_names_: "list[str] | None" = field(default=None, init=False)

    def _new_estimator(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: Any, y_dict: "dict[str, Any] | Any") -> "_MultiTargetBase":
        """Fit one estimator per pollutant present in ``y_dict``.

        Args:
            X: Predictor matrix (DataFrame or array).
            y_dict: ``{pollutant: y}`` mapping, or a bare array for the
                single-pollutant case.

        Returns:
            ``self`` with :attr:`models_` populated.
        """
        try:
            import pandas as pd

            if isinstance(X, pd.DataFrame):
                self.feature_names_ = [str(c) for c in X.columns]
        except Exception:  # pragma: no cover
            pass
        Xv = _coerce_X(X)

        if not isinstance(y_dict, dict):
            y_dict = {self.pollutants[0]: y_dict}

        for pol in self.pollutants:
            if pol not in y_dict or y_dict[pol] is None:
                continue
            y = np.asarray(y_dict[pol], dtype=np.float64).ravel()
            Xc, yc, _ = _clean_xy(Xv, y)
            if Xc.shape[0] < 2:
                _log.warning("Skipping %s: %d finite samples.", pol, Xc.shape[0])
                continue
            est = self._new_estimator()
            est.fit(Xc, yc)
            self.models_[pol] = est
            _log.info("Fitted %s baseline for %s on %d samples.",
                      type(self).__name__, pol, Xc.shape[0])
        if not self.models_:
            raise ValueError(f"No models fitted; y_dict had no labels for {self.pollutants}")
        return self

    def predict(self, X: Any) -> dict[str, np.ndarray]:
        """Predict per-pollutant point concentrations (clipped at 0)."""
        Xv = _coerce_X(X)
        return {
            pol: np.clip(np.asarray(m.predict(Xv), dtype=np.float64).ravel(), 0.0, None)
            for pol, m in self.models_.items()
        }

    @property
    def fitted_pollutants(self) -> tuple[str, ...]:
        """Pollutants with a fitted model, in canonical order."""
        return tuple(p for p in self.pollutants if p in self.models_)


@dataclass
class LinearBaseline(_MultiTargetBase):
    """Standardised ridge regression baseline (cheap, interpretable anchor).

    Each pollutant gets a :class:`sklearn.pipeline.Pipeline` of a median
    imputer -> standard scaler -> ridge regressor. The L2 penalty keeps the fit
    stable under the collinear satellite/met features.

    Attributes:
        alpha: Ridge regularisation strength.
    """

    alpha: float = 1.0

    def _new_estimator(self) -> Any:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=self.seed)),
            ]
        )


@dataclass
class RandomForestBaseline(_MultiTargetBase):
    """Random-forest baseline — robust bagged trees with quantile intervals.

    A strong, decorrelated classical member: it captures the nonlinear
    AOD->surface relationship without the tuning a boosted model needs, and its
    tree spread yields cheap empirical prediction intervals via
    :meth:`predict_quantiles`.

    Attributes:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (``None`` = unbounded).
        min_samples_leaf: Minimum samples per leaf (regularisation).
    """

    n_estimators: int = 400
    max_depth: "int | None" = None
    min_samples_leaf: int = 3

    def _new_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        n_jobs=-1,
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    def predict_quantiles(
        self, X: Any, quantiles: "tuple[float, ...]" = (0.05, 0.5, 0.95)
    ) -> dict[str, dict[float, np.ndarray]]:
        """Empirical prediction quantiles from the per-tree predictions.

        For each pollutant, every tree in the forest predicts ``X``; the
        requested quantiles of that per-row ensemble distribution form the
        interval. This is the quantile-RF intervals referenced in the design.

        Args:
            X: Predictor matrix matching the fit-time columns.
            quantiles: Quantile levels to evaluate.

        Returns:
            ``{pollutant: {alpha: array}}`` of empirical quantile predictions.
        """
        Xv = _coerce_X(X)
        out: dict[str, dict[float, np.ndarray]] = {}
        for pol, pipe in self.models_.items():
            # Pull the fitted RF out of the pipeline and impute X the same way.
            rf = pipe.named_steps["rf"]
            Xi = pipe.named_steps["impute"].transform(Xv)
            # (n_trees, n_samples)
            per_tree = np.stack([t.predict(Xi) for t in rf.estimators_], axis=0)
            q = np.quantile(per_tree, list(quantiles), axis=0)  # (n_q, n_samples)
            q = np.clip(q, 0.0, None)
            out[pol] = {float(a): q[i] for i, a in enumerate(quantiles)}
        return out


def make_single_pollutant_baseline_factory(
    kind: str,
    pollutant: str,
    *,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Return a zero-arg factory for a single-pollutant baseline (CV-ready).

    Adapts the multi-pollutant baselines to the ``model_factory`` protocol of
    :func:`aqi_india.validation.cv.cross_validate` (bare ``y`` in, plain array
    out), so the ladder can be run for a baseline anchor.

    Args:
        kind: ``"linear"`` or ``"rf"``.
        pollutant: The pollutant the models target.
        seed: Random seed.
        **kwargs: Forwarded to the baseline constructor.

    Returns:
        A zero-argument callable returning an estimator with
        ``fit(X, y)`` / ``predict(X) -> np.ndarray``.

    Raises:
        ValueError: If ``kind`` is not recognised.
    """
    kind = kind.lower()
    if kind == "linear":
        base_cls: type[_MultiTargetBase] = LinearBaseline
    elif kind in ("rf", "randomforest", "random_forest"):
        base_cls = RandomForestBaseline
    else:
        raise ValueError(f"Unknown baseline kind {kind!r}; use 'linear' or 'rf'")

    class _Single(base_cls):  # type: ignore[valid-type, misc]
        def fit(self, X: Any, y: Any, **kw: Any) -> "_Single":  # type: ignore[override]
            super().fit(X, {pollutant: y})
            return self

        def predict(self, X: Any) -> np.ndarray:  # type: ignore[override]
            return super().predict(X)[pollutant]

    def factory() -> "_Single":
        return _Single(pollutants=(pollutant,), seed=seed, **kwargs)

    return factory


__all__ = [
    "LinearBaseline",
    "RandomForestBaseline",
    "make_single_pollutant_baseline_factory",
]
