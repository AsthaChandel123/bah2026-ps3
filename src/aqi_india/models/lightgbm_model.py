"""LightGBM surface-concentration model — the runnable Objective-1 learner.

The PS3 design uses a stacking ensemble (SA-ConvLSTM + ST-GNN + LightGBM) as the
final estimator, but LightGBM is the *primary classical learner* (India
AOD->PM2.5 R2 0.85-0.92) and the model the **light, runnable demo** trusts. This
module wraps one LightGBM regressor per surface pollutant in a single
:class:`MultiPollutantLGBM` estimator that:

* fits one gradient-boosting regressor per pollutant in
  :data:`SURFACE_POLLUTANTS` (``pm25, pm10, no2, so2, co, o3``),
* applies a **monotonic constraint on AOD** (more aerosol optical depth never
  decreases predicted PM/concentration — a physically-motivated prior that
  stabilises extrapolation in station-sparse zones),
* exposes ``fit(X, y_dict)`` / ``predict(X) -> dict`` mirroring the per-pollutant
  contract the train/predict stages consume,
* optionally fits **quantile heads** (alpha 0.05 / 0.5 / 0.95) per pollutant so
  prediction intervals are available without a separate UQ pass, and
* offers a SHAP hook (lazy ``import shap``) and gain-based feature importances
  for driver attribution.

``lightgbm`` and ``shap`` are imported lazily inside methods so the module
imports under the light dependency set. The wrapper follows the scikit-learn
``fit``/``predict`` protocol closely enough to drop straight into
:func:`aqi_india.validation.cv.cross_validate` (per-pollutant single-target use).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_log = get_logger("models.lightgbm")

#: Surface pollutants this product predicts (CPCB NAQI drivers). CO is in mg/m3,
#: the rest in ug/m3. Order is canonical and matches the AQI engine.
SURFACE_POLLUTANTS: tuple[str, ...] = ("pm25", "pm10", "no2", "so2", "co", "o3")

#: Default quantile levels for the interval heads (5th / median / 95th).
DEFAULT_QUANTILES: tuple[float, float, float] = (0.05, 0.5, 0.95)

#: Feature-name substrings that should carry a monotone-increasing constraint.
#: AOD is the canonical example: column optical depth is monotonically related
#: to surface particulate load once met is held fixed.
_MONOTONE_UP_KEYS: tuple[str, ...] = ("aod",)


@dataclass
class LGBMParams:
    """Hyperparameters for a single LightGBM regressor.

    Mirrors the keys in ``conf/model/lightgbm.yaml`` so a Hydra config maps onto
    this dataclass field-for-field via :meth:`from_cfg`.

    Attributes:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage applied to each tree's contribution.
        num_leaves: Maximum leaves per tree (main capacity knob).
        max_depth: Maximum tree depth (``-1`` = unbounded).
        subsample: Row sub-sampling fraction (bagging).
        colsample_bytree: Column sub-sampling fraction per tree.
        min_child_samples: Minimum samples per leaf (regularisation).
        reg_lambda: L2 regularisation weight.
        seed: Random seed for determinism.
    """

    n_estimators: int = 1200
    learning_rate: float = 0.03
    num_leaves: int = 63
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 40
    reg_lambda: float = 0.0
    seed: int = 42

    @classmethod
    def from_cfg(cls, cfg: Any) -> "LGBMParams":
        """Build params from a Hydra ``cfg.model`` node (missing keys -> defaults).

        Args:
            cfg: A config node exposing the LightGBM hyperparameter attributes
                (e.g. the composed ``cfg.model``). Attribute access is tolerant:
                absent keys fall back to the dataclass defaults.

        Returns:
            A populated :class:`LGBMParams`.
        """

        def g(name: str, default: Any) -> Any:
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(name, default)
            return getattr(cfg, name, default)

        return cls(
            n_estimators=int(g("n_estimators", cls.n_estimators)),
            learning_rate=float(g("learning_rate", cls.learning_rate)),
            num_leaves=int(g("num_leaves", cls.num_leaves)),
            max_depth=int(g("max_depth", cls.max_depth)),
            subsample=float(g("subsample", cls.subsample)),
            colsample_bytree=float(g("colsample_bytree", cls.colsample_bytree)),
            min_child_samples=int(g("min_child_samples", cls.min_child_samples)),
            reg_lambda=float(g("reg_lambda", cls.reg_lambda)),
            seed=int(g("seed", cls.seed)),
        )

    def to_lgbm_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments for ``lightgbm.LGBMRegressor``."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "subsample_freq": 1,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
            "random_state": self.seed,
            "n_jobs": -1,
            "verbosity": -1,
        }


def monotone_constraints_for(
    feature_names: "list[str] | tuple[str, ...]",
    *,
    up_keys: "tuple[str, ...]" = _MONOTONE_UP_KEYS,
) -> list[int]:
    """Build a LightGBM ``monotone_constraints`` vector for given features.

    A feature gets ``+1`` (monotone increasing) when its (lower-cased) name
    contains any of ``up_keys`` — so ``aod``, ``aod_pbl`` and ``aod_dry`` all
    inherit the physically-motivated "more aerosol never lowers PM" prior — and
    ``0`` (unconstrained) otherwise.

    Args:
        feature_names: Ordered predictor column names (the columns of ``X``).
        up_keys: Substrings that trigger a monotone-increasing constraint.

    Returns:
        A list of ints in ``{-1, 0, 1}``, one per feature, in column order.
    """
    cons: list[int] = []
    for name in feature_names:
        low = str(name).lower()
        cons.append(1 if any(k in low for k in up_keys) else 0)
    return cons


@dataclass
class MultiPollutantLGBM:
    """One LightGBM regressor per surface pollutant, with optional quantile heads.

    The estimator stores, per pollutant, a fitted point model and (optionally) a
    dict of quantile models keyed by alpha. ``fit`` accepts a ``y_dict`` so a
    single call trains every pollutant that has labels; pollutants absent from
    ``y_dict`` (or all-NaN) are simply skipped. ``predict`` returns a dict of
    point predictions; :meth:`predict_quantiles` returns the interval heads.

    Attributes:
        pollutants: Pollutants to model (defaults to :data:`SURFACE_POLLUTANTS`).
        params: Shared :class:`LGBMParams` for every per-pollutant model.
        quantiles: Quantile levels for the interval heads; ``None`` disables them.
        monotone_up_keys: Feature-name substrings to constrain monotone-up.
        feature_names_: Predictor names captured at fit time (set after ``fit``).
        models_: Fitted point models keyed by pollutant.
        quantile_models_: Nested ``{pollutant: {alpha: model}}`` quantile heads.
    """

    pollutants: tuple[str, ...] = SURFACE_POLLUTANTS
    params: LGBMParams = field(default_factory=LGBMParams)
    quantiles: "tuple[float, ...] | None" = DEFAULT_QUANTILES
    monotone_up_keys: tuple[str, ...] = _MONOTONE_UP_KEYS

    feature_names_: "list[str] | None" = field(default=None, init=False)
    models_: dict[str, Any] = field(default_factory=dict, init=False)
    quantile_models_: dict[str, dict[float, Any]] = field(
        default_factory=dict, init=False
    )

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def _coerce_X(self, X: Any) -> tuple[np.ndarray, list[str]]:
        """Return ``(values, feature_names)`` from an array or DataFrame ``X``."""
        try:
            import pandas as pd
        except Exception:  # pragma: no cover - pandas is in the light set
            pd = None  # type: ignore[assignment]

        if pd is not None and isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64), [str(c) for c in X.columns]
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = self.feature_names_ or [f"f{i}" for i in range(arr.shape[1])]
        return arr, list(names)

    def _model_input(self, X: Any) -> Any:
        """Return a model-ready input, preserving fit-time feature names.

        LightGBM validates feature names when it was fitted on a named
        DataFrame; re-wrapping a bare array with :attr:`feature_names_` keeps the
        column schema consistent and silences the sklearn name-mismatch warning.
        """
        Xv, names = self._coerce_X(X)
        if self.feature_names_ and len(self.feature_names_) == Xv.shape[1]:
            import pandas as pd

            return pd.DataFrame(Xv, columns=self.feature_names_)
        return Xv

    def _make_regressor(self, feature_names: list[str], *, objective: str = "regression",
                        alpha: float | None = None) -> Any:
        """Construct a configured ``LGBMRegressor`` (point or quantile head).

        The monotone-AOD constraint is applied to the point ("regression")
        objective only: LightGBM rejects ``monotone_constraints`` under the
        ``quantile`` objective, so the quantile heads are fitted unconstrained.
        """
        from lightgbm import LGBMRegressor

        kwargs = self.params.to_lgbm_kwargs()
        if objective == "quantile":
            kwargs["objective"] = "quantile"
            kwargs["alpha"] = float(alpha) if alpha is not None else 0.5
            # monotone_constraints is incompatible with the quantile objective.
        else:
            kwargs["objective"] = "regression"
            kwargs["monotone_constraints"] = monotone_constraints_for(
                feature_names, up_keys=self.monotone_up_keys
            )
        return LGBMRegressor(**kwargs)

    def fit(
        self,
        X: Any,
        y_dict: "dict[str, Any] | Any",
        *,
        fit_quantiles: bool = True,
    ) -> "MultiPollutantLGBM":
        """Fit one model per pollutant present in ``y_dict``.

        Rows with a NaN label for a given pollutant are dropped from *that*
        pollutant's training set only (so a station-day missing SO2 still trains
        the PM2.5 head). A pollutant with no finite labels is skipped.

        Args:
            X: Predictor matrix (``pandas.DataFrame`` or 2-D array). Column names
                are captured for the monotone constraints and SHAP.
            y_dict: Either a ``{pollutant: y}`` mapping or, when this estimator
                models a single pollutant, a bare target array (treated as the
                first entry of :attr:`pollutants`).
            fit_quantiles: Whether to also fit the quantile heads when
                :attr:`quantiles` is set.

        Returns:
            ``self`` (fitted), with :attr:`models_` populated.
        """
        Xv, names = self._coerce_X(X)
        self.feature_names_ = names

        # Allow a bare array for the single-pollutant case (sklearn-style).
        if not isinstance(y_dict, dict):
            y_dict = {self.pollutants[0]: y_dict}

        for pol in self.pollutants:
            if pol not in y_dict or y_dict[pol] is None:
                continue
            y = np.asarray(y_dict[pol], dtype=np.float64).ravel()
            mask = np.isfinite(y) & np.isfinite(Xv).all(axis=1)
            if mask.sum() < 2:
                _log.warning("Skipping %s: only %d finite samples.", pol, int(mask.sum()))
                continue
            Xp, yp = Xv[mask], y[mask]

            point = self._make_regressor(names)
            point.fit(Xp, yp)
            self.models_[pol] = point

            if fit_quantiles and self.quantiles:
                self.quantile_models_[pol] = {}
                for alpha in self.quantiles:
                    qm = self._make_regressor(names, objective="quantile", alpha=alpha)
                    qm.fit(Xp, yp)
                    self.quantile_models_[pol][float(alpha)] = qm

            _log.info("Fitted LightGBM for %s on %d samples.", pol, int(mask.sum()))

        if not self.models_:
            raise ValueError(
                "No pollutant models were fitted; y_dict had no usable labels for "
                f"{self.pollutants}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, X: Any) -> dict[str, np.ndarray]:
        """Predict point surface concentrations for every fitted pollutant.

        Args:
            X: Predictor matrix matching the fit-time feature columns.

        Returns:
            ``{pollutant: prediction_array}`` for each fitted pollutant. CO is in
            mg/m3, the rest in ug/m3 (predictions are clipped at 0).
        """
        Xin = self._model_input(X)
        out: dict[str, np.ndarray] = {}
        for pol, model in self.models_.items():
            pred = np.asarray(model.predict(Xin), dtype=np.float64).ravel()
            out[pol] = np.clip(pred, 0.0, None)
        return out

    def predict_quantiles(self, X: Any) -> dict[str, dict[float, np.ndarray]]:
        """Predict quantile heads per pollutant for prediction intervals.

        Quantile crossing (a higher alpha predicting a lower value than a lower
        alpha) is repaired by sorting the per-row quantile predictions, so the
        returned interval is always well-ordered.

        Args:
            X: Predictor matrix matching the fit-time feature columns.

        Returns:
            ``{pollutant: {alpha: prediction_array}}`` for each pollutant that
            has quantile heads.
        """
        Xin = self._model_input(X)
        out: dict[str, dict[float, np.ndarray]] = {}
        for pol, qmodels in self.quantile_models_.items():
            alphas = sorted(qmodels)
            preds = np.column_stack(
                [np.asarray(qmodels[a].predict(Xin), dtype=np.float64) for a in alphas]
            )
            preds = np.clip(preds, 0.0, None)
            preds = np.sort(preds, axis=1)  # enforce monotone quantiles
            out[pol] = {a: preds[:, i] for i, a in enumerate(alphas)}
        return out

    # ------------------------------------------------------------------ #
    # Interpretation
    # ------------------------------------------------------------------ #
    def feature_importances(self, *, importance_type: str = "gain") -> "pd.DataFrame":
        """Return a tidy gain/split importance table across pollutants.

        Args:
            importance_type: ``"gain"`` (default) or ``"split"`` — forwarded to
                LightGBM's booster importance.

        Returns:
            A :class:`pandas.DataFrame` indexed by feature name with one column
            per fitted pollutant of importance values (gain-normalised columns
            sum to 1 within each pollutant).
        """
        import pandas as pd

        if not self.models_:
            raise RuntimeError("Call fit() before feature_importances().")
        names = self.feature_names_ or []
        cols: dict[str, np.ndarray] = {}
        for pol, model in self.models_.items():
            booster = model.booster_
            imp = np.asarray(
                booster.feature_importance(importance_type=importance_type),
                dtype=np.float64,
            )
            total = imp.sum()
            cols[pol] = imp / total if total > 0 else imp
        return pd.DataFrame(cols, index=names).sort_index()

    def shap_values(
        self, X: Any, pollutant: str, *, max_samples: int | None = 2000
    ) -> "np.ndarray":
        """Compute SHAP values for one pollutant's model (lazy ``import shap``).

        SHAP is an optional heavy dependency, imported here so the module stays
        light. A ``TreeExplainer`` is used (exact for tree ensembles).

        Args:
            X: Predictor matrix matching the fit-time feature columns.
            pollutant: Which fitted pollutant model to explain.
            max_samples: If set, randomly subsample at most this many rows of
                ``X`` for tractable SHAP computation.

        Returns:
            A SHAP value array of shape ``(n_used, n_features)``.

        Raises:
            KeyError: If ``pollutant`` has no fitted model.
            ImportError: If the ``shap`` package is not installed.
        """
        if pollutant not in self.models_:
            raise KeyError(f"No fitted model for pollutant {pollutant!r}")
        import shap  # lazy: optional heavy dependency

        Xv, _names = self._coerce_X(X)
        if max_samples is not None and Xv.shape[0] > max_samples:
            rng = np.random.default_rng(self.params.seed)
            sel = rng.choice(Xv.shape[0], size=max_samples, replace=False)
            Xv = Xv[sel]
        explainer = shap.TreeExplainer(self.models_[pollutant])
        values = explainer.shap_values(Xv)
        return np.asarray(values)

    @property
    def fitted_pollutants(self) -> tuple[str, ...]:
        """Pollutants that have a fitted point model, in canonical order."""
        return tuple(p for p in self.pollutants if p in self.models_)


def make_single_pollutant_factory(
    pollutant: str,
    params: "LGBMParams | None" = None,
    *,
    feature_names: "list[str] | None" = None,
    quantiles: "tuple[float, ...] | None" = None,
) -> Any:
    """Return a zero-arg factory yielding a single-pollutant LightGBM estimator.

    This adapts :class:`MultiPollutantLGBM` to the ``model_factory`` protocol
    used by :func:`aqi_india.validation.cv.cross_validate`, which fits with a
    bare ``y`` array. The returned estimator's ``predict`` yields a plain array
    (not a dict) so it slots into the CV driver unchanged.

    Args:
        pollutant: The pollutant the factory's models target.
        params: Optional shared hyperparameters.
        feature_names: Optional fixed feature names (for array inputs).
        quantiles: Optional quantile levels (disabled in CV by default).

    Returns:
        A zero-argument callable returning a fresh estimator with
        scikit-learn-style ``fit(X, y)`` / ``predict(X) -> np.ndarray``.
    """

    class _SinglePollutantLGBM(MultiPollutantLGBM):
        def fit(self, X: Any, y: Any, **kw: Any) -> "_SinglePollutantLGBM":  # type: ignore[override]
            if feature_names is not None and self.feature_names_ is None:
                self.feature_names_ = list(feature_names)
            super().fit(X, {pollutant: y}, fit_quantiles=bool(quantiles))
            return self

        def predict(self, X: Any) -> np.ndarray:  # type: ignore[override]
            return super().predict(X)[pollutant]

    def factory() -> _SinglePollutantLGBM:
        return _SinglePollutantLGBM(
            pollutants=(pollutant,),
            params=params or LGBMParams(),
            quantiles=quantiles,
        )

    return factory


__all__ = [
    "SURFACE_POLLUTANTS",
    "DEFAULT_QUANTILES",
    "LGBMParams",
    "MultiPollutantLGBM",
    "monotone_constraints_for",
    "make_single_pollutant_factory",
]
