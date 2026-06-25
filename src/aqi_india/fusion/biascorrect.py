"""Bias correction of satellite / reanalysis fields against CPCB ground truth.

Two complementary, sequential corrections (synthesis blueprint Stage-2 methods
13-14, layered as L3 in the architecture):

* :func:`quantile_map` — regionalized **quantile mapping / CDF matching**. It
  aligns the *whole distribution* (not just the mean) of a satellite / reanalysis
  variable to co-located observations, which is what fixes AQI category
  thresholds and the extreme tail. Regionalization (by AOI / region label, e.g.
  IGP vs Peninsula vs Coastal) lets the transfer function differ where the
  satellite-surface relationship differs.
* :func:`ml_residual` — **ML residual correction** (gradient boosting on
  meteorology / BLH / land-use covariates) applied *downstream* of quantile
  mapping to remove the nonlinear, state-dependent bias that a static CDF map
  cannot. Uses a lazily-imported :mod:`lightgbm`, falling back to scikit-learn's
  :class:`~sklearn.ensemble.HistGradientBoostingRegressor` (light dep) so the
  demo always runs.

All functions are pure (no I/O) and operate on :mod:`numpy` arrays /
:class:`pandas.DataFrame` so they unit-test without heavy deps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = get_logger("fusion.biascorrect")


@dataclass
class QuantileMapper:
    """A fitted (optionally regionalized) empirical quantile-mapping transform.

    The mapper stores, per region, the sorted satellite quantiles and the
    matching observation quantiles; :meth:`transform` interpolates a new
    satellite value onto the observation CDF (with linear tail extrapolation).

    Attributes:
        quantiles: The probability grid used (shape ``(n_q,)``).
        sat_q: Mapping ``region -> satellite quantile values`` ``(n_q,)``.
        obs_q: Mapping ``region -> observation quantile values`` ``(n_q,)``.
        default_region: Region key used when an input region is unseen.
    """

    quantiles: np.ndarray
    sat_q: dict[str, np.ndarray]
    obs_q: dict[str, np.ndarray]
    default_region: str = "__global__"

    def transform(
        self,
        sat: np.ndarray,
        region: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """Map satellite values onto the observation distribution.

        Args:
            sat: Satellite / reanalysis values to correct ``(n,)`` (or scalar-ish).
            region: Per-sample region labels ``(n,)``, a single label, or ``None``
                to use the global mapper.

        Returns:
            The bias-corrected values (NaNs preserved).
        """
        sat = np.asarray(sat, dtype="float64")
        out = np.full(sat.shape, np.nan)
        if region is None or isinstance(region, str):
            key = region if isinstance(region, str) else self.default_region
            return self._apply_one(sat, key)
        region = np.asarray(region)
        for key in np.unique(region):
            mask = region == key
            out[mask] = self._apply_one(sat[mask], str(key))
        return out

    def _apply_one(self, sat: np.ndarray, region: str) -> np.ndarray:
        """Apply the mapping for a single region (falls back to default)."""
        if region not in self.sat_q:
            region = self.default_region
        sq = self.sat_q[region]
        oq = self.obs_q[region]
        finite = np.isfinite(sat)
        out = np.full(sat.shape, np.nan)
        # np.interp does linear interpolation with flat ends; add slope-based
        # extrapolation for the tails so extremes are not clipped.
        out[finite] = np.interp(sat[finite], sq, oq)
        if sq.size >= 2:
            lo = sat < sq[0]
            hi = sat > sq[-1]
            low_slope = (oq[1] - oq[0]) / (sq[1] - sq[0] + 1e-12)
            high_slope = (oq[-1] - oq[-2]) / (sq[-1] - sq[-2] + 1e-12)
            out = np.where(lo & finite, oq[0] + low_slope * (sat - sq[0]), out)
            out = np.where(hi & finite, oq[-1] + high_slope * (sat - sq[-1]), out)
        return out


def quantile_map(
    sat: np.ndarray,
    obs: np.ndarray,
    *,
    region: Sequence[str] | None = None,
    n_quantiles: int = 100,
    min_region_points: int = 30,
) -> QuantileMapper:
    """Fit a regionalized empirical quantile-mapping (CDF-matching) transform.

    Paired ``sat``/``obs`` samples (co-located satellite value and CPCB
    observation) define, per region, an empirical CDF match. A global mapper is
    always fitted as the fallback; regions with fewer than
    ``min_region_points`` valid pairs reuse the global transform.

    Args:
        sat: Satellite / reanalysis values at co-location points ``(n,)``.
        obs: Co-located ground-truth observations ``(n,)``.
        region: Optional per-sample region labels ``(n,)`` (e.g. ``"IGP"``).
        n_quantiles: Number of probability nodes in the quantile grid.
        min_region_points: Minimum valid pairs for a region-specific transform.

    Returns:
        A fitted :class:`QuantileMapper`.

    Raises:
        ValueError: If there are no finite ``sat``/``obs`` pairs.
    """
    sat = np.asarray(sat, dtype="float64")
    obs = np.asarray(obs, dtype="float64")
    valid = np.isfinite(sat) & np.isfinite(obs)
    if not valid.any():
        raise ValueError("quantile_map requires at least one finite (sat, obs) pair")

    probs = np.linspace(0.0, 1.0, n_quantiles)
    sat_q: dict[str, np.ndarray] = {}
    obs_q: dict[str, np.ndarray] = {}

    sat_q["__global__"] = np.quantile(sat[valid], probs)
    obs_q["__global__"] = np.quantile(obs[valid], probs)

    if region is not None:
        region = np.asarray(region)
        for key in np.unique(region[valid]):
            mask = valid & (region == key)
            if int(mask.sum()) >= min_region_points:
                sat_q[str(key)] = np.quantile(sat[mask], probs)
                obs_q[str(key)] = np.quantile(obs[mask], probs)
            else:
                logger.debug(
                    "region %s has %d pairs (<%d); using global QM",
                    key,
                    int(mask.sum()),
                    min_region_points,
                )

    return QuantileMapper(quantiles=probs, sat_q=sat_q, obs_q=obs_q)


@dataclass
class ResidualCorrector:
    """A fitted ML residual-bias corrector.

    Attributes:
        feature_names: Ordered covariate names the model expects.
        backend: ``"lightgbm"`` or ``"sklearn_hgb"``.
        model: The fitted estimator (predicts the residual ``obs - sat``).
    """

    feature_names: list[str]
    backend: str
    model: object = field(repr=False)

    def transform(self, sat: np.ndarray, features) -> np.ndarray:  # noqa: ANN001
        """Apply the residual correction to satellite values.

        Args:
            sat: Satellite / reanalysis values ``(n,)`` (post quantile-mapping).
            features: Covariate matrix ``(n, n_feat)`` or a DataFrame with the
                fitted ``feature_names`` columns.

        Returns:
            Corrected values ``sat + predicted_residual`` (NaN where sat is NaN).
        """
        X = _as_feature_matrix(features, self.feature_names)
        sat = np.asarray(sat, dtype="float64")
        # LightGBM records column names at fit; predict with a matching frame to
        # keep fit/predict symmetric and silence the sklearn name-mismatch warning.
        predict_in: object = X
        if self.backend == "lightgbm":
            import pandas as pd

            predict_in = pd.DataFrame(X, columns=self.feature_names)
        pred_resid = np.asarray(self.model.predict(predict_in), dtype="float64")  # type: ignore[attr-defined]
        out = sat + pred_resid
        out[~np.isfinite(sat)] = np.nan
        return out


def _as_feature_matrix(features, feature_names: Sequence[str]) -> np.ndarray:  # noqa: ANN001
    """Coerce a DataFrame / array of covariates to a float matrix."""
    try:
        import pandas as pd

        if isinstance(features, pd.DataFrame):
            return features[list(feature_names)].to_numpy(dtype="float64")
    except Exception:  # pragma: no cover - pandas always present in light set
        pass
    return np.asarray(features, dtype="float64")


def ml_residual(
    sat: np.ndarray,
    obs: np.ndarray,
    features,  # noqa: ANN001 - DataFrame | ndarray
    *,
    feature_names: Sequence[str] | None = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> ResidualCorrector:
    """Fit a gradient-boosting residual-bias corrector.

    Learns the residual ``obs - sat`` as a function of state covariates
    (meteorology, BLH, RH, land-use, etc.). Prefers :mod:`lightgbm`; if it is
    not installed, falls back to scikit-learn's
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor` so the demo runs on
    the light dependency set.

    Args:
        sat: Satellite / reanalysis values at training points ``(n,)`` (already
            quantile-mapped, conventionally).
        obs: Co-located observations ``(n,)``.
        features: Covariate matrix ``(n, n_feat)`` or a DataFrame.
        feature_names: Column names (required if ``features`` is an ndarray and
            you intend to later pass a DataFrame to ``transform``). Defaults to
            ``f0..f{n-1}`` or the DataFrame's columns.
        n_estimators: Boosting iterations.
        learning_rate: Boosting learning rate.
        random_state: RNG seed.

    Returns:
        A fitted :class:`ResidualCorrector`.

    Raises:
        ValueError: If there are no finite training rows.
    """
    sat = np.asarray(sat, dtype="float64")
    obs = np.asarray(obs, dtype="float64")

    names = _resolve_feature_names(features, feature_names)
    X = _as_feature_matrix(features, names)
    y = obs - sat
    rows = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if not rows.any():
        raise ValueError("ml_residual requires at least one finite training row")
    X, y = X[rows], y[rows]

    try:
        import lightgbm as lgb
        import pandas as pd

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
        )
        # Fit with named columns so predict can supply matching feature names.
        model.fit(pd.DataFrame(X, columns=list(names)), y)
        backend = "lightgbm"
    except Exception as exc:
        logger.info(
            "lightgbm unavailable (%s); using sklearn HistGradientBoosting",
            type(exc).__name__,
        )
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        model.fit(X, y)
        backend = "sklearn_hgb"

    return ResidualCorrector(feature_names=list(names), backend=backend, model=model)


def _resolve_feature_names(features, feature_names) -> list[str]:  # noqa: ANN001
    """Determine ordered feature names from inputs."""
    if feature_names is not None:
        return list(feature_names)
    try:
        import pandas as pd

        if isinstance(features, pd.DataFrame):
            return list(features.columns)
    except Exception:  # pragma: no cover
        pass
    arr = np.asarray(features)
    n_feat = arr.shape[1] if arr.ndim == 2 else 1
    return [f"f{i}" for i in range(n_feat)]


__all__ = [
    "QuantileMapper",
    "quantile_map",
    "ResidualCorrector",
    "ml_residual",
]
