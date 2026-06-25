"""Uncertainty quantification — split-conformal prediction intervals.

The product must report *calibrated* per-pixel uncertainty, not just a point
estimate. The design's UQ layer is quantile heads + (Mondrian/weighted)
split-conformal for distribution-free coverage under spatial shift, plus a
per-pixel applicability / abstention map. This module implements the conformal
core from scratch (no heavy dependency), with an optional MAPIE adapter:

* :func:`split_conformal_calibrate` — fit an absolute-residual conformal
  quantile on a held-out calibration set, giving a symmetric half-width ``q`` for
  a target miscoverage ``alpha`` (marginal coverage ``1 - alpha``).
* :func:`conformalized_quantile_calibrate` — CQR: calibrate the *width* of a
  quantile-head interval so its empirical coverage matches ``1 - alpha`` (adaptive
  intervals that respect heteroscedasticity).
* :class:`SplitConformalRegressor` — wraps any point estimator into one that
  emits intervals; :class:`MondrianConformal` does the same per group (e.g. per
  region / AQI band) so coverage holds within strata, not just on average.
* :func:`picp`, :func:`mpiw`, :func:`coverage_report` — empirical interval
  diagnostics (PICP / MPIW) used on the validation page.
* :func:`applicability_flag` — the abstention map: rows whose features fall
  outside the training feature envelope (or whose interval is too wide) are
  flagged so the product can *decline to predict* in cloud-gap / station-sparse
  zones rather than hallucinate.

Only numpy is needed; ``mapie`` is imported lazily inside its adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

_log = get_logger("models.uq")


# --------------------------------------------------------------------------- #
# Core split-conformal calibration
# --------------------------------------------------------------------------- #
def _finite_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-aligned ``(y_true, y_pred)`` float arrays."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def _conformal_quantile_level(n: int, alpha: float) -> float:
    """Finite-sample-corrected quantile level ceil((n+1)(1-alpha))/n.

    Using the ``(n+1)`` correction guarantees the conformal coverage bound
    ``P(Y in C) >= 1 - alpha`` for exchangeable data. The level is clipped to
    ``[0, 1]`` for tiny calibration sets.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    level = np.ceil((n + 1) * (1.0 - alpha)) / max(n, 1)
    return float(min(max(level, 0.0), 1.0))


def split_conformal_calibrate(
    y_cal: Any, y_cal_pred: Any, *, alpha: float = 0.1
) -> float:
    """Calibrate a symmetric conformal half-width from a calibration set.

    The non-conformity score is the absolute residual ``|y - yhat|``. The
    returned ``q`` is its ``(n+1)(1-alpha)/n`` empirical quantile; the prediction
    interval for a new point is ``[yhat - q, yhat + q]`` and has marginal
    coverage ``>= 1 - alpha`` under exchangeability.

    Args:
        y_cal: True calibration targets.
        y_cal_pred: Point predictions on the calibration set.
        alpha: Target miscoverage (``0.1`` -> 90% intervals).

    Returns:
        The conformal half-width ``q`` (>= 0).
    """
    yt, yp = _finite_pair(y_cal, y_cal_pred)
    if yt.size == 0:
        return float("nan")
    scores = np.abs(yt - yp)
    level = _conformal_quantile_level(yt.size, alpha)
    return float(np.quantile(scores, level, method="higher"))


def conformalized_quantile_calibrate(
    y_cal: Any, lower_cal: Any, upper_cal: Any, *, alpha: float = 0.1
) -> float:
    """Conformalized Quantile Regression (CQR) width correction.

    Given quantile-head predictions ``[lower, upper]`` on the calibration set,
    the CQR score is ``max(lower - y, y - upper)`` (signed interval miss). The
    returned ``q`` widens (or tightens, if negative) both edges so the corrected
    interval ``[lower - q, upper + q]`` attains ``1 - alpha`` coverage while
    keeping the adaptive (heteroscedastic) shape of the quantile heads.

    Args:
        y_cal: True calibration targets.
        lower_cal: Lower quantile-head predictions on the calibration set.
        upper_cal: Upper quantile-head predictions on the calibration set.
        alpha: Target miscoverage.

    Returns:
        The CQR width adjustment ``q`` (may be negative to tighten).
    """
    yt = np.asarray(y_cal, dtype=np.float64).ravel()
    lo = np.asarray(lower_cal, dtype=np.float64).ravel()
    hi = np.asarray(upper_cal, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    yt, lo, hi = yt[mask], lo[mask], hi[mask]
    if yt.size == 0:
        return float("nan")
    scores = np.maximum(lo - yt, yt - hi)
    level = _conformal_quantile_level(yt.size, alpha)
    return float(np.quantile(scores, level, method="higher"))


# --------------------------------------------------------------------------- #
# Conformal regressor wrappers
# --------------------------------------------------------------------------- #
@dataclass
class SplitConformalRegressor:
    """Wrap a point estimator to emit split-conformal prediction intervals.

    The base estimator is fit on a training split, then calibrated on a disjoint
    calibration split (the conformal guarantee needs the calibration data to be
    unseen by the fit). ``predict_interval`` returns ``(lower, point, upper)``.

    Attributes:
        estimator: Any object with sklearn-style ``fit`` / ``predict``.
        alpha: Target miscoverage.
        q_: The fitted conformal half-width (set after :meth:`calibrate`).
    """

    estimator: Any
    alpha: float = 0.1
    q_: float = field(default=float("nan"), init=False)

    def fit(self, X_train: Any, y_train: Any) -> "SplitConformalRegressor":
        """Fit the underlying point estimator on the proper training split."""
        self.estimator.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal: Any, y_cal: Any) -> "SplitConformalRegressor":
        """Calibrate the conformal half-width on a held-out calibration split."""
        y_pred = np.asarray(self.estimator.predict(X_cal), dtype=np.float64).ravel()
        self.q_ = split_conformal_calibrate(y_cal, y_pred, alpha=self.alpha)
        _log.info("Split-conformal half-width q=%.4g (alpha=%.3f)", self.q_, self.alpha)
        return self

    def fit_calibrate(
        self,
        X: Any,
        y: Any,
        *,
        cal_fraction: float = 0.25,
        random_state: int = 42,
    ) -> "SplitConformalRegressor":
        """Convenience: random train/calibration split, then fit + calibrate.

        Args:
            X: Full predictor matrix.
            y: Full target vector.
            cal_fraction: Fraction of rows reserved for calibration.
            random_state: Seed for the split.

        Returns:
            ``self``, fitted and calibrated.
        """
        X = np.asarray(X) if not hasattr(X, "iloc") else X
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n)
        n_cal = max(1, int(round(cal_fraction * n)))
        cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
        X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        X_cal = X.iloc[cal_idx] if hasattr(X, "iloc") else X[cal_idx]
        self.fit(X_tr, y[tr_idx])
        self.calibrate(X_cal, y[cal_idx])
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Point predictions from the underlying estimator."""
        return np.asarray(self.estimator.predict(X), dtype=np.float64).ravel()

    def predict_interval(self, X: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lower, point, upper)`` conformal intervals for ``X``."""
        if not np.isfinite(self.q_):
            raise RuntimeError("Call calibrate()/fit_calibrate() before predict_interval().")
        point = self.predict(X)
        return point - self.q_, point, point + self.q_


@dataclass
class MondrianConformal:
    """Group-conditional (Mondrian) split-conformal — coverage per stratum.

    Marginal conformal coverage can hide poor coverage *within* a region or AQI
    band. Mondrian conformal calibrates a separate half-width per group label, so
    the ``1 - alpha`` guarantee holds conditionally on the group. Groups unseen at
    calibration fall back to the pooled half-width.

    Attributes:
        alpha: Target miscoverage.
        q_by_group_: Calibrated half-width per group label.
        q_global_: Pooled fallback half-width.
    """

    alpha: float = 0.1
    q_by_group_: dict[Any, float] = field(default_factory=dict, init=False)
    q_global_: float = field(default=float("nan"), init=False)

    def calibrate(self, y_cal: Any, y_cal_pred: Any, groups: Any) -> "MondrianConformal":
        """Calibrate one conformal half-width per group label.

        Args:
            y_cal: True calibration targets.
            y_cal_pred: Point predictions on the calibration set.
            groups: Per-row group labels (e.g. region or AQI band).

        Returns:
            ``self`` with :attr:`q_by_group_` and :attr:`q_global_` populated.
        """
        yt = np.asarray(y_cal, dtype=np.float64).ravel()
        yp = np.asarray(y_cal_pred, dtype=np.float64).ravel()
        g = np.asarray(groups).ravel()
        self.q_global_ = split_conformal_calibrate(yt, yp, alpha=self.alpha)
        for label in np.unique(g):
            m = g == label
            self.q_by_group_[label] = split_conformal_calibrate(
                yt[m], yp[m], alpha=self.alpha
            )
        return self

    def predict_interval(
        self, point: Any, groups: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` using each row's group half-width.

        Args:
            point: Point predictions for the query rows.
            groups: Per-row group labels (fallback to global where unseen).

        Returns:
            ``(lower, upper)`` arrays.
        """
        p = np.asarray(point, dtype=np.float64).ravel()
        g = np.asarray(groups).ravel()
        q = np.array(
            [self.q_by_group_.get(lbl, self.q_global_) for lbl in g], dtype=np.float64
        )
        return p - q, p + q


# --------------------------------------------------------------------------- #
# Interval diagnostics
# --------------------------------------------------------------------------- #
def picp(y_true: Any, lower: Any, upper: Any) -> float:
    """Prediction-Interval Coverage Probability — fraction of ``y`` inside.

    Args:
        y_true: Observed values.
        lower: Lower interval bounds.
        upper: Upper interval bounds.

    Returns:
        Empirical coverage in ``[0, 1]`` (NaN if no finite rows).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    inside = (yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])
    return float(np.mean(inside))


def mpiw(lower: Any, upper: Any) -> float:
    """Mean Prediction-Interval Width (sharpness; lower is better at fixed PICP)."""
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    mask = np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    return float(np.mean(hi[mask] - lo[mask]))


def coverage_report(
    y_true: Any, lower: Any, upper: Any, *, alpha: float = 0.1
) -> dict[str, float]:
    """Summarise interval calibration: PICP, MPIW, target, and the coverage gap.

    Args:
        y_true: Observed values.
        lower: Lower interval bounds.
        upper: Upper interval bounds.
        alpha: The miscoverage the interval targets (for the ``target`` field).

    Returns:
        Dict with ``picp``, ``mpiw``, ``target_coverage`` (``1 - alpha``),
        ``coverage_gap`` (PICP minus target) and ``n``.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    p = picp(yt, lower, upper)
    target = 1.0 - alpha
    return {
        "picp": p,
        "mpiw": mpiw(lower, upper),
        "target_coverage": target,
        "coverage_gap": p - target,
        "n": float(np.isfinite(yt).sum()),
    }


# --------------------------------------------------------------------------- #
# Applicability / abstention map
# --------------------------------------------------------------------------- #
def applicability_flag(
    X_train: Any,
    X_query: Any,
    *,
    n_std: float = 3.0,
    max_out_of_range_frac: float = 0.25,
) -> np.ndarray:
    """Flag query rows outside the training feature envelope (abstention map).

    A per-row boolean ``applicable`` is computed by z-scoring each query feature
    against the *training* mean/std and counting how many features fall beyond
    ``n_std`` sigma. A row is flagged **not applicable** (``False``) when more
    than ``max_out_of_range_frac`` of its features are out of range — i.e. the
    model is extrapolating and should abstain. Non-finite query features count
    as out of range.

    Args:
        X_train: Training predictor matrix (defines the envelope).
        X_query: Query predictor matrix to flag.
        n_std: Per-feature z-score threshold for "in range".
        max_out_of_range_frac: Max fraction of out-of-range features before a row
            is declared not applicable.

    Returns:
        Boolean array of shape ``(n_query,)``; ``True`` where the model is
        considered applicable.
    """
    Xt = np.asarray(X_train, dtype=np.float64)
    Xq = np.asarray(X_query, dtype=np.float64)
    if Xt.ndim == 1:
        Xt = Xt.reshape(-1, 1)
    if Xq.ndim == 1:
        Xq = Xq.reshape(-1, 1)
    mu = np.nanmean(Xt, axis=0)
    sd = np.nanstd(Xt, axis=0)
    sd = np.where(sd > 0, sd, 1.0)  # guard constant features
    z = np.abs((Xq - mu) / sd)
    out_of_range = (z > n_std) | ~np.isfinite(Xq)
    frac = out_of_range.mean(axis=1)
    return frac <= max_out_of_range_frac


# --------------------------------------------------------------------------- #
# Optional MAPIE adapter
# --------------------------------------------------------------------------- #
def mapie_split_conformal(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    X_query: Any,
    *,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split-conformal intervals via the optional MAPIE library (lazy import).

    A thin adapter for users who have ``mapie`` installed and prefer its
    implementation; the from-scratch :class:`SplitConformalRegressor` is the
    default and needs no extra dependency. Returns ``(lower, point, upper)``.

    Args:
        estimator: An unfitted sklearn-style regressor.
        X_train: Training predictors.
        y_train: Training targets.
        X_query: Query predictors.
        alpha: Target miscoverage.

    Returns:
        ``(lower, point, upper)`` arrays for ``X_query``.

    Raises:
        ImportError: If ``mapie`` is not installed.
    """
    try:
        from mapie.regression import MapieRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "mapie is not installed; use SplitConformalRegressor for the "
            "dependency-free split-conformal implementation."
        ) from exc

    mapie = MapieRegressor(estimator, method="base", cv="split")
    mapie.fit(X_train, np.asarray(y_train, dtype=np.float64).ravel())
    point, interval = mapie.predict(X_query, alpha=alpha)
    lower = np.asarray(interval[:, 0, 0], dtype=np.float64)
    upper = np.asarray(interval[:, 1, 0], dtype=np.float64)
    return lower, np.asarray(point, dtype=np.float64), upper


__all__ = [
    "split_conformal_calibrate",
    "conformalized_quantile_calibrate",
    "SplitConformalRegressor",
    "MondrianConformal",
    "picp",
    "mpiw",
    "coverage_report",
    "applicability_flag",
    "mapie_split_conformal",
]
