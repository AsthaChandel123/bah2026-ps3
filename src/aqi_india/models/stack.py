"""Ensemble stacking of surface-concentration members.

The PS3 final estimator is a Gaussian-Process-blended stack of SA-ConvLSTM +
ST-GNN + LightGBM (mirrors the SOTA India daily-1km product, R2=0.86). That GP
blender belongs to the heavy `deep` workflow; here we provide the **light,
runnable** stacking layer the demo uses: simple averaging and out-of-fold
weighted blending of the classical members (LightGBM + RandomForest baseline),
each of which honours the ``fit(X, y_dict)`` / ``predict(X) -> dict`` contract.

* :class:`EnsembleStack` — holds named members, fits them all, and combines
  per-pollutant predictions by ``"mean"``, fixed ``"weighted"`` weights, or
  out-of-fold-optimised ``"optimal"`` non-negative weights.
* :func:`fit_oof_weights` — derive per-pollutant blend weights by minimising OOF
  RMSE with a simple projected coordinate search (no SciPy optimiser needed).

The Gaussian-Process meta-blender is documented as future work in
:func:`gp_stack_future` so the architecture intent is explicit in code.

Only numpy is required at import; member estimators bring their own (lazy) deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.logging import get_logger
from .lightgbm_model import SURFACE_POLLUTANTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

_log = get_logger("models.stack")

#: Supported blend strategies.
BLEND_STRATEGIES: tuple[str, ...] = ("mean", "weighted", "optimal")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NaN-safe root-mean-squared error."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def fit_oof_weights(
    member_preds: "list[np.ndarray]",
    y_true: Any,
    *,
    n_iter: int = 200,
    step: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """Find non-negative, sum-to-one blend weights minimising OOF RMSE.

    A light projected coordinate-descent search (no SciPy dependency): starting
    from equal weights, it repeatedly proposes nudging weight mass from one
    member to another and keeps the move if it lowers OOF RMSE, re-normalising to
    the simplex each step. This is robust for the handful of members here and
    avoids a heavy optimiser.

    Args:
        member_preds: List of per-member OOF prediction arrays (all same length).
        y_true: Observed OOF targets.
        n_iter: Number of random coordinate moves to attempt.
        step: Weight mass moved per accepted step.
        random_state: Seed for the proposal RNG.

    Returns:
        A weight vector of shape ``(n_members,)`` on the probability simplex.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.column_stack([np.asarray(p, dtype=np.float64).ravel() for p in member_preds])
    m = P.shape[1]
    w = np.full(m, 1.0 / m)
    best = _rmse(yt, P @ w)
    rng = np.random.default_rng(random_state)
    for _ in range(n_iter):
        i, j = rng.integers(0, m), rng.integers(0, m)
        if i == j or w[i] < step:
            continue
        cand = w.copy()
        cand[i] -= step
        cand[j] += step
        score = _rmse(yt, P @ cand)
        if np.isfinite(score) and score < best:
            w, best = cand, score
    return w


@dataclass
class EnsembleStack:
    """Weighted ensemble of multi-pollutant members (mean / weighted / optimal).

    Each member must expose ``fit(X, y_dict)`` and ``predict(X) -> dict`` (the
    contract shared by :class:`~aqi_india.models.lightgbm_model.MultiPollutantLGBM`
    and the baselines). The stack fits every member on the same data and combines
    their per-pollutant predictions with per-pollutant weights.

    Attributes:
        members: ``{name: estimator}`` mapping of the ensemble members.
        pollutants: Pollutants to combine.
        strategy: ``"mean"``, ``"weighted"`` (uses :attr:`weights`), or
            ``"optimal"`` (OOF-fitted weights via :meth:`fit_optimal_weights`).
        weights: ``{member_name: weight}`` for the ``"weighted"`` strategy.
        weights_: Fitted per-pollutant member weights ``{pollutant: array}``
            (populated by ``"optimal"``).
    """

    members: dict[str, Any]
    pollutants: tuple[str, ...] = SURFACE_POLLUTANTS
    strategy: str = "mean"
    weights: "dict[str, float] | None" = None
    weights_: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.strategy not in BLEND_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {BLEND_STRATEGIES}, got {self.strategy!r}"
            )
        if not self.members:
            raise ValueError("EnsembleStack needs at least one member.")

    @property
    def member_names(self) -> list[str]:
        """Ordered member names."""
        return list(self.members)

    def fit(self, X: Any, y_dict: "dict[str, Any]") -> "EnsembleStack":
        """Fit every member on ``(X, y_dict)``.

        Args:
            X: Predictor matrix.
            y_dict: ``{pollutant: y}`` labels.

        Returns:
            ``self`` with all members fitted.
        """
        for name, est in self.members.items():
            _log.info("Fitting stack member %r ...", name)
            est.fit(X, y_dict)
        return self

    def _member_prediction_cube(self, X: Any) -> dict[str, np.ndarray]:
        """Return ``{pollutant: (n_samples, n_members)}`` prediction stacks."""
        per_member = {name: est.predict(X) for name, est in self.members.items()}
        cube: dict[str, np.ndarray] = {}
        for pol in self.pollutants:
            cols = [
                per_member[name][pol]
                for name in self.member_names
                if pol in per_member[name]
            ]
            if cols:
                cube[pol] = np.column_stack([np.asarray(c, dtype=np.float64) for c in cols])
        return cube

    def _weights_for(self, pol: str, n_members: int) -> np.ndarray:
        """Resolve the weight vector for ``pol`` given the active strategy."""
        if self.strategy == "mean":
            return np.full(n_members, 1.0 / n_members)
        if self.strategy == "weighted":
            wd = self.weights or {}
            w = np.array([wd.get(name, 0.0) for name in self.member_names], dtype=np.float64)
            total = w.sum()
            return w / total if total > 0 else np.full(n_members, 1.0 / n_members)
        # optimal
        if pol in self.weights_:
            return self.weights_[pol]
        return np.full(n_members, 1.0 / n_members)

    def fit_optimal_weights(
        self, oof_member_preds: "dict[str, dict[str, np.ndarray]]", y_oof: "dict[str, Any]"
    ) -> "EnsembleStack":
        """Fit per-pollutant OOF blend weights and switch to the optimal strategy.

        Args:
            oof_member_preds: ``{member_name: {pollutant: oof_pred}}`` out-of-fold
                predictions for each member (e.g. from
                :func:`aqi_india.validation.cv.cross_validate`).
            y_oof: ``{pollutant: oof_true}`` matching OOF targets.

        Returns:
            ``self`` with :attr:`weights_` populated and ``strategy="optimal"``.
        """
        for pol in self.pollutants:
            preds = [
                oof_member_preds[name][pol]
                for name in self.member_names
                if name in oof_member_preds and pol in oof_member_preds[name]
            ]
            if len(preds) < 1 or pol not in y_oof:
                continue
            self.weights_[pol] = fit_oof_weights(preds, y_oof[pol])
            _log.info("Optimal weights for %s: %s", pol, np.round(self.weights_[pol], 3))
        self.strategy = "optimal"
        return self

    def predict(self, X: Any) -> dict[str, np.ndarray]:
        """Blend member predictions into a per-pollutant ensemble prediction.

        Args:
            X: Predictor matrix.

        Returns:
            ``{pollutant: blended_prediction}`` (clipped at 0).
        """
        cube = self._member_prediction_cube(X)
        out: dict[str, np.ndarray] = {}
        for pol, stack in cube.items():
            w = self._weights_for(pol, stack.shape[1])
            out[pol] = np.clip(stack @ w, 0.0, None)
        return out


def gp_stack_future() -> str:
    """Document the production Gaussian-Process meta-blender (future work).

    The final submission blends the OOF predictions of SA-ConvLSTM + ST-GNN +
    LightGBM with a Gaussian-Process regressor under an anisotropic RBF kernel,
    yielding a per-pixel predictive mean *and variance* (the epistemic component
    of the UQ map). It lives in the heavy ``deep`` workflow because it depends on
    the torch members and ``scikit-learn``/``GPy`` GP fitting over the full grid;
    the light demo uses :class:`EnsembleStack` instead. This stub exists so the
    architectural intent is explicit and greppable in the codebase.

    Returns:
        A short human-readable description of the planned GP stacker.
    """
    return (
        "GP-blended stack (future): members=[saconvlstm, gnn, lightgbm]; "
        "blender=GaussianProcessRegressor(kernel=anisotropic RBF); "
        "outputs per-pixel mean + variance; trained on out-of-fold member "
        "predictions; part of the `deep` extra, not the light demo."
    )


__all__ = [
    "BLEND_STRATEGIES",
    "EnsembleStack",
    "fit_oof_weights",
    "gp_stack_future",
]
