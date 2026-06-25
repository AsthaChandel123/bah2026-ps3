"""Validation metrics for surface-concentration and AQI prediction.

A compact, NaN-safe, NumPy-only metrics library used throughout the validation
ladder (leave-station-out, spatiotemporal-blocked CV) and the final hold-out
report. Every function pairs ``y_true`` and ``y_pred``, drops pairs where either
value is non-finite, and returns a plain float.

Provided metrics:
    * ``rmse``  — root mean squared error
    * ``mae``   — mean absolute error
    * ``mbe``   — mean bias error (pred - true)
    * ``pearson_r`` — Pearson correlation coefficient
    * ``r2``    — coefficient of determination (1 - SS_res/SS_tot)
    * ``nmb``   — normalised mean bias  = sum(pred-true)/sum(true)
    * ``nme``   — normalised mean error = sum|pred-true|/sum(true)
    * ``ioa``   — Willmott index of agreement
    * ``rma_slope_intercept`` — reduced-major-axis (Deming) regression via ODR

``metrics_table`` returns all of them as a dict; ``stratified_metrics`` computes
the table within concentration/AQI bands.
"""

from __future__ import annotations

import numpy as np


def _clean_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten, align and drop non-finite pairs from two arrays."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same number of elements, "
            f"got {yt.shape} and {yp.shape}"
        )
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yp - yt)))


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean bias error (mean of pred - true)."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yp - yt))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient (NaN if a series has zero variance)."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    st, sp = yt.std(), yp.std()
    if st == 0.0 or sp == 0.0:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, ``1 - SS_res / SS_tot``.

    Returns NaN when the true series has zero variance (SS_tot == 0).
    """
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def nmb(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised mean bias = sum(pred - true) / sum(true)."""
    yt, yp = _clean_pair(y_true, y_pred)
    denom = np.sum(yt)
    if yt.size == 0 or denom == 0.0:
        return float("nan")
    return float(np.sum(yp - yt) / denom)


def nme(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised mean error = sum|pred - true| / sum(true)."""
    yt, yp = _clean_pair(y_true, y_pred)
    denom = np.sum(yt)
    if yt.size == 0 or denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(yp - yt)) / denom)


def ioa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Willmott index of agreement (1 = perfect agreement, 0 = none).

    ``IOA = 1 - sum((p-o)^2) / sum((|p - mean(o)| + |o - mean(o)|)^2)``.
    """
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    o_mean = np.mean(yt)
    num = np.sum((yp - yt) ** 2)
    denom = np.sum((np.abs(yp - o_mean) + np.abs(yt - o_mean)) ** 2)
    if denom == 0.0:
        return float("nan")
    return float(1.0 - num / denom)


def rma_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Reduced-major-axis (Deming/orthogonal) regression slope and intercept.

    Fits ``y_pred = slope * y_true + intercept`` accounting for error in both
    variables via :mod:`scipy.odr`. A slope < 1 exposes the compression of
    high-concentration episodes that R^2 alone can hide.

    Returns:
        ``(slope, intercept)``; ``(nan, nan)`` if the fit is degenerate.
    """
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size < 2 or yt.std() == 0.0:
        return float("nan"), float("nan")

    from scipy import odr  # local import keeps the module light

    def _linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return beta[0] * x + beta[1]

    # Initial guess from an ordinary least-squares slope.
    slope0 = np.cov(yt, yp, bias=True)[0, 1] / np.var(yt)
    intercept0 = np.mean(yp) - slope0 * np.mean(yt)
    model = odr.Model(_linear)
    data = odr.RealData(yt, yp)
    fit = odr.ODR(data, model, beta0=[slope0, intercept0]).run()
    return float(fit.beta[0]), float(fit.beta[1])


def metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return all core metrics as a dictionary.

    Keys: ``n, rmse, mae, mbe, r, r2, nmb, nme, ioa, rma_slope, rma_intercept``.
    """
    yt, yp = _clean_pair(y_true, y_pred)
    slope, intercept = rma_slope_intercept(yt, yp)
    return {
        "n": float(yt.size),
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "mbe": mbe(yt, yp),
        "r": pearson_r(yt, yp),
        "r2": r2(yt, yp),
        "nmb": nmb(yt, yp),
        "nme": nme(yt, yp),
        "ioa": ioa(yt, yp),
        "rma_slope": slope,
        "rma_intercept": intercept,
    }


def stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bands: list[tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Compute the metrics table within concentration / AQI bands.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        bands: List of ``(low, high)`` band edges (``high`` exclusive). Banding
            is applied on ``y_true``.

    Returns:
        Mapping ``"[low,high)" -> metrics_table`` for each band (only bands with
        at least one valid pair are included).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    out: dict[str, dict[str, float]] = {}
    for low, high in bands:
        sel = np.isfinite(yt) & (yt >= low) & (yt < high)
        if sel.any():
            out[f"[{low:g},{high:g})"] = metrics_table(yt[sel], yp[sel])
    return out


__all__ = [
    "rmse",
    "mae",
    "mbe",
    "pearson_r",
    "r2",
    "nmb",
    "nme",
    "ioa",
    "rma_slope_intercept",
    "metrics_table",
    "stratified_metrics",
]
