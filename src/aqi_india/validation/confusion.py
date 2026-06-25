"""AQI-category confusion matrix and per-class precision / recall.

Surface-AQI skill is not only a regression number: for public-health messaging
what matters is whether the model puts a day in the *right CPCB band*, and in
particular whether it **catches the bad days**. A model with an excellent R can
still systematically under-call Severe episodes (the RMA slope < 1 compression
the metrics module flags), which is exactly the failure that harms users.

This module converts continuous ``y_true`` / ``y_pred`` AQI values into the six
CPCB categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) and
builds a 6x6 confusion matrix with per-class precision, recall and F1. The
report deliberately surfaces **recall for the hazardous classes** (Poor / Very
Poor / Severe) — missing a Severe day is far costlier than a false alarm.

The category edges come straight from :mod:`aqi_india.aqi.breakpoints` (the
single source of truth), so the bands here can never drift from the AQI engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..aqi.breakpoints import CATEGORY_NAMES, category_index

#: The CPCB AQI categories considered hazardous; their recall is emphasised in
#: the report because a missed exceedance is the costly error mode.
HAZARDOUS_CATEGORIES: tuple[str, ...] = ("Poor", "Very Poor", "Severe")


def assign_categories(aqi: Any) -> np.ndarray:
    """Map continuous AQI values to CPCB category indices (0..5).

    Non-finite AQI values map to ``-1`` (invalid / unclassifiable) so callers
    can drop them. The mapping uses :func:`aqi_india.aqi.breakpoints.category_index`
    so it is guaranteed consistent with the AQI engine and the colour LUT.

    Args:
        aqi: Scalar or array of AQI values in ``[0, 500]`` (clamped internally).

    Returns:
        Integer array of category indices (``-1`` where AQI is NaN/inf).
    """
    arr = np.asarray(aqi, dtype=np.float64).ravel()
    out = np.full(arr.shape, -1, dtype=np.int64)
    finite = np.isfinite(arr)
    for i in np.where(finite)[0]:
        out[i] = category_index(float(arr[i]))
    return out


def aqi_category_confusion(
    y_true: Any,
    y_pred: Any,
    *,
    normalize: str | None = None,
) -> dict[str, Any]:
    """Build the 6-class AQI confusion matrix with per-class P / R / F1.

    Rows of the matrix are the **true** category, columns the **predicted**
    category, both ordered as :data:`aqi_india.aqi.breakpoints.CATEGORY_NAMES`
    (Good -> Severe). Pairs where either AQI is non-finite are dropped.

    Args:
        y_true: Observed AQI values (continuous; binned internally).
        y_pred: Predicted AQI values (continuous; binned internally).
        normalize: ``None`` for raw counts, ``"true"`` to normalise each row to
            sum 1 (recall view), ``"pred"`` to normalise each column (precision
            view), or ``"all"`` to normalise the whole matrix.

    Returns:
        Dict with:
            ``labels`` — the six category names,
            ``matrix`` — 6x6 ``int`` (or ``float`` if normalised) array,
            ``support`` — per-true-class sample counts,
            ``per_class`` — ``{name: {precision, recall, f1, support}}``,
            ``hazardous_recall`` — mean recall over Poor/Very Poor/Severe,
            ``accuracy`` — overall fraction correctly categorised,
            ``n`` — number of valid pairs scored.

    Raises:
        ValueError: If ``normalize`` is not one of the accepted values.
    """
    if normalize not in (None, "true", "pred", "all"):
        raise ValueError(
            "normalize must be None, 'true', 'pred' or 'all', "
            f"got {normalize!r}"
        )

    ti = assign_categories(y_true)
    pi = assign_categories(y_pred)
    if ti.shape != pi.shape:
        raise ValueError("y_true and y_pred must have the same length")

    valid = (ti >= 0) & (pi >= 0)
    ti, pi = ti[valid], pi[valid]
    n_classes = len(CATEGORY_NAMES)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(ti, pi):
        cm[t, p] += 1

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diag = np.diag(cm).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, diag / support, np.nan)
        precision = np.where(predicted > 0, diag / predicted, np.nan)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            np.nan,
        )

    per_class: dict[str, dict[str, float]] = {}
    for c, name in enumerate(CATEGORY_NAMES):
        per_class[name] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }

    haz_recalls = [
        per_class[name]["recall"]
        for name in HAZARDOUS_CATEGORIES
        if per_class[name]["support"] > 0
    ]
    hazardous_recall = (
        float(np.nanmean(haz_recalls)) if haz_recalls else float("nan")
    )

    total = cm.sum()
    accuracy = float(diag.sum() / total) if total else float("nan")

    matrix: np.ndarray = cm
    if normalize == "true":
        matrix = np.where(
            support[:, None] > 0, cm / support[:, None], 0.0
        )
    elif normalize == "pred":
        matrix = np.where(
            predicted[None, :] > 0, cm / predicted[None, :], 0.0
        )
    elif normalize == "all":
        matrix = cm / total if total else cm.astype(np.float64)

    return {
        "labels": list(CATEGORY_NAMES),
        "matrix": matrix,
        "support": support,
        "per_class": per_class,
        "hazardous_recall": hazardous_recall,
        "accuracy": accuracy,
        "n": int(total),
    }


__all__ = [
    "HAZARDOUS_CATEGORIES",
    "assign_categories",
    "aqi_category_confusion",
]
