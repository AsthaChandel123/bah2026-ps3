"""Getis-Ord Gi* hot-spot statistic with FDR-controlled significance.

Getis-Ord Gi* is the primary, reviewer-trusted hotspot engine (it is the same
statistic ArcGIS' *Hot Spot Analysis* tool reports). For every grid cell it
compares the local weighted sum of a variable in the cell *and its neighbours*
(``star=True`` includes the focal cell itself) against the global expectation,
returning a standard z-score: large positive z = significant clustering of HIGH
values (a hotspot), large negative z = a coldspot.

Because we test thousands of cells simultaneously, raw per-cell p-values would
yield many false positives. We therefore control the **false discovery rate**
with the Benjamini-Hochberg procedure (blueprint method 34: "FDR p<0.01") and
return a boolean significance mask alongside the z-scores.

The spatial weights graph is built from grid-cell centroids as either KNN-``k``
(default ``k=8``, the queen-contiguity analogue for a regular grid) or a
distance-band graph. ``esda``/``libpysal`` are part of the light dependency set,
so this module imports cleanly without heavy/credentialed deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.logging import get_logger

logger = get_logger("hotspot.getis_ord")


@dataclass
class GiStarResult:
    """Container for a Getis-Ord Gi* run.

    Attributes:
        z: Gi* z-scores, one per input observation (``NaN`` where undefined).
        p: One-sided permutation pseudo p-values per observation (``p_sim``).
        sig: Boolean FDR-significant *hotspot* mask (high-value clustering only,
            ``z > 0`` and BH-significant at ``alpha``).
        sig_cold: Boolean FDR-significant *coldspot* mask (``z < 0``).
        fdr_threshold: The Benjamini-Hochberg p-value cutoff actually applied
            (``0.0`` if nothing was significant).
        alpha: The target FDR level.
    """

    z: np.ndarray
    p: np.ndarray
    sig: np.ndarray
    sig_cold: np.ndarray
    fdr_threshold: float
    alpha: float


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.01) -> tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR control.

    Args:
        p: 1-D array of p-values. ``NaN`` entries are treated as non-significant.
        alpha: Target false discovery rate.

    Returns:
        A tuple ``(reject, threshold)`` where ``reject`` is a boolean mask of
        the hypotheses declared significant and ``threshold`` is the largest
        p-value that passed (``0.0`` if none did).
    """
    p = np.asarray(p, dtype=np.float64)
    reject = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    pv = p[finite]
    m = pv.size
    if m == 0:
        return reject, 0.0

    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    crit = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= crit
    if not passed.any():
        return reject, 0.0

    k_max = np.max(np.flatnonzero(passed))
    threshold = float(ranked[k_max])

    finite_idx = np.flatnonzero(finite)
    reject[finite_idx[pv <= threshold]] = True
    return reject, threshold


def build_weights(coords: np.ndarray, *, k: int = 8, kind: str = "knn", band: float | None = None):
    """Build a ``libpysal`` spatial weights object from point coordinates.

    Args:
        coords: ``(n, 2)`` array of cell-centroid coordinates ``(x, y)`` /
            ``(lon, lat)``.
        k: Number of nearest neighbours for ``kind="knn"``.
        kind: ``"knn"`` (default) or ``"distance_band"``.
        band: Distance threshold for ``kind="distance_band"``. If ``None`` it is
            auto-set to the smallest band guaranteeing each point ``>= 1``
            neighbour (the ``min_threshold_distance``).

    Returns:
        A row-standardised :class:`libpysal.weights.W` object.
    """
    from libpysal import weights

    coords = np.asarray(coords, dtype=np.float64)
    if kind == "knn":
        k_eff = int(min(k, max(1, coords.shape[0] - 1)))
        w = weights.KNN.from_array(coords, k=k_eff)
    elif kind == "distance_band":
        if band is None:
            band = weights.min_threshold_distance(coords)
        w = weights.DistanceBand.from_array(coords, threshold=band, binary=True)
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"unknown weights kind {kind!r}")
    w.transform = "r"
    return w


def getis_ord_gi(
    values: np.ndarray,
    coords: np.ndarray,
    k: int = 8,
    *,
    permutations: int = 999,
    alpha: float = 0.01,
    kind: str = "knn",
    band: float | None = None,
    star: bool = True,
    seed: int = 42,
) -> GiStarResult:
    """Compute the Getis-Ord Gi* statistic with FDR-corrected significance.

    Args:
        values: 1-D array of the (standardised) variable per cell, e.g. the
            robust HCHO z-anomaly from :mod:`aqi_india.hotspot.climatology`.
            ``NaN`` cells are dropped from the analysis and reported as
            non-significant.
        coords: ``(n, 2)`` array of matching cell-centroid coordinates.
        k: KNN neighbour count for the weights graph. Defaults to ``8``.
        permutations: Conditional-randomisation permutations for the p-values.
        alpha: Target false-discovery rate for the Benjamini-Hochberg cutoff.
        kind: Weights graph kind (``"knn"`` or ``"distance_band"``).
        band: Distance threshold when ``kind="distance_band"``.
        star: If ``True`` (default) the focal cell is included in its own
            neighbourhood (the Gi\\* variant).
        seed: RNG seed for reproducible permutation inference.

    Returns:
        A :class:`GiStarResult`. The ``z``/``p`` arrays are aligned to the full
        input length (``NaN`` for dropped cells).
    """
    from esda.getisord import G_Local

    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = values.shape[0]
    if coords.shape[0] != n:
        raise ValueError("values and coords must have matching length")

    z_full = np.full(n, np.nan)
    p_full = np.full(n, np.nan)

    finite = np.isfinite(values)
    idx = np.flatnonzero(finite)
    if idx.size < 3:
        logger.warning("fewer than 3 finite cells; Gi* skipped")
        return GiStarResult(z_full, p_full, np.zeros(n, bool), np.zeros(n, bool), 0.0, alpha)

    w = build_weights(coords[idx], k=k, kind=kind, band=band)
    gi = G_Local(
        values[idx],
        w,
        transform="R",
        permutations=permutations,
        star=star,
        seed=seed,
    )

    z_full[idx] = gi.Zs
    # ``p_sim`` is the one-sided pseudo significance from the conditional
    # permutation simulation (proportion of permuted Gi* at least as extreme as
    # observed). Gi* hotspot detection is an upper-tail question, so this
    # one-sided value is what we FDR-control; the hot/cold split uses the sign
    # of ``Zs``. With 999 permutations the floor is 1/1000 = 0.001, fine for BH.
    p_full[idx] = np.asarray(gi.p_sim, dtype=np.float64)
    np.clip(p_full, 0.0, 1.0, out=p_full)

    reject, threshold = benjamini_hochberg(p_full, alpha=alpha)
    hot = reject & (z_full > 0)
    cold = reject & (z_full < 0)
    logger.info(
        "Gi*: %d cells, %d hotspot / %d coldspot at FDR=%.3g (p<=%.3g)",
        idx.size,
        int(hot.sum()),
        int(cold.sum()),
        alpha,
        threshold,
    )
    return GiStarResult(z_full, p_full, hot, cold, threshold, alpha)


__all__ = ["GiStarResult", "benjamini_hochberg", "build_weights", "getis_ord_gi"]
