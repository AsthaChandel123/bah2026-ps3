"""Local Moran's I (LISA) clustering and outlier classification.

Local Indicators of Spatial Association decompose Moran's I into a per-cell
contribution and assign each significant cell to one of four quadrants of the
Moran scatterplot:

* **HH** (High-High) — a high value surrounded by high neighbours: the *core* of
  a hotspot cluster (consensus vote alongside Getis-Ord Gi*).
* **LL** (Low-Low)   — a low value surrounded by low neighbours (cold core).
* **HL** (High-Low)  — a high value surrounded by low neighbours: a *spatial
  outlier*. For HCHO this flags a **fresh, isolated fire/point source** that has
  not yet diffused into its surroundings — the early-warning signal Gi* (which
  rewards spatial smoothness) tends to under-weight.
* **LH** (Low-High)  — a low value surrounded by high neighbours.

This module wraps :class:`esda.Moran_Local`, returning the HH cluster mask and
the HL fresh-fire outlier mask (blueprint method 35). ``esda``/``libpysal`` are
in the light dependency set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.logging import get_logger
from .getis_ord import build_weights

logger = get_logger("hotspot.lisa")

#: ``esda`` Moran_Local quadrant codes.
_HH, _LH, _LL, _HL = 1, 2, 3, 4

#: Human-readable label per quadrant code (0 = not significant).
QUADRANT_LABELS: dict[int, str] = {0: "ns", _HH: "HH", _LH: "LH", _LL: "LL", _HL: "HL"}


@dataclass
class LisaResult:
    """Container for a Local Moran's I run.

    Attributes:
        i_local: Local Moran's I value per observation (``NaN`` where dropped).
        p: Permutation p-values per observation.
        quadrant: Integer quadrant code per observation (0 = not significant,
            else one of :data:`QUADRANT_LABELS`).
        hh_mask: Boolean mask of significant High-High cluster cells (hotspot
            cores) at ``alpha``.
        hl_mask: Boolean mask of significant High-Low spatial outliers (fresh
            fire / point-source flags) at ``alpha``.
        alpha: Significance level used.
    """

    i_local: np.ndarray
    p: np.ndarray
    quadrant: np.ndarray
    hh_mask: np.ndarray
    hl_mask: np.ndarray
    alpha: float


def local_morans_i(
    values: np.ndarray,
    coords: np.ndarray,
    k: int = 8,
    *,
    permutations: int = 999,
    alpha: float = 0.05,
    kind: str = "knn",
    band: float | None = None,
    seed: int = 42,
) -> LisaResult:
    """Compute Local Moran's I and classify HH cores and HL outliers.

    Args:
        values: 1-D array of the (standardised) variable per cell, e.g. the
            robust HCHO z-anomaly. ``NaN`` cells are dropped and reported
            non-significant.
        coords: ``(n, 2)`` array of matching cell-centroid coordinates.
        k: KNN neighbour count for the weights graph.
        permutations: Conditional-randomisation permutations for the p-values.
        alpha: Significance level for the cluster/outlier masks.
        kind: Weights graph kind (``"knn"`` or ``"distance_band"``).
        band: Distance threshold when ``kind="distance_band"``.
        seed: RNG seed for reproducible permutation inference.

    Returns:
        A :class:`LisaResult` whose arrays are aligned to the full input length.
    """
    from esda.moran import Moran_Local

    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = values.shape[0]
    if coords.shape[0] != n:
        raise ValueError("values and coords must have matching length")

    i_full = np.full(n, np.nan)
    p_full = np.full(n, np.nan)
    q_full = np.zeros(n, dtype=int)

    finite = np.isfinite(values)
    idx = np.flatnonzero(finite)
    if idx.size < 3:
        logger.warning("fewer than 3 finite cells; LISA skipped")
        return LisaResult(
            i_full, p_full, q_full, np.zeros(n, bool), np.zeros(n, bool), alpha
        )

    w = build_weights(coords[idx], k=k, kind=kind, band=band)
    lm = Moran_Local(
        values[idx],
        w,
        transformation="r",
        permutations=permutations,
        seed=seed,
    )

    i_full[idx] = lm.Is
    p_full[idx] = np.asarray(lm.p_sim, dtype=np.float64)

    sig_local = p_full[idx] < alpha
    q_local = np.where(sig_local, lm.q, 0)
    q_full[idx] = q_local

    hh_mask = q_full == _HH
    hl_mask = q_full == _HL
    logger.info(
        "LISA: %d cells, %d HH cores / %d HL outliers at alpha=%.3g",
        idx.size,
        int(hh_mask.sum()),
        int(hl_mask.sum()),
        alpha,
    )
    return LisaResult(i_full, p_full, q_full, hh_mask, hl_mask, alpha)


__all__ = ["LisaResult", "QUADRANT_LABELS", "local_morans_i"]
