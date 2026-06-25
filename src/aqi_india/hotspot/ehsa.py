"""Emerging Hot Spot Analysis (EHSA) over a HCHO space-time cube.

EHSA is the headline *temporal* deliverable for Objective-2: it separates the
**persistent** IGP-industrial HCHO hotspots from the **new / intensifying**
biomass-burning hotspots that appear only during the post-monsoon and
pre-monsoon burning seasons (blueprint method 37).

The procedure, mirroring the ArcGIS EHSA tool:

1.  Slice the cube into time bins (e.g. season-years) and run Getis-Ord Gi* on
    each bin, producing a per-cell, per-bin grid of z-scores and a boolean
    "significant hotspot in this bin" series.
2.  For each cell, run a **trend test on the Gi* z-score time series**:
    * the **modified Mann-Kendall test (Hamed-Rao variance correction)** via
      :mod:`pymannkendall`, which accounts for serial autocorrelation that would
      otherwise inflate significance, and
    * **Sen's (Theil) slope** as the robust magnitude of the trend.
3.  Combine the trend direction, its significance, and how often / when the cell
    was a significant hotspot into one of the canonical EHSA categories.

Output categories (per cell):
    ``no_pattern`` · ``new`` · ``consecutive`` · ``intensifying`` ·
    ``persistent`` · ``diminishing`` · ``sporadic`` · ``oscillating`` ·
    ``historical``

``pymannkendall`` is in the light dependency set; ``esda``/``libpysal`` are
imported lazily through :mod:`aqi_india.hotspot.getis_ord`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.logging import get_logger
from .getis_ord import getis_ord_gi

logger = get_logger("hotspot.ehsa")

#: Canonical EHSA pattern labels.
EHSA_CATEGORIES: tuple[str, ...] = (
    "no_pattern",
    "new",
    "consecutive",
    "intensifying",
    "persistent",
    "diminishing",
    "sporadic",
    "oscillating",
    "historical",
)


@dataclass
class EhsaResult:
    """Container for an Emerging Hot Spot Analysis run.

    Attributes:
        category: Object array of EHSA labels, one per cell.
        trend_z: Mann-Kendall normalised test statistic per cell.
        trend_p: Mann-Kendall p-value per cell.
        sen_slope: Sen's slope of the Gi* z-series per cell.
        hot_fraction: Fraction of bins in which the cell was a significant
            hotspot.
        gi_z: ``(n_bins, n_cells)`` matrix of per-bin Gi* z-scores.
        gi_sig: ``(n_bins, n_cells)`` boolean per-bin hotspot mask.
        bin_labels: The ordered time-bin labels.
    """

    category: np.ndarray
    trend_z: np.ndarray
    trend_p: np.ndarray
    sen_slope: np.ndarray
    hot_fraction: np.ndarray
    gi_z: np.ndarray
    gi_sig: np.ndarray
    bin_labels: list


def _modified_mann_kendall(series: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Run the Hamed-Rao modified Mann-Kendall test on one cell's series.

    Args:
        series: 1-D time series of Gi* z-scores.
        alpha: Significance level passed to ``pymannkendall``.

    Returns:
        ``(z_stat, p_value, sen_slope)``. Returns ``(nan, nan, nan)`` for series
        too short or fully constant/NaN to test.
    """
    import pymannkendall as mk

    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 4 or np.allclose(s, s[0]):
        return np.nan, np.nan, 0.0 if s.size else np.nan
    try:
        res = mk.hamed_rao_modification_test(s, alpha=alpha)
    except (ZeroDivisionError, ValueError):  # pragma: no cover - degenerate series
        return np.nan, np.nan, np.nan
    return float(res.z), float(res.p), float(res.slope)


def _classify(
    hot_series: np.ndarray,
    trend_p: float,
    trend_dir: int,
    *,
    alpha: float,
) -> str:
    """Map one cell's hotspot history + trend to an EHSA category.

    Args:
        hot_series: Boolean per-bin significant-hotspot series (time-ordered).
        trend_p: Mann-Kendall p-value (``NaN`` if untestable).
        trend_dir: Sign of the trend (``+1`` increasing, ``-1`` decreasing,
            ``0`` none).
        alpha: Significance threshold for the trend.

    Returns:
        One of :data:`EHSA_CATEGORIES`.
    """
    hot = np.asarray(hot_series, dtype=bool)
    n = hot.size
    if n == 0 or not hot.any():
        return "no_pattern"

    frac = hot.mean()
    final_hot = bool(hot[-1])
    # Length of the current run of hot bins ending at the final bin.
    run = 0
    for v in hot[::-1]:
        if v:
            run += 1
        else:
            break

    sig_up = np.isfinite(trend_p) and trend_p < alpha and trend_dir > 0
    sig_down = np.isfinite(trend_p) and trend_p < alpha and trend_dir < 0

    # Persistent: hot in (almost) every bin, no significant trend.
    if frac >= 0.9 and not (sig_up or sig_down):
        return "persistent"
    # Intensifying: hot most of the time and significantly increasing, still hot.
    if final_hot and frac >= 0.5 and sig_up:
        return "intensifying"
    # Diminishing: was hot a lot, significantly decreasing.
    if frac >= 0.5 and sig_down:
        return "diminishing"
    # Historical: used to be hot, not hot now, was hot in a clear majority.
    if not final_hot and frac >= 0.5:
        return "historical"
    # New: hot only in the final bin.
    if final_hot and run == 1 and hot[:-1].sum() == 0:
        return "new"
    # Consecutive: a single uninterrupted run ending now, a minority of history.
    if final_hot and run == hot.sum() and run >= 2 and frac < 0.5:
        return "consecutive"
    # Oscillating: hot now but with off-and-on history (down-trend signal present).
    if final_hot and sig_down:
        return "oscillating"
    # Sporadic: intermittently hot, never the majority, not currently a run.
    return "sporadic"


def emerging_hotspot_analysis(
    cube: np.ndarray,
    coords: np.ndarray,
    bin_labels: list,
    k: int = 8,
    *,
    permutations: int = 999,
    gi_alpha: float = 0.01,
    trend_alpha: float = 0.05,
    standardize_bins: bool = True,
    seed: int = 42,
) -> EhsaResult:
    """Run Emerging Hot Spot Analysis over a stacked space-time cube.

    Args:
        cube: ``(n_bins, n_cells)`` array of the variable per time bin and cell
            — typically the seasonal-binned HCHO column (or its spatial anomaly)
            averaged within each bin.
        coords: ``(n_cells, 2)`` array of cell-centroid coordinates.
        bin_labels: Ordered labels for the ``n_bins`` time bins (any type).
        k: KNN neighbour count for the Gi* weights graph.
        permutations: Gi* permutations per bin.
        gi_alpha: FDR level for per-bin Gi* significance.
        trend_alpha: Significance level for the Mann-Kendall trend test.
        standardize_bins: If ``True`` (default) each bin's field is converted to
            a robust *spatial* anomaly before Gi* so the India-wide gradient is
            removed and a persistent blob still registers each bin. Set ``False``
            if the caller already supplies spatially-standardised fields.
        seed: RNG seed (offset per bin for independent inference).

    Returns:
        An :class:`EhsaResult` with a per-cell category and the supporting
        trend statistics.
    """
    from .climatology import spatial_robust_anomaly

    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 2:
        raise ValueError("cube must be 2-D (n_bins, n_cells)")
    n_bins, n_cells = cube.shape
    if len(bin_labels) != n_bins:
        raise ValueError("bin_labels length must match cube's first axis")

    gi_z = np.full((n_bins, n_cells), np.nan)
    gi_sig = np.zeros((n_bins, n_cells), dtype=bool)

    for b in range(n_bins):
        bin_field = spatial_robust_anomaly(cube[b]) if standardize_bins else cube[b]
        res = getis_ord_gi(
            bin_field,
            coords,
            k=k,
            permutations=permutations,
            alpha=gi_alpha,
            seed=seed + b,
        )
        gi_z[b] = res.z
        gi_sig[b] = res.sig

    trend_z = np.full(n_cells, np.nan)
    trend_p = np.full(n_cells, np.nan)
    sen_slope = np.full(n_cells, np.nan)
    category = np.empty(n_cells, dtype=object)
    hot_fraction = gi_sig.mean(axis=0)

    for c in range(n_cells):
        z, p, slope = _modified_mann_kendall(gi_z[:, c], trend_alpha)
        trend_z[c] = z
        trend_p[c] = p
        sen_slope[c] = slope
        direction = int(np.sign(slope)) if np.isfinite(slope) else 0
        category[c] = _classify(
            gi_sig[:, c], p, direction, alpha=trend_alpha
        )

    counts = {cat: int((category == cat).sum()) for cat in EHSA_CATEGORIES}
    logger.info("EHSA over %d bins x %d cells: %s", n_bins, n_cells, counts)

    return EhsaResult(
        category=category,
        trend_z=trend_z,
        trend_p=trend_p,
        sen_slope=sen_slope,
        hot_fraction=hot_fraction,
        gi_z=gi_z,
        gi_sig=gi_sig,
        bin_labels=list(bin_labels),
    )


__all__ = ["EHSA_CATEGORIES", "EhsaResult", "emerging_hotspot_analysis"]
