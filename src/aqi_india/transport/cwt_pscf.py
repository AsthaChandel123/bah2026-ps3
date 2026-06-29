"""Receptor source apportionment: CWT and PSCF (Objective-2).

Two complementary trajectory-statistics map *where* the air that delivers high
HCHO to a receptor came from (blueprint method 46):

* **PSCF** (Potential Source Contribution Function) — the conditional
  probability that a grid cell lies on trajectories associated with
  *above-threshold* receptor concentrations.  ``PSCF = m / n`` where ``n`` is the
  number of trajectory endpoints in the cell and ``m`` the number of those that
  belong to a "polluted" trajectory.  An ``n``-dependent weighting damps cells
  with few endpoints.
* **CWT** (Concentration-Weighted Trajectory) — the trajectory-endpoint-weighted
  *mean concentration* attributed to each cell, giving a quantitative (not just
  probabilistic) source strength.

Both consume the trajectories from :mod:`aqi_india.transport.hysplit` plus the
receptor HCHO value that each trajectory's release corresponds to, and return a
gridded source map (and a tidy DataFrame).  Expected India result: Punjab/Haryana
maxima during the post-monsoon burning season.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..utils.geo import INDIA_BBOX
from ..utils.logging import get_logger
from .hysplit import Trajectory

logger = get_logger("transport.cwt_pscf")


@dataclass
class GriddedSourceMap:
    """A gridded CWT/PSCF source field.

    Attributes:
        lon_edges: 1-D longitude bin edges (length ``ncol + 1``).
        lat_edges: 1-D latitude bin edges (length ``nrow + 1``).
        values: ``(nrow, ncol)`` source field (CWT concentration or PSCF prob).
        counts: ``(nrow, ncol)`` trajectory-endpoint counts per cell.
        kind: ``"cwt"`` or ``"pscf"``.
    """

    lon_edges: np.ndarray
    lat_edges: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    kind: str

    @property
    def lon_centers(self) -> np.ndarray:
        """Longitude bin centres."""
        return 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])

    @property
    def lat_centers(self) -> np.ndarray:
        """Latitude bin centres."""
        return 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])

    def to_frame(self) -> pd.DataFrame:
        """Flatten the grid to a tidy ``lon, lat, value, count`` DataFrame."""
        lon_c = self.lon_centers
        lat_c = self.lat_centers
        lon2d, lat2d = np.meshgrid(lon_c, lat_c)
        return pd.DataFrame(
            {
                "lon": lon2d.ravel(),
                "lat": lat2d.ravel(),
                "value": self.values.ravel(),
                "count": self.counts.ravel(),
            }
        )


def _grid_edges(
    bbox: tuple[float, float, float, float], resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build regular lon/lat bin edges over a bbox.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)``.
        resolution: Cell size in degrees.

    Returns:
        ``(lon_edges, lat_edges)``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_edges = np.arange(min_lon, max_lon + resolution, resolution)
    lat_edges = np.arange(min_lat, max_lat + resolution, resolution)
    return lon_edges, lat_edges


def _stack_endpoints(
    trajectories: Sequence[Trajectory],
    receptor_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten all trajectory endpoints with their receptor concentrations.

    Each trajectory point is an "endpoint" that inherits the receptor
    concentration of its parent trajectory.

    Args:
        trajectories: The back-trajectories.
        receptor_values: One concentration per trajectory (same length/order).

    Returns:
        ``(lons, lats, values)`` flat arrays over all endpoints.

    Raises:
        ValueError: If the lengths differ.
    """
    if len(trajectories) != len(receptor_values):
        raise ValueError("Need one receptor value per trajectory.")
    lons, lats, vals = [], [], []
    for traj, conc in zip(trajectories, receptor_values):
        lons.append(traj.lons)
        lats.append(traj.lats)
        vals.append(np.full(traj.lons.size, conc, dtype=float))
    return (
        np.concatenate(lons),
        np.concatenate(lats),
        np.concatenate(vals),
    )


def _weight_function(counts: np.ndarray) -> np.ndarray:
    """Empirical n-dependent weighting that damps low-count cells.

    The standard PSCF/CWT weight: full weight for well-sampled cells, tapering to
    a small floor where few trajectory endpoints fall (reduces noise artefacts).

    Args:
        counts: Per-cell endpoint counts.

    Returns:
        Per-cell weights in ``[0.15, 1.0]``.
    """
    nmax = counts.max() if counts.size and counts.max() > 0 else 1.0
    w = np.ones_like(counts, dtype=float)
    w[counts <= 0.8 * nmax] = 0.7
    w[counts <= 0.5 * nmax] = 0.42
    w[counts <= 0.2 * nmax] = 0.15
    w[counts == 0] = 0.0
    return w


def cwt(
    trajectories: Sequence[Trajectory],
    receptor_values: Sequence[float],
    *,
    bbox: tuple[float, float, float, float] = INDIA_BBOX,
    resolution: float = 0.5,
    weighted: bool = True,
) -> GriddedSourceMap:
    """Concentration-Weighted Trajectory source map.

    Each cell's value is the endpoint-count-weighted mean of the receptor
    concentrations of all trajectories passing through it, optionally damped by
    the n-dependent weight function.

    Args:
        trajectories: Back-trajectories.
        receptor_values: Receptor concentration per trajectory (HCHO).
        bbox: Grid extent ``(min_lon, min_lat, max_lon, max_lat)``.
        resolution: Cell size in degrees (default 0.5).
        weighted: Apply the n-dependent low-count damping.

    Returns:
        A :class:`GriddedSourceMap` of kind ``"cwt"``.
    """
    lon_edges, lat_edges = _grid_edges(bbox, resolution)
    lons, lats, vals = _stack_endpoints(trajectories, receptor_values)

    counts, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    sums, _, _ = np.histogram2d(
        lats, lons, bins=[lat_edges, lon_edges], weights=vals
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        cwt_field = np.where(counts > 0, sums / counts, np.nan)

    if weighted:
        cwt_field = cwt_field * _weight_function(counts)

    logger.info(
        "CWT over %d endpoints on %dx%d grid (res=%.2f deg)",
        vals.size,
        len(lat_edges) - 1,
        len(lon_edges) - 1,
        resolution,
    )
    return GriddedSourceMap(lon_edges, lat_edges, cwt_field, counts, "cwt")


def pscf(
    trajectories: Sequence[Trajectory],
    receptor_values: Sequence[float],
    *,
    bbox: tuple[float, float, float, float] = INDIA_BBOX,
    resolution: float = 0.5,
    threshold: "float | None" = None,
    threshold_quantile: float = 0.75,
    weighted: bool = True,
) -> GriddedSourceMap:
    """Potential Source Contribution Function probability map.

    A trajectory is "polluted" if its receptor value exceeds ``threshold`` (or
    the ``threshold_quantile`` of the receptor values when ``threshold`` is
    ``None``).  ``PSCF = polluted_endpoints / total_endpoints`` per cell, with the
    optional n-dependent weighting.

    Args:
        trajectories: Back-trajectories.
        receptor_values: Receptor concentration per trajectory.
        bbox: Grid extent.
        resolution: Cell size in degrees.
        threshold: Absolute concentration threshold; if ``None`` use the quantile.
        threshold_quantile: Quantile used when ``threshold`` is ``None``.
        weighted: Apply n-dependent low-count damping.

    Returns:
        A :class:`GriddedSourceMap` of kind ``"pscf"`` with probabilities in
        ``[0, 1]``.
    """
    receptor_values = np.asarray(receptor_values, dtype=float)
    if threshold is None:
        threshold = float(np.quantile(receptor_values, threshold_quantile))
    polluted_mask = receptor_values > threshold

    lon_edges, lat_edges = _grid_edges(bbox, resolution)
    lons, lats, _ = _stack_endpoints(trajectories, receptor_values)
    # Build the polluted-endpoint flag aligned to the stacked endpoints.
    flags = []
    for traj, is_poll in zip(trajectories, polluted_mask):
        flags.append(np.full(traj.lons.size, float(is_poll)))
    flags = np.concatenate(flags)

    counts, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    polluted, _, _ = np.histogram2d(
        lats, lons, bins=[lat_edges, lon_edges], weights=flags
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        pscf_field = np.where(counts > 0, polluted / counts, np.nan)

    if weighted:
        pscf_field = pscf_field * _weight_function(counts)

    logger.info(
        "PSCF threshold=%.3g (%d/%d polluted trajectories)",
        threshold,
        int(polluted_mask.sum()),
        polluted_mask.size,
    )
    return GriddedSourceMap(lon_edges, lat_edges, pscf_field, counts, "pscf")


def peak_source_cell(source_map: GriddedSourceMap) -> dict:
    """Return the coordinates and value of the strongest source cell.

    Args:
        source_map: A CWT or PSCF map.

    Returns:
        Dict with ``lon``, ``lat``, ``value`` and ``count`` of the argmax cell.
    """
    values = source_map.values
    if not np.any(np.isfinite(values)):
        return {"lon": float("nan"), "lat": float("nan"), "value": float("nan")}
    idx = np.unravel_index(np.nanargmax(values), values.shape)
    return {
        "lon": float(source_map.lon_centers[idx[1]]),
        "lat": float(source_map.lat_centers[idx[0]]),
        "value": float(values[idx]),
        "count": float(source_map.counts[idx]),
    }


__all__ = [
    "GriddedSourceMap",
    "cwt",
    "peak_source_cell",
    "pscf",
]
