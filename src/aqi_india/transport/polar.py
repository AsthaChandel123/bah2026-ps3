"""Bivariate polar / pollution-rose plots (Objective-2 directional fingerprint).

Following openair's ``polarPlot``/``pollutionRose``, this module bins a receptor
HCHO series by **wind direction** and **wind speed** and surfaces the directional
signature of the source (blueprint method 47).  A high-HCHO lobe to the
north-west corroborates the trajectory/CWT finding that Punjab/Haryana stubble
burning is the dominant source.

The core :func:`bivariate_polar` is pure NumPy (returns the binned grid) so it is
testable without a display; :func:`plot_polar` renders it with matplotlib on a
polar axis, and :func:`pollution_rose` produces the stacked wind-rose variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

logger = get_logger("transport.polar")


@dataclass
class PolarGrid:
    """Binned bivariate-polar field.

    Attributes:
        direction_edges: Wind-direction bin edges (deg, 0-360).
        speed_edges: Wind-speed bin edges (m/s).
        statistic: ``(n_speed_bins, n_dir_bins)`` aggregated concentration.
        counts: ``(n_speed_bins, n_dir_bins)`` sample counts per bin.
        stat_name: The aggregation used (``"mean"`` etc.).
    """

    direction_edges: np.ndarray
    speed_edges: np.ndarray
    statistic: np.ndarray
    counts: np.ndarray
    stat_name: str


def uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (u, v) wind components to speed and meteorological direction.

    Meteorological convention: direction is the bearing the wind blows **from**,
    in degrees clockwise from north.

    Args:
        u: Eastward wind component (m/s).
        v: Northward wind component (m/s).

    Returns:
        ``(speed, direction)`` arrays; direction in ``[0, 360)``.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, direction


def bivariate_polar(
    speed: np.ndarray,
    direction: np.ndarray,
    concentration: np.ndarray,
    *,
    n_dir_bins: int = 16,
    n_speed_bins: int = 6,
    speed_max: "float | None" = None,
    statistic: str = "mean",
) -> PolarGrid:
    """Bin a concentration series by wind speed and direction.

    Args:
        speed: Wind speed per observation (m/s).
        direction: Wind direction per observation (deg, met "from" convention).
        concentration: Receptor concentration (e.g. HCHO) per observation.
        n_dir_bins: Number of direction bins around the compass.
        n_speed_bins: Number of speed bins.
        speed_max: Upper speed edge; defaults to the 99th percentile of ``speed``.
        statistic: ``"mean"``, ``"median"`` or ``"max"`` aggregation per bin.

    Returns:
        A :class:`PolarGrid`.

    Raises:
        ValueError: If the inputs differ in length or ``statistic`` is unknown.
    """
    speed = np.asarray(speed, dtype=float)
    direction = np.asarray(direction, dtype=float) % 360.0
    concentration = np.asarray(concentration, dtype=float)
    if not (speed.size == direction.size == concentration.size):
        raise ValueError("speed, direction and concentration must be equal length.")
    if statistic not in {"mean", "median", "max"}:
        raise ValueError(f"Unknown statistic '{statistic}'.")

    if speed_max is None:
        finite = speed[np.isfinite(speed)]
        speed_max = float(np.quantile(finite, 0.99)) if finite.size else 1.0
        speed_max = max(speed_max, 1e-6)

    dir_edges = np.linspace(0.0, 360.0, n_dir_bins + 1)
    speed_edges = np.linspace(0.0, speed_max, n_speed_bins + 1)

    stat = np.full((n_speed_bins, n_dir_bins), np.nan)
    counts = np.zeros((n_speed_bins, n_dir_bins))

    di = np.clip(np.digitize(direction, dir_edges) - 1, 0, n_dir_bins - 1)
    si = np.clip(np.digitize(speed, speed_edges) - 1, 0, n_speed_bins - 1)

    valid = np.isfinite(concentration) & np.isfinite(speed) & np.isfinite(direction)
    for s in range(n_speed_bins):
        for d in range(n_dir_bins):
            sel = valid & (si == s) & (di == d)
            counts[s, d] = int(sel.sum())
            if sel.any():
                vals = concentration[sel]
                if statistic == "mean":
                    stat[s, d] = float(np.mean(vals))
                elif statistic == "median":
                    stat[s, d] = float(np.median(vals))
                else:
                    stat[s, d] = float(np.max(vals))

    logger.info(
        "bivariate polar: %d obs -> %dx%d (speed x dir) bins",
        int(valid.sum()),
        n_speed_bins,
        n_dir_bins,
    )
    return PolarGrid(dir_edges, speed_edges, stat, counts, statistic)


def dominant_direction(grid: PolarGrid) -> dict:
    """Direction sector carrying the highest mean concentration.

    Args:
        grid: A :class:`PolarGrid`.

    Returns:
        Dict with ``direction_deg`` (sector centre), ``compass`` and ``value``.
    """
    # Count-weighted mean concentration per direction sector.
    with np.errstate(invalid="ignore"):
        weighted = np.nansum(grid.statistic * grid.counts, axis=0)
        totals = np.nansum(grid.counts, axis=0)
        per_dir = np.where(totals > 0, weighted / totals, np.nan)
    if not np.any(np.isfinite(per_dir)):
        return {"direction_deg": float("nan"), "compass": "?", "value": float("nan")}
    d = int(np.nanargmax(per_dir))
    centre = 0.5 * (grid.direction_edges[d] + grid.direction_edges[d + 1])
    return {
        "direction_deg": float(centre),
        "compass": _compass(centre),
        "value": float(per_dir[d]),
    }


def _compass(bearing: float) -> str:
    """8-point compass label for a bearing in degrees."""
    if not np.isfinite(bearing):
        return "?"
    points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return points[int((bearing % 360) / 45.0 + 0.5) % 8]


def plot_polar(
    grid: PolarGrid,
    *,
    title: str = "HCHO bivariate polar plot",
    cmap: str = "viridis",
    ax: "Optional[object]" = None,
) -> "Figure":
    """Render a :class:`PolarGrid` on a matplotlib polar axis.

    Args:
        grid: The binned field from :func:`bivariate_polar`.
        title: Plot title.
        cmap: Matplotlib colormap name.
        ax: Optional pre-created polar axis; a new figure is made if ``None``.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure

    # Meteorological direction increases clockwise from north.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_edges = np.radians(grid.direction_edges)
    r_edges = grid.speed_edges
    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)
    mesh = ax.pcolormesh(theta_grid, r_grid, grid.statistic, cmap=cmap, shading="flat")
    fig.colorbar(mesh, ax=ax, label="HCHO (mol/m2)", pad=0.1)
    ax.set_title(title)
    return fig


def pollution_rose(
    speed: np.ndarray,
    direction: np.ndarray,
    concentration: np.ndarray,
    *,
    n_dir_bins: int = 16,
    bins: "Optional[list[float]]" = None,
    title: str = "HCHO pollution rose",
) -> "Figure":
    """Stacked-bar pollution rose: HCHO contribution by wind direction.

    Args:
        speed: Wind speed per observation (m/s) — accepted for API parity; the
            rose stacks by concentration bin, not speed.
        direction: Wind direction per observation (deg).
        concentration: Receptor concentration per observation.
        n_dir_bins: Number of direction sectors.
        bins: Concentration bin edges; defaults to data quartiles.
        title: Plot title.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    direction = np.asarray(direction, dtype=float) % 360.0
    concentration = np.asarray(concentration, dtype=float)
    valid = np.isfinite(direction) & np.isfinite(concentration)
    direction, concentration = direction[valid], concentration[valid]

    if bins is None:
        qs = np.quantile(concentration, [0.25, 0.5, 0.75]) if concentration.size else [0]
        bins = [-np.inf, *qs, np.inf]

    dir_edges = np.linspace(0.0, 360.0, n_dir_bins + 1)
    dir_centres = np.radians(0.5 * (dir_edges[:-1] + dir_edges[1:]))
    di = np.clip(np.digitize(direction, dir_edges) - 1, 0, n_dir_bins - 1)
    ci = np.clip(np.digitize(concentration, bins) - 1, 0, len(bins) - 2)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    width = np.radians(360.0 / n_dir_bins)

    bottoms = np.zeros(n_dir_bins)
    cmap = plt.get_cmap("YlOrRd")
    for c in range(len(bins) - 1):
        heights = np.array(
            [np.sum((di == d) & (ci == c)) for d in range(n_dir_bins)], dtype=float
        )
        ax.bar(
            dir_centres,
            heights,
            width=width,
            bottom=bottoms,
            color=cmap((c + 1) / (len(bins) - 1)),
            edgecolor="white",
            linewidth=0.3,
            label=f"bin {c + 1}",
        )
        bottoms += heights

    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize="x-small")
    return fig


__all__ = [
    "PolarGrid",
    "bivariate_polar",
    "dominant_direction",
    "plot_polar",
    "pollution_rose",
    "uv_to_speed_dir",
]
