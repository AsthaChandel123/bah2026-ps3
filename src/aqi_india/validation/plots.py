"""Validation figures — Taylor diagram, 1:1 hexbin, residual map, ladder bar.

The validation page is one of the most score-relevant deliverables, so these
figures are first-class. Everything here is pure matplotlib (plus an optional
GeoPandas/Shapely overlay for the India outline) — no exotic plotting deps — so
the figures render under the light dependency set.

Functions:
    * :func:`taylor_diagram` — a standalone, hand-built Taylor diagram (no
      external Taylor package) plotting one marker per model, summarising
      correlation + normalised standard deviation + (implied) centred RMSD.
    * :func:`one_to_one_hexbin` — density scatter of predicted vs observed with
      the 1:1 line and an R / RMSE / MAE annotation box.
    * :func:`residual_map` — per-CPCB-station residual (pred - obs) bubbles over
      an India outline, diverging colour scale.
    * :func:`cv_ladder_bar` — bar chart of the CV-ladder skill per rung, the
      "leakage-gap" story.

Each function accepts an optional ``ax`` to compose into a larger figure and an
optional ``save_path``; with no ``save_path`` the helper still returns the
Matplotlib ``Figure``/``Axes`` so callers can place it. The convenience
``DEFAULT_FIG_DIR`` points at ``reports/figures``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .metrics import mae as _mae
from .metrics import pearson_r as _pearson_r
from .metrics import rmse as _rmse

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Default output directory for saved validation figures.
DEFAULT_FIG_DIR = Path("reports/figures")


def _save(fig: "Figure", save_path: str | Path | None) -> None:
    """Save ``fig`` to ``save_path`` (creating parent dirs) if a path is given."""
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _clean_xy(
    y_true: Any, y_pred: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten and drop non-finite pairs (matches the metrics module)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


# --------------------------------------------------------------------------- #
# Taylor diagram
# --------------------------------------------------------------------------- #
def taylor_diagram(
    stats: "list[dict[str, Any]] | pd.DataFrame",
    *,
    ref_std: float | None = None,
    ax: "Axes | None" = None,
    save_path: str | Path | None = None,
    title: str = "Taylor diagram",
) -> "Figure":
    """Draw a standalone Taylor diagram (one marker per model).

    A Taylor diagram encodes three statistics of each model against the
    reference in a single polar quarter-plane: the angle is ``arccos(r)``
    (correlation), the radius is the model's standard deviation, and the
    distance to the reference point equals the centred RMSD. It is the standard
    multi-model summary requested in the validation plan.

    Args:
        stats: One entry per model, each providing ``std`` (the model's standard
            deviation, ideally normalised by the observed std) and ``r`` (the
            Pearson correlation with the observations); an optional ``name``
            labels the marker. Accepts a list of dicts or a DataFrame with these
            columns.
        ref_std: Standard deviation of the reference (observations). Defaults to
            ``1.0``, i.e. ``std`` values are assumed already normalised.
        ax: Optional polar Axes to draw into; created if omitted.
        save_path: If given, the figure is written here.
        title: Figure title.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    if hasattr(stats, "to_dict"):  # DataFrame
        records = stats.to_dict(orient="records")  # type: ignore[union-attr]
    else:
        records = list(stats)  # type: ignore[arg-type]

    ref = 1.0 if ref_std is None else float(ref_std)
    std_vals = [float(s["std"]) for s in records]
    smax = max([ref, *std_vals]) * 1.3 if std_vals else ref * 1.3

    if ax is None:
        fig = plt.figure(figsize=(6.5, 6.0))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure

    # Quarter circle: theta in [0, pi/2] maps r in [1, 0].
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rlim(0, smax)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # Correlation gridlines (the angular axis).
    corr_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_thetagrids(
        np.degrees(np.arccos(corr_ticks)),
        labels=[f"{c:g}" for c in corr_ticks],
    )
    ax.text(
        np.radians(45),
        smax * 1.12,
        "Correlation",
        ha="center",
        va="center",
        rotation=-45,
        fontsize=10,
    )
    ax.set_xlabel("Standard deviation")

    # Reference standard-deviation arc and the reference point itself.
    theta_arc = np.linspace(0, np.pi / 2, 100)
    ax.plot(theta_arc, np.full_like(theta_arc, ref), "k--", lw=1, alpha=0.6)
    ax.plot(0, ref, "k*", ms=14, label="Reference")

    # Centred-RMSD contours (circles centred on the reference point).
    rms_levels = np.linspace(0, smax, 5)[1:]
    angle = np.linspace(0, 2 * np.pi, 200)
    for lev in rms_levels:
        x = ref + lev * np.cos(angle)
        y = lev * np.sin(angle)
        rr = np.sqrt(x**2 + y**2)
        tt = np.arctan2(y, x)
        keep = (tt >= 0) & (tt <= np.pi / 2) & (rr <= smax)
        ax.plot(tt[keep], rr[keep], color="0.7", ls=":", lw=0.8)

    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(records):
        r = float(np.clip(s["r"], -1.0, 1.0))
        theta = np.arccos(r)
        ax.plot(
            theta,
            float(s["std"]),
            "o",
            ms=9,
            color=cmap(i % 10),
            label=str(s.get("name", f"model {i}")),
        )

    ax.set_title(title, pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=8)
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# 1:1 hexbin density scatter
# --------------------------------------------------------------------------- #
def one_to_one_hexbin(
    y_true: Any,
    y_pred: Any,
    *,
    gridsize: int = 40,
    ax: "Axes | None" = None,
    save_path: str | Path | None = None,
    title: str = "Predicted vs observed",
    units: str = "",
) -> "Figure":
    """Hexbin density scatter of predicted vs observed with R/RMSE/MAE.

    A density (hexbin) scatter avoids the over-plotting that hides structure in
    a plain scatter of many station-days; the 1:1 line and the metric box make
    bias and compression immediately legible.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        gridsize: Hexbin resolution.
        ax: Optional Axes to draw into; created if omitted.
        save_path: If given, the figure is written here.
        title: Plot title.
        units: Optional unit string appended to the axis labels.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    yt, yp = _clean_xy(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.6))
    else:
        fig = ax.figure

    if yt.size:
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        hb = ax.hexbin(yt, yp, gridsize=gridsize, mincnt=1, cmap="viridis")
        fig.colorbar(hb, ax=ax, label="count")
    else:
        lo, hi = 0.0, 1.0

    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    suffix = f" ({units})" if units else ""
    ax.set_xlabel(f"Observed{suffix}")
    ax.set_ylabel(f"Predicted{suffix}")
    ax.set_title(title)

    r = _pearson_r(yt, yp)
    rm = _rmse(yt, yp)
    ma = _mae(yt, yp)
    ax.text(
        0.05,
        0.95,
        f"n = {yt.size}\nR = {r:.3f}\nRMSE = {rm:.2f}\nMAE = {ma:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8},
    )
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# Residual map over India
# --------------------------------------------------------------------------- #
def residual_map(
    stations: "pd.DataFrame",
    residuals: Any,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    ax: "Axes | None" = None,
    save_path: str | Path | None = None,
    title: str = "Prediction residuals (pred - obs)",
    cmap: str = "RdBu_r",
    max_abs: float | None = None,
) -> "Figure":
    """Plot per-CPCB-station residuals as coloured bubbles over India.

    Spatially mapping the residuals reveals *where* the model is biased — a
    failure invisible in scalar metrics (e.g. systematic IGP under-prediction).
    The India outline is drawn from
    :func:`aqi_india.utils.geo.get_india_boundary` when GeoPandas is available,
    otherwise the INDIA bounding box frames the plot.

    Args:
        stations: Table with at least the ``lat_col`` / ``lon_col`` columns
            (one row per station, aligned with ``residuals``).
        residuals: Per-station residual ``pred - obs`` (same length / order as
            ``stations``).
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        ax: Optional Axes to draw into; created if omitted.
        save_path: If given, the figure is written here.
        title: Plot title.
        cmap: Diverging colormap name.
        max_abs: Symmetric colour limit; defaults to the 98th percentile of
            ``|residual|``.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    from ..utils.geo import INDIA_BBOX

    lat = np.asarray(stations[lat_col], dtype=np.float64)
    lon = np.asarray(stations[lon_col], dtype=np.float64)
    res = np.asarray(residuals, dtype=np.float64).ravel()
    if not (len(lat) == len(lon) == len(res)):
        raise ValueError("stations and residuals must have matching lengths")

    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(res)
    lat, lon, res = lat[mask], lon[mask], res[mask]

    if max_abs is None:
        max_abs = float(np.percentile(np.abs(res), 98)) if res.size else 1.0
        max_abs = max_abs or 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 7.5))
    else:
        fig = ax.figure

    # India outline (best effort; falls back to the bounding box).
    try:
        from ..utils.geo import get_india_boundary

        boundary = get_india_boundary(simplify=True)
        boundary.boundary.plot(ax=ax, color="0.4", lw=0.8)
    except Exception:  # pragma: no cover - offline / geopandas missing
        pass

    sizes = 25 + 55 * (np.abs(res) / max_abs).clip(0, 1)
    sc = ax.scatter(
        lon,
        lat,
        c=res,
        s=sizes,
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        edgecolor="k",
        linewidth=0.3,
        alpha=0.9,
    )
    fig.colorbar(sc, ax=ax, label="residual (pred - obs)", shrink=0.8)

    min_lon, min_lat, max_lon, max_lat = INDIA_BBOX
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# CV-ladder bar chart
# --------------------------------------------------------------------------- #
def cv_ladder_bar(
    table: "pd.DataFrame",
    *,
    metric: str = "r",
    ax: "Axes | None" = None,
    save_path: str | Path | None = None,
    title: str = "CV ladder — skill vs split rigour",
) -> "Figure":
    """Bar chart of out-of-fold skill per CV rung (the leakage-gap story).

    Ordered leakiest-to-strictest, the descending bars make the spatial +
    temporal leakage of random CV visible at a glance: the height drop from
    ``random_kfold`` to ``spatiotemporal_blocked_cv`` is the inflation a naive
    split was hiding.

    Args:
        table: The DataFrame returned by
            :func:`aqi_india.validation.cv.run_cv_ladder`. Must contain a
            ``scheme`` column and an ``oof_<metric>`` column.
        metric: Which out-of-fold metric column to plot (default ``r``).
        ax: Optional Axes to draw into; created if omitted.
        save_path: If given, the figure is written here.
        title: Plot title.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.

    Raises:
        KeyError: If the expected columns are absent from ``table``.
    """
    import matplotlib.pyplot as plt

    col = f"oof_{metric}"
    if "scheme" not in table.columns or col not in table.columns:
        raise KeyError(f"table must contain 'scheme' and '{col}' columns")

    schemes = list(table["scheme"])
    values = np.asarray(table[col], dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
    else:
        fig = ax.figure

    # Colour ramp from leaky (red) to strict (green).
    cmap = plt.get_cmap("RdYlGn")
    n = max(len(schemes), 1)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    bars = ax.bar(range(n), values, color=colors, edgecolor="k", linewidth=0.5)

    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(schemes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"out-of-fold {metric}")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.6)

    # Annotate the headline leakage gap (random -> strictest) if present.
    if "random_kfold" in schemes and len(values) > 1:
        ref = values[schemes.index("random_kfold")]
        last = values[-1]
        if np.isfinite(ref) and np.isfinite(last):
            ax.annotate(
                f"leakage gap = {ref - last:.3f}",
                xy=(n - 1, last),
                xytext=(0.5, max(values) * 0.5 if np.isfinite(max(values)) else 0.5),
                fontsize=9,
                color="darkred",
            )

    _save(fig, save_path)
    return fig


__all__ = [
    "DEFAULT_FIG_DIR",
    "taylor_diagram",
    "one_to_one_hexbin",
    "residual_map",
    "cv_ladder_bar",
]
