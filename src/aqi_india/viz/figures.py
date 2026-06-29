"""Flagship report / PPT figures for the BAH 2026 PS3 product.

This module renders the polished static figures that carry the bulk of the
visualisation score:

* :func:`plot_aqi_map` — a filled India AQI raster on the six-band CPCB colour
  scale, with the state/India boundary overlay, a discrete category colourbar,
  an Indo-Gangetic-Plain (IGP) highlight box and a title.
* :func:`plot_responsible_pollutant_map` — which pollutant drives the AQI in
  each cell (categorical map).
* :func:`plot_hcho_hotspots` — the HCHO column field with confirmed hotspot
  polygons, FIRMS fire points and wind arrows, highlighting the Punjab/Haryana
  stubble belt and the central/Himalayan forest belt (Objective-2 hero figure).
* :func:`plot_timeseries` — a tidy multi-series daily time-series helper.
* :func:`plot_fire_hcho_correlation` — the lagged fire->HCHO cross-correlation
  stem plot with the peak-lag annotation.
* :func:`plot_validation` — composite validation panel delegating to
  :mod:`aqi_india.validation.plots` (hexbin + Taylor).

Design rules honoured here:

* **No hard cartopy dependency.** The India basemap is drawn from
  :func:`aqi_india.utils.geo.get_india_boundary` (GeoPandas). Cartopy is only
  imported lazily if a caller explicitly opts in via ``use_cartopy=True``.
* matplotlib + numpy at module scope; geopandas / xarray imported lazily inside
  functions so the module imports under the light dependency set.
* Every figure can be composed into a caller ``ax`` and/or saved via a single
  :func:`save_figure` helper that writes under ``reports/figures/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from aqi_india.aqi.breakpoints import (
    AQI_CATEGORY_EDGES,
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    POLLUTANTS,
)
from aqi_india.aqi.grid import category_color_lut
from aqi_india.utils.geo import (
    FOREST_BELT_BBOX,
    IGP_BBOX,
    INDIA_BBOX,
    PUNJAB_HARYANA_BBOX,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd
    import pandas as pd
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure

#: Default output directory for saved report figures.
DEFAULT_FIG_DIR = Path("reports/figures")


# --------------------------------------------------------------------------- #
# Save / basemap helpers
# --------------------------------------------------------------------------- #
def save_figure(
    fig: "Figure",
    out_path: str | Path,
    *,
    dpi: int = 200,
    fig_dir: str | Path = DEFAULT_FIG_DIR,
) -> Path:
    """Save a figure, defaulting relative paths under ``reports/figures/``.

    Args:
        fig: The Matplotlib figure to write.
        out_path: Destination path. If relative, it is resolved under
            ``fig_dir``; if absolute, it is used verbatim.
        dpi: Output resolution.
        fig_dir: Base directory for relative ``out_path`` values.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    path = Path(out_path)
    if not path.is_absolute():
        path = Path(fig_dir) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def _aqi_cmap_norm() -> tuple["ListedColormap", "BoundaryNorm"]:
    """Build the discrete six-band CPCB colormap and its boundary norm."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(list(CATEGORY_COLORS), name="cpcb_naqi")
    cmap.set_bad((0, 0, 0, 0))  # transparent for NaN
    norm = BoundaryNorm(AQI_CATEGORY_EDGES, cmap.N, clip=False)
    return cmap, norm


def _grid_extent(da: "xr.DataArray") -> tuple[float, float, float, float]:
    """Return the ``imshow`` extent (lon0, lon1, lat0, lat1) for a lat/lon grid.

    Computes pixel-edge extent from the 1-D ``lat``/``lon`` coordinates so the
    raster aligns with the vector boundary overlay.
    """
    lon = np.asarray(da["lon"].values, dtype=np.float64)
    lat = np.asarray(da["lat"].values, dtype=np.float64)
    dlon = float(np.median(np.diff(lon))) if lon.size > 1 else 0.25
    dlat = float(np.median(np.diff(lat))) if lat.size > 1 else 0.25
    return (
        lon.min() - dlon / 2,
        lon.max() + dlon / 2,
        lat.min() - dlat / 2,
        lat.max() + dlat / 2,
    )


def add_india_basemap(
    ax: "Axes",
    *,
    boundary_color: str = "0.25",
    boundary_lw: float = 0.9,
    use_cartopy: bool = False,
    set_extent: bool = True,
) -> "Axes":
    """Overlay the India national boundary and frame the axes to India.

    The boundary is sourced from :func:`aqi_india.utils.geo.get_india_boundary`
    (Natural Earth if available, else the bundled simplified GeoJSON), so no
    network access or cartopy is required. Cartopy coastlines/borders are added
    only when ``use_cartopy=True`` and the package is importable.

    Args:
        ax: Axes to draw onto.
        boundary_color: Boundary line colour.
        boundary_lw: Boundary line width.
        use_cartopy: If True, additionally request cartopy features (lazy
            import; silently skipped if cartopy is unavailable).
        set_extent: If True, set the axes limits to :data:`INDIA_BBOX`.

    Returns:
        The same ``ax`` (for chaining).
    """
    try:
        from aqi_india.utils.geo import get_india_boundary

        boundary = get_india_boundary(simplify=True)
        boundary.boundary.plot(ax=ax, color=boundary_color, lw=boundary_lw, zorder=5)
    except Exception:  # pragma: no cover - offline / geopandas missing
        pass

    if use_cartopy:  # pragma: no cover - optional heavy dep
        try:
            import cartopy.feature as cfeature

            ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.4")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
        except Exception:
            pass

    if set_extent:
        min_lon, min_lat, max_lon, max_lat = INDIA_BBOX
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_aspect("equal", adjustable="box")
    return ax


def _highlight_box(
    ax: "Axes",
    bbox: tuple[float, float, float, float],
    *,
    color: str,
    label: str | None = None,
    lw: float = 1.6,
    ls: str = "-",
) -> None:
    """Draw a labelled highlight rectangle for an AOI bounding box."""
    from matplotlib.patches import Rectangle

    min_lon, min_lat, max_lon, max_lat = bbox
    rect = Rectangle(
        (min_lon, min_lat),
        max_lon - min_lon,
        max_lat - min_lat,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        linestyle=ls,
        zorder=6,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            min_lon + 0.1,
            max_lat - 0.4,
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
            zorder=7,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "alpha": 0.8},
        )


def _resolve_2d(da: "xr.DataArray", date: Any | None) -> "xr.DataArray":
    """Select a single 2-D ``(lat, lon)`` slice, indexing ``time`` if present."""
    if "time" in da.dims:
        if date is not None:
            da = da.sel(time=date, method="nearest")
        else:
            da = da.isel(time=0)
    extra = [d for d in da.dims if d not in ("lat", "lon")]
    for d in extra:
        da = da.isel({d: 0})
    return da


# --------------------------------------------------------------------------- #
# Objective-1: AQI map
# --------------------------------------------------------------------------- #
def plot_aqi_map(
    aqi_grid: "xr.DataArray | xr.Dataset",
    date: Any | None = None,
    out_path: str | Path | None = None,
    *,
    var: str = "aqi",
    title: str | None = None,
    ax: "Axes | None" = None,
    use_cartopy: bool = False,
    highlight_igp: bool = True,
) -> "Figure":
    """Plot a filled India surface-AQI map on the CPCB six-band colour scale.

    Renders the AQI field as a discrete-coloured raster, overlays the India
    boundary, adds a categorical colourbar (Good..Severe) and an IGP highlight
    box, and titles the figure with the date.

    Args:
        aqi_grid: AQI :class:`~xarray.DataArray`, or a Dataset containing ``var``
            (typically the output of
            :func:`aqi_india.aqi.grid.apply_naqi_grid`). May carry a ``time``
            dimension.
        date: Date to select from the ``time`` axis (label or ``method="nearest"``
            value). Ignored if there is no time dim. Also used in the title.
        out_path: If given, the figure is saved here (relative paths land under
            ``reports/figures/``).
        var: AQI variable name when ``aqi_grid`` is a Dataset.
        title: Figure title; a sensible default including the date is built when
            omitted.
        ax: Optional Axes to draw into; one is created otherwise.
        use_cartopy: Forwarded to :func:`add_india_basemap`.
        highlight_igp: If True, draw the IGP highlight box.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    da = aqi_grid[var] if hasattr(aqi_grid, "data_vars") else aqi_grid
    da = _resolve_2d(da, date)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 8.6))
    else:
        fig = ax.figure

    cmap, norm = _aqi_cmap_norm()
    extent = _grid_extent(da)
    data = np.ma.masked_invalid(np.asarray(da.values, dtype=np.float64))
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )

    add_india_basemap(ax, use_cartopy=use_cartopy)
    if highlight_igp:
        _highlight_box(ax, IGP_BBOX, color="#1f3b8a", label="IGP")

    cbar = fig.colorbar(
        im,
        ax=ax,
        boundaries=AQI_CATEGORY_EDGES,
        ticks=(AQI_CATEGORY_EDGES[:-1] + AQI_CATEGORY_EDGES[1:]) / 2,
        spacing="proportional",
        shrink=0.82,
        pad=0.02,
    )
    cbar.ax.set_yticklabels(CATEGORY_NAMES)
    cbar.set_label("CPCB National Air Quality Index", fontsize=10)

    date_str = _date_label(date if date is not None else _infer_date(da))
    if title is None:
        title = "Daily Surface AQI over India"
        if date_str:
            title += f" — {date_str}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if out_path is not None:
        save_figure(fig, out_path)
    return fig


def plot_responsible_pollutant_map(
    grid: "xr.DataArray | xr.Dataset",
    date: Any | None = None,
    out_path: str | Path | None = None,
    *,
    var: str = "aqi_responsible",
    title: str | None = None,
    ax: "Axes | None" = None,
    use_cartopy: bool = False,
) -> "Figure":
    """Map the responsible (driving) pollutant per cell as a categorical raster.

    Args:
        grid: Responsible-pollutant **code** DataArray (``int16``, ``-1`` =
            invalid) or a Dataset containing ``var`` — i.e. the
            ``aqi_responsible`` output of the NAQI grid engine. The code->name
            mapping is read from the variable's ``pollutant_codes`` attribute
            when present, else assumed to follow :data:`POLLUTANTS` order.
        date: Date to select from a ``time`` axis (also titles the figure).
        out_path: Optional save path (relative -> ``reports/figures/``).
        var: Variable name when ``grid`` is a Dataset.
        title: Figure title; defaulted when omitted.
        ax: Optional Axes to draw into.
        use_cartopy: Forwarded to :func:`add_india_basemap`.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    da = grid[var] if hasattr(grid, "data_vars") else grid
    codes_attr = str(da.attrs.get("pollutant_codes", "")) if hasattr(da, "attrs") else ""
    da = _resolve_2d(da, date)

    # Recover the ordered pollutant names from the codes attribute if available.
    if codes_attr:
        pairs = [p.split(":") for p in codes_attr.split(",") if ":" in p]
        names = [n for _, n in sorted(((int(i), n) for i, n in pairs))]
    else:
        names = list(POLLUTANTS)
    n = len(names)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 8.6))
    else:
        fig = ax.figure

    # A distinct qualitative colour per pollutant; invalid (-1) is masked.
    base = plt.get_cmap("tab10")
    colors = [base(i % 10) for i in range(n)]
    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1.0), cmap.N)

    arr = np.asarray(da.values, dtype=np.float64)
    arr = np.where(arr < 0, np.nan, arr)
    extent = _grid_extent(da)
    ax.imshow(
        np.ma.masked_invalid(arr),
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )

    add_india_basemap(ax, use_cartopy=use_cartopy)
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="k", label=names[i].upper()) for i in range(n)
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, title="Driver")

    date_str = _date_label(date if date is not None else _infer_date(da))
    if title is None:
        title = "AQI Responsible Pollutant"
        if date_str:
            title += f" — {date_str}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if out_path is not None:
        save_figure(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# Objective-2: HCHO hotspots
# --------------------------------------------------------------------------- #
def plot_hcho_hotspots(
    hcho_grid: "xr.DataArray | xr.Dataset",
    hotspots_gdf: "gpd.GeoDataFrame | None" = None,
    fires_gdf: "gpd.GeoDataFrame | None" = None,
    out_path: str | Path | None = None,
    *,
    var: str = "hcho_col",
    date: Any | None = None,
    wind_u: "xr.DataArray | None" = None,
    wind_v: "xr.DataArray | None" = None,
    title: str | None = None,
    ax: "Axes | None" = None,
    use_cartopy: bool = False,
    fire_size: float = 6.0,
    wind_stride: int = 6,
) -> "Figure":
    """Plot the HCHO field with hotspots, FIRMS fires and wind arrows.

    The Objective-2 hero figure: a continuous HCHO column field, confirmed
    hotspot polygons outlined on top, FIRMS fire detections as a point cloud,
    and a quiver field of the 10 m wind. The Punjab/Haryana stubble belt and the
    central/Himalayan forest belt are highlighted to anchor the source-region
    story.

    Args:
        hcho_grid: HCHO column DataArray, or a Dataset containing ``var``
            (``hcho_col`` in the synthetic-data schema, ``mol/m2``). May carry a
            ``time`` axis.
        hotspots_gdf: Confirmed-hotspot polygons (EPSG:4326). Optional.
        fires_gdf: FIRMS fire points with ``lat``/``lon`` (and optional ``frp``
            for marker sizing). Optional.
        out_path: Optional save path (relative -> ``reports/figures/``).
        var: HCHO variable name when ``hcho_grid`` is a Dataset.
        date: Date to select from the time axis (also titles the figure). Fires
            are filtered to this date when they carry a ``date`` column.
        wind_u: Optional 10 m u-wind DataArray (same grid) for quiver arrows.
        wind_v: Optional 10 m v-wind DataArray for quiver arrows.
        title: Figure title; defaulted when omitted.
        ax: Optional Axes to draw into.
        use_cartopy: Forwarded to :func:`add_india_basemap`.
        fire_size: Base marker size for fire points.
        wind_stride: Sub-sampling stride for the wind quiver (every Nth cell).

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    da = hcho_grid[var] if hasattr(hcho_grid, "data_vars") else hcho_grid
    da = _resolve_2d(da, date)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.4, 8.8))
    else:
        fig = ax.figure

    extent = _grid_extent(da)
    data = np.ma.masked_invalid(np.asarray(da.values, dtype=np.float64))
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        interpolation="bilinear",
        zorder=2,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("HCHO tropospheric column (mol m$^{-2}$)", fontsize=10)

    add_india_basemap(ax, use_cartopy=use_cartopy)

    # Source-region highlight boxes.
    _highlight_box(
        ax, PUNJAB_HARYANA_BBOX, color="#0b6e4f", label="Punjab/Haryana", ls="--"
    )
    _highlight_box(ax, FOREST_BELT_BBOX, color="#5b2a86", label="Forest belt", ls=":")

    # Wind arrows.
    if wind_u is not None and wind_v is not None:
        _add_wind_quiver(ax, da, wind_u, wind_v, date=date, stride=wind_stride)

    # Confirmed hotspot polygons.
    if hotspots_gdf is not None and len(hotspots_gdf):
        try:
            hotspots_gdf.boundary.plot(
                ax=ax, color="#08306b", lw=1.4, zorder=8, label="Hotspot"
            )
        except Exception:  # pragma: no cover - geometry edge cases
            pass

    # FIRMS fire points.
    if fires_gdf is not None and len(fires_gdf):
        fires = _filter_fires_to_date(fires_gdf, date)
        if len(fires):
            flon, flat = _fire_coords(fires)
            frp = (
                np.asarray(fires["frp"], dtype=np.float64)
                if "frp" in getattr(fires, "columns", [])
                else None
            )
            sizes = fire_size
            if frp is not None and np.isfinite(frp).any():
                fmax = np.nanmax(frp) or 1.0
                sizes = fire_size + 30.0 * (frp / fmax).clip(0, 1)
            ax.scatter(
                flon,
                flat,
                s=sizes,
                c="#111111",
                marker="^",
                edgecolor="orange",
                linewidth=0.3,
                alpha=0.8,
                zorder=9,
                label="FIRMS fire",
            )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # De-duplicate legend entries.
        seen: dict[str, Any] = {}
        for h, lbl in zip(handles, labels):
            seen.setdefault(lbl, h)
        ax.legend(seen.values(), seen.keys(), loc="lower left", fontsize=8)

    date_str = _date_label(date if date is not None else _infer_date(da))
    if title is None:
        title = "HCHO Hotspots & Biomass-Burning Transport"
        if date_str:
            title += f" — {date_str}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if out_path is not None:
        save_figure(fig, out_path)
    return fig


def _add_wind_quiver(
    ax: "Axes",
    ref: "xr.DataArray",
    wind_u: "xr.DataArray",
    wind_v: "xr.DataArray",
    *,
    date: Any | None,
    stride: int,
) -> None:
    """Overlay a sub-sampled 10 m wind quiver aligned to the HCHO grid."""
    u = _resolve_2d(wind_u, date)
    v = _resolve_2d(wind_v, date)
    lon = np.asarray(ref["lon"].values, dtype=np.float64)
    lat = np.asarray(ref["lat"].values, dtype=np.float64)
    s = max(int(stride), 1)
    lon_s, lat_s = lon[::s], lat[::s]
    uu = np.asarray(u.values, dtype=np.float64)[::s, ::s]
    vv = np.asarray(v.values, dtype=np.float64)[::s, ::s]
    lon_g, lat_g = np.meshgrid(lon_s, lat_s)
    ax.quiver(
        lon_g,
        lat_g,
        uu,
        vv,
        color="0.25",
        alpha=0.6,
        scale=250,
        width=0.0022,
        zorder=4,
    )


def _filter_fires_to_date(fires_gdf: "gpd.GeoDataFrame", date: Any | None):
    """Filter a fire table to a single acquisition date when possible."""
    if date is None or "date" not in getattr(fires_gdf, "columns", []):
        return fires_gdf
    import pandas as pd

    try:
        target = pd.Timestamp(date).normalize()
        dates = pd.to_datetime(fires_gdf["date"]).dt.normalize()
        sel = fires_gdf[dates == target]
        return sel if len(sel) else fires_gdf
    except Exception:  # pragma: no cover - mixed date types
        return fires_gdf


def _fire_coords(fires: "gpd.GeoDataFrame") -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) arrays from a fire GeoDataFrame or lat/lon columns."""
    cols = getattr(fires, "columns", [])
    if "lon" in cols and "lat" in cols:
        return (
            np.asarray(fires["lon"], dtype=np.float64),
            np.asarray(fires["lat"], dtype=np.float64),
        )
    geom = fires.geometry
    return (
        np.asarray(geom.x, dtype=np.float64),
        np.asarray(geom.y, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def plot_timeseries(
    data: "pd.DataFrame",
    *,
    time_col: str = "time",
    value_cols: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    title: str = "Daily time series",
    ylabel: str = "value",
    ax: "Axes | None" = None,
) -> "Figure":
    """Plot one or more daily time series from a tidy (wide) DataFrame.

    Args:
        data: DataFrame with a datetime ``time_col`` and one column per series.
        time_col: Name of the time column.
        value_cols: Columns to plot; defaults to every numeric column except
            ``time_col``.
        out_path: Optional save path (relative -> ``reports/figures/``).
        title: Figure title.
        ylabel: Y-axis label.
        ax: Optional Axes to draw into.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
    else:
        fig = ax.figure

    t = pd.to_datetime(data[time_col])
    if value_cols is None:
        value_cols = [
            c
            for c in data.columns
            if c != time_col and np.issubdtype(np.asarray(data[c]).dtype, np.number)
        ]
    for col in value_cols:
        ax.plot(t, np.asarray(data[col], dtype=np.float64), lw=1.4, marker="o", ms=3, label=col)

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if len(value_cols) > 1:
        ax.legend(fontsize=9)
    fig.autofmt_xdate()

    if out_path is not None:
        save_figure(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# Fire -> HCHO lagged correlation
# --------------------------------------------------------------------------- #
def plot_fire_hcho_correlation(
    lags: Any,
    corr: Any,
    out_path: str | Path | None = None,
    *,
    title: str = "Fire → HCHO lagged cross-correlation",
    ax: "Axes | None" = None,
    sig_level: float | None = None,
) -> "Figure":
    """Stem plot of the lagged fire→HCHO cross-correlation with peak annotation.

    Args:
        lags: Lag values in days (x-axis); positive = HCHO lags fire.
        corr: Cross-correlation coefficient at each lag.
        out_path: Optional save path (relative -> ``reports/figures/``).
        title: Figure title.
        ax: Optional Axes to draw into.
        sig_level: Optional significance threshold; drawn as dashed guide lines.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    lags_arr = np.asarray(lags, dtype=np.float64)
    corr_arr = np.asarray(corr, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
    else:
        fig = ax.figure

    markerline, stemlines, baseline = ax.stem(lags_arr, corr_arr)
    plt.setp(stemlines, color="#1f3b8a", lw=1.2)
    plt.setp(markerline, color="#1f3b8a", ms=5)
    plt.setp(baseline, color="0.4", lw=0.8)

    if corr_arr.size and np.isfinite(corr_arr).any():
        peak = int(np.nanargmax(corr_arr))
        ax.annotate(
            f"peak lag = {lags_arr[peak]:.0f} d\nr = {corr_arr[peak]:.2f}",
            xy=(lags_arr[peak], corr_arr[peak]),
            xytext=(0.62, 0.86),
            textcoords="axes fraction",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "darkred"},
            bbox={"boxstyle": "round", "fc": "white", "ec": "darkred", "alpha": 0.85},
        )

    if sig_level is not None:
        ax.axhline(sig_level, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(-sig_level, color="red", ls="--", lw=0.8, alpha=0.6)

    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Lag (days, HCHO relative to fire)")
    ax.set_ylabel("Cross-correlation r")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if out_path is not None:
        save_figure(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# Validation panel (delegates to validation.plots)
# --------------------------------------------------------------------------- #
def plot_validation(
    y_true: Any,
    y_pred: Any,
    out_path: str | Path | None = None,
    *,
    taylor_stats: "list[dict[str, Any]] | None" = None,
    units: str = "",
    title: str = "Model validation",
) -> "Figure":
    """Composite validation panel: 1:1 hexbin (+ optional Taylor diagram).

    Delegates the actual plotting to :mod:`aqi_india.validation.plots` so the
    hexbin / Taylor implementations live in one place. With ``taylor_stats`` a
    two-panel figure (hexbin + Taylor) is produced; otherwise a single hexbin.

    Args:
        y_true: Observed values (e.g. CPCB hold-out).
        y_pred: Predicted values aligned with ``y_true``.
        out_path: Optional save path (relative -> ``reports/figures/``).
        taylor_stats: Optional per-model ``{"std","r","name"}`` records for the
            Taylor panel (see :func:`aqi_india.validation.plots.taylor_diagram`).
        units: Unit string appended to the hexbin axis labels.
        title: Overall figure suptitle.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    from aqi_india.validation.plots import one_to_one_hexbin, taylor_diagram

    if taylor_stats:
        fig = plt.figure(figsize=(12.0, 5.6))
        ax_hex = fig.add_subplot(1, 2, 1)
        ax_taylor = fig.add_subplot(1, 2, 2, projection="polar")
        one_to_one_hexbin(y_true, y_pred, ax=ax_hex, units=units)
        taylor_diagram(taylor_stats, ax=ax_taylor)
    else:
        fig, ax_hex = plt.subplots(figsize=(6.0, 5.8))
        one_to_one_hexbin(y_true, y_pred, ax=ax_hex, units=units)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if out_path is not None:
        save_figure(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# Small internal date helpers
# --------------------------------------------------------------------------- #
def _infer_date(da: "xr.DataArray") -> Any | None:
    """Best-effort extraction of a scalar date coordinate from a DataArray."""
    if "time" in getattr(da, "coords", {}):
        try:
            return da["time"].values
        except Exception:  # pragma: no cover
            return None
    return None


def _date_label(date: Any | None) -> str:
    """Format a date-like value as ``YYYY-MM-DD`` (empty string if unavailable)."""
    if date is None:
        return ""
    try:
        import pandas as pd

        val = np.atleast_1d(np.asarray(date)).ravel()[0] if np.ndim(date) else date
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return str(date)


# Re-export the LUT helper so callers can `from viz.figures import category_color_lut`.
__all__ = [
    "DEFAULT_FIG_DIR",
    "save_figure",
    "add_india_basemap",
    "category_color_lut",
    "plot_aqi_map",
    "plot_responsible_pollutant_map",
    "plot_hcho_hotspots",
    "plot_timeseries",
    "plot_fire_hcho_correlation",
    "plot_validation",
]
