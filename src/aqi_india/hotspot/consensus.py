"""Three-vote consensus HCHO hotspot detection (Objective-2 entry point).

A single statistic over noisy HCHO is not trustworthy: TROPOMI single-pixel
error is 30-100 %, so each detector has its own failure mode. We therefore
require **agreement of at least two of three independent votes** before a cell
is declared a CONFIRMED hotspot (blueprint method 36):

1.  **Gi\\* significant** — FDR-corrected Getis-Ord Gi* hotspot
    (:mod:`aqi_india.hotspot.getis_ord`).
2.  **LISA HH** — a significant High-High Local Moran's I cluster core
    (:mod:`aqi_india.hotspot.lisa`).
3.  **Exceedance** — the cell's robust z-anomaly exceeds ``z>2`` *and* its mean
    column exceeds the per-cell seasonal 95th-percentile surface
    (:mod:`aqi_india.hotspot.climatology`).

The detectors all consume the **per-season robust z-anomaly** so the India-wide
spatial gradient and seasonal cycle are removed first. The public
:func:`detect_hotspots` returns a :class:`geopandas.GeoDataFrame` of confirmed
hotspot cells (with per-vote metadata) and :func:`run` is the Hydra-config entry
point that also writes ``data/processed/hotspots.geoparquet`` and the delineated
polygons.
"""

from __future__ import annotations

import numpy as np

from ..utils.geo import CRS_WGS84
from ..utils.logging import get_logger
from .climatology import build_climatology
from .getis_ord import getis_ord_gi
from .lisa import local_morans_i

logger = get_logger("hotspot.consensus")

#: Minimum number of votes (of three) for a confirmed hotspot.
DEFAULT_MIN_VOTES: int = 2

#: Robust z-anomaly threshold for the exceedance vote.
DEFAULT_Z_THRESHOLD: float = 2.0


def _grid_to_cells(clim):
    """Reduce a climatology Dataset to a flat per-cell table.

    Uses the *spatial* robust anomaly of the seasonal-mean column (which keeps
    spatial contrast so a persistent blob still stands out), joins the seasonal
    mean column and 95th-percentile surface, and drops all-NaN cells.

    Args:
        clim: Output of :func:`build_climatology`.

    Returns:
        ``(coords, z_spatial, mean_col, p_surface)`` where ``coords`` is
        ``(m, 2)`` of ``(lon, lat)`` for the ``m`` finite cells, plus the
        matching 1-D arrays and a back-index ``(rows, cols)`` per kept cell.
    """
    lat = clim["lat"].values
    lon = clim["lon"].values
    z_spatial = clim["z_spatial"].values  # (lat, lon)
    mean_col = clim["hcho_mean"].values
    # The seasonal percentile surface column name is hcho_p{q}.
    p_name = next(n for n in clim.data_vars if n.startswith("hcho_p"))
    p_surface = clim[p_name].values

    lon2d, lat2d = np.meshgrid(lon, lat)
    finite = np.isfinite(z_spatial)
    rows, cols = np.where(finite)
    coords = np.column_stack([lon2d[finite], lat2d[finite]])
    return (
        coords,
        z_spatial[finite],
        mean_col[finite],
        p_surface[finite],
        (rows, cols),
    )


def detect_hotspots(
    hcho_cube,
    season: str | None = None,
    *,
    var: str = "hcho_col",
    k: int = 8,
    permutations: int = 999,
    gi_alpha: float = 0.01,
    lisa_alpha: float = 0.05,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    min_votes: int = DEFAULT_MIN_VOTES,
    q: float = 95.0,
):
    """Detect confirmed HCHO hotspots by >=2-of-3 consensus.

    Args:
        hcho_cube: HCHO time-cube (:class:`xarray.Dataset` with ``var`` or a
            :class:`xarray.DataArray`) following the DEV_CONTRACT grid schema
            (``time``/``lat``/``lon``, column in ``mol/m2``).
        season: Season to restrict to (e.g. ``"post_monsoon"``); ``None`` uses
            all time steps.
        var: HCHO column variable name when a Dataset is passed.
        k: KNN neighbour count for the Gi*/LISA weights graphs.
        permutations: Permutations for Gi*/LISA inference.
        gi_alpha: FDR level for the Gi* vote.
        lisa_alpha: Significance level for the LISA HH vote.
        z_threshold: Robust z-anomaly threshold for the exceedance vote.
        min_votes: Votes required to confirm a hotspot (default ``2``).
        q: Percentile for the exceedance surface (default ``95``).

    Returns:
        A :class:`geopandas.GeoDataFrame` (EPSG:4326, point geometry) of the
        confirmed hotspot cells with columns ``lon``, ``lat``, ``mean_z``,
        ``vote_gi``, ``vote_lisa``, ``vote_exceed``, ``n_votes`` and
        ``confirmed`` (all confirmed rows are returned).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    clim = build_climatology(hcho_cube, season=season, var=var, q=q)
    coords, z_spatial, mean_col, _p_surface, _idx = _grid_to_cells(clim)

    if coords.shape[0] < 3:
        logger.warning("too few finite cells (%d) for detection", coords.shape[0])
        return gpd.GeoDataFrame(
            columns=[
                "lon", "lat", "mean_z", "vote_gi", "vote_lisa",
                "vote_exceed", "n_votes", "confirmed", "geometry",
            ],
            geometry="geometry",
            crs=CRS_WGS84,
        )

    gi = getis_ord_gi(
        z_spatial, coords, k=k, permutations=permutations, alpha=gi_alpha
    )
    lisa = local_morans_i(
        z_spatial, coords, k=k, permutations=permutations, alpha=lisa_alpha
    )

    vote_gi = gi.sig
    vote_lisa = lisa.hh_mask
    # Exceedance vote ("95th-pct / z>2"): the cell is a strong spatial anomaly
    # (z>2) AND its seasonal-mean column sits in the domain's upper tail
    # (>= the nationwide 95th-percentile level of the mean field).
    domain_p95 = np.nanpercentile(mean_col, q)
    vote_exceed = (z_spatial > z_threshold) & (mean_col >= domain_p95)

    n_votes = (
        vote_gi.astype(int) + vote_lisa.astype(int) + vote_exceed.astype(int)
    )
    confirmed = n_votes >= min_votes

    sel = np.flatnonzero(confirmed)
    logger.info(
        "consensus: %d/%d cells confirmed (Gi*=%d, LISA=%d, exceed=%d)",
        sel.size,
        coords.shape[0],
        int(vote_gi.sum()),
        int(vote_lisa.sum()),
        int(vote_exceed.sum()),
    )

    gdf = gpd.GeoDataFrame(
        {
            "lon": coords[sel, 0],
            "lat": coords[sel, 1],
            "mean_z": z_spatial[sel],
            "vote_gi": vote_gi[sel],
            "vote_lisa": vote_lisa[sel],
            "vote_exceed": vote_exceed[sel],
            "n_votes": n_votes[sel],
            "confirmed": confirmed[sel],
        },
        geometry=[Point(xy) for xy in coords[sel]],
        crs=CRS_WGS84,
    )
    gdf.attrs["season"] = clim.attrs.get("season", "all")
    return gdf


def run(cfg):
    """Hydra entry point: detect hotspots and persist the GeoParquet outputs.

    Reads the fused grid cube from ``cfg.paths.data_processed/grid.nc``, runs
    :func:`detect_hotspots` for the configured season, delineates polygons, and
    writes ``hotspots.geoparquet`` (confirmed cells) plus
    ``hotspot_polygons.geoparquet`` (delineated clusters).

    Args:
        cfg: The composed Hydra config. Uses ``cfg.paths.data_processed``,
            ``cfg.hotspot.*`` (``weights.k``, ``permutations``, ``alpha``,
            ``standardize``) and ``cfg.dates.season`` when present.

    Returns:
        The confirmed-hotspot :class:`geopandas.GeoDataFrame`.
    """
    from pathlib import Path

    import xarray as xr

    from ..utils.io import save_geoparquet
    from .cluster import hotspot_polygons

    processed = Path(cfg.paths.data_processed)
    grid_path = processed / "grid.nc"
    logger.info("loading HCHO cube from %s", grid_path)
    ds = xr.open_dataset(grid_path)

    hot_cfg = getattr(cfg, "hotspot", {})
    k = int(getattr(getattr(hot_cfg, "weights", {}), "k", 8))
    permutations = int(getattr(hot_cfg, "permutations", 999))
    gi_alpha = float(getattr(hot_cfg, "alpha", 0.01))
    season = getattr(getattr(cfg, "dates", {}), "season", None)

    gdf = detect_hotspots(
        ds,
        season=season,
        k=k,
        permutations=permutations,
        gi_alpha=gi_alpha,
    )

    out_cells = processed / "hotspots.geoparquet"
    save_geoparquet(gdf, out_cells)
    logger.info("wrote %d confirmed hotspot cells -> %s", len(gdf), out_cells)

    if len(gdf):
        polys = hotspot_polygons(
            np.column_stack([gdf["lon"].values, gdf["lat"].values]),
            values=gdf["mean_z"].values,
        )
        save_geoparquet(polys, processed / "hotspot_polygons.geoparquet")
        logger.info("wrote %d hotspot polygons", len(polys))

    return gdf


__all__ = [
    "DEFAULT_MIN_VOTES",
    "DEFAULT_Z_THRESHOLD",
    "detect_hotspots",
    "run",
]
