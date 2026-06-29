"""Assemble the tidy feature matrix the surface-concentration models train on.

This is the join hub of the feature stage. It samples the fused gridded cube at
every CPCB station location (nearest grid node via a ``scipy.cKDTree``), derives
the physics-guided channels (PBL-normalized AOD, hygroscopic correction, FNR,
aerosol-type flag, wind speed/direction), attaches the static covariates and
temporal encodings, and folds in fire context — producing one tidy row per
``(h3_res7, time)`` keyed exactly on the universal H3 fusion key.

Two public entry points:

* :func:`build_feature_matrix` (grid, stations, fires=None) -> the station-level
  training table (a pandas / GeoParquet-ready DataFrame).
* :func:`build_grid_features` (grid) -> the full-grid feature stack for inference.

A thin :func:`build` (cfg) wrapper loads the artifacts named in the
DEV_CONTRACT file-location contract and calls :func:`build_feature_matrix`, so the
Demo agent can chain ``aqi_india.features.feature_matrix.build(cfg)``.

All heavy/optional dependencies (xarray, geopandas, scipy) are imported lazily so
the module imports under the light dependency set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import physics, temporal
from .h3_index import DEFAULT_RES, cells_for_arrays
from .met import met_feature_names
from .static_covars import STATIC_COVARIATES, get_static_covariates
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import xarray as xr

_log = get_logger("features.feature_matrix")

#: Satellite column / met variables carried straight through from the grid cube.
_PASSTHROUGH_GRID_VARS: tuple[str, ...] = (
    "aod",
    "no2_col",
    "so2_col",
    "co_col",
    "o3_col",
    "hcho_col",
    "blh",
    "rh",
    "wind_u",
    "wind_v",
    "t2m",
    "ssrd",
)

#: CPCB label columns expected on the station table (DEV_CONTRACT S6.2).
_LABEL_COLS: tuple[str, ...] = ("pm25", "pm10", "no2", "so2", "co", "o3")

#: Physics-derived feature column names added by this module.
PHYSICS_FEATURES: tuple[str, ...] = (
    "aod_pbl",
    "aod_dry",
    "fnr",
    "aerosol_type",
    "wind_speed",
    "wind_dir",
)

#: Temporal feature column names added by this module.
TEMPORAL_FEATURES: tuple[str, ...] = ("doy_sin", "doy_cos")

#: Fire-context feature column names added when fires are supplied.
FIRE_FEATURES: tuple[str, ...] = ("fire_count", "fire_frp_sum")


def _grid_node_arrays(grid: "xr.Dataset") -> tuple[np.ndarray, np.ndarray]:
    """Return flattened (lat, lon) node coordinate arrays for the grid mesh."""
    lat = np.asarray(grid["lat"].values, dtype=np.float64)
    lon = np.asarray(grid["lon"].values, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon, lat)  # (nlat, nlon)
    return lat2d.ravel(), lon2d.ravel()


def _build_kdtree(grid: "xr.Dataset") -> tuple[Any, int, int]:
    """Build a cKDTree over grid nodes; return (tree, nlat, nlon).

    The tree indexes flattened ``(lat, lon)`` node coordinates so a station's
    nearest grid cell is an O(log n) query. Degrees are treated as a planar
    metric here, which is adequate at India's mid-latitudes and the ~0.25 deg
    grid spacing; metric-critical operations reproject to UTM elsewhere.
    """
    from scipy.spatial import cKDTree

    lat_flat, lon_flat = _grid_node_arrays(grid)
    tree = cKDTree(np.column_stack([lat_flat, lon_flat]))
    return tree, grid.sizes["lat"], grid.sizes["lon"]


def _nearest_node_indices(
    grid: "xr.Dataset", lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map station (lat, lon) to nearest grid (i_lat, j_lon) indices via cKDTree."""
    tree, _nlat, nlon = _build_kdtree(grid)
    _dist, flat_idx = tree.query(np.column_stack([lats, lons]), k=1)
    flat_idx = np.asarray(flat_idx, dtype=np.int64)
    i_lat = flat_idx // nlon
    j_lon = flat_idx % nlon
    return i_lat, j_lon


def _sample_grid_at_stations(
    grid: "xr.Dataset",
    static: "xr.Dataset",
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sample dynamic grid vars + static covariates at station nearest-nodes.

    Returns a dict mapping variable name -> array of shape ``(n_time, n_station)``
    for dynamic vars and ``(n_station,)`` for static covariates.
    """
    i_lat, j_lon = _nearest_node_indices(grid, lats, lons)
    out: dict[str, np.ndarray] = {}

    # Dynamic (time, lat, lon) -> (time, station).
    for name in grid.data_vars:
        da = grid[name].transpose("time", "lat", "lon")
        values = np.asarray(da.values, dtype=np.float64)  # (T, nlat, nlon)
        out[name] = values[:, i_lat, j_lon]  # fancy-index -> (T, n_station)

    # Static (lat, lon) -> (station,).
    for name in static.data_vars:
        values = np.asarray(static[name].values, dtype=np.float64)
        out[f"static::{name}"] = values[i_lat, j_lon]
    return out


def _fire_counts_per_cell(
    fires: "pd.DataFrame | Any", res: int = DEFAULT_RES
) -> "pd.DataFrame":
    """Aggregate fire detections to (h3_res7, date) counts and summed FRP.

    Args:
        fires: Fire table with ``lat``/``lon``/``date`` and optional ``frp`` and
            ``h3_res7`` columns (DEV_CONTRACT S6.3).
        res: H3 resolution for the join key.

    Returns:
        DataFrame with columns ``h3_res7``, ``time``, ``fire_count``,
        ``fire_frp_sum``.
    """
    import pandas as pd

    df = fires.copy()
    if "h3_res7" not in df.columns:
        df["h3_res7"] = cells_for_arrays(
            df["lat"].to_numpy(), df["lon"].to_numpy(), res
        )
    time_col = "date" if "date" in df.columns else "time"
    df["time"] = pd.to_datetime(df[time_col]).dt.normalize()
    frp = df["frp"] if "frp" in df.columns else 0.0
    df = df.assign(_frp=frp)
    grouped = df.groupby(["h3_res7", "time"], sort=False).agg(
        fire_count=("_frp", "size"),
        fire_frp_sum=("_frp", "sum"),
    )
    return grouped.reset_index()


def build_feature_matrix(
    grid: "xr.Dataset",
    stations: "pd.DataFrame | Any",
    fires: "pd.DataFrame | Any | None" = None,
    *,
    res: int = DEFAULT_RES,
    static_covariates: "xr.Dataset | None" = None,
    add_lags: bool = True,
    lags: "list[int] | None" = None,
) -> "pd.DataFrame":
    """Build the tidy station-level training feature matrix.

    Joins satellite columns + physics-guided channels + meteorology + static
    covariates + temporal encodings (+ optional fire context) at every CPCB
    station-day, keyed by ``(h3_res7, time)``. Grid values are sampled at each
    station's nearest grid node via a ``scipy.cKDTree``.

    Args:
        grid: Fused gridded ``Dataset`` (DEV_CONTRACT S6.1), dims
            ``(time, lat, lon)``.
        stations: Long-format station table (DEV_CONTRACT S6.2): one row per
            station-day with ``station_id``, ``lat``, ``lon``, ``time`` and the
            CPCB label columns. ``h3_res7`` is computed if absent.
        fires: Optional fire table (DEV_CONTRACT S6.3); when given, per-cell daily
            ``fire_count`` and ``fire_frp_sum`` are merged in.
        res: H3 resolution for the join key (default 7).
        static_covariates: Optional precomputed static covariate Dataset; if
            ``None`` it is synthesized/loaded via :func:`get_static_covariates`.
        add_lags: If True, append 1- and 3-day lags of the satellite columns and
            labels (per station).
        lags: Custom lag list (days); defaults to ``[1, 3]`` when ``add_lags``.

    Returns:
        A tidy :class:`pandas.DataFrame` (one row per station-day) with the
        satellite/met passthrough columns, the :data:`PHYSICS_FEATURES`,
        ``static_*`` covariates, :data:`TEMPORAL_FEATURES`, optional
        :data:`FIRE_FEATURES`, the H3 key ``h3_res7`` and the CPCB labels.
    """
    import pandas as pd

    stations = _ensure_station_frame(stations)
    n_stat_days = len(stations)
    _log.info("Building feature matrix over %d station-days.", n_stat_days)

    # Static covariates aligned to this grid.
    static = (
        static_covariates
        if static_covariates is not None
        else get_static_covariates(grid)
    )

    # Unique station registry -> sample the grid once per (station, all-times).
    registry = (
        stations.drop_duplicates("station_id")[["station_id", "lat", "lon"]]
        .reset_index(drop=True)
    )
    lats = registry["lat"].to_numpy(dtype=np.float64)
    lons = registry["lon"].to_numpy(dtype=np.float64)
    sampled = _sample_grid_at_stations(grid, static, lats, lons)

    grid_times = pd.DatetimeIndex(np.asarray(grid["time"].values)).normalize()

    # Long-format sampled table: one row per (station_id, grid_time).
    long = _samples_to_long(sampled, registry["station_id"].to_numpy(), grid_times)

    # Derive physics-guided + temporal features on the long table.
    long = _add_physics_features(long)
    long = _add_temporal_features(long)

    # Merge sampled features onto the station label table by (station_id, time).
    stations = stations.copy()
    stations["time"] = pd.to_datetime(stations["time"]).dt.normalize()
    matrix = stations.merge(long, on=["station_id", "time"], how="left")

    # H3 universal key.
    if "h3_res7" not in matrix.columns:
        matrix["h3_res7"] = cells_for_arrays(
            matrix["lat"].to_numpy(), matrix["lon"].to_numpy(), res
        )

    # Fire context per (h3_res7, time).
    if fires is not None and len(fires):
        fire_agg = _fire_counts_per_cell(fires, res=res)
        matrix = matrix.merge(fire_agg, on=["h3_res7", "time"], how="left")
        for col in FIRE_FEATURES:
            if col in matrix.columns:
                matrix[col] = matrix[col].fillna(0.0)
    else:
        for col in FIRE_FEATURES:
            matrix[col] = 0.0

    # Temporal lags (per station).
    if add_lags:
        lag_list = lags if lags is not None else [1, 3]
        lag_cols = [
            c for c in ("aod", "no2_col", "hcho_col", "pm25", "pm10") if c in matrix
        ]
        matrix = temporal.add_lag_features(
            matrix, lag_cols, lag_list, group_col="station_id", time_col="time"
        )

    matrix = matrix.sort_values(["time", "station_id"]).reset_index(drop=True)
    _log.info(
        "Feature matrix built: %d rows x %d columns.", len(matrix), matrix.shape[1]
    )
    return matrix


def _ensure_station_frame(stations: Any) -> "pd.DataFrame":
    """Coerce a GeoDataFrame/DataFrame station input to a plain DataFrame.

    If a GeoDataFrame with point geometry is passed but ``lat``/``lon`` columns
    are absent, they are derived from the geometry.
    """
    import pandas as pd

    df = stations.copy()
    has_geometry = hasattr(df, "geometry")
    if "lat" not in df.columns or "lon" not in df.columns:
        if has_geometry:
            df["lon"] = df.geometry.x
            df["lat"] = df.geometry.y
        else:
            raise ValueError("stations must provide lat/lon columns or geometry.")
    # Drop the geometry column (if any) and return a plain DataFrame.
    if has_geometry and df.geometry.name in df.columns:
        df = df.drop(columns=[df.geometry.name])
    return pd.DataFrame(df)


def _samples_to_long(
    sampled: dict[str, np.ndarray],
    station_ids: np.ndarray,
    grid_times: "pd.DatetimeIndex",
) -> "pd.DataFrame":
    """Reshape sampled (time, station) / (station,) arrays into a long table."""
    import pandas as pd

    n_time = len(grid_times)
    n_station = len(station_ids)

    # Repeat station ids/time across the (time x station) cartesian product.
    time_col = np.repeat(grid_times.values, n_station)
    sid_col = np.tile(station_ids, n_time)
    data: dict[str, np.ndarray] = {"station_id": sid_col, "time": time_col}

    for name, arr in sampled.items():
        if name.startswith("static::"):
            out_name = f"static_{name.split('::', 1)[1]}"
            # (station,) -> tile across time.
            data[out_name] = np.tile(np.asarray(arr, dtype=np.float64), n_time)
        else:
            # (time, station) -> flatten row-major (time-major).
            data[name] = np.asarray(arr, dtype=np.float64).reshape(-1)
    return pd.DataFrame(data)


def _add_physics_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add physics-guided channels to a sampled long table (vectorized)."""
    out = df.copy()
    if {"aod", "blh"}.issubset(out.columns):
        out["aod_pbl"] = physics.pbl_normalized_aod(
            out["aod"].to_numpy(), out["blh"].to_numpy()
        )
    if {"aod", "rh"}.issubset(out.columns):
        out["aod_dry"] = physics.hygroscopic_correction(
            out["aod"].to_numpy(), out["rh"].to_numpy()
        )
    if {"hcho_col", "no2_col"}.issubset(out.columns):
        out["fnr"] = physics.fnr(
            out["hcho_col"].to_numpy(), out["no2_col"].to_numpy()
        )
    if {"wind_u", "wind_v"}.issubset(out.columns):
        out["wind_speed"] = physics.wind_speed(
            out["wind_u"].to_numpy(), out["wind_v"].to_numpy()
        )
        out["wind_dir"] = physics.wind_dir(
            out["wind_u"].to_numpy(), out["wind_v"].to_numpy()
        )
    # Aerosol-type flag only if the optional AER_AI / Angstrom columns exist.
    aer_ai = out["aer_ai"].to_numpy() if "aer_ai" in out.columns else None
    angstrom = out["angstrom"].to_numpy() if "angstrom" in out.columns else None
    if aer_ai is not None or angstrom is not None:
        out["aerosol_type"] = physics.aerosol_type_flag(aer_ai, angstrom)
    return out


def _add_temporal_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add day-of-year sin/cos cyclic encodings."""
    out = df.copy()
    feats = temporal.day_of_year_features(out["time"].to_numpy())
    out["doy_sin"] = feats["doy_sin"]
    out["doy_cos"] = feats["doy_cos"]
    return out


def build_grid_features(
    grid: "xr.Dataset",
    *,
    static_covariates: "xr.Dataset | None" = None,
) -> "xr.Dataset":
    """Build the full-grid feature stack for inference (no station sampling).

    Augments the fused cube with the physics-guided channels and temporal
    encodings as gridded data-vars, and broadcasts the static covariates across
    time, yielding a single ``Dataset`` the gridded models predict from.

    Args:
        grid: Fused gridded ``Dataset`` (DEV_CONTRACT S6.1).
        static_covariates: Optional static covariate Dataset; synthesized via
            :func:`get_static_covariates` when ``None``.

    Returns:
        An :class:`xarray.Dataset` containing the original grid vars plus
        ``aod_pbl``, ``aod_dry``, ``fnr``, ``wind_speed``, ``wind_dir``,
        ``aerosol_type`` (when inputs allow), the day-of-year encodings
        ``doy_sin``/``doy_cos`` (dims ``(time,)``), and the static covariates
        broadcast over time.
    """
    import xarray as xr

    out = grid.copy()

    # Physics-guided gridded channels (xarray-vectorized, NaN-safe).
    if {"aod", "blh"}.issubset(out.data_vars):
        out["aod_pbl"] = physics.pbl_normalized_aod(out["aod"], out["blh"])
    if {"aod", "rh"}.issubset(out.data_vars):
        out["aod_dry"] = physics.hygroscopic_correction(out["aod"], out["rh"])
    if {"hcho_col", "no2_col"}.issubset(out.data_vars):
        out["fnr"] = physics.fnr(out["hcho_col"], out["no2_col"])
    if {"wind_u", "wind_v"}.issubset(out.data_vars):
        out["wind_speed"] = physics.wind_speed(out["wind_u"], out["wind_v"])
        out["wind_dir"] = physics.wind_dir(out["wind_u"], out["wind_v"])
    if "aer_ai" in out.data_vars or "angstrom" in out.data_vars:
        aer_ai = out["aer_ai"] if "aer_ai" in out.data_vars else None
        angstrom = out["angstrom"] if "angstrom" in out.data_vars else None
        out["aerosol_type"] = physics.aerosol_type_flag(aer_ai, angstrom)

    # Temporal cyclic encodings along the time axis.
    feats = temporal.day_of_year_features(out["time"].values)
    out["doy_sin"] = xr.DataArray(feats["doy_sin"], dims=("time",), coords={"time": out["time"]})
    out["doy_cos"] = xr.DataArray(feats["doy_cos"], dims=("time",), coords={"time": out["time"]})

    # Static covariates broadcast across time.
    static = (
        static_covariates
        if static_covariates is not None
        else get_static_covariates(grid)
    )
    for name in static.data_vars:
        out[f"static_{name}"] = static[name]

    out.attrs["feature_stage"] = "aqi_india.features.feature_matrix.build_grid_features"
    return out


def build(cfg: Any) -> "pd.DataFrame":
    """Hydra-entrypoint wrapper: load artifacts and build the feature matrix.

    Implements the DEV_CONTRACT chaining contract
    ``aqi_india.features.feature_matrix.build(cfg)``. Reads the fused cube,
    station labels and (optionally) fires from the processed-data locations,
    builds the matrix, and writes it to ``data/processed/feature_matrix.parquet``.

    Args:
        cfg: Composed Hydra config exposing ``cfg.paths.data_processed`` (falls
            back to ``data/processed`` if unavailable).

    Returns:
        The assembled station-level feature matrix.
    """
    from pathlib import Path

    from ..utils.io import load_netcdf, load_parquet, save_parquet

    processed = Path(
        _cfg_get(cfg, "paths", "data_processed", default="data/processed")
    )
    grid = load_netcdf(processed / "grid.nc")
    stations = load_parquet(processed / "stations.parquet")
    fires_path = processed / "fires.parquet"
    fires = load_parquet(fires_path) if fires_path.exists() else None

    matrix = build_feature_matrix(grid, stations, fires)
    out_path = processed / "feature_matrix.parquet"
    save_parquet(matrix, out_path)
    _log.info("Wrote feature matrix to %s", out_path)
    return matrix


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested Hydra/dict config, returning ``default`` on miss."""
    node: Any = cfg
    for key in keys:
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(key, None)
        else:
            node = getattr(node, key, None)
    return node if node is not None else default


def feature_columns(matrix: "pd.DataFrame") -> list[str]:
    """Return the predictor (non-label, non-key) columns of a feature matrix.

    Useful for the model stage to select X without the labels, ids and the H3
    key.

    Args:
        matrix: A feature matrix from :func:`build_feature_matrix`.

    Returns:
        Ordered list of predictor column names.
    """
    exclude = set(_LABEL_COLS) | {
        "station_id",
        "name",
        "lat",
        "lon",
        "h3_res7",
        "region",
        "time",
        "geometry",
    }
    static_cols = [c for c in matrix.columns if c.startswith("static_")]
    candidates = [
        c
        for c in matrix.columns
        if c not in exclude
        and (matrix[c].dtype.kind in "fiu" or c in static_cols)
    ]
    return candidates


__all__ = [
    "PHYSICS_FEATURES",
    "TEMPORAL_FEATURES",
    "FIRE_FEATURES",
    "STATIC_COVARIATES",
    "build_feature_matrix",
    "build_grid_features",
    "build",
    "feature_columns",
    "met_feature_names",
]
