"""Geostatistical interpolation: ordinary / regression kriging with IDW fallback.

This module interpolates scattered observations (e.g. CPCB station residuals, or
sparse valid satellite pixels) onto the analysis grid, returning both an
estimate and an uncertainty field.

Two tiers, matching the project's "always-runnable demo" constraint:

* **Kriging** (ordinary or regression) via a lazily-imported :mod:`pykrige`
  (preferred) or :mod:`gstools` backend — used when the heavy geostatistics
  stack is installed. Regression kriging fits a trend on covariates first, then
  krigs the residuals.
* **IDW fallback** built on :class:`scipy.spatial.cKDTree` that *always* runs on
  the light dependency set. It is also exposed directly as :func:`idw` for fast
  emergency spatial fills (synthesis blueprint Stage-2 method 10).

All public functions take/return plain :mod:`numpy` arrays in EPSG:4326 (lon/lat
in degrees); set ``metric=True`` to do distance math in an approximate local
equirectangular metric projection (recommended for India-scale extents).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = get_logger("fusion.kriging")

_EARTH_RADIUS_KM = 6371.0088


@dataclass
class KrigeResult:
    """Result of an interpolation onto a target grid.

    Attributes:
        estimate: Interpolated field, shape ``(n_lat, n_lon)``.
        variance: Estimation variance / uncertainty, same shape. For IDW this is
            a heuristic based on neighbour distance and spread.
        method: The backend actually used (``"ordinary_kriging"``,
            ``"regression_kriging"`` or ``"idw"``).
    """

    estimate: np.ndarray
    variance: np.ndarray
    method: str


def _project(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float):
    """Approximate local equirectangular projection to kilometres.

    Args:
        lon: Longitudes (deg).
        lat: Latitudes (deg).
        lon0: Reference longitude (deg) for the projection origin.
        lat0: Reference latitude (deg).

    Returns:
        Tuple ``(x_km, y_km)`` of projected coordinates.
    """
    x = np.radians(lon - lon0) * _EARTH_RADIUS_KM * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * _EARTH_RADIUS_KM
    return x, y


def idw(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_val: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    *,
    k: int = 8,
    power: float = 2.0,
    metric: bool = True,
    smoothing: float = 0.0,
) -> KrigeResult:
    """Inverse-distance-weighted interpolation via :class:`scipy.spatial.cKDTree`.

    This always-available fallback runs on the light dependency set and is used
    both as the kriging backstop and as a standalone fast spatial fill.

    Args:
        obs_lon: Observation longitudes, shape ``(n_obs,)``.
        obs_lat: Observation latitudes, shape ``(n_obs,)``.
        obs_val: Observation values, shape ``(n_obs,)`` (NaNs dropped).
        grid_lon: 1-D target longitudes (ascending), shape ``(n_lon,)``.
        grid_lat: 1-D target latitudes (ascending), shape ``(n_lat,)``.
        k: Number of nearest neighbours used per target cell.
        power: IDW power exponent (2 = classic inverse-square).
        metric: If True, do distance math in local-equirectangular km.
        smoothing: Added to squared distances to avoid singularities at
            coincident points (a Shepard-style smoothing factor).

    Returns:
        A :class:`KrigeResult` with ``method == "idw"``.

    Raises:
        ValueError: If there are no finite observations.
    """
    from scipy.spatial import cKDTree

    obs_lon = np.asarray(obs_lon, dtype="float64")
    obs_lat = np.asarray(obs_lat, dtype="float64")
    obs_val = np.asarray(obs_val, dtype="float64")
    finite = np.isfinite(obs_val) & np.isfinite(obs_lon) & np.isfinite(obs_lat)
    obs_lon, obs_lat, obs_val = obs_lon[finite], obs_lat[finite], obs_val[finite]
    if obs_val.size == 0:
        raise ValueError("idw requires at least one finite observation")

    lon0 = float(np.mean(obs_lon))
    lat0 = float(np.mean(obs_lat))
    if metric:
        ox, oy = _project(obs_lon, obs_lat, lon0, lat0)
    else:
        ox, oy = obs_lon, obs_lat
    tree = cKDTree(np.column_stack([ox, oy]))

    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)  # (n_lat, n_lon)
    if metric:
        gx, gy = _project(mesh_lon.ravel(), mesh_lat.ravel(), lon0, lat0)
    else:
        gx, gy = mesh_lon.ravel(), mesh_lat.ravel()
    query_pts = np.column_stack([gx, gy])

    kk = int(min(k, obs_val.size))
    dist, idx = tree.query(query_pts, k=kk)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    w = 1.0 / np.power(dist**2 + smoothing + 1e-12, power / 2.0)
    # Exact hits (distance ~0) dominate; normalise per row.
    wsum = w.sum(axis=1, keepdims=True)
    weights = w / wsum
    neigh_vals = obs_val[idx]
    est = np.sum(weights * neigh_vals, axis=1)

    # Heuristic variance: weighted neighbour spread inflated by mean distance.
    wmean = est[:, None]
    wvar = np.sum(weights * (neigh_vals - wmean) ** 2, axis=1)
    mean_dist = dist.mean(axis=1)
    dist_scale = mean_dist / (np.median(mean_dist) + 1e-12)
    variance = wvar * (1.0 + dist_scale)

    shape = mesh_lon.shape
    return KrigeResult(
        estimate=est.reshape(shape).astype("float32"),
        variance=variance.reshape(shape).astype("float32"),
        method="idw",
    )


def ordinary_kriging(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_val: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    *,
    variogram_model: str = "spherical",
    metric: bool = True,
    idw_k: int = 8,
    idw_power: float = 2.0,
) -> KrigeResult:
    """Ordinary kriging via lazily-imported :mod:`pykrige`, IDW fallback.

    Attempts :class:`pykrige.ok.OrdinaryKriging`; if :mod:`pykrige` is not
    installed (the demo case) it transparently falls back to :func:`idw` so the
    call always succeeds.

    Args:
        obs_lon: Observation longitudes ``(n_obs,)``.
        obs_lat: Observation latitudes ``(n_obs,)``.
        obs_val: Observation values ``(n_obs,)``.
        grid_lon: 1-D ascending target longitudes.
        grid_lat: 1-D ascending target latitudes.
        variogram_model: pykrige variogram model name.
        metric: Use local-km projection for IDW fallback distances.
        idw_k: Neighbour count for the IDW fallback.
        idw_power: IDW power for the fallback.

    Returns:
        A :class:`KrigeResult`.
    """
    obs_lon = np.asarray(obs_lon, dtype="float64")
    obs_lat = np.asarray(obs_lat, dtype="float64")
    obs_val = np.asarray(obs_val, dtype="float64")
    finite = np.isfinite(obs_val) & np.isfinite(obs_lon) & np.isfinite(obs_lat)
    obs_lon, obs_lat, obs_val = obs_lon[finite], obs_lat[finite], obs_val[finite]

    try:
        from pykrige.ok import OrdinaryKriging  # type: ignore

        ok = OrdinaryKriging(
            obs_lon,
            obs_lat,
            obs_val,
            variogram_model=variogram_model,
            enable_plotting=False,
            coordinates_type="geographic" if not metric else "euclidean",
        )
        z, ss = ok.execute("grid", grid_lon, grid_lat)
        return KrigeResult(
            estimate=np.asarray(z, dtype="float32"),
            variance=np.asarray(ss, dtype="float32"),
            method="ordinary_kriging",
        )
    except Exception as exc:
        logger.info("pykrige unavailable (%s); falling back to IDW", type(exc).__name__)
        return idw(
            obs_lon,
            obs_lat,
            obs_val,
            grid_lon,
            grid_lat,
            k=idw_k,
            power=idw_power,
            metric=metric,
        )


def regression_kriging(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_val: np.ndarray,
    obs_covars: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_covars: np.ndarray,
    *,
    trend: Literal["linear", "gbm"] = "linear",
    variogram_model: str = "spherical",
    metric: bool = True,
) -> KrigeResult:
    """Regression kriging: covariate trend + kriged residuals.

    A trend model is fit on ``obs_covars -> obs_val`` (a fast multilinear least
    squares by default, or a lazily-imported gradient-boosting regressor when
    ``trend="gbm"``). The residuals are interpolated with :func:`ordinary_kriging`
    (which itself falls back to IDW), and the gridded trend prediction is added
    back. This matches the blueprint's "GBM trend + OK residuals" recipe.

    Args:
        obs_lon: Observation longitudes ``(n_obs,)``.
        obs_lat: Observation latitudes ``(n_obs,)``.
        obs_val: Observation target values ``(n_obs,)``.
        obs_covars: Covariate matrix at observations ``(n_obs, n_feat)``.
        grid_lon: 1-D ascending target longitudes ``(n_lon,)``.
        grid_lat: 1-D ascending target latitudes ``(n_lat,)``.
        grid_covars: Covariate stack on the grid ``(n_lat, n_lon, n_feat)``.
        trend: ``"linear"`` (OLS) or ``"gbm"`` (gradient boosting).
        variogram_model: Variogram model for the residual kriging.
        metric: Use local-km projection for fallback distances.

    Returns:
        A :class:`KrigeResult` with ``method == "regression_kriging"`` (the
        residual stage may internally have used IDW).
    """
    obs_lon = np.asarray(obs_lon, dtype="float64")
    obs_lat = np.asarray(obs_lat, dtype="float64")
    obs_val = np.asarray(obs_val, dtype="float64")
    obs_covars = np.asarray(obs_covars, dtype="float64")
    grid_covars = np.asarray(grid_covars, dtype="float64")
    n_lat, n_lon, n_feat = grid_covars.shape

    finite = np.isfinite(obs_val) & np.all(np.isfinite(obs_covars), axis=1)
    Xo, yo = obs_covars[finite], obs_val[finite]
    lon_f, lat_f = obs_lon[finite], obs_lat[finite]

    predict: Callable[[np.ndarray], np.ndarray]
    if trend == "gbm":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(random_state=42)
            model.fit(Xo, yo)
            predict = model.predict
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("GBM trend unavailable (%s); using linear", type(exc).__name__)
            trend = "linear"
    if trend == "linear":
        A = np.column_stack([np.ones(len(Xo)), Xo])
        coef, *_ = np.linalg.lstsq(A, yo, rcond=None)

        def predict(feats: np.ndarray) -> np.ndarray:  # noqa: D401
            return np.column_stack([np.ones(len(feats)), feats]) @ coef

    trend_obs = predict(Xo)
    residual = yo - trend_obs

    grid_flat = grid_covars.reshape(-1, n_feat)
    valid_grid = np.all(np.isfinite(grid_flat), axis=1)
    trend_grid = np.full(grid_flat.shape[0], np.nan)
    if valid_grid.any():
        trend_grid[valid_grid] = predict(grid_flat[valid_grid])
    trend_grid = trend_grid.reshape(n_lat, n_lon)

    resid_res = ordinary_kriging(
        lon_f,
        lat_f,
        residual,
        grid_lon,
        grid_lat,
        variogram_model=variogram_model,
        metric=metric,
    )
    estimate = trend_grid + resid_res.estimate
    return KrigeResult(
        estimate=estimate.astype("float32"),
        variance=resid_res.variance,
        method="regression_kriging",
    )


__all__ = [
    "KrigeResult",
    "idw",
    "ordinary_kriging",
    "regression_kriging",
]
