"""Fire -> HCHO correlation, causality and emission-ratio analysis (Objective-2).

Primary (and a fraction of secondary photochemical) HCHO from biomass burning
peaks **0-2 days** after the fire signal.  This module quantifies that link along
several convergent lines (blueprint methods 42-43), because Granger causality
alone is confounded by shared meteorology and summer biogenic isoprene:

* :func:`lagged_xcorr` — deseasonalised lagged cross-correlation of the fire and
  HCHO daily series, expected to peak at a 0-2 day lag.
* :func:`granger` — prewhitened Granger causality test (statsmodels, guarded with
  a graceful fallback) that fire "Granger-causes" HCHO.
* :func:`dhcho_frp_slope` — background-subtracted dHCHO-vs-FRP regression giving
  an emission ratio in **mol HCHO per MW** of fire radiative power.
* :func:`fnr_map` — the HCHO:NO2 ratio (FNR) map that diagnoses the VOC- vs
  NOx-limited photochemical regime.

The public :func:`run` is the Hydra entry point that ties them together on the
synthetic grid + fire table and returns a results dictionary (DEV_CONTRACT §6.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("transport.fire_hcho_corr")

#: Default maximum lag (days) scanned by the cross-correlation.
DEFAULT_MAXLAG: int = 7


@dataclass
class XCorrResult:
    """Lagged cross-correlation result.

    Attributes:
        lags: Integer lag array (positive = HCHO lags fire).
        correlation: Pearson correlation at each lag (deseasonalised series).
        peak_lag: Lag of maximum positive correlation.
        peak_corr: Correlation value at ``peak_lag``.
        in_expected_window: True if ``peak_lag`` falls in the expected 0-2 days.
    """

    lags: np.ndarray
    correlation: np.ndarray
    peak_lag: int
    peak_corr: float
    in_expected_window: bool


def _deseasonalize(x: np.ndarray, window: int = 15) -> np.ndarray:
    """Remove a slow seasonal/trend component via a centred moving average.

    Args:
        x: 1-D series.
        window: Moving-average window in samples; clamped to the series length.

    Returns:
        The anomaly series ``x - moving_average(x)`` (NaNs propagated as 0 mean).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    win = int(min(max(window, 3), n))
    if win % 2 == 0:
        win = max(win - 1, 3)
    kernel = np.ones(win) / win
    # Reflect-pad so the edges are not pulled toward zero.
    pad = win // 2
    padded = np.pad(x, pad, mode="reflect")
    baseline = np.convolve(padded, kernel, mode="valid")
    return x - baseline[:n]


def lagged_xcorr(
    fire_ts: "np.ndarray | pd.Series",
    hcho_ts: "np.ndarray | pd.Series",
    maxlag: int = DEFAULT_MAXLAG,
    *,
    deseasonalize: bool = True,
    season_window: int = 15,
) -> XCorrResult:
    """Deseasonalised lagged cross-correlation between fire and HCHO.

    A positive lag ``k`` correlates ``fire[t]`` with ``hcho[t + k]`` — i.e. how
    strongly today's fire predicts HCHO ``k`` days later.  We expect the peak at
    a 0-2 day lag (primary pyrogenic HCHO plus fast photochemical production).

    Args:
        fire_ts: Daily fire series (count or FRP), length ``n``.
        hcho_ts: Daily receptor HCHO series, length ``n``.
        maxlag: Maximum absolute lag in days to scan.
        deseasonalize: If True, remove a moving-average baseline from both series
            before correlating (the blueprint requirement).
        season_window: Moving-average window (days) for deseasonalisation.

    Returns:
        An :class:`XCorrResult`.

    Raises:
        ValueError: If the two series differ in length or are too short.
    """
    fire = np.asarray(pd.Series(fire_ts).to_numpy(), dtype=float)
    hcho = np.asarray(pd.Series(hcho_ts).to_numpy(), dtype=float)
    if fire.size != hcho.size:
        raise ValueError("fire_ts and hcho_ts must have equal length.")
    if fire.size <= maxlag + 1:
        raise ValueError("Series too short for the requested maxlag.")

    if deseasonalize:
        fire = _deseasonalize(fire, season_window)
        hcho = _deseasonalize(hcho, season_window)

    lags = np.arange(-maxlag, maxlag + 1)
    corr = np.full(lags.size, np.nan)
    for i, k in enumerate(lags):
        if k >= 0:
            a = fire[: fire.size - k]
            b = hcho[k:]
        else:
            a = fire[-k:]
            b = hcho[: hcho.size + k]
        if a.size >= 3 and np.std(a) > 0 and np.std(b) > 0:
            corr[i] = float(np.corrcoef(a, b)[0, 1])

    # Restrict the "peak" search to non-negative lags (causal direction).
    causal = lags >= 0
    causal_corr = np.where(causal, corr, -np.inf)
    peak_idx = int(np.nanargmax(causal_corr))
    peak_lag = int(lags[peak_idx])
    peak_corr = float(corr[peak_idx])
    in_window = 0 <= peak_lag <= 2

    logger.info(
        "lagged xcorr peak r=%.3f at lag=%d day(s) (expected 0-2)",
        peak_corr,
        peak_lag,
    )
    return XCorrResult(
        lags=lags,
        correlation=corr,
        peak_lag=peak_lag,
        peak_corr=peak_corr,
        in_expected_window=in_window,
    )


def _prewhiten(x: np.ndarray) -> np.ndarray:
    """First-difference prewhitening to reduce autocorrelation before Granger.

    Args:
        x: 1-D series.

    Returns:
        The first-differenced series (length ``n - 1``).
    """
    return np.diff(np.asarray(x, dtype=float))


def granger(
    fire_ts: "np.ndarray | pd.Series",
    hcho_ts: "np.ndarray | pd.Series",
    maxlag: int = 3,
    *,
    prewhiten: bool = True,
) -> dict:
    """Prewhitened Granger-causality test that fire predicts HCHO.

    Tests the null that the fire series does **not** Granger-cause the HCHO
    series.  Both series are first-differenced (prewhitened) to strip the shared
    autocorrelation that otherwise inflates the statistic.  Wrapped in a guard so
    a missing/failed statsmodels call degrades to ``available=False`` rather than
    raising.

    Args:
        fire_ts: Daily fire series.
        hcho_ts: Daily receptor HCHO series.
        maxlag: Maximum lag (days) tested.
        prewhiten: If True (default), first-difference both series first.

    Returns:
        Dict with ``available`` (bool), and when available ``min_p_value``,
        ``best_lag`` (lag with the smallest p-value), ``p_values`` (per-lag dict)
        and ``causes`` (``min_p_value < 0.05``).
    """
    fire = np.asarray(pd.Series(fire_ts).to_numpy(), dtype=float)
    hcho = np.asarray(pd.Series(hcho_ts).to_numpy(), dtype=float)
    if prewhiten:
        fire = _prewhiten(fire)
        hcho = _prewhiten(hcho)
    if fire.size != hcho.size or fire.size < 3 * maxlag + 1:
        return {"available": False, "reason": "series too short after prewhitening"}

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:  # pragma: no cover - statsmodels in the light set
        logger.warning("statsmodels unavailable; skipping Granger test")
        return {"available": False, "reason": "statsmodels not importable"}

    # grangercausalitytests(data) tests whether column 1 is caused by column 0,
    # so the array is [hcho (effect), fire (cause)].
    data = np.column_stack([hcho, fire])
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    except TypeError:  # newer statsmodels dropped the verbose kwarg
        res = grangercausalitytests(data, maxlag=maxlag)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Granger test failed: %s", exc)
        return {"available": False, "reason": str(exc)}

    p_values = {
        lag: float(stat[0]["ssr_ftest"][1]) for lag, stat in res.items()
    }
    best_lag = int(min(p_values, key=p_values.get))
    min_p = float(p_values[best_lag])
    logger.info("Granger min p=%.3g at lag=%d", min_p, best_lag)
    return {
        "available": True,
        "p_values": p_values,
        "best_lag": best_lag,
        "min_p_value": min_p,
        "causes": min_p < 0.05,
    }


def dhcho_frp_slope(
    frp: "np.ndarray | pd.Series",
    hcho: "np.ndarray | pd.Series",
    *,
    background_quantile: float = 0.1,
) -> dict:
    """Background-subtracted dHCHO-vs-FRP regression (emission ratio).

    Subtracts a clean-air HCHO background (a low quantile of the receptor series)
    then fits ``dHCHO = slope * FRP + intercept`` by ordinary least squares.  The
    slope is the apparent **emission ratio in (HCHO column units) per MW**.

    Args:
        frp: Fire radiative power series (MW), aligned to ``hcho``.
        hcho: Receptor HCHO column series (mol/m2), same length as ``frp``.
        background_quantile: Quantile of HCHO taken as the clean background that
            is subtracted before the fit (default 0.1).

    Returns:
        Dict with ``slope`` (mol per MW), ``intercept``, ``r`` (Pearson r),
        ``r2``, ``background`` and ``n``. Degenerate input yields NaN slopes.
    """
    frp = np.asarray(pd.Series(frp).to_numpy(), dtype=float)
    hcho = np.asarray(pd.Series(hcho).to_numpy(), dtype=float)
    mask = np.isfinite(frp) & np.isfinite(hcho)
    frp, hcho = frp[mask], hcho[mask]
    if frp.size < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r": float("nan"),
            "r2": float("nan"),
            "background": float("nan"),
            "n": int(frp.size),
        }

    background = float(np.quantile(hcho, background_quantile))
    dhcho = hcho - background

    if np.std(frp) == 0:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r": float("nan"),
            "r2": float("nan"),
            "background": background,
            "n": int(frp.size),
        }

    slope, intercept = np.polyfit(frp, dhcho, 1)
    r = float(np.corrcoef(frp, dhcho)[0, 1]) if np.std(dhcho) > 0 else float("nan")
    logger.info("dHCHO/FRP slope=%.3e mol per MW (r=%.3f)", slope, r)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": r,
        "r2": float(r * r) if np.isfinite(r) else float("nan"),
        "background": background,
        "n": int(frp.size),
    }


def fnr_map(
    hcho: "np.ndarray | xr.DataArray",
    no2: "np.ndarray | xr.DataArray",
    *,
    min_no2: float = 1e-30,
) -> "np.ndarray | xr.DataArray":
    """Formaldehyde-to-NO2 ratio (FNR) map diagnosing the VOC/NOx regime.

    FNR = HCHO / NO2.  Low FNR (< ~1) indicates a VOC-limited (NOx-saturated)
    regime; high FNR (> ~2) an NOx-limited regime, with a transitional band
    between.  Division is guarded against ~0 NO2.

    Args:
        hcho: HCHO column field (array or ``xarray.DataArray``).
        no2: NO2 column field, broadcastable to ``hcho``.
        min_no2: NO2 values at/below this are masked to NaN to avoid blow-ups.

    Returns:
        The FNR field, same type/shape as ``hcho`` (NaN where NO2 is too small).
    """
    try:
        import xarray as xr

        if isinstance(hcho, xr.DataArray) or isinstance(no2, xr.DataArray):
            safe_no2 = no2.where(no2 > min_no2)
            fnr = hcho / safe_no2
            return fnr.rename("fnr")
    except Exception:  # pragma: no cover - xarray is in the light set
        pass

    hcho_a = np.asarray(hcho, dtype=float)
    no2_a = np.asarray(no2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fnr = np.where(no2_a > min_no2, hcho_a / no2_a, np.nan)
    return fnr


def classify_fnr(fnr: float, *, voc_edge: float = 1.0, nox_edge: float = 2.0) -> str:
    """Label an FNR value as a photochemical regime.

    Args:
        fnr: Formaldehyde-to-NO2 ratio value.
        voc_edge: Upper FNR bound of the VOC-limited regime.
        nox_edge: Lower FNR bound of the NOx-limited regime.

    Returns:
        ``"VOC-limited"``, ``"transitional"``, ``"NOx-limited"`` or ``"unknown"``.
    """
    if not np.isfinite(fnr):
        return "unknown"
    if fnr < voc_edge:
        return "VOC-limited"
    if fnr > nox_edge:
        return "NOx-limited"
    return "transitional"


def _receptor_series(
    ds: "xr.Dataset",
    bbox: tuple[float, float, float, float],
    var: str,
) -> pd.Series:
    """Average a gridded variable over a bbox into a daily series.

    Args:
        ds: Gridded cube (DEV_CONTRACT §6.1) with ``time``/``lat``/``lon``.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` receptor area.
        var: Data-variable name (e.g. ``"hcho_col"``).

    Returns:
        A daily ``pd.Series`` indexed by time.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    sub = ds[var].sel(
        lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)
    )
    series = sub.mean(dim=("lat", "lon"), skipna=True).to_series()
    return series


def run(cfg) -> dict:
    """Hydra entry point: fire->HCHO correlation + causality on synthetic data.

    Loads the fused grid cube and fire table, builds AOI-restricted fire and
    receptor-HCHO daily series, then returns the lagged cross-correlation, the
    prewhitened Granger result, the dHCHO/FRP emission-ratio slope and a summary.

    Args:
        cfg: Composed Hydra config. Uses ``cfg.paths.data_processed`` and
            (optionally) ``cfg.aoi.bbox`` for the source region.

    Returns:
        A results dict with keys ``xcorr``, ``granger``, ``emission_ratio``,
        ``n_days``, ``source_aoi`` and ``receptor_bbox``.
    """
    from pathlib import Path

    import xarray as xr

    from ..utils.geo import IGP_BBOX, PUNJAB_HARYANA_BBOX
    from ..utils.io import load_parquet
    from .fire_periods import daily_fire_series

    processed = Path(cfg.paths.data_processed)
    ds = xr.open_dataset(processed / "grid.nc")
    fires = load_parquet(processed / "fires.parquet")

    source_aoi = tuple(getattr(getattr(cfg, "aoi", {}), "bbox", PUNJAB_HARYANA_BBOX))
    receptor_bbox = IGP_BBOX

    fire_daily = daily_fire_series(fires, source_aoi)
    hcho_series = _receptor_series(ds, receptor_bbox, "hcho_col")

    # Align fire and HCHO onto a common daily index.
    fire_s = pd.Series(
        fire_daily["frp_sum"].to_numpy(),
        index=pd.DatetimeIndex(fire_daily["date"]),
    )
    joined = pd.concat(
        {"frp": fire_s, "hcho": hcho_series}, axis=1
    ).dropna()
    n_days = int(len(joined))

    if n_days <= DEFAULT_MAXLAG + 2:
        logger.warning("Only %d aligned days; correlation may be unstable", n_days)

    maxlag = min(DEFAULT_MAXLAG, max(n_days // 3, 1))
    xc = lagged_xcorr(joined["frp"].to_numpy(), joined["hcho"].to_numpy(), maxlag)
    gr = granger(joined["frp"].to_numpy(), joined["hcho"].to_numpy(), min(3, maxlag))
    ratio = dhcho_frp_slope(joined["frp"].to_numpy(), joined["hcho"].to_numpy())

    return {
        "xcorr": {
            "lags": xc.lags.tolist(),
            "correlation": xc.correlation.tolist(),
            "peak_lag": xc.peak_lag,
            "peak_corr": xc.peak_corr,
            "in_expected_window": xc.in_expected_window,
        },
        "granger": gr,
        "emission_ratio": ratio,
        "n_days": n_days,
        "source_aoi": list(source_aoi),
        "receptor_bbox": list(receptor_bbox),
    }


__all__ = [
    "DEFAULT_MAXLAG",
    "XCorrResult",
    "classify_fnr",
    "dhcho_frp_slope",
    "fnr_map",
    "granger",
    "lagged_xcorr",
    "run",
]
