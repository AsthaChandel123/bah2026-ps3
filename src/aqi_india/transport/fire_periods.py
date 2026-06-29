"""Biomass-burning period extraction from FIRMS fire detections (Objective-2).

Stubble- and forest-burning produce *episodic* bursts of fire activity that ride
on top of a strong seasonal cycle.  To attribute downwind HCHO to fires we first
need clean, objective episode windows.  This module:

1.  reduces a FIRMS-surrogate fire point table (schema in ``DEV_CONTRACT`` §6.3)
    to a daily, area-of-interest fire series — both detection **count** and
    summed **FRP** (fire radiative power, MW) which the blueprint mandates we use
    because count-only correlation under-detects evening stubble fires;
2.  removes the seasonal cycle with STL decomposition (statsmodels) and falls
    back to a centred moving-average baseline when statsmodels / a usable period
    is unavailable;
3.  flags an **episode** wherever the FRP residual exceeds ``+2 sigma`` of the
    residual distribution and that exceedance is **sustained for >= 2 days**
    (blueprint method 41); and
4.  tags every episode with the burning **season** it falls in
    (``oct_nov`` post-monsoon Punjab/Haryana stubble, ``apr_may`` pre-monsoon
    forest belt, otherwise ``other``).

The public :func:`extract_episodes` returns the daily series *and* a tidy table
of episode intervals.  Everything runs on the light dependency set with a
graceful fallback when statsmodels is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..utils.geo import bbox_to_polygon
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

logger = get_logger("transport.fire_periods")

#: Default residual exceedance threshold in units of residual sigma.
DEFAULT_SIGMA: float = 2.0

#: Minimum consecutive days above threshold to qualify as an episode.
DEFAULT_MIN_DURATION: int = 2

#: Default STL seasonal period (days) — an annual fire cycle sampled daily.
DEFAULT_PERIOD: int = 365

#: Burning-season month windows (inclusive month numbers).
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "oct_nov": (10, 11),
    "apr_may": (4, 5),
}


@dataclass
class FireEpisode:
    """A single sustained biomass-burning episode.

    Attributes:
        start: First day of the episode (``datetime64``-compatible Timestamp).
        end: Last day of the episode (inclusive).
        duration_days: Number of days spanned (``end - start + 1``).
        peak_date: Day of maximum FRP residual within the episode.
        peak_frp: Total FRP on the peak day (MW).
        total_frp: FRP summed across the episode (MW-days).
        mean_residual_sigma: Mean residual (in sigma units) over the episode.
        season: One of ``"oct_nov"``, ``"apr_may"`` or ``"other"``.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: int
    peak_date: pd.Timestamp
    peak_frp: float
    total_frp: float
    mean_residual_sigma: float
    season: str


@dataclass
class EpisodeResult:
    """Container returned by :func:`extract_episodes`.

    Attributes:
        daily: Daily series with columns ``date``, ``fire_count``, ``frp_sum``,
            ``trend``, ``seasonal``, ``residual``, ``residual_sigma`` and the
            boolean ``is_episode`` flag.
        episodes: Tidy ``DataFrame`` of episode intervals (one row each).
        threshold_sigma: The sigma multiplier used.
        method: ``"stl"`` or ``"moving_average"`` (which baseline was used).
    """

    daily: pd.DataFrame
    episodes: pd.DataFrame
    threshold_sigma: float
    method: str
    intervals: list[FireEpisode] = field(default_factory=list)


def _clip_to_aoi(
    fire_df: "pd.DataFrame | gpd.GeoDataFrame",
    aoi: Optional[tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Restrict the fire table to an AOI bounding box.

    Args:
        fire_df: Fire detections with ``lat``/``lon`` columns (DEV_CONTRACT §6.3).
        aoi: ``(min_lon, min_lat, max_lon, max_lat)`` box, or ``None`` for all.

    Returns:
        A plain ``DataFrame`` (geometry dropped) of the rows inside the AOI.
    """
    df = pd.DataFrame(fire_df).copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")
    if aoi is None:
        return df
    min_lon, min_lat, max_lon, max_lat = aoi
    mask = (
        (df["lon"] >= min_lon)
        & (df["lon"] <= max_lon)
        & (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
    )
    return df.loc[mask].reset_index(drop=True)


def daily_fire_series(
    fire_df: "pd.DataFrame | gpd.GeoDataFrame",
    aoi: Optional[tuple[float, float, float, float]] = None,
    *,
    date_col: str = "date",
    frp_col: str = "frp",
) -> pd.DataFrame:
    """Aggregate fire points to a gap-free daily count + FRP series.

    The series is re-indexed to a *contiguous* daily calendar between the first
    and last detection so the decomposition sees no missing days (missing days
    are genuine zeros — no fire detected).

    Args:
        fire_df: FIRMS-surrogate fire detections (DEV_CONTRACT §6.3).
        aoi: Optional ``(min_lon, min_lat, max_lon, max_lat)`` AOI box.
        date_col: Name of the acquisition-date column.
        frp_col: Name of the fire-radiative-power column.

    Returns:
        ``DataFrame`` with columns ``date`` (daily), ``fire_count`` (int) and
        ``frp_sum`` (float, MW). Empty input yields an empty frame.
    """
    df = _clip_to_aoi(fire_df, aoi)
    if df.empty:
        return pd.DataFrame(columns=["date", "fire_count", "frp_sum"])

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    grouped = (
        df.groupby(date_col)
        .agg(fire_count=(date_col, "size"), frp_sum=(frp_col, "sum"))
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    full_index = pd.date_range(
        grouped["date"].min(), grouped["date"].max(), freq="D"
    )
    grouped = (
        grouped.set_index("date")
        .reindex(full_index, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    grouped["fire_count"] = grouped["fire_count"].astype(int)
    grouped["frp_sum"] = grouped["frp_sum"].astype(float)
    return grouped


def _moving_average_decompose(
    values: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Additive decomposition via a centred moving-average trend.

    A robust, dependency-free fallback for STL: estimate the trend with a
    centred moving average (window = ``period`` clamped to the series length),
    estimate the seasonal term as the per-phase mean of the detrended series,
    and take the residual as the remainder.

    Args:
        values: 1-D series (FRP sum per day).
        period: Seasonal period in samples.

    Returns:
        ``(trend, seasonal, residual)`` arrays, same length as ``values``.
    """
    n = values.size
    win = int(min(max(period, 3), n))
    if win % 2 == 0:  # centred MA needs an odd window
        win = max(win - 1, 3)
    kernel = np.ones(win) / win
    trend = np.convolve(values, kernel, mode="same")
    detrended = values - trend

    if period >= 2 and n >= period:
        phase = np.arange(n) % period
        seasonal = np.zeros(n)
        for p in range(period):
            sel = phase == p
            if sel.any():
                seasonal[sel] = detrended[sel].mean()
        seasonal -= seasonal.mean()  # additive seasonal sums to ~0
    else:
        seasonal = np.zeros(n)

    residual = values - trend - seasonal
    return trend, seasonal, residual


def _stl_decompose(
    series: pd.Series, period: int
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Attempt an STL decomposition; return ``None`` if unavailable.

    Args:
        series: Daily FRP series indexed by date.
        period: Seasonal period in samples (must be ``>= 2`` and ``< n``).

    Returns:
        ``(trend, seasonal, residual)`` arrays, or ``None`` to signal that the
        caller should use the moving-average fallback.
    """
    n = series.size
    if period < 2 or n < 2 * period:
        # STL requires at least two full periods; defer to the fallback.
        return None
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception:  # pragma: no cover - statsmodels is in the light set
        logger.warning("statsmodels unavailable; using moving-average fallback")
        return None
    try:
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        return (
            np.asarray(res.trend),
            np.asarray(res.seasonal),
            np.asarray(res.resid),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("STL failed (%s); using moving-average fallback", exc)
        return None


def _runs_above(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Find index ranges where ``mask`` is True for at least ``min_len`` samples.

    Args:
        mask: Boolean array.
        min_len: Minimum run length to keep.

    Returns:
        List of ``(start_idx, end_idx_inclusive)`` index pairs.
    """
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))
    return runs


def _season_for(date: pd.Timestamp) -> str:
    """Map a date to its burning season tag.

    Args:
        date: A timestamp.

    Returns:
        ``"oct_nov"``, ``"apr_may"`` or ``"other"``.
    """
    for tag, months in SEASON_MONTHS.items():
        if date.month in months:
            return tag
    return "other"


def extract_episodes(
    fire_df: "pd.DataFrame | gpd.GeoDataFrame",
    aoi: Optional[tuple[float, float, float, float]] = None,
    *,
    period: int = DEFAULT_PERIOD,
    sigma: float = DEFAULT_SIGMA,
    min_duration: int = DEFAULT_MIN_DURATION,
    date_col: str = "date",
    frp_col: str = "frp",
) -> EpisodeResult:
    """Extract sustained biomass-burning episodes from a fire table.

    Builds the daily FRP series, removes the seasonal cycle (STL with a
    moving-average fallback), and flags FRP-weighted episodes where the residual
    exceeds ``+sigma`` standard deviations for at least ``min_duration`` days.

    Args:
        fire_df: FIRMS-surrogate fire detections (DEV_CONTRACT §6.3) with
            ``date``, ``lat``, ``lon`` and ``frp`` columns.
        aoi: Optional ``(min_lon, min_lat, max_lon, max_lat)`` AOI box restricting
            the fires considered (e.g. ``PUNJAB_HARYANA_BBOX``).
        period: STL seasonal period in days. When the series is shorter than two
            periods the function auto-shrinks the effective period so short
            synthetic series still decompose.
        sigma: Residual threshold in standard-deviation units (default ``2.0``).
        min_duration: Minimum sustained days above threshold (default ``2``).
        date_col: Name of the acquisition-date column.
        frp_col: Name of the FRP column.

    Returns:
        An :class:`EpisodeResult` with the annotated daily series, an episode
        table and the list of :class:`FireEpisode` objects.

    Raises:
        ValueError: If the AOI clips away every fire and no series can be built.
    """
    daily = daily_fire_series(fire_df, aoi, date_col=date_col, frp_col=frp_col)
    if daily.empty:
        raise ValueError("No fire detections inside the AOI to build a series.")

    series = pd.Series(
        daily["frp_sum"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(daily["date"]),
        name="frp_sum",
    )
    n = series.size

    # Adapt the period so even short synthetic series can be decomposed: pick the
    # largest period <= requested that still leaves two full cycles, else half n.
    eff_period = int(min(period, n // 2)) if n >= 4 else 0
    if eff_period < 2:
        eff_period = max(min(7, n // 2), 0)

    stl = _stl_decompose(series, eff_period) if eff_period >= 2 else None
    if stl is not None:
        trend, seasonal, residual = stl
        method = "stl"
    else:
        trend, seasonal, residual = _moving_average_decompose(
            series.to_numpy(dtype=float), max(eff_period, 1)
        )
        method = "moving_average"

    resid_std = float(np.nanstd(residual))
    if resid_std <= 0 or not np.isfinite(resid_std):
        resid_std = 1.0  # degenerate flat series -> no episodes
    resid_sigma = residual / resid_std

    daily = daily.copy()
    daily["trend"] = trend
    daily["seasonal"] = seasonal
    daily["residual"] = residual
    daily["residual_sigma"] = resid_sigma

    above = resid_sigma > sigma
    runs = _runs_above(above, min_duration)
    daily["is_episode"] = False

    intervals: list[FireEpisode] = []
    for start_i, end_i in runs:
        block = daily.iloc[start_i : end_i + 1]
        daily.iloc[start_i : end_i + 1, daily.columns.get_loc("is_episode")] = True
        peak_idx = block["residual_sigma"].idxmax()
        peak_row = daily.loc[peak_idx]
        start_date = pd.Timestamp(block["date"].iloc[0])
        intervals.append(
            FireEpisode(
                start=start_date,
                end=pd.Timestamp(block["date"].iloc[-1]),
                duration_days=int(end_i - start_i + 1),
                peak_date=pd.Timestamp(peak_row["date"]),
                peak_frp=float(peak_row["frp_sum"]),
                total_frp=float(block["frp_sum"].sum()),
                mean_residual_sigma=float(block["residual_sigma"].mean()),
                season=_season_for(start_date),
            )
        )

    episodes = pd.DataFrame(
        [
            {
                "start": e.start,
                "end": e.end,
                "duration_days": e.duration_days,
                "peak_date": e.peak_date,
                "peak_frp": e.peak_frp,
                "total_frp": e.total_frp,
                "mean_residual_sigma": e.mean_residual_sigma,
                "season": e.season,
            }
            for e in intervals
        ],
        columns=[
            "start",
            "end",
            "duration_days",
            "peak_date",
            "peak_frp",
            "total_frp",
            "mean_residual_sigma",
            "season",
        ],
    )

    logger.info(
        "extracted %d biomass-burning episode(s) via %s over %d days",
        len(intervals),
        method,
        n,
    )
    return EpisodeResult(
        daily=daily,
        episodes=episodes,
        threshold_sigma=sigma,
        method=method,
        intervals=intervals,
    )


def aoi_from_name(name: str) -> tuple[float, float, float, float]:
    """Resolve a named source region to its bounding box.

    Args:
        name: ``"punjab_haryana"``, ``"igp"``, ``"forest_belt"`` or ``"india"``.

    Returns:
        The ``(min_lon, min_lat, max_lon, max_lat)`` box.

    Raises:
        KeyError: If the name is not recognised.
    """
    from ..utils import geo

    table = {
        "india": geo.INDIA_BBOX,
        "igp": geo.IGP_BBOX,
        "punjab_haryana": geo.PUNJAB_HARYANA_BBOX,
        "forest_belt": geo.FOREST_BELT_BBOX,
    }
    if name not in table:
        raise KeyError(f"Unknown AOI '{name}'. Choose from {sorted(table)}.")
    return table[name]


__all__ = [
    "DEFAULT_MIN_DURATION",
    "DEFAULT_PERIOD",
    "DEFAULT_SIGMA",
    "SEASON_MONTHS",
    "EpisodeResult",
    "FireEpisode",
    "aoi_from_name",
    "bbox_to_polygon",
    "daily_fire_series",
    "extract_episodes",
]
