"""Temporal feature engineering and sequence construction.

Air quality has strong cyclic (diurnal/seasonal) structure and multi-day memory:
nighttime boundary-layer collapse, several-day accumulation under stagnant
synoptic conditions, and the post-monsoon stubble-burning ramp. This module turns
the daily time axis into model-ready features:

* **Cyclic encodings** — day-of-year and (optionally) day-of-week as
  ``sin``/``cos`` pairs so the learner sees a smooth, wrap-around calendar.
* **Lag features** — value at ``t - k`` days (pollutant persistence).
* **Rolling statistics** — trailing-window means/std (accumulation, smoothing).
* **Sequence tensors** — ``build_sequences`` packs the gridded cube into
  ``[B, T, C, H, W]`` windows for ConvLSTM/SA-ConvLSTM, and
  ``build_station_sequences`` builds per-station ``[N, T, C]`` windows for an LSTM.

Everything is NumPy/pandas-vectorized and NaN-safe. Heavy frameworks are never
imported — sequence tensors are plain ``numpy.ndarray`` so the demo runs on the
light dependency set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import xarray as xr

#: Length of the year used for the day-of-year cyclic encoding.
_DAYS_IN_YEAR: float = 365.25


def day_of_year_features(
    times: "np.ndarray | pd.DatetimeIndex | xr.DataArray",
) -> dict[str, np.ndarray]:
    """Cyclic ``sin``/``cos`` encodings of the day-of-year.

    Encodes the calendar position so that Dec 31 and Jan 1 are adjacent in
    feature space (no artificial discontinuity at the year boundary).

    Args:
        times: Datetime-like 1-D array/index (``datetime64[ns]``).

    Returns:
        Dict with keys ``doy_sin`` and ``doy_cos`` (float arrays, same length as
        ``times``).
    """
    import pandas as pd

    idx = pd.DatetimeIndex(np.asarray(times).ravel())
    doy = idx.dayofyear.to_numpy(dtype=np.float64)
    angle = 2.0 * np.pi * doy / _DAYS_IN_YEAR
    return {"doy_sin": np.sin(angle), "doy_cos": np.cos(angle)}


def day_of_week_features(
    times: "np.ndarray | pd.DatetimeIndex | xr.DataArray",
) -> dict[str, np.ndarray]:
    """Cyclic ``sin``/``cos`` encodings of the day-of-week (traffic weekly cycle).

    Args:
        times: Datetime-like 1-D array/index.

    Returns:
        Dict with keys ``dow_sin`` and ``dow_cos``.
    """
    import pandas as pd

    idx = pd.DatetimeIndex(np.asarray(times).ravel())
    dow = idx.dayofweek.to_numpy(dtype=np.float64)  # Mon=0..Sun=6
    angle = 2.0 * np.pi * dow / 7.0
    return {"dow_sin": np.sin(angle), "dow_cos": np.cos(angle)}


def add_lag_features(
    df: "pd.DataFrame",
    columns: "list[str]",
    lags: "list[int]",
    *,
    group_col: str | None = None,
    time_col: str = "time",
) -> "pd.DataFrame":
    """Append lagged copies of ``columns`` (value at ``t - lag`` days).

    Rows are sorted by ``time`` (within each group when ``group_col`` is given)
    before shifting so lags respect temporal order. New columns are named
    ``{col}_lag{lag}``. Missing history yields ``NaN``.

    Args:
        df: Long-format table (one row per (entity, time)).
        columns: Columns to lag.
        lags: Positive integer lags in days.
        group_col: Entity key (e.g. ``station_id`` or ``h3_res7``) to lag within;
            ``None`` lags the whole frame as one series.
        time_col: Name of the datetime column used for ordering.

    Returns:
        A copy of ``df`` with the lag columns added.
    """
    out = df.sort_values(([group_col] if group_col else []) + [time_col]).copy()
    grouped = out.groupby(group_col, sort=False) if group_col else None
    for col in columns:
        for lag in lags:
            name = f"{col}_lag{lag}"
            if grouped is not None:
                out[name] = grouped[col].shift(lag)
            else:
                out[name] = out[col].shift(lag)
    return out.loc[df.index] if df.index.equals(out.index) else out


def add_rolling_features(
    df: "pd.DataFrame",
    columns: "list[str]",
    windows: "list[int]",
    *,
    group_col: str | None = None,
    time_col: str = "time",
    stats: "tuple[str, ...]" = ("mean",),
    min_periods: int = 1,
) -> "pd.DataFrame":
    """Append trailing rolling statistics of ``columns``.

    The window is *causal* (uses only past/current values via ``shift(1)`` would
    drop the current; here the window includes the current day) and respects
    group/time ordering. New columns are named ``{col}_roll{window}_{stat}``.

    Args:
        df: Long-format table.
        columns: Columns to aggregate.
        windows: Trailing window sizes in days.
        group_col: Entity key to roll within; ``None`` rolls the whole frame.
        time_col: Datetime column used for ordering.
        stats: Subset of ``{"mean", "std", "min", "max", "sum"}``.
        min_periods: Minimum observations in a window to emit a value.

    Returns:
        A copy of ``df`` with the rolling-statistic columns added.
    """
    out = df.sort_values(([group_col] if group_col else []) + [time_col]).copy()
    for col in columns:
        for window in windows:
            if group_col is not None:
                roller = out.groupby(group_col, sort=False)[col].rolling(
                    window, min_periods=min_periods
                )
                for stat in stats:
                    series = getattr(roller, stat)().reset_index(level=0, drop=True)
                    out[f"{col}_roll{window}_{stat}"] = series
            else:
                roller = out[col].rolling(window, min_periods=min_periods)
                for stat in stats:
                    out[f"{col}_roll{window}_{stat}"] = getattr(roller, stat)()
    return out.loc[df.index] if df.index.equals(out.index) else out


def build_sequences(
    cube: "np.ndarray | xr.Dataset",
    *,
    variables: "list[str] | None" = None,
    window: int = 7,
    stride: int = 1,
    return_times: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Build sliding ``[B, T, C, H, W]`` spatiotemporal tensors for ConvLSTM.

    Stacks the requested gridded variables into a channel axis, then slides a
    length-``window`` temporal window over the daily time axis to produce a batch
    of overlapping sequences. NaNs are preserved (the model/gap-fill handles
    them); no normalization is applied here.

    Args:
        cube: Either an :class:`xarray.Dataset` (dims ``(time, lat, lon)`` per
            variable) or a pre-stacked ``[T, C, H, W]`` ndarray.
        variables: When ``cube`` is a Dataset, the data-vars to stack as channels
            (in order). Defaults to all data-vars.
        window: Temporal length ``T`` of each sequence (days).
        stride: Step between consecutive window starts.
        return_times: If True (and ``cube`` is a Dataset), also return the array
            of window *end* timestamps aligned to each sequence.

    Returns:
        A ``float32`` array of shape ``[B, T, C, H, W]``; if ``return_times`` and
        a Dataset was given, a ``(sequences, end_times)`` tuple.

    Raises:
        ValueError: If ``window`` exceeds the number of available time steps.
    """
    times: np.ndarray | None = None
    if _is_dataset(cube):
        stacked, times = _stack_dataset_to_tchw(cube, variables)
    else:
        stacked = np.asarray(cube, dtype=np.float32)
        if stacked.ndim != 4:
            raise ValueError(
                "Pre-stacked cube must have shape [T, C, H, W]; "
                f"got ndim={stacked.ndim}."
            )

    n_t = stacked.shape[0]
    if window > n_t:
        raise ValueError(
            f"window={window} exceeds number of time steps ({n_t})."
        )
    starts = range(0, n_t - window + 1, stride)
    seqs = np.stack([stacked[s : s + window] for s in starts], axis=0).astype(
        np.float32
    )
    if return_times and times is not None:
        end_times = np.array([times[s + window - 1] for s in starts])
        return seqs, end_times
    return seqs


def _stack_dataset_to_tchw(
    ds: "xr.Dataset", variables: "list[str] | None"
) -> tuple[np.ndarray, np.ndarray]:
    """Stack a ``(time, lat, lon)`` Dataset into a ``[T, C, H, W]`` array."""
    var_names = list(variables) if variables is not None else list(ds.data_vars)
    if not var_names:
        raise ValueError("No variables to stack.")
    channels = []
    for name in var_names:
        da = ds[name].transpose("time", "lat", "lon")
        channels.append(np.asarray(da.values, dtype=np.float32))
    stacked = np.stack(channels, axis=1)  # [T, C, H, W]
    times = np.asarray(ds["time"].values)
    return stacked, times


def build_station_sequences(
    df: "pd.DataFrame",
    feature_cols: "list[str]",
    *,
    target_col: str | None = None,
    group_col: str = "station_id",
    time_col: str = "time",
    window: int = 7,
    stride: int = 1,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]":
    """Build per-station ``[N, T, C]`` temporal windows for an LSTM.

    For each station the rows are sorted by time and a length-``window`` window is
    slid over them; the target (if given) is taken at the *last* step of each
    window (next-step / nowcast convention).

    Args:
        df: Long-format station table (one row per station-day).
        feature_cols: Predictor columns to stack as the channel axis ``C``.
        target_col: Optional label column; when given, ``y`` is returned.
        group_col: Station identifier column.
        time_col: Datetime column used for ordering.
        window: Sequence length ``T`` (days).
        stride: Step between consecutive windows.

    Returns:
        ``(X, groups)`` or ``(X, y, groups)`` where ``X`` is ``[N, T, C]``
        ``float32``, ``y`` is ``[N]`` ``float32`` (window-end target), and
        ``groups`` is ``[N]`` of the station id per sequence (for grouped CV).
    """
    sorted_df = df.sort_values([group_col, time_col])
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    grp_list: list[object] = []

    for station, sdf in sorted_df.groupby(group_col, sort=False):
        feats = sdf[feature_cols].to_numpy(dtype=np.float32)
        targets = (
            sdf[target_col].to_numpy(dtype=np.float32)
            if target_col is not None
            else None
        )
        n_rows = feats.shape[0]
        if n_rows < window:
            continue
        for s in range(0, n_rows - window + 1, stride):
            x_list.append(feats[s : s + window])
            grp_list.append(station)
            if targets is not None:
                y_list.append(targets[s + window - 1])

    if x_list:
        x = np.stack(x_list, axis=0).astype(np.float32)
    else:  # preserve [N, T, C] rank even when empty
        x = np.empty((0, window, len(feature_cols)), dtype=np.float32)
    groups = np.array(grp_list, dtype=object)

    if target_col is not None:
        y = np.array(y_list, dtype=np.float32)
        return x, y, groups
    return x, groups


def _is_dataset(obj: object) -> bool:
    """Return True if ``obj`` is an xarray Dataset (no eager xarray import)."""
    return type(obj).__name__ == "Dataset" and type(obj).__module__.startswith(
        "xarray"
    )


__all__ = [
    "day_of_year_features",
    "day_of_week_features",
    "add_lag_features",
    "add_rolling_features",
    "build_sequences",
    "build_station_sequences",
]
