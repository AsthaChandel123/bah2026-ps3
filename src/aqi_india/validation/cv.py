"""Cross-validation schemes for surface-AQI models — the CV LADDER.

The single most important correctness lever in this product is **how we split
train from test**. Surface pollutant fields are strongly autocorrelated in both
space (neighbouring CPCB stations see the same airshed) and time (today's PM2.5
predicts tomorrow's). A naive random k-fold split therefore leaks information:
a test row's near-twin sits in the training set, so the model "remembers" rather
than generalises, inflating R by 0.10-0.25. Judges and reviewers catch this.

This module implements a *ladder* of progressively stricter splitters, each
exposing a different leakage axis:

    random_kfold          baseline only — present to QUANTIFY the leakage gap
    leave_time_out        hold out whole days (temporal blocks)
    forward_chaining      expanding-window, train-on-past / test-on-future
    leave_station_out     hold out whole stations (a station is never in both
                          train and test — the core fairness guarantee)
    spatial_block_cv      hold out variogram-sized spatial blocks with a buffer
                          gap, so adjacent autocorrelated cells never straddle
                          the split
    spatiotemporal_blocked_cv   GOLD — block in space AND time simultaneously

Each splitter is a **generator** yielding ``(train_idx, test_idx)`` integer
index arrays over the rows of the feature matrix, mirroring the scikit-learn
``split`` protocol but with the domain-specific grouping this problem needs.

:func:`cross_validate` runs one scheme end-to-end against a ``model_factory``
and returns per-fold + aggregate metrics. :func:`run_cv_ladder` runs every
scheme and returns a tidy comparison table — the "leakage-gap story" that is the
headline validation deliverable.

The module imports under the light dependency set (numpy + pandas only at module
top); scikit-learn / lightgbm live behind the user-supplied ``model_factory``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import metrics_table

#: Type alias for a fold: (train indices, test indices) into the feature matrix.
Fold = tuple[np.ndarray, np.ndarray]

#: Type alias for a callable that returns a fresh, untrained estimator exposing
#: scikit-learn-style ``fit(X, y)`` and ``predict(X)`` methods.
ModelFactory = Callable[[], Any]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_2d(x: Any) -> np.ndarray:
    """Return ``x`` as a 2-D float array (rows = samples), preserving order.

    Accepts NumPy arrays and pandas DataFrames/Series; 1-D input becomes a
    single-column matrix.
    """
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _n_rows(x: Any) -> int:
    """Number of sample rows in ``x`` (array or pandas object)."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return len(x)
    return int(np.asarray(x).shape[0])


def _check_groups(groups: Any, n: int, name: str) -> np.ndarray:
    """Validate and return a 1-D group label array of length ``n``."""
    if groups is None:
        raise ValueError(f"{name} requires a `groups` array (got None)")
    g = np.asarray(groups)
    if g.ndim != 1 or g.shape[0] != n:
        raise ValueError(
            f"{name}: `groups` must be 1-D of length {n}, got shape {g.shape}"
        )
    return g


def _digitize_blocks(values: np.ndarray, block_size: float) -> np.ndarray:
    """Assign each value to an integer block index of width ``block_size``.

    Blocks are anchored at ``values.min()`` so the first occupied block is 0.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    origin = np.nanmin(values)
    return np.floor((values - origin) / block_size).astype(np.int64)


# --------------------------------------------------------------------------- #
# Rung 0 — random k-fold (baseline; exposes leakage)
# --------------------------------------------------------------------------- #
def random_kfold(
    X: Any,
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: int = 42,
) -> Iterator[Fold]:
    """Plain random k-fold split — the **leaky baseline**.

    This ignores all spatial and temporal structure and is included *only* to
    quantify the optimistic bias of random CV against the structured rungs. Do
    not report its scores as the model's skill.

    Args:
        X: Feature matrix (array or DataFrame); only the row count is used.
        n_splits: Number of folds.
        shuffle: Whether to shuffle row indices before partitioning.
        random_state: Seed for the shuffle.

    Yields:
        ``(train_idx, test_idx)`` index arrays for each of ``n_splits`` folds.
    """
    n = _n_rows(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    for test_idx in np.array_split(idx, n_splits):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


# --------------------------------------------------------------------------- #
# Rung 1 — temporal splits
# --------------------------------------------------------------------------- #
def leave_time_out(
    X: Any,
    times: Any,
    n_splits: int = 5,
    *,
    block_days: float | None = None,
) -> Iterator[Fold]:
    """Hold out whole time blocks (days) — temporal leave-out CV.

    Unique calendar days are sorted and partitioned into contiguous blocks;
    each fold tests on one block and trains on all others. Because a held-out
    day's rows never appear in training, this removes the same-day temporal
    leak that random CV suffers.

    Args:
        X: Feature matrix; only the row count is used.
        times: Per-row timestamps (anything ``pandas`` can parse to datetime).
        n_splits: Number of temporal folds (ignored if ``block_days`` is set).
        block_days: If given, the width of each temporal block in days; the
            number of folds is then determined by the data span.

    Yields:
        ``(train_idx, test_idx)`` for each temporal block.
    """
    n = _n_rows(X)
    t = pd.to_datetime(pd.Series(np.asarray(times))).to_numpy()
    if t.shape[0] != n:
        raise ValueError("`times` length must match number of rows in X")

    # Map each row to an integer day ordinal.
    day = (t.astype("datetime64[D]") - t.astype("datetime64[D]").min()).astype(
        "timedelta64[D]"
    ).astype(np.int64)

    if block_days is not None:
        block = (day // int(round(block_days))).astype(np.int64)
    else:
        # Partition the unique days into n_splits contiguous groups.
        uniq = np.unique(day)
        edges = np.array_split(uniq, n_splits)
        day_to_block = {}
        for b, group in enumerate(edges):
            for d in group:
                day_to_block[int(d)] = b
        block = np.array([day_to_block[int(d)] for d in day], dtype=np.int64)

    for b in np.unique(block):
        test_mask = block == b
        if not test_mask.any() or test_mask.all():
            continue
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def forward_chaining(
    X: Any,
    times: Any,
    n_splits: int = 5,
    *,
    min_train_days: int = 1,
) -> Iterator[Fold]:
    """Expanding-window temporal CV — train on the past, test on the future.

    Unique days are sorted ascending and divided into ``n_splits + 1``
    contiguous segments. Fold *k* trains on segments ``0..k`` and tests on
    segment ``k+1``, so the training window only ever grows and never sees
    future days. This is the most realistic split for an operational daily
    forecasting product and is strictly harder than :func:`leave_time_out`.

    Args:
        X: Feature matrix; only the row count is used.
        times: Per-row timestamps.
        n_splits: Number of forward folds produced.
        min_train_days: Minimum number of distinct training days required
            before a fold is emitted (small-data guard).

    Yields:
        ``(train_idx, test_idx)`` with all training days strictly before every
        test day in the fold.
    """
    n = _n_rows(X)
    t = pd.to_datetime(pd.Series(np.asarray(times))).to_numpy()
    if t.shape[0] != n:
        raise ValueError("`times` length must match number of rows in X")
    # Work in integer day ordinals so membership tests are unambiguous
    # (datetime64 -> .tolist() yields datetime.date, which breaks set membership
    # against datetime64 row values).
    day = (t.astype("datetime64[D]") - t.astype("datetime64[D]").min()).astype(
        "timedelta64[D]"
    ).astype(np.int64)
    uniq = np.unique(day)
    if uniq.size < n_splits + 1:
        n_splits = max(1, uniq.size - 1)

    segments = np.array_split(uniq, n_splits + 1)
    train_set: set[int] = {int(d) for d in segments[0]}
    for k in range(1, len(segments)):
        test_set = {int(d) for d in segments[k]}
        if len(train_set) >= min_train_days:
            train_mask = np.isin(day, list(train_set))
            test_mask = np.isin(day, list(test_set))
            if train_mask.any() and test_mask.any():
                yield np.where(train_mask)[0], np.where(test_mask)[0]
        train_set |= test_set


# --------------------------------------------------------------------------- #
# Rung 2 — station / group splits
# --------------------------------------------------------------------------- #
def leave_station_out(
    X: Any,
    groups: Any,
    n_splits: int | None = None,
    *,
    random_state: int = 42,
) -> Iterator[Fold]:
    """Group k-fold by ``station_id`` — a station is never in both sets.

    This is the **core fairness guarantee** for a sparse-station product: the
    model is scored only on physical sensors it has never seen, so it cannot
    cheat by memorising a nearby training station. With ``n_splits=None`` this
    degenerates to leave-one-station-out (one fold per station).

    Args:
        X: Feature matrix; only the row count is used.
        groups: Per-row station identifiers (e.g. the ``station_id`` column).
        n_splits: Number of folds; ``None`` -> leave-one-station-out.
        random_state: Seed used to shuffle stations across folds.

    Yields:
        ``(train_idx, test_idx)`` where the test stations are disjoint from the
        train stations.
    """
    n = _n_rows(X)
    g = _check_groups(groups, n, "leave_station_out")
    stations = np.unique(g)
    n_stations = stations.size

    if n_splits is None or n_splits >= n_stations:
        folds = [np.array([s]) for s in stations]  # leave-one-station-out
    else:
        rng = np.random.default_rng(random_state)
        shuffled = stations.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, n_splits)

    for held in folds:
        if held.size == 0:
            continue
        test_mask = np.isin(g, held)
        if not test_mask.any() or test_mask.all():
            continue
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


# --------------------------------------------------------------------------- #
# Rung 3 — spatial block CV
# --------------------------------------------------------------------------- #
def spatial_block_cv(
    X: Any,
    lats: Any,
    lons: Any,
    *,
    block_deg: float = 2.0,
    buffer_deg: float = 0.5,
    checkerboard: int = 4,
    random_state: int = 42,
) -> Iterator[Fold]:
    """Spatial-block CV with a buffer gap (variogram-sized blocks).

    Stations are tiled into square blocks of side ``block_deg`` (chosen to be
    at least the empirical variogram range, so within-block autocorrelation is
    captured *inside* a block rather than straddling the split). Blocks are
    assigned to ``checkerboard`` folds. Crucially, any training station within
    ``buffer_deg`` of a held-out block is **dropped** from that fold, leaving a
    dead zone so spatial autocorrelation cannot leak across the boundary.

    Args:
        X: Feature matrix; only the row count is used.
        lats: Per-row latitudes (degrees).
        lons: Per-row longitudes (degrees).
        block_deg: Side length of each spatial block, in degrees (~ variogram
            range). Larger blocks = stricter spatial separation.
        buffer_deg: Width of the buffer dead-zone around each test block, in
            degrees; training stations inside it are excluded from the fold.
        checkerboard: Number of folds the blocks are assigned to (round-robin
            over a shuffled block list).
        random_state: Seed for the block-to-fold assignment.

    Yields:
        ``(train_idx, test_idx)`` with a buffered spatial gap between them.
    """
    n = _n_rows(X)
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    if lat.shape[0] != n or lon.shape[0] != n:
        raise ValueError("`lats`/`lons` length must match number of rows in X")

    bi = _digitize_blocks(lat, block_deg)
    bj = _digitize_blocks(lon, block_deg)
    block_id = np.array([f"{i}_{j}" for i, j in zip(bi, bj)])

    uniq_blocks = np.unique(block_id)
    rng = np.random.default_rng(random_state)
    order = uniq_blocks.copy()
    rng.shuffle(order)
    fold_of_block = {b: (k % checkerboard) for k, b in enumerate(order)}
    block_fold = np.array([fold_of_block[b] for b in block_id])

    for f in range(checkerboard):
        test_mask = block_fold == f
        if not test_mask.any() or test_mask.all():
            continue
        # Candidate training rows: everything not in the test fold.
        cand = ~test_mask
        # Drop training rows within buffer_deg of ANY test station (dead zone).
        test_lat = lat[test_mask]
        test_lon = lon[test_mask]
        keep_train = cand.copy()
        train_rows = np.where(cand)[0]
        for r in train_rows:
            # Chebyshev distance in degrees to nearest test station.
            d = np.maximum(
                np.abs(lat[r] - test_lat), np.abs(lon[r] - test_lon)
            ).min()
            if d < buffer_deg:
                keep_train[r] = False
        if not keep_train.any():
            continue
        yield np.where(keep_train)[0], np.where(test_mask)[0]


# --------------------------------------------------------------------------- #
# Rung 4 — spatiotemporal blocked CV (GOLD)
# --------------------------------------------------------------------------- #
def spatiotemporal_blocked_cv(
    X: Any,
    lats: Any,
    lons: Any,
    times: Any,
    *,
    block_deg: float = 4.0,
    block_days: float = 14.0,
    n_space_folds: int = 3,
    n_time_folds: int = 3,
    buffer_days: float = 0.0,
    random_state: int = 42,
) -> Iterator[Fold]:
    """Block in space AND time simultaneously — the GOLD-standard splitter.

    Each fold holds out a *space-block x time-block* tile. A training row is
    eligible only if it is neither in a held-out spatial block **nor** in a
    held-out (buffered) temporal block, so the test tile shares neither a
    nearby station nor an adjacent day with anything used to fit the model.
    This is the closest analogue to deploying the model on an unseen region in
    an unseen week and is the number to lead with.

    Args:
        X: Feature matrix; only the row count is used.
        lats: Per-row latitudes (degrees).
        lons: Per-row longitudes (degrees).
        times: Per-row timestamps.
        block_deg: Side of each spatial block in degrees.
        block_days: Width of each temporal block in days.
        n_space_folds: Number of spatial fold groups blocks are assigned to.
        n_time_folds: Number of temporal fold groups time-blocks are assigned to.
        buffer_days: Temporal dead-zone (days) excluded from training around
            each held-out time block, preventing day-to-day temporal leakage.
        random_state: Seed for the fold assignments.

    Yields:
        ``(train_idx, test_idx)`` for each occupied (space-fold, time-fold)
        combination. Test tiles never share a spatial block or an (un-buffered)
        day with their training set.
    """
    n = _n_rows(X)
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    t = pd.to_datetime(pd.Series(np.asarray(times))).to_numpy()
    if not (lat.shape[0] == lon.shape[0] == t.shape[0] == n):
        raise ValueError("`lats`/`lons`/`times` lengths must match rows in X")

    # Spatial blocks -> spatial folds.
    bi = _digitize_blocks(lat, block_deg)
    bj = _digitize_blocks(lon, block_deg)
    sblock = np.array([f"{i}_{j}" for i, j in zip(bi, bj)])
    rng = np.random.default_rng(random_state)
    uniq_s = np.unique(sblock)
    s_order = uniq_s.copy()
    rng.shuffle(s_order)
    s_fold_of = {b: (k % n_space_folds) for k, b in enumerate(s_order)}
    space_fold = np.array([s_fold_of[b] for b in sblock])

    # Temporal blocks (as integer day ordinals) -> temporal folds.
    day = (t.astype("datetime64[D]") - t.astype("datetime64[D]").min()).astype(
        "timedelta64[D]"
    ).astype(np.int64)
    tblock = (day // int(round(block_days))).astype(np.int64)
    uniq_t = np.unique(tblock)
    t_order = uniq_t.copy()
    rng.shuffle(t_order)
    t_fold_of = {b: (k % n_time_folds) for k, b in enumerate(t_order)}
    time_fold = np.array([t_fold_of[b] for b in tblock])

    for sf in range(n_space_folds):
        for tf in range(n_time_folds):
            test_mask = (space_fold == sf) & (time_fold == tf)
            if not test_mask.any():
                continue
            # Spatial separation: training rows must be in a different s-block.
            held_sblocks = set(np.unique(sblock[test_mask]).tolist())
            spatial_ok = np.array([b not in held_sblocks for b in sblock])
            # Temporal separation: drop training rows within buffer_days of any
            # held-out day, and never share a held-out day.
            held_days = np.unique(day[test_mask])
            if buffer_days > 0:
                lo = held_days.min() - buffer_days
                hi = held_days.max() + buffer_days
                temporal_ok = (day < lo) | (day > hi)
            else:
                temporal_ok = ~np.isin(day, held_days)
            train_mask = spatial_ok & temporal_ok & ~test_mask
            if not train_mask.any():
                continue
            yield np.where(train_mask)[0], np.where(test_mask)[0]


# --------------------------------------------------------------------------- #
# Cross-validation driver
# --------------------------------------------------------------------------- #
#: Registry of named schemes -> the keyword arguments each expects beyond X.
_SCHEME_KWARGS: dict[str, tuple[str, ...]] = {
    "random_kfold": (),
    "leave_time_out": ("times",),
    "forward_chaining": ("times",),
    "leave_station_out": ("groups",),
    "spatial_block_cv": ("lats", "lons"),
    "spatiotemporal_blocked_cv": ("lats", "lons", "times"),
}


def _build_folds(
    scheme: str,
    X: Any,
    *,
    times: Any = None,
    groups: Any = None,
    lats: Any = None,
    lons: Any = None,
    scheme_kwargs: dict[str, Any] | None = None,
) -> list[Fold]:
    """Materialise the folds for a named scheme into a list."""
    scheme_kwargs = dict(scheme_kwargs or {})
    if scheme == "random_kfold":
        gen = random_kfold(X, **scheme_kwargs)
    elif scheme == "leave_time_out":
        gen = leave_time_out(X, times, **scheme_kwargs)
    elif scheme == "forward_chaining":
        gen = forward_chaining(X, times, **scheme_kwargs)
    elif scheme == "leave_station_out":
        gen = leave_station_out(X, groups, **scheme_kwargs)
    elif scheme == "spatial_block_cv":
        gen = spatial_block_cv(X, lats, lons, **scheme_kwargs)
    elif scheme == "spatiotemporal_blocked_cv":
        gen = spatiotemporal_blocked_cv(X, lats, lons, times, **scheme_kwargs)
    else:
        raise ValueError(
            f"Unknown CV scheme {scheme!r}; choose from {tuple(_SCHEME_KWARGS)}"
        )
    return list(gen)


def cross_validate(
    model_factory: ModelFactory,
    X: Any,
    y: Any,
    *,
    scheme: str = "leave_station_out",
    groups: Any = None,
    times: Any = None,
    lats: Any = None,
    lons: Any = None,
    scheme_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one CV scheme and return per-fold and aggregate metrics.

    A fresh estimator is built from ``model_factory`` for every fold (so no
    state leaks between folds), fitted on the train indices and scored on the
    test indices using :func:`aqi_india.validation.metrics.metrics_table`. Out
    -of-fold (OOF) predictions are concatenated and scored as a whole, which is
    the most honest single number because every row is predicted exactly once
    by a model that never saw it.

    Args:
        model_factory: Zero-argument callable returning a fresh estimator with
            scikit-learn-style ``fit(X, y)`` / ``predict(X)``.
        X: Feature matrix (array or DataFrame).
        y: Target vector.
        scheme: One of the keys of the scheme registry (e.g.
            ``"leave_station_out"``, ``"spatiotemporal_blocked_cv"``).
        groups: Station identifiers (required for ``leave_station_out``).
        times: Per-row timestamps (required for temporal/ST schemes).
        lats: Per-row latitudes (required for spatial/ST schemes).
        lons: Per-row longitudes (required for spatial/ST schemes).
        scheme_kwargs: Extra keyword arguments forwarded to the splitter.

    Returns:
        Dict with keys:
            ``scheme`` — the scheme name,
            ``n_folds`` — number of folds actually run,
            ``per_fold`` — list of per-fold metric dicts (with ``fold`` and
                ``n_train`` added),
            ``oof`` — metrics over the concatenated out-of-fold predictions,
            ``aggregate`` — mean +/- std of each metric across folds.
    """
    Xm = _as_2d(X)
    yv = np.asarray(y, dtype=np.float64).ravel()
    if Xm.shape[0] != yv.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    folds = _build_folds(
        scheme,
        X,
        times=times,
        groups=groups,
        lats=lats,
        lons=lons,
        scheme_kwargs=scheme_kwargs,
    )

    per_fold: list[dict[str, float]] = []
    oof_true: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []

    for k, (tr, te) in enumerate(folds):
        if tr.size == 0 or te.size == 0:
            continue
        model = model_factory()
        model.fit(Xm[tr], yv[tr])
        pred = np.asarray(model.predict(Xm[te]), dtype=np.float64).ravel()
        row = metrics_table(yv[te], pred)
        row["fold"] = float(k)
        row["n_train"] = float(tr.size)
        per_fold.append(row)
        oof_true.append(yv[te])
        oof_pred.append(pred)

    aggregate = _aggregate_folds(per_fold)
    oof_metrics: dict[str, float] = {}
    if oof_true:
        oof_metrics = metrics_table(
            np.concatenate(oof_true), np.concatenate(oof_pred)
        )

    return {
        "scheme": scheme,
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "oof": oof_metrics,
        "aggregate": aggregate,
    }


def _aggregate_folds(per_fold: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean and population std of each numeric metric across folds.

    Returns a flat dict with ``<metric>_mean`` and ``<metric>_std`` keys for
    the scoring metrics (the bookkeeping ``fold`` column is skipped).
    """
    if not per_fold:
        return {}
    keys = [k for k in per_fold[0] if k != "fold"]
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array(
            [f[key] for f in per_fold if np.isfinite(f.get(key, np.nan))],
            dtype=np.float64,
        )
        if vals.size:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
        else:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# The CV ladder
# --------------------------------------------------------------------------- #
#: Ordered rungs of the ladder, from leakiest (top) to strictest (bottom).
LADDER_SCHEMES: tuple[str, ...] = (
    "random_kfold",
    "leave_time_out",
    "forward_chaining",
    "leave_station_out",
    "spatial_block_cv",
    "spatiotemporal_blocked_cv",
)


def run_cv_ladder(
    model_factory: ModelFactory,
    X: Any,
    y: Any,
    *,
    groups: Any = None,
    times: Any = None,
    lats: Any = None,
    lons: Any = None,
    schemes: Sequence[str] = LADDER_SCHEMES,
    scheme_kwargs: dict[str, dict[str, Any]] | None = None,
    metric: str = "r",
) -> pd.DataFrame:
    """Run every applicable rung of the CV ladder and tabulate the leakage gap.

    For each scheme it has the data to run (a temporal scheme is skipped if no
    ``times`` are supplied, etc.) this fits ``model_factory`` across the folds
    and records the out-of-fold and mean +/- std skill. The resulting table —
    ordered leakiest-to-strictest — is the headline validation deliverable: the
    drop in ``metric`` from ``random_kfold`` down to
    ``spatiotemporal_blocked_cv`` *is* the spatial+temporal leakage the random
    split was hiding.

    Args:
        model_factory: Zero-argument estimator factory (see
            :func:`cross_validate`).
        X: Feature matrix.
        y: Target vector.
        groups: Station ids (enables ``leave_station_out``).
        times: Timestamps (enables temporal/ST schemes).
        lats: Latitudes (enables spatial/ST schemes).
        lons: Longitudes (enables spatial/ST schemes).
        schemes: Which rungs to attempt, in display order.
        scheme_kwargs: Optional ``{scheme_name: kwargs}`` overrides per rung.
        metric: The headline metric column used for the ``leakage_vs_random``
            delta (default Pearson ``r``).

    Returns:
        A :class:`pandas.DataFrame` with one row per scheme and columns
        ``scheme, n_folds, oof_<metric>, <metric>_mean, <metric>_std,
        oof_rmse, oof_mae, leakage_vs_random``.
    """
    scheme_kwargs = scheme_kwargs or {}
    available = {
        "times": times is not None,
        "groups": groups is not None,
        "lats": lats is not None,
        "lons": lons is not None,
    }

    rows: list[dict[str, Any]] = []
    for scheme in schemes:
        required = _SCHEME_KWARGS.get(scheme, ())
        if any(not available.get(req, False) for req in required):
            # Not enough metadata to run this rung; skip it transparently.
            continue
        result = cross_validate(
            model_factory,
            X,
            y,
            scheme=scheme,
            groups=groups,
            times=times,
            lats=lats,
            lons=lons,
            scheme_kwargs=scheme_kwargs.get(scheme),
        )
        agg = result["aggregate"]
        oof = result["oof"]
        rows.append(
            {
                "scheme": scheme,
                "n_folds": result["n_folds"],
                f"oof_{metric}": oof.get(metric, float("nan")),
                f"{metric}_mean": agg.get(f"{metric}_mean", float("nan")),
                f"{metric}_std": agg.get(f"{metric}_std", float("nan")),
                "oof_rmse": oof.get("rmse", float("nan")),
                "oof_mae": oof.get("mae", float("nan")),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty and "random_kfold" in table["scheme"].values:
        ref = table.loc[
            table["scheme"] == "random_kfold", f"oof_{metric}"
        ].iloc[0]
        table["leakage_vs_random"] = ref - table[f"oof_{metric}"]
    elif not table.empty:
        table["leakage_vs_random"] = float("nan")
    return table


def run(cfg: Any) -> dict[str, Any]:
    """Hydra entry point: run the CV ladder from the composed config.

    This is the chaining-contract hook (``aqi_india.validation.cv.run``) the
    Demo agent calls. It loads the feature matrix and labels produced upstream,
    builds a light default estimator (a LightGBM regressor if available, else a
    scikit-learn gradient-boosting tree), runs the full ladder and returns the
    comparison table plus a per-scheme breakdown.

    Heavy/learner imports are deferred to call time so the module stays
    importable under the light dependency set.

    Args:
        cfg: The composed Hydra config (uses ``cfg.paths`` and ``cfg.project``).

    Returns:
        Dict with ``ladder`` (the comparison table as records) and ``schemes``
        (the full :func:`cross_validate` result per rung).
    """
    from pathlib import Path

    from ..utils.io import load_parquet
    from ..utils.logging import get_logger

    log = get_logger("validation.cv")
    seed = int(getattr(getattr(cfg, "project", object()), "seed", 42))

    processed = Path(getattr(cfg.paths, "data_processed", "data/processed"))
    stations = load_parquet(processed / "stations.parquet")

    # Build a simple, dependency-light regression problem from the station
    # table: predict pm25 from the other available pollutant columns. This is a
    # placeholder learner used to demonstrate the ladder; real runs pass the
    # engineered feature matrix and the trained model factory explicitly.
    feature_cols = [
        c
        for c in ("pm10", "no2", "so2", "co", "o3")
        if c in stations.columns
    ]
    target_col = "pm25"
    df = stations.dropna(subset=[target_col, *feature_cols]).reset_index(drop=True)
    X = df[feature_cols]
    y = df[target_col].to_numpy()

    def model_factory() -> Any:
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=200, num_leaves=31, random_state=seed, verbosity=-1
            )
        except Exception:  # pragma: no cover - lightgbm absent fallback
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(random_state=seed)

    ladder = run_cv_ladder(
        model_factory,
        X,
        y,
        groups=df.get("station_id"),
        times=df.get("time"),
        lats=df.get("lat"),
        lons=df.get("lon"),
    )
    log.info("CV ladder complete: %d rungs", len(ladder))
    return {"ladder": ladder.to_dict(orient="records")}


__all__ = [
    "Fold",
    "ModelFactory",
    "LADDER_SCHEMES",
    "random_kfold",
    "leave_time_out",
    "forward_chaining",
    "leave_station_out",
    "spatial_block_cv",
    "spatiotemporal_blocked_cv",
    "cross_validate",
    "run_cv_ladder",
    "run",
]
