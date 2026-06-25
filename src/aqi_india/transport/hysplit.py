"""Back-trajectory transport modelling (Objective-2).

Source attribution of receptor HCHO needs trajectories that trace air parcels
**backward** from the receptor to their origin.  The reference implementation is
HYSPLIT (driven by ERA5/IMDAA met), but HYSPLIT needs ARL meteorology and a
compiled executable that is not available in the light demo environment.  This
module therefore provides two paths behind one API:

* :func:`back_trajectories` — tries a real PySPLIT/HYSPLIT run when the
  executable and ARL data are configured, and otherwise transparently falls back
  to :func:`kinematic_back_trajectories`.
* :func:`kinematic_back_trajectories` — a **real kinematic Lagrangian integrator**
  that steps air parcels backward in time through an ERA5/synthetic ``wind_u`` /
  ``wind_v`` field (DEV_CONTRACT §6.1), converting m/s winds to deg/step on a
  spherical Earth.  It actually runs on the synthetic winds and returns lat/lon
  trajectory paths.

The kinematic core is deliberately simple (single-level horizontal advection with
bilinear space-time interpolation and a midpoint integration step) but it is a
genuine algorithm, not a stub: clustering, CWT and PSCF downstream consume its
output directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = get_logger("transport.hysplit")

#: Mean Earth radius (m) for deg<->metre conversion.
EARTH_RADIUS_M: float = 6_371_000.0

#: Default release heights AGL (m) — boundary layer and just above.
DEFAULT_LEVELS: tuple[int, ...] = (500, 1000)


@dataclass
class Trajectory:
    """A single back-trajectory path.

    Attributes:
        traj_id: Stable identifier ``"<release>_<level>_<start>"``.
        times: Timestamps along the path (descending: from start backward).
        lats: Latitude (deg) at each step.
        lons: Longitude (deg) at each step.
        level: Release height AGL (m) the parcel was launched at.
        release_lat: Receptor latitude (deg).
        release_lon: Receptor longitude (deg).
    """

    traj_id: str
    times: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    level: float
    release_lat: float
    release_lon: float

    def to_frame(self) -> pd.DataFrame:
        """Return the trajectory as a tidy long DataFrame.

        Returns:
            Columns ``traj_id``, ``step``, ``time``, ``lat``, ``lon``, ``level``.
        """
        return pd.DataFrame(
            {
                "traj_id": self.traj_id,
                "step": np.arange(self.lats.size),
                "time": self.times,
                "lat": self.lats,
                "lon": self.lons,
                "level": self.level,
            }
        )


def _bilinear_sample(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    """Bilinearly sample a 2-D ``(lat, lon)`` field at one point.

    Coordinates are assumed ascending and regularly spaced. Out-of-domain points
    are clamped to the nearest edge so a parcel that drifts off-grid still gets a
    sensible wind (it simply stops accelerating off-domain).

    Args:
        field: 2-D array indexed ``[lat, lon]``.
        lats: Ascending 1-D latitude coordinate.
        lons: Ascending 1-D longitude coordinate.
        lat: Query latitude.
        lon: Query longitude.

    Returns:
        The interpolated scalar (NaN-holes fall back to nearest finite value).
    """
    lat = float(np.clip(lat, lats[0], lats[-1]))
    lon = float(np.clip(lon, lons[0], lons[-1]))

    i = int(np.searchsorted(lats, lat) - 1)
    j = int(np.searchsorted(lons, lon) - 1)
    i = int(np.clip(i, 0, lats.size - 2))
    j = int(np.clip(j, 0, lons.size - 2))

    lat0, lat1 = lats[i], lats[i + 1]
    lon0, lon1 = lons[j], lons[j + 1]
    ty = 0.0 if lat1 == lat0 else (lat - lat0) / (lat1 - lat0)
    tx = 0.0 if lon1 == lon0 else (lon - lon0) / (lon1 - lon0)

    f00 = field[i, j]
    f01 = field[i, j + 1]
    f10 = field[i + 1, j]
    f11 = field[i + 1, j + 1]
    corners = np.array([f00, f01, f10, f11], dtype=float)
    if np.any(~np.isfinite(corners)):
        finite = corners[np.isfinite(corners)]
        return float(finite.mean()) if finite.size else 0.0

    top = f00 * (1 - tx) + f01 * tx
    bot = f10 * (1 - tx) + f11 * tx
    return float(top * (1 - ty) + bot * ty)


def _wind_at(
    ds: "xr.Dataset",
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    t: np.datetime64,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    """Sample (u, v) wind at a space-time point with linear time interpolation.

    Args:
        ds: Grid cube exposing ``wind_u``/``wind_v`` (time, lat, lon).
        lats: Ascending latitude coordinate.
        lons: Ascending longitude coordinate.
        times: ``datetime64`` time coordinate (ascending).
        t: Query time.
        lat: Query latitude.
        lon: Query longitude.

    Returns:
        ``(u, v)`` wind components in m/s.
    """
    t = np.datetime64(t)
    t_clamped = min(max(t, times[0]), times[-1])
    k = int(np.searchsorted(times, t_clamped) - 1)
    k = int(np.clip(k, 0, times.size - 2))
    t0, t1 = times[k], times[k + 1]
    span = (t1 - t0) / np.timedelta64(1, "s")
    w = 0.0 if span == 0 else (t_clamped - t0) / np.timedelta64(1, "s") / span

    u0 = _bilinear_sample(ds["wind_u"].isel(time=k).values, lats, lons, lat, lon)
    u1 = _bilinear_sample(ds["wind_u"].isel(time=k + 1).values, lats, lons, lat, lon)
    v0 = _bilinear_sample(ds["wind_v"].isel(time=k).values, lats, lons, lat, lon)
    v1 = _bilinear_sample(ds["wind_v"].isel(time=k + 1).values, lats, lons, lat, lon)
    return (u0 * (1 - w) + u1 * w, v0 * (1 - w) + v1 * w)


def _step_latlon(
    lat: float, lon: float, u: float, v: float, dt_s: float
) -> tuple[float, float]:
    """Advance a parcel by a wind vector over ``dt_s`` seconds (deg output).

    Converts m/s to degrees using the local meridian/parallel length on a
    spherical Earth (the parallel shrinks with cos(latitude)).

    Args:
        lat: Current latitude (deg).
        lon: Current longitude (deg).
        u: Eastward wind (m/s).
        v: Northward wind (m/s).
        dt_s: Time step (s); negative integrates backward in time.

    Returns:
        ``(new_lat, new_lon)`` in degrees.
    """
    dlat = np.degrees(v * dt_s / EARTH_RADIUS_M)
    coslat = max(np.cos(np.radians(lat)), 1e-6)
    dlon = np.degrees(u * dt_s / (EARTH_RADIUS_M * coslat))
    return lat + dlat, lon + dlon


def kinematic_back_trajectories(
    wind_ds: "xr.Dataset",
    release_pts: Sequence[tuple[float, float]],
    start: "str | np.datetime64 | pd.Timestamp",
    hours: int = 120,
    levels: Sequence[int] = DEFAULT_LEVELS,
    *,
    step_minutes: int = 60,
) -> list[Trajectory]:
    """Integrate kinematic back-trajectories through a wind field.

    For each release point and level the parcel is stepped **backward** in time
    using a midpoint (Heun-style) scheme on the horizontal wind, sampling
    ``wind_u``/``wind_v`` with bilinear space + linear time interpolation.

    Args:
        wind_ds: Grid cube with ``wind_u``/``wind_v`` (DEV_CONTRACT §6.1).
        release_pts: Iterable of ``(lat, lon)`` receptor coordinates.
        start: Release time (the trajectory's t=0, integrated backward).
        hours: Back-trajectory length in hours (e.g. 120 = 5 days).
        levels: Release heights AGL (m). Heights only tag the trajectory here
            (single-level horizontal advection); kept for API parity with HYSPLIT.
        step_minutes: Integration step in minutes (default 60).

    Returns:
        A list of :class:`Trajectory` objects (one per release point x level).

    Raises:
        KeyError: If ``wind_ds`` lacks ``wind_u``/``wind_v``.
    """
    if "wind_u" not in wind_ds or "wind_v" not in wind_ds:
        raise KeyError("wind_ds must contain 'wind_u' and 'wind_v' data vars.")

    lats = np.asarray(wind_ds["lat"].values, dtype=float)
    lons = np.asarray(wind_ds["lon"].values, dtype=float)
    times = np.asarray(wind_ds["time"].values, dtype="datetime64[ns]")
    if lats[0] > lats[-1]:  # ensure ascending for the sampler
        lats = lats[::-1]
        wind_ds = wind_ds.isel(lat=slice(None, None, -1))

    start_t = np.datetime64(pd.Timestamp(start), "ns")
    dt_s = -step_minutes * 60.0  # negative => backward in time
    n_steps = int(round(hours * 60 / step_minutes))

    trajectories: list[Trajectory] = []
    for (rlat, rlon) in release_pts:
        for level in levels:
            lat, lon = float(rlat), float(rlon)
            t = start_t
            path_lat = [lat]
            path_lon = [lon]
            path_t = [t]
            for _ in range(n_steps):
                u1, v1 = _wind_at(wind_ds, lats, lons, times, t, lat, lon)
                # Midpoint predictor.
                mlat, mlon = _step_latlon(lat, lon, u1, v1, dt_s / 2)
                mt = t + np.timedelta64(int(dt_s / 2), "s")
                u2, v2 = _wind_at(wind_ds, lats, lons, times, mt, mlat, mlon)
                lat, lon = _step_latlon(lat, lon, u2, v2, dt_s)
                t = t + np.timedelta64(int(dt_s), "s")
                path_lat.append(lat)
                path_lon.append(lon)
                path_t.append(t)

            trajectories.append(
                Trajectory(
                    traj_id=f"{rlat:.2f}_{rlon:.2f}_L{level}_{pd.Timestamp(start_t)}",
                    times=np.array(path_t, dtype="datetime64[ns]"),
                    lats=np.array(path_lat, dtype=float),
                    lons=np.array(path_lon, dtype=float),
                    level=float(level),
                    release_lat=float(rlat),
                    release_lon=float(rlon),
                )
            )

    logger.info(
        "integrated %d kinematic back-trajectories (%dh, levels=%s)",
        len(trajectories),
        hours,
        tuple(levels),
    )
    return trajectories


def _try_pysplit(
    release_pts: Sequence[tuple[float, float]],
    start,
    hours: int,
    levels: Sequence[int],
    hysplit_cfg: Optional[dict],
) -> Optional[list[Trajectory]]:
    """Attempt a real PySPLIT/HYSPLIT run; return ``None`` to fall back.

    This is the clearly-marked external-API adapter. It only runs when a HYSPLIT
    working dir, executable and ARL met directory are supplied via ``hysplit_cfg``
    and the ``pysplit`` package imports. Any failure returns ``None`` so the
    caller uses the kinematic integrator instead.

    Args:
        release_pts: ``(lat, lon)`` receptor coordinates.
        start: Release time.
        hours: Back-trajectory length (hours).
        levels: Release heights AGL (m).
        hysplit_cfg: Dict with ``working_dir``, ``hysplit_exe``, ``meteo_dir``,
            ``meteo_files``; or ``None`` to skip.

    Returns:
        Parsed trajectories, or ``None`` if HYSPLIT is unavailable/failed.
    """
    if not hysplit_cfg:
        return None
    try:  # pragma: no cover - exercised only with a real HYSPLIT install
        import pysplit  # type: ignore

        generator = pysplit.TrajectoryGenerator(
            hysplit_working=hysplit_cfg["working_dir"],
            hysplit=hysplit_cfg["hysplit_exe"],
        )
        ts = pd.Timestamp(start)
        generator.generate_trajectories(
            basename=hysplit_cfg.get("basename", "aqi_back"),
            hysplit_working=hysplit_cfg["working_dir"],
            output_dir=hysplit_cfg["working_dir"],
            years=[ts.year],
            months=[ts.month],
            hours=[ts.hour],
            altitudes=list(levels),
            coordinates=list(release_pts),
            run=-int(hours),
            meteo_bookends=hysplit_cfg.get("meteo_bookends"),
            meteoyr_2digits=False,
            meteo_dir=hysplit_cfg["meteo_dir"],
            get_reverse=False,
            get_clipped=False,
        )
        logger.info("PySPLIT/HYSPLIT trajectories generated; parsing output")
        # Real parsing of HYSPLIT 'tdump' output would happen here.
        return _parse_hysplit_output(hysplit_cfg["working_dir"], levels)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("PySPLIT/HYSPLIT unavailable (%s); kinematic fallback", exc)
        return None


def _parse_hysplit_output(  # pragma: no cover - needs real HYSPLIT output
    working_dir: str, levels: Sequence[int]
) -> list[Trajectory]:
    """Parse HYSPLIT ``tdump`` trajectory files into :class:`Trajectory` objects.

    Args:
        working_dir: Directory containing HYSPLIT ``tdump`` output files.
        levels: Release heights AGL (m) used in the run.

    Returns:
        Parsed trajectories.

    Raises:
        FileNotFoundError: If no ``tdump`` files are found.
    """
    from pathlib import Path

    files = sorted(Path(working_dir).glob("*tdump*"))
    if not files:
        raise FileNotFoundError(f"No HYSPLIT tdump output in {working_dir}")

    trajectories: list[Trajectory] = []
    for fpath in files:
        records: list[tuple] = []
        with open(fpath) as fh:
            lines = fh.readlines()
        # HYSPLIT header: line 1 = n met grids; skip them, then 1 line per
        # trajectory definition, then a '1 PRESSURE' style var line, then data.
        idx = 0
        n_grids = int(lines[idx].split()[0])
        idx += 1 + n_grids
        n_traj = int(lines[idx].split()[0])
        idx += 1 + n_traj
        idx += 1  # variable-count line
        for line in lines[idx:]:
            parts = line.split()
            if len(parts) < 12:
                continue
            year, month, day, hour = (int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]))
            lat, lon = float(parts[9]), float(parts[10])
            year += 2000 if year < 50 else 1900
            records.append((pd.Timestamp(year, month, day, hour), lat, lon))
        if not records:
            continue
        ts, la, lo = zip(*records)
        trajectories.append(
            Trajectory(
                traj_id=fpath.stem,
                times=np.array(ts, dtype="datetime64[ns]"),
                lats=np.array(la, dtype=float),
                lons=np.array(lo, dtype=float),
                level=float(levels[0]),
                release_lat=float(la[0]),
                release_lon=float(lo[0]),
            )
        )
    return trajectories


def back_trajectories(
    release_pts: Sequence[tuple[float, float]],
    start: "str | np.datetime64 | pd.Timestamp",
    hours: int = 120,
    levels: Sequence[int] = DEFAULT_LEVELS,
    *,
    wind_ds: "Optional[xr.Dataset]" = None,
    hysplit_cfg: Optional[dict] = None,
    step_minutes: int = 60,
) -> list[Trajectory]:
    """Compute back-trajectories, preferring HYSPLIT, falling back to kinematic.

    Args:
        release_pts: Iterable of ``(lat, lon)`` receptor coordinates (e.g. Delhi,
            Lucknow, Kanpur, Patna).
        start: Release time (t=0; the integration runs backward from here).
        hours: Back-trajectory length in hours (default 120 = 5 days).
        levels: Release heights AGL (m) (default ``(500, 1000)``).
        wind_ds: Grid cube with ``wind_u``/``wind_v`` for the kinematic fallback.
            Required when HYSPLIT is not configured.
        hysplit_cfg: Optional HYSPLIT configuration dict to attempt a real run.
        step_minutes: Kinematic integration step (minutes).

    Returns:
        A list of :class:`Trajectory` objects.

    Raises:
        ValueError: If neither HYSPLIT succeeds nor ``wind_ds`` is provided.
    """
    real = _try_pysplit(release_pts, start, hours, levels, hysplit_cfg)
    if real is not None:
        return real

    if wind_ds is None:
        raise ValueError(
            "No HYSPLIT config succeeded and no wind_ds given for the kinematic "
            "fallback. Pass wind_ds=<grid cube> to compute trajectories."
        )
    return kinematic_back_trajectories(
        wind_ds,
        release_pts,
        start,
        hours=hours,
        levels=levels,
        step_minutes=step_minutes,
    )


def trajectories_to_frame(trajectories: Sequence[Trajectory]) -> pd.DataFrame:
    """Stack trajectories into one tidy long DataFrame.

    Args:
        trajectories: Trajectories from :func:`back_trajectories`.

    Returns:
        Concatenated long table (see :meth:`Trajectory.to_frame`).
    """
    if not trajectories:
        return pd.DataFrame(
            columns=["traj_id", "step", "time", "lat", "lon", "level"]
        )
    return pd.concat([t.to_frame() for t in trajectories], ignore_index=True)


__all__ = [
    "DEFAULT_LEVELS",
    "EARTH_RADIUS_M",
    "Trajectory",
    "back_trajectories",
    "kinematic_back_trajectories",
    "trajectories_to_frame",
]
