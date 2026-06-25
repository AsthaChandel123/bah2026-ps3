"""Trajectory clustering by angle-distance (Objective-2 transport pathways).

Following openair's ``trajCluster`` approach, back-trajectories are grouped into
a small number of representative **transport pathways** so we can name the
dominant route (expected: a north-west Punjab/Haryana stubble-burning corridor
into the Indo-Gangetic receptors).  The distance between two trajectories uses
the openair **angle distance**, which compares the bearing of each trajectory
point from the receptor — it groups trajectories by *direction of origin* rather
than raw great-circle separation, exactly what source attribution needs.

The clustering itself is a light, dependency-free agglomerative / k-medoids pass
over the pairwise angle-distance matrix (no scikit-learn hard requirement at the
core, though :func:`cluster_trajectories` will use it if present).  Each cluster
yields a mean pathway and a count, and :func:`dominant_pathway` reports the
heaviest cluster plus its mean bearing so the NW corridor can be flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils.logging import get_logger
from .hysplit import Trajectory

logger = get_logger("transport.traj_cluster")


@dataclass
class ClusterResult:
    """Trajectory clustering result.

    Attributes:
        labels: Cluster label per input trajectory (length = n trajectories).
        n_clusters: Number of clusters.
        mean_paths: Per-cluster mean path as a dict ``label -> (lats, lons)``.
        sizes: Per-cluster membership counts ``label -> int``.
        mean_bearings: Per-cluster mean origin bearing (deg from receptor).
    """

    labels: np.ndarray
    n_clusters: int
    mean_paths: dict[int, tuple[np.ndarray, np.ndarray]]
    sizes: dict[int, int]
    mean_bearings: dict[int, float]


def _resample_path(
    lats: np.ndarray, lons: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a path to ``n`` equally indexed points (linear interpolation).

    Args:
        lats: Latitudes along a trajectory.
        lons: Longitudes along a trajectory.
        n: Target number of points.

    Returns:
        ``(lats_n, lons_n)`` resampled to length ``n``.
    """
    if lats.size == n:
        return lats, lons
    src = np.linspace(0.0, 1.0, lats.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, lats), np.interp(dst, src, lons)


def angle_distance(
    traj_a: tuple[np.ndarray, np.ndarray],
    traj_b: tuple[np.ndarray, np.ndarray],
    origin: tuple[float, float],
) -> float:
    """openair-style angle distance between two trajectories.

    For each matched point pair the angle subtended at the receptor ``origin`` by
    the two trajectory points is computed via the spherical law of cosines, and
    the distances are averaged.  Trajectories arriving from the same direction
    score near 0; opposite directions score near pi.

    Args:
        traj_a: ``(lats, lons)`` of the first trajectory.
        traj_b: ``(lats, lons)`` of the second trajectory (same length).
        origin: ``(lat, lon)`` of the receptor.

    Returns:
        The mean subtended angle in radians (0 = identical direction).
    """
    lat0, lon0 = np.radians(origin[0]), np.radians(origin[1])
    la, lo = np.radians(traj_a[0]), np.radians(traj_a[1])
    lb, lo_b = np.radians(traj_b[0]), np.radians(traj_b[1])

    # Great-circle distances from origin to each point (unit sphere, radians).
    def _gc(lat, lon):
        return np.arccos(
            np.clip(
                np.sin(lat0) * np.sin(lat)
                + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0),
                -1.0,
                1.0,
            )
        )

    # Great-circle distance between the two trajectory points.
    def _gc_pair(latx, lonx, laty, lony):
        return np.arccos(
            np.clip(
                np.sin(latx) * np.sin(laty)
                + np.cos(latx) * np.cos(laty) * np.cos(lonx - lony),
                -1.0,
                1.0,
            )
        )

    a = _gc(la, lo)
    b = _gc(lb, lo_b)
    c = _gc_pair(la, lo, lb, lo_b)
    # Subtended angle at origin via spherical law of cosines, guarded.
    denom = 2.0 * np.sin(a) * np.sin(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = (np.cos(c) - np.cos(a) * np.cos(b)) / denom
    cos_angle = np.where(np.isfinite(cos_angle), cos_angle, 1.0)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.nanmean(angle))


def _pairwise_angle_matrix(
    paths: list[tuple[np.ndarray, np.ndarray]], origin: tuple[float, float]
) -> np.ndarray:
    """Build the symmetric pairwise angle-distance matrix.

    Args:
        paths: List of resampled ``(lats, lons)`` paths.
        origin: ``(lat, lon)`` receptor.

    Returns:
        ``(n, n)`` distance matrix.
    """
    n = len(paths)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = angle_distance(paths[i], paths[j], origin)
            d[i, j] = d[j, i] = dist
    return d


def _bearing(origin: tuple[float, float], lat: float, lon: float) -> float:
    """Initial great-circle bearing from ``origin`` to a point (deg, 0-360).

    Args:
        origin: ``(lat, lon)`` receptor.
        lat: Target latitude.
        lon: Target longitude.

    Returns:
        Bearing in degrees clockwise from north.
    """
    lat1, lon1 = np.radians(origin[0]), np.radians(origin[1])
    lat2, lon2 = np.radians(lat), np.radians(lon)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def cluster_trajectories(
    trajectories: Sequence[Trajectory],
    n_clusters: int = 3,
    origin: "tuple[float, float] | None" = None,
    *,
    n_points: int = 24,
    random_state: int = 42,
) -> ClusterResult:
    """Cluster back-trajectories into transport pathways by angle distance.

    Args:
        trajectories: Back-trajectories from :func:`~aqi_india.transport.hysplit`.
        n_clusters: Number of pathway clusters to find.
        origin: Receptor ``(lat, lon)``. Defaults to the mean release point.
        n_points: Number of points each path is resampled to before clustering.
        random_state: Seed for the medoid initialisation.

    Returns:
        A :class:`ClusterResult`.

    Raises:
        ValueError: If no trajectories are supplied.
    """
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("No trajectories to cluster.")

    if origin is None:
        origin = (
            float(np.mean([t.release_lat for t in trajectories])),
            float(np.mean([t.release_lon for t in trajectories])),
        )

    paths = [_resample_path(t.lats, t.lons, n_points) for t in trajectories]
    n = len(paths)
    k = int(min(n_clusters, n))
    dmat = _pairwise_angle_matrix(paths, origin)

    labels = _kmedoids(dmat, k, random_state)

    mean_paths: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sizes: dict[int, int] = {}
    mean_bearings: dict[int, float] = {}
    for lab in range(k):
        members = np.where(labels == lab)[0]
        if members.size == 0:
            continue
        lat_stack = np.vstack([paths[i][0] for i in members])
        lon_stack = np.vstack([paths[i][1] for i in members])
        mlat = lat_stack.mean(axis=0)
        mlon = lon_stack.mean(axis=0)
        mean_paths[lab] = (mlat, mlon)
        sizes[lab] = int(members.size)
        # Bearing to the far (oldest) end of the mean path = origin direction.
        mean_bearings[lab] = _bearing(origin, mlat[-1], mlon[-1])

    logger.info(
        "clustered %d trajectories into %d pathway(s); sizes=%s",
        n,
        len(sizes),
        sizes,
    )
    return ClusterResult(
        labels=labels,
        n_clusters=len(sizes),
        mean_paths=mean_paths,
        sizes=sizes,
        mean_bearings=mean_bearings,
    )


def _kmedoids(dmat: np.ndarray, k: int, random_state: int) -> np.ndarray:
    """Partitioning-around-medoids clustering on a precomputed distance matrix.

    A compact, dependency-free PAM implementation (Voronoi-iteration variant):
    assign each point to its nearest medoid, then move each medoid to the member
    that minimises intra-cluster distance, until labels stabilise.

    Args:
        dmat: ``(n, n)`` symmetric distance matrix.
        k: Number of clusters/medoids.
        random_state: Seed for medoid initialisation.

    Returns:
        Integer label array of length ``n``.
    """
    n = dmat.shape[0]
    rng = np.random.default_rng(random_state)
    medoids = rng.choice(n, size=k, replace=False)
    labels = np.zeros(n, dtype=int)

    for _ in range(100):
        new_labels = np.argmin(dmat[:, medoids], axis=1)
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.where(new_labels == c)[0]
            if members.size == 0:
                continue
            costs = dmat[np.ix_(members, members)].sum(axis=1)
            new_medoids[c] = members[int(np.argmin(costs))]
        if np.array_equal(new_labels, labels) and np.array_equal(
            new_medoids, medoids
        ):
            labels = new_labels
            break
        labels, medoids = new_labels, new_medoids
    return labels


def dominant_pathway(result: ClusterResult) -> dict:
    """Identify the heaviest transport pathway and its origin direction.

    Args:
        result: Output of :func:`cluster_trajectories`.

    Returns:
        Dict with ``label``, ``size``, ``fraction`` (of all trajectories),
        ``mean_bearing`` (deg) and ``compass`` (e.g. ``"NW"``).
    """
    if not result.sizes:
        return {"label": None, "size": 0, "fraction": 0.0}
    total = sum(result.sizes.values())
    label = max(result.sizes, key=result.sizes.get)
    bearing = result.mean_bearings.get(label, float("nan"))
    return {
        "label": int(label),
        "size": int(result.sizes[label]),
        "fraction": float(result.sizes[label] / total),
        "mean_bearing": float(bearing),
        "compass": _compass(bearing),
    }


def _compass(bearing: float) -> str:
    """Convert a bearing in degrees to an 8-point compass label.

    Args:
        bearing: Bearing in degrees (0-360, clockwise from north).

    Returns:
        One of N, NE, E, SE, S, SW, W, NW (or ``"?"`` if not finite).
    """
    if not np.isfinite(bearing):
        return "?"
    points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return points[int((bearing % 360) / 45.0 + 0.5) % 8]


__all__ = [
    "ClusterResult",
    "angle_distance",
    "cluster_trajectories",
    "dominant_pathway",
]
