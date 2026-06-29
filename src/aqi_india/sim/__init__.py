"""Synthetic-data generators for the offline BAH 2026 PS3 demo.

The :mod:`aqi_india.sim.synthetic` module fabricates a deterministic,
physically-plausible **synthetic** India dataset (gridded satellite/met cube,
CPCB-like stations, FIRMS-like fires with a downwind HCHO signal) that stands in
for the real sources the ``aqi_india.ingest`` adapters target, so the full
pipeline is runnable and verifiable offline on the light dependency set.
"""

from __future__ import annotations

from .synthetic import (
    COLUMN_VARS,
    DEFAULT_SEED,
    GRID_VARS,
    SURFACE_POLLUTANTS,
    generate_all,
    inject_fire_hcho,
    make_fires,
    make_grid,
    make_stations,
)

__all__ = [
    "COLUMN_VARS",
    "DEFAULT_SEED",
    "GRID_VARS",
    "SURFACE_POLLUTANTS",
    "generate_all",
    "inject_fire_hcho",
    "make_fires",
    "make_grid",
    "make_stations",
]
