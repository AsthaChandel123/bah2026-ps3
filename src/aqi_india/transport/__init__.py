"""aqi_india.transport — fire-HCHO correlation and transport attribution (Obj-2).

This subpackage closes the Objective-2 chain that links biomass-burning fires to
downwind HCHO over the Indo-Gangetic Plain and forest belts, with a *convergent
multi-line* attribution (the blueprint requirement): the statistical correlation,
the Lagrangian back-trajectory / receptor-apportionment, and the emission
inventory should agree before a source is trusted.

Modules:

* :mod:`~aqi_india.transport.fire_periods` — STL (moving-average fallback)
  extraction of sustained biomass-burning episodes from a FIRMS-surrogate table.
* :mod:`~aqi_india.transport.fire_hcho_corr` — deseasonalised lagged
  cross-correlation, prewhitened Granger causality, dHCHO/FRP emission ratio and
  the HCHO:NO2 (FNR) regime; ``run(cfg)`` is the Hydra entry point.
* :mod:`~aqi_india.transport.hysplit` — back-trajectories, preferring a real
  PySPLIT/HYSPLIT run and falling back to a kinematic Lagrangian integrator on
  synthetic ``wind_u``/``wind_v``.
* :mod:`~aqi_india.transport.traj_cluster` — openair-style angle-distance
  trajectory clustering into transport pathways (the NW corridor).
* :mod:`~aqi_india.transport.cwt_pscf` — CWT / PSCF receptor source maps.
* :mod:`~aqi_india.transport.polar` — bivariate polar / pollution-rose
  directional fingerprints.
* :mod:`~aqi_india.transport.inventory` — FRP-based fire emission inventory with
  a multi-family (FINN/GFAS/QFED) uncertainty bracket.

Everything imports and runs on the light dependency set; HYSPLIT and PySPLIT are
optional and lazy, with documented kinematic fallbacks.
"""

from __future__ import annotations

from .cwt_pscf import GriddedSourceMap, cwt, peak_source_cell, pscf
from .fire_hcho_corr import (
    XCorrResult,
    classify_fnr,
    dhcho_frp_slope,
    fnr_map,
    granger,
    lagged_xcorr,
)
from .fire_hcho_corr import run as fire_hcho_run
from .fire_periods import (
    EpisodeResult,
    FireEpisode,
    aoi_from_name,
    daily_fire_series,
    extract_episodes,
)
from .hysplit import (
    Trajectory,
    back_trajectories,
    kinematic_back_trajectories,
    trajectories_to_frame,
)
from .inventory import (
    EmissionInventory,
    emission_coefficient,
    emission_rates,
    grid_emissions,
    inventory_ensemble,
)
from .inventory import run as inventory_run
from .polar import (
    PolarGrid,
    bivariate_polar,
    dominant_direction,
    plot_polar,
    pollution_rose,
    uv_to_speed_dir,
)
from .traj_cluster import (
    ClusterResult,
    angle_distance,
    cluster_trajectories,
    dominant_pathway,
)

__all__ = [
    # fire_periods
    "EpisodeResult",
    "FireEpisode",
    "aoi_from_name",
    "daily_fire_series",
    "extract_episodes",
    # fire_hcho_corr
    "XCorrResult",
    "classify_fnr",
    "dhcho_frp_slope",
    "fire_hcho_run",
    "fnr_map",
    "granger",
    "lagged_xcorr",
    # hysplit
    "Trajectory",
    "back_trajectories",
    "kinematic_back_trajectories",
    "trajectories_to_frame",
    # traj_cluster
    "ClusterResult",
    "angle_distance",
    "cluster_trajectories",
    "dominant_pathway",
    # cwt_pscf
    "GriddedSourceMap",
    "cwt",
    "peak_source_cell",
    "pscf",
    # polar
    "PolarGrid",
    "bivariate_polar",
    "dominant_direction",
    "plot_polar",
    "pollution_rose",
    "uv_to_speed_dir",
    # inventory
    "EmissionInventory",
    "emission_coefficient",
    "emission_rates",
    "grid_emissions",
    "inventory_ensemble",
    "inventory_run",
]
