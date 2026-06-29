"""aqi_india — Surface AQI mapping and HCHO hotspot detection over India.

BAH 2026 (ISRO) Problem Statement 3. A modular, config-driven pipeline that
fuses Sentinel-5P TROPOMI columns, INSAT-3D AOD, and reanalysis meteorology to
produce daily surface AQI maps over India (Objective 1), and detects biomass-
burning HCHO hotspots with fire correlation and transport attribution
(Objective 2).

The package is importable without any heavy or credentialed dependency: torch,
earthengine-api, cdsapi and similar are lazy-imported inside the functions that
need them. The light dependency set (numpy/pandas/scipy/sklearn/lightgbm/h3/
xarray/geopandas/...) is sufficient to run the demo and the core unit tests.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
