"""CPCB National Air Quality Index (NAQI, 2014) breakpoint tables.

This module encodes the official Central Pollution Control Board (CPCB) National
Air Quality Index breakpoints for the eight scheduled pollutants as immutable
NumPy arrays, together with the AQI category edges, category names, colours and
per-pollutant averaging-period / unit metadata.

The numbers here are the single source of truth for the whole AQI engine. They
intentionally mirror the CPCB worked-example tables exactly. Getting these wrong
(e.g. CO in ug/m3 instead of mg/m3, or EPA/WHO breakpoints) is the most common
fatal error in AQI products, so they are unit-tested against CPCB golden vectors.

References:
    CPCB (2014). National Air Quality Index. Central Pollution Control Board,
    Ministry of Environment, Forest and Climate Change, Government of India.

Conventions:
    * ``AQI_CATEGORY_EDGES`` are the AQI-axis edges of the six bands:
      0-50, 51-100, 101-200, 201-300, 301-400, 401-500.
    * ``BREAKPOINTS[pollutant]`` gives the *lower* concentration edge of each of
      the six bands (length 7: the 6 lower edges plus the upper edge of the top
      band, used to bound the top segment). The 7th value is the CPCB upper cap
      for the top band; above it the AQI is clamped to 500 (the open top band is
      extrapolated linearly using the top-band slope up to the cap).
    * Units: CO is in mg/m3; every other pollutant is in ug/m3.
    * Averaging period: CO and O3 use an 8-hour mean; all others a 24-hour mean.
"""

from __future__ import annotations

import numpy as np

#: AQI-axis edges of the six index bands (length 7).
#: Bands map to [0-50, 51-100, 101-200, 201-300, 301-400, 401-500].
AQI_CATEGORY_EDGES: np.ndarray = np.array(
    [0.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64
)

#: Lower AQI value of each of the six bands, used as ``I_Lo`` in the sub-index
#: interpolation: I = ((I_Hi - I_Lo)/(BP_Hi - BP_Lo)) * (C - BP_Lo) + I_Lo.
AQI_BAND_LO: np.ndarray = np.array([0.0, 51.0, 101.0, 201.0, 301.0, 401.0], dtype=np.float64)
#: Upper AQI value of each of the six bands (``I_Hi``).
AQI_BAND_HI: np.ndarray = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)

#: Maximum AQI value (top of the Severe band).
AQI_MAX: float = 500.0

#: Per-pollutant concentration breakpoints (length 7 each).
#:
#: Indices 0..5 are the lower edges of the six AQI bands; index 6 is the CPCB
#: upper cap of the top (Severe) band. Within the top band the sub-index is
#: interpolated linearly between BREAKPOINTS[...][5] (AQI 401) and [...][6]
#: (AQI 500), then clamped to 500 above the cap.
#:
#: PM2.5/PM10/NO2/SO2/O3/NH3 in ug/m3; CO in mg/m3; Pb in ug/m3.
BREAKPOINTS: dict[str, np.ndarray] = {
    # PM2.5, 24h, ug/m3. CPCB top band 250-380 -> AQI 401-500.
    "pm25": np.array([0.0, 30.0, 60.0, 90.0, 120.0, 250.0, 380.0], dtype=np.float64),
    # PM10, 24h, ug/m3. CPCB top band 430-510 -> AQI 401-500.
    "pm10": np.array([0.0, 50.0, 100.0, 250.0, 350.0, 430.0, 510.0], dtype=np.float64),
    # NO2, 24h, ug/m3. Top band 400-... ; CPCB uses 400 with slope of band 5.
    "no2": np.array([0.0, 40.0, 80.0, 180.0, 280.0, 400.0, 520.0], dtype=np.float64),
    # SO2, 24h, ug/m3. Top band 1600-2620 -> AQI 401-500.
    "so2": np.array([0.0, 40.0, 80.0, 380.0, 800.0, 1600.0, 2620.0], dtype=np.float64),
    # CO, 8h, mg/m3 (NOT ug/m3). Top band 34-... -> AQI 401-500.
    "co": np.array([0.0, 1.0, 2.0, 10.0, 17.0, 34.0, 51.0], dtype=np.float64),
    # O3, 8h, ug/m3. Top band 748-... -> AQI 401-500.
    "o3": np.array([0.0, 50.0, 100.0, 168.0, 208.0, 748.0, 1288.0], dtype=np.float64),
    # NH3, 24h, ug/m3. Top band 1800-2400 -> AQI 401-500.
    "nh3": np.array([0.0, 200.0, 400.0, 800.0, 1200.0, 1800.0, 2400.0], dtype=np.float64),
    # Pb, 24h, ug/m3. Top band 3.5-... -> AQI 401-500.
    "pb": np.array([0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0], dtype=np.float64),
}

#: Canonical ordered list of the eight CPCB pollutants.
POLLUTANTS: tuple[str, ...] = ("pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "pb")

#: Pollutants whose presence is required for a valid NAQI (>=1 of PM2.5 / PM10).
PM_POLLUTANTS: tuple[str, ...] = ("pm25", "pm10")

#: Minimum number of valid pollutant sub-indices required for a valid NAQI.
MIN_VALID_POLLUTANTS: int = 3

#: Averaging period in hours for each pollutant (CO/O3 = 8h, rest = 24h).
AVERAGING_HOURS: dict[str, int] = {
    "pm25": 24,
    "pm10": 24,
    "no2": 24,
    "so2": 24,
    "co": 8,
    "o3": 8,
    "nh3": 24,
    "pb": 24,
}

#: Concentration unit of each pollutant (CO in mg/m3, all others ug/m3).
UNITS: dict[str, str] = {
    "pm25": "ug/m3",
    "pm10": "ug/m3",
    "no2": "ug/m3",
    "so2": "ug/m3",
    "co": "mg/m3",
    "o3": "ug/m3",
    "nh3": "ug/m3",
    "pb": "ug/m3",
}

#: Six CPCB AQI category names (band 0..5).
CATEGORY_NAMES: tuple[str, ...] = (
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
)

#: Official-style hex colours for the six categories (green -> dark red).
CATEGORY_COLORS: tuple[str, ...] = (
    "#009865",  # Good        — green
    "#A3C853",  # Satisfactory— light green
    "#FFF833",  # Moderate    — yellow
    "#F29C33",  # Poor        — orange
    "#E93F33",  # Very Poor   — red
    "#AF2D24",  # Severe      — dark red
)


def category_index(aqi: float) -> int:
    """Return the CPCB category band index (0..5) for an AQI value.

    Args:
        aqi: AQI value in [0, 500]. Values are clamped into range.

    Returns:
        Integer band index where 0=Good ... 5=Severe.
    """
    if np.isnan(aqi):
        raise ValueError("category_index is undefined for NaN AQI")
    a = float(np.clip(aqi, 0.0, AQI_MAX))
    # Bands: [0,50], (50,100], (100,200], (200,300], (300,400], (400,500].
    edges = np.array([50.0, 100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    return int(np.searchsorted(edges, a, side="left"))


def category_name(aqi: float) -> str:
    """Return the CPCB category name for an AQI value."""
    return CATEGORY_NAMES[category_index(aqi)]


def category_color(aqi: float) -> str:
    """Return the hex colour for the CPCB category of an AQI value."""
    return CATEGORY_COLORS[category_index(aqi)]


__all__ = [
    "AQI_CATEGORY_EDGES",
    "AQI_BAND_LO",
    "AQI_BAND_HI",
    "AQI_MAX",
    "BREAKPOINTS",
    "POLLUTANTS",
    "PM_POLLUTANTS",
    "MIN_VALID_POLLUTANTS",
    "AVERAGING_HOURS",
    "UNITS",
    "CATEGORY_NAMES",
    "CATEGORY_COLORS",
    "category_index",
    "category_name",
    "category_color",
]
