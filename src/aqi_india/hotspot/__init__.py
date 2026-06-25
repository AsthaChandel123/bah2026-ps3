"""aqi_india.hotspot — Objective-2 HCHO hotspot detection.

Per-season robust z-anomaly standardisation feeding a three-vote consensus
(Getis-Ord Gi* + LISA HH + 95th-percentile/z>2 exceedance), DBSCAN/HDBSCAN
polygon delineation, and Emerging Hot Spot Analysis (the headline temporal
deliverable). All public entry points import under the light dependency set;
``hdbscan``/``alphashape`` are optional and imported lazily.
"""

from __future__ import annotations

from .climatology import build_climatology, robust_z_anomaly
from .consensus import detect_hotspots
from .ehsa import EHSA_CATEGORIES, emerging_hotspot_analysis
from .getis_ord import getis_ord_gi
from .lisa import local_morans_i

__all__ = [
    "build_climatology",
    "robust_z_anomaly",
    "getis_ord_gi",
    "local_morans_i",
    "emerging_hotspot_analysis",
    "detect_hotspots",
    "EHSA_CATEGORIES",
]
