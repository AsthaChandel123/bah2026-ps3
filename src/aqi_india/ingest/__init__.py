"""Data-source ingest adapters for BAH 2026 PS3 (``aqi_india.ingest``).

Each adapter targets a **real** endpoint documented in ``docs/DATA_SOURCES.md``
and exposes a ``fetch_*`` entry point with a ``synthetic=True`` offline path that
delegates to :mod:`aqi_india.sim.synthetic` and returns the exact
``DEV_CONTRACT`` §6 schema, so the whole pipeline runs without credentials.

Modules:

* :mod:`~aqi_india.ingest.gee_auth` — Earth Engine auth + reduction helpers.
* :mod:`~aqi_india.ingest.s5p` — Sentinel-5P TROPOMI L3 columns (GEE).
* :mod:`~aqi_india.ingest.maiac` — MODIS MAIAC 1 km AOD backbone (GEE).
* :mod:`~aqi_india.ingest.insat_aod` — INSAT-3D/3DR AOD (MOSDAC HDF5, non-GEE).
* :mod:`~aqi_india.ingest.firms` — NASA FIRMS VIIRS 375 m + MODIS active fire.
* :mod:`~aqi_india.ingest.era5_cds` — ERA5 single-levels + BLH/upper winds (CDS).
* :mod:`~aqi_india.ingest.imdaa` — IMDAA 12 km India reanalysis (NCMRWF).
* :mod:`~aqi_india.ingest.merra2` — MERRA-2 AOD + PBLH + speciated PM (earthaccess).
* :mod:`~aqi_india.ingest.cams` — CAMS EAC4/NRT gap-free composition prior (ADS).
* :mod:`~aqi_india.ingest.cpcb` — CPCB OGD labels + unified station DB.
* :mod:`~aqi_india.ingest.aeronet` — AERONET AOD validation ground truth.
* :mod:`~aqi_india.ingest.exporters` — COG / GeoParquet / EE-table exporters.

Submodules are imported lazily via ``__getattr__`` so ``import aqi_india.ingest``
stays light (no heavy/credentialed deps at import time).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "gee_auth",
    "s5p",
    "maiac",
    "insat_aod",
    "firms",
    "era5_cds",
    "imdaa",
    "merra2",
    "cams",
    "cpcb",
    "aeronet",
    "exporters",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from . import (  # noqa: F401
        aeronet,
        cams,
        cpcb,
        era5_cds,
        exporters,
        firms,
        gee_auth,
        imdaa,
        insat_aod,
        maiac,
        merra2,
        s5p,
    )


def __getattr__(name: str) -> Any:
    """Lazily import and return a submodule by name (PEP 562)."""
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
