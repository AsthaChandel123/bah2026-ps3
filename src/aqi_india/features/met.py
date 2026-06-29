"""Meteorological predictor stack for the surface-concentration models.

Meteorology controls the column->surface mapping: boundary-layer height sets the
dilution volume, relative humidity drives hygroscopic aerosol growth, wind
ventilates/transports pollution, temperature and solar radiation drive
photochemistry (O3, secondary HCHO), and precipitation wet-scavenges aerosol.
This module assembles those drivers into a tidy, model-ready predictor stack
derived from the fused gridded ``Dataset`` (schema in DEV_CONTRACT.md S6.1).

.. important::
   **Boundary-layer height and pressure-level/upper winds must come from CDS
   (cdsapi) or MERRA-2 (M2T1NXFLX), NOT from GEE's ERA5 mirror**, which omits
   ``boundary_layer_height`` and pressure-level winds. The fused cube is expected
   to already carry ``blh`` sourced this way (see DATA_SOURCES.md); this module
   only *derives* downstream met features and never silently substitutes a
   GEE-ERA5 BLH.

All derivations are NumPy/xarray-vectorized and NaN-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import physics

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

#: Canonical met predictor variable names emitted by :func:`build_met_stack`.
MET_FEATURES: tuple[str, ...] = (
    "blh",
    "rh",
    "wind_speed",
    "wind_dir",
    "wind_u",
    "wind_v",
    "t2m",
    "ssrd",
    "precip",
    "mslp",
)

#: Sources from which BLH and upper-level winds MUST be obtained (never GEE-ERA5).
BLH_WIND_SOURCES: tuple[str, ...] = ("CDS/ERA5 (cdsapi)", "MERRA-2 M2T1NXFLX")


def derive_wind_features(ds: "xr.Dataset") -> "xr.Dataset":
    """Add ``wind_speed`` and ``wind_dir`` from ``wind_u`` / ``wind_v``.

    Args:
        ds: Gridded dataset carrying ``wind_u`` and ``wind_v`` (m/s).

    Returns:
        A shallow copy of ``ds`` with ``wind_speed`` (m/s) and ``wind_dir``
        (meteorological degrees, direction the wind blows *from*) added. If the
        wind components are absent, ``ds`` is returned unchanged.
    """
    if "wind_u" not in ds or "wind_v" not in ds:
        return ds
    out = ds.copy()
    spd = physics.wind_speed(ds["wind_u"], ds["wind_v"])
    direction = physics.wind_dir(ds["wind_u"], ds["wind_v"])
    out["wind_speed"] = spd.assign_attrs(units="m s-1", long_name="10 m wind speed")
    out["wind_dir"] = direction.assign_attrs(
        units="degrees",
        long_name="10 m wind direction (meteorological, from)",
    )
    return out


def ensure_relative_humidity(ds: "xr.Dataset") -> "xr.Dataset":
    """Ensure an ``rh`` variable exists, deriving it from dewpoint if needed.

    If ``rh`` is already present it is returned untouched. Otherwise, when both
    ``t2m`` and a dewpoint variable (``d2m``) are present, RH is computed with the
    Magnus formula (:func:`physics.relative_humidity_from_dewpoint`).

    Args:
        ds: Gridded dataset.

    Returns:
        ``ds`` (copy) guaranteed to carry ``rh`` (%) when derivable.
    """
    if "rh" in ds:
        return ds
    if "t2m" in ds and "d2m" in ds:
        out = ds.copy()
        rh = physics.relative_humidity_from_dewpoint(ds["t2m"], ds["d2m"])
        out["rh"] = rh.assign_attrs(units="%", long_name="2 m relative humidity")
        return out
    return ds


def build_met_stack(
    ds: "xr.Dataset", *, features: tuple[str, ...] | None = None
) -> "xr.Dataset":
    """Assemble the meteorological predictor stack from the fused grid cube.

    Derives :func:`derive_wind_features` and :func:`ensure_relative_humidity`,
    then selects the canonical met predictors (those present in ``ds``). Variables
    listed in ``features`` but absent from ``ds`` are skipped (the synthetic demo
    cube does not carry ``precip``/``mslp``), so the result is the intersection of
    requested and available predictors.

    Args:
        ds: Fused gridded ``Dataset`` (DEV_CONTRACT S6.1) with dims
            ``(time, lat, lon)``.
        features: Optional subset of :data:`MET_FEATURES` to extract; defaults to
            all of them.

    Returns:
        An :class:`xarray.Dataset` containing the available met predictors, each
        ``float32`` with ``(time, lat, lon)`` dims and descriptive attributes.
    """
    import xarray as xr

    wanted = features if features is not None else MET_FEATURES
    enriched = derive_wind_features(ensure_relative_humidity(ds))

    data_vars: dict[str, xr.DataArray] = {}
    for name in wanted:
        if name in enriched:
            data_vars[name] = enriched[name].astype("float32")
    out = xr.Dataset(data_vars, coords=enriched.coords)
    out.attrs["description"] = "Meteorological predictor stack (aqi_india.features.met)"
    out.attrs["blh_wind_provenance"] = (
        "BLH and upper-level winds must originate from CDS/MERRA-2, not GEE-ERA5; "
        f"valid sources: {', '.join(BLH_WIND_SOURCES)}."
    )
    return out


def met_feature_names(ds: "xr.Dataset") -> list[str]:
    """Return the met predictor names actually derivable from ``ds``.

    Args:
        ds: Fused gridded dataset.

    Returns:
        The ordered subset of :data:`MET_FEATURES` present after derivation.
    """
    enriched = derive_wind_features(ensure_relative_humidity(ds))
    return [name for name in MET_FEATURES if name in enriched]


__all__ = [
    "MET_FEATURES",
    "BLH_WIND_SOURCES",
    "derive_wind_features",
    "ensure_relative_humidity",
    "build_met_stack",
    "met_feature_names",
]
