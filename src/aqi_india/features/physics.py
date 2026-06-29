"""Physics-guided feature engineering — the single biggest pre-ML accuracy lift.

The raw column-to-surface mapping (AOD -> PM2.5, tropospheric column -> surface
mixing ratio) is weak on its own because the link is modulated by first-principle
physics the satellite cannot see directly:

* **Vertical mixing.** A column of aerosol confined to a shallow boundary layer
  produces a far higher *surface* concentration than the same column spread
  through a deep, well-mixed layer. Hence ``AOD / BLH`` (PBL normalization).
* **Hygroscopic growth.** Aerosol optical depth swells non-linearly as relative
  humidity approaches saturation (water uptake), so a fixed surface dry mass maps
  to a larger AOD at high RH. Dividing by ``f(RH)`` recovers the dry signal.
* **Composition / regime.** The ``HCHO / NO2`` ratio (FNR) discriminates the
  VOC-limited from the NOx-limited ozone-production regime; aerosol-type flags
  (from the UV Aerosol Index and the Angstrom exponent) separate coarse dust from
  fine smoke/haze, which have very different mass-extinction efficiencies.

Per the design blueprint these engineered channels lift the AOD->PM2.5 skill from
R^2 ~ 0.4 to ~0.65. Every function here is pure, NumPy/xarray-vectorized and
NaN-safe: invalid or non-physical inputs propagate as ``NaN`` rather than raising
or returning sentinel numbers, matching the package-wide NaN convention.

References:
    Hu et al. (2014); Liu et al. (2005) — AOD/PBL/RH PM2.5 calibration.
    Duncan et al. (2010); Jin et al. (2017) — HCHO/NO2 (FNR) ozone regime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

#: Array-like accepted/returned by the physics kernels (NumPy or xarray).
ArrayLike = Union["np.ndarray", "xr.DataArray", float]

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Mass-extinction-efficiency priors (m^2 / g, ~550 nm, ambient).
# Used to convert AOD into a dry/ambient surface-mass first guess and to weight
# aerosol-type-dependent AOD->PM relationships. Values are literature midpoints;
# they are *priors*, refined by the ML, not hard physical constants.
# ---------------------------------------------------------------------------
MASS_EXTINCTION_EFFICIENCY: dict[str, float] = {
    "sulfate": 5.0,
    "organic_carbon": 4.0,
    "black_carbon": 9.0,
    "dust": 0.6,
    "sea_salt": 3.0,
    "fine_pollution": 4.5,
    "coarse_dust": 0.7,
}

#: Default hygroscopic-growth exponent ``gamma`` in ``f(RH) = (1 - RH/100)^-g``.
#: ~0.6 is a robust IGP/continental-aerosol midpoint (Liu et al., 2005).
DEFAULT_GAMMA: float = 0.6

#: RH is clamped strictly below this (%) to keep ``f(RH)`` finite as RH -> 100.
_RH_CEILING: float = 99.0

#: Integer codes emitted by :func:`aerosol_type_flag`.
AEROSOL_TYPE_CODES: dict[str, int] = {
    "unknown": 0,
    "dust": 1,
    "smoke_absorbing": 2,
    "fine_nonabsorbing": 3,
    "mixed": 4,
}


def _is_xarray(obj: object) -> bool:
    """Return True if ``obj`` is an xarray DataArray (without importing it eagerly)."""
    return type(obj).__name__ == "DataArray" and type(obj).__module__.startswith(
        "xarray"
    )


def _asfloat(arr: ArrayLike) -> np.ndarray | "xr.DataArray":
    """Coerce an input to float, preserving xarray containers.

    NumPy/scalar inputs become ``float64`` ndarrays; xarray DataArrays are
    returned as float DataArrays so coordinates/dims survive the computation.
    """
    if _is_xarray(arr):
        return arr.astype("float64")  # type: ignore[union-attr]
    return np.asarray(arr, dtype=np.float64)


def f_rh(rh: ArrayLike, gamma: float = DEFAULT_GAMMA) -> np.ndarray | "xr.DataArray":
    """Hygroscopic-growth factor ``f(RH) = (1 - RH/100)^(-gamma)``.

    The factor describes how ambient aerosol optical depth grows relative to its
    dry value as relative humidity increases. It diverges as ``RH -> 100`` and is
    undefined for ``RH >= 100``; RH is therefore clamped to ``[0, 99]`` % before
    evaluation and non-finite / out-of-range inputs propagate as ``NaN``.

    Args:
        rh: Relative humidity in percent (0..100).
        gamma: Growth exponent (dimensionless). ~0.6 for continental aerosol.

    Returns:
        The growth factor ``f(RH) >= 1`` (same container type as ``rh``).
    """
    rh_f = _asfloat(rh)
    valid = (rh_f >= 0.0) & (rh_f <= 100.0)
    # Clamp into [0, ceiling] so (1 - RH/100) stays in [1-0.99, 1].
    if _is_xarray(rh_f):
        rh_c = rh_f.clip(0.0, _RH_CEILING)
    else:
        rh_c = np.clip(rh_f, 0.0, _RH_CEILING)
    factor = (1.0 - rh_c / 100.0) ** (-gamma)
    return _where(valid, factor, np.nan)


def _where(cond: object, a: object, b: object) -> np.ndarray | "xr.DataArray":
    """``np.where``-style select that also works for xarray DataArrays."""
    if _is_xarray(cond) or _is_xarray(a):
        import xarray as xr

        return xr.where(cond, a, b)
    return np.where(cond, a, b)


def pbl_normalized_aod(aod: ArrayLike, blh: ArrayLike) -> np.ndarray | "xr.DataArray":
    """PBL-normalized AOD ``= AOD / BLH`` (an aerosol-density proxy).

    Dividing the (vertically integrated) optical depth by the boundary-layer
    height approximates the near-surface aerosol *density*, decoupling the
    surface signal from how deeply the column is mixed. This is the dominant
    physics correction for the AOD->PM2.5 breakdown under IGP winter inversions.

    Args:
        aod: Aerosol optical depth (unitless, >= 0).
        blh: Boundary-layer height in metres (> 0). Pull from CDS/MERRA-2, not
            GEE-ERA5, which lacks BLH.

    Returns:
        ``AOD / BLH`` (units m^-1). Non-finite AOD, or non-positive/non-finite
        BLH, yield ``NaN``.
    """
    aod_f = _asfloat(aod)
    blh_f = _asfloat(blh)
    valid = (blh_f > 0.0) & _isfinite(blh_f) & _isfinite(aod_f)
    safe_blh = _where(valid, blh_f, np.nan)
    return _where(valid, aod_f / safe_blh, np.nan)


def hygroscopic_correction(
    aod: ArrayLike, rh: ArrayLike, gamma: float = DEFAULT_GAMMA
) -> np.ndarray | "xr.DataArray":
    """Hygroscopically-corrected (dry-equivalent) AOD ``= AOD * f(RH)^-1``.

    Removes the relative-humidity-driven optical swelling so the corrected value
    tracks dry aerosol mass, which is what surface PM monitors measure. The
    correction is steep above ~70% RH where ``f(RH)`` rises sharply.

    Args:
        aod: Ambient aerosol optical depth (unitless, >= 0).
        rh: Relative humidity in percent (0..100).
        gamma: Hygroscopic-growth exponent passed to :func:`f_rh`.

    Returns:
        Dry-equivalent AOD ``AOD / f(RH)``. Invalid inputs propagate as ``NaN``.
    """
    aod_f = _asfloat(aod)
    factor = f_rh(rh, gamma=gamma)
    valid = _isfinite(aod_f) & _isfinite(factor) & (factor > 0.0)
    safe_factor = _where(valid, factor, np.nan)
    return _where(valid, aod_f / safe_factor, np.nan)


def fnr(
    hcho_col: ArrayLike,
    no2_col: ArrayLike,
    *,
    min_no2: float = 1e-8,
) -> np.ndarray | "xr.DataArray":
    """Formaldehyde-to-NO2 ratio (FNR) ``= HCHO_col / NO2_col``.

    FNR is the canonical proxy for the ozone-production photochemical regime:
    low FNR (< ~1) indicates a VOC-limited (NOx-saturated) regime typical of city
    cores, high FNR (> ~2) a NOx-limited regime. Columns must share units
    (``mol/m2``); the ratio is dimensionless. Cells where NO2 is below
    ``min_no2`` (numerically zero / noise) are masked to avoid division blow-up.

    Args:
        hcho_col: Tropospheric HCHO column (mol/m2).
        no2_col: Tropospheric NO2 column (mol/m2).
        min_no2: Lower NO2 threshold (mol/m2) below which the ratio is masked.

    Returns:
        The HCHO/NO2 ratio; ``NaN`` where either column is non-finite/negative or
        NO2 < ``min_no2``.
    """
    hcho_f = _asfloat(hcho_col)
    no2_f = _asfloat(no2_col)
    valid = (
        _isfinite(hcho_f)
        & _isfinite(no2_f)
        & (hcho_f >= 0.0)
        & (no2_f >= min_no2)
    )
    safe_no2 = _where(valid, no2_f, np.nan)
    return _where(valid, hcho_f / safe_no2, np.nan)


def _isfinite(arr: object) -> object:
    """``np.isfinite`` that also accepts xarray DataArrays."""
    if _is_xarray(arr):
        import xarray as xr

        return xr.apply_ufunc(np.isfinite, arr)
    return np.isfinite(arr)


def relative_humidity_from_dewpoint(
    t2m: ArrayLike, d2m: ArrayLike, *, kelvin: bool = True
) -> np.ndarray | "xr.DataArray":
    """Relative humidity (%) from 2 m temperature and dewpoint (Magnus formula).

    Uses the Magnus-Tetens saturation-vapour-pressure approximation::

        e_s(T)  = 6.112 * exp(a*T / (b + T))         # T in degC
        RH      = 100 * e_s(Td) / e_s(T)

    with ``a = 17.625``, ``b = 243.04`` degC (Alduchov & Eskridge, 1996). Result
    is clipped to ``[0, 100]``.

    Args:
        t2m: 2 m air temperature.
        d2m: 2 m dewpoint temperature (same units as ``t2m``).
        kelvin: If True, inputs are in Kelvin and are converted to degC; if
            False they are already in degC.

    Returns:
        Relative humidity in percent, clipped to ``[0, 100]``; ``NaN`` for
        non-finite inputs.
    """
    a, b = 17.625, 243.04
    t = _asfloat(t2m)
    td = _asfloat(d2m)
    if kelvin:
        t = t - 273.15
        td = td - 273.15
    valid = _isfinite(t) & _isfinite(td)
    # e_s(Td) / e_s(T) without the common 6.112 prefactor.
    es_t = _exp(a * t / (b + t))
    es_td = _exp(a * td / (b + td))
    rh = 100.0 * es_td / es_t
    rh = _clip(rh, 0.0, 100.0)
    return _where(valid, rh, np.nan)


def _exp(arr: object) -> object:
    """``np.exp`` that also accepts xarray DataArrays."""
    if _is_xarray(arr):
        import xarray as xr

        return xr.apply_ufunc(np.exp, arr)
    return np.exp(arr)


def _clip(arr: object, lo: float, hi: float) -> object:
    """Clip that works for both NumPy arrays and xarray DataArrays."""
    if _is_xarray(arr):
        return arr.clip(lo, hi)  # type: ignore[union-attr]
    return np.clip(arr, lo, hi)


def wind_speed(u: ArrayLike, v: ArrayLike) -> np.ndarray | "xr.DataArray":
    """Wind speed ``= sqrt(u^2 + v^2)`` from u/v wind components.

    Args:
        u: Zonal (eastward) wind component (m/s).
        v: Meridional (northward) wind component (m/s).

    Returns:
        Scalar wind speed (m/s); ``NaN`` where either component is non-finite.
    """
    u_f = _asfloat(u)
    v_f = _asfloat(v)
    valid = _isfinite(u_f) & _isfinite(v_f)
    spd = _sqrt(u_f * u_f + v_f * v_f)
    return _where(valid, spd, np.nan)


def _sqrt(arr: object) -> object:
    """``np.sqrt`` that also accepts xarray DataArrays."""
    if _is_xarray(arr):
        import xarray as xr

        return xr.apply_ufunc(np.sqrt, arr)
    return np.sqrt(arr)


def wind_dir(
    u: ArrayLike, v: ArrayLike, *, convention: str = "meteorological"
) -> np.ndarray | "xr.DataArray":
    """Wind direction in degrees from u/v components.

    Args:
        u: Zonal (eastward) wind component (m/s).
        v: Meridional (northward) wind component (m/s).
        convention: ``"meteorological"`` returns the direction the wind blows
            *from* (0deg = from North, 90deg = from East); ``"oceanographic"``
            returns the direction it blows *toward*.

    Returns:
        Direction in degrees ``[0, 360)``; ``NaN`` for non-finite inputs.
    """
    u_f = _asfloat(u)
    v_f = _asfloat(v)
    valid = _isfinite(u_f) & _isfinite(v_f)
    if convention not in ("meteorological", "oceanographic"):
        raise ValueError(
            "convention must be 'meteorological' or 'oceanographic', "
            f"got {convention!r}"
        )
    if convention == "meteorological":
        # Direction FROM which the wind originates.
        deg = _arctan2(-u_f, -v_f) * (180.0 / np.pi)
    else:
        deg = _arctan2(u_f, v_f) * (180.0 / np.pi)
    deg = (deg + 360.0) % 360.0
    return _where(valid, deg, np.nan)


def _arctan2(y: object, x: object) -> object:
    """``np.arctan2`` that also accepts xarray DataArrays."""
    if _is_xarray(y) or _is_xarray(x):
        import xarray as xr

        return xr.apply_ufunc(np.arctan2, y, x)
    return np.arctan2(y, x)


def aerosol_type_flag(
    aer_ai: ArrayLike | None = None,
    angstrom: ArrayLike | None = None,
    *,
    ai_dust: float = 1.0,
    ai_smoke: float = 1.0,
    angstrom_coarse: float = 0.6,
    angstrom_fine: float = 1.2,
) -> np.ndarray | "xr.DataArray":
    """Classify dominant aerosol type from UV Aerosol Index and Angstrom exponent.

    Combines two orthogonal discriminators:

    * **UV Aerosol Index (AER_AI)** — positive for UV-*absorbing* aerosol
      (mineral dust, biomass-burning smoke), near-zero/negative for
      non-absorbing aerosol (sulfate/nitrate haze, sea salt).
    * **Angstrom exponent** — small (~0) for *coarse* particles (dust), large
      (>~1.2) for *fine* particles (smoke, urban pollution).

    Decision logic (codes in :data:`AEROSOL_TYPE_CODES`):

    ====================================  ==================
    Condition                             Type
    ====================================  ==================
    AI >= ai_dust AND coarse Angstrom     ``dust`` (1)
    AI >= ai_smoke AND fine Angstrom      ``smoke_absorbing`` (2)
    AI < threshold AND fine Angstrom      ``fine_nonabsorbing`` (3)
    finite but ambiguous                  ``mixed`` (4)
    all inputs missing                    ``unknown`` (0)
    ====================================  ==================

    At least one of ``aer_ai`` / ``angstrom`` must be provided. The shared output
    shape is inferred from whichever inputs are arrays.

    Args:
        aer_ai: UV Aerosol Index (unitless), or ``None`` if unavailable.
        angstrom: Angstrom exponent (unitless), or ``None`` if unavailable.
        ai_dust: AI threshold above which absorbing aerosol is flagged as dust
            (when paired with a coarse Angstrom exponent).
        ai_smoke: AI threshold for absorbing smoke (paired with fine Angstrom).
        angstrom_coarse: Angstrom value at/below which particles are coarse.
        angstrom_fine: Angstrom value at/above which particles are fine.

    Returns:
        Integer-coded aerosol-type array (see :data:`AEROSOL_TYPE_CODES`).
    """
    if aer_ai is None and angstrom is None:
        raise ValueError("Provide at least one of aer_ai or angstrom.")

    ai = _asfloat(aer_ai) if aer_ai is not None else None
    ang = _asfloat(angstrom) if angstrom is not None else None

    # Establish a broadcasting template so we can build a full-shape result even
    # when only one discriminator is present.
    template = ai if ai is not None else ang
    code = _full_like_int(template, AEROSOL_TYPE_CODES["unknown"])

    has_ai = ai is not None
    has_ang = ang is not None

    ai_absorbing = (ai >= ai_dust) if has_ai else _false_like(template)
    ai_smoky = (ai >= ai_smoke) if has_ai else _false_like(template)
    ai_clean = (ai < ai_dust) if has_ai else _false_like(template)
    ang_coarse = (ang <= angstrom_coarse) if has_ang else _false_like(template)
    ang_fine = (ang >= angstrom_fine) if has_ang else _false_like(template)

    finite = _isfinite(template)
    if has_ai and has_ang:
        finite = finite & _isfinite(ai) & _isfinite(ang)

    # Order matters: most specific classifications win, then fall back to mixed.
    code = _where(finite, AEROSOL_TYPE_CODES["mixed"], code)
    if has_ai and has_ang:
        code = _where(
            ai_absorbing & ang_coarse, AEROSOL_TYPE_CODES["dust"], code
        )
        code = _where(
            ai_smoky & ang_fine, AEROSOL_TYPE_CODES["smoke_absorbing"], code
        )
        code = _where(
            ai_clean & ang_fine, AEROSOL_TYPE_CODES["fine_nonabsorbing"], code
        )
    elif has_ang:
        code = _where(ang_coarse, AEROSOL_TYPE_CODES["dust"], code)
        code = _where(ang_fine, AEROSOL_TYPE_CODES["fine_nonabsorbing"], code)
    else:  # has_ai only
        code = _where(ai_absorbing, AEROSOL_TYPE_CODES["smoke_absorbing"], code)
        code = _where(ai_clean, AEROSOL_TYPE_CODES["fine_nonabsorbing"], code)

    # Non-finite cells -> unknown.
    code = _where(finite, code, AEROSOL_TYPE_CODES["unknown"])
    return _as_int16(code)


def _full_like_int(template: object, value: int) -> object:
    """Return an int array/DataArray of ``value`` matching ``template``'s shape."""
    if _is_xarray(template):
        import xarray as xr

        return xr.zeros_like(template, dtype="int16") + value
    return np.full(np.shape(template), value, dtype=np.int16)


def _false_like(template: object) -> object:
    """Return an all-False boolean mask matching ``template``'s shape."""
    if _is_xarray(template):
        import xarray as xr

        return xr.zeros_like(template, dtype=bool)
    return np.zeros(np.shape(template), dtype=bool)


def _as_int16(arr: object) -> object:
    """Cast a NumPy/xarray container to int16."""
    if _is_xarray(arr):
        return arr.astype("int16")  # type: ignore[union-attr]
    return np.asarray(arr).astype(np.int16)


def mass_extinction_efficiency(aerosol_type: str) -> float:
    """Return the mass-extinction-efficiency prior (m^2/g) for an aerosol type.

    These priors convert AOD to a surface dry-mass first guess via
    ``PM ~ AOD / (MEE * H * f(RH))`` and seed the ML's per-type AOD->PM mapping.

    Args:
        aerosol_type: One of the keys of :data:`MASS_EXTINCTION_EFFICIENCY`.

    Returns:
        The mass-extinction efficiency (m^2/g).

    Raises:
        KeyError: If ``aerosol_type`` is not a known type.
    """
    key = aerosol_type.lower()
    if key not in MASS_EXTINCTION_EFFICIENCY:
        raise KeyError(
            f"Unknown aerosol type {aerosol_type!r}; "
            f"known: {sorted(MASS_EXTINCTION_EFFICIENCY)}"
        )
    return MASS_EXTINCTION_EFFICIENCY[key]


def mee_for_codes(codes: ArrayLike) -> np.ndarray | "xr.DataArray":
    """Map integer aerosol-type codes to mass-extinction-efficiency priors.

    Provides a vectorized AOD->PM mass weighting channel keyed on the output of
    :func:`aerosol_type_flag`.

    Args:
        codes: Integer aerosol-type codes (see :data:`AEROSOL_TYPE_CODES`).

    Returns:
        Float array of MEE priors (m^2/g); ``NaN`` for the ``unknown`` code.
    """
    # Code -> representative MEE. Unknown -> NaN so it propagates as missing.
    lut = {
        AEROSOL_TYPE_CODES["unknown"]: np.nan,
        AEROSOL_TYPE_CODES["dust"]: MASS_EXTINCTION_EFFICIENCY["coarse_dust"],
        AEROSOL_TYPE_CODES["smoke_absorbing"]: MASS_EXTINCTION_EFFICIENCY[
            "black_carbon"
        ],
        AEROSOL_TYPE_CODES["fine_nonabsorbing"]: MASS_EXTINCTION_EFFICIENCY[
            "fine_pollution"
        ],
        AEROSOL_TYPE_CODES["mixed"]: MASS_EXTINCTION_EFFICIENCY["fine_pollution"],
    }
    codes_f = _asfloat(codes)
    out = _full_like_float(codes_f, np.nan)
    for code, mee in lut.items():
        out = _where(codes_f == code, mee, out)
    return out


def _full_like_float(template: object, value: float) -> object:
    """Return a float array/DataArray of ``value`` matching ``template``'s shape."""
    if _is_xarray(template):
        import xarray as xr

        return xr.full_like(template, value, dtype="float64")
    return np.full(np.shape(template), value, dtype=np.float64)


__all__ = [
    "ArrayLike",
    "MASS_EXTINCTION_EFFICIENCY",
    "DEFAULT_GAMMA",
    "AEROSOL_TYPE_CODES",
    "f_rh",
    "pbl_normalized_aod",
    "hygroscopic_correction",
    "fnr",
    "relative_humidity_from_dewpoint",
    "wind_speed",
    "wind_dir",
    "aerosol_type_flag",
    "mass_extinction_efficiency",
    "mee_for_codes",
]
