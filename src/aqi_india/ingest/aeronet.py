"""AERONET AOD ground-truth adapter for satellite-AOD validation.

AERONET (NASA's Aerosol Robotic Network) sun-photometers provide the reference
spectral AOD used to **validate** satellite AOD (MAIAC, INSAT-3D) before it is
fed into the AOD->PM2.5 branch. Validation follows the standard "expected-error
envelope": a satellite AOD agrees with AERONET when it falls within
``+/- (0.05 + 0.15 * AOD_AERONET)`` (the over-land MODIS EE used throughout the
blueprint).

This adapter fetches the AERONET v3 Web Service (Direct-Sun L2) for the Indian
sites, interpolates the spectral AOD to 550 nm via the Angstrom exponent, and
aggregates to daily means. ``requests`` is lazy-imported. The ``synthetic=True``
path fabricates daily 550 nm AOD at the canonical Indian sites consistent with the
synthetic ``aod`` field so the validation step runs offline.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger("ingest.aeronet")

#: AERONET v3 Direct-Sun web-service endpoint.
AERONET_WEB_SERVICE: str = (
    "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
)

#: Canonical Indian AERONET sites (name -> (lat, lon)) used for AOD validation.
INDIA_AERONET_SITES: dict[str, tuple[float, float]] = {
    "Kanpur": (26.513, 80.232),
    "Gandhi_College": (25.871, 84.128),
    "Jaipur": (26.906, 75.806),
    "Pune": (18.537, 73.805),
    "Gual_Pahari": (28.426, 77.151),
    "Nainital": (29.360, 79.460),
}

#: Expected-error envelope coefficients: |sat - ref| <= EE_ABS + EE_REL*ref.
EE_ABS: float = 0.05
EE_REL: float = 0.15

#: Angstrom-law reference wavelengths (nm) used to interpolate to 550 nm.
_LAMBDA_440: float = 440.0
_LAMBDA_675: float = 675.0
_LAMBDA_550: float = 550.0

DateLike = str | date | datetime


def fetch_aeronet(
    start: DateLike,
    end: DateLike,
    sites: tuple[str, ...] | None = None,
    *,
    synthetic: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Fetch daily AERONET 550 nm AOD at the Indian validation sites.

    Real path: queries the AERONET v3 web service (L2 Direct-Sun) per site over
    ``[start, end)``, interpolates ``AOD_440``/``AOD_675`` to 550 nm via the
    Angstrom exponent, and aggregates to daily means.

    Offline path: fabricates plausible daily 550 nm AOD at the requested sites
    (seasonally varying, IGP-elevated) so the satellite-AOD validation step runs
    without network access.

    Args:
        start: First day (inclusive).
        end: Last day (exclusive).
        sites: Site names to fetch (default: all of :data:`INDIA_AERONET_SITES`).
        synthetic: Offline synthetic path when ``True``.
        seed: Synthetic seed.

    Returns:
        A long table with columns ``site, lat, lon, time, aod_550``.
    """
    site_names = tuple(sites) if sites else tuple(INDIA_AERONET_SITES)
    if synthetic:
        return _synthetic_aeronet(start, end, site_names, seed=seed)
    return _fetch_aeronet_web(start, end, site_names)


def _synthetic_aeronet(
    start: DateLike,
    end: DateLike,
    sites: tuple[str, ...],
    *,
    seed: int,
) -> pd.DataFrame:
    """Fabricate daily 550 nm AERONET AOD at the requested sites (offline path)."""
    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    days = pd.date_range(t0, t1 - pd.Timedelta(days=1), freq="D")
    if len(days) == 0:
        days = pd.DatetimeIndex([t0])
    rng = np.random.default_rng([seed, 99])

    rows: list[dict] = []
    for site in sites:
        if site not in INDIA_AERONET_SITES:
            logger.warning("unknown AERONET site %r; skipping", site)
            continue
        lat, lon = INDIA_AERONET_SITES[site]
        # IGP sites carry higher background AOD; winter (DJF) elevated.
        igp = 1.0 if lat >= 24.0 else 0.5
        doy = days.dayofyear.to_numpy().astype(float)
        season = 1.0 + 0.5 * np.cos(2.0 * np.pi * (doy - 330.0) / 365.25)
        base = 0.2 + 0.5 * igp * season
        noise = rng.normal(0.0, 0.05, len(days))
        aod = np.clip(base + noise, 0.02, 3.0)
        for d, a in zip(days, aod, strict=False):
            rows.append(
                {"site": site, "lat": lat, "lon": lon, "time": d, "aod_550": float(a)}
            )
    df = pd.DataFrame(rows)
    logger.info("fetch_aeronet(synthetic): %d site-days at %d sites", len(df), len(sites))
    return df


def _fetch_aeronet_web(
    start: DateLike, end: DateLike, sites: tuple[str, ...]
) -> pd.DataFrame:
    """Query the AERONET v3 web service per site and interpolate to 550 nm."""
    import requests

    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end) - pd.Timedelta(days=1)
    frames: list[pd.DataFrame] = []
    for site in sites:
        if site not in INDIA_AERONET_SITES:
            continue
        params = {
            "site": site,
            "year": t0.year,
            "month": t0.month,
            "day": t0.day,
            "year2": t1.year,
            "month2": t1.month,
            "day2": t1.day,
            "AOD15": 0,
            "AOD20": 1,  # L2 quality-assured
            "AVG": 20,  # daily averages
            "if_no_html": 1,
        }
        try:
            resp = requests.get(AERONET_WEB_SERVICE, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("AERONET request failed (%s): %s", site, exc)
            continue
        part = _parse_aeronet_csv(resp.text, site)
        if part is not None and len(part):
            frames.append(part)

    if not frames:
        logger.warning("AERONET returned no data for the requested sites/window.")
        return pd.DataFrame(columns=["site", "lat", "lon", "time", "aod_550"])
    return pd.concat(frames, ignore_index=True)


def _parse_aeronet_csv(text: str, site: str) -> pd.DataFrame | None:
    """Parse an AERONET web-service CSV block into a 550 nm daily AOD frame."""
    import io

    # The v3 service prefixes several metadata lines before the CSV header.
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Date(dd:mm:yyyy)")), None
    )
    if header_idx is None:
        return None
    csv = "\n".join(lines[header_idx:])
    try:
        raw = pd.read_csv(io.StringIO(csv))
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return None

    date_col = next((c for c in raw.columns if c.startswith("Date(dd")), None)
    if date_col is None:
        return None
    time = pd.to_datetime(raw[date_col], format="%d:%m:%Y", errors="coerce")
    aod440 = pd.to_numeric(raw.get("AOD_440nm"), errors="coerce")
    aod675 = pd.to_numeric(raw.get("AOD_675nm"), errors="coerce")
    aod550 = _interp_to_550(aod440.to_numpy(), aod675.to_numpy())

    lat, lon = INDIA_AERONET_SITES[site]
    out = pd.DataFrame(
        {"site": site, "lat": lat, "lon": lon, "time": time, "aod_550": aod550}
    ).dropna(subset=["time", "aod_550"])
    out = out.groupby(["site", "lat", "lon", out["time"].dt.normalize()], as_index=False)[
        "aod_550"
    ].mean()
    out = out.rename(columns={"time": "time"})
    return out


def _interp_to_550(aod440: np.ndarray, aod675: np.ndarray) -> np.ndarray:
    """Interpolate AOD to 550 nm via the Angstrom power law from 440/675 nm.

    ``alpha = -ln(AOD_440 / AOD_675) / ln(440/675)`` and
    ``AOD_550 = AOD_440 * (550/440)^(-alpha)``.

    Args:
        aod440: AOD at 440 nm.
        aod675: AOD at 675 nm.

    Returns:
        AOD at 550 nm (NaN where inputs are non-positive/missing).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = -np.log(aod440 / aod675) / np.log(_LAMBDA_440 / _LAMBDA_675)
        aod550 = aod440 * (_LAMBDA_550 / _LAMBDA_440) ** (-alpha)
    valid = (aod440 > 0) & (aod675 > 0)
    return np.where(valid, aod550, np.nan)


def within_expected_error(
    sat_aod: np.ndarray, ref_aod: np.ndarray
) -> np.ndarray:
    """Return a boolean mask of satellite AOD within the AERONET EE envelope.

    Implements ``|sat - ref| <= EE_ABS + EE_REL * ref`` (the over-land MODIS
    expected-error envelope). The fraction of ``True`` is the headline
    satellite-AOD validation metric.

    Args:
        sat_aod: Co-located satellite AOD values.
        ref_aod: AERONET reference AOD values.

    Returns:
        Boolean array; ``True`` where the satellite AOD is within the envelope.
    """
    sat = np.asarray(sat_aod, dtype=np.float64)
    ref = np.asarray(ref_aod, dtype=np.float64)
    envelope = EE_ABS + EE_REL * ref
    return np.abs(sat - ref) <= envelope


__all__ = [
    "AERONET_WEB_SERVICE",
    "INDIA_AERONET_SITES",
    "EE_ABS",
    "EE_REL",
    "fetch_aeronet",
    "within_expected_error",
]
