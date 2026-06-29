"""Fire emission inventory from FRP (Objective-2 transport).

The trajectory / CWT / PSCF lines tell us *where* the air came from; an emission
inventory tells us *how much* a fire emitted.  Cross-checking the statistical and
Lagrangian attribution against an independent **emission-inventory** line is the
blueprint's credibility test (architecture §8.5-§8.6, method 48): convergence of
the statistical + Lagrangian + inventory estimates is the result, not any single
method.

This module converts FIRMS-surrogate fire radiative power (FRP, MW; DEV_CONTRACT
§6.3) into species emission *rates* and gridded emission *fields* using the
FRP-based approach common to GFAS / QFED / FEER:

    emission_rate(species) = FRP * EF(species)                    [mass / time]

where ``EF`` is an emission coefficient (kg species per MW of FRP, derived from
FRP-to-dry-matter combustion rates × the species emission factor of the burned
vegetation).  We default to **cropland-residue** factors (paddy/wheat stubble) —
the dominant Indo-Gangetic post-monsoon source — and expose a small biome table
plus a multi-inventory **ensemble bracket** (FINN / GFAS / GFED-style scalings)
so the inventory uncertainty (factor ~2-3×, blueprint risk #12) can be reported
honestly rather than as a single point estimate.

Everything runs on the light dependency set; nothing here hits the network.  The
public surface is :func:`emission_rates`, :func:`grid_emissions`,
:func:`inventory_ensemble` and the Hydra entry point :func:`run`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np
import pandas as pd

from ..utils.geo import INDIA_BBOX, PUNJAB_HARYANA_BBOX
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

logger = get_logger("transport.inventory")

#: FRP -> dry-matter combustion-rate coefficient (kg dry matter combusted per
#: second, per MW of FRP).  The Wooster et al. (2005) / GFAS conversion is
#: ~0.368 g s^-1 per W = 0.368 kg s^-1 per MW; rounded to the value the GFAS
#: assimilation uses as a global mean.
FRP_TO_DM_KG_PER_S_PER_MW: float = 0.368

#: Species emission factors EF(species) in **grams of species per kilogram of
#: dry matter burned** (g/kg), Akagi et al. (2011) / Andreae & Merlet families.
#: ``crop_residue`` (paddy/wheat stubble) is the default IGP post-monsoon biome;
#: the others let the forest-belt fires use a more appropriate factor.
EMISSION_FACTORS_G_PER_KG: dict[str, dict[str, float]] = {
    # Agricultural crop-residue burning (paddy/wheat stubble) — the dominant
    # Punjab/Haryana post-monsoon source.
    "crop_residue": {
        "hcho": 2.30,
        "nmvoc": 18.0,
        "co": 92.0,
        "co2": 1430.0,
        "pm25": 6.30,
        "ch4": 5.80,
        "no2": 3.10,
        "nh3": 2.20,
    },
    # Tropical / dry-season forest fire (forest-belt cluster).
    "tropical_forest": {
        "hcho": 1.73,
        "nmvoc": 22.0,
        "co": 93.0,
        "co2": 1640.0,
        "pm25": 9.10,
        "ch4": 5.07,
        "no2": 2.55,
        "nh3": 1.33,
    },
    # Savanna / grassland.
    "savanna": {
        "hcho": 1.20,
        "nmvoc": 8.5,
        "co": 63.0,
        "co2": 1690.0,
        "pm25": 5.20,
        "ch4": 1.94,
        "no2": 3.90,
        "nh3": 0.52,
    },
}

#: Default biome used when none is supplied (IGP stubble).
DEFAULT_BIOME: str = "crop_residue"

#: Species the inventory reports by default (HCHO first — it is the Obj-2 target).
DEFAULT_SPECIES: tuple[str, ...] = ("hcho", "nmvoc", "co", "pm25")

#: Relative scaling of three FRP-based inventory families about the central
#: estimate, used by :func:`inventory_ensemble` to bracket the well-known
#: factor ~2-3x inventory disagreement (blueprint risk #12). FINN is treated as
#: the central/primary line; GFAS runs lower, QFED higher (illustrative, but in
#: the documented direction of their published differences).
INVENTORY_SCALINGS: dict[str, float] = {
    "FINN": 1.00,
    "GFAS": 0.70,
    "QFED": 1.80,
}


@dataclass
class EmissionInventory:
    """Gridded fire-emission inventory for one species.

    Attributes:
        species: Emitted species name (e.g. ``"hcho"``).
        lon_edges: 1-D longitude bin edges (length ``ncol + 1``).
        lat_edges: 1-D latitude bin edges (length ``nrow + 1``).
        emission: ``(nrow, ncol)`` emission field (kg, summed over the period).
        biome: Biome whose emission factor was used.
        inventory: Inventory-family label (``"FINN"``/``"GFAS"``/``"QFED"``/...).
        total_kg: Domain-total emission of the species (kg).
    """

    species: str
    lon_edges: np.ndarray
    lat_edges: np.ndarray
    emission: np.ndarray
    biome: str
    inventory: str
    total_kg: float

    @property
    def lon_centers(self) -> np.ndarray:
        """Longitude bin centres."""
        return 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])

    @property
    def lat_centers(self) -> np.ndarray:
        """Latitude bin centres."""
        return 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])

    def to_frame(self) -> pd.DataFrame:
        """Flatten the field to a tidy ``lon, lat, emission`` DataFrame."""
        lon2d, lat2d = np.meshgrid(self.lon_centers, self.lat_centers)
        return pd.DataFrame(
            {
                "lon": lon2d.ravel(),
                "lat": lat2d.ravel(),
                "emission_kg": self.emission.ravel(),
            }
        )


def _factors_for(biome: str) -> dict[str, float]:
    """Return the emission-factor table for a biome (case-insensitive).

    Args:
        biome: Biome key (e.g. ``"crop_residue"``).

    Returns:
        The ``{species: g/kg}`` factor mapping.

    Raises:
        KeyError: If the biome is unknown.
    """
    key = biome.lower()
    if key not in EMISSION_FACTORS_G_PER_KG:
        raise KeyError(
            f"Unknown biome '{biome}'. Choose from "
            f"{sorted(EMISSION_FACTORS_G_PER_KG)}."
        )
    return EMISSION_FACTORS_G_PER_KG[key]


def emission_coefficient(
    species: str,
    biome: str = DEFAULT_BIOME,
    *,
    seconds_per_detection: float = 86_400.0,
) -> float:
    """Mass of a species emitted per MW of FRP over one detection interval.

    Combines the FRP->dry-matter rate with the species emission factor:

        EF_kg_per_MW = FRP_TO_DM[kg/s/MW] * seconds * EF[g/kg] / 1000

    The ``seconds_per_detection`` term turns the per-second combustion rate into
    a per-detection mass; a daily FIRMS detection represents ~1 day of burning,
    so the default is 86 400 s.

    Args:
        species: Species name (must be in the biome's factor table).
        biome: Vegetation type (default cropland residue).
        seconds_per_detection: Burn duration each detection stands for (s).

    Returns:
        Kilograms of ``species`` emitted per MW of FRP per detection interval.

    Raises:
        KeyError: If the species is not in the biome's emission-factor table.
    """
    factors = _factors_for(biome)
    if species not in factors:
        raise KeyError(
            f"No emission factor for species '{species}' in biome '{biome}'."
        )
    ef_g_per_kg = factors[species]
    dm_kg_per_mw = FRP_TO_DM_KG_PER_S_PER_MW * seconds_per_detection
    return dm_kg_per_mw * ef_g_per_kg / 1000.0


def emission_rates(
    fire_df: "pd.DataFrame | gpd.GeoDataFrame",
    *,
    species: "tuple[str, ...] | list[str]" = DEFAULT_SPECIES,
    biome: str = DEFAULT_BIOME,
    frp_col: str = "frp",
    seconds_per_detection: float = 86_400.0,
    scaling: float = 1.0,
) -> pd.DataFrame:
    """Per-detection species emissions from a fire table.

    Multiplies each detection's FRP by the per-MW emission coefficient of every
    requested species, returning a copy of the fire table with one
    ``emis_<species>_kg`` column per species.

    Args:
        fire_df: FIRMS-surrogate fire detections (DEV_CONTRACT §6.3) with an FRP
            column.
        species: Species to compute (default :data:`DEFAULT_SPECIES`).
        biome: Vegetation type whose emission factors apply.
        frp_col: Name of the FRP column (MW).
        seconds_per_detection: Burn duration each detection represents (s).
        scaling: Multiplicative inventory-family scaling (see
            :data:`INVENTORY_SCALINGS`).

    Returns:
        A ``DataFrame`` (geometry dropped) with the original columns plus an
        ``emis_<species>_kg`` column for each species. Empty input -> empty frame
        carrying the expected columns.

    Raises:
        KeyError: If ``frp_col`` is absent or a species has no emission factor.
    """
    df = pd.DataFrame(fire_df).copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")
    out_cols = [f"emis_{s}_kg" for s in species]
    if df.empty:
        for c in out_cols:
            df[c] = pd.Series(dtype=float)
        return df
    if frp_col not in df.columns:
        raise KeyError(f"Fire table has no '{frp_col}' column.")

    frp = df[frp_col].to_numpy(dtype=float)
    for s in species:
        coeff = emission_coefficient(
            s, biome, seconds_per_detection=seconds_per_detection
        )
        df[f"emis_{s}_kg"] = frp * coeff * float(scaling)

    totals = {s: float(df[f"emis_{s}_kg"].sum()) for s in species}
    logger.info(
        "emission rates for %d detections (biome=%s, scaling=%.2f); totals(kg)=%s",
        len(df),
        biome,
        scaling,
        {k: round(v, 1) for k, v in totals.items()},
    )
    return df


def _grid_edges(
    bbox: tuple[float, float, float, float], resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build regular lon/lat bin edges over a bbox.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)``.
        resolution: Cell size in degrees.

    Returns:
        ``(lon_edges, lat_edges)``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_edges = np.arange(min_lon, max_lon + resolution, resolution)
    lat_edges = np.arange(min_lat, max_lat + resolution, resolution)
    return lon_edges, lat_edges


def grid_emissions(
    fire_df: "pd.DataFrame | gpd.GeoDataFrame",
    species: str = "hcho",
    *,
    bbox: tuple[float, float, float, float] = INDIA_BBOX,
    resolution: float = 0.25,
    biome: str = DEFAULT_BIOME,
    inventory: str = "FINN",
    frp_col: str = "frp",
    lat_col: str = "lat",
    lon_col: str = "lon",
    seconds_per_detection: float = 86_400.0,
) -> EmissionInventory:
    """Bin per-detection species emissions onto a regular lon/lat grid.

    Args:
        fire_df: FIRMS-surrogate fire detections with ``lat``/``lon``/FRP.
        species: Species to grid (default ``"hcho"``).
        bbox: Grid extent ``(min_lon, min_lat, max_lon, max_lat)``.
        resolution: Cell size in degrees (default 0.25, matching the cube).
        biome: Vegetation type whose emission factors apply.
        inventory: Inventory-family label; scales emissions by
            :data:`INVENTORY_SCALINGS` when recognised (else 1.0).
        frp_col: FRP column name.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        seconds_per_detection: Burn duration each detection represents (s).

    Returns:
        An :class:`EmissionInventory` for ``species`` (kg per cell, summed over
        all detections in the table).
    """
    scaling = INVENTORY_SCALINGS.get(inventory, 1.0)
    rates = emission_rates(
        fire_df,
        species=(species,),
        biome=biome,
        frp_col=frp_col,
        seconds_per_detection=seconds_per_detection,
        scaling=scaling,
    )
    lon_edges, lat_edges = _grid_edges(bbox, resolution)
    nrow, ncol = lat_edges.size - 1, lon_edges.size - 1
    field = np.zeros((nrow, ncol), dtype=float)

    if not rates.empty:
        emis = rates[f"emis_{species}_kg"].to_numpy(dtype=float)
        lats = rates[lat_col].to_numpy(dtype=float)
        lons = rates[lon_col].to_numpy(dtype=float)
        field, _, _ = np.histogram2d(
            lats, lons, bins=[lat_edges, lon_edges], weights=emis
        )

    total = float(field.sum())
    logger.info(
        "gridded %s emissions: %.1f kg total on %dx%d grid (%s, res=%.2f deg)",
        species,
        total,
        nrow,
        ncol,
        inventory,
        resolution,
    )
    return EmissionInventory(
        species=species,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        emission=field,
        biome=biome,
        inventory=inventory,
        total_kg=total,
    )


def inventory_ensemble(
    fire_df: "pd.DataFrame | gpd.GeoDataFrame",
    species: str = "hcho",
    *,
    biome: str = DEFAULT_BIOME,
    scalings: Optional[Mapping[str, float]] = None,
    frp_col: str = "frp",
    seconds_per_detection: float = 86_400.0,
) -> dict:
    """Bracket the inventory uncertainty across several FRP-based families.

    Runs the domain-total emission of ``species`` under each inventory-family
    scaling (FINN central, GFAS lower, QFED higher by default) so the result can
    be reported with an honest spread rather than a single point — the blueprint
    requires the inventory line to *bracket*, not assert (risk #12).

    Args:
        fire_df: FIRMS-surrogate fire detections with an FRP column.
        species: Species to total (default ``"hcho"``).
        biome: Vegetation type whose emission factors apply.
        scalings: ``{inventory: scale}`` map; defaults to
            :data:`INVENTORY_SCALINGS`.
        frp_col: FRP column name.
        seconds_per_detection: Burn duration each detection represents (s).

    Returns:
        Dict with ``species``, ``biome``, ``totals_kg`` (per-inventory total),
        ``central_kg`` (the FINN/central total), ``low_kg``, ``high_kg`` and
        ``spread_ratio`` (``high/low``).
    """
    scalings = dict(scalings or INVENTORY_SCALINGS)
    totals: dict[str, float] = {}
    for inv, scale in scalings.items():
        rates = emission_rates(
            fire_df,
            species=(species,),
            biome=biome,
            frp_col=frp_col,
            seconds_per_detection=seconds_per_detection,
            scaling=scale,
        )
        totals[inv] = (
            float(rates[f"emis_{species}_kg"].sum()) if not rates.empty else 0.0
        )

    vals = np.array(list(totals.values()), dtype=float)
    low = float(vals.min()) if vals.size else 0.0
    high = float(vals.max()) if vals.size else 0.0
    central = float(totals.get("FINN", float(np.median(vals)) if vals.size else 0.0))
    spread = float(high / low) if low > 0 else float("nan")

    logger.info(
        "%s inventory ensemble: central(FINN)=%.1f kg, range=[%.1f, %.1f], "
        "spread=%.2fx",
        species,
        central,
        low,
        high,
        spread,
    )
    return {
        "species": species,
        "biome": biome,
        "totals_kg": totals,
        "central_kg": central,
        "low_kg": low,
        "high_kg": high,
        "spread_ratio": spread,
    }


def run(cfg) -> dict:
    """Hydra entry point: build the fire HCHO emission inventory on synthetic data.

    Loads the fire table, grids the HCHO emissions over the source region and
    over India, and computes the multi-inventory ensemble bracket so the
    inventory line can be cross-checked against the statistical and Lagrangian
    attribution.

    Args:
        cfg: Composed Hydra config. Uses ``cfg.paths.data_processed`` for inputs.

    Returns:
        A results dict with keys ``species``, ``biome``, ``ensemble``,
        ``source_total_kg``, ``india_total_kg``, ``peak_cell`` and ``n_fires``.
    """
    from pathlib import Path

    from ..utils.io import load_parquet

    processed = Path(cfg.paths.data_processed)
    fires = load_parquet(processed / "fires.parquet")

    biome = str(getattr(getattr(cfg, "transport", object()), "biome", DEFAULT_BIOME))
    species = "hcho"

    india = grid_emissions(
        fires, species, bbox=INDIA_BBOX, resolution=0.25, biome=biome
    )
    source = grid_emissions(
        fires, species, bbox=PUNJAB_HARYANA_BBOX, resolution=0.1, biome=biome
    )
    ensemble = inventory_ensemble(fires, species, biome=biome)

    # Peak emission cell (where the inventory says HCHO is produced most).
    peak_cell: dict = {"lon": float("nan"), "lat": float("nan"), "value": float("nan")}
    if np.any(india.emission > 0):
        idx = np.unravel_index(int(np.argmax(india.emission)), india.emission.shape)
        peak_cell = {
            "lon": float(india.lon_centers[idx[1]]),
            "lat": float(india.lat_centers[idx[0]]),
            "value": float(india.emission[idx]),
        }

    return {
        "species": species,
        "biome": biome,
        "ensemble": ensemble,
        "source_total_kg": source.total_kg,
        "india_total_kg": india.total_kg,
        "peak_cell": peak_cell,
        "n_fires": int(len(fires)),
    }


__all__ = [
    "DEFAULT_BIOME",
    "DEFAULT_SPECIES",
    "EMISSION_FACTORS_G_PER_KG",
    "FRP_TO_DM_KG_PER_S_PER_MW",
    "INVENTORY_SCALINGS",
    "EmissionInventory",
    "emission_coefficient",
    "emission_rates",
    "grid_emissions",
    "inventory_ensemble",
    "run",
]
