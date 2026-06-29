# Data Sources Catalog — BAH 2026 PS3 (Surface AQI & HCHO Hotspots over India)

This document catalogs **85 global satellite, reanalysis, and ground-based datasets** evaluated for ISRO/BAH 2026 Problem Statement 3 — deriving surface air-quality indices and identifying formaldehyde (HCHO) hotspots over India from space.
The sources span polar-orbiting and geostationary trace-gas/aerosol spectrometers, fire and burned-area products, atmospheric-composition and meteorological reanalyses, and surface monitoring networks. They are organized to support **multi-source fusion** (combining complementary instruments into one feature stack), **cross-verification** (independent sensors corroborating the same signal), and **gap-filling** (covering cloud, temporal, spatial-resolution, and missing-pollutant gaps in any single feed).

> Scope: Objective 1 is surface AQI mapping (PM2.5, NO2, SO2, CO, O3) via CNN/LSTM fusion of satellite columns + meteorology + ground labels; Objective 2 is HCHO hotspot detection and its correlation with biomass-burning fire activity over the Indo-Gangetic Plain and forest-fire zones.

## Contents

- [Summary Index](#summary-index)
- [Sentinel / Copernicus (Sentinel-5P TROPOMI, Sentinel-3, Sentinel-2)](#sentinel-copernicus-sentinel-5p-tropomi-sentinel-3-sentinel-2)
- [Geostationary & Indian Satellites (INSAT-3D/3DR/3DS, Himawari, FY-4, GK-2A, GEMS)](#geostationary-indian-satellites-insat-3d3dr3ds-himawari-fy-4-gk-2a-gems)
- [MODIS Suite (Terra / Aqua / Combined)](#modis-suite-terra-aqua-combined)
- [VIIRS Suite (Suomi-NPP, NOAA-20/21)](#viirs-suite-suomi-npp-noaa-2021)
- [Polar-Orbiting UV-Vis Spectrometers (OMI, GOME-2, SCIAMACHY, GOME-1)](#polar-orbiting-uv-vis-spectrometers-omi-gome-2-sciamachy-gome-1)
- [Atmospheric IR / Multispectral Sounders (IASI, AIRS, CrIS, MOPITT, TES)](#atmospheric-ir-multispectral-sounders-iasi-airs-cris-mopitt-tes)
- [Aerosol-Profile & Wind Satellites (CALIPSO, MISR, EarthCARE, Aeolus, GCOM-C, PARASOL)](#aerosol-profile-wind-satellites-calipso-misr-earthcare-aeolus-gcom-c-parasol)
- [Atmospheric-Composition Reanalysis & Forecasts (CAMS, MERRA-2, GEOS-CF, NAAPS, SILAM)](#atmospheric-composition-reanalysis-forecasts-cams-merra-2-geos-cf-naaps-silam)
- [Meteorological Reanalysis (ERA5, IMDAA, MERRA-2, GFS/NCEP)](#meteorological-reanalysis-era5-imdaa-merra-2-gfsncep)
- [Ground-Truth Networks (CPCB, AERONET, OpenAQ, AirNow, SAFAR, PurpleAir)](#ground-truth-networks-cpcb-aeronet-openaq-airnow-safar-purpleair)
- [Fire Detection & Biomass-Burning Emission Inventories](#fire-detection-biomass-burning-emission-inventories)
- [Additional Augmenting Datasets (Terrain, Land-Use, Population, Precipitation, TEMPO/OMPS)](#additional-augmenting-datasets-terrain-land-use-population-precipitation-tempoomps)
- [How These Sources Fill Each Other's Gaps](#how-these-sources-fill-each-others-gaps)

## Summary Index

All 85 datasets at a glance. See the per-family sections below for full provider, access, latency, and role detail.

| # | Dataset | Instrument / Platform | Key Products | Spatial Res | Temporal | Primary Role |
|---|---------|-----------------------|--------------|-------------|----------|--------------|
| 1 | Sentinel-5P TROPOMI HCHO (Formaldehyde, tropospheric column) | TROPOMI (UV-VIS-NIR-SWIR push-broom spectrometer) | L2__HCHO___ orbit NetCDF; formaldehyde_tropospheric_vertical_column (mol/m^2); … | 5.5x3.5 km nadir (since 6 Aug 2019; was 7x3.… | Daily, ~13:30 LT equator crossing; India ~1… | Primary HCHO-hotspot variable (Obj-2) and VOC/secondary-pollutant fea… |
| 2 | Sentinel-5P TROPOMI NO2 (tropospheric + total + stratospheric) | TROPOMI | tropospheric_NO2_column_number_density (mol/m^2); NO2_column_number_density (total); … | 5.5x3.5 km nadir; GEE L3 ~0.01 deg. | Daily, ~13:30 LT. | Top-tier AQI CNN/LSTM feature; column-to-surface via reanalysis PBL h… |
| 3 | Sentinel-5P TROPOMI SO2 (total column) | TROPOMI | SO2_column_number_density (mol/m^2; PBL/1km/7km/15km a-priori); SO2_slant_column; … | 5.5x3.5 km; GEE L3 ~0.01 deg. | Daily. | SO2 AQI sub-index feature; flags point-source-dominated grid cells. |
| 4 | Sentinel-5P TROPOMI CO + CH4 (SWIR combustion/GHG tracers) | TROPOMI SWIR (2.3 um) | CO_column_number_density (mol/m^2); H2O_column_number_density; … | ~7x5.5 km (SWIR); GEE L3 ~0.01 deg. | Daily (CH4 cloud/albedo limited). | CO = robust combustion feature for AQI CO sub-index + fire-HCHO co-tr… |
| 5 | Sentinel-5P TROPOMI O3 (total column + tropospheric TCL) | TROPOMI | O3_column_number_density (total, mol/m^2); O3_effective_temperature; … | Total ~5.5x3.5 km; O3_TCL ~0.5 deg. | Daily (total); TCL 3-day/monthly. | Auxiliary; FNR (HCHO/NO2) feature flags VOC- vs NOx-limited O3 chemis… |
| 6 | Sentinel-5P AER_AI (UV/Absorbing Aerosol Index) + CLOUD | TROPOMI UV (354/388 nm; also 340/380) | absorbing_aerosol_index (unitless); cloud_fraction; … | 5.5x3.5 km; GEE L3 ~0.01 deg. | Daily. | AER_AI = smoke-presence/episode feature; CLOUD = quality/weighting la… |
| 7 | Sentinel-5P AER_LH (Aerosol Layer Height) | TROPOMI O2-A band (760 nm) | aerosol_height (m); aerosol_pressure (Pa); … | ~5.5x3.5 km; GEE L3 ~0.01 deg. | Daily (cloud-free, elevated-aerosol scenes). | Vertical context: tells AQI model whether AAI/AOD aerosol is in PBL (… |
| 8 | Sentinel-3 SLSTR Active Fire & FRP + SYN AOD + SLSTR LST | SLSTR (MWIR 3.74 um S7, TIR, dual-view) + OLCI for SYN… | FRP (MW/pixel); fire detection flag/confidence; … | FRP & LST 1 km; SYN AOD ~4.5 km; OLCI 300 m. | S3A+S3B ~daily India; ~10:00 & ~22:00 LT (co… | FRP = fire-energy covariate for fire-HCHO correlation (Obj-2); AOD =… |
| 9 | Sentinel-2 MSI High-res Fire/Burn-scar (NBR) & Land Cover | MSI (13 bands, 10/20/60 m) | NBR=(B8-B12)/(B8+B12); dNBR burn severity; … | 10 m (B8), 20 m SWIR (B11/B12); burn-scar/fi… | ~5-day revisit (S2A+S2B+S2C); India good. | High-res ground-truth of burned area & fuel/land cover validating coa… |
| 10 | INSAT-3D / 3DR Imager Aerosol Optical Depth (AOD) | 6-band Imager (VIS 0.55, SWIR 1.6, MIR, WV, TIR1/2) | L2 AOD over land+ocean (550nm); ancillary OLR, SST, cloud | VIS/SWIR 1km; MIR/TIR 4km; WV 8km; AOD gridd… | Imager 30 min single sat; 15 min for 3D+3DR;… | PRIMARY mandated AOD input for PS3 Objective-1 surface-AQI maps |
| 11 | GEMS (Geostationary Environment Monitoring Spectrometer) on GK-2B | UV-Vis hyperspectral imaging spectrometer 300-500nm, 0… | L2 NO2 (tropo+total); L2 HCHO column; … | 3.5x8km (NO2/O3/HCHO/AOD) at Seoul; SO2 7x16… | Hourly, ~8 daytime scans/day | GAME-CHANGER: HOURLY column NO2/SO2/HCHO/O3/AOD over India — same spe… |
| 12 | Himawari-8 / Himawari-9 AHI — AOD & Active Fire (FRP) | Advanced Himawari Imager (AHI), 16 bands, 0.5-2km | L2 Dark Target AOD 10km; JAXA P-Tree hourly AOD (ARP); … | AHI 0.5-2km; AOD 10km at nadir; FRP ~2km | Full disk every 10 min; daytime AOD | Secondary 10-min AOD + active-fire timing for E/NE India and Bay-of-B… |
| 13 | FY-4A / FY-4B AGRI — AOD & Fire/Hotspot (FHS) | Advanced Geostationary Radiation Imager (AGRI), 15 ch… | L2 AOD (multichannel MC algorithm); Fire/Hotspot product FHS; … | AGRI 0.5-4km; AOD ~4km; FHS 2km | Full disk every 15 min; daytime AOD | Supplementary 15-min GEO AOD with strong India viewing geometry (105E… |
| 14 | GK-2A (GEO-KOMPSAT-2A) AMI — Aerosol Detection / AOD | Advanced Meteorological Imager (AMI), 16 ch (AHI twin) | Aerosol Detection (dust/haze/ash; 4 day, 3 night types); Visible AOD; … | AMI 0.5-2km; AOD ~ few km | Full disk every 10 min | Aerosol-type discrimination (dust vs haze) and day+night DAOD to flag… |
| 15 | INSAT-3DS Imager — Active Fire & AOD | Improved 6-band Imager + 19-ch Sounder | Active fire detection product; Imager AOD (3D/3DR continuity); … | Imager 1-4km; fire ~4km | Imager ~15-30 min; daytime/24h fire | Native Indian GEO active-fire timing for Objective-2 biomass-burning… |
| 16 | MCD19A2 — MAIAC Land Aerosol Optical Depth (Terra+Aqua combined) | MODIS Terra + Aqua (MAIAC algorithm) | Optical_Depth_047 (AOD @ 0.47um); Optical_Depth_055 (AOD @ 0.55um, primary for PM2.5); … | 1 km (native MAIAC grid) | Daily; one image per Terra and per Aqua over… | Objective-1 backbone: highest-resolution (1km) satellite AOD predicto… |
| 17 | MOD04/MYD04 L2 — Dark Target & Deep Blue AOD | MODIS Terra (MOD04_L2/MOD04_3K) / Aqua (MYD04_L2/MYD04… | Optical_Depth_Land_And_Ocean (DT, 10km); Deep_Blue_Aerosol_Optical_Depth_550_Land (DB); … | 10 km (MOD04_L2) and 3 km (MOD04_3K) at nadir | Twice daily (Terra ~10:30, Aqua ~13:30 LT) | Secondary/validation AOD; Combined DT+DB and Angstrom exponent add co… |
| 18 | MOD14A1/MYD14A1 — Thermal Anomalies & Fire Daily 1km | MODIS Terra (MOD14A1) / Aqua (MYD14A1) | FireMask (class 0-9: 7=low,8=nominal,9=high confidence fire); MaxFRP (Fire Radiative Power); … | 1 km | Daily composite of Terra (10:30/22:30) and A… | Objective-2: gridded 1km fire mask + FRP as raster predictor for HCHO… |
| 19 | FIRMS rasterized fire + MCD14DL/MCD14ML active-fire locations | MODIS Terra+Aqua (MOD14/MYD14 algorithm) | T21 (brightness temp 4um, K); MaxFRP (raster); … | 1 km (centroid of flagged pixel) | Per-overpass NRT (4x/day); MCD14ML monthly a… | PS3-named fire input |
| 20 | MCD64A1 — Burned Area Monthly 500m | MODIS Terra+Aqua (500m SR + 1km active fire) | BurnDate (day-of-year of burn, 1-366); Uncertainty; … | 500 m | Monthly | Objective-2 context: defines spatial extent/timing of biomass-burning… |
| 21 | MOD11A1/MYD11A1 — Land Surface Temperature & Emissivity Daily 1km | MODIS Terra (MOD11A1) / Aqua (MYD11A1) | LST_Day_1km; LST_Night_1km; … | 1 km (also MOD11A2 8-day; MOD11B 6km) | Daily, day+night (Terra 10:30/22:30, Aqua 13… | Auxiliary meteorological/surface predictor in AQI CNN/LSTM models (su… |
| 22 | MCD12Q1 — Land Cover Type Yearly 500m | MODIS Terra+Aqua | LC_Type1 (IGBP 17-class); LC_Type2 (UMD); … | 500 m | Yearly | Static land-use predictor for AQI ML (urban vs rural vs cropland emis… |
| 23 | MOD13Q1/MYD13Q1 — Vegetation Indices (NDVI/EVI) 250m 16-day | MODIS Terra (MOD13Q1) / Aqua (MYD13Q1); MOD13A1 500m,… | NDVI; EVI; … | 250 m (Q1); 500m (A1); 1km (A2) | 16-day composite (Terra+Aqua offset to ~8-da… | Vegetation/phenology predictor in AQI model; NDVI drop after harvest… |
| 24 | VIIRS 375m Active Fire (VNP14IMG / VJ114IMG / VJ214IMG) | VIIRS I-bands (I1-I5, 375m) on Suomi-NPP, NOAA-20, NOA… | VNP14IMG (S-NPP standard); VJ114IMG (NOAA-20); … | 375 m (I-band); detection footprint ~0.14 km… | ~Daily per platform; ~3 day + 3 night VIIRS… | Primary fire-activity input for Objective 2: build daily India/IGP fi… |
| 25 | FIRMS multi-platform VIIRS+MODIS fire fusion | VIIRS (S-NPP/NOAA-20/NOAA-21, 375m) + MODIS (Terra/Aqu… | combined active-fire archive & NRT; MCD14DL (MODIS); … | 375 m (VIIRS) + 1 km (MODIS), merged point d… | 5 platforms => multiple day/night passes; de… | Single fused fire-count layer maximizing detection completeness for t… |
| 26 | VIIRS Deep Blue Aerosol AOD (AERDB_L2) | VIIRS M-bands on S-NPP (AERDB_L2_VIIRS_SNPP) & NOAA-20 | AOT 550nm (Deep Blue land + SOAR ocean); Angstrom exponent; … | 6 km at nadir (increases off-nadir) | Daily (per platform, ~13:30 LT) | Independent polar AOD to feed/validate the AOD->surface-PM2.5 branch… |
| 27 | VIIRS Dark Target Aerosol AOD (AERDT_L2) | VIIRS M-bands, S-NPP & NOAA-20 | AOT 550nm (Dark Target land+ocean, MODIS-heritage algorithm); fine-mode fraction (ocean); … | 6 km at nadir | Daily per platform | Continuity with MODIS Dark Target AOD time series for the surface-AQI… |
| 28 | VNP46A2 Black Marble Nighttime Lights (DNB) | VIIRS Day-Night Band (DNB), S-NPP (+NOAA-20 in VNP46) | Gap_Filled_DNB_BRDF-Corrected_NTL; Latest_High_Quality_Retrieval; … | 500 m | Daily (A2); monthly/annual composites (A3/A4) | Combustion/anthropogenic-emission and urbanization proxy covariate fo… |
| 29 | VIIRS Nightfire (VNF) | VIIRS M-bands (M7-M13 incl | per-source temperature (K); source size; … | ~750 m M-band; sub-pixel source via Planck-c… | Nightly global | Verification layer: independent sub-pixel combustion temperature/FRP… |
| 30 | VNP14A1 Thermal Anomalies/Fire Daily 1km (GEE) | VIIRS M-bands, Suomi-NPP | FireMask; MaxFRP; … | 1 km (SIN grid) | Daily L3 | Convenient gridded GEE fire layer for rapid in-platform fire-HCHO ove… |
| 31 | Aura OMI — Formaldehyde (OMHCHO L2 / OMHCHOd L3) | OMI (Ozone Monitoring Instrument), Aura platform, UV-V… | OMHCHO L2 (1-orbit swath HCHO vertical column); OMHCHOd L3 daily 0.1deg gridded; … | 13 x 24 km nadir (L2); 0.1deg L3 grid | Daily, ~13:30 LT ascending (sun-synchronous)… | Primary multi-decade HCHO baseline overlapping TROPOMI (both ~13:30 L… |
| 32 | Aura OMI — NO2 (OMNO2 / OMNO2d) | OMI / Aura, visible 405-465 nm DOAS | OMNO2 L2 total & tropospheric NO2 column; OMNO2d L3 daily 0.25deg gridded tropospheric NO2; … | 13 x 24 km nadir; 0.25deg L3 | Daily ~13:30 LT | Long NO2 record to contextualize TROPOMI NO2 and as auxiliary AQI pre… |
| 33 | Aura OMI — SO2 (OMSO2 L2 / OMSO2e L3) | OMI / Aura, UV 310.5-340 nm (PCA/PBL & multiple a-prio… | OMSO2 L2 total column (PBL, TRL, TRM, TRU, STL anchored); OMSO2e L3 daily 0.25deg best-pixel; … | 13 x 24 km; 0.25deg L3 | Daily ~13:30 LT | Historic SO2 record to contextualize TROPOMI SO2 over Indian coal/pow… |
| 34 | Aura OMI — Aerosol Index & Absorbing Aerosol (OMAERUV / OMAERUVd) | OMI / Aura, near-UV 354/388 nm two-channel algorithm | UV Aerosol Index (UVAI); Absorbing Aerosol Optical Depth (AAOD); … | 13 x 24 km; 1.0deg L3 | Daily ~13:30 LT | UV Aerosol Index flags absorbing smoke/dust — key co-tracer to confir… |
| 35 | GOME-2 on MetOp-A/B/C (HCHO/NO2/SO2) — AC SAF / DLR / BIRA-IASB | GOME-2 UV-Vis grating spectrometer 240-790 nm, MetOp-A… | Tropospheric HCHO VCD (BIRA-IASB DOAS); Tropospheric & total NO2 (TEMIS/DLR); … | 80 x 40 km (A/B pre-2013); 40 x 40 km (GOME-… | Daily, ~09:30 LT descending (morning overpas… | Extends record to 2007 and adds INDEPENDENT morning (09:30 LT) HCHO -… |
| 36 | QA4ECV harmonized HCHO & NO2 (GOME/SCIAMACHY/GOME-2/OMI) | Multi-sensor: GOME-1(ERS-2), SCIAMACHY(Envisat), GOME-… | Harmonized tropospheric HCHO ECV; Harmonized tropospheric NO2 ECV; … | Native per sensor (40-13 km); L3 gridded | Monthly climatology + per-overpass | Single consistently-retrieved 22-yr multi-sensor record — THE source… |
| 37 | SCIAMACHY on Envisat — HCHO (BIRA-IASB) | SCIAMACHY UV-Vis-NIR 240-2380 nm nadir/limb, DOAS HCHO | Tropospheric HCHO VCD; NO2, SO2 total columns; … | 30 x 60 km nadir | ~6-day global; ~10:00 LT descending | Bridges GOME-1 (pre-2003) to OMI/GOME-2 era; part of the continuous ~… |
| 38 | GOME-1 on ERS-2 — HCHO/NO2 (earliest record) | GOME (Global Ozone Monitoring Experiment) UV-Vis 240-7… | Tropospheric HCHO VCD; Tropospheric NO2; … | 40 x 320 km nadir (coarsest) | ~3-day global; ~10:30 LT | Anchors the earliest (~1996) end of the multi-decade HCHO/NO2 climato… |
| 39 | IASI (Infrared Atmospheric Sounding Interferometer) on MetOp-A/-B/-C | Nadir Fourier-transform IR spectrometer, 645-2760 cm-1… | NH3 total column (ANNI-NH3 v4, neural-network, with averaging kernels); CO total column + coarse profile (FORLI-CO / AC SAF operational); … | 12 km nadir footprint (circular, 4-pixel mat… | Twice daily per platform (~09:30 & 21:30 LT… | Independent twice-daily NH3 & CO total columns over India |
| 40 | AIRS (Atmospheric Infrared Sounder) on Aqua | Grating IR spectrometer 3.7-15.4 um (2378 channels) +… | AIRS3STD/AIRS3STM L3 daily/monthly: T & H2O (RH) profiles (24 levels), O3, CO, CH4 total+profile, surface T; AIRX2RET / AIRS2RET L2 retrievals; … | L2: 45 km nadir; L3 gridded: 1° x 1° | Twice daily (01:30 & 13:30 LT); global daily… | Primary source of free-tropospheric T & H2O (RH) profiles to drive th… |
| 41 | CrIS (Cross-track Infrared Sounder) on SNPP & NOAA-20 (JPSS-1) | FTS IR sounder (LWIR/MWIR/SWIR) + ATMS microwave | CLIMCAPS V2 L2/L3 (SNDRSNIML2/.. , SNDRJ1IML2): T, H2O, O3, CO, CH4, CO2, SO2, N2O, HNO3 profiles; ESSPA-NH3 V1 (SNDRSNIL2ESPNH3): NH3 profile/column; … | L2: ~14 km FOV / FOR; L3: 0.5°-1° | Twice daily per platform (~01:30 & 13:30 LT)… | Operational continuity & ensemble for IASI/AIRS: extra daily CO and N… |
| 42 | MOPITT (Measurements Of Pollution In The Troposphere) on Terra | Gas-correlation radiometer; TIR (4.7 um) + NIR (2.3 um… | MOP02J V9 L2 multispectral (TIR+NIR) CO profiles + total column; MOP02T V9 (TIR-only); … | 22 km x 22 km nadir; L3 gridded 1° x 1° | Near-daily global (~3 days full coverage); 1… | Authoritative long-term CO total-column reference |
| 43 | TES (Tropospheric Emission Spectrometer) on Aura (legacy) | High-spectral-resolution IR FTS (limb/nadir) | O3 profiles; CO profiles; … | ~5 x 8 km footprint, sparse targeted sampling | 16-day global survey (limited duty cycle); n… | Legacy high-spectral-res O3/CO/NH3 reference: useful for historical v… |
| 44 | CALIPSO CALIOP Lidar L2 Aerosol Profile / Layer (V4-51) | CALIOP (dual-wavelength 532+1064nm polarization-sensit… | L2 05kmAPro (extinction/backscatter profiles); L2 05kmALay (layer top/base, AOD); … | 333m horiz native; L2 at 5km; 30-60m vertical | 16-day repeat, nadir curtain only (sparse) | Provides aerosol scale height H and vertical extinction shape to conv… |
| 45 | MISR L2 Aerosol + MINX Plume Height | MISR (9 along-track cameras 0-70deg, 4 bands, 275m) | L2 Aerosol AOD (558nm); Aerosol type / size & non-spherical fraction; … | 275m imaging; L2 aerosol 4.4km; plume height… | Terra ~9-day repeat (2-4d mid-lat), ~10:30 l… | Aerosol-type AOD cross-check for surface AQI; multi-angle constrains… |
| 46 | EarthCARE ATLID L2a (A-PRO: A-AER/A-EBD/A-TC) | ATLID (355nm high-spectral-resolution lidar, polarizat… | A-EBD: extinction, backscatter, depolarization profiles; A-AER large-scale aerosol; … | ~285m vertical (to 100m near surface); ~1-10… | Sun-synch ~10:30; nadir curtain (sparse) | Refreshes/validates vertical extinction-profile & scale-height climat… |
| 47 | Aeolus ALADIN L2B Wind Profiles (+ Aeolus-2) | ALADIN (355nm Doppler wind lidar) | L2B HLOS wind (Rayleigh clear-air); L2B HLOS wind (Mie cloud/aerosol); … | ~87km horiz integration; 0.25-2km vertical b… | Sun-synch ~7-day repeat; single LOS | Independent verification of reanalysis wind fields driving transport… |
| 48 | GCOM-C / SGLI L2-L3 Aerosol (AROT/ARAE) | SGLI (250m-1km multispectral + polarization 380/670/86… | AROT AOD 500nm land+ocean; ARAE Angstrom Exponent (380/500); … | 250m-1km (aerosol L2 1km) | ~2-day global, ~10:30 local | Supplementary daily AOD input + cross-sensor consistency for surface… |
| 49 | PARASOL / POLDER-3 (GRASP) Polarized Aerosol | POLDER-3 (multi-angle, polarized 490/670/865nm, 9 band… | Fine-mode & coarse-mode AOD; Aerosol type, SSA, refractive index (GRASP); … | ~6km | Multi-angle per overpass; 2004-2013 archive | Training data / fine-mode-fraction priors for aerosol-type-aware AOD-… |
| 50 | CAMS Global Reanalysis EAC4 | Reanalysis (IFS-COMPO, 4D-Var) assimilating MODIS/PMAp… | PM2.5 (particulate_matter_2.5um); PM10 (particulate_matter_10um); … | ~80 km native (0.75deg), provided 0.5deg/0.7… | 3-hourly (analysis 00/12 UTC, +3..+9 forecas… | Primary gap-free speciated 3D prior: supplies cloud-free NO2/SO2/CO/O… |
| 51 | CAMS Global Atmospheric Composition Forecasts (NRT) | IFS-COMPO forecast; initial conditions assimilate TROP… | NO2; SO2; … | 0.4deg x 0.4deg (~44 km) | Hourly to lead time +120 h; two cycles/day (… | NRT gap-free prior for operational daily AQI when EAC4 latency too lo… |
| 52 | MERRA-2 Aerosol Diagnostics (M2T1NXAER / tavg1_2d_aer_Nx) | GEOS-5.12.4 reanalysis with GOCART aerosols; assimilat… | AOD 550nm total (TOTEXTTAU) + scattering; Black carbon col/surf mass (BCSMASS); … | 0.5deg lat x 0.625deg lon | Hourly time-averaged (centers 00:30..23:30 U… | Hourly gap-free speciated aerosol prior: derive PM2.5 = 1.375*SO4+1.6… |
| 53 | NASA GEOS-CF (Composition Forecast v1) | GEOS + GEOS-Chem (full tropospheric chemistry); meteor… | Surface NO2; O3; … | 0.25deg x 0.25deg (~25 km) | Hourly (and 15-min inst.); daily 5-day forec… | Highest-res gap-free surface-AQI prior: hourly 0.25deg NO2/O3/PM2.5/C… |
| 54 | CAMS GFAS (Global Fire Assimilation System, biomass-burning emissions) | MODIS Terra+Aqua Fire Radiative Power -> dry matter ->… | FRP; CO emission; … | 0.1deg x 0.1deg | Daily averaged | Emission-based fire constraint for objective-2: regress/correlate GFA… |
| 55 | NAAPS (Navy Aerosol Analysis and Prediction System) | Aerosol transport model assimilating MODIS + VIIRS AOD… | Total AOD; sulfate AOD/conc; … | ~1/3deg (reanalysis); 0.25deg forecast | 6-hourly | Independent secondary aerosol prior for ensemble spread and cross-ver… |
| 56 | SILAM (System for Integrated modeLling of Atmospheric coMposition) | Chemistry-transport + fire (IS4FIRES FRP) model; CAMS… | NO2; O3; … | ~0.1-0.5deg (global ~0.5deg) | Hourly forecast | Independent full-chemistry prior (incl |
| 57 | ERA5 single-levels (hourly) — full variable set via CDS | Global atmospheric reanalysis (IFS Cy41r2, 4D-Var) | reanalysis-era5-single-levels; boundary_layer_height (blh); … | 0.25 deg (~31 km native) | Hourly, 1940-present | Primary global met predictor for AQI ML: BLH (dilution), 10m wind spd… |
| 58 | ERA5 / ERA5-Land Hourly on Google Earth Engine | ERA5 reanalysis (single-level 2D params only) | u_component_of_wind_10m; v_component_of_wind_10m; … | ERA5: ~27.8 km; ERA5-Land: ~11.1 km (0.1 deg) | Hourly; ERA5 1940-present, ERA5-Land 1950-pr… | Fast in-cloud predictor extraction co-gridded with INSAT AOD + S5P co… |
| 59 | IMDAA Regional Reanalysis (12 km) | Unified Model + 4D-Var regional reanalysis (Indian mon… | HPBL/boundary layer height; 10m u/v wind; … | 0.12 deg (~12 km) — highest-res reanalysis o… | Hourly; 63 pressure levels; 1979-2018, exten… | Best-resolution India met predictor: 12 km BLH, wind, T, RH, precip,… |
| 60 | MERRA-2 Single-Level + Surface Flux Diagnostics | GEOS-5 reanalysis with assimilated aerosol (GOCART) | PBLH (M2T1NXFLX); T2M / T10M; … | 0.625 x 0.5 deg (~50-65 km) | Hourly time-averaged (tavg1), 1980-present | Independent PBLH + 10m/50m wind + RH(QV2M) + surface shortwave (SWGDN… |
| 61 | GFS / NCEP (near-real-time meteorology) | GFS global forecast/analysis model | temperature_2m; u/v_component_of_wind_10m; … | 0.25 deg (~28 km) | Analysis + forecasts every 6 h; hourly forec… | Operational/forecast met for DAILY AQI maps when ERA5/IMDAA latency i… |
| 62 | CPCB CAAQMS (Continuous Ambient Air Quality Monitoring Stations) | Reference/equivalent continuous analyzers (BAM/TEOM PM… | PM2.5; PM10; … | Point stations, ~500-560 sites (weighted to… | 15-min raw, hourly aggregated | PRIMARY TRAINING LABELS (y) and RMSE/R/MAE target |
| 63 | AERONET (Aerosol Robotic Network) - India sites | Cimel CE318 sun-sky photometer | Spectral AOD 340/380/440/500/675/870/1020nm; 440-870nm Angstrom exponent; … | Point sites (~8-12 active/historical in Indi… | ~15-min daytime cloud-free; daily averages | VALIDATE satellite AOD (INSAT-3D/3DR Imager AOD, MODIS MAIAC) vs refe… |
| 64 | OpenAQ API v3 | Aggregates reference + low-cost (passes through provid… | PM2.5; PM10; … | All ingested India stations (mirrors CPCB +… | Hourly (also raw measurements, days, years r… | PRIMARY PROGRAMMATIC INGESTION for CPCB-style labels with harmonized… |
| 65 | US Embassy / Consulate AirNow PM2.5 monitors (India) | Reference-grade BAM PM2.5 | PM2.5 (reference-grade) | 5 sites: New Delhi, Mumbai, Kolkata, Chennai… | Hourly | Independent reference-grade PM2.5 validation in 5 metros for PRE-2025… |
| 66 | SAFAR (System of Air Quality & Weather Forecasting And Research) | Continuous analyzers + VOC/BC monitors | PM2.5; PM10; … | ~10 stations each Delhi/Mumbai/Pune/Ahmedaba… | Hourly | Independent metro cross-validation of CPCB labels; rare speciated VOC… |
| 67 | PurpleAir (low-cost sensor network) API v1 | Plantower PMS laser nephelometer (PA-II) | PM2.5 (cf=1 / ATM); PM1; … | Dense where deployed; sparse but present in… | ~2-min / configurable real-time | Spatial DENSIFICATION / gap-fill of PM2.5 between sparse reference st… |
| 68 | Unified Ground-Truth Station Database (build spec) | Multi-network merge | station_id; agency; … | All India ground sites merged | Harmonized to hourly | Single keyed registry mapping every label/validation point to lat/lon… |
| 69 | NASA FIRMS — VIIRS 375m Active Fire (NRT + archive) | VIIRS 375m: S-NPP (VNP14IMGTDL), NOAA-20 (VJ114IMGDL/V… | VNP14IMGTDL_NRT active fire; VJ114IMGDL_NRT (NOAA-20); … | 375m (I-band); nominal 375m pixel footprint | ~2-4 overpasses/day combined across S-NPP+NO… | Primary fire-hotspot input: per-pixel FRP+location feed daily fire-co… |
| 70 | NASA FIRMS — MODIS C6.1 Active Fire | MODIS Terra (MOD14) + Aqua (MYD14), Collection 6.1 | MCD14DL / MOD14/MYD14 active fire (lat/lon, FRP, brightness T21/T31, confidence 0-100, day/night) | 1km (nominal); fire pixel 1km | 4 overpasses/day (Terra ~10:30/22:30, Aqua ~… | Long-baseline fire-count climatology (2000-present) to define normal… |
| 71 | GFAS v1.2 (CAMS Global Fire Assimilation System) | MODIS (Terra+Aqua) FRP assimilation -> top-down emissi… | FRP (frpfire); dry-matter burnt; … | 0.1deg (~11km) | Daily | Core physical fire->VOC link: gridded daily HCHO + NMVOC emission flu… |
| 72 | FINNv2.5 (Fire INventory from NCAR) | MODIS (MCD14ML) + VIIRS 375m active fire; land-cover +… | Per-fire daily emissions: CO, NOx, NMOC/NMVOC, HCHO (speciated), VOCs; VOC speciation for MOZART-4, SAPRC99, GEOS-Chem; … | ~1km per-fire (point), griddable to model re… | Daily | Highest-resolution fire->HCHO emission: per-fire daily speciated HCHO… |
| 73 | GFED5 / GFED4s (Global Fire Emissions Database) | MODIS MCD64A1 burned area (post-2001) + VIIRS active f… | Burned area; Emissions: C, CO2, CO, CH4, NMVOC, HCHO precursors, NOx, OC/BC, PM2.5; … | GFED5 0.25deg; GFED4s 0.25deg | Monthly + daily fractions + 3-hourly/diurnal… | Monthly/seasonal NMVOC & burning climatology to bound expected HCHO s… |
| 74 | QFED v2.x (Quick Fire Emissions Dataset) | MODIS (+VIIRS in newer) FRP, top-down with GFAS-style… | FRP-based emissions: CO, CO2, NOx, NMVOC/VOC incl. HCHO precursors, OC/BC, SO2, PM2.5; flaming/smoldering split (next-gen) | 0.1deg (and 0.25deg) | Daily | Alternative top-down FRP->emission inventory for ensemble/uncertainty… |
| 75 | MODIS MCD64A1 Burned Area | MODIS Terra+Aqua surface reflectance + active fire | BurnDate (day-of-year); Uncertainty; … | 500m | Monthly (daily burn-date within month) | Independent burned-area extent to validate active-fire-derived burnin… |
| 76 | VIIRS Nightfire (VNF) + black-marble nighttime | VIIRS day/night band + M-bands (SWIR/MWIR), nighttime | Per-detection temperature, radiant heat, source area, ESF (gas flaring vs biomass); nighttime fire radiance | ~375-750m | Nightly | Captures night/evening burning that day-overpass MODIS/VIIRS-day miss… |
| 77 | NASA TEMPO + Suomi-NPP/NOAA-20 OMPS (geostationary + OMI-continuity composition) | TEMPO geostationary UV-Vis spectrometer (N | TEMPO hourly NO2/HCHO/O3/SO2/aerosol (N.America); OMPS O3 total column (NMTO3); … | TEMPO ~2.1x4.7 km; OMPS-NM ~17 km (NOAA-20)… | TEMPO hourly daytime (geostationary); OMPS d… | TEMPO: diurnal-HCHO methodology/benchmark reference |
| 78 | TROPOMI CH4 + OCO-2/OCO-3 & GOSAT/GOSAT-2 (combustion GHG + SIF) | TROPOMI SWIR (XCH4); OCO grating spectrometer (XCO2/SI… | S5P XCH4 (ppb); OCO-2/3 XCO2 (ppm); … | S5P CH4 ~7x5.5 km; OCO ~1.3x2.25 km sounding… | S5P CH4 daily (OFFL only); OCO-2 16-day; OCO… | Co-pollutant combustion covariates (CH4) + biogenic proxy (SIF) to cl… |
| 79 | Landsat 8/9 OLI/TIRS (C2 L2) + ECOSTRESS LSTE (high-res thermal/optical) | OLI (VNIR/SWIR)+TIRS thermal; ECOSTRESS PHyTIR thermal… | Landsat 30 m surface reflectance (NDVI/NDBI); Landsat Surface Temperature ST_B10 (30 m); … | Landsat 30 m (TIRS->30 m); ECOSTRESS 70 m | Landsat 8-day combined (16-day each); ECOSTR… | High-res covariates (LST, NDVI, NDBI, imperviousness) for AOD/AQI dow… |
| 80 | SRTM & Copernicus DEM GLO-30 (terrain for downscaling & dispersion) | SRTM C-band radar (2000); CopDEM TanDEM-X interferomet… | Elevation DSM/DTM; derived slope/aspect/TPI/TRI; … | SRTM 30 m (1 arc-sec); CopDEM GLO-30 30 m | Static | Static terrain covariates (elevation, slope, aspect, TPI, valley dept… |
| 81 | ESA WorldCover & MODIS MCD12Q1 land cover (land-use covariate) | Sentinel-1/2 (WorldCover); MODIS Terra/Aqua | WorldCover 11-class 10 m LC (2020 v100, 2021 v200); MODIS IGBP/PFT land cover 500 m annual; … | WorldCover 10 m; MODIS 500 m; Dynamic World… | WorldCover 2020/2021 epochs; MODIS annual; D… | Land-use categorical covariates for LUR + AQI ML; source-region maski… |
| 82 | Population: WorldPop 100m & GPWv4.11 (exposure mapping) | Modeled (census + RS dasymetric) | WorldPop 100 m population count/density; WorldPop age/sex structure; … | WorldPop ~100 m; GPW ~1 km (30 arc-sec) | WorldPop annual 2000-2020; GPW 5-yr 2000-2020 | Exposure-mapping layer; population-weighting of AQI; density as anthr… |
| 83 | VIIRS Black Marble VNP46A2 (nighttime lights / combustion proxy) | VIIRS Day/Night Band (DNB) | Gap-filled BRDF-corrected nighttime radiance (500 m daily); gas-flaring/kiln combustion signal; … | 500 m | Daily (VNP46A2); monthly (VNP46A3) | Anthropogenic-combustion/activity covariate for AQI ML & land-use reg… |
| 84 | GPM IMERG V07 (precipitation / wet scavenging) | GPM constellation (DPR + GMI merged, IR-gauge) | Half-hourly precipitation (mm/hr, precipitationCal); daily accumulations; … | ~11 km (0.1 deg) | 30-min; daily; monthly | Wet-scavenging meteorological covariate (precip, antecedent rain) for… |
| 85 | OSM road & industrial / GHSL built layers (land-use regression) | Crowd-sourced vector + EO-derived built-up | Road network density / distance-to-major-road; industrial & landuse polygons; … | Vector (rasterize to 100 m); GHSL 100 m; WSF… | OSM live; GHSL/WSF epochs | Core land-use-regression predictors (road density/distance, industria… |

## Sentinel / Copernicus (Sentinel-5P TROPOMI, Sentinel-3, Sentinel-2)

*SENTINEL (Copernicus) — atmospheric composition & fire over India for BAH 2026 PS3: Sentinel-5P TROPOMI (NO2/SO2/CO/O3/HCHO/CH4/AER_AI/AER_LH/CLOUD), Sentinel-3 OLCI/SLSTR/SYN (AOD/FRP/LST), Sentinel-2 MSI (high-res burn-scar/NBR/land cover).*

**Top picks:**

- **Sentinel-5P TROPOMI HCHO (COPERNICUS/S5P/OFFL/L3_HCHO) — PS3 Objective-2 hotspot target variable**
- **Sentinel-5P TROPOMI NO2 tropospheric (COPERNICUS/S5P/OFFL/L3_NO2) — strongest surface-AQI NO2/NOx predictor**
- **Sentinel-5P AER_AI UV Aerosol Index — smoke tracer to flag fire-influenced HCHO & cloud-tolerant gap-fill**
- **Sentinel-5P CO+SO2+O3 — complete the multi-pollutant AQI feature stack**
- **Sentinel-3 SLSTR FRP + Sentinel-2 MSI NBR — fire energy (1km) and burn-scar/land-cover (20m) for fire-HCHO correlation**

**Key findings:**

- GEE IDs: COPERNICUS/S5P/{OFFL|NRTI}/L3_{NO2,SO2,CO,O3,HCHO,CH4,AER_AI,AER_LH,CLOUD}. All have NRTI+OFFL except CH4 (OFFL only) and AER_LH (OFFL in GEE). Use OFFL/RPRO for AQI training, NRTI for near-real-time.
- TROPOMI nadir pixel = 5.5x3.5 km for UV-VIS gases (since 6 Aug 2019; 7x3.5 before); CO/CH4 SWIR ~7x5.5 km. GEE L3 regrids to ~0.01 deg via harpconvert bin_spatial. ~13:30 LT, daily India coverage.
- HCHO is the noisiest key product: single-pixel random error 30-100% over low-VOC scenes; MUST average temporally/spatially (8-day-monthly, 0.05-0.1 deg) before hotspot detection. Use qa_value>=0.5 for HCHO (vs >=0.75 NO2).
- qa_value is a continuous 0-1 per-pixel quality field bundling cloud, snow/ice, sun-glint, high-SZA, surface and retrieval-convergence flags: keep NO2 >=0.75; HCHO/SO2/CH4/AER_LH >=0.5.
- AER_AI is the linchpin gap-filler: computed at 354/388 nm (low O3 absorption) so it works THROUGH clouds (daily global) and flags UV-absorbing smoke — used to confirm fire-influenced HCHO and discriminate absorbing vs scattering aerosol vs INSAT/S3 AOD.
- Latency tiers: NRTI ~3h (alerts), OFFL hours-days (standard AQI), RPRO/PAL reprocessed (trend & ML training; fixes NO2 polluted-scene low bias pre-v2.4). Train on RPRO history, infer daily on NRTI/OFFL.
- Fire stack: Sentinel-3 SLSTR FRP (1km, ~10:00 & 22:00 LT, NRT, via CDSE/EUMETSAT not GEE) gives fire ENERGY at 2 extra overpass times; Sentinel-2 MSI NBR/dNBR (20m, 5-day) gives burn-scar ground truth; pair with GEE FIRMS MODIS/VIIRS for daily counts.
- Pipeline: Obj-1 feeds qa-filtered RPRO cloud-weighted S5P columns (NO2,SO2,CO,HCHO + AER_AI/AER_LH) + INSAT-3D AOD + ERA5/IMDAA/MERRA-2 met into CNN/CNN-LSTM mapping columns->CPCB surface (RMSE/R/MAE); S5P CLOUD masks gaps, filled via compositing + ML.

### Sentinel-5P TROPOMI HCHO (Formaldehyde, tropospheric column)

- **Provider:** ESA/EU Copernicus; DLR processor; Google Earth Engine
- **Instrument:** TROPOMI (UV-VIS-NIR-SWIR push-broom spectrometer)
- **Products:** L2__HCHO___ orbit NetCDF; formaldehyde_tropospheric_vertical_column (mol/m^2); tropospheric_HCHO_column_number_density_amf; HCHO_slant_column_number_density; cloud_fraction
- **GEE / API IDs:** `COPERNICUS/S5P/OFFL/L3_HCHO`, `COPERNICUS/S5P/NRTI/L3_HCHO`
- **Spatial resolution:** 5.5x3.5 km nadir (since 6 Aug 2019; was 7x3.5). GEE L3 ~0.01 deg via harpconvert bin_spatial.
- **Temporal resolution:** Daily, ~13:30 LT equator crossing; India ~1 overpass/day.
- **Latency:** NRTI ~3h (alerts); OFFL hours-days (standard AQI/hotspots); RPRO/PAL reprocessed for training & trends.
- **Coverage:** 2018-on (stable ~Dec 2018+); India full daily.
- **Access:** GEE ImageCollection; CDSE OData/STAC (dataspace.copernicus.eu); DLR S5P-PAL (s5p-pal.com) reprocessed v2.4+; NASA GES DISC L2 (S5P_L2__HCHO___HiR).
- **Pipeline role:** Primary HCHO-hotspot variable (Obj-2) and VOC/secondary-pollutant feature for surface-AQI CNN/LSTM (Obj-1); couple with reanalysis winds for transport.
- **Gap-fill / cross-verification role:** Cloud/low-SNR gaps filled by temporal compositing, AER_AI smoke-flagging, FIRMS/SLSTR fire covariates, ML imputation. HCHO fills the VOC/pyrogenic axis NO2/CO/SO2 cannot.
- **India relevance:** Obj-2 core: stubble (Oct-Nov Punjab/Haryana) & forest fires (Mar-May) drive VOC-oxidation HCHO; strong IGP enhancements.
- **Caveats:** LOW SNR: single-pixel random error 30-100% over low-VOC scenes. Needs 8-day-monthly / 0.05-0.1 deg averaging. qa_value>=0.5.

### Sentinel-5P TROPOMI NO2 (tropospheric + total + stratospheric)

- **Provider:** ESA/EU Copernicus; DLR/KNMI; GEE
- **Instrument:** TROPOMI
- **Products:** tropospheric_NO2_column_number_density (mol/m^2); NO2_column_number_density (total); stratospheric_NO2_column_number_density; tropopause_pressure; air_mass_factor_troposphere
- **GEE / API IDs:** `COPERNICUS/S5P/OFFL/L3_NO2`, `COPERNICUS/S5P/NRTI/L3_NO2`
- **Spatial resolution:** 5.5x3.5 km nadir; GEE L3 ~0.01 deg.
- **Temporal resolution:** Daily, ~13:30 LT.
- **Latency:** NRTI ~3h; OFFL hours-days; RPRO/PAL v2.4 fixes polluted-scene low bias — use for training.
- **Coverage:** 2018-on; India daily.
- **Access:** GEE; CDSE OData/STAC; GES DISC (S5P_L2__NO2____HiR).
- **Pipeline role:** Top-tier AQI CNN/LSTM feature; column-to-surface via reanalysis PBL height + ML mapping to CPCB stations.
- **Gap-fill / cross-verification role:** Anchors NOx AQI axis; cloud gaps filled by compositing+ML; INSAT-3D AOD & CO cross-verify pollution episodes.
- **India relevance:** Strongest satellite predictor for surface NO2/NOx over IGP, Delhi-NCR, industrial corridors; correlates with CPCB NO2.
- **Caveats:** Clear-sky LOW bias over polluted scenes pre-v2.4 — use RPRO. Column!=surface: needs PBL/met scaling (ERA5/IMDAA). qa_value>=0.75.

### Sentinel-5P TROPOMI SO2 (total column)

- **Provider:** ESA/EU Copernicus; BIRA-IASB/DLR; GEE
- **Instrument:** TROPOMI
- **Products:** SO2_column_number_density (mol/m^2; PBL/1km/7km/15km a-priori); SO2_slant_column; SO2_column_number_density_amf; cloud_fraction
- **GEE / API IDs:** `COPERNICUS/S5P/OFFL/L3_SO2`, `COPERNICUS/S5P/NRTI/L3_SO2`
- **Spatial resolution:** 5.5x3.5 km; GEE L3 ~0.01 deg.
- **Temporal resolution:** Daily.
- **Latency:** NRTI ~3h; OFFL hours-days; RPRO.
- **Coverage:** 2018-on; India daily.
- **Access:** GEE; CDSE; GES DISC (S5P_L2__SO2____HiR).
- **Pipeline role:** SO2 AQI sub-index feature; flags point-source-dominated grid cells.
- **Gap-fill / cross-verification role:** Fills SO2 AQI sub-index that NO2/CO/HCHO cannot; INSAT AOD + AER_AI corroborate episodes; noise cut by temporal binning.
- **India relevance:** Coal plants, smelters, Singrauli/eastern-IGP industrial belt; episodic plumes.
- **Caveats:** VERY noisy for anthropogenic columns; reliable for strong sources only. qa_value>=0.5, heavy (monthly) averaging. Use PBL a-priori for surface AQI.

### Sentinel-5P TROPOMI CO + CH4 (SWIR combustion/GHG tracers)

- **Provider:** ESA/EU Copernicus; SRON; GEE
- **Instrument:** TROPOMI SWIR (2.3 um)
- **Products:** CO_column_number_density (mol/m^2); H2O_column_number_density; CH4_column_volume_mixing_ratio_dry_air (ppb); aerosol_optical_thickness_SWIR
- **GEE / API IDs:** `COPERNICUS/S5P/OFFL/L3_CO`, `COPERNICUS/S5P/NRTI/L3_CO`, `COPERNICUS/S5P/OFFL/L3_CH4`
- **Spatial resolution:** ~7x5.5 km (SWIR); GEE L3 ~0.01 deg.
- **Temporal resolution:** Daily (CH4 cloud/albedo limited).
- **Latency:** CO: NRTI ~3h + OFFL + RPRO. CH4: OFFL only (no NRTI).
- **Coverage:** 2018-on; India daily (CH4 sparse over bright/wet surfaces).
- **Access:** GEE; CDSE; GES DISC (S5P_L2__CO_____HiR, S5P_L2__CH4____HiR).
- **Pipeline role:** CO = robust combustion feature for AQI CO sub-index + fire-HCHO co-tracer; CH4 = optional auxiliary emission feature.
- **Gap-fill / cross-verification role:** CO+HCHO+AER_AI triad confirms fire attribution; SWIR penetrates thin cloud better than UV products, filling combustion-signal gaps where NO2 saturates.
- **India relevance:** CO = excellent biomass-burning/combustion tracer co-located with HCHO; CH4 = paddy/landfill source co-tracer (not a CPCB pollutant).
- **Caveats:** CO LOW noise, good SNR; CH4 strict qa_value>=0.5, fails over bright/high-aerosol/low-albedo scenes (IGP monsoon gaps). Both total-column: surface needs PBL scaling.

### Sentinel-5P TROPOMI O3 (total column + tropospheric TCL)

- **Provider:** ESA/EU Copernicus; DLR/KNMI; GEE
- **Instrument:** TROPOMI
- **Products:** O3_column_number_density (total, mol/m^2); O3_effective_temperature; tropospheric column L2__O3_TCL (tropics); cloud_fraction
- **GEE / API IDs:** `COPERNICUS/S5P/OFFL/L3_O3`, `COPERNICUS/S5P/NRTI/L3_O3`, `COPERNICUS/S5P/OFFL/L3_O3_TCL`
- **Spatial resolution:** Total ~5.5x3.5 km; O3_TCL ~0.5 deg.
- **Temporal resolution:** Daily (total); TCL 3-day/monthly.
- **Latency:** NRTI ~3h; OFFL hours-days.
- **Coverage:** 2018-on; India daily (total).
- **Access:** GEE; CDSE; GES DISC.
- **Pipeline role:** Auxiliary; FNR (HCHO/NO2) feature flags VOC- vs NOx-limited O3 chemistry for AQI model.
- **Gap-fill / cross-verification role:** Provides photochemical-regime context; gaps filled by ML-deriving O3 surrogate rather than the column.
- **India relevance:** Surface O3 is a CPCB AQI pollutant, but total column is stratosphere-dominated — use HCHO/NO2 ratio (FNR) as O3-regime indicator instead.
- **Caveats:** Total column ~stratospheric: POOR surface-O3 proxy. O3_TCL coarse, tropics-only. Derive surface-O3 surrogate via ML from NO2+HCHO+temperature.

### Sentinel-5P AER_AI (UV/Absorbing Aerosol Index) + CLOUD

- **Provider:** ESA/EU Copernicus; KNMI/DLR; GEE
- **Instrument:** TROPOMI UV (354/388 nm; also 340/380)
- **Products:** absorbing_aerosol_index (unitless); cloud_fraction; cloud_top_pressure/height; cloud_optical_depth; surface_albedo
- **GEE / API IDs:** `COPERNICUS/S5P/NRTI/L3_AER_AI`, `COPERNICUS/S5P/OFFL/L3_AER_AI`, `COPERNICUS/S5P/OFFL/L3_CLOUD`, `COPERNICUS/S5P/NRTI/L3_CLOUD`
- **Spatial resolution:** 5.5x3.5 km; GEE L3 ~0.01 deg.
- **Temporal resolution:** Daily.
- **Latency:** AER_AI & CLOUD: NRTI ~3h + OFFL.
- **Coverage:** 2018-on; India daily INCLUDING cloudy scenes (AER_AI).
- **Access:** GEE; CDSE; GES DISC.
- **Pipeline role:** AER_AI = smoke-presence/episode feature; CLOUD = quality/weighting layer for all S5P columns and gap identification.
- **Gap-fill / cross-verification role:** CRITICAL gap-filler: AAI computable through clouds (daily) fills coverage where AOD/HCHO fail; flags fire-influenced HCHO (pyrogenic vs biogenic); cross-checks INSAT/S3 AOD absorbing-vs-scattering. CLOUD drives compositing/imputation decisions.
- **India relevance:** Positive AAI = UV-absorbing smoke/dust; lights up IGP during stubble burning & NW-India dust. CLOUD quantifies monsoon (JJAS) gaps.
- **Caveats:** AAI qualitative (not concentration); positive=absorbing, negative=scattering/cloud; cannot alone separate smoke vs dust — combine with FRP/CO. CLOUD is a mask/weight, not a pollutant.

### Sentinel-5P AER_LH (Aerosol Layer Height)

- **Provider:** ESA/EU Copernicus; KNMI; GEE
- **Instrument:** TROPOMI O2-A band (760 nm)
- **Products:** aerosol_height (m); aerosol_pressure (Pa); aerosol_optical_depth_760nm; cloud_fraction
- **GEE / API IDs:** `COPERNICUS/S5P/OFFL/L3_AER_LH`
- **Spatial resolution:** ~5.5x3.5 km; GEE L3 ~0.01 deg.
- **Temporal resolution:** Daily (cloud-free, elevated-aerosol scenes).
- **Latency:** OFFL in GEE (~days); NRTI exists product-side.
- **Coverage:** 2018-on; retrieved mainly over thick elevated plumes/dark land.
- **Access:** GEE; CDSE.
- **Pipeline role:** Vertical context: tells AQI model whether AAI/AOD aerosol is in PBL (affects surface) or aloft.
- **Gap-fill / cross-verification role:** Disambiguates column vs surface aerosol — fills the vertical-distribution gap 2D columns lack; supports transport analysis with winds.
- **India relevance:** Plume injection height of stubble-burning smoke — distinguishes near-surface (AQI-relevant) vs lofted long-range transport.
- **Caveats:** Retrieves only for thick elevated layers; sparse over bright land; biased in mixed scenes. qa_value>=0.5.

### Sentinel-3 SLSTR Active Fire & FRP + SYN AOD + SLSTR LST

- **Provider:** EU Copernicus / EUMETSAT (S3A+S3B)
- **Instrument:** SLSTR (MWIR 3.74 um S7, TIR, dual-view) + OLCI for SYN AOD
- **Products:** FRP (MW/pixel); fire detection flag/confidence; SY_2_AOD AOD@550nm; SL_2_LST Land Surface Temperature
- **GEE / API IDs:** `COPERNICUS/S3/OLCI (L1 TOA in GEE)`, `SL_2_FRP___ (CDSE/EUMETSAT)`, `SY_2_AOD___`, `SL_2_LST___`
- **Spatial resolution:** FRP & LST 1 km; SYN AOD ~4.5 km; OLCI 300 m.
- **Temporal resolution:** S3A+S3B ~daily India; ~10:00 & ~22:00 LT (complements 13:30 TROPOMI/MODIS-Aqua, VIIRS).
- **Latency:** NRT ~3h; NTC reprocessed.
- **Coverage:** 2018-on (daytime FRP operational Mar 2022+); India daily.
- **Access:** FRP/AOD/LST L2 via EUMETSAT Data Store / CDSE (NOT first-class in GEE); GEE has S3 OLCI L1 TOA. Pair with GEE FIRMS (MODIS/VIIRS) for daily detection.
- **Pipeline role:** FRP = fire-energy covariate for fire-HCHO correlation (Obj-2); AOD = PM proxy feature; LST = met feature for AQI ML.
- **Gap-fill / cross-verification role:** FRP adds 2 daily overpass times filling single-overpass temporal gap & quantifies energy FIRMS counts lack; AOD cross-verifies INSAT-3D AOD; LST fills surface-temp gaps affecting biogenic HCHO.
- **India relevance:** Independent fire energy for crop-residue & forest fires; mid-morning/evening slots fill MODIS-Terra decline; AOD complements INSAT; LST drives biogenic-VOC/PBL context.
- **Caveats:** FRP/AOD/LST not in GEE catalog (extra ingestion); FRP 1 km coarser than VIIRS 375 m; cloud-obscured fires missed; optical AOD daytime/cloud-limited.

### Sentinel-2 MSI High-res Fire/Burn-scar (NBR) & Land Cover

- **Provider:** EU Copernicus; GEE; ESA WorldCover
- **Instrument:** MSI (13 bands, 10/20/60 m)
- **Products:** NBR=(B8-B12)/(B8+B12); dNBR burn severity; active-fire via SWIR B11/B12; land cover / fuel type; SR Harmonized
- **GEE / API IDs:** `COPERNICUS/S2_SR_HARMONIZED`, `COPERNICUS/S2_HARMONIZED`, `ESA/WorldCover/v200`
- **Spatial resolution:** 10 m (B8), 20 m SWIR (B11/B12); burn-scar/fire ~20 m.
- **Temporal resolution:** ~5-day revisit (S2A+S2B+S2C); India good.
- **Latency:** L2A SR ~hours-1 day; not NRT fire.
- **Coverage:** 2017-on (L2A global 2019+); India full.
- **Access:** GEE (above); CDSE OData/STAC; ESA WorldCover 10 m land cover.
- **Pipeline role:** High-res ground-truth of burned area & fuel/land cover validating coarse HCHO/FRP hotspots (Obj-2).
- **Gap-fill / cross-verification role:** Fills the SPATIAL-resolution gap of 1-5.5 km products: confirms which TROPOMI HCHO / SLSTR-FRP hotspots are real burn scars; land cover separates pyrogenic vs biogenic HCHO sources.
- **India relevance:** Maps exact burn scars/fields in Punjab-Haryana & forest-fire perimeters; land cover/fuel for emission attribution.
- **Caveats:** 5-day revisit + cloud => NOT a daily/operational fire detector; use for post-event burn-area & validation, not real-time timing.

## Geostationary & Indian Satellites (INSAT-3D/3DR/3DS, Himawari, FY-4, GK-2A, GEMS)

*Geostationary & Indian satellites for AOD, fire, and atmospheric composition over India (INSAT-3D/3DR/3DS, Himawari-8/9 AHI, FY-4A/4B AGRI, GK-2A AMI, GEMS/GK-2B)*

**Top picks:**

- **INSAT-3D/3DR Imager AOD (MOSDAC) — native Indian GEO AOD, ~4 km, half-hourly; PS3 mandated AOD input**
- **GEMS on GK-2B — hourly GEO NO2/SO2/HCHO/O3/AOD over Asia; covers IGP/eastern India (75-145E); diurnal game-changer filling TROPOMI's daily gap**
- **Himawari-8/9 AHI — 10-min full-disk AOD + FRP fire; usable only for E/NE India (extreme VZA over W India)**
- **FY-4A/4B AGRI — 15-min AOD + FHS fire over Asia incl. India; FY-4A at 105E has good India geometry**
- **INSAT-3DS active-fire product — newest Indian GEO fire detection complementing VIIRS/MODIS FIRMS**

**Key findings:**

- GEMS (GK-2B) field of regard 75E-145E, 5S-45N covers the Indo-Gangetic Plain and central/eastern India but CUTS OFF western India (W of ~75E: most Rajasthan/Gujarat). Only hourly GEO composition source over India; biggest gap-fill for TROPOMI's once-daily overpass.
- GEMS native res 3.5km(NS)x8km(EW) at Seoul (coarser/oblique over India) for NO2/O3/HCHO/AOD; SO2 7x16km (2x2 co-add); CHOCHO 14x32km (4x4); ~8 daytime scans/day; 300-500nm UV-Vis at 0.6nm.
- Himawari-9 (140.7E) AHI full disk reaches ~80E but India is at extreme viewing-zenith angle; AOD quality drops sharply with large VZA, so AHI AOD is reliable only for NE/E India fringe, not the IGP.
- FY-4A (105E)/FY-4B (133E) AGRI give 15-min AOD and a 2km Fire/Hotspot (FHS) product; FY-4 multichannel AOD RMSE~0.16 over South Asia beats MODIS DT/DB (~0.18). FY-4A at 105E views India far better than Himawari.
- INSAT-3D/3DR Imager AOD is operational+free on MOSDAC & VEDAS (Mishra 2018 validation). Imager: VIS/SWIR 1km, MIR/TIR 4km, WV 8km; AOD ~4-8km; 3D+3DR combined cadence 15-min imager / 30-min sounder.
- INSAT-3DS (launched 2024) adds native Indian GEO active-fire detection (RSESS 2025), complementing polar MODIS/VIIRS FIRMS for sub-hourly fire timing in stubble/forest-burning seasons.
- Geostationary AOD/composition (INSAT, GEMS, FY-4, Himawari, GK-2A) fills TROPOMI/MODIS TEMPORAL gaps (single daily overpass, cloud days) with multiple looks/day, improving CNN-LSTM surface-AQI training and capturing diurnal cycles polar sensors miss.
- Access split: INSAT/3DS->MOSDAC (mosdac.gov.in); GEMS->NIER NESC (nesc.nier.go.kr, web+Open-API)+ESA; Himawari AOD/FRP->JAXA P-Tree + NASA LAADS (XAERDT_L2_AHI) + FIRMS; FY-4->NSMC (nsmc.org.cn); GK-2A AMI->KMA NMSC (nmsc.kma.go.kr). None of the GEO AOD/composition products are GEE-native.

### INSAT-3D / 3DR Imager Aerosol Optical Depth (AOD)

- **Provider:** ISRO / IMD / Space Applications Centre (SAC)
- **Instrument:** 6-band Imager (VIS 0.55, SWIR 1.6, MIR, WV, TIR1/2)
- **Products:** L2 AOD over land+ocean (550nm); ancillary OLR, SST, cloud
- **GEE / API IDs:** `MOSDAC 3DIMG_L2B_AOD`, `MOSDAC 3RIMG_L2B_AOD`
- **Spatial resolution:** VIS/SWIR 1km; MIR/TIR 4km; WV 8km; AOD gridded ~4-8km
- **Temporal resolution:** Imager 30 min single sat; 15 min for 3D+3DR; daytime AOD
- **Latency:** NRT (hours) on MOSDAC
- **Coverage:** Full Indian disk; 3D sub-point ~82E, 3DR ~74E; entire India incl. western India well-viewed
- **Access:** MOSDAC https://mosdac.gov.in (free registration); also VEDAS https://vedas.sac.gov.in; HDF5; not in GEE; co-archived at IMD New Delhi.
- **Pipeline role:** PRIMARY mandated AOD input for PS3 Objective-1 surface-AQI maps. Native Indian GEO AOD with good geometry over ALL India incl. western India where Himawari/GEMS fail; feeds PM2.5/PM10 estimation in CNN/LSTM.
- **Gap-fill / cross-verification role:** Fills TROPOMI/MODIS daily-overpass temporal gap with half-hourly AOD; covers western India that GEMS (cut at 75E) misses; GEO cadence captures dust-storm/diurnal aerosol evolution.
- **India relevance:** ISRO native, PS3-mandated; best-geometry GEO AOD over IGP and western desert dust belt.
- **Caveats:** Daytime-only, cloud-screened gaps; coarser than MODIS MAIAC 1km; AOD bias over bright desert; HDF5 not cloud-optimized.

### GEMS (Geostationary Environment Monitoring Spectrometer) on GK-2B

- **Provider:** NIER / KARI (South Korea); ESA mirror
- **Instrument:** UV-Vis hyperspectral imaging spectrometer 300-500nm, 0.6nm
- **Products:** L2 NO2 (tropo+total); L2 HCHO column; L2 SO2 column; L2 O3 total/tropo/profile; L2 CHOCHO (glyoxal); L2 Aerosol: AOD/UVAI/SSA/ALH; UVI, cloud CRF/CCH, surface reflectance
- **GEE / API IDs:** `NIER NESC Open-API nesc.nier.go.kr/en/html/svc/openapi`, `ESA GEMS distribution`
- **Spatial resolution:** 3.5x8km (NO2/O3/HCHO/AOD) at Seoul; SO2 7x16km; CHOCHO 14x32km; coarser/oblique over India
- **Temporal resolution:** Hourly, ~8 daytime scans/day
- **Latency:** NRT to ~1 day (L2)
- **Coverage:** Asia FOR 75E-145E, 5S-45N. Covers Indo-Gangetic Plain, central & eastern India; EXCLUDES western India (W of ~75E).
- **Access:** NIER Environmental Satellite Center https://nesc.nier.go.kr (web download + Open-API, free registration); ESA mirror; NetCDF L2; not GEE-native.
- **Pipeline role:** GAME-CHANGER: HOURLY column NO2/SO2/HCHO/O3/AOD over India — same species as TROPOMI but hourly. Powers Objective-2 HCHO hotspot detection at intra-day resolution and Objective-1 diurnal AQI.
- **Gap-fill / cross-verification role:** Fills TROPOMI's once-daily (~13:30) temporal gap — resolves morning/afternoon NO2 & HCHO peaks, photochemistry diurnal cycle, biomass-burning HCHO build-up a single polar overpass misses; cross-validates TROPOMI at overpass time.
- **India relevance:** Only hourly GEO composition sensor seeing India; transformative for diurnal AQI over IGP (Delhi/Lucknow/Patna) and HCHO-fire transport.
- **Caveats:** Western India (W of 75E) outside FOR; large view/solar zenith over India inflates noise & footprint; daytime-only; v2.0 validated mainly Korea/Japan/SE-Asia — needs CPCB/India bias-correction.

### Himawari-8 / Himawari-9 AHI — AOD & Active Fire (FRP)

- **Provider:** JMA / JAXA; EUMETSAT; NASA (DT aerosol, FIRMS)
- **Instrument:** Advanced Himawari Imager (AHI), 16 bands, 0.5-2km
- **Products:** L2 Dark Target AOD 10km; JAXA P-Tree hourly AOD (ARP); FRP-PIXEL active fire; FIRMS GEO fire detections
- **GEE / API IDs:** `NASA LAADS XAERDT_L2_AHI_H08`, `NASA LAADS XAERDT_L2_AHI_H09`, `JAXA P-Tree eorc.jaxa.jp/ptree`, `NASA FIRMS (Himawari)`, `GEE: NOAA L1B radiance only`
- **Spatial resolution:** AHI 0.5-2km; AOD 10km at nadir; FRP ~2km
- **Temporal resolution:** Full disk every 10 min; daytime AOD
- **Latency:** NRT (~10-20 min via P-Tree/FIRMS)
- **Coverage:** Full disk 60S-60N, ~80E-160W from 140.7E. India only at far western edge — extreme VZA; reliable AOD only NE/E India fringe.
- **Access:** JAXA P-Tree https://www.eorc.jaxa.jp/ptree (free, FTP/THREDDS); NASA LAADS https://ladsweb.modaps.eosdis.nasa.gov; FIRMS https://firms.modaps.eosdis.nasa.gov; AWS noaa-himawari. AOD/FRP NOT in GEE.
- **Pipeline role:** Secondary 10-min AOD + active-fire timing for E/NE India and Bay-of-Bengal aerosol transport; sub-10-min fire diurnal cycle for Objective-2 over E/NE India.
- **Gap-fill / cross-verification role:** Highest-cadence (10-min) GEO fire/AOD to time biomass-burning onset between polar VIIRS/MODIS passes — but only where India is adequately viewed (E/NE).
- **India relevance:** Limited: extreme VZA over most of India (140.7E sub-point); best for NE India/Myanmar burning fringe.
- **Caveats:** Western & central India effectively unusable for quantitative AOD (VZA/geometric distortion); daytime AOD only; not in GEE.

### FY-4A / FY-4B AGRI — AOD & Fire/Hotspot (FHS)

- **Provider:** NSMC / China Meteorological Administration (CMA)
- **Instrument:** Advanced Geostationary Radiation Imager (AGRI), 15 ch 0.45-13.6um
- **Products:** L2 AOD (multichannel MC algorithm); Fire/Hotspot product FHS; ancillary cloud, LST
- **GEE / API IDs:** `NSMC FY-4 service satellite.nsmc.org.cn`, `FY-4A AGRI L2 AOD`, `FY-4A FHS fire`
- **Spatial resolution:** AGRI 0.5-4km; AOD ~4km; FHS 2km
- **Temporal resolution:** Full disk every 15 min; daytime AOD
- **Latency:** NRT (hours) via NSMC
- **Coverage:** FY-4A sub-point 105E (good India geometry); FY-4B 133E; full disk covers all India, FY-4A favorably placed.
- **Access:** NSMC https://www.nsmc.org.cn / https://satellite.nsmc.org.cn (free registration, may need approval); HDF; not in GEE.
- **Pipeline role:** Supplementary 15-min GEO AOD with strong India viewing geometry (105E) plus a 2km fire product for Objective-2 fire timing.
- **Gap-fill / cross-verification role:** FY-4A at 105E covers India far better than Himawari (140.7E); MC AOD RMSE~0.16 over South Asia beats MODIS — good GEO AOD gap-fill where INSAT unavailable; 15-min fire bridges polar gaps.
- **India relevance:** 105E sub-point gives genuinely usable AOD over India incl. IGP, unlike Himawari; validated over South Asia.
- **Caveats:** Access friction (Chinese portal/approval); FHS 2km coarser than VIIRS 375m; daytime AOD; docs mostly Chinese.

### GK-2A (GEO-KOMPSAT-2A) AMI — Aerosol Detection / AOD

- **Provider:** KMA NMSC / KARI (South Korea)
- **Instrument:** Advanced Meteorological Imager (AMI), 16 ch (AHI twin)
- **Products:** Aerosol Detection (dust/haze/ash; 4 day, 3 night types); Visible AOD; Dust AOD (DAOD) day+night; 52 met products
- **GEE / API IDs:** `KMA NMSC nmsc.kma.go.kr`, `GK2A AMI AOD / AERSL`
- **Spatial resolution:** AMI 0.5-2km; AOD ~ few km
- **Temporal resolution:** Full disk every 10 min
- **Latency:** NRT
- **Coverage:** 128.2E sub-point; East Asia/W-Pacific/Indian Ocean; India at high VZA (similar to Himawari, slightly west).
- **Access:** KMA NMSC https://nmsc.kma.go.kr/enhome (free registration); NetCDF; not in GEE.
- **Pipeline role:** Aerosol-type discrimination (dust vs haze) and day+night DAOD to flag dust events; adds aerosol-type context to AQI modeling.
- **Gap-fill / cross-verification role:** Day+night DAOD adds nighttime dust info daytime-only sensors lack; aerosol-type classification separates dust from anthropogenic haze in PM estimation.
- **India relevance:** Limited quantitative AOD over India (VZA from 128.2E); main value is dust-type detection/transport context.
- **Caveats:** High viewing angle over India; AOD RMSE~0.21 even over E-Asia; weak quantitative use over India.

### INSAT-3DS Imager — Active Fire & AOD

- **Provider:** ISRO / SAC / IMD
- **Instrument:** Improved 6-band Imager + 19-ch Sounder
- **Products:** Active fire detection product; Imager AOD (3D/3DR continuity); Sounder T/humidity profiles
- **GEE / API IDs:** `MOSDAC 3SIMG_L2 products`
- **Spatial resolution:** Imager 1-4km; fire ~4km
- **Temporal resolution:** Imager ~15-30 min; daytime/24h fire
- **Latency:** NRT on MOSDAC
- **Coverage:** Full Indian disk; native good geometry over all India
- **Access:** MOSDAC https://mosdac.gov.in (free registration); HDF5; not in GEE.
- **Pipeline role:** Native Indian GEO active-fire timing for Objective-2 biomass-burning detection; extends 3D/3DR AOD record for Objective-1.
- **Gap-fill / cross-verification role:** Sub-hourly Indian GEO fire complements polar VIIRS/MODIS FIRMS (which miss inter-overpass ignition/peak); validates FY-4/Himawari fire over India with native geometry.
- **India relevance:** Newest ISRO GEO sensor (2024) purpose-built over India; ideal native fire+AOD continuity for an ISRO hackathon.
- **Caveats:** New mission — short record, limited published validation; product maturity evolving; HDF5 not cloud-native.

## MODIS Suite (Terra / Aqua / Combined)

*MODIS (Terra MOD / Aqua MYD / Combined MCD) suite — AOD, active fire, burned area, LST, land cover, vegetation indices. Terra (10:30 LT descending) + Aqua (13:30 LT ascending) give 4 daily overpasses over India (~2 day + 2 night). Backbone: 1km MAIAC AOD (MCD19A2) as the high-res AOD anchor for satellite-PM2.5 over India and the gap-fill/cross-validation reference for coarse INSAT-3D AOD; FIRMS/MOD14 fire for the HCHO biomass-burning objective.*

**Top picks:**

- **MCD19A2 MAIAC 1km AOD (MODIS/061/MCD19A2_GRANULES) — primary high-res AOD backbone for PM2.5 + INSAT AOD gap-fill/validation**
- **FIRMS rasterized fire (FIRMS) + MCD14DL/ML NRT fire locations — fire input for HCHO hotspot + fire-HCHO correlation**
- **MOD14A1/MYD14A1 1km thermal anomalies (MODIS/061/MOD14A1, MODIS/061/MYD14A1) — science-quality fire radiative power**
- **MCD64A1 500m burned area (MODIS/061/MCD64A1) — seasonal burn extent for biomass-burning season masking**
- **MOD11A1/MYD11A1 1km LST + MOD13Q1 250m NDVI/EVI + MCD12Q1 500m land cover — meteo/land predictors for AQI ML models**

**Key findings:**

- MCD19A2 MAIAC AOD at 1km (GEE: MODIS/061/MCD19A2_GRANULES, band Optical_Depth_055) is the backbone high-res satellite AOD for India PM2.5; Indian studies report 5-fold CV R2~0.92, RMSE~11.8 ug/m3 annual after gap-filling.
- Terra (10:30 LT) + Aqua (13:30 LT) give 2 daytime + 2 nighttime overpasses/day; this twice-daily snapshot misses the diurnal cycle, which geostationary INSAT-3D/3DR AOD fills — MODIS in turn cross-validates/downscales coarse INSAT-3D AOD (~4-10km).
- MAIAC + Deep Blue (MOD04/MYD04) retrieve AOD over bright IGP/arid surfaces where Dark Target fails; Angstrom exponent separates fine smoke/anthropogenic from coarse dust for PM2.5 speciation.
- FIRMS (GEE asset 'FIRMS') is rasterized MOD14/MYD14 NRT (<3h latency, not science quality); MOD14A1/MYD14A1 (MODIS/061/MOD14A1, .../MYD14A1) are 1km science-quality FireMask+MaxFRP; MCD14ML is the monthly science-quality point archive for robust fire-HCHO correlation.
- For Objective-2, pair MODIS FIRMS with VIIRS FIRMS (375m, Suomi-NPP/NOAA-20) to fill MODIS 1km/timing gaps and catch small stubble fires; MCD64A1 500m monthly burned area gives total burned extent that point detections miss.
- MODIS LST (MOD11A1/MYD11A1 1km day+night), land cover (MCD12Q1 500m), and NDVI/EVI (MOD13Q1 250m) supply high-res surface/land predictors that add spatial structure to coarse ERA5/IMDAA/MERRA-2 meteorology in the CNN/LSTM AQI model.
- Cloud and monsoon (Jun-Sep) gaps are the main MAIAC limitation; standard fix is MERRA-2 AOD + INSAT AOD fusion to impute gaps before AOD->PM2.5 conversion, applying PBLH/RH correction.
- MOD04/MYD04 (MOD04_L2 10km, MOD04_3K 3km) are not native GEE ImageCollections and must be ingested from LAADS DAAC; the 3km product usefully bridges 1km MAIAC and ~10km INSAT scales for multi-resolution fusion.

### MCD19A2 — MAIAC Land Aerosol Optical Depth (Terra+Aqua combined)

- **Provider:** NASA LP DAAC / processed in Google Earth Engine
- **Instrument:** MODIS Terra + Aqua (MAIAC algorithm)
- **Products:** Optical_Depth_047 (AOD @ 0.47um); Optical_Depth_055 (AOD @ 0.55um, primary for PM2.5); AOD_Uncertainty; FineModeFraction; Column_WV (column water vapor); AOD_QA (bit flags: cloud mask, adjacency, QA); cosSZA/cosVZA/RelAZ/Scattering_Angle/Glint angles
- **GEE / API IDs:** `MODIS/061/MCD19A2_GRANULES`
- **Spatial resolution:** 1 km (native MAIAC grid)
- **Temporal resolution:** Daily; one image per Terra and per Aqua overpass (multiple granules/day stacked as bands)
- **Latency:** Standard science quality, ~days-weeks; not NRT
- **Coverage:** 2000-02-24 to present (Terra); Aqua from 2002
- **Access:** GEE: MODIS/061/MCD19A2_GRANULES (ImageCollection). Also LAADS/LP DAAC HDF (MCD19A2.061). Earthdata: lpcloud-mcd19a2-061.
- **Pipeline role:** Objective-1 backbone: highest-resolution (1km) satellite AOD predictor for surface PM2.5/AQI. Feed Optical_Depth_055 + uncertainty + FMF into CNN/LSTM/CNN-LSTM alongside TROPOMI columns and ERA5/IMDAA/MERRA-2 meteo, calibrate to CPCB PM2.5.
- **Gap-fill / cross-verification role:** Cross-validates and downscales coarser INSAT-3D AOD (~4-10km); fills INSAT spatial detail and provides an independent AOD reference to bias-correct INSAT. AOD_QA enables strict cloud/adjacency filtering; residual cloud gaps filled by MERRA-2/INSAT AOD fusion. Provides the AOD-PM2.5 scaling anchor (PBLH/RH-corrected).
- **India relevance:** MAIAC retrieves AOD over bright IGP/urban/cropland surfaces where Dark Target fails; 1km resolves city-scale gradients across Delhi-NCR and IGP.
- **Caveats:** Cloud/monsoon gaps (Jun-Sep sparse); twice-daily snapshots only — diurnal cycle from geostationary INSAT-3D/3DR needed. Retrieval bias over very bright/snow surfaces.

### MOD04/MYD04 L2 — Dark Target & Deep Blue AOD

- **Provider:** NASA LAADS DAAC (Atmosphere SIPS)
- **Instrument:** MODIS Terra (MOD04_L2/MOD04_3K) / Aqua (MYD04_L2/MYD04_3K)
- **Products:** Optical_Depth_Land_And_Ocean (DT, 10km); Deep_Blue_Aerosol_Optical_Depth_550_Land (DB); AOD_550_Dark_Target_Deep_Blue_Combined; Angstrom_Exponent; Image_Optical_Depth_Land_And_Ocean (3km, MOD04_3K)
- **GEE / API IDs:** `LAADS:MOD04_L2`, `LAADS:MOD04_3K`, `LAADS:MYD04_L2`, `LAADS:MYD04_3K`
- **Spatial resolution:** 10 km (MOD04_L2) and 3 km (MOD04_3K) at nadir
- **Temporal resolution:** Twice daily (Terra ~10:30, Aqua ~13:30 LT)
- **Latency:** Standard; NRT via LANCE
- **Coverage:** 2000 (Terra) / 2002 (Aqua) to present
- **Access:** LAADS DAAC HDF (MOD04_L2, MOD04_3K, MYD04_*). Not natively in GEE catalog as ImageCollection — ingest via download or Earthdata. NRT via LANCE.
- **Pipeline role:** Secondary/validation AOD; Combined DT+DB and Angstrom exponent add coarse-but-robust AOD and particle-size info to PM2.5 model where MAIAC is gap-flagged.
- **Gap-fill / cross-verification role:** Independent algorithm family to cross-check MAIAC and INSAT AOD; Deep Blue fills bright-surface gaps; 3km product bridges 1km MAIAC and 10km/INSAT scales for multi-resolution fusion. Angstrom exponent helps distinguish fine (smoke/anthropogenic) vs coarse (dust) aerosol for PM2.5.
- **India relevance:** Deep Blue critical over bright arid NW India/Thar and IGP where Dark Target underperforms; Combined DT+DB maximizes coverage.
- **Caveats:** Coarser than MAIAC; DT and DB have different biases needing harmonization; no native GEE collection — extra ingestion step.

### MOD14A1/MYD14A1 — Thermal Anomalies & Fire Daily 1km

- **Provider:** NASA LP DAAC
- **Instrument:** MODIS Terra (MOD14A1) / Aqua (MYD14A1)
- **Products:** FireMask (class 0-9: 7=low,8=nominal,9=high confidence fire); MaxFRP (Fire Radiative Power); QA; sample (pixel position)
- **GEE / API IDs:** `MODIS/061/MOD14A1`, `MODIS/061/MYD14A1`
- **Spatial resolution:** 1 km
- **Temporal resolution:** Daily composite of Terra (10:30/22:30) and Aqua (13:30/01:30) overpasses
- **Latency:** Science quality, ~days; NRT counterpart via LANCE
- **Coverage:** 2000-02 (Terra) / 2002-07 (Aqua) to present
- **Access:** GEE: MODIS/061/MOD14A1 and MODIS/061/MYD14A1 (ImageCollection).
- **Pipeline role:** Objective-2: gridded 1km fire mask + FRP as raster predictor for HCHO hotspot detection and fire-HCHO correlation; FRP scales emission intensity.
- **Gap-fill / cross-verification role:** Science-quality complement to NRT FIRMS — backfills/validates FIRMS detections; FRP quantifies burning intensity to correlate against TROPOMI HCHO enhancements and weight transport analysis.
- **India relevance:** Detects Punjab/Haryana paddy-stubble burning (Oct-Nov) and Central/NE India forest fires (Mar-May) feeding HCHO.
- **Caveats:** 1km misses small/cool fires; cloud/smoke obscuration; twice-daily timing misses fires between overpasses (geostationary INSAT/VIIRS help). Confidence thresholding needed to cut false alarms.

### FIRMS rasterized fire + MCD14DL/MCD14ML active-fire locations

- **Provider:** NASA FIRMS / LANCE
- **Instrument:** MODIS Terra+Aqua (MOD14/MYD14 algorithm)
- **Products:** T21 (brightness temp 4um, K); MaxFRP (raster); confidence; line/sample; MCD14DL vector NRT points; MCD14ML monthly science-quality point archive (lat/lon, FRP, confidence, day/night, acq time)
- **GEE / API IDs:** `FIRMS`, `FIRMS_API:MODIS_NRT`, `FIRMS_API:MODIS_SP`, `LAADS:MCD14ML`
- **Spatial resolution:** 1 km (centroid of flagged pixel)
- **Temporal resolution:** Per-overpass NRT (4x/day); MCD14ML monthly archive
- **Latency:** FIRMS NRT < 3 hours (LANCE/ULTRA-NRT); MCD14ML monthly science quality
- **Coverage:** 2000 to present
- **Access:** GEE raster: FIRMS (ImageCollection). FIRMS API: firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/MODIS_NRT/<bbox>/<days>. MCD14ML at LAADS/FTP.
- **Pipeline role:** PS3-named fire input. NRT FIRMS points/raster drive daily HCHO-hotspot fire overlay; MCD14ML monthly gives clean point archive for robust fire-HCHO correlation and clustering (DBSCAN/Getis-Ord on fire+HCHO).
- **Gap-fill / cross-verification role:** FIRMS provides low-latency fire presence to pair with near-real-time TROPOMI HCHO; combine MODIS FIRMS with VIIRS (375m, SUOMI/NOAA-20) FIRMS to fill MODIS coarse/timing gaps and confirm small fires. MCD14ML backfills/QA the NRT stream.
- **India relevance:** Standard fire dataset cited in PS3; FIRMS country/region feeds give India stubble + forest-fire points operationally with VIIRS.
- **Caveats:** NRT is not science quality (geolocation/confidence may shift); only fire presence, not emission flux; clustered fires merge in 1km. Pair with VIIRS for completeness.

### MCD64A1 — Burned Area Monthly 500m

- **Provider:** NASA LP DAAC
- **Instrument:** MODIS Terra+Aqua (500m SR + 1km active fire)
- **Products:** BurnDate (day-of-year of burn, 1-366); Uncertainty; QA; FirstDay/LastDay (burn window)
- **GEE / API IDs:** `MODIS/061/MCD64A1`
- **Spatial resolution:** 500 m
- **Temporal resolution:** Monthly
- **Latency:** Science quality, ~1-2 months lag
- **Coverage:** 2000-11 to present
- **Access:** GEE: MODIS/061/MCD64A1 (ImageCollection). Earthdata lpcloud-mcd64a1-061.
- **Pipeline role:** Objective-2 context: defines spatial extent/timing of biomass-burning seasons to mask/weight HCHO hotspot analysis and aggregate seasonal emission area.
- **Gap-fill / cross-verification role:** Complements instantaneous fire detections (which miss area) by giving total burned area — fills the 'how much burned' gap that point fires omit; BurnDate constrains the temporal window for fire-HCHO correlation.
- **India relevance:** Maps cumulative burned extent in Punjab/Haryana cropland and Central/NE forest belts per season.
- **Caveats:** Underestimates small/fragmented cropland burns (known IGP bias); monthly latency unsuitable for daily NRT; 500m misses field-edge burns.

### MOD11A1/MYD11A1 — Land Surface Temperature & Emissivity Daily 1km

- **Provider:** NASA LP DAAC
- **Instrument:** MODIS Terra (MOD11A1) / Aqua (MYD11A1)
- **Products:** LST_Day_1km; LST_Night_1km; QC_Day/QC_Night; Day_view_time/Night_view_time; Emis_31/Emis_32
- **GEE / API IDs:** `MODIS/061/MOD11A1`, `MODIS/061/MYD11A1`
- **Spatial resolution:** 1 km (also MOD11A2 8-day; MOD11B 6km)
- **Temporal resolution:** Daily, day+night (Terra 10:30/22:30, Aqua 13:30/01:30 LT)
- **Latency:** Standard, ~days
- **Coverage:** 2000 to present
- **Access:** GEE: MODIS/061/MOD11A1 and MODIS/061/MYD11A1.
- **Pipeline role:** Auxiliary meteorological/surface predictor in AQI CNN/LSTM models (surrogate for near-surface T and stability); day-night LST captures diurnal thermal contrast affecting boundary-layer mixing.
- **Gap-fill / cross-verification role:** Provides 1km surface-thermal field that bridges coarse ERA5/MERRA-2 (~9-50km) gridded T — adds high-res spatial structure to reanalysis meteo; view-time bands align AOD/LST/fire to consistent overpass times.
- **India relevance:** Surface temperature over IGP correlates with PBL stability/heat that modulate PM2.5 accumulation and HCHO photochemistry.
- **Caveats:** Cloud gaps (clear-sky only); LST is skin temp, not 2m air temp — needs transfer relationship; emissivity uncertainty over heterogeneous urban surfaces.

### MCD12Q1 — Land Cover Type Yearly 500m

- **Provider:** NASA LP DAAC
- **Instrument:** MODIS Terra+Aqua
- **Products:** LC_Type1 (IGBP 17-class); LC_Type2 (UMD); LC_Type3 (LAI); LC_Type4 (BGC); LC_Type5 (PFT); LC_Prop1-3 + QA
- **GEE / API IDs:** `MODIS/061/MCD12Q1`
- **Spatial resolution:** 500 m
- **Temporal resolution:** Yearly
- **Latency:** Annual, ~1 yr lag
- **Coverage:** 2001 to present
- **Access:** GEE: MODIS/061/MCD12Q1.
- **Pipeline role:** Static land-use predictor for AQI ML (urban vs rural vs cropland emission regimes) and land-use term in land-use-regression component.
- **Gap-fill / cross-verification role:** Provides land-use covariate that explains spatial PM2.5/HCHO heterogeneity reanalysis cannot; cropland mask separates agricultural-burning HCHO sources from urban/industrial HCHO in hotspot attribution.
- **India relevance:** Distinguishes IGP croplands (stubble-burning fuel), urban Delhi-NCR, and forest classes for fire-emission typing.
- **Caveats:** Annual product — no intra-year change; 500m mixes land covers in fragmented Indian landscapes; class confusion in mosaic cropland.

### MOD13Q1/MYD13Q1 — Vegetation Indices (NDVI/EVI) 250m 16-day

- **Provider:** NASA LP DAAC
- **Instrument:** MODIS Terra (MOD13Q1) / Aqua (MYD13Q1); MOD13A1 500m, MOD13A2 1km
- **Products:** NDVI; EVI; VI Quality / pixel reliability; red/NIR/blue/MIR reflectance; view/sun angles; composite day-of-year
- **GEE / API IDs:** `MODIS/061/MOD13Q1`, `MODIS/061/MYD13Q1`, `MODIS/061/MOD13A2`
- **Spatial resolution:** 250 m (Q1); 500m (A1); 1km (A2)
- **Temporal resolution:** 16-day composite (Terra+Aqua offset to ~8-day combined)
- **Latency:** Standard, ~weeks
- **Coverage:** 2000 to present
- **Access:** GEE: MODIS/061/MOD13Q1 and MODIS/061/MYD13Q1.
- **Pipeline role:** Vegetation/phenology predictor in AQI model; NDVI drop after harvest is a temporal cue for biomass-burning onset in HCHO hotspot timing.
- **Gap-fill / cross-verification role:** Provides surface greenness covariate (biogenic VOC/HCHO context, dust-source bareness) absent from atmospheric datasets; helps separate biogenic vs pyrogenic HCHO and constrain fire-fuel availability.
- **India relevance:** Crop phenology/greenness in Punjab-Haryana indicates pre-harvest vs post-harvest (residue) state preceding stubble-burning HCHO peaks.
- **Caveats:** 16-day compositing smooths rapid post-harvest change; cloud-contaminated composites in monsoon; coarse temporal resolution for daily AQI.

## VIIRS Suite (Suomi-NPP, NOAA-20/21)

*VIIRS suite (Suomi-NPP, NOAA-20/JPSS-1, NOAA-21/JPSS-2): 375m active fire, 6km AOD, DNB nighttime lights, Black Marble, Nightfire — the high-res fire + combustion-proxy backbone for PS3 Objective 2 (HCHO-fire correlation) and AOD gap-fill for Objective 1.*

**Top picks:**

- **VNP14IMG/VJ114/VJ214 375m active fire (FIRMS NRT) — highest-res operational fire, core for Punjab/Haryana stubble detection + HCHO-fire correlation**
- **FIRMS multi-platform VIIRS+MODIS fusion — combined fire-count completeness for fire-HCHO regression**
- **VIIRS Deep Blue AERDB AOD 6km — independent AOD to gap-fill/cross-validate INSAT-3D + cloud gaps for surface AQI**
- **VNP46A2 Black Marble DNB nighttime lights — combustion/anthropogenic-emission proxy and urban AQI covariate**
- **VIIRS Nightfire (VNF) — sub-pixel combustion temperature/FRP, brick-kiln & flaring verification**

**Key findings:**

- 375m VIIRS resolves ~7x more fire area per pixel than MODIS 1km (0.14 km2 vs 1 km2), detecting small/cool agricultural-residue fires that MODIS misses — critical for Punjab/Haryana stubble where many fires are <100m and short-lived.
- Three-platform constellation (S-NPP ~1:30, NOAA-20 ~12:40, NOAA-21) gives ~3 daytime + 3 nighttime VIIRS overpasses/day; adding MODIS Terra/Aqua yields 5 platforms => denser diurnal sampling of the short stubble-burn window (afternoon peak) and fewer missed transient fires.
- FIRMS distributes all NRT fire products (VNP14IMGTDL_NRT, VJ114IMGTDL_NRT, VJ214IMGTDL_NRT + MODIS MCD14DL) with <3h latency via CSV/SHP/WMS API and country shapefiles — directly query India bbox for fire-count time series to correlate with TROPOMI HCHO.
- VIIRS fire fusion improves fire-count COMPLETENESS (more true detections, fewer omission errors), reducing under-counting bias in fire-HCHO regression; per-detection FRP enables emission-weighted (not just count-based) correlation with HCHO columns.
- VIIRS AERDB (Deep Blue, land+SOAR ocean) and AERDT (Dark Target) at 6km provide a second polar AOD source to fill INSAT-3D geostationary AOD cloud/striping gaps and cross-validate the AOD->surface-PM input to the CNN/LSTM AQI model.
- VNP46A2 Black Marble (500m, lunar-BRDF + atmospherically corrected, gap-filled) is a stable combustion/anthropogenic proxy — useful AQI covariate over IGP cities and for flagging persistent combustion sources distinct from seasonal crop fires.
- VIIRS Nightfire (NOAA EOG, V4.0) fits Planck curves to multispectral nighttime radiances for sub-pixel source temperature/size/radiant-heat — verifies brick-kiln/industrial/flaring combustion that the contextual VNP14 fire algorithm or daytime AOD can confuse with crop fires.
- GEE hosts NASA/VIIRS/002/VNP14A1 (1km daily fire), VNP46A2 (NTL), and VNP09/VNP13 surfaces; but native 375m AF and 6km AERDB/AERDT AOD are NOT in GEE swath form — pull those from LAADS DAAC / FIRMS directly for full resolution.

### VIIRS 375m Active Fire (VNP14IMG / VJ114IMG / VJ214IMG)

- **Provider:** NASA LANCE / FIRMS, LAADS DAAC, USFS-NASA
- **Instrument:** VIIRS I-bands (I1-I5, 375m) on Suomi-NPP, NOAA-20, NOAA-21
- **Products:** VNP14IMG (S-NPP standard); VJ114IMG (NOAA-20); VJ214IMG (NOAA-21); VNP14IMGTDL_NRT; VJ114IMGTDL_NRT; VJ214IMGTDL_NRT; FireMask; confidence (low/nominal/high); FRP (MW)
- **GEE / API IDs:** `FIRMS/VNP14IMGTDL_NRT`, `FIRMS/VJ114IMGTDL_NRT`, `FIRMS/VJ214IMGTDL_NRT`, `FIRMS/MODIS_MCD14DL_NRT`
- **Spatial resolution:** 375 m (I-band); detection footprint ~0.14 km2 vs MODIS 1 km2
- **Temporal resolution:** ~Daily per platform; ~3 day + 3 night VIIRS overpasses combined; India overpasses ~10:30/13:30 local +/-
- **Latency:** NRT <3 h via FIRMS/LANCE; standard/science quality days later
- **Coverage:** 2012-01-19 (S-NPP) to present; NOAA-20 from 2018; NOAA-21 from 2023
- **Access:** FIRMS API (CSV/SHP/JSON/WMS), area & country requests: firms.modaps.eosdis.nasa.gov/api/area/ and /country/ (MAP_KEY); LAADS DAAC for VNP14IMG/VJ114IMG granules; archive download CSVs per year.
- **Pipeline role:** Primary fire-activity input for Objective 2: build daily India/IGP fire-count + FRP time series and point clusters; spatial join to TROPOMI HCHO grid for fire-HCHO correlation and DBSCAN/Getis-Ord hotspot detection during burning season (Oct-Nov Punjab/Haryana, Mar-May forest fires).
- **Gap-fill / cross-verification role:** Fills MODIS omission gaps for small/cool ag fires; 375m localizes fires within HCHO 5.5x3.5km pixels; FRP enables emission-weighted correlation; multi-platform overpasses fill diurnal/temporal sampling gaps of any single sensor.
- **India relevance:** Highest-res operational detector of Punjab/Haryana stubble fires (often <100m, brief afternoon burns missed by MODIS 1km); standard product for NW-India crop-fire monitoring.
- **Caveats:** Cloud/smoke obscuration causes omission; afternoon overpass can miss morning/evening burns; large fires saturate; confidence classes differ from MODIS — harmonize before fusion.

### FIRMS multi-platform VIIRS+MODIS fire fusion

- **Provider:** NASA FIRMS (LANCE/EOSDIS)
- **Instrument:** VIIRS (S-NPP/NOAA-20/NOAA-21, 375m) + MODIS (Terra/Aqua, 1km)
- **Products:** combined active-fire archive & NRT; MCD14DL (MODIS); VNP14IMGTDL/VJ114/VJ214; per-detection lat/lon, acq time, confidence, FRP, day/night flag
- **GEE / API IDs:** `FIRMS area API`, `FIRMS country API (IND)`, `FIRMS/MODIS_MCD14DL_NRT`, `FIRMS/VNP14IMGTDL_NRT`
- **Spatial resolution:** 375 m (VIIRS) + 1 km (MODIS), merged point dataset
- **Temporal resolution:** 5 platforms => multiple day/night passes; densest available polar-orbiter fire sampling
- **Latency:** NRT <3 h
- **Coverage:** MODIS 2000-/2002-; VIIRS 2012- ; global incl. all India
- **Access:** FIRMS Web/API: firms.modaps.eosdis.nasa.gov (area/country/CSV/SHP/WMS); free MAP_KEY; ArcGIS/QGIS plugins.
- **Pipeline role:** Single fused fire-count layer maximizing detection completeness for the fire-HCHO correlation regression and transport/back-trajectory source attribution.
- **Gap-fill / cross-verification role:** Cross-platform fusion reduces omission error (clouds, overpass timing, sensor saturation) — improves fire-count completeness so HCHO-fire correlation is not biased low by missed fires; MODIS adds long pre-2012 baseline VIIRS lacks.
- **India relevance:** Recommended PS3 fire source; mosaic VIIRS-375m sensitivity with MODIS temporal baseline across IGP and central-India forest belts.
- **Caveats:** Duplicate detections across overlapping platforms must be de-duplicated; differing pixel sizes/confidence schemes require harmonization before counting.

### VIIRS Deep Blue Aerosol AOD (AERDB_L2)

- **Provider:** NASA LAADS DAAC / LANCE (NRT)
- **Instrument:** VIIRS M-bands on S-NPP (AERDB_L2_VIIRS_SNPP) & NOAA-20
- **Products:** AOT 550nm (Deep Blue land + SOAR ocean); Angstrom exponent; single-scattering albedo; QA flags; AERDB_L2_VIIRS_SNPP; AERDB_L2_VIIRS_NOAA20
- **GEE / API IDs:** `LAADS AERDB_L2_VIIRS_SNPP (not native in GEE)`
- **Spatial resolution:** 6 km at nadir (increases off-nadir)
- **Temporal resolution:** Daily (per platform, ~13:30 LT)
- **Latency:** Standard ~days; NRT via LANCE
- **Coverage:** 2012- (S-NPP); global, strong over bright/arid land (better than Dark Target over IGP dusty/urban surfaces)
- **Access:** LAADS DAAC: ladsweb.modaps.eosdis.nasa.gov (AERDB_L2_VIIRS_SNPP); Earthdata search; NRT AERDB from LANCE.
- **Pipeline role:** Independent polar AOD to feed/validate the AOD->surface-PM2.5 branch of the CNN/LSTM surface-AQI model alongside INSAT-3D AOD and MERRA-2.
- **Gap-fill / cross-verification role:** Fills INSAT-3D geostationary AOD gaps (cloud edges, sun-glint, calibration striping) and provides Deep Blue retrievals over bright IGP/Thar surfaces where Dark Target fails; cross-validates AOD magnitude.
- **India relevance:** Deep Blue is the preferred AOD algorithm over bright/dusty Indo-Gangetic and NW-India surfaces; 6km bridges coarse geostationary and fine needs.
- **Caveats:** 6km coarser than MAIAC 1km; single afternoon snapshot misses diurnal cycle (geostationary INSAT complements); cloud gaps remain.

### VIIRS Dark Target Aerosol AOD (AERDT_L2)

- **Provider:** NASA LAADS DAAC / LANCE (NRT)
- **Instrument:** VIIRS M-bands, S-NPP & NOAA-20
- **Products:** AOT 550nm (Dark Target land+ocean, MODIS-heritage algorithm); fine-mode fraction (ocean); QA; AERDT_L2_VIIRS_SNPP
- **GEE / API IDs:** `LAADS AERDT_L2_VIIRS_SNPP (not native in GEE)`
- **Spatial resolution:** 6 km at nadir
- **Temporal resolution:** Daily per platform
- **Latency:** NRT via LANCE; standard days
- **Coverage:** 2012- ; best over dark vegetated/dense surfaces and ocean
- **Access:** LAADS DAAC: AERDT_L2_VIIRS_SNPP; NRT Dark Target now via LANCE.
- **Pipeline role:** Continuity with MODIS Dark Target AOD time series for the surface-AQI AOD input; pairs with AERDB for full-surface AOD coverage.
- **Gap-fill / cross-verification role:** Complements Deep Blue over dark/vegetated surfaces; MODIS-consistent algorithm enables merging VIIRS+MODIS AOD record to fill temporal/cross-sensor gaps.
- **India relevance:** Works over vegetated S/E India and water bodies; weaker over bright NW IGP where AERDB takes over.
- **Caveats:** Fails/low-quality over bright urban/desert IGP surfaces; 6km coarse; cloud-screened gaps.

### VNP46A2 Black Marble Nighttime Lights (DNB)

- **Provider:** NASA / LAADS DAAC / GSFC Black Marble
- **Instrument:** VIIRS Day-Night Band (DNB), S-NPP (+NOAA-20 in VNP46)
- **Products:** Gap_Filled_DNB_BRDF-Corrected_NTL; Latest_High_Quality_Retrieval; Mandatory_Quality_Flag; Snow_Flag; VNP46A1 (raw); VNP46A3/A4 (monthly/annual)
- **GEE / API IDs:** `NASA/VIIRS/002/VNP46A2`, `NASA/VIIRS/002/VNP46A1`
- **Spatial resolution:** 500 m
- **Temporal resolution:** Daily (A2); monthly/annual composites (A3/A4)
- **Latency:** Standard (days-weeks); NRT Black Marble available
- **Coverage:** 2012-01-19 to present; global
- **Access:** GEE: NASA/VIIRS/002/VNP46A2; LAADS DAAC VNP46A2; Black Marble portal (blackmarble.gsfc.nasa.gov).
- **Pipeline role:** Combustion/anthropogenic-emission and urbanization proxy covariate for the surface-AQI CNN/LSTM (population/activity signal correlating with NO2/PM in IGP cities).
- **Gap-fill / cross-verification role:** Stable nightly proxy filling daytime-only sensor gaps; distinguishes persistent anthropogenic combustion from transient seasonal crop fires; flags lit industrial sources.
- **India relevance:** Maps IGP urban combustion/activity gradients; can highlight brick-kiln belts and Diwali/festival combustion spikes affecting AQI.
- **Caveats:** Light != direct pollutant; lunar/cloud/snow contamination (use QA & gap-filled band); not a quantitative emission measure.

### VIIRS Nightfire (VNF)

- **Provider:** NOAA Earth Observation Group (EOG), Colorado School of Mines / Payne Institute
- **Instrument:** VIIRS M-bands (M7-M13 incl. SWIR/MWIR) + DNB, S-NPP & NOAA-20 (J01)
- **Products:** per-source temperature (K); source size; radiant heat (RH); radiant heat intensity; detection footprints; VNF V3.0 / V4.0 nightly; gas-flare catalogs
- **GEE / API IDs:** `EOG VNF (not in GEE; download from eogdata.mines.edu)`
- **Spatial resolution:** ~750 m M-band; sub-pixel source via Planck-curve fit
- **Temporal resolution:** Nightly global
- **Latency:** Near real-time to days (EOG processing)
- **Coverage:** 2012- ; global; nightly
- **Access:** EOG: eogdata.mines.edu/products/vnf/ (nightly CSV/HDF); flare catalogs annual.
- **Pipeline role:** Verification layer: independent sub-pixel combustion temperature/FRP to characterize and validate fire/combustion sources behind HCHO hotspots, separating crop fires from industrial/flaring/brick-kiln emitters.
- **Gap-fill / cross-verification role:** Detects cool/persistent combustion (brick kilns, flares) missed by VNP14 contextual fire algorithm; nighttime detection fills the daytime gap of standard active-fire and AOD products; temperature retrieval disambiguates source type.
- **India relevance:** Identifies IGP brick-kiln clusters and industrial combustion that confound the fire-HCHO signal; nighttime complement to daytime stubble-fire detection.
- **Caveats:** Not the standard ag-fire product (combustion-physics oriented); coarser M-band; cloud-limited; requires EOG-specific processing knowledge.

### VNP14A1 Thermal Anomalies/Fire Daily 1km (GEE)

- **Provider:** NASA LP DAAC / GEE
- **Instrument:** VIIRS M-bands, Suomi-NPP
- **Products:** FireMask; MaxFRP; QA; sample
- **GEE / API IDs:** `NASA/VIIRS/002/VNP14A1`
- **Spatial resolution:** 1 km (SIN grid)
- **Temporal resolution:** Daily L3
- **Latency:** Standard (days)
- **Coverage:** 2012-01-19 to ~present
- **Access:** GEE: NASA/VIIRS/002/VNP14A1; LP DAAC.
- **Pipeline role:** Convenient gridded GEE fire layer for rapid in-platform fire-HCHO overlay and time-series extraction when staying inside Earth Engine.
- **Gap-fill / cross-verification role:** GEE-native fallback when 375m FIRMS pull is impractical; provides gridded MaxFRP for emission-weighting; coarser than native VNP14IMG so pair with FIRMS 375m for small fires.
- **India relevance:** Quick IGP fire-season screening in GEE before pulling full 375m FIRMS detail.
- **Caveats:** 1km gridded (loses 375m advantage for small ag fires); L3 daily compositing; use FIRMS VNP14IMG for stubble-fire completeness.

## Polar-Orbiting UV-Vis Spectrometers (OMI, GOME-2, SCIAMACHY, GOME-1)

*Polar-orbiting UV-Vis nadir spectrometers (HCHO/NO2/SO2) — the long historical record (1995-present) for contextualizing & cross-verifying Sentinel-5P TROPOMI*

**Top picks:**

- **Aura OMI (OMHCHO/OMNO2/OMSO2/OMAERUV) — 2004-present overlap with TROPOMI, the workhorse for HCHO climatology baseline and direct OMI-vs-TROPOMI cross-validation over India**
- **GOME-2 MetOp-A/B/C (BIRA-IASB/TEMIS/AC SAF) — 2007-present, extends record + independent morning-overpass HCHO for diurnal/intercalibration checks**
- **QA4ECV harmonized HCHO & NO2 (GOME/SCIAMACHY/GOME-2/OMI, 1995-2017) — single consistently-retrieved multi-decade ECV record, ideal for robust hotspot thresholds**
- **SCIAMACHY/Envisat HCHO (BIRA-IASB) — 2002-2012, bridges GOME-1 to OMI/GOME-2 era**
- **GOME-1/ERS-2 HCHO — 1996-2003, anchors the earliest end of the ~30-yr climatology**

**Key findings:**

- OMI L2 NO2/SO2/HCHO are NOT in Google Earth Engine — only ozone (TOMS_MERGED / TOMS_MERGED_V4) is. Pull OMHCHO/OMNO2/OMSO2/OMAERUV from NASA GES DISC (Earthdata login, OPeNDAP/Harmony). GEE-native long record for India hotspots is effectively TROPOMI + S5P; OMI must be ingested as external rasters.
- OMI row anomaly (since 2007, worsening) blocks specific cross-track viewing rows — ALWAYS filter on XTrackQualityFlags / row-anomaly flags before gridding, else striping creates false hotspot artifacts. L3 products (OMSO2e, OMHCHOd) pre-apply best-pixel selection.
- BIRA-IASB retrieves HCHO with a CONSISTENT DOAS algorithm across GOME, SCIAMACHY, GOME-2, OMI, TROPOMI — this single-algorithm chain is what makes a true multi-decade intercalibrated HCHO climatology possible (~1996-present) for defining statistically robust India hotspot thresholds.
- QA4ECV (qa4ecv.eu) delivers harmonized, uncertainty-characterized HCHO & NO2 from GOME/SCIAMACHY/GOME-2/OMI (1995-2017) — the best single source for a consistent baseline distribution; TROPOMI HCHO uses the same QA4ECV/S5P algorithm baseline, enabling apples-to-apples cross-verification.
- Coarse resolution is the main caveat: OMI 13x24 km, GOME-2A 40x40 km (80x40 pre-2013), SCIAMACHY 30x60 km, GOME-1 40x320 km vs TROPOMI 3.5x5.5 km. Use coarse sensors for TEMPORAL anomaly thresholds & trend context, NOT for fine spatial hotspot delineation — that stays TROPOMI's job.
- HCHO has low per-pixel SNR; single-overpass columns are noisy. Build the climatology from MONTHLY/SEASONAL means and oversampled L3 grids to get statistically stable IGP/forest-fire-season baselines; define hotspots as anomalies (e.g. >mean+2sigma or percentile) relative to this multi-year monthly climatology.
- Overpass times differ — OMI/Aura ~13:30 LT (same A-train slot as TROPOMI, best for direct comparison), GOME-2/MetOp ~09:30 LT (morning, lower HCHO/different photochemistry). Use OMI for TROPOMI cross-cal; use GOME-2 to characterize diurnal variation and as independent verification.
- Instrument drift/degradation (Envisat ended 2012; OMI degrading; GOME-2A swath halved to 40 km in 2013 tandem ops) means trends need intercalibration corrections. For India objective-2, biomass-burning HCHO enhancements are large enough that fire-season anomaly detection is robust even at coarse resolution.

### Aura OMI — Formaldehyde (OMHCHO L2 / OMHCHOd L3)

- **Provider:** NASA GSFC / SAO (Smithsonian) — GES DISC
- **Instrument:** OMI (Ozone Monitoring Instrument), Aura platform, UV-Vis pushbroom 270-500 nm; HCHO fit 327.5-356.5 nm
- **Products:** OMHCHO L2 (1-orbit swath HCHO vertical column); OMHCHOd L3 daily 0.1deg gridded; Reference Sector Corrected HCHO column; AMF, column uncertainty, fitting RMS
- **GEE / API IDs:** `GES DISC short name: OMHCHO (V003)`, `GES DISC short name: OMHCHOd (L3 daily)`
- **Spatial resolution:** 13 x 24 km nadir (L2); 0.1deg L3 grid
- **Temporal resolution:** Daily, ~13:30 LT ascending (sun-synchronous), global daily coverage
- **Latency:** OFFL ~days; reprocessed Collection 3/4
- **Coverage:** Oct 2004 - present
- **Access:** NASA GES DISC (Earthdata login): disc.gsfc.nasa.gov OMHCHO_003 / OMHCHOd_003; OPeNDAP, Harmony, earthaccess Python. NOT in Google Earth Engine.
- **Pipeline role:** Primary multi-decade HCHO baseline overlapping TROPOMI (both ~13:30 LT) — core dataset for building India HCHO climatology and direct OMI-vs-TROPOMI cross-validation.
- **Gap-fill / cross-verification role:** Fills pre-2018 temporal gap (2004-2017) before TROPOMI; provides ~14-yr record to set statistically robust IGP/fire-season hotspot thresholds and detect long-term trends TROPOMI alone cannot.
- **India relevance:** Same overpass as TROPOMI -> best for cross-cal over IGP; captures historic crop-residue-burning HCHO enhancements (Oct-Nov Punjab/Haryana).
- **Caveats:** 13x24 km too coarse for fine hotspots; row anomaly striping (filter XTrackQualityFlags); low HCHO SNR needs monthly/seasonal averaging; instrument degradation over time.

### Aura OMI — NO2 (OMNO2 / OMNO2d)

- **Provider:** NASA GSFC — GES DISC
- **Instrument:** OMI / Aura, visible 405-465 nm DOAS
- **Products:** OMNO2 L2 total & tropospheric NO2 column; OMNO2d L3 daily 0.25deg gridded tropospheric NO2; Cloud fraction, AMF, scene flags
- **GEE / API IDs:** `GES DISC: OMNO2 (V004)`, `GES DISC: OMNO2d (L3)`
- **Spatial resolution:** 13 x 24 km nadir; 0.25deg L3
- **Temporal resolution:** Daily ~13:30 LT
- **Latency:** OFFL days; reprocessed collections
- **Coverage:** Oct 2004 - present (V4 current; V3 legacy)
- **Access:** GES DISC (OMNO2_004 / OMNO2d_003), OPeNDAP/Harmony/earthaccess. NOT in GEE (use TROPOMI for GEE NO2).
- **Pipeline role:** Long NO2 record to contextualize TROPOMI NO2 and as auxiliary AQI predictor feature; co-emitted with HCHO from combustion/fires (FNR ratio context).
- **Gap-fill / cross-verification role:** Provides 2004-2017 NO2 history for trend baseline; cross-checks TROPOMI NO2 magnitudes over Indian megacities; HCHO/NO2 ratio (FNR) helps interpret ozone-sensitivity regimes.
- **India relevance:** Captures Delhi-NCR & IGP NO2; FNR with HCHO informs O3 chemistry for AQI objective.
- **Caveats:** Coarse res; row anomaly; tropospheric AMF uncertainty over polluted/aerosol-laden IGP; V3->V4/V5 destriping differences.

### Aura OMI — SO2 (OMSO2 L2 / OMSO2e L3)

- **Provider:** NASA GSFC — GES DISC
- **Instrument:** OMI / Aura, UV 310.5-340 nm (PCA/PBL & multiple a-priori levels)
- **Products:** OMSO2 L2 total column (PBL, TRL, TRM, TRU, STL anchored); OMSO2e L3 daily 0.25deg best-pixel; Volcanic & anthropogenic SO2
- **GEE / API IDs:** `GES DISC: OMSO2 (V004)`, `GES DISC: OMSO2e (L3 V004)`
- **Spatial resolution:** 13 x 24 km; 0.25deg L3
- **Temporal resolution:** Daily ~13:30 LT
- **Latency:** OFFL days
- **Coverage:** Oct 2004 - present (V4 / V3)
- **Access:** GES DISC (OMSO2_004 L2, OMSO2e_004 L3), OPeNDAP/Harmony/earthaccess. NOT in GEE.
- **Pipeline role:** Historic SO2 record to contextualize TROPOMI SO2 over Indian coal/power-plant clusters; auxiliary AQI feature.
- **Gap-fill / cross-verification role:** Fills pre-2018 SO2 baseline for thermal-power-belt point sources (Singrauli, IGP); cross-verifies TROPOMI SO2 hotspots.
- **India relevance:** Detects large Indian SO2 point sources for AQI; long record shows emission trends pre/post FGD mandates.
- **Caveats:** PBL SO2 has high noise/low SNR — needs heavy averaging & only large sources detectable; row anomaly; coarse res.

### Aura OMI — Aerosol Index & Absorbing Aerosol (OMAERUV / OMAERUVd)

- **Provider:** NASA GSFC — GES DISC
- **Instrument:** OMI / Aura, near-UV 354/388 nm two-channel algorithm
- **Products:** UV Aerosol Index (UVAI); Absorbing Aerosol Optical Depth (AAOD); Single Scattering Albedo (SSA); OMAERUVd L3 daily 1deg
- **GEE / API IDs:** `GES DISC: OMAERUV (V003)`, `GES DISC: OMAERUVd (L3)`
- **Spatial resolution:** 13 x 24 km; 1.0deg L3
- **Temporal resolution:** Daily ~13:30 LT
- **Latency:** OFFL days
- **Coverage:** Oct 2004 - present
- **Access:** GES DISC (OMAERUV_003 / OMAERUVd_003). UVAI also available in GEE only for TROPOMI (COPERNICUS/S5P/.../L3_AER_AI), not OMI.
- **Pipeline role:** UV Aerosol Index flags absorbing smoke/dust — key co-tracer to confirm biomass-burning HCHO hotspots and separate smoke from dust in India.
- **Gap-fill / cross-verification role:** Independent smoke detection that corroborates FIRMS fires + HCHO enhancement; long UVAI record contextualizes TROPOMI AER_AI for fire-season transport analysis.
- **India relevance:** Distinguishes IGP crop-burning smoke (absorbing) from Thar dust; supports objective-2 transport/correlation.
- **Caveats:** Coarse 1deg L3; UVAI sensitive to aerosol layer height; row anomaly; not a quantitative AOD.

### GOME-2 on MetOp-A/B/C (HCHO/NO2/SO2) — AC SAF / DLR / BIRA-IASB

- **Provider:** EUMETSAT AC SAF (formerly O3M SAF); operational L2 by DLR, scientific HCHO by BIRA-IASB; distributed via TEMIS
- **Instrument:** GOME-2 UV-Vis grating spectrometer 240-790 nm, MetOp-A/B/C
- **Products:** Tropospheric HCHO VCD (BIRA-IASB DOAS); Tropospheric & total NO2 (TEMIS/DLR); SO2 total column; Total O3; L3 daily/monthly gridded (ESSD 2023 product)
- **GEE / API IDs:** `AC SAF product IDs (O3M SAF) via EUMETSAT data store`, `TEMIS GOME-2 NO2/HCHO archives`, `BIRA-IASB h2co.aeronomie.be HCHO`
- **Spatial resolution:** 80 x 40 km (A/B pre-2013); 40 x 40 km (GOME-2A after Jul-2013 tandem); L3 ~0.25-1deg
- **Temporal resolution:** Daily, ~09:30 LT descending (morning overpass)
- **Latency:** NRT hours; reprocessed/offline
- **Coverage:** MetOp-A 2007-2021; MetOp-B 2012-present; MetOp-C 2018-present
- **Access:** TEMIS (temis.nl), AC SAF (acsaf.org), DLR; BIRA-IASB HCHO at h2co.aeronomie.be. L3 ESSD dataset (Copernicus, 2023) with DOI. NOT in GEE.
- **Pipeline role:** Extends record to 2007 and adds INDEPENDENT morning (09:30 LT) HCHO -> intercalibration check and diurnal context vs OMI/TROPOMI 13:30 LT.
- **Gap-fill / cross-verification role:** Three-satellite constellation gives near-daily morning coverage to verify TROPOMI; same BIRA DOAS algorithm as TROPOMI enables consistent HCHO climatology; fills any TROPOMI outage windows.
- **India relevance:** Adds morning-overpass HCHO over IGP; multi-MetOp record strengthens fire-season climatology baseline.
- **Caveats:** Very coarse (40-80 km) -> only regional HCHO, not point hotspots; GOME-2A swath/res change in 2013 breaks homogeneity; morning vs afternoon HCHO differs photochemically.

### QA4ECV harmonized HCHO & NO2 (GOME/SCIAMACHY/GOME-2/OMI)

- **Provider:** QA4ECV consortium (BIRA-IASB, KNMI, MPIC, Uni Bremen) — qa4ecv.eu / TEMIS
- **Instrument:** Multi-sensor: GOME-1(ERS-2), SCIAMACHY(Envisat), GOME-2(MetOp), OMI(Aura)
- **Products:** Harmonized tropospheric HCHO ECV; Harmonized tropospheric NO2 ECV; Full uncertainty budget & averaging kernels; L2 + L3 monthly grids
- **GEE / API IDs:** `QA4ECV HCHO/NO2 via TEMIS download portal (DOI-referenced)`
- **Spatial resolution:** Native per sensor (40-13 km); L3 gridded
- **Temporal resolution:** Monthly climatology + per-overpass
- **Latency:** Archived/reprocessed
- **Coverage:** 1995 - 2017 (continuous, consistently retrieved)
- **Access:** qa4ecv.eu, TEMIS (temis.nl), with DOIs. TROPOMI S5P HCHO/NO2 use the SAME QA4ECV algorithm baseline.
- **Pipeline role:** Single consistently-retrieved 22-yr multi-sensor record — THE source for a statistically robust India HCHO baseline distribution and for apples-to-apples TROPOMI cross-verification.
- **Gap-fill / cross-verification role:** Stitches GOME-1->SCIAMACHY->GOME-2->OMI with harmonized algorithm+uncertainties, removing inter-sensor bias so hotspot thresholds (mean+Nsigma / percentile) are physically consistent; directly comparable to TROPOMI (same baseline).
- **India relevance:** Provides the rigorous multi-decade IGP HCHO climatology needed to define defensible hotspot thresholds for ISRO PS-3.
- **Caveats:** Ends 2017 (no TROPOMI overlap inside QA4ECV itself — bridge via OMI); coarse native resolutions; averaging-kernel/a-priori smoothing must be matched when comparing.

### SCIAMACHY on Envisat — HCHO (BIRA-IASB)

- **Provider:** ESA Envisat; HCHO retrieval BIRA-IASB; distributed via TEMIS/QA4ECV
- **Instrument:** SCIAMACHY UV-Vis-NIR 240-2380 nm nadir/limb, DOAS HCHO
- **Products:** Tropospheric HCHO VCD; NO2, SO2 total columns; Part of GOME->SCIAMACHY continuous HCHO record
- **GEE / API IDs:** `TEMIS / BIRA-IASB SCIAMACHY HCHO`, `ESA Envisat SCIAMACHY archive`
- **Spatial resolution:** 30 x 60 km nadir
- **Temporal resolution:** ~6-day global; ~10:00 LT descending
- **Latency:** Archived/reprocessed
- **Coverage:** Aug 2002 - Apr 2012 (Envisat contact lost)
- **Access:** TEMIS (temis.nl), BIRA-IASB h2co.aeronomie.be, ESA Envisat archive, QA4ECV. NOT in GEE.
- **Pipeline role:** Bridges GOME-1 (pre-2003) to OMI/GOME-2 era; part of the continuous ~16-yr GOME+SCIAMACHY HCHO climatology.
- **Gap-fill / cross-verification role:** Fills 2002-2007 gap before GOME-2/OMI maturity; consistent BIRA DOAS makes it directly mergeable into the multi-decade baseline.
- **India relevance:** Adds early-2000s IGP HCHO context to lengthen the climatology and trend analysis.
- **Caveats:** Coarse 30x60 km; sparse ~6-day revisit (limb/nadir duty cycle); record ends abruptly 2012; lower SNR.

### GOME-1 on ERS-2 — HCHO/NO2 (earliest record)

- **Provider:** ESA ERS-2; retrieval BIRA-IASB/Uni Bremen; via TEMIS/QA4ECV
- **Instrument:** GOME (Global Ozone Monitoring Experiment) UV-Vis 240-790 nm
- **Products:** Tropospheric HCHO VCD; Tropospheric NO2; Total O3, SO2
- **GEE / API IDs:** `TEMIS / QA4ECV GOME-1 HCHO/NO2`, `ESA ERS-2 GOME archive`
- **Spatial resolution:** 40 x 320 km nadir (coarsest)
- **Temporal resolution:** ~3-day global; ~10:30 LT
- **Latency:** Archived/reprocessed
- **Coverage:** 1996 - 2003 (full global; degraded after 2003 tape-recorder loss)
- **Access:** TEMIS (temis.nl), QA4ECV, ESA ERS archive. NOT in GEE.
- **Pipeline role:** Anchors the earliest (~1996) end of the multi-decade HCHO/NO2 climatology for long-term trend context.
- **Gap-fill / cross-verification role:** Extends baseline back ~30 yr total; useful only for very-large-scale/seasonal trends given coarse footprint.
- **India relevance:** Provides decadal-scale baseline for IGP HCHO trend framing; not for hotspot delineation.
- **Caveats:** Very coarse 40x320 km -> regional/continental scale only, no India sub-regional hotspots; near-global coverage lost after 2003; older calibration.

## Atmospheric IR / Multispectral Sounders (IASI, AIRS, CrIS, MOPITT, TES)

*Atmospheric IR/multispectral sounders (trace-gas profilers) complementary to UV-Vis (TROPOMI/INSAT) — IASI, AIRS, CrIS, MOPITT, TES. Add thermal-IR-sensed gases (NH3, CO, CH4, O3), twice-daily sampling, vertical profiles, and a 25-year CO climate record. Primary value for PS3: NH3+CO as biomass-burning co-tracers to corroborate TROPOMI HCHO fire signals; T/RH profiles for AOD->PM2.5 humidity/vertical correction; gap-fill for TROPOMI cloud/SZA gaps over the Indo-Gangetic Plain.*

**Top picks:**

- **IASI on MetOp-B/C (ULB/LATMOS+AC SAF): NH3 & CO twice daily — best biomass-burning co-tracer corroboration of TROPOMI HCHO over IGP/forest-fire zones**
- **MOPITT on Terra (MOP02J/MOP03J V9): 25-yr multispectral CO total-column record — long baseline + near-surface CO sensitivity for fire transport**
- **AIRS on Aqua (AIRS3STD V7, GEE asset NASA/AIRS/AIRS3STD/006): T/RH/O3/CO/CH4 profiles — drives AOD->PM2.5 humidity & boundary-layer correction**
- **CrIS SNPP/NOAA-20 (CLIMCAPS V2 + ESSPA-NH3): continuity of CO/NH3 + T/RH profiles past IASI/AIRS, 2 platforms/day**
- **TES on Aura (legacy 2004-2018): high-spectral-res O3/CO/NH3 profiles for historical validation & averaging-kernel reference**

**Key findings:**

- NH3 is the key gas these sounders add that TROPOMI/INSAT lack: IASI ANNI-NH3 v4 (with averaging kernels) + CrIS ESSPA-NH3 give twice-daily NH3 columns — a strong crop-residue/biomass-burning co-tracer to corroborate TROPOMI HCHO fire hotspots over the IGP (Oct-Nov) and validate fire-HCHO correlation.
- CO is observed by all five (IASI, AIRS, CrIS, MOPITT, TES), giving an independent multi-sensor CO ensemble to cross-check TROPOMI S5P CO; MOPITT's 25-yr (2000-present) multispectral record is the long baseline for burning-season CO anomalies.
- Overpass diversity fills TROPOMI's single ~13:30 LT, cloud-limited sampling: IASI ~09:30/21:30, AIRS/CrIS ~01:30/13:30, MOPITT ~10:30 LT — denser diurnal sampling and gap-fill over cloudy IGP days.
- AIRS3STD (GEE asset NASA/AIRS/AIRS3STD/006) and CrIS CLIMCAPS deliver satellite-observed T & RH profiles — directly usable for the AOD->PM2.5 hygroscopic-growth (f(RH)) and boundary-layer/vertical correction, and to validate ERA5/IMDAA/MERRA-2 met used in the AQI CNN/LSTM.
- MOPITT multispectral 'Joint' (TIR+NIR, MOP02J/MOP03J V9) uniquely adds near-surface CO sensitivity (~2 DOF), aligning CO better with surface AQI than TIR-only AIRS/CrIS/IASI CO retrievals.
- Access split: IASI science products (NH3/CO) from ULB-LATMOS via AERIS (iasi.aeris-data.fr) + AC SAF/EUMETSAT; AIRS/CrIS/TES from NASA GES DISC; MOPITT from NASA Langley ASDC — all free with Earthdata login; only AIRS L3 is mirrored in Google Earth Engine.
- These are all coarse-footprint (12-45 km) / often 1° L3 sounders — use as predictor covariates, anomaly flags, and verification layers, NOT as the high-res AQI target; the CNN/LSTM should ingest them as auxiliary channels alongside fine-res TROPOMI + INSAT AOD.
- TIR retrievals peak in the free/mid-troposphere with low near-surface DOF; rely on supplied averaging kernels when comparing to surface CPCB AQI, and use NH3's thermal-contrast dependence (better in hot IGP afternoons) to your advantage during burning season.

### IASI (Infrared Atmospheric Sounding Interferometer) on MetOp-A/-B/-C

- **Provider:** EUMETSAT / AC SAF; science products by ULB (Université Libre de Bruxelles) + LATMOS, distributed via AERIS/Ether
- **Instrument:** Nadir Fourier-transform IR spectrometer, 645-2760 cm-1, 0.5 cm-1 resolution
- **Products:** NH3 total column (ANNI-NH3 v4, neural-network, with averaging kernels); CO total column + coarse profile (FORLI-CO / AC SAF operational); HCHO total column; O3 total column & profile; Dust/aerosol flags, SO2; T & H2O profiles
- **GEE / API IDs:** `AERIS/Ether: iasi.aeris-data.fr`, `AC SAF: acsaf.org (NH3 demo, CO)`, `EUMETSAT Data Store (operational L2)`
- **Spatial resolution:** 12 km nadir footprint (circular, 4-pixel matrix); ~25 km off-nadir
- **Temporal resolution:** Twice daily per platform (~09:30 & 21:30 LT desc/asc); 3 MetOps = up to 6 overpasses/day; global daily coverage
- **Vertical sensitivity:** TIR: peak sensitivity in free/mid-troposphere (~3-8 km) for CO; NH3 near-surface in warm, high-thermal-contrast conditions (good over hot IGP afternoons). Use supplied averaging kernels.
- **Latency:** NRT (~3 h) via EUMETCast/AC SAF; reprocessed/offline via AERIS
- **Coverage:** Oct 2007-present (MetOp-A retired 2021; -B 2012-, -C 2018-)
- **Access:** Free. ULB/LATMOS NH3 & CO NetCDF from AERIS portal (iasi.aeris-data.fr, registration); AC SAF operational CO/NH3/O3 via EUMETSAT Data Store + EUMETCast NRT; not in GEE catalog (download NetCDF, regrid to India domain).
- **Pipeline role:** Independent twice-daily NH3 & CO total columns over India. NH3 is a strong combustion/agro-burning tracer absent from TROPOMI's gas suite; CO corroborates fire emissions. Co-locate with TROPOMI HCHO + FIRMS to confirm biomass-burning attribution and feed extra predictor channels (NH3, CO) into the AQI CNN/LSTM.
- **Gap-fill / cross-verification role:** Different overpass time (~09:30 & 21:30 LT) than TROPOMI (~13:30) fills temporal/diurnal gaps and TROPOMI cloud/glint gaps; provides NH3 (a TROPOMI gap) and a second CO estimate to cross-check S5P CO over the IGP.
- **India relevance:** IGP NH3 hotspots (fertilizer + crop-residue burning Oct-Nov) and forest-fire CO plumes (Mar-May Central India/Himalaya) directly corroborate HCHO hotspots.
- **Caveats:** Coarse vertical resolution; NH3 retrieval needs thermal contrast (weak at night/winter dawn); ~12-25 km footprint coarser than TROPOMI 5.5x3.5 km.

### AIRS (Atmospheric Infrared Sounder) on Aqua

- **Provider:** NASA JPL / Sounder SIPS; distributed via GES DISC & Earthdata; mirrored in Google Earth Engine
- **Instrument:** Grating IR spectrometer 3.7-15.4 um (2378 channels) + AMSU microwave
- **Products:** AIRS3STD/AIRS3STM L3 daily/monthly: T & H2O (RH) profiles (24 levels), O3, CO, CH4 total+profile, surface T; AIRX2RET / AIRS2RET L2 retrievals; OLR, cloud properties
- **GEE / API IDs:** `GEE: NASA/AIRS/AIRS3STD/006 (L3 daily)`, `GES DISC: AIRS3STD, AIRS3STM, AIRX2RET`, `Giovanni / OPeNDAP / Earthdata`
- **Spatial resolution:** L2: 45 km nadir; L3 gridded: 1° x 1°
- **Temporal resolution:** Twice daily (01:30 & 13:30 LT); global daily L3
- **Vertical sensitivity:** T/H2O profiles to ~1-2 km vertical resolution lower trop; CO peak sensitivity ~500 hPa; full RH profile enables humidity correction.
- **Latency:** Standard ~ days; L3 daily
- **Coverage:** Sep 2002-present (CO/CH4/O3 retrievals robust mid-trop)
- **Access:** Free. GES DISC (Earthdata login) for L2/L3 NetCDF/HDF; OPeNDAP & Giovanni for subsetting; GEE asset NASA/AIRS/AIRS3STD/006 gives ready-to-use daily T/RH/O3/CO over India with no download.
- **Pipeline role:** Primary source of free-tropospheric T & H2O (RH) profiles to drive the AOD->PM2.5 conversion: humidity (f(RH)) hygroscopic-growth correction and vertical-structure/boundary-layer context complementing ERA5/IMDAA/MERRA-2. Also supplies CO/CH4/O3 cross-check fields.
- **Gap-fill / cross-verification role:** 1:30 AM/PM overpasses bracket TROPOMI; provides T/RH satellite-observed profiles to validate reanalysis met used in the AQI model; CH4 & mid-trop CO fill gases coarsely where TROPOMI is cloud-blocked.
- **India relevance:** 1° L3 is coarse for AQI but ideal as a gridded humidity/temperature predictor and for plume-altitude/transport context over IGP and fire belts.
- **Caveats:** L3 1° too coarse for station-scale AQI (use as covariate, not target); near-surface gas sensitivity weak; cloud-cleared FOV reduces yield.

### CrIS (Cross-track Infrared Sounder) on SNPP & NOAA-20 (JPSS-1)

- **Provider:** NASA Sounder SIPS / NOAA; GES DISC & Earthdata
- **Instrument:** FTS IR sounder (LWIR/MWIR/SWIR) + ATMS microwave
- **Products:** CLIMCAPS V2 L2/L3 (SNDRSNIML2/.. , SNDRJ1IML2): T, H2O, O3, CO, CH4, CO2, SO2, N2O, HNO3 profiles; ESSPA-NH3 V1 (SNDRSNIL2ESPNH3): NH3 profile/column; CLIMCAPS L3 gridded daily (0.5°/1°)
- **GEE / API IDs:** `GES DISC: SNDRSNIML2CCPRET (SNPP), SNDRJ1IML2CCPRET (NOAA-20), SNDR*IML3CDCCP (L3), SNDRSNIL2ESPNH3 (NH3)`, `Earthdata Search / OPeNDAP`
- **Spatial resolution:** L2: ~14 km FOV / FOR; L3: 0.5°-1°
- **Temporal resolution:** Twice daily per platform (~01:30 & 13:30 LT); 2 platforms = 4 overpasses/day
- **Vertical sensitivity:** Similar TIR profile sensitivity to AIRS/IASI: CO mid-trop, NH3 near-surface w/ thermal contrast, T/H2O 1-2 km lower-trop.
- **Latency:** Standard ~1-2 days; NRT variants exist
- **Coverage:** SNPP 2012-present; NOAA-20 2018-present (operational continuity for AIRS/IASI era)
- **Access:** Free via GES DISC (Earthdata login), NetCDF; not in GEE — download & regrid. Use CLIMCAPS for CO/CH4/O3/T/RH and ESSPA-NH3 for ammonia.
- **Pipeline role:** Operational continuity & ensemble for IASI/AIRS: extra daily CO and NH3 columns (multi-platform) to strengthen biomass-burning co-tracer corroboration of HCHO; T/RH profiles add to humidity correction; mirrors AIRS 1:30 overpass.
- **Gap-fill / cross-verification role:** Two same-day platforms (SNPP+NOAA-20) densify sampling and fill TROPOMI cloud gaps; CrIS NH3 (ESSPA) independent of IASI NH3 enables cross-sensor NH3 verification.
- **India relevance:** Guarantees the CO/NH3 co-tracer record continues post-IASI/Aqua over India; multi-overpass helps capture fast-evolving fire plumes in IGP.
- **Caveats:** Coarse footprint; NH3 thermal-contrast dependence; L3 coarse for station AQI.

### MOPITT (Measurements Of Pollution In The Troposphere) on Terra

- **Provider:** NASA/NCAR; NASA Langley ASDC (also Earthdata/GES DISC tools)
- **Instrument:** Gas-correlation radiometer; TIR (4.7 um) + NIR (2.3 um) -> multispectral (TIR+NIR 'Joint')
- **Products:** MOP02J V9 L2 multispectral (TIR+NIR) CO profiles + total column; MOP02T V9 (TIR-only); MOP03J/MOP03TM/MOP03NM V9 L3 daily/monthly gridded CO total column
- **GEE / API IDs:** `ASDC: MOP02J_9, MOP02T_9, MOP03J_9, MOP03TM_9, MOP03NM_9`, `Earthdata: larc-cloud-mop03j-9`, `asdc.larc.nasa.gov/project/MOPITT`
- **Spatial resolution:** 22 km x 22 km nadir; L3 gridded 1° x 1°
- **Temporal resolution:** Near-daily global (~3 days full coverage); 10:30 LT overpass
- **Vertical sensitivity:** Joint TIR+NIR provides 2 independent DOF: near-surface + free-trop CO (better surface sensitivity than AIRS/CrIS/IASI CO).
- **Latency:** Standard product (not NRT); long reprocessed record
- **Coverage:** Mar 2000-present — longest single-sensor satellite CO record (~25 yr)
- **Access:** Free via NASA Langley ASDC (Earthdata login), HDF-EOS; L3 daily gridded CO easiest to use; not in GEE — download/regrid to India.
- **Pipeline role:** Authoritative long-term CO total-column reference. Multispectral (TIR+NIR Joint) retrieval adds near-surface CO sensitivity -> better for boundary-layer fire/pollution CO than TIR-only sounders, strengthening the CO co-tracer for HCHO biomass-burning attribution.
- **Gap-fill / cross-verification role:** 25-yr baseline lets you build CO climatology/anomalies to flag burning-season excess; cross-validates TROPOMI & IASI CO; 10:30 LT overpass adds a third diurnal sample.
- **India relevance:** Long record captures IGP & Central-India fire-season CO trends; NIR near-surface sensitivity aligns CO with surface AQI better than pure TIR.
- **Caveats:** Coarse 22 km / 1°; near-daily (not twice-daily); Terra aging; surface-level retrieval still has limited DOF.

### TES (Tropospheric Emission Spectrometer) on Aura (legacy)

- **Provider:** NASA JPL; GES DISC / ASDC archive
- **Instrument:** High-spectral-resolution IR FTS (limb/nadir)
- **Products:** O3 profiles; CO profiles; NH3 profiles/column (early NH3 IR product); CH4, HDO, T/H2O (special obs)
- **GEE / API IDs:** `GES DISC: TES TL2/TL3 O3, CO, NH3 products`, `ASDC archive`
- **Spatial resolution:** ~5 x 8 km footprint, sparse targeted sampling
- **Temporal resolution:** 16-day global survey (limited duty cycle); not daily
- **Vertical sensitivity:** Best legacy vertical resolution among nadir IR sounders for O3/CO; pioneered satellite NH3 profiling.
- **Latency:** Archive only (mission ended 2018)
- **Coverage:** 2004-2018 (global surveys reduced after ~2010)
- **Access:** Free archival via GES DISC/Earthdata, HDF; historical use only (no ongoing data).
- **Pipeline role:** Legacy high-spectral-res O3/CO/NH3 reference: useful for historical validation, averaging-kernel methodology, and establishing pre-2018 IGP burning-season baselines; not for operational daily AQI.
- **Gap-fill / cross-verification role:** Historical cross-calibration anchor for IASI/CrIS NH3 & CO retrievals and for long-term trend context predating TROPOMI (2018+).
- **India relevance:** Provides pre-TROPOMI-era O3/CO/NH3 profiles over India for trend/validation studies.
- **Caveats:** Mission ended 2018; sparse spatial sampling; coarse temporal — not suitable for daily maps.

## Aerosol-Profile & Wind Satellites (CALIPSO, MISR, EarthCARE, Aeolus, GCOM-C, PARASOL)

*Aerosol-profile & wind satellites for vertical structure / transport (AOD->PM2.5 vertical correction + HCHO/smoke transport) — BAH 2026 PS3*

**Top picks:**

- **CALIPSO CALIOP (aerosol vertical profile + scale height for AOD->PM2.5 correction)**
- **MISR + MINX (plume injection height for smoke/HCHO transport)**
- **EarthCARE ATLID (current-era 355nm HSRL profile, CALIPSO successor 2024+)**
- **Aeolus ALADIN / Aeolus-2 (wind profiles for transport, archive 2018-2023)**
- **GCOM-C SGLI (1km daily AOD gap-fill, in GEE)**

**Key findings:**

- AOD->PM2.5 needs vertical correction: surface PM2.5 ~ AOD/(H*f(RH)), where scale height H (CALIOP H63 = altitude holding 63% of column extinction) is a dominant error. Over IGP, AOD>0.7 with lofted smoke (2-5km) breaks AOD-PM2.5 correlation unless corrected.
- CALIPSO/CALIOP (532+1064nm, 30-60m vert, 333m horiz, 16-day, 2006-2023) = canonical vertical-profile + aerosol-subtype source. NOT in GEE; access via NASA ASDC/Earthdata (L2 05kmAPro V4-51). Sparse nadir track -> use as seasonal extinction/scale-height climatology, not daily per-pixel.
- MISR (Terra, 9 angles, 275m) + MINX retrieve smoke/dust PLUME INJECTION HEIGHT + plume winds stereoscopically -> feeds HYSPLIT transport for HCHO+smoke. MISR L2 aerosol (AOD+type, 4.4km) IS in GEE (NASA/MISR/MIL2ASAE/006); MINX heights from JPL + DLR climatology.
- EarthCARE/ATLID (355nm HSRL, May 2024) = current-era CALIPSO successor: extinction/backscatter/depolarization + classification (A-AER/A-EBD/A-TC). L2a open since Mar 2025 via ESA OADS. Refresh/validate profile climatology for 2024+; depolarization separates smoke vs dust.
- Aeolus/ALADIN (355nm Doppler wind lidar, 2018-08 to 2023-04): only spaceborne WIND PROFILES (HLOS Rayleigh clear-air + Mie cloud/aerosol). Archive via ESA VirES/aeolus.services. Use for archive validation of ERA5/IMDAA winds aloft; Aeolus-2 operational in development.
- GCOM-C/SGLI (JAXA, 250m-1km, ~2-day, 2017+): daily AOD500 + Angstrom -> independent gap-fill/cross-check for INSAT-3D & MODIS AOD over India; L3 in GEE (JAXA/GCOM-C/L3), L2 ARNP via G-Portal.
- PARASOL/POLDER-3 (polarized 490/670/865nm, 2004-2013, archive) + GRASP give fine-mode AOD & aerosol type -> separates fine smoke from coarse dust. Historical only; use for algorithm training/fine-mode priors via ICARE / GRASP-Open.

### CALIPSO CALIOP Lidar L2 Aerosol Profile / Layer (V4-51)

- **Provider:** NASA LaRC / CNES
- **Instrument:** CALIOP (dual-wavelength 532+1064nm polarization-sensitive lidar)
- **Products:** L2 05kmAPro (extinction/backscatter profiles); L2 05kmALay (layer top/base, AOD); VFM Vertical Feature Mask; Aerosol Subtype (dust, polluted dust, elevated smoke, polluted/clean continental, marine); Aerosol Scale Height H63; L3 Aerosol Profile monthly
- **GEE / API IDs:** `ASDC: CAL_LID_L2_05kmAPro-Standard-V4-51`, `ASDC: CAL_LID_L2_05kmALay-Standard-V4-51`, `ASDC: CAL_LID_L3_APro-Standard`
- **Spatial resolution:** 333m horiz native; L2 at 5km; 30-60m vertical
- **Temporal resolution:** 16-day repeat, nadir curtain only (sparse)
- **Latency:** Archive (ended 2023-08-01); reprocessed V4-51
- **Coverage:** 2006-06 to 2023-08, global incl. India
- **Access:** NASA ASDC / Earthdata (CAL_LID_L2_05kmAPro-Standard-V4-51), HDF/netCDF via ASDC Search & Subset + OPeNDAP. NOT in GEE.
- **Pipeline role:** Provides aerosol scale height H and vertical extinction shape to convert column AOD (INSAT/MODIS) to surface PM2.5 -- corrects the key vertical error in objective 1.
- **Gap-fill / cross-verification role:** Fills vertical-distribution gap unmeasured by any column sensor; verifies whether high AOD is surface haze (counts for PM2.5) vs elevated transported smoke (does not).
- **India relevance:** Quantifies lofted smoke/dust layers over IGP (2-5km) in burning season; build seasonal extinction-profile & scale-height climatology per IGP/fire region.
- **Caveats:** Sparse nadir track -> not daily/per-pixel; use as climatology/prior. Mission ended 2023; for 2024+ use EarthCARE.

### MISR L2 Aerosol + MINX Plume Height

- **Provider:** NASA JPL / Terra; DLR (climatology)
- **Instrument:** MISR (9 along-track cameras 0-70deg, 4 bands, 275m)
- **Products:** L2 Aerosol AOD (558nm); Aerosol type / size & non-spherical fraction; MINX stereo plume injection height + plume-level winds; MISR Plume Height Climatology (>23000 plumes)
- **GEE / API IDs:** `NASA/MISR/MIL2ASAE/006 (aerosol, GEE)`, `ASDC: MIL2ASAE`
- **Spatial resolution:** 275m imaging; L2 aerosol 4.4km; plume height ~1.1km
- **Temporal resolution:** Terra ~9-day repeat (2-4d mid-lat), ~10:30 local
- **Latency:** Operational AOD; MINX heights manual/semi-auto
- **Coverage:** 2000-present (Terra); MINX event-based
- **Access:** GEE for L2 aerosol; NASA ASDC for HDF; MINX tool + DLR WDC-RSAT climatology
- **Pipeline role:** Aerosol-type AOD cross-check for surface AQI; multi-angle constrains aerosol model.
- **Gap-fill / cross-verification role:** Injection HEIGHT for HCHO/smoke transport (obj 2): initializes HYSPLIT/FLEXPART; verifies if fire emissions stay in boundary layer (local AQI) or loft and transport downwind.
- **India relevance:** Aerosol type discriminates smoke/dust/pollution over IGP; injection heights for crop-residue & forest fires feed transport modeling.
- **Caveats:** Narrow 380km swath -> infrequent coverage of a given fire; MINX needs operator interaction per plume.

### EarthCARE ATLID L2a (A-PRO: A-AER/A-EBD/A-TC)

- **Provider:** ESA / JAXA
- **Instrument:** ATLID (355nm high-spectral-resolution lidar, polarization)
- **Products:** A-EBD: extinction, backscatter, depolarization profiles; A-AER large-scale aerosol; A-TC target classification; A-ICE ice microphysics; ATL_CLA_2A optical characteristics
- **GEE / API IDs:** `ESA: ATL_AER_2A / ATL_EBD_2A / ATL_TC__2A`, `OADS JAXAL2Validated collection`
- **Spatial resolution:** ~285m vertical (to 100m near surface); ~1-10km along-track avg
- **Temporal resolution:** Sun-synch ~10:30; nadir curtain (sparse)
- **Latency:** L2a open since 2025-03; L2b H2 2025
- **Coverage:** 2024-05 onward, global incl. India
- **Access:** ESA EO OADS (ec-pdgs-dissemination1.eo.esa.int/oads). Not in GEE.
- **Pipeline role:** Refreshes/validates vertical extinction-profile & scale-height climatology used in AOD->PM2.5 correction for 2024+ data.
- **Gap-fill / cross-verification role:** Fills CALIPSO post-2023 gap; HSRL extinction verifies CALIPSO scale heights; depolarization separates smoke (low) from dust (high).
- **India relevance:** Current-era replacement for CALIPSO profiles over India for 2024+ burning seasons; HSRL gives accurate extinction without lidar-ratio assumption.
- **Caveats:** New mission; sparse nadir track; validation ongoing (PollyNET). 355nm differs from CALIPSO 532nm sensitivity.

### Aeolus ALADIN L2B Wind Profiles (+ Aeolus-2)

- **Provider:** ESA / Airbus
- **Instrument:** ALADIN (355nm Doppler wind lidar)
- **Products:** L2B HLOS wind (Rayleigh clear-air); L2B HLOS wind (Mie cloud/aerosol); L2A aerosol/cloud optical properties
- **GEE / API IDs:** `ESA: ALD_U_N_2B (L2B wind)`, `ESA: ALD_U_N_2A (aerosol)`, `VirES API / aeolus.services`
- **Spatial resolution:** ~87km horiz integration; 0.25-2km vertical bins; surface to ~30km
- **Temporal resolution:** Sun-synch ~7-day repeat; single LOS
- **Latency:** Archive (2018-08 to 2023-04); reprocessed baselines
- **Coverage:** 2018-2023 global; Aeolus-2 operational planned this decade
- **Access:** ESA VirES for Aeolus (aeolus.services), ESA EO catalogue; netCDF/DBL
- **Pipeline role:** Independent verification of reanalysis wind fields driving transport analysis (objective 2).
- **Gap-fill / cross-verification role:** Fills upper-air wind observation gap; HLOS winds verify reanalysis transport direction/speed for plume trajectories. Live runs use ERA5/IMDAA; Aeolus = archive validation.
- **India relevance:** Wind profiles over India/IGP to validate ERA5/IMDAA/MERRA-2 winds for HCHO/smoke transport, esp. aloft where reanalysis is weakly constrained.
- **Caveats:** Only HLOS (one component), not full vector; sparse track; mission ended 2023 -> archive only until Aeolus-2.

### GCOM-C / SGLI L2-L3 Aerosol (AROT/ARAE)

- **Provider:** JAXA
- **Instrument:** SGLI (250m-1km multispectral + polarization 380/670/865)
- **Products:** AROT AOD 500nm land+ocean; ARAE Angstrom Exponent (380/500); L2 ARNP aerosol; L3 gridded
- **GEE / API IDs:** `JAXA/GCOM-C/L3 namespace (aerosol L3)`, `G-Portal: GCOM-C SGLI L2 ARNP`
- **Spatial resolution:** 250m-1km (aerosol L2 1km)
- **Temporal resolution:** ~2-day global, ~10:30 local
- **Latency:** NRT to standard
- **Coverage:** 2017-12 onward, global incl. India
- **Access:** JAXA G-Portal (free, registration); L3 in GEE
- **Pipeline role:** Supplementary daily AOD input + cross-sensor consistency for surface AQI regression/CNN.
- **Gap-fill / cross-verification role:** Fills spatial/temporal gaps in INSAT-3D AOD (cloud/glint, coarse res); Angstrom adds fine/coarse info absent from single-band AOD.
- **India relevance:** Independent daily 1km AOD over India to cross-check/gap-fill INSAT-3D & MODIS AOD; finer than INSAT for urban gradients.
- **Caveats:** Polar single overpass (vs INSAT geostationary diurnal); cloud gaps remain.

### PARASOL / POLDER-3 (GRASP) Polarized Aerosol

- **Provider:** CNES / ICARE / GRASP-Open
- **Instrument:** POLDER-3 (multi-angle, polarized 490/670/865nm, 9 bands)
- **Products:** Fine-mode & coarse-mode AOD; Aerosol type, SSA, refractive index (GRASP); Fine-mode fraction; Spectral AOD 443-1020
- **GEE / API IDs:** `ICARE: PARASOL/POLDER-3 L2/L3`, `GRASP-Open POLDER data release`
- **Spatial resolution:** ~6km
- **Temporal resolution:** Multi-angle per overpass; 2004-2013 archive
- **Latency:** Archive only (mission 2004-2013)
- **Coverage:** 2004-2013 global incl. India
- **Access:** ICARE Data Center; GRASP-Open POLDER release. Not in GEE.
- **Pipeline role:** Training data / fine-mode-fraction priors for aerosol-type-aware AOD->PM2.5 models.
- **Gap-fill / cross-verification role:** Verifies fine vs coarse partition that single-band AOD cannot resolve; informs aerosol model selection.
- **India relevance:** Historical fine-mode AOD over IGP separates combustion/smoke fine aerosol from coarse dust -- key for attributing AQI/HCHO sources.
- **Caveats:** Historical only (ended 2013) -> no current data; use for method development & climatological priors, not daily ops.

## Atmospheric-Composition Reanalysis & Forecasts (CAMS, MERRA-2, GEOS-CF, NAAPS, SILAM)

*Global atmospheric-composition reanalysis & forecast models (gap-free 3D priors) for surface AQI estimation and HCHO hotspot detection over India — BAH 2026 PS3*

**Top picks:**

- **CAMS EAC4 reanalysis (cams-global-reanalysis-eac4) — primary gap-free speciated prior**
- **MERRA-2 M2T1NXAER (GES DISC, GEE NASA/GSFC/MERRA/aer/2) — hourly speciated PM/AOD prior**
- **NASA GEOS-CF (aqc collection, OPeNDAP) — hourly 0.25deg surface NO2/O3/PM2.5/CO forecast**
- **CAMS GFAS (cams-global-fire-emissions-gfas) — FRP-based fire emissions for HCHO/fire constraint**
- **CAMS NRT global forecast (ECMWF/CAMS/NRT on GEE; cams-global-atmospheric-composition-forecasts on ADS)**

**Key findings:**

- EAC4 and MERRA-2 are GAP-FREE 4D fields (no cloud/orbit holes) — ideal priors to fill TROPOMI/INSAT/MODIS cloud and swath gaps and to supply speciated surface PM (BC/OC/dust/sulfate/sea-salt) that satellite columns cannot resolve directly.
- EAC4 native ~80 km (0.75deg, regridded to 0.5deg on ADS), 3-hourly, 2003-present, ~5 month latency; assimilates MODIS/TROPOMI/IASI/GOME-2 — strong physically-consistent prior but too coarse for IGP gradients, use as downscaling input not final AQI.
- MERRA-2 M2T1NXAER: 0.5x0.625deg, HOURLY (00:30-23:30 UTC center), 1980-present, ~3-week latency; assimilates MODIS/AVHRR/MISR/AERONET AOD — best for hourly aerosol/speciated PM2.5 features feeding CNN-LSTM; available directly in GEE.
- GEOS-CF gives 0.25deg HOURLY surface NO2/O3/SO2/CO/PM2.5/PM10 (aqc) as 5-day forecast + replay — higher-res composition prior than CAMS/MERRA-2 and the strongest single gap-free surface-AQI feature/benchmark; OPeNDAP access, no assimilation of in-situ AQ so independent validation target.
- GFAS (0.1deg daily, FRP->dry-matter->40 species incl. HCHO precursors, CO, OC, BC + plume injection height) gives an EMISSION-based fire constraint complementary to FIRMS hotspot counts — correlate GFAS biomass-burning emissions with TROPOMI HCHO enhancements for objective-2 attribution and transport source terms.
- CAMS global forecast (0.4deg, hourly to T+120, 2015-present, ~day latency) provides NRT gap-free prior incl. HCHO and dust for operational daily AQI when EAC4 latency is too long; assimilates TROPOMI SO2/CO/O3/NO2 + MODIS AOD.
- Workflow: use CAMS/MERRA-2/GEOS-CF reanalysis fields as predictor channels (gap-free priors) alongside INSAT-3D AOD + TROPOMI columns + ERA5/IMDAA met into CNN/LSTM regressors to CPCB surface AQI; reanalysis fills pixels where satellites are cloud-masked.
- NAAPS (US Navy, 1/3deg, 6-hourly, aerosol-only: sulfate/dust/smoke/sea-salt, assimilates MODIS/VIIRS AOD) and SILAM (FMI, ~0.1-0.5deg European/global chemistry-fire forecast incl. HCHO/PM) are independent secondary priors for ensemble spread and cross-verification of CAMS aerosol/fire fields.

### CAMS Global Reanalysis EAC4

- **Provider:** Copernicus Atmosphere Monitoring Service (ECMWF)
- **Instrument:** Reanalysis (IFS-COMPO, 4D-Var) assimilating MODIS/PMAp AOD, TROPOMI/OMI/GOME-2/SCIAMACHY (NO2,SO2,HCHO,CO,O3), IASI/MLS
- **Products:** PM2.5 (particulate_matter_2.5um); PM10 (particulate_matter_10um); NO2; SO2; CO; O3 (total column + profile); HCHO; AOD 550nm (total/dust/OM/BC/SO4/sea-salt); dust AOD; organic matter aerosol mixing ratio; 60 model levels 3D
- **GEE / API IDs:** `cams-global-reanalysis-eac4`, `cams-global-reanalysis-eac4-monthly`
- **Spatial resolution:** ~80 km native (0.75deg), provided 0.5deg/0.75deg on ADS
- **Temporal resolution:** 3-hourly (analysis 00/12 UTC, +3..+9 forecast steps)
- **Latency:** ~4-5 months (consolidated reanalysis)
- **Coverage:** Global, 2003-present
- **Access:** ADS via cdsapi; dataset id 'cams-global-reanalysis-eac4' (single levels) and '-eac4-monthly'; pressure/model-level variants. Need ADS API key (ads.atmosphere.copernicus.eu). NetCDF/GRIB. Also on WEkEO.
- **Pipeline role:** Primary gap-free speciated 3D prior: supplies cloud-free NO2/SO2/CO/O3/HCHO columns + speciated surface PM as predictor channels and to backfill masked satellite pixels for daily India AQI maps.
- **Gap-fill / cross-verification role:** Fills TROPOMI cloud/orbit gaps and INSAT/MODIS cloud gaps with physically-consistent assimilated fields; provides HCHO + speciated PM where satellites are missing; coarse so use as downscaling target not final output.
- **India relevance:** Captures IGP winter aerosol loading and dust; consistent multi-pollutant priors for CNN/LSTM over India.
- **Caveats:** ~80km too coarse for IGP/urban gradients; ~5-month latency unsuitable for NRT — pair with CAMS forecast/GEOS-CF for recent dates.

### CAMS Global Atmospheric Composition Forecasts (NRT)

- **Provider:** Copernicus Atmosphere Monitoring Service (ECMWF)
- **Instrument:** IFS-COMPO forecast; initial conditions assimilate TROPOMI (SO2,CO,O3,NO2), MODIS/PMAp AOD, IASI, GOME-2
- **Products:** NO2; SO2; CO; O3; HCHO; PM2.5; PM10; AOD components (dust/OM/BC/SO4/sea-salt); surface + 3D model levels
- **GEE / API IDs:** `cams-global-atmospheric-composition-forecasts`, `ECMWF/CAMS/NRT`
- **Spatial resolution:** 0.4deg x 0.4deg (~44 km)
- **Temporal resolution:** Hourly to lead time +120 h; two cycles/day (00/12 UTC)
- **Latency:** ~same day (forecast issued ~hours after base time)
- **Coverage:** Global, 2015-present
- **Access:** ADS cdsapi dataset id 'cams-global-atmospheric-composition-forecasts'; GRIB/NetCDF. Also GEE NRT collection ECMWF/CAMS/NRT (AOD + PM + species).
- **Pipeline role:** NRT gap-free prior for operational daily AQI when EAC4 latency too long; supplies HCHO/dust/PM forecast fields and transport context.
- **Gap-fill / cross-verification role:** Fills recent-date gaps (EAC4 not yet available) and cloud-masked satellite pixels; hourly diurnal structure for LSTM temporal features.
- **India relevance:** Operational dust + biomass-burning transport over IGP; HCHO precursor fields for hotspot-season context.
- **Caveats:** Forecast (not analysis) — larger error than EAC4; 0.4deg still coarse; GEE ECMWF/CAMS/NRT discontinued Oct 2025, prefer ADS for current data.

### MERRA-2 Aerosol Diagnostics (M2T1NXAER / tavg1_2d_aer_Nx)

- **Provider:** NASA GMAO (GES DISC)
- **Instrument:** GEOS-5.12.4 reanalysis with GOCART aerosols; assimilates MODIS/MISR/AVHRR + AERONET AOD
- **Products:** AOD 550nm total (TOTEXTTAU) + scattering; Black carbon col/surf mass (BCSMASS); Organic carbon (OCSMASS); Dust (DUSMASS, DUSMASS25); Sulfate (SO4SMASS); Sea salt (SSSMASS, SSSMASS25); PM2.5 derivable from species; column mass densities
- **GEE / API IDs:** `NASA/GSFC/MERRA/aer/2`, `M2T1NXAER`, `GES DISC OPeNDAP goldsmr4.gesdisc.eosdis.nasa.gov`
- **Spatial resolution:** 0.5deg lat x 0.625deg lon
- **Temporal resolution:** Hourly time-averaged (centers 00:30..23:30 UTC)
- **Latency:** ~3 weeks after month end
- **Coverage:** Global, 1980-present
- **Access:** GES DISC OPeNDAP/HTTPS (Earthdata login + .netrc); xarray/pydap; also Earthdata Harmony subsetting. GEE: NASA/GSFC/MERRA/aer/2.
- **Pipeline role:** Hourly gap-free speciated aerosol prior: derive PM2.5 = 1.375*SO4+1.6*OC+BC+DUSMASS25+0.5*SSSMASS25 as a model feature; AOD-to-PM bridge with INSAT AOD.
- **Gap-fill / cross-verification role:** Provides hourly aerosol fields where INSAT/MODIS AOD is cloud-masked; speciation (dust vs smoke vs sulfate) that single-band satellite AOD cannot give — key for AQI composition.
- **India relevance:** Standard reference for IGP speciated aerosol; widely used to separate crop-residue smoke (OC/BC) from Thar dust.
- **Caveats:** Aerosol-only (no NO2/HCHO/O3); 0.5deg coarse; assimilates only AOD (no surface PM) so surface values biased in some regimes.

### NASA GEOS-CF (Composition Forecast v1)

- **Provider:** NASA GMAO
- **Instrument:** GEOS + GEOS-Chem (full tropospheric chemistry); meteorology from GEOS-FP, no in-situ AQ assimilation
- **Products:** Surface NO2; O3; SO2; CO; PM2.5; PM10 (aqc collection); 3D chemistry (chm); met (met) collections; HCHO and ~250 species available
- **GEE / API IDs:** `aqc_tavg_1hr_g1440x721_v1`, `chm_tavg_1hr_g1440x721_v1`, `portal.nccs.nasa.gov/datashare/gmao/geos-cf`
- **Spatial resolution:** 0.25deg x 0.25deg (~25 km)
- **Temporal resolution:** Hourly (and 15-min inst.); daily 5-day forecast + replay analysis
- **Latency:** Forecast same day (~17 UTC); replay analysis lags
- **Coverage:** Global, 2018-present
- **Access:** OPeNDAP at portal.nccs.nasa.gov/datashare/gmao/geos-cf and dataportal.nccs.nasa.gov; THREDDS; NetCDF via xarray. No login. Collections: aqc_tavg_1hr_g1440x721_v1 (surface AQ), chm_inst (3D), met_tavg.
- **Pipeline role:** Highest-res gap-free surface-AQI prior: hourly 0.25deg NO2/O3/PM2.5/CO as strong predictor channels and as an independent (no-AQ-assimilation) benchmark for CPCB validation.
- **Gap-fill / cross-verification role:** Fills cloud/orbit gaps with full-chemistry surface concentrations at finer scale than CAMS/MERRA-2; supplies surface-level fields (vs satellite columns) directly comparable to CPCB.
- **India relevance:** 0.25deg resolves IGP corridor better; independent of CAMS for ensemble/cross-check over India.
- **Caveats:** Free-running chemistry (no AQ obs assimilation) -> regional biases need CPCB bias-correction; PM2.5 from GOCART speciation; HCHO needs chm collection (larger files).

### CAMS GFAS (Global Fire Assimilation System, biomass-burning emissions)

- **Provider:** Copernicus Atmosphere Monitoring Service (ECMWF)
- **Instrument:** MODIS Terra+Aqua Fire Radiative Power -> dry matter -> emission factors
- **Products:** FRP; CO emission; HCHO/precursor (NMVOC) emissions; OC/BC/PM2.5 emission; SO2/NOx emission; 40 pyrogenic species; smoke plume injection height (APT/MAMI)
- **GEE / API IDs:** `cams-global-fire-emissions-gfas`, `ECMWF/CAMS/GFAS_V1_2`
- **Spatial resolution:** 0.1deg x 0.1deg
- **Temporal resolution:** Daily averaged
- **Latency:** ~near-real-time (1-2 days)
- **Coverage:** Global, 2003-present
- **Access:** ADS cdsapi dataset id 'cams-global-fire-emissions-gfas'; GRIB/NetCDF. GEE: ECMWF/CAMS/GFAS_V1_2 (frpfire/cofire/etc.).
- **Pipeline role:** Emission-based fire constraint for objective-2: regress/correlate GFAS biomass-burning emissions (CO, NMVOC, OC) against TROPOMI HCHO enhancements; source term for transport analysis.
- **Gap-fill / cross-verification role:** Quantitative emission complement to FIRMS hotspot counts (intensity, not just detection); injection height informs vertical transport of HCHO precursors; fills attribution gap between fire pixels and observed HCHO columns.
- **India relevance:** Directly quantifies Punjab/Haryana crop-residue and NE/central forest-fire emissions driving IGP HCHO/PM.
- **Caveats:** MODIS-FRP based -> misses small/cloud-obscured fires and overpass-time bias; daily mean only; 0.1deg.

### NAAPS (Navy Aerosol Analysis and Prediction System)

- **Provider:** US Naval Research Laboratory / FNMOC
- **Instrument:** Aerosol transport model assimilating MODIS + VIIRS AOD (NAAPS-RA reanalysis)
- **Products:** Total AOD; sulfate AOD/conc; dust AOD/conc; smoke AOD/conc; sea-salt AOD/conc; surface concentrations
- **GEE / API IDs:** `NAAPS-RA`, `usgodae.org NAAPS`
- **Spatial resolution:** ~1/3deg (reanalysis); 0.25deg forecast
- **Temporal resolution:** 6-hourly
- **Latency:** NRT forecast; reanalysis periodic
- **Coverage:** Global, 2003-present (reanalysis); NRT forecast
- **Access:** NRL Monterey usgodae/NAAPS portals (HTTP/THREDDS); reanalysis via NRL request. NetCDF.
- **Pipeline role:** Independent secondary aerosol prior for ensemble spread and cross-verification of CAMS/MERRA-2 dust/smoke speciation.
- **Gap-fill / cross-verification role:** Provides an alternative gap-free smoke/dust field to corroborate biomass-burning aerosol over IGP during burning season.
- **India relevance:** Useful for dust vs smoke discrimination over IGP; secondary/optional.
- **Caveats:** Aerosol-only; coarser; access less open than CAMS/NASA.

### SILAM (System for Integrated modeLling of Atmospheric coMposition)

- **Provider:** Finnish Meteorological Institute (FMI)
- **Instrument:** Chemistry-transport + fire (IS4FIRES FRP) model; CAMS regional ensemble member
- **Products:** NO2; O3; SO2; CO; PM2.5/PM10; HCHO/VOC; dust; pollen; fire smoke
- **GEE / API IDs:** `silam.fmi.fi thredds`, `SILAM global v5`
- **Spatial resolution:** ~0.1-0.5deg (global ~0.5deg)
- **Temporal resolution:** Hourly forecast
- **Latency:** Same-day forecast
- **Coverage:** Global + regional, NRT
- **Access:** FMI THREDDS (silam.fmi.fi) OPeNDAP/NetCDF; open.
- **Pipeline role:** Independent full-chemistry prior (incl. HCHO) for ensemble and to cross-check CAMS HCHO/fire-smoke fields in objective-2.
- **Gap-fill / cross-verification role:** Alternative gap-free HCHO + smoke transport field; verifies CAMS/GEOS-CF HCHO hotspot patterns and fire-driven transport.
- **India relevance:** Provides independent HCHO/fire-smoke transport for IGP hotspot attribution.
- **Caveats:** Coarser global config; less validated over India than CAMS; secondary.

## Meteorological Reanalysis (ERA5, IMDAA, MERRA-2, GFS/NCEP)

*Meteorological Reanalysis (ML predictors + transport): ERA5, IMDAA, MERRA-2, GFS/NCEP*

**Top picks:**

- **IMDAA (NCMRWF, 12 km) — best India-specific reanalysis; native predictor grid for IGP/forest-fire zones**
- **ERA5 single-levels via CDS API (BLH + pressure-level winds for transport)**
- **ERA5 on GEE (ECMWF/ERA5_LAND/HOURLY + ECMWF/ERA5/HOURLY) — fast cloud predictor extraction at CPCB points**
- **MERRA-2 (M2T1NXSLV + M2T1NXFLX) — PBLH/radiation cross-check**
- **GFS/NCEP (NOAA/GFS0P25) — near-real-time met for operational daily AQI maps**

**Key findings:**

- EXACT AQI-ML met predictors: boundary_layer_height (BLH/PBLH — vertical dilution, strongest met driver of surface PM), 10m wind speed=sqrt(u10^2+v10^2) & dir=atan2(-u,-v), temperature_2m, RH from T2m+dewpoint via Magnus, total_precipitation (washout), SSRD/SSR, surface_pressure/MSLP.
- EXACT HCHO-photochemistry predictors: surface_solar_radiation_downwards (SSRD, drives VOC->HCHO photochemistry) and temperature_2m (biogenic VOC emission + reaction rates); pair with multi-level winds for plume transport.
- CRITICAL GAP: GEE ERA5 collections (ECMWF/ERA5/HOURLY, ECMWF/ERA5_LAND/HOURLY) do NOT contain boundary_layer_height nor pressure-level winds — pull via cdsapi (reanalysis-era5-single-levels for blh; reanalysis-era5-pressure-levels for u/v/T at 925/850/700 hPa) or MERRA-2 M2T1NXFLX PBLH.
- BEST FOR INDIA = IMDAA (NCMRWF, 12 km): highest-res reanalysis over the Indian monsoon region, validated to match Lidar BLH better than ERA5 over the Himalayan foothills; resolves IGP gradients that ERA5's 31 km smears. Use as native training grid.
- LATENCY HIERARCHY: GFS/NCEP (NOAA/GFS0P25, hours) for operational DAILY AQI maps + live fire-plume transport; ERA5 (~5 d ERA5T / 2-3 mo final) and IMDAA (years, ends ~2018-2020) for historical TRAINING + validation.
- GAP-FILL/VERIFY: build a 3-reanalysis ensemble — IMDAA (fine India truth) + ERA5 (global continuous backbone, fills IMDAA's temporal end) + MERRA-2 (independent PBLH/wind/aerosol-coupled check). Inter-model disagreement flags uncertain met cells; GFS bridges the NRT gap.
- TRANSPORT for HCHO hotspots: use ERA5 pressure-level u/v at 850/700 hPa (cdsapi reanalysis-era5-pressure-levels) or IMDAA multi-level winds for back/forward trajectories linking FIRMS fire pixels to downwind S5P HCHO; 10m winds alone miss elevated plumes.
- RH must be DERIVED: no direct RH band in ERA5 single-levels/GEE — compute from temperature_2m + dewpoint_temperature_2m (Magnus); MERRA-2 gives QV2M (specific humidity), GFS gives relative_humidity_2m directly as fallback.

### ERA5 single-levels (hourly) — full variable set via CDS

- **Provider:** ECMWF / Copernicus Climate Change Service (C3S)
- **Instrument:** Global atmospheric reanalysis (IFS Cy41r2, 4D-Var)
- **Products:** reanalysis-era5-single-levels; boundary_layer_height (blh); 10m_u/v_component_of_wind; 2m_temperature; 2m_dewpoint_temperature; surface_pressure / MSLP; total_precipitation; surface_net_solar_radiation (ssr); surface_solar_radiation_downwards (ssrd)
- **GEE / API IDs:** `cdsapi: reanalysis-era5-single-levels`, `cdsapi: reanalysis-era5-pressure-levels`
- **Spatial resolution:** 0.25 deg (~31 km native)
- **Temporal resolution:** Hourly, 1940-present
- **Latency:** ~5 days preliminary (ERA5T), final ~2-3 months
- **Coverage:** Global, full India
- **Access:** cdsapi: c=cdsapi.Client(); c.retrieve('reanalysis-era5-single-levels',{...}). Free CDS account + ~/.cdsapirc (url+key). NetCDF/GRIB. Pressure-level u/v/T (925/850/700 hPa) via reanalysis-era5-pressure-levels.
- **Pipeline role:** Primary global met predictor for AQI ML: BLH (dilution), 10m wind spd/dir, T2m, RH(from T2m+Td), precip (washout), SSRD/SSR (photochemistry). Supplies the GEE-absent vars (BLH, multi-level winds).
- **Gap-fill / cross-verification role:** Gap-fills GEE ERA5 which LACKS boundary_layer_height and pressure-level winds. Continuous 1940-present backbone, no cloud/orbit gaps. Cross-validates IMDAA over IGP.
- **India relevance:** Standard met input in Indian AQI papers; validated over IGP. Coarser than IMDAA but global and continuous.
- **Caveats:** 0.25 deg too coarse for urban BLH/wind gradients; ERA5T preliminary may be revised. BLH only via CDS, not GEE.

### ERA5 / ERA5-Land Hourly on Google Earth Engine

- **Provider:** ECMWF/C3S, hosted by Google Earth Engine
- **Instrument:** ERA5 reanalysis (single-level 2D params only)
- **Products:** u_component_of_wind_10m; v_component_of_wind_10m; temperature_2m; dewpoint_temperature_2m; surface_pressure; total_precipitation / mean_total_precipitation_rate; surface_net_solar_radiation; surface_solar_radiation_downwards (ERA5-Land)
- **GEE / API IDs:** `ECMWF/ERA5_LAND/HOURLY`, `ECMWF/ERA5/HOURLY`, `ECMWF/ERA5_LAND/DAILY_AGGR`
- **Spatial resolution:** ERA5: ~27.8 km; ERA5-Land: ~11.1 km (0.1 deg)
- **Temporal resolution:** Hourly; ERA5 1940-present, ERA5-Land 1950-present (~3-month lag)
- **Latency:** ~2-3 months
- **Coverage:** Global (ERA5-Land = land only), full India
- **Access:** ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate().filterBounds(india); sample at CPCB lat/lon via reduceRegions. RH=f(temperature_2m,dewpoint_temperature_2m) Magnus; wind spd=sqrt(u^2+v^2), dir=atan2.
- **Pipeline role:** Fast in-cloud predictor extraction co-gridded with INSAT AOD + S5P columns at CPCB points for CNN/LSTM. ERA5-Land 11 km gives finer T2m, wind, SSRD than ERA5.
- **Gap-fill / cross-verification role:** Gridded met without local downloads; aligns natively with satellite GEE assets. Does NOT provide BLH or upper-level winds — fetch those from CDS or MERRA-2.
- **India relevance:** Easiest path to a met feature stack on GEE for India AQI CNN/LSTM pipelines.
- **Caveats:** NO boundary_layer_height band and NO pressure-level winds on GEE — critical gap; supplement via cdsapi/MERRA-2. ERA5-Land is land-only.

### IMDAA Regional Reanalysis (12 km)

- **Provider:** NCMRWF (MoES) + UK Met Office + IMD, National Monsoon Mission
- **Instrument:** Unified Model + 4D-Var regional reanalysis (Indian monsoon domain)
- **Products:** HPBL/boundary layer height; 10m u/v wind; 2m temperature; 2m RH / specific humidity; surface pressure / MSLP; total precipitation; downward shortwave / net solar radiation; multi-level u/v/T/q (transport)
- **GEE / API IDs:** `NCMRWF RDS portal (not on GEE)`
- **Spatial resolution:** 0.12 deg (~12 km) — highest-res reanalysis over India
- **Temporal resolution:** Hourly; 63 pressure levels; 1979-2018, extended ~2020+
- **Latency:** Research dataset, not NRT (multi-year lag); registration-gated
- **Coverage:** Indian monsoon region (full India, IGP, Himalayan foothills, NE fire zones)
- **Access:** Register https://rds.ncmrwf.gov.in/ -> Datasets -> IMDAA; download GRIB2 by variable+level+time. No public API/GEE; bulk download then regrid to AOI. 57+ variables, 63 levels.
- **Pipeline role:** Best-resolution India met predictor: 12 km BLH, wind, T, RH, precip, solar resolve IGP gradients better than ERA5 for surface AQI ML. Preferred for fire->HCHO transport at sub-ERA5 scales.
- **Gap-fill / cross-verification role:** Higher-fidelity refinement of ERA5 over India; corrects ERA5 coarse BLH/wind bias (matches Lidar BLH better than ERA5 over Himalayan foothills). Use ERA5/GFS to extend beyond IMDAA's end date.
- **India relevance:** THE best India-specific reanalysis; purpose-built for the monsoon region/IGP; superior BLH agreement vs ERA5.
- **Caveats:** Not real-time and not on GEE -> unusable for operational NRT AQI; manual GRIB download + regridding; coverage ends ~2018-2020 so pair with ERA5/GFS for recent dates.

### MERRA-2 Single-Level + Surface Flux Diagnostics

- **Provider:** NASA GMAO / GES DISC
- **Instrument:** GEOS-5 reanalysis with assimilated aerosol (GOCART)
- **Products:** PBLH (M2T1NXFLX); T2M / T10M; U10M/V10M, U50M/V50M, U2M/V2M; SLP / PS; QV2M (humidity -> RH); PRECTOT (M2T1NXFLX); SWGDN surface incoming shortwave (M2T1NXRAD)
- **GEE / API IDs:** `M2T1NXSLV`, `M2T1NXFLX`, `M2T1NXRAD`, `GEE: NASA/GSFC/MERRA/slv/2`
- **Spatial resolution:** 0.625 x 0.5 deg (~50-65 km)
- **Temporal resolution:** Hourly time-averaged (tavg1), 1980-present
- **Latency:** ~3-4 weeks
- **Coverage:** Global, full India
- **Access:** GES DISC OPeNDAP/Subsetter + Earthdata login; or GEE 'NASA/GSFC/MERRA/slv/2' for SLV. PBLH is in M2T1NXFLX (Earthdata, NOT in GEE slv). AWS: s3 registry.opendata.aws/nasa-m2t1nxslv.
- **Pipeline role:** Independent PBLH + 10m/50m wind + RH(QV2M) + surface shortwave (SWGDN) feature source; uniquely aerosol-coupled (aids AOD-AQI consistency).
- **Gap-fill / cross-verification role:** Cross-validates ERA5/IMDAA BLH and winds (3-model ensemble cuts single-model bias). Backs up ERA5T revisions; coarse, so verification not fine prediction.
- **India relevance:** Used in India aerosol studies; assimilated AOD ties into INSAT AOD validation.
- **Caveats:** Coarsest (~50 km) — too blunt for IGP urban detail; PBLH defined differently than ERA5 BLH (relative cross-check); PBLH absent from GEE slv collection.

### GFS / NCEP (near-real-time meteorology)

- **Provider:** NOAA NCEP
- **Instrument:** GFS global forecast/analysis model
- **Products:** temperature_2m; u/v_component_of_wind_10m; relative_humidity_2m / specific_humidity_2m; total_precipitation_surface; downward_shortwave_radiation_flux; precipitable_water
- **GEE / API IDs:** `NOAA/GFS0P25`, `NOMADS GRIB filter`, `AWS s3://noaa-gfs-bdp-pds`
- **Spatial resolution:** 0.25 deg (~28 km)
- **Temporal resolution:** Analysis + forecasts every 6 h; hourly forecast steps
- **Latency:** Near-real-time (hours)
- **Coverage:** Global, full India
- **Access:** GEE: ee.ImageCollection('NOAA/GFS0P25'). Or NOMADS/AWS GRIB2. NOAA/GFS0P25 does NOT carry HPBL on GEE; for NRT BLH pull GFS HPBL field from NOMADS/AWS GRIB.
- **Pipeline role:** Operational/forecast met for DAILY AQI maps when ERA5/IMDAA latency is too slow. Supplies T, wind, RH, precip, SW radiation for current-day inference and live HCHO plume transport during fire events.
- **Gap-fill / cross-verification role:** Bridges multi-month reanalysis latency so the system produces TODAY's AQI map and live transport; ERA5/IMDAA backfill the training period.
- **India relevance:** Only practical NRT met for an operational India AQI product within the daily-map objective.
- **Caveats:** Forecast (not reanalysis) -> larger error; no post-analysis obs; GFS0P25 on GEE lacks HPBL band (derive or pull from NOMADS GRIB).

## Ground-Truth Networks (CPCB, AERONET, OpenAQ, AirNow, SAFAR, PurpleAir)

*Ground-Truth Networks (training labels, satellite-AOD validation, QA/gap-fill) for BAH 2026 PS3 — Surface AQI & HCHO Hotspots over India*

**Top picks:**

- **CPCB CAAQMS (primary surface labels: PM2.5/PM10/NO2/SO2/CO/O3, hourly, ~500+ stations)**
- **AERONET India sites (reference truth to validate INSAT-3D/MODIS satellite AOD)**
- **OpenAQ API v3 (programmatic harmonized access ingesting CPCB + others)**
- **data.gov.in OGD CPCB API (official keyed JSON/CSV CPCB hourly fallback)**
- **PurpleAir API v1 (low-cost spatial densification & gap-fill)**

**Key findings:**

- CPCB CAAQMS = mandated label source: ~500-560 real-time stations (>400 by 2023, ~530+ by 2025), PM2.5/PM10/NO2/SO2/CO/O3+met at 15-min/hourly. Convert to NAQI via CPCB sub-index breakpoints (worst-of-8). This is the y-label for CNN/LSTM and the RMSE/R/MAE scoring set.
- Two CPCB pull paths: (1) official data.gov.in OGD REST, resource_id 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (api.data.gov.in/resource/...?api-key=KEY&format=json) = latest hourly pollutant min/max/avg + lat/lon; (2) CCR scrapers (cpcbccr-python-client, gsidhu) for full historical LSTM sequences.
- AERONET validates satellite AOD (INSAT-3D/3DR + MODIS/VIIRS) BEFORE AOD->PM2.5 regression. v3 service: print_web_data_v3?site=Kanpur&...&AOD20=1&AVG=10&if_no_html=1 returns L2.0 AOD at 340-1020nm + Angstrom. India sites: Kanpur, Gandhi_College, Jaipur, Pune, Gual_Pahari, Nainital.
- OpenAQ API v3 (api.openaq.org/v3, X-API-Key, free, ~60 req/min): locations->sensors->measurements/hours/days. Best single ingestion layer harmonizing CPCB India stations; but it MIRRORS CPCB, so verify provenance and don't count as independent truth.
- US Embassy/Consulate AirNow PM2.5 monitors (Delhi/Mumbai/Kolkata/Chennai/Hyderabad) were reference-grade independent validators, BUT State Dept TURNED OFF the global embassy network Mar 2025 (funding cut). Treat as historical-only validation (pre-2025), not a live source.
- SAFAR (IITM/IMD): ~10 stations each in Delhi/Mumbai/Pune/Ahmedabad measuring PM/O3/CO/NOx/SO2/BC + VOCs/Benzene. Independent metro cross-check and rare VOC-precursor context for HCHO objective. No clean public API; scrape safar.tropmet.res.in.
- PurpleAir API v1 (api.purpleair.com/v1/sensors, X-API-Key): low-cost PM2.5 at high spatial density, sparse but present in Indian metros. Role: spatial GAP-FILL between sparse reference stations; needs US-EPA/cf correction for Plantower humidity bias before use vs reference grade.
- Build a unified station DB keyed by stable station_id: name, agency, lat/lon, pollutants, cadence, IGP-flag, fire-zone flag. Dedup CPCB-vs-OpenAQ by <1km coord match; tag each station as primary-label/validation/gap-fill so ML never trains and validates on the same physical sensor.

### CPCB CAAQMS (Continuous Ambient Air Quality Monitoring Stations)

- **Provider:** Central Pollution Control Board, MoEFCC, India
- **Instrument:** Reference/equivalent continuous analyzers (BAM/TEOM PM, chemilum NO2, UV O3, NDIR CO, UV-fluor SO2)
- **Products:** PM2.5; PM10; NO2; SO2; CO; O3; NH3/benzene (some); NAQI sub-indices; co-located met
- **GEE / API IDs:** `data.gov.in:3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`, `airquality.cpcb.gov.in/ccr`
- **Spatial resolution:** Point stations, ~500-560 sites (weighted to IGP + metros)
- **Temporal resolution:** 15-min raw, hourly aggregated
- **Latency:** Near-real-time (sub-hour); historical via CCR archive
- **Coverage:** India nationwide; dense over Delhi-NCR & Indo-Gangetic Plain
- **Access:** Official: data.gov.in OGD REST resource_id 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (api.data.gov.in/resource/...?api-key=KEY&format=json/csv). Portal: airquality.cpcb.gov.in/ccr/. Unofficial historical: cpcbccr-python-client, gsidhu/cpcbccr-data-scraper, wrap-away/cpcb-air-quality-api.
- **Pipeline role:** PRIMARY TRAINING LABELS (y) and RMSE/R/MAE target. Convert concentrations->NAQI via CPCB breakpoints (worst sub-index). Station time series feed LSTM/CNN-LSTM; points anchor CNN AQI maps.
- **Gap-fill / cross-verification role:** Defines where truth exists; spatial holes are what satellite maps must fill. QA: flag negative/flatline/stuck values, drop <75% hourly completeness, despike before label use.
- **India relevance:** Mandated national network; official scoring reference for any India surface-AQI product.
- **Caveats:** Uneven coverage (rural/fire zones sparse); gaps & calibration drift; OGD API gives only latest snapshot, need CCR scrape for long history.

### AERONET (Aerosol Robotic Network) - India sites

- **Provider:** NASA GSFC / national PI institutions (IIT Kanpur etc.)
- **Instrument:** Cimel CE318 sun-sky photometer
- **Products:** Spectral AOD 340/380/440/500/675/870/1020nm; 440-870nm Angstrom exponent; precipitable water; inversion: SSA, fine/coarse mode
- **GEE / API IDs:** `aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3`
- **Spatial resolution:** Point sites (~8-12 active/historical in India)
- **Temporal resolution:** ~15-min daytime cloud-free; daily averages
- **Latency:** L1.5 near-real-time; L2.0 QA after recalibration (months)
- **Coverage:** Kanpur, Gandhi_College (Ballia IGP), Jaipur, Pune, Gual_Pahari, Nainital, Dibrugarh, Pantnagar
- **Access:** Web service: aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site=SITE&year=Y&month=M&day=D&AOD20=1&AVG=10&if_no_html=1 (AOD15=1 for L1.5). Inversion: print_web_data_inv_v3.
- **Pipeline role:** VALIDATE satellite AOD (INSAT-3D/3DR Imager AOD, MODIS MAIAC) vs reference; quantify bias/RMSE; interpolate AOD550 via Angstrom to match MODIS/INSAT before AOD->PM2.5 regression.
- **Gap-fill / cross-verification role:** Independent check that AOD predictor feeding the model is unbiased; anchors atmospheric-correction QA. Gandhi College/Kanpur = critical IGP biomass-burning truth.
- **India relevance:** Only reference-grade AOD ground truth in India; IGP sites directly under project focus region.
- **Caveats:** Few sites; daytime/clear-sky only (overlaps satellite clear-sky gaps); L2.0 latency means recent data only at L1.5.

### OpenAQ API v3

- **Provider:** OpenAQ (non-profit aggregator)
- **Instrument:** Aggregates reference + low-cost (passes through provider type)
- **Products:** PM2.5; PM10; NO2; SO2; CO; O3; BC; RH; T
- **GEE / API IDs:** `api.openaq.org/v3`
- **Spatial resolution:** All ingested India stations (mirrors CPCB + extras)
- **Temporal resolution:** Hourly (also raw measurements, days, years rollups)
- **Latency:** Near-real-time mirror of upstream
- **Coverage:** Global; India coverage = CPCB-sourced + extras
- **Access:** Base api.openaq.org/v3, header X-API-Key. Endpoints: /locations, /locations/{id}/sensors, /sensors/{id}/measurements|hours|days, /parameters, /latest. Free key, paginated (limit/page), date_from/date_to.
- **Pipeline role:** PRIMARY PROGRAMMATIC INGESTION for CPCB-style labels with harmonized schema/units/geocoords; simplest path to bulk pull station time series for training.
- **Gap-fill / cross-verification role:** Cross-source reconciliation & QA: compare OpenAQ vs direct-CPCB to catch ingestion errors; metadata separates reference vs low-cost.
- **India relevance:** Easiest standardized access to Indian CPCB data + sensor metadata for DB build.
- **Caveats:** NOT independent of CPCB (mirrors it) - don't treat as separate truth; only data OpenAQ has discovered/been given.

### US Embassy / Consulate AirNow PM2.5 monitors (India)

- **Provider:** US Dept of State / EPA AirNow
- **Instrument:** Reference-grade BAM PM2.5
- **Products:** PM2.5 (reference-grade)
- **GEE / API IDs:** `airnowapi.org`, `aqicn.org/data-platform`
- **Spatial resolution:** 5 sites: New Delhi, Mumbai, Kolkata, Chennai, Hyderabad
- **Temporal resolution:** Hourly
- **Latency:** Historically hourly NRT - DISCONTINUED Mar 2025
- **Coverage:** 5 metros (compound-sited)
- **Access:** Historical only: AirNow API (airnowapi.org keyed) & docs.airnowapi.org; mirrors on aqicn data-platform. Live embassy feed off since Mar 2025 (funding).
- **Pipeline role:** Independent reference-grade PM2.5 validation in 5 metros for PRE-2025 period; cross-check CPCB co-located stations historically.
- **Gap-fill / cross-verification role:** Bias-check against CPCB metro stations (independent instrument/QA chain) for historical train/validation splits.
- **India relevance:** Independent metro truth, now historical archive only - not live for 2025+ products.
- **Caveats:** Network shut off Mar 2025; PM2.5 only; embassy-compound micro-siting bias.

### SAFAR (System of Air Quality & Weather Forecasting And Research)

- **Provider:** IITM Pune / IMD (MoES)
- **Instrument:** Continuous analyzers + VOC/BC monitors
- **Products:** PM2.5; PM10; O3; CO; NOx; SO2; BC; VOCs/Benzene; met + solar
- **GEE / API IDs:** `safar.tropmet.res.in`
- **Spatial resolution:** ~10 stations each Delhi/Mumbai/Pune/Ahmedabad (~40)
- **Temporal resolution:** Hourly
- **Latency:** Near-real-time (portal)
- **Coverage:** 4 metros
- **Access:** Portal safar.tropmet.res.in (map_data.php per city). No clean public API - HTML/JSON scraping required.
- **Pipeline role:** Independent metro cross-validation of CPCB labels; rare speciated VOC/benzene + BC data contextually relevant to HCHO-precursor narrative (objective 2).
- **Gap-fill / cross-verification role:** Secondary truth where present; BC/VOC adds chemical context for HCHO hotspot interpretation, not direct HCHO measurement.
- **India relevance:** Indigenous IITM network in same metros; aligns with Indian-data preference.
- **Caveats:** Only 4 cities; no documented API; coverage/continuity uncertain; not nationwide.

### PurpleAir (low-cost sensor network) API v1

- **Provider:** PurpleAir Inc.
- **Instrument:** Plantower PMS laser nephelometer (PA-II)
- **Products:** PM2.5 (cf=1 / ATM); PM1; PM10; T; RH; pressure
- **GEE / API IDs:** `api.purpleair.com/v1/sensors`
- **Spatial resolution:** Dense where deployed; sparse but present in Indian metros
- **Temporal resolution:** ~2-min / configurable real-time
- **Latency:** Real-time
- **Coverage:** Global crowdsourced; limited India deployment
- **Access:** api.purpleair.com/v1/sensors (multi) and /v1/sensors/{index}; header X-API-Key (read key); optional per-sensor read_key for private. Free read key on request.
- **Pipeline role:** Spatial DENSIFICATION / gap-fill of PM2.5 between sparse reference stations; extra training samples after correction.
- **Gap-fill / cross-verification role:** Fill spatial holes in CPCB coverage; cross-validate satellite PM in unmonitored areas - REQUIRES US-EPA/cf correction for Plantower humidity bias.
- **India relevance:** Adds spatial density in metros where CPCB is sparse, but Indian coverage is thin.
- **Caveats:** Low-cost humidity bias, variable QA, uneven/sparse India coverage; never use raw as label without correction.

### Unified Ground-Truth Station Database (build spec)

- **Provider:** Derived (project-built)
- **Instrument:** Multi-network merge
- **Products:** station_id; agency; lat/lon/elev; pollutants_available; cadence; role tag (label/validation/gapfill); IGP flag; fire-zone flag
- **GEE / API IDs:** `derived`
- **Spatial resolution:** All India ground sites merged
- **Temporal resolution:** Harmonized to hourly
- **Latency:** n/a
- **Coverage:** India
- **Access:** Construct from OpenAQ /locations (CPCB) + data.gov.in OGD + AERONET site list + SAFAR + PurpleAir /sensors bbox(India). Dedup CPCB<->OpenAQ by <1km coord match.
- **Pipeline role:** Single keyed registry mapping every label/validation point to lat/lon/grid-cell so CNN/LSTM joins satellite pixels to ground truth cleanly and avoids train/validate leakage on same sensor.
- **Gap-fill / cross-verification role:** Tags each station primary-label vs independent-validation vs gap-fill; enables spatially-blocked CV (hold out whole stations/regions) for honest RMSE/R/MAE.
- **India relevance:** Operationalizes 'CPCB primary, AERONET validates AOD, OpenAQ programmatic' as one schema.
- **Caveats:** Must reconcile naming/coord mismatches and IST timezone alignment; deduplicate mirrored CPCB records.

## Fire Detection & Biomass-Burning Emission Inventories

*Fire detection + biomass-burning emission inventories (HCHO-fire objective, BAH 2026 PS3 Obj-2)*

**Top picks:**

- **NASA FIRMS VIIRS 375m (S-NPP VNP14IMGTDL + NOAA-20/21 VJ1/VJ2) — primary fire-hotspot/period detector, API NRT**
- **Sentinel-5P TROPOMI HCHO L2/L3 — the target variable (paired, not a fire product)**
- **GFAS v1.2 (CAMS/ECMWF) — daily 0.1deg FRP->HCHO/NMVOC emissions, direct physical fire->VOC link**
- **FINNv2.5 (NCAR) — 1km per-fire daily speciated VOC (incl. HCHO) emissions for hi-res IGP plumes**
- **MODIS MCD64A1 burned-area (GEE MODIS/061/MCD64A1) — independent burned-area verification of active-fire counts**

**Key findings:**

- Detection tier (where/when fires are): FIRMS VIIRS 375m primary + MODIS 1km long baseline; fuse SNPP+NOAA-20+NOAA-21+Nightfire to beat single-sensor swath/overpass gaps and small-fire omission. Emission tier (how much HCHO/VOC): GFAS + FINN + GFED5 + QFED convert fires into HCHO/NMVOC flux linkable to TROPOMI.
- FIRMS API is the NRT pipeline: area-CSV https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[lonW,latS,lonE,latN]/[day_range]/[date]; India bbox ~68,6,98,38; country API uses IND. Free MAP_KEY, 5000 tx/10min. SOURCE *_NRT (recent) + *_SP (archive).
- Define burning PERIODS from daily fire-count series over AOI (Punjab+Haryana bbox/district masks): onset/peak when count>mean+2sigma (MODIS 2000-baseline) or absolute >N/day; FRP-weighted. Windows: post-monsoon Oct-Nov (paddy, dominant Delhi HCHO event), pre-monsoon Apr-May (wheat), Himalayan fires Mar-May, NE Apr-Jun.
- Critical India bias: stubble burning shifted to evening, partly evading MODIS/VIIRS ~13:30 overpasses (NASA-documented) -> add VIIRS Nightfire + GFED5 diurnal/hourly emission fractions to recover true burning and avoid under-attributing next-day HCHO.
- MCD64A1 (500m, GEE MODIS/061/MCD64A1) and GlobFire give independent burned-area to verify active-fire periods, but both 500m+monthly omit small Punjab fields — use as confirmation alongside (never instead of) 375m VIIRS active fire.
- Multi-inventory ensemble (GFAS+QFED top-down FRP vs FINN+GFED5 bottom-up burned-area) brackets emission uncertainty; spread quantifies attribution confidence. GFED5 emissions ~50% higher than GFED4 (burned area +61%), better capturing small ag fires.
- Fire->HCHO workflow: FIRMS FRP/count grid -> GFAS/FINN HCHO+NMVOC flux on matching grid -> co-register to TROPOMI HCHO L3 (~0.05deg) -> ERA5/IMDAA winds for transport/back-trajectory -> Getis-Ord Gi* + DBSCAN clustering + lagged fire-HCHO regression with downwind offset.
- Gap-fill matrix: VIIRS375m fills MODIS small-fire gap; Nightfire fills evening/overpass gap; GFAS/FINN fill FIRMS 'no emission mass'; GFED5 diurnal fractions fill overpass-timing; MCD64A1 verifies fire events; emission ensemble brackets EF uncertainty; FINN 1km fills GFAS/GFED coarseness.

### NASA FIRMS — VIIRS 375m Active Fire (NRT + archive)

- **Provider:** NASA LANCE / EOSDIS (FIRMS)
- **Instrument:** VIIRS 375m: S-NPP (VNP14IMGTDL), NOAA-20 (VJ114IMGDL/VJ1), NOAA-21 (VJ214/VJ2)
- **Products:** VNP14IMGTDL_NRT active fire; VJ114IMGDL_NRT (NOAA-20); VJ214IMGDL_NRT (NOAA-21); lat/lon, FRP(MW), brightness Ti4/Ti5, confidence(l/n/h), day/night flag, acq date+time, scan/track
- **GEE / API IDs:** `FIRMS (GEE, MODIS rasterized)`, `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT`, `VNP14IMGTDL_NRT`, `VJ114IMGTDL_NRT`, `api/area/csv`, `api/country/csv`
- **Spatial resolution:** 375m (I-band); nominal 375m pixel footprint
- **Temporal resolution:** ~2-4 overpasses/day combined across S-NPP+NOAA-20/21 (each ~1:30 LT asc); sub-daily when fused
- **Latency:** NRT ~3h; Ultra-RT ~60min (LANCE); URT/RT/NRT tiers
- **Coverage:** Global incl. all India; archive S-NPP 2012-, NOAA-20 2020-, NOAA-21 2023-
- **Access:** FIRMS API area-CSV: https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[W,S,E,N]/[DAY_RANGE]/[DATE]; SOURCE=VIIRS_SNPP_NRT|VIIRS_NOAA20_NRT|VIIRS_NOAA21_NRT|MODIS_NRT and *_SP (standard/archive). Country API .../api/country/csv/[KEY]/[SRC]/IND/[days]. MAP_KEY free; limit 5000 tx/10min. Bulk archive: firms.modaps.eosdis.nasa.gov/download (SHP/CSV/KML/WFS). GEE rasterized: FIRMS (MODIS only).
- **Pipeline role:** Primary fire-hotspot input: per-pixel FRP+location feed daily fire-count grids and FRP density layers co-registered to TROPOMI HCHO. Defines biomass-burning PERIODS via daily India/Punjab-Haryana fire-count time series; threshold onset/peak (count>mean+2sigma or >N fires/day in AOI). FRP weights fire->HCHO regression.
- **Gap-fill / cross-verification role:** 375m fills MODIS 1km small/cool-fire omission (stubble fires are small); multi-VIIRS (SNPP+N20+N21) fusion fills single-sensor temporal/swath gaps. Active fire fills MCD64A1 burned-area omission of short-lived ag fires. Verifies HCHO enhancement co-locates with real combustion (rejects biogenic/industrial HCHO).
- **India relevance:** Best sensor for Punjab/Haryana stubble (small fields, Oct-Nov paddy peak; Apr-May wheat) and Himalayan/NE forest fires; 375m detects fires MODIS misses.
- **Caveats:** Cloud/smoke obscuration; overpass-time bias (afternoon burning peaks ~13:30 LT but evening-shifted burning missed); confidence filtering needed; no emission mass (counts/FRP only).

### NASA FIRMS — MODIS C6.1 Active Fire

- **Provider:** NASA LANCE / EOSDIS
- **Instrument:** MODIS Terra (MOD14) + Aqua (MYD14), Collection 6.1
- **Products:** MCD14DL / MOD14/MYD14 active fire (lat/lon, FRP, brightness T21/T31, confidence 0-100, day/night)
- **GEE / API IDs:** `FIRMS`, `MODIS_NRT`, `MODIS_SP`, `MCD14DL`
- **Spatial resolution:** 1km (nominal); fire pixel 1km
- **Temporal resolution:** 4 overpasses/day (Terra ~10:30/22:30, Aqua ~13:30/01:30 LT)
- **Latency:** NRT ~3h; URT ~60min; standard collection (archive)
- **Coverage:** Global; Terra 2000-, Aqua 2002- (long climatology)
- **Access:** Same FIRMS API (SOURCE=MODIS_NRT / MODIS_SP). GEE: FIRMS (rasterized MODIS, 1km daily, T21+confidence bands, 2000-).
- **Pipeline role:** Long-baseline fire-count climatology (2000-present) to define normal vs anomalous burning seasons; cross-sensor fusion with VIIRS for completeness; consistent historical record for training/trend.
- **Gap-fill / cross-verification role:** 20+ yr archive fills VIIRS short record for climatological thresholds/anomaly baselines; Terra+Aqua dual overpass fills VIIRS single-time sampling. Verifies VIIRS detections (multi-sensor agreement).
- **India relevance:** Standard long record for IGP stubble trend analysis; NASA notes evening-shift burning now partly missed by 13:30 overpass — motivates VIIRS+geostationary fusion.
- **Caveats:** 1km omits small stubble fires (underdetection vs VIIRS); afternoon overpass misses late-evening burning.

### GFAS v1.2 (CAMS Global Fire Assimilation System)

- **Provider:** ECMWF / Copernicus Atmosphere Monitoring Service (CAMS)
- **Instrument:** MODIS (Terra+Aqua) FRP assimilation -> top-down emissions
- **Products:** FRP (frpfire); dry-matter burnt; 40 species incl. HCHO/CH2O, NMVOC, CO, CO2, OC/BC aerosol, NOx, CH4; injection height
- **GEE / API IDs:** `cams-global-fire-emissions-gfas (ADS)`, `GFAS v1.2`, `frpfire`, `hchofire/ch2ofire`
- **Spatial resolution:** 0.1deg (~11km)
- **Temporal resolution:** Daily
- **Latency:** Near-real-time daily (CAMS operational); 2003-present
- **Coverage:** Global incl. India
- **Access:** Copernicus ADS (Atmosphere Data Store) API/web download, GRIB or netCDF; dataset 'cams-global-fire-emissions-gfas'. CDS-API key.
- **Pipeline role:** Core physical fire->VOC link: gridded daily HCHO + NMVOC emission flux converts FIRMS fire activity into expected HCHO source strength; regress/compare against TROPOMI HCHO columns; drives/constrains transport with ERA5/IMDAA winds.
- **Gap-fill / cross-verification role:** Fills FIRMS gap of 'no emission mass' — turns FRP into species flux. Provides the missing HCHO emission linking fire counts to observed column. Injection height aids vertical attribution. 0.1deg matches TROPOMI ~5.5x3.5km after aggregation.
- **India relevance:** Captures IGP stubble-season HCHO/NMVOC pulses; widely used to attribute north-India Nov pollution to crop-residue burning.
- **Caveats:** Coarse 0.1deg vs field-scale fires; MODIS-FRP based so inherits afternoon-overpass + small-fire bias; emission factors uncertain for ag residue.

### FINNv2.5 (Fire INventory from NCAR)

- **Provider:** NCAR/UCAR ACOM
- **Instrument:** MODIS (MCD14ML) + VIIRS 375m active fire; land-cover + emission factors
- **Products:** Per-fire daily emissions: CO, NOx, NMOC/NMVOC, HCHO (speciated), VOCs; VOC speciation for MOZART-4, SAPRC99, GEOS-Chem; PM2.5/PM10, OC/BC, CO2
- **GEE / API IDs:** `FINNv2.5`, `ds312.9 (NCAR RDA)`, `FINNv2.5.1_modvrs_nrt`, `Zenodo 7868652`
- **Spatial resolution:** ~1km per-fire (point), griddable to model resolution
- **Temporal resolution:** Daily
- **Latency:** NRT product (FINNv2.5.1_modvrs_nrt_*) updated ~daily; final archive 2002-2023+ (RDA ds312.9)
- **Coverage:** Global incl. India
- **Access:** NRT txt: https://www.acom.ucar.edu/acresp/MODELING/finn_emis_txt/ (FINNv2.5.1_modvrs_nrt_*.txt, MODIS & MODIS+VIIRS variants). Archive: NCAR RDA ds312.9 (rda.ucar.edu/datasets/ds312.9). Zenodo 7868652. Explicit speciated HCHO field.
- **Pipeline role:** Highest-resolution fire->HCHO emission: per-fire daily speciated HCHO for fine-scale IGP plume attribution and bottom-up vs TROPOMI top-down comparison; MODIS+VIIRS variant maximizes detection completeness.
- **Gap-fill / cross-verification role:** 1km per-fire detail fills GFAS 0.1deg coarseness for field-level Punjab attribution. Explicit HCHO speciation fills the precursor-species gap. MODIS+VIIRS fusion variant fills single-sensor omission. Independent (bottom-up) cross-check of GFAS top-down.
- **India relevance:** Resolves individual stubble-field plumes in Punjab/Haryana; speciated VOCs support HCHO budget.
- **Caveats:** Bottom-up emission-factor + land-cover-class uncertainty; assumes burned area per detect; small/overlapping fields can saturate; NRT less QC'd than final.

### GFED5 / GFED4s (Global Fire Emissions Database)

- **Provider:** VU Amsterdam / globalfiredata.org (GFED5 Nature Sci Data 2025)
- **Instrument:** MODIS MCD64A1 burned area (post-2001) + VIIRS active fire (post-2022 scaling) + emission factors
- **Products:** Burned area; Emissions: C, CO2, CO, CH4, NMVOC, HCHO precursors, NOx, OC/BC, PM2.5; Monthly + daily fraction + diurnal/hourly fraction climatology (GOES-derived)
- **GEE / API IDs:** `GFED4.1s`, `GFED5`, `globalfiredata.org`, `samapriya/awesome-gee-community-datasets #141`
- **Spatial resolution:** GFED5 0.25deg; GFED4s 0.25deg
- **Temporal resolution:** Monthly + daily fractions + 3-hourly/diurnal cycle
- **Latency:** GFED4s beta NRT (monthly, ~near-current); GFED5 2002-2022/2023, VIIRS-based extension post-2022
- **Coverage:** Global incl. India
- **Access:** globalfiredata.org downloads (HDF5/netCDF); GFED4.1s widely mirrored (incl. GEE community asset, samapriya). GFED5: Nature Sci Data 2025 (s41597-025-06127-w).
- **Pipeline role:** Monthly/seasonal NMVOC & burning climatology to bound expected HCHO source per region/month; daily+diurnal fractions disaggregate to overpass time for TROPOMI matching; baseline anomaly detection of burning seasons.
- **Gap-fill / cross-verification role:** Diurnal emission fractions fill the overpass-timing gap (correct for fires burning outside satellite passes). Burned-area-based emissions complement FRP-based GFAS/QFED (independent method cross-check). Long climatology fills baseline for anomaly thresholds.
- **India relevance:** Standard for IGP seasonal emission budgets; GFED5 burned area 61% larger than GFED4 -> better small-fire (stubble) capture.
- **Caveats:** Coarse 0.25deg; monthly native (daily via fraction); GFED5 post-2022 uses VIIRS scaling (uncertainty); ag-fire EF uncertainty.

### QFED v2.x (Quick Fire Emissions Dataset)

- **Provider:** NASA GSFC GMAO (GEOS system)
- **Instrument:** MODIS (+VIIRS in newer) FRP, top-down with GFAS-style cloud correction
- **Products:** FRP-based emissions: CO, CO2, NOx, NMVOC/VOC incl. HCHO precursors, OC/BC, SO2, PM2.5; flaming/smoldering split (next-gen)
- **GEE / API IDs:** `QFED2.4`, `QFED2.5`, `NTRS 20180005253`
- **Spatial resolution:** 0.1deg (and 0.25deg)
- **Temporal resolution:** Daily
- **Latency:** Operational in GEOS-FP (near-real-time)
- **Coverage:** Global incl. India
- **Access:** NASA GMAO portal / GEOS download (ftp/https), netCDF; QFED2.4/2.5. NTRS doc 20180005253.
- **Pipeline role:** Alternative top-down FRP->emission inventory for ensemble/uncertainty bracketing against GFAS+FINN; HCHO-precursor flux for fire-HCHO attribution.
- **Gap-fill / cross-verification role:** Independent FRP-based estimate to bracket GFAS (both MODIS-FRP) and FINN/GFED (bottom-up) — spread = emission uncertainty. Cloud-correction recovers FRP under partial cloud (gap-fills obscured fires).
- **India relevance:** QFED docs explicitly flag north-India Nov crop-residue emission spikes; used in GEOS forecasts over India.
- **Caveats:** FRP top-down EF uncertainty; MODIS-overpass bias; less openly documented version cadence.

### MODIS MCD64A1 Burned Area

- **Provider:** NASA LP DAAC / GEE
- **Instrument:** MODIS Terra+Aqua surface reflectance + active fire
- **Products:** BurnDate (day-of-year); Uncertainty; QA; First/Last day
- **GEE / API IDs:** `MODIS/061/MCD64A1`, `JRC_GWIS_GlobFire_v2_DailyPerimeters`, `JRC_GWIS_GlobFire_v2_FinalPerimeters`
- **Spatial resolution:** 500m
- **Temporal resolution:** Monthly (daily burn-date within month)
- **Latency:** Standard product, ~months lag; archive 2000-11 to present
- **Coverage:** Global incl. India
- **Access:** GEE: MODIS/061/MCD64A1 (BurnDate band). LP DAAC direct. GlobFire daily events: JRC_GWIS_GlobFire_v2_DailyPerimeters / FinalPerimeters.
- **Pipeline role:** Independent burned-area extent to validate active-fire-derived burning periods and map total area burned per season per district (Punjab/Haryana).
- **Gap-fill / cross-verification role:** Burned-area (effect) cross-checks active-fire (event) detections; confirms persistent burning vs transient hotspots. GlobFire perimeters give fire-event polygons/duration. 500m verifies VIIRS clusters.
- **India relevance:** Quantifies stubble-burned hectares per season; basis for GFED/FINN burned-area emissions over IGP.
- **Caveats:** 500m + monthly misses small/short ag fires (known omission for fragmented Punjab fields) -> use WITH VIIRS, not alone; reflectance-based (cloud/smoke limited).

### VIIRS Nightfire (VNF) + black-marble nighttime

- **Provider:** NOAA / Colorado School of Mines (EOG) — VIIRS Nightfire
- **Instrument:** VIIRS day/night band + M-bands (SWIR/MWIR), nighttime
- **Products:** Per-detection temperature, radiant heat, source area, ESF (gas flaring vs biomass); nighttime fire radiance
- **GEE / API IDs:** `VIIRS Nightfire VNF`, `eogdata.mines.edu`
- **Spatial resolution:** ~375-750m
- **Temporal resolution:** Nightly
- **Latency:** Daily-ish (EOG processing lag)
- **Coverage:** Global incl. India
- **Access:** EOG (eogdata.mines.edu) VNF downloads; nightly CSV/products.
- **Pipeline role:** Captures night/evening burning that day-overpass MODIS/VIIRS-day miss; supplements active-fire counts for true diurnal burning extent.
- **Gap-fill / cross-verification role:** Fills the evening/night burning gap (critical as Punjab burning has shifted later, evading afternoon overpasses) — improves completeness of fire-period detection and prevents under-attribution of next-morning HCHO.
- **India relevance:** Directly addresses NASA-documented evening-shift of Punjab stubble burning that biases daytime sensors.
- **Caveats:** Distinguishes flares vs biomass (filter ESF); smaller flaming fires only; not an emission inventory.

## Additional Augmenting Datasets (Terrain, Land-Use, Population, Precipitation, TEMPO/OMPS)

*Additional global satellites/datasets (composition, terrain, land-use, population, fire/thermal, precipitation) to augment INSAT-3D + Sentinel-5P + reanalysis for surface AQI mapping and HCHO hotspot detection over India — BAH 2026 PS3*

**Top picks:**

- **Copernicus DEM GLO-30 (COPERNICUS/DEM/GLO30) + SRTM (USGS/SRTMGL1_003) — terrain covariates**
- **ESA WorldCover v200 (ESA/WorldCover/v200) + MODIS MCD12Q1 — land-use covariate & LUR base**
- **WorldPop 100m + GPWv411 — exposure mapping / population-weighted AQI**
- **TROPOMI OFFL CH4 + OCO/GOSAT/SIF — combustion vs biogenic HCHO disambiguation**
- **GPM IMERG V07 (NASA/GPM_L3/IMERG_V07) — wet-scavenging covariate & temporal-gap flag**

**Key findings:**

- TEMPO has NO coverage over India — use strictly as methodology/algorithm reference (shared DOAS/AMF retrieval lineage) and an hourly-composition template to estimate the diurnal HCHO-sampling bias of TROPOMI's single ~13:30 LT overpass.
- Source-disambiguation chain for HCHO: TROPOMI HCHO + S5P CH4 (COPERNICUS/S5P/OFFL/L3_CH4) + CO + OCO-2/3 SIF — combustion = co-located HCHO+CH4+CO+XCO2; biogenic = HCHO with high SIF, low CH4/CO. Sharpens fire-HCHO correlation confidence in Objective 2.
- OMPS (CMR OMPS_NPP_NMTO3/NMSO2/UVAI) provides OMI-continuity backup for TROPOMI O3/SO2 plus a UVAI absorbing-aerosol (smoke/dust) flag for burning season; coarse (~17-50 km) and only via Earthdata (not gridded in GEE).
- Core daily-usable GEE ML covariates that downscale 1-7 km columns to neighbourhood AQI and attribute sources: terrain (USGS/SRTMGL1_003, COPERNICUS/DEM/GLO30), land cover (ESA/WorldCover/v200, MODIS/061/MCD12Q1), nightlights (NASA/VIIRS/002/VNP46A2), OSM/GHSL roads&built (JRC/GHSL/P2023A/GHS_BUILT_S).
- Landsat 8/9 C2 L2 (LANDSAT/LC08/C02/T1_L2, ST_B10) + ECOSTRESS (NASA/ECOSTRESS/L2T_LSTE/V2) give 30-70 m LST/NDVI/NDBI for AOD-to-PM2.5 downscaling; Landsat dNBR independently verifies FIRMS active-fire detections via burn scars, cutting false positives.
- Exposure: WorldPop/GP/100m/pop + CIESIN/GPWv411 convert AQI maps to population-weighted exposure to rank IGP megacity hotspots by people affected; density also doubles as an anthropogenic-emission proxy covariate.
- GPM IMERG V07 (NASA/GPM_L3/IMERG_V07) is the wet-scavenging covariate explaining sudden AQI drops (rain washout) and simultaneously flags rainy/cloudy days when TROPOMI and INSAT AOD retrievals are likely missing — a temporal-gap indicator.
- Composition gap-fillers (TEMPO, OMPS, OCO-2/3, GOSAT) are sparse-swath or US-only — use for validation, source attribution, and methodology; terrain/land-use/population/precipitation layers are the daily GEE-native covariates. Together these add ~12+ new sources pushing the catalogue past 30.

### NASA TEMPO + Suomi-NPP/NOAA-20 OMPS (geostationary + OMI-continuity composition)

- **Provider:** NASA/SAO (TEMPO); NASA/NOAA (OMPS)
- **Instrument:** TEMPO geostationary UV-Vis spectrometer (N. America only); OMPS Nadir Mapper+Profiler (OMI/TOMS continuity)
- **Products:** TEMPO hourly NO2/HCHO/O3/SO2/aerosol (N.America); OMPS O3 total column (NMTO3); OMPS SO2 (PCA); OMPS UV Aerosol Index; OMPS O3 profile
- **GEE / API IDs:** `CMR:TEMPO_HCHO_L3`, `CMR:TEMPO_NO2_L3`, `CMR:OMPS_NPP_NMTO3_L3`, `CMR:OMPS_NPP_NMSO2_PCA_L2`, `ASDC Harmony API`
- **Spatial resolution:** TEMPO ~2.1x4.7 km; OMPS-NM ~17 km (NOAA-20) to ~50 km (NPP)
- **Temporal resolution:** TEMPO hourly daytime (geostationary); OMPS daily global
- **Latency:** NRT ~3 h (LANCE) for both; standard products lag days
- **Coverage:** TEMPO N. America ONLY (not India); OMPS global incl. India
- **Access:** TEMPO: NASA ASDC/LANCE, CMR short_name TEMPO_HCHO_L3/TEMPO_NO2_L3, Harmony API (NOT in GEE). OMPS: NASA GES DISC/CMR OMPS_NPP_NMTO3_L3, OMPS_NPP_NMSO2_PCA, LANCE NRT (not gridded in GEE).
- **Pipeline role:** TEMPO: diurnal-HCHO methodology/benchmark reference. OMPS: redundant O3/SO2 column source + UVAI absorbing-aerosol (smoke) covariate/flag.
- **Gap-fill / cross-verification role:** TEMPO fills CONCEPTUAL gap (quantifies diurnal under-sampling bias of polar HCHO, validated hourly template). OMPS fills O3/SO2 instrument-continuity gaps when TROPOMI missing/noisy; UVAI complements MERRA-2 AOD for smoke detection.
- **India relevance:** TEMPO = methodology/algorithm reference only (shared DOAS/AMF retrieval lineage with TROPOMI HCHO/NO2; hourly template to estimate diurnal HCHO bias of single ~13:30 LT overpass). OMPS gives O3/SO2 continuity backup over India + UVAI smoke/dust flag for burning season.
- **Caveats:** TEMPO unusable as input over India (reference only). OMPS coarse (17-50 km), SO2 only for strong sources, neither gridded in GEE — needs Earthdata pipeline.

### TROPOMI CH4 + OCO-2/OCO-3 & GOSAT/GOSAT-2 (combustion GHG + SIF)

- **Provider:** ESA/KNMI/SRON (S5P); NASA (OCO); JAXA/NIES (GOSAT)
- **Instrument:** TROPOMI SWIR (XCH4); OCO grating spectrometer (XCO2/SIF); GOSAT TANSO-FTS (XCH4/XCO2)
- **Products:** S5P XCH4 (ppb); OCO-2/3 XCO2 (ppm); OCO-2/3 SIF (solar-induced fluorescence); GOSAT XCH4/XCO2; GOSAT-2 XCO
- **GEE / API IDs:** `COPERNICUS/S5P/OFFL/L3_CH4`, `CMR:OCO2_L2_Lite_FP_11.1r`, `CMR:OCO3_L2_Lite_FP`, `NIES GOSAT L2`
- **Spatial resolution:** S5P CH4 ~7x5.5 km; OCO ~1.3x2.25 km soundings; GOSAT ~10 km (sparse)
- **Temporal resolution:** S5P CH4 daily (OFFL only); OCO-2 16-day; OCO-3 ISS+SAM snapshots; GOSAT 3-day
- **Latency:** S5P CH4 OFFL ~5 d; OCO Lite ~weeks; GOSAT NIES weeks
- **Coverage:** S5P global incl. India; OCO/GOSAT global but sparse swaths/soundings
- **Access:** GEE: COPERNICUS/S5P/OFFL/L3_CH4. OCO/GOSAT via NASA GES DISC/CMR (OCO2_L2_Lite_FP, OCO3_L2_Lite_FP) and NIES GOSAT DHF (limited/no GEE).
- **Pipeline role:** Co-pollutant combustion covariates (CH4) + biogenic proxy (SIF) to classify HCHO source (combustion vs biogenic) in Objective 2.
- **Gap-fill / cross-verification role:** Disambiguates HCHO: HCHO+CH4+CO+XCO2 co-located => fire/anthropogenic; HCHO+high SIF, low CH4 => biogenic. Independent verification of fire-HCHO attribution.
- **India relevance:** IGP residue-burning/paddy/landfill/oil&gas co-emit CH4+CO+CO2 with HCHO; XCH4 plumes + XCO2 enhancements corroborate combustion; OCO SIF constrains vegetation productivity (biogenic VOC/HCHO).
- **Caveats:** S5P CH4 strict QA, albedo/aerosol-sensitive, sparse over cloudy IGP winter. OCO/GOSAT very sparse swaths — point-validation only, not gridded input.

### Landsat 8/9 OLI/TIRS (C2 L2) + ECOSTRESS LSTE (high-res thermal/optical)

- **Provider:** USGS/NASA (Landsat); NASA JPL/ISS (ECOSTRESS)
- **Instrument:** OLI (VNIR/SWIR)+TIRS thermal; ECOSTRESS PHyTIR thermal radiometer
- **Products:** Landsat 30 m surface reflectance (NDVI/NDBI); Landsat Surface Temperature ST_B10 (30 m); dNBR burn scars; ECOSTRESS LST&emissivity (70 m, multi-time-of-day)
- **GEE / API IDs:** `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`, `NASA/ECOSTRESS/L2T_LSTE/V2`
- **Spatial resolution:** Landsat 30 m (TIRS->30 m); ECOSTRESS 70 m
- **Temporal resolution:** Landsat 8-day combined (16-day each); ECOSTRESS ISS non-sunsync (diurnal sampling)
- **Latency:** Landsat C2 L2 ~1-2 days; ECOSTRESS days-weeks
- **Coverage:** Global incl. India
- **Access:** GEE: LANDSAT/LC08/C02/T1_L2, LANDSAT/LC09/C02/T1_L2 (ST_B10, SR_B*); NASA/ECOSTRESS/L2T_LSTE/V2 (LST band).
- **Pipeline role:** High-res covariates (LST, NDVI, NDBI, imperviousness) for AOD/AQI downscaling + land-use regression; burn-scar verification of fire products.
- **Gap-fill / cross-verification role:** Fills SPATIAL-resolution gap (bridges 1-7 km columns to ~100 m urban exposure); ECOSTRESS fills DIURNAL LST gap of sun-synchronous sensors; dNBR independently verifies FIRMS fires, reducing false positives.
- **India relevance:** 30-70 m LST/NDVI/NDBI downscale coarse AOD/TROPOMI to neighbourhood AQI over Indian cities; ECOSTRESS multi-time LST proxies PBL/mixing; Landsat dNBR maps Himalayan/central-India forest-fire burn scars to validate FIRMS.
- **Caveats:** Landsat 8-16 day revisit + cloud loss (not daily); ECOSTRESS irregular ISS revisit/coverage gaps; thermal stripe/emissivity QA. Covariates, not standalone daily inputs.

### SRTM & Copernicus DEM GLO-30 (terrain for downscaling & dispersion)

- **Provider:** NASA/USGS (SRTM); ESA/Airbus (CopDEM/TanDEM-X)
- **Instrument:** SRTM C-band radar (2000); CopDEM TanDEM-X interferometry
- **Products:** Elevation DSM/DTM; derived slope/aspect/TPI/TRI; valley/basin depth index for cold-pool & dispersion
- **GEE / API IDs:** `USGS/SRTMGL1_003`, `COPERNICUS/DEM/GLO30`, `MERIT/DEM/v1_0_3`
- **Spatial resolution:** SRTM 30 m (1 arc-sec); CopDEM GLO-30 30 m
- **Temporal resolution:** Static
- **Latency:** Static
- **Coverage:** Global; SRTM 60N-56S (all India); CopDEM global
- **Access:** GEE: USGS/SRTMGL1_003 (image); COPERNICUS/DEM/GLO30 (ImageCollection, band DEM); derive slope via ee.Terrain. MERIT/DEM as hydro-corrected alt.
- **Pipeline role:** Static terrain covariates (elevation, slope, aspect, TPI, valley depth) for AQI ML + dispersion/transport modeling.
- **Gap-fill / cross-verification role:** Fills PHYSICAL-process gap: explains spatial AQI variance unresolved by coarse columns; constrains wind-channeling in HCHO transport analysis; flags basins prone to accumulation.
- **India relevance:** IGP topographic confinement + Himalayan foothills control pollutant trapping/transport; terrain drives valley cold pools (Dehradun, Kashmir) and channels smoke — essential dispersion + downscaling covariate.
- **Caveats:** CopDEM is DSM (canopy/buildings bias urban); SRTM voids in steep Himalaya (use CopDEM there); static, no temporal info.

### ESA WorldCover & MODIS MCD12Q1 land cover (land-use covariate)

- **Provider:** ESA (WorldCover); NASA/USGS (MODIS)
- **Instrument:** Sentinel-1/2 (WorldCover); MODIS Terra/Aqua
- **Products:** WorldCover 11-class 10 m LC (2020 v100, 2021 v200); MODIS IGBP/PFT land cover 500 m annual; cropland/urban/forest masks; Dynamic World NRT LC
- **GEE / API IDs:** `ESA/WorldCover/v200`, `ESA/WorldCover/v100`, `MODIS/061/MCD12Q1`, `GOOGLE/DYNAMICWORLD/V1`
- **Spatial resolution:** WorldCover 10 m; MODIS 500 m; Dynamic World 10 m
- **Temporal resolution:** WorldCover 2020/2021 epochs; MODIS annual; Dynamic World near-real-time
- **Latency:** Static/annual; Dynamic World ~days
- **Coverage:** Global incl. India
- **Access:** GEE: ESA/WorldCover/v200, ESA/WorldCover/v100, MODIS/061/MCD12Q1, GOOGLE/DYNAMICWORLD/V1.
- **Pipeline role:** Land-use categorical covariates for LUR + AQI ML; source-region masking for HCHO attribution.
- **Gap-fill / cross-verification role:** Fills LAND-USE attribution gap: maps coarse-column enhancements to source types (cropland=burning, built=traffic, forest=fire/biogenic), improving HCHO source classification and exposure context.
- **India relevance:** Cropland mask (IGP rice-wheat belt) localizes residue-burning HCHO source areas; urban/built anchors traffic/industrial AQI; forest class for forest-fire HCHO zones — key land-use-regression predictor.
- **Caveats:** WorldCover static 2020/21 (use Dynamic World for currency); 10 m class noise; MODIS 500 m coarse for cities.

### Population: WorldPop 100m & GPWv4.11 (exposure mapping)

- **Provider:** WorldPop (Southampton); CIESIN/NASA SEDAC
- **Instrument:** Modeled (census + RS dasymetric)
- **Products:** WorldPop 100 m population count/density; WorldPop age/sex structure; GPWv411 count & density (~1 km); UN-adjusted variants
- **GEE / API IDs:** `WorldPop/GP/100m/pop`, `CIESIN/GPWv411/GPW_Population_Density`, `CIESIN/GPWv411/GPW_Population_Count`, `WorldPop/GP/100m/pop_age_sex`
- **Spatial resolution:** WorldPop ~100 m; GPW ~1 km (30 arc-sec)
- **Temporal resolution:** WorldPop annual 2000-2020; GPW 5-yr 2000-2020
- **Latency:** Static/annual
- **Coverage:** Global incl. India
- **Access:** GEE: WorldPop/GP/100m/pop, WorldPop/GP/100m/pop_age_sex; CIESIN/GPWv411/GPW_Population_Density, .../GPW_Population_Count.
- **Pipeline role:** Exposure-mapping layer; population-weighting of AQI; density as anthropogenic-emission proxy covariate.
- **Gap-fill / cross-verification role:** Fills EXPOSURE/impact gap: turns concentration maps into health-relevant exposure metrics for hotspot ranking; density doubles as emission proxy covariate.
- **India relevance:** Converts surface AQI maps to POPULATION-WEIGHTED exposure; ranks high-exposure IGP megacities (Delhi/Kanpur/Lucknow) and prioritizes HCHO hotspots by people affected.
- **Caveats:** Modeled not measured (dasymetric error in peri-urban); epoch lag (2020) underestimates recent growth; GPW coarse (1 km).

### VIIRS Black Marble VNP46A2 (nighttime lights / combustion proxy)

- **Provider:** NASA (VIIRS Land, GSFC)
- **Instrument:** VIIRS Day/Night Band (DNB)
- **Products:** Gap-filled BRDF-corrected nighttime radiance (500 m daily); gas-flaring/kiln combustion signal; monthly VNP46A3/A4
- **GEE / API IDs:** `NASA/VIIRS/002/VNP46A2`, `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`
- **Spatial resolution:** 500 m
- **Temporal resolution:** Daily (VNP46A2); monthly (VNP46A3)
- **Latency:** ~1 day to days
- **Coverage:** Global incl. India
- **Access:** GEE: NASA/VIIRS/002/VNP46A2 (band Gap_Filled_DNB_BRDF_Corrected_NTL); monthly NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG.
- **Pipeline role:** Anthropogenic-combustion/activity covariate for AQI ML & land-use regression.
- **Gap-fill / cross-verification role:** Fills EMISSION-PROXY gap: nighttime radiance correlates with traffic/industry/kiln combustion not captured by land cover; complements daytime FIRMS fire with persistent-source intensity.
- **India relevance:** Nighttime-lights proxy for urban/industrial combustion & activity over Indian cities; brick-kiln/flaring hotspots; covariate for anthropogenic emission intensity in AQI LUR.
- **Caveats:** Confounded by lighting infrastructure (not pure emissions); moonlight/cloud residuals; saturation in city cores. Proxy only.

### GPM IMERG V07 (precipitation / wet scavenging)

- **Provider:** NASA/JAXA
- **Instrument:** GPM constellation (DPR + GMI merged, IR-gauge)
- **Products:** Half-hourly precipitation (mm/hr, precipitationCal); daily accumulations; monthly IMERG
- **GEE / API IDs:** `NASA/GPM_L3/IMERG_V07`, `NASA/GPM_L3/IMERG_MONTHLY_V07`, `UCSB-CHG/CHIRPS/DAILY`
- **Spatial resolution:** ~11 km (0.1 deg)
- **Temporal resolution:** 30-min; daily; monthly
- **Latency:** Early ~4 h, Late ~14 h, Final ~3.5 months
- **Coverage:** Global 60N-60S incl. India
- **Access:** GEE: NASA/GPM_L3/IMERG_V07 (precipitation/precipitationCal); monthly NASA/GPM_L3/IMERG_MONTHLY_V07; alt UCSB-CHG/CHIRPS/DAILY.
- **Pipeline role:** Wet-scavenging meteorological covariate (precip, antecedent rain) for AQI ML.
- **Gap-fill / cross-verification role:** Fills PROCESS + temporal-gap interpretation: explains sudden column drops (washout vs emission change); rainy/cloudy days flag when TROPOMI/INSAT AOD retrievals are missing.
- **India relevance:** Rain wet-scavenges aerosols/soluble gases (SO2, HCHO) — key driver of day-to-day IGP AQI drops; monsoon/winter-rain timing explains washout; vital ML covariate.
- **Caveats:** ~11 km coarse; IR estimates biased for convective extremes; Final run too slow for NRT (use Early/Late).

### OSM road & industrial / GHSL built layers (land-use regression)

- **Provider:** OpenStreetMap (Overpass/Geofabrik); JRC GHSL; DLR WSF; GRIP roads
- **Instrument:** Crowd-sourced vector + EO-derived built-up
- **Products:** Road network density / distance-to-major-road; industrial & landuse polygons; GHS-BUILT-S built-up surface (100 m); GHS-POP
- **GEE / API IDs:** `JRC/GHSL/P2023A/GHS_BUILT_S`, `JRC/GHSL/P2023A/GHS_POP`, `DLR/WSF/WSF2015/v1`, `OSM Overpass API (offline)`
- **Spatial resolution:** Vector (rasterize to 100 m); GHSL 100 m; WSF 10-30 m
- **Temporal resolution:** OSM live; GHSL/WSF epochs
- **Latency:** OSM live; GHSL/WSF static
- **Coverage:** Global incl. India (variable OSM completeness)
- **Access:** OSM Overpass API / Geofabrik India extract (offline, rasterize -> GEE asset); GEE: JRC/GHSL/P2023A/GHS_BUILT_S, JRC/GHSL/P2023A/GHS_POP, DLR/WSF/WSF2015/v1.
- **Pipeline role:** Core land-use-regression predictors (road density/distance, industrial/built proximity) for fine-scale AQI.
- **Gap-fill / cross-verification role:** Fills MICRO-SCALE SPATIAL gap: encodes sub-km traffic/industrial gradients invisible to 1-7 km columns, sharpening downscaled AQI and local hotspot identification.
- **India relevance:** Distance-to-major-road, road density, industrial proximity are classic LUR predictors of NO2/PM near Indian traffic corridors & industrial clusters (NCR, Kanpur) that coarse satellites cannot resolve.
- **Caveats:** OSM completeness uneven across rural India; not directly in GEE (preprocess to raster); sparse industrial tagging; mostly static.

## How These Sources Fill Each Other's Gaps

No single feed is complete, so the catalog is built around mutual gap-filling, with each family's `gapfill_role` defining how it backstops the others:

- **Cloud and low-SNR gaps.** Sentinel-5P TROPOMI HCHO/NO2 are the core columns but drop out under cloud and over low-VOC, low-signal scenes; temporal compositing (8-day to monthly), UV Aerosol Index smoke-flagging, fire covariates (SLSTR/VIIRS FRP, FIRMS), and ML imputation rebuild those gaps. IR sounders (IASI/AIRS/CrIS NH3, CO) and gap-free reanalysis (CAMS, GEOS-CF, MERRA-2) provide cloud-tolerant priors where the UV-Vis retrievals fail.
- **Temporal gaps.** Polar orbiters give roughly one daily overpass; geostationary platforms (INSAT-3D/3DR, Himawari AHI, FY-4 AGRI, GK-2A, GEMS, TEMPO) and the hourly reanalyses (ERA5, IMDAA) supply sub-daily and diurnal sampling, catching the morning/evening pollution and fire cycles that single overpasses miss.
- **Coarse spatial resolution.** Coarse INSAT-3D AOD and ~0.1° reanalysis grids are sharpened and cross-validated against 1 km MODIS MAIAC AOD, 375 m VIIRS / 1 km MODIS active fire, 20 m Sentinel-2 burn scars, and high-resolution land-use, terrain, population, and nighttime-lights layers used as downscaling predictors.
- **Missing pollutants and vertical structure.** HCHO supplies the VOC/pyrogenic axis that NO2/CO/SO2 cannot; NH3 and CO from IR sounders act as biomass-burning co-tracers corroborating HCHO fire signals; CALIPSO/EarthCARE/MISR aerosol profiles and Aeolus winds add the vertical and transport context needed to convert columnar AOD into surface PM2.5 and to track HCHO/smoke advection.
- **Ground truth and validation.** CPCB CAAQMS, AERONET, OpenAQ, AirNow, SAFAR, and PurpleAir close the loop, providing the surface labels for model training and the independent reference for validating satellite-derived AQI and AOD.

---

*Generated from the PS3 research corpus: 12 families, 85 datasets.*
