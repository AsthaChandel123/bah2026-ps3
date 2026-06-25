# Runbook — `aqi_india`

How to set up, authenticate, configure, run each pipeline stage, run the offline
synthetic demo, run the tests, and reproduce results for the BAH 2026 (ISRO) PS3
project. For the architecture see [`ARCHITECTURE.md`](ARCHITECTURE.md); for the
frozen API and synthetic-data schema see [`DEV_CONTRACT.md`](DEV_CONTRACT.md).

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Setup (venv + editable install + extras)](#2-setup-venv--editable-install--extras)
3. [The offline synthetic demo (no credentials)](#3-the-offline-synthetic-demo-no-credentials)
4. [Credentials — GEE / CDS / Earthdata / MOSDAC](#4-credentials--gee--cds--earthdata--mosdac)
5. [Configuration via Hydra](#5-configuration-via-hydra)
6. [Running each pipeline stage via the `aqi` CLI](#6-running-each-pipeline-stage-via-the-aqi-cli)
7. [Tests](#7-tests)
8. [Reproducing results (DVC)](#8-reproducing-results-dvc)
9. [Data-access table (real sources)](#9-data-access-table-real-sources)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

- **Python 3.11** (the package targets 3.11; other 3.x may work but is untested).
- **git**, and a POSIX shell (`make` is convenient but optional).
- For real-data paths only: a Google Earth Engine account, a Copernicus CDS
  account, a NASA Earthdata login, and a MOSDAC login (see
  [§4](#4-credentials--gee--cds--earthdata--mosdac)). **None of these are needed
  for the offline demo or the tests.**

---

## 2. Setup (venv + editable install + extras)

```bash
git clone <repo-url> bah2026-ps3
cd bah2026-ps3

python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# LIGHT install — runs the full offline demo and all core tests:
pip install -e .

# or via the Makefile (identical):
make setup
```

The package **imports under the light set alone** — heavy/credentialed
dependencies (torch, earthengine-api, cdsapi, …) are lazy-imported inside the
functions that use them. Install extras only when you exercise the corresponding
real-data path:

```bash
pip install -e ".[deep]"        # torch / SA-ConvLSTM / CNN-LSTM / GNN / U-Net inpaint
pip install -e ".[gee]"         # earthengine-api, geemap, cdsapi, earthaccess, STAC
pip install -e ".[serve]"       # titiler, lonboard, pydeck, streamlit
pip install -e ".[geo]"         # pykrige, gstools, scikit-gstat, xesmf, hdbscan
pip install -e ".[transport]"   # pysplit (HYSPLIT)
pip install -e ".[dev]"         # pytest, hypothesis, ruff, mypy
pip install -e ".[deep,gee,serve,geo,transport,dev]"   # everything
```

Verify the install:

```bash
aqi --help        # shows ingest/fuse/features/train/maps/hotspots/transport/validate/serve/demo
aqi info          # prints package version + environment summary
```

---

## 3. The offline synthetic demo (no credentials)

The demo synthesizes a tiny India grid + CPCB-surrogate stations + FIRMS-surrogate
fires on the **light** dependency set and runs the full chain
(fusion → features → model → NAQI maps → hotspots), writing maps and figures
locally. It needs **no GEE/CDS/Earthdata credentials**.

```bash
aqi demo run
# or
make demo
```

Outputs land under the paths fixed by the
[synthetic-data contract](DEV_CONTRACT.md) (§6) and `conf/config.yaml`:

| Output | Path |
| --- | --- |
| Fused gridded cube | `data/processed/grid.nc` |
| Station daily labels | `data/processed/stations.parquet` |
| Fire detections | `data/processed/fires.parquet` |
| Predicted-concentration AQI grid | `data/processed/aqi_grid.nc` |
| Confirmed HCHO hotspot polygons | `data/processed/hotspots.geoparquet` |
| **India AQI + HCHO hotspot figures** | `reports/figures/` |

The flagship demo figures (India surface-AQI map and the HCHO hotspot map with
fire overlay) are written to `reports/figures/` and are the artifacts to surface
in a presentation.

---

## 4. Credentials — GEE / CDS / Earthdata / MOSDAC

Required **only** for real-data ingestion (`aqi ingest …`). Store credentials in
the environment or each tool's config file — **never commit them**.

**Google Earth Engine** (`[gee]` extra):

```bash
earthengine authenticate                 # interactive OAuth (one-time)
# or a service account for headless/CI runs:
export EARTHENGINE_SERVICE_ACCOUNT="svc@project.iam.gserviceaccount.com"
export EARTHENGINE_PRIVATE_KEY="/secure/path/key.json"   # never commit
```

`aqi_india.ingest.gee_auth` initializes `ee` lazily and supports both the OAuth
token and the service-account key path.

**Copernicus CDS** (ERA5, CAMS — `[gee]` extra pulls `cdsapi`). Create
`~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <UID>:<API-KEY>
```

**NASA Earthdata** (MERRA-2, MODIS/VIIRS L2, FIRMS — via `earthaccess`). Create
`~/.netrc`:

```
machine urs.earthdata.nasa.gov login <username> password <password>
```

A **FIRMS map key** (for the fire CSV/area API) is read from `FIRMS_MAP_KEY`:

```bash
export FIRMS_MAP_KEY="<your-firms-map-key>"
```

**MOSDAC** (INSAT-3D AOD) requires a free registered login; downloads are manual
HDF5 placed under `data/raw/insat_aod/` (see the data-access table in
[§9](#9-data-access-table-real-sources)). **CPCB** labels come from the
data.gov.in OGD API and/or the CCR repository.

---

## 5. Configuration via Hydra

All AOIs, dates, dataset IDs, grids, model hyperparameters, AQI breakpoints, and
hotspot settings live in `conf/` — **there are no magic constants in code.** The
root `conf/config.yaml` `defaults` are:

```yaml
defaults:
  - aoi: india
  - dates: train
  - grid: latlon_0p05
  - datasets: cpcb
  - model: cnn_lstm
  - aqi: cpcb_naqi
  - hotspot: getis_ord
```

Override any group on the command line:

```bash
# Train on the Indo-Gangetic Plain over the Oct–Nov burning window with SA-ConvLSTM:
aqi train saconvlstm aoi=igp dates=burn_oct_nov model=saconvlstm

# Detect hotspots over the Punjab–Haryana AOI using ST-DBSCAN settings:
aqi hotspots detect aoi=punjab_haryana hotspot=stdbscan

# Switch the AOD/column source to Sentinel-5P for an ingest run:
aqi ingest s5p datasets=s5p dates=burn_oct_nov
```

Config groups available: `aoi/{india,igp,punjab_haryana,forest_belt}` ·
`dates/{train,burn_oct_nov,burn_apr_may}` ·
`grid/{h3_res7,latlon_0p05,latlon_1km}` ·
`datasets/{s5p,insat_aod,maiac,firms,era5,imdaa,merra2,cams,cpcb}` ·
`model/{saconvlstm,cnn_lstm,lightgbm,gnn,stack}` · `aqi/cpcb_naqi` ·
`hotspot/{getis_ord,lisa,stdbscan,hysplit}`.

Each `datasets/<name>.yaml` is namespaced (`# @package datasets.<name>`) so it
composes under `cfg.datasets.<name>` (e.g. `cfg.datasets.s5p.gee_collections.hcho`).
The composed config exposes `cfg.project.{name,version,seed}`,
`cfg.paths.{data_raw,…,reports}`, `cfg.run.{device,num_workers,log_level}`,
`cfg.aoi.{name,bbox,crs,utm}`, `cfg.dates.{start,end,freq}`, `cfg.grid.{…}`,
`cfg.model.{…}`, `cfg.aqi.{…}`, and `cfg.hotspot.{…}`.

---

## 6. Running each pipeline stage via the `aqi` CLI

Every stage exposes a `run(cfg)` core and a Typer command. The real-data stages
require the relevant extras and credentials from [§4](#4-credentials--gee--cds--earthdata--mosdac).

| Command | Stage | Needs |
| --- | --- | --- |
| `aqi ingest s5p` | Pull S5P L3 columns (QA-masked) + co-locate to CPCB | `[gee]`, GEE |
| `aqi ingest cpcb` | Pull/QA CPCB CAAQMS labels (data.gov.in OGD / CCR) | network |
| `aqi fuse gapfill` | DINEOF → U-Net → kriging gap-fill + QM/ML bias-correct | `[geo]` (`[deep]` for U-Net) |
| `aqi features build` | H3 index + physics/met/static/temporal feature matrix | light (`[gee]`/`[geo]` for real covars) |
| `aqi train lightgbm` | Train the LightGBM tabular baseline | light + `lightgbm` |
| `aqi train saconvlstm` | Train the physics-guided SA-ConvLSTM | `[deep]` |
| `aqi maps daily` | Predict surface concentrations → CPCB NAQI → daily AQI COG | `[deep]` (or LightGBM) |
| `aqi hotspots detect` | z-anomaly → Gi*/LISA/percentile ≥2-of-3 consensus → polygons | light (`[geo]` for HDBSCAN) |
| `aqi transport hysplit` | HYSPLIT back/forward + trajCluster + CWT/PSCF | `[transport]` |
| `aqi validate run` | CV ladder + RMSE/R/MAE + AQI confusion + Taylor | light |
| `aqi serve dashboard` | Launch the Streamlit + TiTiler interactive app | `[serve]` |
| `aqi demo run` | Offline synthetic end-to-end demo | light only |

A typical end-to-end real run (after credentials are set):

```bash
aqi ingest cpcb dates=train
aqi ingest s5p  datasets=s5p dates=train
aqi fuse gapfill
aqi features build
aqi train saconvlstm aoi=india model=saconvlstm
aqi maps daily
aqi validate run
aqi hotspots detect aoi=igp dates=burn_oct_nov
aqi transport hysplit aoi=igp dates=burn_oct_nov
aqi serve dashboard
```

---

## 7. Tests

```bash
make test-core   # pytest tests/test_naqi.py tests/test_h3.py tests/test_metrics.py
make test        # full suite
make lint        # ruff check src tests
```

The core suite runs on the **light** set with no credentials and covers the NAQI
golden vectors (sub-index → max, CO in mg/m³, O3/CO max(8h,1h), ≥3-pollutant + PM
validity), H3 roundtrip, and the metric correctness checks. Keep this gate green.

---

## 8. Reproducing results (DVC)

The pipeline is a DVC DAG (`dvc.yaml`); each node is one CLI command with declared
deps/outs. Artifacts are tracked against a remote so a judge can fetch and rebuild
deterministically:

```bash
dvc pull          # fetch tracked inputs/outputs from the configured remote
dvc repro         # rebuild only what changed, in dependency order
dvc dag           # render the pipeline graph
```

Determinism is enforced by fixed seeds (`cfg.project.seed`, default 42; `torch`,
`numpy`, `random`, `PYTHONHASHSEED`) and the cuDNN deterministic flag. MLflow logs
every training run's params/metrics/artifacts. For full environment parity,
`docker compose up` bakes the geospatial + ML stack (mount GEE/Earthdata/CDS creds
via env, never committed).

---

## 9. Data-access table (real sources)

Pointers to the authoritative sources behind each ingest adapter. Full provider,
latency, and role detail for all 85 datasets are in
[`DATA_SOURCES.md`](DATA_SOURCES.md).

| Source | What | Access | Adapter |
| --- | --- | --- | --- |
| **Sentinel-5P TROPOMI** | NO2/SO2/CO/O3/HCHO/AER_AI/AER_LH/CLOUD L3 columns | GEE `COPERNICUS/S5P/{OFFL,NRTI}/L3_*`; DLR S5P-PAL; GES DISC | `ingest/s5p.py` |
| **MODIS MAIAC AOD** | 1km AOD backbone (`Optical_Depth_055`) | GEE `MODIS/061/MCD19A2_GRANULES` | `ingest/maiac.py` |
| **INSAT-3D/3DR AOD** | PS-mandated diurnal AOD (HDF5) | MOSDAC (registered login), VEDAS | `ingest/insat_aod.py` |
| **FIRMS fire** | VIIRS 375m + MODIS active-fire counts/FRP | GEE `FIRMS`; FIRMS API (`FIRMS_MAP_KEY`) | `ingest/firms.py` |
| **ERA5** | BLH, 10 m wind, RH, T2m, precip, SSRD | CDS `reanalysis-era5-single-levels` (`~/.cdsapirc`); GEE ERA5_LAND (no BLH) | `ingest/era5_cds.py` |
| **IMDAA** | 12 km India regional reanalysis (native training grid) | NCMRWF `nwp.ncmrwf.gov.in/reanalysis` | `ingest/imdaa.py` |
| **MERRA-2** | Speciated aerosol + PBLH (independent met cross-check) | GES DISC / `earthaccess` (`~/.netrc`) | `ingest/merra2.py` |
| **CAMS / GFAS** | Gap-free composition prior + fire-emission flux | ADS (Atmosphere Data Store), `cdsapi` | `ingest/cams.py` |
| **CPCB CAAQMS** | Surface PM/gases — the training labels | data.gov.in OGD API; CCR repository | `ingest/cpcb.py` |
| **AERONET** | AOD input validation (Kanpur, Gandhi College, …) | NASA AERONET web service | `ingest/aeronet.py` |

> India geometry: GADM v4.1 for development; switch to Survey of India / Bhuvan
> boundaries for the final ISRO submission.

---

## 10. Troubleshooting

- **`ImportError` for torch / ee / cdsapi on import** — should never happen; these
  are lazy-imported. If it does, you hit a real-data function without its extra:
  install the matching extra (e.g. `pip install -e ".[deep]"`).
- **GEE `EEException: not initialized`** — run `earthengine authenticate` or set
  the `EARTHENGINE_SERVICE_ACCOUNT` / `EARTHENGINE_PRIVATE_KEY` env vars.
- **CDS / Earthdata 401** — check `~/.cdsapirc` / `~/.netrc`; CDS requires
  accepting the dataset licence once on the web portal.
- **TiTiler import error** — the metapackage was dropped late 2025; install the
  submodules explicitly (`titiler.core`, `titiler.mosaic`) via `pip install -e ".[serve]"`.
- **`aqi <cmd>` prints "wired in later"** — that stage's module is still a stub in
  this build; the offline `aqi demo run` exercises the implemented chain.
- **Empty `reports/figures/`** — run `aqi demo run` first; the demo writes the
  flagship India AQI and HCHO hotspot figures there.
