# DEV_CONTRACT — aqi_india (BAH 2026 PS3, Foundation phase)

**Status:** binding. This document is written by the Foundation agent and is the
single source of truth for the package layout, the public API signatures the
core already exposes, the Hydra config keys, and the synthetic-data schema that
every module agent (and the Demo agent) **must** target so the modules compose.

If you need to deviate from anything here, raise it with the coordinator first —
other agents are coding against these signatures in parallel.

- Package: `aqi_india`, `src/` layout, Python 3.11, build backend hatchling.
- Console entrypoint: `aqi = "aqi_india.cli:app"`.
- Import rule: **the package must import with only the light dependency set.**
  Lazy-import torch / earthengine-api / cdsapi / earthaccess / pykrige / xesmf /
  hdbscan **inside the functions that use them**, never at module top level.
- Dependency groups (in `pyproject.toml`): light set in `[project].dependencies`;
  extras `deep`, `gee`, `serve`, `geo`, `transport`, `dev`.

---

## 1. Package tree

```
src/aqi_india/
├─ __init__.py            # __version__ = "0.1.0"
├─ cli.py                 # Typer root app `app` + sub-Typers (see §5)
├─ data/                  # bundled static assets (offline India boundary, LUTs)
│  ├─ __init__.py
│  └─ india_simplified.geojson
├─ aqi/
│  ├─ breakpoints.py      # CPCB NAQI constant tables  (DONE, frozen)
│  └─ naqi.py             # pure vectorized NAQI engine (DONE, frozen)
├─ features/
│  └─ h3_index.py         # H3 v4/v3 fusion-key wrappers (DONE)
│     # TO BUILD: physics.py met.py static_covars.py temporal.py feature_matrix.py
├─ validation/
│  └─ metrics.py          # RMSE/MAE/MBE/R/R2/NMB/NME/IOA/RMA (DONE)
│     # TO BUILD: cv.py confusion.py plots.py
├─ utils/
│  ├─ io.py geo.py logging.py     (DONE)
├─ ingest/    # TO BUILD: gee_auth.py s5p.py insat_aod.py maiac.py firms.py
│             #           era5_cds.py imdaa.py merra2.py cams.py cpcb.py
│             #           aeronet.py exporters.py
├─ fusion/    # TO BUILD: regrid.py dineof.py inpaint_unet.py kriging.py
│             #           biascorrect_qm.py biascorrect_ml.py harmonize.py
├─ models/    # TO BUILD: datasets.py saconvlstm.py convlstm.py cnn_lstm.py
│             #           lightgbm_model.py gnn.py stack.py losses.py
│             #           train.py predict.py uq.py
├─ hotspot/   # TO BUILD: climatology.py getis_ord.py lisa.py ehsa.py
│             #           cluster.py consensus.py
├─ transport/ # TO BUILD: fire_periods.py fire_hcho_corr.py hysplit.py
│             #           traj_cluster.py cwt_pscf.py polar.py inventory.py
├─ viz/       # TO BUILD: cog_export.py pmtiles.py titiler_app.py
│             #           dashboard_streamlit.py figures.py static_maplibre/
└─ sim/       # TO BUILD: synthetic data generators (see §6) — owned by Demo agent

conf/         # Hydra tree (see §4)
tests/        # test_naqi.py test_h3.py test_metrics.py (DONE, green)
data/{raw,interim,processed,models,artifacts}/   # gitignored except .gitkeep
reports/figures/
```

Every subpackage has an `__init__.py`. Add new modules under the listed homes;
do not introduce new top-level packages without coordinator sign-off.

---

## 2. Frozen public API — DO NOT change these signatures

These are implemented, unit-tested, and depended on by downstream modules.

### `aqi_india.aqi.breakpoints`
```python
AQI_CATEGORY_EDGES: np.ndarray          # [0,50,100,200,300,400,500]
AQI_BAND_LO: np.ndarray                 # [0,51,101,201,301,401]
AQI_BAND_HI: np.ndarray                 # [50,100,200,300,400,500]
AQI_MAX: float = 500.0
BREAKPOINTS: dict[str, np.ndarray]      # length-7 lower edges + top cap per pollutant
POLLUTANTS: tuple[str, ...]             # ("pm25","pm10","no2","so2","co","o3","nh3","pb")
PM_POLLUTANTS: tuple[str, ...]          # ("pm25","pm10")
MIN_VALID_POLLUTANTS: int = 3
AVERAGING_HOURS: dict[str, int]         # CO/O3 = 8, rest = 24
UNITS: dict[str, str]                   # co="mg/m3", rest="ug/m3"
CATEGORY_NAMES: tuple[str, ...]         # Good..Severe
CATEGORY_COLORS: tuple[str, ...]        # 6 hex colors
category_index(aqi: float) -> int       # 0..5
category_name(aqi: float) -> str
category_color(aqi: float) -> str
```
**Canonical pollutant keys** are the strings in `POLLUTANTS`. Use them as
DataFrame column names, xarray data-var names, and dict keys everywhere.

### `aqi_india.aqi.naqi`
```python
sub_index(conc, pollutant: str) -> np.ndarray
    # scalar or array; CO in mg/m3, rest ug/m3; NaN/negative -> NaN; clamp [0,500]
aqi_from_subindices(subindices, pollutants=None, *, return_responsible=False)
    # subindices: dict[str, array] OR array (n_pollutants, *grid) + pollutants list
    # returns aqi array, or (aqi, responsible_object_array) if return_responsible
responsible_pollutant(subindices, pollutants=None) -> np.ndarray
category(aqi: float) -> tuple[str, str]          # (name, hex_color)
compute_aqi(df, *, column_map=None, out_prefix="") -> pd.DataFrame
    # adds {prefix}aqi (float), {prefix}aqi_category (str|None),
    #      {prefix}aqi_responsible (str|None)
compute_aqi_grid(ds, *, var_map=None, aqi_name="aqi",
                 responsible_name="aqi_responsible") -> xr.Dataset
    # returns Dataset with float32 `aqi` and int16 `aqi_responsible` code (-1=invalid);
    # code->name mapping in aqi_responsible.attrs["pollutant_codes"]
```
**Validity rule (do not reimplement elsewhere — call this engine):** a cell is
valid only if `>= 3` pollutant sub-indices are present AND `>= 1` of
`pm25`/`pm10` is present; otherwise AQI is NaN.

### `aqi_india.features.h3_index`
```python
DEFAULT_RES: int = 7
h3_api_version() -> int                          # 3 or 4 (auto-detected)
latlng_to_cell(lat, lon, res=7) -> str
cell_to_latlng(cell) -> tuple[float, float]
cell_to_parent(cell, res) -> str
grid_disk(cell, k=1) -> list[str]
grid_disk_size(k) -> int                         # 3k(k+1)+1
get_resolution(cell) -> int
cells_for_points(df, lat="lat", lon="lon", res=7, *, out_col="h3") -> pd.DataFrame
cells_for_arrays(lats, lons, res=7) -> np.ndarray   # object array, None for non-finite
parents_for_cells(cells, res) -> np.ndarray
neighborhood(cell, k=1) -> list[str]             # alias of grid_disk
assign_grid_cells(grid, lat="lat", lon="lon", res=7) -> xr.DataArray  # dims (lat,lon)
```
**The H3 cell is the universal join key.** Use `out_col="h3_res7"` to match the
station/fire schema column names in §6.

### `aqi_india.validation.metrics`
```python
rmse, mae, mbe, pearson_r, r2, nmb, nme, ioa  : (y_true, y_pred) -> float   # all NaN-safe
rma_slope_intercept(y_true, y_pred) -> tuple[float, float]                  # Deming/RMA
metrics_table(y_true, y_pred) -> dict[str, float]
    # keys: n, rmse, mae, mbe, r, r2, nmb, nme, ioa, rma_slope, rma_intercept
stratified_metrics(y_true, y_pred, bands: list[tuple[float,float]]) -> dict[str, dict]
```

### `aqi_india.utils`
```python
# io.py
ensure_dir(path) -> Path
load_netcdf/save_netcdf(ds, path, **kw)
load_parquet/save_parquet(df, path, **kw)
load_geoparquet/save_geoparquet(gdf, path, **kw)
load_geojson/save_geojson(gdf, path, **kw)
load_yaml/save_yaml(obj, path)
# geo.py
INDIA_BBOX = (68.0, 6.0, 98.0, 38.0)             # (min_lon,min_lat,max_lon,max_lat)
IGP_BBOX = (73.0, 24.0, 88.0, 31.0)
PUNJAB_HARYANA_BBOX = (73.5, 28.5, 77.5, 32.5)
FOREST_BELT_BBOX = (73.0, 18.0, 96.0, 31.0)
CRS_WGS84 = "EPSG:4326";  UTM_43N="EPSG:32643";  UTM_44N="EPSG:32644"
bbox_to_polygon(bbox) -> shapely Polygon
get_india_boundary(simplify=False) -> gpd.GeoDataFrame   # NE if available else bundled
clip_to_india(gdf) -> gpd.GeoDataFrame
# logging.py
get_logger(name=None, level=None) -> logging.Logger      # under "aqi_india" namespace
```

---

## 3. Conventions every module must follow

- **Lazy heavy imports.** `import torch` / `import ee` / `import cdsapi` / etc.
  go *inside* functions. The module must import under the light set alone.
- **Canonical names.** Pollutants: keys of `POLLUTANTS`. Spatial key column:
  `h3_res7`. Time dim/coord: `time` (daily, `datetime64[ns]`). Spatial dims:
  `lat`, `lon` (1-D coords, ascending).
- **CRS** is always EPSG:4326 for stored vector/raster unless a function name
  says otherwise; reproject to UTM 43N/44N only for metric ops.
- **Units.** Surface concentrations in CPCB units (CO `mg/m3`, rest `ug/m3`).
  Satellite columns in `mol/m2`. Convert at ingest, never inside the AQI engine.
- **Each stage** exposes a `run(cfg)` function (takes the composed Hydra config)
  **and** a Typer command already declared in `cli.py` (see §5). Fill the
  command body; keep the signature.
- **NaN handling.** Missing pixels/labels are `NaN`, never sentinel numbers.
- **Logging** via `get_logger("<stage>.<module>")`. No bare `print` in library code.
- **Determinism.** Seed from `cfg.project.seed` (default 42).

---

## 4. Hydra config keys (`conf/`)

Root `config.yaml` `defaults`: `aoi=india, dates=train, grid=latlon_0p05,
datasets=cpcb, model=cnn_lstm, aqi=cpcb_naqi, hotspot=getis_ord`. Override on the
CLI, e.g. `aqi train aoi=igp dates=burn_oct_nov model=saconvlstm`.

Top-level keys available in the composed config:
```
cfg.project.{name,version,seed}
cfg.paths.{data_raw,data_interim,data_processed,models,artifacts,reports}
cfg.run.{device,num_workers,log_level}
cfg.aoi.{name,description,bbox,crs,utm}
cfg.dates.{name,start,end,freq,...}
cfg.grid.{name,kind,resolution|resolution_deg,...}
cfg.datasets.<dataset>.{...}        # NOTE: namespaced — see below
cfg.model.{name,type,...}
cfg.aqi.{breakpoints,category_names,category_colors,min_valid_pollutants,...}
cfg.hotspot.{name,method,...}
```

Config groups and files:
- `aoi/`: `india, igp, punjab_haryana, forest_belt` — each has `bbox` (lon/lat).
- `dates/`: `train, burn_oct_nov, burn_apr_may`.
- `grid/`: `h3_res7` (res 7, parent 4), `latlon_0p05` (0.05deg), `latlon_1km`.
- `datasets/`: `s5p, insat_aod, maiac, firms, era5, imdaa, merra2, cams, cpcb`.
  **Each dataset file uses `# @package datasets.<name>`** so it composes under
  `cfg.datasets.<name>` (e.g. `cfg.datasets.s5p.gee_collections.hcho`). To load a
  source, override `datasets=s5p`. Asset IDs / endpoints / QA thresholds inside.
- `model/`: `saconvlstm, cnn_lstm, lightgbm, gnn, stack`.
- `aqi/cpcb_naqi.yaml`: documents the breakpoint constants (authoritative copy is
  the Python module `aqi_india.aqi.breakpoints`; keep them in sync).
- `hotspot/`: `getis_ord, lisa, stdbscan, hysplit`.

---

## 5. CLI surface (`aqi_india.cli:app`)

Root app `app` (Typer) with `aqi info` and these sub-Typers, each with at least
one stub command that prints a "wired in later" notice today:

| Sub-app     | Command(s) declared            | Module agent fills |
|-------------|--------------------------------|--------------------|
| `ingest`    | `s5p`, `cpcb`                  | ingest             |
| `fuse`      | `gapfill`                      | fusion             |
| `features`  | `build`                        | features           |
| `train`     | `saconvlstm`, `lightgbm`       | models             |
| `maps`      | `daily`                        | viz / models       |
| `hotspots`  | `detect`                       | hotspot            |
| `transport` | `hysplit`                      | transport          |
| `validate`  | `run`                          | validation         |
| `serve`     | `dashboard`                    | viz                |
| `demo`      | `run`                          | sim / Demo agent   |

Add commands by registering them on the existing sub-Typer objects in `cli.py`
(`ingest_app`, `fuse_app`, ...). Keep `aqi --help` working at all times.

---

## 6. SYNTHETIC-DATA SCHEMA (binding — the Demo agent produces this; everyone consumes it)

The `aqi_india.sim` package (owned by the Demo agent) generates a tiny, fully
synthetic India dataset on the light dependency set. All module agents must read
and write exactly these schemas so the pipeline chains end-to-end.

### 6.1 Gridded fields — xarray `Dataset` named `grid`
- **Dims/coords:** `time` (daily `datetime64[ns]`), `lat`, `lon` (ascending 1-D).
- **Domain:** `INDIA_BBOX = (68, 6, 98, 38)` at **0.25 deg** resolution
  (lon 68.0..98.0 step 0.25, lat 6.0..38.0 step 0.25).
- **data_vars** (all `float32`, dims `(time, lat, lon)`):
  ```
  aod        # MAIAC-like 550nm AOD (unitless)
  no2_col    # tropospheric NO2 column (mol/m2)
  so2_col    # SO2 column (mol/m2)
  co_col     # CO column (mol/m2)
  o3_col     # O3 column (mol/m2)
  hcho_col   # tropospheric HCHO column (mol/m2)   <- Objective-2 target field
  blh        # boundary-layer height (m)
  rh         # relative humidity (%)
  wind_u     # 10 m u-wind (m/s)
  wind_v     # 10 m v-wind (m/s)
  t2m        # 2 m temperature (K)
  ssrd       # surface solar radiation downwards (J/m2 or W/m2)
  ```
- **File:** `data/processed/grid.nc` (NetCDF4, via `utils.io.save_netcdf`).
  Raw per-source pulls live under `data/raw/<source>/...`.

### 6.2 Stations — GeoDataFrame / parquet (CPCB surrogate, the labels)
- **Columns** (one row per station-day after the hourly->daily aggregation):
  ```
  station_id : str      # stable unique id
  name       : str
  lat        : float
  lon        : float
  h3_res7    : str       # = latlng_to_cell(lat, lon, 7)
  region     : str       # e.g. "IGP", "Peninsula", "Coastal"
  time       : datetime64[ns]   # daily (present in the per-day table)
  pm25 pm10 no2 so2 co o3 : float   # CPCB units (CO mg/m3, rest ug/m3); NaN if missing
  ```
- A static station registry (no `time`/pollutant columns) may also be written.
- **Files:** `data/processed/stations.parquet` (daily long table) and/or
  `data/processed/stations.geoparquet` (geometry = points, EPSG:4326).
- **Hourly->daily rule:** mean of valid hours, require >= 75% completeness, then
  CO/O3 reported as 8-hour-max, others as 24-hour mean (matches NAQI averaging).

### 6.3 Fires — GeoDataFrame / parquet (FIRMS surrogate)
- **Columns:**
  ```
  date    : datetime64[ns]    # acquisition date (daily)
  lat     : float
  lon     : float
  frp     : float             # fire radiative power (MW)
  sensor  : str               # "VIIRS_SNPP" | "VIIRS_NOAA20" | "MODIS" | ...
  h3_res7 : str               # = latlng_to_cell(lat, lon, 7)
  ```
- **File:** `data/processed/fires.parquet` (or `.geoparquet`, geometry = points).

### 6.4 File-location contract
```
data/raw/<source>/...                 # per-source synthetic/real raw pulls
data/processed/grid.nc                 # the fused gridded cube (§6.1)
data/processed/stations.parquet        # station daily labels (§6.2)
data/processed/fires.parquet           # fire detections (§6.3)
data/processed/aqi_grid.nc             # output of naqi.compute_aqi_grid(predicted)
data/processed/hotspots.geoparquet     # confirmed HCHO hotspot polygons (Obj-2)
data/models/<model_name>/...           # trained model artifacts
data/artifacts/...                     # COGs, PMTiles, validation tables
reports/figures/...                    # report/PPT figures
```

### 6.5 Function names other agents must implement (chaining contract)

So the Demo agent can chain the full pipeline, the following entry points must
exist with these names and accept the composed Hydra `cfg`:

```python
# sim (Demo agent)
aqi_india.sim.generate.make_grid(cfg) -> xr.Dataset            # §6.1, also writes grid.nc
aqi_india.sim.generate.make_stations(cfg) -> gpd.GeoDataFrame  # §6.2
aqi_india.sim.generate.make_fires(cfg) -> gpd.GeoDataFrame     # §6.3

# fusion
aqi_india.fusion.harmonize.run(cfg) -> xr.Dataset              # gap-filled cube
# features
aqi_india.features.feature_matrix.build(cfg) -> "pd.DataFrame | xr.Dataset"
# models
aqi_india.models.train.run(cfg) -> "trained model artifact path"
aqi_india.models.predict.run(cfg) -> xr.Dataset               # predicted surface concentrations
                                                              # data_vars use POLLUTANTS keys
# AQI maps (already available — call the engine, don't reimplement)
aqi_india.aqi.naqi.compute_aqi_grid(predicted_ds) -> xr.Dataset
# hotspots
aqi_india.hotspot.consensus.run(cfg) -> gpd.GeoDataFrame       # hotspots.geoparquet
# transport
aqi_india.transport.fire_hcho_corr.run(cfg) -> dict            # correlation + lag results
# validation
aqi_india.validation.cv.run(cfg) -> dict                       # CV-ladder metrics tables
```

`models.predict.run` **must** output an xarray Dataset whose pollutant data-vars
are named with the canonical `POLLUTANTS` keys (`pm25`, `pm10`, `no2`, `so2`,
`co`, `o3`) so it can be fed directly into `compute_aqi_grid`.

---

## 7. Build & test gate

```bash
make setup        # python -m venv .venv && pip install -e .   (light deps only)
make test-core    # pytest tests/test_naqi.py tests/test_h3.py tests/test_metrics.py
make test         # full suite
make lint         # ruff check src tests
make demo         # aqi demo run
```
The Foundation phase ends green: **46 core tests pass**, `aqi --help` / `aqi info`
work, and the package imports under the light set. Keep this gate green.
