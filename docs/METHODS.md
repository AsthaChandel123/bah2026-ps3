# Methods & Models Catalog — BAH 2026 PS3

This catalog consolidates **118 methods and models** (10 research topics) evaluated for Bharatiya Antariksh Hackathon 2026, Problem Statement 3 (satellite-derived surface AQI maps + HCHO biomass-burning hotspots over India). It spans **30+ distinct techniques** across CPCB AQI computation, AOD→PM2.5 physics, deep learning, classical/geostatistical ML, HCHO hotspot detection, fire–HCHO correlation & transport attribution, multi-source data fusion / gap-filling, validation & uncertainty quantification, fast-platform O(1)/O(log n) compute, and visualization. Each entry records inputs, outputs, libraries, complexity, trade-offs, benchmarks, and a concrete recommendation so the team can pick a defensible stack per pipeline stage.

**At a glance:** Acquisition / Preprocess (8) · Fusion & Gap-fill (11) · AQI Computation (13) · ML Prediction (25) · Hotspot Detection (12) · Fire & Transport (9) · Validation (14) · Fast-Platform / O(1) (12) · Visualization (14).

---

## Method Index by Pipeline Stage

Every method below is grouped under the pipeline stage it serves (derived from its research topic and category). The **Recommended?** column flags the project's preferred choices (`Yes` = adopt, `Maybe` = situational, `No` = baseline/avoid).

### Acquisition / Preprocess  (8)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [Physical eta-scaling: PM2.5 = AOD x eta (column-to-surface conversion)](#physical-eta-scaling-pm25--aod-x-eta-column-to-surface-conversion) | Physical / first-principles | Core relation PM2.5 = eta * AOD. Expanded eta = 1/(MEE * H * f(RH)) * profile, where MEE = mass extinction efficiency (~3-7 m2/g), H = aerosol scale height proxied by PBL/boundary-layer height (BLH), f(RH) = hygroscopic growth (extinctio… | No |
| [GEOS-Chem model-ratio scaling (van Donkelaar geophysical method)](#geos-chem-model-ratio-scaling-van-donkelaar-geophysical-method) | Physical / CTM model-ratio | Compute simulated ratio eta_sim = surface_PM2.5/AOD from a chemical transport model (GEOS-Chem, or MERRA-2/CAMS for India) per grid/time, then PM2.5_sat = AOD_obs * eta_sim. The CTM supplies vertical profile, f(RH), composition and MEE s… | Yes |
| [Geographically Weighted Regression fusion (van Donkelaar V5/V6 calibration)](#geographically-weighted-regression-fusion-van-donkelaar-v5v6-calibration) | Statistical fusion / hybrid | Calibrate geophysical (CTM-ratio) PM2.5 to ground monitors via GWR: local spatially-varying regression of ground PM2.5 on geophysical PM2.5 plus predictors (elevation, land use, NDVI, urban fraction). Corrects heterogeneous bias. This is… | Yes |
| [ChinaHighPM2.5 / GHAP Space-Time Extra-Trees (STET) - Wei et al.](#chinahighpm25--ghap-space-time-extra-trees-stet---wei-et-al) | ML ensemble (tree-based) | Space-Time Extremely Randomized Trees: extra-trees ensemble with explicit spatiotemporal terms (lat, lon, day-of-year, autocorrelation features) fusing AOD, reanalysis meteo, emissions, land use, population. Produces seamless gap-free 1k… | Yes |
| [Empirical two-stage / mixed-effects linear regression](#empirical-two-stage--mixed-effects-linear-regression) | Statistical empirical | Stage 1: calibrate AOD-PM2.5 with day-specific (mixed-effects) or site-specific slopes/intercepts; Stage 2: predict on AOD-missing days via meteorology + spatial smoothing. Includes plain MLR PM2.5 = a + b*AOD + c*BLH + d*RH + e*WS. Clas… | No |
| [Random Forest / Gradient Boosting (tabular ML) - India IGP proven](#random-forest--gradient-boosting-tabular-ml---india-igp-proven) | ML ensemble (tree-based) | RF/XGBoost/LightGBM regressing CPCB PM2.5 on satellite AOD + meteo + land-use + spatiotemporal features. Demonstrated over IGP (16 cities) and Maharashtra at 1km. The realistic high-scoring workhorse baseline for PS3 Objective 1. | Yes |
| [CNN / LSTM / CNN-LSTM deep hybrids](#cnn--lstm--cnn-lstm-deep-hybrids) | ML deep learning (PS3-specified) | CNN captures spatial context (AOD/meteo grids, neighborhood emissions); LSTM/temporal captures pollutant memory and transport lag; CNN-LSTM/ConvLSTM fuses both for spatiotemporal daily PM2.5. Optionally physics-guided (inject AOD/BLH, f(… | Yes |
| [AOD gap-filling preprocessing (cloud/no-retrieval recovery)](#aod-gap-filling-preprocessing-cloudno-retrieval-recovery) | Preprocessing / data fusion | Recover missing AOD (clouds, glint, bright surface, night) before retrieval: fuse multi-sensor AOD (MAIAC Terra+Aqua, VIIRS, INSAT-3D geostationary) and fill with reanalysis (MERRA-2/CAMS) via RF/mean-filter/IDW imputation. Critical beca… | Yes |

### Fusion & Gap-fill  (11)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [DINEOF (Data-Interpolating Empirical Orthogonal Functions)](#dineof-data-interpolating-empirical-orthogonal-functions) | Gap-filling (cloud/orbit) — unsupervised EOF reconstruction | Reconstructs missing pixels in a 3D space-time cube via iterative truncated EOF/SVD: gaps set to mean, decompose, retain optimal modes by cross-validation, refill, iterate to convergence. Label-free, no covariates. Captures dominant spat… | Yes |
| [Spatiotemporal kriging / Gaussian-process interpolation](#spatiotemporal-kriging--gaussian-process-interpolation) | Gap-filling + geostatistical fusion | Models AOD/column or AQI residual as a space-time random field with fitted spatiotemporal covariance/variogram; BLUP prediction fills gaps and interpolates between CPCB stations with kriging variance (uncertainty). Regression-kriging int… | Yes |
| [Random Forest / Gradient Boosting (XGBoost, LightGBM)](#random-forest--gradient-boosting-xgboost-lightgbm) | ML gap-fill + nonlinear fusion/regression | Tree ensembles impute missing satellite AOD/columns and learn nonlinear column->surface relationships from co-located CPCB labels using met + land-use + reanalysis features. In MAIAC benchmarks XGBoost/RF beat kriging/IDW/GAM. Central-In… | Yes |
| [ConvLSTM / U-Net partial-convolution inpainting](#convlstm--u-net-partial-convolution-inpainting) | Deep-learning spatiotemporal gap-filling / inpainting | CNN image-inpainting adapted to satellite cubes: mask-aware partial convolutions in U-Net or 3D nets reconstruct irregular cloud/orbit gaps; ConvLSTM exploits temporal sequence. Applied to global Sentinel-5P CO with errors comparable to… | Yes |
| [Gap-filled MAIAC AOD + MERRA-2 fusion pipeline](#gap-filled-maiac-aod--merra-2-fusion-pipeline) | Applied multi-source AOD gap-fill (proven India recipe) | India-validated recipe: take 1km MODIS MAIAC AOD (MCD19A2), fill cloud gaps with gap-free MERRA-2 AOD and/or ML imputation to get full-coverage daily 1km AOD, then combine with MERRA-2 met + land use in LME/RF/CNN-LightGBM to predict sur… | Yes |
| [Geographically Weighted Regression (GWR) — van Donkelaar / WUSTL ACAG](#geographically-weighted-regression-gwr--van-donkelaar--wustl-acag) | Multi-sensor fusion + ground calibration | WUSTL ACAG global PM2.5: combine multi-instrument AOD (MODIS Terra/Aqua, MISR, SeaWiFS, VIIRS) with GEOS-Chem CTM to get geophysical PM2.5, then calibrate to monitors with GWR whose coefficients vary in space, capturing regional AOD-PM r… | Yes |
| [Bayesian Maximum Entropy (BME)](#bayesian-maximum-entropy-bme) | Geostatistical fusion of hard + soft (uncertain) data | BME rigorously fuses 'hard' CPCB ground data with 'soft' uncertain data (satellite/ML estimates, met) under a max-entropy framework, propagating each source's uncertainty. China national LUR+BME PM2.5 reached R2~0.82, RMSE~4.6; adding so… | Maybe |
| [Optimal Interpolation / Data Assimilation (CAMS 4D-Var)](#optimal-interpolation--data-assimilation-cams-4d-var) | Physics-based assimilation prior | CAMS/ECMWF assimilates satellite columns (S5P TROPOMI NO2/CO/O3, AOD) into the IFS chemistry model via incremental 4D-Var (12h windows, T95/T159), producing gap-free, physically consistent 3D composition fields (EAC4, NRT/forecast). For… | Yes |
| [Bias correction: Quantile Mapping / CDF matching](#bias-correction-quantile-mapping--cdf-matching) | Satellite-vs-ground bias correction (distributional) | Aligns CDF/quantiles of satellite-derived or reanalysis values to co-located CPCB observations, correcting systematic bias, distribution shape and extremes. Variants: empirical QM, parametric, RQUANT, detrended QM. Standard in precip/AQ;… | Yes |
| [ML residual bias correction](#ml-residual-bias-correction) | Satellite-vs-ground bias correction (nonlinear, covariate-aware) | Train ML (RF/GBM/NN) to predict residual (satellite/model minus ground) as a function of met, land use, BLH, time; add back to correct. Captures state-dependent, nonlinear, spatially varying bias QM misses (RH/BLH-driven AOD-PM bias). Us… | Maybe |
| [Deep super-resolution / statistical downscaling (TROPOMI/INSAT -> 1km)](#deep-super-resolution--statistical-downscaling-tropomiinsat---1km) | Spatial downscaling / super-resolution | Bring coarse columns to fine grid: (a) statistical downscaling — regress fine concentration on high-res covariates (land use, roads, NDVI, elevation, population, met) a la LUR/geographic-ML; (b) deep SR — CNN/SRGAN/diffusion or land-cove… | Yes |

### AQI Computation  (13)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [Six AQI categories + colors](#six-aqi-categories--colors) | AQI scale | CPCB 6-band scale 0-500: Good 0-50 (Green #009865/dark green), Satisfactory 51-100 (Light Green/Yellow-green #84CF33), Moderate(ly Polluted) 101-200 (Yellow #FFFF00), Poor 201-300 (Orange #FF9900/Orange), Very Poor 301-400 (Red #FF0000),… | Yes |
| [PM2.5 sub-index (24h)](#pm25-sub-index-24h) | Pollutant breakpoint table | Averaging: 24h mean, units ug/m3. Conc breakpoints [BP_Lo-BP_Hi] -> AQI [I_Lo-I_Hi]: 0-30->0-50; 31-60->51-100; 61-90->101-200; 91-120->201-300; 121-250->301-400; 250-500(extrapolate, >250)->401-500. NAAQS 24h=60. | Yes |
| [PM10 sub-index (24h)](#pm10-sub-index-24h) | Pollutant breakpoint table | 24h mean, ug/m3. 0-50->0-50; 51-100->51-100; 101-250->101-200; 251-350->201-300; 351-430->301-400; 430-500(>430)->401-500. NAAQS 24h=100. | Yes |
| [NO2 sub-index (24h)](#no2-sub-index-24h) | Pollutant breakpoint table | 24h mean, ug/m3. 0-40->0-50; 41-80->51-100; 81-180->101-200; 181-280->201-300; 281-400->301-400; 400-500(>400)->401-500. NAAQS 24h=80. | Maybe |
| [SO2 sub-index (24h)](#so2-sub-index-24h) | Pollutant breakpoint table | 24h mean, ug/m3. 0-40->0-50; 41-80->51-100; 81-380->101-200; 381-800->201-300; 801-1600->301-400; 1600-2000(>1600)->401-500. NAAQS 24h=80. | Maybe |
| [CO sub-index (8h, mg/m3)](#co-sub-index-8h-mgm3) | Pollutant breakpoint table | Averaging: max 8h rolling mean, units mg/m3 (NOT ug/m3). 0-1.0->0-50; 1.1-2.0->51-100; 2.1-10->101-200; 10.1-17->201-300; 17.1-34->301-400; 34-50(>34)->401-500. NAAQS 8h=2 mg/m3. | Yes |
| [O3 sub-index (8h with 1h override)](#o3-sub-index-8h-with-1h-override) | Pollutant breakpoint table | ug/m3. CPCB rule: use max 8h up to AQI 300, then max 1h for higher bands. 0-50->0-50 (8h); 51-100->51-100 (8h); 101-168->101-200 (8h); 169-208->201-300 (8h); 209-748->301-400 (1h); 748-1000(>748)->401-500 (1h). NAAQS 8h=100,1h=180. | Yes |
| [NH3 sub-index (24h)](#nh3-sub-index-24h) | Pollutant breakpoint table | 24h mean, ug/m3. 0-200->0-50; 201-400->51-100; 401-800->101-200; 801-1200->201-300; 1201-1800->301-400; 1800-2400(>1800)->401-500. NAAQS 24h=400. | No |
| [Pb sub-index (24h)](#pb-sub-index-24h) | Pollutant breakpoint table | 24h mean, ug/m3. 0-0.5->0-50; 0.5-1.0->51-100; 1.1-2.0->101-200; 2.1-3.0->201-300; 3.1-3.5->301-400; 3.5-4.0(>3.5)->401-500. NAAQS 24h=1.0. | No |
| [Sub-index linear interpolation formula](#sub-index-linear-interpolation-formula) | Core formula | For pollutant p with concentration Cp in segment [BP_Lo, BP_Hi] mapping to AQI [I_Lo, I_Hi]: Ip = ((I_Hi - I_Lo)/(BP_Hi - BP_Lo)) * (Cp - BP_Lo) + I_Lo. Linear within each segment; piecewise-linear overall. Round to nearest integer. | Yes |
| [Overall AQI = max of sub-indices (>=3 pollutant + PM rule)](#overall-aqi--max-of-sub-indices-3-pollutant--pm-rule) | Aggregation rule | AQI = max over available pollutant sub-indices. Valid ONLY if >=3 pollutants have data AND at least one is PM2.5 or PM10. Otherwise AQI undefined/not reported. The pollutant attaining the max is the 'responsible/prominent pollutant'. | Maybe |
| [O(1) vectorized lookup-table implementation](#o1-vectorized-lookup-table-implementation) | Implementation | Precompute per-pollutant arrays: bp_conc (n_seg+1 edges), I_lo, I_hi, bp_lo, bp_hi. Per grid cell+pollutant: seg=clip(searchsorted(bp_conc,C,'right')-1,0,n_seg-1); Ip=(I_hi[seg]-I_lo[seg])/(bp_hi[seg]-bp_lo[seg])*(C-bp_lo[seg])+I_lo[seg]… | Yes |
| [Comparison: US EPA AQI vs WHO 2021](#comparison-us-epa-aqi-vs-who-2021) | Benchmark / standards comparison | EPA AQI: same piecewise-linear max-of-sub-index method but different categories (USG/Unhealthy/Hazardous), different breakpoints, and ppb/ppm units (CO 8h in ppm, NO2/SO2 1h ppb). WHO 2021 AQGs are health guidelines, not an index: PM2.5… | Maybe |

### ML Prediction  (25)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [CNN (2D spatial regression)](#cnn-2d-spatial-regression) | Spatial / per-day grid estimation | Convolutional layers learn spatial context (neighboring AOD/column gradients, land-use, emission patterns) to map satellite columns -> surface concentration on a single time slice. Image-to-pixel/image-to-image regression. Acts as the sp… | No |
| [LSTM / GRU (temporal sequence)](#lstm--gru-temporal-sequence) | Temporal / per-station time series | Recurrent nets model time evolution of column + met -> surface concentration at a fixed location/station. Captures lagged effects (nighttime boundary-layer collapse, multi-day accumulation episodes in IGP winter). GRU is a lighter LSTM v… | Yes |
| [Hybrid CNN-LSTM](#hybrid-cnn-lstm) | Spatiotemporal (decoupled) | CNN extracts spatial features per timestep, pooled features feed an LSTM modeling temporal evolution. The PS-recommended workhorse; consistently outperforms standalone CNN or LSTM. Decoupled (not fully spatiotemporal-convolutional) so ch… | Yes |
| [ConvLSTM / SA-ConvLSTM (self-attention ConvLSTM)](#convlstm--sa-convlstm-self-attention-convlstm) | Spatiotemporal (coupled, full-grid) | Replaces LSTM matmuls with convolutions so hidden/cell states stay 2D maps -> joint spatial+temporal learning and spatially coherent full-grid output. SA-ConvLSTM adds self-attention to fix ConvLSTM's weak long-range temporal memory. Bes… | Yes |
| [U-Net / Encoder-Decoder (gap-filling, super-resolution, image-to-image)](#u-net--encoder-decoder-gap-filling-super-resolution-image-to-image) | Spatial gap-filling / downscaling | Encoder-decoder with skip connections. Two critical roles: (a) impute cloud/swath gaps in TROPOMI/INSAT columns before the estimator; (b) super-resolve coarse columns/AOD (3.5-7km TROPOMI, ~10km MERRA-2) to 1km. Two-stage 'impute-then-es… | Yes |
| [Transformers / TFT / Informer / iTransformer (long-range temporal)](#transformers--tft--informer--itransformer-long-range-temporal) | Temporal (long-horizon, attention) | Self-attention for long-sequence forecasting. TFT handles multivariate covariates with variable-selection + interpretable attention, applied to AQI. Informer's ProbSparse attention + distilling solves O(n^2) cost. iTransformer attends ac… | Yes |
| [Attention mechanisms (spatial + temporal + cross-attention)](#attention-mechanisms-spatial--temporal--cross-attention) | Add-on module (all architectures) | Spatial attention weights surrounding pixels/stations; temporal attention weights informative lags; cross/channel attention reweights satellite vs met channels. Lightweight add-ons (SE, CBAM, self-attention) that consistently lift CNN-LS… | Yes |
| [Graph Neural Networks (ST-GNN / GCN-LSTM / graph attention)](#graph-neural-networks-st-gnn--gcn-lstm--graph-attention) | Spatiotemporal over irregular station network | Models CPCB stations as graph nodes with edges from distance and wind field (directed dynamic graphs). GCN/GAT propagate along transport pathways; combined with LSTM/temporal conv for time. Excels at sparse irregular networks and at esti… | Maybe |
| [Physics-Informed NN (AirPhyNet / advection-diffusion-constrained)](#physics-informed-nn-airphynet--advection-diffusion-constrained) | Hybrid physics + DL | Embeds advection-diffusion PDE (wind transport + diffusion) as a soft loss or Neural-ODE layer. AirPhyNet = RNN encoder + GNN differential-equation network + decoder. Improves generalization, data efficiency, physical plausibility; reduc… | Yes |
| [Residual / Ensemble / Stacked deep models (ResNet, stacking, MoE)](#residual--ensemble--stacked-deep-models-resnet-stacking-moe) | Meta / robustness | Residual connections stabilize deep estimators; ensembles/stacking blend complementary models (CNN-LSTM + GNN + trees) for variance reduction and uncertainty. In India, stacking ensembles of RF/ExtraTrees/LightGBM are strong, hard-to-bea… | Maybe |
| [HCHO hotspot detection (Getis-Ord Gi*, LISA, DBSCAN + fire-HCHO correlation + back-trajectory transport)](#hcho-hotspot-detection-getis-ord-gi-lisa-dbscan--fire-hcho-correlation--back-trajectory-transport) | Objective-2: statistical/clustering + transport | For HCHO biomass-burning hotspots: Getis-Ord Gi* and Local Moran's I/LISA flag significant HCHO clusters; DBSCAN/ST-DBSCAN clusters fire pixels; pixelwise/lagged Pearson correlation links FIRMS fires to TROPOMI HCHO enhancement; HYSPLIT… | Maybe |
| [Random Forest (RF)](#random-forest-rf) | Tree ensemble (bagging) | Bagged decision trees on AOD + TROPOMI NO2/SO2/CO/O3/HCHO + ERA5/MERRA-2 met + land-use + spatial/temporal coords. Robust, non-parametric, handles non-linear AOD-PM2.5 hygroscopic/PBL effects. OOB gives free CV. Saturates at high PM2.5 (… | Yes |
| [Gradient Boosting (XGBoost / LightGBM / CatBoost)](#gradient-boosting-xgboost--lightgbm--catboost) | Tree ensemble (boosting) | Sequential boosted trees — often the best single classical model for AOD->PM2.5. LightGBM (leaf-wise histogram) scales to national 1km; XGBoost most battle-tested; CatBoost best with categorical (state/land-class) + ordered boosting. Cap… | Yes |
| [Extra Trees (Extremely Randomized Trees)](#extra-trees-extremely-randomized-trees) | Tree ensemble (bagging) | Like RF but random split thresholds and full-sample (no bootstrap) -> lower variance, faster, slightly higher bias. Near-ties RF; valuable as a decorrelated diversity member in a stack. | No |
| [Support Vector Regression (SVR)](#support-vector-regression-svr) | Kernel regression | RBF-kernel epsilon-insensitive regression. Models non-linear AOD-PM2.5 but scales O(n^2-n^3), sensitive to scaling & C/gamma/epsilon. Generally beaten by tree ensembles on tabular RS data; mostly historical baseline for India. | No |
| [Gaussian Process Regression (GPR / Kriging-equivalent)](#gaussian-process-regression-gpr--kriging-equivalent) | Bayesian kernel / geostatistical | Non-parametric Bayesian regression giving predictive mean AND calibrated uncertainty — kriging with a learned covariance. Ideal ENSEMBLE STACKER with anisotropic spatial smoothing (exactly the Indian national 1km product). Naive O(n^3);… | Yes |
| [Ordinary / Universal Kriging](#ordinary--universal-kriging) | Geostatistics | Interpolates station residuals via fitted variogram (OK constant mean; UK trend on covariates). Best for filling spatial gaps between sparse CPCB stations and cloud-gap residual interpolation, not primary satellite-driven prediction. | Maybe |
| [Regression Kriging (RK) / Co-Kriging](#regression-kriging-rk--co-kriging) | Geostatistics (hybrid) | Two-stage: regress PM2.5 on covariates (GLM/RF/GBM), then krige residuals for leftover spatial autocorrelation. Co-Kriging exploits densely-sampled AOD to improve sparsely-sampled PM2.5. Most practical geostatistical hybrid for India. | Yes |
| [Inverse Distance Weighting (IDW)](#inverse-distance-weighting-idw) | Deterministic interpolation | Distance-weighted average of nearby stations. Simple, fast, no fit, but no uncertainty, sensitive to power p, poor where CPCB network sparse/uneven (most of rural India). Baseline/gap-fill only. | No |
| [Geographically Weighted Regression (GWR)](#geographically-weighted-regression-gwr) | Spatial regression | Local regression with spatially-varying coefficients (distance-kernel weighted) — captures spatially non-stationary AOD-PM2.5 relations across India's regimes (IGP vs coastal vs arid). Foundational satellite-PM2.5 method; adding NO2/EVI… | Maybe |
| [Geographically & Temporally Weighted Regression (GTWR)](#geographically--temporally-weighted-regression-gtwr) | Spatiotemporal regression | Extends GWR with a spatiotemporal kernel so coefficients vary in space AND time — directly suited to DAILY surface-AQI mapping. Beats SLR and GWR; neural (GTWNN) and spatiotemporally-weighted-tree variants push further. Strong interpreta… | Maybe |
| [Land Use Regression (LUR)](#land-use-regression-lur) | Empirical regression | Linear regression of pollutant on buffer-aggregated land-use/traffic/emission/satellite predictors. Excellent for LONG-TERM annual NO2/PM exposure surfaces (national India satellite-LUR NO2 exists). Weak for daily dynamics; pair with uni… | Maybe |
| [Generalized Additive Models (GAM)](#generalized-additive-models-gam) | Semi-parametric regression | Sum of smooth spline functions s(AOD)+s(RH)+s(PBLH)+spatial tensor smooth — captures non-linearity while interpretable, giving response curves. Middle ground between linear and black-box trees; supports spatial/temporal smooths. | Maybe |
| [Linear Mixed-Effects Model (LME — Hu/Liu/Lee daily calibration)](#linear-mixed-effects-model-lme--huliulee-daily-calibration) | Hierarchical linear model | Classic two-stage satellite-PM2.5 method: PM2.5 ~ AOD + met with DAY-SPECIFIC random intercepts and slopes, recalibrating AOD-PM2.5 every day. Simple, robust, fast; established benchmark over Indian subcontinent (Mhawish/Dey). Often coup… | Yes |
| [Bayesian Hierarchical Model](#bayesian-hierarchical-model) | Bayesian spatial/spatiotemporal | Full probabilistic hierarchy with spatial (CAR/GP) + temporal random effects + measurement-error layers — propagates uncertainty end-to-end, ideal for fusing multi-source satellite columns of differing accuracy and reporting AQI confiden… | Maybe |

### Hotspot Detection  (12)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [Getis-Ord Gi* (local hot/cold-spot z-score)](#getis-ord-gi-local-hotcold-spot-z-score) | Spatial autocorrelation / canonical hotspot statistic | Per-cell z-score of local weighted HCHO sum vs global mean. High +z + low p = significant hotspot (high-among-high); negative z = coldspot. star=True includes focal cell (Gi*, standard). Permutation p_sim avoids normality. The most defen… | Yes |
| [Local Moran's I / LISA (cluster & outlier)](#local-morans-i--lisa-cluster--outlier) | Spatial autocorrelation / cluster-outlier | Decomposes global Moran's I per cell, classifying significant cells as HH (hotspot core), LL (coldspot), HL (high outlier = isolated fire plume), LH (low in high). Complements Gi* by flagging spatial OUTLIERS that Gi* smooths over. | Yes |
| [Emerging Hot Spot Analysis (EHSA: Gi* + Mann-Kendall)](#emerging-hot-spot-analysis-ehsa-gi--mann-kendall) | Spatio-temporal trend + hotspot | Space-time cube (cell x time-bin) -> Gi* per cell per bin -> Mann-Kendall on each cell's Gi* series -> labels New/Consecutive/Intensifying/Persistent/Diminishing/Sporadic/Oscillating/Historical hot/cold spot. Answers 'where are HCHO hots… | Yes |
| [Mann-Kendall trend + Sen's slope](#mann-kendall-trend--sens-slope) | Temporal trend (non-parametric) | Per-cell non-parametric monotonic trend test (no normality, robust to outliers/gaps) giving direction & significance; Sen's slope gives robust magnitude (HCHO change per season). Underpins EHSA and stand-alone trend maps. | Yes |
| [Percentile / threshold (>90th/95th of climatology)](#percentile--threshold-90th95th-of-climatology) | Threshold / climatology | Flag cells/days where HCHO exceeds a high percentile (90/95/98th) of the local long-term climatology (per-cell/per-season). Simple transparent first-pass hotspot/exceedance mask; basis for anomaly consensus. | Yes |
| [Standardized anomalies / z-scores](#standardized-anomalies--z-scores) | Threshold / climatology | Per-cell (HCHO - clim mean)/std (per season) -> standardized anomaly field. Removes India's spatial gradient & seasonal cycle so one \|z\| threshold is comparable everywhere; ideal input to Gi*/LISA and the consensus vote. | Yes |
| [DBSCAN / HDBSCAN density clustering](#dbscan--hdbscan-density-clustering) | Density clustering (unsupervised) | Cluster high-HCHO cells in (lat,lon[,value]) space into arbitrary-shaped dense hotspot clusters; auto-labels sparse noise. HDBSCAN removes DBSCAN's single-eps sensitivity & handles variable density (compact urban vs diffuse fire plumes). | Yes |
| [ST-DBSCAN (spatiotemporal DBSCAN)](#st-dbscan-spatiotemporal-dbscan) | Spatiotemporal density clustering | DBSCAN with separate spatial (eps1) and temporal (eps2) radii (+ value threshold), clustering high-HCHO exceedances dense in space AND time -> coherent spatio-temporal hotspot episodes (multi-day burning plume tracked across cells). | Yes |
| [K-means / Gaussian Mixture (GMM)](#k-means--gaussian-mixture-gmm) | Partitional / model-based clustering | Partition cells by feature vector (HCHO level, anomaly, trend slope, optional NO2/fire) into K regimes; high-HCHO cluster(s) = hotspot zone. GMM gives soft probabilistic membership & elliptical clusters; good for IGP vs forest-fire regimes. | Yes |
| [Kernel Density Estimation (KDE)](#kernel-density-estimation-kde) | Density surface | Smooth continuous density/intensity surface from high-HCHO points (or value-weighted), revealing hotspot cores as density peaks; standard companion to Gi* in AQ studies for a visually smooth map. | No |
| [EOF / PCA decomposition](#eof--pca-decomposition) | Dimensionality reduction / mode extraction | Decompose HCHO space-time field into orthogonal spatial patterns (EOFs) + temporal amplitudes (PCs); leading EOFs isolate dominant modes (seasonal burning, IGP industrial), high-loading regions of relevant EOFs flag coherent hotspot zones. | Maybe |
| [Spatial scan statistic (Kulldorff / SaTScan)](#spatial-scan-statistic-kulldorff--satscan) | Cluster significance (scan statistic) | Scans circular/elliptical (and space-time cylindrical) windows of varying size, maximizing likelihood ratio of elevated HCHO vs outside with Monte-Carlo p -> significant, geographically explicit most-likely hotspot cluster(s), spatial &… | Yes |

### Fire & Transport  (9)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [Biomass-burning period extraction (thresholds + seasonal decomposition)](#biomass-burning-period-extraction-thresholds--seasonal-decomposition) | Fire time-series preprocessing | Build daily fire-count/FRP series over Punjab-Haryana + forest AOIs from FIRMS (MODIS MCD14ML, VIIRS VNP14IMGML 375m). Define episodes via percentile thresholds (>90th pct), STL/seasonal decomposition (statsmodels STL, prophet) isolating… | Yes |
| [Lagged cross-correlation & Granger causality (fire -> HCHO)](#lagged-cross-correlation--granger-causality-fire---hcho) | Statistical correlation/causality | Align AOI-mean TROPOMI HCHO column (S5P OFFL L3_HCHO, mol/m2) with fire-count/FRP. Cross-correlation over lags 0-7 d finds peak lag (primary + secondary HCHO from NMVOC oxidation lags fire ~0-2 d). Granger test (statsmodels) on stationar… | Maybe |
| [Spatial co-location & emission-ratio analysis (HCHO/NO2, dHCHO vs FRP)](#spatial-co-location--emission-ratio-analysis-hchono2-dhcho-vs-frp) | Spatial / emission-ratio | Co-grid HCHO, NO2 (S5P OFFL L3_NO2) and fire pixels to 0.05-0.1deg. Compute background-subtracted dHCHO and regress vs FRP -> emission slope (mol per MW). Map HCHO/NO2 (FNR) as VOC/NOx regime + burning fingerprint. Bivariate Moran/Local… | Yes |
| [HYSPLIT back/forward trajectory analysis](#hysplit-backforward-trajectory-analysis) | Lagrangian transport | Ensembles of HYSPLIT trajectories: BACK from IGP receptors (Delhi, Lucknow, Kanpur, Patna) to test Punjab/Haryana origin; FORWARD from fire clusters to map downwind HCHO exposure. Drive with GDAS/GFS or (best) ERA5; release 500/1000/1500… | Yes |
| [Trajectory clustering (source-region attribution)](#trajectory-clustering-source-region-attribution) | Lagrangian transport / clustering | Cluster the trajectory ensemble (k-means/hierarchical on great-circle angle-distance or resampled endpoints) into representative pathways. Assign each receptor-day to a cluster; composite TROPOMI HCHO and fire counts by cluster to identi… | Maybe |
| [Concentration-Weighted Trajectory (CWT) & PSCF](#concentration-weighted-trajectory-cwt--pscf) | Receptor source apportionment | Grid the domain; weight cells by trajectory residence time and receptor HCHO. PSCF = fraction of endpoints in a cell tied to HCHO above threshold (75th pct) -> source probability. CWT = residence-weighted mean HCHO -> source-strength map… | Yes |
| [Wind-sector & bivariate polar plots (openair) with ERA5 winds](#wind-sector--bivariate-polar-plots-openair-with-era5-winds) | Meteorological conditional analysis | Pair receptor HCHO with ERA5 10m winds. openair polarPlot (HCHO vs wind speed+direction), pollutionRose, percentileRose reveal the sector and wind-speed regime delivering high HCHO -> low-level confirmation of NW transport from Punjab/Ha… | Yes |
| [FLEXPART Lagrangian dispersion](#flexpart-lagrangian-dispersion) | Lagrangian dispersion (concentration) | Run FLEXPART (or FLEXPART-WRF) backward from IGP receptors for source-receptor sensitivity footprints (s m3 kg-1), then convolve with GFED/FINN HCHO+NMVOC emission flux to predict the HCHO contribution from Punjab/Haryana fires - true co… | Maybe |
| [Emission-inventory cross-check (GFED / FINN VOC)](#emission-inventory-cross-check-gfed--finn-voc) | Emission inventory validation | Cross-validate attribution vs bottom-up inventories: FINNv2.5 (daily, VIIRS375m+MODIS, 0.1deg, NMVOC speciated incl. explicit HCHO) and GFED4.1s/GFED5 (0.25deg burned-area, VOC factors). Compare inventory HCHO/NMVOC flux over Punjab/Hary… | Yes |

### Validation  (14)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [Core point metrics: RMSE, R/R2, MAE, MBE](#core-point-metrics-rmse-rr2-mae-mbe) | Accuracy metric (Obj-1 primary) | RMSE penalizes large errors (pollutant units); MAE robust to outliers; R Pearson corr, R2 explained variance; MBE = mean(pred-obs) signed bias (+ = overprediction). Report all four (PS scores RMSE/R/MAE) plus MBE for bias. RMSE/MAE ratio… | Yes |
| [MBE / NMB / NME (bias & normalized error)](#mbe--nmb--nme-bias--normalized-error) | Bias & normalized error metric | NMB = sum(pred-obs)/sum(obs) %; NME = sum\|pred-obs\|/sum(obs) %. Standard in regulatory AQ evaluation. Published PM2.5 benchmarks (Huang 2021 ACP): NMB within +/-10-20%, NME 35-45% = good model. Cite these to claim regulatory-grade perf… | Yes |
| [IOA (Willmott Index of Agreement) + slope/intercept](#ioa-willmott-index-of-agreement--slopeintercept) | Agreement & regression-fit metric | Refined IOA (Willmott 2012) range -1 to 1; goal>=0.80, criteria>=0.70 (Emery 2017), independent of linear-form assumption. Fit reduced-major-axis (orthogonal/Deming) regression pred~obs: report SLOPE (ideal 1.0; <1 = regression-to-mean c… | Maybe |
| [Random K-fold CV (BASELINE ONLY - leakage-prone)](#random-k-fold-cv-baseline-only---leakage-prone) | Cross-validation (naive baseline) | Random k-fold/LOOCV shuffles all space-time samples. For AQ this LEAKS via spatial+temporal autocorrelation: held-out samples have near-identical neighbors in training -> inflated R/low RMSE not reflecting prediction at new locations/tim… | No |
| [Leave-Location-Out / Spatial-block CV (PRIMARY)](#leave-location-out--spatial-block-cv-primary) | Cross-validation (spatial, recommended) | Withhold ENTIRE stations/spatial blocks; predict into spatially independent regions. Variants: Leave-One-Station-Out (full time series); Spatial-LOO with a BUFFER (dead-zone) removing neighbors within autocorrelation range; spatial-block… | Yes |
| [Leave-Time-Out / forward-chaining temporal CV](#leave-time-out--forward-chaining-temporal-cv) | Cross-validation (temporal, recommended) | Hold out whole periods: leave-one-month/season-out or forward-chaining (train past, test future; no future leakage). Essential since CNN-LSTM uses temporal sequences and the PS targets daily maps incl. unseen days and biomass-burning sea… | Maybe |
| [Spatiotemporal-blocked CV (GOLD STANDARD)](#spatiotemporal-blocked-cv-gold-standard) | Cross-validation (spatiotemporal, gold standard) | Hold out blocks in BOTH space and time simultaneously (spatial block over a month) so test shares neither nearby station NOR adjacent day. Hardest, most honest; this is the daily India-grid deployment reality. Report as a ladder: random… | Yes |
| [Independent hold-out station set (blind test)](#independent-hold-out-station-set-blind-test) | Validation design (independent test) | Reserve a geographically representative ~15-20% of CPCB stations NEVER used in training/CV/tuning, stratified across IGP/coastal/urban/rural and AQI regimes. Final blind report = most credible single number. Distinct from CV folds (model… | Maybe |
| [AERONET validation of satellite AOD](#aeronet-validation-of-satellite-aod) | Reference-data validation (input QA) | Validate INSAT-3D/S5P AOD vs AERONET L2.0 (Indian sites: Kanpur, Gandhi College/IGP, Jaipur, Pune, Gual Pahari). Collocate AERONET +/-30min of overpass, satellite mean over 25-50km. Report R, RMSE, % within Expected-Error EE = +/-(0.05+0… | Yes |
| [CPCB validation of predicted concentrations/AQI](#cpcb-validation-of-predicted-concentrationsaqi) | Reference-data validation (Obj-1 ground truth) | Ground truth = CPCB CAAQMS hourly (~400-540 stations) via CCR/data.gov.in. QA: dedupe, despike, drop calibration flags, daily means/24h AQI via CPCB sub-index breakpoints. Collocate satellite-pixel to station; compute full metric suite u… | Yes |
| [Taylor diagram (multi-model comparison)](#taylor-diagram-multi-model-comparison) | Visualization for model comparison | Polar plot encoding R (angle), normalized std (radius), centered RMSE (distance to reference) for all candidates (CNN, LSTM, CNN-LSTM, RF baseline). Lets judges rank architectures at a glance; model nearest reference wins. Standard in at… | Yes |
| [Target diagram (bias + variability)](#target-diagram-bias--variability) | Visualization for model comparison | Cartesian plot of normalized bias (y) vs signed normalized unbiased-RMSD (x); distance from origin = total normalized RMSE, unit circle = skill threshold. Complements Taylor by showing SIGN of bias Taylor omits. Together they fully chara… | Maybe |
| [Quantile regression / conformal-calibrated intervals](#quantile-regression--conformal-calibrated-intervals) | Uncertainty quantification (recommended) | Quantile heads (LightGBM objective=quantile, GBM quantile loss, or quantile CNN-LSTM) yield heteroscedastic PIs (wider where sparse/cloudy). Wrap with CONFORMAL prediction (split/Mondrian/weighted) for distribution-free finite-sample COV… | Yes |
| [Deep ensembles / MC dropout (deep UQ)](#deep-ensembles--mc-dropout-deep-uq) | Uncertainty quantification (epistemic) | Deep ensemble: N=5-10 CNN-LSTM with different seeds; mean prediction, variance = epistemic uncertainty; best UQ quality and robustness under dataset SHIFT (Lakshminarayanan 2017, beats MC-dropout). MC-dropout (Gal 2016): dropout active a… | Maybe |

### Fast-Platform / O(1)  (12)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [Google Earth Engine (GEE) — server-side lazy planetary-scale compute](#google-earth-engine-gee--server-side-lazy-planetary-scale-compute) | Compute platform (PRIMARY fast platform) | Fastest path: analyst writes lazy server-side expressions; Google's cluster executes globally — effectively O(1) for the analyst (no download, no local cluster). Holds all PS3 inputs as analysis-ready ImageCollections: COPERNICUS/S5P/{OF… | Yes |
| [Uber H3 hexagonal hierarchical index](#uber-h3-hexagonal-hierarchical-index) | Spatial index (O(1) point->cell) | Maps (lat,lng)->hex cell in O(1) via latlng_to_cell(lat,lng,res). Hierarchical: cell_to_parent/children = instant multi-res rollup; grid_disk(cell,k) = O(k^2) neighbor traversal; polygon_to_cells tessellates India/IGP/fire-zone polygons.… | Yes |
| [Google S2 / geohash / quadkey-XYZ tiles](#google-s2--geohash--quadkey-xyz-tiles) | Spatial index (alternatives / tiling keys) | S2 (Hilbert spherical cells) & geohash (base32 prefix) give O(1) point->cell and locality-preserving range keys for DB partition/sort. Quadkey/XYZ (z/x/y) is the web-map standard and the addressing for precomputed COG/tile pyramids. For… | Maybe |
| [R-tree (STRtree) & KD-tree (cKDTree)](#r-tree-strtree--kd-tree-ckdtree) | Spatial index (nearest-neighbor / overlap) | shapely STRtree/rtree give O(log n) bbox-overlap queries (assign stations to admin polygons, clip to India). scipy cKDTree gives O(log n) nearest-station/nearest-pixel queries — critical for IDW interpolation, matching CPCB stations to n… | Maybe |
| [Cloud-Optimized GeoTIFF (COG) + HTTP range requests](#cloud-optimized-geotiff-cog--http-range-requests) | Cloud-native format (raster, partial read) | Internally tiled+overviewed GeoTIFF; clients fetch only needed byte ranges (tiles/zoom) via HTTP range requests — O(1)-ish partial reads instead of full download. Perfect for serving daily India AQI/HCHO maps and on-demand window reads d… | Maybe |
| [Zarr + Kerchunk / VirtualiZarr (chunked N-D arrays)](#zarr--kerchunk--virtualizarr-chunked-n-d-arrays) | Cloud-native format (multidim, lazy out-of-core) | Zarr stores chunked compressed N-D arrays; xarray opens lazily, fetching only requested chunks — ideal for time x lat x lon met/satellite cubes. Kerchunk/VirtualiZarr build a virtual Zarr (JSON/parquet reference) over existing NetCDF/GRI… | Yes |
| [Parquet / GeoParquet (+ Hilbert sort)](#parquet--geoparquet--hilbert-sort) | Cloud-native format (tabular vector, columnar) | Columnar (Geo)Parquet with predicate+projection pushdown and row-group stats enables fast filtered reads over HTTP range requests — read only needed columns/rows. Store CPCB obs, H3-binned pixel tables, fire detections, and ML feature ta… | Yes |
| [STAC catalog + pystac-client + odc-stac](#stac-catalog--pystac-client--odc-stac) | Discovery / indexing layer | STAC standardizes search of imagery by bbox/time/collection. pystac-client queries STAC APIs; odc-stac loads matched Items into an xarray datacube (crop/mosaic/resample/reproject). Use to discover/index Sentinel-5P, MODIS/VIIRS, and your… | Yes |
| [Pangeo stack — xarray + Dask + NumPy vectorization](#pangeo-stack--xarray--dask--numpy-vectorization) | Compute (out-of-core parallel) | xarray (labeled N-D) + Dask (lazy parallel task graphs) + vectorized NumPy: the standard for out-of-core parallel processing of cubes too big for RAM. Use for per-pixel/per-station feature engineering, regridding, compositing, and batche… | Yes |
| [DuckDB + spatial + H3 extensions](#duckdb--spatial--h3-extensions) | Compute (fast tabular geo query engine) | Embedded vectorized OLAP engine querying (Geo)Parquet/CSV in-place (incl. over HTTP range requests), with a spatial extension (ST_*, GeoParquet, ST_Hilbert) and an h3 community extension (h3_latlng_to_cell, h3_cell_to_parent). Ideal fast… | Yes |
| [Precomputed tile pyramids & caching (CDN)](#precomputed-tile-pyramids--caching-cdn) | Serving / latency optimization | For the public daily AQI/HCHO map, precompute XYZ tile pyramids (or cache TiTiler+COG overviews behind a CDN) so every request is an O(1) static-tile/cache hit. PMTiles/MBTiles package the whole pyramid as one range-readable archive serv… | Yes |
| [geemap Python API (GEE bridge)](#geemap-python-api-gee-bridge) | Interface / interactive analysis | geemap wraps earthengine-api with ipyleaflet/folium: visualize ImageCollections, draw AOIs, run reduceRegions, export to COG/GeoPandas. Bridges GEE server-side outputs to the local Pangeo/DuckDB/H3 pipeline (ee_to_geopandas, ee_export_im… | Yes |

### Visualization  (14)

| Method | Category | One-line summary | Recommended? |
| --- | --- | --- | :---: |
| [geemap (GEE-native interactive maps)](#geemap-gee-native-interactive-maps) | Python interactive mapping (jupyter/colab) | ipyleaflet wrapper over Google Earth Engine. Visualizes S5P TROPOMI / INSAT AOD / MODIS ee.Image(Collection) without download. Built-in split-panel maps, linked maps, add_time_slider, ts_inspector for per-pixel HCHO query. Map.addLayer +… | Yes |
| [leafmap](#leafmap) | Python interactive mapping (backend-agnostic) | GEE-decoupled sibling of geemap. add_cog_layer() streams a remote/local COG via TiTiler; split_map(left,right) for before/after; add_time_slider; backends include ipyleaflet, folium, maplibre, plotly, kepler, pydeck. leafmap.maplibregl g… | Yes |
| [folium + branca (TimeSliderChoropleth)](#folium--branca-timesliderchoropleth) | Lightweight static/HTML maps (Leaflet) | Leaflet wrapper producing self-contained HTML — a no-server demo. TimeSliderChoropleth/TimestampedGeoJson animate daily AQI choropleths or fire points over time. ImageOverlay for static AQI PNG+bounds; HeatMap for HCHO density; Dual_map… | Yes |
| [lonboard](#lonboard) | GPU vector rendering in Jupyter (deck.gl+GeoArrow) | Renders millions of points/lines/polygons in-notebook via deck.gl over a binary GeoArrow/GeoParquet pipeline (no GeoJSON text). Benchmark: 3M points in 2.5s where ipyleaflet/pydeck crashed. ScatterplotLayer (FIRMS fires), PathLayer (wind… | Yes |
| [deck.gl + pydeck](#deckgl--pydeck) | WebGL/WebGPU large-data rendering | deck.gl is the JS WebGL2/WebGPU layer engine; pydeck binds it to Python. HeatmapLayer/ScreenGridLayer (HCHO/fire density), HexagonLayer/GridLayer (extruded 3D AQI bins), ArcLayer/LineLayer (pollutant transport source→receptor), TripsLaye… | Yes |
| [kepler.gl (keplergl python)](#keplergl-keplergl-python) | No-code geospatial exploration (deck.gl) | Drag-and-drop deck.gl app; keplergl widget loads GeoDataFrames/CSV. Built-in time-playback animation, point/heatmap/grid/hexbin/arc layers, dual-map split, brushing/filtering. Export config+data to standalone HTML. Fast route to an impre… | Maybe |
| [MapLibre GL JS + PMTiles/COG tiles](#maplibre-gl-js--pmtilescog-tiles) | Production web-GL basemap + raster/vector tiles | Open-source (no token) WebGL engine — the static-export interactive deliverable. Consumes raster XYZ/PMTiles of AQI/HCHO COGs and vector PMTiles (districts/fires). raster-color expression for AQI palette; fill-extrusion 3D; symbol layers… | Yes |
| [TiTiler + rio-tiler over COG](#titiler--rio-tiler-over-cog) | Dynamic raster tile server (cloud-native) | FastAPI app (titiler.core/.application/.mosaic — metapackage dropped late 2025, install submodules) dynamically tiles COGs reading only needed bytes; on-the-fly rescale/colormap/expression. Endpoints: /cog/tiles, /cog/tilejson.json, /cog… | Yes |
| [terracotta](#terracotta) | Lightweight raster tile server (DB-indexed) | Pure-Python XYZ tile server pre-indexing many rasters in SQLite/MySQL by metadata keys (date, pollutant) — fits 'daily AQI, 365 days x 6 pollutants'. On-the-fly colormaps; simpler ops than GeoServer. | Maybe |
| [Streamlit + leafmap/pydeck/plotly](#streamlit--leafmappydeckplotly) | Python dashboard / web app | Fastest Python-to-web-app. st.pydeck_chart (deck.gl), st_folium, leafmap components, st.plotly_chart. st.slider/select_slider drives a date slider re-rendering the daily AQI map + validation panel. streamlit-keplergl for kepler. Multipag… | Yes |
| [Plotly Dash](#plotly-dash) | Python dashboard (production callbacks) | Callback-driven app with precise layout; dash-leaflet (Leaflet incl. TimeSlider, COG via tiles) and dash-deck (deck.gl) for maps; plotly for time series, validation scatter, polar Taylor. Better than Streamlit for tightly-linked interact… | Maybe |
| [HoloViz Panel + GeoViews + Datashader](#holoviz-panel--geoviews--datashader) | Dashboard + server-side big-data rendering | Panel app; GeoViews+hvPlot for xarray-native maps; Datashader rasterizes millions of points/large grids server-side into images (whole-India fire clouds, dense grids, no GPU needed). Native xarray/dask integration matches AQI/HCHO cubes.… | Yes |
| [matplotlib + cartopy + contextily](#matplotlib--cartopy--contextily) | Static cartography for report/PPT | cartopy adds CRS/projection + coastlines/borders to matplotlib (PlateCarree for India + state shapefiles); contextily adds basemap tiles behind GeoDataFrames (Web Mercator). For clean static daily-AQI panels, HCHO hotspot maps, multi-pan… | Yes |
| [Taylor diagram + validation scatter](#taylor-diagram--validation-scatter) | Model validation visualization | Taylor diagram summarizes correlation R, centered RMSD, std-dev ratio of model vs CPCB on one polar plot — directly mirrors PS3 RMSE/R/MAE scoring (SkillMetrics.taylor_diagram or Copin's matplotlib recipe). Pair with 1:1 hexbin density s… | Yes |

---

## Method Catalog by Topic

## CPCB Indian National Air Quality Index (NAQI, 2014) — exact computation, breakpoint constants, and O(1) lookup implementation for surface AQI mapping (BAH 2026 PS3)

*Pipeline stage: **AQI Computation** · 13 methods.*

**Recommended stack:**

- **NumPy vectorized np.searchsorted over per-pollutant breakpoint arrays for O(1) sub-index lookup**
- **Single fused 8-pollutant breakpoint constant table (concentration BP_Lo/BP_Hi -> index I_Lo/I_Hi)**
- **Linear-interpolation sub-index then np.nanmax reduction across pollutant axis with >=3-pollutant + PM mask**
- **xarray/dask for tiled grid (INSAT/TROPOMI) AQI map computation over India**
- **Unit normalization layer: CO in mg/m3, all others ug/m3; O3/CO use max(8h,1h) per CPCB rule**

**Key findings:**

- CPCB NAQI uses 6 non-uniform bands: Good 0-50, Satisfactory 51-100, Moderate 101-200, Poor 201-300, Very Poor 301-400, Severe 401-500 (widths 50,50,100,100,100,100). Colors: Green, Light Green, Yellow, Orange, Red, Maroon.
- Eight pollutants. Averaging: 24h mean for PM2.5, PM10, NO2, SO2, NH3, Pb; max 8h for CO and O3, with O3 switching to max 1h for the upper (>AQI 300) bands and CO carried in mg/m3 (all others ug/m3).
- Exact concentration breakpoints (BP edges) — PM2.5: 0,30,60,90,120,250; PM10: 0,50,100,250,350,430; NO2: 0,40,80,180,280,400; SO2: 0,40,80,380,800,1600; CO(mg/m3): 0,1,2,10,17,34; O3: 0,50,100,168,208,748; NH3: 0,200,400,800,1200,1800; Pb: 0,0.5,1.0,2.0,3.0,3.5. Each maps to AQI edges 0,50,100,200,300,400 (then 500).
- Sub-index formula is piecewise-linear: Ip = ((I_Hi-I_Lo)/(BP_Hi-BP_Lo))*(Cp-BP_Lo)+I_Lo, applied within the segment containing Cp; final AQI = max of available sub-indices.
- Validity rule: AQI computed only if >=3 pollutants have data AND at least one is PM2.5 or PM10; else not reported. The pollutant at the max is the 'prominent/responsible' pollutant — for IGP this is almost always PM2.5/PM10.
- Top (Severe/401-500) segments are open-ended in CPCB for several pollutants; implementations cap them (e.g. PM2.5 250-380->401-500, PM10 430-510, CO 34-50, O3 748-1000) or clamp final AQI to 500.
- O(1) implementation: precompute per-pollutant edge arrays, use np.searchsorted to find the segment (branch-free), apply the linear formula, np.nanmax across the pollutant axis with the >=3+PM mask — fully vectorizable over a daily India grid via numpy/xarray/dask.
- Standards differ: WHO 2021 24h PM2.5 guideline = 15 ug/m3 (half the CPCB Good-band ceiling of 60), and US EPA AQI uses different breakpoints AND units (ppb/ppm, CO 8h in ppm) — so EPA tables and WHO guidelines must NOT be substituted into the CPCB ug/m3/mg/m3 computation that PS3 is scored against.

### Six AQI categories + colors

- **Category:** AQI scale
- **Summary:** CPCB 6-band scale 0-500: Good 0-50 (Green #009865/dark green), Satisfactory 51-100 (Light Green/Yellow-green #84CF33), Moderate(ly Polluted) 101-200 (Yellow #FFFF00), Poor 201-300 (Orange #FF9900/Orange), Very Poor 301-400 (Red #FF0000), Severe 401-500 (Maroon/Dark Red #7E0023). Each band maps to associated health-impact text. Band widths are non-uniform: 50,50,100,100,100,100.
- **Inputs:** Final AQI integer 0-500 (capped at 500)
- **Outputs:** Category label + RGB color
- **Libraries:** `numpy`, `matplotlib`
- **Complexity:** O(1)
- **Pros:** Fixed, exact, O(1); same edges drive both sub-index I_Lo/I_Hi and final colorization.
- **Cons:** CPCB uses non-uniform band widths (50,50,100,100,100,100) — do not assume 100-wide bands.
- **Recommendation:** Hard-code band edges [0,50,100,200,300,400,500] and a 6-color LUT; index = np.searchsorted on the final AQI value for map colorization.

### PM2.5 sub-index (24h)

- **Category:** Pollutant breakpoint table
- **Summary:** Averaging: 24h mean, units ug/m3. Conc breakpoints [BP_Lo-BP_Hi] -> AQI [I_Lo-I_Hi]: 0-30->0-50; 31-60->51-100; 61-90->101-200; 91-120->201-300; 121-250->301-400; 250-500(extrapolate, >250)->401-500. NAAQS 24h=60.
- **Inputs:** 24h PM2.5 ug/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** Primary driver in Indo-Gangetic Plain; satellite proxy = INSAT-3D/3DR AOD + TROPOMI converted via ML.
- **Cons:** Top segment (>250) has no published BP_Hi; implementations cap at 380->500 or clamp AQI to 500.
- **Recommendation:** Mandatory anchor pollutant. Conc array [0,30,60,90,120,250,(380 cap for 500)]; common practice caps the >250 segment using 250-380 ->401-500.

### PM10 sub-index (24h)

- **Category:** Pollutant breakpoint table
- **Summary:** 24h mean, ug/m3. 0-50->0-50; 51-100->51-100; 101-250->101-200; 251-350->201-300; 351-430->301-400; 430-500(>430)->401-500. NAAQS 24h=100.
- **Inputs:** 24h PM10 ug/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** Strongly tied to dust + AOD; good satellite observability.
- **Cons:** >430 segment open-ended; cap at 430-510->401-500.
- **Recommendation:** Second mandatory PM anchor; at least one of PM2.5/PM10 must be present. Conc array [0,50,100,250,350,430,510 cap].

### NO2 sub-index (24h)

- **Category:** Pollutant breakpoint table
- **Summary:** 24h mean, ug/m3. 0-40->0-50; 41-80->51-100; 81-180->101-200; 181-280->201-300; 281-400->301-400; 400-500(>400)->401-500. NAAQS 24h=80.
- **Inputs:** 24h NO2 ug/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** TROPOMI NO2 is high-quality, daily; key urban/IGP signal.
- **Cons:** Column-to-surface conversion needs PBL height + meteorology (ERA5/MERRA-2).
- **Recommendation:** Conc array [0,40,80,180,280,400,520 cap]. Satellite source = Sentinel-5P TROPOMI NO2 tropospheric column -> surface via ML/regression.

### SO2 sub-index (24h)

- **Category:** Pollutant breakpoint table
- **Summary:** 24h mean, ug/m3. 0-40->0-50; 41-80->51-100; 81-380->101-200; 381-800->201-300; 801-1600->301-400; 1600-2000(>1600)->401-500. NAAQS 24h=80.
- **Inputs:** 24h SO2 ug/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** Captures point-source plumes.
- **Cons:** TROPOMI SO2 has high detection limit -> often below noise; rely on reanalysis/CAMS for fill.
- **Recommendation:** Conc array [0,40,80,380,800,1600,2000 cap]. Satellite = TROPOMI SO2 (noisier; useful near power plants/smelters).

### CO sub-index (8h, mg/m3)

- **Category:** Pollutant breakpoint table
- **Summary:** Averaging: max 8h rolling mean, units mg/m3 (NOT ug/m3). 0-1.0->0-50; 1.1-2.0->51-100; 2.1-10->101-200; 10.1-17->201-300; 17.1-34->301-400; 34-50(>34)->401-500. NAAQS 8h=2 mg/m3.
- **Inputs:** 8h CO mg/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** TROPOMI CO daily, good for biomass-burning plume tracking.
- **Cons:** Unit pitfall: mixing mg/m3 vs ug/m3 corrupts the whole AQI; CO is a total-column proxy not surface.
- **Recommendation:** Conc array [0,1,2,10,17,34,50 cap] in mg/m3. CRITICAL unit difference. Satellite = TROPOMI CO total column (mol/m2) -> convert to mg/m3 surface.

### O3 sub-index (8h with 1h override)

- **Category:** Pollutant breakpoint table
- **Summary:** ug/m3. CPCB rule: use max 8h up to AQI 300, then max 1h for higher bands. 0-50->0-50 (8h); 51-100->51-100 (8h); 101-168->101-200 (8h); 169-208->201-300 (8h); 209-748->301-400 (1h); 748-1000(>748)->401-500 (1h). NAAQS 8h=100,1h=180.
- **Inputs:** max 8h and max 1h O3 ug/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** TROPOMI O3 + reanalysis; secondary pollutant, photochemical.
- **Cons:** Dual 8h/1h averaging makes it the only pollutant with mixed-period segments; coding error-prone.
- **Recommendation:** Compute candidate from 8h breakpoints [0,50,100,168,208] and from 1h [208,748,1000]; per CPCB take the larger applicable. Conc array typically [0,50,100,168,208,748,1000 cap].

### NH3 sub-index (24h)

- **Category:** Pollutant breakpoint table
- **Summary:** 24h mean, ug/m3. 0-200->0-50; 201-400->51-100; 401-800->101-200; 801-1200->201-300; 1201-1800->301-400; 1800-2400(>1800)->401-500. NAAQS 24h=400.
- **Inputs:** 24h NH3 ug/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** Agricultural/IGP relevance.
- **Cons:** Sparse monitoring + no good satellite surface product (IASI NH3 is column, coarse).
- **Recommendation:** Conc array [0,200,400,800,1200,1800,2400 cap]. Rarely the driver; few ground monitors report NH3.

### Pb sub-index (24h)

- **Category:** Pollutant breakpoint table
- **Summary:** 24h mean, ug/m3. 0-0.5->0-50; 0.5-1.0->51-100; 1.1-2.0->101-200; 2.1-3.0->201-300; 3.1-3.5->301-400; 3.5-4.0(>3.5)->401-500. NAAQS 24h=1.0.
- **Inputs:** 24h Pb ug/m3
- **Outputs:** Sub-index 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1)
- **Pros:** Completeness with CPCB.
- **Cons:** No satellite proxy; effectively unused in remote-sensing AQI.
- **Recommendation:** Conc array [0,0.5,1.0,2.0,3.0,3.5,4.0 cap]. Not satellite-observable; usually omitted from satellite AQI.

### Sub-index linear interpolation formula

- **Category:** Core formula
- **Summary:** For pollutant p with concentration Cp in segment [BP_Lo, BP_Hi] mapping to AQI [I_Lo, I_Hi]: Ip = ((I_Hi - I_Lo)/(BP_Hi - BP_Lo)) * (Cp - BP_Lo) + I_Lo. Linear within each segment; piecewise-linear overall. Round to nearest integer.
- **Inputs:** Cp + breakpoint LUT
- **Outputs:** Ip integer 0-500
- **Libraries:** `numpy`
- **Complexity:** O(1) per pollutant per grid cell; O(N) over grid, vectorized
- **Pros:** Exact CPCB method; O(1) per cell; fully vectorizable over the grid.
- **Cons:** Edge handling at exact breakpoints and >max-conc must be defined (searchsorted side + clamp).
- **Recommendation:** Vectorize: idx = np.searchsorted(bp_conc[p], Cp, side='right')-1; then apply formula with gathered BP/I_Lo/Hi arrays. Clamp Cp>top to AQI=500.

### Overall AQI = max of sub-indices (>=3 pollutant + PM rule)

- **Category:** Aggregation rule
- **Summary:** AQI = max over available pollutant sub-indices. Valid ONLY if >=3 pollutants have data AND at least one is PM2.5 or PM10. Otherwise AQI undefined/not reported. The pollutant attaining the max is the 'responsible/prominent pollutant'.
- **Inputs:** Stack of 3-8 sub-indices + validity mask
- **Outputs:** Final AQI + responsible pollutant
- **Libraries:** `numpy`
- **Complexity:** O(P) reduction, vectorized
- **Pros:** Simple max reduction; conservative (worst pollutant governs).
- **Cons:** Pure max ignores co-pollutant burden; satellite gaps can break the >=3 rule -> need gap-fill (reanalysis/CAMS) to keep validity over cloudy IGP.
- **Recommendation:** mask = (count_valid>=3) & (PM2.5 valid | PM10 valid); AQI = np.where(mask, np.nanmax(subindex_stack, axis=0), NaN). Track argmax for responsible-pollutant map layer.

### O(1) vectorized lookup-table implementation

- **Category:** Implementation
- **Summary:** Precompute per-pollutant arrays: bp_conc (n_seg+1 edges), I_lo, I_hi, bp_lo, bp_hi. Per grid cell+pollutant: seg=clip(searchsorted(bp_conc,C,'right')-1,0,n_seg-1); Ip=(I_hi[seg]-I_lo[seg])/(bp_hi[seg]-bp_lo[seg])*(C-bp_lo[seg])+I_lo[seg]; clamp [0,500]. Stack pollutants on axis 0, apply max+mask.
- **Inputs:** Aligned per-pollutant concentration grids
- **Outputs:** Daily surface AQI map + responsible-pollutant map
- **Libraries:** `numpy`, `xarray`, `dask`
- **Complexity:** O(1) per cell; O(grid*pollutants) total, fully vectorized
- **Pros:** Truly O(1) per cell, branch-free, GPU/dask-friendly for full-India daily maps.
- **Cons:** Must keep CO in mg/m3 and O3 8h/1h handling in the LUT construction, not the hot loop.
- **Recommendation:** Implement as a single function aqi(conc_dict)->aqi_grid,responsible_grid. Use float32 for grid; precompute constants once (no per-cell branching).

### Comparison: US EPA AQI vs WHO 2021

- **Category:** Benchmark / standards comparison
- **Summary:** EPA AQI: same piecewise-linear max-of-sub-index method but different categories (USG/Unhealthy/Hazardous), different breakpoints, and ppb/ppm units (CO 8h in ppm, NO2/SO2 1h ppb). WHO 2021 AQGs are health guidelines, not an index: PM2.5 5(annual)/15(24h), PM10 15/45, NO2 10/25, O3 100(8h), SO2 40(24h), CO 4 mg/m3. WHO 24h PM2.5=15 is half CPCB Good ceiling (60) -> India bands far more lenient.
- **Inputs:** Standard tables
- **Outputs:** Comparative mapping
- **Complexity:** N/A
- **Pros:** Clarifies why India AQI 'Good' can exceed WHO safe limits; aids interpretation.
- **Cons:** Cross-standard unit/category mixing is a common bug; EPA CO is ppm vs CPCB mg/m3.
- **Recommendation:** Validate in CPCB-AQI units (the scored target); report WHO-exceedance as overlay only. Never substitute EPA ppb/ppm breakpoints into the CPCB ug/m3 + CO-mg/m3 computation.

**SOTA references:**

- CPCB (2014) 'National Air Quality Index' report, Central Pollution Control Board, MoEFCC — official NAQI methodology and breakpoint table (cpcb.nic.in, About_AQI.pdf / FINAL-REPORT_AQI.pdf)
- CPCB AQI portal airquality.cpcb.gov.in / cpcb.nic.in/National-Air-Quality-Index — operational reference and live AQI bulletins
- Breakpoints of different pollutants in IND-AQI (CPCB, 2014), ResearchGate tbl1_315725810 — peer-reproduced breakpoint table
- US EPA Technical Assistance Document for the Reporting of Daily Air Quality (AQI), EPA-454/B-18-007 — EPA breakpoints/method for comparison
- WHO Global Air Quality Guidelines 2021 (PM2.5, PM10, O3, NO2, SO2, CO) — health-based reference limits
- Sentinel-5P TROPOMI L2 product docs (NO2, SO2, CO, O3, HCHO) — column inputs; GEE assets COPERNICUS/S5P/OFFL/L3_* and NRTI variants
- ERA5 (ECMWF) / MERRA-2 (NASA GMAO) / IMDAA reanalysis — PBL height + meteorology for column-to-surface conversion
- MODIS/VIIRS FIRMS active-fire products for fire-HCHO correlation in biomass-burning hotspot analysis

---

## Converting columnar AOD to surface PM2.5 over India (BAH 2026 PS3, Objective 1: daily surface AQI from satellite columns)

*Pipeline stage: **Acquisition / Preprocess** · 8 methods.*

**Recommended stack:**

- **AOD input: MODIS MAIAC MCD19A2 1km (GEE: MODIS/061/MCD19A2_GRANULES, bands Optical_Depth_055/047) as primary; INSAT-3D AOD (ISRO MOSDAC/VEDAS, geostationary ~hourly) to fill polar-orbit gaps; MERRA-2 AOD (M2T1NXAER) / CAMS for residual cloud gaps**
- **Gap-filling stage: multi-sensor + reanalysis RF imputation to make seamless daily 1km AOD with a QA/uncertainty channel**
- **Meteorology: ERA5 single-levels for BLH (GEE: ECMWF/ERA5/HOURLY) + ERA5-Land hourly (GEE: ECMWF/ERA5_LAND/HOURLY) for T/RH/wind/precip; note BLH is in ERA5 NOT ERA5-Land; IMDAA (NCMRWF, 12km India) for native BLH/RH; MERRA-2 for aerosol composition**
- **Trace gases (composition + AQI sub-indices): Sentinel-5P TROPOMI OFFL L3 - NO2 (COPERNICUS/S5P/OFFL/L3_NO2), SO2 (..._SO2), CO (..._CO), O3 (..._O3), HCHO (COPERNICUS/S5P/OFFL/L3_HCHO)**
- **Static covariates: SRTM/Copernicus DEM, MODIS NDVI (MOD13), land use/land cover, road density, population (GHSL/WorldPop), lat/lon + day-of-year sin/cos encodings**
- **Physics-guided features: AOD/BLH (PBL normalization), f(RH) hygroscopic correction ((1-RH)^-g / IMPROVE growth curve), AOD x MEE priors, aerosol-type flags from TROPOMI AER_AI**
- **Ground truth: CPCB CAAQMS continuous stations (~500+) PM2.5/PM10 + sub-index pollutants for train/val; AERONET for AOD QA**
- **Models: (1) LightGBM/RF tabular baseline + airshed clustering; (2) reproduce Wei STET as benchmark; (3) physics-guided ConvLSTM/CNN-LSTM as primary PS3 model, pretrain on MERRA-2 PM2.5 then fine-tune on CPCB; final = ensemble LightGBM + CNN-LSTM**
- **Benchmark/validation: van Donkelaar WUSTL SatPM2.5 V5.GL.06 (1998-2024) and GHAP/CHAP as independent cross-checks; report RMSE/R/MAE with leave-time-out + leave-station-out cross-validation**

**Key findings:**

- PM2.5 = AOD x eta is the core relation; eta is NOT constant (varies 5-10x). It decomposes as eta = 1/(MEE x H x f(RH)) x profile: inverse mass-extinction-efficiency, inverse aerosol scale height (proxied by PBL height BLH), inverse hygroscopic growth f(RH), times PBL-resident column fraction. BLH and RH dominate.
- Raw AOD-PM2.5 correlation over India is weak (R~0.4-0.6); PBL normalization (AOD/BLH) and humidity correction f(RH) (steep above ~70% RH) are the highest-value corrections, lifting physical estimates to R2 ~0.6-0.7 before ML.
- Two SOTA paradigms: (a) van Donkelaar WUSTL GWR-fused geophysical (GEOS-Chem ratio + GWR) -> global SatPM2.5 V5.GL.06 (1998-2024), CV-R2 0.80-0.90, mostly monthly/annual; (b) Wei et al. GHAP Space-Time Extra-Trees -> gap-free 1km DAILY, CV-R2 ~0.92, RMSE ~10.8. STET is the closer PS3 daily template.
- India ML is proven: IGP 16-city Random Forest R2 ~0.94 (RMSE 8-14); Maharashtra 1km RF R2 0.87 (RMSE 12.6); national airshed-clustered RF R2 0.80 (RMSE 23); LongPMInd national daily CV-R2 ~0.77. National models score below city/regional due to monitor sparsity + regime diversity.
- INSAT-3D AOD has poor AERONET correlation vs MODIS, but its geostationary (~hourly) sampling is uniquely valuable to FILL temporal/cloud gaps of polar MODIS/VIIRS. Use INSAT for gap-filling/diurnal context, MAIAC MODIS 1km as the quantitative AOD backbone.
- Dominant errors: cloud/retrieval gaps (MAIAC daily India coverage often <50%); night (no passive AOD, so use reanalysis/temporal models); vertical-profile uncertainty (lofted dust/biomass smoke in IGP winter crop-fire breaks surface-column link); hygroscopic/composition variability; CPCB sparsity.
- Recommended India predictors: gap-filled AOD(550) + BLH + RH(+f(RH)) + T + wind U/V + precip + DEM + NDVI + land use + population + TROPOMI NO2/SO2/CO/O3/HCHO + lat/lon + DOY encodings + physics features AOD/BLH and AOD/f(RH). Add airshed clustering to boost national skill.
- Recommended pipeline: gap-fill AOD (INSAT+MERRA-2 RF fusion) -> physics-guided features (PBL + f(RH) normalization) -> pretrain ConvLSTM on MERRA-2 PM2.5 -> fine-tune on CPCB -> ensemble with LightGBM/STET -> validate leave-station-out + leave-time-out, cross-check vs V5.GL and GHAP. Realistic daily R2 ~0.80-0.90.

### Physical eta-scaling: PM2.5 = AOD x eta (column-to-surface conversion)

- **Category:** Physical / first-principles
- **Summary:** Core relation PM2.5 = eta * AOD. Expanded eta = 1/(MEE * H * f(RH)) * profile, where MEE = mass extinction efficiency (~3-7 m2/g), H = aerosol scale height proxied by PBL/boundary-layer height (BLH), f(RH) = hygroscopic growth (extinction rises steeply above ~70% RH), profile = PBL-resident column fraction. eta also varies with aerosol type. BLH and RH dominate.
- **Inputs:** Columnar AOD (550nm), PBL/boundary-layer height (BLH), relative humidity, aerosol type/composition prior, vertical extinction profile (CALIPSO/model)
- **Outputs:** Surface dry PM2.5 (ug/m3) per pixel
- **Libraries:** `numpy`, `xarray`, `metpy`
- **Complexity:** Low compute; needs BLH, RH, MEE/aerosol-type priors. Sensitive to assumptions.
- **Pros:** Interpretable, no ground labels needed, physically transferable, exposes each error term
- **Cons:** eta varies 5-10x in space/time; MEE and profile assumptions dominate error; poor in mixed/dust regimes
- **Benchmark:** Bare AOD-PM2.5 R~0.4-0.6 over India; rises to R2 0.6-0.7 after BLH + f(RH) normalization. Strong seasonality (IGP winter best).
- **Recommendation:** Use as ENGINEERED FEATURES (AOD/BLH, AOD*f(RH)^-1, aerosol-type flags) fed into ML, not standalone. Best as physics-guided priors for CNN/LSTM.

### GEOS-Chem model-ratio scaling (van Donkelaar geophysical method)

- **Category:** Physical / CTM model-ratio
- **Summary:** Compute simulated ratio eta_sim = surface_PM2.5/AOD from a chemical transport model (GEOS-Chem, or MERRA-2/CAMS for India) per grid/time, then PM2.5_sat = AOD_obs * eta_sim. The CTM supplies vertical profile, f(RH), composition and MEE self-consistently. This is the geophysical backbone of van Donkelaar V4/V5 before ground calibration.
- **Inputs:** Satellite AOD (MODIS/MISR/VIIRS/MAIAC or INSAT-3D), CTM/reanalysis simulated AOD + surface PM2.5 (GEOS-Chem, MERRA-2 M2T1NXAER, CAMS)
- **Outputs:** Geophysical (uncalibrated) surface PM2.5 grid
- **Libraries:** `xarray`, `gcpy`, `numpy`
- **Complexity:** High: requires CTM output or reanalysis aerosol fields; moderate to apply ratio.
- **Pros:** Physically complete eta, consistent composition/vertical/RH, global/gap-free, no local training
- **Cons:** Inherits CTM biases (India emission inventory gaps, dust/biomass errors); coarse native res (0.1-0.5 deg)
- **Benchmark:** Geophysical estimate R2 ~0.6-0.8 vs ground; underpins WUSTL V5.GL. MERRA-2-based ratio over India r~0.9 after calibration.
- **Recommendation:** Use MERRA-2/CAMS ratio (not full GEOS-Chem run) to make a gap-free physical PM2.5 prior, then bias-correct/fuse with CPCB via ML. Strong gap-filler for cloud pixels.

### Geographically Weighted Regression fusion (van Donkelaar V5/V6 calibration)

- **Category:** Statistical fusion / hybrid
- **Summary:** Calibrate geophysical (CTM-ratio) PM2.5 to ground monitors via GWR: local spatially-varying regression of ground PM2.5 on geophysical PM2.5 plus predictors (elevation, land use, NDVI, urban fraction). Corrects heterogeneous bias. This is the van Donkelaar V5.GL.06 (1998-2024) approach producing the global SatPM2.5 product.
- **Inputs:** Geophysical PM2.5, ground PM2.5 (CPCB), geographic/land-use covariates, distance kernels
- **Outputs:** Bias-corrected gridded surface PM2.5 (monthly/annual native; daily possible)
- **Libraries:** `mgwr`, `scikit-learn`, `geopandas`
- **Complexity:** Moderate; needs dense ground network for local weights; weak where stations sparse.
- **Pros:** SOTA global standard, corrects local bias, interpretable spatial coefficients
- **Cons:** Degrades where CPCB sparse; mainly monthly/annual; weaker for daily extremes
- **Benchmark:** Global CV R2 ~0.80-0.90 (V5). Best at high monitor density; India IGP well covered, peninsular/NE sparse.
- **Recommendation:** Use V5.GL as independent validation/benchmark and static bias prior. For DAILY India maps prefer ML (RF/STET/CNN-LSTM); keep GWR-style geographic covariates.

### ChinaHighPM2.5 / GHAP Space-Time Extra-Trees (STET) - Wei et al.

- **Category:** ML ensemble (tree-based)
- **Summary:** Space-Time Extremely Randomized Trees: extra-trees ensemble with explicit spatiotemporal terms (lat, lon, day-of-year, autocorrelation features) fusing AOD, reanalysis meteo, emissions, land use, population. Produces seamless gap-free 1km DAILY PM2.5 (GHAP/CHAP series). Directly analogous to the India daily map PS3 needs.
- **Inputs:** MAIAC AOD 1km, ERA5/MERRA-2 meteo (BLH, RH, T, wind, precip), DEM, NDVI, population, land use, spatiotemporal coordinates
- **Outputs:** Gap-free 1km daily surface PM2.5 (plus PM10/O3/NO2/SO2/CO in CHAP suite)
- **Libraries:** `scikit-learn`, `lightgbm`, `xgboost`, `numpy`
- **Complexity:** Moderate-high compute; mature, documented; needs gap-filled AOD input first.
- **Pros:** SOTA accuracy, gap-free, fast vs deep nets, handles spatiotemporal heterogeneity, proven country-scale
- **Cons:** Needs AOD gap-filling first; trees extrapolate poorly to unseen regimes; less spatial context than CNN
- **Benchmark:** Daily CV-R2 ~0.92, RMSE ~10.8, MAE ~6.3 ug/m3 (China); monthly R2 ~0.80. Among best-published daily 1km.
- **Recommendation:** Strong contender for India daily 1km. Reproduce STET as the tabular benchmark to beat; ensemble with CNN-LSTM; mirror its predictor stack.

### Empirical two-stage / mixed-effects linear regression

- **Category:** Statistical empirical
- **Summary:** Stage 1: calibrate AOD-PM2.5 with day-specific (mixed-effects) or site-specific slopes/intercepts; Stage 2: predict on AOD-missing days via meteorology + spatial smoothing. Includes plain MLR PM2.5 = a + b*AOD + c*BLH + d*RH + e*WS. Classic Beijing/Madrid/Dhaka approach.
- **Inputs:** AOD, BLH, RH, wind speed, temperature, day/site dummies
- **Outputs:** Surface PM2.5 (station-calibrated)
- **Libraries:** `statsmodels`, `pymer4`, `scikit-learn`
- **Complexity:** Low; fast, transparent.
- **Pros:** Transparent, minimal data, fast explanatory baseline
- **Cons:** Misses nonlinearity/interactions; weak national generalization; large RMSE in extremes
- **Benchmark:** Day-specific mixed-effects R2 ~0.6-0.8 regionally; plain MLR R2 ~0.4-0.6. Below tree/DL.
- **Recommendation:** Use only as interpretable baseline and feature-sign sanity check. Not competitive for scored RMSE/R; do not submit as final.

### Random Forest / Gradient Boosting (tabular ML) - India IGP proven

- **Category:** ML ensemble (tree-based)
- **Summary:** RF/XGBoost/LightGBM regressing CPCB PM2.5 on satellite AOD + meteo + land-use + spatiotemporal features. Demonstrated over IGP (16 cities) and Maharashtra at 1km. The realistic high-scoring workhorse baseline for PS3 Objective 1.
- **Inputs:** MAIAC/INSAT AOD, ERA5 BLH+RH+T+wind+precip, DEM, NDVI, population, land use, lat/lon, DOY; optional TROPOMI NO2/SO2/CO/O3/HCHO
- **Outputs:** 1km daily/grid surface PM2.5
- **Libraries:** `lightgbm`, `xgboost`, `scikit-learn`, `shap`
- **Complexity:** Low-moderate compute; very mature tooling.
- **Pros:** High accuracy, fast, SHAP-interpretable, robust to missing patterns, strong India track record
- **Cons:** No spatial-context/temporal-memory; national models lower R2 than city; needs gap-filled AOD
- **Benchmark:** IGP RF R2 ~0.94, RMSE 8-14; Maharashtra R2 0.87, RMSE 12.6; national airshed-clustered RF R2 0.80, RMSE 23 ug/m3.
- **Recommendation:** Mandatory baseline. Add TROPOMI gases + airshed clustering (national R2 ~0.80). Ensemble with CNN-LSTM; use SHAP to justify predictors.

### CNN / LSTM / CNN-LSTM deep hybrids

- **Category:** ML deep learning (PS3-specified)
- **Summary:** CNN captures spatial context (AOD/meteo grids, neighborhood emissions); LSTM/temporal captures pollutant memory and transport lag; CNN-LSTM/ConvLSTM fuses both for spatiotemporal daily PM2.5. Optionally physics-guided (inject AOD/BLH, f(RH) channels) + location encoders. The architecture explicitly named for PS3 scoring.
- **Inputs:** Multichannel rasters: AOD, ERA5/IMDAA/MERRA-2 meteo (BLH,RH,T,U,V,precip), TROPOMI NO2/SO2/CO/O3/HCHO, DEM, NDVI, land use, lat/lon/DOY; CPCB PM2.5 labels
- **Outputs:** Gridded daily surface PM2.5 / AQI maps
- **Libraries:** `pytorch`, `tensorflow`, `xarray`, `rioxarray`
- **Complexity:** High compute (GPU); needs careful patch sampling, gap masks, more data.
- **Pros:** Models spatial context + temporal transport (key for IGP smog), end-to-end, matches PS3 ask
- **Cons:** Data-hungry vs sparse CPCB; harder to interpret; overfit risk; needs gap-free inputs
- **Benchmark:** Comparable-to-better than RF with enough data; ConvLSTM/location-encoder studies beat RF RMSE; overfit risk on ~500 CPCB stations.
- **Recommendation:** Primary PS3 submission model. Physics-guided channels + ConvLSTM; pretrain on MERRA-2 PM2.5 then fine-tune on CPCB; ensemble with LightGBM to win RMSE/R/MAE.

### AOD gap-filling preprocessing (cloud/no-retrieval recovery)

- **Category:** Preprocessing / data fusion
- **Summary:** Recover missing AOD (clouds, glint, bright surface, night) before retrieval: fuse multi-sensor AOD (MAIAC Terra+Aqua, VIIRS, INSAT-3D geostationary) and fill with reanalysis (MERRA-2/CAMS) via RF/mean-filter/IDW imputation. Critical because raw MAIAC daily coverage over India is often under 50%.
- **Inputs:** Multi-sensor AOD (MODIS MAIAC, VIIRS, INSAT-3D), MERRA-2/CAMS AOD, meteo, DEM, NDVI
- **Outputs:** Seamless gap-free daily AOD raster + QA channel
- **Libraries:** `scikit-learn`, `scipy`, `rasterio`, `gdal`
- **Complexity:** Moderate; adds a stage but critical for usable daily maps.
- **Pros:** Enables daily seamless maps; INSAT geostationary high temporal coverage fills MODIS gaps
- **Cons:** Imputed pixels carry extra uncertainty; must flag/propagate it
- **Benchmark:** RF gap-filling restores near-full coverage with small added error; STET/GHAP depend on it for gap-free 1km daily.
- **Recommendation:** REQUIRED first stage. Use INSAT-3D (geostationary, hourly) to fill MODIS/VIIRS polar gaps; backfill with MERRA-2 AOD; carry a QA/uncertainty channel into the PM2.5 model.

**SOTA references:**

- van Donkelaar et al., WUSTL Atmospheric Composition Analysis Group SatPM2.5 V5.GL.06 (1998-2024): GEOS-Chem + multi-sensor AOD + GWR; sites.wustl.edu/acag/surface-pm2-5 ; satpm.org/v5-gl-06
- van Donkelaar et al., Environ. Sci. Technol. 2016/2021 - Global geophysical PM2.5 from AOD with GEOS-Chem ratio + GWR ground calibration (foundational)
- Wei J. et al., Atmos. Chem. Phys. 20, 3273-3289, 2020 - Improved 1km PM2.5 across China using enhanced Space-Time Extremely Randomized Trees (STET)
- Wei J. et al., Remote Sens. Environ. 2021 / ChinaHighPM2.5 (GHAP/CHAP) gap-free 1km daily, CV-R2 0.92; Zenodo 6398971; tapdata.org.cn
- High-Resolution PM2.5 over the Indo-Gangetic Plain by Fusion of Satellite Data, Meteorology, and Land Use, Environ. Sci. Technol. 2020 (doi 10.1021/acs.est.0c01769)
- Vishal et al., SSRN 5358795 - Satellite-AOD + ML for urban air quality over the Indo-Gangetic Plain (RF)
- LongPMInd: Reconstructing long-term (1980-2022) daily ground PM concentrations in India, ESSD 16, 3565, 2024 (national daily CV-R2 ~0.77)
- Airshed Delineation and PM2.5 Estimation across India using ML + spatial clustering (national RF R2 0.71->0.80; ResearchGate 395793667)
- Maharashtra seasonal 1km PM2.5 via RF, Discover Sustainability, Springer 2025 (R2 0.87, RMSE 12.6)
- Zhang & Li et al., ACP 21, 18375, 2021 - Spatiotemporal AOD-PM2.5 relationship (eta): BLH/RH/aerosol-type factors for MAIAC PM2.5 estimation

---

## SOTA Deep Learning for Surface AQI / Pollutant Concentration Estimation from Satellite + Meteorology over India (BAH 2026 PS3: Surface AQI mapping + HCHO biomass-burning hotspots)

*Pipeline stage: **ML Prediction** · 11 methods.*

**Recommended stack:**

- **PRIMARY (Objective-1 gridded daily Surface AQI): U-Net gap-fill/super-resolution preprocessing -> SA-ConvLSTM (ConvLSTM + temporal self-attention + channel attention) with ERA5/IMDAA met covariate channels, producing 1km daily PM2.5/NO2/SO2/CO/O3 -> CPCB-AQI computation. Satisfies PS CNN/LSTM/CNN-LSTM wording while being SOTA.**
- **SECONDARY / ENSEMBLE: GNN (torch-geometric-temporal A3TGCN/GConvLSTM) on CPCB stations with wind-derived dynamic adjacency for station validation + transductive (no-station) estimation + forecasting; LightGBM/RF tabular baseline; blend all three via a stacking meta-learner for best RMSE/R/MAE.**
- **PHYSICS DIFFERENTIATOR (optional, high value): add advection-diffusion PDE residual loss (ERA5/IMDAA winds) to ConvLSTM for transport realism over the Indo-Gangetic Plain.**
- **OBJECTIVE-2 (HCHO hotspots): U-Net HCHO gap-fill -> Getis-Ord Gi*/LISA significant clusters + ST-DBSCAN fire clustering + lagged FIRMS-fire vs TROPOMI-HCHO correlation + HYSPLIT/ERA5 back-trajectory transport, restricted to Oct-Nov and Mar-May burning seasons.**
- **INPUT TENSOR DESIGN: [B,T,C,H,W]; T=7-14 daily steps; C ~15-25 channels = INSAT-3D AOD + TROPOMI NO2/SO2/CO/O3/HCHO (gap-filled) + ERA5/IMDAA BLH/RH/T2m/U/V-wind/pressure/precip + DEM/land-use/NDVI/population/road-density + DOY sin/cos; HxW = tiled India grid (64x64 or 128x128 at 0.05deg/1km).**
- **TRAINING TRICKS: loss = Huber/Charbonnier (robust to PM2.5 tails) optionally + log-transform target + masked loss on valid pixels; per-channel z-score/min-max normalization fit on train only; SPATIAL-BLOCK + LEAVE-STATIONS-OUT CV (never random pixel split) and TEMPORAL holdout (whole months/season); AdamW + cosine/OneCycle LR + early stopping; report R/RMSE/MAE/bias by season and region (IGP vs rest).**
- **DATA / GEE ASSETS: COPERNICUS/S5P/OFFL/L3_HCHO (+ L3_NO2/SO2/CO/O3), ECMWF/ERA5_LAND/HOURLY, MODIS/061/MOD14A1 + FIRMS, MODIS MAIAC AOD (MCD19A2) and INSAT-3D AOD; MERRA-2 (M2T1NXAER) for aerosol reanalysis gap-fill; IMDAA reanalysis (NCMRWF) for India-tuned met.**
- **LIBRARIES: PyTorch + segmentation-models-pytorch (U-Net), custom/ndrplz ConvLSTM, PyTorch Geometric + torch-geometric-temporal (GNN), pytorch-forecasting (TFT), DeepXDE/torchdiffeq (PINN), PySAL/esda + scikit-learn DBSCAN + HYSPLIT/pysplit (hotspots), rioxarray/xarray/rasterio for geodata, Google Earth Engine + geemap for ingestion.**

**Key findings:**

- CNN-LSTM (PS-recommended) consistently beats standalone CNN or LSTM; representative PM2.5 R2=0.91, RMSE=8.2 ug/m3 on clean met-driven data. It is the safe baseline that directly matches PS wording.
- For GRIDDED daily maps (Objective-1), SA-ConvLSTM (ConvLSTM + self-attention) is superior to decoupled CNN-LSTM: hidden states stay 2D giving spatially coherent full-grid output and better spatial RMSE; attention fixes ConvLSTM's weak long-range temporal memory.
- The dominant satellite obstacle is cloud/swath gaps + coarse resolution (TROPOMI 3.5-7km, MERRA-2 ~10km, INSAT AOD). Two-stage 'U-Net impute/super-resolve -> estimate' pipelines (SLNet for NO2) are SOTA: NO2 R2=0.887-0.919 at 1km. Make U-Net gap-filling a mandatory preprocessing stage.
- India benchmarks to beat: IGP RF (AOD+met+land-use, 1km daily) R2=0.87; Lucknow hybrid RT-RF-CNN R2=0.90 RMSE=26.9; IGP MERRA-2 stacking ensemble (Delhi/Kanpur/Lucknow/Patna) R2=0.79-0.82 RMSE 27-31 ug/m3. Tree/stacking ensembles are strong, hard-to-beat tabular baselines.
- GNNs (wind-derived dynamic adjacency over CPCB stations) excel at irregular networks, transport-aware prediction, and estimating concentration where no station exists, but output node values not dense rasters; best as station validation/forecasting and an ensemble member alongside ConvLSTM.
- Physics-informed nets (AirPhyNet, advection-diffusion loss using ERA5/IMDAA winds) improve data-efficiency and physically-plausible extrapolation in sparse-station zones and support HCHO plume transport/source-localization for Objective-2; a high-value optional differentiator.
- TFT/Informer/iTransformer dominate long-horizon temporal forecasting and give interpretable covariate importance (which met/satellite driver matters), but are data-hungry and unnecessary for same-day mapping; reserve TFT for forecasting extension and explainability.
- Objective-2 HCHO hotspots are best handled with statistics/clustering not a predictor net: Getis-Ord Gi*/LISA clusters, ST-DBSCAN fire clustering, lagged FIRMS-HCHO correlation, HYSPLIT/ERA5 back-trajectories; Odisha/Chhattisgarh and IGP are confirmed hotspot zones.

### CNN (2D spatial regression)

- **Category:** Spatial / per-day grid estimation
- **Summary:** Convolutional layers learn spatial context (neighboring AOD/column gradients, land-use, emission patterns) to map satellite columns -> surface concentration on a single time slice. Image-to-pixel/image-to-image regression. Acts as the spatial feature extractor inside every hybrid. Exploits that surface PM2.5/NO2 depends on a spatial neighborhood (transport, urban form), not just the colocated column.
- **Inputs:** [B,C,H,W]. HxW=India grid (0.05deg ~5km or 1km, 64x64 tiles). C=AOD, TROPOMI NO2/SO2/CO/O3/HCHO, met (BLH,RH,T2m,U/V,pressure), DEM, land-use, NDVI, DOY sin/cos.
- **Outputs:** Per-pixel surface concentration (PM2.5 / NO2 / AQI) at station-validated points or full grid [B,1,H,W].
- **Libraries:** `PyTorch`, `torchvision`, `TensorFlow/Keras`, `segmentation-models-pytorch`
- **Complexity:** Low-moderate; ResNet/VGG backbone, 1-10M params; trains in hours on 1 GPU.
- **Pros:** Captures spatial autocorrelation and emission morphology; cheap; strong when temporal dynamics are weak (daily product).
- **Cons:** No temporal memory; sensitive to cloud gaps in input columns; needs spatial CV or it overfits to station locations.
- **Benchmark:** India Lucknow hybrid RT-RF-CNN: R2=0.90, RMSE=26.9 ug/m3 (sub-km PM2.5). Deep-CNN NO2 (Ghahremanloo 2021): CV-R=0.91.
- **Recommendation:** Use as the spatial encoder/backbone, not standalone. Good fast baseline for the daily-map objective; pair with met covariates and report spatial-block CV R/RMSE.

### LSTM / GRU (temporal sequence)

- **Category:** Temporal / per-station time series
- **Summary:** Recurrent nets model time evolution of column + met -> surface concentration at a fixed location/station. Captures lagged effects (nighttime boundary-layer collapse, multi-day accumulation episodes in IGP winter). GRU is a lighter LSTM variant with similar accuracy. Backbone of the temporal half of CNN-LSTM.
- **Inputs:** [B,T,F]. T=lookback (24h hourly or 7-30 daily steps). F per timestep: colocated AOD, TROPOMI columns, BLH, RH, wind, T, precip, prior-day PM2.5. Per-station or per-pixel sequences.
- **Outputs:** Next-step or same-step surface concentration; supports multi-step forecast horizons.
- **Libraries:** `PyTorch (nn.LSTM/GRU)`, `TensorFlow/Keras`, `pytorch-forecasting`, `darts`
- **Complexity:** Low; 1-3 stacked layers, hidden 64-256; fast.
- **Pros:** Strong on temporal lags and episode dynamics; ideal where dense CPCB hourly history exists.
- **Cons:** Ignores spatial structure (each station independent); struggles with long horizons (vanishing memory); needs continuous series (gaps hurt).
- **Benchmark:** Standalone LSTM PM2.5 typically R2 ~0.80-0.88. CNN-LSTM beats LSTM in nearly all studies.
- **Recommendation:** Use as the temporal block inside CNN-LSTM. Standalone only for pure station forecasting, not gridded mapping.

### Hybrid CNN-LSTM

- **Category:** Spatiotemporal (decoupled)
- **Summary:** CNN extracts spatial features per timestep, pooled features feed an LSTM modeling temporal evolution. The PS-recommended workhorse; consistently outperforms standalone CNN or LSTM. Decoupled (not fully spatiotemporal-convolutional) so cheaper than ConvLSTM but loses spatial detail in the temporal stage.
- **Inputs:** [B,T,C,H,W]: CNN applied per t -> sequence of feature vectors -> LSTM. Channels = satellite columns + met + static covariates; T=7-24 steps.
- **Outputs:** Surface concentration / AQI at target time (now-cast) or +1..+n steps.
- **Libraries:** `PyTorch`, `TensorFlow/Keras`, `Keras TimeDistributed + LSTM`
- **Complexity:** Moderate; 2-20M params; trains in hours-day on 1 GPU.
- **Pros:** Captures both space and time; robust, well-documented, judge-friendly (named in PS); good accuracy/cost trade-off.
- **Cons:** Spatial info bottlenecked into a vector before LSTM; less spatially coherent than ConvLSTM for full-grid output.
- **Benchmark:** Generic CNN-LSTM PM2.5: R2=0.91, RMSE=8.2 ug/m3 (clean met-driven dataset). Outperforms CNN/LSTM alone across literature.
- **Recommendation:** Strong safe choice and directly satisfies PS wording. Use as primary baseline; upgrade to ConvLSTM+attention if compute allows.

### ConvLSTM / SA-ConvLSTM (self-attention ConvLSTM)

- **Category:** Spatiotemporal (coupled, full-grid)
- **Summary:** Replaces LSTM matmuls with convolutions so hidden/cell states stay 2D maps -> joint spatial+temporal learning and spatially coherent full-grid output. SA-ConvLSTM adds self-attention to fix ConvLSTM's weak long-range temporal memory. Best fit for daily gridded surface-AQI maps over India.
- **Inputs:** [B,T,C,H,W] preserved throughout. T sequence of co-registered daily/hourly stacks (AOD+columns+met). Output is a map, not a vector.
- **Outputs:** Full-grid surface concentration/AQI map [B,1,H,W] (or [T_out,1,H,W] for forecast).
- **Libraries:** `PyTorch (custom ConvLSTM cell)`, `ndrplz/ConvLSTM_pytorch`, `TensorFlow ConvLSTM2D`
- **Complexity:** Moderate-high; memory-heavy (states are HxW); needs adequate VRAM; tile the India grid.
- **Pros:** Spatially coherent maps, joint spatiotemporal learning, naturally fills small temporal gaps; SOTA for gridded products.
- **Cons:** GPU/VRAM hungry; ConvLSTM alone weak on long horizons (fix with attention); more tuning.
- **Benchmark:** Att-ConvLSTM / SA-ConvLSTM outperform CNN-LSTM/plain ConvLSTM on PM2.5 mapping; typical R2 gains of 0.02-0.05 with better spatial RMSE.
- **Recommendation:** Recommended core for Objective-1 gridded AQI maps. Use SA-ConvLSTM (temporal self-attention) + met covariate channels.

### U-Net / Encoder-Decoder (gap-filling, super-resolution, image-to-image)

- **Category:** Spatial gap-filling / downscaling
- **Summary:** Encoder-decoder with skip connections. Two critical roles: (a) impute cloud/swath gaps in TROPOMI/INSAT columns before the estimator; (b) super-resolve coarse columns/AOD (3.5-7km TROPOMI, ~10km MERRA-2) to 1km. Two-stage 'impute-then-estimate' pipelines (SLNet) are SOTA for NO2.
- **Inputs:** [B,C,H,W] with masked/missing channels + a validity-mask channel; static covariates (DEM, land-use, roads) as downscaling guidance.
- **Outputs:** Gap-filled / super-resolved column map, or directly surface concentration map [B,1,H,W].
- **Libraries:** `segmentation-models-pytorch`, `monai`, `fastai U-Net`, `Keras U-Net`
- **Complexity:** Moderate; 5-30M params (U-Net/ResUNet); standard segmentation training.
- **Pros:** Solves the #1 satellite problem (cloud gaps + coarse resolution); continuous daily maps; pairs with any downstream model.
- **Cons:** Imputation can hallucinate where unconstrained; needs gap-free reference (reanalysis/CTM) to train against; validate filled pixels separately.
- **Benchmark:** SLNet two-stage (1km NO2, England): R2=0.887-0.919. Residual deep nets reconstruct 1km ground-level NO2 over most of mainland China.
- **Recommendation:** Essential preprocessing stage. U-Net to gap-fill TROPOMI HCHO/NO2 and downscale AOD to 1km BEFORE the ConvLSTM/CNN-LSTM estimator. Key cross-dataset gap-filling step.

### Transformers / TFT / Informer / iTransformer (long-range temporal)

- **Category:** Temporal (long-horizon, attention)
- **Summary:** Self-attention for long-sequence forecasting. TFT handles multivariate covariates with variable-selection + interpretable attention, applied to AQI. Informer's ProbSparse attention + distilling solves O(n^2) cost. iTransformer attends across variates. Better than LSTM for multi-day episode forecasting and feature attribution.
- **Inputs:** [B,T_in,F] with known-future covariates (forecast met, calendar) separated from observed. TFT distinguishes static / known-future / observed inputs.
- **Outputs:** Multi-horizon quantile forecasts of surface concentration/AQI; attention weights for interpretability.
- **Libraries:** `pytorch-forecasting (TFT)`, `Informer2020`, `darts`, `HuggingFace time-series`, `NeuralForecast`
- **Complexity:** Moderate-high; attention memory-heavy for long T (Informer mitigates); more data-hungry than RNNs.
- **Pros:** Best long-range temporal modeling; native multivariate covariates; quantile/uncertainty output; interpretable (TFT variable importance).
- **Cons:** Data-hungry, more tuning, weaker on small/gappy datasets; pure temporal (needs spatial wrapper for maps).
- **Benchmark:** TFT and SpatioTemporal-Informer report lower MAE/RMSE than LSTM/GRU on multi-city PM2.5; sparse-attention transformers cut complexity while matching accuracy.
- **Recommendation:** Use TFT for the forecasting extension and interpretable covariate importance. Not needed for core same-day mapping.

### Attention mechanisms (spatial + temporal + cross-attention)

- **Category:** Add-on module (all architectures)
- **Summary:** Spatial attention weights surrounding pixels/stations; temporal attention weights informative lags; cross/channel attention reweights satellite vs met channels. Lightweight add-ons (SE, CBAM, self-attention) that consistently lift CNN-LSTM/ConvLSTM accuracy and add interpretability for judging.
- **Inputs:** Operates on feature maps/sequences inside the backbone; no new external inputs.
- **Outputs:** Reweighted features + attention maps usable as explainability artifacts.
- **Libraries:** `PyTorch nn.MultiheadAttention`, `timm (SE/CBAM)`, `custom`
- **Complexity:** Low add-on cost.
- **Pros:** Cheap accuracy gain; interpretability (which met driver / which neighbor drove a prediction); helps long-range temporal.
- **Cons:** Can overfit on small data; attention maps not always physically faithful.
- **Benchmark:** Spatial+temporal attention encoder-decoder and Att-ConvLSTM consistently beat non-attention baselines in PM2.5 studies.
- **Recommendation:** Add temporal self-attention + channel attention to the ConvLSTM core. High value-per-line; strengthens demo/interpretability.

### Graph Neural Networks (ST-GNN / GCN-LSTM / graph attention)

- **Category:** Spatiotemporal over irregular station network
- **Summary:** Models CPCB stations as graph nodes with edges from distance and wind field (directed dynamic graphs). GCN/GAT propagate along transport pathways; combined with LSTM/temporal conv for time. Excels at sparse irregular networks and at estimating concentration where no station exists. Wind-derived dynamic adjacency encodes transport (key for IGP).
- **Inputs:** Node features [N_stations,T,F] (columns+met per station), adjacency A from geo-distance + wind direction (dynamic). Optional grid as super-nodes.
- **Outputs:** Per-node concentration/forecast; can infer unmonitored locations.
- **Libraries:** `PyTorch Geometric (PyG)`, `DGL`, `torch-geometric-temporal (DCRNN, A3TGCN, GConvLSTM)`
- **Complexity:** Moderate; depends on graph size; dynamic graphs add cost.
- **Pros:** Native fit for irregular CPCB network; encodes physical transport via wind-graph; strong for no-station prediction and forecasting; SOTA on multi-station benchmarks.
- **Cons:** Produces node values not dense rasters (needs interpolation/decoder for full map); graph construction is a design choice; CPCB may be sparse regionally.
- **Benchmark:** Dynamic-graph GCN-LSTM, MSDGNN, SA-GNN, STGATN report lower RMSE than CNN-LSTM on multi-station PM2.5; strong transductive (no-station) prediction.
- **Recommendation:** Excellent secondary model for station-level validation/forecasting and transport-aware prediction. For the gridded deliverable prefer ConvLSTM; consider GNN-stations + ConvLSTM-grid ensemble.

### Physics-Informed NN (AirPhyNet / advection-diffusion-constrained)

- **Category:** Hybrid physics + DL
- **Summary:** Embeds advection-diffusion PDE (wind transport + diffusion) as a soft loss or Neural-ODE layer. AirPhyNet = RNN encoder + GNN differential-equation network + decoder. Improves generalization, data efficiency, physical plausibility; reduces overfitting in sparse-station regions and respects wind transport.
- **Inputs:** Column+met tensors PLUS explicit wind field (U,V) and diffusion coefficients; PDE residual on grid/graph; collocation points.
- **Outputs:** Physically-consistent concentration field + plausible spatial extrapolation.
- **Libraries:** `PyTorch`, `DeepXDE`, `NVIDIA Modulus`, `torchdiffeq (Neural ODE)`, `PyG (AirPhyNet)`
- **Complexity:** High; PDE-loss tuning, ODE solvers, slower/stiffer optimization.
- **Pros:** Physically plausible extrapolation; data-efficient; better in data-scarce zones; supports source-localization (inverse) useful for HCHO attribution.
- **Cons:** Harder to train (loss weighting), more engineering, may underperform pure DL where data is dense; less mature tooling.
- **Benchmark:** AirPhyNet and PINN advection-diffusion models report competitive RMSE with much better extrapolation/data-efficiency than black-box baselines.
- **Recommendation:** High-impact differentiator if time allows. Add a light advection-diffusion residual loss (ERA5/IMDAA winds) to ConvLSTM for IGP transport realism and HCHO plume transport (Objective-2).

### Residual / Ensemble / Stacked deep models (ResNet, stacking, MoE)

- **Category:** Meta / robustness
- **Summary:** Residual connections stabilize deep estimators; ensembles/stacking blend complementary models (CNN-LSTM + GNN + trees) for variance reduction and uncertainty. In India, stacking ensembles of RF/ExtraTrees/LightGBM are strong, hard-to-beat baselines; a deep stack with a tree meta-learner is a pragmatic winner.
- **Inputs:** Per-base-model inputs; meta-learner consumes base predictions + key covariates.
- **Outputs:** Blended concentration/AQI + ensemble-spread uncertainty.
- **Libraries:** `scikit-learn (StackingRegressor)`, `LightGBM`, `XGBoost`, `PyTorch (ResNet)`
- **Complexity:** Additive (cost of all bases); simple meta-learner.
- **Pros:** Best leaderboard robustness; uncertainty via spread; tree baselines very strong on tabular satellite+met features in India.
- **Cons:** Heavier to train/serve; less elegant; diminishing returns if bases correlated.
- **Benchmark:** IGP MERRA-2 stacking (Delhi/Kanpur/Lucknow/Patna): R2=0.79-0.82, RMSE 27-31 ug/m3. IGP RF satellite+met+landuse: R2=0.87, 1km daily.
- **Recommendation:** Always include a LightGBM/RF tabular baseline AND a final stacking ensemble (ConvLSTM + GNN + LightGBM). Safest path to top RMSE/R/MAE on CPCB validation.

### HCHO hotspot detection (Getis-Ord Gi*, LISA, DBSCAN + fire-HCHO correlation + back-trajectory transport)

- **Category:** Objective-2: statistical/clustering + transport
- **Summary:** For HCHO biomass-burning hotspots: Getis-Ord Gi* and Local Moran's I/LISA flag significant HCHO clusters; DBSCAN/ST-DBSCAN clusters fire pixels; pixelwise/lagged Pearson correlation links FIRMS fires to TROPOMI HCHO enhancement; HYSPLIT back/forward trajectories + ERA5 winds attribute and track plume transport into the IGP. Optional autoencoder for HCHO anomaly detection.
- **Inputs:** TROPOMI HCHO L3 (gap-filled), MODIS/VIIRS FIRMS active-fire counts/FRP, ERA5/IMDAA U/V winds, seasonal (Oct-Nov, Mar-May) masks.
- **Outputs:** Significant-hotspot polygons/grids, fire-HCHO correlation maps with lag, transport vectors/trajectories, hotspot time series.
- **Libraries:** `PySAL/esda (Gi*, Moran)`, `scikit-learn (DBSCAN)`, `scipy`, `HYSPLIT / pysplit`, `Google Earth Engine`
- **Complexity:** Low-moderate; geostatistics + trajectory model; no heavy GPU.
- **Pros:** Interpretable, validated methods, directly matches PS Objective-2; robust without large training sets.
- **Cons:** Requires HCHO gap-filling first; correlation != causation (biogenic VOC/temperature confounding); trajectory accuracy bounded by wind reanalysis.
- **Benchmark:** Established synergy: HCHO + fire counts + surface temperature quantifies biomass-burning emissions; Odisha/Chhattisgarh forest belts confirmed HCHO hotspots over India.
- **Recommendation:** Getis-Ord Gi* + ST-DBSCAN for hotspots, lagged fire-HCHO correlation, HYSPLIT/ERA5 transport. Gap-fill HCHO with U-Net first; restrict to burning seasons over IGP and central-India forest zones.

**SOTA references:**

- CNN-LSTM PM2.5 (R2=0.91, RMSE=8.216): pmc.ncbi.nlm.nih.gov/articles/PMC11313410/
- Deep learning ground-level NO2 (Ghahremanloo 2021, CV-R=0.91): agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JD034925
- SLNet two-stage 1km NO2 impute+estimate (R2=0.887-0.919): sciencedirect.com/science/article/pii/S0034425724003390
- IGP high-res PM2.5 fusion satellite+met+land-use (RF, R2=0.87, 1km): pubs.acs.org/doi/10.1021/acs.est.0c01769
- Lucknow hybrid RT-RF-CNN sub-km PM2.5 (R2=0.90, RMSE=26.9): sciencedirect.com/science/article/abs/pii/S1352231024004734
- IGP MERRA-2 + stacking ensemble PM2.5 (R2=0.79-0.82): nature.com/articles/s41598-026-37934-9
- AirPhyNet physics-guided neural ODE: arxiv.org/pdf/2402.03784
- TFT for AQI forecasting: ijetjournal.org/temporal-fusion-transformer-air-quality/ ; SpatioTemporal-Informer PM2.5: pmc.ncbi.nlm.nih.gov/articles/PMC10289464/
- ST-GNN dynamic wind-graph PM2.5 (MSDGNN/SA-GNN/STGATN): sciencedirect.com/science/article/pii/S1364815225000350
- GEE Sentinel-5P HCHO asset COPERNICUS/S5P/OFFL/L3_HCHO: developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_HCHO

---

## Classical ML and geostatistical baselines for satellite-to-surface AQI (AOD/TROPOMI columns -> ground PM2.5/AQI) over India, BAH 2026 PS3

*Pipeline stage: **ML Prediction** · 14 methods.*

**Recommended stack:**

- **Base learners (diverse): LightGBM + XGBoost (primary), Random Forest + Extra Trees (bagged diversity), day-specific Linear Mixed-Effects (Hu/Liu/Dey calibration), GTWR or MGWR (interpretable spatiotemporal), CNN-LSTM/CNN-LSTM-attention DL (spatial context + temporal dynamics, required by PS3).**
- **Meta/blender: Gaussian Process Regression with anisotropic spatial smoothing as the final stacker over base-learner out-of-fold predictions (mirrors SOTA India national 1km product, daily R2=0.86) — yields per-pixel uncertainty maps.**
- **Geostatistical residual layer: Regression Kriging (GBM trend + Ordinary Kriging of residuals); Co-Kriging with dense AOD where CPCB stations sparse; IDW only as emergency gap-fill.**
- **Gap-filling/fusion: fill TROPOMI/INSAT cloud & swath gaps via kriging/GP + reanalysis (ERA5/IMDAA/MERRA-2) covariates; LME/GTWR daily calibration handles missing-AOD days; multi-source columns cross-verify (NO2/SO2/CO/O3/HCHO fill what AOD misses).**
- **Validation: spatial-block + temporal-block (and leave-one-station-out) grouped CV to avoid spatial-temporal autocorrelation leakage; report RMSE/R/MAE per PS3, plus winter-IGP-stratified and rural-vs-urban metrics.**
- **Interpretation: SHAP (LightGBM) for driver attribution; GWR/GTWR coefficient surfaces and GAM partial-effect curves for the AOD-PM2.5-met response story and HCHO hotspot drivers.**
- **Libraries: scikit-learn, lightgbm, xgboost, catboost, SHAP, optuna, mgwr (GWR/MGWR), pykrige + gstools + scikit-gstat, statsmodels/pymer4 (LME), pyGAM, GPyTorch/GPflow, R-INLA/PyMC, PyTorch/TF (CNN-LSTM).**

**Key findings:**

- Gradient boosting (LightGBM/XGBoost) is the strongest single classical learner for AOD->PM2.5 (Asia/India R2 0.85-0.92), typically edging Random Forest (India R2 0.71-0.87); both saturate/bias-low at extreme winter-IGP haze and cannot extrapolate beyond training range.
- The SOTA India national product (daily 1km, 2008-2020, Gupta/Dey et al., PNAS Nexus 2024) is itself a STACKED ENSEMBLE blended by Gaussian Process regression with anisotropic smoothing, daily-validation R2=0.86 — directly validating the recommended tree-models + GP-stacker architecture for PS3.
- The Hu/Liu/Lee day-specific Linear Mixed-Effects calibration (random daily intercept+slope) is the canonical cheap robust satellite-PM2.5 baseline, proven over the Indian subcontinent (3km MODIS AOD explains ~83% of PM2.5 variance; Mhawish/Dey 2020).
- Spatial non-stationarity matters: GWR/MGWR captures region-varying AOD-PM2.5 relations (China CV R2 0.77->0.87 after adding NO2+EVI); GTWR adds temporal variation (beats GWR/SLR) and is the right interpretable model for the DAILY AQI objective.
- Newest frontier merges both worlds: spatiotemporally-weighted tree-based algorithms (GTWR weighting + gradient boosting) reach site-CV R2 ~0.85-0.90 with better interpretability (npj Clim Atmos Sci 2024) — a high-value recommended member.
- Geostatistics is best used as a residual/gap-filling layer, not a primary predictor: Regression Kriging (GBM trend + OK residuals) and Co-Kriging with dense AOD add accuracy and uncertainty; IDW/OK alone fail where the CPCB network is sparse (rural/NE India).
- Feature-importance is consistent across India studies: AOD, RH, PBLH/boundary-layer height, temperature, wind speed, and NO2/EVI dominate; SHAP on LightGBM is the practical tool for driver attribution and HCHO-fire hotspot interpretation.
- CV design is the dominant correctness risk: naive random CV leaks via spatial+temporal autocorrelation and inflates R2 — use spatial-block + temporal-block + leave-one-station-out grouped CV, and report RMSE/R/MAE stratified by season (winter IGP) and urban/rural.

### Random Forest (RF)

- **Category:** Tree ensemble (bagging)
- **Summary:** Bagged decision trees on AOD + TROPOMI NO2/SO2/CO/O3/HCHO + ERA5/MERRA-2 met + land-use + spatial/temporal coords. Robust, non-parametric, handles non-linear AOD-PM2.5 hygroscopic/PBL effects. OOB gives free CV. Saturates at high PM2.5 (winter IGP haze) and cannot extrapolate beyond training range — biases extremes low.
- **Inputs:** AOD, column trace gases, RH/T/wind/PBLH, DEM, NDVI/EVI, road/pop density, DOY/lat/lon
- **Outputs:** Point concentration; permutation & impurity importance; OOB R2
- **Libraries:** `scikit-learn RandomForestRegressor`, `ranger (R)`, `cuML RF (GPU)`
- **Complexity:** O(n_trees * n*log n). Parallel, fast inference, low tuning burden.
- **Pros:** Strong baseline, minimal preprocessing, handles collinearity, OOB CV, interpretable importances
- **Cons:** Saturates at extremes (winter IGP), no spatial autocorrelation, no native uncertainty, memory-heavy
- **Benchmark:** India: R2 0.71-0.87 (Maharashtra R2=0.87 RMSE=12.6; NW India R2=0.71 RMSE=31.6; MERRA-2 recon R2=0.86). Asia ~0.80-0.90.
- **Recommendation:** Core ensemble member and sanity baseline. Use spatial-block + day-grouped CV; quantile RF for intervals.

### Gradient Boosting (XGBoost / LightGBM / CatBoost)

- **Category:** Tree ensemble (boosting)
- **Summary:** Sequential boosted trees — often the best single classical model for AOD->PM2.5. LightGBM (leaf-wise histogram) scales to national 1km; XGBoost most battle-tested; CatBoost best with categorical (state/land-class) + ordered boosting. Captures sharp non-linear interactions (AOD x RH x PBLH).
- **Inputs:** Same feature stack as RF; benefits from engineered spatial/temporal & lag features
- **Outputs:** Point estimate; SHAP importance/interactions; quantile (pinball) intervals
- **Libraries:** `xgboost`, `lightgbm`, `catboost`, `SHAP`, `optuna`
- **Complexity:** LightGBM very fast (histogram bins); needs tuning (depth, lr, reg, early stopping).
- **Pros:** Usually best classical accuracy, native quantile/SHAP, monotonic constraints, fast inference, handles missing internally
- **Cons:** Overfits without early stopping, poor extrapolation, many hyperparameters, leaks if CV not blocked
- **Benchmark:** India/Asia: R2 0.85-0.92, frequently top single learner; multisource RS XGBoost ~0.90 (beats RF).
- **Recommendation:** Primary classical learner and top stacking member. SHAP for hotspot drivers; monotonic constraint on AOD.

### Extra Trees (Extremely Randomized Trees)

- **Category:** Tree ensemble (bagging)
- **Summary:** Like RF but random split thresholds and full-sample (no bootstrap) -> lower variance, faster, slightly higher bias. Near-ties RF; valuable as a decorrelated diversity member in a stack.
- **Inputs:** Same feature stack as RF
- **Outputs:** Point estimate, impurity importance
- **Libraries:** `scikit-learn ExtraTreesRegressor`
- **Complexity:** Faster training than RF (no threshold search); same inference cost.
- **Pros:** Fast, low variance, good ensemble diversity, fewer overfit issues
- **Cons:** Same extrapolation/saturation limits as RF, marginal standalone gain
- **Benchmark:** Asia PM2.5: R2 ~0.80-0.88, comparable to RF; rarely top model but cheap diversity.
- **Recommendation:** Low-cost diversity member in the stack; not a standalone deliverable.

### Support Vector Regression (SVR)

- **Category:** Kernel regression
- **Summary:** RBF-kernel epsilon-insensitive regression. Models non-linear AOD-PM2.5 but scales O(n^2-n^3), sensitive to scaling & C/gamma/epsilon. Generally beaten by tree ensembles on tabular RS data; mostly historical baseline for India.
- **Inputs:** Standardized feature stack; PCA often pre-applied
- **Outputs:** Point estimate only (no native uncertainty)
- **Libraries:** `scikit-learn SVR/LinearSVR`, `ThunderSVM (GPU)`, `Nystroem+SGDRegressor`
- **Complexity:** Training O(n^2)-O(n^3); infeasible at national scale without subsampling/Nystroem.
- **Pros:** Effective high-dim/small-n, robust to outliers via epsilon tube
- **Cons:** Poor scaling, heavy tuning, needs scaling, no uncertainty, usually lower R2 than trees
- **Benchmark:** India/Asia: R2 0.6-0.8, typically below RF/GBM; competitive only on small clean datasets.
- **Recommendation:** Optional weak baseline only; deprioritize for a national 1km product.

### Gaussian Process Regression (GPR / Kriging-equivalent)

- **Category:** Bayesian kernel / geostatistical
- **Summary:** Non-parametric Bayesian regression giving predictive mean AND calibrated uncertainty — kriging with a learned covariance. Ideal ENSEMBLE STACKER with anisotropic spatial smoothing (exactly the Indian national 1km product). Naive O(n^3); use sparse/inducing-point or per-day local GP at scale.
- **Inputs:** Base-learner predictions + lat/lon/time for residual smoothing; or full feature stack
- **Outputs:** Predictive mean + variance (uncertainty maps), kernel length-scales
- **Libraries:** `GPyTorch`, `GPflow`, `scikit-learn GaussianProcessRegressor`
- **Complexity:** Exact O(n^3), mem O(n^2). Use SVGP/SGPR/local-GP; GPyTorch/GPflow scale on GPU.
- **Pros:** Calibrated uncertainty (AQI confidence), natural spatial smoothing, principled ensembling/residual kriging
- **Cons:** Cubic scaling, kernel/likelihood sensitive, needs sparse approx at scale
- **Benchmark:** India national ensemble: GP stacker w/ anisotropic smoothing -> daily R2=0.86 (Gupta/Dey, PNAS Nexus 2024). China Bayesian-GP improves satellite PM2.5.
- **Recommendation:** Final ensemble blender with anisotropic spatial smoothing (mirrors SOTA India product); produces per-pixel uncertainty.

### Ordinary / Universal Kriging

- **Category:** Geostatistics
- **Summary:** Interpolates station residuals via fitted variogram (OK constant mean; UK trend on covariates). Best for filling spatial gaps between sparse CPCB stations and cloud-gap residual interpolation, not primary satellite-driven prediction.
- **Inputs:** Station residuals/values + coords (UK adds AOD/met trend covariates)
- **Outputs:** Interpolated surface + kriging variance
- **Libraries:** `pykrige (Ordinary/UniversalKriging)`, `gstools`, `scikit-gstat`
- **Complexity:** Variogram fit + O(n^3) neighborhood solve; local/moving-window kriging scales.
- **Pros:** BLUP optimality, kriging-variance uncertainty, gap-filling between stations
- **Cons:** Assumes stationarity/isotropy, fails where CPCB sparse (rural/NE India), poor for non-linear drivers
- **Benchmark:** Standalone OK weak where stations sparse; as residual-kriging stage adds +0.02-0.05 R2 over base model in Asia.
- **Recommendation:** Residual-interpolation stage (Regression Kriging) on tree models; fill spatial gaps over IGP network.

### Regression Kriging (RK) / Co-Kriging

- **Category:** Geostatistics (hybrid)
- **Summary:** Two-stage: regress PM2.5 on covariates (GLM/RF/GBM), then krige residuals for leftover spatial autocorrelation. Co-Kriging exploits densely-sampled AOD to improve sparsely-sampled PM2.5. Most practical geostatistical hybrid for India.
- **Inputs:** Covariate stack for trend + station residuals + coords; co-kriging adds AOD field
- **Outputs:** Surface + uncertainty combining deterministic trend & stochastic residual
- **Libraries:** `pykrige (RegressionKriging)`, `gstools`, `scikit-learn (trend)`
- **Complexity:** Regression + kriging cost; moderate. Co-kriging adds cross-variogram fitting.
- **Pros:** Captures trend + autocorrelation, leverages dense AOD via co-kriging, uncertainty
- **Cons:** Residual stationarity assumed, two-stage tuning, cross-variogram fiddly
- **Benchmark:** Asia RK typically R2 0.80-0.90; consistently beats pure regression or pure kriging.
- **Recommendation:** Strong classical pipeline: GBM trend + OK residual kriging; co-krige with dense AOD where stations sparse.

### Inverse Distance Weighting (IDW)

- **Category:** Deterministic interpolation
- **Summary:** Distance-weighted average of nearby stations. Simple, fast, no fit, but no uncertainty, sensitive to power p, poor where CPCB network sparse/uneven (most of rural India). Baseline/gap-fill only.
- **Inputs:** Station coords + values + power p
- **Outputs:** Interpolated surface (no uncertainty)
- **Libraries:** `scipy.interpolate`, `pykrige (IDW)`, `gdal_grid`
- **Complexity:** O(n) per query point; trivial.
- **Pros:** Trivial, fast, no assumptions, quick gap-fill
- **Cons:** No uncertainty, bullseye artifacts, ignores covariates & anisotropy, poor with sparse network
- **Benchmark:** Lowest tier; naive baseline. Underperforms kriging and ML substantially over India.
- **Recommendation:** Naive baseline / emergency gap-filler only; not a deliverable model.

### Geographically Weighted Regression (GWR)

- **Category:** Spatial regression
- **Summary:** Local regression with spatially-varying coefficients (distance-kernel weighted) — captures spatially non-stationary AOD-PM2.5 relations across India's regimes (IGP vs coastal vs arid). Foundational satellite-PM2.5 method; adding NO2/EVI lifts CV R2 markedly.
- **Inputs:** AOD, met, NO2, EVI + coords; bandwidth by AICc/CV
- **Outputs:** Spatially-varying coefficient maps + local R2
- **Libraries:** `mgwr (GWR/MGWR, Python)`, `GWmodel (R)`, `spgwr (R)`
- **Complexity:** O(n^2) bandwidth + local fits; mgwr handles moderate n; slower for daily national.
- **Pros:** Models spatial non-stationarity, interpretable coefficient surfaces, well-established
- **Cons:** Assumes temporal stationarity (use GTWR for daily), local collinearity, slower, limited non-linearity
- **Benchmark:** China GWR CV R2 0.77->0.87 after adding NO2+EVI; India/Asia GWR commonly 0.80-0.87 monthly/annual.
- **Recommendation:** Interpretable spatial member + coefficient-map storytelling; prefer GTWR for daily AQI objective.

### Geographically & Temporally Weighted Regression (GTWR)

- **Category:** Spatiotemporal regression
- **Summary:** Extends GWR with a spatiotemporal kernel so coefficients vary in space AND time — directly suited to DAILY surface-AQI mapping. Beats SLR and GWR; neural (GTWNN) and spatiotemporally-weighted-tree variants push further. Strong interpretable spatiotemporal member.
- **Inputs:** AOD + met + trace gases + space-time coords; spatial & temporal bandwidths
- **Outputs:** Space-time varying coefficients, local/temporal R2
- **Libraries:** `GWmodel (R) gtwr`, `custom spatiotemporal kernel in Python`
- **Complexity:** Higher than GWR (temporal bandwidth); national daily needs tiling/subsampling.
- **Pros:** Captures daily-varying AOD-PM2.5 relation (like LME), interpretable, beats GWR
- **Cons:** Compute-heavy at national daily scale, bandwidth tuning, limited non-linearity vs trees
- **Benchmark:** GTWR > GWR > SLR consistently. GTWNN site-CV R2 ~0.79-0.80; ST-weighted trees site-CV R2 ~0.85-0.90 (npj 2024).
- **Recommendation:** Key interpretable daily member; or use GTWR weighting to build spatiotemporally-weighted LightGBM (current SOTA flavor).

### Land Use Regression (LUR)

- **Category:** Empirical regression
- **Summary:** Linear regression of pollutant on buffer-aggregated land-use/traffic/emission/satellite predictors. Excellent for LONG-TERM annual NO2/PM exposure surfaces (national India satellite-LUR NO2 exists). Weak for daily dynamics; pair with universal kriging of residuals.
- **Inputs:** Road density, pop, land cover, point-source/emission, satellite NO2/AOD in multi-radius buffers
- **Outputs:** Long-term mean surface; coefficient set; (with UK) uncertainty
- **Libraries:** `scikit-learn (LASSO/OLS)`, `statsmodels`, `geopandas/rasterstats`, `pykrige UK`
- **Complexity:** Cheap (OLS/LASSO) once buffers computed; GIS buffer extraction is main cost.
- **Pros:** Interpretable, cheap, great for annual exposure & HCHO source proxies, strong for NO2
- **Cons:** Static (poor daily), overfits with many buffers, residual spatial autocorr
- **Benchmark:** India national satellite-LUR NO2 annual model published; LUR+UK (China national) strong annual R2 ~0.8+. Daily LUR weak.
- **Recommendation:** Long-term/annual AQI layer + HCHO source-apportionment covariates; combine LUR + universal kriging.

### Generalized Additive Models (GAM)

- **Category:** Semi-parametric regression
- **Summary:** Sum of smooth spline functions s(AOD)+s(RH)+s(PBLH)+spatial tensor smooth — captures non-linearity while interpretable, giving response curves. Middle ground between linear and black-box trees; supports spatial/temporal smooths.
- **Inputs:** AOD, met, trace gases, spatial tensor (lat,lon), temporal smooth (DOY)
- **Outputs:** Smooth partial-effect curves + surface; EDF/significance
- **Libraries:** `pyGAM`, `mgcv (R)`, `statsmodels GLMGam`
- **Complexity:** Penalized spline fit, moderate; pyGAM/mgcv scale via thin-plate/tensor smooths.
- **Pros:** Interpretable non-linear curves, spatial smooth, uncertainty bands
- **Cons:** Lower accuracy than boosting, smooth/knot tuning, additive (limited interactions)
- **Benchmark:** Asia PM2.5 GAM R2 ~0.7-0.85; below GBM but far more interpretable response curves.
- **Recommendation:** Interpretable AOD-PM2.5 response curves and smooth member; good for explaining drivers to judges.

### Linear Mixed-Effects Model (LME — Hu/Liu/Lee daily calibration)

- **Category:** Hierarchical linear model
- **Summary:** Classic two-stage satellite-PM2.5 method: PM2.5 ~ AOD + met with DAY-SPECIFIC random intercepts and slopes, recalibrating AOD-PM2.5 every day. Simple, robust, fast; established benchmark over Indian subcontinent (Mhawish/Dey). Often coupled with RF for residuals.
- **Inputs:** AOD + met covariates; grouping = day (and/or site/month) random effects
- **Outputs:** Fixed + daily random coefficients; fitted PM2.5; CV R2
- **Libraries:** `statsmodels MixedLM`, `lme4 (R)`, `pymer4`
- **Complexity:** Cheap REML fit (lme4/statsmodels); scales easily to national daily.
- **Pros:** Captures daily AOD-PM2.5 variability cheaply, interpretable, proven over India
- **Cons:** Linear fixed effects (misses non-linearity), needs AOD present (cloud gaps), normal-RE assumption
- **Benchmark:** 3km MODIS AOD via day-specific LME explains ~83% of PM2.5 variance over Indian subcontinent (2020). LME+RF boosts hourly.
- **Recommendation:** Canonical calibration baseline (Hu/Liu/Dey) and feature/member; couple LME trend + RF residual for fast strong pipeline.

### Bayesian Hierarchical Model

- **Category:** Bayesian spatial/spatiotemporal
- **Summary:** Full probabilistic hierarchy with spatial (CAR/GP) + temporal random effects + measurement-error layers — propagates uncertainty end-to-end, ideal for fusing multi-source satellite columns of differing accuracy and reporting AQI confidence. INLA-SPDE makes large spatiotemporal models tractable.
- **Inputs:** Satellite columns + met + station data with error structure; spatial mesh + time
- **Outputs:** Posterior PM2.5 fields + full predictive uncertainty + parameter posteriors
- **Libraries:** `R-INLA (SPDE)`, `PyMC`, `NumPyro`, `cmdstanpy`
- **Complexity:** MCMC heavy; INLA/SPDE fast deterministic approx scales to national grids.
- **Pros:** Rigorous uncertainty, handles measurement error & data fusion, principled missing-data, spatiotemporal
- **Cons:** Computationally expensive, modeling/prior expertise, slower iteration
- **Benchmark:** China Bayesian-hierarchical GP improves satellite PM2.5 with calibrated uncertainty; INLA-SPDE common for national AQ.
- **Recommendation:** INLA-SPDE for uncertainty-aware data-fusion variant + AQI confidence bands; advanced/stretch member.

**SOTA references:**

- Gupta, Dey et al. (2024), Nationwide daily ambient PM2.5 2008-2020 at 1 km2 in India via ensemble (GP-regression stacker, daily R2=0.86), PNAS Nexus 3(3):pgae088 — India SOTA baseline.
- Mhawish, Dey et al. (2020), Spatiotemporal mixed-effects PM2.5 from MODIS AOD over Indian subcontinent, GIScience & Remote Sensing — day-specific LME, ~R2 0.83.
- npj Climate & Atmospheric Science (2024), Deriving PM2.5 with spatiotemporally weighted tree-based algorithms — GTWR-weighted boosting, site-CV R2 ~0.85-0.90.
- You et al. (2016), GTWR Model for Ground-Level PM2.5 from 500 m AOD, Remote Sensing 8(3):262 — GTWR > GWR > SLR.
- Hu et al. (2014, EST) two-stage land-use/met model & Lee et al. (2011) day-specific LME — foundational daily calibration framework.
- Bi et al. (2017), Improving satellite PM2.5 in China using Gaussian processes in a Bayesian hierarchical setting — GP/Bayesian uncertainty.
- Maharashtra ML study (Discover Sustainability, 2025) — RF R2=0.87, RMSE=12.6, MAE=6.96 over India.
- India national satellite Land-Use-Regression NO2 model (Sci Total Environ, 2024) — national annual NO2 LUR.
- Ma/You et al. — China GWR with NO2+EVI, cross-val R2 0.77->0.87.
- Dey SAANS / CPCB satellite 1km PM2.5 reference platform — validation/benchmark resource for India.

---

## Statistical & clustering HOTSPOT-detection methods for Sentinel-5P HCHO spatio-temporal mapping over India (BAH 2026 PS3 Objective 2: biomass-burning-season HCHO hotspots over the Indo-Gangetic Plain & forest-fire zones)

*Pipeline stage: **Hotspot Detection** · 12 methods.*

**Recommended stack:**

- **Data/IO: Google Earth Engine (COPERNICUS/S5P/OFFL/L3_HCHO band tropospheric_HCHO_column_number_density, ~0.01 deg L3 grid, daily, regrid to 0.05-0.1 deg over India) + xarray/rioxarray + geopandas**
- **Spatial weights: libpysal.weights (Queen contiguity for regular grid, or KNN-8 / DistanceBand) — tune band, document MAUP sensitivity**
- **Primary hotspot: esda.G_Local (Gi*, star=True, permutations=999) + esda.fdr for multiple-comparison control**
- **Cluster/outlier: esda.Moran_Local (LISA HH/LL/HL/LH) + esda.Moran (global) + splot.esda for plots**
- **Temporal: pymannkendall (hamed_rao_modification_test for autocorrelation, seasonal_test, sens_slope) over space-time cube -> EHSA categories**
- **Climatology votes: numpy/xarray per-cell seasonal 95th-percentile exceedance + robust z-anomaly (MAD)**
- **Delineation: hdbscan (haversine) / sklearn.cluster.DBSCAN + shapely/alphashape for hotspot polygons; st_dbscan for multi-day plume episodes**
- **Supporting: eofs/xeofs (mode separation), scipy.stats.gaussian_kde (intensity surface), optional SaTScan/smerc Kulldorff space-time**
- **Attribution: MODIS/VIIRS FIRMS fire counts + ERA5/IMDAA winds -> lagged fire-HCHO correlation (scipy.stats) + ST-DBSCAN episode + wind transport overlay**
- **Compute: vectorized xarray.apply_ufunc / dask for pixelwise MK & Gi*-per-bin**

**Key findings:**

- Consensus pipeline: a cell is a CONFIRMED HCHO hotspot if it passes >=2 of 3 votes — (1) Gi* significant high (FDR p<0.01, star=True, 999 perms), (2) LISA HH (p<0.05), (3) per-cell seasonal 95th-pct / robust z>2 anomaly. Fuses inferential + climatological evidence; suppresses single-method artifacts.
- Gi* is the canonical, reviewer-trusted detector (mirrors ArcGIS Hot Spot Analysis); always run star=True with permutation p_sim + FDR because India-wide grids cause severe multiple-comparison inflation and non-stationarity vs a single global mean.
- Emerging Hot Spot Analysis (Gi* per season-year + Mann-Kendall per cell) is the headline temporal deliverable: separates Persistent/Historical IGP industrial hotspots from New/Intensifying episodic biomass-burning hotspots. No turnkey OSS EHSA — compose esda.G_Local + pymannkendall.
- Use MODIFIED Mann-Kendall (Hamed-Rao) not the original test — atmospheric series are autocorrelated and raw MK over-rejects (false trends); pair with Sen's slope for robust magnitude (mol/m2 per season).
- Standardize first: per-season robust z-anomalies (median/MAD) remove India's strong spatial gradient + seasonal cycle BEFORE Gi*/LISA so one threshold is comparable nationwide and fire plumes don't inflate variance.
- Density clustering (HDBSCAN/DBSCAN, ST-DBSCAN) are DELINEATION/episode tools, not significance tests — apply to the already-significant exceedance mask to extract polygons and multi-day episodes; HDBSCAN haversine handles variable plume density better than DBSCAN's single eps.
- For fire attribution/transport: ST-DBSCAN clusters daily exceedances into multi-day episodes; overlay FIRMS MODIS/VIIRS fires + ERA5/IMDAA winds; compute lagged fire-HCHO correlation and wind back-trajectories to attribute downwind hotspots to upwind burning.
- K-means/GMM, KDE, EOF/PCA, Kulldorff scan are SECONDARY: GMM for multivariate typing (fire vs industrial), KDE for presentation surfaces, EOF to isolate the burning mode and correlate its PC with fires, Kulldorff space-time as optional rigorous 'most-likely cluster' cross-check.

### Getis-Ord Gi* (local hot/cold-spot z-score)

- **Category:** Spatial autocorrelation / canonical hotspot statistic
- **Summary:** Per-cell z-score of local weighted HCHO sum vs global mean. High +z + low p = significant hotspot (high-among-high); negative z = coldspot. star=True includes focal cell (Gi*, standard). Permutation p_sim avoids normality. The most defensible 'hotspot' definition for the PS.
- **Inputs:** Gridded HCHO (daily/seasonal-mean raster -> GeoDataFrame centroids) + spatial weights W (Queen/KNN/DistanceBand). Run per time slice or on the seasonal climatology.
- **Outputs:** Per-cell Gi* z, p (analytic or p_sim), HH/cold class; threshold |z|>1.96/2.58, FDR-corrected. Dissolve significant high cells -> hotspot polygons.
- **Libraries:** `esda.G_Local`, `esda.fdr`, `libpysal.weights (Queen/KNN/DistanceBand)`, `geopandas`, `rasterio/rioxarray`, `numpy`
- **Complexity:** O(n) with sparse W; x n_perm (e.g. 999) for permutations. Fast at India 0.05-0.1 deg grid.
- **Pros:** Standard, interpretable z/p; hot vs cold; direct significance; mirrors ArcGIS Hot Spot Analysis so reviewers trust it.
- **Cons:** Sensitive to W & band distance (MAUP); global-mean reference skewed by fire plumes; multiple-comparison inflation needs FDR; assumes single global mean (non-stationarity).
- **Benchmark:** Used for India SO2/NO2 TROPOMI hotspots (Nature Sci Rep 2024) & Madhya Pradesh multi-pollutant mapping (ScienceDirect 2026); |z|>1.96/2.58.
- **Recommendation:** PRIMARY hotspot engine. Gi* star=True, 999 perms, KNN-8/DistanceBand W, FDR p, classify >99% as core. Run per season (Oct-Nov, Apr-May).

### Local Moran's I / LISA (cluster & outlier)

- **Category:** Spatial autocorrelation / cluster-outlier
- **Summary:** Decomposes global Moran's I per cell, classifying significant cells as HH (hotspot core), LL (coldspot), HL (high outlier = isolated fire plume), LH (low in high). Complements Gi* by flagging spatial OUTLIERS that Gi* smooths over.
- **Inputs:** Same gridded HCHO + W as Gi*. Best on standardized/anomaly field so HH/LL reference the local mean.
- **Outputs:** Per-cell Ii, z, p_sim, quadrant q (1=HH,2=LH,3=LL,4=HL). HH = consensus hotspots; HL = candidate point-source/fresh-fire anomalies.
- **Libraries:** `esda.Moran_Local`, `esda.Moran (global)`, `splot.esda (lisa_cluster, moran_scatterplot)`, `libpysal.weights`, `geopandas`
- **Complexity:** O(n) sparse + permutations; same scale as Gi*.
- **Pros:** Adds outlier semantics (HL fire spikes; LH gaps); pinpoints transition zones; widely accepted; pairs with Gi* for consensus.
- **Cons:** HH boundaries differ slightly from Gi* (cross-product vs sum); p needs FDR; conditional permutation noisy at low autocorrelation; HL/LH interpretation care.
- **Benchmark:** LISA p<0.05, 999 perms standard; ClusterRadar (arXiv 2024) and AQ/crime studies use Gi*+LISA jointly.
- **Recommendation:** Run alongside Gi*. HH quadrant = hotspot consensus vote; report HL cells as 'fresh fire / point-source anomalies' for transport analysis.

### Emerging Hot Spot Analysis (EHSA: Gi* + Mann-Kendall)

- **Category:** Spatio-temporal trend + hotspot
- **Summary:** Space-time cube (cell x time-bin) -> Gi* per cell per bin -> Mann-Kendall on each cell's Gi* series -> labels New/Consecutive/Intensifying/Persistent/Diminishing/Sporadic/Oscillating/Historical hot/cold spot. Answers 'where are HCHO hotspots emerging/intensifying'.
- **Inputs:** Multi-year stack of seasonal/monthly HCHO grids -> space-time cube (n_cells x n_periods) + spatial W + temporal neighbor def.
- **Outputs:** Per-cell EHSA category + MK trend (tau,p) + hotspot frequency. Maps of intensifying vs new vs persistent HCHO hotspots over IGP.
- **Libraries:** `esda.G_Local (loop over bins)`, `pymannkendall (original_test, hamed_rao_modification_test, sens_slope)`, `xarray (space-time cube)`, `numpy`
- **Complexity:** Gi* per bin (n_cells x n_bins) + MK per cell; moderate, fully vectorizable.
- **Pros:** Best single product for PS narrative; separates chronic IGP industrial from episodic fire-season hotspots; ArcGIS-recognized.
- **Cons:** No turnkey OSS EHSA (compose Gi*+pymannkendall); needs >=8-10 time bins; MK assumes monotonic & independence -> use modified MK.
- **Benchmark:** ESRI EHSA categorization standard; modified MK (Hamed-Rao) for autocorrelated environmental series.
- **Recommendation:** HEADLINE deliverable. Compose Gi* (per season-year) + pymannkendall.hamed_rao per cell. Categorize New/Intensifying/Persistent + Sen's slope.

### Mann-Kendall trend + Sen's slope

- **Category:** Temporal trend (non-parametric)
- **Summary:** Per-cell non-parametric monotonic trend test (no normality, robust to outliers/gaps) giving direction & significance; Sen's slope gives robust magnitude (HCHO change per season). Underpins EHSA and stand-alone trend maps.
- **Inputs:** Per-cell HCHO series (monthly/seasonal means). Modified variants for serial autocorrelation; seasonal MK for seasonality.
- **Outputs:** Per-cell trend (inc/dec/none), tau, p, Sen's slope (mol/m2 per year). Pixel-wise trend raster.
- **Libraries:** `pymannkendall (original_test, seasonal_test, hamed_rao_modification_test, yue_wang_modification_test, sens_slope)`, `xarray.apply_ufunc`, `scipy.stats.kendalltau/theilslopes`
- **Complexity:** O(T^2) per cell for slope but cheap; pixel-parallel.
- **Pros:** Robust to non-normal, outlier-heavy HCHO and gaps; standard in RS trend studies; modified/seasonal variants available.
- **Cons:** Monotonic only (misses regime shifts); raw MK over-rejects under autocorrelation -> use Hamed-Rao/Yue-Wang; needs adequate length.
- **Benchmark:** Punjab/Sheikhupura crop-residue-burning GEE study (Springer 2025) uses MK+Sen on TROPOMI; p<0.05.
- **Recommendation:** Use modified MK (Hamed-Rao) + Sen's slope pixelwise for trend maps and as the temporal engine inside EHSA.

### Percentile / threshold (>90th/95th of climatology)

- **Category:** Threshold / climatology
- **Summary:** Flag cells/days where HCHO exceeds a high percentile (90/95/98th) of the local long-term climatology (per-cell/per-season). Simple transparent first-pass hotspot/exceedance mask; basis for anomaly consensus.
- **Inputs:** HCHO climatology per cell (multi-year monthly/seasonal distribution) + current field. Per-cell thresholds preferred over national.
- **Outputs:** Binary exceedance mask / exceedance-frequency map; magnitude above threshold. Cheap hotspot-candidate layer.
- **Libraries:** `numpy.percentile/nanpercentile`, `xarray.quantile + groupby('time.season')`, `scipy.stats`
- **Complexity:** O(n) trivial; one numpy.percentile per cell.
- **Pros:** Transparent, fast, no W/assumptions; per-cell climatology handles India's spatial gradient; intuitive for CPCB framing.
- **Cons:** No significance/spatial-context test (noisy pixels pass); arbitrary threshold; not a spatial cluster method alone.
- **Benchmark:** Common climatological-exceedance baseline; percentile thresholds in ESA/ECMWF anomaly products.
- **Recommendation:** Use as one of three consensus votes. Per-cell seasonal 95th-pct exceedance; require spatial support (>= few contiguous cells).

### Standardized anomalies / z-scores

- **Category:** Threshold / climatology
- **Summary:** Per-cell (HCHO - clim mean)/std (per season) -> standardized anomaly field. Removes India's spatial gradient & seasonal cycle so one |z| threshold is comparable everywhere; ideal input to Gi*/LISA and the consensus vote.
- **Inputs:** Per-cell climatological mean & std (seasonal). Robust variant: (x-median)/IQR or modified z (MAD) to resist fire-plume skew.
- **Outputs:** Continuous z-anomaly raster; threshold (z>2) anomaly mask.
- **Libraries:** `xarray (groupby season, mean/std)`, `scipy.stats.zscore`, `numpy (median/MAD)`
- **Complexity:** O(n) trivial.
- **Pros:** Normalizes heterogeneity so Gi*/percentile behave consistently; robust-z resists outliers; strong visualization layer.
- **Cons:** Assumes stable climatology (needs multi-year baseline); Gaussian z misleading for skewed HCHO -> prefer robust z; std inflated by target plumes.
- **Benchmark:** Standard anomaly framework in atmospheric RS; robust-z (MAD) per NIST guidance.
- **Recommendation:** Compute robust per-season z-anomalies; feed as standardized input to Gi* & LISA and as the third consensus vote (z>2).

### DBSCAN / HDBSCAN density clustering

- **Category:** Density clustering (unsupervised)
- **Summary:** Cluster high-HCHO cells in (lat,lon[,value]) space into arbitrary-shaped dense hotspot clusters; auto-labels sparse noise. HDBSCAN removes DBSCAN's single-eps sensitivity & handles variable density (compact urban vs diffuse fire plumes).
- **Inputs:** Point set of exceedance cells (>95th pct or z>2), coords (haversine or projected m); optionally 3rd dim = HCHO value.
- **Outputs:** Cluster labels (hotspot IDs) + noise (-1); polygons via convex/alpha hull; HDBSCAN adds membership probs & persistence.
- **Libraries:** `sklearn.cluster.DBSCAN/HDBSCAN`, `hdbscan (McInnes)`, `sklearn BallTree (haversine)`, `shapely/alphashape`
- **Complexity:** ~O(n log n) with KD/ball tree; fine for India exceedance points.
- **Pros:** Arbitrary-shaped plumes; no preset count; noise rejection; HDBSCAN robust to density variation & near param-free.
- **Cons:** No statistical significance (descriptive); DBSCAN eps very sensitive; ignores time unless extended; depends on upstream threshold.
- **Benchmark:** Standard for point hotspot delineation; HDBSCAN preferred over DBSCAN for geo density variation.
- **Recommendation:** Use to DELINEATE hotspot polygons from the Gi*/consensus mask, not as significance test. HDBSCAN haversine; min_cluster_size tuned to grid.

### ST-DBSCAN (spatiotemporal DBSCAN)

- **Category:** Spatiotemporal density clustering
- **Summary:** DBSCAN with separate spatial (eps1) and temporal (eps2) radii (+ value threshold), clustering high-HCHO exceedances dense in space AND time -> coherent spatio-temporal hotspot episodes (multi-day burning plume tracked across cells).
- **Inputs:** Exceedance events (lat, lon, time, value); eps1 (km), eps2 (days), min_pts, optional density factor.
- **Outputs:** Spatio-temporal cluster IDs = hotspot EPISODES with start/end + footprint; aligns with fire-event/transport analysis.
- **Libraries:** `st_dbscan (PyPI)`, `sklearn (custom metric)`, `geopandas/movingpandas`
- **Complexity:** Similar to DBSCAN with neighbor pruning on both dims; moderate.
- **Pros:** Captures evolving/migrating plumes (links fire -> downwind HCHO over days); natural for burning-season episodes + transport.
- **Cons:** No turnkey maintained sklearn impl (community st_dbscan); 3 sensitive params; descriptive only (no p-values).
- **Benchmark:** Birant & Kut (2007) ST-DBSCAN; used for spatiotemporal pollution/event episode mining.
- **Recommendation:** TRANSPORT/episode layer: cluster daily exceedances into multi-day plume episodes, overlay FIRMS fires + ERA5 winds for attribution.

### K-means / Gaussian Mixture (GMM)

- **Category:** Partitional / model-based clustering
- **Summary:** Partition cells by feature vector (HCHO level, anomaly, trend slope, optional NO2/fire) into K regimes; high-HCHO cluster(s) = hotspot zone. GMM gives soft probabilistic membership & elliptical clusters; good for IGP vs forest-fire regimes.
- **Inputs:** Per-cell feature matrix (mean HCHO, z-anomaly, Sen slope, optional co-pollutants/fire). Standardized; K via silhouette/BIC.
- **Outputs:** Cluster labels / GMM posterior probs per cell; hotspot = high-HCHO cluster; regime/zonation map.
- **Libraries:** `sklearn.cluster.KMeans/MiniBatchKMeans`, `sklearn.mixture.GaussianMixture`, `sklearn.metrics.silhouette_score`, `yellowbrick`
- **Complexity:** K-means O(nKi); GMM EM heavier but fine at India grid.
- **Pros:** Fast regionalization; GMM gives soft probs & handles correlated features; good for multivariate (HCHO+fire+NO2) typing.
- **Cons:** Assumes spherical (K-means)/Gaussian (GMM) clusters; ignores spatial contiguity (salt-and-pepper); pick K; not significance-based; scaling-sensitive.
- **Benchmark:** Standard unsupervised baseline; secondary vs Gi*/KDE in pollutant hotspot literature.
- **Recommendation:** SECONDARY: multivariate hotspot typing (fire vs industrial regimes), not primary detector. Add spatial coords / post-smooth vs salt-and-pepper.

### Kernel Density Estimation (KDE)

- **Category:** Density surface
- **Summary:** Smooth continuous density/intensity surface from high-HCHO points (or value-weighted), revealing hotspot cores as density peaks; standard companion to Gi* in AQ studies for a visually smooth map.
- **Inputs:** Exceedance point set (optionally HCHO-weighted); kernel + bandwidth (fixed or adaptive).
- **Outputs:** Continuous density raster; contour/percentile hotspot polygons (top density quantiles).
- **Libraries:** `scipy.stats.gaussian_kde`, `sklearn.neighbors.KernelDensity`, `seaborn.kdeplot`, `GEE reduceNeighborhood`
- **Complexity:** O(n*m) grid eval; fast with FFT/tree KDE.
- **Pros:** Smooth intuitive maps; bandwidth controls scale; pairs with Gi* for presentation; value-weighting adds intensity.
- **Cons:** No significance; bandwidth-dependent (over/under-smoothing); edge effects; descriptive only.
- **Benchmark:** KDE+Gi* combo standard in TROPOMI India SO2/NO2 hotspot mapping (Nature Sci Rep 2024).
- **Recommendation:** VISUALIZATION/cross-check layer alongside Gi*; not standalone detector. Value-weighted KDE for intensity surfaces.

### EOF / PCA decomposition

- **Category:** Dimensionality reduction / mode extraction
- **Summary:** Decompose HCHO space-time field into orthogonal spatial patterns (EOFs) + temporal amplitudes (PCs); leading EOFs isolate dominant modes (seasonal burning, IGP industrial), high-loading regions of relevant EOFs flag coherent hotspot zones.
- **Inputs:** De-trended/standardized space-time matrix (cells x time). Optionally varimax-rotated EOFs for localized patterns.
- **Outputs:** Spatial EOF maps + PC series + variance explained; hotspot = high-loading region of burning-related EOF.
- **Libraries:** `eofs (eofs.xarray.Eof)`, `xeofs`, `sklearn.decomposition.PCA`, `scipy.linalg.svd`
- **Complexity:** SVD; fine for India grid with reduced rank.
- **Pros:** Separates overlapping drivers (fire/industry/meteorology); denoises; PC series correlate with fire counts for attribution.
- **Cons:** EOFs statistical not physical (orthogonality artifacts, mode mixing); not per-cell significance; needs rotation; gaps filled first.
- **Benchmark:** Standard atmospheric variability-mode tool; used for trace-gas spatiotemporal decomposition.
- **Recommendation:** SUPPORTING: extract dominant modes to disentangle burning vs anthropogenic HCHO; correlate burning-mode PC with FIRMS fires.

### Spatial scan statistic (Kulldorff / SaTScan)

- **Category:** Cluster significance (scan statistic)
- **Summary:** Scans circular/elliptical (and space-time cylindrical) windows of varying size, maximizing likelihood ratio of elevated HCHO vs outside with Monte-Carlo p -> significant, geographically explicit most-likely hotspot cluster(s), spatial & space-time variants.
- **Inputs:** Cell values + expected (or continuous/Normal model) with coords; add time for space-time. Define max cluster size.
- **Outputs:** Ranked significant clusters: center, radius, time window, relative risk, LLR, Monte-Carlo p. Primary + secondary.
- **Libraries:** `SaTScan (external)`, `smerc (R)`, `custom Kulldorff / libpysal-based scan (Python)`
- **Complexity:** Heavier: many windows x Monte-Carlo reps; manageable on aggregated grid.
- **Pros:** Rigorous cluster-level significance with built-in multiple-testing; native space-time clusters; complements cell-wise Gi*; trusted in epidemiology.
- **Cons:** Circular/elliptical shape bias (misses irregular plumes); SaTScan external GUI/CLI; needs Normal model for values; coarser boundaries.
- **Benchmark:** Kulldorff (1997) scan statistic; epidemiology gold standard; Normal model for measurement data.
- **Recommendation:** OPTIONAL rigor layer: Kulldorff space-time (Normal model) for headline 'most-likely significant cluster' + relative risk; cross-validate consensus hotspots.

**SOTA references:**

- esda.G_Local (Gi*) API — pysal.org/esda/generated/esda.G_Local.html (star, permutations, p_sim, z_sim)
- esda.Moran_Local (LISA HH/LL/HL/LH) & esda.Moran — PySAL esda v2.8.x docs
- pyMannKendall (mmhs013) PyPI/GitHub — hamed_rao_modification, seasonal_test, sens_slope (Hussain & Mahmud, JOSS 2019)
- GEE COPERNICUS/S5P/OFFL/L3_HCHO (tropospheric_HCHO_column_number_density, ~0.01 deg L3, daily); NRTI variant COPERNICUS/S5P/NRTI/L3_HCHO
- Getis & Ord (1992) and Ord & Getis (1995) — local Gi/Gi* statistics with distance
- Anselin (1995) — Local Indicators of Spatial Association (LISA), Geographical Analysis
- Kulldorff (1997) spatial scan statistic / SaTScan; smerc (R) for scan
- Nature Scientific Reports 2024 (s41598-024-72276-4) — Sentinel-5P TROPOMI urban AQ hotspot analysis using Gi*+KDE
- ScienceDirect 2026 (S2352938526001734) — multi-pollutant spatiotemporal dynamics & hotspot mapping, Madhya Pradesh India
- Springer 2025 (s42865-025-00099-w) — spatio-temporal trend (MK+Sen) of crop-residue-burning via GEE, Punjab; Birant & Kut (2007) ST-DBSCAN

---

## Fire-HCHO correlation & transport attribution for biomass-burning HCHO over the Indo-Gangetic Plain (BAH 2026 PS3 Obj-2 final steps)

*Pipeline stage: **Fire & Transport** · 9 methods.*

**Recommended stack:**

- **Fire prep: FIRMS VIIRS VNP14IMGML 375m (primary) + MODIS MCD14ML (climatology) -> pandas/statsmodels STL episode flags per cropland/forest AOI**
- **Correlation: deseasonalized lagged cross-correlation + prewhitened Granger (statsmodels) on AOI-mean TROPOMI HCHO (S5P OFFL L3_HCHO, QA>0.5, cloud<0.4) vs fire-count/FRP, per season**
- **Spatial/emission-ratio: xarray co-grid HCHO+NO2+FRP to 0.05deg; dHCHO/FRP slope, HCHO:NO2 FNR maps, bivariate LISA co-location (esda)**
- **Quick directional check: openair polarPlot + pollutionRose of HCHO vs ERA5 10m winds at IGP receptors**
- **Trajectories: ERA5-driven HYSPLIT via splitr(R)/PySPLIT, 120h back (500&1000m AGL, 13:30 LT) from Delhi/Lucknow/Kanpur/Patna + forward from fire clusters**
- **Pathway attribution: openair trajCluster (angle metric) -> composite HCHO/fire per cluster**
- **Source maps: openair trajLevel CWT + PSCF on 0.25-0.5deg grid**
- **Quantitative attribution: FLEXPART backward footprints x FINNv2.5 HCHO emissions (final figure)**
- **Inventory cross-check: FINNv2.5 daily (NCAR RDA ds312.9, explicit HCHO) primary + GFED4.1s/GFED5 bracket; top-down/bottom-up dHCHO regression**
- **Data access: Google Earth Engine (COPERNICUS/S5P/OFFL/L3_HCHO, .../L3_NO2, ECMWF/ERA5 hourly, FIRMS)**

**Key findings:**

- VIIRS 375m (VNP14IMGML) is essential over Punjab/Haryana - detects ~3-5x more small field fires than MODIS 1km, which systematically misses stubble fires; use MODIS only for the 2002+ climatology.
- TROPOMI HCHO over IGP mixes PRIMARY fire HCHO with SECONDARY photochemical HCHO from co-emitted NMVOC oxidation -> expect 0-2 d fire->HCHO lag; interpret dHCHO/FRP slopes and HCHO:NO2 (FNR) with plume-age in mind; Oct-Nov IGP HCHO ~12.5e-6 mol/m2.
- Granger causality is predictive, not physical - confounded by shared meteorology and summer biogenic isoprene HCHO; prewhiten/deseasonalize, restrict to post-monsoon, and complement with CCM and conditional analysis.
- HYSPLIT trajectories show source PATH not concentration; for quantitative 'X% of receptor HCHO from Punjab fires' use FLEXPART backward footprint x FINN/GFED HCHO emission, or CWT/PSCF maps - validate the three against each other.
- openair (R) is the highest-leverage single tool: trajCluster (pathways), trajLevel (CWT/PSCF), polarPlot/pollutionRose (ERA5-wind directional fingerprint) cover most of the transport workflow with minimal code.
- Drive HYSPLIT/FLEXPART with ERA5 (better IGP/Himalayan boundary layer than 1deg GDAS); release at S5P ~13:30 LT, 500&1000m AGL; expect a dominant NW Punjab/Haryana transport cluster carrying peak HCHO in Oct-Nov.
- FINNv2.5 (Wiedinmyer 2023 GMD) is the best inventory cross-check: daily, VIIRS375m-based, speciated NMVOC with explicit HCHO, 2002-2023 at 0.1deg (NCAR RDA ds312.9); bracket uncertainty with GFED4.1s/GFED5 (can disagree 2-3x).
- Deliverable = a convergent multi-line attribution chain: episode flags -> lagged corr/Granger -> dHCHO/FRP + FNR + LISA -> polar plots -> HYSPLIT cluster+CWT/PSCF -> FLEXPART x FINN; agreement across statistical, Lagrangian, and inventory lines is the credible result, not any single method.

### Biomass-burning period extraction (thresholds + seasonal decomposition)

- **Category:** Fire time-series preprocessing
- **Summary:** Build daily fire-count/FRP series over Punjab-Haryana + forest AOIs from FIRMS (MODIS MCD14ML, VIIRS VNP14IMGML 375m). Define episodes via percentile thresholds (>90th pct), STL/seasonal decomposition (statsmodels STL, prophet) isolating the Oct-Nov rice-stubble and Apr-May wheat/forest peaks, and anomaly = obs - seasonal climatology. Filter confidence, day-night, and land-cover (cropland vs forest).
- **Inputs:** FIRMS active fire (lat,lon,FRP,confidence,date,daynight); land-cover (ESA WorldCover, MCD12Q1); AOI polygons
- **Outputs:** Daily fire-count & summed-FRP series per AOI; binary burning-episode calendar; seasonal/anomaly components
- **Libraries:** `pandas`, `statsmodels (STL)`, `prophet`, `geopandas`, `earthengine-api (FIRMS)`
- **Complexity:** Low; O(N) over fire records; runs on a laptop
- **Pros:** Cheap, transparent, robust episode flags; separates climatology from anomalies; VIIRS resolves sub-MODIS fields
- **Cons:** Cloud/overpass gaps bias counts; FRP saturates; threshold choice subjective; no plume-age info
- **Benchmark:** VIIRS detects ~3-5x more Punjab field fires than MODIS; Oct-Nov Punjab+Haryana counts explain ~78% of Delhi AOD variance (lit.)
- **Recommendation:** Use VIIRS 375m as primary (resolves small field fires), MODIS for 2002+ climatology. Episode = STL-residual > +2 sigma sustained >=2 days; tag cropland vs forest.

### Lagged cross-correlation & Granger causality (fire -> HCHO)

- **Category:** Statistical correlation/causality
- **Summary:** Align AOI-mean TROPOMI HCHO column (S5P OFFL L3_HCHO, mol/m2) with fire-count/FRP. Cross-correlation over lags 0-7 d finds peak lag (primary + secondary HCHO from NMVOC oxidation lags fire ~0-2 d). Granger test (statsmodels) on stationarized (STL-residual/differenced) series, lag via AIC. Deseasonalize both and control for biogenic isoprene HCHO and ERA5 temperature.
- **Inputs:** Daily AOI-mean HCHO (QA>0.5, cloud<0.4); fire counts/FRP; ERA5 T2m, isoprene proxy
- **Outputs:** Lag-resolved correlation + significance; Granger F-stats/p by lag; optimal lag
- **Libraries:** `statsmodels (ccf, grangercausalitytests, adfuller)`, `scipy.stats`, `pandas`, `skccm (CCM)`
- **Complexity:** Low-medium; needs stationarity care
- **Pros:** Quantifies temporal lead-lag & directionality; cheap; answers 'does fire drive HCHO'
- **Cons:** Granger=predictive not physical; confounded by shared met & biogenic HCHO; cloud gaps; autocorrelation inflates significance
- **Benchmark:** Lit. reports significant fire-HCHO/CO co-variation in Oct-Nov-Dec over IGP; HCHO enhancement strongest at 0-2 d lag
- **Recommendation:** Run cross-correlation per season/AOI; report peak-lag r+CI. Granger only on prewhitened stationary residuals; complement with CCM/transfer entropy. Expect peak at 0-1 d lag in Oct-Nov.

### Spatial co-location & emission-ratio analysis (HCHO/NO2, dHCHO vs FRP)

- **Category:** Spatial / emission-ratio
- **Summary:** Co-grid HCHO, NO2 (S5P OFFL L3_NO2) and fire pixels to 0.05-0.1deg. Compute background-subtracted dHCHO and regress vs FRP -> emission slope (mol per MW). Map HCHO/NO2 (FNR) as VOC/NOx regime + burning fingerprint. Bivariate Moran/Local Moran (LISA) for fire-HCHO co-location clusters. Plume-distance gradient separates primary vs secondary HCHO.
- **Inputs:** Gridded HCHO, NO2 (QA filtered); gridded FRP/fire density; clean-day baseline fields
- **Outputs:** dHCHO-FRP slope (emission ratio); FNR maps; bivariate LISA clusters; enhancement maps
- **Libraries:** `xarray`, `rioxarray`, `esda/libpysal (Moran/LISA)`, `scikit-learn`, `earthengine-api`, `harp`
- **Complexity:** Medium; gridding + regression + spatial stats
- **Pros:** Links column enhancement to fire intensity; FNR adds chemistry regime; co-location gives spatial attribution; reuses Obj-1 data
- **Cons:** NO2 short-lived vs partly-secondary HCHO; column not surface; AMF/cloud errors; background definition sensitive
- **Benchmark:** Lit. Oct-Nov IGP HCHO ~12.5e-6 mol/m2 during burning; FNR widely used for VOC/NOx regime over India
- **Recommendation:** Map dHCHO/FRP slope per season; use FNR ~1-2 transition; validate hotspots where high HCHO + high NO2 + active fires coincide (LISA high-high clusters).

### HYSPLIT back/forward trajectory analysis

- **Category:** Lagrangian transport
- **Summary:** Ensembles of HYSPLIT trajectories: BACK from IGP receptors (Delhi, Lucknow, Kanpur, Patna) to test Punjab/Haryana origin; FORWARD from fire clusters to map downwind HCHO exposure. Drive with GDAS/GFS or (best) ERA5; release 500/1000/1500 m AGL, 72-120 h. Intersect endpoints with FIRMS fire density and TROPOMI HCHO swaths for source attribution.
- **Inputs:** HYSPLIT exe + met (GDAS/GFS/ERA5 ARL); receptor coords; release heights/times; fire & HCHO fields
- **Outputs:** Trajectory endpoints (lat,lon,height,t); residence-time over fire regions; attribution overlays
- **Libraries:** `splitr (R)`, `PySPLIT (Python)`, `NOAA HYSPLIT`, `era5->arl (api2arl)`
- **Complexity:** Medium-high; met packing + batch runs
- **Pros:** Standard, validated, free (NOAA ARL); links source to receptor; forward maps plume reach
- **Cons:** Single-particle, no dispersion/chemistry; sensitive to release height & met res; ERA5->ARL overhead; trajectory != concentration
- **Benchmark:** Lit. HYSPLIT shows Punjab smoke transported 200-300 km across IGP; 120h forward at 500-1000m standard for crop-burning attribution
- **Recommendation:** Run 120 h back-trajectories at 500 & 1000 m AGL, release times matched to S5P ~13:30 LT, ERA5-driven (better IGP BL than GDAS). Automate via splitr (R) or PySPLIT.

### Trajectory clustering (source-region attribution)

- **Category:** Lagrangian transport / clustering
- **Summary:** Cluster the trajectory ensemble (k-means/hierarchical on great-circle angle-distance or resampled endpoints) into representative pathways. Assign each receptor-day to a cluster; composite TROPOMI HCHO and fire counts by cluster to identify which pathway (e.g. NW Punjab flow) carries peak HCHO. Choose k via total spatial variance elbow.
- **Inputs:** Trajectory endpoint ensemble; per-trajectory HCHO/fire metrics
- **Outputs:** Cluster-mean trajectories; cluster frequency; HCHO/fire composites per cluster
- **Libraries:** `openair (trajCluster, R)`, `PySPLIT`, `scikit-learn (KMeans/Agglomerative)`, `scipy`
- **Complexity:** Medium; clustering + compositing
- **Pros:** Reduces 1000s of trajectories to interpretable pathways; quantifies dominant HCHO source; openair has trajCluster
- **Cons:** k and metric subjective; clusters smear sub-pathway variability; inherits trajectory errors
- **Benchmark:** openair trajCluster is de-facto standard; angle-distance metric recommended for episodic source attribution
- **Recommendation:** Cluster 120h back-trajectories with openair trajCluster (angle metric) or PySPLIT; expect a dominant NW Punjab/Haryana pathway in Oct-Nov carrying peak HCHO.

### Concentration-Weighted Trajectory (CWT) & PSCF

- **Category:** Receptor source apportionment
- **Summary:** Grid the domain; weight cells by trajectory residence time and receptor HCHO. PSCF = fraction of endpoints in a cell tied to HCHO above threshold (75th pct) -> source probability. CWT = residence-weighted mean HCHO -> source-strength map. Apply Wij smoothing to damp low-count cells. Overlay GFED/FINN emission grids to confirm hotspots coincide with burning.
- **Inputs:** Trajectory endpoint ensemble; receptor HCHO series; grid; optional GFED/FINN emission grid
- **Outputs:** PSCF probability map; CWT weighted-concentration source map; source-region ranking
- **Libraries:** `openair (trajLevel, R)`, `pyPSCF/ZeFir`, `trajstat (MeteoInfo)`, `numpy/xarray`
- **Complexity:** Medium; gridded residence-time accumulation
- **Pros:** Explicit spatial source maps from receptor data; established; complements clustering; ties HCHO to emitting cells
- **Cons:** Trailing-effect & collinear-source artifacts; PSCF threshold sensitivity; needs many trajectories; column-vs-surface mismatch
- **Benchmark:** CWT/PSCF standard for biomass-burning apportionment; TrajStat/openair widely cited for IGP/Delhi studies
- **Recommendation:** Compute CWT (quantitative) + PSCF (probabilistic) on 0.25-0.5deg grid from 120h trajectories + AOI HCHO; Wij smoothing; expect maxima over Punjab/Haryana in Oct-Nov, validated vs FINN.

### Wind-sector & bivariate polar plots (openair) with ERA5 winds

- **Category:** Meteorological conditional analysis
- **Summary:** Pair receptor HCHO with ERA5 10m winds. openair polarPlot (HCHO vs wind speed+direction), pollutionRose, percentileRose reveal the sector and wind-speed regime delivering high HCHO -> low-level confirmation of NW transport from Punjab/Haryana. polarCluster groups episode types. Fast QC before HYSPLIT.
- **Inputs:** Receptor HCHO (or surface PM/AQI); ERA5 u10/v10 -> ws/wd; aligned timestamps
- **Outputs:** Bivariate polar HCHO surfaces; pollution roses by sector; directional source signatures
- **Libraries:** `openair (polarPlot, pollutionRose, percentileRose, polarCluster, R)`, `windrose (Python)`, `matplotlib`
- **Complexity:** Low; one-line openair functions
- **Pros:** Fast, intuitive directional fingerprint; uses ERA5 already in pipeline; QC for trajectory results
- **Cons:** Local wind only (no long path); column-vs-surface; static-receptor assumption; correlation not causation
- **Benchmark:** openair polar/rose plots are the standard first-pass directional attribution tool in air-quality literature
- **Recommendation:** Generate polarPlot + pollutionRose of HCHO (and Obj-1 surface AQI) vs ERA5 winds at IGP receptors; an NW high-HCHO lobe at moderate winds corroborates fire transport.

### FLEXPART Lagrangian dispersion

- **Category:** Lagrangian dispersion (concentration)
- **Summary:** Run FLEXPART (or FLEXPART-WRF) backward from IGP receptors for source-receptor sensitivity footprints (s m3 kg-1), then convolve with GFED/FINN HCHO+NMVOC emission flux to predict the HCHO contribution from Punjab/Haryana fires - true concentration attribution unlike single-particle trajectories. Forward mode disperses fire emissions to map downwind fields.
- **Inputs:** ERA5/GFS or WRF met; receptor boxes; particles; GFED5/FINN HCHO+VOC inventory for convolution
- **Outputs:** Source-receptor footprints; modeled HCHO contribution maps; fire-attributable fraction
- **Libraries:** `FLEXPART / FLEXPART-WRF`, `reflexible/pflexible`, `flexwrf`, `xarray`
- **Complexity:** High; met preprocessing, particle runs, emission convolution
- **Pros:** Rigorous concentration attribution incl. turbulence; footprint x inventory yields % from fires; forward maps plumes
- **Cons:** Heavy setup/compute; ERA5/WRF preprocessing; secondary HCHO chemistry not in passive runs; steep learning curve
- **Benchmark:** FLEXPART reference Lagrangian model (Pisso 2019 GMD); footprint x GFED/FINN standard for quantitative burning attribution
- **Recommendation:** Use FLEXPART backward footprints x FINNv2.5/GFED HCHO emissions for quantitative attribution where HYSPLIT is inconclusive; ERA5/WRF driven. Reserve for the final quantitative figure.

### Emission-inventory cross-check (GFED / FINN VOC)

- **Category:** Emission inventory validation
- **Summary:** Cross-validate attribution vs bottom-up inventories: FINNv2.5 (daily, VIIRS375m+MODIS, 0.1deg, NMVOC speciated incl. explicit HCHO) and GFED4.1s/GFED5 (0.25deg burned-area, VOC factors). Compare inventory HCHO/NMVOC flux over Punjab/Haryana with TROPOMI dHCHO and CWT/footprint maps; top-down/bottom-up ratio flags inventory bias.
- **Inputs:** FINNv2.5 daily (NCAR RDA ds312.9, MOZART speciation); GFED4.1s/GFED5 (netCDF); gridded TROPOMI HCHO
- **Outputs:** Gridded fire HCHO/NMVOC flux; top-down vs bottom-up ratio; inventory-validated attribution; scaling factors
- **Libraries:** `xarray/netCDF4`, `FINN (NCAR RDA ds312.9)`, `GFED4/GFED5 tools`, `regionmask`, `cdo/nco`
- **Complexity:** Medium; inventory regridding + speciation mapping
- **Pros:** Independent bottom-up constraint; FINN has daily speciated HCHO; closes emission->column loop; flags inventory bias
- **Cons:** Emission factors uncertain (crop residue); FINN/GFED disagree 2-3x; coarse grid; injection-height assumptions; no secondary HCHO
- **Benchmark:** FINNv2.5 (Wiedinmyer 2023 GMD): VIIRS375m daily 2002-2023 speciated HCHO; GFED baseline; top-down/bottom-up HCHO ratios in TROPOMI inversion lit. (ACP 2026)
- **Recommendation:** Use FINNv2.5 daily (matches your VIIRS fires, direct HCHO field) as primary; GFED4.1s/GFED5 to bracket uncertainty. Regress TROPOMI dHCHO vs FINN HCHO for a top-down scaling factor.

**SOTA references:**

- Wiedinmyer et al. 2023, GMD 16:3873 - FINN v2.5 (VIIRS375m daily speciated HCHO): https://gmd.copernicus.org/articles/16/3873/2023/ ; data NCAR RDA ds312.9
- ACP 26:733 (2026) - Global VOC emissions from inversion of TROPOMI HCHO+glyoxal: https://acp.copernicus.org/articles/26/733/2026/
- Springer Env Dev Sustain (2026) 10.1007/s10668-026-07726-2 - Satellite+model crop-burning on IGP air quality (HYSPLIT+TROPOMI HCHO/CO)
- Carslaw & Ropkins - openair (trajCluster, trajLevel CWT/PSCF, polarPlot); HYSPLIT appendix: https://openair-project.github.io/book/sections/appendices/appendix-hysplit.html
- splitr (R) https://github.com/envhyf/SplitR ; PySPLIT https://github.com/mscross/pysplit - HYSPLIT automation
- NOAA ARL HYSPLIT (Stein et al. 2015 BAMS); FLEXPART (Pisso et al. 2019 GMD 12:4955) - Lagrangian footprints
- GEE assets: COPERNICUS/S5P/OFFL/L3_HCHO & NRTI/L3_HCHO; .../L3_NO2; ECMWF/ERA5 - https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_HCHO
- GFED4.1s / GFED5 (van der Werf 2017 ESSD; Chen 2023) - burned-area fire emissions w/ VOC factors: https://www.globalfiredata.org/
- NCBI PMC12431154 (2025) Sentinel-2 burned-area Punjab 2022-2024; PMC12823738 stubble burning & Delhi AQ
- FIRMS active fire: MODIS MCD14ML C6.1 (1km), VIIRS VNP14IMGML/VJ114IMGML 375m - https://firms.modaps.eosdis.nasa.gov/

---

## Multi-source data fusion, gap-filling, bias-correction and downscaling for satellite-derived surface AQI and HCHO hotspots over India (BAH 2026 ISRO PS3)

*Pipeline stage: **Fusion & Gap-fill** · 11 methods.*

**Recommended stack:**

- **LAYER 0 — Gap-free physical prior: CAMS EAC4/NRT (NO2/SO2/CO/O3/HCHO/AOD) via ADS cdsapi + MERRA-2 (M2T1NXAER AOD) as always-available background and ML features**
- **LAYER 1 — Multi-satellite harmonization: INSAT-3D/3DR AOD (geostationary, high cadence) + Sentinel-5P TROPOMI NO2/SO2/CO/O3/HCHO (GEE COPERNICUS/S5P/OFFL/L3_*; HCHO=L3_HCHO) + MODIS MAIAC MCD19A2 1km AOD; regrid with xESMF/rioxarray**
- **LAYER 2 — Per-species gap-fill: DINEOF (pydineof) first-pass per cube -> ConvLSTM/U-Net partial-conv (PyTorch) nonlinear cloud/orbit refinement; spatiotemporal kriging fallback with uncertainty**
- **LAYER 3 — Bias correction to CPCB: regionalized quantile mapping (python-cmethods/xclim.sdba) then ML residual correction (LightGBM) on met/BLH/land-use covariates**
- **LAYER 4 — Surface AQI model (objective 1): CNN / LSTM / CNN-LSTM (PyTorch) plus LightGBM/XGBoost baseline mapping gap-filled columns + ERA5/IMDAA/MERRA-2 met + land use -> CPCB; score RMSE/R/MAE**
- **LAYER 5 — Ground anchoring + uncertainty: GWR (mgwr) or BME (BMElib) fusing ML surface estimate (soft) with CPCB (hard); residual-kriging to enforce station agreement and produce uncertainty maps**
- **LAYER 6 — Downscaling to 1km: covariate-guided statistical/geographic downscaling (LUR + mgwr) baseline, optional deep super-resolution (EDSR/SRGAN) using land-use/roads/elevation/NDVI/population priors**
- **HCHO HOTSPOTS (objective 2): gap-filled TROPOMI HCHO + MODIS/VIIRS FIRMS active fire + ERA5/IMDAA winds -> Getis-Ord Gi*/LISA + DBSCAN/ST-DBSCAN clustering, fire-HCHO correlation, HYSPLIT back-trajectory transport**
- **Geospatial/IO backbone: Google Earth Engine + earthengine-api, xarray/rioxarray/rasterio/xESMF, geopandas, dask for India-wide daily cubes**

**Key findings:**

- Layered strategy beats any single method: gap-free reanalysis prior (CAMS/MERRA-2) -> multi-satellite obs -> gap-fill (DINEOF + deep inpainting) -> CPCB bias-correction -> ML surface model -> GWR/BME anchoring -> 1km downscaling. Each layer fills/verifies another.
- Proven India recipe: central-India PM2.5 (Sci Rep 2020) gap-fills MAIAC AOD with gap-free MERRA-2 AOD then ML with met+land-use; directly portable, substituting INSAT-3D AOD as high-cadence source.
- For column gap-fill, DINEOF is the best cheap label-free first pass (gives uncertainty); U-Net partial-conv inpainting (validated on S5P CO) matches statistical accuracy ~1000x faster and preserves nonlinear plumes — ideal combo for cloudy IGP HCHO/NO2.
- XGBoost/RandomForest are the strongest simple baselines for both AOD imputation (beat kriging/IDW/GAM in MDPI RS 2020) and column->surface mapping; use as workhorse before/with the required CNN/LSTM models.
- Ground anchoring is essential via spatially-varying calibration: van Donkelaar/WUSTL GWR (global GWRPM25) for bias and BME (China LUR+BME R2=0.82) to fuse uncertain satellite 'soft' data with CPCB 'hard' data while propagating uncertainty.
- Bias correction should be two-stage: regionalized quantile mapping/CDF matching to fix the whole distribution (critical for AQI category thresholds/extremes) then ML residual correction for state-dependent (RH/BLH-driven) nonlinear bias.
- Downscaling coarse TROPOMI/INSAT to 1km works: covariate-guided statistical/geographic downscaling (land use, roads, elevation, NDVI, population) is the robust baseline; deep SR (NO2 1km, RSE 2025; Vietnam 1km NO2) adds urban gradients but can hallucinate — validate vs CPCB.
- CAMS (ECMWF 4D-Var IFS, assimilates S5P NO2/CO/O3/AOD) gives physically consistent transport-aware gap-free fields; download as prior/feature (do not reimplement DA); its winds + HCHO strengthen objective-2 fire-plume transport analysis.

### DINEOF (Data-Interpolating Empirical Orthogonal Functions)

- **Category:** Gap-filling (cloud/orbit) — unsupervised EOF reconstruction
- **Summary:** Reconstructs missing pixels in a 3D space-time cube via iterative truncated EOF/SVD: gaps set to mean, decompose, retain optimal modes by cross-validation, refill, iterate to convergence. Label-free, no covariates. Captures dominant spatiotemporal variability of AOD/column fields. 2025 work extends to super-resolution gap-free products; transferable to INSAT-3D AOD and TROPOMI column cubes over India where monsoon/winter cloud and orbit swaths create large gaps.
- **Inputs:** Time stack of single variable (daily INSAT AOD or TROPOMI NO2/HCHO column) on fixed grid with NaN gaps; optional covariate field for multivariate DINEOF
- **Outputs:** Gap-free reconstructed cube + per-mode EOF spatial/temporal patterns + cross-validation error estimate
- **Libraries:** `pydineof`, `DINEOF (GHER)`, `DIVAnd.jl`, `scipy SVD`
- **Complexity:** Moderate; O(iterations x SVD). Scales via Lanczos; memory-bound for big cubes
- **Pros:** No labels/covariates required; preserves spatiotemporal coherence; gives error estimate; fast, proven for satellite columns
- **Cons:** Linear (misses sharp plumes); struggles when whole timesteps missing or gap >70-90%; assumes stationary covariance; weak on isolated HCHO/fire plumes
- **Benchmark:** Validated for SST/ocean-colour/AOD; error comparable to kriging at lower cost; 2025 Ocean Science super-res DINEOF
- **Recommendation:** FIRST-PASS gap-filler per TROPOMI species cube (NO2/SO2/CO/O3/HCHO) and INSAT AOD before ML; cheap, label-free, gives uncertainty. Pair with ML for nonlinear residuals.

### Spatiotemporal kriging / Gaussian-process interpolation

- **Category:** Gap-filling + geostatistical fusion
- **Summary:** Models AOD/column or AQI residual as a space-time random field with fitted spatiotemporal covariance/variogram; BLUP prediction fills gaps and interpolates between CPCB stations with kriging variance (uncertainty). Regression-kriging interpolates residuals of a covariate regression. Strong for anchoring sparse CPCB network and merging with satellite trend.
- **Inputs:** Point/grid obs (CPCB AQI, satellite columns) + coords + time; covariates for regression-kriging (elevation, met, land use)
- **Outputs:** Continuous gap-free field + kriging variance map (uncertainty)
- **Libraries:** `PyKrige`, `gstat (R)`, `scikit-gstat`, `GSTools`, `sklearn GP`
- **Complexity:** High at scale: O(n^3) dense; needs local/sparse/fixed-rank approximation for India-wide daily grids
- **Pros:** Principled uncertainty; flexible covariance; regression-kriging fuses covariates; well understood
- **Cons:** Cubic cost; variogram fitting fragile; assumes quasi-stationarity/Gaussianity; poor on sharp plume edges
- **Benchmark:** Standard baseline in MAIAC imputation comparison studies (MDPI RS 2020)
- **Recommendation:** Use for CPCB ground anchoring and residual-kriging of the ML surface-AQI prediction to enforce station agreement and produce uncertainty. Use moving-window kriging for tractability.

### Random Forest / Gradient Boosting (XGBoost, LightGBM)

- **Category:** ML gap-fill + nonlinear fusion/regression
- **Summary:** Tree ensembles impute missing satellite AOD/columns and learn nonlinear column->surface relationships from co-located CPCB labels using met + land-use + reanalysis features. In MAIAC benchmarks XGBoost/RF beat kriging/IDW/GAM. Central-India PM2.5 studies gap-fill MAIAC with MERRA-2 AOD then regress. Robust, fast, strong tabular baseline for PS3 objective-1.
- **Inputs:** Satellite columns (gap-filled), MERRA-2/CAMS AOD & species, ERA5/IMDAA met (BLH, RH, wind, T), elevation, land use, NDVI, population, road density, DOY; labels = CPCB
- **Outputs:** Gap-filled column or surface concentration/AQI per cell + feature importance + quantile uncertainty
- **Libraries:** `xgboost`, `lightgbm`, `scikit-learn`, `catboost`, `quantile-forest`
- **Complexity:** Low-moderate; fast train/inference; easy tuning
- **Pros:** Handles nonlinearity & missing features; strong on tabular; interpretable importances; great baseline
- **Cons:** No native spatial smoothness (blocky); extrapolates poorly beyond training region; ignores spatial autocorrelation unless engineered
- **Benchmark:** Best in MDPI RS 2020 MAIAC imputation comparison; stacking models for continuous PM2.5 (Mekong 2023)
- **Recommendation:** Primary BASELINE for objective-1 and AOD gap-fill; also the fusion blender combining DINEOF + MERRA-2 + met. Report RMSE/R/MAE vs CPCB; use quantile loss for uncertainty.

### ConvLSTM / U-Net partial-convolution inpainting

- **Category:** Deep-learning spatiotemporal gap-filling / inpainting
- **Summary:** CNN image-inpainting adapted to satellite cubes: mask-aware partial convolutions in U-Net or 3D nets reconstruct irregular cloud/orbit gaps; ConvLSTM exploits temporal sequence. Applied to global Sentinel-5P CO with errors comparable to statistical methods but ~1000x faster. Captures nonlinear plume structure better than DINEOF; ideal for TROPOMI HCHO/NO2 swath & cloud gaps over India.
- **Inputs:** Image time series with binary missing-mask per species; optional auxiliary channels (met, AOD, fire). S5P column cube + mask
- **Outputs:** Gap-free reconstructed image sequence; optionally super-resolved
- **Libraries:** `PyTorch`, `keras`, `nvidia partialconv`, `segmentation-models-pytorch`, `monai`
- **Complexity:** High; GPU training; needs cloud-free patches or self-supervised masking for labels
- **Pros:** Models nonlinear/sharp plume structure; very fast inference; handles irregular masks; reusable across species
- **Cons:** Data-hungry; risk of hallucinating under huge gaps; needs self-supervised mask training; uncertainty not native
- **Benchmark:** arXiv 2208.08781 partial-conv on S5P CO: errors ~ statistical, up to 1000x faster; Landsat LST inpainting (AGU 2019)
- **Recommendation:** Use for HCHO/NO2 cloud-gap inpainting (objective-2 continuity) and nonlinear refinement on DINEOF output. Train self-supervised by masking observed pixels.

### Gap-filled MAIAC AOD + MERRA-2 fusion pipeline

- **Category:** Applied multi-source AOD gap-fill (proven India recipe)
- **Summary:** India-validated recipe: take 1km MODIS MAIAC AOD (MCD19A2), fill cloud gaps with gap-free MERRA-2 AOD and/or ML imputation to get full-coverage daily 1km AOD, then combine with MERRA-2 met + land use in LME/RF/CNN-LightGBM to predict surface PM2.5. Maps to PS3: substitute INSAT-3D AOD as high-cadence source, MERRA-2/CAMS as gap-free prior.
- **Inputs:** MAIAC/INSAT-3D AOD (gappy), MERRA-2 AOD (gap-free 0.5x0.625deg), MERRA-2/ERA5 met, land use, CPCB labels
- **Outputs:** Daily gap-free 1km AOD + surface PM2.5/AQI maps
- **Libraries:** `GEE MCD19A2_GRANULES`, `earthengine-api`, `xarray`, `rioxarray`, `lightgbm`
- **Complexity:** Moderate; engineering-heavy (regridding, co-location, QA filtering)
- **Pros:** Proven over central India (Sci Rep 2020); full coverage; leverages reanalysis prior
- **Cons:** MERRA-2 coarse->detail loss; AOD-PM relation varies with RH/BLH/vertical profile; needs hygroscopic & BLH correction
- **Benchmark:** Sci Rep 2020 central-India 3-stage model; ScienceDirect hybrid hourly 1km gap-fill
- **Recommendation:** Backbone AOD layer for objective-1: INSAT-3D AOD gap-filled by MERRA-2/CAMS prior, then ML to surface PM2.5. GEE: MODIS_061_MCD19A2_GRANULES, MODIS/061/MOD04_3K.

### Geographically Weighted Regression (GWR) — van Donkelaar / WUSTL ACAG

- **Category:** Multi-sensor fusion + ground calibration
- **Summary:** WUSTL ACAG global PM2.5: combine multi-instrument AOD (MODIS Terra/Aqua, MISR, SeaWiFS, VIIRS) with GEOS-Chem CTM to get geophysical PM2.5, then calibrate to monitors with GWR whose coefficients vary in space, capturing regional AOD-PM relationship. State-of-art global product (V5.GL.04). Template for fusing INSAT+TROPOMI+MERRA-2 then anchoring to CPCB.
- **Inputs:** Multi-sensor AOD, CTM (GEOS-Chem/MERRA-2/CAMS) species, ground monitors, geographic covariates
- **Outputs:** Calibrated high-res surface PM2.5/AQI with spatially varying calibration
- **Libraries:** `mgwr`, `GWmodel (R)`, `spgwr (R)`, `libpysal/esda`
- **Complexity:** Moderate; GWR bandwidth selection; CTM dependency heavy
- **Pros:** Spatially adaptive bias correction; merges many sensors; globally validated; interpretable local coefficients
- **Cons:** GWR can overfit/collinearity; needs dense monitors for stable local fit; CTM coupling heavy
- **Benchmark:** NASA SEDAC GWRPM25 V5.GL.04 global product; van Donkelaar et al. ES&T
- **Recommendation:** Use GWR (or geographically-weighted RF with spatial features) as GROUND-ANCHORING calibration that corrects satellite/ML AQI to CPCB across India's heterogeneous IGP vs peninsula.

### Bayesian Maximum Entropy (BME)

- **Category:** Geostatistical fusion of hard + soft (uncertain) data
- **Summary:** BME rigorously fuses 'hard' CPCB ground data with 'soft' uncertain data (satellite/ML estimates, met) under a max-entropy framework, propagating each source's uncertainty. China national LUR+BME PM2.5 reached R2~0.82, RMSE~4.6; adding soft data raised R2 ~6%. Ideal for blending confident CPCB points with uncertain satellite columns over India.
- **Inputs:** Hard data (station concentrations) + soft data with PDFs/intervals (satellite/ML estimates, LUR residuals) + space-time covariance
- **Outputs:** Posterior mean field + full predictive PDF / uncertainty
- **Libraries:** `BMElib (MATLAB)`, `STAR-BME`, `pyBME`, `scikit-gstat`
- **Complexity:** High; covariance modelling + soft-data PDF specification; computationally heavy
- **Pros:** Principled uncertainty fusion; uses uncertain satellite data without discarding; beats plain kriging
- **Cons:** Complex to implement/tune; soft-data PDF assumptions sensitive; few maintained libraries
- **Benchmark:** ScienceDirect 2018 China LUR+BME R2=0.82; Springer 2017 BME PM2.5/NO2 N China with satellite
- **Recommendation:** Optional final-fusion layer merging ML/LUR surface estimates (soft) with CPCB (hard) for uncertainty-aware AQI. Use if time permits beyond RF/GWR baseline.

### Optimal Interpolation / Data Assimilation (CAMS 4D-Var)

- **Category:** Physics-based assimilation prior
- **Summary:** CAMS/ECMWF assimilates satellite columns (S5P TROPOMI NO2/CO/O3, AOD) into the IFS chemistry model via incremental 4D-Var (12h windows, T95/T159), producing gap-free, physically consistent 3D composition fields (EAC4, NRT/forecast). For PS3, CAMS is a ready gap-free PRIOR/feature; full DA is out of scope but cheaper OI can nudge model to obs.
- **Inputs:** Background model state + satellite retrievals + obs/background error covariances
- **Outputs:** Analysis: gap-free, mass-consistent 3D fields of NO2/CO/O3/SO2/AOD/HCHO
- **Libraries:** `cdsapi (CAMS ADS)`, `CAMS EAC4`, `DART/PyOSSE for OI`
- **Complexity:** Very high for full 4D-Var; OI variant moderate
- **Pros:** Physically consistent, fully gap-free, multi-species coupled; transport-aware (great for HCHO plumes)
- **Cons:** Full DA infeasible for hackathon; coarse (~40-80km) needs downscaling; emission-inventory biases
- **Benchmark:** ACP 2019 CAMS reanalysis; ACP 2022 S5P CO assimilation; ACP 2019 TROPOMI total ozone
- **Recommendation:** DOWNLOAD CAMS EAC4/NRT (NO2/CO/O3/SO2/AOD/HCHO) as gap-free PRIOR and ML features; do NOT reimplement 4D-Var. Provides physical consistency + transport for HCHO hotspot/transport analysis.

### Bias correction: Quantile Mapping / CDF matching

- **Category:** Satellite-vs-ground bias correction (distributional)
- **Summary:** Aligns CDF/quantiles of satellite-derived or reanalysis values to co-located CPCB observations, correcting systematic bias, distribution shape and extremes. Variants: empirical QM, parametric, RQUANT, detrended QM. Standard in precip/AQ; key to correct INSAT AOD-derived or CAMS surface fields to CPCB before AQI computation.
- **Inputs:** Paired satellite/model series + ground series (per station or per cluster/region)
- **Outputs:** Bias-corrected field matching observed distribution
- **Libraries:** `python-cmethods`, `xclim.sdba`, `scikit-downscale`, `statsmodels ECDF`, `SBCK`
- **Complexity:** Low; per-pixel/region transfer function
- **Pros:** Fixes whole distribution incl. extremes (critical for AQI category thresholds); simple, transparent
- **Cons:** Stationarity assumption; needs overlap; spatial transfer to ungauged cells needs clustering/regionalization
- **Benchmark:** Extensive in satellite precip BC; integrated QM+spatial clustering (MDPI Sustainability 2025) for data-sparse regions
- **Recommendation:** Fast, mandatory bias-correction step on satellite/reanalysis inputs vs CPCB; regionalize transfer functions by IGP/peninsula/coastal clusters to extend to ungauged areas.

### ML residual bias correction

- **Category:** Satellite-vs-ground bias correction (nonlinear, covariate-aware)
- **Summary:** Train ML (RF/GBM/NN) to predict residual (satellite/model minus ground) as a function of met, land use, BLH, time; add back to correct. Captures state-dependent, nonlinear, spatially varying bias QM misses (RH/BLH-driven AOD-PM bias). Used to enhance satellite precip BC with WRF met; same recipe for AQ.
- **Inputs:** Satellite/model value + covariates (met, BLH, RH, land use, DOY) at CPCB sites; target = residual
- **Outputs:** Corrected field + residual map; uncertainty via quantile/ensemble
- **Libraries:** `xgboost`, `lightgbm`, `scikit-learn`, `pytorch`
- **Complexity:** Low-moderate; reuses tabular ML stack
- **Pros:** Nonlinear, covariate- and location-aware; beats static QM where bias is state-dependent
- **Cons:** Needs dense co-location; can overfit; extrapolation risk to unseen regimes
- **Benchmark:** ScienceDirect 2024 ML+WRF satellite-precip BC; JGR 2025 DNN precip BC with topography
- **Recommendation:** Nonlinear bias-correction layer downstream of QM, or fold directly into the RF/CNN surface-AQI model. Best when met covariates strongly modulate bias.

### Deep super-resolution / statistical downscaling (TROPOMI/INSAT -> 1km)

- **Category:** Spatial downscaling / super-resolution
- **Summary:** Bring coarse columns to fine grid: (a) statistical downscaling — regress fine concentration on high-res covariates (land use, roads, NDVI, elevation, population, met) a la LUR/geographic-ML; (b) deep SR — CNN/SRGAN/diffusion or land-cover-prior nets learn coarse->fine. Demonstrated: TROPOMI NO2 full-coverage 1km via deep learning; Vietnam 1km NO2; isoprene SR with land-cover priors. Yields 1km NO2/HCHO/AQI over India.
- **Inputs:** Coarse satellite/reanalysis field + high-res static covariates (land use, roads, elevation, NDVI, population, nightlights) + met; ground labels
- **Outputs:** Fine-resolution (1km) gap-aware concentration/AQI maps
- **Libraries:** `pytorch (EDSR/SRGAN/SRCNN)`, `scikit-downscale`, `mgwr`, `xesmf/rioxarray`, `GEE`
- **Complexity:** Statistical: low-moderate; deep SR: high (GPU, training data)
- **Pros:** Recovers urban/road-scale gradients invisible at native res; covariate-guided SR physically grounded; big city-AQI gain
- **Cons:** Deep SR can hallucinate detail; needs high-res covariates & labels; 1km validation hard with sparse CPCB
- **Benchmark:** RSE 2025 NO2 super-res full-coverage 1km; Frontiers 2023 Vietnam 1km NO2; arXiv 2503.18658 land-cover-prior isoprene SR
- **Recommendation:** Final DOWNSCALING layer: covariate-guided (LUR/geographic-ML) downscaling as robust baseline; optional deep SR for cities. TROPOMI HCHO native ~5.5x3.5km -> 1km with land-use+fire+wind covariates.

**SOTA references:**

- van Donkelaar et al. — WUSTL ACAG SatPM2.5 / NASA SEDAC GWRPM25 V5.GL.04 (multi-sensor AOD + GEOS-Chem + GWR ground calibration, global PM2.5 1998-2022)
- Sci Rep 2020 (s41598-020-79229-7) — Central India (Madhya Pradesh) 3-stage PM2.5: MAIAC AOD gap-filled by MERRA-2 + LME, India-specific template
- MDPI Remote Sensing 2020 12(18):3008 — Comparison of missing-imputation methods for MAIAC AOD (XGBoost/RF beat ST-kriging/IDW/GAM)
- arXiv 2208.08781 — Gap-filling satellite image time series with partial-convolution deep nets; validated on quasi-global Sentinel-5P CO (~1000x faster)
- Alvera-Azcarate & Barth — DINEOF for geophysical data; Ocean Science 2025 (os.copernicus.org/articles/21/787/2025) super-resolution gap-free DINEOF
- Inness et al. ACP 2019 (19/3515) — CAMS reanalysis of atmospheric composition (4D-Var IFS, EAC4); ACP 2022 (22/14355) S5P/TROPOMI CO assimilation
- RSE 2025 (S0034425725003013) — Super-resolution model for full-coverage high-res NO2; Frontiers Env Sci 2023 Vietnam 1km NO2 (NN + land use)
- ScienceDirect 2018 (S0160412017321505) — National PM2.5 by LUR + Bayesian Maximum Entropy, China (R2=0.82, RMSE 4.6); Springer AQAH 2017 BME PM2.5/NO2 N China
- arXiv 2302.10278 — Data-level & decision-level fusion of multi-sensor AOD (MODIS+VIIRS, Deep Blue/Dark Target) for PM2.5, Tehran
- MDPI Sustainability 2025 (17/8321) — Integrated quantile mapping + spatial clustering for robust bias correction in data-sparse regions; arXiv 2503.18658 land-cover-prior emission SR

---

## Validation, cross-validation & uncertainty quantification protocol for BAH 2026 PS3 (satellite-derived surface AQI + HCHO hotspots over India)

*Pipeline stage: **Validation** · 14 methods.*

**Recommended stack:**

- **Metrics: scikit-learn + scipy.stats + custom NMB/NME/IOA + scipy.odr (RMA slope/intercept); report RMSE/MAE/R/R2/MBE/NMB/NME/IOA stratified by pollutant, season, AQI band**
- **CV ladder: sklearn GroupKFold (LOSO) + verde.BlockKFold / spacv / blockCV (variogram-sized buffer) + TimeSeriesSplit forward-chaining + spatiotemporal-blocked CV (CAST knndm)**
- **Independent blind hold-out: stratified ~15-20% CPCB station set locked before modeling**
- **Input QA: AERONET L2 AOD collocation (+/-30min, 25-50km) with %-within-EE +/-(0.05+0.15*AOD); pyaerocom harmonized stats**
- **Ground truth: CPCB CCR / data.gov.in hourly -> official India AQI breakpoints -> regression metrics + AQI-category confusion matrix (Severe recall)**
- **Model comparison viz: SkillMetrics Taylor diagram + target diagram (one marker per model/pollutant)**
- **UQ: LightGBM/CNN-LSTM quantile heads -> MAPIE/crepes Mondrian or weighted split-conformal for guaranteed coverage; optional 5-member deep ensemble; MC-dropout baseline**
- **UQ reporting: per-pixel PI maps + PICP/MPIW/CRPS + applicability/abstention map flagging data-sparse forest & cloud-gap zones**

**Key findings:**

- Random k-fold CV LEAKS for AQ via spatial+temporal autocorrelation and inflates R by ~0.10-0.25 vs spatial CV; using it as the headline result is a known rigor failure. Lead with spatial-block and spatiotemporal-blocked CV - studies confirm conventional spatiotemporal CV substantially overestimates generalizability.
- Report the full CV LADDER (random -> temporal -> spatial -> spatiotemporal); the gap between rungs quantifies leakage and is itself a clarity/rigor differentiator. Spatial blocks need a buffer/dead-zone sized to the empirical variogram range to remove neighbor leakage.
- Spatial generalization is the deployment reality: a daily India-wide grid predicts mostly at pixels with NO nearby station, so Leave-Location-Out / spatial-block CV + an untouched stratified hold-out station set are the credible headline numbers, not random CV.
- Beyond required RMSE/R/MAE add MBE, NMB, NME, IOA, RMA slope/intercept; cite benchmarks (Huang 2021 ACP PM2.5: NMB +/-10-20%, NME 35-45%; Emery 2017 IOA goal 0.80) to claim regulatory grade. Slope<1 exposes ML compression of Severe episodes that R2 masks.
- For Obj-1 deliver BOTH continuous metrics AND an AQI-category confusion matrix with per-class precision/recall, emphasizing Poor/Severe recall - the episodes judges and policymakers care about most.
- Validate the AOD INPUT vs Indian AERONET (Kanpur, Gandhi College/IGP, Jaipur, Pune) using the EE envelope +/-(0.05+0.15*AOD); MAIAC-class targets R>0.8, >=70-85% within EE. Shows input due-diligence before AQI error propagation despite India's sparse (~10-15 site) network.
- For UQ, conformal prediction (Mondrian/weighted for spatial covariate shift) gives distribution-free coverage guarantees and is the recommended primary method; 2024-25 satellite-reanalysis PM2.5 fusion pairs LightGBM + spatial CV + conformal to flag applicability limits. Score intervals via PICP and MPIW.
- Deep ensembles give the best epistemic uncertainty and shift robustness (beats MC-dropout, the cheap retrofit). Best practice: produce heteroscedastic intervals then conformal-calibrate, and publish a per-pixel uncertainty + applicability map highlighting cloud-gap and data-sparse forest regions.

### Core point metrics: RMSE, R/R2, MAE, MBE

- **Category:** Accuracy metric (Obj-1 primary)
- **Summary:** RMSE penalizes large errors (pollutant units); MAE robust to outliers; R Pearson corr, R2 explained variance; MBE = mean(pred-obs) signed bias (+ = overprediction). Report all four (PS scores RMSE/R/MAE) plus MBE for bias. RMSE/MAE ratio ~1 = uniform errors, >>1 = heavy tails. Stratify overall AND per-pollutant AND per-AQI-band since Severe-episode skill matters most.
- **Inputs:** Paired predicted vs observed (CPCB) conc/AQI at matched space-time
- **Outputs:** Scalar metrics + per-station, per-month, per-AQI-class tables
- **Libraries:** `scikit-learn`, `numpy`, `scipy.stats`, `statsmodels`
- **Complexity:** O(N) trivial
- **Pros:** Matches PS scoring (RMSE/R/MAE); universally understood by judges
- **Cons:** R/R2 can look good with large MBE; RMSE dominated by few severe-episode errors
- **Benchmark:** Satellite PM2.5 in IGP: R~0.7-0.85, RMSE 25-45 ug/m3; AQI R2>0.7 competitive
- **Recommendation:** MANDATORY headline table: RMSE, MAE, R, R2, MBE overall + by pollutant, season, AQI band.

### MBE / NMB / NME (bias & normalized error)

- **Category:** Bias & normalized error metric
- **Summary:** NMB = sum(pred-obs)/sum(obs) %; NME = sum|pred-obs|/sum(obs) %. Standard in regulatory AQ evaluation. Published PM2.5 benchmarks (Huang 2021 ACP): NMB within +/-10-20%, NME 35-45% = good model. Cite these to claim regulatory-grade performance.
- **Inputs:** Paired pred/obs concentrations
- **Outputs:** NMB%, NME% per pollutant + benchmark pass/fail flag
- **Libraries:** `numpy`, `pyaerocom`
- **Complexity:** O(N)
- **Pros:** Dimensionless, comparable across pollutants and vs published benchmark envelopes
- **Cons:** Unstable when obs sums near zero; less intuitive than RMSE
- **Benchmark:** PM2.5: NMB |<=10-20%|, NME |<=35-45%| (Huang 2021); O3: NMB +/-5-15% (Emery 2017)
- **Recommendation:** Add NMB/NME columns; cite ACP-2021 China PM2.5 thresholds for regulatory-grade rigor.

### IOA (Willmott Index of Agreement) + slope/intercept

- **Category:** Agreement & regression-fit metric
- **Summary:** Refined IOA (Willmott 2012) range -1 to 1; goal>=0.80, criteria>=0.70 (Emery 2017), independent of linear-form assumption. Fit reduced-major-axis (orthogonal/Deming) regression pred~obs: report SLOPE (ideal 1.0; <1 = regression-to-mean compression) and INTERCEPT. Slope<1 exposes ML smoothing that hides extreme-episode underprediction.
- **Inputs:** Paired pred/obs
- **Outputs:** IOA; RMA slope, intercept + CIs
- **Libraries:** `scipy.odr`, `statsmodels`, `scikit-learn`
- **Complexity:** O(N)
- **Pros:** IOA robust to outliers; slope/intercept expose compression R2 masks
- **Cons:** Use ORTHOGONAL/RMA not OLS (OLS slope biased by x-error); IOA less familiar to lay judges
- **Benchmark:** IOA goal 0.80/criteria 0.70; slope 0.85-1.15, small intercept
- **Recommendation:** Report IOA + RMA slope/intercept; show 1:1 hexbin scatter with fit line and y=x reference.

### Random K-fold CV (BASELINE ONLY - leakage-prone)

- **Category:** Cross-validation (naive baseline)
- **Summary:** Random k-fold/LOOCV shuffles all space-time samples. For AQ this LEAKS via spatial+temporal autocorrelation: held-out samples have near-identical neighbors in training -> inflated R/low RMSE not reflecting prediction at new locations/times. Studies show conventional spatiotemporal CV substantially overestimates generalizability.
- **Inputs:** All paired samples shuffled
- **Outputs:** Inflated metrics (optimistic upper bound only)
- **Libraries:** `sklearn KFold`
- **Complexity:** O(k*train)
- **Pros:** Simple; useful as a labeled optimistic upper bound to contrast against honest CV
- **Cons:** Leakage via autocorrelation -> misleading; judges penalize if used as headline
- **Benchmark:** Random-CV R typically 0.10-0.25 higher than spatial-CV R
- **Recommendation:** Show ONLY alongside spatial/spatiotemporal CV to quantify optimism gap; never as primary claim.

### Leave-Location-Out / Spatial-block CV (PRIMARY)

- **Category:** Cross-validation (spatial, recommended)
- **Summary:** Withhold ENTIRE stations/spatial blocks; predict into spatially independent regions. Variants: Leave-One-Station-Out (full time series); Spatial-LOO with a BUFFER (dead-zone) removing neighbors within autocorrelation range; spatial-block k-fold (100-200km blocks). Mandatory: the PS deploys AQI on a continuous India grid where most pixels have NO nearby station.
- **Inputs:** Stations grouped by location/block; buffer = variogram range
- **Outputs:** Honest spatial-generalization RMSE/R/MAE per held-out block
- **Libraries:** `sklearn GroupKFold`, `verde BlockKFold`, `blockCV`, `spacv`, `mlr3spatiotempcv`
- **Complexity:** O(folds*train) + variogram
- **Pros:** Honest estimate at unmonitored locations; defensible; matches gridded-map deployment
- **Cons:** Lower scores; buffer size needs variogram justification
- **Benchmark:** Spatial-CV R ~0.6-0.78 for satellite PM2.5 over India vs ~0.85 random
- **Recommendation:** PRIMARY Obj-1 validation: spatial-block k-fold with variogram-sized buffer; show fold-assignment map.

### Leave-Time-Out / forward-chaining temporal CV

- **Category:** Cross-validation (temporal, recommended)
- **Summary:** Hold out whole periods: leave-one-month/season-out or forward-chaining (train past, test future; no future leakage). Essential since CNN-LSTM uses temporal sequences and the PS targets daily maps incl. unseen days and biomass-burning season. Forward-chaining prevents the model seeing future autocorrelated days.
- **Inputs:** Samples grouped by day/month/season; ordered for forward-chaining
- **Outputs:** Temporal-generalization metrics; per-month skill curve
- **Libraries:** `sklearn TimeSeriesSplit`, `sktime`
- **Complexity:** O(folds*train)
- **Pros:** Prevents LSTM temporal leakage; reveals seasonal skill drop (Oct-Nov burning) key to both objectives
- **Cons:** Reduced training data in early forward-chaining folds; seasonal imbalance
- **Benchmark:** Report burning season (Oct-Nov, Apr-May) skill separately - errors largest there
- **Recommendation:** Combine with spatial CV; report per-month RMSE/R curve highlighting biomass-burning months.

### Spatiotemporal-blocked CV (GOLD STANDARD)

- **Category:** Cross-validation (spatiotemporal, gold standard)
- **Summary:** Hold out blocks in BOTH space and time simultaneously (spatial block over a month) so test shares neither nearby station NOR adjacent day. Hardest, most honest; this is the daily India-grid deployment reality. Report as a ladder: random (optimistic) -> temporal -> spatial -> spatiotemporal (pessimistic). The full ladder is itself a clarity/rigor differentiator.
- **Inputs:** 2D blocking grid in (space x time)
- **Outputs:** Most-honest deployment-realistic metrics
- **Libraries:** `spacv`, `CAST knndm/nndm`, `blockCV`, `verde`
- **Complexity:** O(folds*train); careful block design
- **Pros:** Eliminates both leakage modes; demonstrates leakage-resistant rigor that wins expert judges
- **Cons:** Lowest scores; over-pessimizes if blocks too large
- **Benchmark:** Inter-rung differences quantify autocorrelation leakage
- **Recommendation:** Report the full CV LADDER in one table; lead headline with spatial+spatiotemporal numbers.

### Independent hold-out station set (blind test)

- **Category:** Validation design (independent test)
- **Summary:** Reserve a geographically representative ~15-20% of CPCB stations NEVER used in training/CV/tuning, stratified across IGP/coastal/urban/rural and AQI regimes. Final blind report = most credible single number. Distinct from CV folds (model selection) - this is the untouched test set, locked before any modeling.
- **Inputs:** Pre-split stratified station list
- **Outputs:** Single blind RMSE/R/MAE table = headline credibility number
- **Libraries:** `sklearn stratified split`
- **Complexity:** Trivial split
- **Pros:** Maximally credible; mirrors true operational deployment; prevents tuning leakage
- **Cons:** Sacrifices ~15-20% of stations from training; must fix split BEFORE modeling
- **Benchmark:** Report alongside spatial-CV to confirm consistency
- **Recommendation:** Lock stratified hold-out station set day 1; report final blind metrics + India residual map.

### AERONET validation of satellite AOD

- **Category:** Reference-data validation (input QA)
- **Summary:** Validate INSAT-3D/S5P AOD vs AERONET L2.0 (Indian sites: Kanpur, Gandhi College/IGP, Jaipur, Pune, Gual Pahari). Collocate AERONET +/-30min of overpass, satellite mean over 25-50km. Report R, RMSE, % within Expected-Error EE = +/-(0.05+0.15*AOD). MAIAC: R>0.8 at >68% of 332 sites, >83-87% within EE. Validates AOD INPUT before it propagates into AQI.
- **Inputs:** AERONET L2 AOD (Angstrom-interp to 550nm), satellite AOD, collocation window
- **Outputs:** AOD R/RMSE + %-within-EE, scatter vs AERONET
- **Libraries:** `AERONET API`, `pyaerocom`, `numpy`
- **Complexity:** Collocation O(N log N)
- **Pros:** Gold-standard AOD reference; EE envelope is accepted AOD QA; quantifies input error feeding AQI
- **Cons:** Sparse Indian AERONET (~10-15 sites) -> limited coverage; cloud gaps
- **Benchmark:** Target R>0.8, RMSE<0.1 AOD, >=70% within EE over India
- **Recommendation:** Add AOD-vs-AERONET panel (R, RMSE, %-in-EE) to prove input quality before AQI results.

### CPCB validation of predicted concentrations/AQI

- **Category:** Reference-data validation (Obj-1 ground truth)
- **Summary:** Ground truth = CPCB CAAQMS hourly (~400-540 stations) via CCR/data.gov.in. QA: dedupe, despike, drop calibration flags, daily means/24h AQI via CPCB sub-index breakpoints. Collocate satellite-pixel to station; compute full metric suite under the CV ladder. Stratify IGP vs rest, urban vs rural, season. AQI = official India National AQI (max sub-index).
- **Inputs:** CPCB hourly conc, station metadata, AQI breakpoints
- **Outputs:** Per-pollutant + composite-AQI tables, AQI-category confusion matrix
- **Libraries:** `pandas`, `CPCB CCR/data.gov.in`, `India AQI calculator`
- **Complexity:** O(N) after collocation
- **Pros:** Authoritative national ground truth aligned to PS scoring; enables AQI-category confusion matrix
- **Cons:** Urban siting bias, missing data, drift; sparse in forest/rural zones
- **Benchmark:** Report regression metrics AND AQI-category accuracy/F1 (esp. Severe recall)
- **Recommendation:** Deliver BOTH continuous metrics AND AQI-category confusion matrix with per-class precision/recall; emphasize Severe recall.

### Taylor diagram (multi-model comparison)

- **Category:** Visualization for model comparison
- **Summary:** Polar plot encoding R (angle), normalized std (radius), centered RMSE (distance to reference) for all candidates (CNN, LSTM, CNN-LSTM, RF baseline). Lets judges rank architectures at a glance; model nearest reference wins. Standard in atmospheric evaluation (Taylor 2001).
- **Inputs:** Per-model R and std ratio vs obs
- **Outputs:** One comparison figure
- **Libraries:** `SkillMetrics`, `matplotlib`
- **Complexity:** Trivial
- **Pros:** Compact, authoritative, instant model ranking; signals domain literacy
- **Cons:** Hides BIAS (centered RMSE removes mean error) -> pair with target diagram
- **Benchmark:** Use to justify final architecture choice
- **Recommendation:** Include one Taylor diagram comparing all architectures; pair with target diagram for bias.

### Target diagram (bias + variability)

- **Category:** Visualization for model comparison
- **Summary:** Cartesian plot of normalized bias (y) vs signed normalized unbiased-RMSD (x); distance from origin = total normalized RMSE, unit circle = skill threshold. Complements Taylor by showing SIGN of bias Taylor omits. Together they fully characterize R, std, bias, RMSE per model.
- **Inputs:** Per-model normalized bias and CRMSD
- **Outputs:** Bias-vs-variability figure with unit-circle threshold
- **Libraries:** `SkillMetrics`, `matplotlib`
- **Complexity:** Trivial
- **Pros:** Reveals over/under-prediction direction; points inside unit circle = skillful; multi-pollutant overview
- **Cons:** Less familiar to non-specialist judges -> annotate
- **Benchmark:** Models inside unit circle pass normalized-skill test
- **Recommendation:** Pair with Taylor; one marker per pollutant/model to show all biases on one panel.

### Quantile regression / conformal-calibrated intervals

- **Category:** Uncertainty quantification (recommended)
- **Summary:** Quantile heads (LightGBM objective=quantile, GBM quantile loss, or quantile CNN-LSTM) yield heteroscedastic PIs (wider where sparse/cloudy). Wrap with CONFORMAL prediction (split/Mondrian/weighted) for distribution-free finite-sample COVERAGE guarantees under spatial covariate shift. 2024-25 satellite-reanalysis PM2.5 work pairs LightGBM + spatial CV + conformal to flag applicability limits. Evaluate via PICP & MPIW.
- **Inputs:** Features + target; quantile levels; spatial calibration set
- **Outputs:** Guaranteed-coverage per-pixel PI maps + applicability/abstention map
- **Libraries:** `LightGBM/XGBoost quantile`, `sklearn GBR(loss=quantile)`, `MAPIE`, `crepes`, `puncc`
- **Complexity:** ~3x training; conformal negligible
- **Pros:** Cheap, heteroscedastic, distribution-free; coverage guarantee; flags where map is untrustworthy
- **Cons:** Quantile crossing possible; exact guarantee needs weighted/Mondrian under shift; wide PIs in sparse forest
- **Benchmark:** PICP within +/-2-3% of nominal under spatial-CV calibration
- **Recommendation:** PRIMARY UQ: quantile base + Mondrian/weighted split-conformal; publish per-pixel uncertainty + applicability map.

### Deep ensembles / MC dropout (deep UQ)

- **Category:** Uncertainty quantification (epistemic)
- **Summary:** Deep ensemble: N=5-10 CNN-LSTM with different seeds; mean prediction, variance = epistemic uncertainty; best UQ quality and robustness under dataset SHIFT (Lakshminarayanan 2017, beats MC-dropout). MC-dropout (Gal 2016): dropout active at inference, ~50-100 passes, cheapest deep UQ, approaches BNN on CRPS/NLL but weaker under shift. Add a variance head for aleatoric.
- **Inputs:** N trained models (ensemble) OR dropout net + T passes
- **Outputs:** Mean + epistemic variance map; CRPS/NLL/PICP/MPIW
- **Libraries:** `PyTorch/TF custom`, `tensorflow-probability`, `laplace-torch`
- **Complexity:** Ensemble Nx training; MC-dropout T-pass inference
- **Pros:** Ensembles: best uncertainty + shift robustness, parallel. MC-dropout: near-free retrofit
- **Cons:** Ensemble Nx compute; MC-dropout underestimates uncertainty, dropout-rate sensitive
- **Benchmark:** Ensembles beat MC-dropout on CRPS/NLL & shift in AQ studies
- **Recommendation:** 5-member deep ensemble for final CNN-LSTM if compute allows; combine mean with conformal for coverage; MC-dropout as cheap baseline.

**SOTA references:**

- Meyer & Pebesma - CAST (knndm/nndm) spatial CV & area-of-applicability; Valavi 2019 blockCV & mlr3spatiotempcv - spatial/temporal blocking with buffering
- AQ-ML spatial-CV: random/spatial/spatiotemporal mobile-monitoring validation (PMC9408314); reviews that conventional spatiotemporal CV overestimates generalizability (arXiv:2012.13867; arXiv:2402.00183)
- Lyapustin/Qin MODIS MAIAC global AOD validation (332 AERONET sites, R>0.8 at >68%, AtmEnv 2021); MODIS/VIIRS AOD over India (Frontiers 2023 11:1158641); EE +/-(0.05+0.15*AOD)
- TROPOMI ground validation: Verhoelst et al. AMT 14:481 (2021) NO2 vs ZSL/MAX-DOAS/Pandonia; ESA S5P-MPC VDAF summaries (NO2/SO2/HCHO/O3)
- Taylor (2001 JGR) Taylor diagram; Jolliff et al. (2009) target diagram; SkillMetrics Python package
- AQ benchmarks: Emery et al. (2017 JAWMA 67:582) NMB/NME/IOA goal-criteria; Huang et al. (2021 ACP 21:2725) China PM2.5 benchmarks; Willmott et al. (2012) refined IOA
- Conformal prediction: Vovk; Angelopoulos & Bates intro; MAPIE/crepes/puncc; arXiv:2604.22787 Conformal PM2.5 under spatial covariate shift (Mondrian/weighted/Geo-conformal)
- Deep UQ: Lakshminarayanan et al. (2017) deep ensembles; Gal & Ghahramani (2016) MC-dropout; arXiv:2112.02622 probabilistic AQ forecasting (PICP/MPIW/CRPS)
- Global ML air-pollution estimation with prediction intervals (PMC12289206); ConvFormer-KDE & multi-source point-interval PM2.5 frameworks (2024-25)
- CPCB CAAQMS / CCR & data.gov.in real-time data; National AQI (India) sub-index breakpoints (CPCB 2014)

---

## Fastest geospatial compute & O(1)/O(log n) access architecture for planetary-scale air-quality data — applied to BAH 2026 PS3 (Surface AQI maps + HCHO hotspots over India)

*Pipeline stage: **Fast-Platform / O(1)** · 12 methods.*

**Recommended stack:**

- **PRIMARY COMPUTE: Google Earth Engine (earthengine-api + geemap) for all global ingestion, temporal compositing, cloud-gap filling, and CPCB-station reduceRegions over S5P/MODIS/ERA5 — O(1) for the analyst**
- **FUSION GRID: Uber H3 (h3-py v4 / h3ronpy) res 7 as common O(1) join key for satellite pixels + CPCB stations + FIRMS fires; grid_disk hotspot neighborhoods, cell_to_parent IGP rollups**
- **NEAREST/OVERLAP: scipy cKDTree (nearest-station matching, IDW gap-fill) + shapely STRtree / geopandas.sindex (clip & join to India admin boundaries)**
- **RASTER FORMAT/SERVING: COG (rio-cogeo/rioxarray) + TiTiler XYZ + PMTiles/CDN for daily national AQI & HCHO maps via HTTP range requests**
- **DATACUBE FORMAT: Zarr + VirtualiZarr/Kerchunk over IMDAA/MERRA-2 NetCDF, and ARCO-ERA5 Zarr directly, opened lazily with xarray**
- **TABULAR FORMAT: GeoParquet (pyarrow/geopandas), Hilbert-sorted, partitioned by date — the fused CNN/LSTM feature matrix keyed by H3 cell**
- **DISCOVERY: STAC (pystac-client + odc-stac + stac-geoparquet) to index exported COGs and pull MODIS/VIIRS/Sentinel-2 supplementary layers**
- **OUT-OF-CORE COMPUTE: Pangeo (xarray + Dask + NumPy + flox) for custom ML preprocessing, regridding, and LSTM time-series windowing**
- **FAST GEO SQL: DuckDB + spatial + h3 extensions as fusion/serving engine over cloud GeoParquet (range-request reads, H3 groupby, hotspot SQL)**
- **ML: CNN/LSTM/CNN-LSTM (TensorFlow/PyTorch) consuming H3/COG-aligned GeoParquet patches; validate vs CPCB on RMSE/R/MAE**

**Key findings:**

- GEE is the single fastest lever: it co-locates ALL PS3 inputs (S5P HCHO/NO2/SO2/CO/O3 at COPERNICUS/S5P/{OFFL,NRTI}/L3_*, MODIS MAIAC AOD MCD19A2, FIRMS, ERA5_LAND) and runs lazy server-side reductions — O(1) analyst effort. Make it the ingestion + station-reduction engine.
- H3 res 7 is the O(1) fusion trick: give every satellite pixel, CPCB station, and fire an integer cell once, so ALL station<->grid<->fire joins become hash groupbys (O(n)) not O(n*m) geometric joins; grid_disk gives hotspot neighborhoods, cell_to_parent gives IGP rollups.
- Gap-filling map: GEE .median compositing fills S5P cloud gaps; cKDTree IDW + ERA5 fill spatial/temporal holes; MODIS MAIAC 1km AOD and geostationary INSAT-3D (high cadence) fill each other's resolution/revisit gaps; FIRMS/VIIRS confirm HCHO burning hotspots; CPCB calibrates all columns.
- Cloud-native partial reads beat downloads: COG + HTTP range requests serve maps reading KBs/tile; Zarr/VirtualiZarr open multi-TB reanalysis lazily (ARCO-ERA5 = serverless ~12TB); GeoParquet + DuckDB read only needed columns/row-groups over HTTP. Never download full files.
- DuckDB (spatial + h3 + httpfs) is the fast zero-infra fusion/serving engine: queries cloud GeoParquet in-place with predicate pushdown, computes H3 aggregation and HCHO hotspot SQL (Getis-Ord/quantile), runs on a laptop in seconds.
- Two-tier architecture: server-side planetary compute in GEE (global, lazy, O(1)-for-analyst) -> export compact H3/COG/GeoParquet artifacts -> local Pangeo/DuckDB/H3 for custom ML preprocessing GEE can't express (normalization, CNN patching, LSTM windowing).
- Serving tier: precomputed PMTiles or CDN-cached TiTiler XYZ tiles make every daily AQI/HCHO map request an O(1) cache hit (<100ms) at national scale.
- Discovery: build a static STAC (+stac-geoparquet) over exported daily COGs for O(log n) spatiotemporal search; use Planetary Computer / Earth Search + odc-stac to pull MODIS/VIIRS as analysis-ready xarray; keep GEE primary for S5P.

### Google Earth Engine (GEE) — server-side lazy planetary-scale compute

- **Category:** Compute platform (PRIMARY fast platform)
- **Summary:** Fastest path: analyst writes lazy server-side expressions; Google's cluster executes globally — effectively O(1) for the analyst (no download, no local cluster). Holds all PS3 inputs as analysis-ready ImageCollections: COPERNICUS/S5P/{OFFL,NRTI}/L3_{HCHO,NO2,SO2,CO,O3} (0.01deg, daily), MODIS MCD19A2 (MAIAC AOD 1km), FIRMS fire, ECMWF/ERA5_LAND/HOURLY. Use filterDate/filterBounds, .median() composites (cloud-gap fill), reduceRegions for CPCB extraction, Export to COG.
- **Inputs:** AOI geometry (India/IGP), date ranges, band names, CPCB station FeatureCollection
- **Outputs:** Composited rasters, per-station reduced tables, exported daily COGs for ML
- **Libraries:** `earthengine-api`, `geemap`, `eemont`, `geopandas`
- **Complexity:** O(1) for analyst (server-side); internally distributed
- **Pros:** Zero egress; all PS3 datasets co-located; lazy reduceRegions over all CPCB stations in one call; trivial cloud-gap fill via compositing; free for research
- **Cons:** Quotas/timeouts on huge reduceRegions (mitigate tileScale 4-16, explicit scale, maxPixels); not for custom CNN inference; export latency
- **Benchmark:** reduceRegions over ~1500 CPCB stations on a daily S5P image returns in seconds-minutes vs hours of local download+sampling
- **Recommendation:** PRIMARY ingestion+reduction engine. Composite, gap-fill, co-locate stations server-side; export compact tables/COGs. scale=1113, tileScale=8, combine mean+stdDev reducers.

### Uber H3 hexagonal hierarchical index

- **Category:** Spatial index (O(1) point->cell)
- **Summary:** Maps (lat,lng)->hex cell in O(1) via latlng_to_cell(lat,lng,res). Hierarchical: cell_to_parent/children = instant multi-res rollup; grid_disk(cell,k) = O(k^2) neighbor traversal; polygon_to_cells tessellates India/IGP/fire-zone polygons. Equal-area, no projection distortion. PS3: bin CPCB stations + satellite pixels + FIRMS fires into common res-7 cells for O(1) station<->grid<->fire joins (replaces geometric spatial join); hex hotspot binning + neighbor smoothing for HCHO.
- **Inputs:** lat/lon points (stations, pixel centroids, fires), AOI polygons, resolution
- **Outputs:** uint64/hex cell IDs, hex-aggregated stats, neighbor sets
- **Libraries:** `h3 (h3-py v4)`, `h3ronpy`, `duckdb h3 extension`
- **Complexity:** O(1) point-in-cell & parent/child; O(k^2) k-ring; O(cells) polyfill
- **Pros:** O(1) integer join key replaces costly point-in-polygon; equal-area hexagons remove pixel-area bias; hierarchy = instant multi-scale hotspots; fast groupby
- **Cons:** Hex cells don't align to raster grid (area-weighting needed); res choice is a tradeoff; not a substitute for true zonal stats on admin polygons
- **Benchmark:** Hex groupby over millions of pixel rows in DuckDB/pandas is ms-seconds; replaces O(n*m) spatial join with O(n) hash on integer key
- **Recommendation:** H3 res 7 as canonical fusion grid: assign every pixel, station, fire a cell ID; do all joins as integer groupby; grid_disk for hotspot stats; cell_to_parent for IGP rollups.

### Google S2 / geohash / quadkey-XYZ tiles

- **Category:** Spatial index (alternatives / tiling keys)
- **Summary:** S2 (Hilbert spherical cells) & geohash (base32 prefix) give O(1) point->cell and locality-preserving range keys for DB partition/sort. Quadkey/XYZ (z/x/y) is the web-map standard and the addressing for precomputed COG/tile pyramids. For PS3, H3 wins for analytics (equal-area), but XYZ/quadkey is the right key for the map-serving tier, and geohash/Hilbert ordering speeds GeoParquet range reads.
- **Inputs:** lat/lon, zoom/cell level
- **Outputs:** S2 token, geohash string, z/x/y quadkey
- **Libraries:** `s2sphere`, `python-geohash`, `mercantile`, `morecantile`
- **Complexity:** O(1) encode; O(log n) range queries on sorted keys
- **Pros:** Locality-preserving keys = fast 1-D range scans; XYZ native to all map clients; Hilbert/geohash sort speeds GeoParquet predicate pushdown
- **Cons:** S2/geohash cells vary in area (lat-dependent) — worse than H3 for equal-area binning; geohash boundary artifacts
- **Benchmark:** Hilbert-sorted GeoParquet bbox queries read only relevant row groups (orders-of-magnitude less I/O)
- **Recommendation:** XYZ/quadkey for tile-serving; Hilbert-sort GeoParquet outputs (DuckDB ST_Hilbert) for fast range reads; default to H3 for analytics.

### R-tree (STRtree) & KD-tree (cKDTree)

- **Category:** Spatial index (nearest-neighbor / overlap)
- **Summary:** shapely STRtree/rtree give O(log n) bbox-overlap queries (assign stations to admin polygons, clip to India). scipy cKDTree gives O(log n) nearest-station/nearest-pixel queries — critical for IDW interpolation, matching CPCB stations to nearest satellite cell, and building validation pairs. In-memory and very fast for ~1500 CPCB stations + India grid.
- **Inputs:** point/polygon geometries, query points
- **Outputs:** candidate index lists, nearest-neighbor indices & distances
- **Libraries:** `shapely.STRtree`, `rtree`, `scipy.spatial.cKDTree`, `geopandas.sindex`
- **Complexity:** Build O(n log n); query O(log n)
- **Pros:** Fastest exact nearest-neighbor for station<->pixel matching & IDW; query_ball_point for radius; mature, light
- **Cons:** In-memory only (not planetary-scale alone); needs metric CRS for true distances
- **Benchmark:** cKDTree nearest-station over 1500 stations + millions of grid points <1s; STRtree ~100x faster than naive O(n*m)
- **Recommendation:** cKDTree for nearest-CPCB-station matching & IDW gap-fill; STRtree/geopandas.sindex for clip/join to India boundaries. Project to UTM 43N/44N for metric distances.

### Cloud-Optimized GeoTIFF (COG) + HTTP range requests

- **Category:** Cloud-native format (raster, partial read)
- **Summary:** Internally tiled+overviewed GeoTIFF; clients fetch only needed byte ranges (tiles/zoom) via HTTP range requests — O(1)-ish partial reads instead of full download. Perfect for serving daily India AQI/HCHO maps and on-demand window reads during ML tiling. Pair with TiTiler for dynamic XYZ tiles and rioxarray for windowed reads.
- **Inputs:** Georeferenced rasters (exported daily AQI/HCHO/AOD grids)
- **Outputs:** Range-readable COGs; dynamic XYZ tiles via TiTiler
- **Libraries:** `rio-cogeo`, `rio-tiler`, `rioxarray`, `titiler`, `gdal`
- **Complexity:** O(1) per-tile range read; O(window) partial reads
- **Pros:** Serve national maps without TB downloads; windowed reads feed CNN patch generation cheaply; works directly on S3/GCS
- **Cons:** Raster-only; rewrite cost to make COGs; many small daily files need a STAC index
- **Benchmark:** Viewport tile reads ~KBs via range request vs multi-MB full GeoTIFF; TiTiler serves tiles in tens of ms
- **Recommendation:** Export every GEE daily product as COG; serve interactive India maps via TiTiler XYZ; use rioxarray windowed reads to cut CNN patches without loading full grids.

### Zarr + Kerchunk / VirtualiZarr (chunked N-D arrays)

- **Category:** Cloud-native format (multidim, lazy out-of-core)
- **Summary:** Zarr stores chunked compressed N-D arrays; xarray opens lazily, fetching only requested chunks — ideal for time x lat x lon met/satellite cubes. Kerchunk/VirtualiZarr build a virtual Zarr (JSON/parquet reference) over existing NetCDF/GRIB (MERRA-2, IMDAA, ERA5) WITHOUT copying. ERA5 is already ARCO Zarr (Google ARCO-ERA5, AWS) for serverless time-series access.
- **Inputs:** NetCDF/GRIB reanalysis (ERA5, MERRA-2, IMDAA), multiband satellite stacks
- **Outputs:** Lazy xarray datacube; per-chunk reads; reference JSON/parquet
- **Libraries:** `zarr`, `xarray`, `kerchunk`, `virtualizarr`, `icechunk`, `fsspec`, `gcsfs/s3fs`
- **Complexity:** O(chunks touched) lazy reads; out-of-core via Dask
- **Pros:** Build time-series datacube over decades of reanalysis without download; perfect chunking for per-pixel LSTM extraction; serverless; pairs with Dask
- **Cons:** Chunk-shape choice critical (time- vs spatial-contiguous); reference-generation step; eventual consistency on object stores
- **Benchmark:** Zarr-on-S3 cut query latency minutes->seconds (AWS); ARCO-ERA5 = serverless access to ~12TB without local copy
- **Recommendation:** Use ARCO-ERA5 Zarr directly; VirtualiZarr-wrap IMDAA/MERRA-2 NetCDF. Chunk time-contiguous for LSTM per-station series, spatial-contiguous for CNN patches.

### Parquet / GeoParquet (+ Hilbert sort)

- **Category:** Cloud-native format (tabular vector, columnar)
- **Summary:** Columnar (Geo)Parquet with predicate+projection pushdown and row-group stats enables fast filtered reads over HTTP range requests — read only needed columns/rows. Store CPCB obs, H3-binned pixel tables, fire detections, and ML feature tables. Hilbert/geohash-sort for spatial locality so bbox queries skip row groups. DuckDB reads it natively.
- **Inputs:** Tabular geo data: station obs, per-pixel feature rows (H3 key), fire points
- **Outputs:** Columnar files queryable in-place by DuckDB/pandas/Spark
- **Libraries:** `pyarrow`, `geopandas`, `geoparquet`, `duckdb`, `dask.dataframe`
- **Complexity:** O(rows matching predicate); row-group skipping ~O(log n) on sorted key
- **Pros:** Tiny I/O for filtered ML reads; H3 cell integer column = fast groupby joins; interoperable; HTTP range reads from cloud
- **Cons:** Not for raster; needs partitioning (date/region) for best pushdown
- **Benchmark:** Column+predicate pushdown reads MBs not GBs; partitioned by date, a single-day query touches one partition
- **Recommendation:** Store the fused ML feature matrix (H3 cell, date, satellite cols, met, CPCB target) as GeoParquet partitioned by date, Hilbert-sorted. This is the train/inference table.

### STAC catalog + pystac-client + odc-stac

- **Category:** Discovery / indexing layer
- **Summary:** STAC standardizes search of imagery by bbox/time/collection. pystac-client queries STAC APIs; odc-stac loads matched Items into an xarray datacube (crop/mosaic/resample/reproject). Use to discover/index Sentinel-5P, MODIS/VIIRS, and your own exported daily COGs (static STAC over GCS). Replaces ad-hoc file lists with O(log n) spatiotemporal queries.
- **Inputs:** bbox (India/IGP), datetime range, collection IDs, cloud-cover filters
- **Outputs:** STAC Items (asset hrefs) -> lazy xarray datacube via odc-stac
- **Libraries:** `pystac-client`, `odc-stac`, `stackstac`, `pystac`, `stac-geoparquet`
- **Complexity:** O(log n) indexed spatiotemporal search; lazy load
- **Pros:** One query discovers all assets for a date+AOI; odc-stac -> analysis-ready xarray; stac-geoparquet makes the catalog a DuckDB-queryable table
- **Cons:** Needs a STAC endpoint (Planetary Computer / Earth Search, or self-host static); S5P L3 coverage in public STACs varies (GEE stays primary for S5P)
- **Benchmark:** stac-geoparquet lets DuckDB filter millions of catalog items in ms vs paginated API calls
- **Recommendation:** Index exported daily COGs with a static STAC (+stac-geoparquet). Use Planetary Computer/Earth Search + odc-stac for MODIS/VIIRS; keep GEE primary for S5P.

### Pangeo stack — xarray + Dask + NumPy vectorization

- **Category:** Compute (out-of-core parallel)
- **Summary:** xarray (labeled N-D) + Dask (lazy parallel task graphs) + vectorized NumPy: the standard for out-of-core parallel processing of cubes too big for RAM. Use for per-pixel/per-station feature engineering, regridding, compositing, and batched windowed reads feeding CNN/LSTM. Complements GEE: GEE does global server-side reductions; Pangeo does custom local/cluster compute GEE can't express.
- **Inputs:** Zarr/COG/NetCDF datacubes, chunked
- **Outputs:** Parallel-computed feature arrays, composites, regridded grids
- **Libraries:** `xarray`, `dask`, `numpy`, `rioxarray`, `xarray-spatial`, `flox`, `xesmf`
- **Complexity:** Parallel O(n/p); lazy graph, chunk-streamed
- **Pros:** Handles larger-than-RAM India cubes; flox accelerates zonal groupby; scales laptop->cluster; vectorized = no Python loops
- **Cons:** Chunking/graph tuning needed; you run the cluster (not managed like GEE)
- **Benchmark:** flox groupby + dask map_blocks parallelize zonal/temporal reductions across cores; out-of-core avoids OOM on multi-GB cubes
- **Recommendation:** Use Pangeo for custom preprocessing GEE can't express (model-specific normalization, multi-source regrid to H3/COG grid, LSTM windowing). Pair Dask with VirtualiZarr cubes.

### DuckDB + spatial + H3 extensions

- **Category:** Compute (fast tabular geo query engine)
- **Summary:** Embedded vectorized OLAP engine querying (Geo)Parquet/CSV in-place (incl. over HTTP range requests), with a spatial extension (ST_*, GeoParquet, ST_Hilbert) and an h3 community extension (h3_latlng_to_cell, h3_cell_to_parent). Ideal fast backend for station<->grid joins, H3 aggregation, hotspot SQL, and serving the ML feature table — all in SQL, zero infra.
- **Inputs:** GeoParquet/Parquet/CSV feature & station tables, H3 keys
- **Outputs:** Aggregated/joined results, GeoParquet outputs, hex stats
- **Libraries:** `duckdb`, `duckdb spatial`, `duckdb h3`, `duckdb httpfs`
- **Complexity:** Vectorized scan after pushdown; hash-join O(n+m); row-group skipping on sorted keys
- **Pros:** Single-binary, no server; reads cloud Parquet via range requests; H3 + spatial + columnar speed in one engine; great fusion/aggregation layer
- **Cons:** Single-node (scale-out limited); raster support nascent (keep rasters in COG/Zarr); memory limits on huge joins
- **Benchmark:** Joins/aggregates millions of geo rows in seconds on a laptop; httpfs reads only needed row groups from cloud
- **Recommendation:** Use DuckDB as fusion/serving engine: ATTACH GeoParquet, compute H3 aggregation + station-grid joins + HCHO hotspot SQL (Getis-Ord/quantile). Output GeoParquet for dashboard.

### Precomputed tile pyramids & caching (CDN)

- **Category:** Serving / latency optimization
- **Summary:** For the public daily AQI/HCHO map, precompute XYZ tile pyramids (or cache TiTiler+COG overviews behind a CDN) so every request is an O(1) static-tile/cache hit. PMTiles/MBTiles package the whole pyramid as one range-readable archive served from object storage + CDN — no tile server needed.
- **Inputs:** Daily COG AQI/HCHO grids
- **Outputs:** XYZ/PMTiles pyramid; CDN-cached tiles
- **Libraries:** `gdal2tiles`, `rio-tiler/titiler`, `pmtiles`, `tippecanoe`, `maplibre`
- **Complexity:** O(1) cached tile fetch
- **Pros:** Sub-100ms national map loads; PMTiles needs only static hosting + range requests; scales via CDN
- **Cons:** Precompute/storage cost per day; stale-on-update (invalidate on new daily run)
- **Benchmark:** Static/cached tile <50ms vs dynamic render; PMTiles single-file range reads avoid millions of small objects
- **Recommendation:** Package daily AQI/HCHO maps as PMTiles (or cache TiTiler behind CDN) for the demo dashboard — instant scalable national serving.

### geemap Python API (GEE bridge)

- **Category:** Interface / interactive analysis
- **Summary:** geemap wraps earthengine-api with ipyleaflet/folium: visualize ImageCollections, draw AOIs, run reduceRegions, export to COG/GeoPandas. Bridges GEE server-side outputs to the local Pangeo/DuckDB/H3 pipeline (ee_to_geopandas, ee_export_image, zonal_stats).
- **Inputs:** GEE assets, AOIs, reducers
- **Outputs:** Interactive maps, exported COGs/GeoJSON/GeoDataFrames, zonal-stat tables
- **Libraries:** `geemap`, `earthengine-api`, `eemont`, `geopandas`
- **Complexity:** Inherits GEE O(1)-for-analyst; local export O(size)
- **Pros:** Fast interactive prototyping of PS3 reductions; one-call export GEE->cloud-native formats; zonal_stats helper for station extraction
- **Cons:** Export size limits; notebook-oriented (script for production)
- **Benchmark:** geemap.zonal_statistics over India admin units returns a downloadable table in one call
- **Recommendation:** Use geemap for dev, station zonal stats, and as the export bridge GEE->COG/GeoParquet feeding DuckDB/H3/Pangeo and the ML stage.

**SOTA references:**

- GEE Sentinel-5P catalog: COPERNICUS/S5P/OFFL/L3_HCHO, /L3_NO2, /L3_SO2, /L3_CO, /L3_O3 (+NRTI/) — developers.google.com/earth-engine/datasets/catalog/sentinel-5p (0.01deg, daily)
- GEE reducers & performance: ee.Image.reduceRegions, reduceRegion; set scale, maxPixels, tileScale; Coding Best Practices — developers.google.com/earth-engine/guides/best_practices
- h3-py v4 (latlng_to_cell, cell_to_parent/children, grid_disk, polygon_to_cells) — uber.github.io/h3-py; h3geo.org; h3ronpy for vectorized numpy/arrow
- DuckDB spatial extension (ST_*, GeoParquet, ST_Hilbert) + h3 community extension — duckdb.org/docs/extensions/spatial; cloudnativegeo.org Hilbert-GeoParquet
- COG + TiTiler + rio-tiler/rio-cogeo for range-request serving — cogeo.org; developmentseed.org/titiler
- Zarr + Kerchunk + VirtualiZarr virtual datacubes — virtualizarr.readthedocs.io; projectpythia.org/kerchunk-cookbook
- ARCO-ERA5 cloud-optimized Zarr — github.com/google-research/arco-era5; GEE ECMWF/ERA5_LAND/HOURLY & ECMWF/ERA5/DAILY
- STAC discovery: pystac-client + odc-stac + stac-geoparquet; Microsoft Planetary Computer / Element84 Earth Search — stacspec.org
- INSAT-3D AOD operational via ISRO MOSDAC/VEDAS; Mishra et al. 2018 JGR 'Retrieval of AOD From INSAT-3D Imager' (doi 10.1029/2017JD028116)
- Pangeo (xarray+Dask+flox) + AWS 'Zarr on S3 cuts query latency minutes->seconds' — pangeo.io

---

## Visualization & web-serving stack for surface AQI / HCHO hotspot maps + time series over India (BAH 2026 PS3 / ISRO) — judges score visualization quality heavily

*Pipeline stage: **Visualization** · 14 methods.*

**Recommended stack:**

- **Data backbone: rioxarray/xarray/rasterio align INSAT AOD + S5P TROPOMI (NO2/SO2/CO/O3/HCHO) + ERA5 to a common grid; write daily AQI/HCHO as Cloud-Optimized GeoTIFFs via rio-cogeo**
- **Notebook EDA + GEE viz: geemap (split-map, time slider, inspector) for S5P/INSAT/MODIS; leafmap.add_cog_layer for self-produced AQI COGs; matplotlib+cartopy+contextily for report figures**
- **INTERACTIVE architecture (primary): Streamlit multipage — date slider drives leafmap/MapLibre COG served by TiTiler (dynamic palette + /point AQI query), pydeck ArcLayer/TripsLayer for fire->HCHO wind transport, lonboard for dense FIRMS points, plotly+Taylor for CPCB validation; deploy HF Spaces / Streamlit Cloud**
- **STATIC-EXPORT architecture (polished public deliverable): COG -> precomputed XYZ/PMTiles (rio-cogeo + tippecanoe vector PMTiles) -> MapLibre GL JS + deck.gl MapboxOverlay, fully static on GitHub Pages / Vercel (no token, no server)**
- **Raster tile server: TiTiler (titiler.core + titiler.mosaic, MosaicJSON for the daily time dim); terracotta as alternative for many date x pollutant keyed rasters; avoid GeoServer for a hackathon**
- **Big-data / xarray-native page (optional): HoloViz Panel + GeoViews + Datashader for dense HCHO/fire fields and the full time-cube with an auto time-slider**
- **Dense vector wow-layer: lonboard (deck.gl+GeoArrow) for whole-India FIRMS fire clouds and wind-vector/back-trajectory fields (3M pts ~2.5s)**
- **Time-series animation: xarray + matplotlib FuncAnimation / xmovie -> burning-season HCHO+fire+wind MP4 for the demo reel; interactive sliders in Streamlit/kepler/folium**
- **Validation viz: SkillMetrics/custom Taylor diagram + 1:1 hexbin density scatter (R/RMSE/MAE annotated) + per-CPCB-station residual map — mirrors PS3 scoring**
- **India geometry: GADM v4.1 (gadm41_IND L1/L2) for dev, switch to Survey of India / Bhuvan boundaries for final ISRO submission; explicit IGP + forest-fire-belt highlight overlays; folium TimeSliderChoropleth single-HTML fallback**

**Key findings:**

- Two-track delivery wins: ship a STATIC MapLibre+PMTiles+deck.gl site (no server, hosts on GitHub Pages/Vercel) AND an INTERACTIVE Streamlit/TiTiler app (live palette + per-pixel AQI /point query + date slider). Keep a folium single-HTML fallback.
- Visualization is heavily scored; lead with three flagship views: animated daily surface-AQI raster over India + IGP inset; HCHO hotspot map with FIRMS fire overlay + deck.gl ArcLayer wind-transport arcs; validation page with Taylor diagram + 1:1 scatter (R/RMSE/MAE vs CPCB).
- lonboard (deck.gl+GeoArrow) is the only in-notebook tool that interactively renders whole-India FIRMS fire clouds and wind-vector fields at 10^5-10^6 pts (3M pts in 2.5s where ipyleaflet/pydeck crashed). Use it for the biomass-burning objective wow factor.
- TiTiler (install submodules titiler.core/.mosaic; metapackage dropped late 2025) dynamically tiles AQI COGs reading only needed bytes, with live colormap/rescale and a /cog/point click-to-query AQI endpoint; MosaicJSON handles the daily stack. terracotta is the simpler date x pollutant alternative.
- Verified GEE asset IDs: HCHO COPERNICUS/S5P/{NRTI,OFFL}/L3_HCHO; NO2 .../L3_NO2; SO2/CO/O3 analogous; AER_AI aerosol index. Use OFFL (better quality, full-orbit) for daily maps, NRTI for low latency; qa_value>0.5 mask. geemap renders these directly with split-map + time slider.
- Taylor diagram (SkillMetrics or Copin's recipe) collapses R, centered RMSD and std-dev ratio into one plot mirroring PS3 RMSE/R/MAE scoring; pair with hexbin 1:1 scatter + per-station CPCB residual map. The single most score-relevant figure.
- deck.gl ArcLayer/TripsLayer (pydeck or MapLibre MapboxOverlay) is the cleanest fire->downwind HCHO transport viz using ERA5/IMDAA winds; HexagonLayer gives extruded 3D AQI bins. For xarray cubes, HoloViz Panel+GeoViews+Datashader renders dense fields with an auto time-slider.
- India cartography caution for ISRO judges: GADM v4.1 (gadm41_IND L1 states / L2 districts) is fastest for dev but may not match the official Indian position; switch to Survey of India / Bhuvan boundaries for submission, and prominently highlight the Indo-Gangetic Plain + forest-fire belts.

### geemap (GEE-native interactive maps)

- **Category:** Python interactive mapping (jupyter/colab)
- **Summary:** ipyleaflet wrapper over Google Earth Engine. Visualizes S5P TROPOMI / INSAT AOD / MODIS ee.Image(Collection) without download. Built-in split-panel maps, linked maps, add_time_slider, ts_inspector for per-pixel HCHO query. Map.addLayer + palettes + add_colorbar. Exports timelapse GIF/MP4 via ee_to_video / cartoee.
- **Inputs:** ee.Image/ee.ImageCollection (S5P, INSAT, MODIS, ERA5), vis params, AOI FeatureCollection
- **Outputs:** Interactive map in notebook; GIF/MP4 timelapse; PNG via cartoee; HTML export
- **Libraries:** `geemap`, `earthengine-api`, `ipyleaflet`, `cartoee`
- **Complexity:** Low — runs in Colab; needs GEE auth
- **Pros:** Zero data egress (compute on GEE); fastest S5P asset-IDs-to-map; split-map AQI-vs-validation; time slider built-in
- **Cons:** Needs GEE; ipyleaflet weak on large vector overlays (use lonboard for fires); not standalone web app
- **Benchmark:** GEE community standard 2024-2025 (CVPR/FOSS4G workshops)
- **Recommendation:** USE for EDA, daily-AQI time slider, HCHO seasonal frames, split-map satellite-AQI vs CPCB. Primary notebook viz tool.

### leafmap

- **Category:** Python interactive mapping (backend-agnostic)
- **Summary:** GEE-decoupled sibling of geemap. add_cog_layer() streams a remote/local COG via TiTiler; split_map(left,right) for before/after; add_time_slider; backends include ipyleaflet, folium, maplibre, plotly, kepler, pydeck. leafmap.maplibregl gives GL/3D + offline HTML export — the bridge to serve self-produced AQI COGs.
- **Inputs:** Local/remote COG URL, GeoDataFrame, xarray DataArray, CSV points
- **Outputs:** Interactive map (chosen backend), HTML export, MapLibre GL scene
- **Libraries:** `leafmap`, `localtileserver`, `rio-tiler`, `maplibre`
- **Complexity:** Low-medium
- **Pros:** One API, swappable backends; add_cog_layer is cleanest way to show YOUR predicted AQI rasters; MapLibre backend = GL + offline HTML
- **Cons:** Some features need heavy optional deps; maplibre backend API still maturing
- **Benchmark:** Widely used 2024-2025; leafmap.org
- **Recommendation:** USE to visualize self-produced AQI/HCHO COGs and for the static MapLibre HTML export deliverable.

### folium + branca (TimeSliderChoropleth)

- **Category:** Lightweight static/HTML maps (Leaflet)
- **Summary:** Leaflet wrapper producing self-contained HTML — a no-server demo. TimeSliderChoropleth/TimestampedGeoJson animate daily AQI choropleths or fire points over time. ImageOverlay for static AQI PNG+bounds; HeatMap for HCHO density; Dual_map for side-by-side. Portable single .html for GitHub Pages submission.
- **Inputs:** GeoJSON (district AQI), PNG overlay + bounds, point CSV (fires), styledict
- **Outputs:** Single portable offline .html (easy to host/submit)
- **Libraries:** `folium`, `branca`, `folium.plugins`
- **Complexity:** Low
- **Pros:** Self-contained HTML = bulletproof demo, no backend; time-animated choropleth out of box; trivial GitHub Pages hosting
- **Cons:** Leaflet (no GPU) slow above ~10k features; raster only as pre-rendered PNG overlay
- **Benchmark:** Mature, ubiquitous
- **Recommendation:** USE for the portable HTML fallback submission (animated daily district AQI + FIRMS fire dots). Reliability insurance.

### lonboard

- **Category:** GPU vector rendering in Jupyter (deck.gl+GeoArrow)
- **Summary:** Renders millions of points/lines/polygons in-notebook via deck.gl over a binary GeoArrow/GeoParquet pipeline (no GeoJSON text). Benchmark: 3M points in 2.5s where ipyleaflet/pydeck crashed. ScatterplotLayer (FIRMS fires), PathLayer (wind vectors/back-trajectories), SolidPolygonLayer (district AQI), HeatmapLayer (HCHO density). Built on anywidget.
- **Inputs:** GeoPandas GeoDataFrame / pyarrow GeoArrow / GeoParquet; per-feature color & size arrays
- **Outputs:** GPU map widget in Jupyter; exportable standalone HTML
- **Libraries:** `lonboard`, `geopandas`, `pyarrow`, `shapely`
- **Complexity:** Medium (GeoArrow/GeoPandas inputs)
- **Pros:** Only viable tool for whole-India FIRMS fire clouds (10^5-10^6 pts) interactively; binary pipeline = instant; impresses judges
- **Cons:** Vector-focused (rasters need separate tile layer); newer API
- **Benchmark:** FOSS4G 2025; 3M pts/2.5s vs crashes (DevelopmentSeed)
- **Recommendation:** USE for dense FIRMS fire overlays and wind-vector/trajectory fields. The wow layer for the biomass-burning objective.

### deck.gl + pydeck

- **Category:** WebGL/WebGPU large-data rendering
- **Summary:** deck.gl is the JS WebGL2/WebGPU layer engine; pydeck binds it to Python. HeatmapLayer/ScreenGridLayer (HCHO/fire density), HexagonLayer/GridLayer (extruded 3D AQI bins), ArcLayer/LineLayer (pollutant transport source→receptor), TripsLayer (animated air-parcel trajectories), BitmapLayer (raster AQI). Pairs with MapLibre/Mapbox basemap.
- **Inputs:** JSON/Arrow feature arrays, getPosition/getColor/getElevation accessors
- **Outputs:** GL canvas in web app or notebook (pydeck)
- **Libraries:** `pydeck`, `deck.gl`, `@deck.gl/layers`, `@deck.gl/aggregation-layers`
- **Complexity:** Medium-high (JS for full control)
- **Pros:** Best-in-class transport ARCs and 3D hexbin AQI; embeds in Dash/Streamlit; ArcLayer shows fire→downwind HCHO transport
- **Cons:** pydeck less performant than lonboard for raw points; full power needs JS
- **Benchmark:** Industry standard (Uber/Foursquare)
- **Recommendation:** USE ArcLayer/TripsLayer for wind transport and HexagonLayer for 3D AQI; embed via pydeck in the dashboard.

### kepler.gl (keplergl python)

- **Category:** No-code geospatial exploration (deck.gl)
- **Summary:** Drag-and-drop deck.gl app; keplergl widget loads GeoDataFrames/CSV. Built-in time-playback animation, point/heatmap/grid/hexbin/arc layers, dual-map split, brushing/filtering. Export config+data to standalone HTML. Fast route to an impressive animated multi-layer map (AQI+fires+wind) without writing render code.
- **Inputs:** CSV/GeoJSON/GeoDataFrame; saved kepler config JSON
- **Outputs:** Interactive app; exportable HTML + config
- **Libraries:** `keplergl`, `pandas`, `geopandas`
- **Complexity:** Low (GUI); config JSON for reproducibility
- **Pros:** Time animation + multi-layer + split-map with zero code; great for demo video; HTML export
- **Cons:** Less programmatic control; config fiddly; heavier HTML
- **Benchmark:** Mature Foursquare/Uber tool
- **Recommendation:** OPTIONAL fast-win: an animated AQI+fire+wind kepler HTML for the pitch/demo reel.

### MapLibre GL JS + PMTiles/COG tiles

- **Category:** Production web-GL basemap + raster/vector tiles
- **Summary:** Open-source (no token) WebGL engine — the static-export interactive deliverable. Consumes raster XYZ/PMTiles of AQI/HCHO COGs and vector PMTiles (districts/fires). raster-color expression for AQI palette; fill-extrusion 3D; symbol layers for wind barbs; deck.gl MapboxOverlay injects deck layers on the same canvas. PMTiles = single-file archive servable from static hosting.
- **Inputs:** Raster tiles (rio-tiler/TiTiler or precomputed), PMTiles, GeoJSON, style.json
- **Outputs:** Fast interactive web map, fully static-hostable
- **Libraries:** `maplibre-gl`, `pmtiles`, `@deck.gl/mapbox`, `rio-cogeo`
- **Complexity:** Medium (JS/HTML)
- **Pros:** No API key, no server with PMTiles; smooth GL; combines raster AQI + vector + deck overlay; cheap hosting
- **Cons:** JS; must precompute tiles for static path
- **Benchmark:** Standard open web-mapping 2025
- **Recommendation:** USE as the STATIC-EXPORT architecture: COG→PMTiles + MapLibre + deck.gl overlay on GitHub Pages/Vercel. Primary polished public deliverable.

### TiTiler + rio-tiler over COG

- **Category:** Dynamic raster tile server (cloud-native)
- **Summary:** FastAPI app (titiler.core/.application/.mosaic — metapackage dropped late 2025, install submodules) dynamically tiles COGs reading only needed bytes; on-the-fly rescale/colormap/expression. Endpoints: /cog/tiles, /cog/tilejson.json, /cog/statistics, /cog/point (per-pixel AQI query); MosaicJSON for daily AQI time stacks. Live-switch NO2/HCHO/AQI palettes via URL.
- **Inputs:** COG/.tif (rio-cogeo converts your AQI rasters), MosaicJSON for time stacks
- **Outputs:** XYZ/WMTS tiles, tilejson, point values, statistics — consumed by MapLibre/leafmap/folium
- **Libraries:** `titiler.core`, `titiler.mosaic`, `rio-tiler`, `rio-cogeo`, `fastapi`
- **Complexity:** Medium (deploy FastAPI)
- **Pros:** Live colormap/rescale; per-pixel AQI /point query; MosaicJSON handles daily time dim; any frontend
- **Cons:** Needs a running server (not pure static); COG conversion step
- **Benchmark:** DevelopmentSeed standard for cloud raster tiling
- **Recommendation:** USE for INTERACTIVE architecture raster backend (dynamic palettes + /point query). For static path, precompute tiles instead.

### terracotta

- **Category:** Lightweight raster tile server (DB-indexed)
- **Summary:** Pure-Python XYZ tile server pre-indexing many rasters in SQLite/MySQL by metadata keys (date, pollutant) — fits 'daily AQI, 365 days x 6 pollutants'. On-the-fly colormaps; simpler ops than GeoServer.
- **Inputs:** Folder of (C)OGs + metadata keys; SQLite index
- **Outputs:** XYZ tiles, colormap-styled PNG
- **Libraries:** `terracotta`
- **Complexity:** Low-medium
- **Pros:** Best ergonomics for keyed time-series stacks (date x pollutant); minimal config; embeddable serverless
- **Cons:** Less feature-rich than TiTiler (no virtual mosaics, fewer expressions)
- **Benchmark:** DHI / established
- **Recommendation:** ALTERNATIVE to TiTiler when data is many discrete daily rasters keyed by (date, pollutant). Pick one tile server.

### Streamlit + leafmap/pydeck/plotly

- **Category:** Python dashboard / web app
- **Summary:** Fastest Python-to-web-app. st.pydeck_chart (deck.gl), st_folium, leafmap components, st.plotly_chart. st.slider/select_slider drives a date slider re-rendering the daily AQI map + validation panel. streamlit-keplergl for kepler. Multipage: (1) Daily AQI, (2) HCHO hotspots, (3) Validation. Deploy on Streamlit Cloud / HF Spaces.
- **Inputs:** Predicted AQI arrays/COGs, CPCB validation DataFrame, fire/wind GeoDataFrames
- **Outputs:** Hosted interactive multi-page dashboard
- **Libraries:** `streamlit`, `streamlit-folium`, `leafmap`, `pydeck`, `plotly`
- **Complexity:** Low
- **Pros:** Minimal code, fast to impressive; date slider + tabs; free hosting; embeds deck/kepler/folium/plotly
- **Cons:** Full-rerun model can lag; less layout control than Dash; state quirks
- **Benchmark:** Dominant hackathon dashboard tool 2024-2025
- **Recommendation:** PRIMARY interactive deliverable: Streamlit + leafmap COG (TiTiler) + pydeck arcs + plotly/Taylor validation. Best effort/impact ratio.

### Plotly Dash

- **Category:** Python dashboard (production callbacks)
- **Summary:** Callback-driven app with precise layout; dash-leaflet (Leaflet incl. TimeSlider, COG via tiles) and dash-deck (deck.gl) for maps; plotly for time series, validation scatter, polar Taylor. Better than Streamlit for tightly-linked interactions (click station -> update scatter/time series).
- **Inputs:** Same as Streamlit; callbacks wire components
- **Outputs:** Interactive multi-panel web app
- **Libraries:** `dash`, `dash-leaflet`, `dash-deck`, `plotly`, `dash-bootstrap-components`
- **Complexity:** Medium
- **Pros:** Fine-grained linked interactions (map click <-> scatter <-> time series); production-grade; dash-deck for GL
- **Cons:** More boilerplate than Streamlit; slower to prototype
- **Benchmark:** Mature enterprise standard
- **Recommendation:** ALTERNATIVE to Streamlit when you need linked click-driven CPCB station drilldowns. Otherwise Streamlit wins on speed.

### HoloViz Panel + GeoViews + Datashader

- **Category:** Dashboard + server-side big-data rendering
- **Summary:** Panel app; GeoViews+hvPlot for xarray-native maps; Datashader rasterizes millions of points/large grids server-side into images (whole-India fire clouds, dense grids, no GPU needed). Native xarray/dask integration matches AQI/HCHO cubes. rasterize()/datashade for instant big-data maps; a time dimension auto-binds to a Panel slider.
- **Inputs:** xarray DataArray/Dataset (model output cube time x lat x lon), GeoDataFrames
- **Outputs:** Interactive Bokeh/Panel dashboard, server-rendered big-data tiles
- **Libraries:** `panel`, `holoviews`, `geoviews`, `hvplot`, `datashader`
- **Complexity:** Medium
- **Pros:** xarray/dask-native (matches data model); Datashader = no-overflow rendering of huge fields; auto time-slider from a dim
- **Cons:** Steeper API/concept curve; styling less turnkey
- **Benchmark:** PyData/Anaconda standard for big geospatial
- **Recommendation:** STRONG fit if outputs are xarray cubes; Datashader handles dense HCHO/fire fields. Use for the heavy time-cube exploration page.

### matplotlib + cartopy + contextily

- **Category:** Static cartography for report/PPT
- **Summary:** cartopy adds CRS/projection + coastlines/borders to matplotlib (PlateCarree for India + state shapefiles); contextily adds basemap tiles behind GeoDataFrames (Web Mercator). For clean static daily-AQI panels, HCHO hotspot maps, multi-panel seasonal composites, IGP zoom insets, report figures. geemap.cartoee bridges GEE->cartopy.
- **Inputs:** 2D arrays/xarray, GeoDataFrames, extent, CRS
- **Outputs:** PNG/PDF/SVG figures (300+ dpi for report)
- **Libraries:** `matplotlib`, `cartopy`, `contextily`, `geopandas`, `cartoee`
- **Complexity:** Low-medium
- **Pros:** Publication quality; full control of palettes/colorbars/IGP insets; required for written-report figures judges score
- **Cons:** Static; cartopy install can be fiddly
- **Benchmark:** Scientific-paper standard
- **Recommendation:** USE for ALL report/PPT figures and IGP highlight insets. Make these crisp — figure quality is scored.

### Taylor diagram + validation scatter

- **Category:** Model validation visualization
- **Summary:** Taylor diagram summarizes correlation R, centered RMSD, std-dev ratio of model vs CPCB on one polar plot — directly mirrors PS3 RMSE/R/MAE scoring (SkillMetrics.taylor_diagram or Copin's matplotlib recipe). Pair with 1:1 hexbin density scatter (predicted vs observed AQI + R/RMSE/MAE annotation), per-station residual maps, bias-by-AQI-bin bars.
- **Inputs:** Paired predicted vs CPCB observed arrays (per station/pollutant/AQI)
- **Outputs:** Taylor diagram, 1:1 scatter, residual/bias maps
- **Libraries:** `skillmetrics`, `matplotlib`, `scipy.stats`, `seaborn`
- **Complexity:** Low-medium
- **Pros:** Single figure communicates the scored metrics; judges instantly see model skill; multi-station markers on one Taylor plot
- **Cons:** Custom styling code; needs clean paired validation set
- **Benchmark:** Standard atmospheric model evaluation (Taylor 2001)
- **Recommendation:** MUST-HAVE validation page: Taylor diagram + 1:1 density scatter (R/RMSE/MAE) per pollutant + CPCB station residual map. Mirrors scoring directly.

**SOTA references:**

- GEE Sentinel-5P catalog: developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_HCHO (+ L3_NO2/SO2/CO/O3, NRTI variants) — verified asset IDs
- lonboard: developmentseed.org/lonboard + github.com/developmentseed/lonboard — GeoArrow/deck.gl, 3M pts/2.5s benchmark; FOSS4G 2025 talk
- TiTiler: developmentseed.org/titiler + github.com/developmentseed/titiler — dynamic COG tiling; CHANGES.md notes metapackage dropped late 2025 (use titiler.core/.application/.mosaic)
- rio-tiler: github.com/cogeotiff/rio-tiler — core raster read/tiling engine under TiTiler
- geemap: geemap.org (split-map, time slider, cartoee, ts_inspector) — GEE Workshop/CVPR 2025 materials
- leafmap: leafmap.org — add_cog_layer, split_map, multi-backend (maplibre/pydeck/kepler)
- MapLibre GL JS + PMTiles (protomaps) — static-hostable WebGL tiles; deck.gl @deck.gl/mapbox MapboxOverlay for hybrid raster+GL-vector
- deck.gl / pydeck (Foursquare/Uber): deck.gl/docs — ArcLayer/TripsLayer/HexagonLayer/HeatmapLayer for transport, 3D AQI, density
- Taylor (2001) diagram; SkillMetrics python (github.com/PeterRochford/SkillMetrics) and Yannick Copin matplotlib recipe — R/RMSE/std summary
- GADM v4.1 (gadm.org, gadm41_IND levels 0/1/2); Survey of India / Bhuvan for official submission boundaries

---

## Recommended Method Stack (Consolidated)

The single strongest, end-to-end recommendation distilled from each topic — a prioritized pipeline from acquisition through serving.

1. **AQI Computation → [O(1) vectorized lookup-table implementation](#o1-vectorized-lookup-table-implementation)** — Implement as a single function aqi(conc_dict)->aqi_grid,responsible_grid. Use float32 for grid; precompute constants once (no per-cell branching).
2. **Acquisition / Preprocess → [CNN / LSTM / CNN-LSTM deep hybrids](#cnn--lstm--cnn-lstm-deep-hybrids)** — Primary PS3 submission model. Physics-guided channels + ConvLSTM; pretrain on MERRA-2 PM2.5 then fine-tune on CPCB; ensemble with LightGBM to win RMSE/R/MAE.
3. **ML Prediction → [ConvLSTM / SA-ConvLSTM (self-attention ConvLSTM)](#convlstm--sa-convlstm-self-attention-convlstm)** — Recommended core for Objective-1 gridded AQI maps. Use SA-ConvLSTM (temporal self-attention) + met covariate channels.
4. **ML Prediction → [Gradient Boosting (XGBoost / LightGBM / CatBoost)](#gradient-boosting-xgboost--lightgbm--catboost)** — Primary classical learner and top stacking member. SHAP for hotspot drivers; monotonic constraint on AOD.
5. **Hotspot Detection → [Getis-Ord Gi* (local hot/cold-spot z-score)](#getis-ord-gi-local-hotcold-spot-z-score)** — PRIMARY hotspot engine. Gi* star=True, 999 perms, KNN-8/DistanceBand W, FDR p, classify >99% as core. Run per season (Oct-Nov, Apr-May).
6. **Fire & Transport → [HYSPLIT back/forward trajectory analysis](#hysplit-backforward-trajectory-analysis)** — Run 120 h back-trajectories at 500 & 1000 m AGL, release times matched to S5P ~13:30 LT, ERA5-driven (better IGP BL than GDAS). Automate via splitr (R) or PySPLIT.
7. **Fusion & Gap-fill → [Random Forest / Gradient Boosting (XGBoost, LightGBM)](#random-forest--gradient-boosting-xgboost-lightgbm)** — Primary BASELINE for objective-1 and AOD gap-fill; also the fusion blender combining DINEOF + MERRA-2 + met. Report RMSE/R/MAE vs CPCB; use quantile loss for uncertainty.
8. **Validation → [Leave-Location-Out / Spatial-block CV (PRIMARY)](#leave-location-out--spatial-block-cv-primary)** — PRIMARY Obj-1 validation: spatial-block k-fold with variogram-sized buffer; show fold-assignment map.
9. **Fast-Platform / O(1) → [Google Earth Engine (GEE) — server-side lazy planetary-scale compute](#google-earth-engine-gee--server-side-lazy-planetary-scale-compute)** — PRIMARY ingestion+reduction engine. Composite, gap-fill, co-locate stations server-side; export compact tables/COGs. scale=1113, tileScale=8, combine mean+stdDev reducers.
10. **Visualization → [Streamlit + leafmap/pydeck/plotly](#streamlit--leafmappydeckplotly)** — PRIMARY interactive deliverable: Streamlit + leafmap COG (TiTiler) + pydeck arcs + plotly/Taylor validation. Best effort/impact ratio.

> Full trade-offs, benchmarks, and alternatives for every choice above are detailed in the [Method Catalog by Topic](#method-catalog-by-topic).

