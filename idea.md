PROBLEM STATEMENT 3

Development of Surface AQI & Identification of HCHO Hotspots over India using Satellite Data
Overview
Air pollution is a major health hazard that's liable for many annual excess deaths worldwide. The concept of air quality index (AQI) that transforms weighted values of individual air pollution related parameters into a single number or set of numbers is widely used for air quality communication and decision making in many countries. Hence the AQI is a rating scale for reporting the Ambient air pollution (CO, NO2, O3, SO2, and PM2.5 etc) recorded at monitoring sites on a particular time scale (e.g., daily). The two main objectives of AQI are (a) to inform and caution the public about the risk of exposure to daily pollution levels and (b) to enforce required regulatory measures by competent authority for immediate local impact. The higher the AQI value, the greater the level of air pollution and health risk. Most of the global population lives further than 100 km from air quality monitors and devoid of air quality information in that area. Hence it is important to develop a surface AQI derived from satellite data to know its spatial distribution on daily basis.

VOC's can be monitored from Global Ozone Monitoring Experiment (GOME), the Scanning Imaging Absorption Spectrometer for Atmospheric Cartography (SCIAMACHY), the Ozone Monitoring Instrument (OMI), and the TROPOspheric Monitoring Instrument (TROPOMI), play pivotal roles in retrieving HCHO columns. Formaldehyde (HCHO) is a key indicator of volatile organic compound (VOC) emissions and plays an important role in ozone formation and atmospheric chemistry. During biomass burning events (such as agricultural residue burning and forest fires), large amounts of VOCs are released, leading to enhanced HCHO concentrations. This problem aims to identify and analyze spatio-temporal hotspots of HCHO over India using satellite observations, with a focus on biomass burning periods.

The National Clean Air Programme (NCAP) is India's long-term, time-bound strategy to reduce air pollution by 20–40% in non-attainment cities. INSAT 3D satellite provides Aerosol Optical Depth data product, from which surface PM2.5 need to be deduced to assess the pollution exposure to the people. The Sustainable Development Goal (SDG) target (11.6) calls for reducing adverse environmental effects in cities. Chintan Shivir 2.0 targets include urban climate, air quality, mitigation, sustainable development. Hence, the problem statement is relevant to ISRO and National priorities.

Objectives
Development of surface Air quality index (AQI) using satellite data — Spatial maps of air quality index over India
High-resolution maps of HCHO hotspots during biomass burning seasons
Identification of major source regions (e.g., Indo-Gangetic Plain, forest fire zones)
Expected Outcomes
Spatial maps of surface AQI over India
Spatial maps of HCHO concentrations and hotspots over India
Identification of major source regions
Analysis on influence of fire activity on HCHO levels through transport
Dataset Required
AOD data from INSAT 3D satellite (https://www.mosdac.gov.in/insat-3d-data-products)
Sentinel-5P (TROPOMI NO2, SO2, CO, O3 & HCHO data) (https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p/ https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/)
Ground based air quality data from CPCB (https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/caaqm-data-repository)
MODIS / VIIRS fire count data (https://firms.modaps.eosdis.nasa.gov/active_fire/)
Reanalysis meteorological data (ERA5/IMDAA/MERRA-2) https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download https://nwp.ncmrwf.gov.in/reanalysis https://disc.gsfc.nasa.gov/datasets?project=MERRA-2
Suggested Tools/Technologies
Python
Google Earth Engine
Python tools for data base creation and use of deep learning models such as CNN, LSTM or hybrid deep learning models such as CNN-LSTM
Expected Solution / Steps to be followed to achieve the objectives
Objective-1

Generate data base of columnar multi pollutant concentrations from satellite data sets
Generate data base of surface measured multi pollutant concentrations (from CPCB) along with meteorological parameters (from IMDAA/ERA-5/MERRA-2)
Develop algorithm using deep learning models to predict ground based multi pollutant concentrations
Validate the predicted concentrations with ground-based data and prepare spatial maps of AQI
Objective-2

Acquire and pre-process satellite HCHO column data
Extract biomass burning periods using fire count datasets
Map spatio-temporal distribution of HCHO concentrations
Identify hotspots using statistical thresholds or clustering methods
Analyse correlation between fire activity and HCHO levels
Assess transport influence using wind/reanalysis data
Evaluation Parameters
Objective-1

Statistical parameters such as RMSE, R, MAE etc. will be used to arrive at the accuracy of predicted parameters
Objective-2

Accuracy and clarity of hotspot detection
Integration of multi-source datasets
Scientific interpretation of results
Visualization quality (Spatial maps over Indian region, time series)
Innovation in methodology
Image representing problem statement
Objective-1

Flow diagram showing columnar satellite data, CPCB surface data, and meteorological data fed into a CNN-LSTM deep learning model to produce spatial maps of surface AQI
Objective-2

Flow diagram showing remote sensing data, fire data, and reanalysis meteorological data processed to produce high-resolution HCHO hotspot maps, source region identification, temporal evolution, and fire-HCHO correlation over India
