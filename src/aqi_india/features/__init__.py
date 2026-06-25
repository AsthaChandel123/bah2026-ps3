"""aqi_india.features — physics-guided feature engineering.

Modules:
    h3_index: H3 universal fusion-key wrappers (Foundation phase).
    physics: physics-guided channels (PBL/RH/FNR/aerosol-type/wind).
    met: meteorological predictor stack from the fused grid cube.
    static_covars: static covariate loaders/synthesizers over the India grid.
    temporal: cyclic/lag/rolling features and sequence builders.
    feature_matrix: the tidy (h3_res7, time) training/inference feature tables.

Submodules are not imported eagerly here so that importing
``aqi_india.features`` stays cheap; import the specific module you need.
"""
