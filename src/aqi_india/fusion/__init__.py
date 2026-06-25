"""Multi-source fusion, gap-filling, bias-correction and regridding.

This subpackage is the core of "using broad data to fill gaps in other data":

* :mod:`~aqi_india.fusion.regrid` — bilinear / nearest regridding onto the common
  lat/lon analysis grid (lazy :mod:`xesmf` for conservative).
* :mod:`~aqi_india.fusion.dineof` — label-free DINEOF (iterative truncated-SVD)
  cloud-gap reconstruction with explained-variance + uncertainty.
* :mod:`~aqi_india.fusion.kriging` — ordinary / regression kriging (lazy
  :mod:`pykrige`) with an always-available cKDTree IDW fallback.
* :mod:`~aqi_india.fusion.inpaint_unet` — partial-conv U-Net inpainting interface
  (lazy :mod:`torch`; DINEOF fallback).
* :mod:`~aqi_india.fusion.biascorrect` — regionalized quantile mapping +
  gradient-boosting ML residual correction against CPCB.
* :mod:`~aqi_india.fusion.harmonize` — multi-sensor stacking with provenance and
  a QA/uncertainty channel; multi-sensor AOD fusion via RF imputation; the
  ``run(cfg)`` chaining entrypoint.
* :mod:`~aqi_india.fusion.gapfill` — :func:`fill_gaps`, the public cascade
  (DINEOF -> IDW -> reanalysis prior) the demo calls.

Everything imports under the light dependency set; heavy backends (torch,
pykrige, xesmf) are imported lazily inside the functions that use them.
"""

from __future__ import annotations

from .gapfill import GapReport, fill_gaps

__all__ = ["fill_gaps", "GapReport"]
