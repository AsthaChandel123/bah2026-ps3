#!/usr/bin/env python
"""Thin wrapper around :mod:`aqi_india.demo` — the offline synthetic demo.

The full orchestration lives inside the package at ``aqi_india.demo`` so the
``aqi demo run`` CLI command and this script share one implementation. Run with:

    python scripts/run_demo.py            # defaults (60 days, 120 stations)
    python scripts/run_demo.py --n-days 45 --seed 7
    aqi demo run                          # equivalent, via the Typer CLI

It produces real artifacts on the light dependency set (no credentials/network):
``reports/figures/obj{1,2}_*.png`` and ``docs/RESULTS.md`` with the run's numbers.
"""

from __future__ import annotations

from aqi_india.demo import main

if __name__ == "__main__":
    main()
