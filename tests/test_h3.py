"""Tests for the H3 fusion-key wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqi_india.features import h3_index

# Delhi, roughly.
DELHI = (28.6139, 77.2090)


def test_api_version_detected() -> None:
    assert h3_index.h3_api_version() in (3, 4)


def test_point_to_cell_is_stable_and_correct_resolution() -> None:
    c1 = h3_index.latlng_to_cell(*DELHI, res=7)
    c2 = h3_index.latlng_to_cell(*DELHI, res=7)
    assert c1 == c2  # deterministic
    assert h3_index.get_resolution(c1) == 7


def test_centroid_roundtrip_is_close() -> None:
    cell = h3_index.latlng_to_cell(*DELHI, res=7)
    lat, lon = h3_index.cell_to_latlng(cell)
    # res-7 cells are ~5 km; centroid must be within ~0.1 deg of the point.
    assert abs(lat - DELHI[0]) < 0.1
    assert abs(lon - DELHI[1]) < 0.1


def test_cell_to_parent_roundtrip() -> None:
    cell = h3_index.latlng_to_cell(*DELHI, res=7)
    parent = h3_index.cell_to_parent(cell, 4)
    assert h3_index.get_resolution(parent) == 4
    # The child's centroid must index back into the same parent.
    lat, lon = h3_index.cell_to_latlng(cell)
    child_parent = h3_index.cell_to_parent(
        h3_index.latlng_to_cell(lat, lon, 7), 4
    )
    assert parent == child_parent


def test_grid_disk_neighborhood_size() -> None:
    cell = h3_index.latlng_to_cell(*DELHI, res=7)
    for k in (0, 1, 2, 3):
        disk = h3_index.grid_disk(cell, k)
        # Hexagonal grid disk has 3k(k+1)+1 cells (pentagon cells excepted).
        assert len(disk) == h3_index.grid_disk_size(k)
        assert cell in disk


def test_cells_for_points_dataframe() -> None:
    df = pd.DataFrame(
        {
            "lat": [28.6139, 19.0760, np.nan],
            "lon": [77.2090, 72.8777, 75.0],
        }
    )
    out = h3_index.cells_for_points(df, res=7, out_col="h3_res7")
    assert out["h3_res7"].iloc[0] == h3_index.latlng_to_cell(28.6139, 77.2090, 7)
    assert out["h3_res7"].iloc[1] is not None
    assert out["h3_res7"].iloc[2] is None  # non-finite coords -> None


def test_cells_for_arrays_and_parents() -> None:
    lats = np.array([28.6139, 19.0760])
    lons = np.array([77.2090, 72.8777])
    cells = h3_index.cells_for_arrays(lats, lons, res=7)
    assert cells.shape == (2,)
    parents = h3_index.parents_for_cells(cells, res=5)
    for child, parent in zip(cells, parents, strict=True):
        assert h3_index.cell_to_parent(child, 5) == parent
