from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from parq_blockmodel.footprint.common import assert_supported_plan_projection, dissolve_row_runs

_FOOTPRINT_KEYS = ("dense", "sparse")


def _default_attributes() -> dict[str, dict[str, Any]]:
    return {
        "footprint_type": {
            "dense": "dense",
            "sparse": "sparse",
        }
    }


def _normalize_attributes(
    attributes: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if attributes is None:
        return _default_attributes()
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping of output column names to dense/sparse mappings.")

    normalized: dict[str, dict[str, Any]] = {}
    for column, values in attributes.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"Attribute mapping for '{column}' must be a mapping with 'dense' and 'sparse' keys.")
        missing = [key for key in _FOOTPRINT_KEYS if key not in values]
        if missing:
            raise ValueError(f"Attribute mapping for '{column}' is missing required keys: {missing}.")
        unexpected = [key for key in values if key not in _FOOTPRINT_KEYS]
        if unexpected:
            raise ValueError(f"Attribute mapping for '{column}' has unexpected keys: {unexpected}.")
        normalized[column] = {
            "dense": values["dense"],
            "sparse": values["sparse"],
        }
    return normalized


def _dense_row_runs(shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    ni, nj, _ = shape
    return [(0, ni, j) for j in range(nj)] if ni > 0 and nj > 0 else []


def _occupied_flat_ij_indices(blockmodel) -> np.ndarray:
    ni, _, _ = blockmodel.geometry.local.shape
    occupied: set[int] = set()

    for block_ids in blockmodel._iter_block_ids():
        if len(block_ids) == 0:
            continue
        i, j, _ = blockmodel.geometry.ijk_from_row_index(block_ids)
        flat = np.asarray(j, dtype=np.int64) * ni + np.asarray(i, dtype=np.int64)
        occupied.update(np.unique(flat).tolist())

    if not occupied:
        return np.empty(0, dtype=np.int64)
    return np.fromiter(occupied, dtype=np.int64, count=len(occupied))


def _flat_ij_indices_to_row_runs(flat_ij: np.ndarray, ni: int) -> list[tuple[int, int, int]]:
    if flat_ij.size == 0:
        return []

    flat_sorted = np.unique(np.asarray(flat_ij, dtype=np.int64))
    flat_sorted.sort()

    j_values = flat_sorted // ni
    i_values = flat_sorted % ni

    runs: list[tuple[int, int, int]] = []
    current_j = int(j_values[0])
    run_start = int(i_values[0])
    previous_i = int(i_values[0])

    for idx in range(1, len(flat_sorted)):
        j = int(j_values[idx])
        i = int(i_values[idx])
        if j == current_j and i == previous_i + 1:
            previous_i = i
            continue
        runs.append((run_start, previous_i + 1, current_j))
        current_j = j
        run_start = i
        previous_i = i

    runs.append((run_start, previous_i + 1, current_j))
    return runs


def to_footprint_geodataframe(
    blockmodel,
    *,
    attributes: Mapping[str, Mapping[str, Any]] | None = None,
):
    import geopandas as gpd

    assert_supported_plan_projection(blockmodel.geometry)

    dense_attributes = _normalize_attributes(attributes)
    ni, nj, nk = blockmodel.geometry.local.shape
    del nk

    dense_geometry = dissolve_row_runs(blockmodel.geometry, _dense_row_runs((ni, nj, 0)))

    if blockmodel.is_sparse:
        sparse_flat_ij = _occupied_flat_ij_indices(blockmodel)
        sparse_geometry = dissolve_row_runs(
            blockmodel.geometry,
            _flat_ij_indices_to_row_runs(sparse_flat_ij, ni=ni),
        )
    else:
        sparse_geometry = dense_geometry

    records = [
        {
            **{column: values["dense"] for column, values in dense_attributes.items()},
            "geometry": dense_geometry,
        },
        {
            **{column: values["sparse"] for column, values in dense_attributes.items()},
            "geometry": sparse_geometry,
        },
    ]

    return gpd.GeoDataFrame(
        records,
        columns=[*dense_attributes.keys(), "geometry"],
        geometry="geometry",
        crs=blockmodel.geometry.srs,
    )
