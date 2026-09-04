from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parq_blockmodel import LocalGeometry, ParquetBlockModel, RegularGeometry, WorldFrame
from parq_blockmodel.utils.geometry_utils import angles_to_axes


pytest.importorskip("geopandas")


def _make_footprint_pbm(
    tmp_path: Path,
    *,
    populated_cells: set[tuple[int, int, int]] | None = None,
    axis_azimuth: float = 0.0,
    axis_dip: float = 0.0,
    axis_plunge: float = 0.0,
) -> ParquetBlockModel:
    axis_u, axis_v, axis_w = angles_to_axes(
        axis_azimuth=axis_azimuth,
        axis_dip=axis_dip,
        axis_plunge=axis_plunge,
    )
    local = LocalGeometry(corner=(100.0, 200.0, 300.0), block_size=(10.0, 20.0, 5.0), shape=(4, 3, 2))
    base_geometry = RegularGeometry(
        local=local,
        world=WorldFrame(
            axis_u=axis_u,
            axis_v=axis_v,
            axis_w=axis_w,
            crs="EPSG:28350",
        ),
    )
    coords = base_geometry.to_dataframe().to_numpy(dtype=float)
    mins = coords.min(axis=0)
    world_origin = tuple(np.where(mins < 0.0, -mins, 0.0))
    geometry = RegularGeometry(
        local=local,
        world=WorldFrame(
            origin=world_origin,
            axis_u=axis_u,
            axis_v=axis_v,
            axis_w=axis_w,
            crs="EPSG:28350",
        ),
    )
    pbm = ParquetBlockModel.from_geometry(geometry=geometry, path=tmp_path / "footprint_extent.pbm")

    if populated_cells is None:
        return pbm

    dense = pbm.read(columns=["block_id"], index="ijk", dense=True).reset_index()
    mask = dense[["i", "j", "k"]].apply(tuple, axis=1).isin(populated_cells)
    subset_indices = np.flatnonzero(mask.to_numpy(dtype=bool))
    table = pq.read_table(pbm.blockmodel_path)
    sparse_table = table.take(pa.array(subset_indices, type=pa.int64()))
    pq.write_table(sparse_table, pbm.blockmodel_path)
    return ParquetBlockModel(pbm.blockmodel_path)


def test_to_footprint_geodataframe_defaults_to_footprint_type_column(tmp_path):
    pbm = _make_footprint_pbm(tmp_path)

    gdf = pbm.to_footprint_geodataframe()

    assert list(gdf.columns) == ["footprint_type", "geometry"]
    assert gdf["footprint_type"].tolist() == ["dense", "sparse"]


def test_to_footprint_geodataframe_supports_attribute_mapping(tmp_path):
    pbm = _make_footprint_pbm(
        tmp_path,
        populated_cells={
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
            (2, 2, 1),
        },
    )

    gdf = pbm.to_footprint_geodataframe(
        attributes={
            "footprint_type": {"dense": "model_extent", "sparse": "populated_extent"},
            "color": {"dense": "#0000FF", "sparse": "#00FF00"},
        }
    )

    assert list(gdf["footprint_type"]) == ["model_extent", "populated_extent"]
    assert list(gdf["color"]) == ["#0000FF", "#00FF00"]


def test_to_footprint_geodataframe_dense_and_sparse_agree_for_full_models(tmp_path):
    pbm = _make_footprint_pbm(tmp_path)

    gdf = pbm.to_footprint_geodataframe()

    assert gdf.geometry.iloc[0].equals(gdf.geometry.iloc[1])


def test_to_footprint_geodataframe_sparse_extent_reflects_only_populated_cells(tmp_path):
    pbm = _make_footprint_pbm(
        tmp_path,
        populated_cells={
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
            (3, 2, 1),
        },
    )

    gdf = pbm.to_footprint_geodataframe()

    dense_area = gdf.geometry.iloc[0].area
    sparse_area = gdf.geometry.iloc[1].area

    assert sparse_area == pytest.approx(3 * 10.0 * 20.0)
    assert sparse_area < dense_area


def test_to_footprint_geodataframe_sparse_extent_can_be_multipart(tmp_path):
    pbm = _make_footprint_pbm(
        tmp_path,
        populated_cells={
            (0, 0, 0),
            (0, 0, 1),
            (3, 2, 0),
        },
    )

    gdf = pbm.to_footprint_geodataframe()

    assert gdf.geometry.iloc[1].geom_type == "MultiPolygon"


def test_to_footprint_geodataframe_supports_rotated_geometry(tmp_path):
    pbm = _make_footprint_pbm(
        tmp_path,
        populated_cells={
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 1),
        },
        axis_azimuth=30.0,
    )

    gdf = pbm.to_footprint_geodataframe()

    assert gdf.crs is not None
    assert gdf.crs.to_string() == "EPSG:28350"
    x0, y0 = gdf.geometry.iloc[0].exterior.coords[0]
    x1, y1 = gdf.geometry.iloc[0].exterior.coords[1]
    assert not np.isclose(y1 - y0, 0.0)
    assert not np.isclose(x1 - x0, 0.0)


def test_to_footprint_geodataframe_rejects_tilted_logical_columns(tmp_path):
    pbm = _make_footprint_pbm(tmp_path, axis_azimuth=20.0, axis_dip=15.0, axis_plunge=0.0)

    with pytest.raises(ValueError, match="projection onto the \\(i, j\\) plane"):
        pbm.to_footprint_geodataframe()


def test_to_footprint_geodataframe_returns_valid_geometries(tmp_path):
    pbm = _make_footprint_pbm(
        tmp_path,
        populated_cells={
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 1),
            (3, 2, 1),
        },
        axis_azimuth=30.0,
    )

    gdf = pbm.to_footprint_geodataframe()

    assert gdf.geometry.is_valid.all()
    assert set(gdf.geometry.geom_type.unique()).issubset({"Polygon", "MultiPolygon"})


def test_to_footprint_geodataframe_sparse_path_does_not_require_dense_read(tmp_path, monkeypatch):
    pbm = _make_footprint_pbm(
        tmp_path,
        populated_cells={
            (0, 0, 0),
            (0, 0, 1),
            (3, 2, 1),
        },
    )
    assert pbm.is_sparse

    def _fail_read(*args, **kwargs):
        raise AssertionError("Sparse footprint path should not call read().")

    monkeypatch.setattr(pbm, "read", _fail_read)

    gdf = pbm.to_footprint_geodataframe()

    assert len(gdf) == 2
