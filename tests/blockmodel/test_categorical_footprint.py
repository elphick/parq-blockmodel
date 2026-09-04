from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parq_blockmodel import LocalGeometry, ParquetBlockModel, RegularGeometry, WorldFrame
from parq_blockmodel.utils.geometry_utils import angles_to_axes


pytest.importorskip("geopandas")


def _make_classification_pbm(
    tmp_path: Path,
    *,
    ordered: bool = False,
    axis_azimuth: float = 0.0,
    axis_dip: float = 0.0,
    axis_plunge: float = 0.0,
) -> ParquetBlockModel:
    axis_u, axis_v, axis_w = angles_to_axes(
        axis_azimuth=axis_azimuth,
        axis_dip=axis_dip,
        axis_plunge=axis_plunge,
    )
    local = LocalGeometry(corner=(100.0, 200.0, 300.0), block_size=(10.0, 20.0, 5.0), shape=(2, 2, 3))
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
    pbm = ParquetBlockModel.from_geometry(geometry=geometry, path=tmp_path / "categorical_footprint.pbm")

    rows = pbm.read(columns=["block_id"], index="ijk", dense=True).reset_index()
    values = np.full(len(rows), pd.NA, dtype=object)

    values[(rows["i"] == 0) & (rows["j"] == 0) & (rows["k"] == 2)] = "Measured"
    values[(rows["i"] == 0) & (rows["j"] == 0) & (rows["k"] == 0)] = "Inferred"
    values[(rows["i"] == 0) & (rows["j"] == 1) & (rows["k"] == 1)] = "Indicated"
    values[(rows["i"] == 1) & (rows["j"] == 0) & (rows["k"] == 0)] = "Inferred"

    if ordered:
        classification = pd.Categorical(
            values,
            categories=["Inferred", "Indicated", "Measured"],
            ordered=True,
        )
    else:
        classification = values

    pbm.write(
        pd.DataFrame({"block_id": rows["block_id"], "classification": classification}),
        merge=True,
    )
    return pbm


def test_to_categorical_geodataframe_respects_explicit_categories_and_ignores_missing(tmp_path):
    pbm = _make_classification_pbm(tmp_path)

    gdf = pbm.to_categorical_geodataframe(
        column="classification",
        categories=["Measured", "Ghost", "Inferred"],
        precedence=["Measured", "Inferred", "Ghost"],
    )

    assert set(gdf["classification"].tolist()) == {"Measured", "Inferred"}
    assert "Ghost" not in set(gdf["classification"].tolist())


def test_to_categorical_geodataframe_applies_precedence_to_overlap(tmp_path):
    pbm = _make_classification_pbm(tmp_path)

    gdf = pbm.to_categorical_geodataframe(
        column="classification",
        precedence=["Inferred", "Indicated", "Measured"],
    )

    assert "Measured" not in set(gdf["classification"].tolist())
    assert set(gdf["classification"].tolist()) == {"Inferred", "Indicated"}


def test_to_categorical_geodataframe_uses_ordered_categorical_order_when_precedence_missing(tmp_path):
    pbm = _make_classification_pbm(tmp_path, ordered=True)

    gdf = pbm.to_categorical_geodataframe(column="classification")

    assert "Measured" not in set(gdf["classification"].tolist())
    assert set(gdf["classification"].tolist()) == {"Inferred", "Indicated"}


def test_to_categorical_geodataframe_handles_all_missing_categories_as_empty(tmp_path):
    pbm = _make_classification_pbm(tmp_path)

    gdf = pbm.to_categorical_geodataframe(
        column="classification",
        categories=["GhostA", "GhostB"],
    )

    assert list(gdf.columns) == ["classification", "geometry"]
    assert gdf.empty


def test_to_categorical_geodataframe_supports_rotated_geometry(tmp_path):
    pbm = _make_classification_pbm(tmp_path, axis_azimuth=30.0)

    gdf = pbm.to_categorical_geodataframe(
        column="classification",
        precedence=["Measured", "Indicated", "Inferred"],
    )

    assert not gdf.empty
    assert gdf.crs is not None
    assert gdf.crs.to_string() == "EPSG:28350"
    first_poly = gdf.geometry.iloc[0]
    x0, y0 = first_poly.exterior.coords[0]
    x1, y1 = first_poly.exterior.coords[1]
    assert not np.isclose(y1 - y0, 0.0)
    assert not np.isclose(x1 - x0, 0.0)


def test_to_categorical_geodataframe_output_polygons_do_not_overlap(tmp_path):
    pbm = _make_classification_pbm(tmp_path)

    gdf = pbm.to_categorical_geodataframe(
        column="classification",
        precedence=["Measured", "Indicated", "Inferred"],
    )
    geometries = gdf.geometry.tolist()

    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            assert geometries[i].intersection(geometries[j]).area <= 1e-9


def test_to_categorical_geodataframe_rejects_tilted_logical_columns(tmp_path):
    pbm = _make_classification_pbm(tmp_path, axis_azimuth=20.0, axis_dip=15.0, axis_plunge=0.0)

    with pytest.raises(ValueError, match="axis_w horizontal components"):
        pbm.to_categorical_geodataframe(column="classification")
