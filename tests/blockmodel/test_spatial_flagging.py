from pathlib import Path

import numpy as np
import pyvista as pv
from shapely.geometry import box

from parq_blockmodel import MeshSolid, ParquetBlockModel, PolygonField, RegularGeometry


def make_pbm(tmp_path: Path) -> ParquetBlockModel:
    geometry = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(3, 3, 3),
    )
    return ParquetBlockModel.from_geometry(geometry=geometry, path=tmp_path / "spatial_flags.pbm")


def test_flag_polygon_persists_named_domain(tmp_path: Path):
    pbm = make_pbm(tmp_path)
    polygon = PolygonField.from_shapely(box(0.0, 0.0, 2.0, 2.0), name="lease_a")

    mask = pbm.flag_polygon(polygon, column="lease_domain")
    out = pbm.read(columns=["block_id", "lease_domain"], index=None)

    expected_inside = mask[out["block_id"].to_numpy(dtype=np.int64)]
    actual_inside = out["lease_domain"].eq("lease_a").to_numpy(dtype=bool)
    np.testing.assert_array_equal(actual_inside, expected_inside)


def test_flag_solid_persists_boolean_column(tmp_path: Path):
    pbm = make_pbm(tmp_path)
    solid = MeshSolid.from_pyvista(
        pv.Cube(bounds=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)).triangulate(),
        name="ore_shell",
    )

    mask = pbm.flag_solid(
        solid,
        column="ore",
        inside_value=True,
        outside_value=False,
        as_categorical=False,
    )
    out = pbm.read(columns=["block_id", "ore"], index=None)

    expected = mask[out["block_id"].to_numpy(dtype=np.int64)]
    actual = out["ore"].to_numpy(dtype=bool)
    np.testing.assert_array_equal(actual, expected)

