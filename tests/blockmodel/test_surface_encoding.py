from pathlib import Path

import numpy as np

from parq_blockmodel import ParquetBlockModel, RasterSurface, RegularGeometry


def make_pbm(tmp_path: Path) -> ParquetBlockModel:
    geometry = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )
    return ParquetBlockModel.from_geometry(geometry=geometry, path=tmp_path / "surface_flags.pbm")


def test_evaluate_surface_persists_column_and_returns_values(tmp_path: Path):
    pbm = make_pbm(tmp_path)
    surface = RasterSurface.from_arrays(
        x_coords=np.array([0.0, 1.0, 2.0], dtype=float),
        y_coords=np.array([0.0, 1.0, 2.0], dtype=float),
        values=np.full((3, 3), 1.5, dtype=float),
        name="topo",
    )

    values = pbm.evaluate_surface(surface, column="topo_fraction")
    out = pbm.read(columns=["block_id", "topo_fraction"], index=None)

    expected = values[out["block_id"].to_numpy(dtype=np.int64)]
    np.testing.assert_allclose(out["topo_fraction"].to_numpy(dtype=float), expected)
    np.testing.assert_allclose(values, np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5], dtype=float))
