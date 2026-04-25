import os
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel import ParquetBlockModel, RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.utils import create_demo_blockmodel
from parq_blockmodel.utils.demo_block_model import create_toy_blockmodel
from parq_blockmodel.utils.geometry_utils import angles_to_axes, rotate_points
from parq_blockmodel.utils.spatial_encoding import decode_world_coordinates, get_world_id_encoding_params


def test_from_empty_parquet_raises_error(tmp_path):
    import pandas as pd
    parquet_path = tmp_path / "bar.parquet"
    parquet_path.touch()
    pd.DataFrame(columns=["x", "y", "z"]).to_parquet(parquet_path)

    with pytest.raises(ValueError, match="Cannot infer block size/shape from degenerate centroid coordinates."):
        ParquetBlockModel.from_parquet(parquet_path)


def test_create_demo_block_model_creates_file(tmp_path):
    filename = tmp_path / "demo_block_model.parquet"
    model = ParquetBlockModel.create_demo_block_model(filename)
    assert isinstance(model, ParquetBlockModel)
    # Canonical ParquetBlockModel files now use a single ".pbm" suffix.
    assert model.blockmodel_path == filename.with_suffix(".pbm")
    assert model.name == filename.stem
    assert filename.exists()
    # Check if the file is a Parquet file
    assert filename.suffix == ".parquet"
    # Clean up the created file
    os.remove(filename) if filename.exists() else None
    os.remove(model.blockmodel_path) if filename.exists() else None


def test_block_model_with_categorical_data(tmp_path):
    # Create a Parquet file with categorical data
    parquet_path = tmp_path / "categorical_block_model.parquet"
    blocks = create_demo_blockmodel()
    blocks['category_col'] = pd.Categorical(['A', 'B', 'C'] * (len(blocks) // 3))

    blocks.to_parquet(parquet_path)

    # Load the block model
    block_model = ParquetBlockModel.from_parquet(parquet_path)

    # Check if the block model is created correctly
    assert isinstance(block_model, ParquetBlockModel)
    assert block_model.blockmodel_path == parquet_path.with_suffix(".pbm")
    assert block_model.name == parquet_path.stem
    assert block_model.data.shape[0] == 27
    assert "category_col" in block_model.columns


def test_sparse_block_model(tmp_path):
    # Create a Parquet file with sparse data
    parquet_path = tmp_path / "sparse_block_model.parquet"
    blocks = create_demo_blockmodel(shape=(4, 4, 4), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0))
    blocks = blocks.query('z!=1.5')  # Drop a single z value to create a sparse dataset
    blocks.to_parquet(parquet_path)
    # Load the block model
    block_model = ParquetBlockModel.from_parquet(parquet_path)
    # Check if the block model is created correctly
    assert isinstance(block_model, ParquetBlockModel)
    assert block_model.blockmodel_path == parquet_path.with_suffix(".pbm")
    assert block_model.name == parquet_path.stem

    block_model.to_dense_parquet(parquet_path.with_suffix('.dense.parquet'), show_progress=True)
    df = pd.read_parquet(parquet_path.with_suffix('.dense.parquet'))
    assert df.shape[0] == 4 * 4 * 4

def test_block_model_with_empty_parquet(tmp_path):
    # Create an empty Parquet file
    parquet_path = tmp_path / "empty_block_model.parquet"
    pd.DataFrame(columns=["x", "y", "z"]).to_parquet(parquet_path)

    # Attempt to load the block model
    with pytest.raises(ValueError, match="Cannot infer block size/shape from degenerate centroid coordinates."):
        ParquetBlockModel.from_parquet(parquet_path)


def test_from_parquet_raises_on_incongruent_block_id_and_xyz(tmp_path):
    parquet_path = tmp_path / "bad_block_id_xyz.parquet"
    blocks = create_demo_blockmodel(shape=(3, 3, 3)).reset_index(drop=False)

    # Deliberately break positional integrity for one row.
    blocks.loc[0, "block_id"] = int(blocks.loc[0, "block_id"]) + 1
    blocks[["block_id", "x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="Inconsistent positional columns"):
        ParquetBlockModel.from_parquet(parquet_path)


def test_read_without_xyz_uses_block_id(tmp_path):
    parquet_path = tmp_path / "source.parquet"
    blocks = create_demo_blockmodel(shape=(2, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    blocks.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=3)
    assert "x" not in pbm.columns
    assert "y" not in pbm.columns
    assert "z" not in pbm.columns
    assert "block_id" in pbm.columns

    df_ijk = pbm.read(columns=["depth"], index="ijk")
    assert list(df_ijk.index.names) == ["i", "j", "k"]
    assert len(df_ijk) == 8

    # xyz index is derived from geometry + block_id when xyz columns are absent.
    df_xyz = pbm.read(columns=["depth"], index="xyz")
    assert list(df_xyz.index.names) == ["x", "y", "z"]
    assert len(df_xyz) == 8


def test_sparse_validation_without_xyz(tmp_path):
    parquet_path = tmp_path / "source_sparse.parquet"
    blocks = create_demo_blockmodel(shape=(3, 3, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    sparse = blocks.iloc[:-2].copy()
    sparse.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)
    assert pbm.is_sparse
    assert pbm.validate_sparse()


def test_block_id_persisted_in_dense_pbm(tmp_path):
    parquet_path = tmp_path / "dense_source.parquet"
    blocks = create_demo_blockmodel(shape=(3, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    blocks.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)
    persisted = pd.read_parquet(pbm.blockmodel_path)

    expected_block_ids = pbm.geometry.row_index_from_xyz(
        blocks["x"].to_numpy(),
        blocks["y"].to_numpy(),
        blocks["z"].to_numpy(),
    ).astype(np.uint32)

    assert "block_id" in persisted.columns
    assert persisted.columns[0] == "block_id"
    assert persisted["block_id"].dtype == np.uint32
    assert np.array_equal(persisted["block_id"].to_numpy(), expected_block_ids)


def test_block_id_persisted_in_sparse_pbm(tmp_path):
    parquet_path = tmp_path / "sparse_source.parquet"
    blocks = create_demo_blockmodel(shape=(3, 3, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    sparse = blocks.drop(index=[1, 4, 11]).reset_index(drop=True)
    sparse.to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=3)
    persisted = pd.read_parquet(pbm.blockmodel_path)

    expected_block_ids = pbm.geometry.row_index_from_xyz(
        sparse["x"].to_numpy(),
        sparse["y"].to_numpy(),
        sparse["z"].to_numpy(),
    ).astype(np.uint32)

    assert pbm.is_sparse
    assert "block_id" in persisted.columns
    assert persisted.columns[0] == "block_id"
    assert persisted["block_id"].is_unique
    assert np.array_equal(persisted["block_id"].to_numpy(), expected_block_ids)


def test_validate_xyz_parquet_chunked(tmp_path):
    parquet_path = tmp_path / "xyz_defined.parquet"
    blocks = create_demo_blockmodel(shape=(4, 3, 2), block_size=(2.0, 1.0, 0.5), corner=(10.0, 20.0, -1.0)).reset_index()
    # Simulate xyz-defined input with no geometry metadata.
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    geometry = ParquetBlockModel.validate_xyz_parquet(parquet_path, chunk_size=5)
    assert geometry.local.shape == (4, 3, 2)
    assert geometry.local.block_size == (2.0, 1.0, 0.5)


def test_block_id_column_matches_geometry_row_indices(tmp_path):
    shape = (3, 2, 4)
    parquet_path = tmp_path / "index_compare.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=shape)

    df = pbm.read(columns=["block_id"], index="ijk")
    i = df.index.get_level_values("i").to_numpy()
    j = df.index.get_level_values("j").to_numpy()
    k = df.index.get_level_values("k").to_numpy()
    expected = pbm.geometry.row_index_from_ijk(i, j, k)

    assert np.array_equal(df["block_id"].to_numpy(), expected)
    assert len(df["block_id"]) == int(np.prod(shape))


def test_world_id_persisted_and_decodable(tmp_path):
    parquet_path = tmp_path / "world_id_source.parquet"
    blocks = create_demo_blockmodel(shape=(3, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(500000.0, 7000000.0, 100.0)).reset_index()
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)
    persisted = pd.read_parquet(pbm.blockmodel_path)

    assert "world_id" in persisted.columns
    assert persisted["world_id"].dtype == np.int64

    offset, scale = get_world_id_encoding_params(pbm.geometry.world_id_encoding)
    x_dec, y_dec, z_dec = decode_world_coordinates(persisted["world_id"].to_numpy(dtype=np.int64), offset=offset, scale=scale)

    x_true, y_true, z_true = pbm.geometry.xyz_from_row_index(persisted["block_id"].to_numpy(dtype=np.uint32))
    np.testing.assert_allclose(x_dec, x_true, atol=0.05)
    np.testing.assert_allclose(y_dec, y_true, atol=0.05)
    np.testing.assert_allclose(z_dec, z_true, atol=0.05)


def test_metadata_contains_schema_version_and_world_id_encoding(tmp_path):
    parquet_path = tmp_path / "world_id_meta_source.parquet"
    blocks = create_demo_blockmodel(shape=(2, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(500000.0, 7000000.0, 50.0)).reset_index()
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)
    geom = pbm.geometry

    assert geom.schema_version == "1.0"
    assert geom.world_id_encoding is not None
    assert geom.world_id_encoding.get("column") == "world_id"

    geom_meta = pq.read_metadata(pbm.blockmodel_path).metadata[b"parq-blockmodel"].decode("utf-8")
    assert "schema_version" in geom_meta
    assert "world_id_encoding" in geom_meta


def test_parquet_blockmodel_exposes_corner_and_origin_properties(tmp_path):
    parquet_path = tmp_path / "origin_property_source.parquet"
    blocks = create_demo_blockmodel(
        shape=(2, 2, 2),
        block_size=(1.0, 1.0, 1.0),
        corner=(100.0, 200.0, 300.0),
    ).reset_index()
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)

    assert pbm.corner == pbm.geometry.local.corner
    assert pbm.origin == pbm.geometry.world.origin


def test_to_pyvista_image_supports_local_and_world_frames(tmp_path):
    parquet_path = tmp_path / "rotated_frame_source.parquet"
    pbm = ParquetBlockModel.create_toy_blockmodel(
        filename=parquet_path,
        shape=(4, 3, 2),
        block_size=(1.0, 1.0, 1.0),
        corner=(10.0, 20.0, 30.0),
        axis_azimuth=30.0,
        axis_dip=0.0,
        axis_plunge=0.0,
    )

    local_grid = pbm.to_pyvista(grid_type="image", attributes=["grade"], frame="local")
    world_grid = pbm.to_pyvista(grid_type="image", attributes=["grade"], frame="world")

    np.testing.assert_allclose(np.asarray(local_grid.direction_matrix), np.eye(3))
    assert not np.allclose(np.asarray(world_grid.direction_matrix), np.eye(3))
    np.testing.assert_array_equal(local_grid.cell_data["grade"], world_grid.cell_data["grade"])


def test_to_pyvista_local_frame_requires_image_grid(tmp_path):
    parquet_path = tmp_path / "frame_guard_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))

    with pytest.raises(ValueError, match="frame='local' is only supported for grid_type='image'."):
        pbm.to_pyvista(grid_type="structured", frame="local")


def test_plot_passes_frame_to_image_pyvista(tmp_path, monkeypatch):
    pv = pytest.importorskip("pyvista")

    parquet_path = tmp_path / "plot_frame_source.parquet"
    pbm = ParquetBlockModel.create_toy_blockmodel(
        filename=parquet_path,
        shape=(2, 2, 1),
        block_size=(1.0, 1.0, 1.0),
        corner=(0.0, 0.0, 0.0),
        axis_azimuth=25.0,
    )
    scalar = "grade"

    mesh = pv.ImageData(dimensions=(3, 3, 2))
    mesh.cell_data[scalar] = np.arange(mesh.n_cells)

    captured: dict[str, object] = {}

    def fake_to_pyvista(self, grid_type="image", attributes=None, frame="world"):
        captured["grid_type"] = grid_type
        captured["attributes"] = list(attributes) if attributes is not None else None
        captured["frame"] = frame
        return mesh

    class FakePlotter:
        def __init__(self):
            self.actors = {}
            self.title = None

        def add_mesh_threshold(self, *args, **kwargs):
            captured["mesh_call"] = "threshold"

        def add_mesh(self, *args, **kwargs):
            captured["mesh_call"] = "mesh"

        def show_axes(self):
            captured["show_axes"] = True

        def add_text(self, text, position=None, font_size=None, name=None):
            if name is not None:
                self.actors[name] = text

        def remove_actor(self, name):
            self.actors.pop(name, None)

        def enable_cell_picking(self, callback=None, show_message=False, through=False):
            captured["picking_enabled"] = True

    monkeypatch.setattr(ParquetBlockModel, "to_pyvista", fake_to_pyvista)
    monkeypatch.setattr(pv, "Plotter", FakePlotter)

    plotter = pbm.plot(scalar=scalar, grid_type="image", frame="local", threshold=False)

    assert isinstance(plotter, FakePlotter)
    assert captured["grid_type"] == "image"
    assert captured["frame"] == "local"
    assert captured["attributes"] == [scalar]
    assert captured["mesh_call"] == "mesh"


def test_from_parquet_rotated_xyz_only_derives_correct_block_ids(tmp_path):
    axis_azimuth = 35.0
    axis_dip = 12.0
    axis_plunge = 8.0
    world_origin = (100.0, 200.0, 50.0)
    shape = (20, 15, 10)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    extent_center = np.array([[10.0, 7.5, 5.0]], dtype=float)
    deposit_center = tuple(
        rotate_points(
            extent_center,
            azimuth=axis_azimuth,
            dip=axis_dip,
            plunge=axis_plunge,
        )[0]
    )
    source_df = create_toy_blockmodel(
        shape=shape,
        block_size=block_size,
        corner=corner,
        axis_azimuth=axis_azimuth,
        axis_dip=axis_dip,
        axis_plunge=axis_plunge,
        deposit_bearing=0.0,
        deposit_dip=0.0,
        deposit_plunge=0.0,
        grade_name="fe",
        grade_min=45.0,
        grade_max=70.0,
        deposit_center=deposit_center,
        deposit_radii=(7.0, 4.5, 3.5),
        noise_rel=1e-3,
        noise_seed=42,
    ).reset_index(drop=False)

    source_df = source_df.drop(columns=["i", "j", "k", "block_id"], errors="ignore")
    source_df["x"] = source_df["x"] + world_origin[0]
    source_df["y"] = source_df["y"] + world_origin[1]
    source_df["z"] = source_df["z"] + world_origin[2]
    source_df = source_df.set_index(["x", "y", "z"]).sort_index()

    source_path = tmp_path / "rotated_xyz_only.parquet"
    source_df.to_parquet(source_path)

    axis_u, axis_v, axis_w = angles_to_axes(
        axis_azimuth=axis_azimuth,
        axis_dip=axis_dip,
        axis_plunge=axis_plunge,
    )
    geometry = RegularGeometry(
        local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
        world=WorldFrame(origin=world_origin, axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
    )

    table = pq.read_table(source_path)
    meta = dict(table.schema.metadata or {})
    meta[b"parq-blockmodel"] = json.dumps(geometry.to_metadata_dict()).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(meta), source_path)

    pbm = ParquetBlockModel.from_parquet(source_path)
    promoted = pbm.read(columns=["block_id"], index="xyz", dense=False)

    x = promoted.index.get_level_values("x").to_numpy(dtype=float)
    y = promoted.index.get_level_values("y").to_numpy(dtype=float)
    z = promoted.index.get_level_values("z").to_numpy(dtype=float)
    expected = pbm.geometry.row_index_from_xyz(x, y, z).astype(np.uint32)
    actual = promoted["block_id"].to_numpy(dtype=np.uint32)

    np.testing.assert_array_equal(actual, expected)


def test_read_ijk_dense_preserves_attribute_mapping_when_positional_columns_not_requested(tmp_path):
    axis_azimuth = 35.0
    axis_dip = 12.0
    axis_plunge = 8.0
    world_origin = (100.0, 200.0, 50.0)
    shape = (20, 15, 10)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    source_df = create_toy_blockmodel(
        shape=shape,
        block_size=block_size,
        corner=corner,
        axis_azimuth=0.0,
        axis_dip=0.0,
        axis_plunge=0.0,
        deposit_bearing=0.0,
        deposit_dip=0.0,
        deposit_plunge=0.0,
        grade_name="fe",
        grade_min=45.0,
        grade_max=70.0,
        deposit_center=(10.0, 7.5, 5.0),
        deposit_radii=(7.0, 4.5, 3.5),
        noise_rel=1e-3,
        noise_seed=42,
    ).reset_index(drop=False)

    rotated_xyz = rotate_points(
        source_df[["x", "y", "z"]].to_numpy(dtype=float),
        azimuth=axis_azimuth,
        dip=axis_dip,
        plunge=axis_plunge,
    )
    source_df["x"] = rotated_xyz[:, 0] + world_origin[0]
    source_df["y"] = rotated_xyz[:, 1] + world_origin[1]
    source_df["z"] = rotated_xyz[:, 2] + world_origin[2]
    source_df = source_df.drop(columns=["i", "j", "k", "block_id"], errors="ignore")
    source_df = source_df.set_index(["x", "y", "z"]).sort_index()

    source_path = tmp_path / "rotated_xyz_for_read.parquet"
    source_df.to_parquet(source_path)

    axis_u, axis_v, axis_w = angles_to_axes(
        axis_azimuth=axis_azimuth,
        axis_dip=axis_dip,
        axis_plunge=axis_plunge,
    )
    geometry = RegularGeometry(
        local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
        world=WorldFrame(origin=world_origin, axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
    )

    table = pq.read_table(source_path)
    meta = dict(table.schema.metadata or {})
    meta[b"parq-blockmodel"] = json.dumps(geometry.to_metadata_dict()).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(meta), source_path)

    pbm = ParquetBlockModel.from_parquet(source_path)

    fe_only = pbm.read(columns=["fe"], index="ijk", dense=True)
    fe_with_block_id = pbm.read(columns=["block_id", "fe"], index="ijk", dense=True)[["fe"]]

    assert fe_only.index.equals(fe_with_block_id.index)
    np.testing.assert_allclose(fe_only["fe"].to_numpy(), fe_with_block_id["fe"].to_numpy())


def test_canonical_pbm_requires_block_id_column(tmp_path):
    parquet_path = tmp_path / "missing_block_id_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))

    table = pq.read_table(pbm.blockmodel_path)
    broken = table.drop_columns(["block_id"])
    pq.write_table(broken, pbm.blockmodel_path)

    with pytest.raises(ValueError, match="Canonical \\.pbm requires a 'block_id' column"):
        ParquetBlockModel(pbm.blockmodel_path)


def test_canonical_pbm_requires_unique_block_id(tmp_path):
    parquet_path = tmp_path / "duplicate_block_id_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))

    table = pq.read_table(pbm.blockmodel_path)
    df = table.to_pandas()
    df.loc[1, "block_id"] = int(df.loc[0, "block_id"])
    patched = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(table.schema.metadata)
    pq.write_table(patched, pbm.blockmodel_path)

    with pytest.raises(ValueError, match="Canonical \\.pbm requires unique block_id values"):
        ParquetBlockModel(pbm.blockmodel_path)


