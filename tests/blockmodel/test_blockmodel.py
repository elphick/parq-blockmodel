import os
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import parq_blockmodel.blockmodel as blockmodel_module
from parq_blockmodel import ParquetBlockModel, RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.reporting import BlockModelReport
from parq_blockmodel.utils import create_demo_blockmodel
from parq_blockmodel.utils.demo_block_model import create_toy_blockmodel
from parq_blockmodel.utils.geometry_utils import angles_to_axes, rotate_points
from parq_blockmodel.utils.spatial_encoding import decode_world_coordinates, get_world_id_encoding_params


class FakeParquetProfileReport:
    HTML = "<html><body>fake report</body></html>"
    last_init: dict[str, object] | None = None
    last_show_notebook: bool | None = None

    def __init__(self, parquet_path, columns=None, batch_size=1, show_progress=True, title="", dataset_metadata=None,
                 column_descriptions=None):
        FakeParquetProfileReport.last_init = {
            "parquet_path": Path(parquet_path),
            "columns": list(columns) if columns is not None else None,
            "batch_size": batch_size,
            "show_progress": show_progress,
            "title": title,
            "dataset_metadata": dataset_metadata,
            "column_descriptions": column_descriptions,
        }
        self.parquet_path = Path(parquet_path)
        self.columns = list(columns) if columns is not None else None
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.title = title
        self.dataset_metadata = dataset_metadata
        self.column_descriptions = column_descriptions
        self.report = SimpleNamespace(
            config=SimpleNamespace(
                html=SimpleNamespace(
                    style=SimpleNamespace(logo=False),
                )
            )
        )

    def profile(self):
        return self

    def save_html(self, output_html: Path) -> None:
        Path(output_html).write_text(self.HTML, encoding="utf-8")

    def to_html(self) -> str:
        return self.HTML

    def show(self, notebook: bool = False):
        FakeParquetProfileReport.last_show_notebook = notebook


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


def test_create_report_returns_wrapper_and_saves_html(tmp_path, monkeypatch):
    parquet_path = tmp_path / "report_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    monkeypatch.setattr(blockmodel_module, "ParquetProfileReport", FakeParquetProfileReport)

    report = pbm.create_report(show_progress=False)

    assert isinstance(report, BlockModelReport)
    assert report.output_path == pbm.blockmodel_path.with_suffix(".html")
    assert pbm.report_path == report.output_path
    assert report.output_path.exists()
    assert report.output_path.read_text(encoding="utf-8") == FakeParquetProfileReport.HTML
    assert report.columns_per_batch == 10
    assert FakeParquetProfileReport.last_init is not None
    assert FakeParquetProfileReport.last_init["columns"] == pbm.columns
    assert FakeParquetProfileReport.last_init["dataset_metadata"] is None
    assert FakeParquetProfileReport.last_init["column_descriptions"] is None


def test_create_report_enriches_schema_metadata(tmp_path, monkeypatch):
    pandera = pytest.importorskip("pandera", reason="pandera not installed")
    DataFrameSchema = pandera.DataFrameSchema
    Column = pandera.Column

    parquet_path = tmp_path / "report_schema_metadata.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    pbm.schema = DataFrameSchema(
        columns={
            "depth": Column(
                float,
                nullable=True,
            )
        },
        strict=False,
    )
    pbm.schema.name = "demo_pbm"
    pbm.schema.title = "Demo Block Model"
    pbm.schema.description = "Synthetic demo dataset."
    pbm.schema.columns["depth"].title = "Depth"
    pbm.schema.columns["depth"].description = "Vertical distance from reference."

    monkeypatch.setattr(blockmodel_module, "ParquetProfileReport", FakeParquetProfileReport)

    pbm.create_report(columns=["depth"], write_html=False, show_progress=False)

    assert FakeParquetProfileReport.last_init is not None
    assert FakeParquetProfileReport.last_init["column_descriptions"] == {
        "depth": "Depth - Vertical distance from reference."
    }
    assert FakeParquetProfileReport.last_init["dataset_metadata"] == {
        "description": "{'name': 'demo_pbm', 'title': 'Demo Block Model', 'description': 'Synthetic demo dataset.'}"
    }


def test_create_report_enriches_schema_metadata_omits_null_dataset_keys(tmp_path, monkeypatch):
    pandera = pytest.importorskip("pandera", reason="pandera not installed")
    DataFrameSchema = pandera.DataFrameSchema
    Column = pandera.Column

    parquet_path = tmp_path / "report_schema_metadata_nulls.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    pbm.schema = DataFrameSchema(
        columns={"depth": Column(float, nullable=True)},
        strict=False,
    )
    pbm.schema.name = "demo_pbm"
    pbm.schema.title = None
    pbm.schema.description = "Synthetic demo dataset."

    monkeypatch.setattr(blockmodel_module, "ParquetProfileReport", FakeParquetProfileReport)

    pbm.create_report(columns=["depth"], write_html=False, show_progress=False)

    assert FakeParquetProfileReport.last_init is not None
    assert FakeParquetProfileReport.last_init["dataset_metadata"] == {
        "description": "{'name': 'demo_pbm', 'description': 'Synthetic demo dataset.'}"
    }


def test_create_report_dataset_metadata_does_not_use_schema_metadata_fallback(tmp_path, monkeypatch):
    pandera = pytest.importorskip("pandera", reason="pandera not installed")
    DataFrameSchema = pandera.DataFrameSchema
    Column = pandera.Column

    parquet_path = tmp_path / "report_schema_dataset_no_fallback.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    pbm.schema = DataFrameSchema(
        columns={"depth": Column(float, nullable=True)},
        strict=False,
        metadata={
            "name": "metadata_name",
            "title": "metadata_title",
            "description": "metadata_description",
        },
    )

    monkeypatch.setattr(blockmodel_module, "ParquetProfileReport", FakeParquetProfileReport)

    pbm.create_report(columns=["depth"], write_html=False, show_progress=False)

    assert FakeParquetProfileReport.last_init is not None
    assert FakeParquetProfileReport.last_init["dataset_metadata"] is None


def test_create_report_column_descriptions_do_not_use_metadata_fallback(tmp_path, monkeypatch):
    pandera = pytest.importorskip("pandera", reason="pandera not installed")
    DataFrameSchema = pandera.DataFrameSchema
    Column = pandera.Column

    parquet_path = tmp_path / "report_schema_column_no_fallback.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    pbm.schema = DataFrameSchema(
        columns={
            "depth": Column(
                float,
                nullable=True,
                metadata={
                    "title": "metadata_title",
                    "description": "metadata_description",
                },
            )
        },
        strict=False,
    )

    monkeypatch.setattr(blockmodel_module, "ParquetProfileReport", FakeParquetProfileReport)

    pbm.create_report(columns=["depth"], write_html=False, show_progress=False)

    assert FakeParquetProfileReport.last_init is not None
    assert FakeParquetProfileReport.last_init["column_descriptions"] is None


def test_create_report_can_save_later_and_disable_chunking(tmp_path, monkeypatch):
    parquet_path = tmp_path / "report_full_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    monkeypatch.setattr(blockmodel_module, "ParquetProfileReport", FakeParquetProfileReport)

    report = pbm.create_report(columns=["depth"], columns_per_batch=None, write_html=False, show_progress=False)

    assert isinstance(report, BlockModelReport)
    assert report.columns_per_batch is None
    assert not report.output_path.exists()
    assert FakeParquetProfileReport.last_init is not None
    assert FakeParquetProfileReport.last_init["batch_size"] is None

    custom_path = tmp_path / "profiles" / "depth.html"
    saved_path = report.save(custom_path)
    assert saved_path == custom_path
    assert custom_path.exists()
    assert custom_path.read_text(encoding="utf-8") == FakeParquetProfileReport.HTML


def test_create_report_memory_budget_derives_column_batch_size(tmp_path, monkeypatch):
    parquet_path = tmp_path / "report_budget_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    selected_columns = pbm.columns[:3]
    estimated_sizes = {
        selected_columns[0]: 100,
        selected_columns[1]: 150,
        selected_columns[2]: 100,
    }

    monkeypatch.setattr(blockmodel_module, "ParquetProfileReport", FakeParquetProfileReport)
    monkeypatch.setattr(
        blockmodel_module,
        "resolve_report_columns_per_batch",
        lambda parquet_file, columns, columns_per_batch, memory_budget: (2, 260),
    )

    report = pbm.create_report(columns=selected_columns, write_html=False, show_progress=False, memory_budget="260B")

    assert isinstance(report, BlockModelReport)
    assert report.columns_per_batch == 2
    assert report.memory_budget_bytes == 260
    assert FakeParquetProfileReport.last_init is not None
    assert FakeParquetProfileReport.last_init["batch_size"] == 2


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
    ).astype(np.int32)

    assert "block_id" in persisted.columns
    assert persisted.columns[0] == "block_id"
    assert persisted["block_id"].dtype == np.int32
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
    ).astype(np.int32)

    assert pbm.is_sparse
    assert "block_id" in persisted.columns
    assert persisted.columns[0] == "block_id"
    assert persisted["block_id"].dtype == np.int32
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

    offset, scale, bits_per_axis = get_world_id_encoding_params(pbm.geometry.world_id_encoding)
    x_dec, y_dec, z_dec = decode_world_coordinates(
        persisted["world_id"].to_numpy(dtype=np.int64),
        offset=offset,
        scale=scale,
        bits_per_axis=bits_per_axis,
    )

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

    assert geom.schema_version == "1.1"
    assert geom.world_id_encoding is not None
    assert geom.world_id_encoding.get("column") == "world_id"

    geom_meta = pq.read_metadata(pbm.blockmodel_path).metadata[b"parq-blockmodel"].decode("utf-8")
    assert "schema_version" in geom_meta
    assert "world_id_encoding" in geom_meta


def test_canonical_special_column_order_defaults(tmp_path):
    parquet_path = tmp_path / "special_order_source.parquet"
    blocks = create_demo_blockmodel(shape=(2, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, chunk_size=4)
    persisted = pd.read_parquet(pbm.blockmodel_path)

    expected_special_order = ["block_id", "world_id", "i", "j", "k", "x", "y", "z"]
    observed_special_order = [col for col in persisted.columns if col in expected_special_order]
    assert observed_special_order == expected_special_order
    assert persisted["block_id"].dtype == np.int32
    assert persisted["world_id"].dtype == np.int64
    assert persisted["i"].dtype == np.int32
    assert persisted["j"].dtype == np.int32
    assert persisted["k"].dtype == np.int32
    assert persisted["x"].dtype == np.float32
    assert persisted["y"].dtype == np.float32
    assert persisted["z"].dtype == np.float32


def test_ensure_spatial_and_validate_special_columns(tmp_path):
    parquet_path = tmp_path / "ensure_spatial_source.parquet"
    blocks = create_demo_blockmodel(shape=(2, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, columns=["depth"], chunk_size=4)
    assert "x" not in pbm.columns
    assert "i" not in pbm.columns

    pbm.ensure_spatial()

    assert {"x", "y", "z", "i", "j", "k"}.issubset(pbm.columns)
    assert pbm.validate_special_columns()


def test_validate_special_columns_detects_invalid_world_id(tmp_path):
    parquet_path = tmp_path / "invalid_world_id_source.parquet"
    blocks = create_demo_blockmodel(shape=(2, 2, 2), block_size=(1.0, 1.0, 1.0), corner=(0.0, 0.0, 0.0)).reset_index()
    blocks[["x", "y", "z", "depth"]].to_parquet(parquet_path, index=False)

    pbm = ParquetBlockModel.from_parquet(parquet_path, chunk_size=4)
    table = pq.read_table(pbm.blockmodel_path)
    tampered = table.to_pandas()
    tampered.loc[0, "world_id"] = tampered.loc[0, "world_id"] + 1
    pq.write_table(
        pa.Table.from_pandas(tampered, preserve_index=False).replace_schema_metadata(table.schema.metadata),
        pbm.blockmodel_path,
    )
    pbm = ParquetBlockModel(blockmodel_path=pbm.blockmodel_path)

    assert not pbm.validate_special_columns(sample_size=0)


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


def test_plot_categorical_passes_pyvista_category_args(tmp_path, monkeypatch):
    pv = pytest.importorskip("pyvista")
    from parq_blockmodel.utils.pyvista.categorical_utils import store_mapping_dict

    parquet_path = tmp_path / "plot_categorical_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    scalar = "depth_category"

    mesh = pv.ImageData(dimensions=(3, 3, 3))
    mesh.cell_data[scalar] = np.array([0.0, np.nan, 0.0, np.nan, 0.0, np.nan, 0.0, np.nan], dtype=float)
    store_mapping_dict(mesh, scalar, {0: "lease_a"})

    captured: dict[str, object] = {}

    def fake_to_pyvista(self, grid_type="image", attributes=None, frame="world"):
        return mesh

    class FakePlotter:
        def __init__(self):
            self.actors = {}
            self.title = None

        def add_mesh_threshold(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def add_mesh(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def show_axes(self):
            pass

        def add_text(self, text, position=None, font_size=None, name=None):
            if name is not None:
                self.actors[name] = text

        def remove_actor(self, name):
            self.actors.pop(name, None)

        def enable_cell_picking(self, callback=None, show_message=False, through=False):
            pass

    monkeypatch.setattr(ParquetBlockModel, "to_pyvista", fake_to_pyvista)
    monkeypatch.setattr(pv, "Plotter", FakePlotter)

    pbm.plot(scalar=scalar, grid_type="image", threshold=False)

    kwargs = captured["kwargs"]
    assert kwargs["categories"] is True
    assert kwargs["annotations"] == {0.0: "lease_a"}
    assert isinstance(kwargs["cmap"], pv.LookupTable)
    assert tuple(kwargs["cmap"].scalar_range) == (-0.5, 0.5)
    assert kwargs["scalar_bar_args"] == {"n_labels": 0, "nan_annotation": True}


def test_plot_categorical_without_nan_keeps_scalarbar_clean(tmp_path, monkeypatch):
    pv = pytest.importorskip("pyvista")
    from parq_blockmodel.utils.pyvista.categorical_utils import store_mapping_dict

    parquet_path = tmp_path / "plot_categorical_no_nan_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    scalar = "depth_category"

    mesh = pv.ImageData(dimensions=(3, 3, 3))
    mesh.cell_data[scalar] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    store_mapping_dict(mesh, scalar, {0: "lease_a"})

    captured: dict[str, object] = {}

    def fake_to_pyvista(self, grid_type="image", attributes=None, frame="world"):
        return mesh

    class FakePlotter:
        def __init__(self):
            self.actors = {}
            self.title = None

        def add_mesh_threshold(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def add_mesh(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def show_axes(self):
            pass

        def add_text(self, text, position=None, font_size=None, name=None):
            if name is not None:
                self.actors[name] = text

        def remove_actor(self, name):
            self.actors.pop(name, None)

        def enable_cell_picking(self, callback=None, show_message=False, through=False):
            pass

    monkeypatch.setattr(ParquetBlockModel, "to_pyvista", fake_to_pyvista)
    monkeypatch.setattr(pv, "Plotter", FakePlotter)

    pbm.plot(scalar=scalar, grid_type="image", threshold=False)

    kwargs = captured["kwargs"]
    assert kwargs["categories"] is True
    assert kwargs["annotations"] == {0.0: "lease_a"}
    assert isinstance(kwargs["cmap"], pv.LookupTable)
    assert tuple(kwargs["cmap"].scalar_range) == (-0.5, 0.5)
    assert kwargs["scalar_bar_args"] == {"n_labels": 0}


def test_plot_picking_shows_categorical_labels(tmp_path, monkeypatch):
    pv = pytest.importorskip("pyvista")
    from parq_blockmodel.utils.pyvista.categorical_utils import store_mapping_dict

    parquet_path = tmp_path / "plot_picking_categorical_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    scalar = "depth_category"

    mesh = pv.ImageData(dimensions=(3, 3, 3))
    mesh.cell_data[scalar] = np.array([0.0, np.nan, 0.0, np.nan, 0.0, np.nan, 0.0, np.nan], dtype=float)
    store_mapping_dict(mesh, scalar, {0: "lease_a"})

    captured: dict[str, object] = {}

    def fake_to_pyvista(self, grid_type="image", attributes=None, frame="world"):
        return mesh

    class FakePlotter:
        def __init__(self):
            self.actors = {}
            self.title = None

        def add_mesh_threshold(self, *args, **kwargs):
            pass

        def add_mesh(self, *args, **kwargs):
            pass

        def show_axes(self):
            pass

        def add_text(self, text, position=None, font_size=None, name=None):
            if name is not None:
                self.actors[name] = text
                captured[name] = text

        def remove_actor(self, name):
            self.actors.pop(name, None)

        def enable_cell_picking(self, callback=None, show_message=False, through=False):
            captured["callback"] = callback

    class FakePickedCell:
        n_cells = 1
        cell_data = {"vtkOriginalCellIds": np.array([0], dtype=np.int64)}

    monkeypatch.setattr(ParquetBlockModel, "to_pyvista", fake_to_pyvista)
    monkeypatch.setattr(pv, "Plotter", FakePlotter)

    pbm.plot(scalar=scalar, grid_type="image", threshold=False, enable_picking=True, picked_attributes=[scalar])

    callback = captured["callback"]
    callback(FakePickedCell())

    assert "lease_a" in captured["cell_info_text"]
    assert "depth_category: 0" not in captured["cell_info_text"]

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
