import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel.geometry import RegularGeometry


def make_geometry():
    return RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 2.0, 3.0),
        shape=(2, 2, 2),
    )


def test_roundtrip_via_attrs():
    geom = make_geometry()
    meta = geom.to_metadata_dict()

    df = pd.DataFrame({"value": [1, 2, 3]})
    df.attrs["parq-blockmodel"] = meta

    geom2 = RegularGeometry.from_attrs(df.attrs)

    assert geom2.corner == geom.corner
    assert geom2.block_size == geom.block_size
    assert geom2.shape == geom.shape
    assert geom2.axis_u == geom.axis_u
    assert geom2.axis_v == geom.axis_v
    assert geom2.axis_w == geom.axis_w
    assert geom2.srs == geom.srs


def test_roundtrip_via_parquet_metadata(tmp_path):
    geom = make_geometry()
    meta = geom.to_metadata_dict()

    # Write a tiny Parquet file with custom key_value_metadata
    table = pa.Table.from_pydict({"value": [1, 2, 3]})
    metadata = {"parq-blockmodel": json.dumps(meta).encode("utf-8")}
    table = table.replace_schema_metadata(metadata)

    path = tmp_path / "test_geom_meta.parquet"
    pq.write_table(table, path)

    pf = pq.ParquetFile(path)
    geom2 = RegularGeometry.from_parquet_metadata(pf.metadata)

    assert geom2.corner == geom.corner
    assert geom2.block_size == geom.block_size
    assert geom2.shape == geom.shape
    assert geom2.axis_u == geom.axis_u
    assert geom2.axis_v == geom.axis_v
    assert geom2.axis_w == geom.axis_w
    assert geom2.srs == geom.srs


def test_from_parquet_metadata_missing_key_raises(tmp_path):
    table = pa.Table.from_pydict({"value": [1, 2, 3]})
    path = tmp_path / "test_geom_no_meta.parquet"
    pq.write_table(table, path)

    pf = pq.ParquetFile(path)

    try:
        RegularGeometry.from_parquet_metadata(pf.metadata)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when geometry metadata is missing")

