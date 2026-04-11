import numpy as np
import pandas as pd

from parq_blockmodel.utils import create_demo_blockmodel


def test_create_blockmodel_default_index_block():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    # Default index_type is expected to be 'block_index'
    df = create_demo_blockmodel(shape, block_size, corner)

    # Check shape and index
    assert isinstance(df, pd.DataFrame)
    assert list(df.index.names) == ["i", "j", "k"]
    assert len(df) == np.prod(shape)

    # Check columns are present
    for col in ["x", "y", "z", "index_c", "index_f", "depth", "depth_category"]:
        assert col in df.columns

    # Check that dx/dy/dz are not present
    assert "dx" not in df.columns
    assert "dy" not in df.columns
    assert "dz" not in df.columns


def test_index_type_world_centroids():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    df = create_demo_blockmodel(
        shape=shape,
        block_size=block_size,
        corner=corner,
        index_type="world_centroids",
    )

    # Index is a MultiIndex on world centroids
    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index.names) == ["x", "y", "z"]

    # Canonical indices should still be available as columns
    for col in ["i", "j", "k"]:
        assert col in df.columns


def test_index_type_range():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    df = create_demo_blockmodel(
        shape=shape,
        block_size=block_size,
        corner=corner,
        index_type="range",
    )

    # Default RangeIndex starting at 0
    assert isinstance(df.index, pd.RangeIndex)
    assert df.index.start == 0
    assert df.index.stop == np.prod(shape)

    # Both i,j,k and x,y,z should be available as columns
    for col in ["i", "j", "k", "x", "y", "z"]:
        assert col in df.columns


def test_ordering():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    df = create_demo_blockmodel(shape, block_size, corner)

    # Check c_order_xyz: C order (xyz fastest to slowest)
    assert np.array_equal(df["index_c"].values, np.arange(np.prod(shape)))

    # Check f_order_zyx: F order (zyx fastest to slowest)
    expected_f_order = (
        np.arange(np.prod(shape)).reshape(shape, order="C").ravel(order="F")
    )
    assert np.array_equal(df["index_f"].values, expected_f_order)


def test_ordering_after_rotation():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    azimuth = 45.0
    dip = 30.0
    plunge = 15.0
    df = create_demo_blockmodel(
        shape, block_size, corner, azimuth=azimuth, dip=dip, plunge=plunge
    )

    # Check c_order_xyz
    assert np.array_equal(df["index_c"].values, np.arange(np.prod(shape)))

    # Check f_order_zyx: F order (zyx fastest to slowest)
    expected_f_order = (
        np.arange(np.prod(shape)).reshape(shape, order="C").ravel(order="F")
    )
    assert np.array_equal(df["index_f"].values, expected_f_order)


def test_geometry_attrs_index_type():
    """geometry.attrs['index_type'] should track the chosen index_type."""
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    for index_type in ["block_index", "world_centroids", "range"]:
        df = create_demo_blockmodel(
            shape=shape,
            block_size=block_size,
            corner=corner,
            index_type=index_type,
        )
        geom = df.attrs.get("geometry", {})
        assert geom.get("index_type") == index_type
        assert tuple(geom.get("shape")) == shape
        assert tuple(geom.get("block_size")) == block_size
        assert tuple(geom.get("corner")) == corner


def test_index_c_matches_ijk_c_order():
    """index_c must be consistent with i,j,k via C-order unravel_index."""
    shape = (3, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    df = create_demo_blockmodel(shape=shape, block_size=block_size, corner=corner)

    # Recover i,j,k from index_c using C-order and compare to stored columns
    rows = df["index_c"].to_numpy()
    i, j, k = np.unravel_index(rows, shape, order="C")

    # When index_type='block_index', i,j,k are the index levels
    assert np.array_equal(df.index.get_level_values("i"), i)
    assert np.array_equal(df.index.get_level_values("j"), j)
    assert np.array_equal(df.index.get_level_values("k"), k)
