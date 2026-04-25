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
    for col in ["x", "y", "z", "block_id", "depth", "depth_category"]:
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


def test_index_type_block_id():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    df = create_demo_blockmodel(
        shape=shape,
        block_size=block_size,
        corner=corner,
        index_type="block_id",
    )

    # Explicit canonical block_id index
    assert list(df.index.names) == ["block_id"]
    assert np.array_equal(df.index.to_numpy(), np.arange(np.prod(shape)))

    # Both i,j,k and x,y,z should be available as columns
    for col in ["i", "j", "k", "x", "y", "z"]:
        assert col in df.columns


def test_ordering():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    df = create_demo_blockmodel(shape, block_size, corner)

    # block_id follows C-order unraveling of logical (i,j,k)
    assert np.array_equal(df["block_id"].values, np.arange(np.prod(shape)))


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

    # Rotation does not affect canonical local block_id ordering
    assert np.array_equal(df["block_id"].values, np.arange(np.prod(shape)))


def test_geometry_attrs_index_type():
    """geometry.attrs['index_type'] should track the chosen index_type."""
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    for index_type in ["block_index", "world_centroids", "block_id"]:
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


def test_block_id_matches_ijk_c_order():
    """block_id must be consistent with i,j,k via C-order unravel_index."""
    shape = (3, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)

    df = create_demo_blockmodel(shape=shape, block_size=block_size, corner=corner)

    # Recover i,j,k from block_id using C-order and compare to stored columns
    rows = df["block_id"].to_numpy()
    i, j, k = np.unravel_index(rows, shape, order="C")

    # When index_type='block_index', i,j,k are the index levels
    assert np.array_equal(df.index.get_level_values("i"), i)
    assert np.array_equal(df.index.get_level_values("j"), j)
    assert np.array_equal(df.index.get_level_values("k"), k)


def test_sorted_world_xyz_is_non_canonical_for_rotated_model():
    """Sorting by world xyz after rotation should not be treated as canonical block order."""
    df = create_demo_blockmodel(
        shape=(3, 3, 2),
        block_size=(1.0, 1.0, 1.0),
        corner=(0.0, 0.0, 0.0),
        azimuth=35.0,
        dip=15.0,
        plunge=5.0,
        index_type="block_id",
    )

    # Canonical local order from block_id index
    canonical = df.index.to_numpy()

    # Sorting by world coordinates defines a different (frame-dependent) order
    world_sorted = df.sort_values(["x", "y", "z"]).index.to_numpy()

    assert not np.array_equal(world_sorted, canonical)

