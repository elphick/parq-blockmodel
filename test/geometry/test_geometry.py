import numpy as np

from parq_blockmodel import RegularGeometry


def test_regular_geometry_init():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 2.0, 3.0),
        shape=(4, 5, 6)
    )
    assert geom.corner == (0.0, 0.0, 0.0)
    assert geom.block_size == (1.0, 2.0, 3.0)
    assert geom.shape == (4, 5, 6)
    assert geom.is_regular

def test_centroid_properties():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
        srs='my_srs'
    )
    np.testing.assert_allclose(geom.centroid_u, [0.5, 1.5])
    np.testing.assert_allclose(geom.centroid_v, [0.5, 1.5])
    np.testing.assert_allclose(geom.centroid_w, [0.5, 1.5])
    assert geom.srs == 'my_srs'

def test_extents_and_bounding_box():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    extents = geom.extents
    assert extents[0] == (0.0, 2.0)
    assert extents[1] == (0.0, 2.0)
    assert extents[2] == (0.0, 2.0)
    assert geom.bounding_box == ((0.0, 2.0), (0.0, 2.0))

def test_to_json_and_from_json():
    geom = RegularGeometry(
        corner=(1.0, 2.0, 3.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    json_str = geom.to_json()
    geom2 = RegularGeometry.from_json(json_str)
    assert geom2.corner == [1.0, 2.0, 3.0]
    assert geom2.block_size == [1.0, 1.0, 1.0]
    assert geom2.shape == [2, 2, 2]

def test_nearest_centroid_lookup():
    geom = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    assert geom.nearest_centroid_lookup(0.6, 0.6, 0.6) == (0.5, 0.5, 0.5)
    assert geom.nearest_centroid_lookup(1.4, 1.4, 1.4) == (1.5, 1.5, 1.5)

def test_is_compatible_true():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    geom2 = RegularGeometry(
        corner=(10.0, 20.0, 30.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    assert geom1.is_compatible(geom2)

def test_is_compatible_false():
    geom1 = RegularGeometry(
        corner=(0.0, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    geom2 = RegularGeometry(
        corner=(0.5, 0.0, 0.0),
        block_size=(1.0, 1.0, 1.0),
        shape=(2, 2, 2)
    )
    assert not geom1.is_compatible(geom2)