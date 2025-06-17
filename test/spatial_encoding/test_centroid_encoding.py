import pytest
import numpy as np

from parq_blockmodel.utils import encode_coordinates, decode_coordinates
from parq_blockmodel.utils.spatial_encoding import MAX_XY_VALUE, MAX_Z_VALUE


def test_encode_decode_float():
    x, y, z = 12.3, 56.7, 90.1
    encoded = encode_coordinates(x, y, z)
    decoded_x, decoded_y, decoded_z = decode_coordinates(encoded)
    assert pytest.approx(decoded_x, 1e-06) == x
    assert pytest.approx(decoded_y, 1e-06) == y
    assert pytest.approx(decoded_z, 1e-06) == z


def test_encode_decode_array():
    x = np.array([12.3, 23.4, 34.5])
    y = np.array([56.7, 67.8, 78.9])
    z = np.array([90.1, 12.3, 23.4])
    encoded = encode_coordinates(x, y, z)
    decoded_x, decoded_y, decoded_z = decode_coordinates(encoded)
    np.testing.assert_almost_equal(decoded_x, x, decimal=6)
    np.testing.assert_almost_equal(decoded_y, y, decimal=6)
    np.testing.assert_almost_equal(decoded_z, z, decimal=6)


def test_max_values():
    x, y, z = MAX_XY_VALUE, MAX_XY_VALUE, MAX_Z_VALUE
    encoded = encode_coordinates(x, y, z)
    decoded_x, decoded_y, decoded_z = decode_coordinates(encoded)
    assert pytest.approx(decoded_x, 1e-06) == x
    assert pytest.approx(decoded_y, 1e-06) == y
    assert pytest.approx(decoded_z, 1e-06) == z


def test_exceed_max_values():
    with pytest.raises(ValueError, match=f"exceeds the maximum supported value of {MAX_XY_VALUE}"):
        encode_coordinates(MAX_XY_VALUE + 0.1, 0, 0)
    with pytest.raises(ValueError, match=f"exceeds the maximum supported value of {MAX_XY_VALUE}"):
        encode_coordinates(0, MAX_XY_VALUE + 0.1, 0)
    with pytest.raises(ValueError, match=f"exceeds the maximum supported value of {MAX_Z_VALUE}"):
        encode_coordinates(0, 0, MAX_Z_VALUE + 0.1)


def test_more_than_one_decimal_place():
    with pytest.raises(ValueError, match="has more than 1 decimal place"):
        encode_coordinates(12.345, 0, 0)
    with pytest.raises(ValueError, match="has more than 1 decimal place"):
        encode_coordinates(0, 56.789, 0)
    with pytest.raises(ValueError, match="has more than 1 decimal place"):
        encode_coordinates(0, 0, 90.123)


def test_random_values():
    num_points: int = int(1e06)
    x = np.round(np.random.uniform(0, MAX_XY_VALUE, num_points), 1)
    y = np.round(np.random.uniform(0, MAX_XY_VALUE, num_points), 1)
    z = np.round(np.random.uniform(0, MAX_Z_VALUE, num_points), 1)
    encoded = encode_coordinates(x, y, z)
    decoded_x, decoded_y, decoded_z = decode_coordinates(encoded)
    np.testing.assert_almost_equal(decoded_x, x, decimal=6)
    np.testing.assert_almost_equal(decoded_y, y, decimal=6)
    np.testing.assert_almost_equal(decoded_z, z, decimal=6)
