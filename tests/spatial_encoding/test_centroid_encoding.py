import pytest
import numpy as np

from parq_blockmodel.utils import (
    decode_coordinates,
    decode_frame_coordinates,
    decode_world_coordinates,
    encode_coordinates,
    encode_frame_coordinates,
    encode_world_coordinates,
    get_id_encoding_params,
    get_world_id_encoding_params,
)
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


def test_frame_encoding_aliases_match_world_encoding():
    x = np.array([500000.1, 500010.2])
    y = np.array([7000000.3, 7000010.4])
    z = np.array([100.5, 110.6])
    offset = (500000.0, 7000000.0, 100.0)

    world_encoded = encode_world_coordinates(x, y, z, offset=offset, scale=10.0)
    frame_encoded = encode_frame_coordinates(x, y, z, offset=offset, scale=10.0)
    np.testing.assert_array_equal(world_encoded, frame_encoded)

    xw, yw, zw = decode_world_coordinates(world_encoded, offset=offset, scale=10.0)
    xf, yf, zf = decode_frame_coordinates(frame_encoded, offset=offset, scale=10.0)
    np.testing.assert_allclose(xw, xf)
    np.testing.assert_allclose(yw, yf)
    np.testing.assert_allclose(zw, zf)


def test_get_id_encoding_params_alias():
    payload = {
        "offset": {"x": 1.0, "y": 2.0, "z": 3.0},
        "quantization": {"scale": 10.0},
    }
    assert get_id_encoding_params(payload) == get_world_id_encoding_params(payload)

