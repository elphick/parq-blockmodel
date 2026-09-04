"""Additional tests for parq_blockmodel.utils.spatial_encoding"""
import numpy as np
import pandas as pd
import pytest

from parq_blockmodel.utils.spatial_encoding import (
    get_id_encoding_params,
    encode_frame_coordinates,
    decode_frame_coordinates,
    multiindex_to_encoded_index,
    encoded_index_to_multiindex,
)
from parq_blockmodel.io.ingest_utils import build_world_id_encoding_from_xyz


# ---------------------------------------------------------------------------
# get_id_encoding_params
# ---------------------------------------------------------------------------

def test_get_id_encoding_params_none():
    offset, scale, bits = get_id_encoding_params(None)
    assert offset == (0.0, 0.0, 0.0)
    assert scale == (10.0, 10.0, 10.0)
    assert bits == (24, 24, 16)


def test_get_id_encoding_params_empty_dict():
    offset, scale, bits = get_id_encoding_params({})
    assert offset == (0.0, 0.0, 0.0)
    assert scale == (10.0, 10.0, 10.0)
    assert bits == (24, 24, 16)


def test_get_id_encoding_params_scale_as_list():
    payload = {
        "offset": {"x": 100.0, "y": 200.0, "z": 5.0},
        "quantization": {"scale": [10.0, 10.0, 10.0]},
    }
    offset, scale, bits = get_id_encoding_params(payload)
    assert offset == (100.0, 200.0, 5.0)
    assert scale == (10.0, 10.0, 10.0)
    assert bits == (24, 24, 16)


# ---------------------------------------------------------------------------
# encode_frame_coordinates – scalar path & error paths
# ---------------------------------------------------------------------------

def test_encode_frame_coordinates_scalar_roundtrip():
    x, y, z = 100.0, 200.0, 5.0
    offset = (100.0, 200.0, 5.0)
    encoded = encode_frame_coordinates(x, y, z, offset=offset, scale=10.0)
    assert isinstance(encoded, int)
    rx, ry, rz = decode_frame_coordinates(encoded, offset=offset, scale=10.0)
    assert np.isclose(float(rx), x)
    assert np.isclose(float(ry), y)
    assert np.isclose(float(rz), z)


def test_encode_frame_coordinates_negative_raises():
    # offset bigger than coordinate → negative quantised value
    with pytest.raises(ValueError, match="negative quantized"):
        encode_frame_coordinates(0.0, 0.0, 0.0, offset=(1.0, 0.0, 0.0), scale=10.0)


def test_encode_frame_coordinates_overflow_raises():
    # coordinate enormously beyond integer range
    big = 1e9
    with pytest.raises(ValueError, match="overflow"):
        encode_frame_coordinates(big, big, big, offset=(0.0, 0.0, 0.0), scale=10.0)


# ---------------------------------------------------------------------------
# multiindex_to_encoded_index & encoded_index_to_multiindex
# ---------------------------------------------------------------------------

def test_multiindex_roundtrip():
    xs = [10.0, 20.0]
    ys = [30.0, 40.0]
    zs = [1.0, 2.0]
    mi = pd.MultiIndex.from_arrays([xs, ys, zs], names=["x", "y", "z"])
    encoded = multiindex_to_encoded_index(mi)
    assert len(encoded) == 2
    assert encoded.name == "encoded_xyz"

    restored = encoded_index_to_multiindex(encoded)
    assert list(restored.names) == ["x", "y", "z"]
    np.testing.assert_allclose(restored.get_level_values("x"), xs, atol=1e-1)
    np.testing.assert_allclose(restored.get_level_values("y"), ys, atol=1e-1)
    np.testing.assert_allclose(restored.get_level_values("z"), zs, atol=1e-1)


def test_build_world_id_encoding_zero_offset_default():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 5.0, 15.0])
    z = np.array([0.0, 1.0, 2.0])
    enc = build_world_id_encoding_from_xyz(x, y, z, scale=10.0)
    # Default builder should produce zero offset
    assert enc["offset"]["x"] == 0.0
    assert enc["offset"]["y"] == 0.0
    assert enc["offset"]["z"] == 0.0
    assert enc["range_after_offset"]["x_min"] == 0.0


def test_build_world_id_encoding_overflow_raises_and_suggests_offset():
    # Construct coords that exceed encoder capacity for zero-offset encoding
    x = np.array([-1000.0, 1701000.0])
    y = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    with pytest.raises(ValueError) as exc:
        build_world_id_encoding_from_xyz(x, y, z, scale=10.0)
    msg = str(exc.value)
    assert "zero-offset" in msg
    # Suggested offset should be present and reflect the floored minimum
    assert "offset.x" in msg
    assert "-1000" in msg
