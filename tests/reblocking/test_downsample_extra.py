"""Additional downsample tests to improve coverage of downsample.py"""
import numpy as np
import pytest

from parq_blockmodel.reblocking.downsample import downsample_attributes, normalise_missing_value


# ---------------------------------------------------------------------------
# sum with fill_ratio
# ---------------------------------------------------------------------------

def test_downsample_sum_with_fill_ratio():
    data = np.ones((2, 2, 2), dtype=float)
    fill = np.ones((2, 2, 2), dtype=float)  # fill ratio = 1.0 always
    attrs = {"data": data, "fill": fill}
    config = {
        "data": {"method": "sum", "fill_ratio": "fill"},
        "fill": {"method": "mean"},
    }
    result = downsample_attributes(attrs, 2, 2, 2, config)
    # sum of 8 ones / fill_ratio 1.0 = 8
    assert np.isclose(result["data"][0, 0, 0], 8.0)


# ---------------------------------------------------------------------------
# weighted_mean with fill_ratio
# ---------------------------------------------------------------------------

def test_downsample_weighted_mean_with_fill_ratio():
    grade = np.ones((2, 2, 2), dtype=float) * 5.0
    mass = np.ones((2, 2, 2), dtype=float)
    fill = np.ones((2, 2, 2), dtype=float)

    attrs = {"grade": grade, "mass": mass, "fill": fill}
    config = {
        "grade": {"method": "weighted_mean", "weight": "mass", "fill_ratio": "fill"},
        "mass": {"method": "sum"},
        "fill": {"method": "mean"},
    }
    result = downsample_attributes(attrs, 2, 2, 2, config)
    assert np.isclose(result["grade"][0, 0, 0], 5.0)


# ---------------------------------------------------------------------------
# mode with all-NaN block → None
# ---------------------------------------------------------------------------

def test_downsample_mode_all_nan_block():
    arr = np.array([[[np.nan, np.nan], [np.nan, np.nan]],
                    [[np.nan, np.nan], [np.nan, np.nan]]], dtype=object)
    attrs = {"cat": arr}
    config = {"cat": {"method": "mode"}}
    result = downsample_attributes(attrs, 2, 2, 2, config)
    assert result["cat"][0, 0, 0] is None


# ---------------------------------------------------------------------------
# replacement map applied
# ---------------------------------------------------------------------------

def test_downsample_replace_zero_with_nan():
    arr = np.zeros((2, 2, 2), dtype=float)
    attrs = {"val": arr}
    config = {"val": {"method": "sum", "replace": {0: np.nan}}}
    result = downsample_attributes(attrs, 2, 2, 2, config)
    assert np.isnan(result["val"][0, 0, 0])


def test_downsample_replace_none_string_becomes_nan():
    assert np.isnan(normalise_missing_value(None))
    assert np.isnan(normalise_missing_value("nan"))
    assert np.isnan(normalise_missing_value("None"))
    assert np.isnan(normalise_missing_value("NA"))
    assert np.isnan(normalise_missing_value("null"))


def test_downsample_replace_numeric_value_unchanged():
    assert normalise_missing_value(99) == 99
    assert normalise_missing_value(3.14) == 3.14


# ---------------------------------------------------------------------------
# invalid downsampling factors
# ---------------------------------------------------------------------------

def test_downsample_invalid_factors():
    arr = np.ones((2, 2, 2))
    config = {"val": {"method": "mean"}}
    with pytest.raises(ValueError, match="positive integers"):
        downsample_attributes({"val": arr}, 0, 1, 1, config)


def test_downsample_non_divisible_dimensions():
    arr = np.ones((3, 3, 3))
    config = {"val": {"method": "mean"}}
    with pytest.raises(ValueError, match="divisible"):
        downsample_attributes({"val": arr}, 2, 2, 2, config)


def test_downsample_weighted_mean_without_fill_ratio():
    """Covers the else branch of 'if fill' inside weighted_mean (line 119)."""
    grade = np.array([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]], dtype=float)
    mass = np.ones((2, 2, 2), dtype=float)

    attrs = {"grade": grade, "mass": mass}
    config = {
        "grade": {"method": "weighted_mean", "weight": "mass"},
        "mass": {"method": "sum"},
    }
    result = downsample_attributes(attrs, 2, 2, 2, config)
    # weighted mean with uniform weights = arithmetic mean
    expected = grade.mean()
    assert np.isclose(result["grade"][0, 0, 0], expected)


def test_downsample_unsupported_method():
    arr = np.ones((2, 2, 2))
    config = {"val": {"method": "unknown_method"}}
    with pytest.raises(ValueError, match="Unsupported aggregation method"):
        downsample_attributes({"val": arr}, 2, 2, 2, config)


