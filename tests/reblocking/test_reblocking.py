# test_reblocking.py

import numpy as np
import pandas as pd
import pytest
from parq_blockmodel.reblocking.downsample import downsample_attributes
from parq_blockmodel.reblocking.upsample import upsample_attributes
from parq_blockmodel.reblocking.conversion import tabular_to_3d_dict, dict_3d_to_tabular, to_numeric
from parq_blockmodel.reblocking.reblocking import _calculate_factors

def test_downsample_attributes_mean():
    arr = np.arange(8).reshape(2,2,2)
    attributes = {'value': arr}
    config = {'value': {'method': 'mean'}}
    result = downsample_attributes(attributes, 2, 2, 2, config)
    assert result['value'].dtype == arr.dtype
    assert result['value'][0,0,0] == np.rint(arr.mean()).astype(arr.dtype)

def test_downsample_attributes_mode():
    arr = np.array([[[1,1],[2,2]],[[1,1],[2,2]]])
    attributes = {'cat': arr}
    config = {'cat': {'method': 'mode'}}
    result = downsample_attributes(attributes, 2, 2, 2, config)
    assert result['cat'][0,0,0] == 1

def test_upsample_attributes_linear():
    arr = np.ones((2,2,2))
    attributes = {'value': arr}
    config = {'value': 'linear'}
    result = upsample_attributes(attributes, 2, 2, 2, config)
    assert result['value'].shape == (4,4,4)
    assert np.allclose(result['value'], 1)


def test_upsample_attributes_preserves_float32_dtype():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    result = upsample_attributes({'value': arr}, 2, 2, 2, {'value': 'linear'})
    assert result['value'].dtype == np.float32


def test_upsample_attributes_requires_config_for_all_attributes():
    attrs = {
        "a": np.ones((2, 2, 2), dtype=float),
        "b": np.ones((2, 2, 2), dtype=float),
    }
    with pytest.raises(ValueError, match="must specify a method for every attribute"):
        upsample_attributes(attrs, 2, 2, 2, {"a": "linear"})


def test_upsample_attributes_nearest_keeps_class_codes_without_intermediate_values():
    class_codes = np.array([0, 2], dtype=np.int32).reshape(2, 1, 1)
    out = upsample_attributes({"class_code": class_codes}, 2, 1, 1, {"class_code": "nearest"})
    flattened = out["class_code"].ravel(order="C")
    assert set(flattened.tolist()) <= {0, 2}
    np.testing.assert_array_equal(flattened, np.array([0, 0, 2, 2], dtype=np.int32))


def test_upsample_attributes_mode_uses_lowest_code_on_tie():
    class_codes = np.array([0, 2], dtype=np.int32).reshape(2, 1, 1)
    out = upsample_attributes({"class_code": class_codes}, 2, 1, 1, {"class_code": "mode"})
    flattened = out["class_code"].ravel(order="C")
    assert flattened[1] == 0


def test_upsample_attributes_linear_remains_available_for_integer_numeric_values():
    values = np.array([0, 2], dtype=np.int32).reshape(2, 1, 1)
    out = upsample_attributes({"value": values}, 2, 1, 1, {"value": "linear"})
    flattened = out["value"].ravel(order="C")
    assert set(flattened.tolist()) == {0, 1, 2}


def test_upsample_attributes_parent_inherits_exact_parent_values():
    values = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    out = upsample_attributes({"value": values}, 2, 2, 2, {"value": "parent"})
    expected = np.repeat(np.repeat(np.repeat(values, 2, axis=0), 2, axis=1), 2, axis=2)
    np.testing.assert_array_equal(out["value"], expected)


def test_upsample_attributes_rejects_unsupported_method():
    arr = np.ones((2, 2, 2), dtype=float)
    with pytest.raises(ValueError, match="Unsupported upsampling method"):
        upsample_attributes({"value": arr}, 2, 2, 2, {"value": "idw"})


def test_upsample_blockmodel_rejects_unsupported_method(tmp_path):
    from parq_blockmodel import ParquetBlockModel

    pbm = ParquetBlockModel.create_demo_block_model(
        tmp_path / "upsample_unsupported_method.parquet", shape=(2, 2, 2)
    )

    with pytest.raises(ValueError, match="Unsupported upsampling method"):
        pbm.upsample((0.5, 0.5, 0.5), upsample_config={"depth": "idw", "depth_category": "mode"})


def test_downsample_attributes_preserves_float32_dtype():
    arr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    result = downsample_attributes({'value': arr}, 2, 2, 2, {'value': {'method': 'mean'}})
    assert result['value'].dtype == np.float32


def test_downsample_attributes_int_overflow_warns_and_keeps_promoted_dtype():
    arr = np.full((2, 2, 2), 100, dtype=np.int8)
    config = {'value': {'method': 'sum'}}

    with pytest.warns(RuntimeWarning, match='exceeds representable range'):
        result = downsample_attributes({'value': arr}, 2, 2, 2, config)

    assert result['value'].dtype == np.float64


def test_downsample_integer_mean_uses_bankers_rounding_on_ties():
    attrs = {
        'tie_to_even': np.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32).reshape(2, 2, 2),
        'tie_to_odd': np.array([3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int32).reshape(2, 2, 2),
    }
    config = {
        'tie_to_even': {'method': 'mean'},
        'tie_to_odd': {'method': 'mean'},
    }

    result = downsample_attributes(attrs, 2, 2, 2, config)

    assert result['tie_to_even'].dtype == np.int32
    assert result['tie_to_odd'].dtype == np.int32
    assert result['tie_to_even'][0, 0, 0] == 2  # np.rint(2.5) -> 2
    assert result['tie_to_odd'][0, 0, 0] == 4   # np.rint(3.5) -> 4

def test_tabular_to_3d_dict_and_back_dense():
    # Create a dense 2x2x2 grid for i, j, k
    i, j, k = np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij')
    df = pd.DataFrame({
        'i': i.ravel(),
        'j': j.ravel(),
        'k': k.ravel(),
        'val': np.arange(1, 9)  # Example values 1..8
    }).set_index(['i', 'j', 'k'])
    arrays, categories = tabular_to_3d_dict(df)
    df2 = dict_3d_to_tabular(arrays, categories)
    assert set(df2['val']) == {1, 2, 3, 4, 5, 6, 7, 8}
    pd.testing.assert_frame_equal(df2, df)

def test_tabular_to_3d_dict_and_back_sparse():
    df = pd.DataFrame({
        'i': [0, 1, 1],
        'j': [0, 0, 1],
        'k': [0, 0, 1],
        'val': [1, 2, 3]
    }).set_index(['i', 'j', 'k'])
    df['val'] = df['val'].astype('Int64')  # Pandas nullable integer type
    arrays, categories = tabular_to_3d_dict(df)
    df2 = dict_3d_to_tabular(arrays, categories)
    assert set(df2['val'].dropna()) == {1, 2, 3}  # Ensure all values are present
    assert df2['val'].dropna().values == df['val'].values
    # assert df2['val'].dtype == df['val'].dtype

def test_upsample_and_downsample_mixed_types():
    # Create mixed type DataFrame
    df = pd.DataFrame({
        'i': [0, 0, 1, 1],
        'j': [0, 1, 0, 1],
        'k': [0, 0, 0, 0],
        'float_val': [1.0, 2.0, 3.0, 4.0],
        'str_val': ['a', 'b', 'a', 'b'],
        'cat_val': pd.Categorical(['dog', 'cat', 'dog', 'cat']),
        'nullable_int': pd.Series([1, None, 3, 4], dtype='Int64')
    }).set_index(['i', 'j', 'k'])
    arrays, _ = tabular_to_3d_dict(df)

    # Upsample by factor 2
    upsampled = upsample_attributes(arrays, 2, 2, 1, {
        'float_val': 'linear',
        'str_val': 'nearest',
        'cat_val': 'nearest',
        'nullable_int': 'nearest'
    })
    # Downsample back by factor 2
    downsampled = downsample_attributes(upsampled, 2, 2, 1, {
        'float_val': {'method': 'mean'},
        'str_val': {'method': 'mode'},
        'cat_val': {'method': 'mode'},
        'nullable_int': {'method': 'mode'}
    })

    # Check shapes
    assert downsampled['float_val'].shape == (2, 2, 1)
    assert downsampled['str_val'].shape == (2, 2, 1)
    assert downsampled['cat_val'].shape == (2, 2, 1)
    assert downsampled['nullable_int'].shape == (2, 2, 1)

    # Check types
    assert downsampled['float_val'].dtype.kind == 'f'
    assert downsampled['str_val'].dtype == object
    assert downsampled['cat_val'].dtype == object or isinstance(downsampled['cat_val'][0,0,0], str)
    assert downsampled['nullable_int'].dtype == object or str(downsampled['nullable_int'].dtype).startswith('Int')

    # Check values
    assert downsampled['str_val'][0,0,0] in [0, 1]  #['a', 'b'] # using codes
    assert downsampled['cat_val'][0,0,0] in [0, 1]  #['dog', 'cat']  # using codes
    assert downsampled['nullable_int'][0,0,0] in [1, 3, 4]
    assert np.isclose(downsampled['float_val'][0,0,0], np.mean([0.5, 1.5, 2.5]))


def test_downsample_attributes_fill_ratio_must_be_single_variable():
    arr = np.ones((2, 2, 2), dtype=float)
    attrs = {
        "a": arr,
        "b": arr,
        "fill_a": arr,
        "fill_b": arr,
    }
    config = {
        "a": {"method": "sum", "fill_ratio": "fill_a"},
        "b": {"method": "sum", "fill_ratio": "fill_b"},
    }

    with pytest.raises(ValueError, match="Only one fill_ratio is supported"):
        downsample_attributes(attrs, 2, 2, 2, config)


def test_downsample_attributes_replace_requires_mapping():
    arr = np.arange(8, dtype=float).reshape(2, 2, 2)
    attrs = {"value": arr}
    config = {"value": {"method": "sum", "replace": "nan"}}

    with pytest.raises(ValueError, match="replace value must be a dictionary"):
        downsample_attributes(attrs, 2, 2, 2, config)


def test_downsample_attributes_weighted_mean_requires_basis_array():
    arr = np.arange(8, dtype=float).reshape(2, 2, 2)
    attrs = {"grade": arr}
    config = {"grade": {"method": "weighted_mean"}}

    with pytest.raises(ValueError, match="Missing basis array"):
        downsample_attributes(attrs, 2, 2, 2, config)


def test_tabular_to_3d_dict_requires_ijk_index_names():
    df = pd.DataFrame({"x": [0], "y": [0], "z": [0], "value": [1]}).set_index(["x", "y", "z"])
    with pytest.raises(ValueError, match="DataFrame index must be a MultiIndex"):
        tabular_to_3d_dict(df)


def test_dict_3d_to_tabular_rejects_shape_mismatch():
    arrays = {
        "a": np.zeros((2, 2, 2)),
        "b": np.zeros((2, 2, 1)),
    }
    with pytest.raises(ValueError, match="Shape mismatch"):
        dict_3d_to_tabular(arrays)


def test_to_numeric_preserves_nullable_integer_dtype():
    arr = pd.array([1, None, 3], dtype="Int64")
    numeric, restore = to_numeric(arr)
    restored = restore(numeric)

    assert numeric.dtype.kind == "f"
    assert str(restored.dtype) == "Int64"


def test_calculate_factors_rejects_mixed_up_and_downsampling(tmp_path):
    from parq_blockmodel import ParquetBlockModel

    pbm = ParquetBlockModel.create_demo_block_model(tmp_path / "factor_mixed.parquet", shape=(4, 4, 4))
    with pytest.raises(ValueError, match="Cannot mix upsampling and downsampling"):
        _calculate_factors(pbm, (2.0, 0.5, 2.0))

