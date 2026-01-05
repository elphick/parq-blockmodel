# test_reblocking.py

import numpy as np
import pandas as pd
import pytest
from parq_blockmodel.reblocking.downsample import downsample_attributes
from parq_blockmodel.reblocking.upsample import upsample_attributes
from parq_blockmodel.reblocking.conversion import tabular_to_3d_dict, dict_3d_to_tabular

def test_downsample_attributes_mean():
    arr = np.arange(8).reshape(2,2,2)
    attributes = {'value': arr}
    config = {'value': {'method': 'mean'}}
    result = downsample_attributes(attributes, 2, 2, 2, config)
    assert np.isclose(result['value'][0,0,0], arr.mean())

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