import numpy as np
import pandas as pd

from parq_blockmodel.utils import create_demo_blockmodel


def test_create_blockmodel():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    df = create_demo_blockmodel(shape, block_size, corner)
    # Check shape and index
    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ['x', 'y', 'z']
    assert len(df) == np.prod(shape)
    # Check columns
    assert 'c_order_xyz' in df.columns
    assert 'f_order_zyx' in df.columns
    assert 'depth' in df.columns
    # Check that dx/dy/dz are not present
    assert 'dx' not in df.columns
    assert 'dy' not in df.columns
    assert 'dz' not in df.columns

def test_ordering():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    df = create_demo_blockmodel(shape, block_size, corner)
    # Check c_order_xyz: C order (xyz fastest to slowest)
    assert np.array_equal(df['c_order_xyz'].values, np.arange(np.prod(shape)))
    # Check f_order_zyx: F order (zyx fastest to slowest)
    expected_f_order = np.arange(np.prod(shape)).reshape(shape, order='C').ravel(order='F')
    assert np.array_equal(df['f_order_zyx'].values, expected_f_order)

def test_ordering_after_rotation():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    azimuth = 45.0
    dip = 30.0
    plunge = 15.0
    df = create_demo_blockmodel(shape, block_size, corner, azimuth, dip, plunge)
    # Check c_order_xyz
    assert np.array_equal(df['c_order_xyz'].values, np.arange(np.prod(shape)))
    # Check f_order_zyx: F order (zyx fastest to slowest)
    expected_f_order = np.arange(np.prod(shape)).reshape(shape, order='C').ravel(order='F')
    assert np.array_equal(df['f_order_zyx'].values, expected_f_order)
