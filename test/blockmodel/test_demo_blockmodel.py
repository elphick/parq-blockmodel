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
    assert 'c_style_xyz' in df.columns
    assert 'f_style_zyx' in df.columns
    assert 'depth' in df.columns
    # Check that dx/dy/dz are not present
    assert 'dx' not in df.columns
    assert 'dy' not in df.columns
    assert 'dz' not in df.columns

