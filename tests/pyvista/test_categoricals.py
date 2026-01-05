import pandas as pd
import numpy as np
import pyvista as pv

from parq_blockmodel import RegularGeometry
from parq_blockmodel.utils import create_demo_blockmodel
from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_image_data
from parq_blockmodel.utils.pyvista.categorical_utils import load_mapping_dict

def test_df_to_pv_image_data_categorical_encoding():
    df = create_demo_blockmodel()  # includes a categorical called 'depth_category'
    geometry = RegularGeometry.from_multi_index(df.index)
    grid = df_to_pv_image_data(df, geometry, categorical_encode=True)

    # Check that the categorical column is encoded as integers
    arr = grid.cell_data['depth_category']
    assert set(arr) == {0, 1}

    # Check that the mapping is stored and can be loaded
    mapping = load_mapping_dict(grid, 'depth_category')
    assert isinstance(mapping, dict)
    assert set(mapping.values()) == {'shallow', 'deep'}