from pathlib import Path

import numpy as np
import pandas as pd

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils.demo_block_model import create_toy_blockmodel


def test_toy_blockmodel():
    grade_name = 'fe'
    grade_min = 5.0
    grade_max = 65.0
    toy_blocks: pd.DataFrame = create_toy_blockmodel(grade_name=grade_name,
                                                     grade_min=grade_min,
                                                     grade_max=grade_max)
    assert isinstance(toy_blocks, pd.DataFrame), "The result should be a pandas DataFrame."
    assert not toy_blocks.empty, "The DataFrame should not be empty."
    assert 'index_c' in toy_blocks.columns, "The DataFrame should contain 'index_c' column."
    assert 'index_f' in toy_blocks.columns, "The DataFrame should contain 'index_f' column."
    assert 'depth' in toy_blocks.columns, "The DataFrame should contain 'depth' column."
    assert grade_name in toy_blocks.columns, "The DataFrame should contain 'grade' column."
    assert toy_blocks[grade_name].mean() >= grade_min, \
        f"The mean of '{grade_name}' should be at least {grade_min}."
    assert toy_blocks[grade_name].mean() <= grade_max, \
        f"The mean of '{grade_name}' should be at most {grade_max}."


def test_toy_blockmodel_parquet(tmpdir):
    from parq_blockmodel.utils.demo_block_model import create_toy_blockmodel

    parquet_filepath = Path(tmpdir) / 'toy_blockmodel.parquet'
    grade_name = 'fe'
    grade_min = 5.0
    grade_max = 65.0
    create_toy_blockmodel(grade_name=grade_name,
                          grade_min=grade_min,
                          grade_max=grade_max,
                          parquet_filepath=parquet_filepath)

    assert parquet_filepath.exists(), "The parquet file should exist."
    toy_blocks = pd.read_parquet(parquet_filepath)
    assert isinstance(toy_blocks, pd.DataFrame), "The result should be a pandas DataFrame."
    assert not toy_blocks.empty, "The DataFrame should not be empty."
    assert 'fe' in toy_blocks.columns, "The DataFrame should contain 'fe' column."

def test_class_method(tmpdir):

    parquet_filepath = Path(tmpdir) / 'toy_blockmodel.parquet'

    pbm = ParquetBlockModel.create_toy_blockmodel(filename=parquet_filepath,
                                                  shape=(100, 75, 50),
                                                  corner=(0.0, 10.0, 0.0),
                                                  deposit_radii=(70, 30, 20),
                                                  deposit_center=tuple(0.5 * np.array((100, 75, 50))))
    assert isinstance(pbm, ParquetBlockModel), "The result should be a ParquetBlockModel instance."
    assert pbm.blockmodel_path == parquet_filepath.with_suffix('.pbm.parquet'), "The filename is unexpected."

    # p = pbm.plot(scalar='grade', threshold=True, enable_picking=True)
    # p.view_xy()
    # p.show()

    import pyvista as pv
    p = pv.Plotter()
    img = pbm.create_heatmap_from_threshold(attribute='grade', threshold=57.0, axis='z', return_array=False)
    p.add_mesh(img)
    p.view_xy()
    p.show_axes()
    p.show_grid()
    p.show(auto_close=False)

    pbm.plot_heatmap(attribute='grade', threshold=57.0, axis='z').show(renderer='browser')