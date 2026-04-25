from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert 'block_id' in toy_blocks.columns, "The DataFrame should contain 'block_id' column."
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


def test_toy_blockmodel_relative_noise_is_reproducible_and_breaks_ties():
    grade_name = "fe"
    kwargs = dict(
        shape=(20, 15, 10),
        block_size=(1.0, 1.0, 1.0),
        corner=(0.0, 0.0, 0.0),
        axis_azimuth=0.0,
        axis_dip=0.0,
        axis_plunge=0.0,
        deposit_bearing=0.0,
        deposit_dip=0.0,
        deposit_plunge=0.0,
        grade_name=grade_name,
        grade_min=45.0,
        grade_max=70.0,
        deposit_center=(10.0, 7.5, 5.0),
        deposit_radii=(8.0, 5.0, 3.0),
        noise_rel=1e-3,
        noise_seed=42,
    )
    df_a = create_toy_blockmodel(**kwargs)
    df_b = create_toy_blockmodel(**kwargs)

    pd.testing.assert_series_equal(df_a[grade_name], df_b[grade_name], check_names=False)
    assert int((df_a[grade_name] == float(df_a[grade_name].max())).sum()) == 1


def test_toy_blockmodel_rejects_both_noise_std_and_noise_rel():
    with pytest.raises(ValueError, match="noise_std or noise_rel"):
        create_toy_blockmodel(noise_std=0.1, noise_rel=1e-3)

@pytest.mark.gui
def test_class_method(tmpdir):

    parquet_filepath = Path(tmpdir) / 'toy_blockmodel.parquet'

    pbm = ParquetBlockModel.create_toy_blockmodel(filename=parquet_filepath,
                                                  shape=(100, 75, 50),
                                                  corner=(0.0, 10.0, 0.0),
                                                  deposit_radii=(70, 30, 20),
                                                  deposit_center=tuple(0.5 * np.array((100, 75, 50))))
    assert isinstance(pbm, ParquetBlockModel), "The result should be a ParquetBlockModel instance."
    assert pbm.blockmodel_path == parquet_filepath.with_suffix('.pbm'), "The filename is unexpected."

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