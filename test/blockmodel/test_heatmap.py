from pathlib import Path

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.utils import create_demo_blockmodel


def test_heatmap_array(tmpdir):
    import numpy as np

    parquet_filepath: Path = Path(tmpdir) / 'test.parquet'
    shape = (3, 3, 3)
    pbm: ParquetBlockModel = ParquetBlockModel.create_demo_block_model(filename=parquet_filepath,
                                                                       shape=shape,
                                                                       )
    heatmap: np.ndarray = pbm.create_heatmap_from_threshold(attribute='c_index',
                                                            threshold=4,
                                                            axis='z',
                                                            return_array=True)

    assert heatmap.shape == (shape[0], shape[1]), "Heatmap shape does not match expected dimensions."
    assert np.all(heatmap >= 0), "Heatmap contains negative values, which is unexpected."
    assert np.all(heatmap <= shape[2]), "Heatmap contains values greater than the number of blocks in the z-axis."
    assert np.sum(heatmap) > 0, "Heatmap is empty, expected some non-zero values."
    print("Heatmap shape:", heatmap.shape)

def test_heatmap_plot(tmpdir):
    parquet_filepath: Path = Path(tmpdir) / 'test.parquet'
    shape = (3, 3, 3)
    pbm: ParquetBlockModel = ParquetBlockModel.create_demo_block_model(filename=parquet_filepath,
                                                                       shape=shape,
                                                                       )
    fig = pbm.plot_heatmap(attribute='c_index',
                           threshold=4,
                           axis='z')

    assert fig is not None, "Heatmap figure is None."
    assert len(fig.data) > 0, "Heatmap figure contains no data."
    print("Heatmap plot created successfully.")