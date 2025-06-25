import pandas as pd

from parq_blockmodel import RegularGeometry
from parq_blockmodel.utils import rotation_to_axis_orientation


def test_rotated_demo_block_model(tmp_path):
    from parq_blockmodel.utils import create_demo_blockmodel

    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    axis_azimuth = 30.0
    axis_dip = 0.0
    axis_plunge = 0.0

    blocks: pd.DataFrame = create_demo_blockmodel(shape=shape, block_size=block_size, corner=corner,
                                                  azimuth=axis_azimuth, dip=axis_dip, plunge=axis_plunge,
                                                  )

    # get the orientation of the axes
    axis_u, axis_v, axis_w = rotation_to_axis_orientation(axis_azimuth=axis_azimuth, axis_dip=axis_dip,
                                                          axis_plunge=axis_plunge)
    # create geometry that aligns with the demo block model
    geometry = RegularGeometry(block_size=block_size, corner=corner, shape=shape,
                               axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)

    assert blocks.index.equals(geometry.to_multi_index())
