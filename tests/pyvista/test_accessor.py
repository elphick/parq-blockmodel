import numpy as np
import pytest

from parq_blockmodel.utils import create_demo_blockmodel
from parq_blockmodel.utils import pandas_accessors # noqa: F401
import pyvista as pv

def dense_f_to_c_order(arr, shape):
    arr_reshaped = arr.reshape(shape, order='F')
    return arr_reshaped.ravel(order='C')

def dense_c_to_f_order(arr, shape):
    arr_reshaped = arr.reshape(shape, order='C')
    return arr_reshaped.ravel(order='F')

def sparse_to_dense(arr, indices, size):
    dense = np.empty(size, dtype=arr.dtype)
    dense[:] = np.nan  # or another fill value
    dense[indices] = arr
    return dense


def test_df_to_image_data_simple():
    df = create_demo_blockmodel().drop(columns=['depth_category'])
    mesh: pv.ImageData = df.to_pyvista(grid_type='image')
    assert isinstance(mesh, pv.ImageData)
    assert df.shape[0] == mesh.n_cells
    assert df.columns.tolist() == list(mesh.cell_data.keys())
    # verify the values match when ordered by index
    for col in df.columns:
        cell_dims = mesh.dimensions - np.array([1, 1, 1])
        np.testing.assert_array_equal(
            dense_c_to_f_order(df[col].values, shape=cell_dims),
            mesh.cell_data[col]
        )

def test_df_to_image_data_categorical():
    # The cat becomes a string in pyvista (numpy),
    # but tests still pass when comparing to cat.
    df = create_demo_blockmodel()  # includes a categorical called 'depth_category'
    mesh: pv.ImageData = df.to_pyvista(grid_type='image')
    assert isinstance(mesh, pv.ImageData)
    assert df.shape[0] == mesh.n_cells
    assert df.columns.tolist() == list(mesh.cell_data.keys())
    # verify the values match when ordered by index
    for col in df.columns:
        cell_dims = mesh.dimensions - np.array([1, 1, 1])
        np.testing.assert_array_equal(
            dense_c_to_f_order(df[col].values, shape=cell_dims),
            mesh.cell_data[col]
        )




@pytest.mark.skip(reason="Quarantined: StructuredGrid path in to_pyvista() not yet implemented; "
                   "unquarantine once grid_type='structured' is supported for sparse models")
def test_df_to_structured_grid():
    df = create_demo_blockmodel()
    # remove some cells to make it non-image
    df = df[df['block_id'] % 2 == 0].copy()
    mesh: pv.ImageData = df.to_pyvista(grid_type='structured')
    assert isinstance(mesh, pv.StructuredGrid)
    assert df.shape[0] == mesh.n_cells
    assert df.columns.tolist() == list(mesh.cell_data.keys())

def test_df_to_image_data_frame_local_vs_world():
    df = create_demo_blockmodel(azimuth=30.0)

    local_mesh: pv.ImageData = df.to_pyvista(grid_type='image', frame='local')
    world_mesh: pv.ImageData = df.to_pyvista(grid_type='image', frame='world')

    np.testing.assert_allclose(np.asarray(local_mesh.direction_matrix), np.eye(3))
    assert not np.allclose(np.asarray(world_mesh.direction_matrix), np.eye(3))
