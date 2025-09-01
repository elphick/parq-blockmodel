from parq_blockmodel.utils import create_demo_blockmodel
from parq_blockmodel.utils import pandas_accessors # noqa: F401

def test_df_to_image_data():
    import pyvista as pv

    df = create_demo_blockmodel()
    mesh: pv.ImageData = df.to_image_data()
    print(mesh)