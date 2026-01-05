from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from parq_blockmodel import ParquetBlockModel

@pd.api.extensions.register_dataframe_accessor("to_pyvista")
class PyVistaAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self,
                 grid_type="image",
                 geometry=None,
                 block_size=None,
                 fill_value=np.nan
                 ) -> "pv.ImageData | pv.StructuredGrid | pv.UnstructuredGrid":
        if grid_type == "image":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_image_data
            if geometry is None:
                from parq_blockmodel import RegularGeometry
                geometry = RegularGeometry.from_multi_index(self._obj.index)
            return df_to_pv_image_data(self._obj, geometry, fill_value=fill_value)
        elif grid_type == "structured":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_structured_grid
            return df_to_pv_structured_grid(self._obj, block_size=block_size)
        elif grid_type == "unstructured":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_unstructured_grid
            if block_size is None:
                raise ValueError("block_size must be provided for unstructured grid.")
            return df_to_pv_unstructured_grid(self._obj, block_size=block_size)
        else:
            raise ValueError(f"Invalid grid_type: {grid_type}. Choose 'image', 'structured', or 'unstructured'.")


@pd.api.extensions.register_dataframe_accessor("to_parquet_blockmodel")
class ParquetBlockModelAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, filename, **kwargs) -> "ParquetBlockModel":
        from parq_blockmodel import ParquetBlockModel
        # You may want to infer geometry or pass additional arguments as needed
        return ParquetBlockModel.from_dataframe(self._obj, filename=filename, **kwargs)