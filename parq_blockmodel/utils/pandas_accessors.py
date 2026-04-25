from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from parq_blockmodel import ParquetBlockModel


def _geometry_from_df(df: pd.DataFrame):
    """Infer a RegularGeometry from a DataFrame.

    Resolution order:
    1. ``df.attrs["parq-blockmodel"]`` – full production metadata.
    2. ``df.attrs["geometry"]`` – lightweight demo-blockmodel metadata.
    3. xyz-indexed MultiIndex – infer from centroid distribution (non-rotated only).

    Raises ValueError if geometry cannot be determined.
    """
    from parq_blockmodel import RegularGeometry
    from parq_blockmodel.geometry import LocalGeometry, WorldFrame

    # 1. Full parq-blockmodel metadata
    if "parq-blockmodel" in df.attrs:
        return RegularGeometry.from_attrs(df.attrs, key="parq-blockmodel")

    # 2. Lightweight demo-blockmodel attrs (produced by create_demo_blockmodel)
    if "geometry" in df.attrs:
        g = df.attrs["geometry"]
        from parq_blockmodel.utils.geometry_utils import angles_to_axes
        rot = g.get("rotation", {})
        axis_u, axis_v, axis_w = angles_to_axes(
            axis_azimuth=rot.get("azimuth", 0.0),
            axis_dip=rot.get("dip", 0.0),
            axis_plunge=rot.get("plunge", 0.0),
        )
        return RegularGeometry(
            local=LocalGeometry(
                corner=g["corner"],
                block_size=g["block_size"],
                shape=g["shape"],
            ),
            world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
        )

    raise ValueError(
        "Cannot determine geometry for the DataFrame. "
        "Attach geometry via df.attrs['parq-blockmodel'], "
        "df.attrs['geometry'], or use an xyz-indexed MultiIndex."
    )


@pd.api.extensions.register_dataframe_accessor("to_pyvista")
class PyVistaAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self,
                 grid_type="image",
                 geometry=None,
                 block_size=None,
                 fill_value=np.nan,
                 frame: str = "world"
                 ) -> "pv.ImageData | pv.StructuredGrid | pv.UnstructuredGrid":
        if grid_type == "image":
            from parq_blockmodel.utils.pyvista.pyvista_utils import df_to_pv_image_data
            if geometry is None:
                geometry = _geometry_from_df(self._obj)
            return df_to_pv_image_data(self._obj, geometry, fill_value=fill_value, frame=frame)
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
        return ParquetBlockModel.from_dataframe(self._obj, filename=filename, **kwargs)