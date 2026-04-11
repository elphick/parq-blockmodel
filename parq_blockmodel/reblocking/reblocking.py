"""
reblocking.py

Module for reblocking operations on block models.
"""
from typing import TYPE_CHECKING
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parq_blockmodel.reblocking.conversion import tabular_to_3d_dict, dict_3d_to_tabular
from parq_blockmodel.reblocking.downsample import downsample_attributes
from parq_blockmodel.reblocking.upsample import upsample_attributes

if TYPE_CHECKING:
    from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.geometry import RegularGeometry


def _validate_params(config, new_block_size):
    if not isinstance(new_block_size, tuple) or len(new_block_size) != 3:
        raise ValueError("new_block_size must be a tuple of three floats (dx, dy, dz).")
    if any(dim <= 0 for dim in new_block_size):
        raise ValueError("All dimensions in new_block_size must be positive.")
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")

def _prepare_arrays_and_index(blockmodel) -> tuple[dict[str, np.ndarray], dict[str, pd.Index]]:
    """Prepare dense attribute arrays and category indices for reblocking.

    This helper reads the blockmodel on a dense ijk grid and converts it
    into 3D arrays suitable for up/downsampling. Geometry-derived columns
    (coordinates and linear indices) are excluded and later recomputed from
    the new :class:`RegularGeometry` to avoid inconsistencies.
    """

    # Read a dense ijk-indexed view of the data. ``ParquetBlockModel.read``
    # already returns a C-order (i, j, k) MultiIndex when ``index="ijk"``,
    # which is exactly what ``tabular_to_3d_dict`` expects.
    #
    # Only load true attributes; drop geometry/linear-index helpers which
    # will be regenerated for the reblocked grid.
    geometry_cols = {"x", "y", "z", "i", "j", "k", "index_c", "index_f"}
    cols = [c for c in blockmodel.columns if c not in geometry_cols]
    df: pd.DataFrame = blockmodel.read(columns=cols, index="ijk", dense=True)
    arrays, categories = tabular_to_3d_dict(df)
    return arrays, categories

def _calculate_factors(blockmodel, new_block_size):
    old_shape = blockmodel.geometry.shape
    old_block_size = blockmodel.geometry.block_size

    factors = []
    for ob, nb in zip(old_block_size, new_block_size):
        factor = nb / ob
        if np.isclose(factor, round(factor)):
            factors.append(round(factor))
        elif np.isclose(1 / factor, round(1 / factor)):
            factors.append(1 / round(1 / factor))
        else:
            raise ValueError("Block sizes must be integer multiples or divisors for up/downsampling.")

    fx, fy, fz = factors
    if all(f >= 1 for f in factors):
        # Downsampling
        new_shape = tuple(int(o // f) for o, f in zip(old_shape, factors))
    elif all(f <= 1 for f in factors):
        # Upsampling
        up_factors = [int(round(1 / f)) for f in factors]
        new_shape = tuple(int(o * uf) for o, uf in zip(old_shape, up_factors))
    else:
        raise ValueError("Cannot mix upsampling and downsampling in one operation.")

    return fx, fy, fz, new_shape

def _build_new_geometry(blockmodel, new_block_size, new_shape) -> RegularGeometry:
    return RegularGeometry(
        corner=blockmodel.geometry.corner,
        block_size=new_block_size,
        shape=new_shape,
        axis_u=blockmodel.geometry.axis_u,
        axis_v=blockmodel.geometry.axis_v,
        axis_w=blockmodel.geometry.axis_w,
    )


def downsample_blockmodel(blockmodel, new_block_size, aggregation_config) -> "ParquetBlockModel":
    """
    Downsample a block model to a coarser grid with specified aggregation methods for each attribute.
    This function supports downsampling of both categorical and numeric attributes.
    Args:
        blockmodel: ParquetBlockModel instance.
        new_block_size: tuple of floats (dx, dy, dz) for the new block size.
        aggregation_config: dict mapping attribute names to aggregation methods.

    Example:
        aggregation_config = {
            'grade': {'method': 'weighted_mean', 'weight': 'dry_mass'},
            'density': {'method': 'weighted_mean', 'weight': 'volume'},
            'dry_mass': {'method': 'sum'},
            'volume': {'method': 'sum'},
            'rock_type': {'method': 'mode'}
        }

    Returns:
        ParquetBlockModel: A new ParquetBlockModel instance with the downsampled grid.
    """
    from parq_blockmodel import ParquetBlockModel

    _validate_params(aggregation_config, new_block_size)

    arrays, categories = _prepare_arrays_and_index(blockmodel)

    fx, fy, fz, new_shape = _calculate_factors(blockmodel, new_block_size)

    if fx > 1 or fy > 1 or fz > 1:
        # Downsampling: expect dict of dicts
        if not all(isinstance(v, dict) for v in (aggregation_config or {}).values()):
            raise ValueError("Downsampling config must be a dict of dicts per attribute.")
        arrays = downsample_attributes(arrays, fx, fy, fz, aggregation_config or {})
    elif fx < 1 or fy < 1 or fz < 1:
        raise ValueError("reblocking factors imply upsampling, not downsampling.")
    elif any(dim < 1 for dim in (fx, fy, fz)) and any(dim > 1 for dim in (fx, fy, fz)):
        raise ValueError("Reblocking factors cannot specify upsample and downsample at the same time.")

    # Convert back to DataFrame (ijk-indexed attributes only)
    new_geometry = _build_new_geometry(blockmodel, new_block_size, new_shape)
    # Use canonical .pbm suffix expected by ParquetBlockModel
    new_path = blockmodel.blockmodel_path.with_name(f"{blockmodel.name}.reblocked.pbm")
    reblocked_df = dict_3d_to_tabular(arrays, categories)

    # assert the indexes are aligned in ijk space
    if not reblocked_df.index.equals(new_geometry.to_multi_index_ijk()):
        raise ValueError("Reblocking ijk index mismatch.")

    # ------------------------------------------------------------------
    # Reconstruct geometry-derived columns for the new grid
    # ------------------------------------------------------------------
    # x, y, z centroids in C-order matching the dense ijk index
    reblocked_df["x"] = new_geometry.centroid_x
    reblocked_df["y"] = new_geometry.centroid_y
    reblocked_df["z"] = new_geometry.centroid_z

    # Linear index helpers following the same conventions as the demo
    # block model utilities (C-order index_c, F-order index_f).
    N = int(np.prod(new_geometry.shape))
    rows = np.arange(N)
    reblocked_df["block_id"] = rows
    reblocked_df["index_c"] = rows
    reblocked_df["index_f"] = rows.reshape(new_geometry.shape, order="C").ravel(order="F")

    table = pa.Table.from_pandas(reblocked_df.reset_index(drop=True), preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"parq-blockmodel"] = json.dumps(new_geometry.to_metadata_dict()).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, new_path)
    return ParquetBlockModel(blockmodel_path=new_path, geometry=new_geometry)




def upsample_blockmodel(blockmodel, new_block_size, interpolation_config) -> "ParquetBlockModel":
    """
    Upsample a block model to a finer grid with specified interpolation methods for each attribute.
    This function supports upsampling of both categorical and numeric attributes.

    Args:
        blockmodel: ParquetBlockModel instance.
        new_block_size: tuple of floats (dx, dy, dz) for the new block size.
        interpolation_config: dict mapping attribute names to interpolation methods.

    Example:
        interpolation_config = {
            'grade': 'linear',
            'density': 'linear',
            'dry_mass': 'linear',
            'volume': 'linear',
            'rock_type': 'nearest'
        }

    Returns:
        ParquetBlockModel: A new ParquetBlockModel instance with the upsampled grid.
    """

    from parq_blockmodel import ParquetBlockModel

    _validate_params(interpolation_config, new_block_size)

    arrays, categories = _prepare_arrays_and_index(blockmodel)

    fx, fy, fz, new_shape = _calculate_factors(blockmodel, new_block_size)

    if fx > 1 or fy > 1 or fz > 1:
        raise ValueError("reblocking factors imply downsampling, not upsampling.")
    elif fx < 1 or fy < 1 or fz < 1:
        # Upsampling: expect dict of strings
        if not all(isinstance(v, str) for v in (interpolation_config or {}).values()):
            raise ValueError("Upsampling config must be a dict of interpolation methods per attribute.")
        arrays = upsample_attributes(arrays, int(1 // fx), int(1 // fy), int(1 // fz), interpolation_config or {})
    elif any(dim < 1 for dim in (fx, fy, fz)) and any(dim > 1 for dim in (fx, fy, fz)):
        raise ValueError("Reblocking factors cannot specify upsample and downsample at the same time.")

    # Convert back to DataFrame (ijk-indexed attributes only)
    new_geometry = _build_new_geometry(blockmodel, new_block_size, new_shape)
    # Use canonical .pbm suffix expected by ParquetBlockModel
    new_path = blockmodel.blockmodel_path.with_name(f"{blockmodel.name}.reblocked.pbm")
    reblocked_df = dict_3d_to_tabular(arrays, categories)

    # assert the indexes are aligned in ijk space
    if not reblocked_df.index.equals(new_geometry.to_multi_index_ijk()):
        raise ValueError("Upsampling ijk index mismatch.")

    # ------------------------------------------------------------------
    # Reconstruct geometry-derived columns for the new grid
    # ------------------------------------------------------------------
    reblocked_df["x"] = new_geometry.centroid_x
    reblocked_df["y"] = new_geometry.centroid_y
    reblocked_df["z"] = new_geometry.centroid_z

    N = int(np.prod(new_geometry.shape))
    rows = np.arange(N)
    reblocked_df["block_id"] = rows
    reblocked_df["index_c"] = rows
    reblocked_df["index_f"] = rows.reshape(new_geometry.shape, order="C").ravel(order="F")

    table = pa.Table.from_pandas(reblocked_df.reset_index(drop=True), preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"parq-blockmodel"] = json.dumps(new_geometry.to_metadata_dict()).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, new_path)
    return ParquetBlockModel(blockmodel_path=new_path, geometry=new_geometry)
