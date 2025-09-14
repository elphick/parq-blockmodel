from typing import Optional

import numpy as np
import pandas as pd

from parq_blockmodel.utils.geometry_utils import dense_ijk_multiindex


def tabular_to_3d_dict(df: pd.DataFrame) -> tuple[
    dict[str, np.ndarray],
    dict[str, pd.Index]]:
    """
    Convert a DataFrame indexed by i, j, k into a dict of 3D arrays.
    Object/categorical columns are converted to codes for efficient storage/interpolation.

    Parameters:
    - df: DataFrame indexed by i, j, k

    Returns:
    - dict of 3D arrays (C-order)
    - dict of pd.Indexes (categories)
    """
    if df.index.names != ['i', 'j', 'k']:
        raise ValueError("DataFrame index must be a MultiIndex with levels ['i', 'j', 'k']")

    shape = tuple(len(df.index.levels[i]) for i in range(3))
    dense_ijk_index = dense_ijk_multiindex(shape)
    if not df.index.equals(dense_ijk_index):
        df = df.reindex(dense_ijk_index)
    arrays = {}
    categories = {}
    for col in df.columns:
        col_data = df[col]
        if pd.api.types.is_object_dtype(col_data) or isinstance(col_data.dtype, pd.CategoricalDtype):
            cat = pd.Categorical(col_data)
            arrays[col] = cat.codes.reshape(shape, order='C')
            categories[col] = cat.categories
        else:
            arrays[col] = np.asarray(col_data).reshape(shape, order='C')

    return arrays, categories


def dict_3d_to_tabular(
        arrays: dict[str, np.ndarray],
        categories: dict[str, pd.Index] = None
) -> pd.DataFrame:
    """
    Convert a dict of dense 3D arrays (C-order) to a DataFrame indexed by i, j, k.
    Optionally reconstruct categorical columns using provided categories.

    Parameters:
    - arrays: dict of 3D arrays (C-order)
    - categories: dict mapping column name to pd.Index of categories (optional)

    Returns:
    - DataFrame indexed by i, j, k
    """

    data = {}
    shape = None
    for col, arr in arrays.items():
        if not shape:
            shape = arr.shape
        if arr.shape != shape:
            raise ValueError(f"Shape mismatch for column {col}: {arr.shape} != {shape}")
        flat = arr.ravel(order='C')
        if categories and col in categories:
            data[col] = pd.Categorical.from_codes(
                np.asarray(flat, dtype=flat.dtype if np.issubdtype(flat.dtype, np.integer) else np.int64),
                categories[col]
            )
        else:
            data[col] = flat

    dense_ijk_index = dense_ijk_multiindex(shape)
    df = pd.DataFrame(data, index=dense_ijk_index)

    return df


def to_numeric(arr, categories=None):
    """
    Convert array to numeric for interpolation.
    Returns numeric array and a restoration function.
    """
    if categories is not None:
        arr_numeric = arr.astype(float)

        def restore_fn(x):
            x = np.round(x).astype(int)
            return pd.Categorical.from_codes(
                np.where(np.isnan(x), -1, x), categories
            )

        return arr_numeric, restore_fn

    if pd.api.types.is_integer_dtype(arr):
        arr_numeric = arr.astype(float)
        if pd.api.types.is_extension_array_dtype(arr):
            # Pandas nullable integer (Int64, Int32, etc.)
            def restore_fn(x):
                return pd.array(x, dtype=arr.dtype)
        elif pd.isna(arr).any():
            # Fallback for legacy nullable integer
            def restore_fn(x):
                return pd.array(x, dtype="Int64")
        else:
            def restore_fn(x):
                return np.round(x).astype(arr.dtype)
        return arr_numeric, restore_fn

    arr_numeric = arr.astype(float)

    def restore_fn(x):
        return x

    return arr_numeric, restore_fn
