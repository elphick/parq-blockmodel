import warnings

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


def _warn_dtype_promotion(original_dtype, operation: str, attribute: str | None, reason: str) -> None:
    attr_text = f" for attribute '{attribute}'" if attribute else ""
    warnings.warn(
        f"{operation}{attr_text}: keeping promoted dtype because {reason} (original dtype: {original_dtype}).",
        RuntimeWarning,
        stacklevel=3,
    )


def to_numeric(arr, categories=None, *, operation: str = "reblocking", attribute: str | None = None):
    """
    Convert array to numeric for interpolation.
    Returns numeric array and a restoration function.
    """
    if categories is not None:
        arr_numeric = np.asarray(arr, dtype=np.float64)

        def restore_fn(x):
            x = np.round(x).astype(int)
            return pd.Categorical.from_codes(
                np.where(np.isnan(x), -1, x), categories
            )

        return arr_numeric, restore_fn

    if pd.api.types.is_integer_dtype(arr):
        arr_numeric = np.asarray(arr, dtype=np.float64)
        is_extension_dtype = pd.api.types.is_extension_array_dtype(arr.dtype)
        target_numpy_dtype = np.dtype(getattr(arr.dtype, "numpy_dtype", arr.dtype))

        def restore_fn(x):
            values = np.asarray(x, dtype=np.float64)
            rounded = np.rint(values)
            finite_mask = np.isfinite(rounded)

            if finite_mask.any():
                info = np.iinfo(target_numpy_dtype)
                finite_values = rounded[finite_mask]
                if finite_values.min() < info.min or finite_values.max() > info.max:
                    _warn_dtype_promotion(
                        arr.dtype,
                        operation=operation,
                        attribute=attribute,
                        reason=(
                            "integer result exceeds representable range "
                            f"[{info.min}, {info.max}]"
                        ),
                    )
                    return values

            if is_extension_dtype:
                rounded_with_na = rounded.copy()
                rounded_with_na[~finite_mask] = np.nan
                return pd.array(rounded_with_na, dtype=arr.dtype)

            if not finite_mask.all():
                _warn_dtype_promotion(
                    arr.dtype,
                    operation=operation,
                    attribute=attribute,
                    reason="integer result contains NaN/Inf",
                )
                return values

            return rounded.astype(target_numpy_dtype, copy=False)

        return arr_numeric, restore_fn

    if pd.api.types.is_float_dtype(arr):
        arr_numeric = np.asarray(arr, dtype=np.float64)
        target_dtype = np.dtype(arr.dtype)

        def restore_fn(x):
            return np.asarray(x, dtype=target_dtype)

        return arr_numeric, restore_fn

    arr_numeric = np.asarray(arr, dtype=np.float64)

    def restore_fn(x):
        return x

    return arr_numeric, restore_fn
