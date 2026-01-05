from typing import Literal

import numpy as np
import pandas as pd
from pandas import CategoricalDtype


def aggregate(df: pd.DataFrame, agg_dict: dict, cat_treatment: Literal['majority', 'proportions'] = 'majority',
              proportions_as_columns: bool = False) -> pd.DataFrame:
    """
    Aggregate a DataFrame using a provided dictionary.

    Args:
        df: The DataFrame to aggregate.
        agg_dict: A dictionary where keys are the columns to be aggregated and values are the weight columns.
        cat_treatment: A string indicating how to treat categorical columns.
                       'majority' returns the majority category, 'proportions' returns the proportions of each category.
        proportions_as_columns: A boolean indicating whether to return category proportions as separate columns.

    Returns:
        pd.DataFrame: The aggregated DataFrame with columns in the same order as the incoming DataFrame.
    """
    result = {}
    weight_columns = set(agg_dict.values())

    for weight_col in weight_columns:
        # Get columns that share the same weight column
        cols_with_weight = [col for col, w_col in agg_dict.items() if w_col == weight_col]
        if cols_with_weight:
            weights = df[weight_col].values
            weighted_values = df[cols_with_weight].values * weights[:, np.newaxis]
            aggregated_values = np.sum(weighted_values, axis=0) / np.sum(weights)
            result.update({col: aggregated_values[i] for i, col in enumerate(cols_with_weight)})

    # Sum columns that are not in the agg_dict
    for col in df.columns:
        if col not in agg_dict:
            if isinstance(df[col].dtype, CategoricalDtype):
                if cat_treatment == 'majority':
                    result[col] = df[col].mode()[0]  # Get the majority category
                elif cat_treatment == 'proportions':
                    proportions = df[col].value_counts(normalize=True).to_dict()
                    if proportions_as_columns:
                        for cat, prop in proportions.items():
                            result[f"{col}_{cat}"] = prop
                    else:
                        result[col] = proportions
            else:
                result[col] = df[col].sum()

    # Create a DataFrame from the result dictionary
    aggregated_df = pd.DataFrame([result])

    # Manage the final column order
    if proportions_as_columns:
        # loop through the columns and add them, extending with cat classes
        final_columns = []
        for col in df.columns:
            if col in result:
                final_columns.append(col)
            elif isinstance(df[col].dtype, CategoricalDtype) and cat_treatment == 'proportions':
                for cat in df[col].cat.categories:
                    final_columns.append(f"{col}_{cat}")
        aggregated_df = aggregated_df[final_columns]
    else:
        # Ensure the columns are in the same order as the incoming DataFrame
        aggregated_df = aggregated_df[df.columns]

    return aggregated_df