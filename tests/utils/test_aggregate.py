"""Tests for parq_blockmodel.utils.pandas.aggregate"""
import numpy as np
import pandas as pd
import pytest
from pandas import CategoricalDtype

from parq_blockmodel.utils.pandas.aggregate import aggregate


def _make_df():
    """Create a small test DataFrame with numeric and categorical columns."""
    df = pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0],
        "weight": [1.0, 1.0, 1.0, 1.0],
        "category": pd.Categorical(["A", "B", "A", "A"]),
    })
    return df


def test_aggregate_weighted_mean_uniform_weights():
    df = _make_df()
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict)
    assert np.isclose(result["value"].iloc[0], 2.5)


def test_aggregate_weighted_mean_non_uniform():
    df = pd.DataFrame({
        "value": [10.0, 20.0],
        "weight": [1.0, 3.0],
        "category": pd.Categorical(["A", "A"]),
    })
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict)
    # (10*1 + 20*3) / (1+3) = 70/4 = 17.5
    assert np.isclose(result["value"].iloc[0], 17.5)


def test_aggregate_sum_for_non_agg_numeric_column():
    df = pd.DataFrame({
        "value": [1.0, 2.0, 3.0],
        "weight": [1.0, 1.0, 1.0],
        "category": pd.Categorical(["A", "B", "A"]),
    })
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict)
    # weight column is not in agg_dict keys (it IS a value), so weight sums: 1+1+1=3
    assert result["weight"].iloc[0] == 3.0


def test_aggregate_categorical_majority():
    df = _make_df()
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict, cat_treatment="majority")
    assert result["category"].iloc[0] == "A"


def test_aggregate_categorical_proportions_as_dict():
    df = _make_df()
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict, cat_treatment="proportions", proportions_as_columns=False)
    proportions = result["category"].iloc[0]
    assert isinstance(proportions, dict)
    assert proportions["A"] == pytest.approx(0.75)
    assert proportions["B"] == pytest.approx(0.25)


def test_aggregate_categorical_proportions_as_columns():
    df = _make_df()
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict, cat_treatment="proportions", proportions_as_columns=True)
    assert "category_A" in result.columns
    assert "category_B" in result.columns
    assert result["category_A"].iloc[0] == pytest.approx(0.75)
    assert result["category_B"].iloc[0] == pytest.approx(0.25)


def test_aggregate_multiple_columns_same_weight():
    df = pd.DataFrame({
        "v1": [1.0, 3.0],
        "v2": [2.0, 4.0],
        "weight": [1.0, 1.0],
        "category": pd.Categorical(["X", "X"]),
    })
    agg_dict = {"v1": "weight", "v2": "weight"}
    result = aggregate(df, agg_dict)
    assert np.isclose(result["v1"].iloc[0], 2.0)
    assert np.isclose(result["v2"].iloc[0], 3.0)


def test_aggregate_column_order_preserved():
    df = _make_df()
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict)
    assert list(result.columns) == list(df.columns)


def test_aggregate_proportions_column_order():
    df = pd.DataFrame({
        "value": [1.0, 2.0],
        "weight": [1.0, 1.0],
        "category": pd.Categorical(["A", "B"]),
    })
    agg_dict = {"value": "weight"}
    result = aggregate(df, agg_dict, cat_treatment="proportions", proportions_as_columns=True)
    # category column expanded; value and weight should still be present
    assert "value" in result.columns
    assert "weight" in result.columns

