"""Tests for parq_blockmodel.utils.temporal.demo_data"""
import numpy as np
import pandas as pd
import pytest

from parq_blockmodel.utils.temporal.demo_data import (
    simulate_depletion_or_drawdown,
    build_waste_dump_time_series,
    sample_point_values,
)


@pytest.fixture()
def timestamps():
    return list(pd.date_range("2024-01-01", periods=3, freq="MS"))


def test_simulate_depletion_basic(timestamps):
    da, volumes = simulate_depletion_or_drawdown(
        variable_name="elevation",
        x_range=(-10, 10),
        y_range=(-10, 10),
        grid_size=5,
        timestamps=timestamps,
        a=5.0,
        b=5.0,
        max_depth=10.0,
        center=(0.0, 0.0),
    )
    assert da.dims == ("time", "y", "x")
    assert da.shape[0] == len(timestamps)
    assert len(volumes) == len(timestamps)
    # All volumes should be positive
    assert all(v >= 0 for v in volumes)


def test_simulate_depletion_delay_edges(timestamps):
    da, volumes = simulate_depletion_or_drawdown(
        variable_name="water_level",
        x_range=(-5, 5),
        y_range=(-5, 5),
        grid_size=5,
        timestamps=timestamps,
        a=3.0,
        b=3.0,
        max_depth=5.0,
        delay_edges=True,
    )
    assert da.name == "water_level"


def test_simulate_depletion_with_netcdf_export(tmp_path, timestamps):
    filepath = tmp_path / "test.nc"
    da, _ = simulate_depletion_or_drawdown(
        variable_name="elevation",
        x_range=(-5, 5),
        y_range=(-5, 5),
        grid_size=5,
        timestamps=timestamps,
        a=3.0,
        b=3.0,
        max_depth=5.0,
        netcdf_filename=filepath,
    )
    assert filepath.exists()


def test_build_waste_dump_time_series(timestamps):
    volumes = [100.0, 200.0, 300.0]
    da = build_waste_dump_time_series(
        volumes=volumes,
        timestamps=timestamps,
        center=(0.0, 0.0),
        cell_size=1.0,
        angle_of_repose=38,
    )
    assert da.dims == ("time", "y", "x")
    assert da.shape[0] == len(timestamps)


def test_build_waste_dump_mismatched_lengths_raises(timestamps):
    with pytest.raises(ValueError, match="same length"):
        build_waste_dump_time_series(
            volumes=[100.0, 200.0],
            timestamps=timestamps,  # 3 items, not 2
            center=(0.0, 0.0),
        )


def test_sample_point_values(timestamps):
    da, _ = simulate_depletion_or_drawdown(
        variable_name="elevation",
        x_range=(-5, 5),
        y_range=(-5, 5),
        grid_size=10,
        timestamps=timestamps,
        a=3.0,
        b=3.0,
        max_depth=5.0,
    )
    # sample_point_values calls .to_pandas() which requires ≤2D; select one time slice
    single_time = da.isel(time=0).to_dataset()
    result = sample_point_values(single_time, variable="elevation", num=3)
    assert hasattr(result, "shape")


