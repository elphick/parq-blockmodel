from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, List, Sequence, Optional

from scipy.optimize import fsolve


def simulate_depletion_or_drawdown(
        variable_name: str,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        grid_size: int,
        timestamps: Sequence[pd.Timestamp],
        a: float,
        b: float,
        max_depth: float,
        center: Tuple[float, float] = (0.0, 0.0),
        delay_edges: bool = False,
        netcdf_filename: Optional[Path] = None
) -> Tuple[xr.DataArray, List[float]]:
    """
    Simulates 3D depletion or drawdown over time using an elliptical or circular footprint.

    Args:
        variable_name: Name of the variable to simulate (e.g., 'elevation' or 'water_level').
        x_range: Tuple defining the min and max x-coordinates of the grid.
        y_range: Tuple defining the min and max y-coordinates of the grid.
        grid_size: Resolution of the grid in both x and y directions.
        timestamps: A list or array of pandas Timestamps for each time step.
        a: Semi-major axis of the elliptical footprint.
        b: Semi-minor axis of the elliptical footprint.
        max_depth: Maximum depth or drawdown at the center.
        center: (x0, y0) coordinates of the pit or drawdown center.
        delay_edges: If True, applies delayed onset at the edges (for drawdown cones).
        netcdf_filename: Optional filepath of the NetCDF file to export the results.

    Returns:
        A tuple containing:
            - xarray.DataArray of shape (time_steps, y, x) with depth/drawdown values.
            - List of total volume extracted at each time step.
    """
    x0, y0 = center
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)

    # Normalized radial distance from center
    r = ((xx - x0) ** 2 / a ** 2) + ((yy - y0) ** 2 / b ** 2)

    # Initialize DataArray and volume list
    depth_array = np.zeros((len(timestamps), grid_size, grid_size))
    volumes = []

    # Time-dependent depth function (slowing sink rate)
    def depth_at_t(t):
        return max_depth * (1 - np.exp(-3 * t / len(timestamps))) / (1 - np.exp(-3))

    for t in range(len(timestamps)):
        d_t = depth_at_t(t + 1)
        if delay_edges:
            delay_factor = np.clip(r - 0.5, 0, 1)
            depth = -d_t * (1 - r) * (1 - delay_factor)
        else:
            depth = -d_t * (1 - r)
        depth[r > 1] = 0  # Outside the ellipse
        depth_array[t] = depth
        volumes.append(np.sum(-depth))  # Volume is positive

    # Create xarray DataArray
    da = xr.DataArray(
        depth_array,
        coords={"time": timestamps, "y": y, "x": x},
        dims=["time", "y", "x"],
        name=variable_name
    )

    # Export to NetCDF
    if netcdf_filename:
        netcdf_filename.parent.mkdir(parents=True, exist_ok=True)
        da.to_netcdf(netcdf_filename)

    return da, volumes


def build_waste_dump_time_series(
        volumes: Sequence[float],
        timestamps: Sequence[pd.Timestamp],
        center: tuple[float, float],
        cell_size: float = 1.0,
        angle_of_repose: float = 38):
    """
    Generate a time-aware xarray DataArray representing a sequence of waste dumps.

    Parameters:
    - volumes: List or array of volumes (m³) for each time step
    - timestamps: List or array of pandas Timestamps for each time step
    - center: Tuple (x, y) coordinates of the dump center
    - cell_size: Grid resolution in meters
    - angle_of_repose: Slope angle in degrees (default 38°)

    Returns:
    - xarray.DataArray with dimensions ('time', 'y', 'x')
    """

    if not len(volumes) == len(timestamps):
        raise ValueError("Volumes and times must have the same length")

    theta = np.radians(angle_of_repose)
    dump_layers = []

    max_radius = 0
    max_height = 0

    # First pass to determine max extent
    for volume in volumes:
        def volume_equation(r_flat):
            h = r_flat / 2
            r_base = r_flat + h / np.tan(theta)
            v_cyl = np.pi * r_flat ** 2 * h
            v_slope = (1 / 3) * np.pi * (r_base ** 2 - r_flat ** 2) * h
            return v_cyl + v_slope - volume

        r_flat_guess = (volume / np.pi) ** (1 / 3)
        r_flat = fsolve(volume_equation, r_flat_guess)[0]
        h = r_flat / 2
        r_base = r_flat + h / np.tan(theta)

        max_radius = max(max_radius, r_base)
        max_height = max(max_height, h)

    grid_radius = int(np.ceil(max_radius / cell_size))
    grid_size = 2 * grid_radius + 1
    x = np.linspace(center[0] - grid_radius * cell_size, center[0] + grid_radius * cell_size, grid_size)
    y = np.linspace(center[1] - grid_radius * cell_size, center[1] + grid_radius * cell_size, grid_size)
    x_grid, y_grid = np.meshgrid(x, y)
    r = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)

    # Second pass to build each time slice
    for volume in volumes:
        def volume_equation(r_flat):
            h = r_flat / 2
            r_base = r_flat + h / np.tan(theta)
            v_cyl = np.pi * r_flat ** 2 * h
            v_slope = (1 / 3) * np.pi * (r_base ** 2 - r_flat ** 2) * h
            return v_cyl + v_slope - volume

        r_flat_guess = (volume / np.pi) ** (1 / 3)
        r_flat = fsolve(volume_equation, r_flat_guess)[0]
        h = r_flat / 2
        r_base = r_flat + h / np.tan(theta)

        z = np.zeros_like(r)
        z[r <= r_flat] = h
        mask_slope = (r > r_flat) & (r <= r_base)
        z[mask_slope] = h - (r[mask_slope] - r_flat) * np.tan(theta)

        dump_layers.append(z)

    # Stack into xarray DataArray
    dump_array = xr.DataArray(
        np.stack(dump_layers),
        coords={
            'time': timestamps,
            'y': y,
            'x': x
        },
        dims=['time', 'y', 'x'],
        attrs={
            'description': 'Time-aware waste dump elevation surfaces',
            'center': center,
            'cell_size': cell_size,
            'angle_of_repose_degrees': angle_of_repose
        }
    )

    return dump_array


def sample_point_values(ds: xr.Dataset, variable: str, num: int) -> pd.DataFrame:
    """Randomly sample from the grid to create a point dataset"""
    return ds[variable].to_pandas().sample(num)


if __name__ == '__main__':
    # Example usage
    ts = list(pd.date_range(start="2024-01-01", end="2025-01-01", freq="MS"))
    data_array, volume_per_step = simulate_depletion_or_drawdown(
        variable_name="elevation",
        x_range=(-100, 100),
        y_range=(-100, 100),
        timestamps=ts,
        grid_size=100,
        a=50,
        b=30,
        max_depth=100,
        center=(0, 0),
        delay_edges=True,
        netcdf_filename=Path("./drawdown_simulation.nc")
    )

    print("Simulation complete. NetCDF file saved as 'drawdown_simulation.nc'.")
    print("Volume extracted per time step:", volume_per_step)

    dump_array = build_waste_dump_time_series(
        volumes=np.array(volume_per_step) * 0.8,  # Assume 80% of extracted volume goes to waste dump
        timestamps=ts,
        center=(0, 0),
        cell_size=2.0,
        angle_of_repose=38
    )
    print("Waste dump time series DataArray created.")

    # sample from elevation grid
    samples: pd.DataFrame = sample_point_values(data_array.to_dataset(), variable='elevation', num=40)
    print(samples.shape)
