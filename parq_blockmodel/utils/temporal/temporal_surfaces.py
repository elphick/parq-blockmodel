from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.animation as animation

from parq_blockmodel.utils.temporal.demo_data import simulate_depletion_or_drawdown

try:
    import xarray as xr
except ImportError:
    xr = None


# if TYPE_CHECKING:
#     from xarray import Dataset

class TemporalSurfaceManager:
    def __init__(self, data: xr.Dataset):
        if xr is None:
            raise ImportError(
                "xarray is required for TemporalSurfaceManager. Please install it via 'pip install xarray'.")
        self.data = data
        self.timestamps = self.data.time.values
        self.bounds = (self.data.x.min().item(), self.data.x.max().item(), self.data.y.min().item(),
                       self.data.y.max().item())
        if 'elevation' in self.data:
            self.data['original_elevation'] = self.data['elevation'].isel(time=0)

    def plot_at_selected_time(
            self,
            variable: str,
            timestamp: pd.Timestamp):
        # Placeholder for actual surface computation logic
        surface: xr.DataArray = self.data[variable].sel(time=timestamp, method='nearest')
        surface.plot()

    def plot_at_location_over_time(
            self,
            variable: str,
            x: float,
            y: float):
        # Placeholder for actual time series extraction logic
        time_series: xr.DataArray = self.data[variable].sel(x=x, y=y, method='nearest')
        time_series.plot()

    def plot_difference_between_times(
            self,
            variable: str,
            time1: pd.Timestamp,
            time2: pd.Timestamp = None,
            direction: str = None
    ):
        """
        Plots the difference between two times for a given attribute.
        If time2 is None, direction must be 'previous' or 'next' to select adjacent timestamp.
        Subtraction order is managed so that result is always time2 - time1.
        """
        timestamps = list(self.timestamps)
        idx = timestamps.index(time1)
        if time2 is None:
            if direction == 'previous' and idx > 0:
                time2 = timestamps[idx - 1]
            elif direction == 'next' and idx < len(timestamps) - 1:
                time2 = timestamps[idx + 1]
            else:
                raise ValueError("Invalid direction or time1 is at the edge of timestamps.")
        surface1: xr.DataArray = self.data[variable].sel(time=time1, method='nearest')
        surface2: xr.DataArray = self.data[variable].sel(time=time2, method='nearest')
        difference = surface2 - surface1
        vmax = abs(difference).max().item()
        difference.plot(cmap="RdBu", vmin=-vmax, vmax=vmax)

    def plot_expression(
            self,
            expression: str,
            time: pd.Timestamp,
            timestamp_format: str = "%Y-%m-%d"):
        """
        Plots the result of a user-defined expression at a given time.
        Example expressions: 'elevation - water_level', 'elevation - elevation.sel(time=t)'
        """
        ds = self.data
        result: xr.Dataset = ds.sel(time=time, method='nearest')
        result = result.eval(expression)
        quadmesh = result.plot()
        quadmesh.axes.set_title(f'{expression} @ time: {pd.Timestamp(time).strftime(timestamp_format)}')

    def create_animation(
            self,
            variable: str,
            output_path: Path,
            fps: int = 2,
            timestamp_format: str = "%Y-%m-%d"):
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter

        fig, ax = plt.subplots()
        data = self.data[variable]
        vmin = float(data.min())
        vmax = float(data.max())

        # Initial plot to create colorbar
        im = data.sel(time=self.timestamps[0]).plot(ax=ax, vmin=vmin, vmax=vmax, add_colorbar=True)
        cbar = im.colorbar

        def update(frame):
            ax.clear()
            im = data.sel(time=self.timestamps[frame]).plot(ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False)
            ax.set_title(f'Time: {pd.Timestamp(self.timestamps[frame]).strftime(timestamp_format)}')            # Re-attach colorbar to the updated image
            if cbar:
                cbar.update_normal(im)

        ani = animation.FuncAnimation(fig, update, frames=len(self.timestamps), repeat=False)
        if output_path.suffix == '.gif':
            ani.save(output_path, writer=PillowWriter(fps=fps))
        elif output_path.suffix in ['.mp4', '.avi']:
            import imageio_ffmpeg
            plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
            ani.save(output_path, writer="ffmpeg", fps=fps)
        plt.close(fig)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ts = list(pd.date_range(start="2024-01-01", end="2025-01-01", freq="MS"))

    # create elevation timeseries
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
        delay_edges=False
    )
    # create water level timeseries
    # create elevation timeseries
    data_array_water, volume_per_step_water = simulate_depletion_or_drawdown(
        variable_name="water_level",
        x_range=(-100, 100),
        y_range=(-100, 100),
        timestamps=ts,
        grid_size=100,
        a=50,
        b=30,
        max_depth=150,
        center=(0, 0),
        delay_edges=True
    )
    # merge the two data arrays into a single dataset
    data_array = xr.merge([data_array, data_array_water])

    tsm = TemporalSurfaceManager(data_array)
    tsm.plot_at_selected_time('elevation', pd.Timestamp("2024-06-01"))
    plt.show()

    tsm.plot_at_location_over_time('elevation', x=0, y=0)
    plt.show()

    tsm.plot_difference_between_times('elevation', pd.Timestamp("2024-06-01"), direction='next')
    plt.show()

    tsm.plot_expression('original_elevation - elevation', time=pd.Timestamp("2024-06-01"))
    plt.show()

    tsm.plot_expression('elevation - water_level', time=pd.Timestamp("2024-06-01"))
    plt.show()

    tsm.create_animation('elevation', Path('./elevation_animation.gif'), fps=2)
    print("Animation saved to elevation_animation.gif")

    tsm.create_animation('elevation', Path('./elevation_animation.mp4'), fps=2)
    print("Animation saved to elevation_animation.mp4")
