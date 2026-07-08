from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, TYPE_CHECKING

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import Texture

from parq_blockmodel.utils.pyvista.categorical_utils import load_mapping_dict
from parq_blockmodel.utils.pyvista.custom_plotter import CustomPlotter

if TYPE_CHECKING:
    from parq_blockmodel.blockmodel import ParquetBlockModel


@dataclass(slots=True)
class BlockModelPlotState:
    mesh: pv.DataSet
    scalar: str
    attributes: list[str]
    categorical_mappings: dict[str, dict[int, str]]
    scalar_is_categorical: bool
    title: str


class BlockModelPlotEngine(Protocol):
    def plot(
        self,
        blockmodel: "ParquetBlockModel",
        *,
        scalar: str,
        grid_type: str = "image",
        frame: str = "world",
        threshold: bool = True,
        show_edges: bool = True,
        show_axes: bool = True,
        enable_picking: bool = False,
        picked_attributes: Optional[list[str]] = None,
        z_up_hotkey: str = "z",
        elevation_raster: Optional[str | Path] = None,
        imagery_raster: Optional[str | Path] = None,
    ) -> Any:
        ...


def _resolve_attributes(
    blockmodel: "ParquetBlockModel",
    scalar: str,
    enable_picking: bool,
    picked_attributes: Optional[list[str]],
) -> list[str]:
    if enable_picking:
        attributes = list(blockmodel.available_attributes if picked_attributes is None else picked_attributes)
    else:
        attributes = [scalar]
    if scalar not in attributes:
        attributes.append(scalar)
    return attributes


def _build_categorical_mappings(mesh: pv.DataSet, attributes: list[str]) -> dict[str, dict[int, str]]:
    mappings: dict[str, dict[int, str]] = {}
    for attr in attributes:
        if f"{attr}_json" not in mesh.field_data:
            continue
        mapping = load_mapping_dict(mesh, attr)
        mappings[attr] = {int(float(key)): str(value) for key, value in mapping.items()}
    return mappings


def prepare_plot_state(
    blockmodel: "ParquetBlockModel",
    *,
    scalar: str,
    grid_type: str = "image",
    frame: str = "world",
    enable_picking: bool = False,
    picked_attributes: Optional[list[str]] = None,
    title: Optional[str] = None,
) -> BlockModelPlotState:
    if scalar not in blockmodel.available_columns:
        raise ValueError(
            f"Column '{scalar}' not found in the ParquetBlockModel."
            f"Available columns are: {blockmodel.available_columns}"
        )

    attributes = _resolve_attributes(blockmodel, scalar, enable_picking, picked_attributes)
    mesh = blockmodel.to_pyvista(grid_type=grid_type, attributes=attributes, frame=frame)
    scalar_is_categorical = f"{scalar}_json" in mesh.field_data
    categorical_mappings = _build_categorical_mappings(mesh, attributes) if enable_picking else {}
    if scalar_is_categorical and scalar not in categorical_mappings:
        categorical_mappings.update(_build_categorical_mappings(mesh, [scalar]))

    return BlockModelPlotState(
        mesh=mesh,
        scalar=scalar,
        attributes=attributes,
        categorical_mappings=categorical_mappings,
        scalar_is_categorical=scalar_is_categorical,
        title=title or blockmodel.name,
    )


def _calculate_deciles(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([], dtype=float)
    return np.percentile(finite, np.arange(10, 100, 10))


def _bin_to_deciles(values: np.ndarray, decile_edges: np.ndarray) -> np.ndarray:
    binned = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return binned
    if decile_edges.size == 0:
        binned[finite_mask] = 0.0
        return binned
    finite_values = np.asarray(values[finite_mask], dtype=float)
    finite_bins = np.digitize(finite_values, decile_edges, right=False)
    finite_bins = np.clip(finite_bins, 0, 9)
    binned[finite_mask] = finite_bins.astype(float)
    return binned


def _colormap_to_lookup_table(colormap: str, n_bins: int = 10) -> pv.LookupTable:
    try:
        import matplotlib.pyplot as plt

        sampled = plt.get_cmap(colormap)(np.linspace(0.0, 1.0, n_bins))
        colors = np.asarray(np.clip(np.rint(sampled * 255.0), 0, 255), dtype=np.uint8)
    except ImportError:
        fallback = np.array(
            [
                [31, 119, 180, 255],
                [255, 127, 14, 255],
                [44, 160, 44, 255],
                [214, 39, 40, 255],
                [148, 103, 189, 255],
                [140, 86, 75, 255],
                [227, 119, 194, 255],
                [127, 127, 127, 255],
                [188, 189, 34, 255],
                [23, 190, 207, 255],
            ],
            dtype=np.uint8,
        )
        colors = np.vstack([fallback[i % len(fallback)] for i in range(n_bins)])
    return pv.LookupTable(values=colors, scalar_range=(-0.5, n_bins - 0.5), nan_color="lightgray")


def _plotter_add_mesh_kwargs(
    state: BlockModelPlotState,
    colormap: str = "viridis",
    discretize_to_deciles: bool = False,
    decile_edges: Optional[np.ndarray] = None,
    scalar_for_coloring: Optional[str] = None,
) -> dict[str, Any]:
    scalar_name = scalar_for_coloring or state.scalar
    add_mesh_kwargs: dict[str, Any] = {
        "scalars": scalar_name,
        "show_edges": True,
    }
    if state.scalar_is_categorical:
        scalar_mapping = state.categorical_mappings.get(state.scalar, {})
        annotations = {float(key): str(value) for key, value in scalar_mapping.items()}
        n_categories = max(len(scalar_mapping), 1)
        scalar_values = np.asarray(state.mesh.cell_data[state.scalar], dtype=float)
        has_nan = bool(np.isnan(scalar_values).any())
        scalar_bar_args = {"n_labels": 0}
        if has_nan:
            scalar_bar_args["nan_annotation"] = True
        base_palette = np.array(
            [
                [31, 119, 180, 255],
                [255, 127, 14, 255],
                [44, 160, 44, 255],
                [214, 39, 40, 255],
                [148, 103, 189, 255],
                [140, 86, 75, 255],
                [227, 119, 194, 255],
                [127, 127, 127, 255],
                [188, 189, 34, 255],
                [23, 190, 207, 255],
            ],
            dtype=np.uint8,
        )
        lut_values = np.vstack([base_palette[i % len(base_palette)] for i in range(n_categories)])
        lut = pv.LookupTable(
            values=lut_values,
            scalar_range=(-0.5, n_categories - 0.5),
            nan_color="lightgray",
        )
        lut.annotations = annotations
        add_mesh_kwargs.update(
            {
                "categories": True,
                "annotations": annotations,
                "cmap": lut,
                "nan_color": "lightgray",
                "nan_opacity": 0.15,
                "scalar_bar_args": scalar_bar_args,
            }
        )
    elif discretize_to_deciles:
        n_bins = 10
        lut = _colormap_to_lookup_table(colormap, n_bins=n_bins)
        if decile_edges is None:
            decile_edges = np.asarray([], dtype=float)
        decile_edges = np.asarray(decile_edges, dtype=float)
        annotations: dict[float, str] = {}
        if decile_edges.size == n_bins - 1:
            labels = [f"<= {decile_edges[0]:.3g}"]
            for i in range(1, n_bins - 1):
                labels.append(f"{decile_edges[i - 1]:.3g} - {decile_edges[i]:.3g}")
            labels.append(f"> {decile_edges[-1]:.3g}")
            annotations = {float(i): labels[i] for i in range(n_bins)}
            lut.annotations = annotations
        scalar_bar_args = {"n_labels": 0, "nan_annotation": True}
        add_mesh_kwargs.update(
            {
                "categories": True,
                "annotations": annotations,
                "cmap": lut,
                "nan_color": "lightgray",
                "nan_opacity": 0.15,
                "scalar_bar_args": scalar_bar_args,
            }
        )
    else:
        add_mesh_kwargs["cmap"] = colormap
    return add_mesh_kwargs


def _cell_info_callback(
    plotter: pv.Plotter,
    state: BlockModelPlotState,
    text_name: str,
):
    def cell_callback(picked_cell):
        if text_name in plotter.actors:
            plotter.remove_actor(text_name)
        if hasattr(picked_cell, "n_cells") and picked_cell.n_cells == 1:
            if "vtkOriginalCellIds" in picked_cell.cell_data:
                cell_id = int(picked_cell.cell_data["vtkOriginalCellIds"][0])
                cell_centers = state.mesh.cell_centers().points
                centroid = cell_centers[cell_id]
                centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                values: dict[str, Any] = {}
                for attr in state.attributes:
                    raw_value = state.mesh.cell_data[attr][cell_id]
                    if attr in state.categorical_mappings:
                        if pd.isna(raw_value):
                            values[attr] = "<NA>"
                        else:
                            code = int(np.rint(float(raw_value)))
                            values[attr] = state.categorical_mappings[attr].get(code, f"<unknown:{code}>")
                    else:
                        values[attr] = raw_value
                msg = f"Cell ID: {cell_id}, {centroid_str}, " + ", ".join(f"{k}: {v}" for k, v in values.items())
            else:
                value = picked_cell.cell_data[state.scalar][0]
                msg = f"Picked cell value: {state.scalar}: {value}"
            plotter.add_text(msg, position="upper_left", font_size=12, name=text_name)
        else:
            plotter.add_text("No valid cell picked.", position="upper_left", font_size=12, name=text_name)

    return cell_callback


def _build_elevation_surface(elevation_raster: str | Path) -> pv.StructuredGrid:
    from parq_blockmodel.surface import RasterSurface

    surface = RasterSurface.from_file(elevation_raster)
    x_coords = np.asarray(surface._x, dtype=float)
    y_coords = np.asarray(surface._y, dtype=float)
    elevation = np.asarray(surface._values, dtype=float)
    points_x, points_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    grid = pv.StructuredGrid(points_x, points_y, elevation)
    grid["elevation"] = elevation.ravel(order="C")
    return grid


def _add_elevation_overlay(
    plotter: pv.Plotter,
    *,
    elevation_raster: Optional[str | Path] = None,
    imagery_raster: Optional[str | Path] = None,
) -> None:
    if elevation_raster is None:
        return

    dem_surface = _build_elevation_surface(elevation_raster)
    mesh_kwargs: dict[str, Any] = {
        "show_edges": False,
        "name": "pbm_dem_surface",
    }
    if imagery_raster is not None:
        dem_surface.texture_map_to_plane(inplace=True, use_bounds=True)
        texture: Texture = pv.read_texture(str(imagery_raster))
        texture = texture.flip_y()
        mesh_kwargs["texture"] = texture
    else:
        mesh_kwargs.update(
            {
                "scalars": "elevation",
                "cmap": "terrain",
                "show_scalar_bar": False,
            }
        )

    actor = plotter.add_mesh(dem_surface, **mesh_kwargs)

    if hasattr(plotter, "add_checkbox_button_widget"):
        def _toggle_dem_surface(visible: bool) -> None:
            actor.SetVisibility(bool(visible))
            if hasattr(plotter, "render"):
                plotter.render()

        plotter.add_checkbox_button_widget(
            _toggle_dem_surface,
            value=True,
            position=(10, 10),
            size=28,
            color_on="forestgreen",
            color_off="gray",
            border_size=2,
        )
        if hasattr(plotter, "add_text"):
            plotter.add_text("DEM",
                             position=(45, 18),
                             font_size=10,
                             name="pbm_dem_toggle_label")

        if hasattr(plotter, "add_slider_widget"):
            def _set_dem_opacity(value: float) -> None:
                actor.GetProperty().SetOpacity(value)
                if hasattr(plotter, "render"):
                    plotter.render()

            slider = plotter.add_slider_widget(
                _set_dem_opacity,
                rng=[0.0, 1.0],
                value=0.8,
                title=None,
                pointa=(0.04, 0.02),
                pointb=(0.2, 0.02),
                slider_width=0.01,
                tube_width=0.003,
                style="modern"
            )

            rep = slider.GetRepresentation()
            rep.SetShowSliderLabel(False)


def render_plotter(
    state: BlockModelPlotState,
    *,
    threshold: bool = True,
    show_edges: bool = True,
    show_axes: bool = True,
    enable_picking: bool = False,
    elevation_raster: Optional[str | Path] = None,
    imagery_raster: Optional[str | Path] = None,
) -> pv.Plotter:
    plotter = CustomPlotter()
    add_mesh_kwargs = _plotter_add_mesh_kwargs(state)
    add_mesh_kwargs["show_edges"] = show_edges

    if threshold:
        plotter.add_mesh_threshold(state.mesh, **add_mesh_kwargs)
    else:
        plotter.add_mesh(state.mesh, **add_mesh_kwargs)

    plotter.title = state.title
    if show_axes:
        plotter.add_axes()

    plotter.set_directional_view(direction='SSE', elevation_deg=30)

    if enable_picking:
        callback = _cell_info_callback(plotter, state, "cell_info_text")
        plotter.disable_picking()
        plotter.enable_cell_picking(callback=callback, show_message=False, through=False)

    _add_elevation_overlay(
        plotter,
        elevation_raster=elevation_raster,
        imagery_raster=imagery_raster,
    )

    return plotter


class PyVistaBlockModelPlotEngine:
    def plot(
        self,
        blockmodel: "ParquetBlockModel",
        *,
        scalar: str,
        grid_type: str = "image",
        frame: str = "world",
        threshold: bool = True,
        show_edges: bool = True,
        show_axes: bool = True,
        enable_picking: bool = False,
        picked_attributes: Optional[list[str]] = None,
        z_up_lock: bool = False,
        z_up_hotkey: str = "z",
        elevation_raster: Optional[str | Path] = None,
        imagery_raster: Optional[str | Path] = None,
    ) -> pv.Plotter:
        state = prepare_plot_state(
            blockmodel,
            scalar=scalar,
            grid_type=grid_type,
            frame=frame,
            enable_picking=enable_picking,
            picked_attributes=picked_attributes,
        )
        return render_plotter(
            state,
            threshold=threshold,
            show_edges=show_edges,
            show_axes=show_axes,
            enable_picking=enable_picking,
            elevation_raster=elevation_raster,
            imagery_raster=imagery_raster,
        )


class TrameBlockModelPlotEngine:
    def __init__(
        self,
        *,
        launch_on_plot: bool = False,
        server_name: str = "parq-blockmodel-trame",
        port: Optional[int] = None,
        host: Optional[str] = None,
        app_name: str = "ParquetBlockModel Viewer",
    ) -> None:
        self.launch_on_plot = launch_on_plot
        self.server_name = server_name
        self.port = port
        self.host = host
        self.app_name = app_name

    def plot(
        self,
        blockmodel: "ParquetBlockModel",
        *,
        scalar: str,
        grid_type: str = "image",
        frame: str = "world",
        threshold: bool = True,
        show_edges: bool = True,
        show_axes: bool = True,
        enable_picking: bool = False,
        picked_attributes: Optional[list[str]] = None,
        z_up_lock: bool = False,
        z_up_hotkey: str = "z",
        elevation_raster: Optional[str | Path] = None,
        imagery_raster: Optional[str | Path] = None,
    ) -> Any:
        del threshold, show_axes, enable_picking, picked_attributes, elevation_raster, imagery_raster
        from parq_blockmodel.visualization.trame_app import BlockModelTrameApp

        app = BlockModelTrameApp(
            blockmodel,
            scalar=scalar,
            grid_type=grid_type,
            frame=frame,
            show_edges=show_edges,
            z_up_lock=z_up_lock,
            z_up_hotkey=z_up_hotkey,
            app_name=self.app_name,
        )
        if self.launch_on_plot:
            return app.launch(server_name=self.server_name, port=self.port, host=self.host)
        return app
