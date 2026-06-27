from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, TYPE_CHECKING

import numpy as np
import pandas as pd
import pyvista as pv

from parq_blockmodel.utils.pyvista.categorical_utils import load_mapping_dict

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


def _plotter_add_mesh_kwargs(state: BlockModelPlotState) -> dict[str, Any]:
    add_mesh_kwargs: dict[str, Any] = {
        "scalars": state.scalar,
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


def render_plotter(
    state: BlockModelPlotState,
    *,
    threshold: bool = True,
    show_edges: bool = True,
    show_axes: bool = True,
    enable_picking: bool = False,
) -> pv.Plotter:
    plotter = pv.Plotter()
    add_mesh_kwargs = _plotter_add_mesh_kwargs(state)
    add_mesh_kwargs["show_edges"] = show_edges

    if threshold:
        plotter.add_mesh_threshold(state.mesh, **add_mesh_kwargs)
    else:
        plotter.add_mesh(state.mesh, **add_mesh_kwargs)

    plotter.title = state.title
    if show_axes:
        plotter.add_axes()

    text_name = "cell_info_text"
    plotter.add_text("", position="upper_left", font_size=12, name=text_name)

    if enable_picking:
        plotter.enable_cell_picking(
            callback=_cell_info_callback(plotter, state, text_name),
            show_message=False,
            through=False,
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
        )
