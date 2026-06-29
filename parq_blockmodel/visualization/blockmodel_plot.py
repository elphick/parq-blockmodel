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
        z_up_lock: bool = False,
        z_up_hotkey: str = "z",
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


def _normalize_z_up_hotkey(hotkey: str) -> str:
    normalized = str(hotkey).strip().lower()
    if len(normalized) != 1 or not normalized.isalpha():
        raise ValueError("z_up_hotkey must be a single alphabetic character, e.g. 'z'.")
    return normalized


def _register_z_up_rotation_lock(plotter: pv.Plotter, hotkey: str = "z") -> None:
    interactor = getattr(plotter, "iren", None)
    if interactor is None or not hasattr(interactor, "add_observer"):
        return

    normalized_hotkey = _normalize_z_up_hotkey(hotkey)
    setattr(plotter, "_pbm_z_up_hotkey", normalized_hotkey)
    if getattr(plotter, "_pbm_z_up_callbacks_registered", False):
        return
    setattr(plotter, "_pbm_z_up_callbacks_registered", True)
    setattr(plotter, "_pbm_z_up_hotkey_down", False)
    setattr(plotter, "_pbm_z_up_turntable_active", False)
    setattr(plotter, "_pbm_z_up_original_style", None)
    setattr(plotter, "_pbm_z_up_original_style_name", "")

    def _extract_key(obj) -> str:
        key_sym_getter = getattr(obj, "GetKeySym", None)
        if callable(key_sym_getter):
            key_sym = key_sym_getter() or ""
            if key_sym:
                return key_sym.lower()
        key_code_getter = getattr(obj, "GetKeyCode", None)
        if callable(key_code_getter):
            key_code = key_code_getter() or ""
            if key_code:
                return key_code.lower()
        return ""

    def _set_camera_z_up():
        camera = getattr(plotter, "camera", None)
        if camera is None:
            return
        camera.SetViewUp(0, 0, 1)
        if hasattr(camera, "OrthogonalizeViewUp"):
            camera.OrthogonalizeViewUp()

    def _enable_turntable_mode():
        if getattr(plotter, "_pbm_z_up_turntable_active", False):
            return
        if not hasattr(interactor, "GetInteractorStyle") or not hasattr(interactor, "SetInteractorStyle"):
            return
        original_style = interactor.GetInteractorStyle()
        original_style_name = ""
        if original_style is not None and hasattr(original_style, "GetClassName"):
            original_style_name = str(original_style.GetClassName() or "")
        setattr(plotter, "_pbm_z_up_original_style", original_style)
        setattr(plotter, "_pbm_z_up_original_style_name", original_style_name)
        if hasattr(plotter, "enable_terrain_style"):
            plotter.enable_terrain_style()
        else:
            terrain_style = pv._vtk.vtkInteractorStyleTerrain()
            if hasattr(terrain_style, "SetDefaultRenderer") and hasattr(plotter, "renderer"):
                terrain_style.SetDefaultRenderer(plotter.renderer)
            interactor.SetInteractorStyle(terrain_style)
        setattr(plotter, "_pbm_z_up_turntable_active", True)
        _set_camera_z_up()
        if hasattr(plotter, "render"):
            plotter.render()

    def _disable_turntable_mode():
        if not getattr(plotter, "_pbm_z_up_turntable_active", False):
            return
        original_style_name = str(getattr(plotter, "_pbm_z_up_original_style_name", "")).lower()
        original_style = getattr(plotter, "_pbm_z_up_original_style", None)
        if "trackball" in original_style_name and hasattr(plotter, "enable_trackball_style"):
            plotter.enable_trackball_style()
        elif "joystick" in original_style_name and hasattr(plotter, "enable_joystick_style"):
            plotter.enable_joystick_style()
        elif "terrain" in original_style_name and hasattr(plotter, "enable_terrain_style"):
            plotter.enable_terrain_style()
        elif original_style is not None and hasattr(interactor, "SetInteractorStyle"):
            interactor.SetInteractorStyle(original_style)
        setattr(plotter, "_pbm_z_up_turntable_active", False)
        setattr(plotter, "_pbm_z_up_original_style", None)
        setattr(plotter, "_pbm_z_up_original_style_name", "")
        _set_camera_z_up()
        if hasattr(plotter, "render"):
            plotter.render()

    def _key_press_callback(obj, event):
        key = _extract_key(obj)
        if key == getattr(plotter, "_pbm_z_up_hotkey", "z"):
            setattr(plotter, "_pbm_z_up_hotkey_down", True)
            _enable_turntable_mode()

    def _key_release_callback(obj, event):
        key = _extract_key(obj)
        if key == getattr(plotter, "_pbm_z_up_hotkey", "z"):
            setattr(plotter, "_pbm_z_up_hotkey_down", False)
            _disable_turntable_mode()

    def _interaction_callback(obj, event):
        if not getattr(plotter, "_pbm_z_up_hotkey_down", False):
            return
        _set_camera_z_up()
        if hasattr(plotter, "render"):
            plotter.render()

    setattr(plotter, "_pbm_z_up_enable_turntable_mode", _enable_turntable_mode)
    setattr(plotter, "_pbm_z_up_disable_turntable_mode", _disable_turntable_mode)

    interactor.add_observer("KeyPressEvent", _key_press_callback)
    interactor.add_observer("CharEvent", _key_press_callback)
    interactor.add_observer("KeyReleaseEvent", _key_release_callback)
    interactor.add_observer("InteractionEvent", _interaction_callback)


def _set_z_up_hotkey_state(plotter: pv.Plotter, is_down: bool) -> None:
    if not getattr(plotter, "_pbm_z_up_callbacks_registered", False):
        return
    down = bool(is_down)
    setattr(plotter, "_pbm_z_up_hotkey_down", down)
    if down:
        enable_cb = getattr(plotter, "_pbm_z_up_enable_turntable_mode", None)
        if callable(enable_cb):
            enable_cb()
        return
    disable_cb = getattr(plotter, "_pbm_z_up_disable_turntable_mode", None)
    if callable(disable_cb):
        disable_cb()


def render_plotter(
    state: BlockModelPlotState,
    *,
    threshold: bool = True,
    show_edges: bool = True,
    show_axes: bool = True,
    enable_picking: bool = False,
    z_up_lock: bool = False,
    z_up_hotkey: str = "z",
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
        if hasattr(plotter, "add_axes"):
            plotter.add_axes()
        elif hasattr(plotter, "show_axes"):
            plotter.show_axes()

    text_name = "cell_info_text"
    plotter.add_text("", position="upper_left", font_size=12, name=text_name)

    if enable_picking:
        plotter.enable_cell_picking(
            callback=_cell_info_callback(plotter, state, text_name),
            show_message=False,
            through=False,
        )

    if z_up_lock:
        _register_z_up_rotation_lock(plotter, hotkey=z_up_hotkey)

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
            z_up_lock=z_up_lock,
            z_up_hotkey=z_up_hotkey,
        )


class TrameBlockModelPlotEngine:
    def __init__(
        self,
        *,
        launch_on_plot: bool = False,
        server_name: str = "parq-blockmodel-trame",
        port: Optional[int] = None,
        app_name: str = "ParquetBlockModel Viewer",
    ) -> None:
        self.launch_on_plot = launch_on_plot
        self.server_name = server_name
        self.port = port
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
    ) -> Any:
        del threshold, show_axes, enable_picking, picked_attributes
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
            return app.launch(server_name=self.server_name, port=self.port)
        return app
