from __future__ import annotations

from importlib import resources
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

import numpy as np
import pyvista as pv

from parq_blockmodel.visualization.blockmodel_plot import BlockModelPlotState, _plotter_add_mesh_kwargs, prepare_plot_state

if TYPE_CHECKING:
    from parq_blockmodel.blockmodel import ParquetBlockModel


@dataclass(slots=True)
class ThresholdRange:
    minimum: float
    maximum: float
    value: float
    step: float


class BlockModelTrameApp:
    """Read-only Trame-friendly block-model session.

    The app keeps a PBM model separate from the rendered scene state. It is
    intentionally backend-first so the same state can be reused by a future
    loaded-PBM view or a hive-directory browser.
    """

    def __init__(
        self,
        blockmodel: "ParquetBlockModel",
        *,
        scalar: Optional[str] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
    ) -> None:
        self.blockmodel = blockmodel
        self.grid_type = grid_type
        self.frame = frame
        self.title = title or blockmodel.name
        self.show_edges = show_edges
        self._initial_scalar = scalar or self._default_scalar()
        self.state: Optional[BlockModelPlotState] = None
        self.threshold: Optional[ThresholdRange] = None
        self.filter_enabled = True
        self.plotter = pv.Plotter(off_screen=True)
        self._remote_view = None
        self._server = None
        self._syncing_state = False
        self._drawer_default_open = True

    @classmethod
    def from_path(
        cls,
        blockmodel_path: str | Path,
        *,
        scalar: Optional[str] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
    ) -> "BlockModelTrameApp":
        from parq_blockmodel.blockmodel import ParquetBlockModel

        return cls(
            ParquetBlockModel(blockmodel_path=Path(blockmodel_path)),
            scalar=scalar,
            grid_type=grid_type,
            frame=frame,
            title=title,
            show_edges=show_edges,
        )

    def _default_scalar(self) -> str:
        return self.blockmodel.available_attributes[0]

    def _attribute_values(self, attribute: str) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Plot state has not been loaded yet.")
        values = np.asarray(self.state.mesh.cell_data[attribute], dtype=float)
        return values[np.isfinite(values)]

    def _build_threshold_range(self, attribute: str) -> ThresholdRange:
        values = self._attribute_values(attribute)
        if values.size == 0:
            return ThresholdRange(minimum=0.0, maximum=1.0, value=0.0, step=0.01)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        span = max(maximum - minimum, 1.0)
        return ThresholdRange(
            minimum=minimum,
            maximum=maximum,
            value=minimum,
            step=span / 200.0,
        )

    def _format_threshold(self, value: float) -> str:
        return f"{value:,.3f}".rstrip("0").rstrip(".")

    def _branding_logo_resource(self):
        return resources.files("parq_blockmodel.assets.branding").joinpath("parq-blockmodel.svg")

    def _branding_logo_data_url(self) -> str:
        with resources.as_file(self._branding_logo_resource()) as brand_logo_path:
            svg_text = Path(brand_logo_path).read_text(encoding="utf-8")
        return "data:image/svg+xml;charset=utf-8," + quote(svg_text)

    def _filtered_mesh(self) -> pv.DataSet:
        if self.state is None:
            raise RuntimeError("Plot state has not been loaded yet.")
        if self.state.scalar_is_categorical:
            return self.state.mesh
        if not self.filter_enabled:
            return self.state.mesh
        return self.state.mesh.threshold(
            self.threshold.value,
            scalars=self.state.scalar,
            preference="cell",
        )

    def _refresh_plot(self) -> None:
        if self.state is None:
            raise RuntimeError("Plot state has not been loaded yet.")
        self.plotter.clear()
        mesh = self._filtered_mesh()
        mesh_kwargs = _plotter_add_mesh_kwargs(self.state)
        mesh_kwargs["show_edges"] = self.show_edges
        self.plotter.add_mesh(mesh, name="blockmodel", **mesh_kwargs)
        self.plotter.title = self.title
        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        if self._remote_view is not None:
            self._remote_view.update()



    def _load_plot_state(self, scalar: str) -> None:
        self.state = prepare_plot_state(
            self.blockmodel,
            scalar=scalar,
            grid_type=self.grid_type,
            frame=self.frame,
            enable_picking=False,
            title=self.title,
        )
        self.threshold = self._build_threshold_range(self.state.scalar)
        self.filter_enabled = True

    def set_attribute(self, attribute: str) -> None:
        self._syncing_state = True
        try:
            self._load_plot_state(attribute)
            self._refresh_plot()
            if self._server is not None and self.threshold is not None:
                self._server.state.active_attribute = attribute
                self._server.state.threshold_min = self.threshold.minimum
                self._server.state.threshold_max = self.threshold.maximum
                self._server.state.threshold = self.threshold.value
                self._server.state.filter_active = True
        finally:
            self._syncing_state = False

    def set_threshold(self, value: float) -> None:
        if self.threshold is None:
            raise RuntimeError("Threshold range has not been initialised.")
        self.threshold.value = float(value)
        self.filter_enabled = True
        self._refresh_plot()
        if self._server is not None:
            self._server.state.threshold = self.threshold.value
            self._server.state.threshold_display = self._format_threshold(self.threshold.value)
            self._server.state.filter_active = True

    def reset_filter(self) -> None:
        if self.threshold is None:
            raise RuntimeError("Threshold range has not been initialised.")
        self.filter_enabled = False
        self.threshold.value = self.threshold.minimum
        self._refresh_plot()
        if self._server is not None:
            self._syncing_state = True
            try:
                self._server.state.threshold = self.threshold.value
                self._server.state.threshold_display = self._format_threshold(self.threshold.value)
                self._server.state.filter_active = False
            finally:
                self._syncing_state = False

    def launch(self, server_name: str = "parq-blockmodel-trame"):
        try:
            from trame.app import get_server
            from trame.ui.vuetify import SinglePageWithDrawerLayout
            from trame.widgets import vtk, vuetify
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Trame dependencies are required for the interactive app. "
                "Install the 'trame', 'trame-vtk', and 'trame-vuetify' packages."
            ) from exc

        server = get_server(server_name, client_type="vue2")
        self._server = server
        state, ctrl = server.state, server.controller
        self._load_plot_state(self._initial_scalar)
        self._refresh_plot()
        state.attribute_options = self.blockmodel.available_attributes
        state.active_attribute = self.state.scalar
        state.threshold_min = self.threshold.minimum
        state.threshold_max = self.threshold.maximum
        state.threshold = self.threshold.value
        state.threshold_display = self._format_threshold(self.threshold.value)
        state.threshold_step = self.threshold.step
        state.filter_active = self.filter_enabled
        state.control_panel = 0
        state.model_name = self.blockmodel.name
        state.model_path = str(self.blockmodel.blockmodel_path)

        @state.change("active_attribute")
        def _attribute_changed(active_attribute=None, **_):
            if self._syncing_state or active_attribute is None:
                return
            self.set_attribute(active_attribute)

        @state.change("threshold")
        def _threshold_changed(threshold=None, **_):
            if self._syncing_state or threshold is None:
                return
            self.set_threshold(threshold)

        @state.change("filter_active")
        def _filter_active_changed(filter_active=None, **_):
            if self._syncing_state or filter_active is None:
                return
            self.filter_enabled = bool(filter_active)

        ctrl.update_attribute = _attribute_changed
        ctrl.update_threshold = _threshold_changed
        ctrl.reset_filter = self.reset_filter

        brand_logo_src = self._branding_logo_data_url()

        with SinglePageWithDrawerLayout(server, show_drawer=self._drawer_default_open, width=340) as layout:
            layout.toolbar.color = "grey lighten-3"
            layout.toolbar.dense = True
            layout.toolbar.dark = False
            if hasattr(layout, "title") and hasattr(layout.title, "set_text"):
                layout.title.set_text("")
            with layout.toolbar:
                vuetify.VImg(src=brand_logo_src, max_width=50, contain=True, classes="mr-2")
                vuetify.VToolbarTitle("ParquetBlockModel Viewer")
                vuetify.VSpacer()
                vuetify.VChip(
                    "{{ model_name }}",
                    label=True,
                    small=True,
                    outlined=True,
                    classes="mr-2",
                )
            with layout.drawer:
                with vuetify.VSheet(classes="pa-4 fill-height"):
                    with vuetify.VCard(elevation=3, classes="mb-4"):
                        with vuetify.VCardText(classes="py-4"):
                            vuetify.VCardTitle(
                                self.blockmodel.name,
                                classes="text-subtitle-1 text-truncate",
                                style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;",
                            )
                            vuetify.VChip(
                                str(self.blockmodel.blockmodel_path),
                                label=True,
                                small=True,
                                outlined=True,
                                classes="text-caption text-truncate",
                                style="max-width: 100%;",
                                title=str(self.blockmodel.blockmodel_path),
                            )
                    with vuetify.VExpansionPanels(v_model=("control_panel", 0), multiple=False, focusable=True):
                        with vuetify.VExpansionPanel():
                            vuetify.VExpansionPanelHeader("Controls")
                            with vuetify.VExpansionPanelContent():
                                vuetify.VSelect(
                                    v_model=("active_attribute", self.state.scalar),
                                    items=("attribute_options", self.blockmodel.available_attributes),
                                    label="Attribute",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    change=ctrl.update_attribute,
                                )
                                vuetify.VSlider(
                                    v_model=("threshold", self.threshold.value),
                                    min=("threshold_min", self.threshold.minimum),
                                    max=("threshold_max", self.threshold.maximum),
                                    step=("threshold_step", self.threshold.step),
                                    label="Threshold",
                                    dense=True,
                                    hide_details=True,
                                    thumb_label=False,
                                    outlined=True,
                                    change=ctrl.update_threshold,
                                )
                                vuetify.VTextField(
                                    v_model=("threshold_display", self._format_threshold(self.threshold.value)),
                                    label="Current value",
                                    readonly=True,
                                    dense=True,
                                    outlined=True,
                                    hide_details=True,
                                    classes="mt-2",
                                )
                                vuetify.VBtn("Reset", click=ctrl.reset_filter, block=True, color="primary")
                        with vuetify.VExpansionPanel():
                            vuetify.VExpansionPanelHeader("View status")
                            with vuetify.VExpansionPanelContent():
                                vuetify.VChip("Read-on-demand", color="success", small=True, outlined=True)
                                vuetify.VSpacer()
                                vuetify.VCheckbox(
                                    v_model=("filter_active", self.filter_enabled),
                                    label="Filter active",
                                    readonly=True,
                                    dense=True,
                                    hide_details=True,
                                )
            with layout.content:
                layout.content.classes = "pa-0 ma-0"
                layout.content.style = "height: calc(100vh - 64px); overflow: hidden;"
                self._remote_view = vtk.VtkRemoteView(
                    self.plotter.ren_win,
                    interactive_ratio=1,
                    style="width: 100%; height: 100%; min-height: 600px;",
                )
                self._remote_view.update()

        state.ready()
        server.start(open_browser=True, show_connection_info=True)
        return server
