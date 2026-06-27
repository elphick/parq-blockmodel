from __future__ import annotations

from importlib import resources
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

import numpy as np
import pyvista as pv

from parq_blockmodel.visualization.blockmodel_plot import BlockModelPlotState, _plotter_add_mesh_kwargs, prepare_plot_state
from parq_blockmodel.visualization.asset_selector import HivePbmCatalog

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
        asset_catalog: Optional[HivePbmCatalog] = None,
        asset_catalog_root: Optional[str | Path] = None,
    ) -> None:
        self.blockmodel = blockmodel
        self.grid_type = grid_type
        self.frame = frame
        self._title_override = title is not None
        self.title = title or blockmodel.name
        self.show_edges = show_edges
        self._initial_scalar = scalar or self._default_scalar()
        self.asset_catalog = asset_catalog
        self.asset_catalog_root = (
            Path(asset_catalog_root).resolve() if asset_catalog_root is not None else None
        )
        self._asset_level_keys: list[str] = []
        self._asset_level_values: dict[str, str] = {}
        self._selected_asset_name = ""
        self._registered_asset_level_indices: set[int] = set()
        self._asset_name_handler_registered = False
        self._source_mode = "hive" if asset_catalog is not None else "file"
        self._source_path = str(self.asset_catalog_root or self.blockmodel.blockmodel_path)
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

    @classmethod
    def from_hive_directory(
        cls,
        root_path: str | Path,
        *,
        scalar: Optional[str] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
    ) -> "BlockModelTrameApp":
        from parq_blockmodel.blockmodel import ParquetBlockModel

        catalog = HivePbmCatalog.discover(root_path)
        first_asset = catalog.assets[0]
        blockmodel = ParquetBlockModel(blockmodel_path=first_asset.path)
        return cls(
            blockmodel,
            scalar=scalar,
            grid_type=grid_type,
            frame=frame,
            title=title,
            show_edges=show_edges,
            asset_catalog=catalog,
            asset_catalog_root=Path(root_path),
        )

    @classmethod
    def from_source_path(
        cls,
        source_path: str | Path,
        *,
        scalar: Optional[str] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
    ) -> "BlockModelTrameApp":
        path = Path(source_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() != ".pbm":
                raise ValueError(f"Selected file must be a .pbm file: {path}")
            return cls.from_path(
                path,
                scalar=scalar,
                grid_type=grid_type,
                frame=frame,
                title=title,
                show_edges=show_edges,
            )
        if path.is_dir():
            return cls.from_hive_directory(
                path,
                scalar=scalar,
                grid_type=grid_type,
                frame=frame,
                title=title,
                show_edges=show_edges,
            )
        raise ValueError(f"Selected path is not a file or directory: {path}")

    def _default_scalar(self) -> str:
        return self.blockmodel.available_attributes[0]

    def _resolve_initial_scalar(self, preferred_scalar: Optional[str] = None) -> str:
        candidate = preferred_scalar or self._initial_scalar
        if candidate in self.blockmodel.available_attributes:
            return candidate
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

    def load_blockmodel(self, blockmodel: "ParquetBlockModel", *, preferred_scalar: Optional[str] = None) -> None:
        self.blockmodel = blockmodel
        if not self._title_override:
            self.title = self.blockmodel.name
        scalar = self._resolve_initial_scalar(preferred_scalar)
        self._initial_scalar = scalar

        self._syncing_state = True
        try:
            self._load_plot_state(scalar)
            self._refresh_plot()
            if self._server is not None and self.threshold is not None and self.state is not None:
                self._server.state.attribute_options = self.blockmodel.available_attributes
                self._server.state.active_attribute = self.state.scalar
                self._server.state.threshold_min = self.threshold.minimum
                self._server.state.threshold_max = self.threshold.maximum
                self._server.state.threshold = self.threshold.value
                self._server.state.threshold_step = self.threshold.step
                self._server.state.threshold_display = self._format_threshold(self.threshold.value)
                self._server.state.filter_active = self.filter_enabled
                self._server.state.model_name = self.blockmodel.name
                self._server.state.model_path = str(self.blockmodel.blockmodel_path)
        finally:
            self._syncing_state = False

    def _asset_selection(self) -> dict[str, str]:
        return {
            key: value
            for key, value in self._asset_level_values.items()
            if value not in ("", None)
        }

    def _refresh_asset_selector_values(self) -> tuple[list[list[str]], list[str]]:
        if self.asset_catalog is None:
            return [], []

        level_options: list[list[str]] = []
        selection_prefix: dict[str, str] = {}
        for key in self._asset_level_keys:
            options = self.asset_catalog.level_options(key, selection_prefix)
            current_value = self._asset_level_values.get(key, "")
            if current_value not in options:
                current_value = options[0] if options else ""
            self._asset_level_values[key] = current_value
            if current_value:
                selection_prefix[key] = current_value
            level_options.append(options)

        name_options = self.asset_catalog.pbm_name_options(self._asset_selection())
        if self._selected_asset_name not in name_options:
            self._selected_asset_name = name_options[0] if name_options else ""
        return level_options, name_options

    def _sync_asset_selector_state(self) -> None:
        if self._server is None:
            return

        state = self._server.state
        state.asset_selector_enabled = self.asset_catalog is not None
        if self.asset_catalog is None:
            state.asset_name_options = []
            state.selected_asset_name = ""
            state.asset_selected_path = ""
            return
        level_options, name_options = self._refresh_asset_selector_values()
        self._syncing_state = True
        try:
            for idx, key in enumerate(self._asset_level_keys):
                setattr(state, f"asset_level_{idx}_options", level_options[idx])
                setattr(state, f"asset_level_{idx}_value", self._asset_level_values.get(key, ""))
            state.asset_name_options = name_options
            state.selected_asset_name = self._selected_asset_name
            try:
                asset = self.asset_catalog.select_asset(self._asset_selection(), self._selected_asset_name)
                state.asset_selected_path = str(asset.path)
            except LookupError:
                state.asset_selected_path = ""
        finally:
            self._syncing_state = False

    def _register_asset_selector_handlers(self) -> None:
        if self._server is None:
            return
        state, ctrl = self._server.state, self._server.controller
        for idx, _ in enumerate(self._asset_level_keys):
            if idx in self._registered_asset_level_indices:
                continue
            self._registered_asset_level_indices.add(idx)
            field_name = f"asset_level_{idx}_value"

            def _make_asset_level_handler(level_index: int, watched_field: str):
                @state.change(watched_field)
                def _asset_level_changed(**kwargs):
                    if self._syncing_state:
                        return
                    if level_index >= len(self._asset_level_keys):
                        return
                    level_key = self._asset_level_keys[level_index]
                    self._asset_level_values[level_key] = str(kwargs.get(watched_field) or "")
                    self._sync_asset_selector_state()
                    self._load_selected_asset()

                return _asset_level_changed

            setattr(ctrl, f"update_asset_level_{idx}", _make_asset_level_handler(idx, field_name))

        if not self._asset_name_handler_registered:
            @state.change("selected_asset_name")
            def _asset_name_changed(selected_asset_name=None, **_):
                if self._syncing_state:
                    return
                self._selected_asset_name = str(selected_asset_name or "")
                self._sync_asset_selector_state()
                self._load_selected_asset()

            ctrl.update_asset_name = _asset_name_changed
            self._asset_name_handler_registered = True

    def _set_asset_selection_from_current_model(self) -> None:
        if self.asset_catalog is None:
            return
        found = self.asset_catalog.find_by_path(self.blockmodel.blockmodel_path)
        if found is None:
            return
        level_map = found.level_map
        self._asset_level_values = {key: level_map.get(key, "") for key in self._asset_level_keys}
        self._selected_asset_name = found.name

    def _load_selected_asset(self) -> None:
        if self.asset_catalog is None or not self._selected_asset_name:
            return
        try:
            selected = self.asset_catalog.select_asset(self._asset_selection(), self._selected_asset_name)
        except LookupError:
            return
        if selected.path.resolve() == self.blockmodel.blockmodel_path.resolve():
            return
        from parq_blockmodel.blockmodel import ParquetBlockModel

        self.load_blockmodel(ParquetBlockModel(blockmodel_path=selected.path))
        self._set_asset_selection_from_current_model()
        self._sync_asset_selector_state()

    def load_source_path(self, source_path: str | Path) -> None:
        from parq_blockmodel.blockmodel import ParquetBlockModel

        path = Path(source_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            if path.suffix.lower() != ".pbm":
                raise ValueError(f"Selected file must be a .pbm file: {path}")
            self.asset_catalog = None
            self.asset_catalog_root = None
            self._asset_level_keys = []
            self._asset_level_values = {}
            self._selected_asset_name = ""
            self._source_mode = "file"
            self._source_path = str(path)
            self.load_blockmodel(ParquetBlockModel(blockmodel_path=path))
            if self._server is not None:
                self._server.state.source_mode = self._source_mode
                self._server.state.source_path_input = self._source_path
                self._server.state.source_status = f"Loaded PBM: {path.name}"
                self._sync_asset_selector_state()
            return

        if not path.is_dir():
            raise ValueError(f"Selected path is not a file or directory: {path}")

        catalog = HivePbmCatalog.discover(path)
        first_asset = catalog.assets[0]
        self.asset_catalog = catalog
        self.asset_catalog_root = path.resolve()
        self._asset_level_keys = list(catalog.level_keys)
        self._asset_level_values = {key: "" for key in self._asset_level_keys}
        self._selected_asset_name = first_asset.name
        self._source_mode = "hive"
        self._source_path = str(path.resolve())
        self.load_blockmodel(ParquetBlockModel(blockmodel_path=first_asset.path))
        self._set_asset_selection_from_current_model()
        if self._server is not None:
            self._server.state.source_mode = self._source_mode
            self._server.state.source_path_input = self._source_path
            self._server.state.source_status = f"Loaded hive directory: {self._source_path}"
            self._register_asset_selector_handlers()
            self._sync_asset_selector_state()

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
        self.load_blockmodel(self.blockmodel, preferred_scalar=self._initial_scalar)
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
        state.source_mode = self._source_mode
        state.source_path_input = self._source_path
        state.source_status = ""
        state.asset_selector_enabled = self.asset_catalog is not None
        if self.asset_catalog is not None:
            self._asset_level_keys = list(self.asset_catalog.level_keys)
            self._asset_level_values = {key: "" for key in self._asset_level_keys}
            self._set_asset_selection_from_current_model()
            self._register_asset_selector_handlers()
            self._sync_asset_selector_state()

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
        ctrl.update_asset_name = lambda **_: None
        if self.asset_catalog is not None:
            self._register_asset_selector_handlers()

        def _load_data_source(**_):
            if self._syncing_state:
                return
            try:
                self.load_source_path(state.source_path_input)
            except (FileNotFoundError, ValueError) as exc:
                state.source_status = str(exc)

        ctrl.load_data_source = _load_data_source

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
                with vuetify.VSheet(classes="pa-2 fill-height"):
                    with vuetify.VExpansionPanels(
                        v_model=("source_panel", [0]),
                        multiple=True,
                        focusable=True,
                        flat=True,
                        classes="mb-2",
                    ):
                        with vuetify.VExpansionPanel():
                            vuetify.VExpansionPanelHeader(
                                "Data source",
                                dense=True,
                                classes="py-1 px-2 text-subtitle-2",
                            )
                            with vuetify.VExpansionPanelContent(classes="pt-1 pb-2 px-2"):
                                vuetify.VTextField(
                                    v_model=("source_path_input", self._source_path),
                                    label="Server path (.pbm file or hive directory)",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
                                )
                                vuetify.VBtn(
                                    "Load",
                                    click=ctrl.load_data_source,
                                    block=True,
                                    color="primary",
                                    classes="mt-2",
                                )
                                vuetify.VChip(
                                    "{{ source_mode }}",
                                    label=True,
                                    small=True,
                                    outlined=True,
                                    classes="mt-2",
                                )
                                vuetify.VChip(
                                    "{{ source_status }}",
                                    label=True,
                                    small=True,
                                    outlined=True,
                                    classes="mt-2 text-caption text-truncate",
                                    style="max-width: 100%;",
                                )
                        with vuetify.VExpansionPanel(v_if=("asset_selector_enabled", False)):
                            vuetify.VExpansionPanelHeader(
                                "Asset selector",
                                dense=True,
                                classes="py-1 px-2 text-subtitle-2",
                            )
                            with vuetify.VExpansionPanelContent(classes="pt-1 pb-2 px-2"):
                                for idx, key in enumerate(self._asset_level_keys):
                                    vuetify.VSelect(
                                        v_model=(
                                            f"asset_level_{idx}_value",
                                            self._asset_level_values.get(key, ""),
                                        ),
                                        items=(f"asset_level_{idx}_options", []),
                                        label=key,
                                        dense=True,
                                        hide_details=True,
                                        outlined=True,
                                        classes="mt-2",
                                        change=getattr(ctrl, f"update_asset_level_{idx}"),
                                    )
                                vuetify.VSelect(
                                    v_model=("selected_asset_name", self._selected_asset_name),
                                    items=("asset_name_options", []),
                                    label="PBM name",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
                                    change=ctrl.update_asset_name,
                                )
                                vuetify.VChip(
                                    "{{ asset_selected_path }}",
                                    label=True,
                                    small=True,
                                    outlined=True,
                                    classes="text-caption text-truncate mt-2",
                                    style="max-width: 100%;",
                                )
                    with vuetify.VExpansionPanels(
                        v_model=("control_panel", 0),
                        multiple=False,
                        focusable=True,
                        flat=True,
                    ):
                        with vuetify.VExpansionPanel():
                            vuetify.VExpansionPanelHeader(
                                "Controls",
                                dense=True,
                                classes="py-1 px-2 text-subtitle-2",
                            )
                            with vuetify.VExpansionPanelContent(classes="pt-1 pb-2 px-2"):
                                vuetify.VSelect(
                                    v_model=("active_attribute", self.state.scalar),
                                    items=("attribute_options", self.blockmodel.available_attributes),
                                    label="Attribute",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
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
                                    classes="mt-2",
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
                                vuetify.VBtn(
                                    "Reset",
                                    click=ctrl.reset_filter,
                                    block=True,
                                    color="primary",
                                    classes="mt-2",
                                )
                        with vuetify.VExpansionPanel():
                            vuetify.VExpansionPanelHeader(
                                "View status",
                                dense=True,
                                classes="py-1 px-2 text-subtitle-2",
                            )
                            with vuetify.VExpansionPanelContent(classes="pt-1 pb-2 px-2"):
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
