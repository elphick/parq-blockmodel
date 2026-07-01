from __future__ import annotations

from importlib import resources
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import pandas as pd
import pyvista as pv

from parq_blockmodel.utils.pyvista.categorical_utils import load_mapping_dict
from parq_blockmodel.visualization.blockmodel_plot import (
    BlockModelPlotState,
    _bin_to_deciles,
    _calculate_deciles,
    _normalize_z_up_hotkey,
    _plotter_add_mesh_kwargs,
    _register_z_up_rotation_lock,
    _set_z_up_hotkey_state,
    prepare_plot_state,
)
from parq_blockmodel.visualization.asset_selector import HivePbmCatalog

if TYPE_CHECKING:
    from parq_blockmodel.blockmodel import ParquetBlockModel


@dataclass(slots=True)
class ThresholdRange:
    minimum: float
    maximum: float
    value: float
    step: float


@dataclass(slots=True)
class DataFilterSlot:
    attribute: str = ""
    is_categorical: bool = False
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.005
    range_values: list[float] = None  # type: ignore[assignment]
    category_options: list[str] = None  # type: ignore[assignment]
    selected_categories: list[str] = None  # type: ignore[assignment]
    preset_min: Optional[float] = None
    preset_max: Optional[float] = None
    preset_categories: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.range_values is None:
            self.range_values = [0.0, 1.0]
        if self.category_options is None:
            self.category_options = []
        if self.selected_categories is None:
            self.selected_categories = []


class BlockModelTrameApp:
    """Read-only Trame-friendly block-model session.

    The app keeps a PBM model separate from the rendered scene state. It is
    intentionally backend-first so the same state can be reused by a future
    loaded-PBM view or a hive-directory browser.
    """

    def __init__(
        self,
        blockmodel: Optional["ParquetBlockModel"] = None,
        *,
        scalar: Optional[str] = None,
        # Backward-compatible alias for first data filter preset
        filter_attribute: Optional[str] = None,
        filter_min: Optional[float] = None,
        filter_max: Optional[float] = None,
        data_filter_1_attribute: Optional[str] = None,
        data_filter_1_min: Optional[float] = None,
        data_filter_1_max: Optional[float] = None,
        data_filter_1_categories: Optional[list[str]] = None,
        data_filter_2_attribute: Optional[str] = None,
        data_filter_2_min: Optional[float] = None,
        data_filter_2_max: Optional[float] = None,
        data_filter_2_categories: Optional[list[str]] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
        z_up_lock: bool = False,
        z_up_hotkey: str = "z",
        app_name: str = "ParquetBlockModel Viewer",
        asset_catalog: Optional[HivePbmCatalog] = None,
        asset_catalog_root: Optional[str | Path] = None,
    ) -> None:
        self.blockmodel = blockmodel
        self.grid_type = grid_type
        self.frame = frame
        self._title_override = title is not None
        self.title = title or (blockmodel.name if blockmodel else "")
        self.show_edges = show_edges
        self.z_up_lock = bool(z_up_lock)
        self.z_up_hotkey = _normalize_z_up_hotkey(z_up_hotkey)
        self.app_name = str(app_name)
        self._initial_scalar = scalar or (self._default_scalar() if blockmodel is not None else "")
        self.asset_catalog = asset_catalog
        self.asset_catalog_root = (
            Path(asset_catalog_root).resolve() if asset_catalog_root is not None else None
        )
        self._asset_level_keys: list[str] = []
        self._asset_level_values: dict[str, str] = {}
        self._selected_asset_name = ""
        self._registered_asset_level_indices: set[int] = set()
        self._asset_name_handler_registered = False
        self._asset_selectors_autofilled = False
        self._source_mode = "hive" if asset_catalog is not None else "file"
        self._source_path = str(self.asset_catalog_root or (self.blockmodel.blockmodel_path if blockmodel else ""))
        self._skip_initial_blockmodel_load = False
        self.state: Optional[BlockModelPlotState] = None
        self.threshold: Optional[ThresholdRange] = None
        self.filter_enabled = False
        self.colormap = "jet"
        self.discretize_deciles = False
        self.available_colormaps: list[str] = []
        self.plotter = pv.Plotter(off_screen=True)
        if self.z_up_lock:
            _register_z_up_rotation_lock(self.plotter, self.z_up_hotkey)
        self._remote_view = None
        self._server = None
        self._syncing_state = False
        self._drawer_default_open = True
        self._attribute_data_range: dict[str, tuple[float, float]] = {}
        self._filter_attribute_options: list[str] = []
        self._filter_attribute_cache: dict[str, np.ndarray] = {}
        self._filter_category_code_to_label: dict[str, dict[float, str]] = {}
        self._data_filters = [
            DataFilterSlot(
                attribute=str(data_filter_1_attribute or filter_attribute or ""),
                preset_min=(
                    float(data_filter_1_min)
                    if data_filter_1_min is not None
                    else (float(filter_min) if filter_min is not None else None)
                ),
                preset_max=(
                    float(data_filter_1_max)
                    if data_filter_1_max is not None
                    else (float(filter_max) if filter_max is not None else None)
                ),
                preset_categories=(
                    list(data_filter_1_categories)
                    if data_filter_1_categories is not None
                    else None
                ),
            ),
            DataFilterSlot(
                attribute=str(data_filter_2_attribute or ""),
                preset_min=float(data_filter_2_min) if data_filter_2_min is not None else None,
                preset_max=float(data_filter_2_max) if data_filter_2_max is not None else None,
                preset_categories=(
                    list(data_filter_2_categories)
                    if data_filter_2_categories is not None
                    else None
                ),
            ),
        ]
        self._view_initialized = False

    @classmethod
    def from_path(
        cls,
        blockmodel_path: str | Path,
        *,
        scalar: Optional[str] = None,
        filter_attribute: Optional[str] = None,
        filter_min: Optional[float] = None,
        filter_max: Optional[float] = None,
        data_filter_1_attribute: Optional[str] = None,
        data_filter_1_min: Optional[float] = None,
        data_filter_1_max: Optional[float] = None,
        data_filter_1_categories: Optional[list[str]] = None,
        data_filter_2_attribute: Optional[str] = None,
        data_filter_2_min: Optional[float] = None,
        data_filter_2_max: Optional[float] = None,
        data_filter_2_categories: Optional[list[str]] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
        z_up_lock: bool = False,
        z_up_hotkey: str = "z",
        app_name: str = "ParquetBlockModel Viewer",
    ) -> "BlockModelTrameApp":
        from parq_blockmodel.blockmodel import ParquetBlockModel

        return cls(
            ParquetBlockModel(blockmodel_path=Path(blockmodel_path)),
            scalar=scalar,
            filter_attribute=filter_attribute,
            filter_min=filter_min,
            filter_max=filter_max,
            data_filter_1_attribute=data_filter_1_attribute,
            data_filter_1_min=data_filter_1_min,
            data_filter_1_max=data_filter_1_max,
            data_filter_1_categories=data_filter_1_categories,
            data_filter_2_attribute=data_filter_2_attribute,
            data_filter_2_min=data_filter_2_min,
            data_filter_2_max=data_filter_2_max,
            data_filter_2_categories=data_filter_2_categories,
            grid_type=grid_type,
            frame=frame,
            title=title,
            show_edges=show_edges,
            z_up_lock=z_up_lock,
            z_up_hotkey=z_up_hotkey,
            app_name=app_name,
        )

    @classmethod
    def from_hive_directory(
        cls,
        root_path: str | Path,
        *,
        scalar: Optional[str] = None,
        filter_attribute: Optional[str] = None,
        filter_min: Optional[float] = None,
        filter_max: Optional[float] = None,
        data_filter_1_attribute: Optional[str] = None,
        data_filter_1_min: Optional[float] = None,
        data_filter_1_max: Optional[float] = None,
        data_filter_1_categories: Optional[list[str]] = None,
        data_filter_2_attribute: Optional[str] = None,
        data_filter_2_min: Optional[float] = None,
        data_filter_2_max: Optional[float] = None,
        data_filter_2_categories: Optional[list[str]] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
        z_up_lock: bool = False,
        z_up_hotkey: str = "z",
        app_name: str = "ParquetBlockModel Viewer",
    ) -> "BlockModelTrameApp":
        catalog = HivePbmCatalog.discover(root_path)
        # For hive mode, don't load any blockmodel initially - user selects via UI
        # Start with blockmodel=None to avoid unnecessary loading of first asset
        app = cls(
            blockmodel=None,
            scalar=scalar,
            filter_attribute=filter_attribute,
            filter_min=filter_min,
            filter_max=filter_max,
            data_filter_1_attribute=data_filter_1_attribute,
            data_filter_1_min=data_filter_1_min,
            data_filter_1_max=data_filter_1_max,
            data_filter_1_categories=data_filter_1_categories,
            data_filter_2_attribute=data_filter_2_attribute,
            data_filter_2_min=data_filter_2_min,
            data_filter_2_max=data_filter_2_max,
            data_filter_2_categories=data_filter_2_categories,
            grid_type=grid_type,
            frame=frame,
            title=title,
            show_edges=show_edges,
            z_up_lock=z_up_lock,
            z_up_hotkey=z_up_hotkey,
            app_name=app_name,
            asset_catalog=catalog,
            asset_catalog_root=Path(root_path),
        )
        # Mark that we should skip loading placeholder blockmodel on launch
        app._skip_initial_blockmodel_load = True
        return app

    @classmethod
    def from_source_path(
        cls,
        source_path: str | Path,
        *,
        scalar: Optional[str] = None,
        filter_attribute: Optional[str] = None,
        filter_min: Optional[float] = None,
        filter_max: Optional[float] = None,
        data_filter_1_attribute: Optional[str] = None,
        data_filter_1_min: Optional[float] = None,
        data_filter_1_max: Optional[float] = None,
        data_filter_1_categories: Optional[list[str]] = None,
        data_filter_2_attribute: Optional[str] = None,
        data_filter_2_min: Optional[float] = None,
        data_filter_2_max: Optional[float] = None,
        data_filter_2_categories: Optional[list[str]] = None,
        grid_type: str = "image",
        frame: str = "world",
        title: Optional[str] = None,
        show_edges: bool = True,
        z_up_lock: bool = False,
        z_up_hotkey: str = "z",
        app_name: str = "ParquetBlockModel Viewer",
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
                filter_attribute=filter_attribute,
                filter_min=filter_min,
                filter_max=filter_max,
                data_filter_1_attribute=data_filter_1_attribute,
                data_filter_1_min=data_filter_1_min,
                data_filter_1_max=data_filter_1_max,
                data_filter_1_categories=data_filter_1_categories,
                data_filter_2_attribute=data_filter_2_attribute,
                data_filter_2_min=data_filter_2_min,
                data_filter_2_max=data_filter_2_max,
                data_filter_2_categories=data_filter_2_categories,
                grid_type=grid_type,
                frame=frame,
                title=title,
                show_edges=show_edges,
                z_up_lock=z_up_lock,
                z_up_hotkey=z_up_hotkey,
                app_name=app_name,
            )
        if path.is_dir():
            return cls.from_hive_directory(
                path,
                scalar=scalar,
                filter_attribute=filter_attribute,
                filter_min=filter_min,
                filter_max=filter_max,
                data_filter_1_attribute=data_filter_1_attribute,
                data_filter_1_min=data_filter_1_min,
                data_filter_1_max=data_filter_1_max,
                data_filter_1_categories=data_filter_1_categories,
                data_filter_2_attribute=data_filter_2_attribute,
                data_filter_2_min=data_filter_2_min,
                data_filter_2_max=data_filter_2_max,
                data_filter_2_categories=data_filter_2_categories,
                grid_type=grid_type,
                frame=frame,
                title=title,
                show_edges=show_edges,
                z_up_lock=z_up_lock,
                z_up_hotkey=z_up_hotkey,
                app_name=app_name,
            )
        raise ValueError(f"Selected path is not a file or directory: {path}")

    def _default_scalar(self) -> str:
        if self.blockmodel is None:
            return ""
        return self.blockmodel.available_attributes[0]

    def _resolve_initial_scalar(self, preferred_scalar: Optional[str] = None) -> str:
        if self.blockmodel is None:
            return ""
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

    def _clear_filter_cache(self) -> None:
        self._filter_attribute_cache = {}
        self._filter_category_code_to_label = {}

    def _is_categorical_attribute(self, attribute: str) -> bool:
        if self.blockmodel is None:
            return False
        dtype = self.blockmodel.column_dtypes.get(attribute)
        if dtype is None:
            return False
        return isinstance(dtype, pd.CategoricalDtype)

    def _is_continuous_attribute(self, attribute: str) -> bool:
        if self.blockmodel is None:
            return False
        dtype = self.blockmodel.column_dtypes.get(attribute)
        if dtype is None:
            return False
        if self._is_categorical_attribute(attribute):
            return False
        try:
            return bool(np.issubdtype(np.dtype(dtype), np.number))
        except TypeError:
            return False

    def _all_filter_attribute_options(self) -> list[str]:
        if self.blockmodel is None:
            return []
        return list(self.blockmodel.available_attributes)

    def _load_filter_category_mapping(self, mesh: pv.DataSet, attribute: str) -> dict[float, str]:
        if f"{attribute}_json" not in mesh.field_data:
            return {}
        raw_mapping = load_mapping_dict(mesh, attribute)
        code_to_label: dict[float, str] = {}
        for raw_code, label in raw_mapping.items():
            code_to_label[float(raw_code)] = str(label)
        return code_to_label

    def _load_filter_attribute_values(self, attribute: str) -> np.ndarray:
        if self.blockmodel is None:
            raise RuntimeError("No blockmodel is loaded.")
        if self.state is not None and attribute in self.state.mesh.cell_data:
            self._filter_category_code_to_label[attribute] = self._load_filter_category_mapping(
                self.state.mesh, attribute
            )
            return np.asarray(self.state.mesh.cell_data[attribute], dtype=float)
        attribute_mesh = self.blockmodel.to_pyvista(
            grid_type=self.grid_type,
            attributes=[attribute],
            frame=self.frame,
        )
        self._filter_category_code_to_label[attribute] = self._load_filter_category_mapping(
            attribute_mesh, attribute
        )
        return np.asarray(attribute_mesh.cell_data[attribute], dtype=float)

    def _get_filter_attribute_values(self, attribute: str) -> np.ndarray:
        cached = self._filter_attribute_cache.get(attribute)
        if cached is not None:
            return cached
        loaded = self._load_filter_attribute_values(attribute)
        self._filter_attribute_cache[attribute] = loaded
        return loaded

    def _build_data_filter_range(self, attribute: str) -> tuple[float, float, float]:
        values = self._get_filter_attribute_values(attribute)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0, 1.0, 0.01
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        span = max(maximum - minimum, 1.0)
        return minimum, maximum, span / 200.0

    def _filter_slot_name(self, slot_index: int) -> str:
        return f"data_filter_{slot_index + 1}"

    def _get_filter_slot(self, slot_index: int) -> DataFilterSlot:
        if slot_index not in (0, 1):
            raise ValueError("data filter slot_index must be 0 or 1.")
        return self._data_filters[slot_index]

    def _reset_filter_slot(self, slot_index: int) -> None:
        slot = self._get_filter_slot(slot_index)
        slot.attribute = ""
        slot.is_categorical = False
        slot.minimum = 0.0
        slot.maximum = 1.0
        slot.step = 0.005
        slot.range_values = [0.0, 1.0]
        slot.category_options = []
        slot.selected_categories = []

    def _set_filter_slot_attribute(
        self,
        slot_index: int,
        attribute: str,
        *,
        selected_min: Optional[float] = None,
        selected_max: Optional[float] = None,
        selected_categories: Optional[list[str]] = None,
    ) -> None:
        slot = self._get_filter_slot(slot_index)
        normalized = str(attribute or "")
        if normalized == "":
            self._reset_filter_slot(slot_index)
            return
        if normalized not in self._filter_attribute_options:
            raise ValueError(f"Filter attribute '{normalized}' is not available.")

        slot.attribute = normalized
        slot.is_categorical = self._is_categorical_attribute(normalized)
        values = self._get_filter_attribute_values(normalized)

        if slot.is_categorical:
            code_to_label = self._filter_category_code_to_label.get(normalized, {})
            ordered_labels = [label for _, label in sorted(code_to_label.items(), key=lambda item: item[0])]
            if not ordered_labels:
                finite_values = values[np.isfinite(values)]
                unique_codes = sorted({float(v) for v in finite_values.tolist()})
                ordered_labels = [str(int(code) if float(code).is_integer() else code) for code in unique_codes]
                code_to_label = {
                    float(code): str(int(code) if float(code).is_integer() else code) for code in unique_codes
                }
                self._filter_category_code_to_label[normalized] = code_to_label
            slot.category_options = ordered_labels
            if selected_categories is None:
                chosen = list(slot.preset_categories or ordered_labels)
            else:
                chosen = list(selected_categories)
            slot.selected_categories = [label for label in chosen if label in slot.category_options]
            slot.minimum = 0.0
            slot.maximum = 1.0
            slot.step = 0.005
            slot.range_values = [0.0, 1.0]
        else:
            minimum, maximum, step = self._build_data_filter_range(normalized)
            low_source = selected_min if selected_min is not None else slot.preset_min
            high_source = selected_max if selected_max is not None else slot.preset_max
            low = minimum if low_source is None else max(minimum, min(maximum, float(low_source)))
            high = maximum if high_source is None else max(minimum, min(maximum, float(high_source)))
            if low > high:
                low, high = high, low
            slot.minimum = minimum
            slot.maximum = maximum
            slot.step = step
            slot.range_values = [low, high]
            slot.category_options = []
            slot.selected_categories = []

        slot.preset_min = None
        slot.preset_max = None
        slot.preset_categories = None

    def _apply_initial_data_filters(self) -> None:
        for idx, slot in enumerate(self._data_filters):
            if slot.attribute:
                self._set_filter_slot_attribute(
                    idx,
                    slot.attribute,
                    selected_min=slot.preset_min,
                    selected_max=slot.preset_max,
                    selected_categories=slot.preset_categories,
                )

    def _data_filter_slot_summary(self, slot: DataFilterSlot) -> str:
        if not slot.attribute:
            return "None"
        if slot.is_categorical:
            selected = slot.selected_categories
            total = len(slot.category_options)
            if not selected:
                return f"{slot.attribute}: 0/{total} selected"
            if len(selected) <= 3:
                return f"{slot.attribute}: {', '.join(selected)}"
            return f"{slot.attribute}: {len(selected)}/{total} selected"
        low, high = slot.range_values
        return f"{slot.attribute}: {self._format_threshold(low)} to {self._format_threshold(high)}"

    def _sync_data_filter_state(self) -> None:
        if self._server is None:
            return
        state = self._server.state
        state.data_filter_attribute_options = list(self._filter_attribute_options)
        for idx, slot in enumerate(self._data_filters):
            prefix = self._filter_slot_name(idx)
            setattr(state, f"{prefix}_attribute", slot.attribute)
            setattr(state, f"{prefix}_is_categorical", slot.is_categorical)
            setattr(state, f"{prefix}_range_min", slot.minimum)
            setattr(state, f"{prefix}_range_max", slot.maximum)
            setattr(state, f"{prefix}_range_step", slot.step)
            setattr(state, f"{prefix}_range", list(slot.range_values))
            setattr(state, f"{prefix}_category_options", list(slot.category_options))
            setattr(state, f"{prefix}_selected_categories", list(slot.selected_categories))
            setattr(state, f"{prefix}_summary", self._data_filter_slot_summary(slot))

    def set_data_filter_attribute(self, slot_index: int, attribute: str) -> None:
        self._set_filter_slot_attribute(slot_index, attribute)
        if self.state is not None:
            self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._syncing_state = True
            try:
                self._sync_data_filter_state()
            finally:
                self._syncing_state = False

    def set_data_filter_range(self, slot_index: int, values: list[float] | tuple[float, float]) -> None:
        slot = self._get_filter_slot(slot_index)
        if not slot.attribute or slot.is_categorical:
            return
        if len(values) != 2:
            raise ValueError("Filter range must have exactly two values.")
        lower = max(slot.minimum, min(slot.maximum, float(values[0])))
        upper = max(slot.minimum, min(slot.maximum, float(values[1])))
        if lower > upper:
            lower, upper = upper, lower
        if lower == slot.range_values[0] and upper == slot.range_values[1]:
            return
        slot.range_values = [lower, upper]
        if self.state is not None:
            self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._syncing_state = True
            try:
                prefix = self._filter_slot_name(slot_index)
                setattr(self._server.state, f"{prefix}_range", list(slot.range_values))
                setattr(self._server.state, f"{prefix}_summary", self._data_filter_slot_summary(slot))
            finally:
                self._syncing_state = False

    def set_data_filter_categories(self, slot_index: int, categories: list[str]) -> None:
        slot = self._get_filter_slot(slot_index)
        if not slot.attribute or not slot.is_categorical:
            return
        slot.selected_categories = [label for label in categories if label in slot.category_options]
        if self.state is not None:
            self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._syncing_state = True
            try:
                prefix = self._filter_slot_name(slot_index)
                setattr(
                    self._server.state,
                    f"{prefix}_selected_categories",
                    list(slot.selected_categories),
                )
                setattr(self._server.state, f"{prefix}_summary", self._data_filter_slot_summary(slot))
            finally:
                self._syncing_state = False

    def reset_data_filter(self, slot_index: Optional[int] = None) -> None:
        if slot_index is None:
            self._reset_filter_slot(0)
            self._reset_filter_slot(1)
        else:
            self._reset_filter_slot(slot_index)
        if self.state is not None:
            self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._syncing_state = True
            try:
                self._sync_data_filter_state()
            finally:
                self._syncing_state = False

    def _refresh_filter_options(self) -> None:
        self._filter_attribute_options = self._all_filter_attribute_options()
        for idx, slot in enumerate(self._data_filters):
            if slot.attribute and slot.attribute not in self._filter_attribute_options:
                self._reset_filter_slot(idx)

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
        cell_count = self.state.mesh.n_cells
        mask = np.ones(cell_count, dtype=bool)

        if not self.state.scalar_is_categorical and self.filter_enabled:
            threshold_values = np.asarray(self.state.mesh.cell_data[self.state.scalar], dtype=float)
            mask &= np.isfinite(threshold_values) & (threshold_values >= self.threshold.value)

        for slot in self._data_filters:
            if not slot.attribute:
                continue
            values = self._get_filter_attribute_values(slot.attribute)
            if values.shape[0] != cell_count:
                continue
            if slot.is_categorical:
                code_to_label = self._filter_category_code_to_label.get(slot.attribute, {})
                selected_labels = set(slot.selected_categories)
                selected_codes = [
                    code
                    for code, label in code_to_label.items()
                    if label in selected_labels
                ]
                if selected_codes:
                    mask &= np.isin(values, np.asarray(selected_codes, dtype=float))
                else:
                    mask &= np.zeros(cell_count, dtype=bool)
            else:
                lower, upper = slot.range_values
                mask &= np.isfinite(values) & (values >= lower) & (values <= upper)

        if bool(np.all(mask)):
            return self.state.mesh
        return self.state.mesh.extract_cells(mask)

    def _mesh_with_decile_scalars(self, mesh: pv.DataSet) -> tuple[pv.DataSet, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Plot state has not been loaded yet.")
        scalar_values = np.asarray(self.state.mesh.cell_data[self.state.scalar], dtype=float)
        decile_edges = _calculate_deciles(scalar_values)
        mesh_values = np.asarray(mesh.cell_data[self.state.scalar], dtype=float)
        decile_bins = _bin_to_deciles(mesh_values, decile_edges)
        display_mesh = mesh.copy(deep=True)
        display_mesh.cell_data["__pbm_decile_bin__"] = decile_bins
        return display_mesh, decile_edges

    def _refresh_plot(self, *, preserve_camera: bool = True) -> None:
        if self.state is None:
            return
        camera_position = None
        if preserve_camera and self._view_initialized:
            camera_position = getattr(self.plotter, "camera_position", None)
        self.plotter.clear()
        mesh = self._filtered_mesh()
        is_decile_mode = self.discretize_deciles and not self.state.scalar_is_categorical
        decile_edges: Optional[np.ndarray] = None
        scalar_for_coloring = self.state.scalar
        if is_decile_mode:
            mesh, decile_edges = self._mesh_with_decile_scalars(mesh)
            scalar_for_coloring = "__pbm_decile_bin__"
        mesh_kwargs = _plotter_add_mesh_kwargs(
            self.state,
            colormap=self.colormap,
            discretize_to_deciles=is_decile_mode,
            decile_edges=decile_edges,
            scalar_for_coloring=scalar_for_coloring,
        )
        mesh_kwargs["show_edges"] = self.show_edges

        if not self.state.scalar_is_categorical and not is_decile_mode:
            mesh_values = np.asarray(self.state.mesh.cell_data[self.state.scalar], dtype=float)
            finite_values = mesh_values[np.isfinite(mesh_values)]
            if finite_values.size > 0:
                mesh_kwargs["clim"] = (float(np.min(finite_values)), float(np.max(finite_values)))

        self.plotter.add_mesh(mesh, name="blockmodel", **mesh_kwargs)
        self.plotter.title = self.title
        self.plotter.add_axes()
        if preserve_camera and camera_position not in (None, []):
            self.plotter.camera_position = camera_position
        else:
            self.plotter.view_isometric()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        self._view_initialized = True
        if self._remote_view is not None:
            self._remote_view.update()

    def _reset_model_view(self) -> None:
        self.blockmodel = None
        self.state = None
        self.threshold = ThresholdRange(minimum=0.0, maximum=1.0, value=0.0, step=0.005)
        self.filter_enabled = False
        self._reset_filter_slot(0)
        self._reset_filter_slot(1)
        self._filter_attribute_options = []
        self._clear_filter_cache()
        self._initial_scalar = ""
        self._view_initialized = False
        self.plotter.clear()
        if self._remote_view is not None:
            self._remote_view.update()
        if self._server is not None:
            state = self._server.state
            state.attribute_options = []
            state.active_attribute = ""
            state.threshold_min = self.threshold.minimum
            state.threshold_max = self.threshold.maximum
            state.threshold = self.threshold.value
            state.threshold_display = self._format_threshold(self.threshold.value)
            state.threshold_step = self.threshold.step
            state.filter_active = False
            state.data_filter_attribute_options = []
            for idx in range(2):
                prefix = self._filter_slot_name(idx)
                setattr(state, f"{prefix}_attribute", "")
                setattr(state, f"{prefix}_is_categorical", False)
                setattr(state, f"{prefix}_range_min", 0.0)
                setattr(state, f"{prefix}_range_max", 1.0)
                setattr(state, f"{prefix}_range_step", 0.005)
                setattr(state, f"{prefix}_range", [0.0, 1.0])
                setattr(state, f"{prefix}_category_options", [])
                setattr(state, f"{prefix}_selected_categories", [])
                setattr(state, f"{prefix}_summary", "None")
            state.model_name = ""
            state.model_path = ""



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

    def load_blockmodel(self, blockmodel: "ParquetBlockModel", *, preferred_scalar: Optional[str] = None, auto_select_scalar: bool = True) -> None:
        self.blockmodel = blockmodel
        if not self._title_override:
            self.title = self.blockmodel.name
        self._clear_filter_cache()
        self._refresh_filter_options()

        self._syncing_state = True
        try:
            # Only load plot state if auto_select_scalar is True (file mode)
            # For hive mode, just populate attributes and wait for user selection
            if auto_select_scalar:
                scalar = self._resolve_initial_scalar(preferred_scalar)
                self._initial_scalar = scalar
                self._load_plot_state(scalar)
                self._refresh_plot(preserve_camera=False)
            else:
                # Hive mode: just prepare attributes, don't load a plot yet
                self._initial_scalar = ""
            self._apply_initial_data_filters()
            
            if self._server is not None:
                self._server.state.attribute_options = self.blockmodel.available_attributes
                self._server.state.model_name = self.blockmodel.name
                self._server.state.model_path = str(self.blockmodel.blockmodel_path)
                
                if auto_select_scalar and self.threshold is not None and self.state is not None:
                    # File mode: populate all threshold/attribute state
                    self._server.state.active_attribute = self.state.scalar
                    self._server.state.threshold_min = self.threshold.minimum
                    self._server.state.threshold_max = self.threshold.maximum
                    self._server.state.threshold = self.threshold.value
                    self._server.state.threshold_step = self.threshold.step
                    self._server.state.threshold_display = self._format_threshold(self.threshold.value)
                    self._server.state.filter_active = self.filter_enabled
                else:
                    # Hive mode: leave attribute empty for user selection
                    self._server.state.active_attribute = ""
                    self._server.state.threshold_min = 0.0
                    self._server.state.threshold_max = 1.0
                    self._server.state.threshold = 0.0
                    if self.threshold is None:
                        self.threshold = ThresholdRange(minimum=0.0, maximum=1.0, value=0.0, step=0.005)
                    self._server.state.threshold_step = self.threshold.step
                    self._server.state.threshold_display = "0"
                    self._server.state.filter_active = False
                self._sync_data_filter_state()
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
            # IMPORTANT: Don't auto-fill to first option on initial load
            # Only auto-fill if user has explicitly selected values AND an option is missing
            if current_value and current_value not in options:
                # Only auto-select if user has already interacted with this level
                if self._asset_selectors_autofilled:
                    current_value = options[0] if options else ""
            # Don't set to first option just because list is empty
            self._asset_level_values[key] = current_value
            if current_value:
                selection_prefix[key] = current_value
            level_options.append(options)

        name_options = self.asset_catalog.pbm_name_options(self._asset_selection())
        # Similar logic for asset name - don't auto-select first PBM on initial load
        if self._selected_asset_name and self._selected_asset_name not in name_options:
            if self._asset_selectors_autofilled:
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
            # IMPORTANT: Set values FIRST before options to prevent Vuetify from clearing v_model
            # when items array is updated
            for idx, key in enumerate(self._asset_level_keys):
                setattr(state, f"asset_level_{idx}_value", self._asset_level_values.get(key, ""))
            state.selected_asset_name = self._selected_asset_name
            # Now set options after values are in place
            for idx, key in enumerate(self._asset_level_keys):
                setattr(state, f"asset_level_{idx}_options", level_options[idx])
            state.asset_name_options = name_options
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
                    # IMPORTANT: Only process if the watched field is actually in kwargs
                    # Trame sometimes calls handlers with empty kwargs for internal events
                    if watched_field not in kwargs:
                        return
                     
                    if self._syncing_state:
                        return
                    if level_index >= len(self._asset_level_keys):
                        return
                    level_key = self._asset_level_keys[level_index]
                    new_value = str(kwargs.get(watched_field) or "")
                    old_value = self._asset_level_values.get(level_key, "")
                     
                    # Only sync if value actually changed
                    if new_value == old_value:
                        return
                     
                    # Update current level
                    self._asset_level_values[level_key] = new_value

                    # Clear all child levels and asset name when parent changes
                    for idx in range(level_index + 1, len(self._asset_level_keys)):
                        self._asset_level_values[self._asset_level_keys[idx]] = ""
                    self._selected_asset_name = ""

                    # Clear model view immediately when selection changes
                    self._reset_model_view()

                    self._asset_selectors_autofilled = True
                    # Let _sync_asset_selector_state manage _syncing_state
                    self._sync_asset_selector_state()
                    # NOTE: Don't call _load_selected_asset() - only load when asset name is selected

                return _asset_level_changed

            setattr(ctrl, f"update_asset_level_{idx}", _make_asset_level_handler(idx, field_name))

        if not self._asset_name_handler_registered:
            @state.change("selected_asset_name")
            def _asset_name_changed(selected_asset_name=None, **_):
                # IMPORTANT: Only process if selected_asset_name is not None
                # Trame sometimes calls handlers with None for internal events
                if selected_asset_name is None:
                    return
                 
                if self._syncing_state:
                    return
                new_name = str(selected_asset_name or "")
                old_name = self._selected_asset_name
                 
                # Only proceed if value actually changed
                if new_name == old_name:
                    return
                 
                self._selected_asset_name = new_name
                self._asset_selectors_autofilled = True

                # Only load when asset name is explicitly selected (not empty)
                if new_name:
                    self._reset_model_view()
                    self._load_selected_asset()
                    # Sync state AFTER loading asset so attribute_options are available
                    self._sync_asset_selector_state()
                else:
                    # Clear canvas if asset name is deselected
                    self._reset_model_view()
                    self._sync_asset_selector_state()

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
        
        # In file mode, skip if asset is already loaded
        # In hive mode with None blockmodel, always load (first time)
        if self._source_mode == "file" and self.blockmodel is not None:
            if selected.path.resolve() == self.blockmodel.blockmodel_path.resolve():
                return
        
        from parq_blockmodel.blockmodel import ParquetBlockModel

        self.load_blockmodel(ParquetBlockModel(blockmodel_path=selected.path), auto_select_scalar=False)
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
            self._asset_selectors_autofilled = False
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
        self.asset_catalog = catalog
        self.asset_catalog_root = path.resolve()
        self._asset_level_keys = list(catalog.level_keys)
        self._asset_level_values = {key: "" for key in self._asset_level_keys}
        self._selected_asset_name = ""
        self._asset_selectors_autofilled = False
        self._source_mode = "hive"
        self._source_path = str(path.resolve())
        # Don't auto-load first asset - let user select from DDLs
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
            self._refresh_plot(preserve_camera=self._view_initialized)
            if self._server is not None and self.threshold is not None:
                self._server.state.active_attribute = attribute
                self._server.state.threshold_min = self.threshold.minimum
                self._server.state.threshold_max = self.threshold.maximum
                self._server.state.threshold = self.threshold.value
                self._server.state.filter_active = self.filter_enabled
                self._sync_data_filter_state()
        finally:
            self._syncing_state = False

    def set_threshold(self, value: float) -> None:
        if self.threshold is None:
            raise RuntimeError("Threshold range has not been initialised.")
        self.threshold.value = float(value)
        self.filter_enabled = True
        self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._server.state.threshold = self.threshold.value
            self._server.state.threshold_display = self._format_threshold(self.threshold.value)
            self._server.state.filter_active = self.filter_enabled

    def set_threshold_from_text(self, text_value: str) -> None:
        if self.threshold is None:
            raise RuntimeError("Threshold range has not been initialised.")
        try:
            value = float(text_value)
            if not (self.threshold.minimum <= value <= self.threshold.maximum):
                if self._server is not None:
                    self._server.state.threshold_display = self._format_threshold(self.threshold.value)
                return
            self.set_threshold(value)
        except (ValueError, TypeError):
            if self._server is not None:
                self._server.state.threshold_display = self._format_threshold(self.threshold.value)

    def reset_filter(self) -> None:
        if self.threshold is None:
            raise RuntimeError("Threshold range has not been initialised.")
        self.filter_enabled = False
        self.threshold.value = self.threshold.minimum
        self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._syncing_state = True
            try:
                self._server.state.threshold = self.threshold.value
                self._server.state.threshold_display = self._format_threshold(self.threshold.value)
                self._server.state.filter_active = False
            finally:
                self._syncing_state = False

    def set_colormap(self, colormap: str) -> None:
        self.colormap = colormap
        self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._server.state.selected_colormap = colormap

    def set_discretize_deciles(self, enabled: bool) -> None:
        self.discretize_deciles = bool(enabled)
        self._refresh_plot(preserve_camera=self._view_initialized)
        if self._server is not None:
            self._server.state.discretize_deciles = self.discretize_deciles

    def _get_available_colormaps(self) -> list[str]:
        try:
            import matplotlib.pyplot as plt
            all_cmaps = list(plt.colormaps())
        except (ImportError, AttributeError):
            all_cmaps = ["viridis", "plasma", "inferno", "cool", "hot"]
        
        useful_cmaps = [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "cool", "hot", "spring", "summer", "autumn", "winter",
            "jet", "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "seismic",
            "twilight", "twilight_shifted", "hsv", "bone", "copper",
            "gray", "hot_r", "cool_r", "Greys", "Blues", "Greens", "Oranges",
            "Reds", "Purples", "RdPu", "BuGn", "BuPu", "GnBu", "OrRd", "PuBu"
        ]
        
        available = [c for c in useful_cmaps if c in all_cmaps]
        if not available:
            available = all_cmaps[:10]
        
        return sorted(available)

    def _build_launch_kwargs(self, port: Optional[int], host: Optional[str]) -> dict[str, object]:
        start_kwargs: dict[str, object] = {"open_browser": True, "show_connection_info": True}
        if port is not None:
            start_kwargs["port"] = port
        if host is not None:
            start_kwargs["host"] = host
        return start_kwargs

    def launch(
        self,
        server_name: str = "parq-blockmodel-trame",
        port: Optional[int] = None,
        host: Optional[str] = None,
    ):
        """Launch the Trame visualization app.
        
        Parameters
        ----------
        server_name : str, optional
            Base name for the Trame server, by default "parq-blockmodel-trame"
            A UUID suffix is appended to ensure unique server instances.
        port : int, optional
            Port number for the server. If None, Trame will auto-assign a port.
            Useful for running multiple instances without server state conflicts.
            Example: launch(port=8080), launch(port=8081), etc.
        host : str, optional
            Host interface to bind to. Use "0.0.0.0" to accept external connections.
        """
        try:
            from trame.app import get_server
            from trame.ui.vuetify import SinglePageWithDrawerLayout
            from trame.widgets import vtk, vuetify
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Trame dependencies are required for the interactive app. "
                "Install the 'trame', 'trame-vtk', and 'trame-vuetify' packages."
            ) from exc
        try:  # Optional: used for browser key-capture hotkeys.
            from trame.widgets import trame as trame_widgets
        except ImportError:  # pragma: no cover
            trame_widgets = None

        # Reset all instance state to ensure fresh launch (not stale from previous launch)
        self._asset_level_values = {}
        self._selected_asset_name = ""
        self._asset_selectors_autofilled = False
        
        # Use unique server name to force fresh Trame server instance (prevents caching)
        unique_server_name = f"{server_name}-{uuid4().hex[:8]}"
        server = get_server(unique_server_name, client_type="vue2")
        self._server = server
        state, ctrl = server.state, server.controller
        
        self.plotter.clear()
        
        self._syncing_state = True
        try:
            # For hive mode, skip initial blockmodel load - user will select via dropdowns
            if not self._skip_initial_blockmodel_load:
                self.load_blockmodel(self.blockmodel, preferred_scalar=self._initial_scalar)
            
            # Initialize available colormaps
            self.available_colormaps = self._get_available_colormaps()
            self._refresh_filter_options()
            self._apply_initial_data_filters()
            
            # Set state only if blockmodel was loaded
            if self.state is not None:
                state.attribute_options = self.blockmodel.available_attributes
                state.active_attribute = self.state.scalar
                state.threshold_min = self.threshold.minimum
                state.threshold_max = self.threshold.maximum
                state.threshold = self.threshold.value
                state.threshold_display = self._format_threshold(self.threshold.value)
                state.threshold_step = self.threshold.step
                state.filter_active = self.filter_enabled
            else:
                # Empty state for hive mode before user selection
                if self.threshold is None:
                    self.threshold = ThresholdRange(minimum=0.0, maximum=1.0, value=0.0, step=0.005)
                state.attribute_options = []
                state.active_attribute = ""
                state.threshold_min = self.threshold.minimum
                state.threshold_max = self.threshold.maximum
                state.threshold = self.threshold.value
                state.threshold_display = self._format_threshold(self.threshold.value)
                state.threshold_step = self.threshold.step
                state.filter_active = False
            self._sync_data_filter_state()
            
            state.colormap_options = self.available_colormaps
            state.selected_colormap = self.colormap
            state.discretize_deciles = self.discretize_deciles
            state.control_panel = 0
            if self.state is not None:
                state.model_name = self.blockmodel.name
                state.model_path = str(self.blockmodel.blockmodel_path)
            else:
                state.model_name = ""
                state.model_path = ""
            state.source_mode = self._source_mode
            state.source_path_input = self._source_path
            state.source_status = ""
            state.asset_selector_enabled = self.asset_catalog is not None
            if self.asset_catalog is not None:
                self._asset_level_keys = list(self.asset_catalog.level_keys)
                self._asset_level_values = {key: "" for key in self._asset_level_keys}
        finally:
            self._syncing_state = False

        @state.change("active_attribute")
        def _attribute_changed(active_attribute=None, **_):
            if self._syncing_state or active_attribute is None or not active_attribute:
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
            self._refresh_plot(preserve_camera=self._view_initialized)

        @state.change("data_filter_1_attribute")
        def _data_filter_1_attribute_changed(data_filter_1_attribute=None, **_):
            if self._syncing_state or data_filter_1_attribute is None:
                return
            self.set_data_filter_attribute(0, str(data_filter_1_attribute))

        @state.change("data_filter_2_attribute")
        def _data_filter_2_attribute_changed(data_filter_2_attribute=None, **_):
            if self._syncing_state or data_filter_2_attribute is None:
                return
            self.set_data_filter_attribute(1, str(data_filter_2_attribute))

        @state.change("data_filter_1_range")
        def _data_filter_1_range_changed(data_filter_1_range=None, **_):
            if self._syncing_state or data_filter_1_range is None:
                return
            self.set_data_filter_range(0, data_filter_1_range)

        @state.change("data_filter_2_range")
        def _data_filter_2_range_changed(data_filter_2_range=None, **_):
            if self._syncing_state or data_filter_2_range is None:
                return
            self.set_data_filter_range(1, data_filter_2_range)

        @state.change("data_filter_1_selected_categories")
        def _data_filter_1_categories_changed(data_filter_1_selected_categories=None, **_):
            if self._syncing_state or data_filter_1_selected_categories is None:
                return
            self.set_data_filter_categories(0, list(data_filter_1_selected_categories))

        @state.change("data_filter_2_selected_categories")
        def _data_filter_2_categories_changed(data_filter_2_selected_categories=None, **_):
            if self._syncing_state or data_filter_2_selected_categories is None:
                return
            self.set_data_filter_categories(1, list(data_filter_2_selected_categories))

        @state.change("selected_colormap")
        def _colormap_changed(selected_colormap=None, **_):
            if self._syncing_state or selected_colormap is None or not selected_colormap:
                return
            self.set_colormap(selected_colormap)

        @state.change("discretize_deciles")
        def _discretize_deciles_changed(discretize_deciles=None, **_):
            if self._syncing_state or discretize_deciles is None:
                return
            self.set_discretize_deciles(bool(discretize_deciles))

        ctrl.update_attribute = _attribute_changed
        ctrl.update_threshold = _threshold_changed
        ctrl.update_colormap = _colormap_changed
        ctrl.update_discretize_deciles = _discretize_deciles_changed
        ctrl.update_data_filter_1_attribute = _data_filter_1_attribute_changed
        ctrl.update_data_filter_2_attribute = _data_filter_2_attribute_changed
        ctrl.update_data_filter_1_range = _data_filter_1_range_changed
        ctrl.update_data_filter_2_range = _data_filter_2_range_changed
        ctrl.update_data_filter_1_categories = _data_filter_1_categories_changed
        ctrl.update_data_filter_2_categories = _data_filter_2_categories_changed
        def _apply_threshold_text(**_):
            if self._syncing_state or self._server is None:
                return
            self.set_threshold_from_text(str(self._server.state.threshold_display))

        ctrl.apply_threshold_text = _apply_threshold_text
        ctrl.reset_filter = self.reset_filter
        ctrl.reset_data_filter_1 = lambda **_: self.reset_data_filter(0)
        ctrl.reset_data_filter_2 = lambda **_: self.reset_data_filter(1)
        ctrl.update_asset_name = lambda **_: None
        def _extract_client_key(event_payload=None, **kwargs) -> str:
            values = []
            if isinstance(event_payload, dict):
                values.extend(
                    [
                        event_payload.get("key"),
                        event_payload.get("keySym"),
                        event_payload.get("keysym"),
                        event_payload.get("code"),
                        event_payload.get("keyCode"),
                    ]
                )
            values.extend(
                [
                    kwargs.get("key"),
                    kwargs.get("keySym"),
                    kwargs.get("keysym"),
                    kwargs.get("code"),
                    kwargs.get("keyCode"),
                ]
            )
            for value in values:
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    code = int(value)
                    if 0 <= code <= 255:
                        return chr(code).lower()
                key_text = str(value).strip().lower()
                if key_text:
                    return key_text
            return ""

        def _on_zup_key_down(event=None, **kwargs):
            if not self.z_up_lock:
                return
            key = _extract_client_key(event, **kwargs)
            if key and key != self.z_up_hotkey:
                return
            _set_z_up_hotkey_state(self.plotter, True)

        def _on_zup_key_up(event=None, **kwargs):
            if not self.z_up_lock:
                return
            key = _extract_client_key(event, **kwargs)
            if key and key != self.z_up_hotkey:
                return
            _set_z_up_hotkey_state(self.plotter, False)

        ctrl.zup_key_down = _on_zup_key_down
        ctrl.zup_key_up = _on_zup_key_up
        if self.asset_catalog is not None:
            self._register_asset_selector_handlers()
            self._syncing_state = True
            try:
                self._sync_asset_selector_state()
            finally:
                self._syncing_state = False

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
                vuetify.VToolbarTitle(self.app_name)
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
                                "Data Filter",
                                dense=True,
                                classes="py-1 px-2 text-subtitle-2",
                            )
                            with vuetify.VExpansionPanelContent(classes="pt-1 pb-2 px-2"):
                                vuetify.VSelect(
                                    v_model=("data_filter_1_attribute", self._data_filters[0].attribute),
                                    items=("data_filter_attribute_options", self._filter_attribute_options),
                                    label="Filter 1 attribute",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
                                    clearable=True,
                                    change=ctrl.update_data_filter_1_attribute,
                                )
                                vuetify.VRangeSlider(
                                    v_if=("!data_filter_1_is_categorical",),
                                    v_model=("data_filter_1_range", list(self._data_filters[0].range_values)),
                                    min=("data_filter_1_range_min", self._data_filters[0].minimum),
                                    max=("data_filter_1_range_max", self._data_filters[0].maximum),
                                    step=("data_filter_1_range_step", self._data_filters[0].step),
                                    label="Filter 1 range",
                                    dense=True,
                                    hide_details=True,
                                    thumb_label=False,
                                    classes="mt-2",
                                    disabled=("data_filter_1_attribute === ''",),
                                    change=ctrl.update_data_filter_1_range,
                                )
                                vuetify.VSelect(
                                    v_if=("data_filter_1_is_categorical",),
                                    v_model=("data_filter_1_selected_categories", list(self._data_filters[0].selected_categories)),
                                    items=("data_filter_1_category_options", list(self._data_filters[0].category_options)),
                                    label="Filter 1 categories",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
                                    multiple=True,
                                    chips=True,
                                    deletable_chips=True,
                                    change=ctrl.update_data_filter_1_categories,
                                )
                                vuetify.VBtn(
                                    "Reset filter 1",
                                    click=ctrl.reset_data_filter_1,
                                    block=True,
                                    color="primary",
                                    classes="mt-2",
                                )
                                vuetify.VChip(
                                    "{{ data_filter_1_summary }}",
                                    label=True,
                                    small=True,
                                    outlined=True,
                                    classes="mt-2 text-caption text-truncate",
                                    style="max-width: 100%;",
                                )
                                vuetify.VSelect(
                                    v_model=("data_filter_2_attribute", self._data_filters[1].attribute),
                                    items=("data_filter_attribute_options", self._filter_attribute_options),
                                    label="Filter 2 attribute",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-4",
                                    clearable=True,
                                    change=ctrl.update_data_filter_2_attribute,
                                )
                                vuetify.VRangeSlider(
                                    v_if=("!data_filter_2_is_categorical",),
                                    v_model=("data_filter_2_range", list(self._data_filters[1].range_values)),
                                    min=("data_filter_2_range_min", self._data_filters[1].minimum),
                                    max=("data_filter_2_range_max", self._data_filters[1].maximum),
                                    step=("data_filter_2_range_step", self._data_filters[1].step),
                                    label="Filter 2 range",
                                    dense=True,
                                    hide_details=True,
                                    thumb_label=False,
                                    classes="mt-2",
                                    disabled=("data_filter_2_attribute === ''",),
                                    change=ctrl.update_data_filter_2_range,
                                )
                                vuetify.VSelect(
                                    v_if=("data_filter_2_is_categorical",),
                                    v_model=("data_filter_2_selected_categories", list(self._data_filters[1].selected_categories)),
                                    items=("data_filter_2_category_options", list(self._data_filters[1].category_options)),
                                    label="Filter 2 categories",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
                                    multiple=True,
                                    chips=True,
                                    deletable_chips=True,
                                    change=ctrl.update_data_filter_2_categories,
                                )
                                vuetify.VBtn(
                                    "Reset filter 2",
                                    click=ctrl.reset_data_filter_2,
                                    block=True,
                                    color="primary",
                                    classes="mt-2",
                                )
                                vuetify.VChip(
                                    "{{ data_filter_2_summary }}",
                                    label=True,
                                    small=True,
                                    outlined=True,
                                    classes="mt-2 text-caption text-truncate",
                                    style="max-width: 100%;",
                                )
                        with vuetify.VExpansionPanel():
                            vuetify.VExpansionPanelHeader(
                                "Controls",
                                dense=True,
                                classes="py-1 px-2 text-subtitle-2",
                            )
                            with vuetify.VExpansionPanelContent(classes="pt-1 pb-2 px-2"):
                                vuetify.VSelect(
                                    v_model=("active_attribute", self.state.scalar if self.state else ""),
                                    items=("attribute_options", self.blockmodel.available_attributes if self.state else []),
                                    label="Attribute",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
                                    change=ctrl.update_attribute,
                                )
                                vuetify.VSelect(
                                    v_model=("selected_colormap", self.colormap),
                                    items=("colormap_options", self.available_colormaps),
                                    label="Colormap",
                                    dense=True,
                                    hide_details=True,
                                    outlined=True,
                                    classes="mt-2",
                                    change=ctrl.update_colormap,
                                )
                                vuetify.VCheckbox(
                                    v_model=("discretize_deciles", self.discretize_deciles),
                                    label="Discretise to deciles",
                                    dense=True,
                                    hide_details=True,
                                    classes="mt-2",
                                    change=ctrl.update_discretize_deciles,
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
                                    label="Threshold value",
                                    dense=True,
                                    outlined=True,
                                    hide_details=True,
                                    classes="mt-2",
                                )
                                vuetify.VBtn(
                                    "Apply",
                                    click=ctrl.apply_threshold_text,
                                    block=True,
                                    color="primary",
                                    classes="mt-2",
                                )
                                vuetify.VBtn(
                                    "Reset threshold",
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
                if self.z_up_lock and trame_widgets is not None:
                    key_trap = trame_widgets.MouseTrap(
                        ZUpKeyDown=ctrl.zup_key_down,
                        ZUpKeyUp=ctrl.zup_key_up,
                    )
                    key_trap.bind("z", "ZUpKeyDown", listen_to="keydown")
                    key_trap.bind("z", "ZUpKeyUp", listen_to="keyup")
                remote_view_kwargs = {
                    "interactive_ratio": 1,
                    "style": "width: 100%; height: 100%; min-height: 600px;",
                }
                if self.z_up_lock:
                    remote_view_kwargs.update(
                        {
                            "interactor_events": ("zup_interactor_events", ["KeyDown", "KeyPress", "KeyUp"]),
                            "KeyDown": ctrl.zup_key_down,
                            "KeyPress": ctrl.zup_key_down,
                            "KeyUp": ctrl.zup_key_up,
                        }
                    )
                self._remote_view = vtk.VtkRemoteView(
                    self.plotter.ren_win,
                    **remote_view_kwargs,
                )
                self._remote_view.update()

        state.ready()
        server.start(**self._build_launch_kwargs(port, host))
        return server
