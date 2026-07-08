import sys
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.visualization import BlockModelTrameApp, TrameBlockModelPlotEngine
from parq_blockmodel.visualization.blockmodel_plot import (
    _add_elevation_overlay,
)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def plot(self, blockmodel, **kwargs):
        self.calls.append((blockmodel, kwargs))
        return "engine-result"


class BaseFakePlotter:
    """Base fake plotter with all required CustomPlotter methods for testing."""
    def __init__(self, *args, **kwargs):
        self.actors = {}
        self.title = None
        self.picking_enabled = False
        self.hotkey_pressed = {'z': False}

    def clear(self):
        self.actors.clear()

    def add_mesh(self, mesh, **kwargs):
        self.actors["blockmodel"] = mesh

    def set_directional_view(self, direction=None, **kwargs):
        return None

    def view_isometric(self):
        return None

    def reset_camera_clipping_range(self):
        return None

    def add_axes(self):
        return None

    def render(self):
        return None

    def show(self, *args, **kwargs):
        return None

    def enforce_z_up(self):
        return None

    def setup_picking_with_callback(self, callback_func):
        return None

    def disable_picking(self):
        return None

    def remove_actor(self, name):
        self.actors.pop(name, None)


def test_blockmodel_plot_delegates_to_custom_engine(tmp_path):
    parquet_path = tmp_path / "engine_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    engine = FakeEngine()

    result = pbm.plot(
        scalar=pbm.available_attributes[0],
        threshold=False,
        engine=engine,
    )

    assert result == "engine-result"
    assert len(engine.calls) == 1
    blockmodel, kwargs = engine.calls[0]
    assert blockmodel is pbm
    assert kwargs["scalar"] == pbm.available_attributes[0]
    assert kwargs["threshold"] is False


def test_blockmodel_plot_forwards_z_up_settings_to_engine(tmp_path):
    parquet_path = tmp_path / "engine_zup_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    engine = FakeEngine()

    pbm.plot(
        scalar=pbm.available_attributes[0],
        engine=engine,
        z_up_lock=True,
        z_up_hotkey="z",
    )

    _, kwargs = engine.calls[0]
    assert kwargs["z_up_lock"] is True
    assert kwargs["z_up_hotkey"] == "z"


def test_blockmodel_plot_forwards_raster_settings_to_engine(tmp_path):
    parquet_path = tmp_path / "engine_raster_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    engine = FakeEngine()

    pbm.plot(
        scalar=pbm.available_attributes[0],
        engine=engine,
        elevation_raster="elev.tif",
        imagery_raster="imagery.tif",
    )

    _, kwargs = engine.calls[0]
    assert kwargs["elevation_raster"] == "elev.tif"
    assert kwargs["imagery_raster"] == "imagery.tif"


def test_add_elevation_overlay_adds_textured_surface_and_toggle(monkeypatch):
    calls = {"add_mesh": [], "toggle": None}

    class FakeActor:
        def __init__(self):
            self.visible = True

        def SetVisibility(self, visible):
            self.visible = bool(visible)

    class FakeSurface:
        def __init__(self):
            self.mapped = False

        def texture_map_to_plane(self, inplace=True, use_bounds=True):
            self.mapped = bool(inplace and use_bounds)

    class FakeTexture:
        def __init__(self):
            self.flipped = False

        def flip_y(self):
            self.flipped = True
            return self

    class FakePlotter(BaseFakePlotter):
        def __init__(self):
            self.render_calls = 0
        def add_mesh(self, mesh, **kwargs):
            calls["add_mesh"].append((mesh, kwargs))
            return FakeActor()

        def add_checkbox_button_widget(self, callback, **kwargs):
            calls["toggle"] = (callback, kwargs)

        def add_text(self, *_args, **_kwargs):
            return None

        def render(self):
            self.render_calls += 1

        def set_directional_view(self, direction=None, **kwargs):
            return None

    fake_surface = FakeSurface()
    fake_texture = FakeTexture()
    fake_plotter = FakePlotter()
    monkeypatch.setattr(
        "parq_blockmodel.visualization.blockmodel_plot._build_elevation_surface",
        lambda _path: fake_surface,
    )
    monkeypatch.setattr(
        "parq_blockmodel.visualization.blockmodel_plot.pv.read_texture",
        lambda _path: fake_texture,
    )

    _add_elevation_overlay(
        fake_plotter,
        elevation_raster="elev.tif",
        imagery_raster="imagery.tif",
    )

    assert fake_surface.mapped is True
    assert len(calls["add_mesh"]) == 1
    _, mesh_kwargs = calls["add_mesh"][0]
    assert mesh_kwargs["texture"] is fake_texture
    assert fake_texture.flipped is True
    assert calls["toggle"] is not None
    callback, _ = calls["toggle"]
    callback(False)
    assert fake_plotter.render_calls == 1


def test_trame_plot_engine_returns_trame_app_with_z_up(tmp_path, monkeypatch):
    parquet_path = tmp_path / "engine_trame_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    class FakeInteractor:
        def __init__(self):
            self.observers = []

        def add_observer(self, event_name, callback):
            self.observers.append((event_name, callback))

    class FakeCamera:
        def SetViewUp(self, *_):
            return None

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.iren = FakeInteractor()
            super().__init__(*args, **kwargs)
            self.camera = FakeCamera()
            self.ren_win = object()
            self.actors = {}
            self.hotkey_pressed = {'z': False}

        def clear(self):
            return None

        def add_mesh(self, *args, **kwargs):
            return None

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)
    engine = TrameBlockModelPlotEngine()
    app = pbm.plot(
        scalar=pbm.available_attributes[0],
        engine=engine,
        z_up_lock=True,
        z_up_hotkey="z",
    )

    assert isinstance(app, BlockModelTrameApp)
    assert app.z_up_lock is True
    assert app.z_up_hotkey == "z"


def test_trame_app_thresholding_updates_filtered_scene_without_mutating_pbm(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    original_columns = list(pbm.columns)

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)
            self.title = None
            self.last_mesh_n_cells = None

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.last_mesh_n_cells = mesh.n_cells
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._refresh_plot()
    base_cells = app.plotter.last_mesh_n_cells

    app.set_threshold(app.threshold.maximum)

    assert app.filter_enabled is True
    assert app.plotter.last_mesh_n_cells <= base_cells
    assert pbm.columns == list(pbm.columns)

    app.reset_filter()
    assert app.filter_enabled is False
    assert app.threshold.value == app.threshold.minimum
    assert pbm.columns == original_columns
    assert app.plotter.last_mesh_n_cells == base_cells


def test_trame_app_data_filter_uses_cached_column_values(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_cache_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    filter_attribute = pbm.available_attributes[0]

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=filter_attribute, show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._refresh_filter_options()

    calls = {"count": 0}
    original_loader = app._load_filter_attribute_values

    def _counting_loader(attribute):
        calls["count"] += 1
        return original_loader(attribute)

    monkeypatch.setattr(app, "_load_filter_attribute_values", _counting_loader)

    app.set_data_filter_attribute(0, filter_attribute)
    low, high = app._data_filters[0].range_values
    mid = (low + high) / 2.0
    app.set_data_filter_range(0, [low, mid])
    app.set_data_filter_range(0, [mid, high])

    assert calls["count"] == 1


def test_trame_app_reuses_cached_values_when_switching_attributes(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_cache_switch_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    assert len(pbm.available_attributes) >= 2
    attr1 = pbm.available_attributes[0]
    attr2 = pbm.available_attributes[1]

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=attr1, show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._refresh_filter_options()

    calls = {"count": 0}
    original_loader = app._load_filter_attribute_values

    def _counting_loader(attribute):
        calls["count"] += 1
        return original_loader(attribute)

    monkeypatch.setattr(app, "_load_filter_attribute_values", _counting_loader)

    app.set_data_filter_attribute(0, attr1)
    app.set_data_filter_attribute(0, attr2)
    app.set_data_filter_attribute(0, attr1)

    assert calls["count"] == 2


def test_trame_app_accepts_initial_data_filter_bounds(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_filter_preset_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    filter_attribute = pbm.available_attributes[0]

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(
        pbm,
        scalar=filter_attribute,
        data_filter_1_attribute=filter_attribute,
        data_filter_1_min=-1.0e9,
        data_filter_1_max=1.0e9,
        show_edges=False,
    )
    app._load_plot_state(app._initial_scalar)
    app._refresh_filter_options()
    app._apply_initial_data_filters()

    slot = app._data_filters[0]
    assert slot.attribute == filter_attribute
    assert slot.range_values[0] == slot.minimum
    assert slot.range_values[1] == slot.maximum


def test_trame_app_initial_preset_filter_keeps_initial_mesh_non_empty(monkeypatch, tmp_path):
    pbm_path = tmp_path / "initial_filter_mesh.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=pbm_path, shape=(4, 4, 4))

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)
            self.last_mesh_n_cells = None

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.last_mesh_n_cells = mesh.n_cells
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(
        pbm,
        scalar="density",
        data_filter_1_attribute="density",
        data_filter_1_min=2.4,
        show_edges=False,
    )
    app.load_blockmodel(pbm, preferred_scalar="density")

    assert app.plotter.last_mesh_n_cells is not None
    assert app.plotter.last_mesh_n_cells > 0


def test_trame_app_respects_initial_threshold_value(monkeypatch, tmp_path):
    pbm_path = tmp_path / "initial_threshold_value.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=pbm_path, shape=(4, 4, 4))

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(
        pbm,
        scalar="density",
        threshold_value=2.6,
        data_filter_1_attribute="density",
        data_filter_1_min=2.4,
        show_edges=False,
    )
    app.load_blockmodel(pbm, preferred_scalar="density")

    assert app.threshold is not None
    assert app.threshold.value == 2.6


def test_trame_file_startup_presets_survive_watcher_replay(monkeypatch, tmp_path):
    parquet_path = tmp_path / "watcher_replay_startup.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(4, 4, 4))
    pbm_path = pbm.blockmodel_path
    scalar_attr = pbm.available_attributes[0]
    scalar_values = pbm.data[scalar_attr].to_numpy(dtype=float, copy=False)
    threshold_value = float(np.nanmedian(scalar_values))
    filter_min = float(np.nanquantile(scalar_values, 0.25))

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.ren_win = object()
            super().__init__(*args, **kwargs)
            self.actors = {}

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyToolbar(DummyContext):
        def __init__(self):
            self.color = None
            self.dense = None
            self.dark = None

    class FakeLayout:
        def __init__(self, *args, **kwargs):
            self.toolbar = DummyToolbar()
            self.title = SimpleNamespace(set_text=lambda *_: None)
            self.drawer = DummyContext()
            self.content = DummyContext()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeState:
        def __init__(self):
            self._callbacks = {}

        def change(self, key):
            def register(callback):
                self._callbacks.setdefault(key, []).append(callback)
                return callback

            return register

        def __setattr__(self, key, value):
            object.__setattr__(self, key, value)

        def ready(self):
            self._ready_called = True

    fake_state = FakeState()
    fake_trame = ModuleType("trame")
    fake_trame_app = ModuleType("trame.app")
    fake_trame_ui = ModuleType("trame.ui")
    fake_trame_ui_vuetify = ModuleType("trame.ui.vuetify")
    fake_trame_widgets = ModuleType("trame.widgets")

    fake_trame_app.get_server = lambda name, client_type=None: SimpleNamespace(
        state=fake_state,
        controller=SimpleNamespace(),
        start=lambda **kwargs: None,
    )
    fake_trame_ui_vuetify.SinglePageWithDrawerLayout = FakeLayout
    fake_trame_widgets.vtk = SimpleNamespace(
        VtkRemoteView=lambda ren_win, **kwargs: SimpleNamespace(update=lambda: None)
    )
    fake_trame_widgets.vuetify = SimpleNamespace(
        VSelect=lambda *args, **kwargs: DummyContext(),
        VSlider=lambda *args, **kwargs: DummyContext(),
        VRangeSlider=lambda *args, **kwargs: DummyContext(),
        VDialog=lambda *args, **kwargs: DummyContext(),
        VBtn=lambda *args, **kwargs: DummyContext(),
        VAppBar=lambda *args, **kwargs: DummyContext(),
        VAppBarNavIcon=lambda *args, **kwargs: DummyContext(),
        VToolbarTitle=lambda *args, **kwargs: DummyContext(),
        VSpacer=lambda *args, **kwargs: DummyContext(),
        VChip=lambda *args, **kwargs: DummyContext(),
        VNavigationDrawer=lambda *args, **kwargs: DummyContext(),
        VSheet=lambda *args, **kwargs: DummyContext(),
        VImg=lambda *args, **kwargs: DummyContext(),
        VCard=lambda *args, **kwargs: DummyContext(),
        VCardText=lambda *args, **kwargs: DummyContext(),
        VCardTitle=lambda *args, **kwargs: DummyContext(),
        VExpansionPanels=lambda *args, **kwargs: DummyContext(),
        VExpansionPanel=lambda *args, **kwargs: DummyContext(),
        VExpansionPanelHeader=lambda *args, **kwargs: DummyContext(),
        VExpansionPanelContent=lambda *args, **kwargs: DummyContext(),
        VMain=lambda *args, **kwargs: DummyContext(),
        VContainer=lambda *args, **kwargs: DummyContext(),
        VIcon=lambda *args, **kwargs: DummyContext(),
        VTextField=lambda *args, **kwargs: DummyContext(),
        VCheckbox=lambda *args, **kwargs: DummyContext(),
    )
    fake_trame_widgets.trame = SimpleNamespace(MouseTrap=lambda **kwargs: SimpleNamespace(bind=lambda *a, **k: None))
    fake_trame.app = fake_trame_app
    fake_trame.ui = fake_trame_ui
    fake_trame.widgets = fake_trame_widgets
    fake_trame_ui.vuetify = fake_trame_ui_vuetify

    monkeypatch.setitem(sys.modules, "trame", fake_trame)
    monkeypatch.setitem(sys.modules, "trame.app", fake_trame_app)
    monkeypatch.setitem(sys.modules, "trame.ui", fake_trame_ui)
    monkeypatch.setitem(sys.modules, "trame.ui.vuetify", fake_trame_ui_vuetify)
    monkeypatch.setitem(sys.modules, "trame.widgets", fake_trame_widgets)
    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp.from_pbm_file(
        pbm_path,
        scalar=scalar_attr,
        threshold_value=threshold_value,
        data_filter_1_attribute=scalar_attr,
        data_filter_1_min=filter_min,
        app_name="File Preset Test",
    )
    app.launch(port=3080, host="0.0.0.0")

    assert app.threshold is not None
    assert np.isclose(float(app.threshold.value), threshold_value)
    assert app._data_filters[0].attribute == scalar_attr
    assert np.isclose(float(app._data_filters[0].range_values[0]), filter_min)

    active_callbacks = fake_state._callbacks.get("active_attribute", [])
    threshold_callbacks = fake_state._callbacks.get("threshold", [])
    filter_callbacks = fake_state._callbacks.get("data_filter_1_attribute", [])
    assert len(active_callbacks) == 1
    assert len(threshold_callbacks) == 1
    assert len(filter_callbacks) == 1

    active_callbacks[0](active_attribute=scalar_attr)
    threshold_callbacks[0](threshold=threshold_value)
    filter_callbacks[0](data_filter_1_attribute=scalar_attr)

    assert np.isclose(float(app.threshold.value), threshold_value)
    assert app._data_filters[0].attribute == scalar_attr
    assert np.isclose(float(app._data_filters[0].range_values[0]), filter_min)


def test_trame_app_supports_categorical_data_filter(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_categorical_filter_source.parquet"
    ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    df = pd.read_parquet(parquet_path)
    categories = pd.Categorical(["A", "B", "C"] * 9)
    df["rock_type"] = categories[: len(df)]
    df.to_parquet(parquet_path, index=False)
    pbm = ParquetBlockModel.from_parquet(parquet_path)

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)
            self.last_mesh_n_cells = None

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.last_mesh_n_cells = mesh.n_cells
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._refresh_filter_options()
    app._refresh_plot()
    base_cells = app.plotter.last_mesh_n_cells

    app.set_data_filter_attribute(0, "rock_type")
    app.set_data_filter_categories(0, ["A"])

    assert app._data_filters[0].is_categorical is True
    assert app.plotter.last_mesh_n_cells < base_cells


def test_trame_app_data_filter_summary_tracks_selected_values(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_summary_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    attr = pbm.available_attributes[0]

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=attr, show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._refresh_filter_options()
    app.set_data_filter_attribute(0, attr)

    summary = app._data_filter_slot_summary(app._data_filters[0])
    assert f"{attr}:" in summary
    assert "to" in summary


def test_trame_app_renders_categorical_attributes(tmp_path, monkeypatch):
    parquet_path = tmp_path / "categorical_source.parquet"
    ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    df = pd.read_parquet(parquet_path)
    categories = pd.Categorical(["A", "B", "C"] * 9)
    df["rock_type"] = categories[: len(df)]
    df.to_parquet(parquet_path, index=False)
    pbm = ParquetBlockModel.from_parquet(parquet_path)

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)
            self.title = None
            self.last_mesh_n_cells = None

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.last_mesh_n_cells = mesh.n_cells
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar="rock_type", show_edges=False)
    app._load_plot_state("rock_type")
    app._refresh_plot()

    assert app.state is not None
    assert app.state.scalar_is_categorical is True
    assert app.plotter.last_mesh_n_cells > 0


def test_trame_app_discretises_continuous_values_to_deciles(tmp_path, monkeypatch):
    parquet_path = tmp_path / "decile_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.actors = {}
            super().__init__(*args, **kwargs)
            self.last_kwargs = {}
            self.last_mesh = None

        def clear(self):
            self.actors.clear()

        def add_mesh(self, mesh, **kwargs):
            self.last_mesh = mesh
            self.last_kwargs = kwargs
            self.actors["blockmodel"] = mesh

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app.set_discretize_deciles(True)

    assert app.plotter.last_mesh is not None
    assert "__pbm_decile_bin__" in app.plotter.last_mesh.cell_data
    assert app.plotter.last_kwargs.get("scalars") == "__pbm_decile_bin__"
    assert app.plotter.last_kwargs.get("categories") is True
    assert "clim" not in app.plotter.last_kwargs
    decile_values = np.asarray(app.plotter.last_mesh.cell_data["__pbm_decile_bin__"], dtype=float)
    finite_values = decile_values[np.isfinite(decile_values)]
    assert finite_values.size > 0
    assert float(np.min(finite_values)) >= 0.0
    assert float(np.max(finite_values)) <= 9.0


def test_trame_example_seeds_temp_demo_when_sample_missing(tmp_path, monkeypatch):
    example_path = Path(__file__).resolve().parents[2] / "examples" / "16_trame_threshold_viewer.py"
    spec = util.spec_from_file_location("xx_trame_threshold_viewer", example_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    launched = {}
    created = []

    def fake_create_toy_blockmodel(*, filename, shape):
        created.append(Path(filename))
        launched["shape"] = shape
        return object()

    class FakeApp:
        def launch(self, **kwargs):
            launched["kwargs"] = kwargs
            launched["launched"] = True

    def fake_from_pbm_file(source_path, **_):
        launched["source_kind"] = "file"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    def fake_from_hive_directory(source_path, **_):
        launched["source_kind"] = "hive"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    monkeypatch.setattr(module.ParquetBlockModel, "create_toy_blockmodel", staticmethod(fake_create_toy_blockmodel))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_pbm_file", staticmethod(fake_from_pbm_file))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_hive_directory", staticmethod(fake_from_hive_directory))

    module.main()

    assert launched["launched"] is True
    assert launched["kwargs"]["port"] == 8080
    assert launched["kwargs"]["host"] == "0.0.0.0"
    assert launched["source_kind"] == "file"
    assert launched["source_path"].name == "example_blocks_constructor.pbm"
    assert launched["entrypoint_kwargs"] == {
        "app_name": "Demo App",
        "scalar": "density",
        "threshold_value": 2.6,
        "data_filter_1_attribute": "density",
        "data_filter_1_min": 2.4,
    }
    assert len(created) == 0


def test_trame_example_skips_launch_during_gallery_build(tmp_path, monkeypatch):
    example_path = Path(__file__).resolve().parents[2] / "examples" / "16_trame_threshold_viewer.py"
    spec = util.spec_from_file_location("trame_threshold_viewer", example_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    launched = {}
    created = []

    def fake_create_toy_blockmodel(*, filename, shape):
        created.append(Path(filename))
        return object()

    class FakeApp:
        def launch(self, **kwargs):
            launched["kwargs"] = kwargs
            launched["launched"] = True

    def fake_from_pbm_file(source_path, **_):
        launched["source_kind"] = "file"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    def fake_from_hive_directory(source_path, **_):
        launched["source_kind"] = "hive"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    monkeypatch.setattr(module.BlockModelTrameApp, "from_pbm_file", staticmethod(fake_from_pbm_file))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_hive_directory", staticmethod(fake_from_hive_directory))
    monkeypatch.setattr(module.ParquetBlockModel, "create_toy_blockmodel", staticmethod(fake_create_toy_blockmodel))
    monkeypatch.setattr(module, "DEMO_SOURCE_KIND", "file")
    monkeypatch.setattr(module.pv, "BUILDING_GALLERY", True, raising=False)

    module.main()

    assert "launched" not in launched
    assert launched["entrypoint_kwargs"] == {
        "app_name": "Demo App",
        "scalar": "density",
        "threshold_value": 2.6,
        "data_filter_1_attribute": "density",
        "data_filter_1_min": 2.4,
    }
    assert len(created) == 0


def test_trame_example_hive_toggle_uses_directory_source(tmp_path, monkeypatch):
    example_path = Path(__file__).resolve().parents[2] / "examples" / "16_trame_threshold_viewer.py"
    spec = util.spec_from_file_location("trame_threshold_viewer", example_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    launched: dict[str, object] = {}
    created: list[Path] = []

    def fake_create_toy_blockmodel(*, filename, shape):
        created.append(Path(filename))
        launched["shape"] = shape
        return object()

    class FakeApp:
        def launch(self, **kwargs):
            launched["kwargs"] = kwargs
            launched["launched"] = True

    def fake_from_pbm_file(source_path, **_):
        launched["source_kind"] = "file"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    def fake_from_hive_directory(source_path, **_):
        launched["source_kind"] = "hive"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    monkeypatch.setattr(module, "DEMO_SOURCE_KIND", "hive")
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(module.ParquetBlockModel, "create_toy_blockmodel", staticmethod(fake_create_toy_blockmodel))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_pbm_file", staticmethod(fake_from_pbm_file))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_hive_directory", staticmethod(fake_from_hive_directory))

    module.main()

    assert launched["launched"] is True
    assert launched["kwargs"]["port"] == 8080
    assert launched["kwargs"]["host"] == "0.0.0.0"
    assert launched["shape"] == (4, 4, 4)
    assert launched["source_kind"] == "hive"
    assert launched["source_path"] == tmp_path / "parq_blockmodel_trame_hive_demo"
    assert launched["entrypoint_kwargs"] == {
        "app_name": "Demo App",
        "scalar": "depth",
        "threshold_value": 2.1,
        "data_filter_1_attribute": "depth",
        "data_filter_1_min": 1.25,
    }
    assert len(created) == 2


def test_trame_example_uses_hive_file_for_file_mode(monkeypatch, tmp_path):
    example_path = Path(__file__).resolve().parents[2] / "examples" / "16_trame_threshold_viewer.py"
    spec = util.spec_from_file_location("trame_threshold_viewer_repo_root", example_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    launched: dict[str, object] = {}

    class FakeApp:
        def launch(self, **kwargs):
            launched["kwargs"] = kwargs
            launched["launched"] = True

    def fake_from_pbm_file(source_path, **_):
        launched["source_kind"] = "file"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    def fake_from_hive_directory(source_path, **_):
        launched["source_kind"] = "hive"
        launched["source_path"] = Path(source_path)
        launched["entrypoint_kwargs"] = _
        return FakeApp()

    monkeypatch.setattr(module.Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(module, "DEMO_SOURCE_KIND", "file")
    monkeypatch.setattr(module.BlockModelTrameApp, "from_pbm_file", staticmethod(fake_from_pbm_file))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_hive_directory", staticmethod(fake_from_hive_directory))

    module.main()

    assert launched["source_kind"] == "file"
    assert launched["source_path"].name == "example_blocks_constructor.pbm"
    assert launched["entrypoint_kwargs"] == {
        "app_name": "Demo App",
        "scalar": "density",
        "threshold_value": 2.6,
        "data_filter_1_min": 2.4,
        "data_filter_1_attribute": "density",
    }


def test_trame_hive_directory_starts_without_blockmodel(tmp_path, monkeypatch):
    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            pass
            super().__init__(*args, **kwargs)

        def clear(self):
            return None

        def add_mesh(self, *args, **kwargs):
            return None

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)
    monkeypatch.setattr(
        "parq_blockmodel.visualization.trame_app.HivePbmCatalog.discover",
        staticmethod(lambda root_path: SimpleNamespace(assets=[])),
    )

    app = BlockModelTrameApp.from_hive_directory(tmp_path)

    assert app.blockmodel is None
    assert app._initial_scalar == ""


def test_trame_reset_model_view_clears_loaded_state(tmp_path, monkeypatch):
    parquet_path = tmp_path / "reset_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.cleared = False
            super().__init__(*args, **kwargs)

        def clear(self):
            self.cleared = True

        def add_mesh(self, *args, **kwargs):
            return None

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._server = SimpleNamespace(state=SimpleNamespace())
    app._reset_model_view()

    assert app.blockmodel is None
    assert app.state is None
    assert app.threshold is not None
    assert app.threshold.value == 0.0
    assert app.filter_enabled is False
    assert app.plotter.cleared is True
    assert app._server.state.active_attribute == ""
    assert app._server.state.model_name == ""


def test_trame_reset_model_view_preserves_filter_presets_for_hive_transition(tmp_path, monkeypatch):
    parquet_path = tmp_path / "hive_transition_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))
    attr = pbm.available_attributes[0]

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.cleared = False
            super().__init__(*args, **kwargs)

        def clear(self):
            self.cleared = True

        def add_mesh(self, *args, **kwargs):
            return None

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(
        blockmodel=None,
        scalar=attr,
        data_filter_1_attribute=attr,
        data_filter_1_min=0.0,
        show_edges=False,
    )
    app._server = SimpleNamespace(state=SimpleNamespace())
    app._refresh_filter_options()
    app._apply_startup_filter_presets_without_data()

    app._reset_model_view(preserve_presets=True)
    app.load_blockmodel(pbm, preferred_scalar=attr)

    assert app._data_filters[0].attribute == attr


def test_trame_refresh_plot_uses_default_camera_on_first_render(tmp_path, monkeypatch):
    parquet_path = tmp_path / "camera_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.set_directional_view_calls = 0
            super().__init__(*args, **kwargs)
            self.camera_position = ("preset",)

        def clear(self):
            return None

        def add_mesh(self, *args, **kwargs):
            return None

        def set_directional_view(self, direction=None, **kwargs):
            self.set_directional_view_calls += 1
            self.camera_position = ("directional_view", self.set_directional_view_calls)

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._refresh_plot(preserve_camera=True)
    assert app.plotter.set_directional_view_calls == 1
    assert app.plotter.camera_position == ("directional_view", 1)

    app.plotter.camera_position = ("custom", 42)
    app._refresh_plot(preserve_camera=True)
    assert app.plotter.set_directional_view_calls == 1
    assert app.plotter.camera_position == ("custom", 42)


def test_trame_launch_requests_vue2_client_type(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.ren_win = object()
            super().__init__(*args, **kwargs)

        def clear(self):
            return None

        def add_mesh(self, *args, **kwargs):
            return None

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class ToolbarContext(DummyContext):
        def __init__(self):
            self.color = None
            self.dense = None
            self.dark = None

        def __setattr__(self, key, value):
            object.__setattr__(self, key, value)
            if key in {"color", "dense", "dark"}:
                calls[f"toolbar_{key}"] = value

    class FakeLayout:
        def __init__(self, server, *args, **kwargs):
            self.title = SimpleNamespace(set_text=lambda text: calls.setdefault("layout_title", text))
            self.toolbar = ToolbarContext()
            self.drawer = DummyContext()
            self.content = DummyContext()
            self.icon = SimpleNamespace(click=None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = {}

    class FakeState:
        def __init__(self):
            self._callbacks = {}

        def change(self, key):
            def register(callback):
                self._callbacks.setdefault(key, []).append(callback)
                return callback

            return register

        def __setattr__(self, key, value):
            object.__setattr__(self, key, value)

        def ready(self):
            self._ready_called = True

    fake_state = FakeState()

    fake_trame = ModuleType("trame")
    fake_trame_app = ModuleType("trame.app")
    fake_trame_ui = ModuleType("trame.ui")
    fake_trame_ui_vuetify = ModuleType("trame.ui.vuetify")
    fake_trame_widgets = ModuleType("trame.widgets")

    def fake_get_server(name, client_type=None):
        calls["name"] = name
        calls["client_type"] = client_type
        return SimpleNamespace(
            state=fake_state,
            controller=SimpleNamespace(),
            start=lambda **kwargs: calls.update({"start_kwargs": kwargs}),
        )

    fake_trame_app.get_server = fake_get_server
    fake_trame_ui_vuetify.SinglePageWithDrawerLayout = FakeLayout
    fake_trame_ui_vuetify.VAppLayout = FakeLayout
    fake_trame_ui_vuetify.SinglePageLayout = FakeLayout
    fake_trame_widgets.vtk = SimpleNamespace(
        VtkRemoteView=lambda ren_win, **kwargs: (
            calls.update({"remote_view_kwargs": kwargs}) or SimpleNamespace(update=lambda: None)
        )
    )
    mouse_bindings = []

    class FakeMouseTrap:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def bind(self, keys, event_name, stop_propagation=False, listen_to=None):
            mouse_bindings.append((keys, event_name, stop_propagation, listen_to))

    def make_widget(*args, **kwargs):
        return DummyContext()

    def make_img(*args, **kwargs):
        calls["logo_src"] = kwargs.get("src")
        calls["logo_max_width"] = kwargs.get("max_width")
        return DummyContext()

    def make_toolbar_title(*args, **kwargs):
        if args:
            calls["toolbar_title"] = args[0]
        return DummyContext()

    fake_trame_widgets.vuetify = SimpleNamespace(
        VSelect=make_widget,
        VSlider=make_widget,
        VRangeSlider=make_widget,
        VDialog=make_widget,
        VBtn=make_widget,
        VAppLayout=FakeLayout,
        VAppBar=make_widget,
        VAppBarNavIcon=make_widget,
        VToolbarTitle=make_toolbar_title,
        VSpacer=make_widget,
        VChip=make_widget,
        VNavigationDrawer=make_widget,
        VSheet=make_widget,
        VImg=make_img,
        VCard=make_widget,
        VCardText=make_widget,
        VCardTitle=make_widget,
        VExpansionPanels=make_widget,
        VExpansionPanel=make_widget,
        VExpansionPanelHeader=make_widget,
        VExpansionPanelContent=make_widget,
        VMain=make_widget,
        VContainer=make_widget,
        VIcon=make_widget,
        VTextField=make_widget,
        VCheckbox=make_widget,
    )
    fake_trame_widgets.trame = SimpleNamespace(MouseTrap=FakeMouseTrap)

    fake_trame.app = fake_trame_app
    fake_trame.ui = fake_trame_ui
    fake_trame.widgets = fake_trame_widgets
    fake_trame_ui.vuetify = fake_trame_ui_vuetify

    monkeypatch.setitem(sys.modules, "trame", fake_trame)
    monkeypatch.setitem(sys.modules, "trame.app", fake_trame_app)
    monkeypatch.setitem(sys.modules, "trame.ui", fake_trame_ui)
    monkeypatch.setitem(sys.modules, "trame.ui.vuetify", fake_trame_ui_vuetify)
    monkeypatch.setitem(sys.modules, "trame.widgets", fake_trame_widgets)
    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)
    monkeypatch.setattr(
        "parq_blockmodel.visualization.trame_app.HivePbmCatalog.discover",
        staticmethod(
            lambda root_path: SimpleNamespace(
                level_keys=(),
                level_options=lambda key, selections=None: [],
                pbm_name_options=lambda selections=None: [],
                select_asset=lambda selections, name: (_ for _ in ()).throw(LookupError("no assets")),
            )
        ),
    )

    from parq_blockmodel.visualization.trame_app import BlockModelTrameApp

    app = BlockModelTrameApp(
        pbm,
        scalar=pbm.available_attributes[0],
        z_up_lock=True,
        z_up_hotkey="z",
        app_name="Custom Viewer",
    )
    app.launch(port=3080, host="0.0.0.0")

    assert calls["client_type"] == "vue2"
    assert calls["start_kwargs"]["open_browser"] is True
    assert calls["start_kwargs"]["show_connection_info"] is True
    assert calls["start_kwargs"]["port"] == 3080
    assert calls["start_kwargs"]["host"] == "0.0.0.0"
    assert calls["layout_title"] == ""
    assert calls["toolbar_title"] == "Custom Viewer"
    assert calls["logo_src"].startswith("data:image/svg+xml;charset=utf-8,")
    assert calls["logo_max_width"] == 50
    assert calls["toolbar_color"] == "grey lighten-3"
    assert calls["toolbar_dark"] is False
    assert ("z", "ZUpKeyDown", False, "keydown") in mouse_bindings
    assert ("z", "ZUpKeyUp", False, "keyup") in mouse_bindings
    assert callable(calls["remote_view_kwargs"]["KeyDown"])
    assert callable(calls["remote_view_kwargs"]["KeyUp"])
    assert fake_state.source_mode == "file"
    assert fake_state.source_path_input == str(pbm.blockmodel_path)
    assert fake_state.model_path == str(pbm.blockmodel_path)
    assert fake_state.active_attribute == pbm.available_attributes[0]
    assert fake_state.attribute_options == list(pbm.available_attributes)


def test_trame_plot_engine_forwards_launch_host(tmp_path, monkeypatch):
    parquet_path = tmp_path / "engine_launch_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    calls: dict[str, object] = {}

    def fake_launch(self, **kwargs):
        calls["kwargs"] = kwargs
        return "launched"

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.BlockModelTrameApp.launch", fake_launch)

    engine = TrameBlockModelPlotEngine(launch_on_plot=True, port=3080, host="0.0.0.0")
    result = engine.plot(pbm, scalar=pbm.available_attributes[0])

    assert result == "launched"
    assert calls["kwargs"]["port"] == 3080
    assert calls["kwargs"]["host"] == "0.0.0.0"


def test_trame_hive_launch_keeps_preset_controls_without_loading_asset(tmp_path, monkeypatch):
    class FakePlotter(BaseFakePlotter):
        def __init__(self, *args, **kwargs):
            self.add_mesh_calls = 0
            super().__init__(*args, **kwargs)
            self.ren_win = object()

        def clear(self):
            return None

        def add_mesh(self, *args, **kwargs):
            self.add_mesh_calls += 1

        def view_isometric(self):
            return None

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    class FakeState:
        def __init__(self):
            self._callbacks = {}

        def change(self, key):
            def register(callback):
                self._callbacks.setdefault(key, []).append(callback)
                return callback

            return register

        def __setattr__(self, key, value):
            object.__setattr__(self, key, value)

        def ready(self):
            self._ready_called = True

    fake_state = FakeState()
    calls = {}

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyToolbar(DummyContext):
        def __init__(self):
            self.color = None
            self.dense = None
            self.dark = None

    class FakeLayout:
        def __init__(self, *args, **kwargs):
            self.toolbar = DummyToolbar()
            self.title = SimpleNamespace(set_text=lambda *_: None)
            self.drawer = DummyContext()
            self.content = DummyContext()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_trame = ModuleType("trame")
    fake_trame_app = ModuleType("trame.app")
    fake_trame_ui = ModuleType("trame.ui")
    fake_trame_ui_vuetify = ModuleType("trame.ui.vuetify")
    fake_trame_widgets = ModuleType("trame.widgets")

    def fake_get_server(name, client_type=None):
        calls["name"] = name
        calls["client_type"] = client_type
        return SimpleNamespace(
            state=fake_state,
            controller=SimpleNamespace(),
            start=lambda **kwargs: calls.update({"start_kwargs": kwargs}),
        )

    fake_trame_app.get_server = fake_get_server
    fake_trame_ui_vuetify.SinglePageWithDrawerLayout = FakeLayout
    fake_trame_widgets.vtk = SimpleNamespace(
        VtkRemoteView=lambda ren_win, **kwargs: SimpleNamespace(update=lambda: None)
    )
    fake_trame_widgets.vuetify = SimpleNamespace(
        VSelect=lambda *args, **kwargs: DummyContext(),
        VSlider=lambda *args, **kwargs: DummyContext(),
        VRangeSlider=lambda *args, **kwargs: DummyContext(),
        VDialog=lambda *args, **kwargs: DummyContext(),
        VBtn=lambda *args, **kwargs: DummyContext(),
        VAppBar=lambda *args, **kwargs: DummyContext(),
        VAppBarNavIcon=lambda *args, **kwargs: DummyContext(),
        VToolbarTitle=lambda *args, **kwargs: DummyContext(),
        VSpacer=lambda *args, **kwargs: DummyContext(),
        VChip=lambda *args, **kwargs: DummyContext(),
        VNavigationDrawer=lambda *args, **kwargs: DummyContext(),
        VSheet=lambda *args, **kwargs: DummyContext(),
        VImg=lambda *args, **kwargs: DummyContext(),
        VCard=lambda *args, **kwargs: DummyContext(),
        VCardText=lambda *args, **kwargs: DummyContext(),
        VCardTitle=lambda *args, **kwargs: DummyContext(),
        VExpansionPanels=lambda *args, **kwargs: DummyContext(),
        VExpansionPanel=lambda *args, **kwargs: DummyContext(),
        VExpansionPanelHeader=lambda *args, **kwargs: DummyContext(),
        VExpansionPanelContent=lambda *args, **kwargs: DummyContext(),
        VMain=lambda *args, **kwargs: DummyContext(),
        VContainer=lambda *args, **kwargs: DummyContext(),
        VIcon=lambda *args, **kwargs: DummyContext(),
        VTextField=lambda *args, **kwargs: DummyContext(),
        VCheckbox=lambda *args, **kwargs: DummyContext(),
    )
    fake_trame_widgets.trame = SimpleNamespace(MouseTrap=lambda **kwargs: SimpleNamespace(bind=lambda *a, **k: None))
    fake_trame.app = fake_trame_app
    fake_trame.ui = fake_trame_ui
    fake_trame.widgets = fake_trame_widgets
    fake_trame_ui.vuetify = fake_trame_ui_vuetify

    monkeypatch.setitem(sys.modules, "trame", fake_trame)
    monkeypatch.setitem(sys.modules, "trame.app", fake_trame_app)
    monkeypatch.setitem(sys.modules, "trame.ui", fake_trame_ui)
    monkeypatch.setitem(sys.modules, "trame.ui.vuetify", fake_trame_ui_vuetify)
    monkeypatch.setitem(sys.modules, "trame.widgets", fake_trame_widgets)
    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.CustomPlotter", FakePlotter)
    monkeypatch.setattr(
        "parq_blockmodel.visualization.trame_app.HivePbmCatalog.discover",
        staticmethod(
            lambda root_path: SimpleNamespace(
                level_keys=(),
                level_options=lambda key, selections=None: [],
                pbm_name_options=lambda selections=None: [],
                select_asset=lambda selections, name: (_ for _ in ()).throw(LookupError("no assets")),
            )
        ),
    )

    from parq_blockmodel.visualization.trame_app import BlockModelTrameApp

    hive_root = tmp_path / "hive"
    hive_root.mkdir()
    app = BlockModelTrameApp.from_hive_directory(
        hive_root,
        scalar="grade",
        threshold_value=2.1,
        data_filter_1_attribute="grade",
        data_filter_1_min=1.25,
        app_name="Custom Viewer",
    )
    app.launch(port=3080, host="0.0.0.0")

    assert fake_state.source_mode == "hive"
    assert fake_state.source_path_input == str(hive_root.resolve())
    assert fake_state.model_path == ""
    assert fake_state.active_attribute == "grade"
    assert fake_state.attribute_options == ["grade"]
    assert fake_state.threshold == 2.1
    assert fake_state.threshold_display == "2.1"
    assert fake_state.data_filter_1_attribute == "grade"
    assert fake_state.data_filter_attribute_options == ["grade"]
    assert fake_state.data_filter_1_range[0] == 1.25
    assert fake_state.data_filter_1_range[1] == 1.25
    assert fake_state.data_filter_1_summary.startswith("grade:")
    active_attribute_callbacks = fake_state._callbacks.get("active_attribute", [])
    assert len(active_attribute_callbacks) == 1
    active_attribute_callbacks[0](active_attribute="grade")
    assert app.blockmodel is None
    assert fake_state.active_attribute == "grade"
    filter_attribute_callbacks = fake_state._callbacks.get("data_filter_1_attribute", [])
    assert len(filter_attribute_callbacks) == 1
    filter_attribute_callbacks[0](data_filter_1_attribute="grade")
    assert app.blockmodel is None
    assert fake_state.data_filter_1_attribute == "grade"
