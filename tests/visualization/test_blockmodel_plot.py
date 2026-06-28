import sys
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.visualization import BlockModelTrameApp, TrameBlockModelPlotEngine
from parq_blockmodel.visualization.blockmodel_plot import _register_z_up_rotation_lock, _set_z_up_hotkey_state


class FakeEngine:
    def __init__(self):
        self.calls = []

    def plot(self, blockmodel, **kwargs):
        self.calls.append((blockmodel, kwargs))
        return "engine-result"


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


def test_register_z_up_rotation_lock_uses_turntable_style_while_held(monkeypatch):
    callbacks = {}
    style_changes = []

    class FakeStyle:
        def __init__(self, name):
            self.name = name

    class FakeInteractor:
        def __init__(self):
            self.current_style = FakeStyle("trackball")

        def add_observer(self, event_name, callback):
            callbacks[event_name] = callback

        def GetInteractorStyle(self):
            return self.current_style

        def SetInteractorStyle(self, style):
            style_changes.append(style)
            self.current_style = style

    class FakeCamera:
        def __init__(self):
            self.view_up = None
            self.orthogonalized = 0

        def SetViewUp(self, x, y, z):
            self.view_up = (x, y, z)

        def OrthogonalizeViewUp(self):
            self.orthogonalized += 1

    class FakePlotter:
        def __init__(self):
            self.iren = FakeInteractor()
            self.camera = FakeCamera()
            self.render_calls = 0
            self.renderer = object()

        def render(self):
            self.render_calls += 1

    class FakeEventSource:
        def __init__(self, key):
            self._key = key

        def GetKeySym(self):
            return self._key

    class FakeTerrainStyle:
        def __init__(self):
            self.default_renderer = None

        def SetDefaultRenderer(self, renderer):
            self.default_renderer = renderer

    monkeypatch.setattr(
        "parq_blockmodel.visualization.blockmodel_plot.pv._vtk.vtkInteractorStyleTerrain",
        FakeTerrainStyle,
    )

    plotter = FakePlotter()
    original_style = plotter.iren.GetInteractorStyle()
    _register_z_up_rotation_lock(plotter, hotkey="z")

    callbacks["KeyPressEvent"](FakeEventSource("z"), None)
    callbacks["InteractionEvent"](None, None)
    callbacks["KeyReleaseEvent"](FakeEventSource("z"), None)

    assert plotter.camera.view_up == (0, 0, 1)
    assert plotter.camera.orthogonalized >= 1
    assert plotter.render_calls >= 2
    assert plotter._pbm_z_up_hotkey_down is False
    assert plotter._pbm_z_up_turntable_active is False
    assert isinstance(style_changes[0], FakeTerrainStyle)
    assert style_changes[-1] is original_style


def test_set_z_up_hotkey_state_toggles_registered_callbacks():
    calls = []

    class FakePlotter:
        pass

    plotter = FakePlotter()
    plotter._pbm_z_up_callbacks_registered = True
    plotter._pbm_z_up_enable_turntable_mode = lambda: calls.append("enable")
    plotter._pbm_z_up_disable_turntable_mode = lambda: calls.append("disable")

    _set_z_up_hotkey_state(plotter, True)
    _set_z_up_hotkey_state(plotter, False)

    assert plotter._pbm_z_up_hotkey_down is False
    assert calls == ["enable", "disable"]


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

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.iren = FakeInteractor()
            self.camera = FakeCamera()
            self.ren_win = object()
            self.actors = {}

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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)
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

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.actors = {}
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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

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


def test_trame_app_renders_categorical_attributes(tmp_path, monkeypatch):
    parquet_path = tmp_path / "categorical_source.parquet"
    ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(3, 3, 3))
    df = pd.read_parquet(parquet_path)
    categories = pd.Categorical(["A", "B", "C"] * 9)
    df["rock_type"] = categories[: len(df)]
    df.to_parquet(parquet_path, index=False)
    pbm = ParquetBlockModel.from_parquet(parquet_path)

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.actors = {}
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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar="rock_type", show_edges=False)
    app._load_plot_state("rock_type")
    app._refresh_plot()

    assert app.state is not None
    assert app.state.scalar_is_categorical is True
    assert app.plotter.last_mesh_n_cells > 0


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
            launched["launched"] = True

    def fake_from_source_path(source_path):
        launched["source_path"] = Path(source_path)
        return FakeApp()

    monkeypatch.setattr(module.Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(module.ParquetBlockModel, "create_toy_blockmodel", staticmethod(fake_create_toy_blockmodel))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_source_path", staticmethod(fake_from_source_path))

    module.main()

    assert launched["launched"] is True
    assert launched["shape"] == (4, 4, 4)
    assert launched["source_path"].is_dir() or launched["source_path"].suffix == ".pbm"
    assert len(created) == 2


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
            launched["launched"] = True

    def fake_from_source_path(source_path):
        launched["source_path"] = Path(source_path)
        return FakeApp()

    monkeypatch.setattr(module.Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(module.ParquetBlockModel, "create_toy_blockmodel", staticmethod(fake_create_toy_blockmodel))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_source_path", staticmethod(fake_from_source_path))
    monkeypatch.setattr(module.pv, "BUILDING_GALLERY", True, raising=False)

    module.main()

    assert "launched" not in launched
    assert len(created) == 2


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
            launched["launched"] = True

    def fake_from_source_path(source_path):
        launched["source_path"] = Path(source_path)
        return FakeApp()

    monkeypatch.setattr(module, "DEMO_SOURCE_KIND", "hive")
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(module.ParquetBlockModel, "create_toy_blockmodel", staticmethod(fake_create_toy_blockmodel))
    monkeypatch.setattr(module.BlockModelTrameApp, "from_source_path", staticmethod(fake_from_source_path))

    module.main()

    assert launched["launched"] is True
    assert launched["shape"] == (4, 4, 4)
    assert launched["source_path"] == tmp_path / "parq_blockmodel_trame_hive_demo"
    assert len(created) == 2


def test_trame_hive_directory_starts_without_blockmodel(tmp_path, monkeypatch):
    class FakePlotter:
        def __init__(self, *args, **kwargs):
            pass

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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)
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

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.cleared = False

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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

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


def test_trame_refresh_plot_uses_default_camera_on_first_render(tmp_path, monkeypatch):
    parquet_path = tmp_path / "camera_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.view_isometric_calls = 0
            self.camera_position = ("preset",)

        def clear(self):
            return None

        def add_mesh(self, *args, **kwargs):
            return None

        def view_isometric(self):
            self.view_isometric_calls += 1
            self.camera_position = ("isometric", self.view_isometric_calls)

        def reset_camera_clipping_range(self):
            return None

        def add_axes(self):
            return None

        def render(self):
            return None

        def show(self, *args, **kwargs):
            return None

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], show_edges=False)
    app._load_plot_state(app._initial_scalar)
    app._refresh_plot(preserve_camera=True)
    assert app.plotter.view_isometric_calls == 1
    assert app.plotter.camera_position == ("isometric", 1)

    app.plotter.camera_position = ("custom", 42)
    app._refresh_plot(preserve_camera=True)
    assert app.plotter.view_isometric_calls == 1
    assert app.plotter.camera_position == ("custom", 42)


def test_trame_launch_requests_vue2_client_type(tmp_path, monkeypatch):
    parquet_path = tmp_path / "trame_source.parquet"
    pbm = ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.ren_win = object()

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

    fake_trame_widgets.vuetify = SimpleNamespace(
        VSelect=make_widget,
        VSlider=make_widget,
        VBtn=make_widget,
        VAppLayout=FakeLayout,
        VAppBar=make_widget,
        VAppBarNavIcon=make_widget,
        VToolbarTitle=make_widget,
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
    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

    from parq_blockmodel.visualization.trame_app import BlockModelTrameApp

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0], z_up_lock=True, z_up_hotkey="z")
    app.launch()

    assert calls["client_type"] == "vue2"
    assert calls["start_kwargs"]["open_browser"] is True
    assert calls["start_kwargs"]["show_connection_info"] is True
    assert calls["layout_title"] == ""
    assert calls["logo_src"].startswith("data:image/svg+xml;charset=utf-8,")
    assert calls["logo_max_width"] == 50
    assert calls["toolbar_color"] == "grey lighten-3"
    assert calls["toolbar_dark"] is False
    assert ("z", "ZUpKeyDown", False, "keydown") in mouse_bindings
    assert ("z", "ZUpKeyUp", False, "keyup") in mouse_bindings
    assert calls["remote_view_kwargs"]["interactor_events"][1] == ["KeyDown", "KeyPress", "KeyUp"]
    assert callable(calls["remote_view_kwargs"]["KeyDown"])
    assert callable(calls["remote_view_kwargs"]["KeyUp"])
