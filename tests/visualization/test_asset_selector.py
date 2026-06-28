from pathlib import Path

import pytest

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.visualization import BlockModelTrameApp, HivePbmCatalog


def _create_demo_pbm(root: Path, relative_parquet_path: str) -> ParquetBlockModel:
    relative_parquet_path = relative_parquet_path.replace("\\", "/")
    parquet_path = root.joinpath(*Path(relative_parquet_path).parts)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    return ParquetBlockModel.create_demo_block_model(filename=parquet_path, shape=(2, 2, 2))


def test_hive_catalog_discovers_levels_and_name_options(tmp_path):
    root = tmp_path / "assets"
    _create_demo_pbm(root, "site=alpha\\period=base\\orebody_a.parquet")
    _create_demo_pbm(root, "site=alpha\\period=opt\\orebody_b.parquet")
    _create_demo_pbm(root, "site=beta\\period=base\\orebody_c.parquet")

    catalog = HivePbmCatalog.discover(root)

    assert catalog.level_keys == ("site", "period")
    assert catalog.level_options("site") == ["alpha", "beta"]
    assert catalog.level_options("period", {"site": "alpha"}) == ["base", "opt"]
    assert catalog.pbm_name_options({"site": "alpha", "period": "base"}) == ["orebody_a"]

    selected = catalog.select_asset({"site": "alpha", "period": "opt"}, "orebody_b")
    assert selected.name == "orebody_b"
    assert selected.level_map == {"site": "alpha", "period": "opt"}


def test_hive_catalog_rejects_ambiguous_name_selection(tmp_path):
    root = tmp_path / "assets"
    _create_demo_pbm(root, "site=alpha\\period=base\\same_name.parquet")
    _create_demo_pbm(root, "site=beta\\period=base\\same_name.parquet")

    catalog = HivePbmCatalog.discover(root)

    with pytest.raises(LookupError, match="not unique"):
        catalog.select_asset({}, "same_name")


def test_trame_app_can_switch_models_from_hive_selection(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    pbm_a = _create_demo_pbm(root, "site=alpha\\period=base\\orebody_a.parquet")
    _create_demo_pbm(root, "site=alpha\\period=opt\\orebody_b.parquet")
    catalog = HivePbmCatalog.discover(root)

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.actors = {}
            self.title = None

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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

    app = BlockModelTrameApp(pbm_a, scalar=pbm_a.available_attributes[0], asset_catalog=catalog)
    app._asset_level_keys = list(catalog.level_keys)
    app._asset_level_values = {"site": "alpha", "period": "opt"}
    app._selected_asset_name = "orebody_b"
    app._load_selected_asset()

    assert app.blockmodel.name == "orebody_b"
    assert app.blockmodel.blockmodel_path.parent.name == "period=opt"


def test_trame_app_load_source_path_switches_between_file_and_hive(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    pbm_file = _create_demo_pbm(tmp_path, "single\\single_model.parquet")
    _create_demo_pbm(root, "site=alpha\\period=base\\orebody_a.parquet")
    _create_demo_pbm(root, "site=alpha\\period=opt\\orebody_b.parquet")

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.actors = {}
            self.title = None

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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

    app = BlockModelTrameApp(pbm_file, scalar=pbm_file.available_attributes[0])
    assert app.asset_catalog is None

    app.load_source_path(root)
    assert app.asset_catalog is not None
    assert app._source_mode == "hive"
    assert app.blockmodel.blockmodel_path.suffix == ".pbm"

    app.load_source_path(pbm_file.blockmodel_path)
    assert app.asset_catalog is None
    assert app._source_mode == "file"
    assert app.blockmodel.blockmodel_path == pbm_file.blockmodel_path


def test_trame_app_load_source_path_rejects_missing_path(tmp_path, monkeypatch):
    pbm_file = _create_demo_pbm(tmp_path, "single\\single_model.parquet")

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.actors = {}
            self.title = None

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

    monkeypatch.setattr("parq_blockmodel.visualization.trame_app.pv.Plotter", FakePlotter)

    app = BlockModelTrameApp(pbm_file, scalar=pbm_file.available_attributes[0])
    with pytest.raises(FileNotFoundError):
        app.load_source_path(tmp_path / "missing")


def test_trame_app_from_source_path_selects_mode(tmp_path):
    hive_root = tmp_path / "assets"
    file_pbm = _create_demo_pbm(tmp_path, "single\\single_model.parquet")
    _create_demo_pbm(hive_root, "site=alpha\\period=base\\orebody_a.parquet")

    app_from_file = BlockModelTrameApp.from_source_path(file_pbm.blockmodel_path)
    assert app_from_file.asset_catalog is None
    assert app_from_file._source_mode == "file"

    app_from_dir = BlockModelTrameApp.from_source_path(hive_root)
    assert app_from_dir.asset_catalog is not None
    assert app_from_dir._source_mode == "hive"
