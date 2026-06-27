"""Trame viewer example with two startup patterns.

This example shows both ways to start the same Trame app:

1. **File pattern**: point to a single ``.pbm`` file.
2. **Hive pattern**: point to a hive-style directory and use the in-app
   selector to drill down by levels and ``pbm.name``.

Use ``DEMO_SOURCE_KIND`` to switch between these patterns in one place.

This example is gallery-safe: it stays in Sphinx-Gallery, but skips launching
the live server while docs are being built.
"""

import logging
import tempfile
from pathlib import Path

import pyvista as pv

from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.visualization import BlockModelTrameApp

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# sphinx_gallery_thumbnail_path = "../docs/_static/branding/parq-blockmodel-gallery-thumbnail.svg"
DEMO_SOURCE_KIND = "file"  # "file" or "hive"


def _seed_temporary_hive_demo() -> Path:
    hive_root = Path(tempfile.gettempdir()) / "parq_blockmodel_trame_hive_demo"
    asset_specs = [
        ("site=alpha", "scenario=base", "alpha_base_orebody.parquet", (4, 4, 4)),
        ("site=beta", "scenario=optimised", "beta_optimised_orebody.parquet", (4, 4, 4)),
    ]
    for site_level, scenario_level, filename, shape in asset_specs:
        parquet_path = hive_root / site_level / scenario_level / filename
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        ParquetBlockModel.create_toy_blockmodel(filename=parquet_path, shape=shape)
    return hive_root


def _resolve_source_path() -> Path:
    local_file = Path.cwd() / "example_blocks_constructor.pbm"
    local_hive = Path.cwd() / "example_hive_assets"
    if DEMO_SOURCE_KIND == "hive":
        if local_hive.exists():
            return local_hive
        return _seed_temporary_hive_demo()

    if local_file.exists():
        return local_file
    hive_root = _seed_temporary_hive_demo()
    return hive_root / "site=alpha" / "scenario=base" / "alpha_base_orebody.pbm"


def main() -> None:
    source_path = _resolve_source_path()
    logger.warning("Launching Trame demo from %s", source_path)
    app = BlockModelTrameApp.from_source_path(source_path)

    if getattr(pv, "BUILDING_GALLERY", False):
        logger.info("Skipping live Trame launch while building the gallery.")
        return

    app.launch()


if __name__ == "__main__":
    main()
