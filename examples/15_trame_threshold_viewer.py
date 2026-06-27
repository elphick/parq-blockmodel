"""Standalone Trame-friendly threshold viewer.

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

# sphinx_gallery_thumbnail_path = "../docs/_static/branding/parq-blockmodel.svg"


def main() -> None:
    source = Path.cwd() / "example_blocks_constructor.pbm"
    if not source.exists():
        demo_dir = Path(tempfile.gettempdir()) / "parq_blockmodel_trame_demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        demo_source = demo_dir / "trame_threshold_demo.parquet"
        logger.warning(
            "example_blocks_constructor.pbm is missing; creating a demo PBM at %s",
            demo_dir,
        )
        pbm = ParquetBlockModel.create_toy_blockmodel(filename=demo_source, shape=(4, 4, 4))
    else:
        pbm = ParquetBlockModel(blockmodel_path=source)

    app = BlockModelTrameApp(pbm, scalar=pbm.available_attributes[0])

    if getattr(pv, "BUILDING_GALLERY", False):
        logger.info("Skipping live Trame launch while building the gallery.")
        return

    app.launch()


if __name__ == "__main__":
    main()
