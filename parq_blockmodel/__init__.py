import os

os.environ["YDATA_SUPPRESS_BANNER"] = "1"

from importlib import metadata
from .blockmodel import ParquetBlockModel
from .geometry import RegularGeometry
from .utils.demo_block_model import create_demo_blockmodel

try:
    __version__ = metadata.version('parq_blockmodel')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass

__all__ = ["ParquetBlockModel", "create_demo_blockmodel"]
