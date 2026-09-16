"""Reporting utilities for block-model profiling."""
import os
os.environ["YDATA_SUPPRESS_BANNER"] = "1"

from .report import BlockModelReport

__all__ = ["BlockModelReport"]

