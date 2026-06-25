"""IO module for parq_blockmodel: data ingestion and canonical writing.

This module provides utilities for reading data from various sources
(Parquet, DataFrame, geometry) and writing it into canonical .pbm files
with embedded geometry metadata.

Classes
-------
IngestWriter
    Stateful writer for streaming Parquet data into canonical .pbm format.

Functions
---------
validate_geometry
    Validate centroid columns against a RegularGeometry.
validate_xyz_parquet
    Validate xyz-defined Parquet input and return inferred geometry.
build_world_id_encoding_from_xyz
    Build default world_id encoding metadata from xyz ranges.
assert_block_id_xyz_consistent
    Validate block_id consistency with xyz coordinates.
"""

from parq_blockmodel.io.ingest_writer import IngestWriter
from parq_blockmodel.io import ingest_utils

__all__ = [
    "IngestWriter",
    "ingest_utils",
]
