"""
Compression Framework
=====================

This example shows the intended lifecycle:

* write a model with the fast/default compression path
* inspect the persisted compression metadata
* promote the same model in place with ``compress()``
"""

from pathlib import Path
import json
import tempfile

import pyarrow.parquet as pq

from parq_blockmodel import ParquetBlockModel

# sphinx_gallery_thumbnail_path = "../docs/_static/branding/parq-blockmodel-gallery-thumbnail.svg"

# %%
# Create a small demo block model.

temp_dir = Path(tempfile.gettempdir()) / "compression_framework_example"
temp_dir.mkdir(parents=True, exist_ok=True)

pbm = ParquetBlockModel.create_demo_block_model(temp_dir / "compression_demo.parquet")
pbm

# %%
# Inspect the active-write compression metadata.

before = pq.read_metadata(pbm.blockmodel_path).metadata or {}
json.loads(before[b"parq-blockmodel"].decode("utf-8"))

# %%
# Rewrite the model in place using archive compression.

pbm.compress(level=7)

# %%
# Check the archived policy after the rewrite.

after = pq.read_metadata(pbm.blockmodel_path).metadata or {}
json.loads(after[b"parq-blockmodel"].decode("utf-8"))
