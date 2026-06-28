<h1 style="display: inline-flex; align-items: center; gap: 0.4rem; margin: 0;">
  <img src="https://raw.githubusercontent.com/elphick/parq-blockmodel/main/docs/_static/branding/parq-blockmodel.svg" alt="parq-blockmodel logo" width="72" style="display: block; margin-top: 20px;" />
  <span>parq-blockmodel</span>
</h1>

[![Run Tests](https://github.com/Elphick/parq-blockmodel/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-blockmodel/actions/workflows/build_and_test.yml)
[![PyPI](https://img.shields.io/pypi/v/parq-blockmodel?logo=python&logoColor=white)](https://pypi.org/project/parq-blockmodel/)
![Coverage](https://raw.githubusercontent.com/elphick/parq-blockmodel/main/docs/_static/badges/coverage.svg)
[![Python Versions](https://img.shields.io/pypi/pyversions/parq-blockmodel)](https://pypi.org/project/parq-blockmodel/)
[![License](https://img.shields.io/github/license/Elphick/parq-blockmodel?cacheSeconds=86400)](https://pypi.org/project/parq-blockmodel/)
[![Publish Docs](https://github.com/Elphick/parq-blockmodel/actions/workflows/docs_to_gh_pages.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-blockmodel/actions/workflows/docs_to_gh_pages.yml)
[![Open Issues](https://img.shields.io/github/issues/Elphick/parq-blockmodel?cacheSeconds=86400)](https://github.com/Elphick/parq-blockmodel/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Elphick/parq-blockmodel?cacheSeconds=86400)](https://github.com/Elphick/parq-blockmodel/pulls)

## Overview
A Python package for efficient storage, manipulation, and analysis of mining block models using Parquet files. 
parq-blockmodel provides tools for reading, writing, indexing, and transforming large-scale block model datasets, 
leveraging the performance of Apache Arrow and Parquet for scalable geoscience data workflows.

## Installation

Install the base package from PyPI:

```bash
pip install parq-blockmodel
```

Install the optional schema validation support when you want to validate block
model attributes with Pandera schemas or load schema definitions from YAML:

```bash
pip install "parq-blockmodel[schema]"
```

## Schema validation

`ParquetBlockModel` accepts an optional `schema=` argument on its main
constructors. You can pass either a Pandera `DataFrameSchema` object or a path
to a YAML schema file, then validate the resulting model in chunks:

```python
from pathlib import Path

from parq_blockmodel import ParquetBlockModel

pbm = ParquetBlockModel.from_parquet(
    Path("path/to/blockmodel.parquet"),
    schema=Path("schemas/blockmodel.schema.yaml"),
)

pbm.validate()
pbm.validate(sample_chunks=1)  # quick spot-check for large models
```

See the [User Guide](https://parq-blockmodel.readthedocs.io/en/stable/user_guide/07_calculated_attributes.html)
for detailed documentation on calculated attributes, including custom lookups and functions.

## Visualization

The block-model plotting path now delegates through `parq_blockmodel.visualization`, which keeps the
rendering logic isolated from `ParquetBlockModel` itself.

```python
from parq_blockmodel import ParquetBlockModel
from parq_blockmodel.visualization import BlockModelTrameApp, TrameBlockModelPlotEngine

pbm = ParquetBlockModel.from_parquet("orebody.parquet")
plotter = pbm.plot(scalar="grade", z_up_lock=True, z_up_hotkey="z")

trame_app_from_plot = pbm.plot(
    scalar="grade",
    engine=TrameBlockModelPlotEngine(),
    z_up_lock=True,
    z_up_hotkey="z",
)

app = BlockModelTrameApp(pbm, scalar="grade", z_up_lock=True, z_up_hotkey="z")
```

With ``z_up_lock=True``, hold ``z`` for turntable-style orbit (yaw/pitch, no roll) with camera up aligned to +Z.

## Geometry operations

`parq-blockmodel` supports three geometry flagging workflows:

* [Polygon flagging](https://parq-blockmodel.readthedocs.io/en/stable/user_guide/08_polygon_flagging.html) for 2D XY regions.
* [Surface encoding](https://parq-blockmodel.readthedocs.io/en/stable/user_guide/09_surface_encoding.html) for 2.5D elevation surfaces (`z = f(x, y)`).
* [Solid flagging](https://parq-blockmodel.readthedocs.io/en/stable/user_guide/10_solid_flagging.html) for closed 3D volumes.