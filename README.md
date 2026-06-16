# parq-blockmodel

<p align="center">
  <img src="https://raw.githubusercontent.com/elphick/parq-blockmodel/main/docs/_static/branding/parq-blockmodel.svg" alt="df-eval icon" width="120" />
</p>

[![Run Tests](https://github.com/Elphick/parq-blockmodel/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-blockmodel/actions/workflows/build_and_test.yml)
[![PyPI](https://img.shields.io/pypi/v/parq-blockmodel.svg?logo=python&logoColor=white)](https://pypi.org/project/parq-blockmodel/)
![Coverage](https://raw.githubusercontent.com/elphick/parq-blockmodel/main/docs/_static/badges/coverage.svg)
[![Python Versions](https://img.shields.io/pypi/pyversions/parq-blockmodel.svg)](https://pypi.org/project/parq-blockmodel/)
[![License](https://img.shields.io/github/license/Elphick/parq-blockmodel.svg?logo=apache&logoColor=white)](https://pypi.org/project/parq-blockmodel/)
[![Publish Docs](https://github.com/Elphick/parq-blockmodel/actions/workflows/docs_to_gh_pages.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-blockmodel/actions/workflows/docs_to_gh_pages.yml)
[![Open Issues](https://img.shields.io/github/issues/Elphick/parq-blockmodel.svg)](https://github.com/Elphick/parq-blockmodel/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Elphick/parq-blockmodel.svg)](https://github.com/Elphick/parq-blockmodel/pulls)


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

See the installation guide and user guide for more detail.