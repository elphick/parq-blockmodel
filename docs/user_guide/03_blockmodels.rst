Block Models
============

``world_id`` interoperability across block models
-------------------------------------------------

For models in the same SRS, ``world_id`` can be used as a stable join key across
multiple block models, provided there is no centroid overlap.

For cross-model uniqueness, keep these constraints aligned:

* same SRS,
* same ``world_id_encoding`` contract (frame, axis order, scale, and bit layout),
* no overlapping block centroids.

When these are true, ``world_id`` is unique across all participating models and can
be decoded back to centroid coordinates.

Decode from ``pandas.DataFrame`` only
-------------------------------------

You do not need :class:`parq_blockmodel.blockmodel.ParquetBlockModel` to decode
``world_id``. A DataFrame with ``world_id`` and ``df.attrs['parq-blockmodel']`` is
enough.

.. code-block:: python

	import pandas as pd
	from parq_blockmodel.utils import get_id_encoding_params, decode_frame_coordinates

	# df has a world_id column and metadata attached in df.attrs
	meta = df.attrs["parq-blockmodel"]
	encoding = meta["world_id_encoding"]

	offset, scale, bits_per_axis = get_id_encoding_params(encoding)
	x, y, z = decode_frame_coordinates(
		df["world_id"].to_numpy(dtype="int64"),
		offset=offset,
		scale=scale,
		bits_per_axis=bits_per_axis,
	)

	decoded = df.assign(x=x, y=y, z=z)

See also: :ref:`geometry-coordinates`.

Validate block model attributes with Pandera
--------------------------------------------

Install the optional schema dependencies before using validation features:

.. code-block:: bash

    pip install "parq-blockmodel[schema]"

You can pass a schema when constructing a
:class:`parq_blockmodel.blockmodel.ParquetBlockModel`:

* ``ParquetBlockModel.from_parquet(..., schema=...)`` validates and coerces
  source chunks while writing the canonical ``.pbm`` file.
* ``ParquetBlockModel.from_dataframe(..., schema=...)`` validates and coerces
  the DataFrame before it is written.
* ``ParquetBlockModel.from_geometry(..., schema=...)`` stores a schema on an
  empty skeleton model for later reuse.

The ``schema`` argument accepts either:

* a Pandera ``DataFrameSchema`` object, or
* a path to a YAML schema file.

.. code-block:: python

    from pathlib import Path

    from parq_blockmodel import ParquetBlockModel

    pbm = ParquetBlockModel.from_parquet(
        Path("orebody.parquet"),
        schema=Path("schemas/orebody.schema.yaml"),
    )

Validation can then be re-run against the canonical ``.pbm`` file at any time:

.. code-block:: python

    pbm.validate()                # validate every chunk
    pbm.validate(sample_chunks=1) # validate only the first chunk

If a schema was not stored on the instance, you can still provide one directly
to :meth:`parq_blockmodel.blockmodel.ParquetBlockModel.validate`:

.. code-block:: python

    pbm.validate(schema=Path("schemas/orebody.schema.yaml"))

YAML schema loading uses ``df_eval.utils.pandera_io_compat`` so the
``schema`` extra is the recommended installation path for both Pandera and the
YAML loader.

Reblocking configuration guide
-----------------------------

For explicit upsampling/downsampling configuration patterns, including
``upsample_config`` methods (``linear``, ``nearest``, ``mode``, ``parent``)
and downsampling ``weighted_mean`` with ``basis``, see
:doc:`06_reblocking`.

