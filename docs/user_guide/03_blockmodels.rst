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

	offset, scale = get_id_encoding_params(encoding)
	x, y, z = decode_frame_coordinates(
		df["world_id"].to_numpy(dtype="int64"),
		offset=offset,
		scale=scale,
	)

	decoded = df.assign(x=x, y=y, z=z)

See also: :ref:`geometry-coordinates`.
