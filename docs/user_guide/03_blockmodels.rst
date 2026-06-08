Block Models
============

``global_id`` interoperability across block models
-------------------------------------------------

For models in the same SRS, ``global_id`` can be used as a stable join key across
multiple block models, provided there is no centroid overlap.

For cross-model uniqueness, keep these constraints aligned:

* same SRS,
* same ``global_id_encoding`` contract (frame, axis order, scale, and bit layout),
* no overlapping block centroids.

When these are true, ``global_id`` is unique across all participating models and can
be decoded back to centroid coordinates.

Decode from ``pandas.DataFrame`` only
-------------------------------------

You do not need :class:`parq_blockmodel.blockmodel.ParquetBlockModel` to decode
``global_id``. A DataFrame with ``global_id`` and ``df.attrs['parq-blockmodel']`` is
enough.

.. code-block:: python

	import pandas as pd
	from parq_blockmodel.utils import get_id_encoding_params, decode_frame_coordinates

	# df has a global_id column and metadata attached in df.attrs
	meta = df.attrs["parq-blockmodel"]
	encoding = meta["global_id_encoding"]

	offset, scale, bits_per_axis = get_id_encoding_params(encoding)
	x, y, z = decode_frame_coordinates(
		df["global_id"].to_numpy(dtype="int64"),
		offset=offset,
		scale=scale,
		bits_per_axis=bits_per_axis,
	)

	decoded = df.assign(x=x, y=y, z=z)

See also: :ref:`geometry-coordinates`.
