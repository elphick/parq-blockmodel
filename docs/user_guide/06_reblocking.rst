Reblocking (Up/Down Sampling)
=============================

Reblocking changes the block size of a regular model:

- **Upsampling** creates a finer grid (smaller blocks).
- **Downsampling** creates a coarser grid (larger blocks).

Configuration is explicit per attribute. This avoids accidental defaults when
continuous and class-like attributes are mixed in one model.

.. note::

   Not all upsampling methods are interpolative. ``mode``, ``nearest``, and
   ``parent`` are class-preserving assignment methods.

Upsampling
----------

Use :meth:`parq_blockmodel.blockmodel.ParquetBlockModel.upsample` with an
``upsample_config`` mapping every attribute to a method.

Supported methods are:

- ``linear``: continuous interpolation (for continuous numeric attributes)
- ``nearest``: nearest-source assignment (class-safe)
- ``mode``: neighborhood class mode with deterministic tie-break
  (lowest code/value)
- ``parent``: inherit value directly from the parent block (exact replication)

Typical mixed configuration:

.. code-block:: python

    upsampled = pbm.upsample(
        new_block_size=(0.5, 0.5, 0.5),
        upsample_config={
            "grade": "linear",          # continuous
            "density": "linear",        # continuous
            "rock_type": "mode",        # categorical / class-like
            "domain_code": "parent",    # integer-coded classes
        },
    )

If any attribute is omitted from ``upsample_config``, upsampling fails
immediately with a clear error.

Downsampling
------------

Use :meth:`parq_blockmodel.blockmodel.ParquetBlockModel.downsample` with an
``aggregation_config`` mapping attributes to method dictionaries.

Common methods include:

- ``mean``
- ``sum``
- ``weighted_mean`` (requires ``basis``)
- ``mode`` (class-like attributes)

Typical mixed configuration:

.. code-block:: python

    downsampled = pbm.downsample(
        new_block_size=(2.0, 2.0, 2.0),
        aggregation_config={
            "grade": {"method": "weighted_mean", "basis": "dry_mass"},
            "dry_mass": {"method": "sum"},
            "volume": {"method": "sum"},
            "rock_type": {"method": "mode"},
            "domain_code": {"method": "mode"},
        },
    )

Handling partially-filled blocks with fill_ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When downsampling sparse or partially-filled block models, use the ``fill_ratio``
key to normalize aggregation by the fraction of blocks actually occupied.

This is useful if child blocks in a region are only partly filled (e.g., from
surface-only meshes or sparse models), and you want to avoid underestimating
aggregated values.

Example: downsampling a grade with fill-aware weighting:

.. code-block:: python

    downsampled = pbm.downsample(
        new_block_size=(2.0, 2.0, 2.0),
        aggregation_config={
            "grade": {"method": "weighted_mean", "basis": "mass", "fill_ratio": "fill_factor"},
            "mass": {"method": "sum", "fill_ratio": "fill_factor"},
            "fill_factor": {"method": "mean"},  # required: aggregate the fill ratio itself
        },
    )

How it works:

- ``fill_factor`` must be an attribute (array) with values in ``[0, 1]`` that
  indicate the fraction of child blocks that are occupied in each coarse block.
- For ``sum`` with ``fill_ratio: "fill_factor"``:
  ``result = (sum of values) / mean(fill_factor)``.
- For ``weighted_mean`` with ``fill_ratio: "fill_factor"``:
  ``result = (sum of (values * basis)) / (mean(fill_factor) * sum of basis)``.

Only one ``fill_ratio`` attribute may be used across all aggregations in
a single downsampling call.  If you omit ``fill_ratio``, aggregations work
on all available child blocks without normalization.

Choosing methods
----------------

Use this rule of thumb:

- **Continuous attributes** (grade, density, porosity): ``linear`` for
  upsample, ``mean``/``weighted_mean`` for downsample.
- **Class-like attributes** (categorical labels, integer-coded domains):
  ``parent``, ``nearest``, or ``mode`` for upsample, ``mode`` for downsample.

IDW interpolation
-----------------

Inverse Distance Weighting (IDW) is not currently implemented.

In this package, adding IDW for upsampling is likely non-trivial because it
would need:

- method wiring across API/config validation,
- deterministic handling for boundary and missing data,
- behavior definition for integer and class-like attributes,
- new regression and integration tests.

Given that, IDW is best treated as a separate scoped feature after the
class-safe upsampling behavior is finalized.



