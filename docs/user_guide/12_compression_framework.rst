Compression framework
=====================

``parq-blockmodel`` uses a simple compression model with two intended phases:

* **active / local writes** favor speed and default to ``snappy``
* **archive / promote rewrites** favor smaller files and faster reads over the network

The public API keeps the default path simple:

.. code-block:: python

   from pathlib import Path
   from parq_blockmodel import ParquetBlockModel

   pbm = ParquetBlockModel.from_parquet(Path("orebody.parquet"))

   # fast write path (default)
   pbm.write(df, compression="fast")

   # explicit fast-write override
   pbm.write(df, compression=5)

   # archive rewrite in place
   pbm.compress(level=7)

Fast writes
-----------

The ``compression`` argument accepts a small user-facing shorthand:

* ``"fast"`` or ``0`` → ``snappy``
* positive integers → ``zstd`` at that level

This keeps the active path predictable while still allowing explicit zstd writes
when you want them.

Archive rewrites
----------------

``ParquetBlockModel.compress()`` rewrites the backing ``.pbm`` file in place.
It streams data through ``ParquetFile.iter_batches()`` and writes a new file
with ``ParquetWriter`` so large models do not need to be loaded fully into
memory.

Use archive rewrites when you want the stored model to prioritize read
performance and compactness rather than write speed.

Schema-driven overrides
-----------------------

Compression policy can also be stored in Pandera schema metadata under
``metadata["parq-blockmodel"]["compression"]``.

Example schema fragment:

.. code-block:: yaml

   metadata:
     parq-blockmodel:
       compression:
         default:
           codec: snappy
         archive:
           codec: zstd
           level: 7
         columns:
           lithology:
             codec: zstd
             level: 9
           density:
             codec: zstd
             level: 3

During active writes, per-column overrides are ignored so the fast path stays
simple. During ``compress()``, the archive policy is applied and column-level
overrides take precedence over the archive default.

Metadata
--------

Compression settings are stored in the ``.pbm`` metadata payload alongside the
geometry and schema YAML so the chosen policy can be recovered later. The
``parq-blockmodel`` payload now carries ``geometry``, ``schema_yaml``, and
``compression`` together.

See also
--------

* Gallery example: :doc:`/auto_examples/16_compression_framework`
* :doc:`07_calculated_attributes`
* :doc:`03_blockmodels`
