Glossary
========

.. glossary::

    Blockmodel
        Or block-model, a 3D representation of rock properties, used to understand
        spatial distribution of resources/reserves and support mine planning.

    Cell
        Or block.  A single unit in a blockmodel, representing a specific volume of rock. Each cell can have multiple
        attributes (or properties) associated with it.

    Block
        See :term:`Cell`.

    Attributes
        Or variables, properties. A specific characteristic or quality of a cell
        in a blockmodel, such as grade, density, or porosity.
        In ``parq-blockmodel``, this usually refers to non-positional block
        properties (not ``i/j/k`` or ``x/y/z``).

    Variables
        See :term:`Attributes`.

    Properties
        See :term:`Attributes`.

    Columns
        Table fields in the parquet representation. Columns include both positional/identity columns
        (for example ``block_id``, ``world_id``, ``i/j/k``, ``x/y/z``) and attribute/property columns.

    ParquetBlockModel
        The main class in ``parq-blockmodel``. It wraps a parquet-backed block
        model, attached geometry metadata, optional schema validation, profiling,
        and visualization access.

    PBM file
        The canonical on-disk container used by ``parq-blockmodel``.
        A ``.pbm`` file is a parquet table with embedded
        ``"parq-blockmodel"`` metadata.

    Persisted columns
        Columns physically stored in the parquet file
        (``ParquetBlockModel.persisted_columns``), including position and
        attribute fields.

    Calculated columns
        Columns that are available for materialization but may not be persisted
        on disk (``ParquetBlockModel.calculated_columns``). These come from
        schema ``df-eval`` definitions and built-in geometry-derived values.

    Validate
        The process of checking/coercing block model data against an attached
        Pandera schema, commonly via ``ParquetBlockModel.validate``.

    Profile report
        An HTML report created with ``ParquetBlockModel.create_report`` for
        review of distributions, completeness, and anomalies.

    View
        Interactive 3D visualization of the block model via
        ``ParquetBlockModel.plot`` (PyVista/Trame engines).

    Regular Blockmodel
        A blockmodel where each cell is of uniform size and shape, typically a cube or rectangular prism.

    Irregular Blockmodel
        A blockmodel where cells can vary in size and shape, allowing for more complex geological structures
        to be represented.  This package does not support irregular blockmodels.