Glossary
========

.. glossary::

    Blockmodel
        Or block-model, a 3D representation of rock properties, used to understand spatial distribution of resources and reserves
        and in mine planning.

    Cell
        Or block.  A single unit in a blockmodel, representing a specific volume of rock. Each cell can have multiple
        attributes (or properties) associated with it.

    Block
        See :term:`Cell`.

    Attributes
        Or variables, properties.  A specific characteristic or quality of a cell in a blockmodel, such as grade, density, or porosity.
        In ``parq-blockmodel``, this usually refers to non-positional block properties (not ``i/j/k`` or ``x/y/z``).

    Variables
        See :term:`Attributes`.

    Properties
        See :term:`Attributes`.

    Columns
        Table fields in the parquet representation. Columns include both positional/identity columns
        (for example ``block_id``, ``world_id``, ``i/j/k``, ``x/y/z``) and attribute/property columns.

    Regular Blockmodel
        A blockmodel where each cell is of uniform size and shape, typically a cube or rectangular prism.

    Irregular Blockmodel
        A blockmodel where cells can vary in size and shape, allowing for more complex geological structures
        to be represented.  This package does not support irregular blockmodels.