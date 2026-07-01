:orphan:

Example Gallery
===============

This gallery provides some examples.

.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The underlying geometry (a.k.a. grid) is the foundation of a block model. Since parq-blockmodel is designed to work with regular geometries,  this example demonstrates the underlying regular geometry by visualising it with PyVista.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_01_regular_geometry_thumb.png
    :alt:

  :doc:`/auto_examples/01_regular_geometry`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regular Geometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Rotation of a block model geometry is a common operation in geoscience applications. By rotating the geometry, we can align it with geological features. Rotation is typically specified in degrees from the cardinal orthonormal axes (x, y, z).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_02_rotated_geometry_thumb.png
    :alt:

  :doc:`/auto_examples/02_rotated_geometry`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rotated Geometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Block models represent 3D data, typically via a 3D array.  3D arrays can be flattened into a 2D tabular representation that can be stored in a parquet file.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_03_visualise_blockmodel_thumb.png
    :alt:

  :doc:`/auto_examples/03_visualise_blockmodel`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Block Models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Minimal example of a rotated block model using the new ijk‑first geometry design.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_04_rotated_blockmodel_thumb.png
    :alt:

  :doc:`/auto_examples/04_rotated_blockmodel`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rotated Blockmodel</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A heatmap allows you to visualize the distribution of a specific attribute across a 3D block model from a 2D perspective.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_05_heatmap_thumb.png
    :alt:

  :doc:`/auto_examples/05_heatmap`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Heatmap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reblocking">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_06_reblocking_thumb.png
    :alt:

  :doc:`/auto_examples/06_reblocking`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reblocking</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Block models can be exported as triangulated surface meshes.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_07_mesh_export_thumb.png
    :alt:

  :doc:`/auto_examples/07_mesh_export`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mesh Export</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows an axis-aligned workflow that starts dense and then creates a sparse model from filtered rows.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_08_sparse_parquet_to_pbm_thumb.png
    :alt:

  :doc:`/auto_examples/08_sparse_parquet_to_pbm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dense and Sparse PBM from XYZ Parquet</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates a rotated centroid-parquet workflow where:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_09_rotated_parquet_to_pbm_thumb.png
    :alt:

  :doc:`/auto_examples/09_rotated_parquet_to_pbm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rotated XYZ Parquet to PBM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Schema-backed calculated attributes keep derived values close to the block model schema. This example derives tonnes from density  volume and then derives contained_metal from tonnes  grade.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_10_calculated_attributes_thumb.svg
    :alt:

  :doc:`/auto_examples/10_calculated_attributes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculated Attributes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Use schema-defined calculated attributes in reblocking configuration. In this example tonnes and contained_metal are defined in Pandera df-eval metadata and materialized for downsampling.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_11_reblocking_calculated_attributes_thumb.svg
    :alt:

  :doc:`/auto_examples/11_reblocking_calculated_attributes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reblocking with Calculated Aggregation Inputs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to register custom resolvers with the df-eval engine when working with schema-defined calculated columns. Two approaches are shown: constructor-time registration (eager) and post-load registration (lazy).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_12_calculated_with_custom_lookups_thumb.svg
    :alt:

  :doc:`/auto_examples/12_calculated_with_custom_lookups`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculated Columns with Custom DictResolver Lookups</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates end-to-end polygon-based flagging:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_13_polygon_flagging_thumb.png
    :alt:

  :doc:`/auto_examples/13_polygon_flagging`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polygon Field Flagging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates end-to-end surface-based encoding:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_14_surface_encoding_thumb.png
    :alt:

  :doc:`/auto_examples/14_surface_encoding`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Surface Encoding (2.5D Elevation)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates end-to-end solid-based flagging:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_15_solid_flagging_thumb.png
    :alt:

  :doc:`/auto_examples/15_solid_flagging`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Solid Flagging (3D Volume)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows both ways to start the same Trame app:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_16_trame_threshold_viewer_thumb.svg
    :alt:

  :doc:`/auto_examples/16_trame_threshold_viewer`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trame viewer example with two startup patterns.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows the intended lifecycle:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_17_compression_framework_thumb.svg
    :alt:

  :doc:`/auto_examples/17_compression_framework`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compression Framework</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/01_regular_geometry
   /auto_examples/02_rotated_geometry
   /auto_examples/03_visualise_blockmodel
   /auto_examples/04_rotated_blockmodel
   /auto_examples/05_heatmap
   /auto_examples/06_reblocking
   /auto_examples/07_mesh_export
   /auto_examples/08_sparse_parquet_to_pbm
   /auto_examples/09_rotated_parquet_to_pbm
   /auto_examples/10_calculated_attributes
   /auto_examples/11_reblocking_calculated_attributes
   /auto_examples/12_calculated_with_custom_lookups
   /auto_examples/13_polygon_flagging
   /auto_examples/14_surface_encoding
   /auto_examples/15_solid_flagging
   /auto_examples/16_trame_threshold_viewer
   /auto_examples/17_compression_framework


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
