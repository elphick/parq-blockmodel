parq\_blockmodel.blockmodel.ParquetBlockModel
=============================================

.. currentmodule:: parq_blockmodel.blockmodel

.. autoclass:: ParquetBlockModel
   :members:
   :special-members: __add__, __mul__

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~ParquetBlockModel.__init__
      ~ParquetBlockModel.compress
      ~ParquetBlockModel.configure_engine
      ~ParquetBlockModel.create_demo_block_model
      ~ParquetBlockModel.create_heatmap_from_threshold
      ~ParquetBlockModel.create_report
      ~ParquetBlockModel.create_toy_blockmodel
      ~ParquetBlockModel.downsample
      ~ParquetBlockModel.ensure_spatial
      ~ParquetBlockModel.ensure_world_id
      ~ParquetBlockModel.evaluate_surface
      ~ParquetBlockModel.flag_polygon
      ~ParquetBlockModel.flag_solid
      ~ParquetBlockModel.from_dataframe
      ~ParquetBlockModel.from_geometry
      ~ParquetBlockModel.from_parquet
      ~ParquetBlockModel.plot
      ~ParquetBlockModel.plot_heatmap
      ~ParquetBlockModel.read
      ~ParquetBlockModel.recompute_spatial
      ~ParquetBlockModel.recompute_world_id
      ~ParquetBlockModel.rename
      ~ParquetBlockModel.to_dense_parquet
      ~ParquetBlockModel.to_glb
      ~ParquetBlockModel.to_ply
      ~ParquetBlockModel.to_pyvista
      ~ParquetBlockModel.triangulate
      ~ParquetBlockModel.upsample
      ~ParquetBlockModel.validate
      ~ParquetBlockModel.validate_sparse
      ~ParquetBlockModel.validate_special_columns
      ~ParquetBlockModel.validate_xyz_parquet
      ~ParquetBlockModel.write
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ParquetBlockModel.INTRINSIC_VOLUME_COLUMN
      ~ParquetBlockModel.PBM_METADATA_KEY
      ~ParquetBlockModel.POSITION_COLUMNS
      ~ParquetBlockModel.SPECIAL_COLUMN_DTYPES
      ~ParquetBlockModel.SPECIAL_COLUMN_ORDER
      ~ParquetBlockModel.available_attributes
      ~ParquetBlockModel.available_columns
      ~ParquetBlockModel.calculated_attributes
      ~ParquetBlockModel.calculated_columns
      ~ParquetBlockModel.centroid_index
      ~ParquetBlockModel.column_categorical_ordered
      ~ParquetBlockModel.corner
      ~ParquetBlockModel.grid
      ~ParquetBlockModel.id_column
      ~ParquetBlockModel.index_columns
      ~ParquetBlockModel.is_sparse
      ~ParquetBlockModel.origin
      ~ParquetBlockModel.persisted_attributes
      ~ParquetBlockModel.persisted_columns
      ~ParquetBlockModel.position_columns
      ~ParquetBlockModel.schema_service
      ~ParquetBlockModel.sparsity
      ~ParquetBlockModel.spatial_columns
      ~ParquetBlockModel.world_id_column
   
   