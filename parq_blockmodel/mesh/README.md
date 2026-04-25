Mesh Module - Triangulated Surface Representations
====================================================

The `parq_blockmodel.mesh` submodule provides functionality to convert block models
into triangulated surface geometry, suitable for visualization, exchange with external
tools, and scientific analysis.

Features
--------

✓ **Triangle mesh generation** from regular (rotated and non-rotated) block models
✓ **PLY format** (canonical) - lossless round-tripping with all attributes
✓ **GLB format** (derived) - single-file output with optional textures
✓ **Sparse block models** - efficient surface-only mesh generation
✓ **Attribute preservation** - grades, rock types, densities as vertex/face properties
✓ **Rotated geometry support** - transparent handling of rotated block grids
✓ **Right-handed coordinates** - guaranteed by axis validation
✓ **Scientific workflows** - explicit units, no implicit encodings

Quick Start
-----------

```python
from pathlib import Path
from parq_blockmodel import ParquetBlockModel

# Load a block model
pbm = ParquetBlockModel(Path("model.pbm"))

# Generate mesh (internal representation)
mesh = pbm.triangulate(attributes=["grade", "density"], surface_only=True)

# Export to PLY (lossless, all data)
pbm.to_ply("model.ply", attributes=["grade", "density"])

# Export to GLB (visualization with texture)
pbm.to_glb("model.glb", texture_attribute="grade", colormap="viridis")
```

Key Concepts
------------

### PLY - Canonical Internal Format

PLY (Polygon File Format) is used for lossless, complete representation of mesh data:

- **Vertices**: 3D coordinates in world space
- **Faces**: Triangle connectivity
- **Attributes**: Per-vertex and per-face properties (grades, rock types, etc.)
- **Metadata**: Geometry parameters, CRS, block indices for traceability
- **Round-tripping**: Write → Read preserves exact data

Use PLY for:
- Scientific workflows requiring data integrity
- Long-term archival of mesh data
- Interchange with other geoscience tools
- Detailed attribute preservation

### GLB - Derived Exchange Format

GLB (glTF 2.0 binary) is optimized for external visualization:

- **Single-file format**: Portable and compact
- **Textures**: Vertex colors from scalar attributes
- **Metadata**: Geometry and attributes in glTF extras
- **Viewer compatibility**: Works with Babylon.js, Three.js, Cesium, etc.

Use GLB for:
- 3D visualization in web viewers
- Sharing with non-technical stakeholders
- Lightweight interchange
- Interactive exploration

### Sparse Block Models

Many geological block models are sparse (not all cells filled):

```python
# Surface-only mesh (exterior faces only) - efficient for sparse models
mesh = pbm.triangulate(surface_only=True, sparse=True)

# Dense mesh (full grid) - includes interior empty voids
mesh = pbm.triangulate(surface_only=False, sparse=False)
```

Sparse + surface-only is typical for mining applications.

### Rotated Geometries

Rotations are transparent to the user:

```python
# Rotation angles (degrees)
pbm = ParquetBlockModel.create_toy_blockmodel(
    filename=Path("rotated.pbm"),
    axis_azimuth=30.0,
    axis_dip=15.0,
    axis_plunge=0.0,
)

# Mesh automatically reflects rotation in world space
mesh = pbm.triangulate()
```

### Attribute Mapping

Block attributes are mapped to mesh elements:

**Vertex attributes**: Each vertex inherits from its parent block
**Face attributes**: Each face inherits from its parent block
**Sparse handling**: Vertices on boundaries may be shared; attribute taken from defining block
**NaN values**: Missing data preserved as NaN in PLY, transparent in GLB

### Units and Coordinates

All coordinates use the units and CRS of the underlying block model:

```python
print(pbm.geometry.corner)      # (x0, y0, z0) in block model units
print(pbm.geometry.block_size)  # (dx, dy, dz) in block model units
print(pbm.geometry.srs)         # CRS information (optional)
```

Conversion (feet to meters, etc.) happens at the block model level.

Data Structure
--------------

### TriangleMesh

```python
@dataclass
class TriangleMesh:
    vertices: np.ndarray                    # (n_vertices, 3) in world space
    faces: np.ndarray                       # (n_faces, 3) triangle indices
    vertex_attributes: dict[str, np.ndarray]  # Per-vertex properties
    face_attributes: dict[str, np.ndarray]    # Per-face properties
    vertex_ijk: Optional[np.ndarray]       # (n_vertices, 3) logical indices
    face_ijk: Optional[np.ndarray]         # (n_faces, 3) logical indices
    metadata: dict                          # Geometry + export metadata
```

### Methods

```python
pbm.triangulate(attributes=None, surface_only=True, sparse=None)
    → TriangleMesh
    
pbm.to_ply(output_path, attributes=None, surface_only=True, sparse=None, binary=False)
    → Path
    
pbm.to_glb(output_path, attributes=None, texture_attribute=None, 
           colormap='viridis', surface_only=True, sparse=None)
    → Path
```

Scientific Workflow Considerations
-----------------------------------

**Explicit over implicit**:
- No hidden encodings in colors
- Attributes are explicit columns
- Metadata carries CRS/units

**Lossless preservation**:
- PLY format preserves all data
- Round-trip capability for verification
- Block indices for traceability

**Performance**:
- Sparse models: only generate surfaces (``surface_only=True``)
- Dense models: include all interior geometry if needed
- Rotated geometries: transparent rotation application

**Compatibility**:
- PLY: Human-readable, ASCII-based
- GLB: Optimized for viewers, single-file
- Both carry metadata for downstream tools

Testing
-------

Comprehensive test suite in `tests/mesh/`:

```bash
pytest tests/mesh/test_mesh.py                    # Core mesh generation
pytest tests/mesh/test_pbm_mesh_integration.py    # ParquetBlockModel integration
```

Documentation
--------------

- **User Guide**: `docs/source/user_guide/mesh_export.rst`
- **API Reference**: `docs/source/api/mesh.rst`
- **Example**: `examples/07_mesh_export.py`

Dependencies
------------

Core mesh module uses only standard/already-included libraries:
- `numpy` (already required)
- `pandas` (already required)
- `parq_blockmodel.geometry` (internal)

Optional for enhanced features:
- `matplotlib` (for colormap support in GLB)
- `plyfile` (for advanced PLY handling; currently using custom writer)

Future Enhancements
-------------------

- [ ] Binary PLY support (reduce file size)
- [ ] Advanced texture generation (bump maps, etc.)
- [ ] Mesh simplification/decimation
- [ ] LOD (Level of Detail) generation
- [ ] Mesh smoothing/refinement algorithms
- [ ] Integration with visualization libraries (PyVista, Trimesh)

License
-------

Same as parq-blockmodel (see LICENSE file).

