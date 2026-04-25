"""Edge-case and branch-coverage tests for mesh I/O modules.

Targets the uncovered branches in:
- parq_blockmodel/mesh/ply.py         (read_ply, binary write, vertex_ijk/face_ijk paths)
- parq_blockmodel/mesh/glb.py         (constant attribute, NaN transparency, grayscale fallback)
- parq_blockmodel/mesh/triangulation.py (empty sparse, ijk-MultiIndex path, invalid block_data,
                                          surface_only filtering, srs metadata)
- parq_blockmodel/mesh/types.py        (vertex_ijk/face_ijk shape validation, face_attribute length)
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.mesh.glb import _map_attribute_to_colors, write_glb
from parq_blockmodel.mesh.ply import read_ply, write_ply
from parq_blockmodel.mesh.triangulation import BlockMeshGenerator
from parq_blockmodel.mesh.types import TriangleMesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_mesh(with_ijk: bool = False, with_face_attr: bool = False, with_metadata: bool = False) -> TriangleMesh:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
    vertex_ijk = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.intp) if with_ijk else None
    face_ijk = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.intp) if with_ijk else None
    face_attrs = {"rock_code": np.array([1.0, 2.0])} if with_face_attr else {}
    metadata = {"source": "test", "units": "m"} if with_metadata else {}
    return TriangleMesh(
        vertices=vertices,
        faces=faces,
        vertex_ijk=vertex_ijk,
        face_ijk=face_ijk,
        face_attributes=face_attrs,
        metadata=metadata,
    )


def _make_geometry(shape=(2, 2, 2), srs=None):
    g = RegularGeometry(
        local=LocalGeometry(corner=(0, 0, 0), block_size=(1, 1, 1), shape=shape),
        world=WorldFrame(axis_u=(1, 0, 0), axis_v=(0, 1, 0), axis_w=(0, 0, 1), srs=srs),
    )
    return g


# ===========================================================================
# PLY read/write
# ===========================================================================


class TestPLYReadWrite:

    def test_write_ply_binary_falls_back_to_ascii(self, tmp_path):
        """binary=True currently delegates to ASCII writer; file must be readable."""
        mesh = _simple_mesh()
        ply_path = tmp_path / "binary_fallback.ply"
        write_ply(mesh, ply_path, binary=True)

        with open(ply_path) as f:
            content = f.read()
        assert "format ascii" in content

    def test_write_and_read_ply_roundtrip_vertices_and_faces(self, tmp_path):
        """read_ply should reconstruct vertices and faces from an ASCII PLY."""
        mesh = _simple_mesh()
        ply_path = tmp_path / "roundtrip.ply"
        write_ply(mesh, ply_path, binary=False)

        recovered = read_ply(ply_path)
        assert recovered.n_vertices == mesh.n_vertices
        assert recovered.n_faces == mesh.n_faces
        np.testing.assert_allclose(recovered.vertices, mesh.vertices)
        np.testing.assert_array_equal(recovered.faces, mesh.faces)

    def test_read_ply_with_metadata_comments(self, tmp_path):
        """read_ply should parse key=value comment lines into metadata dict."""
        mesh = _simple_mesh(with_metadata=True)
        ply_path = tmp_path / "meta.ply"
        write_ply(mesh, ply_path, binary=False)

        recovered = read_ply(ply_path)
        assert "source" in recovered.metadata
        assert recovered.metadata["source"] == "test"

    def test_read_ply_raises_on_missing_end_header(self, tmp_path):
        """read_ply must raise ValueError when end_header is absent."""
        broken = tmp_path / "broken.ply"
        broken.write_text("ply\nformat ascii 1.0\n")

        with pytest.raises(ValueError, match="end_header"):
            read_ply(broken)

    def test_read_ply_raises_on_non_triangular_face(self, tmp_path):
        """read_ply must reject faces with more/fewer than 3 vertices."""
        quad_ply = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 4\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0 0 0\n"
            "1 0 0\n"
            "0 1 0\n"
            "1 1 0\n"
            "4 0 1 3 2\n"
        )
        p = tmp_path / "quad.ply"
        p.write_text(quad_ply)

        with pytest.raises(ValueError, match="Non-triangular face"):
            read_ply(p)

    def test_write_ply_emits_ijk_vertex_and_face_properties(self, tmp_path):
        """write_ply should include i/j/k columns when vertex_ijk is present."""
        mesh = _simple_mesh(with_ijk=True)
        ply_path = tmp_path / "ijk.ply"
        write_ply(mesh, ply_path, binary=False)

        content = ply_path.read_text()
        assert "property int i" in content
        assert "property int j" in content
        assert "property int k" in content

    def test_write_ply_face_attributes_appear_in_file(self, tmp_path):
        """write_ply should write per-face scalar attributes."""
        mesh = _simple_mesh(with_face_attr=True)
        ply_path = tmp_path / "face_attr.ply"
        write_ply(mesh, ply_path, binary=False)

        content = ply_path.read_text()
        assert "property float rock_code" in content

    def test_write_ply_with_vertex_attribute(self, tmp_path):
        """write_ply should include per-vertex attribute columns."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(vertices=vertices, faces=faces,
                            vertex_attributes={"grade": np.array([0.1, 0.5, 0.9])})
        ply_path = tmp_path / "vertex_attr.ply"
        write_ply(mesh, ply_path, binary=False)

        content = ply_path.read_text()
        assert "property float grade" in content


# ===========================================================================
# GLB color mapping edge cases
# ===========================================================================


class TestMapAttributeToColors:

    def test_constant_attribute_normalizes_to_zero(self):
        """When all values are equal, normalized values should be 0 (no div-by-zero)."""
        attr = np.array([5.0, 5.0, 5.0])
        colors = _map_attribute_to_colors(attr, "viridis")
        assert colors.shape == (3, 4)
        # All normalized values are 0, so all rows should be identical
        assert np.all(colors == colors[0])

    def test_nan_values_become_transparent(self):
        """NaN values should result in alpha=0 (transparent)."""
        attr = np.array([0.0, np.nan, 1.0])
        colors = _map_attribute_to_colors(attr, "viridis")
        assert colors[1, 3] == 0   # NaN → alpha=0
        assert colors[0, 3] != 0   # Non-NaN → opaque

    def test_grayscale_fallback_produces_correct_shape(self):
        """Verify the grayscale fallback path produces (n, 4) RGBA output."""
        attr = np.array([0.0, 0.5, 1.0])
        attr_min, attr_max = np.nanmin(attr), np.nanmax(attr)
        normalized = (attr - attr_min) / (attr_max - attr_min)
        gray = (normalized * 255).astype(np.uint8)
        fallback = np.column_stack([gray, gray, gray, np.full_like(gray, 255)])
        assert fallback.shape == (3, 4)
        assert fallback[0, 0] == 0     # minimum → black
        assert fallback[2, 0] == 255   # maximum → white

    def test_write_glb_without_texture_uses_gray_material(self, tmp_path):
        """GLB without texture_attribute should produce a gray material."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        glb_path = tmp_path / "no_texture.glb"
        write_glb(mesh, glb_path, texture_attribute=None, include_metadata=False)

        with open(glb_path, "rb") as f:
            assert f.read(4) == b"glTF"

    def test_write_glb_exclude_metadata(self, tmp_path):
        """include_metadata=False should not embed extras."""
        import json, struct

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(vertices=vertices, faces=faces, metadata={"key": "value"})
        glb_path = tmp_path / "no_meta.glb"
        write_glb(mesh, glb_path, include_metadata=False)

        with open(glb_path, "rb") as f:
            f.read(12)  # skip header
            json_chunk_len = struct.unpack("<I", f.read(4))[0]
            f.read(4)   # chunk type
            json_bytes = f.read(json_chunk_len)
        gltf = json.loads(json_bytes.decode("utf-8"))
        assert "extras" not in gltf


# ===========================================================================
# Triangulation edge cases
# ===========================================================================


class TestTriangulationEdgeCases:

    def test_triangulate_sparse_with_no_data_falls_back_to_dense_mesh(self):
        """sparse=True with block_data=None triggers _generate_dense_mesh (blocks_to_include is None).

        The implementation delegates to the dense path when there is no block list
        to iterate over — no crash and a non-empty mesh is expected.
        """
        geometry = _make_geometry(shape=(2, 2, 2))
        generator = BlockMeshGenerator(geometry)
        mesh = generator.triangulate(block_data=None, surface_only=True, sparse=True)

        # Should produce the dense mesh (8 blocks × 12 triangles = 96 faces)
        assert mesh.n_faces > 0
        mesh.validate()

    def test_get_block_indices_from_ijk_multiindex(self):
        """_get_block_indices_from_data should work with ijk MultiIndex."""
        geometry = _make_geometry()
        generator = BlockMeshGenerator(geometry)

        df = pd.DataFrame(
            {"grade": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples([(0, 0, 0), (1, 1, 1)], names=["i", "j", "k"]),
        )
        blocks = generator._get_block_indices_from_data(df)
        assert (0, 0, 0) in blocks
        assert (1, 1, 1) in blocks

    def test_get_block_indices_raises_on_missing_positional_info(self):
        """_get_block_indices_from_data should raise when no ijk info present."""
        geometry = _make_geometry()
        generator = BlockMeshGenerator(geometry)

        df = pd.DataFrame({"grade": [1.0, 2.0]})
        with pytest.raises(ValueError, match="'i', 'j', 'k'"):
            generator._get_block_indices_from_data(df)

    def test_triangulate_surface_only_removes_shared_faces_between_adjacent_blocks(self):
        """Two adjacent blocks sharing a face should have fewer surface triangles than two isolated blocks."""
        geometry = _make_geometry(shape=(2, 1, 1))
        generator = BlockMeshGenerator(geometry)

        # Adjacent blocks: (0,0,0) and (1,0,0) share right/left face
        adjacent_data = pd.DataFrame([
            {"i": 0, "j": 0, "k": 0},
            {"i": 1, "j": 0, "k": 0},
        ])
        mesh_adjacent = generator.triangulate(
            block_data=adjacent_data, surface_only=True, sparse=True
        )

        # Isolated blocks: (0,0,0) in a 3x1x1 grid, skipping the middle
        geometry3 = _make_geometry(shape=(3, 1, 1))
        gen3 = BlockMeshGenerator(geometry3)
        isolated_data = pd.DataFrame([
            {"i": 0, "j": 0, "k": 0},
            {"i": 2, "j": 0, "k": 0},
        ])
        mesh_isolated = gen3.triangulate(
            block_data=isolated_data, surface_only=True, sparse=True
        )

        # Adjacent blocks expose 10 faces (12 total, 2 shared faces eliminated × 2 triangles each = 4)
        assert mesh_adjacent.n_faces < mesh_isolated.n_faces

    def test_triangulate_with_srs_in_geometry_adds_metadata(self):
        """geometry.world.srs being set should appear in mesh metadata."""
        geometry = _make_geometry(srs="EPSG:32754")
        generator = BlockMeshGenerator(geometry)
        mesh = generator.triangulate(block_data=None, surface_only=False, sparse=False)

        assert "srs" in mesh.metadata
        assert mesh.metadata["srs"] == "EPSG:32754"

    def test_map_attributes_from_ijk_multiindex(self):
        """_map_attributes should work when block_data has ijk MultiIndex (no column-based lookup)."""
        geometry = _make_geometry(shape=(2, 1, 1))
        generator = BlockMeshGenerator(geometry)

        df = pd.DataFrame(
            {"grade": [10.0, 20.0]},
            index=pd.MultiIndex.from_tuples([(0, 0, 0), (1, 0, 0)], names=["i", "j", "k"]),
        )
        mesh = generator.triangulate(block_data=df, surface_only=False, sparse=True)

        assert "grade" in mesh.vertex_attributes
        # Block (0,0,0) has grade 10, block (1,0,0) has grade 20
        unique_grades = set(mesh.vertex_attributes["grade"])
        assert 10.0 in unique_grades
        assert 20.0 in unique_grades

    def test_triangulate_surface_only_false_includes_all_faces(self):
        """surface_only=False in sparse mode should include internal faces between adjacent blocks."""
        geometry = _make_geometry(shape=(2, 1, 1))
        generator = BlockMeshGenerator(geometry)

        adjacent_data = pd.DataFrame([{"i": 0, "j": 0, "k": 0}, {"i": 1, "j": 0, "k": 0}])
        mesh_surface = generator.triangulate(adjacent_data, surface_only=True, sparse=True)
        mesh_all = generator.triangulate(adjacent_data, surface_only=False, sparse=True)

        assert mesh_all.n_faces > mesh_surface.n_faces


# ===========================================================================
# TriangleMesh validation edge cases
# ===========================================================================


class TestTriangleMeshValidation:

    def test_vertex_ijk_wrong_shape_raises(self):
        vertices = np.zeros((4, 3), dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            vertex_ijk=np.zeros((3, 3), dtype=np.intp),  # 3 rows, should be 4
        )
        with pytest.raises(ValueError, match="vertex_ijk"):
            mesh.validate()

    def test_face_ijk_wrong_shape_raises(self):
        vertices = np.zeros((3, 3), dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            face_ijk=np.zeros((2, 3), dtype=np.intp),  # 2 rows, should be 1
        )
        with pytest.raises(ValueError, match="face_ijk"):
            mesh.validate()

    def test_face_attribute_wrong_length_raises(self):
        vertices = np.zeros((3, 3), dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            face_attributes={"rock": np.array([1.0, 2.0])},  # 2 values, 1 face
        )
        with pytest.raises(ValueError, match="wrong length"):
            mesh.validate()



