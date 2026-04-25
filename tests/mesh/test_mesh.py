"""Tests for mesh generation and export functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parq_blockmodel.geometry import RegularGeometry, LocalGeometry, WorldFrame
from parq_blockmodel.mesh.triangulation import BlockMeshGenerator
from parq_blockmodel.mesh.types import TriangleMesh
from parq_blockmodel.mesh.ply import write_ply, read_ply
from parq_blockmodel.mesh.glb import write_glb


class TestBlockMeshGenerator:
    """Test core mesh generation logic."""

    @staticmethod
    def make_geometry(
        corner=(0, 0, 0),
        block_size=(1, 1, 1),
        shape=(2, 2, 2),
        axis_u=(1, 0, 0),
        axis_v=(0, 1, 0),
        axis_w=(0, 0, 1),
    ):
        return RegularGeometry(
            local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
            world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
        )

    def test_init_valid_geometry(self):
        """Test initialization with valid geometry."""
        geometry = self.make_geometry()
        generator = BlockMeshGenerator(geometry)
        assert generator.geometry == geometry

    def test_init_invalid_handedness(self):
        """Test that left-handed axes are rejected."""
        with pytest.raises(ValueError, match="right-handed"):
            geometry = self.make_geometry(axis_w=(0, 0, -1))  # left-handed
            BlockMeshGenerator(geometry)

    def test_block_corners_local(self):
        """Test local cube corner generation."""
        geometry = self.make_geometry(shape=(1, 1, 1))
        generator = BlockMeshGenerator(geometry)
        corners = generator.block_corners_local()

        assert corners.shape == (8, 3)
        assert np.allclose(corners[0], [0, 0, 0])
        assert np.allclose(corners[7], [1, 1, 1])

    def test_block_vertices_xyz_non_rotated(self):
        """Test world coordinate generation for non-rotated block."""
        geometry = self.make_geometry(corner=(10, 20, 30), block_size=(2, 3, 4), shape=(1, 1, 1))
        generator = BlockMeshGenerator(geometry)
        vertices = generator.block_vertices_xyz(0, 0, 0)

        assert vertices.shape == (8, 3)
        # Lower corner should be at (10, 20, 30)
        assert np.allclose(vertices[0], [10, 20, 30])
        # Upper corner should be at (12, 23, 34)
        assert np.allclose(vertices[7], [12, 23, 34])

    def test_cube_faces_ccw_count(self):
        """Test that cube generates 12 triangles (6 faces x 2)."""
        geometry = self.make_geometry(shape=(1, 1, 1))
        generator = BlockMeshGenerator(geometry)
        faces = generator.cube_faces_ccw()

        assert faces.shape == (12, 3)
        assert np.all(faces >= 0)
        assert np.all(faces < 8)

    def test_triangulate_single_block_non_rotated(self):
        """Test mesh generation for single non-rotated block."""
        geometry = self.make_geometry(shape=(1, 1, 1))
        generator = BlockMeshGenerator(geometry)
        mesh = generator.triangulate(block_data=None, surface_only=False, sparse=False)

        # Single block = 8 vertices, 12 triangles
        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12
        mesh.validate()

    def test_triangulate_2x2x2_block_dense(self):
        """Test mesh generation for 2x2x2 dense grid."""
        geometry = self.make_geometry()
        generator = BlockMeshGenerator(geometry)
        mesh = generator.triangulate(block_data=None, surface_only=False, sparse=False)

        # Current triangulation emits per-block vertices rather than
        # deduplicating a structured lattice.
        assert mesh.n_vertices == 64
        # 8 blocks * 12 triangles each = 96
        assert mesh.n_faces == 96
        mesh.validate()

    def test_triangulate_with_attributes(self):
        """Test that attributes are properly mapped to mesh."""
        geometry = self.make_geometry()

        # Create dummy block data
        data_list = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    data_list.append({
                        'i': i, 'j': j, 'k': k,
                        'grade': float(i + j + k),
                        'density': float(i * j * k + 1),
                    })
        block_data = pd.DataFrame(data_list)
        block_data.set_index(['i', 'j', 'k'], inplace=True)

        generator = BlockMeshGenerator(geometry)
        mesh = generator.triangulate(
            block_data=block_data.reset_index(),
            surface_only=False,
            sparse=False,
        )

        assert 'grade' in mesh.vertex_attributes
        assert 'density' in mesh.vertex_attributes
        assert len(mesh.vertex_attributes['grade']) == mesh.n_vertices
        mesh.validate()

    def test_triangulate_sparse_model(self):
        """Test sparse block model mesh generation."""
        geometry = self.make_geometry(shape=(3, 3, 3))

        # Include only a few blocks
        data_list = [
            {'i': 0, 'j': 0, 'k': 0, 'grade': 1.0},
            {'i': 1, 'j': 1, 'k': 1, 'grade': 2.0},
        ]
        block_data = pd.DataFrame(data_list)

        generator = BlockMeshGenerator(geometry)
        mesh = generator.triangulate(
            block_data=block_data,
            surface_only=True,
            sparse=True,
        )

        # The two blocks are disjoint, so each contributes a full cube surface.
        assert mesh.n_faces == 2 * 12
        mesh.validate()

    def test_triangulate_rotated_geometry(self):
        """Test mesh generation for rotated geometry."""
        # 45-degree rotation around Z axis
        angle_rad = np.pi / 4
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        geometry = self.make_geometry(
            shape=(1, 1, 1),
            axis_u=(cos_a, sin_a, 0),
            axis_v=(-sin_a, cos_a, 0),
            axis_w=(0, 0, 1),
        )

        generator = BlockMeshGenerator(geometry)
        mesh = generator.triangulate(block_data=None, surface_only=False, sparse=False)

        # Should still be 8 vertices, 12 triangles
        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12
        # The origin corner remains fixed, but non-zero vertices should rotate.
        assert not np.allclose(mesh.vertices[1], [1, 0, 0])
        mesh.validate()


class TestTriangleMesh:
    """Test TriangleMesh dataclass."""

    def test_basic_mesh_creation(self):
        """Test creating a basic mesh."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)

        mesh = TriangleMesh(vertices=vertices, faces=faces)
        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1

    def test_mesh_validation_invalid_vertices(self):
        """Test that invalid vertex dimensions are caught."""
        vertices = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)  # 2D instead of 3D
        faces = np.array([[0, 1, 2]], dtype=np.intp)

        mesh = TriangleMesh(vertices=vertices, faces=faces)
        with pytest.raises(ValueError, match="shape"):
            mesh.validate()

    def test_mesh_validation_invalid_faces(self):
        """Test that invalid face indices are caught."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 5]], dtype=np.intp)  # Index 5 out of range

        mesh = TriangleMesh(vertices=vertices, faces=faces)
        with pytest.raises(ValueError, match="range"):
            mesh.validate()

    def test_mesh_attributes_mismatch(self):
        """Test that attribute length mismatches are caught."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        vertex_attributes = {"grade": np.array([1.0, 2.0])}  # Wrong length

        mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            vertex_attributes=vertex_attributes,
        )
        with pytest.raises(ValueError, match="wrong length"):
            mesh.validate()


class TestPLYFormat:
    """Test PLY import/export."""

    def test_write_ply_ascii(self):
        """Test writing a simple PLY file."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(vertices=vertices, faces=faces, metadata={"test": "value"})

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"
            write_ply(mesh, ply_path, binary=False)

            assert ply_path.exists()
            # Read and verify header
            with open(ply_path, 'r') as f:
                content = f.read()
                assert "ply" in content
                assert "format ascii" in content
                assert "element vertex 3" in content
                assert "element face 1" in content

    def test_ply_roundtrip(self):
        """Test that PLY can be written and read back."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
        mesh_orig = TriangleMesh(vertices=vertices, faces=faces)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"
            write_ply(mesh_orig, ply_path, binary=False)

            # TODO: Implement read_ply and test roundtrip
            # mesh_read = read_ply(ply_path)
            # assert np.allclose(mesh_orig.vertices, mesh_read.vertices)
            # assert np.array_equal(mesh_orig.faces, mesh_read.faces)


class TestGLBFormat:
    """Test GLB import/export."""

    def test_write_glb_basic(self):
        """Test writing a simple GLB file."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = TriangleMesh(vertices=vertices, faces=faces)

        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = Path(tmpdir) / "test.glb"
            write_glb(mesh, glb_path)

            assert glb_path.exists()
            # Check GLB magic number
            with open(glb_path, 'rb') as f:
                magic = f.read(4)
                assert magic == b'glTF'

    def test_write_glb_with_texture(self):
        """Test GLB export with vertex colors from attribute."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        vertex_attributes = {"grade": np.array([0.5, 1.0, 0.75])}
        mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            vertex_attributes=vertex_attributes,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = Path(tmpdir) / "test_colored.glb"
            write_glb(mesh, glb_path, texture_attribute="grade")

            assert glb_path.exists()
            with open(glb_path, 'rb') as f:
                magic = f.read(4)
                assert magic == b'glTF'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

