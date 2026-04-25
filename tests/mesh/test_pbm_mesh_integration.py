"""Integration tests for mesh export from ParquetBlockModel."""

import tempfile
from pathlib import Path

import pytest

from parq_blockmodel import ParquetBlockModel


@pytest.mark.integration
class TestParquetBlockModelMesh:
    """Test mesh export methods on ParquetBlockModel."""

    @pytest.fixture
    def sample_pbm(self, tmp_path):
        """Create a sample block model for testing."""
        pbm_path = tmp_path / "test.pbm"
        pbm = ParquetBlockModel.create_toy_blockmodel(
            filename=pbm_path,
            shape=(3, 3, 3),
            block_size=(1.0, 1.0, 1.0),
            corner=(0.0, 0.0, 0.0),
        )
        yield pbm

    def test_triangulate_basic(self, sample_pbm):
        """Test basic mesh generation."""
        mesh = sample_pbm.triangulate()

        assert mesh.n_vertices > 0
        assert mesh.n_faces > 0
        mesh.validate()

    def test_triangulate_with_attributes(self, sample_pbm):
        """Test mesh generation with attributes."""
        mesh = sample_pbm.triangulate(attributes=["grade"])

        assert "grade" in mesh.vertex_attributes
        assert len(mesh.vertex_attributes["grade"]) == mesh.n_vertices

    def test_triangulate_invalid_attribute(self, sample_pbm):
        """Test that invalid attributes are rejected."""
        with pytest.raises(ValueError, match="not found"):
            sample_pbm.triangulate(attributes=["nonexistent_attr"])

    def test_to_ply_basic(self, sample_pbm):
        """Test PLY export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"
            result = sample_pbm.to_ply(ply_path)

            assert result == Path(ply_path)
            assert ply_path.exists()

    def test_to_ply_with_attributes(self, sample_pbm):
        """Test PLY export with attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"
            sample_pbm.to_ply(ply_path, attributes=["grade"])

            assert ply_path.exists()
            with open(ply_path, 'r') as f:
                content = f.read()
                assert "grade" in content

    def test_to_glb_basic(self, sample_pbm):
        """Test GLB export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = Path(tmpdir) / "test.glb"
            result = sample_pbm.to_glb(glb_path)

            assert result == Path(glb_path)
            assert glb_path.exists()

    def test_to_glb_with_texture(self, sample_pbm):
        """Test GLB export with texture attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = Path(tmpdir) / "test_textured.glb"
            sample_pbm.to_glb(glb_path, texture_attribute="grade", colormap="viridis")

            assert glb_path.exists()
            # Verify it's a valid GLB (magic number)
            with open(glb_path, 'rb') as f:
                magic = f.read(4)
                assert magic == b'glTF'

    def test_to_glb_invalid_texture_attribute(self, sample_pbm):
        """Test that invalid texture attributes are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = Path(tmpdir) / "test.glb"
            with pytest.raises(ValueError, match="not found"):
                sample_pbm.to_glb(glb_path, texture_attribute="nonexistent")

    def test_surface_only_flag(self, sample_pbm):
        """Test that surface_only flag affects mesh generation."""
        mesh_surface = sample_pbm.triangulate(surface_only=True)
        mesh_full = sample_pbm.triangulate(surface_only=False)

        # Surface-only should typically have fewer faces
        assert mesh_surface.n_faces <= mesh_full.n_faces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

