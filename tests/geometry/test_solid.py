from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from parq_blockmodel import LocalGeometry, MeshSolid, RegularGeometry, WorldFrame
from parq_blockmodel.mesh.ply import write_ply
from parq_blockmodel.mesh.types import TriangleMesh


def add_cube_3dfaces(target, origin, size=1.0, layer=None):
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + size, y0 + size, z0 + size
    v000 = (x0, y0, z0)
    v100 = (x1, y0, z0)
    v110 = (x1, y1, z0)
    v010 = (x0, y1, z0)
    v001 = (x0, y0, z1)
    v101 = (x1, y0, z1)
    v111 = (x1, y1, z1)
    v011 = (x0, y1, z1)
    dxfattribs = {"layer": layer} if layer is not None else None

    target.add_3dface([v000, v100, v110, v010], dxfattribs=dxfattribs)
    target.add_3dface([v001, v101, v111, v011], dxfattribs=dxfattribs)
    target.add_3dface([v000, v001, v011, v010], dxfattribs=dxfattribs)
    target.add_3dface([v100, v101, v111, v110], dxfattribs=dxfattribs)
    target.add_3dface([v000, v100, v101, v001], dxfattribs=dxfattribs)
    target.add_3dface([v010, v110, v111, v011], dxfattribs=dxfattribs)


def make_geometry(
    corner=(0.0, 0.0, 0.0),
    block_size=(1.0, 1.0, 1.0),
    shape=(3, 3, 3),
    axis_u=(1.0, 0.0, 0.0),
    axis_v=(0.0, 1.0, 0.0),
    axis_w=(0.0, 0.0, 1.0),
):
    return RegularGeometry(
        local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
        world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
    )


def cube_mesh(bounds: tuple[float, float, float, float, float, float]) -> TriangleMesh:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    vertices = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 5, 1],
            [0, 4, 5],
            [1, 6, 2],
            [1, 5, 6],
            [2, 7, 3],
            [2, 6, 7],
            [3, 4, 0],
            [3, 7, 4],
        ],
        dtype=np.intp,
    )
    return TriangleMesh(vertices=vertices, faces=faces)


def write_mesh(mesh: TriangleMesh, path: Path) -> Path:
    write_ply(mesh, path, binary=False)
    return path


def test_mesh_solid_evaluate_cube_mask(tmp_path):
    geometry = make_geometry(shape=(3, 3, 3))
    ply_path = write_mesh(cube_mesh((0.0, 2.0, 0.0, 2.0, 0.0, 2.0)), tmp_path / "cube.ply")
    solid = MeshSolid.from_ply(ply_path)

    mask = solid.evaluate(geometry)

    i, j, k = geometry.ijk_from_row_index(np.arange(np.prod(geometry.local.shape)))
    expected = (i <= 1) & (j <= 1) & (k <= 1)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, expected)


def test_mesh_solid_boundary_points_are_inside(tmp_path):
    geometry = make_geometry(shape=(1, 1, 1))
    # Centroid is (0.5, 0.5, 0.5), exactly on xmin/ymin/zmin boundaries.
    ply_path = write_mesh(cube_mesh((0.5, 1.5, 0.5, 1.5, 0.5, 1.5)), tmp_path / "boundary_cube.ply")
    solid = MeshSolid.from_ply(ply_path)

    mask = solid.evaluate(geometry)

    np.testing.assert_array_equal(mask, np.array([True], dtype=bool))


def test_mesh_solid_evaluate_rotated_geometry(tmp_path):
    angle = np.deg2rad(45.0)
    axis_u = (float(np.cos(angle)), float(np.sin(angle)), 0.0)
    axis_v = (float(-np.sin(angle)), float(np.cos(angle)), 0.0)
    geometry = make_geometry(shape=(2, 2, 2), axis_u=axis_u, axis_v=axis_v)

    x0, y0, z0 = geometry.centroid_x[0], geometry.centroid_y[0], geometry.centroid_z[0]
    ply_path = write_mesh(
        cube_mesh((x0 - 0.1, x0 + 0.1, y0 - 0.1, y0 + 0.1, z0 - 0.1, z0 + 0.1)),
        tmp_path / "tiny_cube.ply",
    )
    solid = MeshSolid.from_ply(ply_path)

    mask = solid.evaluate(geometry)

    expected = np.zeros(8, dtype=bool)
    expected[0] = True
    np.testing.assert_array_equal(mask, expected)


def test_mesh_solid_from_ply_rejects_non_watertight_mesh(tmp_path):
    open_mesh = TriangleMesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.intp),
    )
    ply_path = write_mesh(open_mesh, tmp_path / "open_mesh.ply")

    with pytest.raises(ValueError, match="watertight"):
        MeshSolid.from_ply(ply_path)


def test_mesh_solid_evaluate_accepts_geometry_owner(tmp_path):
    class GeometryOwner:
        def __init__(self, geometry):
            self.geometry = geometry

    geometry = make_geometry(shape=(1, 1, 1))
    ply_path = write_mesh(cube_mesh((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)), tmp_path / "owner_cube.ply")
    solid = MeshSolid.from_ply(ply_path, name="ore_shell")

    mask = solid.evaluate(GeometryOwner(geometry))

    assert solid.name == "ore_shell"
    np.testing.assert_array_equal(mask, np.array([True], dtype=bool))


def test_mesh_solid_from_pyvista_surface():
    geometry = make_geometry(shape=(2, 2, 2))
    surface = pv.Sphere(radius=0.25, center=(0.5, 0.5, 0.5)).triangulate()

    solid = MeshSolid.from_pyvista(surface, name="sphere")
    mask = solid.evaluate(geometry)

    assert solid.name == "sphere"
    assert mask.dtype == bool
    assert mask.shape == (8,)
    assert mask.any()


def test_mesh_solid_from_pyvista_rejects_non_triangulated():
    with pytest.raises(ValueError, match="triangulated"):
        MeshSolid.from_pyvista(pv.Cube())


def test_mesh_solid_evaluate_with_chunking(monkeypatch):
    geometry = make_geometry(shape=(3, 3, 3))
    solid = MeshSolid.from_pyvista(pv.Cube(bounds=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)).triangulate())

    monkeypatch.setattr(MeshSolid, "EVALUATE_CHUNK_SIZE", 5)
    mask = solid.evaluate(geometry)

    i, j, k = geometry.ijk_from_row_index(np.arange(np.prod(geometry.local.shape)))
    expected = (i <= 1) & (j <= 1) & (k <= 1)
    np.testing.assert_array_equal(mask, expected)


def test_mesh_solid_from_dxf_by_layer(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    geometry = make_geometry(shape=(2, 2, 2))

    doc = ezdxf.new()
    msp = doc.modelspace()
    add_cube_3dfaces(msp, origin=(0.0, 0.0, 0.0), size=1.0, layer="ore_a")
    add_cube_3dfaces(msp, origin=(5.0, 5.0, 5.0), size=1.0, layer="ore_b")
    dxf_path = tmp_path / "solids_by_layer.dxf"
    doc.saveas(dxf_path)

    solid = MeshSolid.from_dxf(dxf_path, name="ore_a", by="layer")
    mask = solid.evaluate(geometry)

    expected = np.zeros(8, dtype=bool)
    expected[0] = True
    np.testing.assert_array_equal(mask, expected)


def test_mesh_solid_from_dxf_requires_name_when_multiple(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")

    doc = ezdxf.new()
    msp = doc.modelspace()
    add_cube_3dfaces(msp, origin=(0.0, 0.0, 0.0), size=1.0, layer="ore_a")
    add_cube_3dfaces(msp, origin=(5.0, 5.0, 5.0), size=1.0, layer="ore_b")
    dxf_path = tmp_path / "multi_layer.dxf"
    doc.saveas(dxf_path)

    with pytest.raises(ValueError, match="Multiple DXF solid candidates"):
        MeshSolid.from_dxf(dxf_path, by="layer")


def test_mesh_solid_from_dxf_by_block(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    geometry = make_geometry(shape=(2, 2, 2))

    doc = ezdxf.new()
    ore_block = doc.blocks.new("ore_shell")
    add_cube_3dfaces(ore_block, origin=(0.0, 0.0, 0.0), size=1.0)
    waste_block = doc.blocks.new("waste_shell")
    add_cube_3dfaces(waste_block, origin=(5.0, 5.0, 5.0), size=1.0)

    msp = doc.modelspace()
    msp.add_blockref("ore_shell", (0.0, 0.0, 0.0))
    msp.add_blockref("waste_shell", (0.0, 0.0, 0.0))

    dxf_path = tmp_path / "solids_by_block.dxf"
    doc.saveas(dxf_path)

    solid = MeshSolid.from_dxf(dxf_path, name="ore_shell", by="block")
    mask = solid.evaluate(geometry)

    expected = np.zeros(8, dtype=bool)
    expected[0] = True
    np.testing.assert_array_equal(mask, expected)
