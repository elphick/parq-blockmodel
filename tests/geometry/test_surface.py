from pathlib import Path

import numpy as np
import pytest

from parq_blockmodel import LocalGeometry, MeshSurface, RasterSurface, RegularGeometry, WorldFrame
from parq_blockmodel.mesh.ply import write_ply
from parq_blockmodel.mesh.types import TriangleMesh


def make_geometry(
    corner=(0.0, 0.0, 0.0),
    block_size=(1.0, 1.0, 1.0),
    shape=(2, 2, 2),
    axis_u=(1.0, 0.0, 0.0),
    axis_v=(0.0, 1.0, 0.0),
    axis_w=(0.0, 0.0, 1.0),
):
    return RegularGeometry(
        local=LocalGeometry(corner=corner, block_size=block_size, shape=shape),
        world=WorldFrame(axis_u=axis_u, axis_v=axis_v, axis_w=axis_w),
    )


def constant_raster_surface(value: float = 1.5) -> RasterSurface:
    x_coords = np.array([0.0, 1.0, 2.0], dtype=float)
    y_coords = np.array([0.0, 1.0, 2.0], dtype=float)
    values = np.full((3, 3), value, dtype=float)
    return RasterSurface.from_arrays(x_coords=x_coords, y_coords=y_coords, values=values, name="topo")


def plane_mesh_surface(value: float = 1.5) -> TriangleMesh:
    vertices = np.array(
        [
            [0.0, 0.0, value],
            [2.0, 0.0, value],
            [2.0, 2.0, value],
            [0.0, 2.0, value],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)
    return TriangleMesh(vertices=vertices, faces=faces)


def test_raster_surface_evaluate_fraction_below_surface():
    geometry = make_geometry()
    surface = constant_raster_surface()

    values = surface.evaluate(geometry)

    expected = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5], dtype=float)
    np.testing.assert_allclose(values, expected)


def test_raster_surface_rejects_invalid_grid_shape():
    with pytest.raises(ValueError, match="shape"):
        RasterSurface.from_arrays(
            x_coords=np.array([0.0, 1.0], dtype=float),
            y_coords=np.array([0.0, 1.0, 2.0], dtype=float),
            values=np.ones((2, 2), dtype=float),
        )


def test_raster_surface_from_file_roundtrip(tmp_path: Path):
    path = tmp_path / "topography.npz"
    np.savez(
        path,
        x=np.array([0.0, 1.0, 2.0], dtype=float),
        y=np.array([0.0, 1.0, 2.0], dtype=float),
        values=np.full((3, 3), 1.5, dtype=float),
        name="topo",
    )

    surface = RasterSurface.from_file(path)

    assert surface.name == "topo"
    np.testing.assert_allclose(surface.values, np.full((3, 3), 1.5, dtype=float))


def test_raster_surface_evaluate_rotated_geometry():
    angle = np.deg2rad(45.0)
    axis_u = (1.0, 0.0, 0.0)
    axis_v = (0.0, float(np.cos(angle)), float(np.sin(angle)))
    axis_w = (0.0, float(-np.sin(angle)), float(np.cos(angle)))
    geometry = make_geometry(shape=(1, 1, 2), axis_u=axis_u, axis_v=axis_v, axis_w=axis_w)
    surface = RasterSurface.from_arrays(
        x_coords=np.array([0.0, 1.0, 2.0], dtype=float),
        y_coords=np.array([-1.0, 0.0, 1.0], dtype=float),
        values=np.full((3, 3), 1.0, dtype=float),
        name="topo",
    )

    values = surface.evaluate(geometry)

    half_z = 0.5 * (
        geometry.local.block_size[0] * abs(geometry.world.axis_u[2])
        + geometry.local.block_size[1] * abs(geometry.world.axis_v[2])
        + geometry.local.block_size[2] * abs(geometry.world.axis_w[2])
    )
    expected = np.clip((1.0 - (geometry.centroid_z - half_z)) / (2.0 * half_z), 0.0, 1.0)
    np.testing.assert_allclose(values, expected)


def test_mesh_surface_evaluate_fraction_below_surface(tmp_path: Path):
    geometry = make_geometry()
    mesh_path = tmp_path / "plane.ply"
    write_ply(plane_mesh_surface(), mesh_path, binary=False)
    surface = MeshSurface.from_ply(mesh_path, name="topo")

    values = surface.evaluate(geometry)

    expected = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5], dtype=float)
    np.testing.assert_allclose(values, expected)


def test_mesh_surface_rejects_invalid_projection(tmp_path: Path):
    invalid = TriangleMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.intp),
    )
    mesh_path = tmp_path / "invalid_surface.ply"
    write_ply(invalid, mesh_path, binary=False)

    with pytest.raises(ValueError, match="projected area"):
        MeshSurface.from_ply(mesh_path)
