from pathlib import Path

import geopandas as gpd
import pytest
import pyvista as pv
from pyvista.examples import load_channels


# Quarantine: DXF/SHP export helpers (pv_structured_grid_to_dxf,
# mesh_to_2d_polygon_with_holes) are not yet implemented.
# Exit condition: implement the export helpers and enable end-to-end DXF/SHP
# workflow (tracked in todo.rst).

@pytest.mark.skip(reason="Quarantined: pv_structured_grid_to_dxf not implemented; "
                          "unquarantine once DXF export workflow is complete")
def test_pv_to_dxf_export():
    mesh: pv.ImageData = load_channels()
    mesh: pv.UnStructuredGrid = mesh.threshold(value=0.1, invert=False)

    filepath: Path = Path('test_mesh.dxf')
    pv_structured_grid_to_dxf(mesh, filepath)
    assert filepath.exists(), f"DXF file was not created at {filepath}"

    gdf: gpd.GeoDataFrame = gpd.read_file(filepath)
    print(gdf.shape)


@pytest.mark.skip(reason="Quarantined: mesh_to_2d_polygon_with_holes not implemented; "
                          "unquarantine once SHP export workflow is complete")
def test_pv_to_shp_export():
    mesh: pv.ImageData = load_channels()
    mesh: pv.UnStructuredGrid = mesh.threshold(value=0.1, invert=False)

    filepath: Path = Path('test_mesh.shp')
    mesh_to_2d_polygon_with_holes(mesh=mesh)
    assert filepath.exists(), f"File was not created at {filepath}"

    gdf: gpd.GeoDataFrame = gpd.read_file(filepath)

    import matplotlib.pyplot as plt
    gdf.plot(edgecolor='blue', facecolor='none', figsize=(8, 8))
    plt.title("GeoDataFrame Polygons")
    plt.show()

    print(gdf.shape)
