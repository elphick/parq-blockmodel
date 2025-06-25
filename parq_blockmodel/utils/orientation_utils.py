import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


def calculate_orientation(
        blocks: pd.DataFrame,
        query: Optional[str] = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    """
    Calculate the orientation (bearing, dip, plunge) of block centroids.

    Args:
        blocks (pd.DataFrame): DataFrame containing block centroids with x, y, z columns.
        query (Optional[str]): Optional query string to filter the DataFrame.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
        Either 3 vectors (u, v, w) or a DataFrame with bearing, dip, and plunge in degrees.
    """
    # Apply the query if provided
    if query:
        blocks = blocks.query(query)

    # Ensure required columns are present
    if not {'x', 'y', 'z'}.issubset(blocks.columns):
        raise ValueError("The DataFrame must contain 'x', 'y', and 'z' columns.")

    # Extract coordinates
    x = blocks['x'].values
    y = blocks['y'].values
    z = blocks['z'].values

    # Calculate differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    # Calculate bearing (azimuth) in degrees
    bearing = np.degrees(np.arctan2(dy, dx)) % 360

    # Calculate dip in degrees
    horizontal_distance = np.sqrt(dx ** 2 + dy ** 2)
    dip = np.degrees(np.arctan2(dz, horizontal_distance))

    # Calculate plunge in degrees
    vector_magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    plunge = np.degrees(np.arcsin(dz / vector_magnitude))

    # Optionally return as vectors (u, v, w)
    u = dx / vector_magnitude
    v = dy / vector_magnitude
    w = dz / vector_magnitude

    # Return as a DataFrame with angles
    orientation_df = pd.DataFrame({
        'bearing': bearing,
        'dip': dip,
        'plunge': plunge
    })

    return orientation_df


def generate_block_model_with_ellipse(
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        spacing: float,
        ellipse_center: Tuple[float, float],
        semi_major_axis: float,
        semi_minor_axis: float,
        orientation_angle: float,
        grade_min: float = 50,
        grade_max: float = 70
) -> pd.DataFrame:
    """
    Generate a regular block model with an elliptical distribution of 'fe' grades.

    Args:
        x_range (Tuple[float, float]): Range of x coordinates.
        y_range (Tuple[float, float]): Range of y coordinates.
        z_range (Tuple[float, float]): Range of z coordinates.
        spacing (float): Spacing between blocks.
        ellipse_center (Tuple[float, float]): Center of the ellipse (x, y).
        semi_major_axis (float): Semi-major axis of the ellipse.
        semi_minor_axis (float): Semi-minor axis of the ellipse.
        orientation_angle (float): Orientation angle of the ellipse in degrees.
        grade_min (float): Minimum of the 'grade' attribute.
        grade_max (float): Maximum of the 'grade' attribute.

    Returns:
        pd.DataFrame: DataFrame containing the block model with 'x', 'y', 'z', and 'fe' columns.
    """
    # Generate grid coordinates
    x = np.arange(x_range[0], x_range[1], spacing)
    y = np.arange(y_range[0], y_range[1], spacing)
    z = np.arange(z_range[0], z_range[1], spacing)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Flatten the grid
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = zz.ravel()

    # Calculate distances from the ellipse center
    angle_rad = np.radians(orientation_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Rotate coordinates to align with the ellipse orientation
    x_rot = cos_angle * (x_flat - ellipse_center[0]) + sin_angle * (y_flat - ellipse_center[1])
    y_rot = -sin_angle * (x_flat - ellipse_center[0]) + cos_angle * (y_flat - ellipse_center[1])

    # Calculate normalized distance from the ellipse center
    distance = (x_rot / semi_major_axis) ** 2 + (y_rot / semi_minor_axis) ** 2

    # Assign grades based on distance (closer to center = higher grade)
    grade = grade_max - (grade_max - grade_min) * np.clip(distance, 0, 1)

    # Create the block model DataFrame
    block_model = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'grade': grade
    })

    return block_model


def generate_block_model_with_3d_ellipse(
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        spacing: float,
        ellipse_center: Tuple[float, float, float],
        semi_major_axis: float,
        semi_minor_axis: float,
        semi_minor_axis_z: float,
        orientation_angle: float,
        grade_min: float = 50,
        grade_max: float = 70
) -> pd.DataFrame:
    """
    Generate a regular block model with a 3D elliptical distribution of 'fe' grades.

    Args:
        x_range (Tuple[float, float]): Range of x coordinates.
        y_range (Tuple[float, float]): Range of y coordinates.
        z_range (Tuple[float, float]): Range of z coordinates.
        spacing (float): Spacing between blocks.
        ellipse_center (Tuple[float, float, float]): Center of the ellipse (x, y, z).
        semi_major_axis (float): Semi-major axis of the ellipse in the x-y plane.
        semi_minor_axis (float): Semi-minor axis of the ellipse in the x-y plane.
        semi_minor_axis_z (float): Semi-minor axis of the ellipse in the z direction.
        orientation_angle (float): Orientation angle of the ellipse in degrees.
        grade_min (float): Minimum 'fe' grade.
        grade_max (float): Maximum 'fe' grade.

    Returns:
        pd.DataFrame: DataFrame containing the block model with 'x', 'y', 'z', and 'fe' columns.
    """
    # Generate grid coordinates
    x = np.arange(x_range[0], x_range[1], spacing)
    y = np.arange(y_range[0], y_range[1], spacing)
    z = np.arange(z_range[0], z_range[1], spacing)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Flatten the grid
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = zz.ravel()

    # Calculate distances from the ellipse center
    angle_rad = np.radians(orientation_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Rotate coordinates to align with the ellipse orientation in the x-y plane
    x_rot = cos_angle * (x_flat - ellipse_center[0]) + sin_angle * (y_flat - ellipse_center[1])
    y_rot = -sin_angle * (x_flat - ellipse_center[0]) + cos_angle * (y_flat - ellipse_center[1])
    z_rot = z_flat - ellipse_center[2]

    # Calculate normalized distance from the 3D ellipse center
    distance = ((x_rot / semi_major_axis) ** 2 +
                (y_rot / semi_minor_axis) ** 2 +
                (z_rot / semi_minor_axis_z) ** 2)

    # Assign grades based on distance (closer to center = higher grade)
    grade = grade_max - (grade_max - grade_min) * np.clip(distance, 0, 1)

    # Create the block model DataFrame
    block_model = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'grade': grade
    })

    return block_model


import pyvista as pv
import numpy as np


def visualize_block_model_with_threshold():
    # Parameters for the block model
    x_range = (0, 100)
    y_range = (0, 100)
    z_range = (0, 50)
    spacing = 10
    ellipse_center = (50, 50)
    semi_major_axis = 30
    semi_minor_axis = 20
    orientation_angle = 45  # degrees
    grade_min = 50
    grade_max = 70

    # Generate the block model
    block_model = generate_block_model_with_ellipse(
        x_range, y_range, z_range, spacing,
        ellipse_center, semi_major_axis, semi_minor_axis,
        orientation_angle, grade_min, grade_max
    )

    # Create a PyVista point cloud for the block model
    points = block_model[['x', 'y', 'z']].values
    fe_values = block_model['grade'].values
    point_cloud = pv.PolyData(points)
    point_cloud['grade'] = fe_values

    # Generate the 2D ellipse points
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = semi_major_axis * np.cos(theta)
    ellipse_y = semi_minor_axis * np.sin(theta)

    # Rotate the ellipse to match the orientation angle
    angle_rad = np.radians(orientation_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotated_x = cos_angle * ellipse_x - sin_angle * ellipse_y + ellipse_center[0]
    rotated_y = sin_angle * ellipse_x + cos_angle * ellipse_y + ellipse_center[1]

    # Create a PyVista line for the ellipse
    ellipse_points = np.column_stack((rotated_x, rotated_y, np.zeros_like(rotated_x)))
    ellipse_line = pv.PolyData(ellipse_points).delaunay_2d()

    # Plot the block model and the ellipse
    plotter = pv.Plotter()

    # # Add the point cloud with coloring by 'fe'
    # plotter.add_mesh(
    #     point_cloud, scalars='fe', cmap='viridis', point_size=5, render_points_as_spheres=True, label='Block Model'
    # )

    # Add the ellipse
    plotter.add_mesh(ellipse_line, color='red', line_width=2, label='Ellipse')

    # Add a threshold slider for 'grade'
    plotter.add_mesh_threshold(
        point_cloud, scalars='grade', invert=False, title="Filter by grade"
    )

    plotter.add_legend()
    plotter.show()

import pyvista as pv
import numpy as np

def visualize_block_model_with_3d_ellipse_and_slider():
    # Parameters for the block model
    x_range = (0, 100)
    y_range = (0, 100)
    z_range = (0, 50)
    spacing = 10
    ellipse_center = (50, 50, 25)
    semi_major_axis = 30
    semi_minor_axis = 20
    semi_minor_axis_z = 15
    orientation_angle = 45  # degrees
    grade_min = 50
    grade_max = 70

    # Generate the block model
    block_model = generate_block_model_with_3d_ellipse(
        x_range, y_range, z_range, spacing,
        ellipse_center, semi_major_axis, semi_minor_axis,
        semi_minor_axis_z, orientation_angle, grade_min, grade_max
    )

    # Create a PyVista point cloud for the block model
    points = block_model[['x', 'y', 'z']].values
    fe_values = block_model['grade'].values
    point_cloud = pv.PolyData(points)
    point_cloud['grade'] = fe_values

    # Generate the 3D ellipse surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    u, v = np.meshgrid(u, v)
    x = semi_major_axis * np.cos(u) * np.sin(v)
    y = semi_minor_axis * np.sin(u) * np.sin(v)
    z = semi_minor_axis_z * np.cos(v)

    # Rotate the ellipse to match the orientation angle
    angle_rad = np.radians(orientation_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    x_rot = cos_angle * x - sin_angle * y + ellipse_center[0]
    y_rot = sin_angle * x + cos_angle * y + ellipse_center[1]
    z_rot = z + ellipse_center[2]

    # Create a PyVista StructuredGrid for the ellipse
    grid = pv.StructuredGrid(x_rot, y_rot, z_rot)

    # Plot the block model and the 3D ellipse
    plotter = pv.Plotter()
    plotter.add_mesh(
        grid, color='red', opacity=0.3, label='3D Ellipse'
    )
    plotter.add_mesh_threshold(
        point_cloud, scalars='grade', invert=False, title="Filter by grade", cmap='viridis'
    )
    plotter.add_legend()
    plotter.add_axes()
    plotter.show()

if __name__ == '__main__':
    # Run the visualization
    # visualize_block_model_with_threshold()

    visualize_block_model_with_3d_ellipse_and_slider()

