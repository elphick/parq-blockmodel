import numpy as np
from scipy.spatial.transform import Rotation as R

# Step 1: Create a regular 3D block model (grid of points)
def create_block_model(size=(5, 4, 3)):
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    z = np.linspace(-1, 1, size[2])
    grid = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T
    return grid

# Step 2: Apply known rotation to the block model
def apply_rotation(points, angles_deg):
    rotation = R.from_euler('xyz', angles_deg, degrees=True)
    rotated_points = rotation.apply(points)
    return rotated_points, rotation.as_matrix()

# Step 3: Estimate rotation using SVD
def estimate_rotation(original, rotated):
    # Center the point clouds
    original_centered = original - np.mean(original, axis=0)
    rotated_centered = rotated - np.mean(rotated, axis=0)

    # Compute cross-covariance matrix
    H = original_centered.T @ rotated_centered

    # Perform SVD
    U, _, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T

    # Ensure a proper rotation (det = +1)
    if np.linalg.det(R_est) < 0:
        Vt[2, :] *= -1
        R_est = Vt.T @ U.T

    return R_est

# Step 4: Extract Euler angles from estimated rotation matrix
def extract_euler_angles(rotation_matrix):
    rotation = R.from_matrix(rotation_matrix)
    angles_deg = rotation.as_euler('xyz', degrees=True)
    return angles_deg

# Simulate the process
original_points = create_block_model()
true_angles = [0, -45, 0]  # Rotation around X, Y, Z in degrees
rotated_points, true_rotation_matrix = apply_rotation(original_points, true_angles)

# Estimate rotation matrix using SVD
estimated_rotation_matrix = estimate_rotation(original_points, rotated_points)

# Extract Euler angles from estimated rotation matrix
estimated_angles = extract_euler_angles(estimated_rotation_matrix)

# Output the results
print("True Rotation Angles (degrees):", true_angles)
print("Estimated Rotation Angles (degrees):", np.round(estimated_angles, 2))

