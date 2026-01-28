import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_best_fit_transform(source_points, target_points):
    """
    Compute the rigid transformation that best aligns the source points to the target points.

    Parameters:
    - source_points: Numpy array of shape (N, D) representing the source points.
    - target_points: Numpy array of shape (N, D) representing the target points.

    Returns:
    - rotation_matrix: Rotation matrix of shape (D, D).
    - translation_vector: Translation vector of shape (D,).
    """
    assert (
        source_points.shape[1] == target_points.shape[1]
    ), "Source and target must have the same dimensionality."

    # Compute centroids of the source and target points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points by subtracting the centroid
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid

    # Compute the covariance matrix
    covariance_matrix = np.dot(centered_source.T, centered_target)

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Compute the rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    # Compute the translation vector
    translation_vector = target_centroid - np.dot(rotation_matrix, source_centroid)

    return rotation_matrix, translation_vector


def iterative_closest_point(
    source_cloud, target_cloud, max_iterations=20, tolerance=0.001
):
    """
    Align the source point cloud to the target point cloud using the Iterative Closest Point (ICP) algorithm.

    Parameters:
    - source_cloud: Numpy array of shape (M, D) representing the source point cloud.
    - target_cloud: Numpy array of shape (N, D) representing the target point cloud.
    - max_iterations: Maximum number of iterations for the ICP algorithm.
    - tolerance: Convergence criterion based on change in alignment error.

    Returns:
    - aligned_source: Transformed source point cloud aligned to the target.
    - cumulative_rotation: Cumulative rotation matrix.
    - cumulative_translation: Cumulative translation vector.
    """
    # Initialize cumulative transformations
    cumulative_rotation = np.eye(source_cloud.shape[1])
    cumulative_translation = np.zeros(source_cloud.shape[1])

    previous_error = 0

    for iteration in range(max_iterations):
        # Find the closest points in the target cloud for each point in the source cloud
        nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(target_cloud)
        distances, indices = nearest_neighbors.kneighbors(source_cloud)
        closest_points = target_cloud[indices.flatten()]

        # Compute the best-fit transform
        rotation_matrix, translation_vector = compute_best_fit_transform(
            source_cloud, closest_points
        )

        # Update the cumulative transformations
        cumulative_rotation = rotation_matrix @ cumulative_rotation
        cumulative_translation = (
            translation_vector + rotation_matrix @ cumulative_translation
        )

        # Apply the transformation to the source cloud
        source_cloud = np.dot(source_cloud, rotation_matrix.T) + translation_vector

        # Compute the mean error
        mean_error = np.mean(distances.flatten())

        # Check for convergence
        if abs(previous_error - mean_error) < tolerance:
            break
        previous_error = mean_error

    return source_cloud, cumulative_rotation, cumulative_translation


# # Example usage
# source_point_cloud = np.array([[2, 4, 1], [3, 5, 2], [1, 3, 0], [4, 6, 3]])  # Source point cloud
# target_point_cloud = np.array([[2, 3, 1], [3, 4, 2], [1, 2, 0]])  # Target point cloud

# # Align the source point cloud to the target point cloud
# aligned_source, cumulative_rotation, cumulative_translation = iterative_closest_point(source_point_cloud, target_point_cloud)

# print("Aligned Source Point Cloud:\n", aligned_source)
# print("Cumulative Rotation Matrix:\n", cumulative_rotation)
# print("Cumulative Translation Vector:\n", cumulative_translation)
