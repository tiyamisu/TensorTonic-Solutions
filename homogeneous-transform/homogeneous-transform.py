import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    single_point = (points.ndim == 1)
    if single_point:
        points = points.reshape(1, 3)

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    # Apply transform
    transformed = (T @ points_h.T).T

    # Extract x, y, z
    transformed = transformed[:, :3]

    if single_point:
        return transformed[0].tolist()
    return transformed.tolist()