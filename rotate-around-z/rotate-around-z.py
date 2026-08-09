import numpy as np

def rotate_around_z(points, theta):
    points = np.array(points, dtype=float)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])

    return np.dot(points, rotation_matrix.T)