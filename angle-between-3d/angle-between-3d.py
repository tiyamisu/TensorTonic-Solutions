import numpy as np

def angle_between_3d(v, w):
    v = np.array(v, dtype=float)
    w = np.array(w, dtype=float)

    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)

    # Zero vector → NaN
    if norm_v == 0 or norm_w == 0:
        return np.nan

    cos_theta = np.dot(v, w) / (norm_v * norm_w)

    # Prevent floating-point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)