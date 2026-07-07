def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """

    x = np.asarray(x, dtype=float)

    if x.ndim == 3:
        # (C, H, W)
        return np.mean(x, axis=(1, 2))

    elif x.ndim == 4:
        # (N, C, H, W)
        return np.mean(x, axis=(2, 3))

    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")