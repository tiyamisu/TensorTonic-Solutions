def max_pooling_2d(X, pool_size):
    """
    Returns non-overlapping maximum-pooled windows.
    """
    H, W = len(X), len(X[0])
    out_h = H // pool_size
    out_w = W // pool_size
    result = []
    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_val = X[i * pool_size][j * pool_size]
            for pi in range(pool_size):
                for pj in range(pool_size):
                    val = X[i * pool_size + pi][j * pool_size + pj]
                    if val > max_val:
                        max_val = val
            row.append(max_val)
        result.append(row)
    return result
