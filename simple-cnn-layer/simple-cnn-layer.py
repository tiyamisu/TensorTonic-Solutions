def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """

    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape

    H_out = H - KH + 1
    W_out = W_in - KW + 1

    out = np.zeros((N, C_out, H_out, W_out), dtype=float)

    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    region = x[n, :, i:i+KH, j:j+KW]
                    out[n, c_out, i, j] = np.sum(region * W[c_out]) + b[c_out]

    return out