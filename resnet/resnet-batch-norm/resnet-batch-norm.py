import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    x = np.asarray(x, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    gamma1 = np.asarray(gamma1, dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    eps = 1e-5

    if mode == "post":
        # Conv 1
        z1 = x @ W1

        # BatchNorm 1
        mean1 = np.mean(z1, axis=0)
        var1 = np.var(z1, axis=0)

        z1 = (z1 - mean1) / np.sqrt(var1 + eps)
        z1 = gamma1 * z1 + beta1

        # ReLU
        h = np.maximum(0, z1)

        # Conv 2
        z2 = h @ W2

        # BatchNorm 2
        mean2 = np.mean(z2, axis=0)
        var2 = np.var(z2, axis=0)

        z2 = (z2 - mean2) / np.sqrt(var2 + eps)
        z2 = gamma2 * z2 + beta2

        # Skip connection + ReLU
        output = np.maximum(0, z2 + x)

    else:
        # Pre-activation:
        # BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Add

        # BatchNorm 1
        mean1 = np.mean(x, axis=0)
        var1 = np.var(x, axis=0)

        h = (x - mean1) / np.sqrt(var1 + eps)
        h = gamma1 * h + beta1

        # ReLU
        h = np.maximum(0, h)

        # Conv 1
        h = h @ W1

        # BatchNorm 2
        mean2 = np.mean(h, axis=0)
        var2 = np.var(h, axis=0)

        h = (h - mean2) / np.sqrt(var2 + eps)
        h = gamma2 * h + beta2

        # ReLU
        h = np.maximum(0, h)

        # Conv 2
        h = h @ W2

        # Skip connection
        output = h + x

    return {
        "output": output.tolist(),
        "mode": mode
    }