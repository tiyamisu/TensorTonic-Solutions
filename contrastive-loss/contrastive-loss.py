import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)
    y: array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)

    # Convert single vectors to batch of size 1
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    d = np.linalg.norm(a - b, axis=1)

    loss = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)

    if reduction == "sum":
        return float(np.sum(loss))
    else:  # "mean"
        return float(np.mean(loss))