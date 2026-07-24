import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)

    # Squared L2 distances for each sample
    d_ap = np.sum((anchor - positive) ** 2, axis=-1)
    d_an = np.sum((anchor - negative) ** 2, axis=-1)

    # Per-sample triplet loss
    loss = np.maximum(0.0, d_ap - d_an + margin)

    # Return average loss as a Python float
    return float(np.mean(loss))