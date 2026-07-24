import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """

    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    # Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # Numerical stability
    S = S - np.max(S, axis=1, keepdims=True)

    exp_S = np.exp(S)

    # Softmax denominator
    denom = np.sum(exp_S, axis=1)

    # Positive pair scores (diagonal)
    numer = np.diag(exp_S)

    # Mean InfoNCE loss
    loss = -np.mean(np.log(numer / denom))

    return float(loss)