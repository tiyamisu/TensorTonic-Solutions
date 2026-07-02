import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)

    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)

    # Scaled dropout mask
    dropout_pattern = (rand >= p).astype(float) / (1 - p)

    # Apply mask
    output = x * dropout_pattern

    return output, dropout_pattern