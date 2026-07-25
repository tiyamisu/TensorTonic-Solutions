import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle dataset and yield mini-batches.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if rng is None:
        rng = np.random.default_rng()

    indices = np.arange(len(X))
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    n = len(X)

    for start in range(0, n, batch_size):
        end = start + batch_size

        if end > n and drop_last:
            break

        yield X[start:end], y[start:end]